"""Event waveforms: the detection's provenance is written once in the data table, not on every row.

MEASURED DEFECT (assessment, run on a real window; 190 detected events). The Event waveforms
table carried a ``provenance_json`` column holding the detection's whole provenance (58,111
characters, 63 KB once a CSV has quoted it) on EVERY row. The Ensemble view has one row per event
per point: 190 events x 120 points = 22,800 rows = a 1.44 GB table (63 KB per row), the Event-count
heatmap's 1,790 rows = 113 MB. So *Snapshot to Figures*, *Export bundle* and the batch export all
stopped at "The frozen figure exceeds the 512 MiB storage limit.", and a composition could never
be made from the figure. The text is the same on every row, so it is written once now: on the
table's first row, with every other row's cell empty. The column keeps its name and its place, so a
CSV still carries the provenance; every other column of every row is what it was.

Synthetic data only: a 172-event ensemble (the case the assessment measured: 172 x 120 = 20,640
rows) whose detection provenance is made as large as a real one's.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from nanoscope.core.analysis_workspace import EVENT_PLOTS_ARTIFACT_ID
from nanoscope.core.figure_library import figures as saved_figures
from nanoscope.core.figure_library import make_figure
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.session import Session
from nanoscope.gui import figure_batch_export
from nanoscope.gui.event_plot_panel import DENSITY, ENSEMBLE, EventPlotWindow
from nanoscope.gui.figure_bundle import (MAX_SNAPSHOT_BYTES, FigureSnapshot, capture_figure_snapshot,
                                         write_figure_bundle)
from nanoscope.gui.figure_composition import compose_figures
from nanoscope.gui.main_window import MainWindow

from . import test_event_plot as helpers

N_EVENTS = 172
MARKER = "SOURCE-PROVENANCE-MARKER"
#: The columns of the two tables, in order, exactly as the untouched code wrote them (read off it,
#: and unchanged: a reader's script finds every column where it always was).
ENSEMBLE_COLUMNS = [
    "part", "trace_index", "event_row_id", "event_row_id_json", "identity_available", "event_id",
    "event_recording", "event_pore_id", "sample_index", "time", "event_current", "mean_current",
    "median_current", "band_lower", "band_upper", "n_events_at_point", "view",
    "source_identity_available", "source_doc_id", "source_channel", "source_units",
    "provenance_json", "normalize_time", "align", "n_points", "pad_percent", "edge_skipped_count",
    "no_finite_sample_count", "ensemble_subset", "align_polarity", "align_polarity_source",
    "band_kind", "time_display", "time_display_unit",
]
DENSITY_COLUMNS = [
    "part", "x_bin", "y_bin", "x_left", "x_right", "y_left", "y_right", "x_center", "y_center",
    "count", "event_row_id", "event_row_id_json", "x_value", "y_value", "included_in_view",
    "view", "source_identity_available", "source_doc_id", "source_channel", "source_units",
    "provenance_json", "x_metric", "y_metric", "log_x", "log_y", "bins", "colormap", "title",
    "x_label", "y_label", "event_id", "event_recording", "event_pore_id", "x_left_display",
    "x_left_display_unit", "x_right_display", "x_right_display_unit", "x_center_display",
    "x_center_display_unit", "x_value_display", "x_value_display_unit", "y_left_display",
    "y_left_display_unit", "y_right_display", "y_right_display_unit", "y_center_display",
    "y_center_display_unit", "y_value_display", "y_value_display_unit",
]


def _large_provenance() -> dict:
    """About as much text as a real detection records (its measurement-quality block is most of
    the 58 KB), with one marker string so the tests can count how often it is written."""
    return {"measurement_quality": {
        "marker": MARKER,
        "checks": [{"check": f"check-{i}", "status": "ok", "value": i / 7.0,
                    "detail": "a measured quantity " * 3} for i in range(520)]}}


@pytest.fixture
def window(qapp, qtbot, tmp_path, monkeypatch):
    # The helper's trace holds 24 events; a 172-event table needs a longer one.
    monkeypatch.setattr(helpers, "N_SAMPLES", 215_000)
    session = Session()
    doc, events = helpers._doc_and_events(tmp_path, n_events=N_EVENTS)
    events.provenance.update(_large_provenance())
    doc_id = session.add_document(doc)
    helpers._install_detection(session, doc_id, events)
    page = EventPlotWindow(session)
    qtbot.addWidget(page)
    page.set_source(doc_id, 0, events)
    return page


def _expected_provenance(window) -> dict:
    return dict(window._eventset.provenance)


def _csv_bytes(frame: pd.DataFrame) -> int:
    return len(frame.to_csv(index=False).encode("utf-8"))


def _assert_provenance_once(window, frame: pd.DataFrame) -> None:
    """Checked first by every test that writes a file: a table that repeats the text on every
    row is a gigabyte, and that is the one thing none of them may be asked to write."""
    column = frame["provenance_json"]
    assert column.iloc[0] != "" and (column.iloc[1:] == "").all()
    assert json.loads(column.iloc[0]) == _expected_provenance(window)
    assert column.str.count(MARKER).sum() == 1


def test_the_precondition_is_the_measured_case(window):
    window.set_view(ENSEMBLE)
    traces = window.ensemble_traces()[1]
    assert traces.shape[0] == N_EVENTS
    assert len(window._plot_data()) == N_EVENTS * window._npoints.value() == 20_640
    assert len(json.dumps(_expected_provenance(window))) > 55_000


def test_ensemble_table_holds_the_provenance_once_and_is_small(window):
    window.set_view(ENSEMBLE)
    frame = window._plot_data()
    assert len(frame) == 20_640
    _assert_provenance_once(window, frame)
    # On the untouched code a row of this table was 63 KB, and the table 1.4 GB. Now a row is the
    # event x point values and the short constants each row has always carried: 380 bytes here,
    # 7.8 MB for the 20,640 rows. The bounds leave room for a column or two, not for the text.
    size = _csv_bytes(frame)
    assert size < 16 * 1024 * 1024, f"the ensemble CSV is {size:,} bytes"
    assert size / len(frame) < 1_000


def test_density_table_holds_the_provenance_once_and_is_small(window):
    window.set_view(DENSITY)
    frame = window._plot_data()
    assert {"density_bin", "source_event"} == set(frame["part"])
    assert (frame["part"] == "source_event").sum() == N_EVENTS
    _assert_provenance_once(window, frame)
    # 567 bytes a row, 1.0 MB for the 1,772 rows (113 MB on the untouched code, for 190 events).
    size = _csv_bytes(frame)
    assert size < 3 * 1024 * 1024, f"the heatmap CSV is {size:,} bytes"
    assert size / len(frame) < 1_500


@pytest.mark.parametrize("view,columns", [(ENSEMBLE, ENSEMBLE_COLUMNS), (DENSITY, DENSITY_COLUMNS)])
def test_every_other_column_and_every_row_is_unchanged(window, view, columns):
    window.set_view(view)
    frame = window._plot_data()
    # every column is there, in its place, under its name; the text column is still a text column
    assert list(frame.columns) == columns
    assert frame["provenance_json"].dtype == object
    # the short constants are still on every row, as the CSV has always had them
    for name in ("view", "source_doc_id", "source_channel", "source_units"):
        assert frame[name].notna().all()
    assert (frame["view"] == ("density" if view == DENSITY else "ensemble")).all()
    assert frame["source_doc_id"].nunique() == 1


@pytest.mark.parametrize("view", [ENSEMBLE, DENSITY])
def test_the_exported_csv_carries_the_provenance_once(window, tmp_path, view):
    window.set_view(view)
    _assert_provenance_once(window, window._plot_data())
    out = window.export_data(str(tmp_path / f"{view}.csv"))
    assert out is not None
    raw = out.read_text(encoding="utf-8")
    assert raw.count(MARKER) == 1
    assert out.stat().st_size < 16 * 1024 * 1024
    back = pd.read_csv(out, low_memory=False)   # the text is in one cell: one type, not chunk by chunk
    assert len(back) == len(window._plot_data())
    assert json.loads(back["provenance_json"].iloc[0]) == _expected_provenance(window)


@pytest.mark.parametrize("view", [ENSEMBLE, DENSITY])
def test_snapshot_and_bundle_succeed_and_hold_the_provenance_once(window, tmp_path, view):
    window.set_view(view)
    _assert_provenance_once(window, window._plot_data())
    snapshot = capture_figure_snapshot(
        window.figure_export_target(), data=window._plot_data(),
        provenance={"details": window.details_text()})
    metadata, blobs = snapshot.to_storage()
    total = sum(len(value) for value in blobs.values())
    assert total < MAX_SNAPSHOT_BYTES // 16, f"the frozen figure is {total:,} bytes"
    # Once in the data table ...
    assert snapshot.data_csv_bytes.count(MARKER.encode()) == 1
    assert snapshot.dataframe()["provenance_json"].ne("").sum() == 1
    # ... and once in the figure's own provenance record (the Details text).
    assert json.loads(snapshot.provenance_json_bytes)["details"].count(MARKER) == 1
    bundle = write_figure_bundle(tmp_path / f"{view}.zip", snapshot)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("data.csv").count(MARKER.encode()) == 1
        assert json.loads(archive.read("provenance.json"))["details"].count(MARKER) == 1
        assert sum(info.file_size for info in archive.infolist()) < MAX_SNAPSHOT_BYTES // 16


def test_the_library_snapshot_of_a_172_event_ensemble_saves_batches_and_reads_back(window, tmp_path):
    """The route Analysis Explorer > Export > Snapshot to Figures takes (the same capture
    the bundle and the batch export use), then a project save and reopen."""
    window.set_view(ENSEMBLE)
    _assert_provenance_once(window, window._plot_data())
    owner = SimpleNamespace(
        _session=window._session, _figure_workspace=None,
        _figure_page=lambda _page_id: (EVENT_PLOTS_ARTIFACT_ID, window),
        _workspace_kwarg=MainWindow._workspace_kwarg,
        _analysis_explorer_status=lambda _page_id, _page: ("CURRENT", "current"))
    analysis, snapshot, source, live, status = MainWindow._capture_figure(owner, "events")
    assert len(snapshot.dataframe()) == 20_640
    metadata, blobs = snapshot.to_storage()
    entry = make_figure(name="Event overlay", page_id="event_waveforms", source_label=source,
                        analysis=analysis, snapshot_metadata=metadata, snapshot_blobs=blobs,
                        live_state=live, capture_status=status)
    # the batch export's own size gate for a captured figure
    assert figure_batch_export._snapshot_bytes(snapshot) < MAX_SNAPSHOT_BYTES // 16
    path = save_project(tmp_path / "events.nanoscope", ProjectState(
        workspace=AnalyticalWorkspace(artifacts=[entry])))
    back = saved_figures(load_project(path).workspace)[0]
    loaded = FigureSnapshot.from_storage(back.payload["snapshot"], back.blobs)
    assert loaded == snapshot
    assert loaded.dataframe()["provenance_json"].ne("").sum() == 1
    assert path.stat().st_size < 32 * 1024 * 1024


def test_the_two_views_can_be_composed_into_one_figure(window):
    """A composition keeps every panel's exact table, so it was refused outright while a panel's
    table was a gigabyte. Each embedded table now holds the provenance once."""
    snapshots = []
    for view in (ENSEMBLE, DENSITY):
        window.set_view(view)
        _assert_provenance_once(window, window._plot_data())
        snapshots.append(capture_figure_snapshot(
            window.figure_export_target(), data=window._plot_data(),
            provenance={"details": window.details_text()}))
    composed = compose_figures(snapshots, ["a", "b"], 2)
    resources = dict(composed.additional_resources)
    assert resources["panels/panel_001/data.csv"].count(MARKER.encode()) == 1
    assert resources["panels/panel_002/data.csv"].count(MARKER.encode()) == 1
    assert sum(len(value) for value in composed.to_storage()[1].values()) < MAX_SNAPSHOT_BYTES // 8


@pytest.mark.parametrize("view", [ENSEMBLE, DENSITY])
def test_a_snapshot_saved_before_this_change_still_reads_back_and_bundles(qtbot, tmp_path, view):
    """Earlier versions wrote the provenance on every row (a small table was small enough to save).
    Reading a stored snapshot never looks at that column, so those figures open, read back exactly
    and export as they did."""
    page = helpers._fed_window(qtbot, tmp_path, n_events=5)
    page.set_view(view)
    old = page._plot_data()
    old["provenance_json"] = json.dumps(page._eventset.provenance)     # on every row, as it was
    snapshot = capture_figure_snapshot(page.figure_export_target(), data=old,
                                       provenance={"details": page.details_text()})
    metadata, blobs = snapshot.to_storage()
    back = FigureSnapshot.from_storage(metadata, blobs)
    pd.testing.assert_frame_equal(back.dataframe(), old, check_exact=True)
    assert back.dataframe()["provenance_json"].ne("").all()
    bundle = write_figure_bundle(tmp_path / "old.zip", back)
    with zipfile.ZipFile(bundle) as archive:
        assert pd.read_csv(archive.open("data.csv"))["provenance_json"].notna().all()
