"""Acceptance workflows for real two-recording figures and frozen layout revisions.

The recording files and known event tables are synthetic; comparison generation,
source discovery, project restore, Figure Library, composer and exporters are real.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtGui, QtWidgets

from nanoscope.analysis.events import EventSet, SCHEMA
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.project import load_project
from nanoscope.gui.analysis_explorer import PAGE_COMPARISON_FIGURES as PAGE
from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.figure_composition_revision import extract_composition
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_export_options import resolve_export_options


def _save_evidence(path, name):
    if directory := os.environ.get("NANOSCOPE_BETA_FIGURE_OUTPUT"):
        target = Path(directory) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _known_events(offset):
    rows = []
    for i in range(4):
        row = {name: False if dtype == "bool" else "" if dtype == "string" else 0
               for name, (dtype, _, _) in SCHEMA.items()}
        row.update(event_id=i, start_idx=100 + 200*i, end_idx=120 + 201*i,
                   dwell_s=.0004 + .00002*i, depth_pA=float(offset + 10*(i+1)),
                   amplitude_pA=float(offset + 10*(i+1)), baseline_pA=250.,
                   frac_depth=(offset + 10*(i+1))/250., ecd_pA_s=.01*(i+1),
                   snr=5.+i, accepted=i != 1)
        rows.append(row)
    return pd.DataFrame(rows).astype({name: spec[0] for name, spec in SCHEMA.items()})


@pytest.fixture
def recorded_pair(qtbot, tmp_path, monkeypatch):
    def unexpected(_parent, title, message, *args, **kwargs):
        pytest.fail(f"Unexpected dialog: {title}: {message}")
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", unexpected)
    paths = []
    for condition in ("control", "treated"):
        folder = tmp_path / condition
        folder.mkdir()
        path = folder / "recording.csv"  # Identical names must retain distinct source identities.
        pd.DataFrame({"time": np.arange(10000)/50000.,
                      "current": 250 + np.sin(np.arange(10000)/25)}).to_csv(path, index=False)
        paths.append(path)
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window.open_paths([str(p) for p in paths])
    qtbot.waitUntil(lambda: window._active_open is None and not window._open_queue, timeout=20000)
    assert len(window._session.documents()) == 2
    expected = {}
    for doc in window._session.documents():
        frame = _known_events(0 if doc.meta.path.parent.name == "control" else 100)
        window._session.set_calibration(doc.doc_id, 0, ChannelCalibration(1., 0., "pA"),
                                        write_sidecar=False)
        result = AnalysisResult(module_id="detect.schmitt", module_version="beta-fixture",
            doc_id=doc.doc_id, channel=0, params={},
            calibration_fingerprint=window._session.calibration(doc.doc_id, 0).fingerprint(),
            payload=EventSet(frame, {"calibration": {"target_units": "pA"}}))
        window._session.add_result(result)
        expected[doc.source_key] = frame.copy(deep=True)
    sources = window._plots_sources()
    assert len(sources) == len({source.key for source in sources}) == 2
    assert len({source.label for source in sources}) == 2
    yield window, expected
    window._dirty = False
    if window._figure_library is not None:
        window._figure_library.close()


def _generate_pair(window, kind):
    window._act_comparison_figures.trigger()
    page = window._comparison_figures_window
    assert page is not None
    page.set_config({"kind": kind, "source_keys": [s.key for s in window._plots_sources()],
                     "accepted_only": True, "x": "dwell_s", "y": "depth_pA",
                     "metric": "depth_pA", "features": ["dwell_s", "depth_pA", "snr"],
                     "points": "strip"})
    page._workflow.primary_actions[0].click()
    assert page.has_data() and not page._stale
    return page


def _snapshot(entry):
    return FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_two_same_named_recordings_export_and_restore_exact_comparison(
        recorded_pair, qtbot, monkeypatch, tmp_path, kind):
    window, expected = recorded_pair
    page = _generate_pair(window, kind)
    observations = page._tables["observations"]
    assert len(observations) == 6
    for source in window._plots_sources():
        actual = observations.loc[observations.source_key == source.key]
        target = expected[source.doc_key].loc[lambda frame: frame.accepted]
        assert actual.source_row.tolist() == [0, 2, 3]
        assert actual.event_id.tolist() == [0, 2, 3]
        for column in ("depth_pA", "dwell_s", "snr"):
            if column in actual:
                np.testing.assert_array_equal(actual[column], target[column])
    plotted = page._plot_data().copy(deep=True)
    csv = page.save_csv(tmp_path / f"{kind}.csv")
    exported = pd.read_csv(csv)
    assert len(exported) == len(plotted)
    assert set(exported.source_key) == set(plotted.source_key)
    for column in ("x", "y"):
        if column in plotted:
            np.testing.assert_allclose(exported[column], plotted[column], equal_nan=True)
    entry = window._save_analysis_figure(PAGE, f"Two recordings — {kind}")
    frozen = _snapshot(entry)
    pd.testing.assert_frame_equal(frozen.dataframe(), plotted, check_exact=True)
    image_path = window._figure_library.export_current(tmp_path / f"{kind}.png")
    image = QtGui.QImage(str(image_path))
    assert not image.isNull() and image.width() > 200 and image.height() > 200
    _save_evidence(image_path, f"two-recordings-{kind}.png")
    bundle = window._figure_library.export_current(tmp_path / f"{kind}.zip", bundle=True)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for key, filename in manifest["files"].items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == manifest["sha256"][key]
    _save_evidence(bundle, f"two-recordings-{kind}.zip")
    project = tmp_path / f"{kind}.nanoscope"
    assert window._write_project(str(project))
    saved = load_project(project)
    assert len(saved.documents) == 2 and len(saved.results) == 2
    monkeypatch.setattr(ComparisonFiguresWindow, "analyze",
                        lambda self: pytest.fail("Restore regenerated a comparison"))
    other = MainWindow(recovery_enabled=False)
    qtbot.addWidget(other)
    other._begin_restore(saved)
    qtbot.waitUntil(lambda: not other._restoring, timeout=20000)
    assert len(other._session.documents()) == 2
    other._act_comparison_figures.trigger()
    pd.testing.assert_frame_equal(other._comparison_figures_window._plot_data(), plotted, check_exact=True)
    restored = other._figure_entries()[0]
    assert restored.blobs == entry.blobs
    for source in other._plots_sources():
        pd.testing.assert_frame_equal(source.df, expected[source.doc_key], check_exact=True)
    other._dirty = False
    other.close()


def test_comparison_composition_revision_works_after_raw_sources_are_removed(
        recorded_pair, qtbot, monkeypatch, tmp_path):
    window, expected = recorded_pair
    originals = []
    for kind in ("facets", "events"):
        _generate_pair(window, kind)
        originals.append(window._save_analysis_figure(PAGE, kind))
    originals_before = [(deepcopy(entry.payload), dict(entry.blobs)) for entry in originals]
    library = window._figure_library
    approved = []

    def compose(dialog):
        dialog._name.setText("Two-recording paper figure")
        dialog._columns.setValue(2)
        dialog._table.item(0, 1).setText("A: events")
        dialog._table.item(1, 1).setText("B: facets")
        dialog._preview_btn.click()
        assert dialog.snapshot is not None, dialog._message.text()
        approved.append(dialog.snapshot)
        dialog._save_btn.click()
        return dialog.result()

    monkeypatch.setattr(FigureComposerDialog, "exec", compose)
    composite = library.compose_current()
    assert composite is not None
    assert _snapshot(composite).png_bytes == approved[0].png_bytes
    embedded, layout = extract_composition(_snapshot(composite))
    assert layout["columns"] == 2 and len(embedded) == 2
    assert {panel.snapshot.png_bytes for panel in embedded} == {_snapshot(e).png_bytes for e in originals}
    project = tmp_path / "composition.nanoscope"
    assert window._write_project(str(project))
    saved = load_project(project)
    for path in expected:
        Path(path).unlink()  # Generated temporary input only, proving frozen revision independence.
    other = MainWindow(recovery_enabled=False)
    qtbot.addWidget(other)
    monkeypatch.setattr(ComparisonFiguresWindow, "analyze",
                        lambda self: pytest.fail("Frozen revision regenerated analysis"))
    other._begin_restore(saved)
    qtbot.waitUntil(lambda: not other._restoring, timeout=20000)
    assert not other._session.documents()
    other._open_figure_library()
    restored_library = other._figure_library
    restored_library.refresh(composite.artifact_id)
    before_revision = deepcopy(restored_library.current_entry())

    def revise(dialog):
        dialog._table.selectRow(1)
        dialog._up.click()
        dialog._columns.setValue(1)
        dialog._spacing.setValue(31)
        dialog._name.setText("Journal revision")
        dialog._preview_btn.click()
        assert dialog.snapshot is not None, dialog._message.text()
        dialog._save_btn.click()
        return dialog.result()

    monkeypatch.setattr(FigureComposerDialog, "exec", revise)
    revised = restored_library.revise_current()
    assert revised is not None and revised.artifact_id != composite.artifact_id
    revised_snapshot = _snapshot(revised)
    children, revised_layout = extract_composition(revised_snapshot)
    assert revised_layout["columns"] == 1 and revised_layout["spacing"] == 31
    for child, original in zip(children, reversed(embedded)):
        assert child.snapshot.to_storage() == original.snapshot.to_storage()
    assert revised_snapshot.provenance()["revision"]["revision_of"] == composite.artifact_id
    parent = next(entry for entry in other._figure_entries() if entry.artifact_id == composite.artifact_id)
    assert parent.payload == before_revision.payload and parent.blobs == before_revision.blobs
    for original, (payload, blobs) in zip(originals, originals_before):
        assert original.payload == payload and original.blobs == blobs
    for extension in (".png", ".pdf", ".svg"):
        options = resolve_export_options((revised_snapshot.width, revised_snapshot.height),
            extension=extension, native=False, width=180,
            height=180*revised_snapshot.height/revised_snapshot.width, dpi=150)
        out = restored_library.export_current(tmp_path / ("revision" + extension), options=options)
        assert out is not None and out.stat().st_size > 100
        _save_evidence(out, "composition-revision" + extension)
    bundle = restored_library.export_current(tmp_path / "revision.zip", bundle=True)
    _save_evidence(bundle, "composition-revision.zip")
    revised_project = tmp_path / "revised.nanoscope"
    assert other._write_project(str(revised_project))
    roundtrip = load_project(revised_project)
    final_entry = next(entry for entry in roundtrip.workspace.artifacts if entry.artifact_id == revised.artifact_id)
    assert final_entry.blobs == revised.blobs
    assert len(roundtrip.documents) == 2 and len(roundtrip.results) == 2
    other._dirty = False
    other.close()


def test_submillisecond_comparison_tick_labels_do_not_overlap(recorded_pair):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from nanoscope.plot.comparison_figures import render_comparison

    window, _ = recorded_pair
    page = _generate_pair(window, "facets")
    original = page._tables["plotted"].copy(deep=True)
    figure = render_comparison(page._snapshot, page._tables, page._appearance)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    for axis in figure.axes:
        low, high = axis.get_xlim()
        labels = [label for value, label in zip(axis.get_xticks(), axis.get_xticklabels())
                  if low <= value <= high and label.get_visible() and label.get_text()]
        boxes = [label.get_window_extent(renderer) for label in labels]
        assert len(boxes) >= 3
        assert all(a.x1 + 1 <= b.x0 for a, b in zip(boxes, boxes[1:])), [label.get_text() for label in labels]
        assert axis.get_xlabel() == "Dwell (s)"
    pd.testing.assert_frame_equal(page._tables["plotted"], original, check_exact=True)
