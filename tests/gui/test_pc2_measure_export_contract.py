"""What the two Measure-panel exports are allowed to CLAIM, and the door that reaches one.

Three defects from the 2026-09-02 skeptic pass on the Measure panel, each reproduced
before it was closed (against a copy of ``gui/measure_panel.py`` with the fix hunks
reverted, imported through ``PYTHONPATH`` — never by editing the shipped file back):

PC-UNITS  the brand-new processed-trace export wrote its samples under a column named
          ``current_pA`` and stamped the sidecar ``"units": "pA"`` even when ``normalize``
          was in the chain — i.e. over z-scores, in which a 100 pA open pore reads 0.29.
          The panel's own basis note already warned the user on screen; the file did not.
USABLE-1  nothing in the app reached ``export_processed_trace``: no button, no menu item,
          no shortcut. It worked, and only a test could open it.
PC-C2     the measurements CSV dropped ``signal_basis``. The stamp is chosen in the GUI,
          rides the project and survives a restore, and then the ONE artefact the user
          takes to their analysis lost it — handing back exactly the raw/processed
          ambiguity the column was added to remove.

The two rules every test here is written against:

* naming a unit the numbers do not have is worse than naming none, and
* nothing that a RAW-units export already wrote may move — a chain with no ``normalize``
  in it still writes ``current_pA``, and the measurements CSV's first 32 fields are still
  byte-for-byte what ``measure.write_measurements_csv`` would have written.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.conditioning import (
    BASIS_PROCESSED,
    BASIS_RAW,
    SIGNAL_BASIS_COLUMN,
    conditioned_trace,
)
from nanoscope.analysis.measure import MEASUREMENT_SCHEMA, write_measurements_csv
from nanoscope.core import project as project_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import measure_panel as measure_panel_mod
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
SPAN = (20_000, 24_000)

#: A LINEAR chain: notch + detrend + low-pass. Every stage is linear in the samples, so a
#: picoamp stays a picoamp and the export must still say ``pA``. This is the "nothing
#: moved" control for PC-UNITS.
LINEAR_CHAIN = {
    "notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0, "notch_harmonics": 3,
    "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.05,
    "filter_cutoff_hz": 5000.0, "filter_order": 8, "filter_kind": "bessel",
}

#: The same chain WITH the one stage that takes the units with it.
NORMALIZED_CHAIN = dict(LINEAR_CHAIN, normalize_enabled=True, normalize_mode="zscore")


def _trace(seed: int = 3) -> np.ndarray:
    """Open pore at 100 pA + 60 Hz hum + slow drift + one 30 pA blockade."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / FS
    x = (100.0 + 1.5 * rng.normal(size=N)
         + 2.0 * np.sin(2 * np.pi * 60.0 * t)
         + 3.0 * np.sin(2 * np.pi * 0.4 * t))
    x[SPAN[0]:SPAN[1]] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, arr: np.ndarray, *, name: str = "A.dat", units: str = "pA"):
    info = ChannelInfo(index=0, name="current", raw_units=units, kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _bracket(window: MainWindow, doc_id: str, channel: int, i0: int, i1: int) -> None:
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    window._trace_view.set_measure_region(
        window._trace_view.x_for_index(doc_id, channel, i0),
        window._trace_view.x_for_index(doc_id, channel, i1))
    assert window._trace_view.measure_bounds() == (i0, i1)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    # The csv tests below pin the panel's OWN measure-on choice; in the app the Analysis panel
    # supplies the table's basis instead (test_measure_tab_preview.py), so detach it here.
    w._analysis_panel.measure_panel.set_basis_provider(None)
    w.resize(1512, 900)          # the owner's display
    w.show()
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# PC-UNITS — the column is named after the unit the samples ACTUALLY have
# ================================================================================================
def test_a_normalized_export_never_calls_its_column_picoamps(window, tmp_path):
    """``normalize`` leaves z-scores. Pre-fix the header read ``time_s,current_pA`` and the
    sidecar said ``"units": "pA"`` over numbers whose 100 pA open pore reads 0.29."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(NORMALIZED_CHAIN)
    mp.set_measurement_basis(BASIS_PROCESSED)

    out = mp.export_processed_trace(str(tmp_path / "norm.csv"), whole=True)
    assert out is not None
    df = pd.read_csv(out["data"])
    meta = json.loads(out["meta"].read_text())

    # The samples really are dimensionless — this is the fact the labels have to match.
    want = conditioned_trace(np.asarray(window._session.calibrated(doc_id, channel).data),
                             FS, NORMALIZED_CHAIN)
    # (the low-pass runs after normalize, so the std is near 1 rather than exactly 1 —
    # what matters is that the 100 pA open pore is now a number of order one)
    raw_mean = float(np.asarray(window._session.calibrated(doc_id, channel).data).mean())
    assert 90.0 < raw_mean < 110.0                      # the channel really is ~100 pA
    assert abs(float(want.mean())) < 0.05 and float(np.abs(want).max()) < 10.0

    assert list(df.columns) != ["time_s", "current_pA"], "z-scores exported as picoamps"
    current = [c for c in df.columns if c != "time_s"]
    assert len(current) == 1, f"the CSV must stay two columns, got {list(df.columns)}"
    assert "dimensionless" in current[0]
    assert np.array_equal(df[current[0]].to_numpy(dtype=np.float32), want)

    assert meta["units"] != "pA"
    assert "dimensionless" in meta["units"]
    # ...and the unit is said where the user is looking, not only in the sidecar.
    assert "dimensionless" in mp._status.text()


def test_a_linear_chain_still_exports_picoamps_and_nothing_moved(window, tmp_path):
    """The control. Notch/detrend/low-pass are linear, so a filtered picoamp is a picoamp:
    this export is byte-for-byte the one the pre-fix build wrote."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(LINEAR_CHAIN)

    out = mp.export_processed_trace(str(tmp_path / "lin.csv"), whole=True)
    df = pd.read_csv(out["data"])
    assert list(df.columns) == ["time_s", "current_pA"]
    assert json.loads(out["meta"].read_text())["units"] == "pA"
    want = conditioned_trace(np.asarray(window._session.calibrated(doc_id, channel).data),
                             FS, LINEAR_CHAIN)
    assert np.array_equal(df["current_pA"].to_numpy(dtype=np.float32), want)


@pytest.mark.parametrize(("units", "params", "expected"), [
    ("pA", None, "pA"),                                   # no chain: unchanged
    ("pA", {"filter_cutoff_hz": 5000.0}, "pA"),           # linear stages: unchanged
    ("nA", {"normalize_enabled": False}, "nA"),           # the flag off is not the stage
    ("pA", {"normalize_enabled": True}, "pA normalized (dimensionless)"),
    ("", {"normalize_enabled": True}, "normalized (dimensionless)"),
    ("", None, ""),                                       # unknown stays unknown
])
def test_processed_units_label_says_what_it_knows_and_no_more(units, params, expected):
    assert measure_panel_mod.processed_units_label(units, params) == expected


def test_a_units_label_with_a_comma_cannot_split_the_header(window, tmp_path):
    """Units are free text on the Calibration tab. ``write_processed_trace`` builds the
    header as ``"current_" + units``, so a comma in it would write a three-field header
    over two-column rows and the file would not parse."""
    _install(window, _doc(tmp_path, _trace(), units="pA, corrected"))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(LINEAR_CHAIN)

    out = mp.export_processed_trace(str(tmp_path / "comma.csv"), whole=True)
    df = pd.read_csv(out["data"])
    assert len(df.columns) == 2, f"header split into {list(df.columns)}"
    assert "," not in json.loads(out["meta"].read_text())["units"]


# ================================================================================================
# USABLE-1 — the door
# ================================================================================================
def test_the_measure_panel_has_a_button_that_exports_the_processed_trace(
        window, tmp_path, monkeypatch):
    """Pre-fix, no button/menu item/shortcut anywhere in the app called
    ``export_processed_trace``: a feature only a test could open."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(LINEAR_CHAIN)

    buttons = [b for b in mp.findChildren(QtWidgets.QPushButton)
               if "processed" in b.text().lower()]
    assert len(buttons) == 1, "the Measure panel offers no way to export the processed trace"
    button = buttons[0]
    assert button.isEnabled()

    target = tmp_path / "from_the_button.csv"
    monkeypatch.setattr(measure_panel_mod, "ask_export_path", lambda *a, **k: target)
    button.click()

    assert target.exists() and target.with_suffix(".meta.json").exists()
    want = conditioned_trace(np.asarray(window._session.calibrated(doc_id, channel).data),
                             FS, LINEAR_CHAIN)
    assert np.array_equal(pd.read_csv(target)["current_pA"].to_numpy(dtype=np.float32), want)


def test_the_button_says_why_when_there_is_nothing_to_export(window, tmp_path):
    """It never greys: the gate is a SENTENCE, and the tooltip is where it lives until the
    click puts it in a dialog."""
    mp = window._analysis_panel.measure_panel
    button = next(b for b in mp.findChildren(QtWidgets.QPushButton)
                  if "processed" in b.text().lower())
    assert "open a recording" in button.toolTip().lower()

    _install(window, _doc(tmp_path, _trace()))
    assert "no filter stage is enabled" in button.toolTip().lower()

    mp.set_conditioning(LINEAR_CHAIN)
    assert mp.processed_export_gate() == ""
    assert "sidecar" in button.toolTip()      # now it describes what will be written


def test_the_door_costs_the_dock_no_width_and_moves_nothing_above_it(window, tmp_path):
    """A new control is only worth having if it does not push the Inspector wider.

    Measured at the owner's 1512x900 with the button in place: the panel's minimum width
    is 438 px, exactly what it was without it, because the button SPANS the three columns
    the add/remove/export row already floors — and every one of those controls is still at
    the same pixel. What moved is the table's top edge, one button row down, which is the
    unavoidable cost of any new control and cost it no visible rows at this size."""
    _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    QtWidgets.QApplication.processEvents()
    button = mp._export_processed_button

    # It sits BELOW the cursor tools and the statistics, in the main column (the old
    # measurement list is a group of its own, shown only while it holds rows).
    assert button.geometry().top() >= mp.action_widget.geometry().bottom()
    # ...and it asks for no width the panel did not already ask for.
    assert button.minimumSizeHint().width() <= mp.minimumSizeHint().width()
    assert mp.minimumSizeHint().width() <= 440      # the old measured floor, which only fell


# ================================================================================================
# PC-C2 — the basis stamp rides the measurements CSV
# ================================================================================================
def _add_row(window, doc_id: int, channel: int, basis: str) -> None:
    mp = window._analysis_panel.measure_panel
    mp.set_measurement_basis(basis)
    _bracket(window, doc_id, channel, *SPAN)
    assert mp.add_current()


def test_the_measurements_csv_carries_the_basis_after_the_frozen_32(window, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    _add_row(window, doc_id, channel, BASIS_RAW)
    mp.set_conditioning(LINEAR_CHAIN)
    _add_row(window, doc_id, channel, BASIS_PROCESSED)

    out = mp.export_csv(str(tmp_path / "m.csv"))
    df = pd.read_csv(out)

    # The 32-column contract, in its frozen order, THEN the additive stamp — last.
    assert list(df.columns) == [*MEASUREMENT_SCHEMA, SIGNAL_BASIS_COLUMN]
    assert list(df[SIGNAL_BASIS_COLUMN]) == [BASIS_RAW, BASIS_PROCESSED]
    # The two rows are genuinely different measurements, so the stamp is load-bearing.
    assert df["amplitude"].iloc[0] != df["amplitude"].iloc[1]


def test_the_first_32_fields_are_what_the_schema_writer_would_have_written(window, tmp_path):
    """Additive means ADDITIVE: every existing column, value and position unchanged."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(LINEAR_CHAIN)
    _add_row(window, doc_id, channel, BASIS_PROCESSED)

    mine = pd.read_csv(mp.export_csv(str(tmp_path / "new.csv")))
    theirs = pd.read_csv(
        write_measurements_csv(mp.measurements_df, tmp_path / "schema_only.csv"))

    assert list(theirs.columns) == list(MEASUREMENT_SCHEMA)
    pd.testing.assert_frame_equal(mine[list(MEASUREMENT_SCHEMA)], theirs)
    # And the raw text: the schema export is a strict prefix of every row of the new one.
    for new_line, old_line in zip((tmp_path / "new.csv").read_text().splitlines(),
                                  (tmp_path / "schema_only.csv").read_text().splitlines()):
        assert new_line.startswith(old_line + ",")


def test_a_project_saved_without_the_column_still_loads_and_exports_it_blank(
        window, tmp_path):
    """The back-compat proof. A ``.nanoscope`` written before the stamp existed carries a
    32-column measurements table; it must restore, and export as ``''`` = "not recorded" —
    never as ``raw``, which nobody wrote down."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    _add_row(window, doc_id, channel, BASIS_RAW)

    old = mp.measurements_df.drop(columns=[SIGNAL_BASIS_COLUMN])   # the pre-column table
    assert list(old.columns) == list(MEASUREMENT_SCHEMA)
    path = tmp_path / "old.nanoscope"
    project_mod.save_project(path, project_mod.ProjectState(measurements=old))
    restored = project_mod.load_project(path).measurements
    assert SIGNAL_BASIS_COLUMN not in restored.columns   # the archive really is the old one

    mp.set_measurements_df(restored)                     # MainWindow's restore door
    assert len(mp.measurements_df) == len(old)
    assert list(mp.measurements_df[SIGNAL_BASIS_COLUMN]) == [""]

    df = pd.read_csv(mp.export_csv(str(tmp_path / "old.csv")))
    assert list(df.columns) == [*MEASUREMENT_SCHEMA, SIGNAL_BASIS_COLUMN]
    stamps = df[SIGNAL_BASIS_COLUMN].fillna("").tolist()
    assert stamps == [""], f"a stamp nobody wrote: {stamps}"
