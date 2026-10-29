"""Measuring what is on screen: the Measure tool's raw-vs-processed basis, and the export.

The audit's finding, verbatim: *the processed signal can be SEEN and nothing else.* The
preview drew the true full-chain trace while ``measure_panel._measure`` measured
``session.calibrated(...).data`` — the raw channel — and no export of the processed
samples existed anywhere in ``gui/``. These tests drive a real ``MainWindow`` at the
owner's own 1512x900 and hold the three things that fix it:

  - the basis is an EXPLICIT choice and the numbers follow it: a processed reading equals
    ``measure_selection`` on ``conditioning.conditioned_trace`` to the last bit, and a raw
    reading is byte-for-byte what this panel has always produced (no measured number
    moved);
  - a saved row RECORDS the basis, an old project's rows load as "not recorded" rather
    than being back-stamped "raw", and the panel says which basis it is on the line under
    the readout;
  - *Export processed trace* writes a real CSV + provenance sidecar whose samples are the
    array that was on screen;

and two failure shapes that have cost this project rounds before:

  - NOTHING MOVES: switching basis, or losing the chain entirely, must not shift the
    readout under the pointer — the Processed item greys, it does not disappear, and the
    basis line is two lines tall whatever it says;
  - NO SILENT FALLBACK: a chosen basis that becomes unavailable blanks the readout and
    says why, instead of quietly measuring the other signal under the same label.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.conditioning import (
    BASIS_PROCESSED,
    BASIS_RAW,
    BASIS_UNKNOWN,
    SIGNAL_BASIS_COLUMN,
    chain_description,
    conditioned_trace,
)
from nanoscope.analysis.measure import measure_selection
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 200_000
SPAN = (80_000, 84_000)

#: A chain with something for every kind of stage to do, and a low-pass on top — the
#: shape of the owner's own Preprocess forms.
CHAIN = {
    "notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0, "notch_harmonics": 3,
    "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.05,
    "filter_cutoff_hz": 5000.0, "filter_order": 8, "filter_kind": "bessel",
}

#: The owner's display. Every reach measurement in this file is taken at exactly this.
WINDOW_SIZE = (1512, 900)


def _trace(seed: int = 3) -> np.ndarray:
    """200 k float32 at 50 kHz: open pore at 100 pA + hum + slow drift + one blockade.

    The hum and the drift are what make the raw and processed readings genuinely
    different numbers — on a clean synthetic trace the two bases would agree and the
    test would pass without measuring anything.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / FS
    x = (100.0 + 1.5 * rng.normal(size=N)
         + 2.0 * np.sin(2 * np.pi * 60.0 * t)
         + 3.0 * np.sin(2 * np.pi * 0.4 * t))
    x[SPAN[0]:SPAN[1]] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, arr: np.ndarray, *, name: str = "A.dat", fs: float | None = FS):
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,))
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
    """Put the cursors on [i0, i1) exactly as a drag would leave them."""
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
    # These tests pin the panel's OWN measure-on choice. In the app the Analysis panel supplies
    # the target table's basis instead and the choice is hidden (test_measure_tab_preview.py), so
    # the owner is detached to reach the panel as it stands on its own.
    w._analysis_panel.measure_panel.set_basis_provider(None)
    w.resize(*WINDOW_SIZE)          # the owner's display, for every reach measurement
    w.show()
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# 1. the numbers follow the chosen basis
# ================================================================================================
def test_a_processed_reading_is_the_conditioned_trace_and_a_raw_reading_is_not(
        window, tmp_path):
    """The whole complaint in one assertion pair: with a chain configured, *Processed*
    reports ``measure_selection`` of ``conditioned_trace`` — the same array the preview
    curve draws — and the two bases genuinely disagree on this trace."""
    arr = _trace()
    doc_id, channel = _install(window, _doc(tmp_path, arr))
    mp = window._analysis_panel.measure_panel
    _bracket(window, doc_id, channel, *SPAN)

    raw_stats = dict(mp.current_measurement())
    assert mp.measurement_basis() == BASIS_RAW          # the shipped default

    mp.set_conditioning(CHAIN)
    mp.set_measurement_basis(BASIS_PROCESSED)
    got = mp.current_measurement()

    want = measure_selection(
        conditioned_trace(np.asarray(window._session.calibrated(doc_id, channel).data),
                          FS, CHAIN),
        FS, SPAN[0], SPAN[1], amplitude_method="median", units="pA",
        max_samples=None)
    for key in ("amplitude", "level", "baseline", "std", "area_ecd", "snr"):
        assert got[key] == want[key], key
    # ...and it is a different measurement from the raw one, not a relabelled copy.
    assert got["std"] != raw_stats["std"]
    assert abs(got["baseline"] - raw_stats["baseline"]) > 1e-6


def test_the_raw_basis_measures_exactly_what_this_panel_always_measured(window, tmp_path):
    """NO MEASURED NUMBER MOVED. On the default basis the panel's dict must equal
    ``measure_selection`` on the calibrated channel, key for key — the call this panel
    made before the basis choice existed."""
    arr = _trace()
    doc_id, channel = _install(window, _doc(tmp_path, arr))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(CHAIN)          # a chain is configured but NOT selected
    _bracket(window, doc_id, channel, *SPAN)

    got = mp.current_measurement()
    want = measure_selection(
        np.asarray(window._session.calibrated(doc_id, channel).data),
        FS, SPAN[0], SPAN[1], amplitude_method="median", units="pA", max_samples=None)
    assert {k: got[k] for k in want} == want


def test_losing_the_chain_blanks_the_readout_instead_of_measuring_the_other_signal(
        window, tmp_path):
    """NO SILENT FALLBACK. The user unticks the last stage while *Processed* is selected:
    the readout must go blank with a reason, not quietly become a raw reading under a
    label that still says processed."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(CHAIN)
    mp.set_measurement_basis(BASIS_PROCESSED)
    _bracket(window, doc_id, channel, *SPAN)
    assert mp.current_measurement() is not None

    mp.set_conditioning({})             # every stage switched off on the Preprocess tab
    assert mp.current_measurement() is None
    assert mp.measurement_basis() == BASIS_PROCESSED    # the CHOICE is not overwritten
    assert "no filter stage is enabled" in mp.refusal_note()
    assert mp._add_button.isEnabled() is False


# ================================================================================================
# 2. the row records the basis; an old project still loads
# ================================================================================================
def test_a_saved_row_records_which_signal_it_was_measured_on(window, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    _bracket(window, doc_id, channel, *SPAN)
    assert mp.add_current() is True                     # raw
    mp.set_conditioning(CHAIN)
    mp.set_measurement_basis(BASIS_PROCESSED)
    assert mp.add_current() is True                     # processed

    df = mp.measurements_df
    assert list(df[SIGNAL_BASIS_COLUMN]) == [BASIS_RAW, BASIS_PROCESSED]
    # The first row's stamp survived the second append (append_measurement re-coerces the
    # frame it is handed, which drops columns outside the frozen schema).
    assert df["amplitude"].iloc[0] != df["amplitude"].iloc[1]
    assert "processed" in mp._status.text()
    # The table SHOWS it — a stamp nobody can read is not a record.
    assert mp._model.column_index(SIGNAL_BASIS_COLUMN) >= 0


def test_an_old_saved_table_loads_and_reads_as_not_recorded(window, tmp_path):
    """A project saved before this column existed: it must restore, and its rows must NOT
    be back-stamped 'raw'. They were raw — but nobody wrote it down, and the difference
    between a record and an inference is the whole reason for the column."""
    from nanoscope.analysis.measure import empty_measurements_df

    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    _bracket(window, doc_id, channel, *SPAN)
    mp.add_current()
    legacy = mp.measurements_df.drop(columns=[SIGNAL_BASIS_COLUMN])
    assert SIGNAL_BASIS_COLUMN not in legacy.columns    # a genuinely pre-column frame

    mp.set_measurements_df(legacy)
    back = mp.measurements_df
    assert len(back) == 1
    assert back[SIGNAL_BASIS_COLUMN].iloc[0] == BASIS_UNKNOWN
    # every frozen column still there, unchanged
    assert set(empty_measurements_df().columns) <= set(back.columns)
    assert back["amplitude"].iloc[0] == legacy["amplitude"].iloc[0]

    # ...and a stamped row round-trips through the same door (the project restore path).
    stamped = legacy.copy()
    stamped[SIGNAL_BASIS_COLUMN] = pd.Series([BASIS_PROCESSED], dtype="string")
    mp.set_measurements_df(stamped)
    assert mp.measurements_df[SIGNAL_BASIS_COLUMN].iloc[0] == BASIS_PROCESSED


# ================================================================================================
# 3. the panel says so, where the numbers are read — and nothing moves
# ================================================================================================
def test_the_basis_line_names_the_signal_and_the_chain(window, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    assert "RAW" in mp.basis_note()

    mp.set_conditioning(CHAIN)
    mp.set_measurement_basis(BASIS_PROCESSED)
    note = mp.basis_note()
    assert "PROCESSED" in note
    assert chain_description(CHAIN) in note
    assert mp._basis_note.toolTip() == note      # the reserved two lines can clip a chain


def test_switching_basis_never_moves_the_readout_under_the_pointer(window, tmp_path):
    """The failure shape three rounds of a previous campaign were lost to. Every state of
    this control — raw, processed, processed-with-no-chain — must leave the readout table
    at the same y, and the Processed option must GREY rather than vanish."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    panel = mp
    ys, counts = [], []
    for step in ("raw", "chain", "processed", "chain off", "back to raw"):
        if step == "chain":
            mp.set_conditioning(CHAIN)
        elif step == "processed":
            mp.set_measurement_basis(BASIS_PROCESSED)
        elif step == "chain off":
            mp.set_conditioning({})
        elif step == "back to raw":
            mp.set_measurement_basis(BASIS_RAW)
        panel.updateGeometry()
        qapp_process(window)
        ys.append(mp._readout_table.mapTo(panel, mp._readout_table.rect().topLeft()).y())
        counts.append(mp._basis_combo.count())
    assert len(set(ys)) == 1, dict(zip(("raw", "chain", "processed", "off", "raw2"), ys))
    assert counts == [2] * 5

    # Greyed, not gone: the item is still there and still says what it is.
    idx = mp._basis_combo.findData(BASIS_PROCESSED)
    item = mp._basis_combo.model().item(idx)
    assert item is not None and item.isEnabled() is False     # no chain configured
    mp.set_conditioning(CHAIN)
    assert mp._basis_combo.model().item(idx).isEnabled() is True


def qapp_process(window) -> None:
    from PySide6 import QtWidgets
    QtWidgets.QApplication.processEvents()


def test_the_basis_row_fits_the_inspector_dock_at_1512x900(window, tmp_path):
    """REACH. The Inspector dock opens 460 px wide (a 420 px viewport) on the owner's
    1512x900 display, and this project has shipped controls at x=1712 twice. The whole
    'measure on' row — label plus combo — must fit inside that viewport at its minimum,
    so the control is usable without dragging a horizontal scrollbar nobody advertises."""
    _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(CHAIN)
    qapp_process(window)

    combo_w = mp._basis_combo.minimumSizeHint().width()
    label = mp._basis_combo.parentWidget().layout()  # the form layout that owns the row
    assert label is not None
    row_w = combo_w + mp.fontMetrics().horizontalAdvance("measure on") + 24
    assert row_w <= 420, f"'measure on' row needs {row_w} px inside a 420 px viewport"
    # The reserved note is exactly two lines tall and never collapses to nothing.
    assert mp._basis_note.height() == mp._basis_note.minimumHeight() > 0


# ================================================================================================
# 4. getting the samples out
# ================================================================================================
def test_export_processed_trace_writes_the_array_that_was_on_screen(window, tmp_path):
    arr = _trace()
    doc_id, channel = _install(window, _doc(tmp_path, arr))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(CHAIN)
    mp.set_measurement_basis(BASIS_PROCESSED)

    out = mp.export_processed_trace(str(tmp_path / "proc.csv"), whole=True)
    assert out is not None and out["data"].exists() and out["meta"].exists()

    want = conditioned_trace(np.asarray(window._session.calibrated(doc_id, channel).data),
                             FS, CHAIN)
    df = pd.read_csv(out["data"])
    assert len(df) == want.size
    assert np.array_equal(df["current_pA"].to_numpy(dtype=np.float32), want)

    meta = json.loads(out["meta"].read_text())
    assert meta["chain"] == chain_description(CHAIN)
    assert meta["sample_rate_hz"] == FS and meta["units"] == "pA"
    assert meta["span"] == "whole recording" and meta["n_samples"] == want.size
    # Reproducible: the sidecar's own params rebuild the exported samples.
    assert np.array_equal(
        conditioned_trace(np.asarray(window._session.calibrated(doc_id, channel).data),
                          meta["sample_rate_hz"], meta["params"]), want)
    assert "processed samples" in mp._status.text()


def test_the_cursor_selection_is_what_gets_exported_when_there_is_one(window, tmp_path):
    """A 15 M-sample recording is a ~250 MB CSV. When the cursors bracket a span, that
    span is what the user means — and the file says which it was."""
    doc_id, channel = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(CHAIN)
    _bracket(window, doc_id, channel, *SPAN)

    out = mp.export_processed_trace(str(tmp_path / "cut.csv"))
    df = pd.read_csv(out["data"])
    assert len(df) == SPAN[1] - SPAN[0]
    assert df["time_s"].iloc[0] == pytest.approx(SPAN[0] / FS)
    meta = json.loads(out["meta"].read_text())
    assert meta["span"] == "cursor selection"
    assert (meta["start_idx"], meta["end_idx"]) == SPAN
    # The exported samples are a SLICE of the whole-recording chain, not the chain re-run
    # on the cut: every stage here is a filter with memory, so the two differ at the edges
    # and only one of them is what the user was looking at.
    want = conditioned_trace(np.asarray(window._session.calibrated(doc_id, channel).data),
                             FS, CHAIN)[SPAN[0]:SPAN[1]]
    assert np.array_equal(df["current_pA"].to_numpy(dtype=np.float32), want)


def test_the_export_refuses_with_a_reason_when_there_is_no_processed_trace(
        window, tmp_path):
    """A gate string, not a silent no-op: the menu item this feeds needs a sentence to
    grey itself with."""
    mp = window._analysis_panel.measure_panel
    assert "Open a recording" in mp.processed_export_gate()

    _install(window, _doc(tmp_path, _trace()))
    assert "No filter stage is enabled" in mp.processed_export_gate()

    mp.set_conditioning(CHAIN)
    assert mp.processed_export_gate() == ""
    assert mp.export_processed_trace(str(tmp_path / "ok.csv"), whole=True) is not None


def test_a_recording_with_no_sample_rate_cannot_offer_a_processed_basis(window, tmp_path):
    """Every stage is designed in Hz and seconds. A legacy no-rate file must grey the
    option and say so, not raise and not silently measure raw under a processed label."""
    _install(window, _doc(tmp_path, _trace(), name="norate.dat", fs=None))
    mp = window._analysis_panel.measure_panel
    mp.set_conditioning(CHAIN)
    assert mp.processed_basis_available() is False
    assert "no sample rate" in mp.processed_export_gate()
    idx = mp._basis_combo.findData(BASIS_PROCESSED)
    assert mp._basis_combo.model().item(idx).isEnabled() is False
