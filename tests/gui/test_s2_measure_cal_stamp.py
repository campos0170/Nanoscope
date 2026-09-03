"""Regression tests for MEAS2-01: saved rows measured under a superseded calibration.

The flip side of MEAS-01 (test_beta_measure_calibration_live_stats.py): the LIVE readout
follows a calibration edit, but the SAVED list used to be left with nothing — the same
30 pA span measured under gain 1 / gain 2 / (gain 0.001, nA) read 30 / 60 / 0.03 in
adjacent rows with no marking, no units column, and a CSV export that pooled them
permanently. The fix follows the app's own convention for detections (a calibration
fingerprint stamped on the result, amber stale marking, never a silent rescale): each
row stamps ``units``/``cal_gain``/``cal_offset``, rows whose stamp no longer matches the
live calibration are marked in the table (amber foreground + a ⚠ on the row header + a
tooltip saying to re-measure), and the CSV carries the stamp columns.

Driven through the real widgets: a MainWindow, its MeasurePanel, the shared TraceView,
and the Session.set_calibration door the Calibration panel's Apply uses.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
_BLOCKADE = (40_000, 60_000)               # one clean square: depth exactly 30, no noise


def _square_trace() -> np.ndarray:
    x = np.full(N, 100.0, dtype=np.float32)
    x[_BLOCKADE[0]:_BLOCKADE[1]] = 70.0
    return x


def _doc(tmp_path, *, name: str = "cal.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(_square_trace(),))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


def _enable_measure(window: MainWindow, doc_id: str, i0: int, i1: int):
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[i0]), float(vals[i1]))
    return mp


def _apply_cal(window: MainWindow, doc_id: str, channel: int, *,
               gain: float, offset: float = 0.0, units: str = "pA") -> None:
    """The exact door the Calibration panel's Apply uses."""
    window._session.set_calibration(
        doc_id, channel,
        ChannelCalibration(gain=gain, offset=offset, target_units=units,
                           provenance=Provenance.MEASURED),
        write_sidecar=False)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _row_marked(mp, source_row: int) -> dict:
    """How the model presents ``source_row``: foreground/tooltip on a cell, row-header text."""
    model = mp._model
    index = model.index(source_row, model.column_index("amplitude"))
    return {
        "foreground": model.data(index, Qt.ForegroundRole),
        "tooltip": model.data(index, Qt.ToolTipRole),
        "row_header": str(model.headerData(source_row, Qt.Vertical, Qt.DisplayRole)),
    }


# ================================================================================================
# the stamp is saved
# ================================================================================================

def test_saved_row_stamps_units_and_calibration(window, tmp_path):
    doc_id, ch = _install(window, _doc(tmp_path))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    assert mp.add_current() is True

    row = mp.measurements_df.iloc[-1]
    assert str(row["units"]) == "pA"
    assert float(row["cal_gain"]) == 1.0
    assert float(row["cal_offset"]) == 0.0


def test_stamp_follows_the_calibration_in_force(window, tmp_path):
    doc_id, ch = _install(window, _doc(tmp_path))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    _apply_cal(window, doc_id, ch, gain=0.001, units="nA")
    assert mp.add_current() is True

    row = mp.measurements_df.iloc[-1]
    assert str(row["units"]) == "nA"
    assert float(row["cal_gain"]) == 0.001
    assert float(row["amplitude"]) == pytest.approx(0.03)


# ================================================================================================
# rows under a superseded calibration are marked; the numbers are never rescaled
# ================================================================================================

def test_calibration_edit_marks_earlier_rows_and_not_the_fresh_one(window, tmp_path):
    """The dossier scenario: the same span saved under gain 1, gain 2 and (0.001, nA).
    After the last edit the first two rows are visibly stale — amber foreground, a ⚠ on
    the row header, a tooltip that says to re-measure — and the third is unmarked. The
    saved amplitudes stay exactly what was measured (30 / 60 / 0.03): records, not
    rescaled."""
    doc_id, ch = _install(window, _doc(tmp_path))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)

    mp.add_current()                                      # row 0: gain 1, pA -> 30
    _apply_cal(window, doc_id, ch, gain=2.0)
    mp.add_current()                                      # row 1: gain 2, pA -> 60
    _apply_cal(window, doc_id, ch, gain=0.001, units="nA")
    mp.add_current()                                      # row 2: 0.001, nA -> 0.03

    amps = [float(a) for a in mp.measurements_df["amplitude"]]
    assert amps == pytest.approx([30.0, 60.0, 0.03])      # never silently rescaled

    stale0, stale1, fresh = (_row_marked(mp, r) for r in (0, 1, 2))
    for marked in (stale0, stale1):
        assert marked["foreground"] is not None, "stale row has no visible state"
        assert "⚠" in marked["row_header"], "stale row lacks the non-colour marker"
        assert "re-measure" in str(marked["tooltip"]).lower(), \
            "the marking does not say what to do"
    assert fresh["foreground"] is None
    assert "⚠" not in fresh["row_header"]
    assert fresh["tooltip"] in (None, "")

    # ...and the same stale rows through the VIEW's proxy (what is actually on screen).
    proxy = mp._proxy
    headers = [str(proxy.headerData(r, Qt.Vertical, Qt.DisplayRole))
               for r in range(proxy.rowCount())]
    assert sum("⚠" in h for h in headers) == 2


def test_reverting_the_calibration_clears_the_marks(window, tmp_path):
    doc_id, ch = _install(window, _doc(tmp_path))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    mp.add_current()
    _apply_cal(window, doc_id, ch, gain=2.0)
    assert 0 in mp._model.stale_ids

    _apply_cal(window, doc_id, ch, gain=1.0, units="pA")
    assert mp._model.stale_ids == frozenset(), \
        "the calibration matches the stamp again; a standing mark would be stale state"


def test_calibration_edit_on_the_measured_doc_marks_rows_even_from_another_target(
        window, tmp_path):
    """Staleness is judged per ROW against the row's own document, not against the
    current target: an edit on recording A must mark A's saved rows while B is
    selected."""
    doc_a = _doc(tmp_path, name="a.dat")
    doc_b = _doc(tmp_path, name="b.dat")
    a_id, ch = _install(window, doc_a)
    mp = _enable_measure(window, a_id, *_BLOCKADE)
    mp.add_current()                                       # measured on A under gain 1

    b_id = window._session.add_document(doc_b)
    window._create_default_layers(doc_b)
    window._on_selection(b_id, 0)                          # now working on B

    _apply_cal(window, a_id, ch, gain=2.0)                 # A's calibration moves
    assert 0 in mp._model.stale_ids, \
        "a calibration edit on a non-current document left its saved rows unmarked"


def test_legacy_rows_without_a_stamp_are_unknown_not_stale(window, tmp_path):
    """A project saved before the stamp columns existed restores with NaN stamps —
    those rows CANNOT be compared, and per the fingerprint rule they read as unknown:
    no stale mark (which would claim knowledge nobody has), no silent 'matches'."""
    doc_id, ch = _install(window, _doc(tmp_path))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    mp.add_current()
    legacy = mp.measurements_df.drop(columns=["units", "cal_gain", "cal_offset", "t0_s"])
    mp.set_measurements_df(legacy)                         # the project-restore door

    assert np.isnan(float(mp.measurements_df.iloc[0]["cal_gain"]))
    assert mp._model.stale_ids == frozenset()
    marked = _row_marked(mp, 0)
    assert marked["foreground"] is None and "⚠" not in marked["row_header"]


# ================================================================================================
# the CSV export discloses the stamps
# ================================================================================================

def test_export_csv_carries_the_stamp_columns(window, tmp_path):
    doc_id, ch = _install(window, _doc(tmp_path))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    mp.add_current()
    _apply_cal(window, doc_id, ch, gain=0.001, units="nA")
    mp.add_current()

    out = mp.export_csv(str(tmp_path / "mixed.csv"))
    assert out is not None
    df = pd.read_csv(out)
    for col in ("units", "cal_gain", "cal_offset", "t0_s"):
        assert col in df.columns, col
    # The two rows are distinguishable downstream — the whole point.
    assert list(df["units"]) == ["pA", "nA"]
    assert list(df["cal_gain"]) == pytest.approx([1.0, 0.001])
