"""Regression tests for MEAS3-01 and MEAS3-03: the manual-measurement table's two
provenance stories say one thing each, and a mark it can no longer verify does not
simply disappear.

MEAS3-01 — ONE TOOLTIP, TWO CONVERSIONS. The Files-dock Δt spin
(``Session.set_layer_time_offset``) shifts a layer along the time axis to align it with
another recording; the trace is then drawn at ``t₀ + nudge``. The ``t_start_s`` header
tooltip said, in one breath, *"add t₀ = 66.3034 s to a number here to find it on the
trace"* and *"t_start_s + t0_s is that row's start on the trace axis"* — while the row's
``t0_s`` held 61.3034. Both sentences claimed to land on the trace axis and they landed
5 s apart, which is the arithmetic the user is asked to do by hand. The fix names the
clock each sentence lands on (the drawn axis vs the INSTRUMENT clock, which is what a
saved stamp records) and states the nudge that separates them.

MEAS3-03 — A MARK THAT STOPPED BEING CHECKABLE. ``_refresh_stale_rows`` classifies a row
whose recording is not open as *unknown*, correctly and per the panel's own docstring —
but the model rendered only two states, so "nobody can check this" was drawn exactly
like "checked, and current". Closing a recording therefore erased a ⚠ the user had been
shown a second earlier, from rows whose numbers had not changed at all.

Driven through a real MainWindow, its real MeasurePanel and the real Session doors the
Files dock and the Calibration panel use.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 200_000
_BLOCKADE = (40_000, 60_000)
T0_APO = 61.3034                           # the owner's real −600 mV ABF t₀
NUDGE = 5.0                                # a plain, checkable alignment shift


def _doc(tmp_path, *, name: str = "apo.dat", t0_s: float = T0_APO) -> TraceDocument:
    x = np.full(N, 100.0, dtype=np.float32)
    x[_BLOCKADE[0]:_BLOCKADE[1]] = 70.0
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         t0_s=t0_s, channels=(info,))
    return TraceDocument(meta=meta, channels=(x,))


def _install(window: MainWindow, doc: TraceDocument) -> str:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id


def _measure_and_save(window: MainWindow, doc_id: str):
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[_BLOCKADE[0]]),
                                          float(vals[_BLOCKADE[1]]))
    assert mp.add_current() is True
    return mp


def _nudge(window: MainWindow, doc_id: str, dt_s: float) -> None:
    """The door the Files-dock Δt spin turns (file_dock._on_offset_changed)."""
    window._session.set_layer_time_offset(f"{doc_id}#0", dt_s)
    window._analysis_panel.measure_panel._sync_target_scope()


def _header(mp, name: str, role):
    model = mp._model
    return str(model.headerData(model.column_index(name), Qt.Horizontal, role))


def _row_state(mp, row: int = 0) -> dict:
    model = mp._model
    idx = model.index(row, model.column_index("amplitude"))
    return {
        "header": str(model.headerData(row, Qt.Vertical, Qt.DisplayRole)),
        "coloured": model.data(idx, Qt.ForegroundRole) is not None,
        "tooltip": str(model.data(idx, Qt.ToolTipRole) or ""),
        "stale_ids": sorted(model.stale_ids),
    }


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# MEAS3-01: the two clock statements name two clocks
# ================================================================================================

def test_a_nudged_layer_does_not_make_one_tooltip_give_two_conversions(window, tmp_path):
    """The tooltip's two halves are about two different clocks now, and each says which:
    the drawn axis (t₀ + nudge) and the instrument clock (what a saved row stamps). The
    failure mode was that both claimed the trace axis while differing by the nudge."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)
    _nudge(window, doc_id, NUDGE)

    tip = _header(mp, "t_start_s", Qt.ToolTipRole)
    row = mp.measurements_df.iloc[0]

    # the drawn-axis half still names the offset the curve is drawn through...
    assert f"{T0_APO + NUDGE:.4f}" in tip
    # ...and the pooled half no longer claims t_start_s + t0_s reaches that axis.
    pooled = tip.split("\n\n")[1]
    assert "t0_s" in pooled
    assert "trace axis" not in pooled, \
        f"the pooled sentence still promises the drawn axis: {pooled!r}"
    assert "INSTRUMENT CLOCK" in pooled.upper()
    # the arithmetic the pooled sentence promises is true:
    assert float(row["t_start_s"]) + float(row["t0_s"]) == pytest.approx(
        _BLOCKADE[0] / FS + T0_APO)


def test_the_nudge_is_named_rather_than_left_as_the_difference(window, tmp_path):
    """A number that is on screen, is persisted in the project and is exactly the gap
    between the two statements does not get to be implicit."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)

    assert "alignment" not in _header(mp, "t_start_s", Qt.ToolTipRole).lower(), \
        "an un-nudged layer must not carry the caveat"

    _nudge(window, doc_id, NUDGE)
    tip = _header(mp, "t_start_s", Qt.ToolTipRole)
    assert "+5.0000" in tip
    assert "alignment" in tip.lower()

    _nudge(window, doc_id, -2.5)
    assert "-2.5000" in _header(mp, "t_start_s", Qt.ToolTipRole)


def test_the_t0_column_tooltip_promises_the_clock_the_stamp_is_actually_on(
        window, tmp_path):
    """The stamp deliberately excludes the nudge (a row records what was measured, and
    the nudge is a display alignment). The tooltip now says which clock that lands on,
    so following it is not wrong by the nudge."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)
    _nudge(window, doc_id, NUDGE)

    tip = _header(mp, "t0_s", Qt.ToolTipRole)
    assert "t_start_s + t0_s" in tip
    assert "INSTRUMENT CLOCK" in tip.upper()
    assert "nudge" in tip.lower() or "display shift" in tip.lower()
    assert float(mp.measurements_df.iloc[0]["t0_s"]) == pytest.approx(T0_APO), \
        "the stamp itself must not move with a display alignment"


def test_a_nudge_that_exactly_cancels_t0_does_not_produce_an_agreement_claim(
        window, tmp_path):
    """The worst variant: nudge = -t₀ makes the drawn offset 0, so the header used to
    lose its badge and assert the clocks agreed — while the row's t0_s was off by the
    whole t₀."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)
    _nudge(window, doc_id, -T0_APO)

    tip = _header(mp, "t_start_s", Qt.ToolTipRole)
    assert f"{-T0_APO:+.4f}" in tip, \
        "the shift that made the two clocks coincide on screen is not named"
    assert "alignment" in tip.lower()


# ================================================================================================
# MEAS3-03: three states, because there are three answers
# ================================================================================================

def _apply_gain(window: MainWindow, doc_id: str, gain: float) -> None:
    window._session.set_calibration(
        doc_id, 0, ChannelCalibration(gain=gain, offset=0.0, target_units="pA",
                                      provenance=Provenance.MEASURED),
        write_sidecar=False)


def test_closing_the_recording_keeps_the_superseded_mark_and_says_why(window, tmp_path):
    """The row's numbers did not change when its document closed, so neither does what
    was true about them. What DID change is that the app can no longer re-check it, and
    that is what the tooltip now says instead of the marking vanishing."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)
    _apply_gain(window, doc_id, 2.0)

    marked = _row_state(mp)
    assert marked["stale_ids"] == [0] and "⚠" in marked["header"] and marked["coloured"]

    window._session.close_document(doc_id)
    after = _row_state(mp)

    assert "⚠" in after["header"], "closing the recording erased a warning it was shown"
    assert after["coloured"]
    assert after["tooltip"] != marked["tooltip"], \
        "the tooltip must not still claim a live comparison"
    assert "no longer open" in after["tooltip"] or "cannot be repeated" in after["tooltip"]
    # the row itself is untouched — the mark describes it, it does not alter it
    row = mp.measurements_df.iloc[0]
    assert float(row["cal_gain"]) == 1.0
    assert float(row["amplitude"]) == pytest.approx(30.0)


def test_a_row_nobody_can_check_is_not_drawn_like_a_row_that_was_checked(window, tmp_path):
    """The third state. A row whose recording is closed and which was never found stale
    is UNKNOWN — it is not claiming to match, and it does not wear the alarm either."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)

    checked = _row_state(mp)
    assert checked["header"].strip() == "1" and not checked["coloured"]
    assert checked["tooltip"] == ""

    window._session.close_document(doc_id)
    unknown = _row_state(mp)

    assert unknown["header"] != checked["header"], \
        "'cannot be judged' renders identically to 'checked and current'"
    assert "?" in unknown["header"]
    assert not unknown["coloured"], "an absent verdict is not an alarm"
    assert "not open" in unknown["tooltip"]
    assert "NOT known to match" in unknown["tooltip"]
    assert unknown["stale_ids"] == [], "unknown must never be presented as stale"


def test_a_legacy_row_without_a_stamp_reads_as_unknown_on_the_surface_too(
        window, tmp_path):
    """The panel's docstring has always said such rows "read as unknown, not as matching
    and not as stale". That was true of the classification and false of the rendering."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)
    legacy = mp.measurements_df.drop(columns=["units", "cal_gain", "cal_offset", "t0_s"])
    mp.set_measurements_df(legacy)                     # the project-restore door

    state = _row_state(mp)
    assert "?" in state["header"] and "⚠" not in state["header"]
    assert not state["coloured"]
    assert "before the stamp columns existed" in state["tooltip"]
    assert state["stale_ids"] == []


def test_reopening_the_recording_restores_a_live_verdict(window, tmp_path):
    """The state is re-derived, never latched: a row that becomes checkable again gets
    the answer the live calibration gives, including "current" after a revert."""
    doc_id = _install(window, _doc(tmp_path))
    mp = _measure_and_save(window, doc_id)
    _apply_gain(window, doc_id, 2.0)
    window._session.close_document(doc_id)
    assert "⚠" in _row_state(mp)["header"]

    doc_id = _install(window, _doc(tmp_path))          # same path, same doc_key
    _apply_gain(window, doc_id, 1.0)                   # back to what the row was measured under
    mp._refresh_stale_rows()

    state = _row_state(mp)
    assert state["header"].strip() == "1", f"a live, matching row still marked: {state}"
    assert not state["coloured"] and state["tooltip"] == ""
