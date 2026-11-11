"""Annotate mode: one drag across an event adds it, one double-click adds it, one Edit ▸ Undo step
each (docs/CURSOR_ERGONOMICS_DESIGN.md §4).

Driven through a real, shown MainWindow with real mouse events sent to the trace's viewport, so
pyqtgraph's own scene delivers the drag (the helpers are ``trace_rig``'s). Synthetic
recording only: a ``TraceDocument`` whose path does not exist.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import json
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.trace_view import annotate_cursor

from .trace_rig import (  # noqa: F401  (free_mouse is a fixture used by name)
    Qt, drag, drag_move, free_mouse, hover, point_at, press, release, send_mouse)

FS = 50_000.0
N = 120_000
I0 = -1000.0
DEPTH = 300.0
EVENT = (20_000, 20_400)
SECOND = (60_000, 60_250)
#: Two events 60 samples apart: one drag over both fits as one wide span.
PAIR = ((90_000, 90_200), (90_260, 90_460))


#: Short events with a sub-level: 110 samples, 300 pA deep, whose last (or first) 33 samples
#: are a 100 pA shoulder; and a 400-sample event with an unannotated 60-sample neighbour 45
#: samples before it. Only the tests that ask for them (``levels=``) have them.
END_SHOULDER = (100_000, 100_110)
START_SHOULDER = (105_000, 105_110)
LONE = (110_000, 110_400)
SHOULDERS = ((100_000, 100_110, 100.0), (100_000, 100_077, 200.0),
             (105_000, 105_110, 100.0), (105_033, 105_110, 200.0),
             (110_000, 110_400, DEPTH), (109_895, 109_955, DEPTH))


def _signal(seed=0, levels=()) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in (EVENT, SECOND, *PAIR):
        x[s:e] += DEPTH
    for s, e, depth in levels:
        x[s:e] += depth
    return x.astype(np.float32)


def build_window(qtbot, tmp_path, seed: int = 0, levels=()):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "annotate.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(seed, levels),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    # The offscreen screen is 800x800; the trace has to be inside it for the pointer helpers.
    win.resize(800, 760)
    win.show()
    QtWidgets.QApplication.processEvents()
    return win, doc_id


@pytest.fixture
def rig(qapp, qtbot, tmp_path, free_mouse):
    win, doc_id = build_window(qtbot, tmp_path)
    yield win, doc_id
    win.hide()


def table(win, doc_id):
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return None if res is None else res.payload.df


def rows(win, doc_id) -> int:
    df = table(win, doc_id)
    return 0 if df is None else len(df)


def target_vb(win, doc_id):
    return win._trace_view.pane_for_layer(f"{doc_id}#0").plot.vb


def zoom(win, doc_id, lo: int, hi: int) -> None:
    tv = win._trace_view
    tv.zoom_x_to(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, hi))
    QtWidgets.QApplication.processEvents()


def at(win, doc_id, sample: int, yfrac: float = 0.5) -> QtCore.QPointF:
    tv = win._trace_view
    return point_at(tv, target_vb(win, doc_id), tv.x_for_index(doc_id, 0, sample), yfrac)


def armed(win, doc_id, lo: int, hi: int) -> None:
    """Zoomed to [lo, hi) and Annotate on, through the Scan strip's button."""
    zoom(win, doc_id, lo, hi)
    QTest.mouseClick(win._annotate_button, Qt.LeftButton)
    QtWidgets.QApplication.processEvents()
    assert win._trace_keys.annotate() and win._act_annotate.isChecked()


def settle(qtbot, ms: int = 30) -> None:
    qtbot.wait(ms)


# ================================================================================================
# one drag, one event, one undo step
# ================================================================================================
def test_one_drag_adds_one_fitted_event_and_one_undo_step(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 19_000, 21_500)
    view_before = tuple(tv.anchor_plot().vb.viewRange()[0])

    # Drawn loosely, about 60 samples outside each step.
    drag(tv, at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70))
    settle(qtbot)

    df = table(win, doc_id)
    assert df is not None and len(df) == 1
    row = df.iloc[0]
    assert abs(int(row["start_idx"]) - EVENT[0]) <= 3 and abs(int(row["end_idx"]) - EVENT[1]) <= 3
    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    assert "Added event" in win.statusBar().currentMessage()
    assert tv.measure_empty()                                    # the next drag starts clean
    assert tuple(tv.anchor_plot().vb.viewRange()[0]) == pytest.approx(view_before)   # no zoom

    win._act_undo.trigger()                                      # ONE step takes it out
    settle(qtbot)
    assert rows(win, doc_id) == 0
    win._act_redo.trigger()
    settle(qtbot)
    assert rows(win, doc_id) == 1


@pytest.mark.parametrize("seed, margin", [(0, 60), (2, 60), (2, 64)])
def test_a_loose_annotate_drag_fits_the_steps(qapp, qtbot, tmp_path, free_mouse, seed, margin):
    """Zoomed so the event fills a third of the pane, a drag overshooting each step by more than
    the 12-pixel search window: seed 0 refused a 37-sigma step as "not significant against the
    noise", and seed 2 added a row starting 35 samples early. A drawn span is searched across
    0.45 of its own width, so it lands on the steps."""
    win, doc_id = build_window(qtbot, tmp_path, seed)
    try:
        tv = win._trace_view
        armed(win, doc_id, 19_600, 20_800)
        assert margin / tv.target_px_samples() > 12          # beyond 12 px of this zoom
        drag(tv, at(win, doc_id, EVENT[0] - margin), at(win, doc_id, EVENT[1] + margin))
        settle(qtbot)
        df = table(win, doc_id)
        assert df is not None and len(df) == 1, win.statusBar().currentMessage()
        row = df.iloc[0]
        assert abs(int(row["start_idx"]) - EVENT[0]) <= 2
        assert abs(int(row["end_idx"]) - EVENT[1]) <= 2
        assert row["bounds_source"] == "start=fit:step;end=fit:step"
        assert json.loads(row["fit_json"])["drawn"] is True
    finally:
        win.hide()


@pytest.mark.parametrize("event", [END_SHOULDER, START_SHOULDER])
@pytest.mark.parametrize("margin", [2, 6])
def test_a_tight_drag_on_a_short_event_with_a_shoulder_adds_it(
        qapp, qtbot, tmp_path, free_mouse, event, margin):
    """On a 110-sample event the drawn window (0.45 x span) is shorter than the samples rule 4
    reads past it, so the inner-edge guard saw the event's own deeper level there as a step
    "further in" and refused a split sitting on the real step: a drag drawn on the steps was
    told to draw closer. The split's own step is now read first."""
    win, doc_id = build_window(qtbot, tmp_path, levels=SHOULDERS)
    try:
        armed(win, doc_id, event[0] - 200, event[0] + 400)
        drag(win._trace_view, at(win, doc_id, event[0] - margin), at(win, doc_id, event[1] + margin))
        settle(qtbot)
        df = table(win, doc_id)
        assert df is not None and len(df) == 1, win.statusBar().currentMessage()
        assert abs(int(df.iloc[0]["start_idx"]) - event[0]) <= 2
        assert abs(int(df.iloc[0]["end_idx"]) - event[1]) <= 2
    finally:
        win.hide()


def test_a_drag_beside_an_unannotated_neighbour_adds_only_the_drawn_event(
        qapp, qtbot, tmp_path, free_mouse):
    """A drawn window reaches 0.45 of the span outward too, so a 60-sample neighbour 45 samples
    before the drawn event was fitted into it, and the add was refused as "more than one
    event". When the drawn span holds one event, the edges are fitted as a pointed span's."""
    win, doc_id = build_window(qtbot, tmp_path, levels=SHOULDERS)
    try:
        armed(win, doc_id, LONE[0] - 400, LONE[0] + 800)
        drag(win._trace_view, at(win, doc_id, LONE[0] - 5), at(win, doc_id, LONE[1] + 5))
        settle(qtbot)
        df = table(win, doc_id)
        assert df is not None and len(df) == 1, win.statusBar().currentMessage()
        assert abs(int(df.iloc[0]["start_idx"]) - LONE[0]) <= 2
        assert abs(int(df.iloc[0]["end_idx"]) - LONE[1]) <= 2
    finally:
        win.hide()


#: A 400-sample event with an unannotated 100-sample neighbour 40 samples before (or after) it.
CLOSE = (110_000, 110_400)
CLOSE_BEFORE = ((110_000, 110_400, DEPTH), (109_860, 109_960, DEPTH))
CLOSE_AFTER = ((110_000, 110_400, DEPTH), (110_440, 110_540, DEPTH))


@pytest.mark.parametrize("width", [4_000, 8_000])
@pytest.mark.parametrize("levels", [CLOSE_BEFORE, CLOSE_AFTER], ids=["before", "after"])
def test_a_drag_beside_a_close_neighbour_at_a_coarse_zoom_adds_only_the_drawn_event(
        qapp, qtbot, tmp_path, free_mouse, levels, width):
    """trace2-1. At 12 and 24 samples/px the pointed refit's window (12 px) still reached a
    neighbour 40 samples off, so the fit took its step and the add was refused as "more than
    one event", though the drag held one. The refit now stops at the baseline gap before the
    neighbour (``fit_span_edges(bounds=)``)."""
    win, doc_id = build_window(qtbot, tmp_path, levels=levels)
    try:
        armed(win, doc_id, CLOSE[0] - width // 3, CLOSE[0] - width // 3 + width)
        drag(win._trace_view, at(win, doc_id, CLOSE[0] - 10), at(win, doc_id, CLOSE[1] + 10))
        settle(qtbot)
        df = table(win, doc_id)
        assert df is not None and len(df) == 1, win.statusBar().currentMessage()
        assert abs(int(df.iloc[0]["start_idx"]) - CLOSE[0]) <= 2
        assert abs(int(df.iloc[0]["end_idx"]) - CLOSE[1]) <= 2
    finally:
        win.hide()


#: Near the recording start: a 400-sample event with an unannotated 100-sample neighbour 40
#: samples before it. On seed 0's noise a pointed far edge drawn ~300 samples past the event
#: fits a step that is not there.
EARLY = (300, 700)
EARLY_BEFORE = ((300, 700, DEPTH), (160, 260, DEPTH))


def test_a_loose_far_edge_beside_a_close_neighbour_adds_the_event_at_its_own_steps(
        qapp, qtbot, tmp_path, free_mouse):
    """trace2-1, the refit beside the neighbour. At 24 samples/px a drag from just before the
    event to ~300 samples past it was refitted pointed, bounded at the gap before the
    neighbour: the far edge took a step 29 samples past the event's end, and the event was
    added as [300, 729) with both edges "fit:step" (build 5 refused it). The bounded refit is
    now drawn first, and a refit is kept only when it holds one event."""
    win, doc_id = build_window(qtbot, tmp_path, levels=EARLY_BEFORE)
    try:
        armed(win, doc_id, 0, 8_000)
        drag(win._trace_view, at(win, doc_id, EARLY[0] - 10), at(win, doc_id, EARLY[1] + 300))
        settle(qtbot)
        df = table(win, doc_id)
        assert df is not None and len(df) == 1, win.statusBar().currentMessage()
        assert abs(int(df.iloc[0]["start_idx"]) - EARLY[0]) <= 2
        assert abs(int(df.iloc[0]["end_idx"]) - EARLY[1]) <= 2
    finally:
        win.hide()


def test_a_drag_far_out_on_one_side_says_to_draw_closer(rig, qtbot):
    """The start drawn 1,000 samples before the event: its window is open pore only. The span
    holds the step, so the sentence says the edge was drawn too far out, not that the step is
    lost in the noise, and points at drawing closer or C instead of at S (which finds no step
    there either)."""
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 18_500, 21_000)
    drag(tv, at(win, doc_id, EVENT[0] - 1_000), at(win, doc_id, EVENT[1] + 20))
    settle(qtbot)
    assert rows(win, doc_id) == 0
    message = win.statusBar().currentMessage()
    assert message.startswith("The start edge is not fitted (no step near the drawn edge)")
    assert message.endswith("Draw closer, press C over the event, or press D twice to add it "
                            "as placed.")
    assert tv.measure_bounds() is not None                       # kept, as every refusal is


def test_two_drags_are_two_rows_and_two_undo_steps(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 19_000, 21_500)
    drag(tv, at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70))
    settle(qtbot)
    zoom(win, doc_id, 59_000, 61_500)
    drag(tv, at(win, doc_id, SECOND[0] - 60), at(win, doc_id, SECOND[1] + 70))
    settle(qtbot)
    assert rows(win, doc_id) == 2
    win._act_undo.trigger()
    settle(qtbot)
    assert rows(win, doc_id) == 1
    win._act_undo.trigger()
    settle(qtbot)
    assert rows(win, doc_id) == 0


def test_a_drag_over_baseline_adds_nothing_and_keeps_the_span(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 39_000, 41_500)
    drag(tv, at(win, doc_id, 40_000), at(win, doc_id, 40_500))
    settle(qtbot)
    assert rows(win, doc_id) == 0
    assert tv.measure_bounds() is not None                       # kept, to fix or add as placed
    assert "not fitted" in win.statusBar().currentMessage()


def test_shift_at_release_keeps_the_span_and_d_adds_it(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 19_000, 21_500)
    p0, p1 = at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70)
    drag(tv, p0, p1, release_it=False)
    release(tv, p1, modifiers=Qt.ShiftModifier)
    settle(qtbot)
    assert rows(win, doc_id) == 0
    assert "Kept for adjusting" in win.statusBar().currentMessage()
    lo, hi = tv.measure_bounds()
    assert abs(lo - EVENT[0]) <= 3 and abs(hi - EVENT[1]) <= 3   # fitted, waiting
    win._trace_keys.add()
    assert rows(win, doc_id) == 1


def test_double_click_in_annotate_mode_catches_and_adds(rig, qtbot):
    win, doc_id = rig
    armed(win, doc_id, 59_000, 61_500)
    pos = at(win, doc_id, (SECOND[0] + SECOND[1]) // 2)
    hover(win._trace_view, pos)
    press(win._trace_view, pos)
    release(win._trace_view, pos)
    send_mouse(win._trace_view, QtCore.QEvent.MouseButtonDblClick, pos, Qt.LeftButton,
               Qt.LeftButton)
    release(win._trace_view, pos)
    settle(qtbot)
    df = table(win, doc_id)
    assert df is not None and len(df) == 1
    assert abs(int(df.iloc[0]["start_idx"]) - SECOND[0]) <= 3
    assert abs(int(df.iloc[0]["end_idx"]) - SECOND[1]) <= 3
    win._act_undo.trigger()
    settle(qtbot)
    assert rows(win, doc_id) == 0


def test_option_drag_box_zooms_in_annotate_mode(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 19_000, 21_500)
    lo0, hi0 = tv.anchor_plot().vb.viewRange()[0]
    drag(tv, at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70),
         modifiers=Qt.AltModifier)
    settle(qtbot)
    lo1, hi1 = tv.anchor_plot().vb.viewRange()[0]
    assert (hi1 - lo1) < 0.5 * (hi0 - lo0)
    assert rows(win, doc_id) == 0


def test_esc_mid_drag_cancels_and_the_release_adds_nothing(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 19_000, 21_500)
    p0, p1 = at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70)
    drag(tv, p0, p1, release_it=False)
    win._trace_keys.escape()
    release(tv, p1)
    settle(qtbot)
    assert rows(win, doc_id) == 0


def test_the_add_runs_after_the_release_returns(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 19_000, 21_500)
    p0, p1 = at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70)
    drag(tv, p0, p1, release_it=False)
    # The release alone, with no event processing after it: the add must not have run inside.
    ev = QtCore.QPointF(p1)
    from PySide6 import QtGui
    QtWidgets.QApplication.sendEvent(
        tv.viewport(), QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, ev,
                                         QtCore.QPointF(tv.viewport().mapToGlobal(ev.toPoint())),
                                         Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    assert rows(win, doc_id) == 0
    settle(qtbot)
    assert rows(win, doc_id) == 1


def test_a_drag_across_two_close_events_adds_nothing(rig, qtbot):
    win, doc_id = rig
    tv = win._trace_view
    armed(win, doc_id, 89_500, 91_000)
    # Today's fit takes the first event's start and the last one's finish for this drag, both
    # "fitted" (probed over gaps of 20-200 samples): only the one-event check stops the add.
    drag(tv, at(win, doc_id, PAIR[0][0] - 20), at(win, doc_id, PAIR[1][1] + 20))
    settle(qtbot)
    lo, hi = tv.measure_bounds()
    assert abs(lo - PAIR[0][0]) <= 3 and abs(hi - PAIR[1][1]) <= 3
    assert rows(win, doc_id) == 0
    assert "more than one event" in win.statusBar().currentMessage()
    assert tv.measure_bounds() is not None                       # D still adds it as one
    win._trace_keys.add()
    assert rows(win, doc_id) == 1


def test_a_drag_across_one_event_passes_the_one_event_check(rig):
    """The check's other side: a single event is never called two."""
    win, doc_id = rig
    keys = win._trace_keys
    for s, e in (EVENT, SECOND, PAIR[0], PAIR[1]):
        assert not keys._holds_more_than_one_event((doc_id, 0), s, e)
    assert keys._holds_more_than_one_event((doc_id, 0), PAIR[0][0], PAIR[1][1])


# ================================================================================================
# the doors, the label, the pointer
# ================================================================================================
def test_four_doors_toggle_one_state(rig, qtbot):
    win, doc_id = rig
    strip = win._annotate_button
    tab = win._analysis_panel.measure_panel.annotate_button()
    assert strip.defaultAction() is win._act_annotate is tab.defaultAction()
    assert tab.isVisible() or not tab.isHidden()
    QTest.mouseClick(strip, Qt.LeftButton)                       # the Scan strip
    assert win._trace_keys.annotate() and tab.isChecked() and win._act_annotate.isChecked()
    tab.click()                                                  # the Measure tab
    assert not win._trace_keys.annotate() and not strip.isChecked()
    win._act_annotate.trigger()                                  # View ▸ Annotate (⌘⇧E)
    assert win._trace_keys.annotate() and strip.isChecked() and tab.isChecked()
    win._trace_keys.toggle_annotate()                            # E on the trace
    assert not win._act_annotate.isChecked() and not strip.isChecked()
    win._trace_keys.toggle_annotate()
    assert strip.isChecked()
    win._analysis_panel.measure_panel.set_active(False)          # Measure off drops the mode
    assert not win._trace_keys.annotate()
    assert not (strip.isChecked() or tab.isChecked() or win._act_annotate.isChecked())
    assert win._act_annotate.shortcut().toString() == "Ctrl+Shift+E"
    assert win._act_annotate.text().replace("&", "") == "Annotate: drag adds an event"


def test_annotate_mode_is_off_in_a_new_window(rig, qtbot, tmp_path):
    win, _doc_id = rig
    win._trace_keys.set_annotate(True)
    assert win._trace_keys.annotate()
    second, _ = build_window(qtbot, tmp_path / "second")
    try:
        assert not second._trace_keys.annotate() and not second._act_annotate.isChecked()
        assert not second._trace_view.draw_mode()
    finally:
        second.hide()


def test_the_label_says_annotate_while_the_mode_is_on(rig, qtbot):
    win, _ = rig
    win._trace_keys.set_annotate(True)
    qtbot.wait(200)
    assert "ANNOTATE: drag adds" in win._trace_view.target_label_text()
    win._trace_keys.set_annotate(False)
    qtbot.wait(200)
    assert "ANNOTATE" not in win._trace_view.target_label_text()


def test_annotate_pointer_differs_from_the_keys_cross(rig):
    win, doc_id = rig
    tv = win._trace_view
    vb = target_vb(win, doc_id)
    tv.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
    tv._apply_target_cursor()
    assert vb.cursor().shape() in (Qt.CrossCursor, Qt.ArrowCursor)
    win._trace_keys.set_annotate(True)
    assert vb.cursor().shape() == Qt.BitmapCursor
    assert vb.cursor().pixmap().cacheKey() == annotate_cursor().pixmap().cacheKey()
    win._trace_keys.set_annotate(False)
    assert vb.cursor().shape() != Qt.BitmapCursor
