"""Draw mode on the trace (``TraceView.set_draw_mode`` and the drag/forward/preview/settle path).

What a plain left-drag does is the whole point: box-zoom today, a drawn Measure span in draw mode,
and Option inverts it in both states. The mask's body forwards its drags, its two edge handles keep
resizing, another pane does nothing and says so, a right-click or a rebuild cancels a draw (also
one started on the mask body), a 20-frame draw costs the Measure panel at most two recomputes, and
the mode survives rebuilds and view-mode changes while the target pane exists.

Everything runs on a synthetic recording (see ``test_trace_rig``) with real mouse events sent to
the trace's viewport, so pyqtgraph's own scene delivers the drags.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.layer import ViewMode
from nanoscope.gui import measure_panel as measure_panel_module

from .test_trace_rig import (  # noqa: F401  (fixtures are used by name)
    CURRENT, FS, N, Qt, Recorder, build_rig, double_click, drag, drag_move, free_mouse, hover,
    point_at, press, release, rig, send_mouse, zoom_middle)


def _armed(rig, *, draw: bool = True, mask: tuple[float, float] | None = None) -> None:
    """Measure on with no mask (``seed=False``), optionally a mask at ``mask`` (data x), and draw
    mode on."""
    rig.panel.set_active(True, seed=False)
    if mask is not None:
        rig.view.set_measure_region(*mask)
    if draw:
        assert rig.view.set_draw_mode(True)
    QtWidgets.QApplication.processEvents()


def _recorders(view):
    return dict(started=Recorder(view.sigDrawStarted), released=Recorder(view.sigDrawReleased),
                cancelled=Recorder(view.sigDrawCancelled), refused=Recorder(view.sigDrawRefused),
                cleared=Recorder(view.sigMeasureCleared), mode=Recorder(view.sigDrawModeChanged))


def _sample_of(rig, x: float) -> int:
    xs = rig.xs()
    return int(np.clip(np.searchsorted(xs, x - 0.5 / FS), 0, xs.size))


def _x_span(rig):
    lo, hi = rig.x_range()
    return lo, hi


# ================================================================================================
# the gesture: draw on empty trace, on the mask body; edges keep resizing
# ================================================================================================
def test_a_drag_on_empty_trace_draws_a_span_in_draw_mode(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    rec = _recorders(rig.view)
    before = rig.x_range()

    xa, xb = c - 0.20, c + 0.10
    drag(rig.view, point_at(rig.view, rig.target_vb, xa), point_at(rig.view, rig.target_vb, xb))

    assert len(rec["started"]) == 1 and len(rec["released"]) == 1 and len(rec["cancelled"]) == 0
    doc_id, channel, lo, hi, px = rec["released"].calls[0]
    assert (doc_id, channel) == rig.view.measure_target() == (rig.doc_id, CURRENT)
    assert isinstance(lo, int) and isinstance(hi, int) and isinstance(px, float)
    # The span runs from where the pointer was PRESSED to where it was released (within a pixel:
    # 14 samples per pixel at this zoom), not from the first frame the scene called a drag.
    assert abs(lo - _sample_of(rig, xa)) <= 20 and abs(hi - _sample_of(rig, xb)) <= 20
    assert rig.view.measure_bounds() == (lo, hi)
    assert 0 <= lo < hi <= N
    # ...at the zoom in force: 1 s over ~700 px at 10 kHz is about 14 samples per pixel.
    assert px == pytest.approx(rig.view.target_px_samples())
    assert 5.0 < px < 25.0
    # No box zoom happened.
    assert rig.x_range() == pytest.approx(before, abs=1e-9)
    assert rig.view.measure_empty() is False


def test_a_plain_drag_still_box_zooms_when_draw_mode_is_off(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    rec = _recorders(rig.view)
    xa, xb = c - 0.10, c + 0.10
    drag(rig.view, point_at(rig.view, rig.target_vb, xa), point_at(rig.view, rig.target_vb, xb))
    lo, hi = rig.x_range()
    assert lo == pytest.approx(xa, abs=0.02) and hi == pytest.approx(xb, abs=0.02)
    assert len(rec["started"]) == len(rec["released"]) == 0
    assert rig.view.measure_empty() is True


def test_a_drag_on_the_mask_body_draws_a_new_span_and_does_not_move_the_mask(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.15, c + 0.15))
    rec = _recorders(rig.view)
    before = rig.view.measure_bounds()

    # Press in the middle of the body, drag right past the mask's right edge.
    drag(rig.view, point_at(rig.view, rig.target_vb, c), point_at(rig.view, rig.target_vb, c + 0.40))

    assert len(rec["started"]) == 1 and len(rec["released"]) == 1
    lo, hi = rec["released"].calls[0][2:4]
    assert abs(lo - _sample_of(rig, c)) <= 20 and abs(hi - _sample_of(rig, c + 0.40)) <= 20
    assert rig.view.measure_bounds() == (lo, hi) != before
    # A MOVE would have kept the 0.30 s width; the new span is 0.40 s.
    assert (hi - lo) / FS == pytest.approx(0.40, abs=0.02)


def test_the_same_body_drag_moves_the_mask_when_draw_mode_is_off(rig):
    """The baseline the previous test inverts: nothing else about a body drag changed."""
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    rig.view.set_measure_region(c - 0.15, c + 0.15)
    rec = _recorders(rig.view)
    lo0, hi0 = rig.view.measure_bounds()

    drag(rig.view, point_at(rig.view, rig.target_vb, c), point_at(rig.view, rig.target_vb, c + 0.10))

    lo1, hi1 = rig.view.measure_bounds()
    assert abs((hi1 - lo1) - (hi0 - lo0)) <= 2         # moved, not resized (sample rounding)
    assert lo1 - lo0 == pytest.approx(0.10 * FS, abs=15)
    assert len(rec["started"]) == len(rec["released"]) == 0


def test_the_edge_handles_still_resize_in_draw_mode(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.15, c + 0.15))
    rec = _recorders(rig.view)
    lo0, hi0 = rig.view.measure_bounds()

    # Grab the RIGHT circular handle (vertical middle of the line) and pull it right.
    drag(rig.view, point_at(rig.view, rig.target_vb, c + 0.15), point_at(rig.view, rig.target_vb, c + 0.30))

    lo1, hi1 = rig.view.measure_bounds()
    assert lo1 == lo0                                  # the other edge did not move
    assert (hi1 - hi0) / FS == pytest.approx(0.15, abs=0.02)
    assert len(rec["started"]) == len(rec["released"]) == 0


# ================================================================================================
# Option inverts the drag in BOTH states
# ================================================================================================
def test_option_drag_draws_a_span_when_draw_mode_is_off(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    rec = _recorders(rig.view)
    before = rig.x_range()

    drag(rig.view, point_at(rig.view, rig.target_vb, c - 0.1), point_at(rig.view, rig.target_vb, c + 0.2),
         modifiers=Qt.AltModifier)

    assert len(rec["started"]) == 1 and len(rec["released"]) == 1
    assert rig.x_range() == pytest.approx(before, abs=1e-9)
    assert rig.view.measure_empty() is False


def test_option_drag_box_zooms_when_draw_mode_is_on(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    rec = _recorders(rig.view)
    xa, xb = c - 0.10, c + 0.10

    drag(rig.view, point_at(rig.view, rig.target_vb, xa), point_at(rig.view, rig.target_vb, xb),
         modifiers=Qt.AltModifier)

    lo, hi = rig.x_range()
    assert lo == pytest.approx(xa, abs=0.02) and hi == pytest.approx(xb, abs=0.02)
    assert len(rec["started"]) == len(rec["released"]) == 0
    assert rig.view.measure_empty() is True


def test_option_drag_on_the_mask_body_draws_with_draw_mode_off(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    rig.view.set_measure_region(c - 0.15, c + 0.15)
    rec = _recorders(rig.view)

    drag(rig.view, point_at(rig.view, rig.target_vb, c), point_at(rig.view, rig.target_vb, c + 0.30),
         modifiers=Qt.AltModifier)

    assert len(rec["started"]) == 1 and len(rec["released"]) == 1
    lo, hi = rig.view.measure_bounds()
    assert (hi - lo) / FS == pytest.approx(0.30, abs=0.02)


def test_option_with_draw_mode_on_gives_the_mask_body_its_old_meaning_back(rig):
    """Option inverts in BOTH states, and that includes the mask body: draw mode on + Option is a
    plain move of the mask (the box zoom's counterpart on the body)."""
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.15, c + 0.15))
    rec = _recorders(rig.view)
    lo0, hi0 = rig.view.measure_bounds()

    drag(rig.view, point_at(rig.view, rig.target_vb, c), point_at(rig.view, rig.target_vb, c + 0.10),
         modifiers=Qt.AltModifier)

    lo1, hi1 = rig.view.measure_bounds()
    assert abs((hi1 - lo1) - (hi0 - lo0)) <= 2 and lo1 > lo0
    assert len(rec["started"]) == len(rec["released"]) == 0


def test_releasing_option_half_way_does_not_change_what_the_drag_is(rig):
    """The modifier is read once, at the start."""
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    rec = _recorders(rig.view)
    before = rig.x_range()
    p0, p1 = point_at(rig.view, rig.target_vb, c - 0.2), point_at(rig.view, rig.target_vb, c + 0.2)
    hover(rig.view, p0, Qt.AltModifier)
    press(rig.view, p0, modifiers=Qt.AltModifier)
    drag_move(rig.view, drag_pos(p0, p1, 0.3), modifiers=Qt.AltModifier)
    drag_move(rig.view, drag_pos(p0, p1, 0.6))                      # Option let go
    drag_move(rig.view, drag_pos(p0, p1, 1.0))
    release(rig.view, p1)
    assert len(rec["started"]) == 1 and len(rec["released"]) == 1
    assert rig.x_range() == pytest.approx(before, abs=1e-9)


def test_a_double_click_on_the_mask_body_in_draw_mode_still_asks_to_catch(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.15, c + 0.15))
    before = rig.view.measure_bounds()
    catch = Recorder(rig.view.sigCatchRequested)
    double_click(rig.view, point_at(rig.view, rig.target_vb, c))
    assert len(catch) == 1 and rig.view.measure_bounds() == before


# ================================================================================================
# another pane does nothing, and says so
# ================================================================================================
def test_a_drag_on_another_pane_does_nothing_and_names_the_pane_to_draw_on(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    rec = _recorders(rig.view)
    before = rig.x_range()

    drag(rig.view, point_at(rig.view, rig.other_vb, c - 0.2), point_at(rig.view, rig.other_vb, c + 0.2))

    assert len(rec["started"]) == len(rec["released"]) == 0
    assert [call[0] for call in rec["refused"].calls] == ["Draw on the I pane"]
    assert rig.x_range() == pytest.approx(before, abs=1e-9)       # and it did not box-zoom either
    assert rig.view.measure_empty() is True


def test_option_drag_on_another_pane_is_refused_the_same_way(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    rec = _recorders(rig.view)
    drag(rig.view, point_at(rig.view, rig.other_vb, c - 0.2), point_at(rig.view, rig.other_vb, c + 0.2),
         modifiers=Qt.AltModifier)
    assert [call[0] for call in rec["refused"].calls] == ["Draw on the I pane"]


def test_an_option_drag_with_measure_off_says_to_turn_measure_on(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    rec = _recorders(rig.view)
    drag(rig.view, point_at(rig.view, rig.target_vb, c - 0.2), point_at(rig.view, rig.target_vb, c + 0.2),
         modifiers=Qt.AltModifier)
    assert len(rec["refused"]) == 1 and "Measure" in rec["refused"].calls[0][0]
    assert len(rec["started"]) == 0


# ================================================================================================
# cancel: right-click, a click-sized release, and a rebuild -- also for a draw begun on the mask body
# ================================================================================================
def _drag_with_right_click(rig, p0, p1, *, release_it=True):
    hover(rig.view, p0)
    press(rig.view, p0)
    steps = 6
    for k in range(1, steps + 1):
        t = k / steps
        drag_move(rig.view, QtCore.QPointF(p0.x() + (p1.x() - p0.x()) * t, p0.y()))
        if k == 3:
            send_mouse(rig.view, QtCore.QEvent.MouseButtonPress, drag_pos(p0, p1, t), Qt.RightButton,
                       Qt.LeftButton | Qt.RightButton)
            send_mouse(rig.view, QtCore.QEvent.MouseButtonRelease, drag_pos(p0, p1, t), Qt.RightButton,
                       Qt.LeftButton)
    if release_it:
        release(rig.view, p1)


def drag_pos(p0, p1, t):
    return QtCore.QPointF(p0.x() + (p1.x() - p0.x()) * t, p0.y())


def test_a_right_click_cancels_a_draw_started_on_empty_trace(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    rec = _recorders(rig.view)
    p0, p1 = point_at(rig.view, rig.target_vb, c - 0.2), point_at(rig.view, rig.target_vb, c + 0.2)

    _drag_with_right_click(rig, p0, p1)

    assert len(rec["started"]) == 1 and len(rec["cancelled"]) == 1 and len(rec["released"]) == 0
    assert rig.view.measure_empty() is True                    # the span it had started is gone
    assert len(rec["cleared"]) == 1                            # ...and the readout was told
    assert rig.view.draw_in_progress() is False
    assert rig.view.draw_mode() is True                        # cancelling never leaves the mode


def test_a_right_click_cancels_a_draw_started_on_the_mask_body_and_restores_the_mask(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.15, c + 0.15))
    before = rig.view.measure_bounds()
    rec = _recorders(rig.view)
    p0, p1 = point_at(rig.view, rig.target_vb, c), point_at(rig.view, rig.target_vb, c + 0.35)

    _drag_with_right_click(rig, p0, p1)

    assert len(rec["started"]) == 1 and len(rec["cancelled"]) == 1 and len(rec["released"]) == 0
    assert rig.view.measure_bounds() == before                  # the old mask, exactly
    assert rig.view.measure_empty() is False


def test_the_frames_after_a_cancel_are_ignored_until_the_release(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.15, c + 0.15))
    before = rig.view.measure_bounds()
    rec = _recorders(rig.view)
    p0, p1 = point_at(rig.view, rig.target_vb, c), point_at(rig.view, rig.target_vb, c + 0.35)
    _drag_with_right_click(rig, p0, p1, release_it=False)
    for t in (0.7, 0.8, 0.9, 1.0):
        drag_move(rig.view, drag_pos(p0, p1, t))
    assert rig.view.measure_bounds() == before
    release(rig.view, p1)
    assert rig.view.measure_bounds() == before
    assert len(rec["released"]) == 0 and len(rec["cancelled"]) == 1


def test_a_release_narrower_than_three_pixels_is_a_click_and_emits_only_the_cancel(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    rec = _recorders(rig.view)
    p0 = point_at(rig.view, rig.target_vb, c)
    far = point_at(rig.view, rig.target_vb, c + 0.15)
    near = QtCore.QPointF(p0.x() + 1.5, p0.y())

    hover(rig.view, p0)
    press(rig.view, p0)
    drag_move(rig.view, far)
    drag_move(rig.view, near)
    release(rig.view, near)

    assert len(rec["started"]) == 1 and len(rec["cancelled"]) == 1 and len(rec["released"]) == 0
    assert rig.view.measure_empty() is True


@pytest.mark.parametrize("on_body", [False, True])
def test_a_rebuild_mid_draw_cancels_it_and_does_not_bring_a_span_or_a_seed_back(rig, on_body):
    c = sum(zoom_middle(rig, 0.5)) / 2
    mask = (c - 0.15, c + 0.15) if on_body else None
    _armed(rig, mask=mask)
    before = rig.view.measure_bounds()
    rec = _recorders(rig.view)
    p0 = point_at(rig.view, rig.target_vb, c if on_body else c - 0.2)
    p1 = point_at(rig.view, rig.target_vb, c + 0.35)

    hover(rig.view, p0)
    press(rig.view, p0)
    for t in (0.25, 0.5):
        drag_move(rig.view, drag_pos(p0, p1, t))
    assert rig.view.draw_in_progress()
    rig.session.set_view_mode(ViewMode.STACKED)                 # a rebuild: new panes, new ViewBoxes
    QtWidgets.QApplication.processEvents()
    assert len(rec["cancelled"]) == 1 and rig.view.draw_in_progress() is False
    # ...and the scene keeps sending the rest of the drag to the ViewBox it removed.
    for t in (0.75, 1.0):
        drag_move(rig.view, drag_pos(p0, p1, t))
    release(rig.view, p1)

    assert len(rec["released"]) == 0 and len(rec["cancelled"]) == 1
    assert rig.view.measure_bounds() == before                  # the old mask, or still none
    assert rig.view.measure_empty() is (not on_body)
    assert rig.view.draw_mode() is True                         # the target pane survived


# ================================================================================================
# preview / settle: a 20-frame draw is not 20 recomputes
# ================================================================================================
def test_a_twenty_frame_draw_costs_the_measure_panel_at_most_two_recomputes(rig, monkeypatch):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.05, c + 0.05))
    calls = []
    real = measure_panel_module.measure_selection

    def counting(*args, **kwargs):
        calls.append(args[2:4])
        return real(*args, **kwargs)

    monkeypatch.setattr(measure_panel_module, "measure_selection", counting)
    rec = _recorders(rig.view)
    changed = Recorder(rig.view.sigMeasureChanged)
    finished = Recorder(rig.view.sigMeasureFinished)

    drag(rig.view, point_at(rig.view, rig.target_vb, c - 0.30), point_at(rig.view, rig.target_vb, c + 0.30),
         frames=20)

    assert len(rec["released"]) == 1
    assert len(changed) >= 18                                    # the frames did stream
    assert len(finished) == 1                                    # ...and the release settled once
    assert 1 <= len(calls) <= 2, calls
    lo, hi = rig.view.measure_bounds()
    assert calls[-1] == (lo, hi)                                 # the readout is the final span
    assert rig.panel.current_measurement()["start_idx"] == lo


def test_a_receiver_that_settles_the_release_itself_is_not_settled_twice(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    finished = Recorder(rig.view.sigMeasureFinished)
    rig.view.sigDrawReleased.connect(
        lambda doc, ch, lo, hi, px: rig.view.set_measure_region(
            rig.xs()[lo + 5], rig.xs()[hi - 5]))                  # a "fit": moves it and settles
    drag(rig.view, point_at(rig.view, rig.target_vb, c - 0.2), point_at(rig.view, rig.target_vb, c + 0.2))
    assert len(finished) == 1


def test_a_release_with_no_receiver_settles_exactly_once(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    finished = Recorder(rig.view.sigMeasureFinished)
    drag(rig.view, point_at(rig.view, rig.target_vb, c - 0.2), point_at(rig.view, rig.target_vb, c + 0.2))
    assert len(finished) == 1 and finished.calls[0] == rig.view.measure_bounds()


def test_a_draw_past_the_end_of_the_recording_clips_to_the_loaded_data(rig):
    zoom_middle(rig, 0.5)
    _armed(rig)
    rec = _recorders(rig.view)
    e0, e1 = rig.view.full_x_extent()
    zoom_x = rig.x_range()
    # Zoom to the tail so the end of the data is on screen, then drag off it to the right.
    rig.view.zoom_x_to(e1 - 0.6, e1 + 0.3)
    QtWidgets.QApplication.processEvents()
    drag(rig.view, point_at(rig.view, rig.target_vb, e1 - 0.3), point_at(rig.view, rig.target_vb, e1 + 0.2))
    lo, hi = rec["released"].calls[0][2:4]
    assert hi == N and lo < N
    # ...and the MASK is clamped too, not only the indices read off it: its right edge sits at
    # the legal cursor extent (one sample past the last sample), not out in the blank margin.
    assert rig.view.measure_x_range()[1] == pytest.approx(
        rig.view.measure_cursor_extent(rig.doc_id, CURRENT)[1], abs=1e-9)
    _ = zoom_x


# ================================================================================================
# the mode: needs a target, survives what the target survives
# ================================================================================================
def test_draw_mode_needs_a_measure_target(rig):
    rec = _recorders(rig.view)
    assert rig.view.set_draw_mode(True) is False
    assert rig.view.draw_mode() is False
    assert len(rec["refused"]) == 1 and len(rec["mode"]) == 0


def test_draw_mode_survives_a_view_mode_change_and_a_rebuild(rig):
    zoom_middle(rig, 0.5)
    _armed(rig)
    rec = _recorders(rig.view)
    assert rig.session.view_mode is ViewMode.OVERLAY             # the default
    rig.session.set_view_mode(ViewMode.STACKED)
    QtWidgets.QApplication.processEvents()
    rig.view.rebuild()
    rig.session.set_view_mode(ViewMode.OVERLAY)
    QtWidgets.QApplication.processEvents()
    assert rig.view.draw_mode() is True and len(rec["mode"]) == 0
    # ...and it still draws on the NEW ViewBoxes.
    c = sum(rig.x_range()) / 2
    drag(rig.view, point_at(rig.view, rig.target_vb, c - 0.1), point_at(rig.view, rig.target_vb, c + 0.1))
    assert len(rec["released"]) == 1


def test_draw_mode_drops_when_the_target_pane_is_gone(rig):
    _armed(rig)
    rec = _recorders(rig.view)
    rig.session.set_layer_visible(f"{rig.doc_id}#{CURRENT}", False)     # the target pane leaves
    QtWidgets.QApplication.processEvents()
    assert rig.view.draw_mode() is False
    assert rec["mode"].calls == [(False,)]


def test_draw_mode_drops_when_measure_is_turned_off(rig):
    _armed(rig)
    rec = _recorders(rig.view)
    rig.panel.set_active(False)
    assert rig.view.draw_mode() is False and rec["mode"].calls == [(False,)]


def test_draw_mode_stays_on_across_a_discarded_mask(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig, mask=(c - 0.1, c + 0.1))
    assert rig.view.discard_measure_region() is True
    assert rig.view.draw_mode() is True and rig.view.measure_empty() is True


# ================================================================================================
# double-click: catch in draw mode, today's mask outside it
# ================================================================================================
def test_a_double_click_in_draw_mode_asks_to_catch_and_places_no_mask(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    catch = Recorder(rig.view.sigCatchRequested)
    placed = Recorder(rig.view.sigMeasurePlacementRequested)

    double_click(rig.view, point_at(rig.view, rig.target_vb, c))

    assert len(catch) == 1 and len(placed) == 0
    doc_id, channel, sample, px = catch.calls[0]
    assert (doc_id, channel) == (rig.doc_id, CURRENT)
    assert abs(sample - _sample_of(rig, c)) <= 20 and 5.0 < px < 25.0
    assert rig.view.measure_empty() is True                        # no 20% mask appeared
    assert rig.view.measure_bounds() is None


def test_a_double_click_outside_draw_mode_still_places_the_mask(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    catch = Recorder(rig.view.sigCatchRequested)
    placed = Recorder(rig.view.sigMeasurePlacementRequested)

    double_click(rig.view, point_at(rig.view, rig.target_vb, c))

    assert len(placed) == 1 and len(catch) == 0
    assert placed.calls[0][0] == pytest.approx(c, abs=0.02)
    lo, hi = rig.view.measure_bounds()                              # the panel placed its 20%
    assert (hi - lo) / FS == pytest.approx(0.20 * 1.0, abs=0.03)


def test_a_double_click_on_another_pane_in_draw_mode_is_refused(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    catch = Recorder(rig.view.sigCatchRequested)
    refused = Recorder(rig.view.sigDrawRefused)
    double_click(rig.view, point_at(rig.view, rig.other_vb, c))
    assert len(catch) == 0 and refused.calls == [("Draw on the I pane",)]


# ================================================================================================
# the caption does not block a drag beneath it
# ================================================================================================
def test_a_drag_that_starts_under_the_caption_still_draws(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _armed(rig)
    rig.view.set_target_label("→ Manual · 37 hand-made · 2 in view · draw")
    QtWidgets.QApplication.processEvents()
    label = rig.view._target_label
    assert label is not None and label.isVisible()
    rec = _recorders(rig.view)
    # Press at the middle of the caption (scene -> viewport), which sits over the data.
    centre = label.mapToScene(label.boundingRect().center())
    p0 = QtCore.QPointF(rig.view.mapFromScene(centre))
    p1 = QtCore.QPointF(p0.x() + 300.0, p0.y())

    drag(rig.view, p0, p1)

    assert len(rec["started"]) == 1 and len(rec["released"]) == 1
