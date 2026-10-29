"""Moving along the trace: the horizontal swipe, the wheel, zoom about the pointer, the window step,
and the pointer helpers the trace keys are built on.

A two-finger swipe used to pan the AMPLITUDE whichever way it went. Now a horizontal one pans time,
with the axis chosen once per gesture from the accumulated ``pixelDelta`` (Qt reports each event
along whichever component dominates THAT event, so an unlocked diagonal swipe alternates between
time and amplitude), and a mouse wheel still pans amplitude unless it is horizontal or Shift is
held. Zoom takes an anchor: the data x under the pointer stays where it is on screen.
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.layer import ViewMode

from .test_trace_rig import (  # noqa: F401  (fixtures are used by name)
    CURRENT, FS, N, Qt, Recorder, build_rig, drag, free_mouse, global_of, hover, make_document,
    point_at, rig, swipe, wheel, zoom_middle)


def _centre(rig):
    return point_at(rig.view, rig.target_vb, sum(rig.x_range()) / 2)


def _dpp(rig) -> float:
    """Data units per screen pixel on the target pane."""
    lo, hi = rig.x_range()
    return (hi - lo) / rig.target_vb.width()


def _pinned(rig):
    """Freeze the y range so a time pan cannot be confused with an autorange."""
    zoom_middle(rig, 0.5)
    rig.target_vb.setYRange(90.0, 110.0, padding=0)
    QtWidgets.QApplication.processEvents()
    return rig.x_range(), rig.y_range()


# ================================================================================================
# the swipe: time, amplitude, and the lock
# ================================================================================================
def test_a_horizontal_swipe_pans_time_and_not_amplitude(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    swipe(rig.view, _centre(rig), [(-30, 0)] * 4)

    nx0, nx1 = rig.x_range()
    assert nx1 - nx0 == pytest.approx(x1 - x0, abs=1e-9)                       # a pan, not a zoom
    # Four updates of 30 px of content travel to the LEFT: the trace follows the fingers, so the
    # window moves later in time by 120 px worth of seconds.
    assert nx0 - x0 == pytest.approx(120 * (x1 - x0) / rig.target_vb.width(), rel=0.01)
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-9)


def test_a_swipe_the_other_way_goes_back(rig):
    (x0, x1), _ = _pinned(rig)
    swipe(rig.view, _centre(rig), [(40, 0)] * 3)
    assert rig.x_range()[0] < x0


def test_a_diagonal_swipe_stays_on_the_axis_it_started_on(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    # The first updates are mostly horizontal; later ones are mostly VERTICAL. Unlocked, Qt's
    # per-event dominant component would flip the pan to amplitude half way through.
    steps = [(-20, 3), (-4, 25), (-5, 30), (-3, 18), (-6, 22)]
    swipe(rig.view, _centre(rig), steps)

    nx0, nx1 = rig.x_range()
    assert nx0 - x0 == pytest.approx(sum(-s[0] for s in steps) * (x1 - x0) / rig.target_vb.width(), rel=0.01)
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-9)


def test_a_vertical_start_locks_amplitude_and_ignores_the_horizontal_drift(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    swipe(rig.view, _centre(rig), [(2, -20), (25, -5), (30, -4), (18, -3)])
    assert rig.x_range() == pytest.approx((x0, x1), abs=1e-9)
    assert rig.y_range()[0] != pytest.approx(y0)                               # amplitude moved
    assert rig.y_range()[1] - rig.y_range()[0] == pytest.approx(y1 - y0, abs=1e-9)


def test_a_vertical_swipe_moves_amplitude_exactly_as_the_old_arithmetic_did(rig):
    """Qt gives a pixel-precise device two angle units per pixel; the old pan used the angle."""
    (x0, x1), (y0, y1) = _pinned(rig)
    swipe(rig.view, _centre(rig), [(0, 60)])
    # 120 angle units = one notch = one tenth of the pane height, the signal moving UP.
    assert rig.y_range()[0] - y0 == pytest.approx(-0.1 * (y1 - y0), abs=1e-6)


def test_the_axis_is_chosen_only_once_six_pixels_have_accumulated(rig):
    (x0, x1), _ = _pinned(rig)
    dpp = (x1 - x0) / rig.target_vb.width()
    swipe(rig.view, _centre(rig), [(-2, 0), (-2, 0)])
    assert rig.x_range()[0] == pytest.approx(x0, abs=1e-12)                    # 4 px: undecided
    swipe(rig.view, _centre(rig), [(-3, 0)], begin=False)
    # ...and the travel held back is not lost: all 7 px are applied on the chosen axis.
    assert rig.x_range()[0] - x0 == pytest.approx(7 * dpp, rel=0.01)


def test_a_small_nudge_that_never_reaches_six_pixels_moves_nothing(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    swipe(rig.view, _centre(rig), [(1, 1), (1, 2), (-1, 1)], end=True)
    assert rig.x_range() == pytest.approx((x0, x1), abs=1e-12)
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-12)


def test_a_new_scroll_begin_resets_the_lock(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    swipe(rig.view, _centre(rig), [(-30, 0), (-30, 0)], end=True)              # locked on time
    mid = rig.x_range()
    assert mid[0] > x0
    # The next gesture's FIRST event carries no movement at all, and is the only thing that says
    # "this is a new swipe": the reset has to happen before the zero-delta early return.
    swipe(rig.view, _centre(rig), [(2, -20), (3, -25)])
    assert rig.x_range() == pytest.approx(mid, abs=1e-12)
    assert rig.y_range()[0] != pytest.approx(y0)


def test_momentum_keeps_the_axis_the_swipe_locked(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    c = _centre(rig)
    swipe(rig.view, c, [(-30, 2), (-30, 3)], end=True)                         # fingers lift
    after_swipe = rig.x_range()
    for px, py in ((-12, 14), (-9, 20), (-6, 25)):                             # the fling, diagonal
        wheel(rig.view, c, pixel=(px, py), angle=(2 * px, 2 * py), phase=Qt.ScrollMomentum)
    assert rig.x_range()[0] > after_swipe[0]
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-9)


def test_a_pause_ends_a_gesture_for_hardware_that_does_not_frame_them(rig, monkeypatch):
    import nanoscope.gui.trace_view as tv

    (x0, x1), (y0, y1) = _pinned(rig)
    clock = [100.0]
    monkeypatch.setattr(tv.time, "monotonic", lambda: clock[0])
    c = _centre(rig)
    wheel(rig.view, c, pixel=(-30, 0), angle=(-60, 0), phase=Qt.ScrollUpdate)  # no Begin ever seen
    wheel(rig.view, c, pixel=(-30, 0), angle=(-60, 0), phase=Qt.ScrollUpdate)
    moved = rig.x_range()[0]
    assert moved > x0
    clock[0] += 5.0                                                            # a long pause
    wheel(rig.view, c, pixel=(0, 30), angle=(0, 60), phase=Qt.ScrollUpdate)
    wheel(rig.view, c, pixel=(0, 30), angle=(0, 60), phase=Qt.ScrollUpdate)
    assert rig.x_range()[0] == pytest.approx(moved, abs=1e-12)
    assert rig.y_range()[0] != pytest.approx(y0)


def test_a_device_with_pixels_but_no_phase_decides_each_event_for_itself(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    c = _centre(rig)
    wheel(rig.view, c, pixel=(-40, 3), angle=(-80, 6))                         # NoScrollPhase
    assert rig.x_range()[0] > x0 and rig.y_range() == pytest.approx((y0, y1), abs=1e-9)


def test_a_swipe_cannot_carry_the_window_off_the_end_of_the_recording(rig):
    e0, e1 = rig.view.full_x_extent()
    rig.view.zoom_x_to(e1 - 1.0, e1)
    QtWidgets.QApplication.processEvents()
    swipe(rig.view, _centre(rig), [(-200, 0)] * 6)
    assert rig.x_range()[1] == pytest.approx(e1, abs=1e-6)
    rig.view.zoom_x_to(e0, e0 + 1.0)
    QtWidgets.QApplication.processEvents()
    swipe(rig.view, _centre(rig), [(200, 0)] * 6)
    assert rig.x_range()[0] == pytest.approx(e0, abs=1e-6)


def test_a_swipe_that_lands_on_the_other_pane_pans_the_same_linked_axis(rig):
    (x0, x1), _ = _pinned(rig)
    swipe(rig.view, point_at(rig.view, rig.other_vb, sum((x0, x1)) / 2), [(-30, 0)] * 3)
    assert rig.x_range()[0] > x0


# ================================================================================================
# the mouse wheel
# ================================================================================================
def test_a_vertical_wheel_still_pans_amplitude(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    wheel(rig.view, _centre(rig), angle=(0, 120))
    assert rig.x_range() == pytest.approx((x0, x1), abs=1e-12)
    assert rig.y_range()[0] - y0 == pytest.approx(-0.1 * (y1 - y0), abs=1e-6)  # the signal moves up
    wheel(rig.view, _centre(rig), angle=(0, -120))
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-6)


def test_a_horizontal_wheel_pans_time_by_a_tenth_of_the_view_per_notch(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    wheel(rig.view, _centre(rig), angle=(-120, 0))
    assert rig.x_range()[0] - x0 == pytest.approx(0.1 * (x1 - x0), rel=1e-6)
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-9)
    wheel(rig.view, _centre(rig), angle=(120, 0))
    assert rig.x_range()[0] == pytest.approx(x0, abs=1e-9)


def test_shift_with_a_vertical_wheel_pans_time(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    wheel(rig.view, _centre(rig), angle=(0, -120), modifiers=Qt.ShiftModifier)
    assert rig.x_range()[0] > x0
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-9)


def test_a_bare_test_double_event_is_still_a_vertical_wheel(rig):
    """tests/test_zoom_navigation.py drives the ViewBox with an object that only has delta()."""
    (x0, x1), (y0, y1) = _pinned(rig)

    class Wheel:
        def __init__(self, d):
            self._d = d

        def delta(self):
            return self._d

        def accept(self):
            pass

        def ignore(self):
            pass

    rig.target_vb.wheelEvent(Wheel(120))
    assert rig.x_range() == pytest.approx((x0, x1), abs=1e-12)
    assert rig.y_range()[0] < y0


def test_zero_delta_events_change_nothing(rig):
    (x0, x1), (y0, y1) = _pinned(rig)
    wheel(rig.view, _centre(rig), phase=Qt.ScrollEnd)
    wheel(rig.view, _centre(rig), angle=(0, 0))
    assert rig.x_range() == pytest.approx((x0, x1), abs=1e-12)
    assert rig.y_range() == pytest.approx((y0, y1), abs=1e-12)


# ================================================================================================
# zoom about the pointer
# ================================================================================================
def _fraction(rig, x: float) -> float:
    lo, hi = rig.x_range()
    return (x - lo) / (hi - lo)


def test_zoom_in_about_an_anchor_keeps_the_anchor_where_it_is_on_screen(rig):
    lo, hi = zoom_middle(rig, 1.0)
    anchor = lo + 0.8 * (hi - lo)
    before = _fraction(rig, anchor)
    assert rig.bar.zoom_in(anchor) is True
    nlo, nhi = rig.x_range()
    assert nhi - nlo == pytest.approx(0.5 * (hi - lo), rel=1e-9)
    assert _fraction(rig, anchor) == pytest.approx(before, abs=1e-9)
    # ...and zooming back out about the same point returns the window.
    assert rig.bar.zoom_out(anchor) is True
    assert rig.x_range() == pytest.approx((lo, hi), abs=1e-9)


def test_zoom_without_an_anchor_still_zooms_about_the_centre(rig):
    lo, hi = zoom_middle(rig, 1.0)
    assert rig.bar.zoom_in() is True
    nlo, nhi = rig.x_range()
    assert 0.5 * (nlo + nhi) == pytest.approx(0.5 * (lo + hi), abs=1e-9)


def test_the_scan_strip_buttons_keep_the_centre(rig):
    """``clicked`` hands its ``checked`` flag to a slot with room for it (PySide really does): that
    must not become an anchor of 0.0. The view is put where x = 0.0 IS inside it, so a flag read
    as an anchor would pull the zoom to the start instead of the centre."""
    rig.view.zoom_x_to(0.0, 2.0)
    QtWidgets.QApplication.processEvents()
    lo, hi = rig.x_range()
    assert lo <= 0.0 <= hi
    centre = 0.5 * (lo + hi)
    rig.bar._zoom_in_button.click()
    nlo, nhi = rig.x_range()
    assert nhi - nlo == pytest.approx(0.5 * (hi - lo), rel=1e-9)
    assert 0.5 * (nlo + nhi) == pytest.approx(centre, abs=1e-9)
    rig.bar._zoom_out_button.click()
    nlo, nhi = rig.x_range()
    assert 0.5 * (nlo + nhi) == pytest.approx(centre, abs=1e-9)
    rig.view.zoom_x_to(0.0, 2.0)
    assert rig.bar.zoom_in(False) is True                          # a bare flag is not an anchor
    nlo, nhi = rig.x_range()
    assert 0.5 * (nlo + nhi) == pytest.approx(centre, abs=1e-9)


def test_an_anchor_outside_the_view_falls_back_to_the_centre(rig):
    lo, hi = zoom_middle(rig, 1.0)
    centre = 0.5 * (lo + hi)
    for bad in (lo - 0.5, hi + 0.5, float("nan"), None):
        zoom_middle(rig, 1.0)
        assert rig.bar.zoom_in(bad) is True
        nlo, nhi = rig.x_range()
        assert 0.5 * (nlo + nhi) == pytest.approx(centre, abs=1e-9), bad


def test_near_the_end_of_the_recording_the_clamp_moves_the_window(rig):
    e0, e1 = rig.view.full_x_extent()
    rig.view.zoom_x_to(e1 - 2.0, e1)
    QtWidgets.QApplication.processEvents()
    anchor = e1 - 0.05
    assert rig.bar.zoom_out(anchor) is True
    nlo, nhi = rig.x_range()
    assert nhi == pytest.approx(e1, abs=1e-6) and nhi - nlo == pytest.approx(4.0, rel=1e-6)


def test_a_zoom_in_never_goes_below_two_samples_and_still_anchors(rig):
    zoom_middle(rig, 1e-3)
    lo, hi = rig.x_range()
    anchor = lo + 0.25 * (hi - lo)
    for _ in range(12):
        rig.bar.zoom_in(anchor)
    nlo, nhi = rig.x_range()
    assert nhi - nlo == pytest.approx(2.0 / FS, rel=1e-6)
    assert nlo <= anchor <= nhi


def test_step_window_is_the_ninety_percent_step(rig):
    lo, hi = zoom_middle(rig, 0.5)
    assert rig.bar.step_window(+1) is True
    nlo, nhi = rig.x_range()
    assert nlo - lo == pytest.approx(0.9 * (hi - lo), rel=1e-6)
    assert rig.bar.step_window(-1) is True and rig.bar.step_window(0) is True


# ================================================================================================
# the pointer helpers
# ================================================================================================
def _pointer_to(rig, vb, x, yfrac=0.5):
    hover(rig.view, point_at(rig.view, vb, x, yfrac))


def test_pointer_helpers_report_the_sample_under_the_pointer_on_the_target_pane(rig):
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    c = sum(rig.x_range()) / 2
    x = c + 0.123
    _pointer_to(rig, rig.target_vb, x)

    got = rig.view.pointer_sample()

    assert got is not None
    doc_id, channel, index, px = got
    assert (doc_id, channel) == (rig.doc_id, CURRENT)
    assert index == pytest.approx(x * FS, abs=20)                    # within a pixel (14 samples)
    assert px == pytest.approx(rig.view.target_px_samples())
    lo, hi = rig.x_range()
    assert px == pytest.approx((hi - lo) / rig.target_vb.width() * FS, rel=1e-9)
    assert rig.view.pointer_data_x() == pytest.approx(x, abs=0.01)


def test_the_pointer_helpers_are_none_off_the_target_plot_area(rig):
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    c = sum(rig.x_range()) / 2
    _pointer_to(rig, rig.other_vb, c)                                 # the other pane
    assert rig.view.pointer_sample() is None and rig.view.pointer_data_x() is None
    r = rig.target_vb.sceneBoundingRect()                             # the left axis, in the widget
    axis = QtCore.QPointF(rig.view.mapFromScene(QtCore.QPointF(r.left() - 12, r.center().y())))
    hover(rig.view, axis)
    assert rig.view.pointer_sample() is None
    QtWidgets.QApplication.processEvents()
    from PySide6 import QtGui
    QtGui.QCursor.setPos(QtCore.QPoint(-4000, -4000))                 # nowhere near the trace
    assert rig.view.pointer_sample() is None


def test_the_pointer_helpers_are_none_while_a_window_covers_the_trace(rig):
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    c = sum(rig.x_range()) / 2
    p = point_at(rig.view, rig.target_vb, c)
    _pointer_to(rig, rig.target_vb, c)
    assert rig.view.pointer_sample() is not None
    cover = QtWidgets.QWidget(None, Qt.Tool | Qt.FramelessWindowHint)
    g = global_of(rig.view, p)
    cover.setGeometry(g.x() - 50, g.y() - 50, 100, 100)
    cover.show()
    cover.raise_()
    QtWidgets.QApplication.processEvents()
    try:
        if QtWidgets.QApplication.widgetAt(g) is rig.view.viewport():
            pytest.skip("this platform plugin does not stack a second top-level window over the first")
        assert rig.view.pointer_sample() is None
        assert rig.view.pointer_data_x() is None
    finally:
        cover.hide()


def test_the_pointer_helpers_need_a_target(rig):
    zoom_middle(rig, 0.5)
    c = sum(rig.x_range()) / 2
    _pointer_to(rig, rig.target_vb, c)
    assert rig.view.pointer_sample() is None and rig.view.target_px_samples() is None
    rig.view.set_annotation_target(rig.doc_id, CURRENT)               # a fallback target stands in
    assert rig.view.pointer_sample() is not None and rig.view.target_px_samples() is not None


def test_a_sample_index_is_through_the_targets_own_axis_not_the_first_layers(qapp, free_mouse):
    """Two recordings share the pA pane in Overlay; the first layer is NOT the target and sits on
    another time offset. Index, draw and px must all follow the target."""
    rig = build_rig(qapp)
    other = make_document(name="second.dat")
    rig.session.add_document(other)
    rig.session.add_layer(other.doc_id, CURRENT)
    qapp.processEvents()
    assert rig.session.view_mode is ViewMode.OVERLAY
    assert (rig.view.pane_for_layer(f"{other.doc_id}#{CURRENT}")
            is rig.view.pane_for_layer(f"{rig.doc_id}#{CURRENT}"))      # one shared pane
    rig.session.set_layer_time_offset(f"{other.doc_id}#{CURRENT}", 0.25)  # (rebuilds the panes)
    qapp.processEvents()
    rig.view.set_measure_active(other.doc_id, CURRENT, True, seed=False)
    zoom_middle(rig, 0.5)
    vb = rig.view.pane_for_layer(f"{other.doc_id}#{CURRENT}").plot.vb      # fetched AFTER the rebuild
    c = sum(rig.x_range()) / 2
    hover(rig.view, point_at(rig.view, vb, c + 0.1))

    doc_id, channel, index, _px = rig.view.pointer_sample()

    assert (doc_id, channel) == (other.doc_id, CURRENT)
    assert index == pytest.approx((c + 0.1 - 0.25) * FS, abs=20)      # NOT (c + 0.1) * FS
    rig.view.set_draw_mode(True)
    released = Recorder(rig.view.sigDrawReleased)
    drag(rig.view, point_at(rig.view, vb, c), point_at(rig.view, vb, c + 0.2))
    got_doc, _ch, lo, hi, _px = released.calls[0]
    assert got_doc == other.doc_id
    assert lo == pytest.approx((c - 0.25) * FS, abs=20) and hi == pytest.approx((c + 0.2 - 0.25) * FS, abs=20)
    rig.view.hide()


def test_pointer_index_is_clamped_to_the_recording(rig):
    e0, e1 = rig.view.full_x_extent()
    rig.panel.set_active(True, seed=False)
    rig.view.zoom_x_to(e1 - 0.5, e1 + 0.5)
    QtWidgets.QApplication.processEvents()
    _pointer_to(rig, rig.target_vb, e1 + 0.4)
    _doc, _ch, index, _px = rig.view.pointer_sample()
    assert index == float(N)


def test_every_helper_is_inert_with_nothing_open(qapp, free_mouse):
    """The placeholder screen: no panes, no target. Nothing may raise."""
    from nanoscope.core.session import Session
    from nanoscope.gui.trace_view import TraceView

    view = TraceView(Session())
    view.resize(500, 300)
    view.show()
    qapp.processEvents()
    assert view.pointer_sample() is None and view.pointer_data_x() is None
    assert view.target_px_samples() is None
    assert view.set_draw_mode(True) is False and view.draw_mode() is False
    assert view.cancel_draw() is False and view.draw_in_progress() is False
    assert view.discard_measure_region() is False and view.measure_empty() is False
    view.set_target_label("→ Manual")                      # no target pane: nothing to caption
    assert view._target_label is None and view.target_label_text() == "→ Manual"
    view.set_annotation_target("nobody", 0)                 # a target that is not drawn
    assert view.pointer_sample() is None
    view.setFocus(Qt.MouseFocusReason)                      # the focus handlers survive no panes
    view.clearFocus()
    view.rebuild()
    view.hide()
