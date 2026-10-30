"""S9-08 — With Measure off the trace still has a key target: focus follows the pointer onto it,
C catches there, and Shift+Up / Shift+Down zoom about the pointer.

MEASURED DEFECT (sweep 9, a six-event synthetic recording, Measure off as a recording opens).
``TraceView._target_ref`` is the Measure target while Measure is on and otherwise the annotation
target, which only ``TraceView.set_annotation_target`` names -- and nothing in the app called it.
So with Measure off the trace had no key target at all. The pointer crossing onto an event left
the keyboard on the workflow chip the window starts on (``_StageChip``), and C, pressed there as
USER_GUIDE section 8 says ("Point at the event and press C"; C "Turns Measure on if it is off"),
did nothing: Measure stayed off and the status line stayed empty. With the trace focused by a click
and the pointer a fifth of the way across the plot, Shift+Up zoomed about the centre instead
(``pointer_data_x()`` was None) and the time under the pointer moved by +0.06 s.

THE FIX. ``MainWindow._retarget_selection`` hands the trace the same resolved (recording, current
channel) it hands the Navigator and the Scan bar -- Analysis's target, the one C, S, F and D act on
-- so the focus rule, the pointer helpers, the cross cursor and the caption work before the first
mask, as ``set_annotation_target``'s docstring intends. ``MainWindow._on_document_closed`` clears
it with the recording it names, so the rule has nothing to fire on until a selection names a
target again; with no recording open there is no pane for it either. While Measure is on, its own
target wins, as before. The new target takes nothing from a holder that owns keys: with Measure off
the pointer still leaves the docked Event inspector its keyboard (S9-03), so A still accepts after
the crossing.

These tests drive the real MainWindow at a size that lies inside the offscreen screen
(``QApplication.widgetAt``, which the pointer helpers use, finds nothing beyond it): recordings
opened and selected through ``_on_selection`` (the Files dock's door) and closed through
``_close_document`` (its Close button's), the pointer through real mouse moves on the trace's
viewport, and C, Shift+Up, Shift+Down and the inspector's R and A as key presses to whatever holds
the keyboard. Measure is left off, as a recording opens, except in the controls that turn it on.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

from ._inspector_hosting import (  # noqa: F401  (fixtures are used by name)
    FS, I0, N, SPANS, accepted, build_window, close_window, docked, private_settings, settle,
    signal)
from .test_trace_rig import free_mouse, hover, point_at  # noqa: F401

#: The event the detector was not handed: on the trace, not in the table.
MISSED = 3


# ================================================================================================
# builders
# ================================================================================================

def _open(window, tmp_path, name: str, *, detect: bool = True, with_voltage: bool = False) -> str:
    """Open the six-event recording and select it, as the Files dock does. With ``detect`` a
    Schmitt table that lacks event ``MISSED``; with ``with_voltage`` a command-voltage channel 0
    in front of the current (channel 1), as an ABF has."""
    current = signal()
    infos = [ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)]
    channels = [current]
    if with_voltage:
        infos = [ChannelInfo(index=0, name="command", raw_units="mV", kind=ChannelKind.VOLTAGE),
                 ChannelInfo(index=1, name="current", raw_units="pA", kind=ChannelKind.CURRENT)]
        channels = [np.full(N, -100.0, dtype=np.float32), current]
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=tuple(infos))
    doc = TraceDocument(meta=meta, channels=tuple(channels))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    if detect:
        seen = current.copy()
        s0, s1 = SPANS[MISSED]
        seen[s0:s1] += 300.0                     # the detector is handed the trace without it
        frame, provenance = detect_events(seen, FS, baseline=I0, min_dwell_s=1e-4)
        assert len(frame) == len(SPANS) - 1
        ch = len(channels) - 1
        window._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=ch,
            params={"min_dwell_s": 1e-4},
            calibration_fingerprint=window._session.calibration(doc_id, ch).fingerprint(),
            payload=EventSet(frame, provenance)))
    settle()
    return doc_id


def _window(qtbot, tmp_path, *, measure: bool = False, **kw):
    window = build_window(qtbot, size=(780, 700), detection=False)
    window.move(0, 0)
    settle()
    doc_id = _open(window, tmp_path, "s9_08.dat", **kw)
    if measure:
        window._analysis_panel.measure_panel.set_active(True, seed=False)
        settle()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=5_000)
    return window, window._trace_view, doc_id


def _vb(tv, doc_id: str, channel: int = 0):
    return tv.pane_for_layer(f"{doc_id}#{channel}").plot.vb


def _cross_onto(tv, vb, *, frac: float = 0.5, x: float | None = None) -> QtCore.QPointF:
    """The hand comes from the left axis onto ``vb``'s plot area: at data ``x``, or a fraction
    ``frac`` of its width. Returns where it stopped, in scene coordinates."""
    r = vb.sceneBoundingRect()
    hover(tv, QtCore.QPointF(tv.mapFromScene(QtCore.QPointF(r.left() - 12.0, r.center().y()))))
    inside = (point_at(tv, vb, x) if x is not None
              else QtCore.QPointF(tv.mapFromScene(
                  QtCore.QPointF(r.left() + frac * r.width(), r.center().y()))))
    hover(tv, inside)
    settle(10)
    return tv.mapToScene(inside.toPoint())


def _cursor(vb):
    return vb.cursor().shape() if vb.hasCursor() else None


# ================================================================================================
# focus follows the pointer, and the cross says the keys are live
# ================================================================================================

@pytest.mark.parametrize("measure", [False, True], ids=["measure_off", "measure_on_control"])
def test_focus_follows_the_pointer_onto_the_trace(
        private_settings, qtbot, free_mouse, tmp_path, measure):
    window, tv, doc_id = _window(qtbot, tmp_path, measure=measure)
    vb = _vb(tv, doc_id)
    focus = QtWidgets.QApplication.focusWidget()
    if focus is not None:
        focus.clearFocus()
    settle()
    assert not tv.hasFocus()

    _cross_onto(tv, vb)

    assert tv.hasFocus(), (
        f"the pointer is on the trace but the keyboard is on "
        f"{type(QtWidgets.QApplication.focusWidget()).__name__}: C, S, F, G and E are dead")
    assert tv.last_focus_reason() == Qt.MouseFocusReason
    assert _cursor(vb) == Qt.CrossCursor                 # ...and the cross says so
    close_window(window)


def test_point_at_a_missed_event_and_press_c_with_measure_off(
        private_settings, qtbot, free_mouse, tmp_path):
    """The guide's step 2 in a fresh window: no clearFocus, the window's own first focus."""
    window, tv, doc_id = _window(qtbot, tmp_path)
    measure = window._analysis_panel.measure_panel
    assert not measure.is_active()
    s0, s1 = SPANS[MISSED]
    tv.zoom_x_to(tv.x_for_index(doc_id, 0, s0 - 3_000), tv.x_for_index(doc_id, 0, s1 + 3_000))
    settle()
    assert not tv.hasFocus()

    _cross_onto(tv, _vb(tv, doc_id), x=tv.x_for_index(doc_id, 0, (s0 + s1) // 2))
    holder = QtWidgets.QApplication.focusWidget()
    assert holder is tv, f"the keyboard stayed on {type(holder).__name__}"
    QTest.keyClick(holder, Qt.Key_C)
    settle()

    assert measure.is_active(), "C did not reach the trace (Measure is still off)"
    bounds = tv.measure_bounds()
    assert bounds is not None, window.statusBar().currentMessage()
    assert abs(bounds[0] - s0) <= 10 and abs(bounds[1] - s1) <= 10, bounds
    assert window.statusBar().currentMessage().startswith("Caught")
    close_window(window)


def test_with_measure_off_the_docked_event_inspector_keeps_its_keys(
        private_settings, qtbot, free_mouse, tmp_path):
    """Guard for the two fixes together. Measure off is the state a recording opens in and a
    curation pass runs in, and the pointer now takes the keyboard there too -- but never from the
    docked Event inspector (S9-03). Measured with this fix alone, on the old registration: the
    crossing moved the keyboard to the trace."""
    window, tv, doc_id = _window(qtbot, tmp_path)
    insp, _dock = docked(window)
    assert not window._analysis_panel.measure_panel.is_active()
    event = int(window._analysis_panel.current_detection().payload.df["event_id"].iloc[1])
    insp.show_event(event)
    insp.focus_curation_surface()                    # what MainWindow does once it is shown
    settle()
    QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key_R)
    settle()
    assert not accepted(window, event)               # curating: R rejected the event on screen

    _cross_onto(tv, _vb(tv, doc_id))

    holder = QtWidgets.QApplication.focusWidget()
    assert holder is not None and (holder is insp or insp.isAncestorOf(holder)), (
        f"the crossing moved the keyboard to {type(holder).__name__}")
    QTest.keyClick(holder, Qt.Key_A)
    settle()
    assert accepted(window, event)
    close_window(window)


# ================================================================================================
# Shift+Up / Shift+Down zoom about the pointer
# ================================================================================================

@pytest.mark.parametrize("measure", [False, True], ids=["measure_off", "measure_on_control"])
@pytest.mark.parametrize("key, factor", [(Qt.Key_Up, 0.5), (Qt.Key_Down, 2.0)],
                         ids=["shift_up", "shift_down"])
def test_shift_up_and_down_zoom_about_the_pointer_over_the_trace(
        private_settings, qtbot, free_mouse, tmp_path, measure, key, factor):
    window, tv, doc_id = _window(qtbot, tmp_path, measure=measure)
    vb = _vb(tv, doc_id)
    e0, e1 = tv.full_x_extent()
    c = 0.5 * (e0 + e1)
    tv.zoom_x_to(c - 0.2, c + 0.2)
    settle()
    scene_pos = _cross_onto(tv, vb, frac=0.2)        # a fifth of the way across the plot
    tv.setFocus(Qt.MouseFocusReason)                 # as a click on the trace gives it
    settle()
    x_before = float(vb.mapSceneToView(scene_pos).x())
    lo, hi = (float(v) for v in vb.viewRange()[0])

    QTest.keyClick(tv, key, Qt.ShiftModifier)
    settle()

    lo2, hi2 = (float(v) for v in vb.viewRange()[0])
    assert hi2 - lo2 == pytest.approx((hi - lo) * factor, rel=1e-6)
    x_after = float(vb.mapSceneToView(scene_pos).x())
    assert x_after == pytest.approx(x_before, abs=(hi - lo) * 1e-3), (
        f"the time under the pointer moved by {x_after - x_before:+.4f} s "
        f"(pointer_data_x()={tv.pointer_data_x()})")
    close_window(window)


# ================================================================================================
# the target is the one the keys act on, and it follows the selection
# ================================================================================================

def test_the_key_target_is_the_resolved_current_channel_not_the_voltage_row(
        private_settings, qtbot, free_mouse, tmp_path):
    """Selecting an ABF's command voltage is still an analysis of its current: the trace takes
    the keys on the current's pane, where C, S, F and D will act, and not on the voltage's."""
    window, tv, doc_id = _window(qtbot, tmp_path, detect=False, with_voltage=True)
    assert window._analysis_panel.current_target() == (doc_id, 1)
    assert tv.annotation_target() == (doc_id, 1)
    tv.clearFocus()
    settle()

    _cross_onto(tv, _vb(tv, doc_id, 0))
    assert not tv.hasFocus()                         # the voltage pane: no keys act there
    _cross_onto(tv, _vb(tv, doc_id, 1))
    assert tv.hasFocus()
    close_window(window)


def test_the_key_target_follows_the_selection_and_goes_with_its_recording(
        private_settings, qtbot, free_mouse, tmp_path):
    window, tv, doc_a = _window(qtbot, tmp_path, detect=False)
    doc_b = _open(window, tmp_path, "s9_08_b.dat", detect=False)     # opened and selected
    assert tv.annotation_target() == (doc_b, 0)
    tv.clearFocus()
    _cross_onto(tv, _vb(tv, doc_b), frac=0.3)
    assert tv.hasFocus()
    sample = tv.pointer_sample()
    assert sample is not None and sample[:2] == (doc_b, 0)

    window._close_document(doc_b)                    # the Files dock's Close button
    settle()
    # The survivor is drawn, and nothing is selected: the keys have no target until a
    # selection names one.
    assert tv.annotation_target() is None
    assert window._analysis_panel.current_target() is None
    tv.clearFocus()
    _cross_onto(tv, _vb(tv, doc_a))
    assert not tv.hasFocus()

    window._on_selection(doc_a, 0)                   # a click on its row
    settle()
    assert tv.annotation_target() == (doc_a, 0)
    tv.clearFocus()
    _cross_onto(tv, _vb(tv, doc_a), frac=0.6)
    assert tv.hasFocus()

    window._close_document(doc_a)                    # the last one
    settle()
    assert tv.annotation_target() is None and not tv.panes()
    close_window(window)
