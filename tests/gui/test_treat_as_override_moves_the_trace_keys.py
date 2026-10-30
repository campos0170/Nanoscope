"""A "Treat this channel as" override that moves the current channel moves the trace keys with it.

MEASURED DEFECT (listed under "Left alone" in 3693928; present since S9-08 gave the trace a key
target with Measure off). A synthetic two-channel recording whose header has the channels the
wrong way round, "IN0" declared current and "IN1" declared voltage, with the events on IN1. The
person fixes it on the Calibration tab: IN1 "Treat this channel as: current", Apply, then IN0
"voltage", Apply. The second Apply moves the analysis to IN1 (``AnalysisPanel._on_calibration_
changed``), Measure panel and all, so C, S, F and D act on IN1. Nothing else followed:
``TraceView.annotation_target()``, the Navigator and the Scan bar stayed on IN0, now a voltage.
With Measure off, how a recording opens, the pointer crossing onto IN1's plot left the keyboard
on the workflow chip (``_StageChip``) and ``pointer_sample()`` was None there, while crossing onto
IN0's plot took the keys; the caption sat on IN0's pane and the Navigator drew IN0. Selecting the
recording again put all of it on IN1. A second current channel re-declared a voltage did the
same with one Apply.

THE FIX. ``MainWindow._follow_analysis_target``, on ``Session.calibrationChanged`` after the
panel's own slot: when the panel's target has moved to another channel of the recording the keys
are on, it points the Navigator, the Scan bar and the trace's keys there
(``_point_at_annotation_target``, the steps ``_retarget_selection`` takes for a selection) and
re-gates Measure, which repaints the caption.

UNCHANGED, AND PINNED BY THE TWO CONTROLS: a mask on the old channel goes, as on a selection (the
Measure panel turns Measure off when its target moves), and an override that leaves the target
where it is moves nothing and keeps the mask.

The window is the real MainWindow at a size inside the offscreen screen (``QApplication.widgetAt``,
which the pointer helpers use, finds nothing beyond it). Rows are selected through
``_on_selection`` (the Files dock's door), the override goes through the Calibration tab's own
combo and Apply button (its sidecar lands beside the tmp_path recording), the pointer through real
mouse moves on the trace's viewport, and the keys are key presses to whatever holds the keyboard.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

from ._inspector_hosting import (  # noqa: F401  (fixtures are used by name)
    FS, N, SPANS, build_window, close_window, private_settings, settle, signal)
from .test_trace_rig import free_mouse, hover, point_at  # noqa: F401

#: The event the C tests catch: the widest of the six.
EVENT = SPANS[2]


# ================================================================================================
# builders
# ================================================================================================

def _open(qtbot, tmp_path, name: str, infos, channels, row: int):
    """The real window on a two-channel recording, row ``row`` selected, the window active."""
    window = build_window(qtbot, size=(780, 700), detection=False)
    window.move(0, 0)
    settle()
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=tuple(infos))
    doc = TraceDocument(meta=meta, channels=tuple(channels))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, row)
    settle()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=5_000)
    return window, window._trace_view, doc_id


def _treat_as(window, row: int, kind: ChannelKind, units: str) -> None:
    """Select ``row``, then "Treat this channel as" ``kind`` in ``units``, and Apply."""
    doc_id = window._current_selection[0]
    window._on_selection(doc_id, row)
    settle()
    cp = window._calibration_panel
    cp._treat_as.setCurrentIndex(cp._treat_as.findData(kind))
    cp._units.setText(units)
    cp._btn_apply.click()
    settle(50)
    assert cp.status_text().startswith("Applied"), cp.status_text()
    assert window._session.effective_kind(doc_id, row) is kind


def _swapped(qtbot, tmp_path):
    """The header has the channels the wrong way round: "IN0" (declared current) carries the
    command, "IN1" (declared voltage) the events. Fixed in two Applies, as the Calibration tab
    allows, ending with IN0's row selected; the keys should end on IN1."""
    infos = (ChannelInfo(index=0, name="IN0", raw_units="pA", kind=ChannelKind.CURRENT),
             ChannelInfo(index=1, name="IN1", raw_units="mV", kind=ChannelKind.VOLTAGE))
    window, tv, doc_id = _open(qtbot, tmp_path, "swapped.dat", infos,
                               (np.full(N, -100.0, dtype=np.float32), signal()), row=1)
    _treat_as(window, 1, ChannelKind.CURRENT, "pA")
    assert window._analysis_panel.current_target() == (doc_id, 0)    # IN0 still answers first
    window._on_selection(doc_id, 0)
    settle()
    return SimpleNamespace(window=window, tv=tv, doc_id=doc_id, old=0, new=1, row=0,
                           fix=lambda: _treat_as(window, 0, ChannelKind.VOLTAGE, "mV"))


def _second_current(qtbot, tmp_path):
    """Two declared current channels, the second selected; then it is re-declared a voltage."""
    infos = (ChannelInfo(index=0, name="I0", raw_units="pA", kind=ChannelKind.CURRENT),
             ChannelInfo(index=1, name="I1", raw_units="pA", kind=ChannelKind.CURRENT))
    window, tv, doc_id = _open(qtbot, tmp_path, "two.dat", infos, (signal(), signal()), row=1)
    return SimpleNamespace(window=window, tv=tv, doc_id=doc_id, old=1, new=0, row=1,
                           fix=lambda: _treat_as(window, 1, ChannelKind.VOLTAGE, "mV"))


SHAPES = {"swapped_header": _swapped, "second_current_declared_voltage": _second_current}


def _vb(tv, doc_id: str, channel: int):
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


def _drop_focus(tv) -> None:
    focus = QtWidgets.QApplication.focusWidget()
    if focus is not None:
        focus.clearFocus()
    settle()
    assert not tv.hasFocus()


def _zoom_onto_event(tv, doc_id: str, channel: int) -> None:
    s0, s1 = EVENT
    tv.zoom_x_to(tv.x_for_index(doc_id, channel, s0 - 3_000),
                 tv.x_for_index(doc_id, channel, s1 + 3_000))
    settle()


def _caught(window, tv) -> None:
    s0, s1 = EVENT
    bounds = tv.measure_bounds()
    assert bounds is not None, window.statusBar().currentMessage()
    assert abs(bounds[0] - s0) <= 10 and abs(bounds[1] - s1) <= 10, bounds
    assert window.statusBar().currentMessage().startswith("Caught")


def _manual_rows(window, doc_id: str, channel: int) -> int:
    res = window._session.results.get(doc_id, channel, MANUAL_MODULE_ID)
    return 0 if res is None else len(res.payload.df)


def _cursor(vb):
    return vb.cursor().shape() if vb.hasCursor() else None


# ================================================================================================
# the keys follow the override to the new current channel
# ================================================================================================

@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_keys_navigator_and_scan_bar_follow_the_override(
        private_settings, qtbot, free_mouse, tmp_path, shape):
    s = SHAPES[shape](qtbot, tmp_path)
    window, tv, doc_id = s.window, s.tv, s.doc_id
    assert tv.annotation_target() == (doc_id, s.old)

    s.fix()

    new = (doc_id, s.new)
    assert window._analysis_panel.current_target() == new           # what C, S, F, D act on
    assert tv.annotation_target() == new, (
        f"the keys stayed on channel {tv.annotation_target()[1]}, which is no longer the current")
    assert window._navigator._target == new
    assert window._navigator._overview_layer().channel == s.new
    assert window._pan_scrollbar._target == new                     # G and Shift+G step it
    # ...which is exactly where selecting the recording again puts them.
    window._on_selection(doc_id, s.row)
    settle()
    assert (tv.annotation_target(), window._navigator._target,
            window._pan_scrollbar._target) == (new, new, new)
    close_window(window)


def test_the_pointer_takes_the_keys_on_the_new_current_pane_and_c_catches_there(
        private_settings, qtbot, free_mouse, tmp_path):
    s = _swapped(qtbot, tmp_path)
    window, tv, doc_id = s.window, s.tv, s.doc_id
    s.fix()
    measure = window._analysis_panel.measure_panel
    assert not measure.is_active()
    _zoom_onto_event(tv, doc_id, s.new)
    middle = (EVENT[0] + EVENT[1]) // 2

    _drop_focus(tv)
    _cross_onto(tv, _vb(tv, doc_id, s.old), x=tv.x_for_index(doc_id, s.old, middle))
    assert not tv.hasFocus(), "the keys went live over the old channel, now a voltage"

    _drop_focus(tv)
    new_vb = _vb(tv, doc_id, s.new)
    _cross_onto(tv, new_vb, x=tv.x_for_index(doc_id, s.new, middle))
    holder = QtWidgets.QApplication.focusWidget()
    assert holder is tv, (
        f"the pointer is on the new current channel but the keyboard is on "
        f"{type(holder).__name__}: C, S, F, G and E are dead there")
    assert tv.last_focus_reason() == Qt.MouseFocusReason
    assert _cursor(new_vb) == Qt.CrossCursor
    sample = tv.pointer_sample()
    assert sample is not None and sample[:2] == (doc_id, s.new)

    QTest.keyClick(holder, Qt.Key_C)
    settle()
    assert measure.is_active()
    _caught(window, tv)
    close_window(window)


def test_shift_up_zooms_about_the_pointer_on_the_new_current_pane(
        private_settings, qtbot, free_mouse, tmp_path):
    s = _swapped(qtbot, tmp_path)
    window, tv, doc_id = s.window, s.tv, s.doc_id
    s.fix()
    assert not window._analysis_panel.measure_panel.is_active()
    vb = _vb(tv, doc_id, s.new)
    e0, e1 = tv.full_x_extent()
    c = 0.5 * (e0 + e1)
    tv.zoom_x_to(c - 0.2, c + 0.2)
    settle()
    scene_pos = _cross_onto(tv, vb, frac=0.2)        # a fifth of the way across the plot
    tv.setFocus(Qt.MouseFocusReason)                 # as a click on the trace gives it
    settle()
    x_before = float(vb.mapSceneToView(scene_pos).x())
    lo, hi = (float(v) for v in vb.viewRange()[0])

    QTest.keyClick(tv, Qt.Key_Up, Qt.ShiftModifier)
    settle()

    lo2, hi2 = (float(v) for v in vb.viewRange()[0])
    assert hi2 - lo2 == pytest.approx((hi - lo) * 0.5, rel=1e-6)
    x_after = float(vb.mapSceneToView(scene_pos).x())
    assert x_after == pytest.approx(x_before, abs=(hi - lo) * 1e-3), (
        f"the time under the pointer moved by {x_after - x_before:+.4f} s "
        f"(pointer_data_x()={tv.pointer_data_x()})")
    close_window(window)


def test_the_caption_and_the_arrow_move_to_the_new_current_pane(
        private_settings, qtbot, free_mouse, tmp_path):
    s = _swapped(qtbot, tmp_path)
    window, tv, doc_id = s.window, s.tv, s.doc_id
    _drop_focus(tv)
    qtbot.waitUntil(lambda: tv._target_label is not None
                    and tv._target_label.parentItem() is _vb(tv, doc_id, s.old), timeout=2_000)

    s.fix()

    new_vb, old_vb = _vb(tv, doc_id, s.new), _vb(tv, doc_id, s.old)
    qtbot.waitUntil(lambda: tv._target_label is not None
                    and tv._target_label.parentItem() is new_vb, timeout=2_000)
    assert tv.target_label_text().startswith("→ ")
    # The keys are not live (the keyboard is elsewhere): the arrow says so on the new pane, and
    # the old pane keeps its own cursor.
    assert _cursor(new_vb) == Qt.ArrowCursor
    assert _cursor(old_vb) is None
    close_window(window)


# ================================================================================================
# controls: what the override already did, and does not do
# ================================================================================================

def test_a_mask_on_the_old_channel_goes_as_on_a_selection(
        private_settings, qtbot, free_mouse, tmp_path):
    """Measure on with a caught mask on the old current: the override turns Measure off and the
    mask goes with it, as selecting another channel does. D then has no mask to add."""
    s = _second_current(qtbot, tmp_path)
    window, tv, doc_id = s.window, s.tv, s.doc_id
    _zoom_onto_event(tv, doc_id, s.old)
    middle = (EVENT[0] + EVENT[1]) // 2
    window._trace_keys.catch(at=(doc_id, s.old, float(middle), float(tv.target_px_samples())))
    settle()
    _caught(window, tv)

    s.fix()

    assert not window._analysis_panel.measure_panel.is_active()
    assert tv.measure_bounds() is None
    tv.setFocus(Qt.MouseFocusReason)
    settle()
    QTest.keyClick(tv, Qt.Key_D)
    settle()
    assert window.statusBar().currentMessage().startswith("No mask")
    assert _manual_rows(window, doc_id, s.old) == 0
    assert _manual_rows(window, doc_id, s.new) == 0
    close_window(window)


def test_an_override_that_leaves_the_target_moves_nothing(
        private_settings, qtbot, free_mouse, tmp_path):
    """The command row selected (Analysis on the current), a mask caught, then the command
    channel declared a current as well: Analysis stays where it was, so the keys, the Navigator,
    the Scan bar and the mask stay too, and D adds the mask to the channel it was caught on."""
    infos = (ChannelInfo(index=0, name="V", raw_units="mV", kind=ChannelKind.VOLTAGE),
             ChannelInfo(index=1, name="I", raw_units="pA", kind=ChannelKind.CURRENT))
    window, tv, doc_id = _open(qtbot, tmp_path, "stands.dat", infos,
                               (np.full(N, -100.0, dtype=np.float32), signal()), row=0)
    target = (doc_id, 1)
    assert window._analysis_panel.current_target() == target
    _zoom_onto_event(tv, doc_id, 1)
    middle = (EVENT[0] + EVENT[1]) // 2
    window._trace_keys.catch(at=(doc_id, 1, float(middle), float(tv.target_px_samples())))
    settle()
    _caught(window, tv)
    bounds = tv.measure_bounds()

    _treat_as(window, 0, ChannelKind.CURRENT, "pA")

    assert window._analysis_panel.current_target() == target
    assert tv.annotation_target() == target
    assert window._navigator._target == target and window._pan_scrollbar._target == target
    assert window._analysis_panel.measure_panel.is_active()
    assert tv.measure_bounds() == bounds
    tv.setFocus(Qt.MouseFocusReason)
    settle()
    QTest.keyClick(tv, Qt.Key_D)
    settle()
    assert _manual_rows(window, doc_id, 1) == 1, window.statusBar().currentMessage()
    close_window(window)
