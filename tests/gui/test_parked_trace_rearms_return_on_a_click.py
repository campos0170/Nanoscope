"""A click on the trace wakes Return, Enter and Esc after the window parked the keyboard there.

MEASURED DEFECT (found by the cross-group review of the sweep-9 fixes; present since e2c92cb).
The trace keys' aliases act only when the trace got focus from the mouse:
``TraceKeyController._alias`` reads ``TraceView.last_focus_reason``, which only
``TraceView.focusInEvent`` wrote. Hiding the Results dock with the keyboard in its table parks focus
on the trace with ``OtherFocusReason`` (``MainWindow._park_results_focus``), on purpose: a habitual
Return right after closing a dock must not add. But a click on the trace then gave no focus-in --
the trace already had focus -- so the reason stayed ``OtherFocusReason``. Measured after the click,
with a mask caught on the event at samples 28000-28700: Return filed nothing and Esc left the mask
standing, while D filed it as event 0; in draw mode, Esc left a draw going. USER_GUIDE section 8
says the two act after a click. They came back only once focus left the trace and returned.

THE FIX. ``TraceView.mousePressEvent`` records ``MouseFocusReason`` for a press on a trace that
already has focus: the reason Qt gives with the focus for the same press when the trace has not.

NOT CHANGED, AND WHY: the pointer arriving on a trace that already has focus (the reviewer's
optional second half). The trace grows into a hidden dock's place under a pointer that never moved
toward it. Measured with that half patched in, at 780x700 and 1512x847: with the pointer resting
on the Results panel's header (the rows above its table, 8, 24 and 40 px below its top) and the
panel hidden from the keyboard (Ctrl+Shift+R), the target plot slid under the pointer and a 1 px
twitch armed Return and Esc in all six positions -- the habitual press the park exists to stop.
Resting on the table's own rows, it armed in none, and the panel's close button lands in the
trace's left-axis gutter, 65 px from the plot. D and X stay live either way, and one click wakes
the aliases.

The window is the real MainWindow at a size inside the offscreen screen (``QApplication.widgetAt``
finds nothing beyond it), on a synthetic six-event recording with no detection, so an add files
into Manual. Keys are real key presses to the focused trace and the click a real QTest click. The
pointer for C is stood in for by a patched ``TraceView.pointer_sample`` (as in test_trace_keys), so
no test points at the trace unless pointing is what it is about.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

from ._inspector_hosting import (  # noqa: F401  (fixtures are used by name)
    FS, N, SPANS, build_window, close_window, private_settings, settle, signal)
from .test_trace_rig import drag_move, free_mouse, hover, lerp, point_at  # noqa: F401

#: The event every test catches: the widest of the six.
EVENT = SPANS[2]


# ================================================================================================
# builders
# ================================================================================================

def _window(qtbot, tmp_path):
    """The real MainWindow on the six-event recording, selected as the Files dock does, the view
    zoomed onto ``EVENT``; the window active and the Results dock open."""
    window = build_window(qtbot, size=(780, 700), detection=False)
    window.move(0, 0)
    settle()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "parked.dat", format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(signal(),))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    settle()
    tv = window._trace_view
    s0, s1 = EVENT
    tv.zoom_x_to(tv.x_for_index(doc_id, 0, s0 - 3_000), tv.x_for_index(doc_id, 0, s1 + 3_000))
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=5_000)
    settle()
    assert window._results_dock.isVisible()
    return window, tv, doc_id


def _park(window, tv) -> None:
    """The keyboard in the Results table, then the Results toggle (Ctrl+Shift+R) hides the dock:
    the window parks the keyboard on the trace, not on a pointer's behalf."""
    table = window._analysis_panel._table.view
    table.setFocus(Qt.MouseFocusReason)
    settle()
    assert table.hasFocus()
    window._act_toggle_results.trigger()
    settle()
    assert window._results_dock.isHidden()
    assert tv.hasFocus() and tv.last_focus_reason() == Qt.OtherFocusReason


def _vb(tv, doc_id: str):
    return tv.pane_for_layer(f"{doc_id}#0").plot.vb


def _at(tv, doc_id: str, sample: int) -> QtCore.QPointF:
    """The viewport position of ``sample`` at mid-height of the target plot."""
    return point_at(tv, _vb(tv, doc_id), tv.x_for_index(doc_id, 0, sample))


def _key(tv, key, modifier=Qt.NoModifier) -> None:
    assert QtWidgets.QApplication.focusWidget() is tv
    QTest.keyClick(tv, key, modifier)
    settle()


def _catch(window, tv, doc_id: str) -> None:
    """C over the middle of ``EVENT``, through a stood-in pointer."""
    s0, s1 = EVENT
    px = float(tv.target_px_samples())
    tv.pointer_sample = lambda: (doc_id, 0, float((s0 + s1) // 2), px)
    _key(tv, Qt.Key_C)
    bounds = tv.measure_bounds()
    assert bounds is not None, window.statusBar().currentMessage()
    assert abs(bounds[0] - s0) <= 10 and abs(bounds[1] - s1) <= 10, bounds


def _rows(window, doc_id: str) -> int:
    res = window._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return 0 if res is None else len(res.payload.df)


# ================================================================================================
# the fix
# ================================================================================================

def test_a_click_on_the_parked_trace_wakes_return_and_esc(
        private_settings, qtbot, free_mouse, tmp_path):
    window, tv, doc_id = _window(qtbot, tmp_path)
    _park(window, tv)
    _catch(window, tv, doc_id)
    _key(tv, Qt.Key_Return)
    assert _rows(window, doc_id) == 0                # parked: Return is still asleep

    # The click USER_GUIDE section 8 asks for, on the baseline beside the event.
    QTest.mouseClick(tv.viewport(), Qt.LeftButton, Qt.NoModifier,
                     _at(tv, doc_id, EVENT[0] - 1_500).toPoint())
    settle()
    assert tv.hasFocus() and tv.last_focus_reason() == Qt.MouseFocusReason

    _key(tv, Qt.Key_Escape)
    assert tv.measure_bounds() is None and tv.measure_empty()
    assert window._analysis_panel.measure_panel.is_active()
    assert _rows(window, doc_id) == 0                # Esc discarded; it added nothing

    _catch(window, tv, doc_id)                       # (C refuses an event already filed)
    _key(tv, Qt.Key_Return)
    assert _rows(window, doc_id) == 1, window.statusBar().currentMessage()
    assert window.statusBar().currentMessage().startswith("Added event")
    assert tv.measure_bounds() is None               # an add clears the mask
    close_window(window)


def test_esc_cancels_a_draw_started_on_the_parked_trace(
        private_settings, qtbot, free_mouse, tmp_path):
    """The press that starts a draw is a click on the trace too."""
    window, tv, doc_id = _window(qtbot, tmp_path)
    _park(window, tv)
    _key(tv, Qt.Key_E)                               # draw mode: E is a key, not an alias
    assert tv.draw_mode()
    p0 = _at(tv, doc_id, EVENT[0] - 600)
    p1 = _at(tv, doc_id, EVENT[1] + 600)
    hover(tv, p0)                                    # the scene hands a drag only to what it saw
    QTest.mousePress(tv.viewport(), Qt.LeftButton, Qt.NoModifier, p0.toPoint())
    for t in (0.25, 0.5, 0.75, 1.0):
        drag_move(tv, lerp(p0, p1, t))
    assert tv.draw_in_progress()

    _key(tv, Qt.Key_Escape)
    assert not tv.draw_in_progress()
    QTest.mouseRelease(tv.viewport(), Qt.LeftButton, Qt.NoModifier, p1.toPoint())
    settle()
    assert tv.measure_bounds() is None and tv.draw_mode()   # nothing drawn; the mode stays
    close_window(window)


# ================================================================================================
# controls: the park still does its job
# ================================================================================================

def test_a_bare_park_leaves_return_and_esc_asleep_and_d_awake(
        private_settings, qtbot, free_mouse, tmp_path):
    """No click and no pointing: what the park exists for."""
    window, tv, doc_id = _window(qtbot, tmp_path)
    _park(window, tv)
    _catch(window, tv, doc_id)
    bounds = tv.measure_bounds()

    _key(tv, Qt.Key_Return)
    _key(tv, Qt.Key_Return, Qt.ShiftModifier)
    _key(tv, Qt.Key_Escape)

    assert _rows(window, doc_id) == 0
    assert tv.measure_bounds() == bounds             # Esc left the mask alone
    assert tv.last_focus_reason() == Qt.OtherFocusReason
    _key(tv, Qt.Key_D)                               # the keys themselves are live
    assert _rows(window, doc_id) == 1, window.statusBar().currentMessage()
    close_window(window)


def test_a_twitch_over_a_trace_that_grew_under_the_pointer_wakes_nothing(
        private_settings, qtbot, free_mouse, tmp_path):
    """The pointer rests on the Results panel's header; Ctrl+Shift+R hides the panel and the
    target plot slides under the pointer, which never moved toward the trace. A 1 px twitch
    there is not pointing at the trace (see the module docstring)."""
    window, tv, doc_id = _window(qtbot, tmp_path)
    body = window._results_dock.widget()
    panel = QtCore.QRect(body.mapToGlobal(QtCore.QPoint(0, 0)), body.size())
    _park(window, tv)
    vp = tv.viewport()
    plot = tv.mapFromScene(_vb(tv, doc_id).sceneBoundingRect()).boundingRect()
    under = panel.intersected(QtCore.QRect(vp.mapToGlobal(plot.topLeft()), plot.size()))
    assert not under.isEmpty(), "the target plot no longer grows into the Results panel's place"
    rest = QtCore.QPointF(vp.mapFromGlobal(under.center()))

    hover(tv, rest)
    hover(tv, rest + QtCore.QPointF(1.0, 0.0))
    assert tv.pointer_data_x() is not None           # the pointer is on the target plot
    _catch(window, tv, doc_id)
    bounds = tv.measure_bounds()
    _key(tv, Qt.Key_Return)
    _key(tv, Qt.Key_Escape)

    assert _rows(window, doc_id) == 0 and tv.measure_bounds() == bounds
    assert tv.last_focus_reason() == Qt.OtherFocusReason
    close_window(window)
