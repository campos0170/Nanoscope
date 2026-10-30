"""S9-03 — The pointer crossing the trace never takes the keyboard from the docked Event inspector.

MEASURED DEFECT (sweep 9, a six-event synthetic recording, the Event inspector docked beside the
trace, Measure on). ``MainWindow._wire`` registered ``self._inspector`` as the trace's key owner:
the right-hand *Inspector* panel, with its Calibration, File Info and Analysis pages. The docked
Event inspector lives in a dock of its own, ``self._inspector_dock``, tabbed beside that one, so
the focus-follows-pointer rule (``TraceView._focus_holder_is_free``) guarded the wrong dock. With
the inspector's waveform focused and its curation keys live (R had just rejected the event on
screen), the pointer moving across the trace handed the trace the keyboard
(``MouseFocusReason``): A no longer reached the inspector, and with a mask standing on the trace
the label hotkey ``d`` (type=dna) became the trace's D and filed the mask as a hand-made event in
the Schmitt table (6 -> 7 rows) while the inspector's event got no label -- the case design 6.1
names. The same registration did the opposite inside the Inspector panel: a push button there, or
the panel's scroll area after a click on empty space (``InspectorScrollArea``, StrongFocus on every
style), kept the keyboard when the pointer arrived, so C, S, F and D were dead after pointing.

THE FIX. The Event inspector's dock registers as the trace's key owner where it is built
(``MainWindow._inspector_dock_widget``). It is built once and kept, so the one registration holds
through every later dock, a close and reopen, and a trip to the inspector's own window and back.
The Inspector panel is no longer registered: its keyless holders (buttons, a closed combo box, the
scroll area) give way to the pointer like any other, and its text fields, spin boxes and tables
keep their keys by type, as they do anywhere in the window. In its own window the inspector needs
no registration: the rule acts only while the main window is active.

These tests drive the real MainWindow: the inspector docked as its toolbar's Dock beside trace
does (``set_curation_host`` with ``present=True``), closed with its dock's close button and
reopened with Analysis ▸ Event inspector…, sent to its own window and back with that toolbar
button; focus put where the app puts it (``focus_curation_surface``) or where a click puts it;
the pointer crossing onto the Measure pane through real mouse moves on the trace's viewport; and
A, R and the label hotkey as key presses to whatever holds the keyboard. They assert who holds it
and what the keys did to the table.
"""

from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.labels import LabelVocabulary
from nanoscope.analysis.scoring import hand_made_mask

from ._inspector_hosting import (  # noqa: F401  (fixtures are used by name)
    accepted, build_window, close_window, docked, private_settings, settle)
from .test_trace_rig import free_mouse, hover  # noqa: F401


# ================================================================================================
# helpers
# ================================================================================================

def _arm(window, qtbot):
    """Measure on with no mask, the main window active: the trace has its target pane."""
    window._analysis_panel.measure_panel.set_active(True, seed=False)
    settle()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=5_000)
    tv = window._trace_view
    target = tv.measure_target()
    assert target is not None
    return tv, tv.pane_for_layer(f"{target[0]}#{target[1]}").plot.vb


def _cross_onto(tv, vb) -> None:
    """The hand moves from the left axis onto the Measure pane's plot area."""
    r = vb.sceneBoundingRect()
    hover(tv, QtCore.QPointF(tv.mapFromScene(QtCore.QPointF(r.left() - 12.0, r.center().y()))))
    hover(tv, QtCore.QPointF(tv.mapFromScene(r.center())))
    settle(10)


def _inside(widget, holder) -> bool:
    return holder is not None and (holder is widget or widget.isAncestorOf(holder))


def _event_ids(window) -> list[int]:
    return [int(i) for i in window._analysis_panel.current_detection().payload.df["event_id"]]


def _bring_back(window, route: str):
    """Dock the inspector, then take it along ``route``; return (inspector, dock)."""
    insp, dock = docked(window)
    if route == "closed_and_reopened":
        dock.close()                                   # the dock's close button
        settle(50)
        assert dock.isHidden()
        window._act_event_inspector.trigger()          # Analysis ▸ Event inspector…
        settle(50)
    elif route == "back_from_its_own_window":
        insp._host_btn.click()                         # Own window
        settle(50)
        assert insp.isWindow()
        insp._host_btn.click()                         # Dock beside trace
        settle(50)
    assert dock.widget() is insp and not insp.isWindow() and not dock.isHidden()
    return insp, dock


# ================================================================================================
# the docked Event inspector keeps the keyboard
# ================================================================================================

@pytest.mark.parametrize("route", ["docked", "closed_and_reopened", "back_from_its_own_window"])
def test_a_still_accepts_in_the_docked_inspector_after_the_pointer_crosses_the_trace(
        private_settings, qtbot, free_mouse, route):
    window = build_window(qtbot)
    insp, _dock = _bring_back(window, route)
    tv, vb = _arm(window, qtbot)
    ids = _event_ids(window)
    insp.show_event(ids[2])
    insp.focus_curation_surface()                      # what MainWindow does once it is shown
    settle()
    assert _inside(insp, QtWidgets.QApplication.focusWidget())
    QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key_R)
    settle()
    assert not accepted(window, ids[2])                # curating: R rejected the event on screen

    _cross_onto(tv, vb)

    holder = QtWidgets.QApplication.focusWidget()
    assert _inside(insp, holder), (
        f"the crossing moved the keyboard to {type(holder).__name__} "
        f"(trace has focus: {tv.hasFocus()})")
    QTest.keyClick(holder, Qt.Key_A)
    settle()
    assert accepted(window, ids[2])
    close_window(window)


def test_a_label_hotkey_on_d_labels_the_inspector_event_and_adds_nothing(
        private_settings, qtbot, free_mouse):
    """Design 6.1's own example: a mask stands on the trace, and ``d`` is a label hotkey."""
    window = build_window(qtbot)
    vocab = LabelVocabulary()
    vocab.add("type", "dna", hotkey="d")
    window._label_vocab = vocab                        # what the Event labels dialog's OK does
    window._push_vocabulary()
    insp, _dock = docked(window)
    tv, vb = _arm(window, qtbot)
    doc_id, ch = tv.measure_target()
    tv.set_measure_region(tv.x_for_index(doc_id, ch, 35_000), tv.x_for_index(doc_id, ch, 35_300))
    settle()
    assert tv.measure_bounds() is not None
    ids = _event_ids(window)
    n0 = len(ids)
    insp.show_event(ids[1])
    insp.focus_curation_surface()
    settle()

    _cross_onto(tv, vb)
    QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key_D)
    settle(50)

    df = window._analysis_panel.current_detection().payload.df
    assert len(df) == n0 and int(hand_made_mask(df).sum()) == 0, (
        f"d filed the trace's mask as a hand-made event (rows {n0} -> {len(df)})")
    flag = str(df.loc[df["event_id"] == ids[1], "flag"].iloc[0])
    assert "type=dna" in flag, f"the inspector's event was not labelled: flag={flag!r}"
    assert tv.measure_bounds() is not None             # ...and the mask is still standing
    close_window(window)


def test_the_inspector_in_its_own_window_keeps_its_keys_too(private_settings, qtbot, free_mouse):
    """Guard for the other host: the rule acts only while the MAIN window is active, so the
    inspector's own window needs no registration."""
    window = build_window(qtbot)
    insp, _dock = docked(window)
    tv, vb = _arm(window, qtbot)
    insp._host_btn.click()                             # Own window
    settle(50)
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is insp, timeout=5_000)
    ids = _event_ids(window)
    insp.show_event(ids[2])
    insp.focus_curation_surface()
    settle()
    QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key_R)
    settle()
    assert not accepted(window, ids[2])

    _cross_onto(tv, vb)

    holder = QtWidgets.QApplication.focusWidget()
    assert _inside(insp, holder) and not tv.hasFocus()
    QTest.keyClick(holder, Qt.Key_A)
    settle()
    assert accepted(window, ids[2])
    close_window(window)


# ================================================================================================
# the Inspector panel is not a key owner: its holders answer by type
# ================================================================================================

@pytest.mark.parametrize("kind", [QtWidgets.QPushButton, QtWidgets.QComboBox],
                         ids=["push_button", "closed_combo"])
def test_a_keyless_holder_in_the_inspector_panel_gives_way_to_the_pointer(
        private_settings, qtbot, free_mouse, kind):
    window = build_window(qtbot)
    tv, vb = _arm(window, qtbot)
    holders = [w for w in window._inspector.findChildren(kind)
               if w.isVisible() and w.isEnabled()
               and not (isinstance(w, QtWidgets.QComboBox) and w.isEditable())]
    assert holders
    holder = holders[0]
    # setFocus, not a click: on a Mac neither takes focus from a click, but Tab lands on both.
    holder.setFocus(Qt.TabFocusReason)
    settle()
    assert QtWidgets.QApplication.focusWidget() is holder

    _cross_onto(tv, vb)

    assert tv.hasFocus(), f"focus stayed on a {type(holder).__name__} in the Inspector panel"
    assert tv.last_focus_reason() == Qt.MouseFocusReason
    close_window(window)


def test_a_click_on_empty_inspector_panel_space_gives_way_to_the_pointer(
        private_settings, qtbot, free_mouse):
    """A click on a label or a frame of the Analysis page gives the keyboard to the panel's
    scroll area (StrongFocus on every style): nothing there takes C, S, F or D."""
    window = build_window(qtbot)
    tv, vb = _arm(window, qtbot)
    area = next(a for a in window._inspector.findChildren(QtWidgets.QScrollArea)
                if a.isVisible() and type(a).__name__ == "InspectorScrollArea")
    viewport = area.viewport()
    target = local = None
    for y in range(5, viewport.height() - 5, 7):
        for x in (viewport.width() - 4, viewport.width() // 2, 6):
            w = viewport.childAt(QtCore.QPoint(x, y))
            if (w is not None and w.focusPolicy() == Qt.NoFocus
                    and isinstance(w, (QtWidgets.QLabel, QtWidgets.QGroupBox, QtWidgets.QFrame))
                    and not isinstance(w, QtWidgets.QAbstractScrollArea)):
                target, local = w, w.mapFrom(viewport, QtCore.QPoint(x, y))
                break
        if target is not None:
            break
    assert target is not None
    QTest.mouseClick(target, Qt.LeftButton, Qt.NoModifier, local)
    settle()
    holder = QtWidgets.QApplication.focusWidget()
    assert isinstance(holder, QtWidgets.QAbstractScrollArea) and _inside(window._inspector, holder)

    _cross_onto(tv, vb)

    assert tv.hasFocus(), f"focus stayed on {type(holder).__name__} in the Inspector panel"
    close_window(window)


@pytest.mark.parametrize("page, kind", [
    ("Calibration", QtWidgets.QLineEdit),
    ("Analysis", QtWidgets.QAbstractSpinBox),
    ("File Info", QtWidgets.QAbstractItemView),
], ids=["text_field", "spin_box", "table"])
def test_the_inspector_panels_fields_and_tables_keep_their_keys(
        private_settings, qtbot, free_mouse, page, kind):
    """Guard: unregistering the panel takes nothing from the holders that own bare keys. They
    keep them by type, as the same widgets would anywhere in the window."""
    window = build_window(qtbot)
    tv, vb = _arm(window, qtbot)
    tabs = window._inspector_tabs
    tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == page))
    settle()
    field = next(w for w in window._inspector.findChildren(kind)
                 if w.isVisible() and w.isEnabled() and not isinstance(w, QtWidgets.QHeaderView))
    field.setFocus(Qt.MouseFocusReason)
    settle()
    holder = QtWidgets.QApplication.focusWidget()
    assert _inside(field, holder)

    _cross_onto(tv, vb)

    assert QtWidgets.QApplication.focusWidget() is holder and not tv.hasFocus()
    close_window(window)
