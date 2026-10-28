"""S8-KEYS2 - A / R / Space take the key from the control that has the keyboard.

MEASURED DEFECT (sweep 8, keyboard lens, the owner's -600 mV detection, 317 events). The Event
inspector and the Event gallery bind A, R, Space (and the label digits, T, the page keys) as
``Qt.WidgetWithChildrenShortcut``, which fires wherever the focus sits inside the window. Qt
evaluates a shortcut BEFORE ``keyPressEvent`` and the focused widget keeps a key only if it
accepts the ``ShortcutOverride`` Qt sends first; a button does not, and neither does the list
of an open dropdown (a separate top-level ``Qt.Popup`` window that the shortcut context walks
straight through to its owner). So:

  A. Tab to "Show in trace" / "Gallery..." / "Reject all" and press Space: the button was never
     pressed and the event on screen was TOGGLED (accepted True -> False).
  B. Open the Filter dropdown and type r / a / Space: the filter did not change and the event
     on screen was rejected / accepted / toggled instead.
  C. Gallery: Space on a Tab-focused "Accept selected" / "Reject selected" flipped the selected
     tiles (accepted ones became REJECTED under "Accept selected"), and "r" in the open Show
     dropdown rejected the selection instead of choosing "Rejected only". The same open list
     lost its PgDn / End to the page keys, which turned the page and dropped the selection.
     An open menu is the same popup: "a" typed into the gallery's Select menu accepted every
     selected tile.
  D. The other half of the same conflict: the controls the inspector had taken out of the Tab
     chain so that Space could not hit them -- Advance after decision, Dock beside trace, the
     Label combo with Apply and Clear, Details -- had no keyboard route at all (the gallery's
     Advance after decision box neither).

THE FIX. ``gui/curation_keys.CurationKeyGuard`` accepts the ``ShortcutOverride`` for the key
the control that holds the keyboard uses itself: Space for a button or check box, and every bare
key for an open dropdown list or menu. Both windows install it (``_key_guard``). Everywhere
else the curation keys are untouched -- on the trace, the tile view, the window background, a
closed combo box that merely holds the focus, a button for A / R. D is the policy half: the
controls take Tab focus again (``Qt.TabFocus``, not ``StrongFocus``: a mouse click on them
still takes no focus, as before).

WHAT THESE TESTS DRIVE. The real MainWindow with a real detection, the inspector in both hosts
(own window, docked) and the gallery; keys are delivered the way the platform delivers them
(``QTest.keyClick`` on the active top-level's QWindow, so the shortcut map and the
ShortcutOverride run before the focus widget sees the key), Tab walks are real Tab presses, and
the assertions are what the user sees: which event is accepted, which signal the button emitted,
which entry the dropdown highlights, which filter is chosen.

Tests marked GUARD pass on the unfixed code by construction; they pin gestures this change must
not disturb. Menus are asked directly (a synthetic ShortcutOverride sent to each QMenu): the
offscreen platform deactivates a window the moment a menu pops up over it and closes the menu
again, so an open menu cannot be driven reliably there; dropdown lists, which it does keep open,
are driven for real.
"""

from __future__ import annotations

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    accepted, build_window, close_window, docked, private_settings, settle)

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.labels import LabelVocabulary

# ================================================================================================
# builders and a keyboard that behaves like the platform's
# ================================================================================================

def _frame(window):
    return window._analysis_panel.current_detection().payload.df


def _ids(window) -> list[int]:
    return [int(i) for i in _frame(window)["event_id"]]


def _n_accepted(window) -> int:
    return int(_frame(window)["accepted"].sum())


def _flag(window, event_id: int) -> str:
    df = _frame(window)
    return str(df.loc[df["event_id"] == event_id, "flag"].iloc[0])


def _open_inspector(qtbot, host: str, *, labels: int = 1):
    window = build_window(qtbot)
    vocab = LabelVocabulary()
    for value in ("translocation", "junk")[:labels]:
        vocab.add("type", value)
    window._label_vocab = vocab
    window._push_vocabulary()
    if host == "dock":
        insp, _dock = docked(window)
    else:
        window._open_event_inspector()
        insp = window._event_inspector
        settle()
    return window, insp


def _activate(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Make ``widget``'s top-level the active window and return it."""
    top = widget.window()
    top.show()
    top.raise_()
    top.activateWindow()
    for _ in range(40):
        settle(10)
        if QtWidgets.QApplication.activeWindow() is top:
            break
    assert QtWidgets.QApplication.activeWindow() is top, "setup: the window never became active"
    return top


def _press(top: QtWidgets.QWidget, key, mods=Qt.NoModifier) -> None:
    """One key press + release delivered the way the platform delivers it: to the ACTIVE
    QWindow, so the shortcut map and ShortcutOverride run before the focus widget sees it."""
    QTest.keyClick(top.windowHandle(), key, mods)
    settle(30)


def _focus_name() -> str:
    f = QtWidgets.QApplication.focusWidget()
    if f is None:
        return "nothing"
    text = f.text() if hasattr(f, "text") and not isinstance(f, QtWidgets.QLineEdit) else ""
    return f"{type(f).__name__} {text!r}".strip()


def _tab_to(top: QtWidgets.QWidget, widget: QtWidgets.QWidget, limit: int = 160) -> int:
    """Real Tab presses until the keyboard is on ``widget``; returns how many it took."""
    for n in range(limit):
        if QtWidgets.QApplication.focusWidget() is widget:
            return n
        _press(top, Qt.Key_Tab)
    pytest.fail(f"Tab never reached {type(widget).__name__} "
                f"{getattr(widget, 'text', lambda: '')()!r} in {limit} presses "
                f"(the keyboard ended on {_focus_name()})")


def _start_of_chain(insp) -> None:
    """The inspector's first Tab stop; every control of the window comes after it."""
    insp._index_spin.setFocus(Qt.TabFocusReason)
    settle()


def _open_dropdown(combo: QtWidgets.QComboBox) -> None:
    combo.setFocus(Qt.OtherFocusReason)
    combo.showPopup()
    for _ in range(40):
        settle(10)
        if QtWidgets.QApplication.activePopupWidget() is not None:
            break
    assert QtWidgets.QApplication.activePopupWidget() is not None, "setup: no dropdown opened"


def _close_dropdown() -> None:
    popup = QtWidgets.QApplication.activePopupWidget()
    if popup is not None:
        popup.close()
        settle(60)


def _highlighted(combo: QtWidgets.QComboBox) -> str:
    """The entry the OPEN list has under its highlight."""
    row = combo.view().currentIndex().row()
    return combo.itemText(row)


# ================================================================================================
# A. a Tab-focused button is pressed by Space  (inspector, both hosts)
# ================================================================================================

@pytest.mark.parametrize("host", ["window", "dock"])
def test_space_on_a_tab_focused_inspector_button_presses_it_and_toggles_nothing(
        private_settings, qtbot, host):
    window, insp = _open_inspector(qtbot, host)
    ids = _ids(window)
    traced: list[int] = []
    gallery_asked: list[int] = []
    insp.traceRequested.connect(traced.append)
    insp.galleryRequested.connect(lambda: gallery_asked.append(1))

    def back_on_event_2() -> QtWidgets.QWidget:
        """A press may have raised the gallery or the main window: close it, come back."""
        if window._event_gallery is not None:
            window._event_gallery.close()
            settle(40)
        insp.show_event(ids[2])
        settle()
        return _activate(insp)

    # "Show in trace": a QPushButton the finder tabbed to.
    top = back_on_event_2()
    assert accepted(window, ids[2])
    _start_of_chain(insp)
    _tab_to(top, insp._trace_btn)
    _press(top, Qt.Key_Space)
    assert traced == [ids[2]], "Space did not press 'Show in trace'"
    assert accepted(window, ids[2]), "Space on 'Show in trace' toggled the event on screen"

    # "Gallery...": same.
    top = back_on_event_2()
    _start_of_chain(insp)
    _tab_to(top, insp._gallery_btn)
    _press(top, Qt.Key_Space)
    assert gallery_asked == [1], "Space did not press 'Gallery...'"
    assert accepted(window, ids[2]), "Space on 'Gallery...' toggled the event on screen"

    # A tool button the toolbar tabs through: "next" moves on and leaves the decision alone.
    top = back_on_event_2()
    _start_of_chain(insp)
    _tab_to(top, insp._next_btn)
    _press(top, Qt.Key_Space)
    assert insp.current_event_id() == ids[3], "Space did not press the next-event button"
    assert accepted(window, ids[2]) and accepted(window, ids[3])

    # "Reject all": the batch button writes the WHOLE column, not one toggled event.
    top = back_on_event_2()
    _start_of_chain(insp)
    _tab_to(top, insp._reject_all_btn)
    _press(top, Qt.Key_Space)
    assert _n_accepted(window) == 0, (
        f"Space on 'Reject all' left {_n_accepted(window)} events accepted")
    close_window(window)


@pytest.mark.parametrize("host", ["window", "dock"])
def test_a_and_r_still_curate_from_a_focused_button_and_space_from_the_surface(
        private_settings, qtbot, host):
    """GUARD. A button uses Space, not A / R; the waveform uses none of them."""
    window, insp = _open_inspector(qtbot, host)
    top = _activate(insp)
    ids = _ids(window)
    insp.show_event(ids[2])
    _start_of_chain(insp)
    _tab_to(top, insp._trace_btn)
    _press(top, Qt.Key_R)
    assert not accepted(window, ids[2])
    _press(top, Qt.Key_A)
    assert accepted(window, ids[2])
    insp.focus_curation_surface()
    settle()
    _press(top, Qt.Key_Space)
    assert not accepted(window, ids[2]), "Space on the waveform stopped toggling"
    _press(top, Qt.Key_Space)
    assert accepted(window, ids[2])
    close_window(window)


@pytest.mark.parametrize("host", ["window", "dock"])
def test_a_click_on_a_toolbar_button_still_takes_the_keyboard_for_the_curation_keys(
        private_settings, qtbot, host):
    """GUARD. A click on a toolbar button takes the keyboard out of the 'Event N' box (which
    swallows letters) into the window, so A / R curate right after it. A focused button is
    pressed by Space, but it uses neither A nor R."""
    window, insp = _open_inspector(qtbot, host)
    top = _activate(insp)
    ids = _ids(window)
    insp.show_event(ids[1])
    insp._index_spin.setFocus(Qt.TabFocusReason)
    settle()
    QTest.mouseClick(insp._next_btn, Qt.LeftButton)
    settle()
    assert insp.current_event_id() == ids[2]
    _press(top, Qt.Key_R)
    assert not accepted(window, ids[2]), "R after a click on 'next' did not reach curation"
    _press(top, Qt.Key_A)
    assert accepted(window, ids[2])
    close_window(window)


# ================================================================================================
# B. an open dropdown keeps the keys typed into it  (inspector, both hosts)
# ================================================================================================

@pytest.mark.parametrize("host", ["window", "dock"])
@pytest.mark.parametrize("key,entry", [(Qt.Key_R, "Rejected only"), (Qt.Key_A, "Accepted only"),
                                      (Qt.Key_Space, None)], ids=["r", "a", "space"])
def test_an_open_filter_dropdown_keeps_what_is_typed_into_it(private_settings, qtbot, host,
                                                              key, entry):
    window, insp = _open_inspector(qtbot, host)
    top = _activate(insp)
    ids = _ids(window)
    insp.show_event(ids[2])
    combo = insp._filter_combo
    before = (accepted(window, ids[2]), _n_accepted(window), combo.currentText())
    _open_dropdown(combo)
    _press(top, key)
    after = (accepted(window, ids[2]), _n_accepted(window), combo.currentText())
    assert after == before, f"a key typed into the open Filter list curated: {before} -> {after}"
    if entry is not None:
        assert _highlighted(combo) == entry, (
            f"the open list did not get the letter: it highlights {_highlighted(combo)!r}")
        _press(top, Qt.Key_Return)                   # and the user can choose what they typed
        assert combo.currentText() == entry
        assert insp._filter == entry
    _close_dropdown()
    close_window(window)


# ================================================================================================
# D. the controls the Tab chain lost are reachable again  (inspector, both hosts)
# ================================================================================================

@pytest.mark.parametrize("host", ["window", "dock"])
def test_every_control_the_inspector_took_out_of_the_tab_chain_is_reachable_and_works(
        private_settings, qtbot, host):
    window, insp = _open_inspector(qtbot, host, labels=2)
    top = _activate(insp)
    ids = _ids(window)
    insp.show_event(ids[2])

    for name in ("_advance_check", "_label_combo", "_label_apply_btn", "_label_clear_btn",
                 "_details_btn"):
        widget = getattr(insp, name)
        assert widget.isVisible(), name
        _start_of_chain(insp)
        _tab_to(top, widget)                               # a keyboard route exists...
        assert not (widget.focusPolicy() & Qt.ClickFocus), (
            f"{name}: a mouse click must still leave the keyboard on the curation surface")

    # ...and Space does the control's own job there, not the event's.
    was = insp.advance_after_decision()
    _start_of_chain(insp)
    _tab_to(top, insp._advance_check)
    _press(top, Qt.Key_Space)
    assert insp.advance_after_decision() is (not was), "Space did not tick 'Advance after decision'"
    assert accepted(window, ids[2]) and insp.current_event_id() == ids[2]
    _press(top, Qt.Key_Space)
    assert insp.advance_after_decision() is was

    shown = insp.details_shown()
    _start_of_chain(insp)
    _tab_to(top, insp._details_btn)
    _press(top, Qt.Key_Space)
    assert insp.details_shown() is (not shown), "Space did not toggle Details"
    assert accepted(window, ids[2])

    # The Label row: choose with the arrow keys, Apply and Clear with Space.
    _start_of_chain(insp)
    _tab_to(top, insp._label_combo)
    first = insp._label_combo.currentIndex()
    _press(top, Qt.Key_Down)
    assert insp._label_combo.currentIndex() == first + 1, (
        "the arrow keys do not reach the Label combo")
    _press(top, Qt.Key_Up)
    assert insp._label_combo.currentIndex() == first
    _tab_to(top, insp._label_apply_btn)
    _press(top, Qt.Key_Space)
    assert "type=translocation" in _flag(window, ids[2]), "Space did not press Apply"
    assert accepted(window, ids[2]), "Space on Apply also toggled the event"
    _start_of_chain(insp)
    _tab_to(top, insp._label_clear_btn)
    _press(top, Qt.Key_Space)
    assert "type=translocation" not in _flag(window, ids[2]), "Space did not press Clear"
    close_window(window)


def test_dock_beside_trace_is_reachable_by_tab_and_pressed_by_space(private_settings, qtbot):
    window, insp = _open_inspector(qtbot, "window")
    top = _activate(insp)
    asked: list[int] = []
    insp.hostToggleRequested.disconnect()          # MainWindow's answer would move the window
    insp.hostToggleRequested.connect(lambda: asked.append(1))
    ids = _ids(window)
    insp.show_event(ids[2])
    _start_of_chain(insp)
    _tab_to(top, insp._host_btn)
    assert not (insp._host_btn.focusPolicy() & Qt.ClickFocus)
    _press(top, Qt.Key_Space)
    assert asked == [1], "Space did not press 'Dock beside trace'"
    assert accepted(window, ids[2]), "Space on 'Dock beside trace' toggled the event"
    close_window(window)


# ================================================================================================
# C. the gallery  (a button, the open Show list, the Advance box)
# ================================================================================================

def _open_gallery(qtbot, *, rejected: int = 2):
    """The wired gallery over the six-event detection, with the first ``rejected`` events
    rejected, so that 'flip' and 'accept' / 'reject' give different totals."""
    window = build_window(qtbot)
    df = _frame(window)
    df.loc[df["event_id"].isin(_ids(window)[:rejected]), "accepted"] = False
    window._act_event_gallery.trigger()
    settle(100)
    gal = window._event_gallery
    assert gal is not None and gal.isVisible()
    top = _activate(gal)
    gal.view.setFocus(Qt.OtherFocusReason)
    settle()
    _press(top, Qt.Key_A, Qt.ControlModifier)
    assert len(gal.view.selectionModel().selectedIndexes()) == len(_ids(window)), (
        "setup: Ctrl+A did not select the page")
    return window, gal, top


@pytest.mark.parametrize("which", ["accept", "reject"])
def test_space_on_a_tab_focused_gallery_batch_button_presses_it_instead_of_flipping_tiles(
        private_settings, qtbot, which):
    """Two of the six events start rejected, so 'press the button' (all accepted / all
    rejected) and 'flip every tile' (two accepted either way) give different totals."""
    window, gal, top = _open_gallery(qtbot)
    n = len(_ids(window))
    assert _n_accepted(window) == n - 2
    button = gal.accept_selected_btn if which == "accept" else gal.reject_selected_btn
    pressed: list[int] = []
    button.clicked.connect(lambda *_: pressed.append(1))
    _tab_to(top, button)
    _press(top, Qt.Key_Space)
    assert pressed == [1], f"Space did not press the {which}-selected button"
    want = n if which == "accept" else 0
    assert _n_accepted(window) == want, (
        f"{which}-selected left {_n_accepted(window)} of {n} accepted, expected {want}: "
        "Space flipped the tiles instead")
    close_window(window)


@pytest.mark.parametrize("key,entry", [(Qt.Key_R, "Rejected only"), (Qt.Key_A, "Accepted only"),
                                      (Qt.Key_Space, None)], ids=["r", "a", "space"])
def test_an_open_show_dropdown_keeps_what_is_typed_into_it(private_settings, qtbot, key, entry):
    window, gal, top = _open_gallery(qtbot)
    combo = gal.filter_box
    before = (_n_accepted(window), combo.currentText(), gal.selected_event_ids())
    _open_dropdown(combo)
    _press(top, key)
    after = (_n_accepted(window), combo.currentText(), gal.selected_event_ids())
    assert after == before, f"a key typed into the open Show list curated: {before} -> {after}"
    if entry is not None:
        assert _highlighted(combo) == entry, (
            f"the open list did not get the letter: it highlights {_highlighted(combo)!r}")
        _press(top, Qt.Key_Return)
        assert combo.currentText() == entry
    _close_dropdown()
    close_window(window)


def test_the_page_keys_belong_to_an_open_gallery_dropdown(private_settings, qtbot):
    """The same cause, one key over: PgDn / End in the open Show list turned the page and
    dropped the selection while the list sat open."""
    window, gal, top = _open_gallery(qtbot)
    gal._model.set_page_size(2)
    gal._sync_controls()
    settle()
    gal.view.setFocus(Qt.OtherFocusReason)
    _press(top, Qt.Key_A, Qt.ControlModifier)
    selected = gal.selected_event_ids()
    assert selected and gal.page() == 0
    _open_dropdown(gal.filter_box)
    for key in (Qt.Key_PageDown, Qt.Key_End):
        _press(top, key)
        assert gal.page() == 0, "a page key typed into the open list turned the page"
        assert gal.selected_event_ids() == selected
    _close_dropdown()
    # ...and with the list closed they page again (GUARD).
    gal.view.setFocus(Qt.OtherFocusReason)
    settle()
    _press(top, Qt.Key_PageDown)
    assert gal.page() == 1
    close_window(window)


def test_the_gallery_advance_box_is_reachable_by_tab_and_toggled_by_space(private_settings,
                                                                           qtbot):
    window, gal, top = _open_gallery(qtbot)
    box = gal._advance_check
    assert not (box.focusPolicy() & Qt.ClickFocus)
    was = gal.advance_after_decision()
    _tab_to(top, box)
    _press(top, Qt.Key_Space)
    assert gal.advance_after_decision() is (not was), "Space did not tick 'Advance after decision'"
    assert _n_accepted(window) == len(_ids(window)) - 2, "Space on the box also curated tiles"
    _press(top, Qt.Key_Space)
    assert gal.advance_after_decision() is was
    close_window(window)


def test_the_gallery_keeps_its_curation_keys_on_the_tile_view(private_settings, qtbot):
    """GUARD. With the tile view focused R rejects the selection and Space flips it."""
    window, gal, top = _open_gallery(qtbot)
    n = len(_ids(window))
    gal.view.setFocus(Qt.OtherFocusReason)
    settle()
    _press(top, Qt.Key_R)
    assert _n_accepted(window) == 0
    _press(top, Qt.Key_Space)
    assert _n_accepted(window) == n
    close_window(window)


# ================================================================================================
# every dropdown of both windows, and the rule's edges
# ================================================================================================

def _dropdowns(window_widget: QtWidgets.QWidget) -> list[QtWidgets.QComboBox]:
    return [c for c in window_widget.findChildren(QtWidgets.QComboBox)
            if c.isVisibleTo(window_widget) and c.isEnabled() and c.count() > 1]


@pytest.mark.parametrize("which", ["inspector", "gallery"])
def test_no_dropdown_of_either_window_lets_a_letter_through_to_curation(private_settings, qtbot,
                                                                         which):
    """Every QComboBox the window has, not only the Filter and Show boxes the finding names:
    a dropdown added later must not bring the defect back."""
    if which == "inspector":
        window, insp = _open_inspector(qtbot, "window", labels=2)
        target, top = insp, _activate(insp)
        insp.show_event(_ids(window)[2])
    else:
        window, target, top = _open_gallery(qtbot)
    boxes = _dropdowns(target)
    assert len(boxes) >= 3, f"setup: only {len(boxes)} dropdowns found in the {which}"
    for box in boxes:
        before = (_n_accepted(window), box.currentText())
        _open_dropdown(box)
        for key in (Qt.Key_R, Qt.Key_A, Qt.Key_Space):
            _press(top, key)
        assert (_n_accepted(window), box.currentText()) == before, (
            f"the {which}'s dropdown at {box.toolTip()[:40]!r} gave its keys to curation")
        _close_dropdown()
        top = _activate(target)
    close_window(window)


def _override(key, text: str, mods=Qt.NoModifier) -> QtGui.QKeyEvent:
    return QtGui.QKeyEvent(QtCore.QEvent.ShortcutOverride, key, mods, text)


def _kept_by(receiver: QtCore.QObject, event: QtGui.QKeyEvent) -> bool:
    """Whether ``receiver`` (or a filter on it) accepts the ShortcutOverride, i.e. whether Qt
    would deliver the key to it rather than to a shortcut."""
    event.setAccepted(False)
    QtWidgets.QApplication.sendEvent(receiver, event)
    return event.isAccepted()


@pytest.mark.parametrize("which", ["inspector", "gallery"])
def test_every_menu_of_both_windows_keeps_what_is_typed_into_it(private_settings, qtbot, which):
    """An open menu is the same popup as an open dropdown: 'a' typed into the gallery's Select
    menu accepted every selected tile (measured: 12 -> 14 accepted of 14) instead of reaching
    the menu's own first-letter search. The menu is asked directly, with the override Qt sends
    to the popup that receives a key: the offscreen platform deactivates a window the moment a
    menu pops up over it and closes the menu again, so an open menu cannot be driven here."""
    if which == "inspector":
        window, target = _open_inspector(qtbot, "window")
    else:
        window, target, _top = _open_gallery(qtbot)
    menus = target.findChildren(QtWidgets.QMenu)
    assert menus, f"setup: the {which} has no menu"
    for menu in menus:
        name = menu.objectName() or ", ".join(a.text() for a in menu.actions())
        assert _kept_by(menu, _override(Qt.Key_A, "a")), f"{which} menu {name!r} lets 'a' through"
        assert _kept_by(menu, _override(Qt.Key_Space, " ")), f"{which} menu {name!r}: Space"
        assert not _kept_by(menu, _override(Qt.Key_Z, "z", Qt.ControlModifier)), (
            f"{which} menu {name!r} keeps a Ctrl chord from its shortcut")
    close_window(window)


def test_chords_and_closed_combos_are_not_claimed(qapp, qtbot):
    """The guard claims bare keys only, and only for a button (Space) or an OPEN dropdown;
    a closed combo box that merely holds the focus keeps yielding to the window's keys (the
    gallery opens with its Show box focused, and a mouse pick leaves the Filter box focused,
    so claiming letters there would turn the very next A / R into a filter change)."""
    from nanoscope.gui.curation_keys import CurationKeyGuard

    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    central = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(central)
    button, combo = QtWidgets.QPushButton("press"), QtWidgets.QComboBox()
    combo.addItems(["alpha", "bravo", "charlie"])
    lay.addWidget(button)
    lay.addWidget(combo)
    win.setCentralWidget(central)
    fired: list[str] = []
    for seq in ("Space", "R", "Ctrl+Z"):
        sc = QtGui.QShortcut(QtGui.QKeySequence(seq), win)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(lambda s=seq: fired.append(s))
    clicks: list[int] = []
    button.clicked.connect(lambda *_: clicks.append(1))
    CurationKeyGuard(win)
    top = _activate(win)

    button.setFocus(Qt.TabFocusReason)
    settle()
    _press(top, Qt.Key_Space)
    _press(top, Qt.Key_R)
    assert clicks == [1] and fired == ["R"], (clicks, fired)      # Space is the button's, R is not

    fired.clear()
    combo.setFocus(Qt.TabFocusReason)
    settle()
    _press(top, Qt.Key_R)
    _press(top, Qt.Key_Space)
    assert fired == ["R", "Space"] and combo.currentIndex() == 0   # closed: the window's keys

    fired.clear()
    _open_dropdown(combo)
    _press(top, Qt.Key_Z, Qt.ControlModifier)
    assert fired == ["Ctrl+Z"], "a Ctrl chord must still reach its shortcut with a list open"
    _press(top, Qt.Key_R)
    assert fired == ["Ctrl+Z"], "a bare key typed into an open list reached a shortcut"
    _close_dropdown()
