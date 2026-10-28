"""Which key belongs to the control that has the keyboard, and which to a curation shortcut.

THE DEFECT (sweep 8, S8-KEYS2)
==============================
The Event inspector and the Event gallery bind BARE keys -- A, R, Space, T, the label digits
and hotkeys, and in the gallery the page keys -- as ``Qt.WidgetWithChildrenShortcut``, so that
they answer wherever the keyboard focus sits inside the window (and not for the events table
behind it; see ``_install_shortcuts`` in either module). Qt evaluates a shortcut BEFORE
``keyPressEvent``: the focused widget keeps a key only if it accepts the ``ShortcutOverride``
Qt sends first. A line edit and a spin box accept it for typing; a button does not, and
neither does the list of an open dropdown. Measured:

* Tab to "Show in trace" / "Reject all" / "Reject selected" and press Space, the one key that
  presses a focused button: the button was never pressed and the event (or the selected
  tiles) was toggled instead.
* Open the Filter or Show dropdown and type "r": the list never saw the letter and the event
  was rejected. A dropdown's list is a separate top-level ``Qt.Popup`` window, but this
  shortcut context walks up through popups to their owner, so the shortcut matched. An open
  menu is the same popup: "a" typed into the gallery's Select menu accepted the selected tiles.

THE RULE
========
:class:`CurationKeyGuard` accepts the ``ShortcutOverride`` for the key the control that holds
the keyboard uses itself, so Qt delivers the key to the control and the shortcut stays out:

* an OPEN DROPDOWN'S LIST OR MENU keeps every bare key -- the letters it searches by, Space, and
  the navigation keys the gallery binds to its pages. It is a popup: whatever is typed while it
  is open is meant for it, and a page turn behind it would also drop the selection.
* a BUTTON keeps Space (push, tool and check boxes alike: a focused button is pressed by it).
* nothing else does. The waveform, the tile view and the window background keep the curation
  keys; a button does not use A or R; a closed combo box that merely holds the focus (the
  gallery opens with its Show box focused, and a mouse pick leaves the Filter box focused) does
  not take them either, or the next A / R after choosing a filter would change the filter.
  Chords (Ctrl / Alt / Meta) are never a control's own typing, so Undo, First unreviewed and
  the others stay on their shortcuts.

Line edits and spin boxes need nothing: Qt already lets them keep what they type.

Imports nothing from the app but Qt: it is shared by two windows that may not import each other.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

Qt = QtCore.Qt

__all__ = ["CurationKeyGuard"]

_CHORD = Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier


class CurationKeyGuard(QtCore.QObject):
    """Accept the ``ShortcutOverride`` for the keys the control that has the keyboard uses.

    Installed on a curation window and on the popup of every combo box and menu in it. The
    window is enough for a button: an override the focused widget declines travels up its
    parents, so the window sees it and asks who has the focus. A dropdown's list or a menu is
    its own top-level window and the override stops there, so each popup is hooked itself --
    found with ``findChildren`` once the window is built, which names only widgets this code
    made (a dropdown added to the window later is covered with no edit here). An override that
    reaches a popup is, by construction, one the popup is receiving the key for."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        window.installEventFilter(self)
        for combo in window.findChildren(QtWidgets.QComboBox):
            combo.view().window().installEventFilter(self)
        for menu in window.findChildren(QtWidgets.QMenu):
            menu.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if event.type() != QtCore.QEvent.Type.ShortcutOverride or event.modifiers() & _CHORD:
            return False
        if isinstance(obj, QtWidgets.QWidget) and obj.windowType() == Qt.Popup:
            keeps = True
        else:
            focus = QtWidgets.QApplication.focusWidget()
            keeps = isinstance(focus, QtWidgets.QAbstractButton) and event.key() == Qt.Key_Space
        if keeps:
            event.accept()
            return True
        return False
