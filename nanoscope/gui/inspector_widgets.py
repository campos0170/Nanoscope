"""Layout-only building blocks for the inspector; never own analysis parameters.

Plus the one key rule of the inspector's panels (:func:`install_run_key`): Ctrl/⌘+Return or
Ctrl/⌘+keypad Enter presses the run button of the panel the keyboard is in. It owns no
parameter either: it commits the focused field and clicks a button the panel already has.
"""
from __future__ import annotations

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

#: The run key: Ctrl+Return, and Ctrl with the keypad's Enter (a different key to Qt). A plain
#: keypad Enter is not one of them. On a Mac "Ctrl" is ⌘.
RUN_KEYS: tuple[str, ...] = ("Ctrl+Return", "Ctrl+Enter")
#: What a run button carries while there are edits it has not run.
PENDING_MARK = " ●"


def _item_view_holding(field: QtWidgets.QWidget) -> QtWidgets.QAbstractItemView | None:
    """The item view whose cell editor ``field`` is (it sits in the view's viewport), or
    None for an ordinary field."""
    holder = field.parentWidget()
    view = holder.parentWidget() if holder is not None else None
    if isinstance(view, QtWidgets.QAbstractItemView) and holder is view.viewport():
        return view
    return None


def _focusable(widget: QtWidgets.QWidget) -> bool:
    return (shiboken6.isValid(widget) and widget.isVisible() and widget.isEnabled()
            and widget.focusPolicy() != QtCore.Qt.NoFocus)


def _commit_focused_field(scope: QtWidgets.QWidget) -> None:
    """Commit what is typed in the focused field inside ``scope``, as leaving it would.

    A spin box with keyboard tracking off and a line edit commit on focus-out (or Return),
    and a shortcut takes the key before the field sees it, so without this the run would use
    the value the field held before the typing. The field gets the focus back afterwards, so
    the user stays where they were.

    A table's open cell editor is committed by giving its view the focus: the delegate
    commits and closes the editor on that focus-out, and the view keeps the keyboard, as
    Return in the cell would leave it. Any other field that is gone or hidden once it lost
    the focus hands it to its nearest focusable, visible ancestor inside ``scope``."""
    focus = QtWidgets.QApplication.focusWidget()
    if focus is None or not (focus is scope or scope.isAncestorOf(focus)):
        return
    field = focus
    if isinstance(field.parentWidget(), QtWidgets.QAbstractSpinBox):
        field = field.parentWidget()            # a spin box's own line edit
    typed = (isinstance(field, (QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit))
             or (isinstance(field, QtWidgets.QComboBox) and field.isEditable()))
    if not typed:
        return
    view = _item_view_holding(field)
    if view is not None:
        view.setFocus(QtCore.Qt.OtherFocusReason)
        return
    ancestors = []                              # from the field's parent up to ``scope``
    if focus is not scope:
        parent = focus.parentWidget()
        while parent is not None and parent is not scope:
            ancestors.append(parent)
            parent = parent.parentWidget()
        ancestors.append(scope)
    focus.clearFocus()
    if shiboken6.isValid(focus) and focus.isVisible():
        focus.setFocus(QtCore.Qt.OtherFocusReason)
        return
    for widget in ancestors:
        if _focusable(widget):
            widget.setFocus(QtCore.Qt.OtherFocusReason)
            return


def press_run_button(scope: QtWidgets.QWidget, button: QtWidgets.QAbstractButton) -> None:
    """The run key's action: nothing while ``button`` is disabled; otherwise commit the
    focused field inside ``scope``, then click ``button`` -- the exact path of a click."""
    if not button.isEnabled():
        return
    _commit_focused_field(scope)
    button.click()


class _FollowEnabled(QtCore.QObject):
    """Keeps run-key shortcuts enabled exactly while their button is. A disabled shortcut
    is skipped, so with the button greyed out the key reaches the focused widget as it did
    before there was a run key (a table's cell editor commits on it, for one)."""

    def __init__(self, button: QtWidgets.QAbstractButton,
                 shortcuts: list[QtGui.QShortcut]) -> None:
        super().__init__(button)
        self._shortcuts = shortcuts
        self._sync(button)
        button.installEventFilter(self)

    def _sync(self, button: QtWidgets.QAbstractButton) -> None:
        on = button.isEnabled()
        for shortcut in self._shortcuts:
            if shiboken6.isValid(shortcut) and shortcut.isEnabled() != on:
                shortcut.setEnabled(on)

    def eventFilter(self, obj, event):  # noqa: N802 -- Qt API
        if event.type() == QtCore.QEvent.EnabledChange:
            self._sync(obj)
        return False


def install_run_key(scope: QtWidgets.QWidget,
                    button: QtWidgets.QAbstractButton) -> list[QtGui.QShortcut]:
    """Make :data:`RUN_KEYS` press ``button`` while the keyboard is inside ``scope``.

    ``Qt.WidgetWithChildrenShortcut``, never a window shortcut: the key belongs to the panel
    that has the focus, so two scopes that do not contain one another can never both match
    one key press (Qt fires neither of an ambiguous pair), and a hidden scope (a tab not in
    front) matches nothing. The shortcuts are enabled only while ``button`` is, so a greyed
    run button never swallows the key."""
    shortcuts = []
    for key in RUN_KEYS:
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), scope)
        shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        shortcut.setAutoRepeat(False)
        shortcut.activated.connect(lambda s=scope, b=button: press_run_button(s, b))
        shortcuts.append(shortcut)
    _FollowEnabled(button, shortcuts)
    return shortcuts


class InspectorScrollArea(QtWidgets.QScrollArea):
    """A bounded viewport: another page's content cannot dictate its height."""

    def __init__(self, content: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumSize(0, 0)
        self.setWidget(content)

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(240, 320)

    def minimumSizeHint(self):  # noqa: N802
        return QtCore.QSize(0, 0)


class Disclosure(QtWidgets.QWidget):
    """Show/hide an existing widget without disabling it or changing its values."""

    def __init__(self, title: str, content: QtWidgets.QWidget, *, expanded=False, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.toggle.setStyleSheet('QToolButton { text-align: left; padding: 4px; }')
        self.content = content
        layout.addWidget(self.toggle)
        layout.addWidget(content)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool):
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.content.setVisible(expanded)


class PinnedActions(InspectorScrollArea):
    """Keep actions above vertical scrolling without imposing a dock-width floor.

    At exceptionally narrow saved widths the row can scroll horizontally. Its
    scrollbar gets its own height, so it never paints over the buttons.
    """

    def __init__(self, content: QtWidgets.QWidget, parent=None):
        super().__init__(content, parent)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        content.installEventFilter(self)
        self._sync_height()

    def _sync_height(self):
        content = self.widget()
        height = content.sizeHint().height()
        if content.minimumSizeHint().width() > self.viewport().width():
            height += self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(max(0, height))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._sync_height()

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QtCore.QEvent.LayoutRequest:
            self._sync_height()
        return super().eventFilter(watched, event)

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(0, self.height())


def host_layout(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    """Transfer an existing layout and its live controls to a plain container."""
    host = QtWidgets.QWidget()
    host.setLayout(layout)
    return host
