"""Choose many columns in one go: a checklist shared by every column chooser.

The Results table's **Columns…** used to be a ``QMenu`` of checkable actions, and Qt hides
a menu before it fires the entry that was clicked, so each opening changed ONE column
(hiding five took five openings and ten clicks). It also listed the frame's schema order
and raw names while the table showed the scientific order and its own header labels, and
it let every column be hidden. The Comparison pair matrix had a checkable list on which
only the 18 px indicator toggled and the two-to-five rule surfaced at Generate.

:class:`ColumnChecklist` is the one answer for all of them: a click anywhere on a row
toggles it, Space toggles the current row and the arrows move, a filter appears on long
lists, a count line says how many are on, and Show all / Hide all / Default act on the
rows the filter shows. ``changed`` is emitted ONCE PER GESTURE, so a host writes its
settings and redraws once, not once per row. Optional bounds (``minimum``/``maximum``)
disable the rows that would cross them and say why, on the row and on the count line.

:class:`ColumnChooserPopup` hosts a checklist in a ``Qt.Popup`` frame for a button. A
popup frame, not ``QMenu`` + ``QWidgetAction``: measured offscreen, a menu hosting the
checklist closed by itself during idle waits while the frame stayed up through every
gesture, and a menu takes the arrow and letter keys for its own navigation, which fights
the filter box.

A leaf module: it imports Qt only, never a sibling gui module, so ``panels.py`` may use it
(the same standing as ``inspector_widgets`` and ``numeric_text``). No pyqtgraph, and no
event filter anywhere: the row toggle is a delegate ``editorEvent`` override.
"""
from __future__ import annotations

import weakref
from collections.abc import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt

#: Item roles: the raw key, the text the filter matches (label AND key), the row's own
#: tooltip (kept while a bound note covers it), and the separator marker.
_KEY_ROLE = Qt.UserRole
_FILTER_ROLE = Qt.UserRole + 1
_TIP_ROLE = Qt.UserRole + 2
_SEPARATOR_ROLE = Qt.UserRole + 3

_ROW_FLAGS = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
#: A row a bound holds: greyed, so neither a click nor Space can toggle it.
_HELD_FLAGS = Qt.ItemIsSelectable | Qt.ItemIsUserCheckable


class _RowToggleDelegate(QtWidgets.QStyledItemDelegate):
    """A left-button release ANYWHERE on a checkable row toggles it; Qt's own delegate
    toggles only inside the 13 px indicator. Space and Select keep Qt's own toggle.

    A double-click is one toggle, not two: Qt 6 delivers Press, Release, DblClick,
    Release, and both releases reach this delegate, so the release that follows a
    double-click is swallowed rather than flipping the row straight back.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._swallow_release = False

    def editorEvent(self, event, model, option, index):  # noqa: N802 (Qt override)
        flags = index.flags()
        if not (flags & Qt.ItemIsUserCheckable) or not (flags & Qt.ItemIsEnabled):
            return super().editorEvent(event, model, option, index)
        kind = event.type()
        if kind == QtCore.QEvent.MouseButtonPress:
            self._swallow_release = False   # never carried past the gesture it belongs to
        elif kind == QtCore.QEvent.MouseButtonDblClick:
            self._swallow_release = True
            return True
        if kind == QtCore.QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._swallow_release:
                self._swallow_release = False
                return True
            state = Qt.CheckState(index.data(Qt.CheckStateRole))
            model.setData(index, Qt.Unchecked if state == Qt.Checked else Qt.Checked,
                          Qt.CheckStateRole)
            return True
        return super().editorEvent(event, model, option, index)


class ColumnChecklist(QtWidgets.QWidget):
    """Choose many keys at once. ``changed(list[str])`` carries the checked keys in entry
    order, once per gesture: a row click, Space, Show all, Hide all or Default.

    ``noun`` names the rows ("columns", "features"). For "columns" the words are "shown",
    Show all and Hide all; for anything else "selected", Select all and Clear.
    ``minimum``/``maximum`` bound what a GESTURE may leave checked: the rows that would
    cross a bound are disabled with a tooltip that says why, and the count line says it
    too. :meth:`set_checked` is the host's own restore and is not bounded. ``keep`` names
    the keys a bulk Hide/Clear leaves checked when the minimum needs one (the Results
    table keeps ``accepted``). ``filter_from``: the filter box shows from this many
    entries up.
    """

    changed = QtCore.Signal(list)

    #: Rows the list shows before it scrolls (see :meth:`set_visible_rows`).
    VISIBLE_ROWS = 16

    def __init__(self, *, noun: str = "columns", minimum: int = 0, maximum: int | None = None,
                 filter_from: int = 12, keep: Iterable[str] = (), parent=None):
        super().__init__(parent)
        self._noun = str(noun)
        self._minimum = max(0, int(minimum))
        self._maximum = None if maximum is None else max(self._minimum, int(maximum))
        self._filter_from = int(filter_from)
        self._keep = tuple(str(k) for k in keep)
        self._columns = self._noun == "columns"
        self._entries: list[tuple[str, str, str]] = []
        self._defaults: set[str] = set()
        self._separators: dict[str, str] = {}
        self._last: list[str] = []
        self._quiet = False
        self._visible_rows = self.VISIBLE_ROWS
        #: True in a popup: the list shrinks to the rows the filter shows. Inline it keeps
        #: one height, so typing in the filter never moves the controls below it.
        self.shrink_to_rows = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.filter = QtWidgets.QLineEdit()
        # Named so a host that wires every QLineEdit it holds into an Undo history (the
        # Plots window's _wire_history) can leave this one out: filtering is not an edit.
        self.filter.setObjectName("columnChecklistFilter")
        self.filter.setPlaceholderText(f"Filter {self._noun}…")
        self.filter.setClearButtonEnabled(True)
        self.filter.setAccessibleName(f"Filter {self._noun}")
        self.filter.setToolTip(f"Shows the {self._noun} whose name or key contains this text. "
                               "The buttons below then act on those rows only.")
        self.filter.setVisible(False)
        layout.addWidget(self.filter)
        self.model = QtGui.QStandardItemModel(self)
        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterRole(_FILTER_ROLE)
        self.view = QtWidgets.QListView()
        self.view.setModel(self.proxy)
        self.view.setItemDelegate(_RowToggleDelegate(self.view))
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.view.setUniformItemSizes(True)
        self.view.setAccessibleName(f"{self._noun.capitalize()} to show"
                                    if self._columns else f"{self._noun.capitalize()} to use")
        layout.addWidget(self.view, 1)
        self.count = QtWidgets.QLabel()
        self.count.setStyleSheet("color:#8b9199;")
        self.count.setWordWrap(True)
        layout.addWidget(self.count)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(4)
        self.all_btn = QtWidgets.QPushButton()
        self.none_btn = QtWidgets.QPushButton()
        self.default_btn = QtWidgets.QPushButton()
        for button in (self.all_btn, self.none_btn, self.default_btn):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.filter.textChanged.connect(self._on_filter_changed)
        self.filter.returnPressed.connect(self._focus_list)
        self.model.itemChanged.connect(self._on_item_changed)
        self.all_btn.clicked.connect(lambda: self._bulk(lambda _key: True))
        self.none_btn.clicked.connect(lambda: self._bulk(lambda _key: False))
        self.default_btn.clicked.connect(lambda: self._bulk(lambda key: key in self._defaults))
        self._sync()

    # -- data in / out ---------------------------------------------------------------------
    def set_entries(self, entries, checked: Iterable[str], defaults: Iterable[str] = ()) -> None:
        """``entries``: ``(key, label, tooltip)`` in the order the user sees them, with the
        label the user sees. Silent: no ``changed``."""
        self._entries = [(str(key), str(label if label is not None else key), str(tip or ""))
                         for key, label, tip in entries]
        self._defaults = {str(k) for k in defaults}
        self._rebuild({str(k) for k in checked})

    def checked(self) -> list[str]:
        """The checked keys, in entry order."""
        out = []
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if not item.data(_SEPARATOR_ROLE) and item.checkState() == Qt.Checked:
                out.append(item.data(_KEY_ROLE))
        return out

    def set_checked(self, keys: Iterable[str]) -> None:
        """Check exactly ``keys``. Silent (no ``changed``) and not bounded: a host restoring
        its own saved state is not a gesture."""
        wanted = {str(k) for k in keys}
        self._quiet = True
        try:
            for item in self._rows():
                state = Qt.Checked if item.data(_KEY_ROLE) in wanted else Qt.Unchecked
                if item.checkState() != state:
                    item.setCheckState(state)
        finally:
            self._quiet = False
        self._last = self.checked()
        self._sync()

    def set_separator_before(self, key: str, caption: str) -> None:
        """Show an unselectable caption row (e.g. "Identifiers") above ``key``. It is kept
        across :meth:`set_entries` and hidden while a filter is active."""
        self._separators[str(key)] = str(caption)
        if self._entries:
            self._rebuild(set(self.checked()))

    def set_visible_rows(self, rows: int) -> None:
        """How many rows the list shows before it scrolls."""
        self._visible_rows = max(1, int(rows))
        self._fit_height()

    def keys(self) -> list[str]:
        """Every entry key, in entry order."""
        return [key for key, _label, _tip in self._entries]

    def shown_keys(self) -> list[str]:
        """The keys of the rows the filter shows, in entry order (separators excluded)."""
        out = []
        for row in range(self.proxy.rowCount()):
            index = self.proxy.index(row, 0)
            if not index.data(_SEPARATOR_ROLE):
                out.append(index.data(_KEY_ROLE))
        return out

    # -- internals -------------------------------------------------------------------------
    def _rows(self):
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if not item.data(_SEPARATOR_ROLE):
                yield item

    def _rebuild(self, checked: set[str]) -> None:
        self._quiet = True
        try:
            self.model.clear()
            for key, label, tip in self._entries:
                caption = self._separators.get(key)
                if caption:
                    sep = QtGui.QStandardItem(caption)
                    sep.setFlags(Qt.NoItemFlags)
                    sep.setData(True, _SEPARATOR_ROLE)
                    sep.setData("", _FILTER_ROLE)   # an active filter hides it
                    font = sep.font()
                    font.setBold(True)
                    sep.setFont(font)
                    self.model.appendRow(sep)
                item = QtGui.QStandardItem(label)
                item.setData(key, _KEY_ROLE)
                item.setData(f"{label} {key}", _FILTER_ROLE)
                item.setData(tip or (key if key != label else ""), _TIP_ROLE)
                item.setToolTip(item.data(_TIP_ROLE))
                item.setFlags(_ROW_FLAGS)
                item.setCheckState(Qt.Checked if key in checked else Qt.Unchecked)
                self.model.appendRow(item)
        finally:
            self._quiet = False
        filtering = len(self._entries) >= self._filter_from
        if not filtering and self.filter.text():
            self.filter.clear()
        self.filter.setVisible(filtering)
        self._last = self.checked()
        self._sync()

    def _on_item_changed(self, item) -> None:
        # itemChanged also fires for the flag and tooltip edits _sync makes; only a change
        # of the checked set is a gesture.
        if self._quiet or item.data(_SEPARATOR_ROLE):
            return
        keys = self.checked()
        if keys == self._last:
            return
        self._last = keys
        self._sync()
        self.changed.emit(list(keys))

    def _bulk(self, rule) -> None:
        """One gesture, one ``changed``: only the rows the filter shows are touched, and the
        bounds hold (a Hide all under a minimum keeps ``keep``, then what was on)."""
        before = self.checked()
        target = set(before)
        for key in self.shown_keys():
            if rule(key):
                target.add(key)
            else:
                target.discard(key)
        order = self.keys()
        if len(target) < self._minimum:
            for key in (*self._keep, *before, *order):
                if len(target) >= self._minimum:
                    break
                if key in order:
                    target.add(key)
        if self._maximum is not None and len(target) > self._maximum:
            ranked = ([k for k in order if k in target and k in before]
                      + [k for k in order if k in target and k not in before])
            target = set(ranked[:self._maximum])
        self._quiet = True
        try:
            for item in self._rows():
                state = Qt.Checked if item.data(_KEY_ROLE) in target else Qt.Unchecked
                if item.checkState() != state:
                    item.setCheckState(state)
        finally:
            self._quiet = False
        keys = self.checked()
        if keys == self._last:
            self._sync()
            return
        self._last = keys
        self._sync()
        self.changed.emit(list(keys))

    def _singular(self) -> str:
        return self._noun[:-1] if self._noun.endswith("s") else self._noun

    def _bound_notes(self) -> tuple[str, str]:
        """(row tooltip at the minimum, row tooltip at the maximum)."""
        state = "visible" if self._columns else "selected"
        if self._minimum == 1:
            low = f"At least one {self._singular()} stays {state}"
        else:
            low = f"At least {self._minimum} {self._noun} stay {state}"
        high = (f"At most {self._maximum} {self._noun}: untick one to choose another"
                if self._maximum is not None else "")
        return low, high

    def _sync(self) -> None:
        """Count line, the bound guard on each row, and the buttons' scope wording."""
        keys = self.checked()
        n, total = len(keys), len(self._entries)
        at_min = self._minimum > 0 and n <= self._minimum
        at_max = self._maximum is not None and n >= self._maximum
        low, high = self._bound_notes()
        self._quiet = True
        try:
            for item in self._rows():
                on = item.checkState() == Qt.Checked
                blocked = (on and at_min) or (not on and at_max)
                flags = _HELD_FLAGS if blocked else _ROW_FLAGS
                if item.flags() != flags:
                    item.setFlags(flags)
                tip = (low if on else high) if blocked else item.data(_TIP_ROLE)
                if item.toolTip() != tip:
                    item.setToolTip(tip)
        finally:
            self._quiet = False
        if self._maximum is not None:
            text = f"{n} of {total} selected · pick {self._minimum}–{self._maximum}"
        elif self._columns:
            text = f"{n} of {total} {self._noun} shown"
        else:
            text = f"{n} of {total} {self._noun} selected"
        note = ""
        if total and at_max:
            note = "untick one to choose another"
        elif total and at_min and self._maximum is None:   # "pick a–b" already says it
            note = low[0].lower() + low[1:]
        self.count.setText(f"{text} · {note}" if note else text)
        self.count.setToolTip(high if note and at_max else low if note else "")
        filtered = bool(self.filter.text())
        if self._columns:
            words = ("Show these", "Hide these") if filtered else ("Show all", "Hide all")
        else:
            words = ("Select these", "Clear these") if filtered else ("Select all", "Clear")
        self.all_btn.setText(words[0])
        self.none_btn.setText(words[1])
        self.default_btn.setText("Default for these" if filtered else f"Default {self._noun}")
        scope = f"the {self._noun} the filter shows" if filtered else f"every {self._singular()}"
        self.all_btn.setToolTip(f"Tick {scope}.")
        self.none_btn.setToolTip(f"Untick {scope}."
                                 + (f" {low}." if self._minimum else ""))
        self.default_btn.setToolTip(f"Put {scope} back to the default choice.")
        # "Select all" past a maximum could only be refused, so a bounded pick has none.
        self.all_btn.setVisible(self._maximum is None or self._maximum >= total)
        self._fit_height()

    def _fit_height(self) -> None:
        shown = self.proxy.rowCount() if self.shrink_to_rows else self.model.rowCount()
        rows = max(1, min(shown, self._visible_rows))
        per_row = self.view.sizeHintForRow(0) if self.proxy.rowCount() else -1
        if per_row <= 0:
            per_row = self.view.fontMetrics().height() + 6
        height = rows * per_row + 2 * self.view.frameWidth() + 2
        if self.view.height() != height or self.view.minimumHeight() != height:
            self.view.setFixedHeight(height)

    def _on_filter_changed(self, text: str) -> None:
        self.proxy.setFilterFixedString(text)
        self._sync()

    def _focus_list(self) -> None:
        self.view.setFocus(Qt.OtherFocusReason)
        for row in range(self.proxy.rowCount()):
            index = self.proxy.index(row, 0)
            if not index.data(_SEPARATOR_ROLE):
                self.view.setCurrentIndex(index)
                return


class ColumnChooserPopup(QtWidgets.QFrame):
    """A :class:`ColumnChecklist` in a ``Qt.Popup`` frame, opened by a button.

    It stays up while rows are toggled and closes on Esc or a click outside (Qt's own
    popup rule). It opens below its button and flips above it when the screen has no room
    below. ``aboutToShow`` is emitted first, so the owner refills the rows from live state;
    keyboard focus then lands in the filter when it is shown, else on the list. Closing
    hands the focus back to the button (macOS leaves it nowhere otherwise), and a click on
    that same button closes the popup instead of reopening it.
    """

    aboutToShow = QtCore.Signal()

    def __init__(self, checklist: ColumnChecklist | None = None, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.checklist = checklist if checklist is not None else ColumnChecklist()
        self.checklist.shrink_to_rows = True
        self.checklist.setMinimumWidth(280)
        self._anchor = None   # weakref to the button that opened it
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(0)
        # The frame follows the checklist's size, so a filter that shows three rows gives
        # a three-row popup rather than a full-height one with empty space under it.
        layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        layout.addWidget(self.checklist)
        self._footer = QtWidgets.QHBoxLayout()
        self._footer.setContentsMargins(6, 0, 6, 2)
        self._footer.setSpacing(4)
        self._footer.addStretch(1)
        layout.addLayout(self._footer)
        self.checklist.filter.textChanged.connect(self._refit)

    def _refit(self, *_args) -> None:
        self.layout().activate()
        self.adjustSize()

    def add_footer_button(self, text: str, tooltip: str = "") -> QtWidgets.QPushButton:
        """A host action under the checklist (the Results table's Default order)."""
        button = QtWidgets.QPushButton(text)
        button.setAutoDefault(False)
        button.setToolTip(tooltip)
        self._footer.insertWidget(self._footer.count() - 1, button)
        return button

    def anchor(self) -> QtWidgets.QWidget | None:
        button = self._anchor() if self._anchor is not None else None
        try:
            return button if button is not None and button.isVisible() else None
        except RuntimeError:   # the button's C++ half is gone
            return None

    def show_below(self, button: QtWidgets.QWidget) -> None:
        self._anchor = weakref.ref(button)
        self.setAttribute(Qt.WA_NoMouseReplay, False)
        self.aboutToShow.emit()
        self._place(button)
        self.show()
        # The list's row height is only final once it is polished on screen; a popup
        # flipped above its button is placed again so its bottom edge still meets it.
        self._refit()
        self._place(button)
        if not self.checklist.filter.isHidden():
            self.checklist.filter.setFocus(Qt.PopupFocusReason)
        else:
            self.checklist.view.setFocus(Qt.PopupFocusReason)
            if not self.checklist.view.currentIndex().isValid():
                self.checklist._focus_list()

    def _place(self, button: QtWidgets.QWidget) -> None:
        self.checklist._fit_height()
        self.layout().activate()
        self.adjustSize()
        screen = (button.screen() or QtGui.QGuiApplication.primaryScreen()).availableGeometry()
        below = button.mapToGlobal(QtCore.QPoint(0, button.height()))
        top = button.mapToGlobal(QtCore.QPoint(0, 0)).y() - self.height()
        y = below.y() if below.y() + self.height() <= screen.bottom() + 1 else max(screen.top(), top)
        x = max(screen.left(), min(below.x(), screen.right() + 1 - self.width()))
        if self.pos() != QtCore.QPoint(x, y):
            self.move(x, y)

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        # A press on the opening button while open would close the popup and then reach
        # the button, whose click would open it again at once: say "not replayed", as
        # QComboBox and QMenu do for their own button.
        if not self.rect().contains(event.position().toPoint()):
            button = self.anchor()
            if button is not None and button.rect().contains(
                    button.mapFromGlobal(event.globalPosition().toPoint())):
                self.setAttribute(Qt.WA_NoMouseReplay, True)
        super().mousePressEvent(event)

    def hideEvent(self, event):  # noqa: N802 (Qt override)
        super().hideEvent(event)
        button = self.anchor()
        if button is None:
            return
        focus = QtWidgets.QApplication.focusWidget()
        if focus is None or focus is self or self.isAncestorOf(focus):
            button.setFocus(Qt.PopupFocusReason)
