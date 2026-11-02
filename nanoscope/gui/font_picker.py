"""The one figure-font control, and the Qt side of the figure-font policy.

``plot.figure_fonts`` is the policy: Arial first and the default, then Helvetica, Times New
Roman and Courier New, and a stack of metric-compatible substitutes for each. This module is
what the windows use of it:

* :class:`FontPicker` -- a combo that lists exactly that list and an ``Other font…`` row, in
  place of the 180-odd-family lists the Appearance section, Distributions' Style group, the
  Text style… popover and the composer each used to carry;
* :class:`FontListDialog` -- the list behind ``Other font…``: every installed family, with a
  filter box;
* :func:`figure_family`, :func:`figure_qfont` and :func:`html_font_family` -- what a figure is
  DRAWN with when its family is unset (Arial) or not installed (its first installed
  substitute, instead of the application font, which on macOS is a private system face).

Nothing here changes what a window stores: an unset family stays unset, and a family the
list does not have (an old project's ``Menlo``) is shown as one extra row and kept as it is.
"""
from __future__ import annotations

import re

from PySide6 import QtCore, QtGui, QtWidgets

from ..plot import figure_fonts as ff

__all__ = ["DEFAULT_ROW_LABEL", "FontListDialog", "FontPicker", "drawn_in_note", "figure_family",
           "figure_qfont", "html_font_family", "installed_families", "is_installed",
           "listable_families"]

#: The first row's words. It stores the UNSET value (``""``), so a project keeps saying
#: "default" and a later change of the default (the proposal's styles) reaches it.
DEFAULT_ROW_LABEL = f"{ff.DEFAULT_FAMILY} (default)"

#: Marks the ``Other font…`` row, so it is found by what it is and not by where it sits.
_OTHER_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 1

#: Characters that would end or corrupt a quoted ``font-family`` inside the single-quoted
#: ``style='…'`` attribute pyqtgraph writes its label HTML with.
_UNSAFE = re.compile(r"""['"<>;{}\\]""")

#: What Qt falls back to for a family none of whose substitutes is installed, by generic.
#: Monospace, not TypeWriter: Qt's typewriter hint picks a face that is not fixed-pitch.
_STYLE_HINTS = {"sans-serif": QtGui.QFont.StyleHint.SansSerif,
                "serif": QtGui.QFont.StyleHint.Serif,
                "monospace": QtGui.QFont.StyleHint.Monospace}


# -- what is installed, and what a figure is drawn in ----------------------------------------
def installed_families() -> list[str]:
    """Every family Qt lists, private ones included (they cannot be told apart from a
    missing one by name alone). The one place the drawing paths ask, so a test can replace
    it with the families of another computer."""
    return list(QtGui.QFontDatabase.families())


def listable_families() -> list[str]:
    """The families a person can name: the installed ones less the system-internal ones
    (a leading ``.``, or Qt's ``isPrivateFamily``), as ``QFontComboBox`` always listed them, and
    less the faces of symbols (``figure_fonts.SYMBOL_FAMILIES``), which are no font for text.
    A family an older project stored is still shown: the picker gives it a row of its own."""
    return [f for f in installed_families()
            if not f.startswith(".") and not QtGui.QFontDatabase.isPrivateFamily(f)
            and not ff.is_symbol_family(f)]


def is_installed(family: str | None) -> bool:
    """Whether this computer has ``family`` (case-insensitive). An unset family is Arial's."""
    name = ff.effective_family(family)
    return (name.casefold() in {f.casefold() for f in installed_families()}
            or QtGui.QFontDatabase.hasFamily(name))


def figure_family(family: str | None) -> str:
    """The name to give ``QFont.setFamily``: ``family`` (Arial when unset) if this computer
    has it, else the first of its fallback stack that it has -- so a missing Arial draws in
    Liberation Sans or Arimo, not in the system UI font."""
    return ff.resolve_installed(family, installed_families())


def drawn_in_note(words: str, family: str) -> str:
    """``words`` (a main-list row's text) with the family this computer really draws ``family``
    in, when that is another one: ``Helvetica (drawn in Arial on this computer)``, ``Arial
    (default; drawn in Liberation Sans on this computer)``. Text only -- what the row stores is
    the family, as before. Asked of :func:`figure_family`, not of the font database's
    ``hasFamily``, which also answers for a system alias."""
    drawn = figure_family(family)
    if drawn.casefold() == ff.effective_family(family).casefold():
        return words
    note = f"drawn in {drawn} on this computer"
    return f"{words[:-1]}; {note})" if words.endswith(")") else f"{words} ({note})"


def figure_qfont(family: str | None, base: QtGui.QFont | None = None) -> QtGui.QFont:
    """``base`` (the application font when omitted: size and weight are the caller's) in the
    figure family. The style hint is the family's generic, so a family none of whose
    substitutes is installed still falls back to a face of its own kind (a serif for Times
    New Roman) rather than to the system UI font."""
    font = QtGui.QFont(base) if base is not None else QtGui.QFont()
    font.setFamily(figure_family(family))
    font.setStyleHint(_STYLE_HINTS[ff.generic_family(family)])
    return font


def html_font_family(family: str | None) -> str:
    """The ``font-family`` for label HTML (pyqtgraph's ``LabelItem`` family option, an
    ``AxisItem``'s ``labelStyle``): the family's stack from ``figure_fonts``, but only the
    names this computer has, in order, and no generic keyword.

    Both are left out on purpose. Qt draws a glyph the first family lacks (the subscript zero
    of ``I₀`` in Times New Roman) in the next name of the list that has it, and measured on
    macOS a name it cannot find, like ``serif``, stands for the application font there: a
    private face that the exported PDF then embeds. With only names that are here the glyph
    comes from a later one that has it, else from the font Qt picks for it (Lucida Grande for
    that one). When none of the stack is here the list is the one family Qt picks for the
    figure's style hint, which is the face the tick numbers get too.

    The names that contain a space are in DOUBLE quotes: that HTML is written ``style='…'`` and
    a single quote would end the attribute (measured: the family is then silently dropped). A
    name that cannot be quoted safely is not used; the stack falls back to Arial's.

    With no ``QGuiApplication`` there is no font database to ask (Qt aborts the process rather
    than raising), so the whole stack is returned, quoted, for a caller that only builds text."""
    if family is not None and _UNSAFE.search(str(family)):
        family = None
    if QtGui.QGuiApplication.instance() is None:
        return ff.css_font_family(family, quote='"')
    have = {name.casefold() for name in installed_families()}
    names = [name for name in ff.fallback_stack(family)      # _STYLE_HINTS' keys: the generics
             if name not in _STYLE_HINTS and name.casefold() in have]
    if not names:
        names = [QtGui.QFontInfo(figure_qfont(family)).family()]
    return ", ".join(f'"{name}"' if " " in name else name for name in names)


# -- the "Other font…" list --------------------------------------------------------------------
class _FilterEdit(QtWidgets.QLineEdit):
    """The filter box. Up and Down move through the list under it, as in a search box."""

    stepped = QtCore.Signal(int)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - our own widget, not a view filter
        if event.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
            self.stepped.emit(-1 if event.key() == QtCore.Qt.Key.Key_Up else 1)
            event.accept()
            return
        super().keyPressEvent(event)


class FontListDialog(QtWidgets.QDialog):
    """Every installed family (none private or dot-prefixed, and no face of symbols such as
    Wingdings), a filter box, and a sample line in the highlighted family -- so a Greek or Å
    glyph can be checked before it is chosen.

    Modal through ``exec()``; :meth:`family` is the pick (None for Cancel or no row)."""

    def __init__(self, parent=None, *, current: str | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureFontListDialog")
        self.setWindowTitle("Other font")
        self.resize(340, 440)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Fonts installed on this computer:"))
        self._filter = _FilterEdit()
        self._filter.setObjectName("figureFontFilter")
        self._filter.setPlaceholderText("Filter fonts")
        self._filter.setClearButtonEnabled(True)
        self._filter.setAccessibleName("Filter fonts")
        layout.addWidget(self._filter)
        self._list = QtWidgets.QListWidget()
        self._list.setObjectName("figureFontList")
        self._list.setAccessibleName("Installed fonts")
        for name in dict.fromkeys(listable_families()):
            self._list.addItem(name)
        layout.addWidget(self._list, 1)
        self._sample = QtWidgets.QLabel("Aa  τ μ Δ  Å  0123")
        self._sample.setObjectName("figureFontSample")
        self._sample.setMinimumHeight(36)
        layout.addWidget(self._sample)
        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self._ok = self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._ok.setDefault(True)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._filter.textChanged.connect(self._apply_filter)
        self._filter.stepped.connect(self._step)
        self._list.currentItemChanged.connect(self._current_changed)
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())
        if not (current and self.select_family(current)):
            self._current_changed()

    def families(self) -> list[str]:
        """The families the list shows right now (the filter applied), in list order."""
        return [self._list.item(row).text() for row in range(self._list.count())
                if not self._list.item(row).isHidden()]

    def family(self) -> str | None:
        item = self._list.currentItem()
        return item.text() if item is not None and not item.isHidden() else None

    def select_family(self, family: str) -> bool:
        """Highlight ``family`` (case-insensitive), clearing the filter if it hides the row.
        False when this computer does not list it."""
        wanted = str(family).strip().casefold()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.text().casefold() == wanted:
                if item.isHidden():
                    self._filter.clear()
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)
                return True
        return False

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().casefold())
        current = self._list.currentItem()
        if current is None or current.isHidden():
            first = next((self._list.item(row) for row in range(self._list.count())
                          if not self._list.item(row).isHidden()), None)
            self._list.setCurrentItem(first)         # None clears it: nothing to pick
            if first is None:
                self._current_changed()
        elif current is not None:
            self._list.scrollToItem(current)

    def _step(self, delta: int) -> None:
        rows = [row for row in range(self._list.count()) if not self._list.item(row).isHidden()]
        if not rows:
            return
        here = self._list.currentRow()
        at = rows.index(here) if here in rows else (-1 if delta > 0 else len(rows))
        self._list.setCurrentRow(rows[max(0, min(len(rows) - 1, at + delta))])

    def _current_changed(self, *_args) -> None:
        name = self.family()
        self._ok.setEnabled(name is not None)
        if name is not None:
            font = QtGui.QFont(name)
            font.setPointSize(16)
            self._sample.setFont(font)


# -- the picker --------------------------------------------------------------------------------
class FontPicker(QtWidgets.QComboBox):
    """Arial (default), Helvetica, Times New Roman, Courier New, a separator, Other font….

    :meth:`value` is the family the figure should have: ``""`` for the first row (the unset
    value -- a project keeps saying "default"), else the family. :attr:`changed` says a person
    (or code that picks a row, as a test does) chose a different one; :meth:`set_value` moves
    the row to show a stored value and NEVER says anything, however the caller has blocked or
    not blocked this widget's signals, and restores the block it found.

    A stored family the list does not have -- an old project's ``Menlo``, an Other font pick --
    gets one extra row, right under the four, carrying its exact name; a family this computer
    lacks says so (``Calibri — not on this computer``) and is still kept. There is never more
    than one: another off-list family takes its place. An installed family's row stays when a
    listed row is picked, so it can be picked again; a missing family's goes, because it would
    advertise a font this computer cannot draw. A stored ``"courier new"`` shows the Courier
    New row and a stored ``"Arial"`` the first one, and :meth:`value` goes on returning what
    was stored, spelling and all, until a row is picked: showing a value never rewrites it.

    ``Other font…`` opens :class:`FontListDialog`; Cancel puts the previous row back. The
    dialog is modal, and a modal window closes any popup it is opened from, so a picker that
    sits in one (the Text style… popover) hides that popup while the dialog runs and shows
    it again after.

    ``first_row`` is ``(words, stored value)`` of the first row. The default is "Arial
    (default)" storing ``""``; the Text style… popover, where "unset" is the Custom font box
    being off and not Arial, passes ``("Arial", "Arial")``.

    A main-list family this computer lacks is still a row (a figure styled on a Mac opens
    unchanged elsewhere), and the row says what it is drawn in here: ``Helvetica (drawn in
    Arial on this computer)`` on Windows. :func:`drawn_in_note` writes that.
    """

    #: The family now chosen (``""`` = the first row's unset value).
    changed = QtCore.Signal(str)
    _other_requested = QtCore.Signal()

    def __init__(self, parent=None, *,
                 first_row: tuple[str, str] = (DEFAULT_ROW_LABEL, "")) -> None:
        super().__init__(parent)
        self.setObjectName("figureFontPicker")
        self.setAccessibleName("Font")
        self.setMaxVisibleItems(12)
        self.setMinimumContentsLength(12)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for index, family in enumerate(ff.FIGURE_FAMILIES):
            words, stored = first_row if index == 0 else (family, family)
            text = drawn_in_note(words, family)       # a family this computer lacks says so
            self.addItem(text, stored)
            self.setItemData(index, QtGui.QFont(family), QtCore.Qt.ItemDataRole.FontRole)
            if text != words:             # the narrow list may elide the note: hover reads it
                self.setItemData(index, text, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.insertSeparator(self.count())
        self.addItem(ff.OTHER_FONT_LABEL, "")
        self.setItemData(self.count() - 1, True, _OTHER_ROLE)
        self.setToolTip(
            f"Font for the figure's text. The first row is the figure default, "
            f"{ff.DEFAULT_FAMILY}. {ff.OTHER_FONT_LABEL} lists every font installed on this "
            f"computer except the system's private fonts and symbol fonts such as Wingdings.")
        self._has_extra = False
        self._asking = False
        self._unset = first_row[1]         # what the first row stores, and "no value" means
        self._value = self._unset          # the value last SET or picked, as written...
        self._value_row = 0                # ...and the row it was shown on
        self.currentIndexChanged.connect(self._on_index_changed)
        # Queued: the dialog opens after the pick has finished (a combo popup is still
        # closing while its own ``activated`` is delivered).
        self._other_requested.connect(self._choose_other, QtCore.Qt.ConnectionType.QueuedConnection)

    # -- the value ---------------------------------------------------------------------------
    def value(self) -> str:
        """The family chosen: what was last SET, as written, while its row is still the
        current one; else what the current row stores (``""`` for the unset first row)."""
        row = self.currentIndex()
        if row == self._value_row:
            return self._value
        data = self.itemData(row)
        return "" if data is None else str(data)

    def set_value(self, value: str | None) -> None:
        """Show ``value`` -- silently. ``None``, ``""`` and ``"Default"`` are the first row."""
        was_blocked = self.blockSignals(True)
        try:
            row = self._main_row(value)
            if row is None:
                stored = str(value).strip()
                self._show_extra(stored)
                row = len(ff.FIGURE_FAMILIES)
            else:
                stored = self._unset if ff.is_default(value) else str(value).strip()
                self._retire_extra()
            self.setCurrentIndex(row)
        finally:
            self.blockSignals(was_blocked)
        self._value, self._value_row = stored, row

    @staticmethod
    def _main_row(value: str | None) -> int | None:
        """The index among the four main rows, or None for a family the list does not have."""
        if ff.is_default(value):
            return 0
        if not ff.is_listed(value):
            return None
        return ff.FIGURE_FAMILIES.index(ff.canonical_family(value))

    def _show_extra(self, family: str) -> None:
        row = len(ff.FIGURE_FAMILIES)
        text = family if is_installed(family) else f"{family} — not on this computer"
        if not self._has_extra:
            self.insertItem(row, text, family)
            self._has_extra = True
        else:
            self.setItemText(row, text)
            self.setItemData(row, family)
        self.setItemData(row, QtGui.QFont(family), QtCore.Qt.ItemDataRole.FontRole)

    def _retire_extra(self) -> None:
        """Take the extra row down if it names a family this computer lacks. An installed
        family's row stays: it was chosen on purpose, and can be chosen again."""
        if self._has_extra and not is_installed(self.itemData(len(ff.FIGURE_FAMILIES))):
            self.removeItem(len(ff.FIGURE_FAMILIES))
            self._has_extra = False

    def has_extra_row(self) -> bool:
        """Whether the list carries the extra row for a family it does not have."""
        return self._has_extra

    # -- picks -------------------------------------------------------------------------------
    def _on_index_changed(self, index: int) -> None:
        if index < 0:
            return
        if self.itemData(index, _OTHER_ROLE):
            self._other_requested.emit()
            return
        if self._has_extra and index != len(ff.FIGURE_FAMILIES):
            # A pick of a listed row ends the extra row of a missing family. That row sits
            # BELOW the four, so taking it out moves no row above it, and the combo's own
            # signals stay quiet for the removal (it would renumber the current row and say so).
            was_blocked = self.blockSignals(True)
            try:
                self._retire_extra()
            finally:
                self.blockSignals(was_blocked)
        data = self.itemData(index)
        self._value, self._value_row = ("" if data is None else str(data)), index
        self.changed.emit(self._value)

    def _choose_other(self) -> None:
        if self._asking or not self.itemData(self.currentIndex(), _OTHER_ROLE):
            return
        self._asking = True
        before = self._value
        try:
            picked = self._ask(before)
        finally:
            self._asking = False
        if picked is not None and ff.is_listed(picked) and (
                ff.canonical_family(picked) == ff.DEFAULT_FAMILY):
            picked = self._unset              # Arial from the long list is the first row's value
        self.set_value(before if picked is None else picked)       # silent; Cancel = before
        if picked is not None and self._value != before:
            self.changed.emit(self._value)

    def _ask(self, current: str) -> str | None:
        """Run the list; the family picked, or None for Cancel."""
        popup = self.window()
        in_popup = popup is not None and popup.windowType() == QtCore.Qt.WindowType.Popup
        owner = popup.parentWidget() if in_popup and popup.parentWidget() is not None else popup
        dialog = FontListDialog(owner.window() if owner is not None else None,
                                current=current or None)
        if in_popup:
            popup.hide()
        try:
            accepted = bool(dialog.exec())
            return dialog.family() if accepted else None
        finally:
            if in_popup:
                popup.show()
            dialog.deleteLater()
