"""Friendly column labels while existing scientific callers retain stable keys."""
from PySide6 import QtCore, QtWidgets
from ..core.quantity_display import column_entry, quantity_for


class ScientificColumnCombo(QtWidgets.QComboBox):
    """The native plotting API's currentText/setCurrentText still exchange raw keys.

    Qt paints the model's friendly DisplayRole. UserRole and Python accessors carry
    the schema key; labels can change without changing a saved recipe or selection.
    """
    def addItems(self, keys):  # noqa: N802
        for key in keys:
            raw, label, tip = column_entry(str(key), quantity_for(str(key)))
            super().addItem(label, raw)
            self.setItemData(self.count() - 1, tip, QtCore.Qt.ItemDataRole.ToolTipRole)

    def selection_key(self, index):
        """Raw alias consumed by the shared raising programmatic setter."""
        return str(self.itemData(index) or '')

    def currentText(self):  # noqa: N802
        return str(self.currentData() or '')

    def findText(self, text, flags=QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchCaseSensitive):  # noqa: N802
        index = self.findData(str(text))
        return index if index >= 0 else super().findText(text, flags)

    def setCurrentText(self, text):  # noqa: N802
        index = self.findText(str(text))
        if index >= 0:
            self.setCurrentIndex(index)
