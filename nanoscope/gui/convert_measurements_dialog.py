"""Pick which old manual measurements become events (docs/MANUAL_EVENT_TABLE_DESIGN.md, 10).

The list shows every measurement the project holds, ticked where it can be placed on the channel
on screen and greyed, with the reason, where it cannot. A ticked row becomes an accepted,
reviewed hand-made event in the Manual table; nothing is converted unticked, because a
measurement that was a baseline or a noise check is not an event and the event model has no
neutral state. The originals stay in the measurement list.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..analysis.measurement_conversion import ConversionRow

__all__ = ["ConvertMeasurementsDialog"]

_COLUMNS = ("", "Id", "Recording", "Channel", "Samples", "Label", "Status")


class ConvertMeasurementsDialog(QtWidgets.QDialog):
    """A checklist of :class:`ConversionRow`; ``selected_ids`` after ``exec`` is the answer."""

    def __init__(self, rows: list[ConversionRow], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert old measurements to events")
        self.setMinimumWidth(760)
        self._rows = list(rows)
        root = QtWidgets.QVBoxLayout(self)
        n_ok = sum(1 for r in rows if r.placeable)
        intro = QtWidgets.QLabel(
            f"{len(rows)} saved measurement{'s' if len(rows) != 1 else ''}; {n_ok} can be placed "
            "on the channel now selected. A ticked row becomes an accepted, reviewed hand-made "
            "event in this channel's Manual table, measured the way that table measures its "
            "events, so its numbers can differ from the stored measurement. Untick anything "
            "that was a baseline or noise check. The measurement list itself is unchanged, and "
            "Edit ▸ Undo takes the conversion back as one step.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        self.table = QtWidgets.QTableWidget(len(rows), len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for i, r in enumerate(rows):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
                           if r.placeable else QtCore.Qt.NoItemFlags)
            check.setCheckState(QtCore.Qt.Checked if r.placeable else QtCore.Qt.Unchecked)
            self.table.setItem(i, 0, check)
            status = r.reason or ("numbers will change (another calibration or basis)"
                                  if r.numbers_change else "ready")
            for j, text in enumerate((str(r.measurement_id), r.recording, r.channel,
                                      f"{r.start:,} – {r.end:,}", r.label, status), start=1):
                item = QtWidgets.QTableWidgetItem(text)
                if not r.placeable:
                    item.setFlags(QtCore.Qt.NoItemFlags)
                self.table.setItem(i, j, item)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()
        root.addWidget(self.table)
        buttons = QtWidgets.QDialogButtonBox()
        self.convert_button = buttons.addButton("Convert", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def selected_ids(self) -> list[int]:
        """The measurement ids whose box is ticked."""
        return [r.measurement_id for i, r in enumerate(self._rows)
                if r.placeable and self.table.item(i, 0).checkState() == QtCore.Qt.Checked]
