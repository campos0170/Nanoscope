"""Read-only About/build details with an explicit, local clipboard action."""
from PySide6 import QtCore, QtWidgets

from ..build_info import build_info, diagnostic_summary


class BuildInfoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About NanoScope · build and diagnostics")
        self.resize(760, 560)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout = QtWidgets.QVBoxLayout(self)
        record = build_info()
        heading = QtWidgets.QLabel(f"NanoScope {record.get('version', 'unknown')}\n{record.get('build_id', 'unavailable')}")
        heading.setTextFormat(QtCore.Qt.PlainText)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        note = QtWidgets.QLabel("This identifies the app currently running—not another app or DMG on disk. "
                               "Copy diagnostics copies the text below locally; nothing is sent or uploaded.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.details = QtWidgets.QPlainTextEdit(diagnostic_summary(record))
        self.details.setReadOnly(True)
        self.details.setAccessibleName("Local build and runtime diagnostics")
        layout.addWidget(self.details, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.copy_button = buttons.addButton("Copy diagnostics", QtWidgets.QDialogButtonBox.ActionRole)
        self.copy_button.clicked.connect(self.copy_diagnostics)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def copy_diagnostics(self):
        QtWidgets.QApplication.clipboard().setText(self.details.toPlainText())


def show_build_info(parent=None):
    """Modeless, owned by its parent; the caller may retain the returned dialog."""
    dialog = BuildInfoDialog(parent)
    dialog.show()
    return dialog
