"""Read-only About/build details with an explicit, local clipboard action."""
from PySide6 import QtCore, QtWidgets

from ..build_info import build_info, diagnostic_summary


class BuildInfoDialog(QtWidgets.QDialog):
    """Modeless, parent-owned. Never ``WA_DeleteOnClose``: MainWindow's stop-holders
    teardown sweep finds every QObject in its tree via ``findChildren`` and calls
    ``close()`` on each window-like one as a stop mechanism (see
    ``MainWindow._stop_holders``). That is a hide, not a delete, for every surface
    window in this app on purpose (``main_window.py`` documents the same invariant
    for Plots/Explorer/Inspector) -- ``WA_DeleteOnClose`` here turned an ordinary
    Close click, or that teardown sweep, into a C++-side delete under a Python
    wrapper that ``_open_build_info`` may still hold (see ``MainWindow._build_info_dialog``),
    which is a use-after-free waiting for its next signal. Guarded by
    ``test_fup_wiring_and_suite_green.py::test_no_gui_widget_turns_close_into_a_delete``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About NanoScope · build and diagnostics")
        self.resize(760, 560)
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
    """Modeless, owned by its parent; the caller may retain the returned dialog.

    Always constructs a fresh dialog -- callers that want at most one open instance
    (MainWindow does) must retain and reuse the return value themselves; this
    function has no static state to reuse across calls."""
    dialog = BuildInfoDialog(parent)
    dialog.show()
    return dialog
