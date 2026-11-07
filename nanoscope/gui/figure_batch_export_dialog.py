"""Choices and truthful per-file results for frozen serial figure export."""
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from .export_paths import ask_export_directory


class BatchExportOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent, names):
        super().__init__(parent)
        self.setWindowTitle('Export selected figures')
        self.resize(560, 520)
        self._selection = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('Selected figures'))
        self.names = QtWidgets.QPlainTextEdit()
        self.names.setReadOnly(True)
        self.names.setPlainText('\n'.join(str(name) for name in names))
        self.names.setMaximumHeight(140)
        layout.addWidget(self.names)
        note = QtWidgets.QLabel('Exports committed pictures without running analysis. Resizing scales all '
            'artwork, including fonts. Column widths come from each captured style; reference widths '
            'are not publication requirements. Saved snapshots keep their captured look.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        folder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(folder); row.setContentsMargins(0, 0, 0, 0)
        self.directory = QtWidgets.QLineEdit()
        self.directory.setPlaceholderText('Choose an existing export folder')
        browse = QtWidgets.QPushButton('Choose…')
        browse.clicked.connect(self._browse)
        row.addWidget(self.directory, 1); row.addWidget(browse)
        form.addRow('Folder', folder)
        formats = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(formats); row.setContentsMargins(0, 0, 0, 0)
        self.formats = {}
        for key, label in (('pdf', 'PDF'), ('svg', 'SVG'), ('png', 'PNG'), ('zip', 'Bundle')):
            box = QtWidgets.QCheckBox(label); self.formats[key] = box
            box.setChecked(key == 'pdf'); box.toggled.connect(self._validate)
            row.addWidget(box)
        form.addRow('Formats', formats)
        self.sizing = QtWidgets.QComboBox()
        for label, value in (('Default canvas from each style (Native if unset)', 'style'),
                             ('Native captured size', 'native'),
                             ('Single column from each style', 'single'),
                             ('Double column from each style', 'double')):
            self.sizing.addItem(label, value)
        form.addRow('Image size', self.sizing)
        self.dpi = QtWidgets.QSpinBox(); self.dpi.setRange(36, 2400); self.dpi.setValue(300)
        form.addRow('PNG DPI', self.dpi)
        layout.addLayout(form)
        self.warning = QtWidgets.QLabel('')
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save |
                                                   QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setText('Continue')
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.directory.textChanged.connect(self._validate)
        self.sizing.currentIndexChanged.connect(self._validate)
        self._validate()

    def _browse(self):
        path = ask_export_directory(self, suggestion=self.directory.text())
        if path is not None: self.directory.setText(str(path))

    def values(self):
        directory = Path(self.directory.text().strip()).expanduser()
        if not self.directory.text().strip() or not directory.is_dir():
            raise ValueError('Choose an existing export folder.')
        formats = tuple(key for key, box in self.formats.items() if box.isChecked())
        if not formats: raise ValueError('Choose at least one output format.')
        return directory.absolute(), formats, self.sizing.currentData(), self.dpi.value()

    def _validate(self, *_args):
        if not hasattr(self, 'buttons'): return
        self.dpi.setEnabled(self.formats['png'].isChecked() and self.sizing.currentData() != 'native')
        try:
            self.values(); message = ''
        except ValueError as exc:
            message = str(exc)
        self.warning.setText(message)
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setEnabled(not message)

    def accept(self):
        try: self._selection = self.values()
        except ValueError:
            self._validate(); return
        super().accept()


def choose_batch_export_options(parent, names):
    dialog = BatchExportOptionsDialog(parent, tuple(names))
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dialog._selection


def show_batch_export_results(parent, results):
    """Show completed file statuses without a modal success prompt or inferred totals."""
    results = tuple(results)
    held = getattr(parent, '_figure_export_results', []) if parent is not None else []
    for old in tuple(held):
        if getattr(old, '_batch_results_closed', False):
            held.remove(old); old.deleteLater()
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle('Figure export results')
    dialog.resize(720, 460)
    dialog.setModal(False)
    dialog._batch_results_closed = False
    layout = QtWidgets.QVBoxLayout(dialog)
    counts = {key: sum(result.status == key for result in results) for key in ('written', 'failed', 'cancelled')}
    summary = QtWidgets.QLabel(f"{counts['written']} file(s) written · {counts['failed']} failed · {counts['cancelled']} cancelled")
    layout.addWidget(summary)
    detail = QtWidgets.QPlainTextEdit()
    detail.setReadOnly(True)
    detail.setObjectName('figureExportResults')
    detail.setPlainText('\n\n'.join(
        f'{result.name} — {result.status}\n{result.path}' + (f'\n{result.error}' if result.error else '')
        for result in results))
    layout.addWidget(detail, 1)
    close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    close.rejected.connect(dialog.reject); layout.addWidget(close)
    held.append(dialog)
    if parent is not None: parent._figure_export_results = held
    dialog.finished.connect(lambda _code: setattr(dialog, '_batch_results_closed', True))
    dialog.show()
    return dialog
