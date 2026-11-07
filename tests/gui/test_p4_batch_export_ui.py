"""Batch controls operate on selected IDs even on Home, without activating rows."""
from pathlib import Path
from types import SimpleNamespace
from PySide6 import QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.figure_batch_export import BatchExportResult
from nanoscope.gui.figure_batch_export_dialog import BatchExportOptionsDialog, show_batch_export_results
from .p2_shell_rig import Controller, record


def test_restoring_live_and_saved_selection_emits_once_without_activation(qtbot):
    live = record('Live')
    owner = Controller(live)
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    saved = SimpleNamespace(artifact_id='saved', payload={'name': 'Snapshot'})
    explorer.set_saved_figure_provider(lambda: (saved,)); explorer.refresh_figures()
    figures = explorer.figures_list()
    activations, selections = [], []
    figures.liveActivated.connect(activations.append); figures.savedActivated.connect(activations.append)
    figures.selectionChanged.connect(lambda: selections.append(True))
    figures.restore_selection((live.spec.figure_id, 'saved'), current_id='saved')
    assert activations == [] and selections == [True]
    assert figures.selected_ids() == (live.spec.figure_id,) and figures.selected_ids('saved') == ('saved',)
    assert figures.current_id('snapshot') == 'saved'
    explorer.show_home()
    bar = explorer.toolbar(); bar._fill_export_menu()
    action = next(action for action in bar._export_menu.actions() if action.text() == 'Export selected figures…')
    assert bar.export_button.isEnabled() and action.isEnabled() and explorer.page_count() == 0
    fired = []
    explorer.exportSelectedRequested.connect(lambda: fired.append(True)); action.trigger()
    assert fired == [True] and activations == []
    figures.restore_selection(())
    bar._fill_export_menu()
    assert not next(a for a in bar._export_menu.actions() if a.text() == 'Export selected figures…').isEnabled()


def test_batch_options_validate_existing_folder_and_default_to_pdf_style_canvas(qtbot, tmp_path):
    dialog = BatchExportOptionsDialog(None, ('First', 'Second'))
    qtbot.addWidget(dialog)
    assert dialog.names.toPlainText() == 'First\nSecond'
    assert not dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).isEnabled()
    dialog.directory.setText(str(tmp_path))
    assert dialog.values() == (tmp_path, ('pdf',), 'style', 300)
    dialog.formats['pdf'].setChecked(False)
    assert not dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).isEnabled()
    dialog.formats['png'].setChecked(True); dialog.sizing.setCurrentIndex(dialog.sizing.findData('single'))
    assert dialog.dpi.isEnabled()


def test_result_dialog_reports_actual_statuses_and_is_nonmodal(qtbot, tmp_path):
    owner = QtWidgets.QWidget(); qtbot.addWidget(owner)
    results = (BatchExportResult('a', 'First', tmp_path/'a.pdf', 'written'),
               BatchExportResult('b', 'Second', tmp_path/'b.pdf', 'failed', 'Disk unavailable'))
    dialog = show_batch_export_results(owner, results)
    assert not dialog.isModal()
    text = dialog.findChild(QtWidgets.QPlainTextEdit, 'figureExportResults').toPlainText()
    assert 'First — written' in text and 'Second — failed' in text and 'Disk unavailable' in text
    assert str(tmp_path/'b.pdf') in text
    dialog.close()
