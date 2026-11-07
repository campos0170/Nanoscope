"""Project toolbar aliases use the Figures commands and fit the narrow toolbar."""
from PySide6 import QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer
from . import p1_hosting as H
from .p2_shell_rig import Controller, record


def hosted(qtbot):
    value = record(kind='iv')
    owner = Controller(value)
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    page = H.SyntheticPlotPage('iv')
    explorer.show_page(value.spec.figure_id, page, kind='iv')
    explorer.resize(840, 560)
    explorer.show()
    qtbot.waitExposed(explorer)
    QtWidgets.QApplication.processEvents()
    return explorer, owner, value, page


def test_new_and_duplicate_toolbar_aliases_share_menu_selection_and_existing_commands(qtbot):
    explorer, owner, value, page = hosted(qtbot)
    bar = explorer.toolbar(); figures = explorer.figures_list()
    assert bar.new_figure_button.menu() is figures.new_menu
    assert bar.new_figure_button.isVisible()
    assert bar.widgetForAction(bar.duplicate_action).isVisible()
    assert bar.duplicate_action.isEnabled()
    bar.duplicate_action.trigger()
    assert ('duplicate', value.spec.figure_id) in owner.calls and page.updates == 0
    assert not bar.reset_figure_button.isVisible()
    assert bar._reset_from_menu.isVisible() and bar._reset_from_menu.isEnabled()
    assert bar.minimumSizeHint().width() <= 840
    assert bar.update_button.isVisible() and bar.export_button.isVisible()
    bar.new_figure_button.menu().actions()[0].trigger()
    assert any(call[:3] == ('create', 'distributions', {'plot_type': 'Histogram'}) for call in owner.calls)


def test_snapshot_names_keep_compatible_signal_routes_and_standalone_labels(qtbot):
    explorer, owner, value, _page = hosted(qtbot)
    bar = explorer.toolbar(); bar.set_library_enabled(True)
    seen = []
    explorer.saveFigureRequested.connect(lambda key: seen.append(('snapshot', key)))
    explorer.saveBundleRequested.connect(lambda key: seen.append(('bundle', key)))
    explorer.figureLibraryRequested.connect(lambda: seen.append(('details',)))
    bar._fill_export_menu()
    actions = {action.text(): action for action in bar._export_menu.actions()}
    for label in ('Snapshot to Figures…', 'Bundle…', 'Snapshot details…'):
        assert actions[label].isEnabled()
        actions[label].trigger()
    assert seen == [('snapshot', value.spec.figure_id), ('bundle', value.spec.figure_id), ('details',)]
    assert 'Open Figure Library' not in actions
    explorer.bind_figure_workspace(None)
    assert all(not action.isVisible() for action in bar.leading_actions)
    assert not bar._reset_from_menu.isVisible()
    bar._fill_export_menu()
    assert 'Open Figure Library' in [action.text() for action in bar._export_menu.actions()]
