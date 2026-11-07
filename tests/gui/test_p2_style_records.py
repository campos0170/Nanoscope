"""Style reaches unloaded records and binds batch Undo to durable revision identity."""
from PySide6 import QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.plot_appearance import PlotAppearance

from .p2_shell_rig import Controller, record


def _explorer(qtbot):
    owner = Controller(record("A"), record("B"), record("C"))
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    return explorer, owner


def test_selected_and_all_style_unloaded_records_without_building_surfaces(qtbot):
    explorer, owner = _explorer(qtbot)
    ids = [rec.spec.figure_id for rec in owner.records()]
    owner.set_selected(ids[1:])
    result = explorer.apply_look(PlotAppearance(title_size=9), scope="selected", keep_hand=False)
    assert result.applied == tuple(ids[1:]) and not result.failed
    assert [call[1] for call in owner.calls if call[0] == "style"] == ids[1:]
    assert explorer.page_count() == 0
    result = explorer.apply_look(PlotAppearance(title_size=8), scope="all")
    assert result.applied == tuple(ids) and explorer.page_count() == 0
    assert explorer._style.batch_intact()
    assert explorer._style.undo_last_batch() == 3
    assert explorer._style.last_batch() is None


def test_batch_survives_surface_change_but_not_edit_or_record_replacement(qtbot):
    explorer, owner = _explorer(qtbot)
    ids = [rec.spec.figure_id for rec in owner.records()]
    explorer.apply_look(PlotAppearance(title_size=8), scope="all")
    # A residency-only notification carries the same revision token and remains undoable.
    owner.recordChanged.emit(ids[0])
    owner.recordsChanged.emit()
    assert explorer._style.batch_intact()
    owner._generation[ids[0]] += 1
    owner.recordChanged.emit(ids[0])
    assert not explorer._style.batch_intact() and explorer._style.undo_last_batch() == 0
    explorer.apply_look(PlotAppearance(title_size=7), scope="all")
    replacement = record("Replacement")
    replacement.spec.figure_id = ids[1]
    owner._records[ids[1]] = replacement
    owner.recordsChanged.emit()
    assert not explorer._style.batch_intact() and explorer._style.undo_last_batch() == 0


def test_project_rebind_drops_batch_and_record_menu_has_explicit_scopes(qtbot):
    explorer, owner = _explorer(qtbot)
    explorer.apply_look(PlotAppearance(title_size=8), scope="all")
    explorer._style_menu._sync()
    assert explorer._style_menu.scope_menu("selected").menuAction().isVisible()
    assert explorer._style_menu.scope_menu("all").title() == "All live figures"
    explorer.bind_figure_workspace(Controller(record("Next project")))
    assert explorer._style.last_batch() is None
    owner.recordsChanged.emit()
    QtWidgets.QApplication.processEvents()
    assert explorer._style.undo_last_batch() == 0
