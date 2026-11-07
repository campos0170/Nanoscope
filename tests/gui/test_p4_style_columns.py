"""Editable physical widths become pinned style revisions, never implicit resizes."""
import pytest
from PySide6 import QtWidgets

from nanoscope.core.figure_spec import thaw_json
from nanoscope.core.figure_style import FigureStyle
from nanoscope.gui.figure_export_style import column_presets
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.project_style_editor import ProjectStyleEditor
from .test_p2_workspace_controller import rig  # noqa: F401


def test_column_editor_preserves_exact_untouched_fields_and_native_look(qtbot):
    style = FigureStyle('project.precise', 1, 'Precise',
        look={'title_size_pt': 9.1234567, 'text_roles': {'title': {'bold': True}},
              'future': {'literal': [None, 'preserve']}},
        columns={'single_mm': 88.123456789, 'double_mm': 177.123456789,
                 'basis': 'imported', 'label': 'Lab values', 'future': {'height': 61.23}})
    dialog = ProjectStyleEditor(style)
    qtbot.addWidget(dialog)
    assert dialog.look() == thaw_json(style.look)
    assert dialog.columns() == thaw_json(style.columns)
    dialog.column_widths['single_mm'].setValue(92.125)
    assert dialog.columns() == {**thaw_json(style.columns), 'single_mm': 92.125,
                                'basis': 'user-defined', 'label': 'User-defined columns'}
    assert dialog.look() == thaw_json(style.look)
    dialog.column_widths['double_mm'].setValue(0)
    assert 'double_mm' not in dialog.columns()
    assert dialog.columns()['future'] == {'height': 61.23}


def test_editing_existing_look_keeps_its_columns_without_an_explicit_edit(rig):
    owner = rig.owner
    original = owner.save_style('Lab', look={'title_size_pt': 9.},
        columns={'single_mm': 85., 'double_mm': 170., 'extra': {'keep': True}})
    revised = owner.save_style('Lab', overwrite=True, look={'title_size_pt': 10.})
    assert revised.columns == original.columns and revised.revision == 2
    before = owner.style_registry()
    with pytest.raises(ValueError):
        owner.save_style('Lab', overwrite=True, look={}, columns={'single_mm': 0})
    assert owner.style_registry() == before


def test_new_width_revision_does_not_rewrite_pinned_existing_figure_or_hydrate(rig, monkeypatch):
    owner = rig.owner
    monkeypatch.setattr(owner, 'activate', lambda _key: False)
    first = owner.save_style('Paper', look={'title_size_pt': 7.},
        columns={'single_mm': 85., 'double_mm': 170.})
    owner.set_default_style(first.ref)
    key = owner.create('iv')
    previous = owner.get(key)
    revised = owner.save_style('Paper', overwrite=True, look=thaw_json(first.look),
        columns={'single_mm': 92., 'double_mm': 185., 'basis': 'user-defined'})
    assert owner.get(key) == previous and not rig.built
    pinned = FigureStyle.from_dict(owner.get(key).spec.presentation.style_snapshot)
    assert [width for _, width in column_presets(pinned)[0]] == [85., 170.]
    next_key = owner.create('iv')
    newest = FigureStyle.from_dict(owner.get(next_key).spec.presentation.style_snapshot)
    assert newest.ref == revised.ref
    assert [width for _, width in column_presets(newest)[0]] == [92., 185.]
    owner.install_workspace(owner.capture_workspace())
    assert FigureStyle.from_dict(owner.get(key).spec.presentation.style_snapshot).columns == first.columns
    assert next(style for style in owner.available_styles() if style.name == 'Paper') == revised
    assert not rig.built


def test_save_current_look_keeps_that_figures_widths_instead_of_the_new_project_default(rig, monkeypatch):
    owner = rig.owner
    monkeypatch.setattr(owner, 'activate', lambda _key: False)
    style = owner.save_style('Figure widths', look={'title_size_pt': 9.},
                             columns={'single_mm': 96., 'double_mm': 190.})
    key = owner.create('iv')
    assert owner.apply_look_to_figure(key, PlotAppearance.from_mapping(bind_native_style({}, style)))
    owner._current = key
    copied = owner.save_style('Copied look')
    assert copied.columns == style.columns
    assert copied.columns != owner.default_style().columns
    assert not rig.built


def test_manager_forwards_columns_as_one_new_revision(qtbot, rig, monkeypatch):
    from nanoscope.gui.analysis_explorer import AnalysisExplorer
    from nanoscope.gui.plot_styles import _ManageLooks
    owner = rig.owner
    first = owner.save_style('Editable widths', look={'title_size_pt': 7.},
                             columns={'single_mm': 89., 'double_mm': 183.})
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    manager = _ManageLooks(explorer, explorer._style)
    qtbot.addWidget(manager)
    manager.refill()
    manager.list.setCurrentRow(next(i for i in range(manager.list.count())
        if manager.list.item(i).text() == first.name))
    def edit(dialog):
        dialog.column_widths['single_mm'].setValue(95.)
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(ProjectStyleEditor, 'exec', edit)
    manager._edit()
    changed = next(style for style in owner.available_styles() if style.name == first.name)
    assert changed.revision == 2 and changed.columns['single_mm'] == 95.
    assert changed.columns['double_mm'] == 183. and changed.look == first.look
    assert not rig.built
