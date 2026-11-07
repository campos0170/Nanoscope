"""Physical canvas defaults are optional style metadata, never analytical layout edits."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nanoscope.core.figure_style import FigureStyle
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.figure_export_style import canvas_dimensions, export_style_for
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plot_export_options import ExportOptionsDialog
from .test_p2_workspace_controller import rig  # noqa: F401


@pytest.mark.parametrize('value', (0, -.1, .01, 2001, float('nan'), float('inf'), True))
def test_canvas_size_validates_known_values_and_leaves_undefined_native(value):
    with pytest.raises(ValueError):
        FigureStyle('project.invalid', 1, 'Invalid', {'canvas_width_mm': value})
    assert canvas_dimensions(FigureStyle('project.native', 1, 'Native', {}), (800, 400)) is None


def test_export_defaults_use_explicit_size_or_one_dimension_with_captured_aspect(qtbot):
    style = FigureStyle('project.canvas', 1, 'Canvas', {'canvas_width_mm': 85., 'canvas_height_mm': 60.})
    dialog = ExportOptionsDialog(native_size=(800, 400), figure_style=style); qtbot.addWidget(dialog)
    options = dialog.options()
    assert (options.physical_width_mm, options.physical_height_mm) == (85., 60.)
    assert not dialog.aspect_check.isChecked()
    width_only = replace(style, look={'canvas_width_mm': 85.})
    assert canvas_dimensions(width_only, (800, 400)) == (85., 42.5)
    height_only = replace(style, look={'canvas_height_mm': 60.})
    assert canvas_dimensions(height_only, (800, 400)) == (120., 60.)
    native = ExportOptionsDialog(native_size=(800, 400), figure_style=replace(style, look={}))
    qtbot.addWidget(native)
    assert native.options().native
    # Existing explicit initial dimensions keep priority over style defaults.
    explicit = ExportOptionsDialog(native_size=(800, 400), figure_style=style,
                                   initial_format='pdf', initial_width=72, initial_height=144)
    qtbot.addWidget(explicit)
    assert explicit.options().physical_width_mm == pytest.approx(25.4)
    assert explicit.options().physical_height_mm == pytest.approx(50.8)


def test_canvas_style_revisions_remain_pinned_through_dormancy_project_save_and_undo(rig, monkeypatch):
    owner = rig.owner; monkeypatch.setattr(owner, 'activate', lambda _key: False)
    first = owner.save_style('Physical canvas', look={'canvas_width_mm': 85., 'canvas_height_mm': 60.})
    owner.set_default_style(first.ref)
    key = owner.create('iv'); original = owner.get(key)
    revised = owner.save_style('Physical canvas', overwrite=True, look={'canvas_width_mm': 90.})
    assert original.spec.presentation.style_ref == first.ref.to_dict()
    second = owner.create('iv')
    assert owner.get(second).spec.presentation.style_ref == revised.ref.to_dict()
    look = PlotAppearance.from_mapping(bind_native_style({}, revised))
    assert owner.apply_look_to_figure(key, look, keep_hand=False)
    assert owner.undo(key)
    assert owner.get(key).spec.presentation == original.spec.presentation
    workspace = owner.capture_workspace(); owner.install_workspace(workspace)
    assert owner.get(key).spec.presentation == original.spec.presentation and not rig.built


def test_canvas_export_resolves_figure_override_without_modifying_binding():
    style = FigureStyle('project.canvas', 1, 'Canvas', {'canvas_width_mm': 85.})
    native = bind_native_style({}, style)
    native['style_binding']['overrides']['canvas_width_mm'] = 91.
    owner = SimpleNamespace(appearance_state=lambda: native)
    effective = export_style_for(owner)
    assert canvas_dimensions(effective, (800, 400)) == (91., 45.5)
    assert native['style_binding']['snapshot']['look']['canvas_width_mm'] == 85.


def test_editor_canvas_values_are_separate_from_column_presets_and_clear_to_native(qtbot):
    from nanoscope.gui.project_style_editor import ProjectStyleEditor
    style = FigureStyle('project.canvas', 1, 'Canvas', {'canvas_width_mm': 85.12345},
                        columns={'single_mm': 89., 'double_mm': 183.})
    dialog = ProjectStyleEditor(style); qtbot.addWidget(dialog)
    assert dialog.look()['canvas_width_mm'] == 85.12345  # untouched precision is preserved
    dialog.canvas_sizes['canvas_height_mm'].setValue(61.25)
    assert dialog.look()['canvas_height_mm'] == 61.25
    assert dialog.columns() == dict(style.columns)
    dialog.canvas_sizes['canvas_width_mm'].setValue(0)
    assert 'canvas_width_mm' not in dialog.look()
