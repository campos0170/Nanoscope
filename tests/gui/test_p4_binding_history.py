"""Style metadata-only edits remain restorable Undo steps."""
from dataclasses import replace
from nanoscope.core.figure_style import FigureStyle
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget, _describe_step


def test_column_revision_only_history_restores_and_undoes_exact_values(qtbot):
    old = FigureStyle('project.paper', 1, 'Paper', columns={})
    new = replace(old, revision=2, columns={'single_mm': 89.})
    before = PlotAppearance.from_mapping(bind_native_style({}, old))
    after = PlotAppearance.from_mapping(bind_native_style({}, new))
    assert before.to_dict().keys() == after.to_dict().keys()
    first = PlotAppearanceWidget(); second = PlotAppearanceWidget()
    qtbot.addWidget(first); qtbot.addWidget(second)
    first.set_state(before); first.clear_history(); first.apply_appearance(after)
    capsule = first.capture_history_capsule()
    second.set_state(after); second.restore_history_capsule(capsule)
    assert 'Paper' in second.undo_button.toolTip()
    assert second.undo() and second.state() == before
    assert second.redo() and second.state() == after
    assert _describe_step(after, replace(after, style_binding=None)) == 'figure style reset'


def test_basis_only_history_description_does_not_dump_or_reinterpret_metadata():
    before = PlotAppearance()
    after = replace(before, quantity_display={'version': 1, 'axes': {}})
    assert _describe_step(before, after) == 'display units'
