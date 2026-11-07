"""Figure-local series overrides travel with Undo, but not reusable looks/recipes."""
from dataclasses import replace
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget


def test_series_state_roundtrip_and_look_boundaries(qtbot):
    state = PlotAppearance(title='Literal', series_styles={'source':{'color':'#123456','line_width_pt':.625}})
    assert PlotAppearance.from_mapping(state.to_dict()) == state
    assert state.style_only().series_styles is None
    assert state.template_form().series_styles is None
    assert state.restyled(PlotAppearance(font_family='Arial')).series_styles == state.series_styles
    widget=PlotAppearanceWidget();qtbot.addWidget(widget)
    widget.set_state(state);widget.clear_history()
    changed=replace(state,series_styles={'source':{'visible':False}})
    widget.apply_appearance(changed)
    assert 'series formatting' in widget.undo_button.toolTip()
    assert widget.undo() and widget.state() == state
    assert widget.redo() and widget.state() == changed
