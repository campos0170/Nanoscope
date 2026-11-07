"""Fractional native look values survive projection, persistence and history."""
from dataclasses import replace

import nanoscope  # noqa: F401 bootstrap before Qt
import pytest

from nanoscope.gui.plots_panel import (
    PlotConfig, StyleSettings, plot_config_from_json, plot_config_to_json)
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.figure_adapter_plots import _restyled_config
from . import p1_hosting as H


def test_fractional_legacy_style_config_round_trip():
    config = replace(PlotConfig(), style=replace(StyleSettings(), line_width=.75,
        point_size=3.5, font_size_pt=8.25, title_font_size_pt=9.5,
        axis_label_font_size_pt=8.25, tick_font_size_pt=7.5, legend_font_size_pt=7.25))
    assert plot_config_from_json(plot_config_to_json(config)) == config
    old = plot_config_to_json(PlotConfig())
    old["style"]["line_width"] = 1
    old["style"]["point_size"] = 7
    assert plot_config_from_json(old).style.line_width == 1.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_geometry_is_not_a_style_size(value):
    with pytest.raises(ValueError):
        replace(StyleSettings(), line_width=value)


def test_fractional_style_matches_dormant_projection_and_one_live_undo(qtbot):
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    page._history_commit()
    before = page.config()
    look = PlotAppearance(line_width=.75, marker_size=3.5, title_size=9.5,
                          axis_size=8.25, tick_size=7.5, legend_size=7.25)
    expected = _restyled_config(before, look, False)
    assert page.inspector_adapter().apply_look(look, False)
    assert page.config() == expected
    assert page._line_width.value() == .75
    assert page._point_size.value() == 3.5
    assert page._qfont(page._tick_pt()).pointSizeF() == 7.5
    assert page._data_pen().widthF() == .75
    assert page.undo_config() and page.config() == before


def test_style_reference_changes_are_inside_the_native_undo_step(qtbot):
    from nanoscope.core.figure_style import builtin_styles
    from nanoscope.gui.figure_style_binding import bind_native_style
    page = H.page_builders()["distributions"]()
    qtbot.addWidget(page)
    page._history_commit()
    before = page.config()
    first = builtin_styles()[0]
    look = PlotAppearance.from_mapping(bind_native_style({}, first))
    assert page.inspector_adapter().apply_look(look, False)
    applied = page.config()
    assert applied.style.style_binding["ref"]["style_id"] == first.style_id
    same_look_new_identity = replace(first, style_id="project.same-look", name="A separate style")
    second = PlotAppearance.from_mapping(bind_native_style({}, same_look_new_identity))
    assert page.inspector_adapter().apply_look(second, False)
    assert page.config().style.style_binding["ref"]["style_id"] == "project.same-look"
    assert page.undo_config() and page.config() == applied
    assert page.undo_config() and page.config() == before
    assert page.redo_config() and page.config() == applied


def test_shared_text_roles_draw_without_creating_explicit_overrides(qtbot, monkeypatch):
    import pyqtgraph as pg
    from nanoscope.core.figure_style import FigureStyle
    from nanoscope.gui.figure_style_binding import bind_native_style
    from nanoscope.gui.plot_appearance import apply_plot_appearance
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    plot.setTitle("Apo")
    plot.setLabel("bottom", "Dwell")
    plot.addLegend()
    item = plot.plot([1., 2.], [3., 4.], name="Sample")
    look = FigureStyle(style_id="project.roles", revision=1, name="Roles", look={
        "line_width_pt": .625, "title_size_pt": 9.5,
        "text_roles": {"title": {"bold": True}, "axis": {"italic": True},
                       "ticks": {"color": "#006600"}, "legend": {"bold": True},
                       "fit_label": {"size_pt": 8.25, "color": "#a00000"},
                       "caption": {"italic": True}}})
    state = PlotAppearance.from_mapping(bind_native_style({}, look))
    before = state.to_dict()
    monkeypatch.setattr(item, "setData", lambda *args, **kw: pytest.fail("style recomputed geometry"))
    apply_plot_appearance(plot, state, plot_widget=widget, data_items=(item,))
    assert plot.titleLabel.opts["bold"] is True
    assert plot.titleLabel.opts["size"] == "9.5pt"
    assert plot.getAxis("bottom").labelStyle["font-style"] == "italic"
    assert plot.getAxis("bottom").textPen().color().name() == "#006600"
    assert plot.legend.items[0][1].opts["bold"] is True
    assert item.opts["pen"].widthF() == pytest.approx(.625 * 96 / 72)
    assert state.effective_fit_label_state().size_pt == 8.25
    assert state.effective_fit_label_state().color == "#a00000"
    assert state.effective_text_style("annotation.provenance").italic is True
    assert state.text_style("title").is_default()
    assert state.to_dict() == before


def test_small_fractional_shared_geometry_is_not_dropped_by_legacy_bounds(qtbot):
    from nanoscope.core.figure_style import FigureStyle
    from nanoscope.gui.figure_style_binding import bind_native_style
    from nanoscope.gui.plot_appearance import PlotAppearanceWidget
    style = FigureStyle(style_id="project.fine", revision=1, name="Fine", look={
        "line_width_pt": .125, "marker_size_pt": .5, "title_size_pt": 5.5})
    look = PlotAppearance.from_mapping(bind_native_style({}, style))
    assert look.line_width == pytest.approx(1 / 6)
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    widget.apply_look(look)
    assert widget.state().title_size == 5.5
    config = _restyled_config(PlotConfig(), look, False)
    assert config.style.line_width == pytest.approx(1 / 6)
    assert config.style.title_font_size_pt == 5.5
    assert plot_config_from_json(plot_config_to_json(config)) == config



def test_shared_fractional_controls_display_and_edit_three_decimals_without_other_changes(qtbot):
    from nanoscope.core.figure_style import FigureStyle
    from nanoscope.gui.figure_style_binding import bind_native_style
    from nanoscope.gui.plot_appearance import PlotAppearanceWidget
    from PySide6 import QtCore
    style = FigureStyle(style_id="project.precise-controls", revision=1, name="Precise controls",
        look={"title_size_pt": 9.125, "axis_size_pt": 8.25, "line_width_pt": .46875})
    # The PG compatibility field is .625px; the physical canonical width stays .46875pt.
    look = PlotAppearance.from_mapping(bind_native_style({}, style))
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    widget.apply_look(look)
    before = widget.state()
    assert widget.title_size.value() == 9.125 and widget.title_size.text() == "9.125"
    assert widget.line_width.value() == .625 and widget.line_width.text() == "0.625"
    assert widget.axis_size.value() == 8.25 and widget.axis_size.text() == "8.250"
    widget.grid.setCurrentIndex(widget.grid.findData(True))
    widget._close_burst()
    assert widget.state().title_size == 9.125 and widget.state().line_width == .625
    assert widget.state().style_binding["ref"] == before.style_binding["ref"]
    assert widget.state().style_binding["overrides"] == {"grid": True}
    widget.undo()
    assert widget.state() == before
    field = widget.title_size.lineEdit()
    field.selectAll()
    qtbot.keyClicks(field, "8.375")
    qtbot.keyPress(field, QtCore.Qt.Key_Return)
    widget._close_burst()
    assert widget.state().title_size == 8.375
    assert widget.state().style_binding["overrides"]["title_size_pt"] == 8.375
    widget.undo()
    assert widget.state() == before


def test_text_and_project_style_fields_show_exact_three_decimal_sizes(qtbot):
    from nanoscope.gui.figure_text import TextStyleFields, TextStyle
    from nanoscope.gui.project_style_editor import ProjectStyleEditor
    from nanoscope.core.figure_style import FigureStyle
    fields = TextStyleFields(live=True)
    qtbot.addWidget(fields)
    fields.set_style(TextStyle(size_pt=9.125))
    assert fields.size.value() == 9.125
    dialog = ProjectStyleEditor(FigureStyle(style_id="project.precise-ui", revision=1,
        name="Precise", look={"line_width_pt": .625, "title_size_pt": 9.125}))
    qtbot.addWidget(dialog)
    assert dialog.sizes["line_width_pt"].value() == .625
    assert dialog.sizes["title_size_pt"].value() == 9.125
