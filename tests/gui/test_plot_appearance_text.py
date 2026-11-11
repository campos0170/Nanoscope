"""The per-element text model of ``PlotAppearance``: presets, templates, Undo tooltips,
composition and the Text-first layout. Synthetic data only."""
from __future__ import annotations

from dataclasses import fields
import json

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from nanoscope.core.figure_style import builtin_styles, style_binding
from nanoscope.gui import plot_appearance
from nanoscope.gui.figure_composition import capture_cartesian_model
from nanoscope.gui.figure_text import AxisPin, TextStyle
from nanoscope.gui.fit_label import FitLabelItem, FitLabelState, FitReadout
from nanoscope.gui.plot_appearance import (PlotAppearance, PlotAppearanceWidget,
                                           _describe_step, apply_plot_appearance)
from nanoscope.gui.plot_templates import decode_window_template, encode_window_template


def _widget(qtbot, **kwargs):
    widget = PlotAppearanceWidget(**kwargs)
    qtbot.addWidget(widget)
    return widget


def test_a_preset_carries_the_look_and_never_the_wording(qtbot):
    """Plan G2 test 7. Before: the preset stored the title and stamped it on the target."""
    source = _widget(qtbot)
    source.apply_appearance(PlotAppearance(title="I-V of pore 3", x_label="V", title_size=16)
                            .with_text_style("title", TextStyle(text="I-V of pore 3", bold=True))
                            .with_text_style("legend.entry.Apo", TextStyle(text="Apo 1")))
    try:
        assert source.save_user_preset("Look only", overwrite=True)
        saved = PlotAppearanceWidget.user_preset_states()["Look only"]
        assert saved == {"title_size": 16.0, "text": {"title": {"bold": True}}}
        stored = json.loads(QtCore.QSettings().value(PlotAppearanceWidget._PRESETS_KEY))
        assert "title" not in stored["Look only"] and "x_label" not in stored["Look only"]
        target = _widget(qtbot)
        target.apply_appearance(PlotAppearance(title="Noise of pore 3", y_label="PSD"))
        assert target.apply_user_preset("Look only")
        state = target.state()
        assert (state.title, state.x_label, state.y_label) == ("Noise of pore 3", None, "PSD")
        assert state.title_size == 16 and state.text_style("title").bold is True
        # a shipped preset keeps the wording too, and its Undo step still names it
        target.preset.activated.emit(1)
        assert target.state().title == "Noise of pore 3"
        assert target.undo_button.toolTip().startswith("Undo: Publication · light preset")
    finally:
        source.delete_user_preset("Look only")


def test_an_old_preset_that_saved_a_title_no_longer_stamps_it(qtbot):
    settings = QtCore.QSettings()
    key = PlotAppearanceWidget._PRESETS_KEY
    settings.setValue(key, json.dumps({"Old look": {"title": "Stamped", "grid": True}}))
    try:
        target = _widget(qtbot)
        target.apply_appearance(PlotAppearance(title="Mine"))
        assert target.apply_user_preset("Old look")
        assert target.state().title == "Mine" and target.state().grid is True
    finally:
        settings.remove(key)


def test_a_template_keeps_wording_and_styles_and_drops_legend_entries():
    """Plan G2 test 8 (owner Q-T6)."""
    state = (PlotAppearance(title="Fig 2", x_label="Voltage (mV)")
             .with_text_style("title", TextStyle(text="Fig 2", italic=True))
             .with_text_style("legend.entry.Apo", TextStyle(text="Apo 1", bold=True))
             .with_axis_pin("bottom", AxisPin(min=0, max=200)))
    payload = encode_window_template("noise-spectrum", {"fft": 4096}, state.to_dict())
    assert "legend.entry.Apo" not in payload["appearance"]["text"]
    assert payload["appearance"]["title"] == "Fig 2"
    _settings, appearance = decode_window_template("noise-spectrum", payload)
    back = PlotAppearance.from_mapping(appearance)
    assert back.title == "Fig 2" and back.x_label == "Voltage (mV)"
    assert back.text_style("title").italic is True
    assert not any(key.startswith("legend.entry.") for key, _ in (back.text or ()))
    assert back.axis_pin("bottom") == AxisPin(min=0, max=200)


def test_a_views_own_pins_travel_like_a_single_views_and_reach_only_that_view():
    """figures2-1: a multi-view window keeps a pin under the view it was typed on
    (``<view>.<side>``). A project and a template keep it as they keep a one-view window's
    pin; a preset keeps none; each view draws its own, and a bare pin (saved before pins were
    kept per view) only on the primary view."""
    state = (PlotAppearance(title="Noise")
             .with_axis_pin("psd.bottom", AxisPin(min=10, max=1000))
             .with_axis_pin("allpoints.left", AxisPin(tick_major=2)))
    assert state.to_dict()["axes"] == {"allpoints.left": {"tick_major": 2.0},
                                       "psd.bottom": {"min": 10.0, "max": 1000.0}}
    assert PlotAppearance.from_mapping(state.to_dict()) == state
    payload = encode_window_template("noise-spectrum", {"fft": 4096}, state.to_dict())
    _settings, appearance = decode_window_template("noise-spectrum", payload)
    assert PlotAppearance.from_mapping(appearance).axes == state.axes
    assert state.style_only().axes is None
    assert state.for_view("psd").axes == (("bottom", AxisPin(min=10, max=1000)),)
    assert state.for_view("allpoints").axes == (("left", AxisPin(tick_major=2)),)
    assert state.for_view("spectrogram").axes is None
    assert state.for_view("") == state
    bare = PlotAppearance.from_mapping({"axes": {"bottom": {"min": 10, "max": 1000},
                                                 "Bad.bottom": {"min": 1, "max": 2},
                                                 "psd.middle": {"min": 1, "max": 2}}})
    assert bare.axes == (("bottom", AxisPin(min=10, max=1000)),)
    assert bare.for_view("psd", primary="psd").axis_pin("bottom").range() == (10, 1000)
    assert bare.for_view("allpoints", primary="psd").axes is None
    assert _describe_step(PlotAppearance(), PlotAppearance().with_axis_pin(
        "psd.bottom", AxisPin(min=10, max=1000))) == "x axis range 10–1000"
    assert _describe_step(bare, bare.with_axis_pin("bottom", AxisPin()).with_axis_pin(
        "psd.bottom", AxisPin(min=20, max=500))) == "x axis range 20–500"


def test_every_field_has_step_words_and_new_fields_are_named_readably():
    """Plan G2 test 9: _describe_step reads _STEP_WORDS for every field that moves."""
    dedicated = {"style_binding", "quantity_display", "series_styles", "source_appearance"}
    assert {f.name for f in fields(PlotAppearance)} <= set(plot_appearance._STEP_WORDS) | dedicated
    # Structured fields have dedicated human-readable history descriptions.
    for key, expected in (("quantity_display", "display units"),
                          ("series_styles", "series formatting"),
                          ("source_appearance", "source names and colors")):
        assert _describe_step(PlotAppearance(), PlotAppearance(**{key: {}})) == expected
    from nanoscope.core.figure_style import FigureStyle, style_binding
    styled_binding = PlotAppearance(style_binding=style_binding(FigureStyle("test", 1, "Test look")))
    assert _describe_step(PlotAppearance(), styled_binding) == "figure style Test look · revision 1"
    assert _describe_step(styled_binding, PlotAppearance()) == "figure style reset"
    empty = PlotAppearance()
    assert _describe_step(empty, empty.with_text_style("title", TextStyle(bold=True))) == "title bold on"
    assert _describe_step(empty, empty.with_text_style(
        "axis.left.label", TextStyle(color="#d55e00"))) == "y label colour #d55e00"
    assert _describe_step(empty, empty.with_axis_pin(
        "bottom", AxisPin(min=0.5, max=4))) == "x axis range 0.5–4"
    assert _describe_step(empty, empty.with_axis_pin(
        "left", AxisPin(tick_major=0.5))) == "y axis major tick 0.5"
    assert _describe_step(empty, PlotAppearance(
        fit_label=FitLabelState(visible=False))) == "fit label hidden"
    styled = empty.with_text_style("title", TextStyle(text="Fig 2", bold=True))
    assert _describe_step(empty, styled) == "title ‘Fig 2’, bold on"
    assert _describe_step(empty, PlotAppearance(background="dark")) == "background automatic → dark"


def test_saved_text_styles_round_trip_and_bad_entries_are_dropped():
    state = PlotAppearance.from_mapping({
        "text": {"title": {"bold": True, "text": "ignored: the title box owns it"},
                 "axis.left.label": {"color": "not a colour", "size_pt": 12},
                 "nonsense key": {"bold": True}, "colorbar.label": "not a mapping"},
        "axes": {"bottom": {"min": 4, "max": 1}, "left": {"tick_major": 0.5}, "middle": {}},
        "fit_label": {"visible": False, "placement": "nowhere"}})
    assert state.text == (("axis.left.label", TextStyle(size_pt=12.0)),
                          ("title", TextStyle(bold=True)))
    assert state.axes == (("left", AxisPin(tick_major=0.5)),)
    assert state.fit_label == FitLabelState(visible=False)
    assert PlotAppearance.from_mapping(state.to_dict()) == state
    hash(state)


def test_composition_refuses_a_plot_whose_view_box_carries_a_label(qtbot):
    """Plan G2 test 14 (D10). Before: the model replayed without the fit's numbers."""
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(500, 400)
    widget.show()
    qtbot.waitExposed(widget)
    plot = widget.getPlotItem()
    plot.plot([0, 1, 2], [1, 3, 2])
    assert capture_cartesian_model(plot) is not None
    label = FitLabelItem(plot.getViewBox())
    label.apply_state(FitLabelState(), [FitReadout({"equation": "y = x"})])
    QtWidgets.QApplication.processEvents()
    assert capture_cartesian_model(plot) is None
    label.apply_state(FitLabelState(visible=False), [])
    assert capture_cartesian_model(plot) is not None, "a hidden label draws nothing"


def test_composition_refuses_styles_its_model_cannot_replay(qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(500, 400)
    widget.show()
    qtbot.waitExposed(widget)
    plot = widget.getPlotItem()
    plot.plot([0, 1, 2], [1, 3, 2])
    apply_plot_appearance(plot, PlotAppearance().with_text_style("title", TextStyle(bold=True)))
    assert capture_cartesian_model(plot) is not None
    apply_plot_appearance(plot, PlotAppearance().with_text_style("colorbar.label",
                                                                 TextStyle(bold=True)))
    assert capture_cartesian_model(plot) is None
    apply_plot_appearance(plot, PlotAppearance().with_axis_pin("left", AxisPin(tick_major=1)))
    assert capture_cartesian_model(plot) is None

    # The same rule for a figure bound to a figure style, as every figure of a project is. The
    # style's family and sizes fill in every role, and composition asks only what the AUTHOR set
    # per element: the style alone, or a legend row the author made bold, is replayed from what is
    # drawn; a row given its own font family, or a style on the colour bar's label, is not.
    legend_widget = pg.PlotWidget()
    qtbot.addWidget(legend_widget)
    legend_widget.resize(500, 400)
    legend_widget.show()
    qtbot.waitExposed(legend_widget)
    framed = legend_widget.getPlotItem()
    framed.addLegend()
    framed.plot([0, 1, 2], [1, 3, 2], name="apo")
    bound = PlotAppearance(style_binding=style_binding(builtin_styles()[0]))
    row = "legend.entry.apo"
    apply_plot_appearance(framed, bound)
    assert capture_cartesian_model(framed) is not None, "a figure style alone is nothing to replay"
    apply_plot_appearance(framed, bound.with_text_style(row, TextStyle(bold=True)))
    assert capture_cartesian_model(framed) is not None
    apply_plot_appearance(framed, bound.with_text_style(row, TextStyle(family="Times New Roman")))
    assert capture_cartesian_model(framed) is None
    apply_plot_appearance(framed, bound.with_text_style("colorbar.label", TextStyle(bold=True)))
    assert capture_cartesian_model(framed) is None


def test_the_text_rows_come_first(qtbot):
    """Plan G2 test 15 (ux H2): Title above the preset and font rows, compact or not."""
    for compact in (False, True):
        widget = _widget(qtbot, compact=compact)
        layout = widget.layout()
        row = lambda w: layout.getItemPosition(layout.indexOf(w))[0]   # noqa: E731
        assert row(widget.title) < row(widget.x_label) < row(widget.y_label)
        assert row(widget.y_label) < row(widget.preset) < row(widget.font_family)


def test_automatic_is_the_one_word_and_placeholders_name_the_automatic_text(qtbot):
    widget = _widget(qtbot)
    for combo in (widget.background, widget.grid, widget.legend_visible):
        assert combo.itemText(0) == "Automatic"
    widget.set_automatic_text("x_label", "Dwell (ms)")
    assert widget.x_label.placeholderText() == "Automatic: Dwell (ms)"


def test_the_fit_label_section_appears_when_the_host_has_a_fit_label(qtbot):
    widget = _widget(qtbot)
    widget.show()
    assert not widget.fit_label_group.isVisible()
    widget.set_fit_label_available(True)
    assert widget.fit_label_group.isVisible()
    widget.fit_label_fields["r2"].setChecked(False)
    assert widget.state().fit_label_state().fields == ("equation", "n")
    widget.fit_label_show.setChecked(False)
    assert widget.state().fit_label_state().visible is False
