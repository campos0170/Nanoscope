"""Matplotlib styles change existing artists, never their scientific geometry."""
from copy import deepcopy

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nanoscope.plot.matplotlib_targets import (
    apply_text_overrides, bind_panel, capture_targets, panel_view,
    remember_automatic_text, storage_key,
)


def canvas():
    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    bind_panel(axis, "source-a/metric-x", label="Recording A")
    axis.plot([1., 2., 3.], [3., 2., 4.], label="Original source", linewidth=.75)
    axis.set_title("Automatic title")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Current (pA)")
    legend = figure.legend()
    legend._nanoscope_source_keys = ["recording-source-key"]
    remember_automatic_text(figure)
    return figure, axis


def test_target_identity_is_independent_of_words_and_capture_is_detached():
    figure, axis = canvas()
    figure.canvas.draw()
    first, axes = capture_targets(figure, figure.canvas.get_renderer())
    original = tuple((t.panel, t.key) for t in first)
    key = storage_key("source-a/metric-x", "title")
    style = {"text": {key: {"text": "β $price$", "size_pt": 7.5, "bold": True}}}
    apply_text_overrides(figure, style)
    figure.canvas.draw()
    after, _ = capture_targets(figure, figure.canvas.get_renderer())
    assert tuple((t.panel, t.key) for t in after) == original
    assert axis.title.get_text() == "β $price$"
    assert axis.title.get_parse_math() is False
    assert axis.title.get_fontsize() == 7.5
    assert next(t for t in after if t.key == "title" and t.panel != "figure").automatic == "Automatic title"
    assert axes[0].shown_range == tuple(axis.get_xlim())
    figure.clear()
    assert first[0].rect and all(type(v) is float for v in first[0].rect)
    assert "legend.entry.recording-source-key" in {t.key for t in first}


def test_style_and_range_leave_coordinates_unchanged_and_tick_text_is_not_editable():
    figure, axis = canvas()
    before = deepcopy(axis.lines[0].get_xydata())
    state = {"text": {storage_key("source-a/metric-x", "axis.bottom.ticks"):
                       {"text": "fabricated numbers", "size_pt": 8.25}},
             "axes": {panel_view("source-a/metric-x") + ".bottom": {"min": 1.25, "max": 2.5, "tick_major": .25}}}
    apply_text_overrides(figure, state)
    figure.canvas.draw()
    targets, _ = capture_targets(figure, figure.canvas.get_renderer())
    ticks = next(t for t in targets if t.key == "axis.bottom.ticks")
    assert ticks.capabilities == frozenset({"style"})
    assert all(t.get_text() != "fabricated numbers" for t in axis.get_xticklabels())
    np.testing.assert_array_equal(axis.lines[0].get_xydata(), before)
    assert axis.get_xlim() == (1.25, 2.5)


def test_event_identity_survives_axes_order_and_pair_labels_keep_their_capability():
    figure = Figure(figsize=(4, 3))
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    left._nanoscope_event_panel_identity = "a" * 64
    right._nanoscope_event_panel_identity = "b" * 64
    bind_panel(left, "ignored-ordinal", label="Event A", labels_locked="Feature units belong to the matrix")
    left.set_xlabel("Dwell (s)")
    figure.canvas.draw()
    first, _ = capture_targets(figure, figure.canvas.get_renderer())
    descriptor = next(t for t in first if t.panel == "event:" + "a" * 64 and t.key == "axis.bottom.label")
    assert "edit" not in descriptor.capabilities
    key = storage_key(descriptor.panel, descriptor.key)
    assert len(key.removeprefix("annotation.")) <= 60
    figure.delaxes(right)
    figure.add_axes(right)
    second, _ = capture_targets(figure, figure.canvas.get_renderer())
    assert key == storage_key(next(t for t in second if t.panel == descriptor.panel and t.key == descriptor.key).panel, descriptor.key)


def test_retained_study_binding_and_element_reset_preserve_data_and_extensions():
    from nanoscope.core.figure_style import FigureStyle
    from nanoscope.plot.study_figure_style import StudyFigureStyle
    figure, axis = canvas()
    styler = StudyFigureStyle(figure)
    original_title = axis.title.get_text()
    original_weight = axis.title.get_fontweight()
    original_range = tuple(axis.get_xlim())
    original_xy = axis.lines[0].get_xydata().copy()
    shared = FigureStyle(style_id="project.paper", revision=2, name="Paper", look={
        "title_size_pt": 7.25, "line_width_pt": .75,
        "text_roles": {"title": {"italic": True}}})
    binding = {"version": 1, "ref": shared.ref.to_dict(), "snapshot": shared.to_dict(), "overrides": {}}
    key = storage_key("source-a/metric-x", "title")
    appearance = {"text": {key: {"text": "Ω $literal$", "bold": True}},
                  "axes": {panel_view("source-a/metric-x") + ".bottom": {"min": 1.2, "max": 2.2}}}
    styler.apply(appearance=appearance, style_binding=binding, future_extension={"kept": 1})
    assert axis.title.get_text() == "Ω $literal$"
    assert axis.title.get_fontsize() == 7.25
    assert axis.title.get_fontstyle() == "italic"
    assert axis.lines[0].get_linewidth() == .75
    np.testing.assert_array_equal(axis.lines[0].get_xydata(), original_xy)
    styler.apply()
    assert axis.title.get_text() == original_title
    assert axis.title.get_fontweight() == original_weight
    assert tuple(axis.get_xlim()) == original_range
    np.testing.assert_array_equal(axis.lines[0].get_xydata(), original_xy)


def test_shared_physical_sizes_are_not_clipped_to_legacy_study_widget_limits():
    from nanoscope.core.figure_style import FigureStyle, style_binding
    from nanoscope.plot.study_figure_style import StudyFigureStyle
    figure, axis = canvas()
    style = FigureStyle(style_id="fine-print", revision=1, name="Fine print", look={
        "title_size_pt": 5.5, "line_width_pt": .125, "marker_size_pt": .5})
    StudyFigureStyle(figure).apply(style_binding=style_binding(style))
    assert axis.title.get_fontsize() == 5.5
    assert axis.lines[0].get_linewidth() == .125
    assert axis.lines[0].get_markersize() == .5


def test_axis_off_marginal_exposes_its_title_without_invisible_axis_targets():
    figure, axis = canvas()
    axis.set_axis_off()
    figure.canvas.draw()
    targets, axes = capture_targets(figure, figure.canvas.get_renderer())
    panel = [t for t in targets if t.panel == "source-a/metric-x"]
    assert any(t.key == "title" for t in panel)
    assert not any(t.key.startswith("axis.") for t in panel)
    assert axes == ()
