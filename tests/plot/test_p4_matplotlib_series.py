"""Per-series presentation reuses frozen or retained scientific geometry."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from nanoscope.core.figure_style import FigureStyle, style_binding
from nanoscope.plot import comparison_figures as comparison
from nanoscope.plot.study_figure_style import StudyFigureStyle
from .test_comparison_figures import prepare, identical_tables


@pytest.mark.parametrize('kind', ['facets', 'pairs', 'events'])
def test_comparison_series_identity_sizes_visibility_and_frozen_tables(kind, monkeypatch):
    snapshot, tables = prepare(kind, show_box=True)
    before = deepcopy(snapshot)
    originals = {key: frame.copy(deep=True) for key, frame in tables.items()}
    descriptors = comparison.comparison_series(snapshot)
    assert [entry.identity_key for entry in descriptors] == ['a', 'b']
    assert all(entry.identity_kind == 'source' for entry in descriptors)
    monkeypatch.setattr(comparison, 'prepare_comparison', lambda *a, **k: pytest.fail('recomputed'))
    binding = style_binding(FigureStyle(style_id='test.palette', revision=1, name='Palette',
                                        look={'palette': ['#ff0000', '#00ff00']}))
    native = comparison.render_comparison(snapshot, tables, {'style_binding': binding})
    # Applying a reusable palette alone cannot relabel/recolour source identities.
    expected = {tuple(to_rgba(source['color'])[:3]) for source in snapshot['sources']}
    actual = {tuple(collection.get_facecolors()[0, :3]) for ax in native.axes
              for collection in ax.collections if len(collection.get_facecolors())}
    assert actual == expected
    appearance = {'style_binding': binding, 'series_styles': {
        descriptors[0].series_id: {'color': '#8833aa', 'line_width_pt': 2.75, 'marker_size_pt': 6.5},
        descriptors[1].series_id: {'visible': False}}}
    figure = comparison.render_comparison(snapshot, tables, appearance,
        source_appearance={'a': {'color': '#112233', 'label': 'Renamed only'}})
    FigureCanvasAgg(figure).draw()
    collections = [item for ax in figure.axes for item in ax.collections]
    assert collections
    for item in collections:
        np.testing.assert_allclose(item.get_facecolors()[0, :3], to_rgba('#8833aa')[:3])
        np.testing.assert_allclose(item.get_sizes(), [6.5 ** 2])
    for ax in figure.axes:
        for item in [*ax.patches, *ax.lines]:
            assert item.get_linewidth() == 2.75
    for ax, panel in zip(figure.axes, snapshot['panels']):
        np.testing.assert_allclose(ax.get_xlim(), panel['xlim'])
        np.testing.assert_allclose(ax.get_ylim(), panel['ylim'])
    if figure.legends:
        assert figure.legends[0]._nanoscope_source_keys == ['a']
        assert figure.legends[0].legend_handles[0].get_markersize() == 6.5
    assert comparison.comparison_series(snapshot, {'a': {'label': 'New label'}})[0].series_id == descriptors[0].series_id
    assert snapshot == before
    identical_tables(tables, originals)
    native.clear(); figure.clear()


def retained():
    figure = Figure(figsize=(5, 3))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    line, = axis.plot([1., 2., 3.], [2., 1., 4.], 'o-', color='#147d92', alpha=.4, label='Trace')
    points = axis.scatter([1., 2.], [1., 3.], color='#ca7627', label='Events')
    axis.legend()
    return figure, axis, line, points


def test_retained_series_resets_visibility_sizes_and_linked_legend_without_touching_data():
    figure, axis, line, points = retained()
    styler = StudyFigureStyle(figure)
    descriptors = {entry.identity_key: entry for entry in styler.series_descriptors()}
    before = (line.get_xydata().copy(), points.get_offsets().copy(), points.get_sizes().copy(),
              line.get_linewidth(), line.get_markersize(), axis.get_xlim(), axis.get_ylim())
    binding = style_binding(FigureStyle(style_id='test.palette', revision=1, name='Palette',
                                        look={'palette': ['#ff0000', '#00ff00']}))
    styler.apply(style_binding=binding)
    assert to_rgba(line.get_color()) == to_rgba('#147d92')
    entries = {descriptors['#147d92'].series_id: {'color': '#8833aa', 'line_width_pt': 2.75,
                                               'marker_size_pt': 6.5, 'visible': False},
               descriptors['#ca7627'].series_id: {'color': '#225577', 'marker_size_pt': 9.5}}
    styler.apply(appearance={'series_styles': entries}, colors={'#147d92': '#112233'})
    assert to_rgba(line.get_color()) == to_rgba('#8833aa')
    assert line.get_alpha() == .4 and not line.get_visible()
    assert line.get_linewidth() == 2.75 and line.get_markersize() == 6.5
    assert not axis.get_legend().legend_handles[0].get_visible()
    assert not axis.get_legend().get_texts()[0].get_visible()
    np.testing.assert_allclose(points.get_sizes(), [9.5 ** 2])
    np.testing.assert_array_equal(line.get_xydata(), before[0])
    np.testing.assert_array_equal(points.get_offsets(), before[1])
    assert (axis.get_xlim(), axis.get_ylim()) == before[-2:]
    styler.apply()
    assert line.get_visible() and axis.get_legend().get_texts()[0].get_visible()
    assert line.get_linewidth() == before[3] and line.get_markersize() == before[4]
    np.testing.assert_array_equal(points.get_sizes(), before[2])
    assert to_rgba(line.get_color()) == to_rgba('#147d92')
    figure.clear()


def test_multicolour_artist_offers_only_safe_group_controls_and_keeps_other_colours():
    figure = Figure(); FigureCanvasAgg(figure)
    axis = figure.subplots()
    collection = axis.scatter([1., 2.], [3., 4.], c=['#123456', '#abcdef'])
    styler = StudyFigureStyle(figure)
    target = next(entry for entry in styler.series_descriptors() if entry.identity_key == '#123456')
    assert target.capabilities == ('color',)
    before = collection.get_offsets().copy()
    styler.apply(appearance={'series_styles': {target.series_id: {'color': '#ff0088'}}})
    np.testing.assert_allclose(collection.get_facecolors(), [to_rgba('#ff0088'), to_rgba('#abcdef')])
    np.testing.assert_array_equal(collection.get_offsets(), before)
    figure.clear()
