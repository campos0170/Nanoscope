"""Shared style defaults cannot erase categorical source identity colors."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import pyqtgraph as pg

from nanoscope.core.figure_style import builtin_styles, style_binding
from nanoscope.core.sources import PlotSource
from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
from nanoscope.gui.figure_style_binding import bind_native_style, project_native


def colors(page):
    result = {}
    for item, primitive in zip(page._data_items, page._snapshot['primitives']):
        color = (pg.mkBrush(item.opts['symbolBrush']).color() if primitive['symbols']
                 else pg.mkPen(item.opts['pen']).color())
        actual = color.name()
        key = primitive['source_key']
        assert key not in result or result[key] == actual
        result[key] = actual
        if primitive['fill']:
            assert pg.mkBrush(item.opts['fillBrush']).color().name() == actual
            assert pg.mkBrush(item.opts['fillBrush']).color().alpha() == 60
    return result


@pytest.mark.parametrize('kind', ['ecdf', 'box'])
def test_bound_style_keeps_three_sources_and_per_source_override_without_analysis(qtbot, monkeypatch, kind):
    from nanoscope.gui import distribution_summary_panel as module
    source_colors = {'a': '#0072b2', 'b': '#d55e00', 'c': '#009e73'}
    sources = [PlotSource(key, key.upper(), pd.DataFrame({
        'dwell_s': [.001, .002, .004, .006, .02], 'accepted': [True] * 5}), colour=color)
        for key, color in source_colors.items()]
    page = DistributionSummaryWindow()
    qtbot.addWidget(page)
    page.set_sources(sources, selected_keys=list(source_colors))
    page.set_config({'kind': kind, 'metric': 'dwell_s'})
    assert page.analyze()
    before = {name: frame.copy(deep=True) for name, frame in page._tables.items()}
    geometry = [tuple(np.asarray(values).copy() for values in item.getData()) for item in page._data_items]
    monkeypatch.setattr(module, 'summarize_distribution', lambda *a, **k: pytest.fail('Style recomputed summary'))
    page._plot.setRange(xRange=(.0001, .1), yRange=(.1, .9), padding=0)
    bounds = deepcopy(page._plot.viewRange())
    native = bind_native_style({}, builtin_styles()[0])
    assert native['data_color'] == '#0072b2'
    page.set_appearance_state(native)
    assert colors(page) == source_colors
    page.set_source_appearance_state({'b': {'color': '#cc22bb'}})
    expected = {**source_colors, 'b': '#cc22bb'}
    assert colors(page) == expected
    for name, frame in before.items():
        pd.testing.assert_frame_equal(frame, page._tables[name], check_exact=True)
    for item, (x, y) in zip(page._data_items, geometry):
        np.testing.assert_array_equal(item.getData()[0], x)
        np.testing.assert_array_equal(item.getData()[1], y)
    np.testing.assert_allclose(page._plot.viewRange(), bounds, atol=1e-12, rtol=0)
    payload, tables = page.snapshot_state()
    restored = DistributionSummaryWindow()
    qtbot.addWidget(restored)
    restored.restore_state(payload, tables)
    assert colors(restored) == expected


def test_explicit_figure_and_legacy_global_colors_keep_their_existing_meaning(qtbot):
    page = DistributionSummaryWindow()
    qtbot.addWidget(page)
    sources = [PlotSource(key, key, pd.DataFrame({'dwell_s': [.001, .002], 'accepted': [True, True]}),
                          colour=color) for key, color in [('a', '#0072b2'), ('b', '#009e73')]]
    page.set_sources(sources, selected_keys=['a', 'b'])
    assert page.analyze()
    # A saved pre-shared-style global override was deliberate and remains visible.
    page.set_appearance_state({'data_color': '#8833aa'})
    assert colors(page) == {'a': '#8833aa', 'b': '#8833aa'}
    # The new explicit figure override also wins; inherited style defaults do not.
    style = builtin_styles()[0]
    page.set_appearance_state(project_native(
        {'style_binding': style_binding(style, {'data_color': '#445566'})}))
    assert colors(page) == {'a': '#445566', 'b': '#445566'}
    page.set_source_appearance_state({'b': {'color': '#ff5500'}})
    assert colors(page) == {'a': '#445566', 'b': '#ff5500'}
