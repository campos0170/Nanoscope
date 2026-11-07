"""Legacy recipe units cross the one native writer boundary exactly once."""
from copy import deepcopy
from dataclasses import replace

import pytest
from PySide6 import QtWidgets

from nanoscope.gui.plots_panel import (PlotConfig, plot_config_from_json,
                                      plot_config_to_json, migrate_plot_quantity)
from . import p1_hosting as H


@pytest.mark.parametrize('unit,lo,hi', [('s', '0.001', '0.005'), ('µs', '1000', '5000')])
def test_legacy_template_keeps_words_and_raw_geometry_while_converting_fit_and_pins_once(qtbot, monkeypatch, unit, lo, hi):
    payload = plot_config_to_json(PlotConfig())
    payload['version'] = 3
    payload['histogram']['column'] = 'dwell_s'
    payload['axes'].update(time_unit=unit, x_min=lo, x_max=hi, manual_x=[.001, .005])
    payload['axes'].pop('quantity_display', None)
    payload['fit'].update(range_unit=unit, range_lo=lo, range_hi=hi)
    payload['style'].update(title='Literal dwell (s)', x_label='τ from my notes (µs)')
    original = deepcopy(payload)
    recipe = plot_config_from_json(payload).as_template()
    assert recipe.axes.time_unit == unit and recipe.fit.range_lo == lo
    page = H.page_builders()['distributions'](); qtbot.addWidget(page)
    QtWidgets.QApplication.processEvents(); qtbot.wait(350)
    installed = page._installable(page.config().with_template(recipe))
    raw_before = {source.key: source.df.copy(deep=True) for source in page._sources.values()}
    with monkeypatch.context() as guard:
        guard.setattr(page, '_redraw', lambda: pytest.fail('Recipe initialization ran an analysis'))
        page._set_config(installed, draw=False)
        config = page.config()
        assert config.axes.time_unit == config.fit.range_unit == 'ms'
        assert (float(config.axes.x_min), float(config.axes.x_max)) == (1., 5.)
        assert (float(config.fit.range_lo), float(config.fit.range_hi)) == (1., 5.)
        assert config.axes.manual_x == (.001, .005)
        assert config.style.title == 'Literal dwell (s)'
        assert config.style.x_label == 'τ from my notes (µs)'
        assert migrate_plot_quantity(config) == config
        # The New path installs canonical appearance with a second draw=False call.
        page._set_config(replace(config, style=config.style), draw=False)
        assert page.config() == config
        QtWidgets.QApplication.processEvents(); qtbot.wait(350)
    assert payload == original
    for source in page._sources.values():
        assert source.df.equals(raw_before[source.key])


@pytest.mark.parametrize('unit,lo', [('s', '0.001'), ('µs', '1000')])
def test_new_recipe_initializer_clears_old_view_pins_without_losing_fit_unit_basis(unit, lo):
    payload = plot_config_to_json(PlotConfig())
    payload['version'] = 3
    payload['histogram']['column'] = 'dwell_s'
    payload['axes'].update(time_unit=unit, x_min=lo, manual_x=[.001, .005])
    payload['axes'].pop('quantity_display', None)
    payload['fit'].update(range_unit=unit, range_lo=lo)
    template = plot_config_from_json(payload).as_template()
    # Exactly MainWindow's empty-New policy; it preserves the scientific fit window.
    template = replace(template, axes=replace(template.axes, x_min='', x_max='',
        y_min='', y_max='', manual_x=None, manual_y=None))
    installed = migrate_plot_quantity(PlotConfig().with_template(template))
    assert installed.axes.x_min == '' and installed.axes.manual_x is None
    assert installed.fit.range_unit == 'ms' and float(installed.fit.range_lo) == 1.
    assert migrate_plot_quantity(installed) == installed
