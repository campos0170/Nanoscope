"""Retained Matplotlib axes use physical display limits without rebuilding science."""
from copy import deepcopy
from dataclasses import replace

import nanoscope  # noqa: F401
import pytest

from .test_p3_matplotlib_editing import plotted, panel_controller, assert_tables
from .test_comparison_figures_panel import forbid_prepare
from nanoscope.gui.figure_text import AxisPin
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.plot.matplotlib_targets import panel_view


def test_comparison_format_range_is_ms_and_undo_preserves_raw_frozen_values(qtbot, monkeypatch):
    page = plotted(qtbot)
    raw = deepcopy(page._tables)
    forbid_prepare(monkeypatch)
    controller = panel_controller(page)
    axis = controller.host.axis_state('bottom')
    assert 'ms' in axis.label_auto
    descriptor = next(a for a in controller.axes() if a.side == 'bottom')
    assert descriptor.basis.quantity.factor == 1000.
    mark = page._appearance_widget.history_mark()
    controller.host.apply_axis('bottom', replace(axis, range=(1., 10.)))
    assert controller.host.axis_state('bottom').shown_range == pytest.approx((1., 10.))
    assert page._appearance_widget.history_mark() == mark + 1
    key = panel_view(controller.panel) + '.bottom'
    assert page._appearance.axis_pin(key).range() == (1., 10.)
    assert page._appearance.quantity_display['axes'][key]['quantity']['display_unit'] == 'ms'
    assert_tables(page._tables, raw)
    assert page._appearance_widget.undo()
    assert controller.host.axis_state('bottom').range is None
    assert_tables(page._tables, raw)


def test_old_comparison_pin_keeps_seconds_picture_until_edited_as_ms(qtbot, monkeypatch):
    page = plotted(qtbot)
    forbid_prepare(monkeypatch)
    controller = panel_controller(page)
    key = panel_view(controller.panel) + '.bottom'
    page.set_appearance_state(PlotAppearance(title='Literal time (s)', axes=((key, AxisPin(.001, .01)),)))
    current = controller.host.axis_state('bottom')
    assert current.range == pytest.approx((1., 10.))
    assert current.shown_range == pytest.approx((1., 10.))
    controller.host.apply_axis('bottom', replace(current, range=(2., 8.), tick_major=2.))
    assert controller.host.axis_state('bottom').shown_range == pytest.approx((2., 8.))
    assert page._appearance.title == 'Literal time (s)'
