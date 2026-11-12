"""The Study figure editor shows and takes the incubation axis in hours, and old pins still land."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.quantity_display import AxisDisplayBasis, quantity_for
from nanoscope.gui.study_figure_window import StudyFigureWindow
from nanoscope.plot import incubation_figures
from nanoscope.plot.matplotlib_targets import panel_identity, panel_view
from nanoscope.plot.study_figure_style import capture_figure


def _incubation_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(incubation_figures, '_distribution_set', lambda *a, **k: tmp_path / 'distribution.png')
    rng = np.random.default_rng(3)
    events, summary = [], []
    for k, h in enumerate((2, 10, 30)):
        tag = f'rec_{h}h'
        events.append(pd.DataFrame({'tag': tag, 'condition': f'{h} h', 'dwell_ms': rng.lognormal(0, .6, 12),
                                    'fracblock': rng.normal(.3, .05, 12), 'aggregation_fracblock_global': rng.normal(.3, .05, 12),
                                    'f_sust_global': rng.normal(.3, .05, 12), 'accepted': True}))
        summary.append(dict(tag=tag, pore_nm=20., pore_id='p', incubation_h=float(h), rate_per_s=1. + k,
                            lo=.8 + k, hi=1.3 + k, n_events=12, gate_frac_of_I0=.08))
    tables = {'gate_transfer_sweep': pd.DataFrame([dict(definition='native', tag=s['tag'], floor=.1, rate=s['rate_per_s'])
                                                   for s in summary])}
    with capture_figure('incubation_rates_and_medians') as captured:
        incubation_figures.comparison_figures(pd.concat(events, ignore_index=True), pd.DataFrame(summary), tables, tmp_path)
    return captured


def _open_editor(qtbot, captured, **kwargs):
    page = StudyFigureWindow(captured.figure, captured.data, {}, name='incubation', provenance={}, **kwargs)
    qtbot.addWidget(page)
    page.show()
    qtbot.wait(50)
    return page, next(c for c in page._text_controllers if c.panel != 'figure')


def test_editor_names_the_incubation_axis_in_hours_and_takes_a_range_in_hours(qtbot, tmp_path, monkeypatch):
    page, controller = _open_editor(qtbot, _incubation_figure(tmp_path, monkeypatch))
    axis = controller.host.axis_state('bottom')
    assert axis.label_auto == 'Incubation (h)'
    assert axis.shown_range == pytest.approx((.6, 31.4))
    controller.host.apply_axis('bottom', replace(axis, range=(0., 40.), tick_major=10.))
    assert page._figure.axes[0].get_xlim() == (0., 40.)
    assert controller.host.axis_state('bottom').shown_range == (0., 40.)
    pins = page._appearance_widget.state().to_dict()
    key = panel_view(controller.panel) + '.bottom'
    assert pins['axes'][key] == {'min': 0., 'max': 40., 'tick_major': 10.}
    assert pins['quantity_display']['axes'][key]['quantity']['display_unit'] == 'h'


def test_a_style_saved_with_pins_in_ms_opens_on_the_same_hours(qtbot, tmp_path, monkeypatch):
    # Saved before incubation was presented in hours: 0-30 h and a tick every 10 h were pinned in ms,
    # together with the millisecond basis they were written in.
    captured = _incubation_figure(tmp_path, monkeypatch)
    key = panel_view(panel_identity(captured.figure.axes[0], 0)) + '.bottom'
    old = AxisDisplayBasis(quantity_for('incubation_h', declared_unit='h', quantity='duration', display_unit='ms'))
    saved = {'appearance': {'axes': {key: {'min': 0., 'max': 108000000., 'tick_major': 36000000.}},
                            'quantity_display': {'version': 1, 'axes': {key: old.to_dict()}}}}
    page, controller = _open_editor(qtbot, captured, style=saved)
    axis = controller.host.axis_state('bottom')
    assert (axis.range, axis.tick_major, axis.shown_range) == ((0., 30.), 10., (0., 30.))
    assert page._figure.axes[0].get_xlim() == (0., 30.)
