"""Display units are reversible while scientific geometry and raw tables stay fixed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nanoscope.core.quantity_display import display_to_view
from nanoscope.plot import figures, incubation_figures, incubation_galleries, signal_study_figures
from nanoscope.plot.quantity_axes import figure_quantity_metadata, time_axis


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_visible_minor_ticks_convert_without_revealing_suppressed_labels():
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots()
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, position: 'native' if value == 2000 else ''))
    time_axis(ax, 'bottom', 'dwell_s', stored_unit='µs')
    formatter = ax.xaxis.get_minor_formatter()
    formatter.set_locs([1000, 2000])
    assert formatter(2000) == '2'
    assert formatter(1000) == ''


def _events():
    return pd.DataFrame({'dwell_s': np.geomspace(.0001, .004, 40),
        'depth_pA': np.linspace(10., 30., 40) + np.sin(np.arange(40)),
        'frac_depth': np.linspace(.1, .3, 40), 'accepted': True})


class _KnownFit:
    params = {'tau_s': .001}
    ci = {'tau_s': (.0008, .0012)}
    meta = {'t_min': .0001, 't_max': .004}
    n = 40

    def predict(self, x):
        x = np.asarray(x)
        return np.where(x >= .0001, np.exp(-(x - .0001) / .001) / .001, 0.)


def test_dwell_fit_counts_raw_seconds_and_millisecond_annotations(monkeypatch):
    monkeypatch.setattr(figures, '_stats_module', lambda: pytest.fail('presentation refitted data'))
    events = _events(); before = events.copy(deep=True)
    fit = _KnownFit(); before_fit = deepcopy((fit.params, fit.ci, fit.meta))
    fig, ax = figures.dwell_histogram(events, fit=fit)
    data = fig._nanoscope_source_data
    np.testing.assert_allclose(ax.patches[0].get_data().edges[:-1], data.bin_left_s * 1e6)
    assert int(data['count'].sum()) == len(events)
    curve = next(line for line in ax.lines if line.get_label() == 'exponential fit')
    assert curve.get_xdata()[0] == pytest.approx(100.)  # retained microseconds
    assert ax.xaxis.get_major_formatter()(100.) == '0.1'
    assert display_to_view(2., ax._nanoscope_quantity_axes['bottom']) == pytest.approx(2000.)
    assert ax.get_xlabel() == 'Dwell time (ms)'
    assert ax.get_ylabel() == 'events per bin'
    text = '\n'.join(item.get_text() for item in ax.texts)
    assert ' = 1 [0.8, 1.2] ms' in text
    assert '0.1 ms' in text and '4 ms' in text and r'\mu' not in text
    assert any(np.allclose(line.get_xdata(), [100., 100.]) for line in ax.lines
               if len(line.get_xdata()) == 2)
    assert (fit.params, fit.ci, fit.meta) == before_fit
    pd.testing.assert_frame_equal(events, before)
    details = figures.figure_details(fig)
    assert details['plots'][0]['fit']['params']['tau_s'] == .001
    assert details['plots'][0]['quantity_display'] == figure_quantity_metadata(fig)


def test_scatter_retains_microsecond_positions_and_source_seconds():
    events = _events()
    fig, ax = figures.scatter_dwell_depth(events, density=False)
    scatter = next(item for item in ax.collections if len(item.get_offsets()) == len(events))
    np.testing.assert_array_equal(scatter.get_offsets()[:, 0], events.dwell_s * 1e6)
    np.testing.assert_array_equal(fig._nanoscope_source_data.dwell_s, events.dwell_s)
    assert ax.xaxis.get_major_formatter()(1000.) == '1'
    assert 'ms' in ax.get_xlabel()


def test_trace_clock_shading_and_source_coordinates_are_not_rescaled():
    x = 12.5 + np.arange(100) / 1000
    channel = SimpleNamespace(data=np.arange(100.), units='pA', info=SimpleNamespace(name='trace'))
    axis = SimpleNamespace(values=x, unit='s', is_time=True)
    fig, ax = figures.trace_preview(channel, axis,
        events=pd.DataFrame({'t_start_s': [.02], 'dwell_s': [.003]}), baseline=np.zeros(100))
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), x)
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), x)
    patch = ax.patches[0]
    assert patch.get_x() == pytest.approx(12.52)
    assert patch.get_width() == pytest.approx(.003)
    assert ax.xaxis.get_major_formatter()(12.52) == '12520'
    assert '12500.0' in ax.get_xlabel() and 'ms' in ax.get_xlabel()
    data = fig._nanoscope_source_data
    np.testing.assert_array_equal(data.x, x)
    np.testing.assert_array_equal(data.x_axis_origin_s, np.full(100, 12.5))
    assert data.t0_s.isna().all()


def test_sample_axis_does_not_gain_an_invented_clock():
    channel = SimpleNamespace(data=np.arange(20.), units='pA', info=SimpleNamespace(name='trace'))
    fig, ax = figures.trace_preview(channel, SimpleNamespace(values=np.arange(20.), unit='sample', is_time=False))
    assert ax.get_xlabel() == 'sample'
    assert figure_quantity_metadata(fig)['axes'] == []
    assert 't0_s' not in fig._nanoscope_source_data


@pytest.mark.parametrize('scale,coordinate_factor', [(1000., 1.), (1., 1000.)])
def test_joint_density_raw_columns_are_canonical_and_display_columns_explicit(scale, coordinate_factor):
    events = _events(); before = events.copy(deep=True)
    fig, ax = figures.joint_density({'Apo α': events}, x_scale=scale, gridsize=20,
        xlabel='Typed dwell wording', ylabel='Typed current wording')
    data = fig._nanoscope_source_data; columns = data.attrs['display_columns']
    np.testing.assert_array_equal(data.dwell_s, events.dwell_s)
    np.testing.assert_array_equal(data.depth_pA, events.depth_pA)
    np.testing.assert_array_equal(data[columns['display_x']], events.dwell_s * 1000)
    assert set(data[columns['display_x_unit']]) == {'ms'}
    assert set(data[columns['display_y_unit']]) == {'pA'}
    basis = ax._nanoscope_quantity_axes['bottom']
    assert basis.quantity.factor == coordinate_factor
    assert display_to_view(2., basis) == pytest.approx(.002 * scale)
    assert ax.get_xlabel() == 'Typed dwell wording' and ax.get_ylabel() == 'Typed current wording'
    assert data.attrs['quantity_display'] == figure_quantity_metadata(fig)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ['Apo α  (n = 40)']
    pd.testing.assert_frame_equal(events, before)


def _capture(monkeypatch, module):
    captured = {}
    def save(fig, folder, name, source=None, **kwargs):
        captured[name] = (fig, None if source is None else source.copy(deep=True))
        return Path(folder) / (name + '.png')
    monkeypatch.setattr(module, '_save', save)
    return captured


def test_incubation_log_dwell_and_per_decade_density_are_not_converted_twice(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, incubation_figures)
    events = pd.DataFrame({'condition': ['Apo'] * 30, 'dwell_ms': np.geomspace(.1, 10., 30),
                          'fracblock': np.linspace(.1, .8, 30) + .03 * np.sin(np.arange(30))})
    incubation_figures._distribution_set(events, tmp_path, 'distribution', 'fracblock', 'Fractional blockade')
    fig, data = captured['distribution']; scatter, _, density = fig.axes[:3]
    np.testing.assert_array_equal(scatter.collections[0].get_offsets()[:, 0], np.log10(events.dwell_ms))
    assert scatter.xaxis.get_major_formatter()(-1.) == '-1'
    assert density.xaxis.get_major_formatter()(-1.) == '-1'
    assert density.yaxis.get_major_formatter()(2.) == '2'
    assert density._nanoscope_quantity_axes['left'].quantity.factor == 1.
    curves = pd.read_csv(tmp_path / 'distribution.kde.csv')
    dwell = curves[curves.metric == 'dwell_ms']
    np.testing.assert_allclose(density.lines[0].get_xdata(), dwell.x, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(density.lines[0].get_ydata(), dwell.density, rtol=1e-14, atol=1e-14)
    pd.testing.assert_frame_equal(data, events)
    fig.clear()


def test_incubation_hours_are_drawn_and_labelled_in_hours(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, incubation_figures)
    monkeypatch.setattr(incubation_figures, '_distribution_set', lambda *a, **k: tmp_path / 'distribution.png')
    events = pd.DataFrame({'condition': ['Apo'] * 4, 'tag': ['one'] * 4, 'dwell_ms': [1., 2., 3., 4.],
        'fracblock': [.1, .2, .3, .4], 'f_sust_global': [.1, .2, .3, .4]})
    summary = pd.DataFrame({'tag': ['one'], 'pore_nm': [10.], 'incubation_h': [2.],
        'rate_per_s': [4.], 'lo': [3.], 'hi': [5.], 'n_events': [4], 'gate_frac_of_I0': [.1]})
    tables = {'gate_transfer_sweep': pd.DataFrame({'definition': ['native'], 'tag': ['one'],
                                                'floor': [.1], 'rate': [4.]})}
    incubation_figures.comparison_figures(events, summary, tables, tmp_path)
    fig, data = captured['incubation_rates_and_medians']
    assert fig.axes[0].lines[0].get_xdata()[0] == 2.
    assert fig.axes[0].xaxis.get_major_formatter()(2.) == '2'
    assert fig.axes[0].get_xlabel() == 'Incubation (h)'
    assert display_to_view(2., fig.axes[0]._nanoscope_quantity_axes['bottom']) == 2.
    assert fig.axes[2].get_ylabel() == 'Dwell (ms)'
    assert fig.axes[0].get_ylabel() == 'Accepted event rate (/s)'
    pd.testing.assert_frame_equal(data, summary)
    for figure, _ in captured.values(): figure.clear()


def test_trace_gallery_exports_original_seconds_and_additive_display_values(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, incubation_galleries)
    raw = np.zeros(500); raw[200] = -10
    analysis = SimpleNamespace(tag='apo', raw=raw + 100, detrended=raw, fs_hz=1000.)
    incubation_galleries._trace_context(analysis, SimpleNamespace(baseline_pA=0., noise_pA=1.), tmp_path)
    fig, _ = captured['trace_context_apo']
    points = pd.read_csv(tmp_path / 'trace_context_apo_points.csv')
    overview = points[points.panel == 1]
    np.testing.assert_allclose(overview.time_s, overview.sample_idx / 1000, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(overview.display_time, overview.time_s)
    np.testing.assert_allclose(overview.display_time_ms, overview.time_s * 1000)
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), overview.time_s)
    assert fig.axes[0].xaxis.get_major_formatter()(.2) == '200'
    assert fig.axes[2].xaxis.get_major_formatter()(-20.) == '-20'
    methods = json.loads((tmp_path / 'trace_context_apo_methods.json').read_text())
    assert methods['quantity_display'] == figure_quantity_metadata(fig)
    fig.clear()


def test_signal_study_original_and_relative_clocks_keep_separate_geometry(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, signal_study_figures)
    catalog = pd.DataFrame({'kind': ['spike'], 'accepted': [True], 'start_idx': [200], 'end_idx': [209],
        'original_onset_s': [1.2], 'onset_s': [.2], 'dwell_ms': [10.], 'depth_sust_pA': [5.],
        'f_sust': [.1], 'I0_local_pA': [100.]})
    before = catalog.copy(deep=True)
    raw = np.zeros(500); raw[200:210] = -5.
    analysis = SimpleNamespace(tag='apo', raw=raw + 100, detrended=raw, fs_hz=1000., duration_s=.5,
        catalog=catalog, provenance={'metadata': {'retained_ranges': [[0, 500, 1000, 1500]]}})
    signal_study_figures.recording_figures(analysis, None, tmp_path)
    trace, data = captured['signal_apo_trace']
    assert trace.axes[0].lines[0].get_xdata()[0] == 1.
    assert trace.axes[0].xaxis.get_major_formatter()(1.2) == '1200'
    np.testing.assert_array_equal(data.original_time_s, trace.axes[1].lines[0].get_xdata())
    examples, _ = captured['signal_apo_examples']
    assert examples.axes[0].lines[0].get_xdata()[0] == -100.
    assert examples.axes[0].xaxis.get_major_formatter()(-100.) == '-100'
    assert '1200.00000 ms' in examples.axes[0].get_title()
    dependence, _ = captured['signal_apo_time_dependence']
    assert dependence.axes[3].get_ylabel() == 'Accepted events / s'
    assert dependence.axes[3].xaxis.get_major_formatter()(.2) == '200'
    pd.testing.assert_frame_equal(catalog, before)
    for figure, _ in captured.values(): figure.clear()
