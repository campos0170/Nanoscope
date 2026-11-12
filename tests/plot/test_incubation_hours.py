"""Incubation time is drawn in hours, the unit it is stored and typed in; every other time stays in ms.

The study figures used to convert the stored hours to milliseconds: 2, 10 and 30 h were drawn at
7.2e+06, 3.6e+07 and 1.08e+08 under "Incubation (ms)", with tick labels that ran into each other.
"""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from nanoscope.core.figure_quantity_basis import migrate_appearance_basis, read_quantity_bases
from nanoscope.core.quantity_display import AxisDisplayBasis, display_to_view, quantity_for
from nanoscope.plot import incubation_figures, signal_study_figures
from nanoscope.plot.matplotlib_targets import apply_text_overrides, panel_identity, panel_view


def _capture(monkeypatch, module):
    captured = {}

    def save(fig, folder, name, source=None, **kwargs):
        captured[name] = fig
        return Path(folder) / (name + '.png')
    monkeypatch.setattr(module, '_save', save)
    return captured


def _tick_labels(axis):
    """(text, pixel box) of the major tick labels inside the view, left to right, after a real draw."""
    axis.figure.canvas.draw()
    low, high = sorted(axis.get_view_interval())
    ticks = [t for t in axis.get_major_ticks() if low <= t.get_loc() <= high and t.label1.get_text()]
    return sorted(((t.label1.get_text(), t.label1.get_window_extent()) for t in ticks), key=lambda item: item[1].x0)


def _collisions(axis):
    labels = _tick_labels(axis)
    return [(a, b) for (a, box_a), (b, box_b) in zip(labels, labels[1:]) if box_a.x1 > box_b.x0]


def _comparison_inputs(hours, n=12):
    rng = np.random.default_rng(3)
    events, summary = [], []
    for k, h in enumerate(hours):
        tag = f'rec_{h:g}h'
        events.append(pd.DataFrame({'tag': tag, 'condition': f'{h:g} h', 'dwell_ms': rng.lognormal(0, .6, n),
                                    'fracblock': rng.normal(.3, .05, n),
                                    'aggregation_fracblock_global': rng.normal(.3, .05, n),
                                    'f_sust_global': rng.normal(.3, .05, n), 'accepted': True}))
        summary.append(dict(tag=tag, pore_nm=20., pore_id='p', incubation_h=float(h), rate_per_s=1. + k,
                            lo=.8 + k, hi=1.3 + k, n_events=n, gate_frac_of_I0=.08))
    sweep = pd.DataFrame([dict(definition='native', tag=row['tag'], floor=.1, rate=row['rate_per_s']) for row in summary])
    return pd.concat(events, ignore_index=True), pd.DataFrame(summary), {'gate_transfer_sweep': sweep}


def _rates_figure(tmp_path, monkeypatch, hours):
    captured = _capture(monkeypatch, incubation_figures)
    monkeypatch.setattr(incubation_figures, '_distribution_set', lambda *a, **k: tmp_path / 'distribution.png')
    events, summary, tables = _comparison_inputs(hours)
    incubation_figures.comparison_figures(events, summary, tables, tmp_path)
    return captured['incubation_rates_and_medians'], summary


@pytest.mark.parametrize('hours', [(2, 10, 30), (2, 10, 18, 30), (.5, 1, 2, 4, 8, 16, 24, 48, 72), (2,), (.25, .5), (1000, 2000)])
def test_incubation_is_drawn_at_its_hours_with_an_hours_label_and_ticks_that_do_not_collide(tmp_path, monkeypatch, hours):
    fig, _ = _rates_figure(tmp_path, monkeypatch, hours)
    for ax in fig.axes:
        drawn = {float(x) for line in ax.lines for x in line.get_xdata()}
        assert {float(h) for h in hours} <= drawn
        assert max(drawn) == max(hours)
        assert ax.get_xlabel() == 'Incubation (h)'
        basis = ax._nanoscope_quantity_axes['bottom']
        assert (basis.quantity.stored_unit, basis.quantity.display_unit, basis.quantity.factor) == ('h', 'h', 1.)
        for h in hours:
            assert ax.xaxis.get_major_formatter()(float(h)) == f'{h:g}'
            assert display_to_view(float(h), basis) == float(h)
        texts = [text for text, _ in _tick_labels(ax.xaxis)]
        assert texts and not any('e+' in text for text in texts), texts
        assert _collisions(ax.xaxis) == []
    # Only incubation changed unit: the dwell axis of the third panel is still milliseconds.
    assert fig.axes[2].get_ylabel() == 'Dwell (ms)'
    fig.clear()


def test_incubation_points_keep_their_stored_hours(tmp_path, monkeypatch):
    fig, summary = _rates_figure(tmp_path, monkeypatch, (2, 10, 30))
    before = summary.copy(deep=True)
    points = [line for line in fig.axes[0].lines if line.get_marker() == 'o']
    assert [float(line.get_xdata()[0]) for line in points] == [2., 10., 30.]
    pd.testing.assert_frame_equal(summary, before)
    fig.clear()


def test_incubation_clusters_are_named_in_hours(tmp_path, monkeypatch):
    captured = _capture(monkeypatch, incubation_figures)
    rng = np.random.default_rng(1)
    n = 60
    table = pd.DataFrame({'PC1': rng.normal(size=n), 'PC2': rng.normal(size=n), 'kmeans3': rng.integers(0, 3, n),
                          'hdbscan': rng.integers(-1, 2, n), 'incubation_h': np.repeat([2., 10., 30.], n // 3),
                          'log_dwell': rng.normal(size=n), 'tag': np.repeat(['a', 'b', 'c'], n // 3)})
    incubation_figures.clustering_figures(table, tmp_path, prefix='pooled')
    ax = captured['pooled_pca'].axes[2]
    assert ax.get_title() == 'Incubation (h)'
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['2 h (n=20)', '10 h (n=20)', '30 h (n=20)']
    captured['pooled_pca'].clear()


def _signal_result(hours):
    tags = [f'rec_{h:g}h' for h in hours]
    rng = np.random.default_rng(2)
    count = len(hours)
    summary = pd.DataFrame(dict(tag=tags, incubation_h=[float(h) for h in hours], pore_id=['p'] * count,
                                rate_per_s=np.arange(count) + 1., duty=.01 * np.arange(1, count + 1),
                                dwell_med=np.arange(count) + 2., depth_sust_med=np.arange(count) + 8.,
                                f_sust_med=.08 + .01 * np.arange(count), G_nS=1.))
    events = pd.concat([pd.DataFrame(dict(tag=t, dwell_ms=rng.lognormal(1, .5, 20), depth_sust_pA=rng.normal(9, 1, 20),
                                          f_sust=rng.normal(.09, .01, 20))) for t in tags], ignore_index=True)
    pairs = pd.DataFrame(dict(tag_a=[tags[0]], tag_b=[tags[1]], nominal_nm=[10.], rate_ratio_b_over_a=[2.],
                              conductance_ratio_b_over_a=[1.], duty_ratio_b_over_a=[2.]))
    types = pd.DataFrame([dict(tag=t, kind=k, n_accepted=5, rate_per_s=1.) for t in tags for k in ('spike', 'plateau')])
    return SimpleNamespace(summary=summary, events=events, tables={'nominal_pairs': pairs, 'signal_types': types})


def _signal_figures(tmp_path, monkeypatch, hours):
    captured = _capture(monkeypatch, signal_study_figures)
    signal_study_figures.comparison_figures(_signal_result(hours), [], tmp_path)
    return captured


def test_signal_study_names_incubation_in_hours_under_each_point_and_in_the_legend(tmp_path, monkeypatch):
    captured = _signal_figures(tmp_path, monkeypatch, (2, 10, 30))
    for name, panels in (('signal_comparison_summary', 6), ('signal_comparison_event_types', 2)):
        fig = captured[name]
        assert len(fig.axes) == panels
        fig.canvas.draw()
        for ax in fig.axes:
            assert [t.get_text() for t in ax.get_xticklabels()] == ['2 h', '10 h', '30 h']
    fig = captured['signal_comparison_distributions']
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ['2 h; n=20', '10 h; n=20', '30 h; n=20']
    # Dwell is a duration like any other: its labels are still milliseconds.
    assert fig.axes[0].get_xlabel() == 'Dwell (ms)'
    for fig in captured.values():
        fig.clear()


MANY = (.5, 1, 2, 4, 8, 12, 16, 24, 36, 48, 60, 72, 84, 96, 120, 144)


@pytest.mark.parametrize('hours', [(2, 10, 30), MANY[:8], MANY[:10], MANY[:12], MANY])
def test_signal_study_labels_under_the_points_do_not_collide(tmp_path, monkeypatch, hours):
    captured = _signal_figures(tmp_path, monkeypatch, hours)
    turned = {}
    for name in ('signal_comparison_summary', 'signal_comparison_event_types'):
        rotations = []
        for ax in captured[name].axes:
            assert [t.get_text() for t in ax.get_xticklabels()] == [f'{h:g} h' for h in hours]
            assert _collisions(ax.xaxis) == [], (name, hours)
            rotations.append({t.get_rotation() for t in ax.get_xticklabels()})
        # A figure turns its labels together or not at all, and only when level ones would touch.
        assert all(each == rotations[0] for each in rotations)
        turned[name] = rotations[0]
    assert turned['signal_comparison_summary'] == ({0.} if len(hours) <= 10 else {90.})
    for fig in captured.values():
        fig.clear()


def test_a_figure_style_saved_with_the_old_millisecond_pins_still_lands_on_the_same_hours(tmp_path, monkeypatch):
    # Before this change the incubation axis was presented in ms, so a range and a tick spacing
    # pinned in the figure editor were saved in ms together with that basis. They must keep
    # meaning 0-30 h and every 10 h, both when the figure is rebuilt and when the editor migrates them.
    fig, _ = _rates_figure(tmp_path, monkeypatch, (2, 10, 30))
    ax = fig.axes[0]
    view = panel_view(panel_identity(ax, 0)) + '.bottom'
    old = AxisDisplayBasis(quantity_for('incubation_h', declared_unit='h', quantity='duration', display_unit='ms'))
    assert old.quantity.factor == 3600000.
    saved = {'axes': {view: {'min': 0., 'max': 108000000., 'tick_major': 36000000.}},
             'quantity_display': {'version': 1, 'axes': {view: old.to_dict()}}}
    apply_text_overrides(fig, saved)
    assert ax.get_xlim() == (0., 30.)
    assert [text for text, _ in _tick_labels(ax.xaxis)] == ['0', '10', '20', '30']
    migrated = migrate_appearance_basis(saved, {view: ax._nanoscope_quantity_axes['bottom']})
    assert [migrated['axes'][view][k] for k in ('min', 'max', 'tick_major')] == pytest.approx([0., 30., 10.])
    assert read_quantity_bases(migrated['quantity_display'])[view].quantity.display_unit == 'h'
    fig.clear()
