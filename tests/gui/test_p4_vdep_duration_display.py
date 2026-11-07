"""Duration provenance presentation cannot alter the core's grouping or stored basis."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from nanoscope.analysis.voltage_dep import VDepSeries
from nanoscope.core.sources import DEFAULT_DETECTION_MODULE_ID
from nanoscope.gui.voltage_dep_panel import VDepWindow, _duration_group_label_ms


def series(groups, metric='n_events'):
    return VDepSeries(metric=metric, points=(), depends_on_baseline=False,
        depends_on_gain=False, baseline_groups=(), merge_groups=(), gain_groups=(),
        warnings=(), duration_groups=groups)


@pytest.mark.parametrize('metric', ['n_events', 'capture_rate_hz'])
def test_duration_provenance_converts_seconds_groups_without_regrouping(metric):
    groups = (('0.400 s', ('Apo A', 'Apo B')), ('40.000 s', ('Apo C',)),
              ('unknown', ('Apo D',)))
    value = series(groups, metric)
    before = deepcopy(value)
    owner = SimpleNamespace(_series_module_id=DEFAULT_DETECTION_MODULE_ID)
    lines = VDepWindow._provenance_lines(owner, value)
    shown = '\n'.join(lines)
    assert '400 ms — Apo A, Apo B' in shown
    assert '40000 ms — Apo C' in shown
    assert 'unknown — Apo D' in shown
    assert ('[not read by this metric]' in next(line for line in lines if 'Duration:' in line)) == (metric != 'n_events')
    assert value == before and value.duration_groups == groups
    assert VDepWindow._provenance_lines(owner, value) == lines  # repeated display never reconverts


def test_shared_duration_caption_is_ms_and_unknown_labels_remain_literal():
    groups = (('0.400 s', ('Apo A', 'Apo B')),)
    value = series(groups)
    assert VDepWindow._figure_method_segments(value) == ['over 400 ms each']
    assert value.duration_groups == groups
    for text in ('unknown', '0.400 sample', 'author wording', '400 ms'):
        assert _duration_group_label_ms(text) == text
