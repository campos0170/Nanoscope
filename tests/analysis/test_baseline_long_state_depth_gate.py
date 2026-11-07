"""Separate coherent long-state evidence from the stricter headline population."""
import numpy as np
import pytest

from nanoscope.analysis import baseline as B


def scene(*, weak_depth=4., weak_dwell=.6, strong=False):
    fs = 1000.
    x = 100. + np.random.default_rng(613129).normal(0., .2, 5000)
    base = np.full(x.size, 100.)
    mask = np.zeros(x.size, bool)
    a, b = 1000, 1000 + round(weak_dwell * fs)
    x[a:b] -= weak_depth
    base[a:b] -= .875 * weak_depth
    mask[a:b] = True
    if strong:
        x[3000:3600] -= 20.
        mask[3000:3600] = True
    return x, base, mask, fs


def measure(x, base, mask, fs, **kwargs):
    return B.tracked_intrusion(x, base, mask, fs, min_depth_pA=5.,
                              worst_min_dwell_s=.3, **kwargs)


def test_moderate_long_state_is_not_lost_when_headline_population_is_empty():
    args = scene()
    assert measure(*args) is None  # unchanged historical direct-call contract
    got = measure(*args, worst_min_depth_pA=3.5)
    assert got is not None
    assert got['fraction'] is None and got['intrusion_pA'] is None
    assert got['depth_pA'] is None and got['n_states'] == 0
    assert got['worst_fraction'] == pytest.approx(.875, abs=.01)
    assert got['worst_dwell_s'] == .6 and got['worst_t_s'] == 1.


def test_moderate_long_state_cannot_hide_behind_clean_high_snr_headline_events():
    args = scene(strong=True)
    old = measure(*args)
    got = measure(*args, worst_min_depth_pA=3.5)
    assert old['fraction'] == got['fraction'] == 0.
    assert old['n_states'] == got['n_states'] == 1
    assert old['worst_fraction'] == 0.
    assert got['worst_fraction'] > B.LONG_STATE_INTRUSION_MAX
    assert got['worst_t_s'] == 1.


@pytest.mark.parametrize('depth,dwell', [(0., .6), (2., .6), (4., .1)])
def test_noise_weak_states_and_short_moderate_states_do_not_become_long_state_evidence(depth, dwell):
    assert measure(*scene(weak_depth=depth, weak_dwell=dwell),
                   worst_min_depth_pA=3.5) is None


def test_worst_state_lower_gate_does_not_claim_correct_baseline_rides():
    x, base, mask, fs = scene(strong=True)
    base[:] = 100.
    got = measure(x, base, mask, fs, worst_min_depth_pA=3.5)
    assert got['fraction'] == got['worst_fraction'] == 0.


def test_long_state_threshold_constants_do_not_weaken_headline_or_riding_limits():
    assert B.INTRUSION_MIN_DEPTH_SIGMA == 5.
    assert B.INTRUSION_LONG_MIN_DEPTH_SIGMA == 3.5
    assert B.BASELINE_INTRUSION_MAX == .03
    assert B.LONG_STATE_INTRUSION_MAX == .5


def test_background_attenuation_reports_worst_only_evidence_without_fabricated_median(monkeypatch):
    fs = 1000.
    x = 100. + np.random.default_rng(761039).normal(0., 1., 5000)
    base = np.full(x.size, 100.)
    mask = np.zeros(x.size, bool); mask[1000:1600] = True
    x[mask] -= 4.; base[mask] -= 3.5
    # Supply the known independent state support, leaving actual depth, noise,
    # duration and intrusion arithmetic untouched.
    monkeypatch.setattr(B, '_rough_pass', lambda *a, **k: (x - 100., mask, 1., 1.))
    health = B._detrend_health(x, base, fs, B.BaselineConfig(method='constant'), -1)
    row = health.check('detrend_attenuation')
    assert row.level == 'warn' and row.value is None
    assert 'headline attenuation median is not measurable' in row.message
    assert 'worst eligible state' in row.message
