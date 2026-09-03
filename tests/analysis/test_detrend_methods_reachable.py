"""Every `baseline.METHODS` entry, exercised through BOTH kernel doors.

Recon for this round said the GUI's detrend combo offers only three of them and that
``DetectionPanel.set_detrend_method`` silently ignores a string it does not know -- so
a protocol that recalls ``'masked_median'`` runs the BLOCK MEDIAN with no warning and
no visible difference in the panel. That is a panel defect and the panel is another
agent's file. These tests establish the other half of the claim: that the KERNEL side
is sound for all seven, so the fix really is a combo entry and a setter, not new
estimator work.

They also pin the two facts the panel most needs and cannot see from its own side:
which methods read ``window_s`` and which ignore it, and which refuse ``two_pass``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanoscope.analysis.baseline import (
    MASKED_METHODS,
    METHODS,
    estimate_baseline,
    remove_baseline_wander,
)

FS = 50_000.0
DEPTH = 40.0

#: Per-method shape kwargs, so each estimator runs at ITS OWN parameters rather than
#: at whatever the block methods happen to default to.
SHAPE_KWARGS: dict[str, dict] = {
    "masked_median": dict(rough_s=0.5, base_s=0.1, decim=50, mask_sigma=3.5,
                          dilate_ms=1.0),
    "robust_median": dict(decim=50),
    "savgol": dict(sg_poly=3, sg_decim=50),
}


def _trace(n=int(8.0 * FS), *, i0=-200.0, noise=3.0, seed=4):
    """A drifting conductive recording with 0.5 ms, 40 pA events every 100 ms."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = i0 + 10.0 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0.0, noise, n)
    starts = range(int(0.1 * FS), n - 30, int(0.1 * FS))
    for i in starts:
        x[i: i + 25] -= DEPTH
    return x.astype(np.float32), list(starts)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_answers_through_estimate_baseline(method):
    x, _ = _trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = estimate_baseline(x, FS, method=method, window_s=0.5, polarity=1,
                                 **SHAPE_KWARGS.get(method, {}))
    assert base.shape == x.shape
    assert base.dtype == np.float64
    assert np.all(np.isfinite(base))
    # It is a BASELINE: it sits at the open-pore level, not at zero and not inside
    # the events (which are 40 pA below it).
    assert -230.0 < float(np.median(base)) < -180.0


@pytest.mark.parametrize("method", METHODS)
def test_every_method_answers_through_remove_baseline_wander(method):
    """The DETREND door -- the one the panel's combo actually drives."""
    x, _ = _trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = remove_baseline_wander(x, FS, method=method, window_s=0.5, polarity=1,
                                     **SHAPE_KWARGS.get(method, {}))
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", [*MASKED_METHODS, "savgol"])
def test_the_three_unoffered_estimators_keep_the_events_they_are_meant_to_keep(method):
    """Soundness, not merely non-crashing. A baseline that dived into the events would
    still return a finite full-length array; this measures that a 40 pA event is still
    ~40 pA deep against the estimate."""
    x, starts = _trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = estimate_baseline(x, FS, method=method, window_s=0.5, polarity=1,
                                 **SHAPE_KWARGS.get(method, {}))
    dev = np.asarray(x, dtype=np.float64) - base
    depths = [float(-dev[i + 5: i + 20].mean()) for i in starts[2:-2]]
    assert np.median(depths) == pytest.approx(DEPTH, rel=0.15), np.median(depths)


# ---------------------------------------------------------------------------
# what the panel cannot see from its own side
# ---------------------------------------------------------------------------


def test_masked_median_IGNORES_window_s_so_the_detrend_spinbox_does_nothing_for_it():
    """``'masked_median'`` takes ``rough_s``/``base_s``, never ``window_s`` -- the
    field rides on `BaselineConfig` only so the methods share one object. A panel that
    offers this method behind the existing "window" spinbox is offering a control that
    changes NOTHING, which is worse than not offering it: the user turns the knob, the
    numbers do not move, and nothing says why."""
    x, _ = _trace()
    kw = dict(method="masked_median", polarity=1, **SHAPE_KWARGS["masked_median"])
    a = estimate_baseline(x, FS, window_s=0.05, **kw)
    b = estimate_baseline(x, FS, window_s=5.0, **kw)
    assert np.array_equal(a, b)


def test_robust_median_and_savgol_DO_read_window_s():
    """The other two do, so the same spinbox is meaningful for them. Stated as a pair
    with the test above because the panel needs to know they differ."""
    x, _ = _trace()
    for method, extra in (("robust_median", SHAPE_KWARGS["robust_median"]),
                          ("savgol", SHAPE_KWARGS["savgol"])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = estimate_baseline(x, FS, method=method, window_s=0.1, polarity=1, **extra)
            b = estimate_baseline(x, FS, method=method, window_s=1.0, polarity=1, **extra)
        assert not np.array_equal(a, b), method


@pytest.mark.parametrize("method", MASKED_METHODS)
def test_the_masked_methods_refuse_the_legacy_two_pass_flag(method):
    """They are already two-pass, with two decoupled windows; the legacy flag is a
    DIFFERENT algorithm. A panel that exposes both must gate one on the other."""
    x, _ = _trace()
    with pytest.raises(ValueError, match=r"does not take two_pass=True"):
        estimate_baseline(x, FS, method=method, two_pass=True, polarity=1,
                          **SHAPE_KWARGS[method])


@pytest.mark.parametrize("method", MASKED_METHODS)
def test_the_masked_methods_refuse_a_trace_shorter_than_their_tracking_window(method):
    """`_require_length`, and it is a REFUSAL rather than a silent constant. The panel
    has to be ready to surface this: it is the one error these two raise that the
    block methods do not."""
    x, _ = _trace(n=int(0.02 * FS))
    with pytest.raises(ValueError, match=r"recording is \d+ samples"):
        estimate_baseline(x, FS, method=method, window_s=0.5, polarity=1,
                          **SHAPE_KWARGS[method])
