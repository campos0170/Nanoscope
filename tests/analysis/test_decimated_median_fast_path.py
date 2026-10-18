"""``_decimated_median`` uses a rolling order statistic, not ndimage's O(n*w) rank filter.

The values must stay identical to the hand-computed median the ferritin contract digests
were derived against (``np.median`` over a symmetric-padded sliding window), and inputs the
old ndimage path accepted (non-finite samples, a window longer than the signal) must still
be accepted.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from nanoscope.analysis.ml_preprocess import _decimated_median

FS = 50_000.0


def _hand_median(x, fs, win_s, decim):
    n = len(x)
    xs = np.ascontiguousarray(x[::decim], dtype=np.float64)
    w = int(float(win_s) * float(fs) / decim)
    if w % 2 == 0:
        w += 1
    r = w // 2
    padded = np.pad(xs, (r, r), mode="symmetric")
    b = np.median(sliding_window_view(padded, w), axis=1)
    xp = np.arange(0, n, decim)[: len(b)]
    return np.interp(np.arange(n), xp, b).astype(np.float32)


@pytest.mark.parametrize("win_s,decim", [(0.01, 1), (0.03, 10), (0.1, 50), (0.5, 50)])
@pytest.mark.parametrize("kind", ["gauss", "quantised"])
def test_matches_the_hand_computed_median(win_s, decim, kind):
    rng = np.random.default_rng(7)
    x = rng.normal(-30.0, 5.0, 60_000)
    if kind == "quantised":
        x = np.round(x / 0.305176) * 0.305176
    np.testing.assert_array_equal(
        _decimated_median(x, FS, win_s, decim), _hand_median(x, FS, win_s, decim))


def test_non_finite_input_still_returns_instead_of_raising():
    rng = np.random.default_rng(8)
    x = rng.normal(-30.0, 5.0, 20_000)
    x[500] = np.nan
    out = _decimated_median(x, FS, 0.01, 1)
    assert out.shape == x.shape and out.dtype == np.float32


def test_window_longer_than_the_signal_still_returns():
    x = np.random.default_rng(9).normal(-30.0, 5.0, 100)
    out = _decimated_median(x, FS, 0.05, 1)
    assert out.shape == x.shape


def test_a_long_window_is_not_quadratic():
    """400k samples x 2501 taps took ~8 s through ndimage's rank filter and ~0.15 s now; the
    bound leaves ~20x headroom so it only fires if the rank filter comes back."""
    x = np.random.default_rng(10).normal(-30.0, 5.0, 400_000)
    t0 = time.perf_counter()
    _decimated_median(x, FS, 0.05, 1)
    assert time.perf_counter() - t0 < 3.0
