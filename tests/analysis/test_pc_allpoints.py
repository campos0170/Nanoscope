"""The all-points amplitude histogram kernel: the levels, read before any detector runs.

:mod:`nanoscope.analysis.allpoints` is a pure numpy kernel — no Qt, no Session, no
detection, no provenance and no schema. Every assertion here is on synthetic float32 at
50 kHz whose levels and occupancies were CHOSEN, so "it found the blocked level" is an
exact number rather than an impression.

The two that carry the most weight:

* a level holding one sample in a thousand is still found beside one holding 999 — that
  rarity is the normal case for translocations and is why the view exists at all;
* a single clean level reports ONE level. The first cut of the peak finder reported four,
  all within 0.7 pA of each other, because counting noise makes a smooth peak's top jitter.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.allpoints import (
    DEFAULT_BINS,
    all_points_histogram,
    level_peaks,
)


def _two_level(n_open: int, n_blocked: int, *, open_i: float = 100.0,
               blocked_i: float = 60.0, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """A pore at ``open_i`` that spends ``n_blocked`` samples at ``blocked_i``."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(open_i, sigma, n_open),
                           rng.normal(blocked_i, sigma, n_blocked)]).astype(np.float32)


def test_every_finite_sample_lands_in_exactly_one_bin():
    """The counts are the recording: no sample is dropped and none is counted twice."""
    x = _two_level(90_000, 10_000)
    hist = all_points_histogram(x)
    assert hist.counts.sum() == x.size == hist.n_finite == hist.n_binned
    assert hist.counts.size == DEFAULT_BINS
    assert hist.edges.size == DEFAULT_BINS + 1


def test_the_two_levels_of_a_blocking_pore_are_found_and_their_gap_is_the_depth():
    """Open and blocked, both recovered, with ΔI read off as the distance between them."""
    hist = all_points_histogram(_two_level(90_000, 10_000))
    levels = level_peaks(hist)

    assert len(levels) == 2
    assert levels[0]["center"] == pytest.approx(100.0, abs=2 * hist.bin_width)
    assert levels[1]["center"] == pytest.approx(60.0, abs=2 * hist.bin_width)
    depth = levels[0]["center"] - levels[1]["center"]
    assert depth == pytest.approx(40.0, abs=4 * hist.bin_width)


def test_a_level_holding_one_sample_in_a_thousand_is_still_found():
    """The rare-event case, which is the ONLY case that matters for translocations.

    1 000 blocked samples in a million: the blocked peak's tallest bin is ~1/900th the
    height of the open-pore peak's, which is exactly the regime a linear count axis and a
    fraction-of-the-mode threshold both lose.
    """
    hist = all_points_histogram(_two_level(999_000, 1_000, seed=3))
    levels = level_peaks(hist)

    assert len(levels) == 2
    assert levels[1]["center"] == pytest.approx(60.0, abs=3 * hist.bin_width)
    assert levels[1]["count"] * 900 < levels[0]["count"]     # genuinely tiny beside it


def test_a_single_clean_level_reports_one_level_and_not_four():
    """Counting noise on one peak's top is not four levels — the smoothing regression.

    Measured before the default smoothing width was raised: a single N(100, 1) trace
    reported 99.69, 99.95, 100.16 and 100.38 pA as four separate "levels".
    """
    rng = np.random.default_rng(1)
    hist = all_points_histogram(rng.normal(100.0, 1.0, 200_000).astype(np.float32))
    levels = level_peaks(hist)

    assert len(levels) == 1
    # 0.3 pA, not two bin widths: the MODE of a sampled Gaussian is not its mean, and the
    # level this reports is a bin, well inside the level's own 1 pA spread.
    assert levels[0]["center"] == pytest.approx(100.0, abs=0.3)


def test_a_third_level_is_reported_when_a_third_level_is_there():
    """The rejection above is about noise, not about a ceiling on how many levels exist."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(100.0, 1.0, 80_000),
                        rng.normal(80.0, 1.0, 10_000),
                        rng.normal(60.0, 1.0, 10_000)]).astype(np.float32)
    centers = sorted(lv["center"] for lv in level_peaks(all_points_histogram(x)))
    assert len(centers) == 3
    for got, want in zip(centers, (60.0, 80.0, 100.0)):
        assert got == pytest.approx(want, abs=0.5)


def test_nonfinite_samples_are_counted_and_never_piled_onto_an_edge_bin():
    """A NaN has no amplitude; binning it would invent a level the pore never sat at."""
    x = _two_level(50_000, 5_000)
    x[:100] = np.nan

    hist = all_points_histogram(x)

    assert hist.n_nonfinite == 100
    assert hist.n_total == x.size
    assert hist.counts.sum() == x.size - 100


def test_clipping_the_range_reports_the_samples_it_pushed_out_of_the_picture():
    """A histogram over a clipped span discards samples, so it must say how many.

    One saturating sample sets the whole range otherwise: here it stretches an 8 pA-wide
    signal across 900 pA, and every real sample lands in seven of 512 bins.
    """
    rng = np.random.default_rng(2)
    x = rng.normal(100.0, 1.0, 100_000).astype(np.float32)
    x[500] = 1000.0                                # one spike

    full = all_points_histogram(x)
    clipped = all_points_histogram(x, clip_quantile=1e-4)

    # the spike stretched the range from 8 pA to 900 pA: the entire pore signal is now in
    # seven of 512 bins, which is the picture the clip exists to prevent
    assert np.count_nonzero(full.counts) <= 8
    assert full.clipped is False and full.n_above == 0
    assert clipped.clipped is True
    assert clipped.n_above >= 1                    # the spike is OUTSIDE and it is counted
    assert clipped.n_binned == x.size - clipped.n_below - clipped.n_above
    assert clipped.hi < 200.0                      # and the picture is readable again


def test_an_explicit_span_wins_over_the_quantile_and_still_tallies_what_it_excluded():
    x = _two_level(50_000, 5_000)
    hist = all_points_histogram(x, span=(90.0, 110.0), clip_quantile=0.1)
    assert (hist.lo, hist.hi) == (90.0, 110.0)
    assert hist.n_below >= 5_000                   # the whole blocked level is outside
    assert hist.n_binned + hist.n_below + hist.n_above == hist.n_finite


def test_a_constant_channel_still_has_something_to_draw():
    """A dead or saturated channel is a real answer, not a crash."""
    hist = all_points_histogram(np.full(10_000, 42.0, dtype=np.float32))
    # float32 edges would ALL have come back as exactly 42.0 here (a zero-width bin);
    # the kernel bins in float64 for exactly this reason
    assert hist.bin_width > 0.0
    assert hist.counts.sum() == 10_000
    assert hist.mode_center == pytest.approx(42.0, abs=1e-3)


def test_an_all_nan_span_refuses_rather_than_drawing_an_empty_picture():
    with pytest.raises(ValueError, match="no finite samples"):
        all_points_histogram(np.full(1000, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="at least one sample"):
        all_points_histogram(np.empty(0, dtype=np.float32))


def test_the_histogram_of_a_span_never_sees_more_than_the_span():
    """A 2 s look at a long recording histograms 100 000 samples, not the recording."""
    rng = np.random.default_rng(9)
    big = rng.normal(100.0, 1.0, 3_000_000).astype(np.float32)
    span = big[100_000:200_000]
    hist = all_points_histogram(span)
    assert hist.n_total == 100_000
    assert span.base is big                        # a view, not a copy
