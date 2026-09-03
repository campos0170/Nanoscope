"""B4DA-04 — ``filter_limited`` must not be judged on the dwell the filter INFLATED.

The rule was ``dwell_s < 2 * filter_rise_s`` where ``dwell_s`` is measured on the
FILTERED trace -- the quantity ``filters.lowpass``'s own docstring says the filter
biases long ("dwell times of short events are biased LONG"). So the filter carried its
victims over the line exactly when it was destroying them. Measured, 20 square 200 us /
100 pA events at 100 kHz with 10 pA of noise:

    cutoff  rise    2*rise  true dwell  MEASURED dwell  depth    flagged
     1 kHz  481 us  962 us    200 us       935 us       21.4 pA   18/20
     2 kHz  241 us  481 us    200 us       545 us       35.7 pA    0/20   <-- 64% low
     5 kHz   97 us  193 us    200 us       320 us       68.6 pA    0/20

and the unflagged rows are precisely the ones the fits KEEP, because
``dwell_fit_inputs`` and the Analysis panel drop filter-limited rows by default -- with
``n_excluded_filter_limited = 0`` in the fit's provenance, so the fit reported itself
clean. Which rows got flagged also depended on the NOISE level (0/20 at 2 pA, 18/20 at
10 pA, 19/20 at 20 pA) rather than on the filter, because a low-pass crushes the noise
and with it the Schmitt exit threshold.

The flag is now judged on the width at HALF THE EVENT'S OWN PEAK, which a symmetric
zero-phase filter leaves at the true duration while the event is resolved and floors at
its own kernel width once it is not. It is never longer than ``dwell_s``, so the rule
can only have become stricter: everything the old one caught is still caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.filters import lowpass, measure_rise_time
from nanoscope.analysis.modules import dwell_fit_inputs

FS = 100_000.0
I0, DEPTH = 1000.0, 100.0


def _square(dwell_s: float, *, sigma: float = 10.0, seed: int = 51, n_ev: int = 20):
    """``n_ev`` square blockades of EXACTLY ``dwell_s``, on a flat noisy open pore."""
    n = int(FS * 4.0)
    pad, step = 5_000, (n - 10_000) // n_ev
    dw = int(round(dwell_s * FS))
    x = np.full(n, I0)
    for i in range(n_ev):
        x[pad + i * step : pad + i * step + dw] -= DEPTH
    return x + np.random.default_rng(seed).normal(0.0, sigma, n)


def _run(sig, cutoff, *, kind="bessel", order=8):
    return detect_events(sig, FS, min_dwell_s=1e-5, filter_cutoff_hz=cutoff,
                         filter_order=order, filter_kind=kind)


# ---------------------------------------------------------------------------
# the regime the flag exists for
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cutoff", [1_000.0, 2_000.0])
def test_events_under_the_resolution_floor_are_flagged_however_the_filter_stretched_them(cutoff):
    """True dwell below 2*rise => every row flagged, at cutoffs where the old rule
    (2 kHz) reported 0 of 20 while the depth was 64% low."""
    sig = _square(200e-6)
    rise = measure_rise_time(FS, cutoff, order=8, kind="bessel")
    assert 200e-6 < 2 * rise, "fixture must sit under the floor for this to mean anything"
    df, prov = _run(sig, cutoff)
    assert len(df) >= 20
    assert int(df["filter_limited"].sum()) == len(df), (
        "unflagged rows below the resolution floor",
        df.loc[~df["filter_limited"], ["dwell_s", "depth_pA"]])
    # And the depth really is wrong, which is why the flag has to fire.
    assert float(df["depth_pA"].median()) < 0.75 * DEPTH


@pytest.mark.parametrize("cutoff", [5_000.0, 10_000.0, 20_000.0])
def test_resolved_events_are_not_flagged(cutoff):
    """The other half of the contract: a rule that flags everything says nothing.

    At 5 kHz the noiseless plateau of these events reaches 99.8 of 100 pA -- they ARE
    resolved -- so a flag there would be a false statement about them.
    """
    sig = _square(200e-6)
    df, _prov = _run(sig, cutoff)
    flagged = int(df["filter_limited"].sum())
    assert flagged <= 0.2 * len(df), (cutoff, flagged, len(df))


def test_long_events_are_never_filter_limited_at_any_cutoff():
    """5 ms events through a 1 kHz filter: 10x the floor, resolved, and left alone."""
    sig = _square(5e-3)
    for cutoff in (1_000.0, 2_000.0, 5_000.0, 10_000.0):
        df, _ = _run(sig, cutoff)
        long_rows = df[df["dwell_s"] > 4e-3]
        assert int(long_rows["filter_limited"].sum()) == 0, (cutoff, long_rows)


def test_the_verdict_no_longer_depends_on_the_noise_level():
    """Same filter, same true events, four noise levels.

    The old rule answered 0/20, 2/20, 18/20, 19/20 across these, because the noise --
    not the filter -- decided where the Schmitt trigger let go of the event.
    """
    verdicts = []
    for sigma in (2.0, 5.0, 10.0, 20.0):
        df, _ = _run(_square(200e-6, sigma=sigma, seed=7), 1_000.0)
        verdicts.append(int(df["filter_limited"].sum()) / max(len(df), 1))
    assert all(v == 1.0 for v in verdicts), verdicts


def test_a_different_filter_kind_and_rate_reaches_the_same_verdict():
    """An independent path: 250 kHz, gaussian kernel, negative open pore, 400 us events
    at a 1.5 kHz cutoff -- the case that shipped 12 of 13 rows unflagged at -50.7%."""
    fs, i0, depth, dwell = 250_000.0, -1500.0, 300.0, 400e-6
    n = int(fs * 2.0)
    pad, step, dw = 5_000, (n - 10_000) // 13, int(round(dwell * fs))
    x = np.full(n, i0)
    for i in range(13):
        x[pad + i * step : pad + i * step + dw] += depth   # resistive on a negative pore
    x = x + np.random.default_rng(777).normal(0.0, 15.0, n)

    rise = measure_rise_time(fs, 1_500.0, kind="gaussian")
    assert dwell < 2 * rise
    df, prov = detect_events(x, fs, min_dwell_s=1e-5, filter_cutoff_hz=1_500.0,
                             filter_kind="gaussian")
    assert int(df["filter_limited"].sum()) == len(df), df[["dwell_s", "depth_pA"]]

    # ...and the fit now SEES the exclusion instead of reporting itself clean.
    inp = dwell_fit_inputs((df, prov))
    assert inp.n_excluded_filter_limited == len(df)


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------
def test_the_flag_is_a_superset_of_the_old_dwell_rule():
    """The half-max width can never exceed the dwell, so no row the old contract
    flagged may come back unflagged. Checked across the whole sweep at once."""
    for sig in (_square(200e-6), _square(1e-3), _square(5e-3)):
        for cutoff in (1_000.0, 5_000.0, 20_000.0):
            df, prov = _run(sig, cutoff)
            old = df["dwell_s"].to_numpy() < 2 * prov["filter_rise_s"]
            assert (df["filter_limited"].to_numpy() >= old).all(), (cutoff, df)


def test_an_unfiltered_hand_span_keeps_the_dwell_answer():
    """No filter smearing, no difference: on a square event the half-max width IS the
    dwell, so the hand-spans path (``characterize`` called directly) is unchanged."""
    n = 20_000
    cur = np.full(n, I0)
    cur[1_000:1_010] -= DEPTH        # 10 samples = 100 us
    cur[5_000:9_000] -= DEPTH        # 4000 samples = 40 ms
    cur = cur + np.random.default_rng(3).normal(0.0, 5.0, n)
    base = np.full(n, I0)
    starts, ends = np.array([1_000, 5_000]), np.array([1_010, 9_000])

    plain = characterize(cur, base, starts, ends, FS, -1, noise_rms=5.0)
    assert not plain["filter_limited"].any()
    with_rise = characterize(cur, base, starts, ends, FS, -1, noise_rms=5.0,
                             filter_rise_s=1e-4)
    assert list(with_rise["filter_limited"]) == [True, False]


def test_a_zero_rise_time_flags_nothing():
    """``filter_rise_s=0`` is a filter that smears nothing; the width test must not
    invent a flag out of it (0 < 0 is False for every row)."""
    n = 20_000
    cur = np.full(n, I0)
    cur[1_000:1_010] -= DEPTH
    cur = cur + np.random.default_rng(4).normal(0.0, 5.0, n)
    df = characterize(cur, np.full(n, I0), np.array([1_000]), np.array([1_010]), FS, -1,
                      noise_rms=5.0, filter_rise_s=0.0)
    assert not df["filter_limited"].any()


def test_a_span_with_no_excursion_falls_back_to_the_dwell_rule():
    """A hand-drawn span over flat baseline has no half-max level to measure. Refusing
    to guess one leaves the pre-existing rule in charge rather than inventing a width."""
    n = 20_000
    cur = np.full(n, I0) + np.random.default_rng(5).normal(0.0, 0.0, n)  # exactly flat
    df = characterize(cur, np.full(n, I0), np.array([1_000]), np.array([1_010]), FS, -1,
                      noise_rms=5.0, filter_rise_s=1e-4)
    # dwell 100 us < 2*rise 200 us -> flagged, exactly as the old rule said.
    assert bool(df["filter_limited"].iloc[0]) is True


def test_lowpassing_the_trace_outside_the_detector_gives_the_same_verdict():
    """The physics, isolated: filter the trace FIRST, hand the spans over by hand.

    The plateau really is crushed (71 pA of 100 at 2 kHz), so this is what the flag is
    for -- and it fires whether the low-pass ran inside ``detect_events`` or before it.
    """
    n = int(FS * 0.5)
    cur = np.full(n, I0)
    starts = np.arange(10_000, n - 10_000, 8_000)[:5]
    for s in starts:
        cur[s : s + 20] -= DEPTH                      # 200 us
    filtered = np.asarray(lowpass(cur, FS, 2_000.0), dtype=np.float64)
    rise = measure_rise_time(FS, 2_000.0)
    df = characterize(filtered, np.full(n, I0), starts, starts + 20, FS, -1,
                      noise_rms=1.0, filter_rise_s=rise)
    assert bool(df["filter_limited"].all())
    assert float(df["depth_pA"].median()) < 0.8 * DEPTH
