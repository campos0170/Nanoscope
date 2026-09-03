"""The CUSUM EVENT DETECTOR (`analysis.detect_cusum`), against known answers.

NOT `analysis.cusum`, which is the sub-state level fitter -- see
``tests/analysis/test_cusum.py`` for that one. This file tests the detector.

Most of what follows is pinned to the SAMPLE, not to a tolerance, because the
sequential recurrence's bookkeeping (`candidate_on` / `candidate_off`, Page's
changepoint estimate) is the part a re-implementation gets subtly wrong, and a
one-sample error there is invisible in an event count while it biases every dwell
fit downstream. The exact-answer cases come from the owner's own extracted spec and
were re-derived here by hand:

  T1  noiseless square -> EXACT [(500, 599)] at four different (k, h)
  T2  an excursion shallower than k -> nothing, at any duration
  T3  a 60-sample detection delay -> the onset is still the true first sample
  T4  a recovery that never arrives -> the open event is emitted, and the app
      additionally flags it clipped_by_boundary and de-accepts it
  T5  the merge boundary, to the sample (gap 9 keeps, gap 10 fuses)
  T6  merge BEFORE min-duration: two 3-sample fragments become one 8-sample event,
      where filtering first would return zero
  T7  the event-level floor: with it the event closes on time, without it the event
      runs off the end of the trace
  T8  the frozen level does not track -- a running-mean "improvement" returns a
      visibly different answer, computed here so the contrast is not a claim
  T9  polarity: the owner's own recording geometry (negative open pore, blockade
      moves the SIGNED current UP), where a port of the notebook's hardcoded
      ``median - current`` finds ZERO events
 T11  one NaN refuses at the door instead of stalling the state machine
 T12  the 20-column schema contract, exclusive end_idx included

THE FIXTURE'S NOISE IS DETERMINISTIC AND ITS SIGMA IS EXACTLY 1.4826 pA, so the
sigma-unit thresholds mean an exact number of pA and the boundaries stay exact. See
`_dithered_trace`.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.detect_cusum import (
    CUSUM_LEVEL_FLOOR_OFFSET,
    DEFAULT_CUSUM_H,
    DEFAULT_CUSUM_K,
    cusum_onset_recovery_regions,
    detect_events_cusum,
)
from nanoscope.analysis.events import SCHEMA
from nanoscope.analysis.filters import noise_rms

FS = 50_000.0

#: ``noise_rms`` of `_dithered_trace`'s residual, exactly: the dither is the repeating
#: pattern (+1, -1, 0) pA, whose median is 0 and whose median |deviation| is 1 pA, so
#: ``1.4826 * MAD`` is 1.4826 pA on the nose -- asserted below rather than assumed.
DITHER_SIGMA = 1.4826

_DITHER = np.array([1.0, -1.0, 0.0])

#: Alternating +-1 nano-pA on the event floor. Its ONLY job is to stop `characterize`
#: flagging every synthetic event ``saturated``: that rule fires on 3+ consecutive
#: samples sitting on EXACTLY the extreme value of the analysed array, which a
#: perfectly flat synthetic event does by construction and a real amplifier does only
#: when it rails. At 1e-9 pA against a 1.4826 pA sigma it is 7e-10 sigma -- it moves no
#: statistic, no median and no MAD, and every exact-boundary assertion below still
#: holds to the sample.
_EVENT_EPS = 1e-9


def _dithered_trace(
    n: int,
    spans: list[tuple[int, int]],
    *,
    depth_sigma: float = 10.0,
    i0: float = -1000.0,
    direction: float = -1.0,
) -> np.ndarray:
    """A trace whose CUSUM boundaries are EXACT, with a real (non-zero) noise scale.

    ``spans`` are INCLUSIVE ``(start, end)`` sample pairs. Two properties make the
    boundaries exact rather than approximate, and both are load-bearing:

    * the dither is +-1 pA against a 1.4826 pA sigma, i.e. +-0.675 sigma, which is
      BELOW the drift ``k = 0.75``, so the onset statistic sits at exactly zero on
      every open-pore sample and Page's candidate lands on the event's first sample;
    * the event samples carry NO dither, so the frozen event level equals the event's
      z exactly and the recovery statistic sits at exactly zero on every in-event
      sample -- which puts ``candidate_off`` on the first open-pore sample and the
      inclusive end on the event's true last sample.

    ``direction=-1`` puts the event BELOW the open pore (polarity -1); ``+1`` puts it
    above, which on a negative ``i0`` is an ordinary blockade of |I| (see T9).
    """
    cur = i0 + np.resize(_DITHER, n)
    alt = np.where(np.arange(n) % 2 == 0, _EVENT_EPS, -_EVENT_EPS)
    for s, e in spans:
        cur[s:e + 1] = i0 + direction * depth_sigma * DITHER_SIGMA + alt[s:e + 1]
    return cur


def _run(cur, **kw):
    """`detect_events_cusum` with the gates OFF unless a test turns one on.

    ``cusum_h=6`` rather than the calibrated 28 wherever a test needs short events:
    the detection delay is ``h/(depth-k)`` samples, and at h=28 a 3-sample event
    cannot trigger at all. Tests that care about the shipped operating point say so.
    """
    kw.setdefault("baseline", -1000.0)
    kw.setdefault("cusum_k", 0.75)
    kw.setdefault("cusum_h", 6.0)
    kw.setdefault("min_dwell_s", 0.0)
    kw.setdefault("merge_gap_s", None)
    return detect_events_cusum(cur, FS, **kw)


def _spans(df) -> list[tuple[int, int]]:
    """``[(start_idx, end_idx), ...]`` -- EXCLUSIVE ends, the schema's convention."""
    return list(zip(df["start_idx"].tolist(), df["end_idx"].tolist()))


# ================================================================================
# the sequential core, on synthetic z with known answers
# ================================================================================

@pytest.mark.parametrize("k,h", [(0.5, 6.0), (0.75, 28.0), (2.5, 6.0), (2.5, 24.0)])
def test_t1_noiseless_square_gives_exact_boundaries(k, h):
    """T1. THE regression for the candidate bookkeeping: both boundaries exact, at
    four operating points spanning the whole calibration grid."""
    z = np.zeros(2000)
    z[500:600] = 6.0
    assert cusum_onset_recovery_regions(z, k, h).regions == [(500, 599)]


@pytest.mark.parametrize("dwell", [10, 100, 1000])
def test_t2_an_excursion_shallower_than_k_is_invisible(dwell):
    """T2. ``k`` is the depth floor of the detector: an excursion of 0.4 sigma cannot
    grow a statistic that subtracts 0.5 every sample, however long it lasts. Pins the
    SIGN of the drift subtraction -- flip it and this fires on the open pore."""
    z = np.zeros(3000)
    z[500:500 + dwell] = 0.4
    assert cusum_onset_recovery_regions(z, 0.5, 6.0).regions == []


def test_t3_onset_is_pages_estimate_not_the_crossing_sample():
    """T3. A 0.6-sigma event with k=0.5, h=6 takes 6/(0.6-0.5) = 60 samples to
    trigger, and the returned onset is still the true first sample. An implementation
    that returns the crossing sample gives (5060, ...) and is 1.2 ms late."""
    z = np.zeros(20_000)
    z[5000:15_000] = 0.6
    assert cusum_onset_recovery_regions(z, 0.5, 6.0).regions == [(5000, 14_999)]


def test_t4_an_event_that_never_recovers_is_emitted_open():
    """T4. The trace ends mid-event: the region is EMITTED, closed at the last
    sample, and counted under n_open_at_end. Dropping it would hide a clog."""
    z = np.zeros(1000)
    z[800:] = 6.0
    found = cusum_onset_recovery_regions(z, 0.5, 6.0)
    assert found.regions == [(800, 999)]
    assert found.n_open_at_end == 1


def test_a_flat_trace_produces_nothing_and_no_open_event():
    found = cusum_onset_recovery_regions(np.zeros(5000), 0.75, 28.0)
    assert found.regions == [] and found.n_open_at_end == 0


def test_regions_are_sorted_and_never_overlap():
    """Guaranteed by construction (a two-state machine cannot be in two events at
    once), so this asserts the construction rather than a filter."""
    rng = np.random.default_rng(11)
    z = rng.normal(0.0, 1.0, 200_000)
    for s in range(1000, 199_000, 977):        # dense, deliberately awkward spacing
        z[s:s + 40] += 8.0
    regions = cusum_onset_recovery_regions(z, 0.75, 28.0).regions
    assert len(regions) > 100
    for (s0, e0), (s1, _) in zip(regions, regions[1:]):
        assert s0 <= e0 < s1


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_non_positive_k_or_h_is_refused(bad):
    z = np.zeros(100)
    with pytest.raises(ValueError, match="cusum_k"):
        cusum_onset_recovery_regions(z, bad, 6.0)
    with pytest.raises(ValueError, match="cusum_h"):
        cusum_onset_recovery_regions(z, 0.75, bad)


def test_the_chunked_scan_agrees_with_the_plain_recurrence():
    """The vectorised closed form ``g = S - min(S)`` must equal the per-sample
    recurrence it replaces -- including ACROSS chunk boundaries, which is where a
    carried running minimum or a carried candidate index goes wrong. Checked against
    a literal transcription of the spec's pseudocode on a trace far longer than the
    1024-sample starting chunk."""
    rng = np.random.default_rng(5)
    z = rng.normal(0.0, 1.0, 60_000)
    for s in range(2000, 58_000, 1300):
        z[s:s + 90] += 5.0
    k, h = 0.75, 28.0

    # the spec's recurrence, transcribed sample by sample
    out, i, n = [], 0, z.size
    while i < n:
        g, cand, start, cross = 0.0, i, None, None
        for j in range(i, n):
            prev = g
            g += z[j] - k
            if g <= 0:
                g, cand = 0.0, j + 1
            elif prev == 0:
                cand = j
            if g >= h:
                start, cross = max(0, min(cand, j)), j
                break
        if start is None:
            break
        level = max(float(z[start:cross + 1].mean()), k + CUSUM_LEVEL_FLOOR_OFFSET)
        g2, cand2, closed = 0.0, cross + 1, None
        for j in range(cross + 1, n):
            prev = g2
            g2 += (level - z[j]) - k
            if g2 <= 0:
                g2, cand2 = 0.0, j + 1
            elif prev == 0:
                cand2 = j
            if g2 >= h:
                closed = j
                break
        if closed is None:
            out.append((start, n - 1))
            break
        out.append((start, max(start, cand2 - 1)))
        i = closed + 1

    assert cusum_onset_recovery_regions(z, k, h).regions == out
    assert len(out) > 20                       # the fixture actually exercised it


# ================================================================================
# T7 -- the event-level floor
# ================================================================================

def test_t7_the_event_level_floor_is_load_bearing():
    """T7. A 10,000-sample 0.6-sigma event at k=0.5, h=6: the onset takes 60 samples,
    so the raw level is 0.5 + 6/60 = 0.6 and the recovery drift would be 0.1 sigma,
    needing 60 samples to close. The floor lifts the level to 0.75 and the drift to
    0.25, which closes in 24. The trace is cut off 40 samples after the event, i.e.
    inside the un-floored latency and outside the floored one -- so removing the
    floor is the difference between a closed event and one that swallows the rest of
    the recording."""
    n = 15_040
    z = np.zeros(n)
    z[5000:15_000] = 0.6

    floored = cusum_onset_recovery_regions(z, 0.5, 6.0)
    assert floored.regions == [(5000, 14_999)]
    assert floored.n_open_at_end == 0
    assert floored.regions[0][1] != n - 1

    unfloored = cusum_onset_recovery_regions(z, 0.5, 6.0, level_floor_offset=0.0)
    assert unfloored.regions == [(5000, n - 1)]
    assert unfloored.n_open_at_end == 1


def test_the_raw_event_level_is_always_above_k():
    """WHY the floor can never be a no-op check on a negative drift: the onset
    statistic is zero at ``start-1`` and never resets before it crosses, so
    ``count*(level - k) >= h`` and the raw level is STRICTLY above k. The floor
    therefore binds exactly when the detection delay exceeds ``4h`` samples, and what
    it bounds is the closing LATENCY, not the sign of the drift. Asserted over a grid
    so a future edit cannot quietly reintroduce the "drift <= 0" reading."""
    for depth in (0.55, 0.6, 0.8, 2.0, 10.0):
        z = np.zeros(40_000)
        z[10_000:30_000] = depth
        k, h = 0.5, 6.0
        regions = cusum_onset_recovery_regions(
            z, k, h, level_floor_offset=-np.inf   # floor removed entirely
        ).regions
        assert regions, f"depth {depth} did not trigger"
        start, _ = regions[0]
        # re-derive the crossing the way the core does, then the level
        g, cross = 0.0, None
        for j in range(start, z.size):
            g += z[j] - k
            if g >= h:
                cross = j
                break
        level = float(z[start:cross + 1].mean())
        assert level > k
        assert level >= k + h / (cross - start + 1) - 1e-12


# ================================================================================
# T8 -- the frozen level does not track
# ================================================================================

def _running_mean_variant(z, k, h):
    """The "improvement" this codebase must never accept: identical in every respect
    except that the recovery reference is the running mean of the event SO FAR.

    Written out here so the T8 assertion is a measured contrast and not a claim.
    """
    n, out, i = z.size, [], 0
    while i < n:
        g, cand, start, cross = 0.0, i, None, None
        for j in range(i, n):
            prev = g
            g += z[j] - k
            if g <= 0:
                g, cand = 0.0, j + 1
            elif prev == 0:
                cand = j
            if g >= h:
                start, cross = max(0, min(cand, j)), j
                break
        if start is None:
            break
        total, count = float(z[start:cross + 1].sum()), cross - start + 1
        g2, cand2, closed = 0.0, cross + 1, None
        for j in range(cross + 1, n):
            total += z[j]
            count += 1
            level = max(total / count, k + CUSUM_LEVEL_FLOOR_OFFSET)   # <-- TRACKS
            prev = g2
            g2 += (level - z[j]) - k
            if g2 <= 0:
                g2, cand2 = 0.0, j + 1
            elif prev == 0:
                cand2 = j
            if g2 >= h:
                closed = j
                break
        if closed is None:
            out.append((start, n - 1))
            break
        out.append((start, max(start, cand2 - 1)))
        i = closed + 1
    return out


def test_t8_the_frozen_level_does_not_track_the_trace():
    """T8. A 6-sigma level for 50 samples stepping down to 5.0 sigma for 400, at the
    SHIPPED operating point (k=0.75, h=28).

    The frozen reference is 6.0, so 5.0 sits 1.0 below it and 0.25 sigma/sample of
    recovery evidence accumulates: the event closes 112 samples into the shallow
    level and a second onset picks the rest up. A running mean instead decays toward
    5.0 within ~19 samples, the drift dies before ``h`` is reached, and the whole
    thing comes back as ONE span.

    Both answers are pinned. The point is not that one is right -- it is that they
    are DIFFERENT, so a future "improvement" to a running mean cannot pass silently.
    """
    z = np.zeros(2000)
    z[500:550] = 6.0
    z[550:950] = 5.0

    frozen = cusum_onset_recovery_regions(z, DEFAULT_CUSUM_K, DEFAULT_CUSUM_H).regions
    assert frozen == [(500, 549), (662, 949)]

    tracking = _running_mean_variant(z, DEFAULT_CUSUM_K, DEFAULT_CUSUM_H)
    assert tracking == [(500, 949)]
    assert frozen != tracking


def test_a_deep_event_with_a_shallow_step_stays_one_event_when_the_step_is_close():
    """The other half of the same mechanism, and the reason the freeze exists at all:
    a step that stays WITHIN ``k`` of the frozen level accumulates no recovery
    evidence, so a two-level event whose levels are close is one span."""
    z = np.zeros(2000)
    z[500:550] = 6.0
    z[550:750] = 5.5              # 0.5 below the frozen 6.0, i.e. inside k = 0.75
    assert cusum_onset_recovery_regions(z, 0.75, 28.0).regions == [(500, 749)]


# ================================================================================
# the app-facing detector: fixture sanity, then the gates
# ================================================================================

def test_the_shipped_defaults_are_the_exact_label_calibrated_operating_point():
    """The four numbers the GUI will read off this module, pinned. They are the
    owner's exact-label calibration on apo-ferritin at -600 mV (k=0.75, h=28, 0.50 ms
    merge, 1.00 ms minimum) -- NOT either of the two weak-label points, which his own
    write-up reports at F1 0.074 and 0.397. Changing any of them changes what the
    Detection tab does by default, so it has to be a deliberate edit here."""
    from nanoscope.analysis import detect_cusum as m

    assert m.DEFAULT_CUSUM_K == 0.75
    assert m.DEFAULT_CUSUM_H == 28.0
    assert m.DEFAULT_CUSUM_MERGE_GAP_S == 5e-4
    assert m.DEFAULT_CUSUM_MIN_DWELL_S == 1e-3
    assert m.CUSUM_LEVEL_FLOOR_OFFSET == 0.25

    # and the signature actually defaults to them
    import inspect

    params = inspect.signature(m.detect_events_cusum).parameters
    assert params["cusum_k"].default == 0.75
    assert params["cusum_h"].default == 28.0
    assert params["merge_gap_s"].default == 5e-4
    assert params["min_dwell_s"].default == 1e-3


def test_the_fixture_has_the_exact_noise_scale_the_other_tests_assume():
    cur = _dithered_trace(2000, [(500, 599)])
    assert noise_rms(cur - (-1000.0)) == pytest.approx(DITHER_SIGMA, rel=1e-12)


def test_a_clean_square_event_is_recovered_with_exact_bounds_and_right_numbers():
    cur = _dithered_trace(2000, [(500, 599)], depth_sigma=10.0)
    df, prov = _run(cur)
    assert _spans(df) == [(500, 600)]                      # exclusive end
    assert df["dwell_s"].iloc[0] == pytest.approx(100 / FS)
    assert df["depth_pA"].iloc[0] == pytest.approx(10 * DITHER_SIGMA, rel=1e-6)
    assert prov["detector"] == "cusum_sequential"
    assert prov["n_cusum_firings"] == 1 and prov["n_events"] == 1
    assert bool(df["accepted"].iloc[0]) and df["flag"].iloc[0] == ""


def test_a_trace_with_no_events_returns_an_empty_frame_with_valid_provenance():
    cur = _dithered_trace(50_000, [])
    df, prov = _run(cur, cusum_k=DEFAULT_CUSUM_K, cusum_h=DEFAULT_CUSUM_H)
    assert len(df) == 0
    assert list(df.columns) == list(SCHEMA)
    assert prov["n_events"] == 0 and prov["n_cusum_firings"] == 0
    # the keys a reader (and EventSet.rate_hz) needs are present even with no rows
    assert prov["analyzed_duration_s"] == pytest.approx(50_000 / FS)
    assert prov["fs"] == FS and prov["n_samples"] == 50_000
    assert prov["cusum_k"] == DEFAULT_CUSUM_K and prov["cusum_h"] == DEFAULT_CUSUM_H


def test_t5_the_merge_boundary_is_exact():
    """T5. Spans (500,599) and (610,699) have exactly 10 samples strictly between
    them. Nine keeps them apart, ten fuses them. Pins the -1 in the notebook's gap
    formula against the app's exclusive ends (``next.start - prev.end_exclusive``)."""
    cur = _dithered_trace(2000, [(500, 599), (610, 699)])

    df, prov = _run(cur, merge_gap_s=9 / FS)
    assert _spans(df) == [(500, 600), (610, 700)]
    assert prov["n_merged"] == 0

    df, prov = _run(cur, merge_gap_s=10 / FS)
    assert _spans(df) == [(500, 700)]
    assert prov["n_merged"] == 1


def test_t6_merge_happens_before_the_minimum_duration_filter():
    """T6. THE test that proves the owner's stated ordering was honoured. Two
    3-sample events 2 samples apart, with a 5-sample floor: merged first they become
    one 8-sample event that survives; filtered first, both die and the answer is
    zero."""
    cur = _dithered_trace(2000, [(500, 502), (505, 507)])

    df, prov = _run(cur, merge_gap_s=2 / FS, min_dwell_s=5 / FS)
    assert _spans(df) == [(500, 508)]
    assert prov["n_cusum_firings"] == 2
    assert prov["n_merged"] == 1
    assert prov["n_rejected_short"] == 0

    df_no_merge, prov_no_merge = _run(cur, merge_gap_s=None, min_dwell_s=5 / FS)
    assert len(df_no_merge) == 0
    assert prov_no_merge["n_rejected_short"] == 2


def test_the_minimum_duration_gate_is_the_notebooks_rounded_sample_count():
    """``max(1, round(min_dwell_s * fs))`` samples, inclusive-length comparison --
    the owner's arithmetic, stamped as ``min_dwell_samples`` so a reader never has to
    re-derive it. A 100-sample event survives a 100-sample floor and dies at 101."""
    cur = _dithered_trace(2000, [(500, 599)])
    df, prov = _run(cur, min_dwell_s=100 / FS)
    assert prov["min_dwell_samples"] == 100 and len(df) == 1
    df, prov = _run(cur, min_dwell_s=101 / FS)
    assert prov["min_dwell_samples"] == 101 and len(df) == 0
    assert prov["n_rejected_short"] == 1


def test_the_occupancy_floor_refuses_a_runaway_merge():
    """The SHARED brake, applied here too. Forty clean well-separated events under a
    merge window wider than their spacing would otherwise become one row that is
    almost entirely open pore -- the failure documented on the Schmitt path (dwell
    312.40 ms, depth 0.70 pA, accepted, blank flag). The floor splits the group back
    into singletons and records the refusal."""
    spans = [(1000 + i * 4000, 1000 + i * 4000 + 60) for i in range(40)]
    cur = _dithered_trace(200_000, spans)
    df, prov = _run(cur, merge_gap_s=0.1)              # 5000 samples: swallows everything
    assert prov["n_merges_refused"] >= 1
    assert prov["n_merged"] == 0                       # nothing was actually folded away
    assert len(df) == 40
    assert all(o < prov["min_merge_occupancy"] for o in prov["refused_merge_occupancy"])


def test_events_overlapping_an_excluded_region_are_dropped_and_counted():
    cur = _dithered_trace(100_000, [(10_000, 10_099), (60_000, 60_099)])
    df, prov = _run(cur, exclude_regions=[(0.0, 0.5)])   # first 25,000 samples
    assert _spans(df) == [(60_000, 60_100)]
    assert prov["n_rejected_excluded"] == 1
    assert prov["excluded_duration_s"] == pytest.approx(0.5)
    assert prov["analyzed_duration_s"] == pytest.approx((100_000 - 25_000) / FS)


def test_an_exclusion_covering_everything_is_refused():
    cur = _dithered_trace(10_000, [(1000, 1099)])
    with pytest.raises(ValueError, match="removes the entire recording"):
        _run(cur, exclude_regions=[(0.0, 1.0)])


# ================================================================================
# T4 (app half), T9, T11, T12 -- integration with the app's contracts
# ================================================================================

def test_t4_app_half_an_open_event_is_kept_flagged_and_not_accepted():
    """The notebook keeps the open event with no marker of any kind. The app keeps
    it too -- and `characterize` flags it and de-accepts it, so the user can see that
    their trace ended mid-translocation."""
    cur = _dithered_trace(20_000, [(15_000, 19_999)])
    df, prov = _run(cur)
    assert len(df) == 1
    assert int(df["end_idx"].iloc[0]) == 20_000
    assert "clipped_by_boundary" in df["flag"].iloc[0]
    assert not bool(df["accepted"].iloc[0])
    assert prov["n_open_at_end"] == 1
    assert prov["n_clipped_boundary"] == 1


def test_an_over_long_event_is_flagged_by_max_dwell_not_dropped():
    """The notebook has NO maximum event length: a clog comes back as one enormous
    region, silently. The app's answer is `characterize`'s flag-and-de-accept."""
    # The event must stay a MINORITY of the trace, or it becomes the median and there
    # is no residual left to measure a noise scale from.
    cur = _dithered_trace(300_000, [(10_000, 60_000)])
    df, _prov = _run(cur, min_dwell_s=1e-4, max_dwell_s=0.05)
    assert len(df) == 1
    assert "dwell>max_dwell_s" in df["flag"].iloc[0]
    assert not bool(df["accepted"].iloc[0])


def test_t9_the_owners_own_recording_geometry_is_detected():
    """T9. THE MOST IMPORTANT INTEGRATION TEST HERE. Open pore at -1000 pA; the
    blockade reduces |I|, so the SIGNED current moves UP and polarity is +1. The
    notebook's hardcoded ``z = (median - current)/sigma`` has the opposite sign and
    returns nothing at all -- asserted below, so the failure mode is on record."""
    cur = _dithered_trace(2000, [(500, 599)], direction=+1.0)

    df, prov = _run(cur, baseline=-1000.0)
    assert prov["polarity"] == +1
    assert prov["polarity_source"] == "measured"
    assert _spans(df) == [(500, 600)]
    assert df["depth_pA"].iloc[0] > 0            # always a positive magnitude

    # the notebook's sign, on the same data
    resid = cur - (-1000.0)
    z_notebook = -resid / noise_rms(resid)
    assert cusum_onset_recovery_regions(z_notebook, 0.75, 6.0).regions == []


def test_a_wrong_provided_polarity_finds_nothing_rather_than_lying():
    cur = _dithered_trace(2000, [(500, 599)], direction=+1.0)
    df, prov = _run(cur, polarity=-1)
    assert prov["polarity"] == -1 and prov["polarity_source"] == "provided"
    assert len(df) == 0


@pytest.mark.parametrize("bad", [0, 7, 2, -3])
def test_a_polarity_that_is_not_a_direction_is_refused(bad):
    cur = _dithered_trace(2000, [(500, 599)])
    with pytest.raises(ValueError):
        _run(cur, polarity=bad)


@pytest.mark.parametrize("where", [0, 999, 1500, 1999])
def test_t11_one_non_finite_sample_refuses_at_the_door(where):
    """T11. The notebook's state machine STALLS on a NaN: every ``>= h`` and ``<= 0``
    comparison goes False and it silently stops detecting from that sample onward.
    This path refuses instead, with the same message `detect_events` gives."""
    cur = _dithered_trace(2000, [(500, 599)])
    cur[where] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _run(cur)
    cur[where] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _run(cur)


def test_a_trace_with_no_measurable_noise_is_refused():
    with pytest.raises(ValueError, match="no measurable noise"):
        _run(np.full(2000, -1000.0))


def test_t12_the_schema_contract():
    """T12. Same assertion the Schmitt and ML paths carry: exactly the 20 columns,
    the declared dtypes, no +/-Inf in any float column, and an EXCLUSIVE end_idx
    (which is what makes ``dwell_s == (end_idx - start_idx)/fs`` true -- get the
    inclusive->exclusive conversion wrong and every dwell is one sample short)."""
    cur = _dithered_trace(100_000, [(i * 5000 + 1000, i * 5000 + 1200) for i in range(15)])
    df, _prov = _run(cur)
    assert len(df) == 15
    assert list(df.columns) == list(SCHEMA)
    for col, (dtype, _unit, _desc) in SCHEMA.items():
        assert df[col].dtype == dtype, f"{col} is {df[col].dtype}, schema says {dtype}"
    floats = [c for c, (d, _, _) in SCHEMA.items() if d == "float64"]
    assert not np.isinf(df[floats].to_numpy(dtype=float)).any()
    dwell = (df["end_idx"].to_numpy() - df["start_idx"].to_numpy()) / FS
    assert np.allclose(df["dwell_s"].to_numpy(), dwell)


def test_provenance_carries_the_keys_a_reader_and_a_rebuild_need():
    cur = _dithered_trace(50_000, [(10_000, 10_500)])
    df, prov = _run(cur, merge_gap_s=5e-4, max_dwell_s=0.05, recording="r.abf",
                    pore_id="p1", units="pA")
    for key in (
        "nanoscope_version", "detector", "detected_at", "recording", "source",
        "pore_id", "units", "fs", "n_samples", "duration_s", "analyzed_duration_s",
        "exclude_regions", "excluded_duration_s", "min_dwell_s", "max_dwell_s",
        "merge_gap_s", "noise_rms_pA", "noise_method", "polarity",
        "polarity_requested", "polarity_source", "polarity_measured",
        "polarity_evidence", "polarity_warning", "cusum_k", "cusum_h",
        "cusum_level_floor", "cusum_sigma_source", "n_cusum_firings", "n_merged",
        "n_rejected_short", "n_rejected_excluded", "n_open_at_end",
        "n_clipped_boundary", "n_events", "n_filter_limited",
        "analysed_start_idx", "analysed_stop_idx",
    ):
        assert key in prov, f"provenance has no {key!r}"
    assert prov["cusum_level_floor"] == pytest.approx(0.75 + CUSUM_LEVEL_FLOOR_OFFSET)
    assert prov["analysed_start_idx"] == 0 and prov["analysed_stop_idx"] == 50_000
    assert df["recording"].iloc[0] == "r.abf" and df["pore_id"].iloc[0] == "p1"


def test_the_shared_baseline_estimator_is_used_and_stamped():
    """Not a local median: the shared `characterisation_baseline`, whose fingerprint
    is what makes a CUSUM row and a Schmitt row on the same recording comparable."""
    cur = _dithered_trace(200_000, [(i * 10_000 + 2000, i * 10_000 + 2300) for i in range(15)])
    df, prov = detect_events_cusum(cur, FS, cusum_k=0.75, cusum_h=6.0, min_dwell_s=0.0)
    assert prov["baseline_provided"] is False
    assert "baseline_fingerprint" in prov
    assert len(df) == 15
    assert df["baseline_pA"].abs().min() > 900.0


def test_a_provided_baseline_stamps_no_fingerprint():
    cur = _dithered_trace(20_000, [(5000, 5200)])
    _df, prov = _run(cur)
    assert prov["baseline_provided"] is True
    assert prov["baseline_method"] == "provided"
    assert "baseline_fingerprint" not in prov


def test_a_low_pass_marks_short_events_filter_limited():
    """Gaussian noise here, not the dither: a 5 kHz low-pass annihilates a 3-sample
    dither pattern outright and leaves a residual with no measurable noise at all."""
    rng = np.random.default_rng(41)
    n = 200_000
    cur = np.full(n, -1000.0) + rng.normal(0.0, 10.0, n)
    for i in range(15):                        # 0.16 ms events, near the rise time
        s = i * 10_000 + 2000
        cur[s:s + 8] -= 800.0
    df, prov = detect_events_cusum(
        cur, FS, baseline=-1000.0, min_dwell_s=0.0, merge_gap_s=None,
        filter_cutoff_hz=5000.0,
    )
    assert prov["filter_cutoff_hz"] == 5000.0
    assert prov["filter_kind"] == "bessel" and prov["filter_zero_phase"] is True
    assert prov["filter_order"] == 8
    assert prov["filter_rise_s"] > 0
    assert len(df) >= 15
    # THE CONTRACT: the flag is set on exactly the rows the rise time makes it true
    # for, and provenance counts the rows rather than a parallel prediction.
    expected = df["dwell_s"].to_numpy() < 2.0 * prov["filter_rise_s"]
    assert np.all(df["filter_limited"].to_numpy(dtype=bool)[expected])
    assert prov["n_filter_limited"] == int(df["filter_limited"].sum())
    assert prov["n_filter_limited"] > 0


def test_detection_is_cancellable():
    from nanoscope.analysis.cancel import OperationCancelled

    cur = _dithered_trace(400_000, [(i * 10_000 + 2000, i * 10_000 + 2300) for i in range(30)])
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(OperationCancelled):
        _run(cur, cancel=cancel)


# ================================================================================
# behaviour on realistic noisy data
# ================================================================================

def _noisy_trace(n=400_000, *, dwell=200, depth_pA=200.0, noise=15.0, spacing=8000, seed=3):
    rng = np.random.default_rng(seed)
    cur = np.full(n, -1000.0) + rng.normal(0.0, noise, n)
    starts = list(range(4000, n - 4000, spacing))
    for s in starts:
        cur[s:s + dwell] -= depth_pA
    return cur, starts


def test_deep_short_and_shallow_long_events_are_both_found():
    """A fixed-threshold detector trades these two off; a sequential test should get
    both. Deep+short (13 sigma, 0.4 ms) and shallow+long (2.7 sigma, 8 ms) at the
    SHIPPED operating point, each within a couple of samples of the truth."""
    for dwell, depth in ((20, 200.0), (400, 40.0)):
        cur, starts = _noisy_trace(dwell=dwell, depth_pA=depth, spacing=8000)
        df, _prov = detect_events_cusum(
            cur, FS, baseline=-1000.0, cusum_k=DEFAULT_CUSUM_K,
            cusum_h=DEFAULT_CUSUM_H, min_dwell_s=0.5 * dwell / FS,
            merge_gap_s=5e-4,
        )
        found = df["start_idx"].to_numpy()
        matched = sum(np.any(np.abs(found - s) <= 5) for s in starts)
        assert matched >= 0.9 * len(starts), (
            f"dwell {dwell}, depth {depth}: matched {matched}/{len(starts)}"
        )
        # and the dwell is right, not merely present
        assert df["dwell_s"].median() == pytest.approx(dwell / FS, rel=0.15)


def test_pure_noise_yields_essentially_nothing_at_the_shipped_operating_point():
    rng = np.random.default_rng(19)
    cur = np.full(400_000, -1000.0) + rng.normal(0.0, 15.0, 400_000)
    df, prov = detect_events_cusum(cur, FS, baseline=-1000.0)
    assert len(df) == 0, f"{len(df)} false events on pure noise: {prov['n_cusum_firings']} firings"


def test_an_event_with_internal_structure_stays_one_event():
    """A translocation that steps between depths mid-passage, at the shipped
    operating point. The recovery statistic is referenced to the OBSERVED level, so a
    step that stays within ``k`` of it accumulates no recovery evidence and the event
    stays whole -- 8 ms of 300/295/300 pA comes back as ONE row of the right dwell,
    not three."""
    rng = np.random.default_rng(31)
    n = 200_000
    cur = np.full(n, -1000.0) + rng.normal(0.0, 10.0, n)
    planted = list(range(5000, n - 5000, 10_000))
    for s in planted:
        cur[s:s + 100] -= 300.0
        cur[s + 100:s + 300] -= 295.0          # 0.5 sigma shallower: inside k = 0.75
        cur[s + 300:s + 400] -= 300.0
    df, _prov = detect_events_cusum(
        cur, FS, baseline=-1000.0, min_dwell_s=2e-3, merge_gap_s=5e-4,
    )
    # The MEDIAN DWELL is the assertion that matters: 400 samples means the internal
    # step was not read as a recovery. The count is allowed one or two extra rows
    # (measured 19-21 over five seeds) because the frozen level is estimated from a
    # short window and occasionally overshoots -- see the note in
    # `cusum_onset_recovery_regions` and `test_deep_events_fragment_and_a_wider_merge_repairs_them`.
    assert df["dwell_s"].median() == pytest.approx(400 / FS, rel=0.05)
    assert len(planted) <= len(df) <= len(planted) + 2


def test_one_operating_point_closes_events_of_very_different_depths():
    """WHY the recovery reference is state-dependent at all. A single ABSOLUTE
    recovery threshold has to be tuned per depth; referencing it to the observed
    level means one (k, h) closes a 3-sigma and a 30-sigma event equally well.
    Ten-fold depth range, one operating point, every dwell right."""
    rng = np.random.default_rng(53)
    n = 300_000
    cur = np.full(n, -1000.0) + rng.normal(0.0, 10.0, n)
    depths = [30.0, 60.0, 100.0]                   # 3, 6 and 10 sigma
    planted = []
    for i, s in enumerate(range(5000, n - 5000, 10_000)):
        cur[s:s + 300] -= depths[i % 3]
        planted.append(s)
    df, _prov = detect_events_cusum(
        cur, FS, baseline=-1000.0, min_dwell_s=2e-3, merge_gap_s=5e-4,
    )
    found = df["start_idx"].to_numpy()
    matched = sum(np.any(np.abs(found - s) <= 10) for s in planted)
    assert matched >= len(planted) - 1, f"{matched}/{len(planted)}"
    assert df["dwell_s"].median() == pytest.approx(300 / FS, rel=0.05)
    # the shallow third was found, not merely counted: both ends of the range appear
    assert df["depth_pA"].min() < 40.0 and df["depth_pA"].max() > 90.0


def test_deep_events_fragment_and_a_wider_merge_repairs_them():
    """THE DETECTOR'S ONE PRACTICAL QUIRK, pinned so nobody rediscovers it in anger.

    The frozen level is a mean of ``ceil(h/(depth-k))`` samples, so a VERY deep event
    gets a 2-sample estimate whose standard error is 0.7 sigma -- and whenever it
    overshoots by more than ``k`` the recovery drift turns positive INSIDE the event
    and the event is emitted as two fragments. Measured here: 29 planted 30-sigma
    events produce 36 raw firings (24 % split). The 0.5 ms default gap is too narrow
    to bridge them; a 5 ms gap recovers the right dwell.

    This is the algorithm as specified -- the spec is explicit that the level window
    is the detection delay and that there is no minimum-sample requirement -- so the
    test documents it rather than working around it.
    """
    rng = np.random.default_rng(77)
    n = 300_000
    cur = np.full(n, -1000.0) + rng.normal(0.0, 10.0, n)
    planted = list(range(5000, n - 5000, 10_000))
    for s in planted:
        cur[s:s + 300] -= 300.0                    # 30 sigma: a 2-sample level window

    _df, narrow = detect_events_cusum(
        cur, FS, baseline=-1000.0, min_dwell_s=0.0, merge_gap_s=None)
    assert narrow["n_cusum_firings"] > len(planted)

    wide, prov = detect_events_cusum(
        cur, FS, baseline=-1000.0, min_dwell_s=0.0, merge_gap_s=5e-3)
    assert prov["n_merged"] > 0
    assert wide["dwell_s"].median() == pytest.approx(300 / FS, rel=0.05)


def test_a_trace_that_begins_mid_event_is_caught_and_flagged_not_silently_lost():
    """The received wisdom is that a change detector cannot see an event that has
    already started. HALF RIGHT, and the half that is wrong is the one that matters:
    ``z`` is already positive at sample 0, so the statistic accumulates from the first
    sample and Page's candidate lands on 0. The row comes back with EXACT bounds,
    flagged ``clipped_by_boundary`` and NOT accepted -- correct, because the true
    onset is outside the array.

    The blind spot that IS real needs the depressed level to move the baseline, which
    is a property of the baseline estimate and not of this test."""
    cur = _dithered_trace(20_000, [(0, 4000)])
    df, prov = _run(cur)
    assert _spans(df) == [(0, 4001)]
    assert "clipped_by_boundary" in df["flag"].iloc[0]
    assert not bool(df["accepted"].iloc[0])
    assert prov["n_clipped_boundary"] == 1
