"""The CUSUM detector's provenance audit trail must BALANCE.

The sibling of ``tests/analysis/test_detect_audit_trail.py``, which pins the same
discipline for the Schmitt trigger. That file is NOT edited and NOT imported from:
its ``BALANCE_KEYS`` are Schmitt-specific (``n_candidate_runs`` and
``n_rejected_below_enter`` have no counterpart here), so the two identities are
spelled out separately on purpose.

WHY THE IDENTITY IS SHORTER HERE. The Schmitt trigger emits a candidate run on the
EXIT threshold and then tests its peak against the ENTER threshold -- and 98.5 % of
its candidates die at that second test, which is the rejection that went uncounted on
the owner's -600 mV recording and forced ``n_rejected_below_enter`` into existence. A
CUSUM region does not exist until the sequential test has ALREADY passed, so there is
no such thing as a candidate that failed, and no term for one. What is left::

    n_cusum_firings
      - n_merged             (folded into a neighbour by merge_gap_s)
      - n_rejected_short     (shorter than min_dwell_s, applied AFTER the merge)
      - n_rejected_excluded  (overlapped an excluded region)
      == n_events

asserted through every gate alone and in combination, so a future gate added without
a counter cannot pass this file.

FOUR COUNTERS ARE DELIBERATELY NOT TERMS, and each is tested for that below:

  ``n_clipped_boundary``  the row is KEPT (flagged, de-accepted), so it is already
                          inside ``n_events``
  ``n_open_at_end``       likewise -- it is the detector's own name for "the trace
                          ran out before the recovery test fired", and the row it
                          describes is kept
  ``n_merges_refused``    a refused merge drops nothing; its events survive as
                          singletons and are already inside ``n_events``
  ``n_merges_marginal``   a kept merge, recorded only so it is auditable

The counters are provenance keys, never EventSet columns: the 20-column schema is
frozen and this detector does not touch a single detection number.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.detect_cusum import detect_events_cusum

FS = 50_000.0

#: The four terms of the audit identity, in the order the gates are applied.
BALANCE_KEYS = (
    "n_cusum_firings",
    "n_merged",
    "n_rejected_short",
    "n_rejected_excluded",
)


def _balance(prov: dict) -> tuple[int, int]:
    """``(firings - every rejection, n_events)`` -- equal iff the trail balances."""
    missing = [k for k in BALANCE_KEYS if k not in prov]
    assert not missing, f"the audit trail has no counter for {missing}"
    accounted = prov["n_cusum_firings"] - sum(prov[k] for k in BALANCE_KEYS[1:])
    return int(accounted), int(prov["n_events"])


def _assert_balances(prov: dict) -> None:
    accounted, n_events = _balance(prov)
    assert accounted == n_events, (
        f"{prov['n_cusum_firings']} firings - "
        + " - ".join(f"{prov[k]} {k}" for k in BALANCE_KEYS[1:])
        + f" = {accounted}, but n_events = {n_events}: "
        f"{n_events - accounted} regions vanished with no counter"
    )


def _trace(
    *, n_events: int = 60, depth_pA: float = 300.0, dwell_n: int = 200,
    noise_pA: float = 15.0, n: int = 500_000, seed: int = 3,
) -> np.ndarray:
    """A flat -1000 pA baseline with square downward events on Gaussian noise.

    The same fixture shape the Schmitt audit file uses, so the two trails are
    exercised against comparable data.
    """
    rng = np.random.default_rng(seed)
    cur = np.full(n, -1000.0, dtype=np.float64) + rng.normal(0.0, noise_pA, n)
    step = n // (n_events + 1)
    for i in range(n_events):
        s = (i + 1) * step
        cur[s:s + dwell_n] -= depth_pA
    return cur


# ================================================================================================
# 1. it balances on an ordinary detection
# ================================================================================================

def test_the_trail_balances_on_a_plain_detection():
    _df, prov = detect_events_cusum(_trace(), FS, baseline=-1000.0, min_dwell_s=0.0,
                                    merge_gap_s=None)
    assert prov["n_events"] > 50
    _assert_balances(prov)


def test_the_trail_balances_on_pure_noise():
    """Nothing is found; every counter is zero and the identity still closes. The
    early-return path writes provenance separately from the characterised path, and
    that is exactly where a missing key hides."""
    rng = np.random.default_rng(17)
    cur = np.full(400_000, -1000.0) + rng.normal(0.0, 15.0, 400_000)
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0)
    assert prov["n_events"] == 0
    _assert_balances(prov)


def test_the_trail_balances_when_nothing_is_detected_at_all():
    cur = np.full(50_000, -1000.0) + 1e-3 * np.arange(50_000)
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, cusum_k=50.0,
                                    cusum_h=500.0)
    assert prov["n_events"] == 0 and prov["n_cusum_firings"] == 0
    _assert_balances(prov)


# ================================================================================================
# 2. it balances through every gate, alone
# ================================================================================================

def test_the_trail_balances_when_min_dwell_rejects():
    cur = _trace(dwell_n=6, n_events=80)
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, min_dwell_s=1e-3,
                                    merge_gap_s=None)
    assert prov["n_rejected_short"] > 0
    assert prov["n_events"] == 0
    _assert_balances(prov)


def test_the_trail_balances_when_a_merge_absorbs_fragments():
    cur = np.full(200_000, -1000.0)
    rng = np.random.default_rng(5)
    cur += rng.normal(0.0, 10.0, cur.size)
    for s in range(20_000, 180_000, 20_000):     # pairs 0.4 ms apart -> one merge each
        cur[s:s + 500] -= 300.0
        cur[s + 520:s + 1000] -= 300.0
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, merge_gap_s=1e-3,
                                    min_dwell_s=0.0)
    assert prov["n_merged"] > 0
    _assert_balances(prov)


def test_the_trail_balances_when_an_exclusion_drops_events():
    cur = _trace(n_events=60)
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0,
                                    exclude_regions=[(0.0, 4.0)], min_dwell_s=0.0,
                                    merge_gap_s=None)
    assert prov["n_rejected_excluded"] > 0
    _assert_balances(prov)


def test_the_trail_balances_when_the_occupancy_floor_refuses_a_merge():
    """A refused merge is the one event in this pipeline that changes a counter
    (``n_merged``) without dropping anything, so it is the one most likely to be
    subtracted twice by a future edit."""
    cur = _trace(n_events=40, dwell_n=20, n=400_000)
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, merge_gap_s=0.3,
                                    min_dwell_s=0.0)
    assert prov["n_merges_refused"] > 0
    _assert_balances(prov)


# ================================================================================================
# 3. it balances with every gate at once, and at every operating point
# ================================================================================================

def test_the_trail_balances_with_every_gate_at_once():
    cur = np.full(300_000, -1000.0)
    rng = np.random.default_rng(23)
    cur += rng.normal(0.0, 12.0, cur.size)
    for s in range(10_000, 290_000, 10_000):
        cur[s:s + 400] -= 300.0
        cur[s + 420:s + 800] -= 300.0
        cur[s + 5_000:s + 5_004] -= 300.0        # too short for min_dwell_s
    _df, prov = detect_events_cusum(
        cur, FS, baseline=-1000.0, merge_gap_s=1e-3, min_dwell_s=2e-3,
        exclude_regions=[(0.0, 1.0)])
    assert prov["n_merged"] > 0
    assert prov["n_rejected_short"] > 0
    assert prov["n_rejected_excluded"] > 0
    _assert_balances(prov)


@pytest.mark.parametrize("k,h", [(0.25, 4.0), (0.75, 28.0), (2.5, 6.0), (2.5, 24.0)])
def test_the_trail_balances_at_every_operating_point_on_the_calibration_grid(k, h):
    """Including the two weak-label points the owner's own grid search produced, which
    fire on very different amounts of the trace: at k=0.25/h=4 the detector fires on
    noise, at k=2.5/h=24 it barely fires at all, and the trail must close at both."""
    _df, prov = detect_events_cusum(
        _trace(depth_pA=120.0), FS, baseline=-1000.0, cusum_k=k, cusum_h=h,
        merge_gap_s=5e-4, min_dwell_s=1e-3)
    _assert_balances(prov)


@pytest.mark.parametrize("polarity", [None, -1])
def test_the_trail_balances_however_polarity_was_decided(polarity):
    _df, prov = detect_events_cusum(_trace(), FS, baseline=-1000.0, polarity=polarity,
                                    merge_gap_s=5e-4, min_dwell_s=1e-3)
    _assert_balances(prov)


# ================================================================================================
# 4. what the counters must NOT be
# ================================================================================================

def test_clipped_boundary_rows_are_inside_n_events_not_a_rejection():
    """A clipped row is KEPT (flagged, de-accepted), so counting it as a rejection
    would break the identity in the other direction."""
    cur = _trace(n_events=20)
    cur[:3_000] -= 300.0                          # an event running off the front edge
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, min_dwell_s=0.0,
                                    merge_gap_s=None)
    assert prov["n_clipped_boundary"] >= 1
    _assert_balances(prov)


def test_an_event_open_at_the_end_of_the_trace_is_counted_but_not_subtracted():
    """``n_open_at_end`` names the detector's own failure to close an event; the row
    it describes is kept and flagged, so it is inside ``n_events`` already."""
    cur = _trace(n_events=20)
    cur[-3_000:] -= 300.0                         # still in an event when the trace ends
    df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, min_dwell_s=0.0,
                                   merge_gap_s=None)
    assert prov["n_open_at_end"] == 1
    assert prov["n_clipped_boundary"] >= 1
    assert not bool(df["accepted"].iloc[-1])
    _assert_balances(prov)


def test_open_at_end_and_clipped_boundary_are_separate_claims():
    """They answer different questions -- "did the DETECTOR run out of trace" versus
    "does the ROW touch the edge of the analysed span" -- and a min-dwell rejection
    can leave the first true and the second false. If they were ever collapsed into
    one key, this is what would be lost."""
    cur = _trace(n_events=10, n=200_000)
    cur[-3:] -= 300.0                             # a 3-sample event open at the end
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, min_dwell_s=1e-3,
                                    merge_gap_s=None)
    assert prov["n_open_at_end"] == 1             # the detector did run out of trace
    assert prov["n_clipped_boundary"] == 0        # ...but the row was dropped as short
    assert prov["n_rejected_short"] >= 1
    _assert_balances(prov)


def test_refused_merges_are_reported_but_are_not_a_rejection():
    cur = _trace(n_events=40, dwell_n=20, n=400_000)
    _df, prov = detect_events_cusum(cur, FS, baseline=-1000.0, merge_gap_s=0.3,
                                    min_dwell_s=0.0)
    assert prov["n_merges_refused"] > 0
    assert prov["n_merged"] == 0                  # nothing was actually folded away
    assert len(prov["refused_merge_occupancy"]) == prov["n_merges_refused"]
    _assert_balances(prov)


def test_the_counters_are_provenance_only_and_the_schema_is_untouched():
    """The 20-column EventSet schema is frozen: new information goes in provenance."""
    from nanoscope.analysis.events import SCHEMA

    df, prov = detect_events_cusum(_trace(), FS, baseline=-1000.0, min_dwell_s=1e-3,
                                   merge_gap_s=5e-4)
    assert list(df.columns) == list(SCHEMA)
    for key in BALANCE_KEYS:
        assert key in prov
        assert key not in df.columns
    assert "n_rejected_below_enter" not in prov, (
        "the Schmitt trigger's enter-threshold rejection has no meaning here; a key "
        "that is always zero is worse than no key"
    )


def test_the_detector_token_does_not_collide_with_the_level_fitter():
    """`analysis.cusum` is the sub-state LEVEL FITTER. A bare 'cusum' in an exported
    provenance record would be unreadable a year from now."""
    _df, prov = detect_events_cusum(_trace(n_events=5, n=100_000), FS,
                                    baseline=-1000.0, min_dwell_s=0.0)
    assert prov["detector"] == "cusum_sequential"
