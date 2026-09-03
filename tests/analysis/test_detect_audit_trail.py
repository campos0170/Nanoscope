"""The Schmitt detector's provenance audit trail must BALANCE.

REGRESSION FOR THE apo-run-2026-08-19 FINDING **Low-23**.

On the owner's -600 mV apoferritin recording the stored provenance read::

    n_candidate_runs      18477
    n_events                276
    n_merged                  0
    n_rejected_short          0
    n_rejected_excluded       0
    n_clipped_boundary        1

18,201 candidate runs -- 98.5 % of everything the trigger saw -- disappeared with
no counter anywhere. They were rejected at the Schmitt ENTER-threshold test: a run
that crossed the exit threshold ``k_out·noise`` but whose peak never reached
``k_in·noise`` is noise and is dropped. That is by far the largest rejection the
detector makes and it was the only one that was not counted, so the one record
that is supposed to let a reader reconstruct a detection could not be reconciled
against its own event count.

``n_rejected_below_enter`` closes it. The identity this file pins::

    n_candidate_runs
      - n_rejected_below_enter
      - n_merged
      - n_rejected_short
      - n_rejected_excluded
      == n_events

``n_clipped_boundary`` is deliberately NOT a term: those rows are kept in the table
(flagged and de-accepted), so they are already inside ``n_events``.

The identity is asserted through every gate -- alone and in combination -- so a
future gate added without a counter cannot pass this file. The counters are
provenance keys, never EventSet columns: the 20-column schema is frozen and this
change does not touch a single detection number.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events

FS = 50_000.0

#: The five terms of the audit identity, in the order the gates are applied.
BALANCE_KEYS = (
    "n_candidate_runs",
    "n_rejected_below_enter",
    "n_merged",
    "n_rejected_short",
    "n_rejected_excluded",
)


def _balance(prov: dict) -> tuple[int, int]:
    """``(candidates - every rejection, n_events)`` -- equal iff the trail balances."""
    missing = [k for k in BALANCE_KEYS if k not in prov]
    assert not missing, f"the audit trail has no counter for {missing}"
    accounted = prov["n_candidate_runs"] - sum(prov[k] for k in BALANCE_KEYS[1:])
    return int(accounted), int(prov["n_events"])


def _assert_balances(prov: dict) -> None:
    accounted, n_events = _balance(prov)
    assert accounted == n_events, (
        f"{prov['n_candidate_runs']} candidates - "
        + " - ".join(f"{prov[k]} {k}" for k in BALANCE_KEYS[1:])
        + f" = {accounted}, but n_events = {n_events}: "
        f"{n_events - accounted} runs vanished with no counter"
    )


def _trace(
    *, n_events: int = 60, depth_pA: float = 300.0, dwell_n: int = 200,
    noise_pA: float = 15.0, n: int = 500_000, seed: int = 3,
) -> np.ndarray:
    """A flat -1000 pA baseline with square downward events on Gaussian noise."""
    rng = np.random.default_rng(seed)
    cur = np.full(n, -1000.0, dtype=np.float64) + rng.normal(0.0, noise_pA, n)
    step = n // (n_events + 1)
    for i in range(n_events):
        s = (i + 1) * step
        cur[s:s + dwell_n] -= depth_pA
    return cur


# ================================================================================================
# 1. the counter exists, and it is the big one
# ================================================================================================

def test_the_enter_threshold_rejection_is_counted():
    """THE Low-23 REGRESSION. Noise alone produces thousands of exit-threshold runs
    and (essentially) no events; every one of them must be counted somewhere."""
    rng = np.random.default_rng(17)
    cur = np.full(400_000, -1000.0) + rng.normal(0.0, 15.0, 400_000)
    _df, prov = detect_events(cur, FS, baseline=-1000.0)

    assert "n_rejected_below_enter" in prov, "the enter-threshold gate still has no counter"
    assert prov["n_candidate_runs"] > 1000, "the fixture did not produce candidate runs"
    # This is the gate the -600 mV recording lost 98.5 % of its candidates to.
    assert prov["n_rejected_below_enter"] > 0.9 * prov["n_candidate_runs"]
    _assert_balances(prov)


def test_the_trail_balances_on_a_plain_detection():
    _df, prov = detect_events(_trace(), FS, baseline=-1000.0)
    assert prov["n_events"] > 50
    _assert_balances(prov)


# ================================================================================================
# 2. it balances through every gate
# ================================================================================================

def test_the_trail_balances_when_min_dwell_rejects():
    cur = _trace(dwell_n=6, n_events=80)
    _df, prov = detect_events(cur, FS, baseline=-1000.0, min_dwell_s=1e-3)
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
    _df, prov = detect_events(cur, FS, baseline=-1000.0, merge_gap_s=1e-3)
    assert prov["n_merged"] > 0
    _assert_balances(prov)


def test_the_trail_balances_when_an_exclusion_drops_events():
    cur = _trace(n_events=60)
    _df, prov = detect_events(cur, FS, baseline=-1000.0, exclude_regions=[(0.0, 4.0)])
    assert prov["n_rejected_excluded"] > 0
    _assert_balances(prov)


def test_the_trail_balances_with_every_gate_at_once():
    cur = np.full(300_000, -1000.0)
    rng = np.random.default_rng(23)
    cur += rng.normal(0.0, 12.0, cur.size)
    for s in range(10_000, 290_000, 10_000):
        cur[s:s + 400] -= 300.0
        cur[s + 420:s + 800] -= 300.0
        cur[s + 5_000:s + 5_004] -= 300.0        # too short for min_dwell_s
    _df, prov = detect_events(
        cur, FS, baseline=-1000.0, merge_gap_s=1e-3, min_dwell_s=2e-3,
        exclude_regions=[(0.0, 1.0)])
    assert prov["n_merged"] > 0
    assert prov["n_rejected_short"] > 0
    assert prov["n_rejected_excluded"] > 0
    assert prov["n_rejected_below_enter"] > 0
    _assert_balances(prov)


def test_the_trail_balances_when_nothing_is_detected_at_all():
    """The early-return path (no surviving runs) writes provenance separately."""
    cur = np.full(50_000, -1000.0)
    _df, prov = detect_events(cur + 1e-9 * np.arange(50_000), FS, baseline=-1000.0,
                              k_in=50.0, k_out=40.0)
    assert prov["n_events"] == 0
    _assert_balances(prov)


@pytest.mark.parametrize("k_in", [3.0, 5.0, 8.0])
def test_the_trail_balances_at_every_enter_threshold(k_in):
    _df, prov = detect_events(_trace(depth_pA=120.0), FS, baseline=-1000.0,
                              k_in=k_in, k_out=2.0)
    _assert_balances(prov)


# ================================================================================================
# 3. what the counter must NOT be
# ================================================================================================

def test_clipped_boundary_rows_are_inside_n_events_not_a_rejection():
    """A clipped row is KEPT (flagged, de-accepted), so counting it as a rejection
    would break the identity in the other direction."""
    cur = _trace(n_events=20)
    cur[:3_000] -= 300.0                          # an event running off the front edge
    _df, prov = detect_events(cur, FS, baseline=-1000.0)
    assert prov["n_clipped_boundary"] >= 1
    _assert_balances(prov)


def test_the_counters_are_provenance_only_and_the_schema_is_untouched():
    """The 20-column EventSet schema is frozen: new information goes in provenance."""
    from nanoscope.analysis.events import SCHEMA

    df, prov = detect_events(_trace(), FS, baseline=-1000.0)
    assert list(df.columns) == list(SCHEMA)
    assert "n_rejected_below_enter" in prov
    assert "n_rejected_below_enter" not in df.columns
