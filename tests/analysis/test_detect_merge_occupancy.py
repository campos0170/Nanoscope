"""The occupancy floor must brake the DETECTOR's ``merge_gap_s``, not only the post-hoc
:mod:`nanoscope.analysis.merge`.

THE GAP THESE TESTS CLOSE. The floor was added to ``analysis.merge`` after a run of 40
clean 0.4 ms / 300 pA events spaced 8 ms apart, merged at a 10 ms window -- the very
setting :mod:`nanoscope.gui.merge_panel`'s own text recommends -- collapsed into ONE row
of dwell 312.40 ms and depth 0.70 pA, accepted, with a blank flag (the depth is a median
over a span that is 96 % open pore, so the true 300 pA blockade is simply out-voted).
It guarded the post-hoc path only. The Detection tab's ``merge_gap_s`` performs the SAME
grouping, by the same shared :func:`~nanoscope.analysis.detect.group_starts_by_gap`, and
had no floor at all -- so the identical pathology was still reachable one spinbox away,
with no refusal, no status and no provenance token.

What is pinned here:

* the reproduced train does NOT collapse at detect time, and nothing in the table has a
  dwell anywhere near the 312 ms row;
* a genuinely shattered event still merges at the SAME window (the floor must brake the
  runaway case without disabling the feature it brakes);
* the refusal is COUNTED in provenance -- ``n_merged == 0`` alone cannot distinguish
  "nothing was close enough" from "something was and it was declined";
* the counter is deliberately NOT a term of the audit identity (a refused merge drops
  nothing), which `test_detect_audit_trail.py` independently enforces;
* the primitives are still importable from ``analysis.merge`` after moving down into
  ``analysis.detect`` (the import direction forbids the reverse), so every existing
  ``from nanoscope.analysis.merge import MIN_MERGE_OCCUPANCY`` still resolves.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events

FS = 50_000.0
I0 = -1000.0
DEPTH = 300.0

#: The reproduced case, in samples at 50 kHz: 0.4 ms events (20 samples) whose starts sit
#: 8 ms (400 samples) apart, so the gap between them is 380 samples -- inside a 10 ms
#: (500-sample) merge window, which is exactly why the gap rule swept all 40 into one
#: group. Occupancy is 40*20 / (39*400 + 20) = 0.051.
_N_TRAIN = 40
_DWELL_N = 20
_PITCH_N = 400
_MERGE_GAP_S = 0.01


def _trace(spans: list[tuple[int, int]], n: int, *, noise_pA: float = 8.0,
           seed: int = 4) -> np.ndarray:
    """A flat ``I0`` baseline with square downward blockades on Gaussian noise."""
    rng = np.random.default_rng(seed)
    cur = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, noise_pA, n)
    for s, e in spans:
        cur[s:e] -= DEPTH
    return cur


def _train() -> tuple[np.ndarray, list[tuple[int, int]]]:
    spans = [
        (10_000 + i * _PITCH_N, 10_000 + i * _PITCH_N + _DWELL_N)
        for i in range(_N_TRAIN)
    ]
    return _trace(spans, n=40_000), spans


# ================================================================================
# 1. THE REPRODUCTION, at detect time
# ================================================================================

def test_a_spaced_train_does_not_collapse_at_detect_time():
    """40 isolated events, merge_gap_s = 10 ms -> 40 rows, not one 312 ms row."""
    cur, spans = _train()
    df, prov = detect_events(cur, FS, baseline=I0, merge_gap_s=_MERGE_GAP_S)

    # Un-merged control: this is what the detector sees before any grouping.
    df_raw, _ = detect_events(cur, FS, baseline=I0)
    assert len(df_raw) == _N_TRAIN, "the fixture itself did not produce 40 events"

    assert len(df) == _N_TRAIN, (
        f"the 8 ms-spaced train collapsed to {len(df)} row(s) at detect time"
    )
    assert prov["n_merged"] == 0
    # The pathology by its most visible symptom: the whole train is 312 ms wide and each
    # real event is 0.4 ms. Anything over a few ms here is glued-together open pore.
    assert df["dwell_s"].max() < 5e-3
    # ...and the depth is not diluted towards the baseline by the open pore between them.
    assert df["depth_pA"].min() > 0.5 * DEPTH


def test_the_refusal_is_counted_in_provenance():
    """``n_merged == 0`` is also what "nothing was close enough" reports; the refusal
    has to be visible as itself."""
    from nanoscope.analysis.detect import MIN_MERGE_OCCUPANCY

    cur, _spans = _train()
    _df, prov = detect_events(cur, FS, baseline=I0, merge_gap_s=_MERGE_GAP_S)

    assert prov["n_merges_refused"] == 1, "one group was proposed, one must be refused"
    assert len(prov["refused_merge_occupancy"]) == 1
    occ = prov["refused_merge_occupancy"][0]
    assert occ < MIN_MERGE_OCCUPANCY
    # 40 * 20 samples of event over a 15,620-sample span.
    assert occ == pytest.approx((_N_TRAIN * _DWELL_N) / (39 * _PITCH_N + _DWELL_N),
                                abs=0.02)

    # A detection with nothing close enough to merge reports the OTHER zero.
    far = _trace([(5_000, 5_200), (30_000, 30_200)], n=40_000)
    _df2, prov2 = detect_events(far, FS, baseline=I0, merge_gap_s=1e-3)
    assert prov2["n_merged"] == 0 and prov2["n_merges_refused"] == 0

    # And a run with no merge requested at all still carries the counters at zero.
    _df3, prov3 = detect_events(far, FS, baseline=I0)
    assert prov3["n_merges_refused"] == 0
    assert prov3["refused_merge_occupancy"] == []


# ================================================================================
# 2. the feature the floor brakes still works
# ================================================================================

def test_a_genuinely_shattered_event_still_merges_at_the_same_window():
    """Three fragments a fraction of a ms apart ARE one translocation: same 10 ms window,
    and this one must merge. A brake that also stops the intended case is not a brake."""
    spans = [(10_000, 10_400), (10_420, 10_800), (10_830, 11_200)]   # ~0.4 ms gaps
    cur = _trace(spans, n=40_000)

    raw, _ = detect_events(cur, FS, baseline=I0)
    assert len(raw) == 3, "the fixture did not shatter into three fragments"

    df, prov = detect_events(cur, FS, baseline=I0, merge_gap_s=_MERGE_GAP_S)
    assert len(df) == 1
    assert prov["n_merged"] == 2
    assert prov["n_merges_refused"] == 0
    assert df["dwell_s"].iloc[0] == pytest.approx(1200 / FS, abs=5 / FS)


def test_a_refused_group_leaves_its_members_to_the_later_gates():
    """A refusal returns singletons to the pipeline; it does not drop them. So the dwell
    gate still sees all 40, and rejecting them there is counted as a dwell rejection."""
    cur, _spans = _train()
    df, prov = detect_events(
        cur, FS, baseline=I0, merge_gap_s=_MERGE_GAP_S, min_dwell_s=1e-3
    )
    assert len(df) == 0
    assert prov["n_rejected_short"] == _N_TRAIN
    assert prov["n_merges_refused"] == 1
    # Had the refusal been treated as a rejection, these 40 would have been counted twice.
    assert prov["n_merged"] == 0


def test_the_floor_is_measured_on_occupancy_not_on_the_number_of_fragments():
    """Two events, the same count, on either side of the floor: the dense pair merges and
    the sparse pair does not."""
    dense = _trace([(10_000, 10_400), (10_450, 10_850)], n=40_000)     # occupancy 0.94
    sparse = _trace([(10_000, 10_400), (12_000, 12_400)], n=40_000)    # occupancy 0.33

    df_d, prov_d = detect_events(dense, FS, baseline=I0, merge_gap_s=0.05)
    assert len(df_d) == 1 and prov_d["n_merges_refused"] == 0

    df_s, prov_s = detect_events(sparse, FS, baseline=I0, merge_gap_s=0.05)
    assert len(df_s) == 2 and prov_s["n_merges_refused"] == 1


# ================================================================================
# 3. the move did not break the public spelling
# ================================================================================

def test_the_primitives_are_still_importable_from_analysis_merge():
    """They now live in ``analysis.detect`` (``detect`` cannot import ``merge`` -- that is
    a cycle -- so the floor had to move DOWN), and are re-exported here. The GUI and the
    existing tests import them by this path."""
    import nanoscope.analysis.detect as detect
    import nanoscope.analysis.merge as merge
    from nanoscope.analysis.merge import (       # the import IS the assertion
        LOW_OCCUPANCY_TOKEN,        # noqa: F401
        MARGINAL_MERGE_OCCUPANCY,   # noqa: F401
        MIN_MERGE_OCCUPANCY,        # noqa: F401
        span_occupancy,             # noqa: F401
    )

    for name in ("MIN_MERGE_OCCUPANCY", "MARGINAL_MERGE_OCCUPANCY",
                 "LOW_OCCUPANCY_TOKEN", "span_occupancy"):
        assert name in merge.__all__, f"{name} left merge's public API"
        assert getattr(merge, name) is getattr(detect, name), (
            f"{name} is a COPY, not the shared definition"
        )


def test_both_merge_paths_share_one_floor_implementation():
    """The refusal decision must not be re-derived; ``merge`` delegates to ``detect``."""
    import numpy as _np

    from nanoscope.analysis.detect import split_low_occupancy_groups
    from nanoscope.analysis.merge import _apply_occupancy_floor

    starts = _np.array([0, 1000, 2000])
    ends = _np.array([100, 1100, 2100])
    mask = _np.array([True, False, False])
    a = _apply_occupancy_floor(starts, ends, mask)
    b = split_low_occupancy_groups(starts, ends, mask)
    assert _np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
    assert a[0].all(), "a 5 %-occupied group must come back as singletons"


def test_the_two_paths_agree_on_the_train():
    """The point of moving the floor rather than copying it: the Detection tab's
    ``merge_gap_s`` and the post-hoc merge must reach the SAME verdict on the same
    events at the same window. They did not before -- one refused and one did not."""
    from nanoscope.analysis.events import EventSet
    from nanoscope.analysis.merge import merge_by_gap

    cur, _spans = _train()
    df_dt, prov_dt = detect_events(cur, FS, baseline=I0, merge_gap_s=_MERGE_GAP_S)

    df_raw, prov_raw = detect_events(cur, FS, baseline=I0)
    out = merge_by_gap(EventSet(df_raw, prov_raw), cur, FS, max_gap_s=_MERGE_GAP_S)

    assert out.n_merged == prov_dt["n_merged"] == 0
    assert out.n_refused == prov_dt["n_merges_refused"] == 1
    assert out.refused_occupancy[0] == pytest.approx(
        prov_dt["refused_merge_occupancy"][0], abs=1e-3)
    assert list(out.eventset.df["start_idx"]) == list(df_dt["start_idx"])
    assert list(out.eventset.df["end_idx"]) == list(df_dt["end_idx"])
