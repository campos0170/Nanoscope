"""CF2-4 — a symmetric filter's edge lobes must not flip the measured polarity.

The regression, measured: a clean 5-blockade trace detects 5 events unfiltered
(polarity -1, confidence 1.0). Savitzky-Golay at the panel's DEFAULTS (1 ms
window, order 3) leaves one brief opposite-sign lobe flanking each event edge
-- the response plot's own documented behaviour -- so the run-count polarity
vote saw 5 down runs vs 10 up lobes, flipped to +1 at confidence 0.33 with
verdict 'measured', and detection searched the wrong direction: 0 of 5 real
events, with ``polarity_warning`` None because the gate fired only on verdict
'undetermined'. Two invariants broken at once: a low-confidence measurement was
treated as decisive, and the warning channel could not fire on the exact case
it exists for.

The fix, pinned here: a run vote inside the ambiguous band (confidence below
``_RUN_VOTE_DECISIVE``) is arbitrated by the coherent SAMPLE mass -- the
statistic the lobes cannot flip (1734 down vs 177 up samples on this fixture)
-- and every arbitration outcome that is not a corroboration carries a
``polarity_warning`` naming the likely cause. A decisive run vote is untouched,
evidence dict and all, which is what keeps unfiltered behaviour byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.detect import (
    _RUN_VOTE_DECISIVE,
    detect_events,
    detect_polarity,
)
from nanoscope.analysis.filters import savgol
from nanoscope.analysis.modules import _build_warnings

FS = 100_000.0
_BLOCKADES = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
              (70_000, 70_200), (85_000, 85_350)]


def _five_blockade_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    """The CF2-4 fixture: +100 pA baseline, 1.5 pA noise, five 30 pA blockades
    of 2-4 ms -- square events longer than the 1 ms savgol default window, i.e.
    exactly the shape the filter table's guidance says savgol is for."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _BLOCKADES:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _savgol_defaults(x: np.ndarray) -> np.ndarray:
    """The Preprocess panel's savgol defaults: 1 ms window, polyorder 3."""
    return savgol(x, FS, window_s=1e-3, polyorder=3)


# ---------------------------------------------------------------------------
# the regression itself
# ---------------------------------------------------------------------------
def test_savgol_at_defaults_still_detects_all_five_blockades():
    x = _five_blockade_trace()

    df0, prov0 = detect_events(x, FS)
    assert len(df0) == 5 and prov0["polarity"] == -1, "unfiltered fixture drifted"

    df, prov = detect_events(_savgol_defaults(x), FS)
    assert prov["polarity"] == -1, (
        "the filter's own edge lobes flipped the measured polarity again"
    )
    assert len(df) == 5, (
        f"savgol at defaults reported {len(df)} of 5 real blockades"
    )


def test_the_arbitration_is_stamped_and_the_warning_names_the_filter():
    df, prov = detect_events(_savgol_defaults(_five_blockade_trace()), FS)
    ev = prov["polarity_evidence"]

    # The overruled run vote is on the record, not silently replaced: both
    # statistics and the decider are in the evidence.
    assert ev["decided_by"] == "coherent_samples"
    assert ev["run_confidence"] < _RUN_VOTE_DECISIVE
    assert ev["sample_confidence"] >= _RUN_VOTE_DECISIVE
    assert ev["confidence"] == pytest.approx(ev["sample_confidence"])
    assert ev["verdict"] == "measured"
    assert ev["n_down_samples"] > ev["n_up_samples"]  # the mass the lobes can't flip

    # The honesty channel fires, with an actionable sentence naming the likely
    # cause and both ways out.
    w = prov["polarity_warning"]
    assert w is not None
    assert "Savitzky-Golay" in w
    assert "polarity override" in w
    assert "disable the filter" in w


def test_the_warning_reaches_the_module_warning_surface():
    """The Stats panel prints ``res.warnings``; ``_build_warnings`` is the one
    assembler of that tuple, so the polarity sentence must survive it."""
    df, prov = detect_events(_savgol_defaults(_five_blockade_trace()), FS)
    warns = _build_warnings(prov, df)
    assert any("polarity" in w for w in warns), warns
    assert any("Savitzky-Golay" in w for w in warns), warns


# ---------------------------------------------------------------------------
# unfiltered behaviour is byte-identical: a decisive run vote stamps nothing
# ---------------------------------------------------------------------------
def test_a_decisive_run_vote_leaves_the_evidence_dict_untouched():
    x = _five_blockade_trace()
    pol, ev = detect_polarity(x, 100.0)
    assert pol == -1
    assert ev["confidence"] == 1.0
    assert ev["verdict"] == "measured"
    # The arbitration keys exist ONLY inside the ambiguous band; their absence
    # here is the byte-identical guarantee for every decisive measurement.
    assert "decided_by" not in ev
    assert "run_confidence" not in ev
    assert "sample_confidence" not in ev
    assert "corroborated_by_samples" not in ev

    df, prov = detect_events(x, FS)
    assert prov["polarity_warning"] is None
    assert len(df) == 5


# ---------------------------------------------------------------------------
# the arbitration's other outcomes, pinned on hand-built votes
# ---------------------------------------------------------------------------
def _vote_trace(*, n_up_runs: int, up_len: int, n_down_runs: int, down_len: int,
                amp: float = 20.0, seed: int = 1) -> np.ndarray:
    """Zero-baseline trace with the requested coherent runs each way, spaced far
    apart so each pulse is exactly one run."""
    rng = np.random.default_rng(seed)
    n = 200_000
    x = rng.normal(0.0, 1.0, n)
    pos = 5_000
    for _ in range(n_up_runs):
        x[pos:pos + up_len] += amp
        pos += up_len + 5_000
    for _ in range(n_down_runs):
        x[pos:pos + down_len] -= amp
        pos += down_len + 5_000
    assert pos < n
    return x


def test_ambiguous_vote_with_no_decisive_sample_mass_keeps_the_runs_but_warns():
    # 3 up runs vs 2 down runs (confidence 0.2) and near-equal sample mass
    # (90 up vs 60 down, confidence 0.2): genuinely ambiguous both ways.
    x = _vote_trace(n_up_runs=3, up_len=30, n_down_runs=2, down_len=30)
    pol, ev = detect_polarity(x, 0.0)
    assert pol == 1                       # the run winner stands...
    assert ev["decided_by"] == "runs"
    assert "corroborated_by_samples" not in ev
    assert ev["confidence"] < _RUN_VOTE_DECISIVE

    df, prov = detect_events(x, FS, baseline=0.0)
    w = prov["polarity_warning"]          # ...but never silently.
    assert w is not None
    assert "low confidence" in w
    assert "polarity override" in w


def test_weak_run_vote_corroborated_by_sample_mass_stands_without_a_warning():
    # 3 up runs vs 2 down runs (confidence 0.2), but the up runs carry the
    # sample mass too (600 vs 60): two independent statistics agree, so the
    # measurement is fine and no advisory fires.
    x = _vote_trace(n_up_runs=3, up_len=200, n_down_runs=2, down_len=30)
    pol, ev = detect_polarity(x, 0.0)
    assert pol == 1
    assert ev["decided_by"] == "runs"
    assert ev["corroborated_by_samples"] is True

    df, prov = detect_events(x, FS, baseline=0.0)
    assert prov["polarity_warning"] is None


def test_a_tie_is_resolved_by_a_decisive_sample_mass():
    # 2 long down runs vs 2 short up runs: the run vote is a dead tie (which
    # used to default silently to -1 at confidence 0), but the sample mass
    # (400 down vs 60 up) is decisive and resolves it as a measurement.
    x = _vote_trace(n_up_runs=2, up_len=30, n_down_runs=2, down_len=200)
    pol, ev = detect_polarity(x, 0.0)
    assert pol == -1
    assert ev["verdict"] == "measured"
    assert ev["decided_by"] == "coherent_samples"
    assert ev["confidence"] >= _RUN_VOTE_DECISIVE


def test_a_provided_polarity_needs_no_advisory_about_the_vote_it_overrode():
    """The user's explicit override IS the instruction; the arbitration still
    runs for the evidence record, but no warning second-guesses the user."""
    df, prov = detect_events(_savgol_defaults(_five_blockade_trace()), FS,
                             polarity=-1)
    assert prov["polarity_source"] == "provided"
    assert len(df) == 5
    assert prov["polarity_warning"] is None
