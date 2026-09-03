"""B4DA-01 — the notch's own startup transient must not arrive as EVENTS.

``filters.notch`` used to call ``sosfiltfilt`` at scipy's DEFAULT pad: 21 samples for a
3-line cascade (0.1 ms at 200 kHz) on a notch whose ring-down time constant is
1/(pi * its width) -- 159 ms at Q=30 / 60 Hz. ``sosfiltfilt`` starts each pass assuming
the signal was CONSTANT at the first sample forever, which for mains hum is wrong by the
full hum amplitude, so the conditioned trace began with a one-sided transient whose
amplitude scaled linearly with the hum removed (measured: 142.9 pA of head excursion for
a 300 pA / 50 Hz line, against 0.0003 pA at the tail). Detection then reported that
ringing as ordinary ACCEPTED events -- 66 rows where 20 events existed, 45 of the extras
accepted with an empty flag, the deepest deeper than a real blockade.

The fix is at the filter: the pad is sized from the notch's OWN poles
(:func:`~nanoscope.analysis.filters.notch_settle_s`), so the transient has decayed to
1e-4 of the hum before the first real sample. Everything downstream -- the live preview,
the surrogate studies, the ML preprocess chain, detection -- gets the settled trace,
because they all go through this one kernel.

The tests below pin (1) the artefact itself, on a hum-only trace where every deviation
from DC is the filter's own doing, (2) the consequence, through the whole module path,
and (3) the honesty channel for the case the pad CANNOT fix: a trace shorter than the
ring-down, where the head is still unsettled and the result now says so.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.filters import notch, notch_settle_s, notch_unsettled_s
from nanoscope.analysis.modules import (
    DetectSchmitt,
    notch_settling_from_provenance,
    preprocess_for_detection,
    register_all,
)

FS = 100_000.0
I0, DEPTH, SIGMA = 1000.0, 100.0, 5.0
NOTCH = {"notch_enabled": True, "notch_freq_hz": 50.0, "notch_q": 30.0,
         "notch_harmonics": 3}


def _trace(hum_pA: float, *, seed: int = 1, dur_s: float = 4.0):
    """20 square 1 ms blockades on a humming open pore, plus the clean truth."""
    n = int(FS * dur_s)
    pad, step, dw = 5_000, (n - 10_000) // 20, 100
    clean = np.full(n, I0)
    for i in range(20):
        clean[pad + i * step : pad + i * step + dw] -= DEPTH
    clean = clean + np.random.default_rng(seed).normal(0.0, SIGMA, n)
    t = np.arange(n) / FS
    return clean + hum_pA * np.sin(2 * np.pi * 50.0 * t), clean


# ---------------------------------------------------------------------------
# 1. the artefact, isolated
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hum_pA", [10.0, 30.0, 100.0, 300.0])
def test_notch_leaves_no_head_transient_at_any_hum_amplitude(hum_pA):
    """A pure hum notched to nothing must leave nothing -- at the HEAD as at the tail.

    The old failure was exactly linear in the hum removed (0.476 * amplitude), so a
    sweep is the honest shape of this test: a pad that happens to be long enough for a
    small line and not for a big one would pass a single-amplitude version.
    """
    n = int(FS * 4.0)
    t = np.arange(n) / FS
    out = np.asarray(notch(hum_pA * np.sin(2 * np.pi * 50.0 * t), FS, 50.0,
                           q=30.0, harmonics=3), dtype=np.float64)
    head = float(np.abs(out[: int(0.1 * FS)]).max())
    tail = float(np.abs(out[-int(0.1 * FS) :]).max())
    # Before the fix: 4.76 / 14.29 / 47.63 / 142.89 pA of head excursion.
    assert head < 0.01 * hum_pA, (
        f"notch head transient {head:.3f} pA for a {hum_pA:g} pA line "
        f"(tail {tail:.4f} pA)")


def test_the_settling_time_is_the_notch_s_own_ring_down():
    """It is read off the designed poles, not guessed -- so it tracks Q and frequency.

    A narrower notch (higher Q) rings longer, and a 50 Hz line rings longer than a
    60 Hz one at the same Q, because both make the notch narrower in absolute Hz.
    """
    assert notch_settle_s(FS, 60.0, q=30.0, harmonics=3) == pytest.approx(1.47, abs=0.05)
    assert notch_settle_s(FS, 50.0, q=30.0) > notch_settle_s(FS, 60.0, q=30.0)
    assert notch_settle_s(FS, 60.0, q=60.0) > notch_settle_s(FS, 60.0, q=30.0)
    # It is a TIME, so it does not move when the same filter is sampled faster.
    assert notch_settle_s(2 * FS, 60.0, q=30.0) == pytest.approx(
        notch_settle_s(FS, 60.0, q=30.0), rel=1e-3)


# ---------------------------------------------------------------------------
# 2. the consequence: the events table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hum_pA", [30.0, 300.0])
def test_conditioning_hum_through_the_app_s_notch_finds_the_real_events_only(hum_pA):
    """The defect as the user met it: enable the notch, get phantom rows.

    30 pA is where it began to bite (a 6-sigma pickup, i.e. the condition that makes a
    user reach for the notch at all); 300 pA is the brief's 3x-depth case, which
    reported 66 events for 20 with 45 accepted fabrications.
    """
    raw, _clean = _trace(hum_pA)
    cond = np.asarray(preprocess_for_detection(raw, FS, NOTCH), dtype=np.float64)
    df, prov = detect_events(cond, FS, min_dwell_s=1e-4)

    truth = {5_000 + i * ((int(FS * 4.0) - 10_000) // 20) for i in range(20)}
    matched = sum(
        1 for s in df["start_idx"].to_numpy()
        if min(abs(int(s) - int(t)) for t in truth) <= 200
    )
    assert prov["n_events"] == 20, (
        f"{prov['n_events']} rows for 20 events at a {hum_pA:g} pA line",
        df.loc[:, ["t_start_s", "dwell_s", "depth_pA", "accepted", "flag"]].head(30))
    assert matched == 20
    early = df[df["t_start_s"] < 0.5]
    assert len(early) == 3, ("rows clustered in the unsettled head", len(early))


def test_the_conditioned_trace_is_as_quiet_at_the_head_as_at_the_tail():
    """The residual, measured the way the report measured it: std over 0.1 s, both ends.

    Before: 83.30 pA at the head against 0.39 pA at the tail, on a 5 pA noise floor.
    """
    raw, clean = _trace(300.0)
    cond = np.asarray(preprocess_for_detection(raw, FS, NOTCH), dtype=np.float64)
    resid = cond - clean
    head = float(resid[: int(0.1 * FS)].std())
    tail = float(resid[-int(0.1 * FS) :].std())
    assert head < 3.0 * tail + 1.0, f"head {head:.2f} pA vs tail {tail:.2f} pA"
    assert head < SIGMA


def test_module_path_with_the_notch_enabled_stamps_what_it_could_settle(context_factory):
    """Through ``DetectSchmitt.run`` -- the object the GUI's Run button drives."""
    register_all()
    raw, _ = _trace(100.0)
    res = DetectSchmitt().run(
        context_factory(raw.astype(np.float32), fs=FS, params=dict(NOTCH)))
    prov = res.payload.provenance
    assert prov["notch_enabled"] is True
    assert prov["notch_settle_s"] == pytest.approx(notch_settle_s(FS, 50.0, q=30.0), rel=1e-9)
    # 4 s of trace against a ~1.8 s ring-down: the pad fits, nothing is unsettled,
    # and therefore nothing is said about it.
    assert prov["notch_unsettled_s"] == 0.0
    assert not any("settle" in w for w in res.warnings), res.warnings
    assert len(res.payload.df) == 20


# ---------------------------------------------------------------------------
# 3. the case the pad cannot fix -- and the refusal that names its real cause
# ---------------------------------------------------------------------------
def test_a_trace_too_short_to_settle_says_so_rather_than_pretending(context_factory):
    """A hand-cut 0.2 s segment cannot hold a 1.8 s pad. That is disclosed, twice:
    the kernel warns, and the result carries the number on a user surface."""
    raw, _ = _trace(100.0, dur_s=0.2)
    unsettled = notch_unsettled_s(FS, raw.size, 50.0, q=30.0, harmonics=3)
    assert unsettled > 0.0

    with pytest.warns(RuntimeWarning, match="still carry the filter's own startup"):
        notch(raw, FS, 50.0, q=30.0, harmonics=3)

    register_all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = DetectSchmitt().run(
            context_factory(raw.astype(np.float32), fs=FS, params=dict(NOTCH)))
    prov = res.payload.provenance
    assert prov["notch_unsettled_s"] == pytest.approx(unsettled, rel=1e-9)
    said = [w for w in res.warnings if "startup transient" in w]
    assert len(said) == 1, res.warnings
    assert "may be the filter ringing rather than translocations" in said[0]


def test_notch_settling_from_provenance_says_nothing_when_the_notch_did_not_run():
    """No notch, no claim -- the helper is the one computation behind both surfaces."""
    assert notch_settling_from_provenance({"fs": FS, "n_samples": 1000}) is None
    assert notch_settling_from_provenance(
        {"notch_enabled": True, "fs": FS}) is None       # nothing to measure against
    settle, unsettled = notch_settling_from_provenance(
        {"notch_enabled": True, "fs": FS, "n_samples": int(FS * 10),
         "notch_freq_hz": 60.0, "notch_q": 30.0, "notch_harmonics": 3})
    assert unsettled == 0.0 and settle > 0.0
