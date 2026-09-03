"""The load-time probe and the proposal rule.

WHAT IS BEING PINNED
====================
The app can no longer ship one baseline default, because two recordings in the
owner's own evidence need windows 60x apart: the apo ``_0006_`` file holds a 521 ms
partial clog and needs ``rough_s = 2.0``, while ``ferritin_data3`` is a gating pore
stepping 296 pA and needs 0.03. So the app MEASURES each recording and PROPOSES.

The acceptance test for the rule is that it reproduces the four known-good parameter
sets from the data alone -- see
:func:`test_the_rule_reproduces_the_known_good_parameter_sets`. Three do so exactly;
the fourth (``ferritin_data1``) is an honest partial and the test says so.

The constants are a MATCHED SET, and the tests below are written to fail if any one
of them is changed alone:

* `PROBE_CLOSE_GAP_S` at 1 ms is the only value at which the apo clog is measured
  whole (497 ms) AND ferritin_data2 still proposes its shipped 0.5 s. At gap 0 the
  clog fragments into runs of <= 26 ms and ``rough_s`` comes out 20x under-sized; at
  4 ms data2's distinct events are welded together and it proposes 1.0 s.
* `ROUGH_MULTIPLE` is 3, not the spec's 4, because `longest_state_s` is one statistic
  applied to all four recordings and reads ~1.3x higher than the spec's per-file
  numbers, which came from three different threshold and closing settings.

WHAT IS DELIBERATELY NOT TESTED
===============================
That the proposal is OPTIMAL. It is not an optimiser and must never become one: the
reference's own QC triple is monotone in window length, so a search over it selects
the shortest window there is -- exactly the defect this whole change removes. See
:func:`test_the_qc_triple_prefers_the_degenerate_parameters`, which asserts that the
degenerate set really does score better, i.e. that auto-tuning would be wrong.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope

FS = 50_000.0

CSV = Path("/Users/abrahamcampos/Desktop/Research-Data/FerritinModeler/CSVData")
APO = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")

#: The four parameter sets the rule must reproduce from the data alone.
#: ``ferritin_data1``'s notebook setting is a SINGLE-PASS ``robust_median`` at 0.5 s,
#: which is not a ``rough_s`` at all -- the rule's rough window happens to land on the
#: same number, and no measurement here chooses between the one- and two-pass
#: estimator (the spec gives no rule for it either, and leaves that row unticked).
KNOWN_GOOD = {
    "ferritin_data2": (0.5, 0.1),
    "ferritin_data3": (0.03, 0.01),
    "apo_0006": (2.0, 0.5),
}


# ===========================================================================
# synthetic fixtures
# ===========================================================================
def synth(*, n=1_500_000, i0=-100.0, noise=5.0, seed=11,
          events=((3000, 300),), depth=60.0, long_state=None, drift=0.0,
          drift_period_s=2.0) -> np.ndarray:
    """A trace with POISSON-arriving events, optionally one long state and drift.

    Arrival times are exponential, not periodic, and that is load-bearing rather than
    decorative: perfectly periodic events beat against the probe's 20 ms gating block
    grid, and an event period that is an exact multiple of it makes the block medians
    alternate deterministically. Measured, a 12 ms event every 60 ms (exactly three
    blocks) scores 9.79 on the gating statistic at only 20% occupancy, where the same
    occupancy with Poisson arrivals scores 0.80. Real translocations arrive at random,
    so the periodic fixture was testing an artefact of the fixture.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = i0 + rng.normal(0.0, noise, n)
    if drift:
        x += drift * np.sin(2 * np.pi * t / drift_period_s)
    for gap, dwell in events:
        s = 2000
        while s < n - dwell - 2000:
            x[s:s + dwell] -= depth
            s += max(int(rng.exponential(max(gap - dwell, 1))) + dwell, dwell + 1)
    if long_state is not None:
        start, samples, dep = long_state
        x[start:start + samples] -= dep
    return x.astype(np.float32)


def gating(*, n=1_200_000, level_s=0.035, step=290.0, seed=5) -> np.ndarray:
    """A pore whose OPEN level steps between two conductances every ``level_s``."""
    rng = np.random.default_rng(seed)
    x = np.full(n, -400.0) + rng.normal(0.0, 14.0, n)
    k = int(level_s * FS)
    hi = False
    for s in range(0, n, k):
        if hi:
            x[s:s + k] += step
        hi = not hi
    return x.astype(np.float32)


# ===========================================================================
# 1. THE GRID
# ===========================================================================
@pytest.mark.parametrize("value,want", [
    (0.583, 0.5),      # ferritin_data1's 3 x 194 ms
    (0.505, 0.5),      # ferritin_data2's 3 x 168 ms
    (1.488, 2.0),      # apo's 3 x 496 ms -- the tight one
    (0.0095, 0.01),    # ferritin_data3's level dwell / 3
    (0.033, 0.03),     # the notebook's own data3 rough window
    (0.125, 0.1),      # 0.5 / 4
    (0.5, 0.5),        # 2.0 / 4
])
def test_the_grid_snaps_the_measured_values_onto_the_notebooks_own_numbers(value, want):
    assert P.snap_to_grid(value) == pytest.approx(want)


def test_the_grid_is_log_nearest_and_not_linear_nearest():
    """0.033 s is ferritin_data3's shipped rough window. LINEAR-nearest on a 1-2-3-5
    grid still gives 0.03, but on the 1-2-5 grid the spec's prose implies it gives
    0.05 -- 1.5x wrong, and unable to express 0.03 at all. Both halves matter."""
    assert 3.0 in P.GRID_MANTISSAS, "a 1-2-5 grid cannot express ferritin_data3's 0.03"
    # log-nearest puts 1.488 above the 1.0/2.0 boundary (geometric mean 1.414);
    # linear-nearest would put it below (arithmetic mean 1.5) and propose 1.0.
    assert P.snap_to_grid(1.488) == 2.0
    assert 1.488 < 1.5 and 1.488 > np.sqrt(2.0)


@pytest.mark.parametrize("value", [0.25, 0.3, 0.31, 0.35])
def test_the_grid_returns_the_decimal_it_prints_not_one_ulp_above_it(value):
    """REGRESSION: the same window must have ONE identity.

    ``BaselineConfig.fingerprint`` hashes the fields' ``repr``, so a float one ULP
    off is a different baseline configuration as far as every comparability check in
    this app is concerned. The grid was built as ``m * 10.0 ** decade``, and
    ``3 * 10.0**-1`` is ``0.30000000000000004`` — while the Baseline panel's
    4-decimal spin box hands back ``0.3``. Same window, two fingerprints
    (``sha256:af17c2c64914`` vs ``sha256:b64a75ebdf29`` on the owner's −500 mV file,
    with all 20 event columns bit-identical), so anything stamping the raw proposal
    made two identically-windowed recordings read NOT COMPARABLE in Analysis ▸ Plots
    and in the voltage-dependence window — a false alarm in the one feature whose
    whole job is to be believed.

    0.3 is the ONLY point on this grid where the two spellings differ, and no
    derived integer moves either way (``derive_decim`` 200 and ``w`` 75 at 50 kHz,
    checked from 1 kHz to 1 MHz).
    """
    snapped = P.snap_to_grid(value)
    assert snapped == 0.3
    assert repr(snapped) == "0.3", f"one ULP off: {snapped!r}"


def test_a_proposal_and_the_panels_own_spin_box_agree_on_the_fingerprint():
    """The end the regression is about: the probe's proposed 1.0/0.3 s window and the
    same window typed into the panel must be the SAME baseline configuration."""
    from nanoscope.analysis.baseline import DEFAULT_BASELINE

    proposed = DEFAULT_BASELINE.replace(
        rough_s=P.snap_to_grid(1.0), base_s=P.snap_to_grid(1.0 / P.ROUGH_BASE_RATIO),
        decim=None)
    typed = DEFAULT_BASELINE.replace(rough_s=1.0, base_s=0.3, decim=None)
    assert proposed.fingerprint(50_000.0) == typed.fingerprint(50_000.0)
    # and it is the fingerprint the -500 mV reference run recorded
    assert proposed.fingerprint(50_000.0) == "sha256:b64a75ebdf29"


@pytest.mark.parametrize("fs", [1e3, 5e4, 2.5e5, 1e6])
def test_the_grid_change_moves_no_window_the_estimator_actually_uses(fs):
    """The identity changed; the measurement must not. Both spellings of every grid
    point give the same stride and the same window length in samples."""
    from nanoscope.analysis.baseline import derive_decim

    for decade in range(-4, 3):
        for m in P.GRID_MANTISSAS:
            literal, product = float(f"{m:g}e{decade}"), m * 10.0 ** decade
            if not (P.GRID_MIN_S <= literal <= P.GRID_MAX_S):
                continue
            try:
                d_lit = derive_decim(literal, fs)
            except ValueError:
                # Too few samples for a decimated median at this rate; the estimator
                # refuses BOTH spellings, which is itself the agreement being checked.
                with pytest.raises(ValueError):
                    derive_decim(product, fs)
                continue
            d_prod = derive_decim(product, fs)
            assert d_lit == d_prod
            assert int(literal * fs / d_lit) == int(product * fs / d_prod)


def test_the_grid_clamps_rather_than_proposing_something_unrunnable():
    assert P.snap_to_grid(1e-9) == P.GRID_MIN_S
    assert P.snap_to_grid(1e6) == P.GRID_MAX_S
    assert P.snap_to_grid(float("nan")) == P.GRID_MIN_S
    assert P.snap_to_grid(-1.0) == P.GRID_MIN_S


# ===========================================================================
# 2. THE PROBE -- refusals and the honest fallback
# ===========================================================================
def test_a_non_finite_trace_is_refused_before_anything_is_measured():
    """2.7: a NaN reaching the kernel makes its sigma NaN, which makes the mask
    threshold all-False, which SILENTLY DISABLES the entire event mask and returns a
    finite, plausible, 12 pA-wrong baseline. The probe must refuse first, not
    propose windows measured against a disabled mask."""
    x = synth(n=200_000)
    x[123] = np.nan
    with pytest.raises(ValueError, match="finite"):
        P.probe_timescales(x, FS)


def test_a_recording_too_short_to_see_a_state_falls_back_and_says_so():
    """The proposal must never be a shrunken probe window. R5: shrinking a window to
    fit is exactly the degenerate auto-tune this design refuses."""
    probe = P.probe_timescales(synth(n=20_000), FS)      # 0.4 s
    assert probe.usable is False
    proposal = P.propose_baseline(probe)
    assert proposal.config is B.DEFAULT_BASELINE, "the fallback must be the app's ONE literal"
    assert proposal.param_source == "default"
    assert not proposal.confident
    assert "0.4" in proposal.reason and "default" in proposal.reason.lower()


def test_the_fallback_is_the_default_by_identity_not_by_respelling():
    """If the fallback ever became ``BaselineConfig(rough_s=0.5, base_s=0.1)`` it
    would be a SECOND baseline literal, free to drift from `DEFAULT_BASELINE` --
    which is the class of defect ``test_baseline_seam.py``'s guards exist to stop."""
    for probe in (None, P.probe_timescales(synth(n=20_000), FS)):
        assert P.propose_baseline(probe).config is B.DEFAULT_BASELINE


def test_a_constant_trace_is_refused_rather_than_proposing_from_zero_noise():
    probe = P.probe_timescales(np.full(1_000_000, -100.0), FS)
    assert probe.usable is False
    assert "noise" in probe.notes[0]
    assert P.propose_baseline(probe).config is B.DEFAULT_BASELINE


def test_the_probe_polls_the_cancellation_token():
    """C: every long-running kernel in this app polls the ambient token. A 200 ms
    probe that could not be cancelled would hold a Cancel click for its whole
    duration on a long recording."""
    x = synth(n=1_500_000)
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            P.probe_timescales(x, FS, polarity=-1)


# ===========================================================================
# 3. THE THREE MEASUREMENTS
# ===========================================================================
def test_the_sigma_is_the_one_sided_upper_mad_and_the_delta_estimator_is_not():
    """2.6(a). The delta estimator reads ~7x low on a low-passed trace, and every
    quantity built on it -- mask fraction, dwell percentiles, the gating flag -- comes
    out garbage. This asserts the probe uses the one that survives."""
    x = synth(n=600_000, noise=6.0, events=((2500, 400),), depth=80.0)
    probe = P.probe_timescales(x, FS, polarity=-1)
    # The tolerance is 25%, not 5%, and the slack is a MEASURED bias rather than
    # noise: at ~16% event occupancy this reads 6.9 against a true 6.0, because the
    # residual's median is pulled down by the events while its upper half is not, so
    # the one-sided spread about that median widens. It is the price of the estimator
    # that survives events at all, it is in the safe direction (a wider sigma masks
    # LESS, never more), and on the owner's real files it lands within 10% low
    # (data1 6.26 against a QC noise of 6.75; apo 4.60 against 4.84).
    assert probe.sigma_pA == pytest.approx(6.0, rel=0.25)
    assert probe.sigma_pA > 6.0, "the bias is upward; if it inverts, re-derive it"
    # ...and the estimator it rejects agrees HERE, on white noise, which is precisely
    # why its 7x error stayed invisible until it was measured on the owner's real,
    # already-low-passed exports.
    delta = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2.0)
    assert delta == pytest.approx(6.0, rel=0.15)


def test_a_long_shallow_state_is_found_only_because_of_the_closing_step():
    """2.6(c), the measured heart of the probe. The apo file's 521 ms clog is shallow
    enough that 17% of its samples do not cross 3.5 sigma, so WITHOUT closing it
    fragments and the longest state reads ~1/20th of the truth.

    Reproduced on a synthetic with the same shape: a 500 ms state at ~4.5 sigma."""
    n = 1_500_000
    x = synth(n=n, noise=5.0, events=((4000, 200),), depth=60.0,
              long_state=(700_000, 25_000, 22.0))       # 500 ms at ~4.4 sigma
    probe = P.probe_timescales(x, FS, polarity=-1)

    # Same residual, same threshold; the ONLY difference is the closing gap.
    from nanoscope.analysis.baseline_probe import _closed_dwells
    cfg = B.DEFAULT_BASELINE.replace(rough_s=probe.probe_rough_s,
                                     base_s=probe.probe_rough_s / 4, decim=None)
    res = B.characterisation_baseline(np.asarray(x, np.float64), FS, config=cfg,
                                      polarity=-1, health=False, cache=False)
    resid = np.asarray(x, np.float64) - res.baseline
    mask = resid < -P.PROBE_MASK_SIGMA * probe.sigma_pA
    bare = float(np.percentile(_closed_dwells(mask, FS, 0.0), P.LONGEST_STATE_Q))

    assert probe.longest_state_s > 4.0 * bare, (
        f"closing recovered {probe.longest_state_s * 1e3:.0f} ms against "
        f"{bare * 1e3:.0f} ms bare -- if the ratio ever falls to ~1 the closing has "
        "stopped being load-bearing and PROBE_CLOSE_GAP_S must be re-derived rather "
        "than left as decoration")
    # ...and only the closed answer proposes a window that clears the state.
    assert P.snap_to_grid(P.ROUGH_MULTIPLE * probe.longest_state_s) >= 1.0
    assert P.snap_to_grid(P.ROUGH_MULTIPLE * bare) < 0.5


def test_the_gating_flag_separates_a_stepping_pore_from_a_busy_one():
    """A pore whose OPEN level steps is a different problem from a pore with long
    events, and the rule INVERTS between them: the stepping pore's long states are
    baseline levels to follow, not events to mask."""
    busy = P.probe_timescales(synth(n=1_200_000, events=((3000, 600),), depth=90.0),
                              FS, polarity=-1)
    steps = P.probe_timescales(gating(), FS, polarity=-1)
    assert not busy.gating, busy.gating_score
    assert steps.gating, steps.gating_score
    assert steps.gating_score > P.GATING_THRESHOLD > busy.gating_score


def test_the_gating_flag_cries_wolf_above_a_quarter_occupancy():
    """A MEASURED LIMIT of the flag, asserted rather than hoped away.

    The statistic is the spread of 20 ms block medians. Once the pore is blocked for
    more than ~25% of the time by events that are long compared with that block, the
    block median itself lands in the blockade often enough to look like a second
    conductance level. Measured with Poisson arrivals at 90 pA over 6 pA noise:

    ========  ======  ======  ======  ======
    dwell     5%      10%     20%     30%
    ========  ======  ======  ======  ======
    2 ms      0.13    0.22    0.30    0.38
    6 ms      0.07    0.41    0.45    0.68
    12 ms     0.06    0.07    0.80    **8.9**
    30 ms     0.06    0.07    0.44    **12.0**
    ========  ======  ======  ======  ======

    Two things make this acceptable rather than a defect. First, all four real
    recordings sit far away: 1.7 (data1, 15% masked), 4.7 (data2), 0.5 (apo),
    13.8 (the genuinely gating data3) -- 1.7x margin either side of the threshold.
    Second, 25-30% occupancy is already inside the regime where 2.8 says the
    ESTIMATOR itself starts to fail (35% duty -> 3.7 pA error, 60% -> a sign flip), so
    a recording that trips this is one the user has to look at anyway.

    And it is the sharpest argument for propose-and-confirm there is: this is exactly
    the case an auto-applying app would get silently wrong.
    """
    heavy = P.probe_timescales(
        synth(n=1_200_000, noise=6.0, events=((2000, 600),), depth=90.0),
        FS, polarity=-1)
    assert heavy.mask_fraction > 0.25
    assert heavy.gating, (
        "if this stops firing the limit has moved and the table above is stale; "
        "re-measure it rather than deleting the test")


def test_the_gating_flag_is_normalised_by_the_within_block_noise_not_by_sigma():
    """The one-sided upper MAD OVER-READS 3-4x on exactly the recording the flag has
    to catch (53.5 pA against a true 13.8 on ferritin_data3), so normalising the
    gating statistic by it would drag the stepping pore's score DOWN below the busy
    pore's -- the flag would be strongest where it is least needed."""
    steps = P.probe_timescales(gating(), FS, polarity=-1)
    assert steps.sigma_pA > 2.0 * steps.noise_pA, (
        "the synthetic must reproduce the over-read, or this test proves nothing")
    would_be = steps.gating_score * steps.noise_pA / steps.sigma_pA
    assert would_be < steps.gating_score


def test_a_stepping_pore_gets_a_level_dwell_and_a_busy_one_does_not():
    steps = P.probe_timescales(gating(level_s=0.035), FS, polarity=-1)
    assert steps.level_dwell_s is not None
    assert steps.level_dwell_s == pytest.approx(0.035, rel=0.6)
    busy = P.probe_timescales(synth(n=1_200_000), FS, polarity=-1)
    assert busy.level_dwell_s is None


def test_the_probe_measures_in_the_frame_the_estimator_will_run_in():
    """Everything the probe measures is ONE-SIDED. Handed the wrong polarity it
    measures the noise tail instead of the events, so the two answers must differ --
    and the enhancement (upward) case must be measured by mirroring, not ignored."""
    x = synth(n=1_200_000, events=((3000, 400),), depth=70.0)
    down = P.probe_timescales(x, FS, polarity=-1)
    up = P.probe_timescales(-x, FS, polarity=+1)
    assert down.longest_state_s == pytest.approx(up.longest_state_s, rel=0.25)
    wrong = P.probe_timescales(x, FS, polarity=+1)
    assert wrong.mask_fraction < 0.5 * down.mask_fraction


def test_polarity_is_derived_on_a_detrended_trace_not_from_the_global_median():
    """`derive_polarity` centres on the GLOBAL MEDIAN and returns +1 on the owner's own
    drifting raw traces -- it counts the drift, not the events. The probe must go
    through `baseline.measured_polarity`, which subtracts a coarse rolling median
    first, or a drifting recording is probed in the wrong frame entirely."""
    x = synth(n=1_500_000, events=((4000, 300),), depth=60.0, drift=300.0,
              drift_period_s=3.0)
    assert B.derive_polarity(x) == +1, "the fixture must reproduce the trap"
    assert B.measured_polarity(x, FS) == -1
    assert P.probe_timescales(x, FS).polarity == -1


# ===========================================================================
# 4. THE RULE
# ===========================================================================
def test_the_rule_never_proposes_anything_but_the_two_windows():
    """Structural, not stylistic: a probe that could set `mask_sigma` would be a
    SECOND baseline default, and `DEFAULT_BASELINE` would stop being the app's one
    baseline literal."""
    probe = P.probe_timescales(synth(n=1_200_000), FS, polarity=-1)
    cfg = P.propose_baseline(probe).config
    d = B.DEFAULT_BASELINE
    assert (cfg.method, cfg.mask_sigma, cfg.dilate_ms, cfg.window_s,
            cfg.despike_size) == (d.method, d.mask_sigma, d.dilate_ms, d.window_s,
                                  d.despike_size)
    assert cfg.decim is None, "decim is DERIVED at run time, never proposed as a knob"


def test_the_rule_keeps_the_rough_to_base_ratio_the_spec_requires():
    for probe in (P.probe_timescales(synth(n=1_200_000), FS, polarity=-1),
                  P.probe_timescales(gating(), FS, polarity=-1)):
        cfg = P.propose_baseline(probe).config
        assert 2.5 <= cfg.rough_s / cfg.base_s <= 5.5, (cfg.rough_s, cfg.base_s)


def test_the_gating_branch_sizes_the_base_window_under_the_level_dwell():
    """On a stepping pore the binding constraint is that the baseline can FOLLOW a
    conductance step, so base_s must sit below the level dwell -- the opposite of the
    dwell branch, where rough_s is sized ABOVE the longest state."""
    probe = P.probe_timescales(gating(level_s=0.035), FS, polarity=-1)
    cfg = P.propose_baseline(probe).config
    assert cfg.base_s < probe.level_dwell_s
    assert "level" in P.propose_baseline(probe).reason.lower()


def test_the_dwell_branch_sizes_the_rough_window_over_the_longest_state():
    probe = P.probe_timescales(
        synth(n=1_500_000, long_state=(700_000, 25_000, 60.0)), FS, polarity=-1)
    cfg = P.propose_baseline(probe).config
    assert cfg.rough_s >= 2.0 * probe.longest_state_s, (cfg.rough_s,
                                                        probe.longest_state_s)


def test_a_stepping_pore_whose_steps_cannot_be_resolved_still_gets_a_real_answer():
    """A partial measurement is not a failed one. The dwell branch is still a better
    answer than the global default, and the reason line must SAY which branch ran."""
    probe = P.probe_timescales(synth(n=1_200_000), FS, polarity=-1)
    stepped = type(probe)(**{**probe.__dict__, "gating": True, "level_dwell_s": None})
    proposal = P.propose_baseline(stepped)
    assert proposal.confident
    assert proposal.config is not B.DEFAULT_BASELINE
    assert "could not be measured" in proposal.reason


def test_the_proposal_carries_its_own_param_source_into_params():
    probe = P.probe_timescales(synth(n=1_200_000), FS, polarity=-1)
    proposal = P.propose_baseline(probe)
    params = proposal.params()
    cfg, source = B.baseline_config_from_params(params)
    assert source == "proposed"
    assert cfg.rough_s == proposal.config.rough_s
    # exactly the PARAMETER keys -- ``baseline_fingerprint`` is derived from the
    # parameters and from fs, and a derived value in a params dict is one free to
    # disagree with the parameters beside it (see the GUI file's regression test,
    # where letting it through raised on every detection run).
    assert set(params) == set(B.BASELINE_PARAM_KEYS) | {"baseline_param_source"}


def test_the_summary_and_the_reason_come_from_one_object():
    """The numbers the user reads and the numbers the estimator runs must be produced
    by one function from one value -- a dialog that formatted its own would be free to
    show something the config does not say."""
    probe = P.probe_timescales(synth(n=1_200_000), FS, polarity=-1)
    proposal = P.propose_baseline(probe)
    rows = dict(P.proposal_summary(proposal, FS))
    assert rows["Proposed rough window"] == f"{proposal.config.rough_s:g} s"
    assert rows["Proposed base window"] == f"{proposal.config.base_s * 1e3:g} ms"
    assert rows["Decimation (derived)"] == str(B.derive_decim(proposal.config.base_s, FS))


# ===========================================================================
# 5. WHY IT MAY NOT AUTO-APPLY
# ===========================================================================
def test_the_qc_triple_prefers_the_degenerate_parameters():
    """2.6(b), asserted as the reason the app must ASK.

    A degenerate ``rough_s=0.01 / base_s=0.002 / decim=1`` set scores BETTER than a
    sane one on open-pore level, drift and noise -- shrinking the window monotonically
    improves all three. Optimising them selects the shortest window there is, which is
    precisely the failure the masked estimator exists to fix. This test exists so that
    anyone who later adds an auto-tune has to delete an assertion that says why not.
    """
    x = np.asarray(synth(n=800_000, events=((3000, 400),), depth=70.0), np.float64)
    sane = B.characterisation_baseline(x, FS, polarity=-1, cache=False)
    degen_cfg = B.DEFAULT_BASELINE.replace(rough_s=0.01, base_s=0.002, decim=1)
    degen = B.characterisation_baseline(x, FS, config=degen_cfg, polarity=-1,
                                        cache=False)
    assert abs(degen.health.open_pore_level_pA) <= abs(sane.health.open_pore_level_pA)
    assert degen.health.open_pore_drift_pA <= sane.health.open_pore_drift_pA
    assert degen.health.noise_pA <= sane.health.noise_pA


# ===========================================================================
# 6. COST
# ===========================================================================
def test_the_probe_stays_inside_its_budget_on_a_three_million_sample_recording():
    """2.6: "a ~200 ms probe". Measured on the owner's own files: 106 ms at 1.2 M,
    167 at 2.3 M, 233 at 3.0 M, 579 at 8.6 M -- about 70 ms per million samples, so
    the spec's budget holds at ~3 M and a 172 s recording costs half a second. The
    budget here is generous because CI machines are not this one; what it is really
    guarding is an accidental order-of-magnitude regression (a probe that forgot to
    decimate, or one that ran the health block)."""
    x = synth(n=3_000_000)
    P.probe_timescales(x[:100_000], FS, polarity=-1)     # warm the kernels
    t0 = time.perf_counter()
    probe = P.probe_timescales(x, FS, polarity=-1)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"{elapsed:.3f} s for 3 M samples"
    assert probe.elapsed_s == pytest.approx(elapsed, rel=0.5)


# ===========================================================================
# 7. THE ACCEPTANCE TEST -- the four known-good sets, from the data alone
# ===========================================================================
def _corpus(tag: str) -> np.ndarray:
    from nanoscope.analysis.ml_preprocess import notch_sequential
    df = pd.read_csv(CSV / f"{tag}.csv.gz", usecols=["current_raw_pA"])
    raw = np.asarray(df["current_raw_pA"], np.float32)
    del df
    return notch_sequential(raw, FS)


def _apo() -> np.ndarray:
    from nanoscope.analysis.ml_preprocess import notch_sequential
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document
    doc = open_document(str(APO))
    # THREE channels on this file (ch0 all-zero, ch1 the command voltage x10, ch2 the
    # current) against two in the ferritin corpus. Never hard-code the index.
    cur = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT]
    return notch_sequential(np.asarray(doc.channels[cur[0].index], np.float32), FS)


@pytest.mark.corpus
@pytest.mark.parametrize("tag", ["ferritin_data2", "ferritin_data3", "apo_0006"])
def test_the_rule_reproduces_the_known_good_parameter_sets(tag):
    """THE ACCEPTANCE TEST. From the recording alone, with no metadata and no
    notebook, the rule must land on the parameters the owner arrived at by hand.

    Measured:

    =============== ========= ======== ======= ========== ==========
    recording       longest   gating   level   proposed   shipped
    =============== ========= ======== ======= ========== ==========
    ferritin_data2  163 ms    no 4.7   --      0.5 / 0.1  0.5 / 0.1
    ferritin_data3  (gating)  yes 13.8 28.5 ms 0.03/0.01  0.03/0.01
    apo ``_0006_``  496 ms    no 0.5   --      2.0 / 0.5  2.0 / 0.5
    =============== ========= ======== ======= ========== ==========
    """
    if tag == "apo_0006":
        if not APO.exists():
            pytest.skip("the apo recording is not on this machine")
        x = _apo()
    else:
        if not (CSV / f"{tag}.csv.gz").exists():
            pytest.skip("the ferritin corpus is not on this machine")
        x = _corpus(tag)

    proposal = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    want_rough, want_base = KNOWN_GOOD[tag]
    assert proposal.confident, proposal.reason
    assert proposal.config.rough_s == pytest.approx(want_rough)
    assert proposal.config.base_s == pytest.approx(want_base)
    # ...and the branch it took is the physically right one for that recording.
    assert proposal.probe.gating is (tag == "ferritin_data3")


@pytest.mark.corpus
@pytest.mark.parametrize("tag", ["ferritin_data2", "ferritin_data3", "apo_0006"])
def test_the_rule_still_lands_when_the_probe_MEASURES_the_polarity(tag):
    """The same acceptance test with ``polarity=None`` -- the path the app takes when
    the detection form has not settled on a direction.

    Everything the probe measures is ONE-SIDED, so a wrong direction is not a small
    error: handed ``+1``, the apo recording's decisive 496 ms clog is measured as
    27.8 ms and the rule proposes ``rough_s = 0.1`` instead of 2.0 -- twenty times
    under-sized, and ``confident=True`` either way, which is the exact failure the
    probe exists to prevent. ``ferritin_data2``'s longest state goes 163 -> 187 ms and
    ``ferritin_data3``'s 264 -> 14 ms.

    The probe used to take its direction from `baseline.measured_polarity` at the
    app's DEFAULT window while running its own baseline at a 5 s one -- two frames,
    and on ``ferritin_data3`` the 5 s frame answers ``+1`` because nothing in it
    crosses 5 sigma at all. The direction no longer depends on either window.

    HONEST LABEL: this is a GUARD, not the regression test. It passes against a
    reverted tree too, because the app's default window happens to answer -1 on these
    three recordings -- which is the whole point of the defect, that it was luck. The
    regression test is
    ``test_masked_baseline.py::test_the_door_reads_the_corpus_polarity_from_the_data_at_every_configuration``,
    which fails on ``ferritin_data3`` against the revert with ``{1, -1}``.
    """
    if tag == "apo_0006":
        if not APO.exists():
            pytest.skip("the apo recording is not on this machine")
        x = _apo()
    else:
        if not (CSV / f"{tag}.csv.gz").exists():
            pytest.skip("the ferritin corpus is not on this machine")
        x = _corpus(tag)

    probe = P.probe_timescales(x, FS)           # polarity MEASURED, not supplied
    assert probe.polarity == -1
    proposal = P.propose_baseline(probe)
    want_rough, want_base = KNOWN_GOOD[tag]
    assert proposal.confident, proposal.reason
    assert proposal.config.rough_s == pytest.approx(want_rough)
    assert proposal.config.base_s == pytest.approx(want_base)
    assert proposal.probe.gating is (tag == "ferritin_data3")


@pytest.mark.corpus
def test_ferritin_data1_is_the_honest_partial():
    """The fourth set, stated truthfully rather than claimed.

    ``ferritin_data1``'s shipped estimator is the SINGLE-PASS ``robust_median`` at a
    0.5 s window. The rule's rough window lands on the same 0.5 s, but a
    ``robust_median`` window and a two-pass ``rough_s`` are not the same parameter,
    and NOTHING the probe measures chooses between the one- and two-pass estimator.
    The spec gives no rule for that choice either and leaves this row unticked.
    """
    if not (CSV / "ferritin_data1.csv.gz").exists():
        pytest.skip("the ferritin corpus is not on this machine")
    proposal = P.propose_baseline(
        P.probe_timescales(_corpus("ferritin_data1"), FS, polarity=-1))
    assert proposal.config.rough_s == pytest.approx(0.5)
    assert proposal.config.method == "masked_median", (
        "the probe proposes the two-pass estimator; the notebook used the single "
        "pass on this file, and that difference is not measured anywhere")


@pytest.mark.corpus
def test_the_closing_gap_and_the_rough_multiple_are_a_matched_pair():
    """Change one and the other stops reproducing the known-good sets.

    Measured on the apo file (p99.9 of the closed dwells, ms):
    gap 0 -> 24.6, 0.4 -> 339, 1 -> 497, 2 -> 500.  On ferritin_data2:
    gap 0 -> 101, 0.4 -> 147, 1 -> 168, 2 -> 227, 4 -> 258.  Only at 1 ms is the apo
    clog whole AND data2 still under the 0.707 s grid boundary that keeps it at 0.5.
    """
    if not APO.exists() or not (CSV / "ferritin_data2.csv.gz").exists():
        pytest.skip("the owner's recordings are not on this machine")
    from nanoscope.analysis.baseline_probe import _closed_dwells

    apo_probe = P.probe_timescales(_apo(), FS, polarity=-1)
    assert apo_probe.longest_state_s == pytest.approx(0.496, abs=0.03)
    assert P.snap_to_grid(P.ROUGH_MULTIPLE * apo_probe.longest_state_s) == 2.0

    d2 = P.probe_timescales(_corpus("ferritin_data2"), FS, polarity=-1)
    assert d2.longest_state_s == pytest.approx(0.163, abs=0.02)
    assert P.snap_to_grid(P.ROUGH_MULTIPLE * d2.longest_state_s) == 0.5
    # the boundary that a bigger gap (or a bigger multiple) would cross
    assert P.ROUGH_MULTIPLE * d2.longest_state_s < np.sqrt(0.5 * 1.0)
    assert _closed_dwells is not None


# ===========================================================================
# 8. THE STRUCTURAL GUARD
# ===========================================================================
_SRC = Path(B.__file__).resolve().parent


def test_the_probe_module_constructs_no_baseline_config():
    """The seam's guard (``tests/analysis/test_baseline_seam.py``) bans a
    `BaselineConfig` literal outside ``baseline.py``, and the probe is no exception --
    it is a resolver, not a second owner of the default. Every config it produces is
    ``DEFAULT_BASELINE.replace(...)``, which is what makes
    :func:`test_the_rule_never_proposes_anything_but_the_two_windows` true by
    construction rather than by convention."""
    tree = ast.parse((_SRC / "baseline_probe.py").read_text())
    calls = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) == "BaselineConfig"
                  or getattr(n.func, "attr", None) == "BaselineConfig")]
    assert not calls, f"BaselineConfig constructed at lines {calls}"


def test_the_probe_module_only_ever_replaces_the_three_fields_it_owns():
    """A ``replace(mask_sigma=...)`` would slip past the guard above while being the
    same defect. Checked on the KEYWORD NAMES, so it cannot be spelled around."""
    tree = ast.parse((_SRC / "baseline_probe.py").read_text())
    allowed = {"rough_s", "base_s", "decim"}
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "replace"):
            continue
        for kw in node.keywords:
            if kw.arg not in allowed:
                bad.append(f"line {node.lineno}: {kw.arg}")
    assert not bad, (f"the probe sets {bad}; it owns rough_s/base_s/decim and nothing "
                     "else -- everything else stays at DEFAULT_BASELINE's value")


def test_the_probe_reaches_the_estimator_only_through_the_door():
    """No module outside ``baseline.py`` may call the raw estimator. The probe runs a
    baseline of its own and is the most tempting place to shortcut."""
    tree = ast.parse((_SRC / "baseline_probe.py").read_text())
    banned = {"estimate_baseline", "baseline_masked_median", "baseline_robust_median",
              "_kernel_baseline"}
    hits = [f"{getattr(n.func, 'id', None) or getattr(n.func, 'attr', None)}"
            f"@{n.lineno}" for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) in banned
                 or getattr(n.func, "attr", None) in banned)]
    assert not hits, f"the probe must go through characterisation_baseline; got {hits}"


def test_the_upper_mad_has_exactly_one_definition():
    """`_rough_pass` (the health block) and the probe both need it, and two copies of
    a robust sigma is two places for a 7x error to hide."""
    x = np.asarray(synth(n=200_000), np.float64)
    rough = x - B.decimated_median(x, FS, 0.5, 50)
    _r, _m, _s, upper = B._rough_pass(x, FS, config=B.DEFAULT_BASELINE.replace(decim=50))
    assert upper == B.upper_mad_sigma(rough)
    src = (_SRC / "baseline.py").read_text()
    assert src.count("def upper_mad_sigma") == 1
    assert src.count("1.4826 * np.median(upper") == 1
