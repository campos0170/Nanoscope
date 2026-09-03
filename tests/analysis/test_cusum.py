"""Tests for CUSUM sub-state level fitting (analysis/cusum.py).

The anchor assertions: a stepped event resolves into the RIGHT NUMBER of levels with
boundaries on the steps and depths on the same scale as ``events.SCHEMA['depth_pA']``;
and a genuinely one-level event is never split. Everything else here defends a
documented contract -- the sign convention, the noise-scale ladder, the schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.cusum import (
    DEFAULT_CUSUM,
    LEVEL_SCHEMA,
    CusumConfig,
    cusum_changepoints,
    empty_levels,
    fit_event_levels,
    fit_levels_1d,
    levels_summary,
)
from nanoscope.analysis.events import SCHEMA as EVENT_SCHEMA
from nanoscope.analysis.events import empty_df

FS = 50_000.0
SIGMA = 1.0


# ================================================================================================
# Fixtures / helpers
# ================================================================================================
def _staircase(depths, n_each, sigma=SIGMA, seed=0) -> np.ndarray:
    """A deviation-space event: constant `depths`, `n_each` samples each, + noise."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.full(n_each, d, dtype=np.float64) for d in depths])
    return x + rng.normal(0.0, sigma, x.size)


def _event_frame(spans, baseline_pA=100.0, accepted=True) -> pd.DataFrame:
    """A minimal events.SCHEMA frame carrying just what fit_event_levels reads."""
    df = empty_df()
    rows = []
    for i, (a, b) in enumerate(spans):
        row = {c: (0 if d == "int64" else np.nan if d == "float64"
                   else False if d == "bool" else "")
               for c, (d, _, _) in EVENT_SCHEMA.items()}
        row.update(event_id=i, start_idx=a, end_idx=b, baseline_pA=baseline_pA,
                   accepted=accepted, recording="r", pore_id="p")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.astype({c: d for c, (d, _, _) in EVENT_SCHEMA.items()})[list(EVENT_SCHEMA)]


# ================================================================================================
# The core algorithm: does it find the steps, and does it invent any?
# ================================================================================================
def test_two_level_event_splits_on_the_step():
    x = _staircase([20.0, 35.0], 200)
    cps = cusum_changepoints(x, sigma=SIGMA)
    assert cps.size == 1
    assert abs(int(cps[0]) - 200) <= 3, "boundary must land on the step"


def test_three_level_event_finds_both_steps():
    x = _staircase([10.0, 25.0, 15.0], 150)
    cps = cusum_changepoints(x, sigma=SIGMA)
    assert cps.size == 2
    assert abs(int(cps[0]) - 150) <= 3
    assert abs(int(cps[1]) - 300) <= 3


def test_transition_slivers_stay_rare_across_seeds():
    """The failure mode the warm-up and the boundary prune exist to suppress.

    A single seed proves nothing here: the sliver is noise-driven, so it appears in a
    few percent of trials and a one-shot test would pass on a broken build most of the
    time. The rate is 0% now that _drop_unearned_segments tests whole SEGMENTS rather
    than individual boundaries; it was 2% with the per-boundary prune alone and 12% with
    neither. The bound stays loose so this measures the failure mode, not the seed set.
    """
    wrong = sum(cusum_changepoints(_staircase([10.0, 25.0, 15.0], 150, seed=s),
                                   sigma=SIGMA).size != 2
                for s in range(200))
    assert wrong / 200 <= 0.10, f"{wrong}/200 staircases resolved to the wrong level count"


def test_short_levels_keep_their_boundaries():
    """Counting levels is not enough -- the boundaries have to be in the right place.

    Before the warm-up window was capped at a third of the remaining samples, a warm-up
    longer than the level (16 vs 12 samples) gave a perfect 100% COUNT while every
    boundary sat 8 samples off -- deterministically, every trial, because the fit could
    not look before sample 16. A count-only test called that the best configuration
    available. Hence: assert POSITION.
    """
    errs = []
    for s in range(80):
        cps = cusum_changepoints(_staircase([10.0, 25.0, 10.0], 12, seed=s), sigma=SIGMA)
        if cps.size == 2:
            errs.append(max(abs(int(cps[0]) - 12), abs(int(cps[1]) - 24)))
    assert len(errs) >= 60, "a 12-sample level must still be resolvable at all"
    assert float(np.median(errs)) <= 1.0, f"boundaries systematically off: {np.median(errs)}"


def test_the_warmup_window_is_capped_by_the_event_length():
    """The cap is what makes ``warmup_samples`` an upper bound rather than a fixed cost.

    Without it a short event is spent unarmed and the answer follows the constant
    instead of the data: on the owner's apo-ferritin recording (median event 20
    samples) the multilevel fraction moved 30% -> 21% across warm-up values. With the
    cap a warm-up far longer than the event still resolves it.
    """
    x = _staircase([10.0, 30.0], 15, seed=2)          # a 30-sample event, warm-up 12
    huge = CusumConfig(warmup_samples=64)             # far longer than the event itself
    cps = cusum_changepoints(x, sigma=SIGMA, config=huge)
    assert cps.size == 1, "an over-long warm-up must not blind the fit to a real step"
    assert abs(int(cps[0]) - 15) <= 3


def test_a_flat_event_is_one_level():
    assert cusum_changepoints(_staircase([20.0], 400), sigma=SIGMA).size == 0


def test_pure_noise_is_not_split_at_the_default_threshold():
    """The false-split rate the CusumConfig.threshold_sigma docstring claims (0% at h=6).

    Guards the DEFAULT specifically: lowering it is exactly the change that would
    start manufacturing sub-states out of noise, and this is the test that says so.
    """
    rng = np.random.default_rng(7)
    split = sum(cusum_changepoints(rng.normal(0.0, SIGMA, 400), sigma=SIGMA).size > 0
                for _ in range(300))
    assert split == 0, f"{split}/300 pure-noise segments were split into sub-states"


def test_a_step_below_step_sigma_is_not_resolved():
    """step_sigma is a floor with teeth: a 0.5-sigma step must not register at 2.0."""
    x = _staircase([0.0, 0.5], 300, seed=3)
    assert cusum_changepoints(x, sigma=SIGMA).size == 0


def test_detection_delay_is_corrected_not_merely_bounded():
    """The backtracking estimator is the reason boundaries are usable.

    The raw crossing index sits ~h/(step_sigma/2) samples late every time; the median
    error after correction is 0. A regression that dropped the backtrack would still
    "find 2 levels" and would still pass a count-only test, so assert the POSITION.
    """
    errs = []
    for seed in range(40):
        x = _staircase([0.0, 3.0], 200, seed=seed)
        cps = cusum_changepoints(x, sigma=SIGMA)
        if cps.size == 1:
            errs.append(abs(int(cps[0]) - 200))
    assert len(errs) >= 38
    assert float(np.median(errs)) <= 1.0


# ================================================================================================
# Guards: min length, max levels, degenerate input.
# ================================================================================================
def test_min_level_samples_is_respected():
    cfg = CusumConfig(min_level_samples=50)
    # A real step 10 samples from the end cannot open a 50-sample level.
    x = np.concatenate([np.zeros(390), np.full(10, 40.0)])
    starts, stops = fit_levels_1d(x, sigma=SIGMA, config=cfg)
    assert starts.size == 1, "a level shorter than min_level_samples must not be opened"
    lengths = stops - starts
    assert (lengths >= cfg.min_level_samples).all()


def test_max_levels_caps_the_fit():
    cfg = CusumConfig(max_levels=3)
    x = _staircase([0.0, 20.0, 0.0, 20.0, 0.0, 20.0], 100)
    starts, _ = fit_levels_1d(x, sigma=SIGMA, config=cfg)
    assert starts.size <= cfg.max_levels


def test_too_short_and_non_finite_events_yield_exactly_one_level():
    short = np.full(4, 10.0)
    starts, stops = fit_levels_1d(short, sigma=SIGMA)
    assert starts.tolist() == [0] and stops.tolist() == [4]

    poisoned = _staircase([10.0, 30.0], 200)
    poisoned[50] = np.nan
    starts, stops = fit_levels_1d(poisoned, sigma=SIGMA)
    assert starts.tolist() == [0] and stops.tolist() == [poisoned.size], (
        "refusing to fit must look like one level, not like a missing event")


def test_empty_input_yields_no_levels():
    starts, stops = fit_levels_1d(np.empty(0), sigma=SIGMA)
    assert starts.size == 0 and stops.size == 0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_a_non_positive_sigma_is_refused(bad):
    with pytest.raises(ValueError, match="sigma"):
        cusum_changepoints(np.zeros(100), sigma=bad)


@pytest.mark.parametrize("kwargs", [
    {"step_sigma": 0.0}, {"step_sigma": -1.0}, {"threshold_sigma": 0.0},
    {"min_level_samples": 0}, {"max_levels": 0},
])
def test_invalid_config_is_refused_at_construction(kwargs):
    with pytest.raises(ValueError):
        CusumConfig(**kwargs)


def test_fingerprint_changes_with_every_field_that_moves_a_level():
    base = CusumConfig()
    assert base.fingerprint() == CusumConfig().fingerprint()
    for kwargs in ({"step_sigma": 2.5}, {"threshold_sigma": 7.0},
                   {"min_level_samples": 5}, {"max_levels": 8}):
        assert CusumConfig(**kwargs).fingerprint() != base.fingerprint(), kwargs


# ================================================================================================
# The event-set path: schema, sign convention, noise ladder.
# ================================================================================================
def _two_level_recording(polarity: int, baseline: float = 100.0):
    """A trace with ONE two-level event in it, built in absolute-current space."""
    rng = np.random.default_rng(11)
    n = 2_000
    sig = np.full(n, baseline, dtype=np.float64) + rng.normal(0.0, SIGMA, n)
    # dev = polarity*(current - baseline) must be POSITIVE inside the event.
    sig[500:700] += polarity * 20.0
    sig[700:900] += polarity * 35.0
    return sig


@pytest.mark.parametrize("polarity", [1, -1])
def test_level_depths_are_positive_for_both_polarities(polarity):
    """The sign convention is the whole reason depth is comparable across recordings."""
    sig = _two_level_recording(polarity)
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900)]),
                              baseline=100.0, polarity=polarity, sigma=SIGMA)
    assert len(levels) == 2
    assert (levels["depth_pA"] > 0).all(), "depth must be positive whichever way current went"
    assert levels["depth_pA"].iloc[0] == pytest.approx(20.0, abs=1.5)
    assert levels["depth_pA"].iloc[1] == pytest.approx(35.0, abs=1.5)


def test_level_table_matches_LEVEL_SCHEMA_exactly():
    sig = _two_level_recording(1)
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900)]),
                              baseline=100.0, polarity=1, sigma=SIGMA)
    assert list(levels.columns) == list(LEVEL_SCHEMA)
    for col, (dtype, _, _) in LEVEL_SCHEMA.items():
        assert levels[col].dtype == np.dtype(dtype) if dtype != "string" else True
    assert list(empty_levels().columns) == list(LEVEL_SCHEMA)


def test_derived_columns_are_consistent_with_the_level_geometry():
    sig = _two_level_recording(1)
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900)]),
                              baseline=100.0, polarity=1, sigma=SIGMA)
    # Levels tile the event exactly, with no gap and no overlap.
    assert int(levels["start_idx"].iloc[0]) == 500
    assert int(levels["end_idx"].iloc[-1]) == 900
    assert int(levels["end_idx"].iloc[0]) == int(levels["start_idx"].iloc[1])
    # dwell/t_start are on the recording clock, as events.SCHEMA's are.
    widths = (levels["end_idx"] - levels["start_idx"]) / FS
    assert levels["dwell_s"].to_numpy() == pytest.approx(widths.to_numpy())
    assert levels["t_start_s"].iloc[0] == pytest.approx(500 / FS)
    # frac_depth against the parent event's baseline; step_pA undefined on the first.
    assert np.isnan(levels["step_pA"].iloc[0])
    assert levels["step_pA"].iloc[1] == pytest.approx(
        levels["depth_pA"].iloc[1] - levels["depth_pA"].iloc[0])
    assert levels["frac_depth"].iloc[0] == pytest.approx(levels["depth_pA"].iloc[0] / 100.0)
    assert (levels["n_levels"] == 2).all()
    assert levels["level_idx"].tolist() == [0, 1]


def test_level_pA_is_absolute_current_not_deviation():
    """level_pA and depth_pA answer different questions; conflating them is silent."""
    sig = _two_level_recording(-1, baseline=100.0)   # blockades go DOWN from 100 pA
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900)]),
                              baseline=100.0, polarity=-1, sigma=SIGMA)
    assert (levels["level_pA"] < 100.0).all(), "absolute current sits below the baseline"
    assert (levels["depth_pA"] > 0).all()


def test_rejected_events_are_excluded_by_default_and_includable():
    sig = _two_level_recording(1)
    ev = _event_frame([(500, 900)], accepted=False)
    assert fit_event_levels(sig, FS, ev, baseline=100.0, polarity=1, sigma=SIGMA).empty
    kept = fit_event_levels(sig, FS, ev, baseline=100.0, polarity=1, sigma=SIGMA,
                            accepted_only=False)
    assert len(kept) == 2


def test_sigma_comes_from_detection_provenance_when_not_given():
    """Same ladder span_noise_rms uses: the table's own noise_rms_pA wins.

    A provenance sigma large enough to swamp the 15 pA step must SUPPRESS the split --
    which is the observable proof the number was actually used.
    """
    sig = _two_level_recording(1)
    ev = _event_frame([(500, 900)])
    levels = fit_event_levels(sig, FS, ev, baseline=100.0, polarity=1,
                              provenance={"noise_rms_pA": 200.0})
    assert (levels["n_levels"] == 1).all()
    # ...and an honest sigma from the same door recovers the two levels.
    levels = fit_event_levels(sig, FS, ev, baseline=100.0, polarity=1,
                              provenance={"noise_rms_pA": SIGMA})
    assert (levels["n_levels"] == 2).all()


def test_an_explicit_sigma_beats_provenance():
    sig = _two_level_recording(1)
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900)]), baseline=100.0,
                              polarity=1, sigma=SIGMA,
                              provenance={"noise_rms_pA": 200.0})
    assert (levels["n_levels"] == 2).all()


def test_a_scalar_and_an_array_baseline_agree():
    sig = _two_level_recording(1)
    ev = _event_frame([(500, 900)])
    a = fit_event_levels(sig, FS, ev, baseline=100.0, polarity=1, sigma=SIGMA)
    b = fit_event_levels(sig, FS, ev, baseline=np.full(sig.size, 100.0),
                         polarity=1, sigma=SIGMA)
    pd.testing.assert_frame_equal(a, b)


def test_many_events_are_ordered_and_keyed_by_event_id():
    sig = _two_level_recording(1)
    sig[1_200:1_400] += 25.0            # a second, single-level event
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900), (1_200, 1_400)]),
                              baseline=100.0, polarity=1, sigma=SIGMA)
    assert levels["event_id"].tolist() == [0, 0, 1]
    assert levels["level_idx"].tolist() == [0, 1, 0]
    assert levels["recording"].tolist() == ["r", "r", "r"]


def test_an_empty_event_table_yields_an_empty_level_table():
    out = fit_event_levels(np.zeros(100), FS, empty_df(), baseline=0.0, polarity=1, sigma=1.0)
    assert out.empty and list(out.columns) == list(LEVEL_SCHEMA)


@pytest.mark.parametrize("kwargs,match", [
    ({"fs": 0.0}, "fs"),
    ({"polarity": 0}, "polarity"),
])
def test_fit_event_levels_refuses_impossible_arguments(kwargs, match):
    args = dict(signal=np.zeros(100), fs=FS, events=_event_frame([(10, 90)]),
                baseline=0.0, polarity=1, sigma=1.0)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fit_event_levels(args.pop("signal"), args.pop("fs"), args.pop("events"), **args)


def test_a_mismatched_baseline_length_is_refused():
    with pytest.raises(ValueError, match="baseline"):
        fit_event_levels(np.zeros(100), FS, _event_frame([(10, 90)]),
                         baseline=np.zeros(50), polarity=1, sigma=1.0)


def test_levels_summary_reports_how_much_structure_was_found():
    sig = _two_level_recording(1)
    sig[1_200:1_400] += 25.0
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900), (1_200, 1_400)]),
                              baseline=100.0, polarity=1, sigma=SIGMA)
    s = levels_summary(levels)
    assert s["n_events_fitted"] == 2
    assert s["n_levels_total"] == 3
    assert s["frac_multilevel"] == pytest.approx(0.5)
    assert s["mean_levels_per_event"] == pytest.approx(1.5)
    assert levels_summary(empty_levels())["n_events_fitted"] == 0


# ================================================================================================
# Regressions for the defects an adversarial review of this module confirmed.
# ================================================================================================
def test_a_level_shorter_than_the_warmup_window_keeps_its_true_boundary():
    """The warm-up used to be a blind spot that silently overrode min_level_samples.

    ``_arm`` consumes the first w samples to seed mu and the backtrack anchors start at
    the end of that window, so no boundary could land closer than w to a segment start.
    A 6-sample first level with a 6-sigma step WAS detected (495/500) but reported at 12
    in every trial -- dwell_s 2x too long, depth_pA a 50/50 mixture of two real states,
    and the next level's step_pA half the true step. Four columns wrong at once, in
    exactly the case this feature exists to measure.
    """
    for cut in (3, 4, 6, 10):
        reported = []
        for seed in range(60):
            rng = np.random.default_rng(seed)
            x = np.concatenate([np.zeros(cut), np.full(300 - cut, 6.0)])
            cps = cusum_changepoints(x + rng.normal(0.0, SIGMA, 300), sigma=SIGMA)
            if cps.size == 1:
                reported.append(int(cps[0]))
        assert len(reported) >= 55, f"a {cut}-sample level at 6 sigma must be detected"
        assert abs(float(np.median(reported)) - cut) <= 1.0, (
            f"true cut {cut} reported at {np.median(reported)} — pinned to the warm-up window")


def test_max_levels_of_one_does_not_split():
    """The runaway guard ran AFTER the append, so max_levels=1 emitted 2 levels — and
    CusumConfig explicitly sanctions 1. A stored fit then had params saying "do not
    split" over a payload that had split."""
    x = _staircase([0.0, 10.0, 0.0, 10.0], 100)
    for max_levels in (1, 2, 3, 4):
        starts, _ = fit_levels_1d(x, sigma=SIGMA, config=CusumConfig(max_levels=max_levels))
        assert starts.size <= max_levels, f"max_levels={max_levels} produced {starts.size}"
    assert fit_levels_1d(x, sigma=SIGMA, config=CusumConfig(max_levels=1))[0].size == 1


def test_event_ids_come_from_the_events_not_from_a_row_counter():
    """The old fallback numbered by OUTPUT ROW, so a 2-level event followed by a 1-level
    event came back as ids [0, 1, 2] and one event's second level became its own event."""
    rng = np.random.default_rng(3)
    sig = np.full(2_000, 100.0) + rng.normal(0.0, SIGMA, 2_000)
    sig[500:700] += 20.0
    sig[700:900] += 35.0          # event A: two levels
    sig[1_200:1_400] += 25.0      # event B: one level
    events = _event_frame([(500, 900), (1_200, 1_400)])

    indexed = events.set_index("event_id")          # event_id as the INDEX, not a column
    levels = fit_event_levels(sig, FS, indexed, baseline=100.0, polarity=1, sigma=SIGMA)
    assert levels["event_id"].tolist() == [0, 0, 1]
    assert levels_summary(levels)["n_events_fitted"] == 2


def test_level_times_are_on_the_event_tables_own_clock():
    """characterize writes t_start_s = start_idx/fs with no recording-start term, so a
    t0_s offset here would put every level seconds away from its own event."""
    rng = np.random.default_rng(5)
    sig = np.full(2_000, 100.0) + rng.normal(0.0, SIGMA, 2_000)
    sig[500:900] += 25.0
    levels = fit_event_levels(sig, FS, _event_frame([(500, 900)]),
                              baseline=100.0, polarity=1, sigma=SIGMA)
    assert levels["t_start_s"].iloc[0] == pytest.approx(500 / FS)
    import inspect
    assert "t0_s" not in inspect.signature(fit_event_levels).parameters


def test_events_outside_the_signal_are_refused_not_clipped():
    """Clipping made a mismatched (events, signal) pair look like a successful fit.
    Pairing a table with the wrong trace is a caller bug and must say so."""
    rng = np.random.default_rng(6)
    sig = np.full(800, 100.0) + rng.normal(0.0, SIGMA, 800)
    with pytest.raises(ValueError, match="do not describe the same data"):
        fit_event_levels(sig, FS, _event_frame([(500, 900)]),
                         baseline=100.0, polarity=1, sigma=SIGMA)


# ================================================================================================
# The autocorrelation correction. A z-test that assumes i.i.d. samples on a LOW-PASSED
# trace is the defect that reported 11 levels for a rectangle.
# ================================================================================================
def _lowpassed_noise(n: int, *, fs: float = 50_000.0, cutoff: float = 10_000.0, seed: int = 0):
    from scipy.signal import bessel, sosfiltfilt
    sos = bessel(8, cutoff / (0.5 * fs), output="sos")
    return sosfiltfilt(sos, np.random.default_rng(seed).normal(0.0, 1.0, n))


def test_white_noise_needs_no_correction():
    """tau must be ~1 on independent samples, or the fix would penalise the case that
    was never broken."""
    from nanoscope.analysis.cusum import integrated_autocorr_time
    x = np.random.default_rng(0).normal(0.0, 1.0, 20_000)
    assert integrated_autocorr_time(x, n_scale=20) == pytest.approx(1.0, abs=0.2)


def test_tau_is_not_set_by_its_own_cutoff():
    """THE bug in the first version of this estimator.

    Summed to the first non-positive rho, tau had no stopping point on a real trace --
    the owner's ACF never becomes non-positive (0.977 at lag 1, still 0.479 at lag 40,
    because the residual carries baseline drift), so tau simply tracked its own horizon:
    10, 18, 31, 53, 85, 141 at max_lag 5..200. A number set by its cutoff is not a
    measurement. The Bartlett-weighted form must be stable across the scale instead.
    """
    from nanoscope.analysis.cusum import integrated_autocorr_time
    x = _lowpassed_noise(20_000)
    taus = [integrated_autocorr_time(x, n_scale=ns) for ns in (10, 20, 64, 200)]
    assert min(taus) > 1.5, "a filtered trace IS correlated; tau must exceed 1"
    assert max(taus) / min(taus) < 2.0, f"tau tracks its own horizon: {taus}"


def test_a_flat_filtered_event_is_not_split(  ):
    """The headline defect, at its root: correlated noise with NO step in it.

    Uncorrected, event-length windows of purely low-passed noise were split 96% of the
    time (mean 3.7 spurious levels). With tau supplied they must not be split at all.
    """
    from nanoscope.analysis.cusum import integrated_autocorr_time
    x = _lowpassed_noise(20_000, seed=3)
    tau = integrated_autocorr_time(x, n_scale=200)
    sigma = float(np.std(x))
    split = sum(cusum_changepoints(x[i * 200:(i + 1) * 200], sigma=sigma, tau=tau).size > 0
                for i in range(80))
    assert split == 0, f"{split}/80 flat filtered windows were split into sub-states"


def test_a_real_step_survives_the_correction():
    """The correction must not simply refuse to find anything: a genuine step well above
    the noise still has to be resolved."""
    from nanoscope.analysis.cusum import integrated_autocorr_time
    base = _lowpassed_noise(4_000, seed=5)
    tau = integrated_autocorr_time(base, n_scale=200)
    sigma = float(np.std(base))
    found = 0
    for i in range(40):
        seg = base[i * 100:(i + 1) * 100].copy()
        seg[50:] += 8.0 * sigma                      # an unambiguous step
        cps = cusum_changepoints(seg, sigma=sigma, tau=tau)
        found += (cps.size == 1 and abs(int(cps[0]) - 50) <= 6)
    assert found >= 34, f"only {found}/40 unambiguous steps survived the correction"


def test_three_levels_resolve_at_a_realistically_measured_tau():
    """Deep structure survives the correction when tau is measured on clean filtered
    noise (tau ~5): both boundaries of a 3-level event are still found."""
    from nanoscope.analysis.cusum import integrated_autocorr_time
    base = _lowpassed_noise(6_000, seed=7)
    tau = integrated_autocorr_time(base, n_scale=100)
    sigma = float(np.std(base))
    ok = 0
    for i in range(40):
        seg = base[i * 100:(i + 1) * 100].copy()
        seg[:33] += 11.0 * sigma
        seg[33:66] += 20.0 * sigma
        seg[66:] += 15.0 * sigma
        ok += cusum_changepoints(seg, sigma=sigma, tau=tau).size == 2
    assert ok >= 30, f"only {ok}/40 three-level events resolved at tau={tau:.1f}"


def test_an_overlarge_tau_silently_drops_the_weaker_step():
    """THE KNOWN LIMITATION, pinned as a measured relationship rather than a claim.

    tau is estimated from the whole-trace residual ``signal - baseline``, which still
    carries baseline DRIFT. Drift inflates the ACF at exactly the lags the Bartlett
    weight cannot fully discount, so on a real recording tau comes out larger than the
    within-event noise justifies (16.6 measured, against ~5.7 for the filter alone) and
    the WEAKER of two steps falls under threshold — a genuine 3-level event is reported
    as 2, or as 1.

    This test states the sensitivity directly: same event, same data, only tau varies.
    It is the algorithm behaving correctly given a wrong input, which is why the fix
    belongs in the ESTIMATOR (measure tau on open-pore stretches only, excluding detected
    events) and not in the gates. When that lands, the high-tau branch here should be
    revisited — deep structure at low effective-N is currently UNDER-reported.
    """
    rng = np.random.default_rng(0)
    seg = np.concatenate([np.full(33, 140.0), np.full(33, 240.0), np.full(34, 180.0)])
    seg = seg + rng.normal(0.0, 12.0, seg.size)
    honest = cusum_changepoints(seg, sigma=12.0, tau=5.7)
    inflated = cusum_changepoints(seg, sigma=12.0, tau=16.6)
    assert honest.size == 2, "at a filter-justified tau both steps are found"
    assert inflated.size < 2, "an inflated tau drops the weaker step — the open defect"
