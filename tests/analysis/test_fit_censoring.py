"""What the distribution fits are allowed to stay quiet about.

Three bugs live here, and each one puts a wrong number in a paper without ever
raising an error or looking wrong on the plot:

* **B4a — right truncation.** ``detect_events(max_dwell_s=...)`` de-accepts every
  event above the limit, which truncates the dwell sample on the RIGHT. The
  censored MLE modelled only the left truncation, so tau came back short
  (measured -24% at ``t_max = 2.5 tau``) while ``summary()`` still printed
  "hard left-truncation ... (eta == 1 above it)" — a false statement about the
  data. Curation that rejects long events does the same thing without any
  parameter to read.
* **B4b — the depth floor.** The Schmitt trigger defines an event as a run of
  ``dev > k_out*noise`` and ``depth_pA`` is the MEDIAN in-event deviation, so the
  depth sample is truncated at ~``thr_out``. The Gaussian mixture is untruncated,
  so a population sitting on that floor comes back with mu biased high and sigma
  biased low, with an empty warnings list.
* **B4c — filter-limited events.** Events whose dwell is within ~2x the filter
  rise time have attenuated depths and long-biased dwells. They were fed to both
  fits with no exclusion and no warning.

The other half of every test here is the invariant that the FIX must not break:
the untruncated, unguarded paths — the k=1 closed form ``mean(t) - t_min``, its
exact cross-check and its vectorised bootstrap — must stay numerically identical,
so no result anyone already has silently moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, empty_df
from nanoscope.analysis.modules import (
    depth_floor_from_provenance,
    dwell_fit_inputs,
    fit_depth,
    fit_dwell,
    resolve_t_min,
)
from nanoscope.analysis.stats import fit_depth_gaussians, fit_dwell_exponential

FS = 250_000.0
TAU = 2e-4
T_MIN = 2e-5


# ================================================================================================
# builders
# ================================================================================================

def _exp_dwells(n: int, *, tau: float = TAU, t_min: float = T_MIN, seed: int = 0) -> np.ndarray:
    """A left-truncated exponential sample: the untruncated tau IS ``tau``."""
    return t_min + np.random.default_rng(seed).exponential(tau, n)


def _events_frame(dwell: np.ndarray, *, frac_depth=None, filter_limited=None,
                  accepted=None) -> pd.DataFrame:
    n = dwell.size
    df = empty_df().reindex(range(n)).reset_index(drop=True)
    df["dwell_s"] = dwell
    df["event_id"] = np.arange(n, dtype=np.int64)
    df["frac_depth"] = 0.2 if frac_depth is None else frac_depth
    df["filter_limited"] = (np.zeros(n, dtype=bool) if filter_limited is None
                            else np.asarray(filter_limited, dtype=bool))
    df["accepted"] = (np.ones(n, dtype=bool) if accepted is None
                      else np.asarray(accepted, dtype=bool))
    df["flag"] = ""
    df["recording"] = "synthetic"
    df["pore_id"] = "p1"
    return df


def _planted_trace(*, noise: float, depth_mu: float, depth_sd: float, seed: int,
                   seconds: float = 20.0, tau: float = 3e-4):
    """A real current trace with planted blockades, for driving the REAL detector."""
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    x = np.full(n, 1000.0) + rng.normal(0.0, noise, n)
    pos, planted = 2000, 0
    while True:
        d = max(6, int(rng.exponential(tau) * FS))
        if pos + d + 3000 >= n:
            break
        x[pos:pos + d] -= abs(rng.normal(depth_mu, depth_sd))
        planted += 1
        pos += d + int(rng.integers(2000, 6000))
    return np.ascontiguousarray(x, dtype=np.float32), planted


# ================================================================================================
# the invariant: the untruncated path must not move
# ================================================================================================

def test_untruncated_k1_is_still_the_exact_closed_form():
    t = _exp_dwells(4000, seed=1)
    fit = fit_dwell_exponential(t, t_min=T_MIN, n_boot=200)

    # mean(t) - t_min is the exact left-truncated MLE, and the fit must BE it.
    assert fit.params["tau_1"] == pytest.approx(float(np.mean(t)) - T_MIN, rel=0, abs=0)
    assert fit.meta["closed_form"] is True
    assert fit.meta["t_max"] is None
    assert fit.meta["right_truncated"] is False
    assert fit.model == "exponential (left-truncated MLE)"
    assert fit.meta["censoring"] == f"hard left-truncation at t_min={T_MIN:g} s (eta == 1 above it)"


def test_untruncated_bootstrap_is_bit_for_bit_reproducible():
    """The vectorised closed-form bootstrap is the calibrated one; same seed, same CI."""
    t = _exp_dwells(600, seed=2)
    a = fit_dwell_exponential(t, t_min=T_MIN, n_boot=500, seed=7)
    b = fit_dwell_exponential(t, t_min=T_MIN, n_boot=500, seed=7)
    assert a.ci["tau_1"] == b.ci["tau_1"]
    assert a.meta["n_boot_ok"] == 500


def test_depth_fit_without_a_floor_is_unchanged_and_silent():
    x = np.abs(np.random.default_rng(3).normal(0.25, 0.02, 500))
    plain = fit_depth_gaussians(x, n_boot=50)
    assert plain.meta["warnings"] == []
    assert plain.meta["depth_floor"] is None
    assert plain.meta["floor_biased"] is False
    # A floor far below the data changes nothing at all.
    with_floor = fit_depth_gaussians(x, depth_floor=1e-4, n_boot=50)
    assert with_floor.params == plain.params
    assert with_floor.meta["warnings"] == []


# ================================================================================================
# B4a — right truncation
# ================================================================================================

def test_right_truncation_recovers_tau_that_the_left_only_fit_loses():
    """The measured bug: t_max at 2.5 tau costs 24% of tau if it is not modelled."""
    t = _exp_dwells(200_000, seed=4)
    t_max = 2.5 * TAU
    t = t[t <= t_max]
    assert t.size > 1000

    naive = fit_dwell_exponential(t, t_min=T_MIN, n_boot=0)
    modelled = fit_dwell_exponential(t, t_min=T_MIN, t_max=t_max, n_boot=0)

    assert naive.params["tau_1"] < 0.85 * TAU          # the documented -24%
    assert modelled.params["tau_1"] == pytest.approx(TAU, rel=0.05)


def test_right_truncated_summary_stops_claiming_left_truncation_only():
    t = _exp_dwells(5000, seed=5)
    t_max = 3.0 * TAU
    t = t[t <= t_max]

    fit = fit_dwell_exponential(t, t_min=T_MIN, t_max=t_max, n_boot=0)
    text = fit.summary()

    assert "eta == 1 above it" not in text
    assert "DOUBLE truncation" in text
    assert "NOT left-truncation only" in text
    assert "right truncation at" in text and "MODELLED" in text
    assert fit.meta["right_truncated"] is True
    assert fit.model == "exponential (doubly-truncated MLE)"
    # ...and the density is zero outside the observable window it just declared.
    assert float(fit.predict(np.array([t_max * 1.5]))[0]) == 0.0


def test_right_truncation_rejects_events_above_the_stated_limit():
    t = _exp_dwells(500, seed=6)
    with pytest.raises(ValueError, match="longer than the stated upper limit"):
        fit_dwell_exponential(t, t_min=T_MIN, t_max=float(t.max()) * 0.5, n_boot=0)


def test_fit_dwell_models_max_dwell_s_from_provenance():
    """The GUI default is max_dwell_s = 0 = off; when it IS set, the fit must know."""
    x, _ = _planted_trace(noise=8.0, depth_mu=60.0, depth_sd=15.0, seed=7)
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, max_dwell_s=6e-4, k_in=5, k_out=2)
    events = EventSet(df, prov)

    res = fit_dwell(events, doc_id="d", channel=0, n_boot=0)
    fit = res.payload

    assert fit.meta["t_max"] == pytest.approx(6e-4)
    assert fit.meta["right_truncated"] is True
    assert "DOUBLE truncation" in fit.meta["censoring"]
    assert any("right truncation at" in w for w in fit.meta["warnings"])
    assert res.warnings, "the warning must reach the AnalysisResult the controller stores"

    # The correction MOVES the number: same events, no declared limit -> shorter tau.
    blind = fit_dwell(EventSet(df, {k: v for k, v in prov.items() if k != "max_dwell_s"}),
                      doc_id="d", channel=0, n_boot=0)
    assert blind.payload.meta["t_max"] is None
    assert blind.payload.params["tau_1"] < fit.params["tau_1"]


def test_curation_that_rejects_long_events_is_detected_from_the_data():
    """No max_dwell_s anywhere — the evidence is that the rejects are all long ones."""
    dwell = _exp_dwells(400, seed=8)
    order = np.argsort(dwell)
    accepted = np.ones(dwell.size, dtype=bool)
    accepted[order[-25:]] = False            # the user hand-rejected the 25 longest
    df = _events_frame(dwell, accepted=accepted)

    fit = fit_dwell((df, {"min_dwell_s": T_MIN, "fs": FS}),
                    doc_id="d", channel=0, n_boot=0).payload

    note = next(w for w in fit.meta["warnings"] if w.startswith("RIGHT CENSORING"))
    assert "25 rejected event(s)" in note
    assert "biased SHORT" in note
    # summary() must stop asserting left-truncation-only.
    assert "right-censored by curation" in fit.meta["censoring"]
    assert "right-censored by curation" in fit.summary()
    # ...but the likelihood is NOT silently changed on a guessed cut point.
    assert fit.meta["t_max"] is None


def test_no_right_censoring_warning_when_the_rejects_are_not_the_long_ones():
    dwell = _exp_dwells(400, seed=9)
    order = np.argsort(dwell)
    accepted = np.ones(dwell.size, dtype=bool)
    accepted[order[:25]] = False             # the SHORT ones were rejected
    df = _events_frame(dwell, accepted=accepted)

    fit = fit_dwell((df, {"min_dwell_s": T_MIN, "fs": FS}),
                    doc_id="d", channel=0, n_boot=0).payload
    assert not any(w.startswith("RIGHT CENSORING") for w in fit.meta["warnings"])
    assert "right-censored" not in fit.meta["censoring"]


# ================================================================================================
# B4b — the depth floor
# ================================================================================================

def test_depth_floor_warning_fires_on_a_floor_truncated_detector_sample():
    """Planted mu=50 sigma=20 pA against a 10 pA noise floor: thr_out cuts the low tail."""
    x, planted = _planted_trace(noise=10.0, depth_mu=50.0, depth_sd=20.0, seed=10)
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, k_in=5, k_out=2)
    events = EventSet(df, prov)

    fit = fit_depth(events, doc_id="d", channel=0, n_boot=50).payload

    # The floor divides by the same baseline ``frac_depth`` did -- the events' OWN
    # ``baseline_pA``, not ``provenance['baseline_median_pA']`` (N4).
    floor = depth_floor_from_provenance(prov, baseline_pA=df["baseline_pA"])
    assert floor == pytest.approx(
        prov["threshold_out_pA"] / df["baseline_pA"].abs().max()
    )
    assert fit.meta["depth_floor"] == pytest.approx(floor)
    assert fit.meta["floor_biased"] is True
    note = fit.meta["warnings"][0]
    assert note.startswith("DEPTH FLOOR")
    assert "biased HIGH" in note and "biased LOW" in note
    assert "DEPTH FLOOR" in fit.summary()
    # The bias itself: sigma_1 is pulled well below the planted 0.02.
    assert fit.params["sigma_1"] < 0.9 * (20.0 / 1000.0)


def test_depth_floor_stays_silent_on_a_clean_sample():
    x, _ = _planted_trace(noise=3.0, depth_mu=200.0, depth_sd=15.0, seed=11)
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, k_in=5, k_out=2)

    fit = fit_depth(EventSet(df, prov), doc_id="d", channel=0, n_boot=50).payload

    assert fit.meta["depth_floor"] is not None, "the floor must still be recorded"
    assert fit.meta["floor_biased"] is False
    assert fit.meta["warnings"] == []
    assert "DEPTH FLOOR" not in fit.summary()


def test_depth_floor_is_thr_out_not_thr_in():
    """The trigger holds an event while dev > k_out*noise; the floor is the EXIT one."""
    prov = {"threshold_out_pA": 20.0, "threshold_in_pA": 50.0, "baseline_median_pA": -1000.0}
    assert depth_floor_from_provenance(
        prov, baseline_pA=[-1000.0, -1000.0]) == pytest.approx(0.02)
    assert depth_floor_from_provenance(prov, column="depth_pA") == pytest.approx(20.0)
    assert depth_floor_from_provenance({}, column="frac_depth") is None


# ================================================================================================
# N4 — the frac_depth floor must use the SAME denominator frac_depth does.
#
# ``frac_depth = depth_pA / |baseline_pA|`` with each event's OWN local open-pore
# level. The floor divided ``threshold_out_pA`` by ``provenance['baseline_median_pA']``
# -- the median of the baseline TRACK over the whole recording -- instead. Whenever
# the open-pore level is not constant those are different numbers, and on the user's
# -600 mV recording they are wildly different: global median 34.6 pA against per-event
# levels of 39-199 pA, so the floor came out 3.5x too high and 965 of 1579 detected
# events (61.1%) sat below their own supposed floor. Every one of them had cleared the
# trigger, so that was arithmetically impossible -- and ``DEPTH_FLOOR_WARNING`` fired
# on it (mass_below_floor = 0.42), calling a fit floor-biased on the strength of it.
#
# ``depth_pA`` was right the whole time (0/1579 below ``thr_out``), which is the
# tell: same detector, same events, two answers, so the denominator was the bug.
# ================================================================================================

def _stepped_baseline_trace(*, seed: int = 21, seconds: float = 20.0):
    """A trace whose open pore CHANGES LEVEL, with the events on one side of it.

    This is the shape of the real recording that exposed N4: the baseline track
    spends most of its length at a small |current| and the events all sit where it
    is large, so ``median(baseline_track)`` is nowhere near the level any event was
    actually measured against. A flat-baseline trace (``_planted_trace``) cannot
    show the bug at all -- the two denominators agree there, which is exactly why
    the existing tests were green.

    The step edge itself is a genuine excursion, so it is handed to
    ``exclude_regions`` the way a user would exclude it; ``STEP_EXCLUDE`` is that
    window in seconds.

    Returns ``(signal, planted)``.
    """
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    split = int(n * 0.35)                      # events live in the first 35% only
    x = np.empty(n)
    x[:split] = -1000.0                        # open pore, 1000 pA
    x[split:] = -60.0                          # pore mostly shut for the rest
    x += rng.normal(0.0, 5.0, n)

    pos, planted = 2000, 0
    while True:
        d = max(8, int(rng.exponential(3e-4) * FS))
        if pos + d + 3000 >= split:
            break
        x[pos:pos + d] += abs(rng.normal(150.0, 20.0))  # blockade: toward zero
        planted += 1
        pos += d + int(rng.integers(2000, 6000))
    assert planted >= 25, planted
    return np.ascontiguousarray(x, dtype=np.float32), planted


#: The step in :func:`_stepped_baseline_trace`, as an ``exclude_regions`` window.
STEP_EXCLUDE = [(20.0 * 0.35 - 0.15, 20.0 * 0.35 + 0.15)]


def test_no_detected_event_can_sit_below_the_depth_floor():
    """The invariant the old floor broke: every event cleared ``thr_out``, so no
    event's ``frac_depth`` can be below the floor quoted in its own units."""
    x, _ = _stepped_baseline_trace()
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, k_in=5, k_out=2,
                             exclude_regions=STEP_EXCLUDE)
    assert len(df) >= 20

    floor = depth_floor_from_provenance(prov, baseline_pA=df["baseline_pA"])
    below = int((df["frac_depth"] < floor).sum())
    assert below == 0, (
        f"{below}/{len(df)} events are below the frac_depth floor {floor:.4g} "
        f"they were supposedly detected above"
    )

    # ...and the pA floor, which was never wrong, agrees event for event.
    pa_floor = depth_floor_from_provenance(prov, column="depth_pA")
    assert int((df["depth_pA"] < pa_floor).sum()) == 0
    assert below == int((df["depth_pA"] < pa_floor).sum())


def test_the_depth_floor_divides_by_the_events_own_baseline_not_the_global_median():
    """Names the two denominators and asserts which one is used.

    On this trace the recording-wide median baseline is an order of magnitude below
    the level the events were measured against, so the two answers are far apart and
    only one of them can be the floor of the sample being fit.
    """
    x, _ = _stepped_baseline_trace()
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, k_in=5, k_out=2,
                             exclude_regions=STEP_EXCLUDE)

    global_median = abs(float(prov["baseline_median_pA"]))
    event_levels = df["baseline_pA"].abs()
    assert global_median < 0.5 * float(event_levels.min()), (
        "fixture no longer separates the two denominators"
    )

    got = depth_floor_from_provenance(prov, baseline_pA=df["baseline_pA"])
    assert got == pytest.approx(float(prov["threshold_out_pA"]) / float(event_levels.max()))
    assert got < float(prov["threshold_out_pA"]) / global_median


def test_the_depth_floor_warning_does_not_fire_on_a_clean_stepped_baseline_fit():
    """The user-visible half: no fabricated DEPTH FLOOR warning.

    The planted depths here are ~0.08 dI/I0 against a floor near 0.004, so nothing
    is truncated and the fit must say nothing. The old global-median floor sat ABOVE
    most of the sample and reported a large ``mass_below_floor``.
    """
    x, _ = _stepped_baseline_trace()
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, k_in=5, k_out=2,
                             exclude_regions=STEP_EXCLUDE)
    fit = fit_depth(EventSet(df, prov), doc_id="d", channel=0, n_boot=0).payload

    assert fit.meta["mass_below_floor"] < 0.01, fit.meta["mass_below_floor"]
    assert fit.meta["floor_biased"] is False
    assert not any(w.startswith("DEPTH FLOOR") for w in fit.meta["warnings"])
    assert "DEPTH FLOOR" not in fit.summary()


def test_the_floor_handed_to_the_fit_is_the_one_the_fitted_rows_imply():
    """``depth_fit_inputs`` drops filter-limited rows from the sample; the floor has
    to be derived from the rows that SURVIVE, or it qualifies a different sample
    than the one that was fit."""
    from nanoscope.analysis.modules import depth_fit_inputs

    x, _ = _stepped_baseline_trace()
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, k_in=5, k_out=2,
                             filter_cutoff_hz=8_000.0,
                             exclude_regions=STEP_EXCLUDE)
    assert bool(df["filter_limited"].any()), "fixture must have filter-limited rows"

    inp = depth_fit_inputs(EventSet(df, prov))
    kept = df[df["accepted"] & ~df["filter_limited"]]
    assert len(kept) < len(df)
    assert inp.depth_floor == pytest.approx(
        float(prov["threshold_out_pA"]) / float(kept["baseline_pA"].abs().max())
    )
    assert int((inp.values < inp.depth_floor).sum()) == 0


# ================================================================================================
# B4c — filter-limited events
# ================================================================================================

def _filter_limited_detection(seed: int = 12):
    """A detection with a slow enough low-pass that some events are filter-limited."""
    x, _ = _planted_trace(noise=5.0, depth_mu=60.0, depth_sd=15.0, seed=seed, tau=TAU)
    df, prov = detect_events(x, FS, min_dwell_s=T_MIN, filter_cutoff_hz=10_000.0,
                             k_in=5, k_out=2)
    assert int(df["filter_limited"].sum()) > 10, "fixture must actually produce them"
    return EventSet(df, prov)


def test_filter_limited_events_are_excluded_by_default_and_counted():
    events = _filter_limited_detection()
    n_limited = int(events.df["filter_limited"].sum())

    fit = fit_dwell(events, doc_id="d", channel=0, n_boot=0).payload

    assert fit.meta["n_excluded_filter_limited"] == n_limited
    assert fit.meta["use_filter_limited"] is False
    assert fit.n == len(events.df) - n_limited
    note = next(w for w in fit.meta["warnings"] if "filter-limited" in w)
    assert f"{n_limited} filter-limited event(s) excluded" in note
    assert str(n_limited) in fit.summary()


def test_the_toggle_puts_filter_limited_events_back():
    """The toggle restores the rows, and restoring them is tau-NEUTRAL by derivation.

    This test used to end with ``kept.tau > excluded.tau`` ("a different, longer
    tau, which is the whole point of excluding them"). That ordering is not a
    theorem, and after B4DA-04 tightened the flag the fixture measures the
    opposite sign (kept 0.2462 ms vs excluded 0.2631 ms). The resolution is the
    mathematics, not a flipped inequality:

    DERIVATION. The fitter's estimator is the exact left-truncated exponential
    MLE, ``tau_hat = mean(t) - t_min`` (pinned by
    ``test_untruncated_k1_is_still_the_exact_closed_form``). For an exponential,
    memorylessness gives ``T - c | T > c ~ Exp(tau)`` for EVERY cut ``c``, so

        E[ mean(t | t > c) - c ] = tau            for every c.

    * kept fit:      all rows,     t_min = min_dwell_s          -> estimates tau
    * excluded fit:  rows above c, t_min = c = 2*filter_rise_s  -> estimates tau
      (the floor move is pinned by the sibling test
      ``test_excluding_filter_limited_moves_the_detection_floor_to_2x_rise``)

    Moving ``t_min`` WITH the exclusion is therefore exactly what makes dropping
    the shortest events tau-neutral: both fits estimate the SAME tau, and no
    systematic ordering between them exists in either direction. The difference
    ``tau_excl - tau_kept`` is a mean-zero statistic with standard deviation
    ~ ``sqrt(tau^2/n_kept + tau^2/n_excl)`` (the truncated MLE's own variance),
    plus whatever the flag's misalignment with a pure dwell cut contributes: the
    flag is per-event physics, not a dwell threshold, so in this fixture flagged
    dwells run up to 0.204 ms while unflagged ones start at 0.156 ms, and the
    thinned overlap band moves the excluded mean slightly without the likelihood
    knowing. Measured here: diff = +0.0169 ms = 1.43 combined standard errors --
    a sign the old assertion turned into a contract.

    So the ESTIMATOR is correct and stays; the INEQUALITY was the bug. What is
    actually derivable, and what is asserted instead: each fit is the exact
    closed form of its own sample and floor, the two samples genuinely differ,
    and the two taus agree to within sampling error of the one tau they both
    estimate.
    """
    events = _filter_limited_detection()
    n_limited = int(events.df["filter_limited"].sum())
    assert n_limited > 0

    kept = fit_dwell(events, doc_id="d", channel=0, n_boot=0,
                     use_filter_limited=True).payload

    assert kept.meta["n_excluded_filter_limited"] == 0
    assert kept.meta["use_filter_limited"] is True
    assert kept.n == len(events.df)
    excluded = fit_dwell(events, doc_id="d", channel=0, n_boot=0).payload
    assert excluded.n == len(events.df) - n_limited

    # Each fit is the exact closed-form MLE of ITS OWN sample and floor -- the
    # invariant the old inequality was groping at. ``mean(t) - t_min``, bit-exact.
    dwell_all = events.df["dwell_s"].to_numpy(dtype=float)
    assert kept.meta["t_min"] == pytest.approx(float(events.provenance["min_dwell_s"]))
    assert kept.params["tau_1"] == pytest.approx(
        float(dwell_all.mean()) - float(kept.meta["t_min"]), rel=1e-12)
    inp = dwell_fit_inputs(events)          # the excluded fit's own inputs
    assert excluded.meta["t_min"] == pytest.approx(
        2.0 * float(events.provenance["filter_rise_s"]))
    assert excluded.params["tau_1"] == pytest.approx(
        float(np.mean(inp.dwell)) - float(inp.t_min), rel=1e-12)

    # The toggle really changes the estimate (different sample, different floor)...
    assert kept.params["tau_1"] != excluded.params["tau_1"]
    # ...but NOT systematically: both estimate the same tau, so they must agree to
    # within sampling error. 4 combined standard errors gives ~2.8x margin over the
    # fixture's measured 1.43 -- and a real estimator regression (the -24% bias at
    # t_max = 2.5 tau that opened this file) is an order of magnitude outside it.
    se = np.sqrt(kept.params["tau_1"] ** 2 / kept.n
                 + excluded.params["tau_1"] ** 2 / excluded.n)
    assert abs(kept.params["tau_1"] - excluded.params["tau_1"]) < 4.0 * se


def test_excluding_filter_limited_moves_the_detection_floor_to_2x_rise():
    """Dropping the short events without moving t_min would re-create the B1 bias."""
    events = _filter_limited_detection()
    rise = float(events.provenance["filter_rise_s"])

    inp = dwell_fit_inputs(events)

    assert inp.t_min == pytest.approx(2.0 * rise)
    assert inp.t_min_source == "2*filter_rise_s"
    assert inp.t_min > float(events.provenance["min_dwell_s"])
    assert float(inp.dwell.min()) >= inp.t_min
    assert any("detection floor for this fit is therefore" in n for n in inp.notes)

    # With them included the floor is the detector's own min_dwell_s again.
    kept = dwell_fit_inputs(events, use_filter_limited=True)
    assert kept.t_min == pytest.approx(events.provenance["min_dwell_s"])
    assert kept.t_min_source == "min_dwell_s"


def test_filter_limited_exclusion_applies_to_the_depth_fit_too():
    events = _filter_limited_detection()
    n_limited = int((events.df["filter_limited"].to_numpy(dtype=bool)
                     & events.df["accepted"].to_numpy(dtype=bool)).sum())

    fit = fit_depth(events, doc_id="d", channel=0, n_boot=50).payload

    assert fit.meta["n_excluded_filter_limited"] == n_limited
    assert fit.n == int(events.df["accepted"].sum()) - n_limited
    assert any("attenuated" in w for w in fit.meta["warnings"])


# ================================================================================================
# B1 — one resolution of t_min, shared by every consumer
# ================================================================================================

def test_resolve_t_min_prefers_the_detector_gate_then_fs_then_the_data():
    t_min, source, note = resolve_t_min({"min_dwell_s": 5e-5, "fs": FS}, observed_min=9e-5)
    assert (t_min, source, note) == (5e-5, "min_dwell_s", None)

    t_min, source, note = resolve_t_min({"fs": FS}, observed_min=9e-5)
    assert t_min == pytest.approx(1 / FS) and source == "1/fs" and note

    t_min, source, note = resolve_t_min({}, observed_min=9e-5)
    assert t_min == 9e-5 and source == "observed_min"
    assert "min_dwell_s" in note

    with pytest.raises(ValueError, match="cannot infer t_min"):
        resolve_t_min({})


def test_fit_dwell_uses_the_detector_gate_not_the_shortest_observed_event():
    dwell = _exp_dwells(500, seed=13) + 3e-5     # every event well above the real gate
    df = _events_frame(dwell)

    fit = fit_dwell((df, {"min_dwell_s": T_MIN, "fs": FS}),
                    doc_id="d", channel=0, n_boot=0).payload

    assert fit.meta["t_min"] == T_MIN
    assert fit.meta["t_min_source"] == "min_dwell_s"
    assert fit.params["tau_1"] == pytest.approx(float(dwell.mean()) - T_MIN)
