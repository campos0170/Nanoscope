"""Is this pooled voltage series comparable? -- the invariant, in the pure core.

REGRESSION FOR THE apo-run-2026-08-19 FINDINGS **High-1** AND **High-2**.

The owner drove ``Analysis > Voltage dependence`` over three apoferritin recordings
at -500 / -600 / -700 mV and found the one surface built to COMPARE recordings
breaking the rule the rest of the app enforces:

* **High-1.** It offered "Median depth" and "Median dI/I0" and fitted them across all
  three, although the three had been detected under the three different baseline
  window pairs the probe proposed for them -- fingerprints ``sha256:b64a75ebdf29``,
  ``sha256:ce8ae31f10bd`` and ``sha256:1a29aabf5fb1``. Neither
  ``nanoscope/analysis/voltage_dep.py`` nor ``nanoscope/gui/voltage_dep_panel.py``
  contained the string ``fingerprint`` or ``baseline``. Measured cost: re-detecting
  the -700 mV recording under the -600 mV windows moved its median ``frac_depth``
  by 11.3 % (1.810 -> 2.014) and reordered the three-point trend.
* **High-2.** The fitted capture-rate ``V0`` was 0.653 V on the raw detections and
  0.441 V after a 10 ms merge -- a 33 % swing in the headline number -- with nothing
  anywhere recording which of the two was on show.

The invariant both fixes land on: **a pooled comparison carries the provenance of
what it pooled, and refuses silence when the inputs are not comparable.** Its two
halves live here and nowhere else:

  1. :class:`~nanoscope.analysis.voltage_dep.VDepPoint` carries the detection's
     ``baseline_fingerprint`` and merge state, so the provenance cannot be dropped
     between gathering a point and pooling it;
  2. :func:`~nanoscope.analysis.voltage_dep.vdep_comparability` is the single place
     that turns those into a verdict, and it is asked PER METRIC -- amplitudes
     (depth, dI/I0) are measured against the baseline, a rate and a dwell are not.
     Merge state is checked for every metric, because a merge moves them all.

Nothing here is allowed to hide a point or refuse a fit: the pool comes back whole,
with the reasons attached. That is asserted too.

Every fixture is a hand-built EventSet with the provenance written explicitly, so
the verdict is checked against exactly what went in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import voltage_dep
from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)

# The three real fingerprints from the apoferritin run, so the failure this file
# pins is addressed by its own identifiers.
FP_500 = "sha256:b64a75ebdf29"
FP_600 = "sha256:ce8ae31f10bd"
FP_700 = "sha256:1a29aabf5fb1"

# The measured capture rates from that run: raw detections vs the same three sets
# after a 10 ms gap merge (SUMMARY.md section 2a).
RAW_RATES = {-500.0: 4.783, -600.0: 4.610, -700.0: 6.497}
MERGED_RATES = {-500.0: 3.451, -600.0: 3.992, -700.0: 5.433}


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path: Path, *, nominal_mV: float, name: str, n: int = 2048) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -30.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _constant_df(*, n: int, dwell_s: float, depth_pA: float, frac_depth: float) -> pd.DataFrame:
    """``n`` identical rows -> every median is exactly the injected value."""
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 100 * i, "end_idx": 100 * i + 50,
            "t_start_s": 1e-3 * i, "dwell_s": float(dwell_s),
            "baseline_pA": -30.0, "depth_pA": float(depth_pA),
            "frac_depth": float(frac_depth), "ecd_pA_s": 0.05,
            "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False,
            "accepted": True, "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _install(
    session: Session,
    doc_id: str,
    df: pd.DataFrame,
    *,
    duration_s: float,
    fingerprint: str | None,
    merged_window_s: float | None = None,
) -> None:
    """Store a ``detect.schmitt`` result whose provenance says how it was measured."""
    prov: dict = {"analyzed_duration_s": float(duration_s), "recording": "rec"}
    if fingerprint is not None:
        prov["baseline_fingerprint"] = fingerprint
        prov["baseline_method"] = "masked_median"
    if merged_window_s is not None:
        prov["merged"] = {"rule": "gap", "window_s": float(merged_window_s),
                          "n_before": len(df) + 1, "n_after": len(df)}
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=EventSet(df, prov)))


def _ladder(
    tmp_path: Path,
    fingerprints: dict[float, str | None],
    *,
    rates: dict[float, float] | None = None,
    merged_window_s: float | None = None,
) -> Session:
    """The apoferritin ladder: one recording per voltage, each with its own fingerprint."""
    session = Session()
    rates = rates or RAW_RATES
    for mV, fp in fingerprints.items():
        doc_id = session.add_document(_doc(tmp_path, nominal_mV=mV, name=f"apo_{int(-mV)}.dat"))
        v = mV / 1000.0
        df = _constant_df(n=40, dwell_s=(v + 1.2) * 1e-3,
                          depth_pA=-100.0 * v, frac_depth=-0.1 * v)
        _install(session, doc_id, df, duration_s=40.0 / rates[mV],
                 fingerprint=fp, merged_window_s=merged_window_s)
    return session


_THREE_FINGERPRINTS = {-500.0: FP_500, -600.0: FP_600, -700.0: FP_700}
_ONE_FINGERPRINT = {-500.0: FP_600, -600.0: FP_600, -700.0: FP_600}

_AMPLITUDE_METRICS = ("median_depth", "median_frac_depth")
_BASELINE_FREE_METRICS = ("capture_rate_hz", "median_dwell_s", "n_events")


# ================================================================================================
# 1. the point carries what measured it
# ================================================================================================

def test_a_gathered_point_carries_its_baseline_fingerprint(tmp_path):
    """High-1's root: the provenance must not be dropped between the EventSet and the pool."""
    session = _ladder(tmp_path, _THREE_FINGERPRINTS)
    pts = voltage_dep.vdep_points(session, metric="median_depth")
    assert [p.baseline_fingerprint for p in pts] == [FP_500, FP_600, FP_700]


def test_a_gathered_point_carries_its_merge_state(tmp_path):
    """High-2's root: 'was this merged?' is a property of the pooled set, and travels."""
    unmerged = voltage_dep.vdep_points(_ladder(tmp_path, _ONE_FINGERPRINT),
                                       metric="capture_rate_hz")
    merged = voltage_dep.vdep_points(
        _ladder(tmp_path, _ONE_FINGERPRINT, rates=MERGED_RATES, merged_window_s=0.010),
        metric="capture_rate_hz")
    assert {p.merge_state for p in unmerged} == {voltage_dep.UNMERGED}
    assert {p.merge_state for p in merged} == {"merged: gap 10 ms"}


def test_a_detection_with_no_fingerprint_reads_as_unknown_not_as_a_match(tmp_path):
    """An ABSENT fingerprint is not evidence of agreement -- it is the absence of evidence."""
    session = _ladder(tmp_path, {-500.0: None, -600.0: None})
    pts = voltage_dep.vdep_points(session, metric="median_depth")
    assert [p.baseline_fingerprint for p in pts] == [voltage_dep.UNKNOWN_BASELINE] * 2


def test_merge_state_reads_both_places_a_merge_is_recorded():
    """The detector's own ``merge_gap_s`` and a post-hoc merge are both merges."""
    assert voltage_dep.merge_state(None) == voltage_dep.UNMERGED
    assert voltage_dep.merge_state({}) == voltage_dep.UNMERGED
    assert voltage_dep.merge_state({"merge_gap_s": None}) == voltage_dep.UNMERGED
    assert voltage_dep.merge_state({"merge_gap_s": 0.005}) == "merged: detector gap 5 ms"
    assert voltage_dep.merge_state(
        {"merged": {"rule": "gap", "window_s": 0.010}}) == "merged: gap 10 ms"
    assert voltage_dep.merge_state(
        {"merged": {"rule": "region", "region_s": (1.0, 2.0)}}) == "merged: region"
    assert voltage_dep.merge_state(
        {"merge_gap_s": 0.002, "merged": {"rule": "gap", "window_s": 0.010}}
    ) == "merged: detector gap 2 ms + gap 10 ms"


# ================================================================================================
# 2. High-1 -- an amplitude pooled across baselines refuses to be silent
# ================================================================================================

@pytest.mark.parametrize("metric", _AMPLITUDE_METRICS)
def test_pooling_an_amplitude_across_three_baselines_warns(tmp_path, metric):
    """THE High-1 REGRESSION. Before the fix this returned a fit and said nothing."""
    series = voltage_dep.vdep_series(_ladder(tmp_path, _THREE_FINGERPRINTS), metric=metric)
    assert series.comparable is False
    assert series.warnings, "a pooled amplitude across three baselines warned nothing"
    body = "\n".join(series.warnings)
    for fp in (FP_500, FP_600, FP_700):
        assert fp in body, f"the warning does not name {fp}"
    # It must say what to do, not merely that something is wrong.
    assert "Re-detect" in body
    assert "capture rate" in body and "median dwell" in body


@pytest.mark.parametrize("metric", _AMPLITUDE_METRICS)
def test_pooling_an_amplitude_within_one_baseline_warns_only_where_the_gain_bites(
        tmp_path, metric):
    """The BASELINE check must not degenerate into 'always complain' — and the GAIN
    check must fire underneath it, on exactly the metric that reads the gain.

    THIS TEST USED TO ASSERT ``series.warnings == ()`` FOR BOTH METRICS, and that was
    the M3 defect, written down. Every recording in ``_ladder`` carries the identity
    calibration ``(1.0, 0.0)`` — i.e. nobody has calibrated anything, and each pore's
    picoamps are its own raw scale (standing caveat 1: the open pore reads −27 to −52 pA
    where a 22 nm pore should be ~1000× larger). A median **depth in pA** pooled across
    three such pores is a fit over three different units, and it used to be drawn in
    silence. ``median dI/I0`` divides the gain out and is untouched, which is the whole
    reason the check is per metric rather than per plot.

    So the shape asserted here is deliberately the MIRROR of what it replaced: not
    "no warnings", but exactly one warning, from the gain half, naming its own subject
    and its own remedy — pinned as tightly as the silence used to be, and failing just
    as loudly if the gain rule is removed as it used to fail when the rule arrived.
    """
    series = voltage_dep.vdep_series(_ladder(tmp_path, _ONE_FINGERPRINT), metric=metric)

    # The baseline half is silent for BOTH metrics: one fingerprint, one group.
    assert series.baseline_groups == ((FP_600, ("apo_500.dat", "apo_600.dat", "apo_700.dat")),)
    assert not [w for w in series.warnings if "NOT COMPARABLE" in w], (
        "the baseline/merge check fired inside a single baseline configuration")

    reads_gain = voltage_dep.depends_on_gain(metric)
    assert reads_gain is (metric == "median_depth"), (
        "the gain question is per metric: an absolute pA reads the gain, a ratio does not")
    assert series.depends_on_gain is reads_gain

    if not reads_gain:
        # median dI/I0 — a ratio. Nothing at all to say.
        assert series.warnings == ()
        assert series.comparable is True
        return

    # median depth (pA) — the pooled amplitude that IS in per-pore units.
    assert series.comparable is False
    assert len(series.warnings) == 1, (
        f"expected exactly the gain caution, got {series.warnings!r}")
    (gain_warning,) = series.warnings
    assert gain_warning.startswith("CAUTION — comparing median depth across recordings:")
    assert "UNCALIBRATED" in gain_warning
    # It must say what to do, not merely that something is wrong — and the two things it
    # offers must be the two that survive an uncalibrated gain.
    assert "median dwell" in gain_warning and "median dI/I0" in gain_warning
    # ...and the identity gain is REPORTED as such, not printed as a real calibration.
    assert series.gain_groups == (
        ("uncalibrated", ("apo_500.dat", "apo_600.dat", "apo_700.dat")),)


@pytest.mark.parametrize("metric", _BASELINE_FREE_METRICS)
def test_rate_dwell_and_count_pool_across_different_baselines(tmp_path, metric):
    """PER METRIC, not blanket: the Schmitt edges are set by the noise ladder, so a
    rate and a dwell survive three different baseline windows. Disabling the whole
    panel would have thrown away the run's strongest result (dwell vs |V| is flat)."""
    series = voltage_dep.vdep_series(_ladder(tmp_path, _THREE_FINGERPRINTS), metric=metric)
    assert series.depends_on_baseline is False
    assert series.warnings == ()
    # ... and the baselines are still REPORTED, just not held against the metric.
    assert len(series.baseline_groups) == 3


@pytest.mark.parametrize("metric", _AMPLITUDE_METRICS)
def test_an_amplitude_with_no_fingerprint_anywhere_warns(tmp_path, metric):
    series = voltage_dep.vdep_series(_ladder(tmp_path, {-500.0: None, -700.0: None}),
                                     metric=metric)
    assert series.comparable is False
    assert "nothing to check them against" in "\n".join(series.warnings)


@pytest.mark.parametrize("metric", _AMPLITUDE_METRICS)
def test_an_amplitude_with_one_unstamped_recording_warns(tmp_path, metric):
    """Known + unknown is a MISMATCH, not a majority vote."""
    series = voltage_dep.vdep_series(
        _ladder(tmp_path, {-500.0: FP_600, -600.0: FP_600, -700.0: None}), metric=metric)
    assert series.comparable is False
    assert voltage_dep.UNKNOWN_BASELINE in "\n".join(series.warnings)


def test_depends_on_baseline_answers_per_metric():
    assert voltage_dep.depends_on_baseline("median_depth") is True
    assert voltage_dep.depends_on_baseline("median_frac_depth") is True
    for metric in _BASELINE_FREE_METRICS:
        assert voltage_dep.depends_on_baseline(metric) is False
    with pytest.raises(ValueError):
        voltage_dep.depends_on_baseline("median_snr")


def test_a_new_median_metric_is_baseline_dependent_until_proved_otherwise(monkeypatch):
    """FAILS CLOSED. The next amplitude added to this plot (median SNR, median ECD)
    must inherit the warning without anyone remembering to opt it in."""
    monkeypatch.setattr(voltage_dep, "METRICS", voltage_dep.METRICS + ("median_snr",))
    monkeypatch.setattr(
        voltage_dep, "_MEDIAN_COLUMN", {**voltage_dep._MEDIAN_COLUMN, "median_snr": "snr"})
    assert voltage_dep.depends_on_baseline("median_snr") is True


# ================================================================================================
# 3. High-2 -- the merge state of the pooled sets is carried, and a mismatch warns
# ================================================================================================

def test_the_fitted_v0_moves_with_the_merge_and_the_series_says_which_it_is(tmp_path):
    """THE High-2 REGRESSION, on the run's own numbers.

    The measured capture rates give V0 = 0.6530 V raw and 0.4407 V after a 10 ms
    merge -- the 33 % swing in the headline number. Both fits are correct; what was
    missing is any record of WHICH event sets produced the number, and that record
    is what this asserts."""
    raw = voltage_dep.vdep_series(
        _ladder(tmp_path, _ONE_FINGERPRINT, rates=RAW_RATES), metric="capture_rate_hz")
    merged = voltage_dep.vdep_series(
        _ladder(tmp_path, _ONE_FINGERPRINT, rates=MERGED_RATES, merged_window_s=0.010),
        metric="capture_rate_hz")

    def v0(series):
        fit = voltage_dep.vdep_fit([abs(p.v_volts) for p in series.points],
                                   [p.value for p in series.points], model="exponential")
        return fit["params"]["V0"]

    assert v0(raw) == pytest.approx(0.6530, abs=5e-4)
    assert v0(merged) == pytest.approx(0.4407, abs=5e-4)
    assert abs(v0(raw) - v0(merged)) / v0(raw) > 0.3          # the 33 % swing

    # Two different headline numbers, and the series now says which set each is from.
    assert raw.merge_groups == ((voltage_dep.UNMERGED,
                                 ("apo_500.dat", "apo_600.dat", "apo_700.dat")),)
    assert merged.merge_groups == (("merged: gap 10 ms",
                                    ("apo_500.dat", "apo_600.dat", "apo_700.dat")),)
    assert raw.merge_groups != merged.merge_groups


@pytest.mark.parametrize("metric", _AMPLITUDE_METRICS + _BASELINE_FREE_METRICS)
def test_a_merged_set_pooled_with_an_unmerged_one_warns_for_every_metric(tmp_path, metric):
    """A merge moves the count, the rate, the dwell AND the depth, so this check is
    the one that IS blanket."""
    session = Session()
    for mV, window in ((-500.0, 0.010), (-600.0, None), (-700.0, 0.010)):
        doc_id = session.add_document(_doc(tmp_path, nominal_mV=mV, name=f"apo_{int(-mV)}.dat"))
        v = mV / 1000.0
        _install(session, doc_id,
                 _constant_df(n=40, dwell_s=(v + 1.2) * 1e-3, depth_pA=-100.0 * v,
                              frac_depth=-0.1 * v),
                 duration_s=10.0, fingerprint=FP_600, merged_window_s=window)
    series = voltage_dep.vdep_comparability(
        voltage_dep.vdep_points(session, metric=metric), metric=metric)
    assert series.comparable is False
    body = "\n".join(series.warnings)
    assert "merge state" in body
    assert "apo_600.dat" in body and "apo_500.dat" in body


# ================================================================================================
# 4. the verdict never censors the data
# ================================================================================================

def test_an_incomparable_pool_is_returned_whole(tmp_path):
    """"Do not simply disable the panel." Every point survives the verdict."""
    session = _ladder(tmp_path, _THREE_FINGERPRINTS)
    series = voltage_dep.vdep_series(session, metric="median_frac_depth")
    assert series.comparable is False
    assert len(series.points) == 3
    assert [p.value for p in series.points] == pytest.approx([0.05, 0.06, 0.07])
    # and it still fits -- the numbers are qualified, not withheld
    fit = voltage_dep.vdep_fit([p.v_volts for p in series.points],
                               [p.value for p in series.points], model="linear")
    assert fit["params"]["slope"] == pytest.approx(-0.1, rel=1e-6)


def test_a_single_recording_cannot_disagree_with_anything(tmp_path):
    series = voltage_dep.vdep_series(_ladder(tmp_path, {-500.0: FP_500}),
                                     metric="median_depth")
    assert len(series.points) == 1
    assert series.warnings == ()


def test_an_empty_session_is_comparable_and_empty(tmp_path):
    series = voltage_dep.vdep_series(Session(), metric="median_depth")
    assert series.points == ()
    assert series.warnings == ()


def test_vdep_series_rejects_an_unknown_metric(tmp_path):
    with pytest.raises(ValueError, match="unknown metric"):
        voltage_dep.vdep_series(_ladder(tmp_path, _ONE_FINGERPRINT), metric="median_snr")
    with pytest.raises(ValueError, match="unknown metric"):
        voltage_dep.vdep_comparability((), metric="median_snr")
