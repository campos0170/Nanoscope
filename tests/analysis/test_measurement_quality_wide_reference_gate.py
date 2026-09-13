"""B08: a clean-looking near-flank fraction is not evidence when the estimator's
own wide-reference health checks already say its state/level tracking is wrong.

The per-event ``baseline_intrusion`` check (measurement_quality.py) compares a
short LEFT flank against a short RIGHT flank of the SAME tracked baseline. Both
flanks can land on the same ramp (an isolated long event the masked-median
baseline rides halfway into) or the same locked-in level (a high-duty pore whose
estimator inverted) and therefore agree with each other while being equally
wrong -- reporting ``baseline_intrusion_state == "assessed"`` with a fraction
near zero on exactly the two canonical Automatic-mode baseline failures. See
``test_measurement_quality_advisory.py`` for the additive-QC contract this
module also has to keep: short, low-duty events with a healthy baseline must
keep reporting a clean local fraction and no advisory.
"""
import numpy as np
import pandas as pd
import pytest

import nanoscope.analysis.detect as detect_module
from nanoscope.analysis.detect import detect_events

FS_A, I0, DEPTH, SIGMA = 50_000.0, -500.0, 100.0, 5.0


def _isolated_long_events_trace(fs=FS_A, total_s=6.0, n_ev=3, dwell_s=0.5, seed=7):
    """3 x 500 ms, 100 pA blockades on a -500 pA open pore, 5 pA noise, 50 kHz.

    The masked-median baseline (window_s comparable to the 500 ms states) rides
    into every one of them; detection returns only the transition-edge fragments.
    """
    rng = np.random.default_rng(seed)
    n = int(total_s * fs)
    x = np.full(n, I0) + rng.normal(0.0, SIGMA, n)
    for i in range(n_ev):
        s = int((total_s / (n_ev + 1) * (i + 1) - dwell_s / 2) * fs)
        x[s:s + int(dwell_s * fs)] += DEPTH
    return x.astype(np.float32)


def _high_duty_trace(fs=10_000.0, total_s=1.0, n_ev=30, dwell_s=0.02, seed=0):
    """30 x 20 ms blockades at 60% duty, 10 kHz -- the estimator locks onto the
    blocked level and both polarity votes invert consistently."""
    rng = np.random.default_rng(seed)
    n = int(total_s * fs)
    x = np.full(n, I0) + rng.normal(0.0, SIGMA, n)
    for i in range(n_ev):
        s = int((total_s / (n_ev + 1) * (i + 1) - dwell_s / 2) * fs)
        x[s:s + int(dwell_s * fs)] += DEPTH
    return x.astype(np.float32)


def _clean_low_duty_trace(n=1_000_000, fs=FS_A, seed=11):
    """20 short (1 ms) events at ~0.005% duty -- a healthy Automatic baseline."""
    rng = np.random.default_rng(seed)
    x = rng.normal(I0, SIGMA, n)
    for s in np.linspace(2000, n - 2000, 20, dtype=int):
        x[s:s + 50] += DEPTH
    return x.astype(np.float32)


@pytest.mark.parametrize(
    "trace_fn,fs,label",
    [
        (_isolated_long_events_trace, FS_A, "case A: isolated long events"),
        (_high_duty_trace, 10_000.0, "case B: high duty"),
    ],
)
def test_canonical_baseline_failures_flag_intrusion_suspected_not_a_reassuring_zero(
    trace_fn, fs, label,
):
    x = trace_fn()
    df, prov = detect_events(x, fs, min_dwell_s=5e-4)
    mq = prov["measurement_quality"]
    assert len(df) > 0, f"{label}: detector produced no rows to assess"

    assessed = [r for r in mq["events"] if r["baseline_intrusion_state"] == "assessed"]
    assert assessed, f"{label}: expected at least one locally-assessed fragment"
    # This is the defect: every assessed fragment used to read a small |fraction|
    # with no flag, even though the wide-reference health block already knows the
    # tracked baseline sits inside these states (or on the wrong level).
    assert all(r["baseline_intrusion_fraction"] is not None for r in assessed)
    assert all(
        "baseline_intrusion_suspected" in r["quality_flags"] for r in assessed
    ), f"{label}: assessed fragment(s) missing baseline_intrusion_suspected: {assessed}"
    assert mq["n_intrusion_advisories"] == len(assessed)
    assert mq["status"] == "review_required"
    assert all(r["status"] == "review_required" for r in assessed)
    assert any("baseline_intrusion" in msg or "recording-level" in msg
               or "Baseline health" in msg for msg in mq["messages"])


def test_clean_short_low_duty_events_are_unaffected_by_the_wide_reference_gate():
    x = _clean_low_duty_trace()
    df, prov = detect_events(x, FS_A, polarity=1, min_dwell_s=2e-5)
    mq = prov["measurement_quality"]
    assert len(df) == 20
    assert mq["status"] == "no_issue_detected"
    assert mq["n_intrusion_advisories"] == 0
    for row in mq["events"]:
        assert row["baseline_intrusion_state"] == "assessed"
        assert row["baseline_intrusion_fraction"] is not None
        assert abs(row["baseline_intrusion_fraction"]) <= 0.1
        assert not row["quality_flags"]
        assert row["status"] == "no_issue_detected"


def test_wide_reference_gate_never_touches_detector_numeric_or_acceptance_output():
    """Advisory-only: the new gate reads baseline_health but must not perturb a
    single detector column or provenance key it does not itself own."""
    x = _isolated_long_events_trace()
    before = x.copy()
    actual_df, actual_prov = detect_events(x, FS_A, min_dwell_s=5e-4)
    assert "measurement_quality" in actual_prov
    assert actual_prov["measurement_quality"]["n_intrusion_advisories"] > 0

    real_builder = detect_module.build_measurement_quality
    detect_module.build_measurement_quality = lambda *a, **k: None
    try:
        expected_df, expected_prov = detect_events(x, FS_A, min_dwell_s=5e-4)
    finally:
        detect_module.build_measurement_quality = real_builder

    pd.testing.assert_frame_equal(actual_df, expected_df, check_exact=True)
    np.testing.assert_array_equal(x, before)
    for prov in (actual_prov, expected_prov):
        prov.pop("measurement_quality", None)
        prov.pop("detected_at", None)
    assert actual_prov == expected_prov
