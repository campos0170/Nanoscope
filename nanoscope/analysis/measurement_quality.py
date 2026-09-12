"""Advisory measurement checks, separate from detection and human curation.

These checks never alter an event, its acceptance, or an analytical default.
Agreement can share a wrong reference, and undetected events are not examined.
Consumers must use ``validated_measurement_quality`` before showing QC as current.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .baseline import BaselineConfig, MASKED_METHODS, baseline_intrusion
from .open_pore_reference import OPEN_PORE_REFERENCE_PARAM_KEY

REPORT_KEY = "measurement_quality"
REPORT_VERSION = 1
INTRUSION_ADVISORY_FRACTION = 0.1  # Existing baseline_intrusion severe boundary.
_LIMITATION = (
    "Advisory checks only: no flag and polarity agreement do not validate the "
    "physical open-pore reference, prove detection completeness, or constitute "
    "human review. Accepted is a separate curation state."
)
_MEASUREMENT_COLUMNS = (
    "event_id", "start_idx", "end_idx", "t_start_s", "t_end_s", "dwell_s",
    "baseline_pA", "depth_pA", "frac_depth", "mean_dev_pA", "max_dev_pA",
    "ecd_pA_s", "snr", "rise_s", "fall_s", "n_levels", "filter_limited",
)
_INPUT_KEYS = (
    "fs", "n_samples", "noise_rms_pA", "exclude_regions",
    *(f"baseline_{name}" for name in BaselineConfig._FIELDS),
    "baseline_fingerprint", "baseline_polarity", "baseline_median_pA",
    "baseline_provided", "baseline_reference_applied", "baseline_estimator_bypassed",
    "baseline_open_pore_reference", "baseline_fallback", "baseline_health", "baseline_warnings",
    "polarity", "polarity_requested", "polarity_source", "polarity_measured",
    "polarity_evidence", "polarity_warning", "filter_cutoff_hz", "filter_kind",
    "filter_order", "filter_zero_phase", "filter_rise_s",
)


def _fingerprint(df: pd.DataFrame, provenance: Mapping[str, Any]) -> str:
    """Conservative measurement identity; human accept/reject is not QC."""
    digest = hashlib.sha256()
    for name in (*_MEASUREMENT_COLUMNS, *sorted(c for c in df if c.startswith("amplitude_"))):
        if name not in df:
            continue
        digest.update(name.encode())
        if pd.api.types.is_numeric_dtype(df[name].dtype):
            values = df[name].to_numpy(dtype="<f8", na_value=np.nan).copy()
            values[np.isnan(values)] = np.nan  # Canonical NaN payload.
            digest.update(values.tobytes())
        else:
            digest.update(json.dumps(df[name].fillna("").astype(str).tolist()).encode())
    # Baseline source labels and unrelated new metadata do not change a measurement.
    inputs = {key: provenance.get(key) for key in _INPUT_KEYS}
    digest.update(json.dumps(inputs, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _empty_report(state: str, message: str) -> dict[str, Any]:
    return {
        "version": REPORT_VERSION, "state": state, "status": "not_assessed",
        "recording_flags": [f"measurement_quality_{state}"],
        "messages": [message], "events": [], "limitations": _LIMITATION,
    }


def validated_measurement_quality(
    df: pd.DataFrame, provenance: Mapping[str, Any], *, copy_report: bool = True,
) -> dict[str, Any]:
    """Return current QC or explicit missing/stale state after measurement edits.

    A stale report retains its old snapshot under ``previous_report`` for audit,
    never as current event evidence. Merge, resize, remeasurement, subsets,
    manual amplitude and changed-reference paths cannot silently reuse fresh QC.
    """
    report = provenance.get(REPORT_KEY)
    if not isinstance(report, Mapping):
        return _empty_report("missing", "Measurement quality has not been assessed for this event table.")
    if (
        report.get("state") == "stale" and report.get("status") == "not_assessed"
        and report.get("events") == [] and "previous_report" in report
    ):
        # Getter-persisted stale state is already validated. Do not nest snapshots
        # or copy a large prior report just to show its short Details message.
        return copy.deepcopy(dict(report)) if copy_report else report
    if not measurement_quality_is_current(df, provenance):
        stale = _empty_report(
            "stale", "Measurement quality is stale after measurement or reference changes; "
            "rerun detection to refresh these advisory checks. Existing acceptance is unchanged.",
        )
        stale["previous_report"] = copy.deepcopy(dict(report))
        return stale
    return copy.deepcopy(dict(report)) if copy_report else report


def measurement_quality_is_current(df: pd.DataFrame, provenance: Mapping[str, Any]) -> bool:
    """Cheap getter guard: hash numeric inputs without copying per-event records."""
    report = provenance.get(REPORT_KEY)
    return bool(
        isinstance(report, Mapping) and report.get("version") == REPORT_VERSION
        and report.get("state") == "current"
        and report.get("measurement_fingerprint") == _fingerprint(df, provenance)
    )


def report_recording_messages(df: pd.DataFrame, provenance: Mapping[str, Any]) -> tuple[str, ...]:
    """Details-only messages, including explicit stale/unavailable labels."""
    report = validated_measurement_quality(df, provenance, copy_report=False)
    return tuple(dict.fromkeys([*report.get("messages", []), report["limitations"]]))


def invalidate_measurement_quality(provenance: dict[str, Any], reason: str) -> None:
    """Explicit invalidation for lifecycle changes not visible in numeric columns."""
    if isinstance(provenance.get(REPORT_KEY), Mapping):
        report = copy.deepcopy(dict(provenance[REPORT_KEY]))
        report.update(state="stale", stale_reason=str(reason))
        provenance[REPORT_KEY] = report


def build_measurement_quality(
    current: np.ndarray, baseline: np.ndarray, df: pd.DataFrame, fs: float,
    provenance: Mapping[str, Any], *, flank_s: float = 0.005, pad_s: float = 0.0002,
) -> dict[str, Any]:
    """Check polarity consistency and local baseline-following without mutation.

    Intrusion reuses ``baseline_intrusion`` but requires complete two-sided flanks,
    each free of other detected events and excluded spans. Unknown/contaminated
    flanks are explicitly unassessed, not a reassuring zero. User-reference modes
    bypass estimator-intrusion checks: correctness requires independent evidence.
    """
    x, b = np.asarray(current), np.asarray(baseline)
    if x.ndim != 1 or x.shape != b.shape or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Quality assessment requires matching 1-D current/baseline and positive fs.")
    flags: list[str] = []
    messages: list[str] = []
    supplied_reference = bool(provenance.get("baseline_reference_applied"))
    recipe = provenance.get(OPEN_PORE_REFERENCE_PARAM_KEY) or {}
    mode = recipe.get("mode", "user_reference") if supplied_reference else (
        "provided_array" if provenance.get("baseline_provided") else "automatic"
    )
    used, estimated = provenance.get("polarity"), provenance.get("baseline_polarity")
    evidence = provenance.get("polarity_evidence") or {}
    method = provenance.get("baseline_method")
    polarity_sensitive = method in (*MASKED_METHODS, "percentile", "asls") or bool(provenance.get("baseline_two_pass"))
    if supplied_reference:
        polarity_state = "not_applicable_user_reference"
        messages.append(
            "The user-supplied open-pore reference bypasses automatic baseline construction. "
            "Its correctness is not certified by these checks; verify signed current, "
            "calibration and applicability to this recording."
        )
    elif method in ("constant", "block_median", "savgol") and not provenance.get("baseline_two_pass"):
        polarity_state = "not_applicable_polarity_independent_baseline"
    elif estimated not in (-1, 1) or used not in (-1, 1) or not polarity_sensitive:
        polarity_state = "not_assessed"
        flags.append("baseline_polarity_not_assessed")
        messages.append("Baseline-stage polarity is unavailable; consistency with detection cannot be assessed.")
    elif int(estimated) != int(used):
        polarity_state = "disagreement"
        flags.append("baseline_detector_polarity_disagreement")
        messages.append(
            f"Baseline construction used polarity {int(estimated):+d}, but detection used "
            f"{int(used):+d}. This can indicate an ambiguous reference or high occupancy. "
            "Inspect the recording and consider a justified known-current or selected-interval reference."
        )
    else:
        polarity_state = "agreement_not_validation"
    if evidence.get("verdict") in ("undetermined", "tie") or not evidence:
        flags.append("detector_polarity_undetermined")
        messages.append("Excursion polarity is not independently determined from this recording; the chosen direction is not confirmation.")
    if provenance.get("polarity_warning"):
        flags.append("detector_polarity_advisory")
        messages.append(str(provenance["polarity_warning"]))
    if provenance.get("baseline_warnings"):
        flags.append("baseline_estimator_advisory")
        messages.extend(str(w) for w in provenance["baseline_warnings"])

    starts = np.asarray(df["start_idx"], dtype=np.int64)
    ends = np.asarray(df["end_idx"], dtype=np.int64)
    if np.any(starts < 0) or np.any(ends > x.size) or np.any(ends <= starts):
        raise ValueError("Quality assessment requires valid half-open event spans.")
    pad = max(1, int(round(pad_s * fs)))
    flank = max(2, int(round(flank_s * fs)))
    spans = list(zip(starts.tolist(), ends.tolist()))
    spans.extend(
        (max(0, int(float(s) * fs)), min(x.size, int(float(e) * fs)))
        for s, e in provenance.get("exclude_regions", [])
    )
    spans.sort()
    occupied_start = np.asarray([s for s, _ in spans], dtype=np.int64)
    occupied_end = np.maximum.accumulate([e for _, e in spans]) if spans else np.empty(0)

    def overlaps(s: int, e: int) -> bool:
        ix = int(np.searchsorted(occupied_start, e, side="left")) - 1
        return ix >= 0 and occupied_end[ix] > s

    fractions = np.full(len(df), np.nan)
    if len(df) and not supplied_reference and used in (-1, 1):
        fractions = baseline_intrusion(
            x, b, (starts, ends), fs, int(used), flank_s=flank_s, pad_s=pad_s,
        )["per_event"]
    records = []
    for pos, (s, e) in enumerate(zip(starts, ends)):
        event_flags = list(flags)
        ls, le, rs, re = s - pad - flank, s - pad, e + pad, e + pad + flank
        if supplied_reference:
            intrusion_state = "not_applicable_user_reference"
        elif ls < 0 or re > x.size:
            intrusion_state = "not_assessed_incomplete_flanks"
        elif overlaps(int(ls), int(le)) or overlaps(int(rs), int(re)):
            intrusion_state = "not_assessed_contaminated_flanks"
        elif not np.isfinite(fractions[pos]):
            intrusion_state = "not_assessed_undefined_depth"
        else:
            intrusion_state = "assessed"
        value = float(fractions[pos]) if intrusion_state == "assessed" else None
        if value is not None and value > INTRUSION_ADVISORY_FRACTION:
            event_flags.append("baseline_intrusion_suspected")
        elif intrusion_state.startswith("not_assessed"):
            event_flags.append("baseline_intrusion_not_assessed")
        warnings = [f for f in event_flags if not f.endswith(("not_assessed", "undetermined"))]
        records.append({
            "event_id": int(df.iloc[pos]["event_id"]) if "event_id" in df else pos,
            "start_idx": int(s), "end_idx": int(e),
            "status": "review_required" if warnings else (
                "not_assessed" if event_flags or supplied_reference else "no_issue_detected"
            ),
            "quality_flags": event_flags,
            "baseline_intrusion_state": intrusion_state,
            "baseline_intrusion_fraction": value,
        })
    n_intrusion = sum("baseline_intrusion_suspected" in r["quality_flags"] for r in records)
    n_unassessed = sum(r["baseline_intrusion_state"].startswith("not_assessed") for r in records)
    if not records:
        flags.append("no_events_measurement_quality_unassessed")
        messages.append(
            "No detected events: event measurement quality and missed-event completeness "
            "cannot be assessed. A zero count does not demonstrate a correct baseline."
        )
    if n_intrusion:
        messages.append(
            f"{n_intrusion} event(s) show local baseline-following above the 10% advisory "
            "threshold relative to flanking baseline estimates. This suggests depth bias, "
            "not a calibrated estimate of true amplitude error. Inspect the reference."
        )
    if n_unassessed:
        messages.append(
            f"Local baseline intrusion was not assessable for {n_unassessed} event(s): "
            "flanks were incomplete, overlapped detected/excluded spans, or depth was undefined."
        )
    warning_flags = [f for f in flags if not f.endswith(("not_assessed", "undetermined", "unassessed"))]
    return {
        "version": REPORT_VERSION, "state": "current",
        "status": "review_required" if warning_flags or n_intrusion else (
            "not_assessed" if flags or supplied_reference or n_unassessed else "no_issue_detected"
        ),
        "reference_mode": mode, "baseline_detector_polarity_state": polarity_state,
        "recording_flags": list(dict.fromkeys(flags)),
        "messages": list(dict.fromkeys(messages)), "events": records,
        "n_events": len(records), "n_intrusion_advisories": n_intrusion,
        "n_intrusion_unassessed": n_unassessed,
        "intrusion_advisory_fraction": INTRUSION_ADVISORY_FRACTION,
        "intrusion_flank_s": float(flank_s), "intrusion_pad_s": float(pad_s),
        "limitations": _LIMITATION,
        "measurement_fingerprint": _fingerprint(df, provenance),
    }
