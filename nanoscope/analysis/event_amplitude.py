"""Explicit, optional amplitude measurements for already-detected events.

The frozen EventSet ``depth_pA`` remains the detector's median deviation. Nothing
here detects, trims, filters, fits, or changes a SCHEMA column. Amplitudes use the
caller's analysis signal and samplewise baseline. A horizontal level is an
equivalent annotation about the mean baseline, not necessarily a raw statistic.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .events import SCHEMA
from .measure import _mode_level

METHODS = ("median", "mean", "peak", "mode", "selected_median", "selected_mean", "manual")
ESTIMATOR_VERSION = 1

# Deliberately NOT part of events.SCHEMA. Nullable indices mean an unmeasured or
# stale row stays blank without inventing a valid sample interval.
AMPLITUDE_COLUMNS: dict[str, tuple[str, str, str]] = {
    "amplitude_pA": ("float64", "pA", "explicit amplitude; see amplitude_units"),
    "amplitude_method": ("string", "-", "explicit amplitude estimator"),
    "amplitude_level_pA": ("float64", "pA", "equivalent level about the mean baseline"),
    "amplitude_baseline_pA": ("float64", "pA", "mean baseline in the measured interval"),
    "amplitude_start_idx": ("Int64", "sample", "first measured sample"),
    "amplitude_end_idx": ("Int64", "sample", "one past the last measured sample"),
    "amplitude_units": ("string", "-", "declared units of signal and baseline"),
    "amplitude_provenance_json": ("string", "-", "estimator, interval, signal identity, original row fingerprint"),
    "amplitude_status": ("string", "-", "current or stale; blank means no optional measurement"),
}
_NUMERIC_COLUMNS = tuple(k for k, (dtype, _, _) in AMPLITUDE_COLUMNS.items()
                         if dtype in ("float64", "Int64"))
_FINGERPRINT_COLUMNS = tuple(k for k in SCHEMA
                             if k not in ("event_id", "recording", "pore_id", "accepted", "flag"))


def _index(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer sample index")
    try:
        number, integer = float(value), int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer sample index") from exc
    if not np.isfinite(number) or integer != number or integer < 0:
        raise ValueError(f"{label} must be a nonnegative integer sample index")
    return integer


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _json_value(value: Any, *, identity: bool = False) -> Any:
    """JSON values only; never stringify arbitrary objects or persist paths."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Provenance keys must be strings")
            if identity and re.search(r"(^|_)(path|filename|directory|uri|url)s?($|_)", key.lower()):
                raise ValueError("Amplitude provenance must use identities, not file paths")
            result[key] = _json_value(item, identity=identity)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, identity=identity) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            if identity:
                raise ValueError("Provenance numbers must be finite")
            return None
        return float(value)
    if isinstance(value, str):
        if identity and (value.startswith(("/", "~/", "file:"))
                         or re.match(r"^[A-Za-z]:[\\/]", value)):
            raise ValueError("Amplitude provenance must not contain file paths")
        return value
    raise ValueError("Provenance must contain JSON-compatible values")


def _fingerprint(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    return {key: _json_value(row[key]) for key in _FINGERPRINT_COLUMNS if key in row}


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _roundtrip_equal(left: Any, right: Any) -> bool:
    """Permit decimal CSV parser roundoff, not a scientific remeasurement.

    The authoritative values remain in the JSON audit. Pandas' default CSV parser
    can move the companion numeric columns a few ulps on reload.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _roundtrip_equal(left[k], right[k]) for k in left)
    if isinstance(left, float) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=4e-15, abs_tol=1e-15)
    return left == right


def measure_event_amplitude(
    current: np.ndarray,
    baseline: np.ndarray | float,
    row: Mapping[str, Any] | pd.Series,
    *,
    polarity: int,
    method: str,
    units: str,
    selection: tuple[int, int] | None = None,
    manual_level: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a chosen amplitude, without modifying inputs.

    Current and array baseline cover the full trace; an explicitly supplied scalar
    baseline is also accepted. Selected estimators require a contained, absolute,
    half-open interval and never clip. Transitions are included; nonfinite measured
    samples are refused. ``peak`` is the largest signed excursion, not an absolute
    outlier. ``mode`` uses the manual-measurement FD histogram estimator (16–512
    bins, 64 for zero IQR). Manual level is on the same analysis signal and refers
    to mean baseline over the full event. Negative amplitudes are refused.

    Provenance names signal/baseline/conditioning identities, never file paths.
    Invalid input raises ValueError before any table is written.
    """
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"Unknown amplitude method: {method!r}")
    if isinstance(polarity, (bool, np.bool_)) or polarity not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")
    polarity = int(polarity)
    if not isinstance(units, str) or not units.strip():
        raise ValueError("Signal units must be explicitly supplied")
    if not isinstance(row, (Mapping, pd.Series)):
        raise ValueError("An event row is required")
    try:
        event_start = _index(row["start_idx"], "start_idx")
        event_end = _index(row["end_idx"], "end_idx")
    except KeyError as exc:
        raise ValueError("The event row must supply sample bounds") from exc
    try:
        signal = np.asarray(current)
        base = np.asarray(baseline)
        # Numeric recording arrays can be inspected without promoting the
        # entire recording for a measurement of a short event. Retain the
        # previous conversion/validation for object and string inputs.
        if signal.dtype.kind not in "biufc":
            signal = np.asarray(current, dtype=np.float64)
        if base.dtype.kind not in "biufc":
            base = np.asarray(baseline, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Current and baseline must be numeric") from exc
    if signal.ndim != 1 or not (0 <= event_start < event_end <= signal.size):
        raise ValueError("Event bounds must be a nonempty interval in the full trace")
    if base.ndim != 0 and (base.ndim != 1 or base.shape != signal.shape):
        raise ValueError("Baseline must be scalar or match the full current trace")
    start, end = event_start, event_end
    if method.startswith("selected_"):
        if not isinstance(selection, (tuple, list)) or len(selection) != 2:
            raise ValueError("Selected estimators require a half-open sample interval")
        start, end = (_index(selection[0], "selection start"),
                      _index(selection[1], "selection end"))
        if not event_start <= start < end <= event_end:
            raise ValueError("The measurement interval must be contained in the event")
    elif selection is not None:
        raise ValueError("A selection is only valid with a selected estimator")
    if method != "manual" and manual_level is not None:
        raise ValueError("manual_level is only valid for manual amplitude")
    window = np.asarray(signal[start:end], dtype=np.float64)
    baseline_window = np.asarray(base if base.ndim == 0 else base[start:end], dtype=np.float64)
    if not np.isfinite(window).all() or not np.isfinite(baseline_window).all():
        raise ValueError("All measured signal and baseline samples must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        baseline_mean = _finite(np.mean(baseline_window), "Mean baseline")
        dev = polarity * (window - baseline_window)
    if not np.isfinite(dev).all():
        raise ValueError("Samplewise excursions must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        if method in ("median", "selected_median"):
            amplitude = float(np.median(dev))
        elif method in ("mean", "selected_mean"):
            amplitude = float(np.mean(dev))
        elif method == "peak":
            amplitude = float(np.max(dev))
        elif method == "mode":
            amplitude = _mode_level(dev)
        else:
            amplitude = polarity * (_finite(manual_level, "Manual level") - baseline_mean)
        amplitude = _finite(amplitude, "Amplitude")
        level = _finite(baseline_mean + polarity * amplitude, "Equivalent level")
    if amplitude < 0:
        raise ValueError("Amplitude must be nonnegative for the selected polarity")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("Provenance must be a mapping")
    metadata = {
        "estimator_version": ESTIMATOR_VERSION, "method": method,
        "polarity": polarity, "units": units.strip(),
        "event_bounds": [event_start, event_end], "interval": [start, end],
        "interval_scope": "selected_interval" if method.startswith("selected_") else "whole_event",
        "sample_count": end - start,
        "baseline_reference": "mean_of_samplewise_baseline_in_interval",
        "level_definition": "baseline_mean_plus_polarity_times_amplitude",
        "amplitude": amplitude, "baseline_mean": baseline_mean, "level": level,
        "event_fingerprint": _fingerprint(row),
        "signal_identity": _json_value(provenance or {}, identity=True),
    }
    if method == "mode":
        metadata["histogram_estimator"] = "freedman_diaconis_16_512_bins_zero_iqr_64"
    return {
        "amplitude_pA": amplitude, "amplitude_method": method,
        "amplitude_level_pA": level, "amplitude_baseline_pA": baseline_mean,
        "amplitude_start_idx": start, "amplitude_end_idx": end,
        "amplitude_units": units.strip(),
        "amplitude_provenance_json": _dump(metadata), "amplitude_status": "current",
    }


def amplitude_is_current(
    row: Mapping[str, Any] | pd.Series,
    provenance: Mapping[str, Any] | None = None,
) -> bool:
    """Validate row fingerprint and, optionally, its original signal identity.

    Extra current identity keys are allowed; all originally recorded keys must
    still match. Without provenance this cannot discover external trace changes.
    Remeasure/mask/merge callers must additionally invalidate explicitly.
    """
    try:
        if row.get("amplitude_status") != "current":
            return False
        meta = json.loads(row["amplitude_provenance_json"])
        if not isinstance(meta, dict) or meta.get("estimator_version") != ESTIMATOR_VERSION:
            return False
        method = row["amplitude_method"]
        if method not in METHODS or meta["method"] != method:
            return False
        polarity = meta["polarity"]
        if isinstance(polarity, bool) or polarity not in (-1, 1):
            return False
        start, end = (_index(row["amplitude_start_idx"], "start"),
                      _index(row["amplitude_end_idx"], "end"))
        event_start, event_end = (_index(row["start_idx"], "event start"),
                                  _index(row["end_idx"], "event end"))
        if not event_start <= start < end <= event_end:
            return False
        selected = method.startswith("selected_")
        if not selected and (start, end) != (event_start, event_end):
            return False
        if (meta["event_bounds"] != [event_start, event_end]
                or meta["interval"] != [start, end] or meta["sample_count"] != end - start
                or meta["interval_scope"] != ("selected_interval" if selected else "whole_event")
                or not _roundtrip_equal(meta["event_fingerprint"], _fingerprint(row))):
            return False
        units = row["amplitude_units"]
        if not isinstance(units, str) or not units.strip() or meta["units"] != units:
            return False
        amp = _finite(row["amplitude_pA"], "Amplitude")
        baseline = _finite(row["amplitude_baseline_pA"], "Baseline")
        level = _finite(row["amplitude_level_pA"], "Level")
        if (amp < 0 or not _roundtrip_equal(meta["amplitude"], amp)
                or not _roundtrip_equal(meta["baseline_mean"], baseline)
                or not _roundtrip_equal(meta["level"], level)
                or not _roundtrip_equal(baseline + polarity * amp, level)):
            return False
        identity = meta["signal_identity"]
        if not isinstance(identity, dict) or _json_value(identity, identity=True) != identity:
            return False
        if provenance is not None:
            current_identity = _json_value(provenance, identity=True)
            if not isinstance(current_identity, dict):
                return False
            if any(key not in current_identity or current_identity[key] != value
                   for key, value in identity.items()):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _position(df: pd.DataFrame, row_index: Any) -> int:
    position = _index(row_index, "row position")
    if position >= len(df) or not df.columns.is_unique:
        raise ValueError("Invalid row position or duplicate DataFrame columns")
    return position


def _blank(dtype: str) -> Any:
    return np.nan if dtype == "float64" else pd.NA


def _optional_series(df: pd.DataFrame, column: str) -> pd.Series:
    dtype = AMPLITUDE_COLUMNS[column][0]
    values = df[column] if column in df else [_blank(dtype)] * len(df)
    return pd.Series(pd.array(values, dtype=dtype), index=df.index, name=column)


def _equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(left == right)


def write_event_amplitude(
    df: pd.DataFrame, row_index: int, measurement: Mapping[str, Any],
) -> bool:
    """Write optional extras in-place at a POSITION, atomically on refusal.

    Returns whether values changed. Invalid input returns False without creating
    even empty columns. No core measurement or curation column is assigned.
    """
    try:
        position = _position(df, row_index)
        if not isinstance(measurement, Mapping) or set(measurement) != set(AMPLITUDE_COLUMNS):
            return False
        row = df.iloc[position].to_dict()
        row.update(measurement)
        if not amplitude_is_current(row):
            return False
        staged = {key: _optional_series(df, key) for key in AMPLITUDE_COLUMNS}
        changed = any(key not in df or not _equal(df[key].iloc[position], value)
                      for key, value in measurement.items())
        for key, value in measurement.items():
            staged[key].iloc[position] = value
    except (TypeError, ValueError, OverflowError, IndexError):
        return False
    if changed:
        for key, series in staged.items():
            df[key] = series
    return changed


def clear_event_amplitude(df: pd.DataFrame, row_index: int) -> bool:
    """Clear only this row's optional extras; retain optional columns."""
    try:
        position = _position(df, row_index)
        staged = {key: _optional_series(df, key) for key in AMPLITUDE_COLUMNS if key in df}
        changed = any(not pd.isna(series.iloc[position]) for series in staged.values())
        for key, series in staged.items():
            series.iloc[position] = _blank(AMPLITUDE_COLUMNS[key][0])
    except (TypeError, ValueError, OverflowError, IndexError):
        return False
    if changed:
        for key, series in staged.items():
            df[key] = series
    return changed


def invalidate_event_amplitudes(
    df: pd.DataFrame, rows: Sequence[int] | None = None, reason: str = "event_remeasured",
) -> int:
    """Blank stale numbers, retaining estimator/provenance as an audit trail.

    Rows are positional; an invalid list raises before changes occur. Legacy
    frames are no-ops, not migrations. Return newly invalidated row count.
    """
    if not any(key in df for key in AMPLITUDE_COLUMNS):
        return 0
    positions = list(range(len(df))) if rows is None else sorted({_position(df, row) for row in rows})
    _json_value(reason, identity=True)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An invalidation reason is required")
    targets = []
    for position in positions:
        row = df.iloc[position]
        existing = any(key in df and not pd.isna(row[key]) for key in AMPLITUDE_COLUMNS)
        has_numbers = any(key in df and not pd.isna(row[key]) for key in _NUMERIC_COLUMNS)
        if existing and (has_numbers or not _equal(row.get("amplitude_status"), "stale")):
            targets.append(position)
    if not targets:
        return 0
    staged = {key: _optional_series(df, key) for key in AMPLITUDE_COLUMNS if key in df}
    staged.setdefault("amplitude_status", _optional_series(df, "amplitude_status"))
    for position in targets:
        for key in _NUMERIC_COLUMNS:
            if key in staged:
                staged[key].iloc[position] = _blank(AMPLITUDE_COLUMNS[key][0])
        staged["amplitude_status"].iloc[position] = "stale"
        if "amplitude_provenance_json" in staged:
            try:
                metadata = json.loads(staged["amplitude_provenance_json"].iloc[position])
                if isinstance(metadata, dict):
                    metadata["invalidation_reason"] = reason
                    staged["amplitude_provenance_json"].iloc[position] = _dump(metadata)
            except (TypeError, ValueError):
                pass  # Keep malformed legacy provenance; never present it as current.
    for key, series in staged.items():
        df[key] = series
    return len(targets)
