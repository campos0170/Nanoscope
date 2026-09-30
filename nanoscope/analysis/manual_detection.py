"""Editable, independently executable stages for a prepared student recording.

The stages retain probabilities and all candidates. Their defaults reproduce the
original Zn StudentInference incubation contract, while every exposed setting is
consumed by its own stage. This module does not invoke the automated study runner.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import medfilt

from .cancel import CancelFlag, check_cancelled, propagate_cancel
from .filters import robust_normalize
from .incubation_detection import (
    PreparedRecording, RecordingAnalysis, REFERENCE_NOTEBOOK_SHA256,
    _fingerprint, _quiet_stats,
)

STAGE_VERSION = "student-manual-stages-v1"
IMPORTED_EVENTS_VERSION = "curated-workspace-events-v1"
TAIL_POLICIES = ("exclude_last_start", "include_last_start", "anchor_end")
_ATTR = "manual_detection"


def _number(value, label: str, *, minimum=0.0, positive=False) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not np.isfinite(value) or value < minimum or (positive and value <= minimum):
        relation = ">" if positive else ">="
        raise ValueError(f"{label} must be finite and {relation} {minimum:g}")
    return value


def _integer(value, label: str, *, minimum=1) -> int:
    number = _number(value, label, minimum=minimum)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


@dataclass(frozen=True)
class InferenceSettings:
    clip_sigma: float = 30.0
    window_samples: int = 2048
    stride_samples: int = 1024
    batch_size: int = 64
    tail_policy: str = "exclude_last_start"
    normalization: str = "robust_sigma"

    def __post_init__(self):
        object.__setattr__(self, "clip_sigma", _number(self.clip_sigma, "clip_sigma", positive=True))
        for name, minimum in (("window_samples", 2), ("stride_samples", 1), ("batch_size", 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        if self.stride_samples > self.window_samples:
            raise ValueError("stride_samples must not exceed window_samples")
        if self.tail_policy not in TAIL_POLICIES:
            raise ValueError(f"tail_policy must be one of {TAIL_POLICIES}")
        if self.normalization != "robust_sigma":
            raise ValueError("normalization must be robust_sigma (trimmed median/MAD)")


@dataclass(frozen=True)
class DecodeSettings:
    enter: float = 0.4
    exit: float = 0.15
    smooth: int = 3
    merge_ms: float = 0.05
    min_ms: float = 0.1

    def __post_init__(self):
        for name in ("enter", "exit", "merge_ms", "min_ms"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if not 0 <= self.exit <= self.enter <= 1 or self.enter == 0:
            raise ValueError("Probability thresholds must satisfy 0 <= exit <= enter <= 1 and enter > 0")
        object.__setattr__(self, "smooth", _integer(self.smooth, "smooth"))
        if self.smooth % 2 != 1:
            raise ValueError("smooth must be an odd positive number of samples; 1 disables smoothing")


@dataclass(frozen=True)
class MeasurementSettings:
    min_depth_sigma: float = 4.0
    i0_block_s: float = 0.1
    i0_window_s: float = 5.0

    def __post_init__(self):
        object.__setattr__(self, "min_depth_sigma", _number(self.min_depth_sigma, "min_depth_sigma"))
        for name in ("i0_block_s", "i0_window_s"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))


@dataclass
class InferenceResult:
    probability: np.ndarray
    coverage: dict
    provenance: dict


def _progress(callback, fraction, message):
    if callback is not None:
        callback(float(fraction), message)


def _array_hash(values: np.ndarray, *, cancel=None) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    if not array.size:
        return digest.hexdigest()
    data = memoryview(array).cast("B")
    for start in range(0, len(data), 1024 * 1024):
        check_cancelled(cancel)
        digest.update(data[start:start + 1024 * 1024])
    return digest.hexdigest()


def recording_identity(recording: PreparedRecording, *, cancel=None) -> dict:
    """Identity of the actual prepared arrays used by every dependent stage."""
    check_cancelled(cancel)
    if not isinstance(recording.tag, str) or not recording.tag.strip():
        raise ValueError("Recording requires a nonempty tag")
    fs = _number(recording.fs_hz, "sample rate", positive=True)
    raw, detrended = np.asarray(recording.raw), np.asarray(recording.detrended)
    if (raw.ndim != 1 or detrended.ndim != 1 or not len(raw)
            or len(raw) != len(detrended)):
        raise ValueError("Raw and prepared traces must be nonempty one-dimensional arrays of equal length")
    if any(a.dtype.kind not in "fiu" for a in (raw, detrended)):
        raise ValueError("Raw and prepared traces must contain real numeric current samples")
    if (not np.isfinite(raw).all() or not np.isfinite(detrended).all()
            or np.any(np.abs(raw) > np.finfo(np.float32).max)
            or np.any(np.abs(detrended) > np.finfo(np.float32).max)):
        raise ValueError("Raw and prepared traces must be finite and representable as float32")
    duration = _number(recording.duration_s, "recording duration", positive=True)
    if not np.isclose(duration, len(raw) / fs, rtol=1e-12, atol=1e-12):
        raise ValueError("Recording duration must equal sample count / sample rate")
    return {
        "tag": recording.tag, "n_samples": len(raw), "fs_hz": fs,
        "duration_s": duration,
        "raw_sha256": _array_hash(np.asarray(raw, np.float64), cancel=cancel),
        "detrended_sha256": _array_hash(np.asarray(detrended, np.float32), cancel=cancel),
        "array_hash_format": "dtype.str + JSON shape + contiguous native bytes",
    }


def _source_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for nested in value.values():
            yield from _source_records(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _source_records(nested)


def _verify_sources(provenance, *, cancel=None):
    seen = set()
    for original in _source_records(provenance):
        check_cancelled(cancel)
        key = (original["path"], original["sha256"])
        if key in seen:
            continue
        seen.add(key)
        current = _fingerprint(original["path"], cancel)
        if current["sha256"] != original["sha256"]:
            raise ValueError(f"Source changed since this stage was created: {Path(original['path']).name}")


def _assert_recording(recording, expected, *, cancel=None):
    if recording_identity(recording, cancel=cancel) != expected:
        raise ValueError("Prepared recording changed; rerun the dependent stages")


def _starts(n: int, settings: InferenceSettings):
    window, stride = settings.window_samples, settings.stride_samples
    if n < window:
        return []
    stop = n - window + (settings.tail_policy != "exclude_last_start")
    starts = list(range(0, stop, stride))
    if settings.tail_policy == "anchor_end" and (not starts or starts[-1] != n - window):
        starts.append(n - window)
    return starts


@propagate_cancel
def run_inference(recording: PreparedRecording, model_path, settings: InferenceSettings,
                  *, progress=None, cancel: CancelFlag | None = None) -> InferenceResult:
    """Infer and retain probabilities without decoding or measuring events.

    Every policy uses complete windows only. Inputs shorter than the selected
    model window have no coverage; ``exclude_last_start`` also excludes n == W.
    """
    if not isinstance(settings, InferenceSettings):
        raise TypeError("settings must be InferenceSettings")
    identity = recording_identity(recording, cancel=cancel)
    _verify_sources(recording.provenance, cancel=cancel)
    model_fp = _fingerprint(model_path, cancel)
    _progress(progress, 0.04, "Loading selected student model on CPU")
    from .ml_detect import clear_model_cache, load_keras_model
    clear_model_cache()
    wrapper = load_keras_model(model_fp["path"])
    check_cancelled(cancel)
    if wrapper.channels != 1:
        raise ValueError("Manual student inference requires a one-channel per-sample model")
    if int(wrapper.window) != settings.window_samples:
        raise ValueError(f"Selected window {settings.window_samples} does not match model input window {wrapper.window}")
    _progress(progress, 0.08, "Normalizing the complete prepared trace")
    normalized = robust_normalize(np.asarray(recording.detrended, np.float32),
                                  clip_sigma=settings.clip_sigma)
    check_cancelled(cancel)
    starts = _starts(len(normalized), settings)
    predictions = np.zeros(len(normalized), np.float32)
    counts = np.zeros(len(normalized), np.float32)
    window = settings.window_samples
    for first in range(0, len(starts), settings.batch_size):
        check_cancelled(cancel)
        batch_starts = starts[first:first + settings.batch_size]
        batch = np.asarray([normalized[s:s + window, None] for s in batch_starts], np.float32)
        values = np.asarray(wrapper.model.predict(batch, verbose=0), np.float32)
        check_cancelled(cancel)
        if values.shape != (len(batch_starts), window, 1):
            raise ValueError(f"Student must return (batch, window, 1), got {values.shape}")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Student returned non-finite or out-of-range probabilities")
        for start, output in zip(batch_starts, values[..., 0]):
            predictions[start:start + window] += output
            counts[start:start + window] += 1
        _progress(progress, 0.12 + 0.8 * (first + len(batch_starts)) / len(starts),
                  f"Student inference: {first + len(batch_starts):,}/{len(starts):,} windows")
    probability = predictions / np.maximum(counts, 1e-8)
    covered_stop = starts[-1] + window if starts else 0
    coverage = {
        "window_samples": window, "stride_samples": settings.stride_samples,
        "batch_size": settings.batch_size, "n_windows": len(starts),
        "tail_policy": settings.tail_policy, "short_input_policy": "no_padding; no complete window means zero coverage",
        "covered_start_idx": 0 if starts else None,
        "covered_stop_idx_exclusive": covered_stop,
        "uncovered_tail_samples": len(normalized) - covered_stop,
        "covered_sample_count": int((counts > 0).sum()),
        "max_overlap_count": int(counts.max()) if len(counts) else 0,
        "coverage_policy": settings.tail_policy,
    }
    _assert_recording(recording, identity, cancel=cancel)
    _verify_sources(recording.provenance, cancel=cancel)
    _verify_sources({"model": model_fp}, cancel=cancel)
    probability.setflags(write=False)
    provenance = {
        **deepcopy(recording.provenance), "stage_version": STAGE_VERSION,
        "stage": "inference", "recording_identity": identity,
        "model": model_fp, "inference_settings": asdict(settings), "inference": deepcopy(coverage),
        "probability_sha256": _array_hash(probability, cancel=cancel),
        "reference_notebook_sha256": REFERENCE_NOTEBOOK_SHA256,
        "cpu_only": True, "model_cache_policy": "clear_before_load; verify content before and after inference",
        "warnings": ([f"{coverage['uncovered_tail_samples']} tail samples have no model coverage; their probabilities are zero."]
                     if coverage["uncovered_tail_samples"] else []),
    }
    _progress(progress, 1.0, "Inference complete; inspect probabilities before decoding")
    return InferenceResult(probability, coverage, provenance)


def _geometry_hash(frame: pd.DataFrame, *, cancel=None):
    return _array_hash(frame[["start_idx", "end_idx"]].to_numpy(dtype=np.int64), cancel=cancel)


def _decode_sampling(fs_hz, settings, *, legacy=False):
    """Record sample rounding separately from the user's requested durations."""
    requested = fs_hz * settings.min_ms / 1000.0
    merge = fs_hz * settings.merge_ms / 1000.0
    if not np.isfinite(requested) or not np.isfinite(merge):
        raise ValueError("Decode durations exceed the supported sample range")
    if legacy:
        minimum = int(requested)
    else:
        # Treat only floating-point noise around an integer as exact. Ordinary
        # nonintegral durations must round up so no shorter event is admitted.
        nearest = round(requested)
        tolerance = 8 * math.ulp(requested)
        snapped = nearest if abs(requested - nearest) <= tolerance else requested
        minimum = max(1 if settings.min_ms > 0 else 0, math.ceil(snapped))
    return dict(fs_hz=fs_hz, minimum_samples=minimum,
        effective_minimum_ms=minimum / fs_hz * 1000.0,
        minimum_duration_rounding=("legacy_truncate" if legacy else "ceiling_with_8ulp_integer_tolerance"),
        minimum_integer_tolerance_ulps=0 if legacy else 8,
        merge_samples=int(merge), merge_rounding="truncate; inclusive end-index distance")


@propagate_cancel
def find_candidates(recording: PreparedRecording, inference: InferenceResult,
                    settings: DecodeSettings, *, cancel: CancelFlag | None = None) -> pd.DataFrame:
    """Decode all fired regions, retaining them before any amplitude gate."""
    if not isinstance(settings, DecodeSettings):
        raise TypeError("settings must be DecodeSettings")
    _assert_recording(recording, inference.provenance.get("recording_identity"), cancel=cancel)
    _verify_sources(inference.provenance, cancel=cancel)
    probability = np.asarray(inference.probability)
    if probability.shape != recording.detrended.shape or probability.dtype != np.float32:
        raise ValueError("Inference probability must be a float32 array matching the prepared recording")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Inference probability must contain finite values between zero and one")
    if _array_hash(probability, cancel=cancel) != inference.provenance.get("probability_sha256"):
        raise ValueError("Inference probabilities changed; rerun inference")
    smooth = medfilt(probability, settings.smooth) if len(probability) >= settings.smooth else probability
    check_cancelled(cancel)
    above_exit = (smooth >= settings.exit).astype(np.int8)
    above_enter = smooth >= settings.enter
    changes = np.diff(above_exit, prepend=0, append=0)
    regions = []
    sampling = _decode_sampling(recording.fs_hz, settings)
    merge_samples = sampling["merge_samples"]
    minimum = sampling["minimum_samples"]
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        check_cancelled(cancel)
        if not above_enter[start:stop].any():
            continue
        start, end = int(start), int(stop - 1)
        if regions and start - regions[-1][1] <= merge_samples:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))
    frame = pd.DataFrame([(s, e) for s, e in regions if e - s + 1 >= minimum],
                         columns=["start_idx", "end_idx"], dtype=np.int64)
    frame.attrs[_ATTR] = {
        "stage_version": STAGE_VERSION,
        "inference_provenance": deepcopy(inference.provenance),
        "decode_settings": asdict(settings),
        "decode_sampling": sampling,
        "candidate_geometry_sha256": _geometry_hash(frame, cancel=cancel),
        "index_convention": "start_idx and end_idx inclusive; end_s exclusive",
    }
    _assert_recording(recording, inference.provenance["recording_identity"], cancel=cancel)
    _verify_sources(inference.provenance, cancel=cancel)
    check_cancelled(cancel)
    return frame


def _local_i0(raw, quiet, fs, at, settings: MeasurementSettings, *, cancel=None):
    raw = np.asarray(raw, np.float32)
    block = max(int(settings.i0_block_s * fs), 1)
    n_blocks = max(len(raw) // block, 1)
    medians = np.empty(n_blocks, np.float64)
    for i in range(n_blocks):
        check_cancelled(cancel)
        sl = slice(i * block, (i + 1) * block)
        selected = quiet[sl]
        values = raw[sl][selected] if selected.any() else raw[sl]
        medians[i] = np.median(values) if values.size else np.nan
    good = np.isfinite(medians)
    if not good.any():
        raise ValueError("No finite raw-current block medians are available for local I0")
    medians = np.interp(np.arange(n_blocks), np.flatnonzero(good), medians[good])
    width = max(int(round(settings.i0_window_s * fs / block)), 1)
    if width % 2 == 0:
        width += 1
    if width > 1_000_001:
        raise ValueError("I0 window exceeds one million block samples; increase block duration or shorten the window")
    # The multidimensional path avoids SciPy's unsafe short-input 1-D rank-filter
    # optimization without allocating a width-sized sliding window per block.
    smooth = median_filter(medians[None, :], size=(1, width), mode="nearest")[0]
    centers = (np.arange(n_blocks) + 0.5) * block
    values = np.interp(np.asarray(at, float), centers, smooth).astype(np.float32)
    drift = float((np.abs(smooth).max() - np.abs(smooth).min()) / max(abs(np.median(smooth)), 1e-9))
    return np.abs(values), drift, {"block_samples": block, "n_complete_blocks": len(raw) // block,
                                 "effective_window_blocks": width, "effective_window_s": width * block / fs,
                                 "partial_final_block_policy": "omitted unless it is the only block",
                                 "interpolation": "block centers evaluated at candidate onsets",
                                 "quiet_empty_block_policy": "median of all raw samples in block"}


def _measure_bounds(recording, bounds, settings, *, progress=None, cancel=None):
    """Shared study measurements for already validated inclusive event bounds."""
    _progress(progress, 0.05, "Estimating the recording's quiet baseline and noise")
    detrended = np.asarray(recording.detrended, np.float32)
    baseline, noise, quiet = _quiet_stats(detrended)
    check_cancelled(cancel)
    local_i0, drift, i0_detail = _local_i0(recording.raw, quiet, recording.fs_hz,
        bounds[:, 0], settings, cancel=cancel)
    global_i0 = abs(float(np.median(np.asarray(recording.raw, np.float32))))
    if not np.isfinite(global_i0) or global_i0 <= 0:
        raise ValueError("Recording-wide raw median I0 is zero, negative, or non-finite")
    numeric = ["onset_s", "end_s", "dwell_ms", "duration_ms", "depth_peak_pA", "depth_sust_pA",
               "peak_sigma", "sust_sigma", "I0_local_pA", "I0_global_pA", "f_peak", "f_sust",
               "fracblock", "depth_pA", "depth_sigma", "depth_sust_sigma"]
    rows = []
    for index, (start, end) in enumerate(bounds):
        check_cancelled(cancel)
        body = detrended[start:end + 1]
        peak = baseline - float(body.min())
        sustained = baseline - float(np.median(body))
        denominator = float(local_i0[index])
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"Local raw I0 reference is zero, negative, or non-finite for candidate {index}")
        dwell = (end - start + 1) / recording.fs_hz * 1000.0
        passed = bool(sustained >= settings.min_depth_sigma * noise)
        rows.append(dict(tag=recording.tag, event_id=f"{recording.tag}:{index}",
            start_idx=int(start), end_idx=int(end), onset_s=start / recording.fs_hz,
            end_s=(end + 1) / recording.fs_hz, dwell_ms=dwell, duration_ms=dwell,
            depth_peak_pA=peak, depth_sust_pA=sustained, peak_sigma=peak / noise,
            sust_sigma=sustained / noise, I0_local_pA=float(local_i0[index]), I0_global_pA=global_i0,
            f_peak=peak / denominator, f_sust=sustained / denominator, fracblock=sustained / denominator,
            depth_pA=peak, depth_sigma=peak / noise, depth_sust_sigma=sustained / noise,
            passes_gate=passed, accepted=passed))
        if index % 64 == 0:
            _progress(progress, 0.2 + 0.65 * (index + 1) / max(len(bounds), 1),
                      f"Measuring candidate {index + 1:,}/{len(bounds):,}")
    catalog = pd.DataFrame(rows, columns=["tag", "event_id", "start_idx", "end_idx", *numeric,
                                         "passes_gate", "accepted"])
    for name in numeric:
        catalog[name] = catalog[name].astype(float)
    for name in ("start_idx", "end_idx"):
        catalog[name] = catalog[name].astype(np.int64)
    for name in ("passes_gate", "accepted"):
        catalog[name] = catalog[name].astype(bool)
    return catalog, baseline, noise, quiet, drift, i0_detail


@propagate_cancel
def measure_candidates(recording: PreparedRecording, candidates: pd.DataFrame,
                       settings: MeasurementSettings, *, progress=None,
                       cancel: CancelFlag | None = None) -> RecordingAnalysis:
    """Measure every candidate and initialize acceptance from the sustained gate.

    Candidates must come from ``find_candidates`` with their ``DataFrame.attrs``
    preserved. This prevents reusing probabilities or bounds on another trace.
    """
    if not isinstance(settings, MeasurementSettings):
        raise TypeError("settings must be MeasurementSettings")
    stage = candidates.attrs.get(_ATTR)
    if not isinstance(stage, dict) or stage.get("stage_version") != STAGE_VERSION:
        raise ValueError("Candidates require their decoding-stage provenance; run Find candidates again")
    previous = stage.get("inference_provenance", {})
    identity = previous.get("recording_identity")
    _assert_recording(recording, identity, cancel=cancel)
    _verify_sources(previous, cancel=cancel)
    if not {"start_idx", "end_idx"}.issubset(candidates.columns):
        raise ValueError("Candidates require start_idx and end_idx columns")
    for name in ("start_idx", "end_idx"):
        if candidates[name].dtype.kind not in "iu":
            raise ValueError("Candidate bounds must be integer sample indices")
    bounds = candidates[["start_idx", "end_idx"]].to_numpy(np.int64)
    if (np.any(bounds < 0) or np.any(bounds >= len(recording.detrended))
            or np.any(bounds[:, 1] < bounds[:, 0])
            or (len(bounds) > 1 and np.any(bounds[1:, 0] <= bounds[:-1, 1]))):
        raise ValueError("Candidate bounds must be ordered, non-overlapping inclusive intervals within the recording")
    if _geometry_hash(candidates, cancel=cancel) != stage.get("candidate_geometry_sha256"):
        raise ValueError("Candidate bounds changed after decoding; run Find candidates again")
    catalog, baseline, noise, quiet, drift, i0_detail = _measure_bounds(
        recording, bounds, settings, progress=progress, cancel=cancel)
    decode = stage["decode_settings"]
    model = previous["model"]
    stamp = dict(detector_file=Path(model["path"]).name, detector_md5=model["md5"][:12],
                 enter_thr=decode["enter"], exit_thr=decode["exit"], merge_ms=decode["merge_ms"],
                 min_event_ms=decode["min_ms"], min_depth_sigma=settings.min_depth_sigma,
                 window=previous["inference_settings"]["window_samples"],
                 i0_mode=f"rolling_median_{settings.i0_window_s:g}s")
    for name, value in stamp.items():
        catalog[name] = value
    _assert_recording(recording, identity, cancel=cancel)
    _verify_sources(previous, cancel=cancel)
    provenance = {
        **deepcopy(previous), "stage": "measurement", "recipe_id": STAGE_VERSION,
        "decode_settings": deepcopy(decode), "measurement_settings": asdict(settings),
        "decode_sampling": deepcopy(stage.get("decode_sampling")) if "decode_sampling" in stage else
            _decode_sampling(recording.fs_hz, DecodeSettings(**decode), legacy=True),
        "candidate_geometry_sha256": stage["candidate_geometry_sha256"],
        "index_convention": stage["index_convention"],
        "amplitude_baseline": "whole prepared trace, 5-sigma then 3-sigma quiet median",
        "gate_statistic": "baseline minus event median >= min_depth_sigma times quiet MAD",
        "baseline_pA": baseline, "noise_mad_pA": noise,
        "quiet_sample_count": int(quiet.sum()), "i0_relative_drift": drift,
        "i0_detail": i0_detail, "candidate_count": len(catalog),
        "accepted_count": int(catalog.accepted.sum()),
        "acceptance_source": "sustained-depth gate; no manual curation applied",
        "event_id_convention": "tag:zero-based candidate ordinal within this decoding stage",
    }
    catalog.attrs[_ATTR] = deepcopy(stage)
    check_cancelled(cancel)
    _progress(progress, 1.0, f"Measured {len(catalog):,} candidates; {provenance['accepted_count']:,} pass the gate")
    return RecordingAnalysis(recording.tag, recording.raw, recording.detrended, recording.fs_hz,
                             recording.duration_s, catalog, provenance)


def _imported_integer_column(frame, name):
    """Read native integer identities/bounds without float rounding or uint wrap."""
    values = frame[name]
    if not len(values):
        return np.empty(0, dtype=np.int64)
    if (values.isna().any() or not pd.api.types.is_integer_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)):
        raise ValueError(f"Imported {name} must contain integer values")
    if (values < 0).any() or (values > np.iinfo(np.int64).max).any():
        raise ValueError(f"Imported {name} must contain nonnegative int64 values")
    return values.to_numpy(dtype=np.int64, copy=True)


@propagate_cancel
def measure_imported_events(recording: PreparedRecording, events: pd.DataFrame,
                            settings: MeasurementSettings, *, source_provenance: dict,
                            progress=None, cancel: CancelFlag | None = None) -> RecordingAnalysis:
    """Measure a snapshot of native workspace events without running a detector.

    Native bounds are half-open and IDs are native nonnegative integers. The
    returned study catalog uses inclusive ends, stable ``tag:source:<event_id>``
    identities, and the source's explicit acceptance decisions. ``passes_gate``
    independently reports the study's sustained-depth gate; it never overwrites
    imported curation. Every original column is retained with a ``source_``
    prefix so native and incubation measurement definitions remain distinct.

    The GUI bridge verifies the source binding, calibration, recording clock and
    full-recording exposure before calling this function. Here we validate the
    prepared arrays and imported event contract and record a detached snapshot
    of that source evidence. Repeating measurement with new study settings keeps
    the imported geometry and curation unchanged.
    """
    if not isinstance(settings, MeasurementSettings):
        raise TypeError("settings must be MeasurementSettings")
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a DataFrame")
    if not isinstance(source_provenance, dict):
        raise TypeError("source_provenance must be a dictionary")
    if (not events.columns.is_unique
            or any(not isinstance(column, str) for column in events.columns)):
        raise ValueError("Imported event column names must be unique strings")
    required = {"event_id", "start_idx", "end_idx", "accepted"}
    if not required.issubset(events.columns):
        raise ValueError(f"Imported events require columns {sorted(required - set(events.columns))}")
    identity = recording_identity(recording, cancel=cancel)
    _verify_sources(recording.provenance, cancel=cancel)
    source = deepcopy(events)
    for column in source.columns:
        check_cancelled(cancel)
        if pd.api.types.is_object_dtype(source[column].dtype):
            source[column] = pd.Series([deepcopy(value) for value in source[column]],
                                       index=source.index, dtype=object)
    evidence = deepcopy(source_provenance)
    ids = _imported_integer_column(source, "event_id")
    starts = _imported_integer_column(source, "start_idx")
    stops = _imported_integer_column(source, "end_idx")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Imported event IDs must be unique within the source")
    accepted = source["accepted"]
    if accepted.isna().any() or any(not isinstance(v, (bool, np.bool_)) for v in accepted):
        raise ValueError("Imported accepted values must be explicit boolean decisions")
    if np.any(stops <= starts) or np.any(stops > len(recording.detrended)):
        raise ValueError("Imported bounds must be nonempty half-open intervals within the recording")
    order = np.argsort(starts, kind="stable")
    source = source.iloc[order].reset_index(drop=True)
    ids, starts, stops = ids[order], starts[order], stops[order]
    if len(starts) > 1 and np.any(starts[1:] < stops[:-1]):
        raise ValueError("Imported event bounds must not overlap")
    bounds = np.column_stack((starts, stops - 1))
    catalog, baseline, noise, quiet, drift, i0_detail = _measure_bounds(
        recording, bounds, settings, progress=progress, cancel=cancel)
    catalog["event_id"] = pd.Series(
        [f"{recording.tag}:source:{value}" for value in ids], dtype=object)
    catalog["accepted"] = source["accepted"].to_numpy(dtype=bool)
    for column in source.columns:
        check_cancelled(cancel)
        catalog["source_" + column] = source[column].copy(deep=True)
    # Keep the identifier column native even for an empty or nullable source.
    catalog["source_event_id"] = ids
    provenance = {
        **deepcopy(recording.provenance), "stage": "measurement",
        "recipe_id": IMPORTED_EVENTS_VERSION, "input_mode": "imported_events",
        "recording_identity": identity, "metadata": deepcopy(recording.metadata),
        "event_polarity": recording.provenance.get("event_polarity",
            recording.metadata.get("event_polarity", 1)),
        "measurement_settings": asdict(settings),
        "candidate_geometry_sha256": _geometry_hash(catalog, cancel=cancel),
        "index_convention": "start_idx and end_idx inclusive; end_s exclusive",
        "amplitude_baseline": "whole prepared trace, 5-sigma then 3-sigma quiet median",
        "gate_statistic": "baseline minus event median >= min_depth_sigma times quiet MAD",
        "baseline_pA": baseline, "noise_mad_pA": noise,
        "quiet_sample_count": int(quiet.sum()), "i0_relative_drift": drift,
        "i0_detail": i0_detail, "candidate_count": len(catalog),
        "accepted_count": int(catalog.accepted.sum()),
        "acceptance_source": "explicit imported workspace acceptance; study gate is advisory",
        "event_id_convention": "tag:source:native integer event ID; stable across remeasurement",
        "imported_events": {
            "version": IMPORTED_EVENTS_VERSION,
            "source_provenance": evidence,
            "source_index_convention": "start_idx inclusive; end_idx exclusive",
            "source_columns": {name: "source_" + name for name in source.columns},
            "source_event_ids_sha256": _array_hash(ids, cancel=cancel),
            "source_acceptance_sha256": _array_hash(catalog.accepted.to_numpy(), cancel=cancel),
            "order": "stable sort by start_idx; source IDs and row decisions preserved",
            "measurement_policy": "incubation measurements recomputed; original values retained in source_ columns",
        },
    }
    _assert_recording(recording, identity, cancel=cancel)
    _verify_sources(recording.provenance, cancel=cancel)
    check_cancelled(cancel)
    _progress(progress, 1., f"Measured {len(catalog):,} imported events; {provenance['accepted_count']:,} accepted by the source")
    return RecordingAnalysis(recording.tag, recording.raw, recording.detrended,
                             recording.fs_hz, recording.duration_s, catalog, provenance)
