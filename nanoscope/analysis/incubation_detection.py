"""Versioned port of the Zn student notebook's *incubation* detector.

This consumes prepared CSV data, not an ABF or a new preprocessing recipe. The
legacy sliding-window tail and inclusive merge arithmetic are deliberate: changing
either changes the scientific reference. Every candidate, including rejected ones,
is retained. Importing this module does not import TensorFlow.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import medfilt

from .cancel import CancelFlag, check_cancelled
from .filters import robust_normalize

Progress = Callable[[float, str], None]
RECIPE_ID = "zn-student-incubation-notebook-v1"
REFERENCE_NOTEBOOK_SHA256 = "92572b5aaaac28c360aacf05d113439a91e03bc30025e3f2f5fadae545469c01"
CONTRACT = {
    "enter": 0.4, "exit": 0.15, "merge_ms": 0.05, "min_ms": 0.1,
    "smooth": 3, "min_depth_sigma": 4.0, "clip_sigma": 30.0,
    "onset_sigma": 0, "offset_sigma": 0, "i0_window_s": 5.0,
    "i0_block_s": 0.1, "batch_size": 64,
    "window_policy": "range(0, n-window, window//2); no padding; uncovered predictions zero",
    "index_convention": "start_idx and end_idx inclusive; end_s exclusive",
}


@dataclass
class PreparedRecording:
    tag: str
    raw: np.ndarray  # true CSV sign, parsed float64 (also needed by notebook cell 51)
    detrended: np.ndarray  # oriented events-down, float32
    fs_hz: float
    duration_s: float
    metadata: dict
    provenance: dict


@dataclass
class RecordingAnalysis:
    tag: str
    raw: np.ndarray
    detrended: np.ndarray
    fs_hz: float
    duration_s: float
    catalog: pd.DataFrame
    provenance: dict


def _progress(callback: Progress | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(float(fraction), message)


def _fingerprint(path: str | Path, cancel: CancelFlag | None = None) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_cancelled(cancel)
            sha.update(chunk)
            md5.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path.name}")
    return {"path": str(path), "sha256": sha.hexdigest(), "md5": md5.hexdigest(),
            "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def _positive_finite(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive and finite") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _validate_time_chunk(time: np.ndarray, start_idx: int, fs: float) -> dict:
    """Check the sample clock at float64 or the notebook export's float32 precision.

    The preprocessing notebook casts time to float32 before pandas writes CSV.
    Its shortest decimal strings need not equal index/fs in float64 (and long
    traces can have repeated timestamps). Require the exact same float32 bins,
    not an arbitrary relative tolerance, while retaining the original tight
    float64 comparison for higher-precision prepared inputs.
    """
    expected = np.arange(start_idx, start_idx + len(time), dtype=np.float64) / fs
    if not np.isfinite(expected).all():
        raise ValueError("metadata fs_hz cannot describe a finite sample clock")
    error = np.abs(time - expected)
    precise = error <= max(1e-10, 1e-4 / fs)
    with np.errstate(over="ignore", invalid="ignore"):
        stored32 = time.astype(np.float32)
        expected32 = expected.astype(np.float32)
        quantized = np.isfinite(stored32) & np.isfinite(expected32) & (stored32 == expected32)
    bad = np.flatnonzero(~(precise | quantized))
    if len(bad):
        sample = start_idx + int(bad[0])
        raise ValueError(
            "retained_time_s must start at zero and match uniformly spaced metadata fs_hz "
            f"at sample {sample}, including the notebook's float32 export precision"
        )
    finite32 = expected32[np.isfinite(expected32)]
    return {
        "rows_requiring_float32_equivalence": int((~precise & quantized).sum()),
        "max_abs_deviation_s": float(error.max()) if len(error) else 0.0,
        "max_float32_quantum_s": float(np.spacing(finite32).max()) if len(finite32) else 0.0,
    }


def load_prepared_recording(trace_path, metadata_path, *, progress: Progress | None = None,
                            cancel: CancelFlag | None = None) -> PreparedRecording:
    """Validate/load prepared pA CSV and metadata, without applying any filters.

    Time must describe zero-origin, uniformly sampled data at metadata fs_hz,
    allowing the original preprocessing notebook's float32 CSV quantization.
    Event positions always use sample/fs, never the rounded CSV timestamp.
    """
    check_cancelled(cancel)
    trace_fp = _fingerprint(trace_path, cancel)
    meta_fp = _fingerprint(metadata_path, cancel)
    metadata = json.loads(Path(meta_fp["path"]).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Recording metadata must be a JSON object")
    tag = metadata.get("tag")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("Recording metadata requires a nonempty tag")
    fs = _positive_finite(metadata.get("fs_hz"), "metadata fs_hz")
    polarity = metadata.get("event_polarity", 1)
    if isinstance(polarity, bool) or polarity not in (-1, 1):
        raise ValueError("metadata event_polarity must be +1 or -1")
    if metadata.get("csv_file") and Path(str(metadata["csv_file"])).name != Path(trace_fp["path"]).name:
        raise ValueError("Metadata csv_file does not identify the selected prepared recording")
    expected_n = metadata.get("n_samples")
    if expected_n is not None and (isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n <= 0):
        raise ValueError("metadata n_samples must be a positive integer")
    raw_parts, detr_parts = [], []
    n = 0
    clock = {"rows_requiring_float32_equivalence": 0, "max_abs_deviation_s": 0.0,
             "max_float32_quantum_s": 0.0, "nonincreasing_source_steps": 0}
    previous_time = None
    # Was "time_s" -- export_prepared() writes that column as the ORIGINAL, pre-
    # truncation source time (source_indices(...)/fs), not a zero-origin clock, so any
    # recording prepared with SourceOptions.keep_start_s/keep_stop_s/exclude_s failed
    # the zero-origin check below on every reopen. "retained_time_s" is the column
    # export_prepared writes as the actual zero-origin, contiguous sample_index/fs
    # clock this validation (and the docstring above) require.
    required = ["retained_time_s", "current_raw_pA", "current_detr_pA"]
    _progress(progress, 0.0, f"Reading prepared recording {tag}")
    # Parse using pandas' notebook-default numeric parser, then cast, rather than
    # asking the CSV parser for float32 (which can round differently).
    with pd.read_csv(trace_fp["path"], compression="infer", usecols=required,
                     chunksize=250_000) as reader:
        for frame in reader:
            check_cancelled(cancel)
            try:
                time = frame["retained_time_s"].to_numpy(dtype=np.float64)
                raw = frame["current_raw_pA"].to_numpy(dtype=np.float64)
                detr = frame["current_detr_pA"].to_numpy(dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("Prepared time and current columns must be numeric") from exc
            if not all(np.isfinite(a).all() for a in (time, raw, detr)) or np.any(np.abs(raw) > np.finfo(np.float32).max):
                raise ValueError("Prepared recording contains NaN, infinity, or float32 overflow")
            chunk_clock = _validate_time_chunk(time, n, fs)
            clock["rows_requiring_float32_equivalence"] += chunk_clock["rows_requiring_float32_equivalence"]
            for name in ("max_abs_deviation_s", "max_float32_quantum_s"):
                clock[name] = max(clock[name], chunk_clock[name])
            if len(time):
                clock["nonincreasing_source_steps"] += int((np.diff(time) <= 0).sum())
                if previous_time is not None:
                    clock["nonincreasing_source_steps"] += int(time[0] <= previous_time)
                previous_time = float(time[-1])
            raw_parts.append(raw)
            detr_parts.append(detr)
            n += len(time)
            _progress(progress, min(n / expected_n, 0.99) if expected_n else 0.0,
                      f"Read {n:,} prepared samples from {tag}")
    if n == 0:
        raise ValueError("Prepared recording is empty")
    if expected_n is not None and n != expected_n:
        raise ValueError(f"Metadata n_samples={expected_n} but prepared recording contains {n}")
    duration = n / fs
    if "duration_s" in metadata:
        declared = _positive_finite(metadata["duration_s"], "metadata duration_s")
        if not np.isclose(declared, duration, rtol=1e-9, atol=1e-9):
            raise ValueError("Metadata duration_s does not equal n_samples/fs_hz")
    raw = np.concatenate(raw_parts)
    detrended = np.concatenate(detr_parts)
    if polarity == -1:
        detrended = -detrended
    clock.update(mode="float32_export_equivalent" if clock["rows_requiring_float32_equivalence"] else "float64_compatible",
                 analysis_clock="sample_idx / metadata fs_hz; rounded CSV time does not position events",
                 max_abs_deviation_samples=clock["max_abs_deviation_s"] * fs,
                 precision_limit="Irregularity within a float32 timestamp bin is not distinguishable from exporter quantization.")
    provenance = {"trace": trace_fp, "metadata_file": meta_fp, "metadata": metadata,
                  "n_samples": n, "fs_hz": fs, "duration_s": duration,
                  "event_polarity": int(polarity), "detrended_orientation": "events_down",
                  "raw_orientation": "unchanged_csv_sign", "preprocessing": "prepared_csv_unchanged",
                  "time_validation": "zero-origin sample/fs: tight float64 tolerance or exact notebook float32 export equivalence",
                  "time_validation_detail": clock}
    check_cancelled(cancel)
    _progress(progress, 1.0, f"Prepared recording {tag} validated")
    return PreparedRecording(tag, raw, detrended, fs, duration, metadata, provenance)


def _predict_notebook(model, signal: np.ndarray, window: int, *, progress=None, cancel=None):
    """Literal notebook overlap arithmetic, including its uncovered tail."""
    if window < 2:
        raise ValueError("Notebook-compatible inference requires a fixed model window >= 2")
    check_cancelled(cancel)
    normalized = robust_normalize(signal, clip_sigma=30.0)
    check_cancelled(cancel)
    n = len(signal)
    predictions = np.zeros(n, np.float32)
    counts = np.zeros(n, np.float32)
    starts = range(0, n - window, window // 2)
    total = len(starts)
    batch, indices = [], []
    completed = 0

    def flush():
        nonlocal completed
        if not batch:
            return
        check_cancelled(cancel)
        outputs = np.asarray(model.predict(np.asarray(batch, np.float32), verbose=0))
        check_cancelled(cancel)
        if outputs.shape != (len(batch), window, 1):
            raise ValueError(f"Student must return (batch, window, 1), got {outputs.shape}")
        if not np.isfinite(outputs).all() or np.any((outputs < 0) | (outputs > 1)):
            raise ValueError("Student returned non-finite or out-of-range probabilities")
        for start, values in zip(indices, outputs[..., 0]):
            predictions[start:start + window] += values
            counts[start:start + window] += 1
        completed += len(batch)
        _progress(progress, completed / total, f"Student inference: {completed:,}/{total:,} windows")
        batch.clear()
        indices.clear()

    for start in starts:
        check_cancelled(cancel)
        batch.append(normalized[start:start + window][:, None])
        indices.append(start)
        if len(batch) == 64:
            flush()
    flush()
    check_cancelled(cancel)
    covered_stop = starts[-1] + window if total else 0
    return predictions / np.maximum(counts, 1e-8), {
        "window_samples": window, "stride_samples": window // 2, "n_windows": total,
        "covered_start_idx": 0 if total else None, "covered_stop_idx_exclusive": covered_stop,
        "uncovered_tail_samples": n - covered_stop,
        "coverage_policy": CONTRACT["window_policy"],
    }


def _student_regions(probability: np.ndarray, fs: float) -> list[tuple[int, int]]:
    smooth = medfilt(probability, kernel_size=3) if len(probability) >= 3 else probability
    above_exit = (smooth >= 0.15).astype(np.int8)
    above_enter = smooth >= 0.4
    diff = np.diff(above_exit, prepend=0, append=0)
    candidates = [(int(s), int(e - 1)) for s, e in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1))
                  if above_enter[s:e].any()]
    merged = []
    merge_samples = int(fs * 0.05 / 1000.0)
    for start, end in candidates:
        if merged and start - merged[-1][1] <= merge_samples:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    minimum = int(fs * 0.1 / 1000.0)
    return [(s, e) for s, e in merged if e - s + 1 >= minimum]


def _quiet_stats(signal: np.ndarray):
    # This is the incubation gate's estimator, not normalization's guarded one.
    m0 = float(np.median(signal))
    s0 = 1.4826 * float(np.median(np.abs(signal - m0)))
    k0 = np.abs(signal - m0) < 5 * s0
    if not k0.any():
        raise ValueError("Prepared recording has no measurable quiet noise for the notebook depth gate")
    m1 = float(np.median(signal[k0]))
    s1 = 1.4826 * float(np.median(np.abs(signal[k0] - m1)))
    quiet = np.abs(signal - m1) < 3 * s1
    if not quiet.any():
        raise ValueError("Prepared recording has no quiet samples after the notebook noise trim")
    baseline = float(np.median(signal[quiet]))
    noise = 1.4826 * float(np.median(np.abs(signal[quiet] - baseline)))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Prepared recording has no positive finite noise for the notebook depth gate")
    return baseline, noise, quiet


def _local_i0(raw, quiet, fs, at, *, cancel=None):
    # Notebook local_I0: omit a final partial block (unless it is the only block).
    raw = np.asarray(raw, np.float32)
    block = max(int(0.1 * fs), 1)
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
        global_median = float(np.median(raw))
        return np.full(len(at), global_median, np.float32), global_median, 0.0
    medians = np.interp(np.arange(n_blocks), np.flatnonzero(good), medians[good])
    width = max(int(round(5.0 * fs / block)), 1)
    if width % 2 == 0:
        width += 1
    if len(medians) < width:
        # SciPy 1.15's 1-D rank-filter fast path can corrupt memory for a
        # footprint larger than this short input. Explicit nearest padding has
        # exactly the same odd-window median definition, including at edges.
        padded = np.pad(medians, width // 2, mode="edge")
        smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=-1)
    else:
        # THE (1, w) FORM IS NOT A STYLE CHOICE: scipy.ndimage.median_filter 1.15 is
        # non-deterministic on a plain 1-D input above a footprint threshold (same
        # issue the branch above works around for short inputs) -- see the identical
        # workaround in event_features.py, ml_preprocess.py and manual_detection.py.
        smooth = median_filter(medians[None, :], size=(1, width), mode="nearest")[0]
    centers = (np.arange(n_blocks) + 0.5) * block
    drift = float((np.abs(smooth).max() - np.abs(smooth).min()) / max(abs(np.median(smooth)), 1e-9))
    return np.interp(np.asarray(at, float), centers, smooth).astype(np.float32), float(np.median(smooth)), drift


_NUMERIC_COLUMNS = ["onset_s", "end_s", "dwell_ms", "duration_ms", "depth_peak_pA", "depth_sust_pA",
                    "peak_sigma", "sust_sigma", "I0_local_pA", "I0_global_pA", "f_peak", "f_sust",
                    "fracblock", "depth_pA", "depth_sigma", "depth_sust_sigma"]


def _build_catalog(recording: PreparedRecording, regions, baseline, noise, quiet, *, cancel=None):
    onsets = np.asarray([s for s, _ in regions], dtype=int)
    local_i0, _, drift = _local_i0(recording.raw, quiet, recording.fs_hz, onsets, cancel=cancel)
    local_i0 = np.abs(local_i0)
    global_i0 = abs(float(np.median(recording.raw.astype(np.float32))))
    if not np.isfinite(global_i0) or global_i0 <= 0:
        raise ValueError(
            f"{recording.tag}: the recording's whole-trace median raw current is at/near "
            "zero pA, so there is no usable open-pore reference for fractional depth.")
    rows = []
    for i, (start, end) in enumerate(regions):
        check_cancelled(cancel)
        body = recording.detrended[start:end + 1]
        peak = baseline - float(body.min())
        sustained = baseline - float(np.median(body))
        if not np.isfinite(local_i0[i]) or local_i0[i] <= 0:
            raise ValueError(
                f"{recording.tag}: the candidate at sample {start} has a zero/non-finite "
                "local open-pore reference, so its fractional depth cannot be measured.")
        denominator = float(local_i0[i])
        dwell = (end - start + 1) / recording.fs_hz * 1000.0
        rows.append(dict(tag=recording.tag, start_idx=start, end_idx=end,
                         onset_s=start / recording.fs_hz, end_s=(end + 1) / recording.fs_hz,
                         dwell_ms=dwell, duration_ms=dwell, depth_peak_pA=peak, depth_sust_pA=sustained,
                         peak_sigma=peak / noise, sust_sigma=sustained / noise,
                         I0_local_pA=float(local_i0[i]), I0_global_pA=global_i0,
                         f_peak=peak / denominator, f_sust=sustained / denominator,
                         fracblock=sustained / denominator, passes_gate=bool(sustained >= 4.0 * noise),
                         depth_pA=peak, depth_sigma=peak / noise, depth_sust_sigma=sustained / noise))
    catalog = pd.DataFrame(rows, columns=["tag", "start_idx", "end_idx", *_NUMERIC_COLUMNS, "passes_gate"])
    for column in _NUMERIC_COLUMNS:
        catalog[column] = catalog[column].astype(float)
    for column in ("start_idx", "end_idx"):
        catalog[column] = catalog[column].astype(np.int64)
    catalog["passes_gate"] = catalog["passes_gate"].astype(bool)
    return catalog, drift


def analyze_recording(trace_path, metadata_path, model_path, *, progress: Progress | None = None,
                      cancel: CancelFlag | None = None) -> RecordingAnalysis:
    """Run the frozen student's original incubation recipe, retaining all candidates.

    ``progress(fraction, message)`` receives 0…1 progress. ``cancel`` accepts a
    zero-argument predicate or threading.Event and raises OperationCancelled.
    Inputs are hashed and rechecked before publication; no reference catalog is read.
    """
    recording = load_prepared_recording(trace_path, metadata_path,
        progress=lambda f, m: _progress(progress, 0.18 * f, m), cancel=cancel)
    check_cancelled(cancel)
    model_fp = _fingerprint(model_path, cancel)
    _progress(progress, 0.19, "Loading the frozen student on CPU")
    from .ml_detect import clear_model_cache, load_keras_model  # lazy; loader forces CPU
    # The generic interactive cache is keyed by path/mtime, not file content.
    # A restored timestamp must not pair new provenance with stale model weights.
    # Active workers own their wrappers independently of this cache.
    clear_model_cache()
    wrapper = load_keras_model(model_fp["path"])
    if wrapper.channels != 1:
        raise ValueError("The incubation recipe requires a one-channel student model")
    check_cancelled(cancel)
    baseline, noise, quiet = _quiet_stats(recording.detrended)
    check_cancelled(cancel)
    probability, coverage = _predict_notebook(wrapper.model, recording.detrended, wrapper.window,
        progress=lambda f, m: _progress(progress, 0.22 + 0.63 * f, m), cancel=cancel)
    _progress(progress, 0.86, "Measuring every candidate and its sustained-depth gate")
    regions = _student_regions(probability, recording.fs_hz)
    check_cancelled(cancel)
    catalog, drift = _build_catalog(recording, regions, baseline, noise, quiet, cancel=cancel)
    stamp = dict(detector_file=Path(model_fp["path"]).name, detector_md5=model_fp["md5"][:12],
                 enter_thr=0.4, exit_thr=0.15, merge_ms=0.05, min_event_ms=0.1,
                 min_depth_sigma=4.0, window=wrapper.window, i0_mode="rolling_median_5s")
    for key, value in stamp.items():
        catalog[key] = value
    _progress(progress, 0.95, "Verifying source identities before publishing results")
    for original in (recording.provenance["trace"], recording.provenance["metadata_file"], model_fp):
        current = _fingerprint(original["path"], cancel)
        if current["sha256"] != original["sha256"]:
            raise ValueError(f"Input changed during analysis: {Path(original['path']).name}")
    provenance = {**recording.provenance, "model": model_fp, "recipe_id": RECIPE_ID,
                  "reference_notebook_sha256": REFERENCE_NOTEBOOK_SHA256,
                  "reference_cells": [1, 4, 5, 6, 7, 15], "contract": dict(CONTRACT),
                  "inference": coverage, "cpu_only": True,
                  "model_cache_policy": "clear_before_load; fresh weights for hashed model bytes",
                  "baseline_pA": baseline, "noise_mad_pA": noise,
                  "quiet_sample_count": int(quiet.sum()), "i0_relative_drift": drift,
                  "candidate_count": len(catalog), "accepted_count": int(catalog.passes_gate.sum()),
                  "warnings": [f"Notebook v1 leaves {coverage['uncovered_tail_samples']} tail samples without model coverage; predictions there are zero."]
                              if coverage["uncovered_tail_samples"] else []}
    check_cancelled(cancel)
    _progress(progress, 1.0, f"{recording.tag}: {provenance['accepted_count']:,} accepted events")
    return RecordingAnalysis(recording.tag, recording.raw, recording.detrended, recording.fs_hz,
                             recording.duration_s, catalog, provenance)
