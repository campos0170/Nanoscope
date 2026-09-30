"""Interactive raw-recording preparation, independent of inference or study recipes.

All parameters are supplied by the caller. The notebook-compatible reader choice
changes only ADC arithmetic precision; the application's ordinary ABF reader and
its historical results retain their existing scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .cancel import CancelFlag, check_cancelled
from .incubation_detection import PreparedRecording, Progress, _fingerprint, _progress
from .ml_preprocess import (MLPreprocessConfig, baseline_masked_median,
                            baseline_robust_median, baseline_savgol,
                            despike_median, notch_sequential)
from ..io.abf import AbfReader
from .study_signal import SourceOptions, retain_samples, shrinking_baseline, source_indices

READER_SCALINGS = ("notebook_float32", "standard")


@dataclass
class RawPreparedRecording(PreparedRecording):
    """Prepared data plus true notched signal and baseline, both in native pA sign."""
    filtered: np.ndarray
    baseline: np.ndarray


def _tag(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Tag must start with a letter or digit and contain only letters, digits, _, - or .")
    if value.endswith(".") or len(value) > 120:
        raise ValueError("Tag must be at most 120 characters and must not end with a dot")
    return value


def _reader(path) -> AbfReader:
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".abf":
        raise ValueError("Raw preparation currently accepts ABF recordings")
    return AbfReader(path)


def _sample_rate(rate) -> float:
    if rate is None or not np.isfinite(rate) or rate <= 0:
        raise ValueError("Raw preparation requires a positive known sample rate")
    rate = float(rate)
    nearest = round(rate)
    # Neo's reciprocal interval can report 50000.00000000001 for an exact 50 kHz
    # ABF clock. Remove representational error only, never round a fractional rate.
    if nearest > 0 and abs(rate - nearest) <= 16 * np.spacing(float(nearest)):
        return float(nearest)
    return rate


def inspect_raw(path, *, cancel: CancelFlag | None = None) -> dict:
    """Inspect channel identities and the complete raw source without loading samples."""
    before = _fingerprint(path, cancel)
    reader = _reader(before["path"])
    meta = reader.probe().meta
    after = _fingerprint(before["path"], cancel)
    if before != after:
        raise ValueError("Raw source changed while inspecting it")
    return {"path": before["path"], "source": before, "raw_file": before,
            "fs_hz": _sample_rate(meta.sample_rate_hz),
            "reported_fs_hz": meta.sample_rate_hz, "n_samples": int(meta.n_samples),
            "t0_s": float(meta.t0_s), "format": meta.format_name,
            "channels": [{"index": c.index, "name": c.name, "units": c.raw_units}
                         for c in meta.channels],
            "notes": list(meta.notes), "issues": list(meta.issues)}


def _read_channel(reader, info, channel, scaling, progress, cancel):
    """Use Neo's validated ABF decoding, choosing ADC arithmetic explicitly."""
    native = reader._open_reader()
    if int(native.segment_count(0)) != 1:
        raise ValueError("Prepare one continuous recording at a time; this ABF contains multiple sweeps")
    n = info["n_samples"]
    result = np.empty(n, np.float32)
    descriptors = native.header["signal_channels"]
    gain, offset = descriptors[channel]["gain"], descriptors[channel]["offset"]
    for start in range(0, n, 250_000):
        check_cancelled(cancel)
        stop = min(n, start + 250_000)
        counts = native.get_analogsignal_chunk(block_index=0, seg_index=0,
                                               i_start=start, i_stop=stop)
        if scaling == "notebook_float32":
            # pyabf's scalar multiply/add on float32 data uses float32 arithmetic.
            # Neo's float64 gain ARRAY gives a wider product rounded afterwards.
            # A one-ULP ADC difference can change a nonlinear baseline mask.
            values = counts[:, channel].astype(np.float32)
            values = np.multiply(values, np.float32(gain))
            values = np.add(values, np.float32(offset))
        else:
            values = native.rescale_signal_raw_to_float(counts, dtype="float32")[:, channel]
        if len(values) != stop - start or not np.isfinite(values).all():
            raise ValueError("Raw current is truncated or contains NaN/Inf")
        result[start:stop] = values
        _progress(progress, 0.05 + 0.15 * stop / n, f"Read {stop:,} of {n:,} raw samples")
    return result


def _csv_raw_values(raw, cancel=None):
    """Retain the reference CSV's shortest float32 decimal as a float64 value.

    The original notebook subsequently parses raw current in float64 for its
    population appendix. Casting the binary float32 directly to float64 would
    differ from the decimal export it actually consumed. This round trip changes
    no float32 current value and is explicitly recorded in provenance.
    """
    result = np.empty(len(raw), np.float64)
    for start in range(0, len(raw), 250_000):
        check_cancelled(cancel)
        stop = min(len(raw), start + 250_000)
        result[start:stop] = raw[start:stop].astype(str).astype(np.float64)
    return result


def _quality(detrended, fs):
    median = np.median(detrended)
    noise = 1.4826 * np.median(np.abs(detrended - median))
    mask = binary_dilation(detrended < median - 3.5 * noise,
                           iterations=max(1, int(0.2e-3 * fs)))
    quiet = detrended[~mask]
    w = max(1, int(0.5 * fs)); n = len(quiet) // w * w
    medians = np.median(quiet[:n].reshape(-1, w), axis=1) if n else np.array([0.0])
    return {"qc_mad_noise_pA": float(noise),
            "qc_open_pore_level_pA": float(np.median(quiet)) if len(quiet) else None,
            "qc_open_pore_drift_pA": float(medians.std()),
            "qc_overshoot": float((detrended > 4 * noise).sum() /
                                  max(int((detrended < -4 * noise).sum()), 1)),
            "qc_quiet_fraction": float((~mask).mean()),
            "qc_method": "3.5 MAD downward mask, 0.2 ms dilation; drift from groups of 0.5*fs unmasked samples"}


def prepare_raw(path, channel: int, tag: str, config: MLPreprocessConfig, *,
                event_polarity: int = 1, reader_scaling: str = "notebook_float32",
                source_options: SourceOptions | None = None,
                progress: Progress | None = None,
                cancel: CancelFlag | None = None) -> RawPreparedRecording:
    """Prepare the whole selected raw pA channel using user-supplied settings.

    No model normalization is applied. Native raw, filtered and baseline signs
    are retained; only ``detrended`` is oriented events-down for analysis.
    Cancellation is checked between filters, baseline operations and I/O chunks;
    a SciPy median call itself finishes before cancellation can be observed.
    """
    tag = _tag(tag)
    source_options = source_options or SourceOptions()
    if not isinstance(source_options, SourceOptions): raise TypeError("source_options must be SourceOptions")
    if not isinstance(config, MLPreprocessConfig):
        raise TypeError("config must be an MLPreprocessConfig")
    if isinstance(event_polarity, bool) or event_polarity not in (-1, 1):
        raise ValueError("event_polarity must be +1 (downward) or -1 (upward)")
    if reader_scaling not in READER_SCALINGS:
        raise ValueError(f"reader_scaling must be one of {READER_SCALINGS}")
    check_cancelled(cancel)
    _progress(progress, 0.0, "Inspecting raw source")
    info = inspect_raw(path, cancel=cancel)
    if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)) or not 0 <= channel < len(info["channels"]):
        raise ValueError("Select an existing current channel index")
    selected = info["channels"][int(channel)]
    if selected["units"] != "pA":
        raise ValueError(f"Selected channel {selected['name']} declares {selected['units']!r}; preparation requires current in pA")
    fs = info["fs_hz"]
    if config.notch_enabled and any(f >= fs / 2 for f in config.notch_freqs):
        raise ValueError("Every enabled notch line must be strictly below Nyquist")
    raw32 = _read_channel(_reader(path), info, int(channel), reader_scaling, progress, cancel)
    if source_options.add_channel:
        other = source_options.add_channel - 1
        if other == channel or not 0 <= other < len(info['channels']):
            raise ValueError('Additional current channel must exist and differ from the selected channel')
        if info['channels'][other]['units'] != 'pA' and not source_options.add_channel_as_pa:
            raise ValueError('Additional channel is not labeled pA. Verify its physical meaning, then explicitly enable the pA interpretation override for a mislabeled process export.')
        addition = _read_channel(_reader(path),info,other,reader_scaling,progress,cancel)
        raw32 = (raw32.astype(np.float64)+addition).astype(np.float32)
        del addition
    raw32, retained_ranges = retain_samples(raw32,fs,source_options)
    if len(raw32) == 0:
        raise ValueError("Raw recording is empty")
    filtered = raw32.copy()
    lines = config.notch_freqs if config.notch_enabled else ()
    for index, line in enumerate(lines):
        check_cancelled(cancel)
        filtered = notch_sequential(filtered, fs, (line,), config.notch_q)
        if not np.isfinite(filtered).all():
            raise ValueError("Notch filtering produced NaN/Inf")
        _progress(progress, 0.2 + 0.2 * (index + 1) / len(lines), f"Removed line {line:g} Hz")
    if lines and config.notch_float32:
        filtered = filtered.astype(np.float32)
    check_cancelled(cancel)
    _progress(progress, 0.45, "Estimating the event-masked baseline" if config.baseline_method == "masked_median" else "Estimating baseline")
    oriented = filtered if event_polarity == 1 else -filtered
    method = config.baseline_method
    if source_options.median_engine == "shrinking" and method in ("masked_median", "robust_median"):
        detr, base = shrinking_baseline(oriented, fs, config, cancel=cancel)
    elif method == "masked_median":
        detr, base = baseline_masked_median(oriented, fs, rough_s=config.masked_rough_s,
                    base_s=config.masked_base_s, decim=config.baseline_decim,
                    mask_sigma=config.masked_mask_sigma, dilate_ms=config.masked_dilate_ms)
    elif method == "robust_median":
        detr, base = baseline_robust_median(oriented, fs,
                    window_s=config.baseline_window_s, decim=config.baseline_decim)
    elif method == "savgol":
        detr, base = baseline_savgol(oriented, fs, window_s=config.baseline_window_s,
                                    poly=config.baseline_poly)
    else:
        detr, base = oriented.astype(np.float32), np.zeros(len(oriented), np.float32)
    check_cancelled(cancel)
    detr = np.asarray(despike_median(detr, size=config.despike_size), np.float32)
    baseline = base if event_polarity == 1 else -base
    if not np.isfinite(detr).all() or not np.isfinite(baseline).all():
        raise ValueError("Baseline preparation produced NaN/Inf")
    _progress(progress, 0.8, "Checking prepared current and source identity")
    qc = _quality(detr, fs)
    raw = _csv_raw_values(raw32, cancel)
    if _fingerprint(path, cancel) != info["source"]:
        raise ValueError("Raw source changed during preparation")
    params = dict(rough_s=config.masked_rough_s, base_s=config.masked_base_s,
                  decim=config.baseline_decim, mask_sigma=config.masked_mask_sigma,
                  dilate_ms=config.masked_dilate_ms)
    metadata = {"tag": tag, "filename": Path(info["path"]).name, "fs_hz": fs,
        "n_samples": len(raw), "duration_s": len(raw) / fs,
        "abf_channel": int(channel), "channel_name": selected["name"], "units": "pA",
        "n_channels": len(info["channels"]), "reader": f"NanoScope ABF ({reader_scaling})",
        "reader_scaling": reader_scaling, "event_polarity": int(event_polarity),
        "truncate_s": None, "segment_s": [source_options.keep_start_s, source_options.keep_stop_s or info["n_samples"]/fs], "derived_from": None,
        "retained_ranges": retained_ranges, "source_options": source_options.__dict__,
        "additional_channel_header_units": info["channels"][source_options.add_channel-1]["units"] if source_options.add_channel else None,
        "additional_channel_interpreted_units": "pA" if source_options.add_channel else None,
        "exclude_s": source_options.exclusions(), "median_engine": source_options.median_engine,
        "raw_operation": f"channel {channel+1}" + (f" + channel {source_options.add_channel}" if source_options.add_channel else ""),
        "instrument_origin_s": info["t0_s"], "raw_source": info["source"],
        "baseline_method": method, "masked_median_params": params if method == "masked_median" else None,
        "sg_window_s": config.baseline_window_s if method != "masked_median" else None,
        "sg_poly": config.baseline_poly, "despike_samples": config.despike_size,
        "notch_hz": list(lines), "notch_q": config.notch_q,
        "notch_float32": config.notch_float32,
        "raw_mean_pA": float(raw32.mean()), "raw_std_pA": float(raw32.std()),
        "detr_mean_pA": float((event_polarity * detr).mean()),
        "detr_std_pA": float(detr.std()), "csv_file": f"{tag}.csv.gz", **qc}
    provenance = {"raw_file": info["source"], "source_info": info,
        "preprocessing": "manual_raw_preparation_v1", "preparation_config": config.to_dict(),
        "normalization_applied": False, "reader_scaling": reader_scaling,
        "raw_numeric_representation": "native float32 shortest CSV decimal parsed to float64",
        "detrended_orientation": "events_down", "raw_orientation": "unchanged_source_sign",
        "event_polarity": int(event_polarity), "n_samples": len(raw),
        "fs_hz": fs, "duration_s": len(raw) / fs, "metadata": metadata,
        "baseline_orientation": "unchanged_source_sign", "calibration": {"gain": 1.0, "offset": 0.0},
        "sample_range": [0, len(raw)], "retained_ranges": retained_ranges, "source_unchanged_after_preparation": True}
    check_cancelled(cancel)
    _progress(progress, 1.0, f"Prepared {tag}: {len(raw):,} samples")
    return RawPreparedRecording(tag, raw, detr, fs, len(raw) / fs, metadata,
                                provenance, filtered, baseline)


def export_prepared(recording: PreparedRecording, folder, *,
                    progress: Progress | None = None,
                    cancel: CancelFlag | None = None) -> tuple[Path, Path]:
    """Export a new notebook-compatible trace/metadata pair, never overwrite files."""
    tag = _tag(recording.tag)
    check_cancelled(cancel)
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    trace, metadata_path = folder / f"{tag}.csv.gz", folder / f"{tag}_metadata.json"
    if trace.exists() or metadata_path.exists():
        raise FileExistsError("Prepared output already exists; choose a new tag or folder")
    raw = np.asarray(recording.raw, np.float32)
    detr = np.asarray(recording.detrended, np.float32)
    polarity = recording.metadata.get("event_polarity", 1)
    if isinstance(polarity, bool) or polarity not in (-1, 1):
        raise ValueError("Prepared event_polarity must be +1 or -1")
    if raw.ndim != 1 or len(raw) == 0 or detr.shape != raw.shape:
        raise ValueError("Prepared raw and detrended arrays must have the same nonempty shape")
    if not np.isfinite(raw).all() or not np.isfinite(detr).all():
        raise ValueError("Prepared currents must contain only finite samples")
    fs = _sample_rate(recording.fs_hz)
    baseline = getattr(recording, "baseline", None)
    if baseline is not None and (np.shape(baseline) != raw.shape or not np.isfinite(baseline).all()):
        raise ValueError("Prepared baseline must be finite and match the current array")
    metadata = dict(recording.metadata, tag=tag, fs_hz=fs, n_samples=len(raw),
                    duration_s=len(raw) / fs, csv_file=trace.name,
                    preparation_provenance=recording.provenance)
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=folder) as stage:
            stage = Path(stage); staged_trace = stage / trace.name
            # One compression stream: zero timestamp avoids changing the gzip header
            # between identical exports. Chunking polls cancellation during large CSV I/O.
            import gzip
            with staged_trace.open("wb") as binary:
                with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as zipped:
                    import io
                    with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                        for start in range(0, len(raw), 250_000):
                            check_cancelled(cancel)
                            stop = min(len(raw), start + 250_000)
                            columns = {"time_s": source_indices(recording, np.arange(start, stop)) / fs,
                                       "retained_time_s": np.arange(start, stop) / fs,
                                "current_raw_pA": raw[start:stop],
                                "current_detr_pA": polarity * detr[start:stop]}
                            if baseline is not None:
                                columns["baseline_pA"] = np.asarray(baseline[start:stop], np.float32)
                            pd.DataFrame(columns).to_csv(text, index=False, header=start == 0)
                            _progress(progress, 0.95 * stop / len(raw), f"Saved {stop:,} of {len(raw):,} prepared samples")
            (stage / metadata_path.name).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            check_cancelled(cancel)
            for staged, target in ((staged_trace, trace), (stage / metadata_path.name, metadata_path)):
                # Hard-link publication is atomic and fails if a racing writer already
                # created this destination; unlike replace(), it cannot overwrite them.
                os.link(staged, target); published.append(target)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    _progress(progress, 1.0, f"Exported {tag}")
    return trace, metadata_path
