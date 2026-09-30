"""Numerical reference tests for the original student incubation contract."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace
import threading

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import median_filter
from scipy.signal import medfilt

from nanoscope.analysis import incubation_detection as detection
from nanoscope.analysis.cancel import OperationCancelled


def _notebook_normalize(signal):
    # Literal numeric reference from original notebook cell 4, not the app helper.
    x = np.asarray(signal, np.float32)
    m0 = float(np.median(x)); s0 = 1.4826 * float(np.median(np.abs(x - m0)))
    k = np.abs(x - m0) < 5 * max(s0, 1e-12)
    m1 = float(np.median(x[k])) if k.any() else m0
    s1 = 1.4826 * float(np.median(np.abs(x[k] - m1))) if k.any() else s0
    q = np.abs(x - m1) < 3 * max(s1, 1e-12)
    med = float(np.median(x[q])) if q.sum() > 100 else m1
    mad = 1.4826 * float(np.median(np.abs(x[q] - med))) if q.sum() > 100 else s1
    if not np.isfinite(mad) or mad <= 0:
        mad = s0 if s0 > 0 else 1.0
    return (np.clip((x - med) / mad, -30.0, 30.0) / 30.0).astype(np.float32)


class FakeStudent:
    def predict(self, batch, verbose=0):
        return np.where(batch < -0.05, 0.9, 0.01).astype(np.float32)


def _notebook_prediction(model, signal, window):
    sig = _notebook_normalize(signal)
    preds = np.zeros(len(sig), np.float32)
    counts = np.zeros(len(sig), np.float32)
    # Equivalent single-window calls make overlap enumeration independent of the
    # implementation's batching while this deterministic fake has no batch state.
    for start in range(0, len(sig) - window, window // 2):
        y = model.predict(sig[start:start + window][None, :, None], verbose=0)[0, :, 0]
        preds[start:start + window] += y
        counts[start:start + window] += 1
    return preds / np.maximum(counts, 1e-8), counts


def _notebook_regions(preds, fs):
    smooth = medfilt(preds, 3) if len(preds) >= 3 else preds
    # Direct run scanner, independent of the port's vectorized boundary extraction.
    regions, start = [], None
    for i in range(len(smooth) + 1):
        above = i < len(smooth) and smooth[i] >= 0.15
        if above and start is None:
            start = i
        if not above and start is not None:
            if np.any(smooth[start:i] >= 0.4):
                regions.append((start, i - 1))
            start = None
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= int(fs * 0.05 / 1000.0):
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s + 1 >= int(fs * 0.1 / 1000.0)]


def _notebook_i0(raw, quiet, fs, at):
    # Original cell 1's arithmetic, including truncated final block.
    raw = np.asarray(raw, np.float32); q = np.asarray(quiet, bool)
    blk = max(int(0.1 * fs), 1); n = len(raw); nb = max(n // blk, 1)
    bm = np.empty(nb, np.float64)
    for i in range(nb):
        sl = slice(i * blk, (i + 1) * blk)
        qs = q[sl]
        seg = raw[sl][qs] if qs.any() else raw[sl]
        bm[i] = np.median(seg) if seg.size else np.nan
    good = np.isfinite(bm)
    bm = np.interp(np.arange(nb), np.flatnonzero(good), bm[good])
    w = max(int(round(5.0 * fs / blk)), 1)
    if w % 2 == 0:
        w += 1
    # Same nearest-median operator through SciPy's multidimensional path:
    # its 1-D fast path is unsafe when this synthetic series is shorter than w.
    sm = median_filter(bm[None, :], size=(1, w), mode="nearest")[0]
    return np.interp(np.asarray(at, float), (np.arange(nb) + 0.5) * blk, sm).astype(np.float32)


def _files(tmp_path, *, polarity=1, n=2049, fs=10_000):
    signal = np.resize(np.array([-0.15, -0.1, 0.0, 0.1, 0.15], np.float32), n)
    signal[100:140] = -2.0
    signal[400:440] = -0.4
    signal[-1] = -5.0
    signal *= polarity
    raw = -200.0 - np.arange(n) * 0.00123456789 + signal
    # retained_time_s is the column load_prepared_recording actually validates (see
    # export_prepared): identical to time_s here since this synthetic file is never
    # truncated, matching what a real untruncated export produces.
    frame = pd.DataFrame(dict(time_s=np.arange(n) / fs, retained_time_s=np.arange(n) / fs,
                              current_raw_pA=raw, current_detr_pA=signal))
    trace = tmp_path / "zn_30h.csv.gz"
    frame.to_csv(trace, index=False)
    metadata = tmp_path / "zn_30h_metadata.json"
    metadata.write_text(json.dumps(dict(tag="zn_30h", fs_hz=fs, n_samples=n,
                                       duration_s=n / fs, event_polarity=polarity,
                                       csv_file=trace.name, baseline_method="prepared_test")))
    model = tmp_path / "student.keras"
    model.write_bytes(b"fake model identity; loading is monkeypatched")
    return trace, metadata, model, frame


def _fake_loader(monkeypatch, model=None, window=64):
    from nanoscope.analysis import ml_detect
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda path: SimpleNamespace(
        model=model or FakeStudent(), window=window, channels=1))


@pytest.mark.parametrize("window,n", [(8, 7), (8, 8), (8, 9), (8, 16), (5, 17), (64, 5000)])
def test_inference_matches_reference_and_discloses_exact_uncovered_tail(window, n):
    x = np.random.default_rng(4).normal(size=n).astype(np.float32)
    expected, counts = _notebook_prediction(FakeStudent(), x, window)
    actual, coverage = detection._predict_notebook(FakeStudent(), x, window)
    assert np.array_equal(actual, expected)
    assert coverage["uncovered_tail_samples"] == int((counts == 0).sum())
    assert coverage["covered_stop_idx_exclusive"] == (np.flatnonzero(counts)[-1] + 1 if counts.any() else 0)


@pytest.mark.parametrize("gap,expected", [(4, [(10, 25)]), (5, [])])
def test_legacy_merge_then_minimum_duration_uses_inclusive_arithmetic(gap, expected):
    p = np.zeros(60, np.float32)
    p[10:16] = 0.4
    p[16 + gap:22 + gap] = 0.4
    # Each six-sample event is too short at 100 kHz; only the merged pair survives.
    assert detection._student_regions(p, 100_000) == expected
    assert detection._student_regions(p, 100_000) == _notebook_regions(p, 100_000)


@pytest.mark.parametrize("polarity", [-1, 1])
def test_full_pipeline_preserves_candidates_gate_values_i0_and_hashes(tmp_path, monkeypatch, polarity):
    trace, metadata, model, _ = _files(tmp_path, polarity=polarity)
    _fake_loader(monkeypatch)
    progress = []
    got = detection.analyze_recording(trace, metadata, model, progress=lambda f, m: progress.append(f))
    parsed = pd.read_csv(trace)
    raw = parsed.current_raw_pA.to_numpy()  # preserve original float64 for later notebook features
    sig = polarity * parsed.current_detr_pA.to_numpy(np.float32)
    assert np.array_equal(got.raw, raw)
    assert np.array_equal(got.detrended, sig)
    m0 = float(np.median(sig)); s0 = 1.4826 * float(np.median(np.abs(sig - m0)))
    k0 = np.abs(sig - m0) < 5 * s0
    m1 = float(np.median(sig[k0])); s1 = 1.4826 * float(np.median(np.abs(sig[k0] - m1)))
    quiet = np.abs(sig - m1) < 3 * s1
    baseline = float(np.median(sig[quiet])); noise = 1.4826 * float(np.median(np.abs(sig[quiet] - baseline)))
    probability, _ = _notebook_prediction(FakeStudent(), sig, 64)
    regions = _notebook_regions(probability, 10_000)
    assert list(zip(got.catalog.start_idx, got.catalog.end_idx)) == regions
    i0 = np.abs(_notebook_i0(raw, quiet, 10_000, [s for s, _ in regions]))
    for i, row in enumerate(got.catalog.itertuples()):
        body = sig[row.start_idx:row.end_idx + 1]
        peak = baseline - float(body.min()); sustained = baseline - float(np.median(body))
        assert row.depth_peak_pA == peak
        assert row.depth_sust_pA == sustained
        assert row.sust_sigma == sustained / noise
        assert row.I0_local_pA == float(i0[i])
        assert row.fracblock == sustained / max(float(i0[i]), 1e-9)
        assert row.passes_gate == (sustained >= 4 * noise)
        assert row.duration_ms == (row.end_idx - row.start_idx + 1) / 10_000 * 1000
        assert row.end_s == (row.end_idx + 1) / 10_000
    assert got.catalog.passes_gate.any() and (~got.catalog.passes_gate).any()
    assert got.provenance["inference"]["uncovered_tail_samples"] == 1
    assert got.provenance["noise_mad_pA"] == noise
    assert got.provenance["recipe_id"] == detection.RECIPE_ID
    assert got.provenance["trace"]["sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert got.provenance["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert progress == sorted(progress) and progress[-1] == 1.0


def test_i0_matches_reference_with_drift_quiet_gaps_and_partial_last_block():
    raw = np.linspace(-100, -300, 1307)
    raw[-7:] = -9000  # intentionally omitted final partial block at this fs
    quiet = np.ones(len(raw), bool); quiet[400:500] = False
    at = np.array([0, 199, 800, 1299, 1306])
    actual, _, drift = detection._local_i0(raw, quiet, 1000, at)
    assert np.array_equal(actual, _notebook_i0(raw, quiet, 1000, at))
    assert drift >= 0


@pytest.mark.parametrize("field,value", [("fs_hz", 0), ("fs_hz", float("nan")),
    ("event_polarity", 0), ("event_polarity", True), ("n_samples", 999), ("duration_s", 88),
    ("csv_file", "other.csv.gz")])
def test_rejects_metadata_mismatch_before_model_loading(tmp_path, field, value):
    trace, meta, _, _ = _files(tmp_path)
    payload = json.loads(meta.read_text()); payload[field] = value; meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        detection.load_prepared_recording(trace, meta)


@pytest.mark.parametrize("column,value", [("retained_time_s", 0.123), ("current_raw_pA", np.inf),
                                        ("current_detr_pA", np.nan)])
def test_rejects_invalid_trace_columns(tmp_path, column, value):
    trace, meta, _, frame = _files(tmp_path)
    frame.loc[50, column] = value; frame.to_csv(trace, index=False)
    with pytest.raises(ValueError):
        detection.load_prepared_recording(trace, meta)


def test_original_float32_csv_clock_at_128_seconds_matches_sample_bins():
    # Actual first-run failure: sample 6,400,002 is exported as 128.00005,
    # although its scientific sample/fs clock is 128.00004 seconds.
    expected = np.arange(6_400_000, 6_400_100, dtype=np.float64) / 50_000
    exported = np.array([str(value) for value in expected.astype(np.float32)], float)
    assert exported[2] == 128.00005
    assert not np.allclose(exported, expected, rtol=0, atol=2e-9)
    clock = detection._validate_time_chunk(exported, 6_400_000, 50_000)
    assert clock['rows_requiring_float32_equivalence'] > 0
    assert clock['max_abs_deviation_s'] == np.abs(exported-expected).max()


@pytest.mark.parametrize('distortion', ['dropped_timestamp','two_sample_shift','clock_reset','rate_drift'])
def test_quantized_clock_still_refuses_irregularity_beyond_export_precision(distortion):
    expected = np.arange(6_400_000, 6_400_100, dtype=np.float64) / 50_000
    exported = np.array([str(value) for value in expected.astype(np.float32)], float)
    if distortion == 'dropped_timestamp':
        exported[20] = exported[19]
    elif distortion == 'two_sample_shift':
        exported[20] += 2 / 50_000
    elif distortion == 'clock_reset':
        exported[20:] -= exported[20]
    else:
        exported *= 1.0001
    with pytest.raises(ValueError, match='sample'):
        detection._validate_time_chunk(exported, 6_400_000, 50_000)


def test_loader_accepts_exported_float32_time_across_chunks_and_keeps_sample_clock(tmp_path):
    trace, metadata, _, frame = _files(tmp_path, n=250_005, fs=777)
    frame['retained_time_s'] = frame.retained_time_s.astype(np.float32)
    frame.to_csv(trace, index=False)
    loaded = detection.load_prepared_recording(trace, metadata)
    assert loaded.duration_s == len(frame) / 777
    assert loaded.fs_hz == 777
    assert np.array_equal(loaded.raw, pd.read_csv(trace).current_raw_pA.to_numpy(float))
    clock = loaded.provenance['time_validation_detail']
    assert clock['mode'] == 'float32_export_equivalent'
    assert clock['rows_requiring_float32_equivalence'] > 0
    assert clock['max_abs_deviation_samples'] < 1
    # A dropped timestamp right across the 250,000-row parser boundary remains
    # an error; chunking cannot reset the expected sample index.
    frame.loc[250_000, 'retained_time_s'] = frame.loc[249_999, 'retained_time_s']
    frame.to_csv(trace, index=False)
    with pytest.raises(ValueError, match='sample 250000'):
        detection.load_prepared_recording(trace, metadata)


def test_short_recording_has_typed_empty_catalog_and_full_uncovered_provenance(tmp_path, monkeypatch):
    trace, meta, model, _ = _files(tmp_path, n=40)
    _fake_loader(monkeypatch)
    got = detection.analyze_recording(trace, meta, model)
    assert got.catalog.empty
    assert got.catalog.passes_gate.dtype == bool
    assert got.catalog.start_idx.dtype == np.int64
    assert got.provenance["inference"]["uncovered_tail_samples"] == 40


def test_constant_noise_is_rejected_instead_of_publishing_nan_gate(tmp_path):
    with pytest.raises(ValueError, match="quiet noise"):
        detection._quiet_stats(np.zeros(1000, np.float32))


def test_zero_global_i0_is_rejected_instead_of_dividing_by_the_1e9_clamp():
    raw = np.array([-1.0, 1.0] * 500, np.float64)  # whole-trace median raw current is exactly 0
    recording = detection.PreparedRecording("zero_global_i0", raw, raw.astype(np.float32),
                                            1000.0, len(raw) / 1000.0, {}, {})
    quiet = np.ones(len(raw), bool)
    with pytest.raises(ValueError, match="open-pore reference"):
        detection._build_catalog(recording, [], 0.0, 1.0, quiet)


def test_zero_local_i0_at_one_candidate_is_rejected_instead_of_dividing_by_the_1e9_clamp(monkeypatch):
    raw = np.linspace(-150.0, -50.0, 500, np.float64)  # nonzero whole-trace reference
    detrended = np.zeros(500, np.float32)
    detrended[10:21] = -5.0
    quiet = np.ones(500, bool)
    recording = detection.PreparedRecording("zero_local_i0", raw, detrended, 1000.0, 0.5, {}, {})
    monkeypatch.setattr(detection, "_local_i0", lambda *a, **k: (np.array([0.0], np.float32), 0.0, 0.0))
    with pytest.raises(ValueError, match="local open-pore reference"):
        detection._build_catalog(recording, [(10, 20)], 0.0, 1.0, quiet)


def test_cancel_before_read_and_during_inference_publishes_nothing(tmp_path, monkeypatch):
    with pytest.raises(OperationCancelled):
        detection.analyze_recording("not-read", "not-read", "not-read", cancel=lambda: True)
    trace, meta, model, _ = _files(tmp_path)
    token = threading.Event()
    class CancellingStudent(FakeStudent):
        def predict(self, batch, verbose=0):
            token.set()
            return super().predict(batch, verbose)
    _fake_loader(monkeypatch, CancellingStudent())
    with pytest.raises(OperationCancelled):
        detection.analyze_recording(trace, meta, model, cancel=token)


def test_input_changed_during_model_call_is_not_published(tmp_path, monkeypatch):
    trace, meta, model, _ = _files(tmp_path)
    class MutatingStudent(FakeStudent):
        def predict(self, batch, verbose=0):
            payload = json.loads(meta.read_text()); payload["changed"] = True
            meta.write_text(json.dumps(payload))
            return super().predict(batch, verbose)
    _fake_loader(monkeypatch, MutatingStudent())
    with pytest.raises(ValueError, match="Input changed during analysis"):
        detection.analyze_recording(trace, meta, model)


def test_invalid_model_probabilities_are_refused():
    class InvalidStudent:
        def predict(self, batch, verbose=0):
            return np.full(batch.shape, np.nan)
    with pytest.raises(ValueError, match="probabilities"):
        detection._predict_notebook(InvalidStudent(), np.linspace(-1, 1, 100, dtype=np.float32), 8)


def test_same_mtime_model_replacement_cannot_reuse_old_cached_weights(tmp_path, monkeypatch):
    from nanoscope.analysis import ml_detect
    trace, meta, model, _ = _files(tmp_path)
    old_stat = model.stat()
    cache = {"wrapper": SimpleNamespace(model=object(), window=64, channels=1)}
    fresh = SimpleNamespace(model=FakeStudent(), window=64, channels=1)
    model.write_bytes(b"new model bytes with restored old mtime")
    os.utime(model, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert model.stat().st_mtime_ns == old_stat.st_mtime_ns
    calls = []
    def clear():
        calls.append("clear")
        cache.clear()
    def load(path):
        calls.append("load")
        return cache.setdefault("wrapper", fresh)
    monkeypatch.setattr(ml_detect, "clear_model_cache", clear)
    monkeypatch.setattr(ml_detect, "load_keras_model", load)
    got = detection.analyze_recording(trace, meta, model)
    assert calls == ["clear", "load"]
    assert got.provenance["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert got.provenance["model_cache_policy"].startswith("clear_before_load")
    assert got.catalog.passes_gate.any()
