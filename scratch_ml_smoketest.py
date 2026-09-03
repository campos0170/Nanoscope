#!/usr/bin/env python3
"""
Smoke test: prove the ported inference pipeline reproduces the trained Keras
raw / global-raw nanopore event-detector faithfully.

Run:  /opt/anaconda3/bin/python3 /Users/abrahamcampos/Desktop/nanoscope/scratch_ml_smoketest.py

Pipeline for the RAW model (input (None,3072,1), single raw current channel):
  1. load model (compile=False, no custom objects)
  2. read ~300k samples from middle of a real ABF via neo
  3. detrend (notch 60/120/180 Hz + SavGol 500 ms) -> global-raw normalize (max-abs, NO mean sub)
  4. predict_full_signal_batched (window=3072, step=1536, overlap-average)
  5. smooth (median) -> extract_regions_hysteresis (enter=0.5, exit=0.3) -> regions
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import time
import functools
import numpy as np

# Force CPU: the Apple Metal pluggable device aborts on this mixed_float16 LSTM.
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception as _e:
    print("could not hide GPU:", _e)

print = functools.partial(print, flush=True)  # noqa: A001 (unbuffered for crash safety)

# ---------------------------------------------------------------------------
# Config / deployment defaults
# ---------------------------------------------------------------------------
MODEL_PATH = "/Users/abrahamcampos/Downloads/student_raw_global_raw_best.keras"
ABF_PATH   = "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf"
FS         = 50_000

APO_NOTCH_HZ   = (60.0, 120.0, 180.0)
APO_NOTCH_Q    = 30.0
APO_SAVGOL_MS  = 500.0
APO_SAVGOL_POLY = 3

N_SAMPLES = 300_000  # 6 s @ 50 kHz


# ---------------------------------------------------------------------------
# Detrend (PART A of the apo contract) — VERBATIM from training notebook cell 12
# ---------------------------------------------------------------------------
def detrend_apo_contract(current, fs):
    from scipy.signal import iirnotch as _iirnotch, filtfilt as _filtfilt, savgol_filter as _savgol
    current_detr = np.asarray(current, np.float32)
    _sig = current_detr.copy()
    for _h in APO_NOTCH_HZ:
        if _h < fs / 2.0:
            _b, _a = _iirnotch(_h, APO_NOTCH_Q, fs)
            _sig = _filtfilt(_b, _a, _sig)
    _w = int(APO_SAVGOL_MS * fs / 1000.0)
    _w = _w + 1 if _w % 2 == 0 else _w
    _maximum_w = len(_sig) - 1 if len(_sig) % 2 == 0 else len(_sig)
    _w = min(_w, _maximum_w)
    if _w <= APO_SAVGOL_POLY:
        raise ValueError("Recording is too short for the configured SavGol detrending contract.")
    current_detr = (_sig - _savgol(_sig, window_length=_w, polyorder=APO_SAVGOL_POLY)).astype(np.float32)
    return current_detr


# ---------------------------------------------------------------------------
# Global-raw normalization (PART B) — divide by max-abs, NO mean subtraction
# ---------------------------------------------------------------------------
def global_raw_normalize(signal):
    signal = np.asarray(signal, np.float32)
    teacher_raw_scale = float(np.max(np.abs(signal)) + 1e-8)
    return (signal / teacher_raw_scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Sliding-window inference — VERBATIM from Untitled46.ipynb cell 9
# ---------------------------------------------------------------------------
def predict_full_signal_batched(model, signal, fs=50_000, window=2048, step=1024,
                                 batch_size=64, normalize_per_window=False):
    x = np.asarray(signal, np.float32)
    n = len(x)
    starts = np.arange(0, max(1, n - window + 1), step, dtype=int)
    out = np.zeros(n, np.float32)
    counts = np.zeros(n, np.float32)
    batch, idxs = [], []
    for s in starts:
        win = x[s:s + window]
        if len(win) < window:
            win = np.pad(win, (0, window - len(win)))
        if normalize_per_window:
            mu = np.mean(win)
            sigma = np.std(win) + 1e-8
            win = (win - mu) / sigma
        batch.append(win[:, None])
        idxs.append(s)
        if len(batch) >= batch_size:
            batch_arr = np.array(batch, np.float32)
            preds = model.predict(batch_arr, verbose=0).squeeze(-1)
            for p, st in zip(preds, idxs):
                e = min(st + window, n)
                out[st:e] += p[:e - st]
                counts[st:e] += 1
            batch.clear()
            idxs.clear()
    if batch:
        batch_arr = np.array(batch, np.float32)
        preds = model.predict(batch_arr, verbose=0).squeeze(-1)
        for p, st in zip(preds, idxs):
            e = min(st + window, n)
            out[st:e] += p[:e - st]
            counts[st:e] += 1
    counts[counts == 0] = 1.0
    preds_full = (out / counts).astype(np.float32)
    return preds_full


def run_model_on_signal(model, signal, fs, batch_size=64):
    model_window = model.input_shape[1]
    print(f"Model expects window_len = {model_window}")
    preds = predict_full_signal_batched(
        model=model, signal=signal, fs=fs,
        window=model_window, step=model_window // 2,
        batch_size=batch_size, normalize_per_window=False)
    return np.squeeze(np.array(preds))


# ---------------------------------------------------------------------------
# Decode — VERBATIM from training notebook cell 6
# ---------------------------------------------------------------------------
def smooth_signal(x, kernel_size=51):
    from scipy.signal import medfilt
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(x, kernel_size) if len(x) >= kernel_size else x


def extract_regions_hysteresis(prob, enter=0.5, exit_thr=0.3):
    prob = np.asarray(prob)
    above_exit  = (prob >= exit_thr).astype(np.int8)
    above_enter = prob >= enter
    diff   = np.diff(above_exit, prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]
    regions = []
    for s, e in zip(starts, ends):
        if above_enter[s:e].any():
            regions.append((s, e - 1))
    return regions


def merge_events_within_radius(regions, fs, radius_ms=0.1):
    if not regions:
        return []
    merge_samples = int(fs * radius_ms / 1000.0)
    merged = []
    current_start, current_end = regions[0]
    for start, end in regions[1:]:
        if start - current_end <= merge_samples:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def filter_short_events(regions, fs, min_ms=0.1):
    min_samples = int(fs * min_ms / 1000.0)
    return [(s, e) for s, e in regions if (e - s + 1) >= min_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1: load model")
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)
    print(f"  input_shape  = {model.input_shape}")
    print(f"  output_shape = {model.output_shape}")

    print("=" * 70)
    print("STEP 2: read ABF slice via neo")
    from neo.rawio import AxonRawIO
    r = AxonRawIO(filename=ABF_PATH)
    r.parse_header()
    total = r.get_signal_size(block_index=0, seg_index=0)
    print(f"  total samples in file = {total}  ({total / FS:.2f} s @ {FS} Hz)")
    A = max(0, (total - N_SAMPLES) // 2)
    B = A + N_SAMPLES
    raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=A, i_stop=B)
    cur = r.rescale_signal_raw_to_float(raw, dtype='float32')[:, 2]  # ch 2 = current (pA)
    print(f"  slice [{A}:{B}] -> {len(cur)} samples; raw current pA "
          f"min/mean/max = {cur.min():.2f} / {cur.mean():.2f} / {cur.max():.2f}")

    print("=" * 70)
    print("STEP 3: detrend (notch + SavGol) -> global-raw normalize")
    detr = detrend_apo_contract(cur, FS)
    print(f"  detrended pA min/mean/max = {detr.min():.2f} / {detr.mean():.4f} / {detr.max():.2f}")
    norm = global_raw_normalize(detr)
    print(f"  normalized  min/mean/max = {norm.min():.4f} / {norm.mean():.6f} / {norm.max():.4f}")

    print("=" * 70)
    print("STEP 4: predict_full_signal_batched (overlap-average stitch)")
    t0 = time.time()
    preds = run_model_on_signal(model, norm, FS, batch_size=64)
    dt = time.time() - t0
    preds = preds.astype(np.float32)
    print(f"  inference wall-clock = {dt:.2f} s for {len(cur)} samples "
          f"({len(cur)/FS:.1f} s of data) -> {len(cur)/FS/dt:.2f}x realtime")
    print(f"  preds length = {len(preds)}")
    print(f"  per-sample prob min/mean/max = {preds.min():.4f} / {preds.mean():.4f} / {preds.max():.4f}")

    print("=" * 70)
    print("STEP 5: decode -> smooth -> hysteresis -> merge -> filter")
    smoothed = smooth_signal(preds, kernel_size=9)
    raw_regions = extract_regions_hysteresis(smoothed, enter=0.5, exit_thr=0.3)
    merged = merge_events_within_radius(raw_regions, fs=FS, radius_ms=0.1)
    final = filter_short_events(merged, fs=FS, min_ms=0.1)
    print(f"  raw hysteresis regions = {len(raw_regions)}")
    print(f"  after merge(0.1ms)     = {len(merged)}")
    print(f"  after min-dur(0.1ms)   = {len(final)}")

    print("-" * 70)
    print("REPORT")
    print(f"  model I/O: input {model.input_shape}, output {model.output_shape}")
    print(f"  event regions found in {len(cur)/FS:.0f} s slice: {len(final)}")
    print("  first few events (start, end samples | dwell samples | dwell ms):")
    for s, e in final[:8]:
        dwell = e - s + 1
        print(f"    ({s:>7}, {e:>7}) | {dwell:>6} | {dwell/FS*1000:.3f} ms")
    print(f"  prob trace min/mean/max = {preds.min():.4f} / {preds.mean():.4f} / {preds.max():.4f}")
    print(f"  inference wall-clock = {dt:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
