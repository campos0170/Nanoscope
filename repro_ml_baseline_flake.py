"""MINIMAL REPRO — the ML baseline is NON-DETERMINISTIC, and so is the test.

    /opt/anaconda3/bin/python3 repro_ml_baseline_flake.py

Runs the body of
tests/analysis/test_ml_preprocess.py::test_baseline_robust_median_matches_reference_bit_for_bit
under the thread pressure a full pytest run has (Qt/TensorFlow worker threads).
Expect several FAILs out of 400.
"""
import sys, pathlib, threading, time
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nanoscope.analysis import ml_preprocess as mp

FS = 50_000.0

def _trace(n=60_000, seed=7):
    rng = np.random.default_rng(seed); t = np.arange(n) / FS
    x = 1500.0 + 40.0 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0.0, 4.0, n)
    for k in range(8):
        s = (n // 9) * (k + 1); x[s:s + 300] -= 60.0
    return x.astype(np.float32)

def _ref_robust_median(sig, fs, window_s=0.5, decim=50):      # the TEST's reference
    n = len(sig); xs = sig[::decim]
    w = int(window_s * fs / decim)
    if w % 2 == 0: w += 1
    b = median_filter(xs, size=w, mode="reflect")
    xp = np.arange(0, n, decim)[:len(b)]
    baseline = np.interp(np.arange(n), xp, b).astype(np.float32)
    return (sig - baseline).astype(np.float32), baseline

x = _trace().astype(np.float64)

stop = threading.Event()
def churn():
    while not stop.is_set():
        b = [np.empty(1 << 14) for _ in range(20)]; del b
for _ in range(4):
    threading.Thread(target=churn, daemon=True).start()
time.sleep(0.2)

fails = 0; worst = 0.0; N = 400
for i in range(N):
    _, got_b = mp.baseline_robust_median(x, FS, window_s=0.5, decim=50)
    _, ref_b = _ref_robust_median(x, FS, window_s=0.5, decim=50)
    if not np.array_equal(got_b, ref_b):
        fails += 1
        worst = max(worst, float(np.abs(got_b - ref_b).max()))
        if fails <= 3:
            d = np.flatnonzero(got_b != ref_b)
            print(f"  iter {i}: {len(d)} samples differ, first at {d[0]}, "
                  f"got {got_b[d[0]]!r} vs ref {ref_b[d[0]]!r}")
stop.set()
print(f"\nassert_array_equal(got_b, ref_b) FAILED {fails}/{N} times "
      f"({100*fails/N:.2f}%), max |delta| = {worst:.3f} pA")
sys.exit(1 if fails else 0)
