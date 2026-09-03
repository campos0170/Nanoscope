"""ADV4-02 -- the detrend must not amplify baseline wander silently.

THE DEFECT. ``remove_baseline_wander``'s implied response is ``H(f) = 1 - M(f)``
where ``M`` is the moving block-median: for wander periods between roughly half the
window and the window, ``M(f)`` is phase-INVERTED (the sign flip of a moving
average's first sinc side lobe), so subtracting the estimate ADDS wander. Measured:
80 pA of 3 Hz wander became a 110.6 pA peak (+38%) under the default 0.5 s window.
No warning was emitted, ``collect_baseline_warnings`` stayed empty, and DetectSchmitt
stamped ``detrend_enabled=True`` / ``detrend_effective=True`` -- provenance claimed
the stage ran while the stage had made the very thing it exists to remove worse, and
nothing downstream can see that from the data (the trace simply wanders more).

THE INVARIANT RESTORED. The kernel measures its own output against its own input
(``_block_median_wander``: medians of window/8 blocks, same grid on both sides, two
O(n) passes) and raises ``AmplifiedWanderWarning`` -- a ``BaselineWarning`` subclass,
so it rides the existing ``collect_baseline_warnings`` sink into
``prov['detrend_warnings']`` and ``res.warnings`` exactly as the no-op detrend's
``ConstantBaselineWarning`` already does. ``prov['detrend_amplified']`` is the
structural form, mirroring ``detrend_effective``. A detrend doing its legitimate job
(slow drift, window wider than the wander period) and a detrend with nothing to do
(flat trace) stay silent -- the threshold `DETREND_AMPLIFIED_PEAK_RATIO` sits between
the measured separations (defect 1.35-1.37; legitimate 0.58-0.95).
"""

from __future__ import annotations

import os
import warnings

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from conftest import make_context
from nanoscope.analysis import baseline as B
from nanoscope.analysis.modules import DetectSchmitt

FS = 50_000.0


def _fast_wander(n: int = 1_000_000, *, amp: float = 80.0, noise: float = 0.0,
                 events: bool = False, seed: int = 5) -> np.ndarray:
    """3 Hz wander (period 0.33 s) against the 0.5 s default window -- the
    phase-inversion band."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = 1000.0 + amp * np.sin(2 * np.pi * 3.0 * t)
    if noise:
        x = x + rng.normal(0.0, noise, n)
    if events:
        for s in range(20_000, n - 5_000, 45_000):
            x[s:s + 300] -= 80.0
    return np.ascontiguousarray(x, dtype=np.float32)


def test_the_kernel_warns_when_its_output_is_worse_than_its_input():
    x = _fast_wander()
    with pytest.warns(B.AmplifiedWanderWarning) as rec:
        out = B.remove_baseline_wander(x, FS, window_s=0.5)
    msg = str(rec[0].message)
    # The warning names the numbers it measured, the mechanism, and a real remedy.
    assert "AMPLIFIED" in msg
    assert "pA in ->" in msg and "pA out" in msg
    assert "high-pass" in msg
    # ...and the amplification it is warning about is real: the output really does
    # carry a bigger peak excursion than the 80 pA that went in.
    mid = np.asarray(out[100_000:900_000], dtype=np.float64)
    assert float(np.abs(mid - np.median(mid)).max()) > 1.10 * 80.0


def test_the_warning_reaches_the_collect_sink():
    """The sink is what DetectSchmitt reads -- a warning that only went through
    ``warnings.warn`` would die in a filter and never reach a surface."""
    x = _fast_wander()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # even with stdlib warnings muted...
        with B.collect_baseline_warnings() as sink:
            B.remove_baseline_wander(x, FS, window_s=0.5)
    assert any(isinstance(w, B.AmplifiedWanderWarning) for w in sink)


def test_a_legitimate_detrend_and_a_flat_trace_stay_silent():
    rng = np.random.default_rng(7)
    n = 1_000_000
    t = np.arange(n) / FS
    slow = 1000.0 + 40.0 * np.sin(2 * np.pi * t / 2.0) + rng.normal(0.0, 5.0, n)
    flat = 1000.0 + rng.normal(0.0, 5.0, n)
    for x in (slow, flat):
        for s in range(20_000, n - 5_000, 45_000):
            x[s:s + 300] -= 80.0
        with B.collect_baseline_warnings() as sink:
            B.remove_baseline_wander(
                np.ascontiguousarray(x, dtype=np.float32), FS, window_s=0.5)
        assert not any(isinstance(w, B.AmplifiedWanderWarning) for w in sink), [
            str(w) for w in sink]


def test_the_shipped_detection_surface_says_it_and_stamps_it():
    """DetectSchmitt with the detrend enabled on the amplifying trace: the sentence
    reaches ``res.warnings`` (the Stats panel's surface) and provenance carries the
    structural key -- two surfaces, one truth."""
    x = _fast_wander(noise=5.0, events=True)
    ctx = make_context(x, fs=FS, params={"detrend_enabled": True,
                                         "detrend_window_s": 0.5,
                                         "min_dwell_s": 1e-4})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DetectSchmitt().run(ctx)
    prov = res.payload.provenance
    assert prov["detrend_enabled"] is True
    assert prov["detrend_amplified"] is True
    assert any("AMPLIFIED the baseline wander" in w for w in prov["detrend_warnings"])
    assert any("AMPLIFIED the baseline wander" in w for w in res.warnings)


def test_a_quiet_detrend_stamps_detrend_amplified_false():
    rng = np.random.default_rng(9)
    n = 1_000_000
    t = np.arange(n) / FS
    x = 1000.0 + 40.0 * np.sin(2 * np.pi * t / 2.0) + rng.normal(0.0, 5.0, n)
    for s in range(20_000, n - 5_000, 45_000):
        x[s:s + 300] -= 80.0
    ctx = make_context(np.ascontiguousarray(x, dtype=np.float32), fs=FS,
                       params={"detrend_enabled": True, "detrend_window_s": 0.5,
                               "min_dwell_s": 1e-4})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DetectSchmitt().run(ctx)
    prov = res.payload.provenance
    assert prov["detrend_amplified"] is False
    assert not any("AMPLIFIED" in w for w in res.warnings)
