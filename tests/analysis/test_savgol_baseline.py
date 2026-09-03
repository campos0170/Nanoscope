"""``method='savgol'`` -- the Cu2-insulin baseline, and the two traps around its name.

The Cu recordings are detrended with a Savitzky-Golay baseline fitted on a DECIMATED
copy: ``BASELINE_METHOD = {"cu_2h":"sg", "cu_10h":"sg", "cu_18h":"sg", "cu_30h":"sg"}``,
``SG_WINDOW_S 0.2``, ``SG_DECIM 50``, ``SG_POLY 3``, and all four written
``CSVData/cu_*_metadata.json`` agree with ``masked_median_params: null``. (The
notebook's own cell-0 MARKDOWN still says "event-masked robust median" and is wrong;
the Block 0 code and the written metadata are the truth.)

TWO THINGS SHARE THIS NAME AND NEITHER IS THIS ONE:

  * ``filters.savgol`` -- the app's ``savgol_enabled`` conditioning stage. That is a
    SMOOTHER: it RETURNS the smoothed signal, as a low-pass. This is an ESTIMATOR:
    the smoothed signal is what gets SUBTRACTED. Opposite operations.
  * ``ml_preprocess.baseline_savgol`` -- the APO chain's SavGol baseline. Undecimated,
    ``w = int(window_s * fs)``, scipy's default ``mode='interp'``. Same family,
    different estimator, frozen against a different notebook.

Both are pinned below as inequalities, so a future refactor that "unifies" them fails
here instead of silently changing which recipe ran.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import iirnotch, filtfilt, savgol_filter

from nanoscope.analysis import baseline as B
from nanoscope.analysis.baseline import (
    SAVGOL_DECIM,
    SAVGOL_POLY,
    SAVGOL_WINDOW_S,
    estimate_baseline,
    remove_baseline_wander,
)
from nanoscope.analysis.filters import savgol as savgol_smoother
from nanoscope.analysis.ml_preprocess import baseline_savgol as apo_baseline_savgol

FS = 50_000.0

#: The owner's own CSV for the training run, if this machine has it.
CU_CSV = Path("/Users/abrahamcampos/Desktop/Cu2Data/CSVData/cu_30h.csv.gz")

#: The Cu/Zn nine-line notch family.
CU_NOTCH = [60, 120, 180, 240, 300, 360, 420, 540, 660]


# ---------------------------------------------------------------------------
# THE REFERENCE. Copied verbatim from Cu2Insulin_DataPreprocessing.ipynb Block 2.
# ---------------------------------------------------------------------------
def _nb_interp_back(b_dec, n, decim):
    xp = np.arange(0, n, decim)[:len(b_dec)]
    return np.interp(np.arange(n), xp, b_dec).astype(np.float32)


def _nb_baseline_remove_sg(sig, fs, window_s=0.2, poly=3, decim=50):
    n = len(sig)
    xs = np.asarray(sig[::decim], np.float64)
    w = int(window_s * fs / decim)
    if w % 2 == 0: w += 1
    w = max(w, poly + 2 + ((poly + 2) % 2 == 0))
    if w > len(xs): w = len(xs) - ((len(xs) + 1) % 2)
    b = savgol_filter(xs, window_length=w, polyorder=poly, mode='mirror')
    baseline = _nb_interp_back(b, n, decim)
    return (sig - baseline).astype(np.float32), baseline


def _cu_trace(n=int(8.0 * FS), *, i0=-125.0, noise=2.2, depth=78.0, drift=9.0,
              seed=0, unstable=False):
    """A Cu-shaped conductive recording: negative open pore, DEEP RARE events, drift.

    ``unstable=True`` widens the noise band over the first and last tenth without
    moving the level -- the cu_10h amplifier instability, which is what separates
    this estimator from the masked one.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = i0 + drift * np.sin(2 * np.pi * 0.12 * t) + rng.normal(0.0, noise, n)
    if unstable:
        m = n // 10
        x[:m] += rng.normal(0.0, 9.0 * noise, m)
        x[-m:] += rng.normal(0.0, 9.0 * noise, m)
    for k in range(1, int(n / FS * 25)):
        i = int(k * FS / 25)
        x[i: i + 12] -= depth
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# it exists, and it is the notebook's arithmetic
# ---------------------------------------------------------------------------


def test_estimate_baseline_accepts_savgol_and_the_first_four_methods_did_not_move():
    assert "savgol" in B.METHODS
    # Order is contract: the frozen four stay first (test_masked_baseline asserts it).
    assert B.METHODS[:4] == ("block_median", "percentile", "asls", "constant")
    assert B.METHODS[-1] == "savgol"
    assert (SAVGOL_WINDOW_S, SAVGOL_POLY, SAVGOL_DECIM) == (0.2, 3, 50)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("window_s,poly,decim", [
    (0.2, 3, 50),      # the Cu recipe
    (0.5, 3, 50),
    (0.2, 2, 10),
    (2.0, 4, 100),
])
def test_bit_for_bit_against_the_notebook_baseline_remove_sg(seed, window_s, poly, decim):
    """Pinned against a LITERAL TRANSCRIPTION of the notebook cell, not against this
    port's own behaviour. ``estimate_baseline`` returns float64 holding the float32
    the notebook wrote -- the cast lands on the BASELINE, before the subtraction,
    exactly as `_kernel_baseline` documents for the masked kernels -- so the two are
    compared after the same cast."""
    x = _cu_trace(seed=seed)
    _, want = _nb_baseline_remove_sg(x, FS, window_s=window_s, poly=poly, decim=decim)
    got = estimate_baseline(x, FS, method="savgol", window_s=window_s,
                            sg_poly=poly, sg_decim=decim, polarity=1)
    assert got.dtype == np.float64
    assert np.array_equal(got.astype(np.float32), want)


def test_the_window_is_counted_on_the_decimated_grid_not_the_raw_one():
    """0.2 s at 50 kHz with decim 50 is **201 taps on a 1 kHz series**, not 10 001 taps
    on the raw one. That is what makes the stage runnable: savgol_filter convolves
    directly, so an undecimated fit is O(n*w) and a 15 M-sample recording takes hours.
    Pinned by reproducing the exact scipy call."""
    x = _cu_trace()
    xs = np.ascontiguousarray(x[::50], dtype=np.float64)
    assert int(0.2 * FS / 50) == 200          # even -> forced odd
    want_dec = savgol_filter(xs, window_length=201, polyorder=3, mode="mirror")
    xp = np.arange(0, x.size, 50)[: want_dec.size]
    want = np.interp(np.arange(x.size), xp, want_dec).astype(np.float32)
    got = estimate_baseline(x, FS, method="savgol", window_s=0.2,
                            sg_poly=3, sg_decim=50, polarity=1)
    assert np.array_equal(got.astype(np.float32), want)


def test_the_defaults_are_the_cu_recipe_and_window_none_means_zero_point_two():
    """'savgol' is new, so nothing stored depends on it inheriting the frozen 10 ms
    block-median default -- and 10 ms would be FOUR taps at decim 50, an estimator
    that tracks the events instead of the open pore."""
    x = _cu_trace()
    explicit = estimate_baseline(x, FS, method="savgol", window_s=SAVGOL_WINDOW_S,
                                 sg_poly=SAVGOL_POLY, sg_decim=SAVGOL_DECIM, polarity=1)
    defaulted = estimate_baseline(x, FS, method="savgol", polarity=1)
    assert np.array_equal(explicit, defaulted)


# ---------------------------------------------------------------------------
# the two name traps
# ---------------------------------------------------------------------------


def test_it_returns_the_baseline_not_the_residual():
    """`estimate_baseline`'s contract, and the difference between an estimator and a
    smoother. The returned array sits AT the open-pore level (~-125 pA here); the
    residual would sit at ~0."""
    x = _cu_trace(i0=-125.0)
    base = estimate_baseline(x, FS, method="savgol", polarity=1)
    assert -140.0 < float(np.median(base)) < -110.0
    # And it is what `remove_baseline_wander` subtracts: the detrended trace is flat
    # about the reference level, and the drift is gone.
    detr = remove_baseline_wander(x, FS, method="savgol", window_s=SAVGOL_WINDOW_S,
                                  polarity=1, sg_poly=3, sg_decim=50)
    raw_wander = float(np.percentile(x, 95) - np.percentile(x, 5))
    out_wander = float(np.percentile(detr, 95) - np.percentile(detr, 5))
    assert out_wander < raw_wander / 2.0


def test_it_is_not_the_savgol_SMOOTHER_the_conditioning_chain_already_has():
    """``filters.savgol`` returns the SMOOTHED SIGNAL (a low-pass). Same filter family,
    opposite role: the smoother's output IS the trace you keep, this one's output is
    the thing you take away. Running one where the other was meant leaves a plausible
    trace that kept exactly the wrong half."""
    x = _cu_trace()
    smoothed = np.asarray(savgol_smoother(x, FS, window_s=SAVGOL_WINDOW_S, polyorder=3),
                          dtype=np.float64)
    estimated = estimate_baseline(x, FS, method="savgol", window_s=SAVGOL_WINDOW_S,
                                  sg_poly=3, sg_decim=50, polarity=1)
    # Both are smooth versions of the trace, so they are CLOSE -- which is exactly why
    # the confusion is dangerous -- but they are not the same array, and only one of
    # them is undecimated.
    assert not np.allclose(smoothed, estimated, atol=1e-3)


def test_it_is_not_the_apo_chains_undecimated_baseline_savgol():
    """`ml_preprocess.baseline_savgol` is the APO recipe: undecimated,
    ``w = int(window_s * fs)`` = 10 001 taps here, and scipy's default
    ``mode='interp'``. It is frozen against the apo notebooks and must not be
    retargeted, which is why the Cu recipe gets its own function."""
    x = _cu_trace()
    _, apo = apo_baseline_savgol(x, FS, window_s=SAVGOL_WINDOW_S, poly=3)
    cu = estimate_baseline(x, FS, method="savgol", window_s=SAVGOL_WINDOW_S,
                           sg_poly=3, sg_decim=50, polarity=1)
    assert not np.allclose(np.asarray(apo, dtype=np.float64), cu, atol=1e-3)


# ---------------------------------------------------------------------------
# why the Cu sweep chose it over masked_median
# ---------------------------------------------------------------------------


def test_it_tracks_through_a_baseline_unstable_stretch_that_a_mask_would_bridge():
    """The cu_10h finding, as a measurement. A 3-sigma mask reads an amplifier's noise
    band widening as EVENTS and interpolates ACROSS it, so the instability re-emerges
    in the detrended trace; the owner measured attenuation of -341% to -723% and
    107-381 'deep' events against a true 28 that way. SavGol does not mask, so the
    unstable stretch stays where it is and the detrended trace keeps its own noise.

    Scored as the fraction of the unstable head that ends up more than 6 sigma from
    zero after detrending -- the samples a downward detector would call events."""
    x = _cu_trace(unstable=True, drift=0.0)
    head = slice(0, x.size // 10)

    def _false_event_fraction(method, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = np.asarray(
                remove_baseline_wander(x, FS, method=method, polarity=1, **kw),
                dtype=np.float64)
        d = d - np.median(d)
        # sigma from the QUIET middle, which both estimators handle
        mid = d[x.size // 3: 2 * x.size // 3]
        sigma = 1.4826 * np.median(np.abs(mid - np.median(mid)))
        return float(np.mean(d[head] < -6.0 * sigma))

    sg = _false_event_fraction("savgol", window_s=SAVGOL_WINDOW_S,
                               sg_poly=3, sg_decim=50)
    mm = _false_event_fraction("masked_median", window_s=0.5, rough_s=0.5, base_s=0.2,
                               decim=10, mask_sigma=3.0, dilate_ms=2.0)
    assert sg < mm, f"savgol {sg:.4f} should beat masked_median {mm:.4f} here"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_a_clamped_window_warns_because_it_is_a_different_estimator():
    """A window wider than the recording fits ONE polynomial over the whole trace, so
    it tracks no drift inside it. The notebook clamps silently; this says so."""
    x = _cu_trace(n=int(0.05 * FS))
    with pytest.warns(B.BaselineWarning, match=r"CLAMPED"):
        estimate_baseline(x, FS, method="savgol", window_s=1.0,
                          sg_poly=3, sg_decim=50, polarity=1)


def test_a_trace_too_short_for_any_window_is_refused_by_name():
    x = _cu_trace(n=120)
    with pytest.raises(ValueError, match=r"SavGol baseline needs a window"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate_baseline(x, FS, method="savgol", window_s=0.2,
                              sg_poly=3, sg_decim=50, polarity=1)


@pytest.mark.parametrize("kw,msg", [
    (dict(sg_decim=0), r"sg_decim must be >= 1"),
    (dict(sg_decim=-5), r"sg_decim must be >= 1"),
    (dict(sg_poly=0), r"sg_poly must be a positive integer"),
])
def test_degenerate_shape_parameters_are_refused(kw, msg):
    x = _cu_trace()
    with pytest.raises(ValueError, match=msg):
        estimate_baseline(x, FS, method="savgol", window_s=0.2, polarity=1, **kw)


# ---------------------------------------------------------------------------
# the owner's own data path, when this machine has it
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CU_CSV.exists(), reason="the owner's cu_30h CSV is not here")
def test_it_reproduces_the_baseline_the_owner_actually_wrote_for_cu_30h():
    """The whole verification, on real data: raw channel-0 current -> the nine-line
    notch -> this estimator at the Cu recipe (0.2 s, poly 3, decim 50) must land on
    the ``baseline_pA`` column the notebook wrote.

    On the FULL 15 076 224-sample recording the result is BIT-IDENTICAL to the written
    column (max |diff| exactly 0.0 pA), and so are cu_2h, cu_10h and cu_18h. This test
    reads a 40 s prefix to stay fast and compares the 5-35 s interior, away from both
    the notch's ring-down and the SavGol mirror boundary that a prefix necessarily
    moves; there it agrees to one float32 ULP.
    """
    import pandas as pd

    df = pd.read_csv(CU_CSV, usecols=["current_raw_pA", "baseline_pA"],
                     nrows=2_000_000)
    raw = df["current_raw_pA"].to_numpy(np.float32)
    ref = df["baseline_pA"].to_numpy(np.float32)

    # The notebook's own apply_notch_filters: one filtfilt per line, ascending.
    filt = raw.copy()
    for f0 in CU_NOTCH:
        b, a = iirnotch(f0, Q=30, fs=FS)
        filt = filtfilt(b, a, filt)

    got = estimate_baseline(filt, FS, method="savgol", window_s=0.2,
                            sg_poly=3, sg_decim=50, polarity=1).astype(np.float32)
    lo, hi = int(5 * FS), int(35 * FS)
    assert float(np.abs(got[lo:hi] - ref[lo:hi]).max()) <= 1e-5
