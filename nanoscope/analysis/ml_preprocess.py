"""Notebook-faithful preprocessing + decode for the ML event detector.

This module is the ML detector's OWN signal chain. It is deliberately NOT the
app's general-purpose "Preprocess" tab chain
(:func:`nanoscope.analysis.modules.preprocess_for_detection`): a learned model
only behaves as measured when it is fed the *same* pipeline it was trained on,
so the ML detector runs this and nothing else.

TWO MOLECULES, TWO PIPELINES
----------------------------
The presets here cover two different samples, not two parameter tastes:

* **Ferritin** (holo, iron core) -- ``notebook1`` / ``notebook2`` / ``notebook3``.
* **Apo-ferritin** (the empty shell) -- ``apo_nb_500mV`` / ``apo_nb_600mV`` /
  ``apo_nb_700mV`` (the real apo notebook) and ``apo_raw_global`` (the older
  raw-global apo model).

Running a ferritin-trained model on apo data, or vice versa, is a category
error, which is why every label is molecule-first.

Provenance of the FERRITIN numbers
----------------------------------
The three ``StudentModel*.ipynb`` notebooks do **no** preprocessing themselves --
they load a precomputed ``current_detr_pA`` column from ``CSVData/<tag>.csv.gz``.
That column is produced by ``FerritinModeler/dataLoader.ipynb`` (Blocks 0/2/3),
which is therefore the real preprocessing contract. The windowing, the
normalization and the decode come from the ``StudentModel*.ipynb`` notebooks
themselves (``student_predict`` and ``postprocess_predictions``).

Shared by all three ferritin notebooks (verified identical, and verified
BIT-EXACT against the persisted training columns):

* **Notch** -- 60 / 120 / 180 Hz, ``Q=30``, applied as THREE SEQUENTIAL
  ``filtfilt`` passes in ascending order. NOT one cascaded second-order-section
  filter: each ``filtfilt`` does its own edge padding, so a cascade is a
  numerically different signal (see :func:`notch_sequential`). The app's general
  :func:`nanoscope.analysis.filters.notch` *does* cascade -- which is exactly why
  the ML detector must not use it. The float64 ``filtfilt`` output is carried
  into the baseline stage UNROUNDED (``notch_float32=False``).
* **Normalization** -- ``x / (max|x| + 1e-8)`` over the WHOLE signal, no mean
  subtraction (``normalize_center=False``), applied BEFORE windowing
  (``student_predict``).
* **Windowing** -- ``W = 2048``, stride ``W // 2 = 1024``, loop
  ``range(0, n - W, step)`` so the tail is uncovered (probability 0);
  recombination is ``sum / np.maximum(counts, 1e-8)``.
* **enter_threshold** -- 0.5.

Provenance of the APO numbers
-----------------------------
``ApoFerritin_LSTM_Cluster_Validation_RAM_Safe.ipynb`` (cells 9-16), which --
unlike the ferritin notebooks -- preprocesses INLINE and is self-contained.
"RAM_Safe" refers to batching *events* downstream; nothing in the signal path is
chunked, so every quantity (notch padding, SavGol envelope, the mean and
``max|x|``) is computed over the WHOLE recording, exactly as this module does.
The chain is notch (identical to ferritin's, but the float32 round in
``apply_notch_filter_harmonics`` IS kept -- ``notch_float32=True``) -> SavGol
25001-sample / poly-3 ``mode='interp'`` detrend -> ``(x - mean) / (max|x - mean|
+ 1e-8)`` over the whole recording (``normalize_center=True``; the mean
subtraction is the one real difference from ferritin) -> ``W = 3072`` stride 1536
with ``np.arange(0, n - W + 1, step)`` (``last_start_inclusive=True``) -> a
SINGLE strict threshold with NO probability smoothing and NO merge inside the
decoder, hand-tuned per bias.

Reproducing that chain on the user's real -600 mV recording gives
``sha256(model_input) = 382b0b027600043534f8ed169c259d2ae2c2e079e73099abefa93ab9adfcfded``
and 236 events -- both matching the notebook's stored outputs exactly.

``"apo_raw_global"`` (formerly ``"apo_legacy"``, still accepted as an alias) is
the app's pre-existing ML chain: SavGol 500 ms / poly-3 baseline, window 3072 /
stride 1536, anchored tail, ``x/(max|x|+1e-8)``. That PREPROCESSING is the
genuine training contract of ``student_raw_global_raw_best.keras``
(``FerritinModeler/FinalModel.ipynb`` cell 23 builds exactly that filename from
``EXP_TAG="student_raw"`` + ``USE_GLOBAL_NORM=True``), a model the user holds --
so the preset stays selectable. Its DECODE, however, is not from any executed
call: ``smoothing_kernel=9 / enter 0.5 / exit 0.3 / merge 0.1 ms / min 0.1 ms``
are verbatim the *signature defaults* of ``postprocess_predictions``, which every
call site in every apo notebook overrides. See :data:`DECODE_PRESETS`.
All four apo/ferritin SavGol notes aside, ``apo_raw_global`` is WRONG for the
three ferritin models: the SavGol polynomial bows around deep/long events and
reintroduces the positive-lobe artifact ``dataLoader.ipynb`` switched away from.

Everything in here is pure numpy/scipy -- no Qt, no TensorFlow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import filtfilt, iirnotch, medfilt, savgol_filter

from .acid_incubation_numerics import median_filter_1d
from .filters import MODEL_INPUT_CLIP_SIGMA, robust_normalize

__all__ = [
    "NOTCH_FREQS",
    "NOTCH_Q",
    "BASELINE_METHODS",
    "NORMALIZE_SCOPES",
    "MLPreprocessConfig",
    "MLDecodeConfig",
    "PREPROCESS_PRESETS",
    "DECODE_PRESETS",
    "PRESET_NAMES",
    "PRESET_LABELS",
    "PRESET_MOLECULE",
    "PRESET_ALIASES",
    "MOLECULES",
    "DEFAULT_PRESET",
    "resolve_preset_name",
    "preset_for_window",
    "notch_sequential",
    "baseline_robust_median",
    "baseline_masked_median",
    "baseline_savgol",
    "despike_median",
    "apply_ml_preprocess",
    "normalization_center",
    "normalization_scale",
    "normalize_global",
    "smooth_probability",
    "extract_regions_hysteresis",
    "merge_events_within_radius",
    "filter_short_events",
    "snap_onsets_to_deep",
    "trim_offsets_to_deep",
    "decode_ml_regions",
    "gate_regions_by_amplitude",
]

# The mains line + harmonics every notebook notches, and the notch quality factor.
NOTCH_FREQS: tuple[float, ...] = (60.0, 120.0, 180.0)
NOTCH_Q: float = 30.0

#: Baseline estimators :func:`apply_ml_preprocess` accepts, in preset order.
BASELINE_METHODS: tuple[str, ...] = ("robust_median", "masked_median", "savgol", "none")

#: Where ``max|x|`` for the global-raw normalization is measured.
#:
#: ``"recording"`` -- over the WHOLE channel, which is what training did
#: (``student_predict`` normalizes ``current_detr``, the whole recording). This is
#: the default, and it makes the model input independent of the user's zoom.
#: ``"range"`` -- over the analysed sub-range only. Kept selectable (and stamped
#: into provenance) because it is what the app used to do implicitly.
NORMALIZE_SCOPES: tuple[str, ...] = ("recording", "range")

#: How the model input is put on a dimensionless scale.
#:
#: ``"maxabs"`` is the ORIGINAL and remains the default, so every stored preset, every
#: saved run and every project written before this field existed reproduces bit-for-bit:
#: ``(x - center) / (max|x - center| + 1e-8)`` (`normalize_global`).
#:
#: ``"robust_sigma"`` is `filters.robust_normalize` -- a twice-trimmed median/MAD, then
#: clip to +-``normalize_clip_sigma`` and divide by it. It REPLACED max-abs in the
#: owner's six host-guest / Zn2 / Cu2 student notebooks for a documented reason: the
#: max-abs divisor is a SINGLE EXTREME SAMPLE and it ranged 8.8 to 42.6 pA across those
#: recordings, so ordinary noise reached the model at 0.049 for the training domain but
#: 0.146 for the host -- host NOISE arriving at the amplitude the model had been taught
#: to call an EVENT. In sigma units the noise floor is 1/clip in every recording by
#: construction, so a model transfers between recordings instead of inheriting each
#: file's extreme value.
NORMALIZE_METHODS: tuple[str, ...] = ("maxabs", "robust_sigma")

#: The method every pre-existing preset and stored run is on. Do not change it: it is
#: what makes the new field invisible to everything that came before.
DEFAULT_NORMALIZE_METHOD: str = "maxabs"

#: Default clip bound / divisor for ``"robust_sigma"``, in noise sigmas. Both the clip
#: AND the divisor, which is what puts the noise floor at exactly 1/30 in every
#: recording -- the one number those models were trained against.
DEFAULT_NORMALIZE_CLIP_SIGMA: float = float(MODEL_INPUT_CLIP_SIGMA)


# ---------------------------------------------------------------------------
# Notch
# ---------------------------------------------------------------------------
def notch_sequential(
    sig: np.ndarray,
    fs: float,
    freqs: Sequence[float] = NOTCH_FREQS,
    q: float = NOTCH_Q,
) -> np.ndarray:
    """``dataLoader.ipynb``'s ``apply_notch_filters``: one ``filtfilt`` per line,
    applied in the order given.

    Verbatim::

        out = sig.copy()
        for f0 in freqs:
            b, a = iirnotch(f0, Q=Q, fs=fs)
            out  = filtfilt(b, a, out)

    IT IS THE PADDING, NOT THE TOPOLOGY, and the earlier wording here conflated the
    two. Stacking the biquads into one SOS and calling ``sosfiltfilt`` once is the SAME
    FILTER to 8.8e-5 pA [MEASURED: six lines, Q=30, 50 kHz, same pad] -- a cascade of
    notches commutes and composes exactly, as any LTI cascade does. What differs by
    tens of pA is that each ``filtfilt`` applies its own ``3 * max(len(a), len(b))``
    odd-extension padding to the *current* signal, so an N-pass cascade pads N times off
    N different arrays while one stacked call pads once. The difference is negligible in
    the interior and 20-40 pA near the edges -- and windowed inference reads the edges.

    So this form is kept because it is the NOTEBOOKS' OWN and the trained models were
    fed its exact edge behaviour, NOT because the app needs a sequential-cascade
    topology anywhere else. `filters.notch` stacks and filters once, deliberately; the
    two are not in tension.
    (``tests/analysis/test_notch_explicit_lines.py`` carries both measurements.)

    Lines at or above Nyquist are skipped (scipy would raise); the notebooks never
    hit that case at 50 kHz, and skipping keeps a low-rate recording runnable.
    Returns ``filtfilt``'s natural float64, exactly as the notebook's
    ``current_filt`` is float64 before the baseline step casts back to float32.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive, finite sample rate, got {fs!r}")
    q = float(q)
    if not np.isfinite(q) or q <= 0:
        raise ValueError(f"q must be positive and finite, got {q!r}")
    lines = tuple(float(f) for f in freqs)
    bad = [f for f in lines if not np.isfinite(f) or f <= 0]
    if bad:
        raise ValueError(f"notch frequencies must be positive and finite, got {bad}")
    out = np.asarray(sig).copy()
    nyq = fs / 2.0
    for f0 in lines:
        if f0 >= nyq:
            continue
        b, a = iirnotch(f0, Q=q, fs=fs)
        out = filtfilt(b, a, out)
    return out


# ---------------------------------------------------------------------------
# Baseline estimators (dataLoader.ipynb Block 2)
# ---------------------------------------------------------------------------
def _decimated_median(
    x: np.ndarray, fs: float, win_s: float, decim: int
) -> np.ndarray:
    """The decimated rolling median both estimators are built from.

    Verbatim from ``dataLoader.ipynb``::

        n = len(x); xs = x[::decim]; w = int(win_s * fs / decim)
        if w % 2 == 0: w += 1
        b = median_filter(xs, size=w, mode='reflect')
        xp = np.arange(0, n, decim)[:len(b)]
        return np.interp(np.arange(n), xp, b).astype(np.float32)

    The three details that decide bit-exactness: the tap count is computed on the
    DECIMATED grid (``win_s * fs / decim``) and forced ODD by adding one, the
    median filter uses ``mode='reflect'``, and the result is interpolated back to
    full length and cast to float32.

    At 50 kHz: 0.5 s / decim 50 -> 500 -> **501** taps; 0.1 s / decim 50 -> 100 ->
    **101**; 0.03 s / decim 10 -> 150 -> **151**; 0.01 s / decim 10 -> 50 -> **51**.

    TWO deliberate deviations from the notebook text, both for reproducibility:

    1. The decimated view is materialised as a **contiguous** copy before filtering
       (a strided view is not something to hand a C loop that assumes otherwise),
       in **float64**. For an ODD window the median is an order statistic -- it is
       exactly one of the input samples -- so widening the accumulator cannot change
       the value; the notebook itself is always in float64 here anyway, because the
       signal reaching it is ``filtfilt`` output.

    2. ``median_filter`` is called on a ``(1, m)`` array with ``size=(1, w)`` and the
       row taken back out, instead of on the 1-D array with ``size=w``.

    Deviation 2 is a WORKAROUND FOR A REAL BUG, and it is the load-bearing one.
    ``scipy.ndimage.median_filter`` 1.15 is NON-DETERMINISTIC on a **1-D** input once
    the footprint passes ~261 taps: repeated calls on the identical array return
    different values. It is **not** confined to float32 (float64 fails too), it is
    **not** confined to tests, and it is **not** rare -- under the concurrent numpy
    allocation a real session produces it fires on roughly one call in five.
    Measured on the user's -600 mV recording (20 s, 1e6 samples, ``notebook1``
    preset): ``baseline_robust_median`` disagreed with a hand-computed rolling
    median in 13 of 40 runs by up to **7.3 pA**, ``apply_ml_preprocess`` returned
    **6 different answers** in 40 runs spanning **6.4 pA**, and a fixed 5-sigma
    decode over those answers counted anywhere from **337 to 469** events.

    The ``(1, m)`` form takes a different path inside ndimage and is stable: 0 wrong
    in 600 calls under the same load, and bit-identical to the hand-computed median
    (``np.median`` over a ``sliding_window_view`` of a ``symmetric``-padded copy,
    which is what ``mode='reflect'`` means) at every tap count the presets use.

    Since 2026-10-18 the usual case -- finite input, window no longer than the signal --
    runs through :func:`~nanoscope.analysis.acid_incubation_numerics.median_filter_1d`
    (a rolling order statistic over that same symmetric pad) instead: the same values, about
    40x faster at the 2501-tap Host-guest / Zn windows. The ``(1, m)`` call above remains the
    path for non-finite input and for a window longer than the signal.

    Because the 1-D path could return the wrong answer at any time, a digest taken
    through it proves nothing about the run that produced it -- the four ferritin
    contract digests that ride on this function were re-derived after this fix, in a
    quiet single-purpose process, and cross-checked against that hand-computed
    median. See ``tests/analysis/test_ml_preprocess.py``.
    """
    decim = int(decim)
    if decim < 1:
        raise ValueError(f"decim must be >= 1, got {decim}")
    n = len(x)
    xs = np.ascontiguousarray(x[::decim], dtype=np.float64)
    w = int(float(win_s) * float(fs) / decim)
    if w % 2 == 0:
        w += 1
    if len(xs) >= w and np.isfinite(xs).all():
        # Same values as the (1, m) ndimage call below -- an odd-window median is one of the
        # input samples and the padding is the same half-sample reflection -- but a rolling
        # order statistic instead of ndimage's O(n*w) rank filter: 0.6 s against 31 s at
        # 1.5M samples x 2501 taps (the Host-guest / Zn protocols on a 300 s recording).
        # Non-finite input and windows longer than the signal keep the ndimage path, which
        # never raised on them.
        b = median_filter_1d(xs, w, "reflect")
    else:
        b = median_filter(xs[None, :], size=(1, w), mode="reflect")[0]
    xp = np.arange(0, n, decim)[: len(b)]
    return np.interp(np.arange(n), xp, b).astype(np.float32)


def baseline_robust_median(
    sig: np.ndarray, fs: float, *, window_s: float = 0.5, decim: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """``baseline_remove_robust_median`` -- the **notebook 1** baseline.

    A rolling median is outlier-resistant, so deep/long events do not drag it the
    way a Savitzky-Golay polynomial does (no positive-lobe overshoot). It is
    estimated on a decimated copy (the baseline is < 1 Hz, so decimation is
    lossless) and linearly interpolated back.

    ``ferritin_data1`` uses ``window_s=0.5`` (``SG_WINDOW_OVERRIDE``) with the
    default ``decim=50`` -> 501 taps at 50 kHz.

    Returns ``(detrended, baseline)``, both float32.
    """
    x = np.asarray(sig)
    baseline = _decimated_median(x, fs, window_s, decim)
    return (x - baseline).astype(np.float32), baseline


def baseline_masked_median(
    sig: np.ndarray,
    fs: float,
    *,
    rough_s: float = 0.5,
    base_s: float = 0.1,
    decim: int = 50,
    mask_sigma: float = 3.5,
    dilate_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``baseline_remove_masked_median`` -- the **notebook 2 / 3** baseline.

    For recordings with BOTH long events and fast baseline movement, where a plain
    median window cannot be short enough to track the drift without clipping the
    events:

    1. rough median baseline (``rough_s``) to expose the events;
    2. mask samples below ``mask_sigma`` x MAD noise, dilated by ``dilate_ms``;
    3. linearly interpolate the SIGNAL across the masked gaps;
    4. SHORT-window median (``base_s``) of the gap-filled signal -> a baseline that
       tracks drift without being pulled by events.

    The noise estimate and the mask threshold are GLOBAL (one MAD over the whole
    input), which is why ``dataLoader.ipynb`` truncates a recording *before*
    filtering rather than slicing the finished trace -- and why this app measures
    the baseline over the whole channel when the normalization scope is
    ``"recording"``.

    Dilation is ``iterations=max(1, int(dilate_ms * 1e-3 * fs))`` -- 50 iterations
    at 50 kHz for the 1.0 ms default.

    Returns ``(detrended, baseline)``, both float32.
    """
    x = np.asarray(sig)
    rough = x - _decimated_median(x, fs, rough_s, decim)
    noise = 1.4826 * np.median(np.abs(rough - np.median(rough)))
    evm = binary_dilation(
        rough < np.median(rough) - float(mask_sigma) * noise,
        iterations=max(1, int(float(dilate_ms) * 1e-3 * float(fs))),
    )
    idx = np.arange(len(x))
    filled = x.copy()
    if evm.any() and (~evm).sum() > 2:
        filled[evm] = np.interp(idx[evm], idx[~evm], x[~evm])
    baseline = _decimated_median(filled, fs, base_s, decim)
    return (x - baseline).astype(np.float32), baseline


def baseline_savgol(
    sig: np.ndarray,
    fs: float,
    *,
    window_s: float = 0.5,
    poly: int = 3,
    mode: str = "interp",
) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay polynomial baseline -- the APO chains ONLY.

    Used by both apo lineages: the apo notebook's ``remove_baseline_savgol``
    (``window_ms=500.0``, ``polyorder=3``, scipy's default ``mode='interp'``) and
    the older ``apo_raw_global`` path, whose SavGol stage this reproduces exactly
    -- including its window arithmetic: ``w = int(window_s * fs)`` forced odd, then
    clamped to the largest odd length <= ``len(sig)``.

    That clamp is the one deliberate departure from the apo notebook, which
    computes ``25001`` unconditionally. It cannot fire on any real recording
    (25001 samples is 0.5 s at 50 kHz); it only stops scipy raising on a trace
    shorter than the window, where the notebook would simply crash.

    Do NOT reach for this for the FERRITIN models. ``dataLoader.ipynb`` moved off
    it precisely because the polynomial bows around deep/long events and injects
    positive-lobe artifacts (measured overshoot 0.094 -> 0.943 on ``data2``; on
    ``data3`` it fails outright, correlation 0.40 with 6.4x inflated noise).
    ``dataLoader.ipynb``'s own (unused by any of the three notebooks) ``sg`` method
    used ``mode='mirror'`` and no length clamp -- hence the ``mode`` knob. The apo
    notebook's *amplitude* appendix also uses ``mode='mirror'``, but that is a
    separate pA-domain chain that never reaches the model; the model-input path is
    ``'interp'``.

    Returns ``(detrended, baseline)``: detrended float32, baseline as SavGol
    returned it.
    """
    x = np.asarray(sig)
    poly = int(poly)
    w = int(float(window_s) * float(fs))
    if w % 2 == 0:
        w += 1
    max_w = len(x) - 1 if len(x) % 2 == 0 else len(x)
    w = min(w, max_w)
    if w <= poly:
        raise ValueError(
            f"recording is too short for the {window_s * 1000:g} ms / poly-{poly} "
            f"SavGol baseline (need > {poly + 1} samples, got {len(x)})"
        )
    baseline = savgol_filter(x, window_length=w, polyorder=poly, mode=mode)
    return (x - baseline).astype(np.float32), baseline


def despike_median(sig: np.ndarray, *, size: int = 3) -> np.ndarray:
    """``despike_median`` -- post-detrend removal of 1-2 sample spike artifacts.

    ``size=3`` kills isolated 1-sample spikes, ``size=5`` also catches 2-sample
    ones. Safe because real events span many more samples than ``size``.
    ``size`` of 0/None/1 is a no-op (the notebook's ``if not size or size < 2``).
    Only ``ferritin_data2`` uses it (``DESPIKE_SAMPLES = {"ferritin_data2": 3}``,
    ~225 such spikes).

    Uses the same ``(1, m)`` call shape as :func:`_decimated_median`, for the same
    reason. At size 3-5 the 1-D path happens to be stable (0 wrong in 200 calls under
    allocation load, and bit-identical to the ``(1, m)`` result on the golden trace),
    so this changes no number -- but there is no reason to keep a second call site on
    a form that is known to be able to return the wrong answer.
    """
    if not size or int(size) < 2:
        return np.asarray(sig)
    x = np.asarray(sig)
    return median_filter(
        x[None, :], size=(1, int(size)), mode="reflect"
    )[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLPreprocessConfig:
    """Every knob of the ML detector's signal chain, as one auditable value.

    Order of operations in :func:`apply_ml_preprocess`::

        notch (sequential) -> [float32 round] -> baseline removal -> despike

    Normalization is deliberately NOT part of that function: its scale (and its
    centre) may have to be measured over a different span than the slice being
    detrended (see :data:`NORMALIZE_SCOPES`), so the caller applies it via
    :func:`normalize_global` with an explicit :func:`normalization_scale` /
    :func:`normalization_center`.
    """

    preset: str = "custom"
    notch_enabled: bool = True
    notch_freqs: tuple[float, ...] = NOTCH_FREQS
    notch_q: float = NOTCH_Q
    #: Round the notch output back to float32 before the baseline stage.
    #:
    #: ``filtfilt`` returns float64 (its ``b``/``a`` are float64), so passes 2 and 3
    #: run in float64 either way. The question is only what the BASELINE estimator is
    #: handed. The apo notebook's ``apply_notch_filter_harmonics`` ends with
    #: ``return filtered.astype(np.float32)``, so its SavGol runs on float32 and
    #: accumulates in float32; ``dataLoader.ipynb`` has no such cast, so the ferritin
    #: chain keeps float64 all the way into the baseline step. Measured difference on
    #: the real -600 mV apo recording: ``max|d| = 3.05e-05 pA`` in the detrended
    #: signal (five orders below the 0.61 pA ADC step) -- immaterial physically, but
    #: it is the difference between bit-exact and not, so it is a switch, not a guess.
    notch_float32: bool = False
    baseline_method: str = "robust_median"
    #: ``robust_median`` / ``savgol`` window, seconds.
    baseline_window_s: float = 0.5
    #: ``savgol`` polynomial order.
    baseline_poly: int = 3
    #: Decimation for the median-based estimators.
    baseline_decim: int = 50
    #: ``masked_median`` rough (event-exposing) window, seconds.
    masked_rough_s: float = 0.5
    #: ``masked_median`` short (drift-tracking) window, seconds.
    masked_base_s: float = 0.1
    masked_mask_sigma: float = 3.5
    masked_dilate_ms: float = 1.0
    #: Post-detrend median de-spike size in samples; 0 = off.
    despike_size: int = 0
    normalize: bool = True
    normalize_scope: str = "recording"
    #: WHICH normalization (:data:`NORMALIZE_METHODS`). ``"maxabs"`` is the default and
    #: is what every preset, stored run and QSettings-saved user preset written before
    #: this field existed reproduces bit-for-bit -- `from_dict` drops unknown keys and
    #: fills missing ones from these defaults, so an old dict round-trips unchanged.
    #:
    #: ``"robust_sigma"`` selects `filters.robust_normalize`, which is NOT an affine map
    #: (it clips), so :attr:`normalize_center` and the ``scale``/``center`` pair
    #: `normalize_global` takes DO NOT APPLY to it -- it estimates its own centre and
    #: scale from the array it is handed. `model_input_normalization` is the one place
    #: that distinction is made.
    normalize_method: str = DEFAULT_NORMALIZE_METHOD
    #: Clip bound AND divisor for ``"robust_sigma"``, in noise sigmas. Ignored by
    #: ``"maxabs"``. See :data:`DEFAULT_NORMALIZE_CLIP_SIGMA` for why the two cannot be
    #: set apart.
    normalize_clip_sigma: float = DEFAULT_NORMALIZE_CLIP_SIGMA
    #: Subtract the mean BEFORE dividing, i.e. ``(x - mean) / (max|x - mean| + 1e-8)``.
    #:
    #: The apo notebook's ``normalize_signal`` (cell 10) does this; ``student_predict``
    #: (ferritin) does NOT -- it is the pure rescale ``x / (max|x| + 1e-8)``. The
    #: residual mean after a SavGol detrend is tiny (-0.0031 pA on the real -600 mV
    #: recording, 1.2e-05 of full scale) and measurably changed 0 of 236 events there,
    #: but it is what the executed code does, so it is reproduced rather than
    #: "corrected" toward the ferritin form.
    normalize_center: bool = False
    #: Whether inference anchors a final window at ``n - W`` so the tail is covered.
    #: The notebooks do NOT (their loops leave the tail at probability 0); the app's
    #: legacy path does. Material at the end of a recording.
    #:
    #: AND AT THE END OF EVERY ANALYSED SUB-RANGE, which is the part that surprises.
    #: ``n`` is the length of the slice handed to inference, not of the file, so with
    #: ``anchor_tail=False`` the last ``(n - W) % step`` samples of the SLICE sit at
    #: probability 0 -- up to ``step - 1`` = 1535 samples, 30.7 ms at 50 kHz. A
    #: translocation straddling the end of a "Visible view" range is therefore not
    #: merely truncated, it disappears. Measured on the real -600 mV recording: the
    #: event at 1736334..1736999 is found by a whole-file run and absent from a run
    #: over ``[1600000, 1736600)``, whose last window stops at slice sample 135168 of
    #: 136600. This is the notebook's own convention faithfully reproduced (its
    #: recordings have the same blind tail), NOT a scope bug -- the normalization
    #: centre and scale still come from the whole recording. Analyse the whole
    #: recording when an event near the range edge matters.
    anchor_tail: bool = False
    #: Whether the window-start loop may START at ``n - W`` (bound ``n - W + 1``).
    #:
    #: The two notebook families differ here, and only here, on the tail:
    #: ferritin's ``range(0, n - W, step)`` (False) vs apo's
    #: ``np.arange(0, max(1, n - W + 1), step)`` (True). They are identical UNLESS
    #: ``(n - W) % step == 0``, in which case ferritin emits one window fewer and
    #: leaves a whole stride at probability 0 while apo covers the recording exactly.
    #: Ignored when :attr:`anchor_tail` is set (anchoring already reaches ``n - W``).
    last_start_inclusive: bool = False

    def __post_init__(self) -> None:
        def finite_float(name: str, *, positive: bool = False,
                         nonnegative: bool = False) -> float:
            raw = getattr(self, name)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number, got {raw!r}") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {raw!r}")
            if positive and value <= 0:
                raise ValueError(f"{name} must be > 0, got {raw!r}")
            if nonnegative and value < 0:
                raise ValueError(f"{name} must be >= 0, got {raw!r}")
            return value

        def exact_int(name: str, *, minimum: int) -> int:
            raw = getattr(self, name)
            value = finite_float(name)
            if not value.is_integer():
                raise ValueError(f"{name} must be an integer, got {raw!r}")
            integer = int(value)
            if integer < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {raw!r}")
            return integer

        if self.baseline_method not in BASELINE_METHODS:
            raise ValueError(
                f"baseline_method must be one of {BASELINE_METHODS}, "
                f"got {self.baseline_method!r}"
            )
        if self.normalize_scope not in NORMALIZE_SCOPES:
            raise ValueError(
                f"normalize_scope must be one of {NORMALIZE_SCOPES}, "
                f"got {self.normalize_scope!r}"
            )
        if self.normalize_method not in NORMALIZE_METHODS:
            raise ValueError(
                f"normalize_method must be one of {NORMALIZE_METHODS}, "
                f"got {self.normalize_method!r}"
            )
        finite_float("normalize_clip_sigma", positive=True)
        finite_float("notch_q", positive=True)
        exact_int("baseline_decim", minimum=1)
        exact_int("baseline_poly", minimum=1)
        exact_int("despike_size", minimum=0)
        for name in ("baseline_window_s", "masked_rough_s", "masked_base_s"):
            finite_float(name, positive=True)
        finite_float("masked_mask_sigma", positive=True)
        finite_float("masked_dilate_ms", nonnegative=True)
        # Normalize the freq container so equality/round-trip are stable.
        try:
            notch_freqs = tuple(float(f) for f in self.notch_freqs)
        except (TypeError, ValueError) as exc:
            raise ValueError("notch_freqs must be a sequence of finite positive numbers") from exc
        bad = [f for f in notch_freqs if not np.isfinite(f) or f <= 0]
        if bad:
            raise ValueError(
                f"notch_freqs must contain only finite positive frequencies, got {bad}"
            )
        if self.notch_enabled and not notch_freqs:
            raise ValueError(
                "notch_freqs must name at least one line when notch_enabled is True"
            )
        object.__setattr__(self, "notch_freqs", notch_freqs)

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict (``notch_freqs`` becomes a list) for params/provenance."""
        d = asdict(self)
        d["notch_freqs"] = [float(f) for f in self.notch_freqs]
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MLPreprocessConfig | None":
        """Rebuild from :meth:`to_dict`. ``None``/empty in -> ``None`` out, so a
        caller can write ``from_dict(params.get(...)) or DEFAULT``."""
        if not data:
            return None
        if isinstance(data, cls):
            return data
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "notch_freqs" in kwargs:
            kwargs["notch_freqs"] = tuple(float(f) for f in kwargs["notch_freqs"])
        return cls(**kwargs)

    def replace(self, **kw: Any) -> "MLPreprocessConfig":
        return replace(self, **kw)

    def deviations_from(self, other: "MLPreprocessConfig") -> tuple[str, ...]:
        """Field names whose values differ from ``other`` (``preset`` excluded).

        The UI uses this to say "custom (2 changes from Notebook 2)" -- consistency
        with training matters, so a deviation must be visible, not silent.
        """
        return tuple(
            f.name
            for f in fields(self)
            if f.name != "preset" and getattr(self, f.name) != getattr(other, f.name)
        )

    def describe(self) -> str:
        """A one-line human summary, used in provenance and the UI."""
        parts = []
        if self.notch_enabled:
            lines = "/".join(f"{f:g}" for f in self.notch_freqs)
            rounded = ", float32-rounded" if self.notch_float32 else ""
            parts.append(
                f"notch {lines} Hz Q={self.notch_q:g} (sequential filtfilt{rounded})"
            )
        else:
            parts.append("no notch")
        if self.baseline_method == "robust_median":
            parts.append(
                f"robust median baseline {self.baseline_window_s:g}s decim {self.baseline_decim}"
            )
        elif self.baseline_method == "masked_median":
            parts.append(
                f"masked median baseline rough {self.masked_rough_s:g}s -> base "
                f"{self.masked_base_s:g}s decim {self.baseline_decim} "
                f"mask {self.masked_mask_sigma:g}sigma dilate {self.masked_dilate_ms:g}ms"
            )
        elif self.baseline_method == "savgol":
            parts.append(
                f"SavGol baseline {self.baseline_window_s * 1000:g}ms poly{self.baseline_poly}"
            )
        else:
            parts.append("no baseline removal")
        if self.despike_size and self.despike_size >= 2:
            parts.append(f"despike median size {self.despike_size}")
        if self.normalize and self.normalize_method == "robust_sigma":
            # The centre/scale words the max-abs branch uses would be wrong here: this
            # kernel re-estimates both itself, twice-trimmed, and then CLIPS.
            parts.append(
                f"normalize twice-trimmed median/MAD, clip +-"
                f"{self.normalize_clip_sigma:g} sigma then divide by it, over the "
                f"{self.normalize_scope} (bounded to [-1, +1])"
            )
        elif self.normalize:
            formula = (
                "(x-mean)/(max|x-mean|+1e-8)" if self.normalize_center
                else "x/(max|x|+1e-8)"
            )
            parts.append(f"normalize {formula} over the {self.normalize_scope}")
        if self.anchor_tail:
            parts.append("tail window anchored")
        elif self.last_start_inclusive:
            parts.append("starts arange(0, n-W+1, step) (apo notebook)")
        else:
            parts.append("tail uncovered (notebook)")
        return "; ".join(parts)


@dataclass(frozen=True)
class MLDecodeConfig:
    """The ``postprocess_predictions`` contract from the StudentModel notebooks.

    Pipeline (order matters, and it is the notebook's order)::

        median-smooth -> hysteresis(enter/exit) -> merge -> onset snap
        -> offset trim -> min-duration -> AMPLITUDE GATES -> regions

    ``min_event_ms`` runs LAST **within postprocess_predictions** so the duration
    test uses the final, merged + trimmed extent -- as the notebook comment
    insists. The three amplitude gates run LATER STILL: they are not part of
    ``postprocess_predictions`` at all. In cell 15 of every StudentInference
    notebook they are applied to the list of regions that function RETURNS, i.e.
    strictly after the min-duration filter, and they measure the FINAL merged
    extent (``sig[s:e+1]``). See :func:`decode_ml_regions`.
    """

    preset: str = "custom"
    smoothing_kernel: int = 9
    enter_threshold: float = 0.5
    exit_threshold: float = 0.15
    merge_radius_ms: float = 0.5
    min_event_ms: float = 1.0
    #: Onset snap: pull the onset back to where the current entered the blockade
    #: (< ``onset_sigma`` x MAD noise below the median). 0 disables it.
    onset_sigma: float = 0.0
    onset_max_ms: float = 5.0
    onset_gap_tol: int = 5
    #: Offset trim: snap the offset to the last sample deeper than
    #: ``offset_sigma`` x MAD noise. 0 disables it.
    offset_sigma: float = 0.0

    # -- the three AMPLITUDE gates (cell 15) ---------------------------------
    # ALL THREE DEFAULT TO 0.0 == OFF, and that is load-bearing rather than
    # tidy: every preset stored before these fields existed, every saved run,
    # and every QSettings-persisted user preset comes back through
    # ``from_dict`` without these keys and must reproduce its old behaviour
    # BIT-FOR-BIT. A default of 4.0 here would silently re-gate every historic
    # apo and ferritin result the moment it was re-opened.
    #
    # ``0.0`` means "this clause did not run", NOT "a threshold of zero" -- a
    # zero threshold would still reject an upward (negative-depth) excursion.
    #: Sustained-depth gate in event-excluded MADs, on ``median`` over the event
    #: (never the deepest sample). 0 disables it. See
    #: :func:`nanoscope.analysis.event_features.sustained_gate`, which is where
    #: it is actually measured -- there is one implementation, not two.
    min_depth_sigma: float = 0.0
    #: ABSOLUTE sustained-depth floor in pA. Pore-independent and blunt; it does
    #: not widen when the recording's noise does. 0 disables it.
    min_depth_pa: float = 0.0
    #: ABSOLUTE sustained dI/I0 floor against the TIME-LOCAL open-pore current.
    #: Needs the RAW trace, which a probability-only decode does not have. 0
    #: disables it.
    min_fracblock: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.enter_threshold) <= 1.0):
            raise ValueError(
                f"enter_threshold must be in (0, 1], got {self.enter_threshold}"
            )
        if not (0.0 <= float(self.exit_threshold) <= float(self.enter_threshold)):
            raise ValueError(
                "exit_threshold must satisfy 0 <= exit <= enter, got "
                f"exit={self.exit_threshold}, enter={self.enter_threshold}"
            )
        if float(self.merge_radius_ms) < 0 or float(self.min_event_ms) < 0:
            raise ValueError("merge_radius_ms / min_event_ms must be >= 0")
        if int(self.onset_gap_tol) < 0:
            raise ValueError(f"onset_gap_tol must be >= 0, got {self.onset_gap_tol}")
        # The gates are validated the way every other field here is -- on the way
        # in, so an impossible pipeline is refused where it is DIALLED rather than
        # halfway through a five-minute inference run. NaN is rejected explicitly:
        # ``nan >= x`` is False for every event, so a NaN threshold is a silent
        # "detected nothing" that looks exactly like a quiet recording.
        for _name in ("min_depth_sigma", "min_depth_pa", "min_fracblock"):
            _v = float(getattr(self, _name))
            if not np.isfinite(_v) or _v < 0:
                raise ValueError(
                    f"{_name} must be finite and >= 0 (0 = gate off), "
                    f"got {getattr(self, _name)!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MLDecodeConfig | None":
        if not data:
            return None
        if isinstance(data, cls):
            return data
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def replace(self, **kw: Any) -> "MLDecodeConfig":
        return replace(self, **kw)

    def deviations_from(self, other: "MLDecodeConfig") -> tuple[str, ...]:
        return tuple(
            f.name
            for f in fields(self)
            if f.name != "preset" and getattr(self, f.name) != getattr(other, f.name)
        )

    def describe(self) -> str:
        parts = [
            f"median smooth k={self.smoothing_kernel}",
            f"hysteresis enter {self.enter_threshold:g} / exit {self.exit_threshold:g}",
            f"merge <= {self.merge_radius_ms:g} ms",
        ]
        if self.onset_sigma:
            parts.append(
                f"onset snap {self.onset_sigma:g}sigma (<= {self.onset_max_ms:g} ms, "
                f"gap tol {self.onset_gap_tol})"
            )
        if self.offset_sigma:
            parts.append(f"offset trim {self.offset_sigma:g}sigma")
        parts.append(f"drop < {self.min_event_ms:g} ms")
        # The gates come after the duration filter in the string because they come
        # after it in the pipeline. A provenance stamp that reorders the stages is
        # a provenance stamp that cannot be used to reproduce the run.
        gates = []
        if self.min_depth_sigma:
            gates.append(f">= {self.min_depth_sigma:g}sigma sustained")
        if self.min_depth_pa:
            gates.append(f">= {self.min_depth_pa:g} pA sustained")
        if self.min_fracblock:
            gates.append(f">= {self.min_fracblock:g} dI/I0 (time-local I0)")
        if gates:
            parts.append("amplitude gate " + " AND ".join(gates))
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
#: The APO NOTEBOOK preprocessing contract, shared verbatim by all three biases.
#:
#: ``ApoFerritin_LSTM_Cluster_Validation_RAM_Safe.ipynb`` cell 12 calls exactly one
#: ``preprocess_signal`` on the full recording for every trace; only the DECODE is
#: per-bias. Reproducing this chain on the real -600 mV recording gives
#: ``sha256 = 382b0b02...fded`` for the model input and, re-running the notebook's
#: own inference + decode on it, its stored 236 events (and 982 on -700 mV).
#:
#: THE MODEL INPUT IS BIT-EXACT; THE EVENT INDICES ARE NOT, AND CANNOT BE. Only
#: the preprocessing above is deterministic across machines -- inference is not.
#: The student runs in mixed float16, so its probability trace differs in the low
#: bits between the notebook's Colab device and any other, and near a threshold as
#: low as 0.1 that moves a boundary by a sample or three. Measured: the notebook
#: PRINTS its shortest -600 mV event as start 1736337 / end 1736999 (662 samples);
#: re-executing the notebook's own cells verbatim on this machine gives start
#: 1736334 / end 1736999 (665) -- and not as a tie-break, ``preds[1736334]`` is
#: 0.156 here, well clear of 0.1. So the reproducible claims are the model-input
#: digest and the event COUNT; a published onset index is not one of them, and
#: this app lands on exactly the same indices as a verbatim re-run of the notebook
#: in the same environment.
_APO_NB_PRE = MLPreprocessConfig(
    notch_enabled=True,
    notch_freqs=NOTCH_FREQS,
    notch_q=NOTCH_Q,
    notch_float32=True,          # apply_notch_filter_harmonics ends in .astype(float32)
    baseline_method="savgol",
    baseline_window_s=0.5,       # int(500 ms * 50 kHz / 1000) -> 25000 -> odd -> 25001
    baseline_poly=3,
    despike_size=0,              # no de-spike, no low-pass, no clipping anywhere
    normalize=True,
    normalize_scope="recording",
    normalize_center=True,       # (x - mean) / (max|x - mean| + 1e-8)
    anchor_tail=False,
    last_start_inclusive=True,   # np.arange(0, max(1, n - W + 1), step)
)

#: The preprocessing chain each trained model was fed, keyed by preset name.
#:
#: ==============  ================================================================
#: preset           baseline / despike
#: ==============  ================================================================
#: notebook1        robust median, window 0.5 s, decim 50 (501 taps @ 50 kHz); no despike
#: notebook2        masked median, rough 0.5 s / base 0.1 s, decim 50 (501 / 101 taps),
#:                  mask 3.5 sigma, dilate 1.0 ms (50 iterations @ 50 kHz); despike size 3
#: notebook3        masked median, rough 0.03 s / base 0.01 s, decim 10 (151 / 51 taps),
#:                  mask 3.5 sigma, dilate 1.0 ms; no despike
#: apo_nb_*         SavGol 500 ms / poly 3 (25001 taps @ 50 kHz), mode='interp'; no
#:                  despike; float32-rounded notch; MEAN-CENTRED normalization;
#:                  window starts reach n - W. Identical for all three biases.
#: apo_raw_global   SavGol 500 ms / poly 3; no despike; tail window anchored
#: ==============  ================================================================
#:
#: All of them notch 60/120/180 Hz at Q=30 as three sequential ``filtfilt`` passes
#: and normalize over the whole recording. The apo notebook presets are the only
#: ones that subtract the mean first (:attr:`MLPreprocessConfig.normalize_center`)
#: and the only ones that round the notch output to float32.
PREPROCESS_PRESETS: dict[str, MLPreprocessConfig] = {
    "notebook1": MLPreprocessConfig(
        preset="notebook1",
        baseline_method="robust_median",
        baseline_window_s=0.5,
        baseline_decim=50,
        despike_size=0,
        anchor_tail=False,
    ),
    "notebook2": MLPreprocessConfig(
        preset="notebook2",
        baseline_method="masked_median",
        masked_rough_s=0.5,
        masked_base_s=0.1,
        baseline_decim=50,
        masked_mask_sigma=3.5,
        masked_dilate_ms=1.0,
        despike_size=3,
        anchor_tail=False,
    ),
    "notebook3": MLPreprocessConfig(
        preset="notebook3",
        baseline_method="masked_median",
        masked_rough_s=0.03,
        masked_base_s=0.01,
        baseline_decim=10,
        masked_mask_sigma=3.5,
        masked_dilate_ms=1.0,
        despike_size=0,
        anchor_tail=False,
    ),
    "apo_nb_500mV": _APO_NB_PRE.replace(preset="apo_nb_500mV"),
    "apo_nb_600mV": _APO_NB_PRE.replace(preset="apo_nb_600mV"),
    "apo_nb_700mV": _APO_NB_PRE.replace(preset="apo_nb_700mV"),
    "apo_raw_global": MLPreprocessConfig(
        preset="apo_raw_global",
        baseline_method="savgol",
        baseline_window_s=0.5,
        baseline_poly=3,
        despike_size=0,
        anchor_tail=True,
    ),
    # -- the three student pipelines that use robust-sigma normalization -------
    #
    # All three come from the owner's OWN notebooks and were cross-checked against the
    # ``*_metadata.json`` his code writes beside every processed recording. They share
    # the model-input normalization (`filters.robust_normalize`, clip 30) that REPLACED
    # ``x/max|x|`` in all six of those notebooks, and they differ in exactly two places:
    # the notch line set and the baseline estimator.
    #
    # THE NOTCH LINE SETS ARE NOT SERIES. Both were found by a Welch census and both
    # have GAPS, which is why they are spelled out: host-guest hums at 60/120/180/300/
    # 420/540 with 240, 360 and 480 quiet; the Zn/Cu rig at 60/120/180/240/300/360/420/
    # 540/660 with 480 absent.
    "hg_student": MLPreprocessConfig(
        preset="hg_student",
        notch_enabled=True,
        notch_freqs=(60.0, 120.0, 180.0, 300.0, 420.0, 540.0),
        notch_q=30.0,
        baseline_method="masked_median",
        masked_rough_s=0.5,
        masked_base_s=0.02,
        baseline_decim=10,
        masked_mask_sigma=3.5,
        masked_dilate_ms=0.2,
        despike_size=0,
        normalize=True,
        normalize_scope="recording",
        normalize_method="robust_sigma",
        normalize_clip_sigma=DEFAULT_NORMALIZE_CLIP_SIGMA,
        anchor_tail=False,
    ),
    "zn_student": MLPreprocessConfig(
        preset="zn_student",
        notch_enabled=True,
        notch_freqs=(60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0),
        notch_q=30.0,
        # A Savitzky-Golay baseline was TESTED AND REJECTED on this data: the Zn events
        # are shallow and abundant, so a polynomial envelope dives into them. The
        # event-masked median is the estimator that survived, and it is not a default.
        baseline_method="masked_median",
        masked_rough_s=0.5,
        masked_base_s=0.2,
        baseline_decim=10,
        masked_mask_sigma=3.0,
        masked_dilate_ms=2.0,
        despike_size=0,
        normalize=True,
        normalize_scope="recording",
        normalize_method="robust_sigma",
        normalize_clip_sigma=DEFAULT_NORMALIZE_CLIP_SIGMA,
        anchor_tail=False,
    ),
    # SAVITZKY-GOLAY, NOT MASKED MEDIAN, and the disagreement is on the record: the
    # Cu notebook's own cell-0 MARKDOWN still says "event-masked robust median" and is
    # WRONG. Block 0's ``BASELINE_METHOD`` dict says ``"sg"`` for all four cu_*
    # recordings and all four written ``cu_*_metadata.json`` agree, with
    # ``masked_median_params`` null. The code and the written metadata are the truth;
    # the prose is stale. (Cu events are DEEP and RARE, which is why the estimator that
    # was rejected for Zn is the right one here.)
    "cu_student": MLPreprocessConfig(
        preset="cu_student",
        notch_enabled=True,
        notch_freqs=(60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0),
        notch_q=30.0,
        baseline_method="savgol",
        baseline_window_s=0.2,
        baseline_poly=3,
        baseline_decim=50,
        despike_size=0,
        normalize=True,
        normalize_scope="recording",
        normalize_method="robust_sigma",
        normalize_clip_sigma=DEFAULT_NORMALIZE_CLIP_SIGMA,
        anchor_tail=False,
    ),
}

#: The decode contract each model's notebook used, keyed by the same preset names.
#:
#: ==============  =======  =====  =====  ========  =======  ==========  ==========
#: preset           smooth  enter   exit  merge ms  min ms   onset snap  offset trim
#: ==============  =======  =====  =====  ========  =======  ==========  ==========
#: notebook1              9    0.5   0.15       0.5      1.0  off         off
#: notebook2              9    0.5   0.15       0.5      1.0  off         off
#: notebook3              5    0.5   0.20       0.2     0.15  2.0 sigma   3.0 sigma
#: apo_nb_500mV           1    0.2   0.20       0.0     15.0  off         off
#: apo_nb_600mV           1    0.1   0.10       0.0      1.0  off         off
#: apo_nb_700mV           1    0.2   0.20       0.0      1.0  off         off
#: apo_raw_global         9    0.5   0.30       0.1      0.1  off         off
#: hg_student             3    0.4   0.15      0.05      0.1  off         off
#: zn_student             3    0.4   0.15      0.05      0.1  off         off
#: cu_student             5    0.5   0.15       0.1     0.05  off         off
#: ==============  =======  =====  =====  ========  =======  ==========  ==========
#:
#: ...plus the AMPLITUDE GATES, which only the three student presets use and which
#: run AFTER the min-duration filter on the final merged extent (see
#: :func:`decode_ml_regions`); ``0`` is off everywhere else:
#:
#: ==============  ===============  ============  ==============
#: preset          min_depth_sigma  min_depth_pa  min_fracblock
#: ==============  ===============  ============  ==============
#: hg_student                  4.0           off             off
#: zn_student                  4.0           off             off
#: cu_student                  4.0          10.0            0.05
#: ==============  ===============  ============  ==============
#:
#: notebook3's onset snap uses ``onset_max_ms=5.0`` and ``gap_tol=5`` (the
#: ``postprocess_predictions`` defaults its ``STUDENT_PP`` leaves untouched).
#:
#: THE APO NOTEBOOK DECODE IS NOT THE FERRITIN DECODE. ``extract_event_regions``
#: (cell 13) is a bare run-length decode of ``preds > threshold``: NO probability
#: smoothing at all (``predict_full_signal_batched``'s docstring calls its output
#: "Smoothed" -- it is not; it is a plain overlap-average), NO hysteresis (one
#: threshold, not an enter/exit pair), and NO merge inside the decoder. The
#: three-field spelling above expresses that with the fields this module already
#: has, and it was verified to reproduce the notebook REGION-FOR-REGION on the real
#: data (236/236 on -600 mV, 828/828 on -500 mV):
#:
#: * ``smoothing_kernel=1`` -- :func:`smooth_probability` short-circuits at <= 1.
#: * ``enter == exit`` -- hysteresis with equal thresholds IS a single threshold.
#: * ``merge_radius_ms=0.0`` -- hysteresis regions are always >= 2 samples apart,
#:   so a 0 ms radius can never merge.
#:
#: One residual, measured, immaterial difference: the notebook tests ``preds >
#: threshold`` (strict) where :func:`extract_regions_hysteresis` tests ``>=``.
#: On the real recordings EXACTLY ZERO samples equal 0.1 or 0.2, so no region moves.
#: (The notebook also exempts a run still open at the end of the trace from its
#: min-length test; this app applies the test to every region. That can only ever
#: affect one region, at the very end of a recording, and dropping an under-length
#: event there is the defensible behaviour.)
#:
#: THE THRESHOLDS ARE PER-RECORDING AND HAND-TUNED, not a canonical apo constant --
#: cells 14/15/16 use 0.2 / 0.1 / 0.2 with min 1 / 1 / 15 ms, and cell 8's own
#: comment admits "you used 0.084 in that script for -500 mV". Hence three presets
#: named after the bias they were tuned at, rather than one invented average.
#: Merging and dwell filtering in that notebook are a SEPARATE, later analysis step
#: (cell 54: radius 1.5 / 20 / 40 ms, dwell 0.3-60 / 5-40 / 5-40 ms), empirically a
#: no-op at these radii (cell 48 printed 236 == 236), and they are deliberately NOT
#: folded into detection here -- the app has its own merge/filter stages for that.
#:
#: ONE KNOWN, MEASURED DEVIATION (notebook 1 / notebook 2 only). Those two
#: notebooks call ``postprocess_predictions(..., signal=current_detr,
#: offset_sigma=0, onset_sigma=0)``. The notebook gates the onset snap on
#: ``if onset_sigma:`` but does NOT gate the offset trim, so at ``offset_sigma=0``
#: it still runs ``_trim_offsets_to_deep`` with ``deep_level = median(signal) -
#: 0 * noise == median(signal)`` -- i.e. it snaps each offset back to the region's
#: last sample below the median. This app treats ``offset_sigma == 0`` as OFF,
#: which is what the notebook author plainly intended (0 is the documented
#: "disabled" value, and trimming at the bare median is not a physical operation).
#: The divergence is real but negligible: measured against a verbatim
#: reimplementation of ``postprocess_predictions`` on the actual student
#: probability traces, 0 of 210 regions moved on ``ferritin_data1`` and 1 of 376
#: on ``ferritin_data2``, with a maximum offset shift of ONE sample (20 us against
#: a ~6 ms median dwell). Regions never move on the onset side, the event COUNT
#: never changes, and the app's offset is never EARLIER than the notebook's --
#: ``test_nb1_nb2_deviate_from_the_notebook_only_on_the_offset_side`` pins exactly
#: that characterisation (the samples the app keeps are precisely the trailing run
#: at-or-above ``median(signal)``). notebook3 is unaffected and byte-faithful (its
#: ``offset_sigma`` is 3.0, so both implementations trim identically) --
#: ``test_notebook3_decode_is_verbatim_postprocess_predictions``.
DECODE_PRESETS: dict[str, MLDecodeConfig] = {
    "notebook1": MLDecodeConfig(
        preset="notebook1",
        smoothing_kernel=9,
        enter_threshold=0.5,
        exit_threshold=0.15,
        merge_radius_ms=0.5,
        min_event_ms=1.0,
        onset_sigma=0.0,
        offset_sigma=0.0,
    ),
    "notebook2": MLDecodeConfig(
        preset="notebook2",
        smoothing_kernel=9,
        enter_threshold=0.5,
        exit_threshold=0.15,
        merge_radius_ms=0.5,
        min_event_ms=1.0,
        onset_sigma=0.0,
        offset_sigma=0.0,
    ),
    "notebook3": MLDecodeConfig(
        preset="notebook3",
        smoothing_kernel=5,
        enter_threshold=0.5,
        exit_threshold=0.2,
        merge_radius_ms=0.2,
        min_event_ms=0.15,
        onset_sigma=2.0,
        onset_max_ms=5.0,
        onset_gap_tol=5,
        offset_sigma=3.0,
    ),
    # -- apo notebook, one entry per hand-tuned bias (cells 14 / 15 / 16) -------
    "apo_nb_500mV": MLDecodeConfig(
        preset="apo_nb_500mV",
        smoothing_kernel=1,          # no probability smoothing at all
        enter_threshold=0.2,
        exit_threshold=0.2,          # enter == exit -> a single threshold
        merge_radius_ms=0.0,         # merging is a separate later step, not decode
        min_event_ms=15.0,
        onset_sigma=0.0,
        offset_sigma=0.0,
    ),
    "apo_nb_600mV": MLDecodeConfig(
        preset="apo_nb_600mV",
        smoothing_kernel=1,
        enter_threshold=0.1,
        exit_threshold=0.1,
        merge_radius_ms=0.0,
        min_event_ms=1.0,
        onset_sigma=0.0,
        offset_sigma=0.0,
    ),
    "apo_nb_700mV": MLDecodeConfig(
        preset="apo_nb_700mV",
        smoothing_kernel=1,
        enter_threshold=0.2,
        exit_threshold=0.2,
        merge_radius_ms=0.0,
        min_event_ms=1.0,
        onset_sigma=0.0,
        offset_sigma=0.0,
    ),
    # -- the older raw-global apo model ----------------------------------------
    # WARNING, and it is why this preset was renamed: these five decode numbers are
    # verbatim the SIGNATURE DEFAULTS of ``postprocess_predictions`` (identical in
    # ApoStudentModelMerged c11, HypotheticalStudentModel_Apo c11,
    # StudentDetector_Morphology2D_Apo3 c6 and _v2_5 c6), and EVERY call site in
    # every one of those notebooks overrides them. They are not the decode of any
    # executed run. FinalModel.ipynb -- which actually trains
    # student_raw_global_raw_best -- decodes with a SINGLE threshold plus a 1 ms
    # minimum, thresholds hand-tuned per cell (0.35 / 0.15 / 0.55 / 0.6 / 0.084 /
    # 0.25 / 0.8). They are preserved unchanged only so an existing saved run keeps
    # reproducing; treat them as a starting point, not as a contract.
    "apo_raw_global": MLDecodeConfig(
        preset="apo_raw_global",
        smoothing_kernel=9,
        enter_threshold=0.5,
        exit_threshold=0.3,
        merge_radius_ms=0.1,
        min_event_ms=0.1,
        onset_sigma=0.0,
        offset_sigma=0.0,
    ),
    # -- host-guest / Zn2 / Cu2: THE REAL DETECTOR_CONTRACT --------------------
    #
    # SOURCE. Cell 1 of each StudentInference notebook, read directly:
    #
    #   hg_student  ~/Desktop/HostGuest_StudentInference_corrected.ipynb
    #   zn_student  ~/Desktop/Zn2IonCondition/Zn2Insulin_StudentInference.ipynb
    #   cu_student  ~/Desktop/Cu2Data/Cu2Insulin_StudentInference.ipynb
    #
    # =========  ======  =====  =====  =======  ======  =====  ======  =========
    # preset     smooth  enter   exit  merge ms  min ms  sigma   pA    dI/I0
    # =========  ======  =====  =====  =======  ======  =====  ======  =========
    # hg_student      3    0.4   0.15     0.05     0.1    4.0   off      off
    # zn_student      3    0.4   0.15     0.05     0.1    4.0   off      off
    # cu_student      5    0.5   0.15     0.1     0.05    4.0   10.0     0.05
    # =========  ======  =====  =====  =======  ======  =====  ======  =========
    #
    # WHY THERE IS A SINGLE CONTRACT AT ALL -- the drift story, in cell 1's own
    # words: these numbers "were repeated at each call site and had drifted: the
    # single-recording event summary ran enter=0.5 while _xbias_events (_xinc_events
    # for Zn/Cu) ran enter=0.4, so THAT SUMMARY DESCRIBED DIFFERENT DETECTIONS FROM
    # EVERY DOWNSTREAM COMPARISON BUILT ON IT." The fix in the notebooks was to
    # define ``DETECTOR_CONTRACT`` once and pass it everywhere. These dict entries
    # are this app's copy of that single definition, and they are the reason the
    # gate itself is a call into ``event_features`` rather than a local
    # reimplementation -- a second copy of a number is how the first drift happened.
    #
    # ONSET SNAP / OFFSET TRIM ARE OFF, AND THE DEFAULTS ARE NOT WHAT RAN. The
    # signature of ``postprocess_predictions`` (cell 4) reads
    # ``offset_sigma=3.0, onset_sigma=2.0, onset_max_ms=5.0`` -- but EVERY REAL CALL
    # SITE, cells 8 and 15 in all three notebooks, passes ``offset_sigma=0,
    # onset_sigma=0``. Verified in all six call sites. Shipping the signature
    # defaults here would ship a decode that has never been executed on this data
    # (which is the exact mistake ``apo_raw_global`` above documents).
    #
    # -- CU'S RAISED ENTER (0.5, where hg and zn use 0.4) ---------------------
    # Deliberate, and cell 1 says why, so it is repeated here rather than left to be
    # "tidied" back to 0.4 for consistency with its siblings. The Cu recordings are
    # detected by the ZN-TRAINED student (no Cu recording is in-domain), and that
    # student's response to PURE NOISE peaks at 0.49 (measured) -- which sits right
    # on a 0.4 gate. Deep-event recall on cu_30h is IDENTICAL across enter
    # 0.4 / 0.5 / 0.6 / 0.7, so the raise costs nothing and buys margin. (Cell 1's
    # comment quotes that flat recall as 81%; the transfer note above it re-measured
    # 76% on the current Savitzky-Golay CSVs after the masked-median ones were
    # superseded. Both numbers describe the same result -- flat across thresholds --
    # and the 81% figure no longer describes any data on disk.) Precision here is
    # carried by MIN_DEPTH_SIGMA, not by the probability gate.
    #
    # -- CU'S TWO ABSOLUTE GATES (added 2026-08-26) ---------------------------
    # MIN_EVENT_MS is the DWELL filter; these two are its amplitude counterpart, and
    # they exist because A SIGMA-SCALED GATE ALONE DOES NOT STOP BASELINE
    # FLUCTUATIONS BEING CALLED EVENTS: MIN_DEPTH_SIGMA scales with each recording's
    # own noise, so wherever the baseline wanders or the noise band widens, the gate
    # widens with it and a fluctuation still clears it. The two below are absolute --
    # they do not move when the baseline misbehaves:
    #
    #   min_depth_pa = 10.0   Blunt and pore-independent. Calibrated to sit just
    #                         above the typical 4-sigma gate (7-10 pA on these pores)
    #                         and far below the real events (20-140 pA).
    #   min_fracblock = 0.05  The PRINCIPLED one: a sustained dI/I0 floor against the
    #                         TIME-LOCAL I0, which is what partially normalizes pore
    #                         size, and which bites hardest exactly where the sigma
    #                         gate is weakest -- cu_10h has I0 = 316 pA, so its
    #                         4-sigma gate is only dI/I0 ~ 0.023, low enough to admit
    #                         baseline wander wholesale. 0.05 sits below every
    #                         recording's event core (per-tag median dI/I0
    #                         0.24 / 0.11 / 0.28 / 0.62 for cu_2h/10h/18h/30h) and
    #                         matches F_MIN in Cu2Insulin_IntraEventAnalysis.ipynb,
    #                         so the two notebooks agree about what an event is.
    #
    # hg and zn leave both at 0.0 (off) because their cell 1 does not define them --
    # their DETECTOR_CONTRACT carries min_depth_sigma alone. That is a real
    # difference between the projects, not an omission to be harmonised away.
    #
    # ORDER: all three gates run AFTER the min-duration filter, on the final merged
    # extent. See :func:`decode_ml_regions`; the notebook's structure is what fixes
    # this, not a preference.
    "hg_student": MLDecodeConfig(
        preset="hg_student",
        smoothing_kernel=3,
        enter_threshold=0.4,
        exit_threshold=0.15,
        merge_radius_ms=0.05,
        min_event_ms=0.1,          # host-guest dwells are 0.1-2 ms
        onset_sigma=0.0,           # cells 8 + 15 pass onset_sigma=0 explicitly
        offset_sigma=0.0,          # ...and offset_sigma=0
        min_depth_sigma=4.0,       # SUSTAINED depth (median over the event)
    ),
    "zn_student": MLDecodeConfig(
        preset="zn_student",
        smoothing_kernel=3,
        enter_threshold=0.4,
        exit_threshold=0.15,
        merge_radius_ms=0.05,
        min_event_ms=0.1,
        onset_sigma=0.0,
        offset_sigma=0.0,
        min_depth_sigma=4.0,
    ),
    "cu_student": MLDecodeConfig(
        preset="cu_student",
        smoothing_kernel=5,
        enter_threshold=0.5,       # RAISED from 0.4 -- see the noise-peak note above
        exit_threshold=0.15,
        merge_radius_ms=0.1,
        min_event_ms=0.05,
        onset_sigma=0.0,
        offset_sigma=0.0,
        min_depth_sigma=4.0,
        min_depth_pa=10.0,         # absolute, pore-independent
        min_fracblock=0.05,        # absolute, vs the TIME-LOCAL I0
    ),
}

#: Preset order for the UI combo -- grouped by molecule, ferritin first.
PRESET_NAMES: tuple[str, ...] = (
    "notebook1",
    "notebook2",
    "notebook3",
    "apo_nb_500mV",
    "apo_nb_600mV",
    "apo_nb_700mV",
    "apo_raw_global",
    "hg_student",
    "zn_student",
    "cu_student",
)

#: Old preset keys -> their current name, so a stored run or a saved project that
#: names a superseded preset still resolves instead of raising.
#:
#: ``apo_legacy`` was renamed to ``apo_raw_global`` when the REAL apo notebook
#: pipeline landed. The rename is the point: "legacy" implied "the apo pipeline,
#: older version", which it is not -- it is the training contract of a DIFFERENT
#: model (``student_raw_global_raw_best.keras``, no mean subtraction, anchored
#: tail), carrying a decode that was never executed anywhere. Reinterpreting an old
#: ``apo_legacy`` choice as one of the notebook presets would silently change
#: results, so the alias maps it to the same numbers it always had.
PRESET_ALIASES: dict[str, str] = {"apo_legacy": "apo_raw_global"}

#: The molecules the presets cover, in UI order.
MOLECULES: tuple[str, ...] = (
    "ferritin", "apo-ferritin", "CB7 host-guest", "Zn2-insulin", "Cu2-insulin",
)

#: UI labels for the presets.
#: User-facing preset names. MOLECULE FIRST, deliberately: the StudentModel
#: notebooks were trained on **ferritin** (holo, iron core) while the apo presets
#: are for **apo-ferritin** (the empty shell) -- a different molecule, not merely
#: a different parameter set. Naming these "Notebook 1/2/3" hid that and invited
#: running a ferritin-trained model on apo data (or vice versa).
#:
#: The three ferritin presets are three different RECORDINGS, not three
#: revisions: nb3's pore gates between two levels ~296 pA apart with a ~35 ms
#: level dwell, which is why its baseline window is 30 ms rather than 500 ms and
#: why it alone enables onset snap / offset trim. Pick the one whose recording
#: behaves like yours.
#:
#: The three ``apo_nb_*`` presets share ONE preprocessing chain and differ only in
#: the notebook's hand-tuned decode threshold + minimum duration, so they are named
#: after the bias each was tuned at. Pick the one matching your recording's bias.
PRESET_LABELS: dict[str, str] = {
    "notebook1": "Ferritin — nb1 (steady baseline, ferritin_data1)",
    "notebook2": "Ferritin — nb2 (spiky, masked baseline, ferritin_data2)",
    "notebook3": "Ferritin — nb3 (gating pore, 30 ms baseline, ferritin_data3)",
    "apo_nb_500mV": "Apo-ferritin — −500 mV (notebook, thr 0.20 / min 15 ms)",
    "apo_nb_600mV": "Apo-ferritin — −600 mV (notebook, thr 0.10 / min 1 ms)",
    "apo_nb_700mV": "Apo-ferritin — −700 mV (notebook, thr 0.20 / min 1 ms)",
    "apo_raw_global": (
        "Apo-ferritin — raw-global model (student_raw_global_raw_best, W=3072)"
    ),
    # The "— DECODE UNCALIBRATED" suffix these three used to carry is GONE, and it
    # had to go: the decode is now the notebooks' own DETECTOR_CONTRACT (cell 1 of
    # each StudentInference notebook), not a neutral placeholder. Leaving a warning
    # up after the thing it warned about was fixed teaches the user to ignore the
    # next one. What the labels say instead is the one decode number that most
    # distinguishes these three from each other and from the presets above -- the
    # sustained-depth gate, which does nearly all of the selection work (on zn_30h:
    # 11004 fired regions become 1076 gated events).
    "hg_student": (
        "CB7 host-guest — student chain (6-line notch, masked median, robust σ), "
        "decode enter 0.40 / ≥4σ sustained"
    ),
    "zn_student": (
        "Zn2-insulin — student chain (9-line notch, masked median, robust σ), "
        "decode enter 0.40 / ≥4σ sustained"
    ),
    "cu_student": (
        "Cu2-insulin — student chain (9-line notch, SavGol baseline, robust σ), "
        "decode enter 0.50 / ≥4σ, ≥10 pA, ≥0.05 ΔI/I0"
    ),
}

#: Which molecule each preset was built for. Used to warn when the loaded model
#: and the selected preset disagree; NOT used to block the combination -- the
#: user knows their sample better than a heuristic does.
PRESET_MOLECULE: dict[str, str] = {
    "notebook1": "ferritin",
    "notebook2": "ferritin",
    "notebook3": "ferritin",
    "apo_nb_500mV": "apo-ferritin",
    "apo_nb_600mV": "apo-ferritin",
    "apo_nb_700mV": "apo-ferritin",
    "apo_raw_global": "apo-ferritin",
    "hg_student": "CB7 host-guest",
    "zn_student": "Zn2-insulin",
    "cu_student": "Cu2-insulin",
}

#: Default preset for the app when nothing else is known.
#:
#: -600 mV is the apo notebook's OWN reference bias -- its ``StandardScaler`` and
#: its UMAP are fit on -600 mV alone and the other two biases are transformed into
#: that space -- so it is the notebook's own choice of canonical recording, not
#: ours. It also supersedes ``apo_legacy`` as the default: an untouched panel now
#: runs the pipeline that was actually executed and validated, rather than a decode
#: scraped from function-signature defaults.
DEFAULT_PRESET: str = "apo_nb_600mV"


def resolve_preset_name(name: str | None) -> str | None:
    """Map a possibly-superseded preset key to its current one.

    Known names pass through; :data:`PRESET_ALIASES` entries are translated;
    anything unknown (or ``None``) comes back as ``None`` so the caller can decide
    -- never silently substituted for a preset with different numbers.
    """
    if not name:
        return None
    key = str(name)
    if key in PREPROCESS_PRESETS:
        return key
    return PRESET_ALIASES.get(key)


def preset_for_window(window: int | None) -> str | None:
    """The preset a model of input width ``window`` was most likely trained with.

    2048 is the three StudentModel (ferritin) notebooks' ``STUDENT_WINDOW``; 3072
    is both apo models. Anything else (or unknown) returns ``None`` so the caller
    keeps whatever it had.

    THE WIDTH CANNOT IDENTIFY A PIPELINE, only narrow it, and callers must say so
    rather than imply the guess is a match:

    * 2048 -> all three ferritin presets, AND an apo student exists at that width
      (``ApoStudentModelMerged``), so 2048 is not even proof of the molecule;
    * 3072 -> both apo lineages -- the notebook detector
      (``student_global_norm_bias_invariant_v0_5ish_best``, mean-centred) and the
      raw-global one (``student_raw_global_raw_best``, no mean subtraction,
      anchored tail). They need DIFFERENT preprocessing and they share a width.
    """
    if window is None:
        return None
    w = int(window)
    if w == 2048:
        return "notebook1"
    if w == 3072:
        return DEFAULT_PRESET
    return None


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def apply_ml_preprocess(
    sig: np.ndarray, fs: float, config: MLPreprocessConfig
) -> np.ndarray:
    """notch -> baseline removal -> despike, per ``config``. Returns float32.

    This is the DETRENDED signal (``current_detr_pA`` in the notebooks). It is NOT
    normalized: the normalization scale can come from a wider span than ``sig``
    (see :data:`NORMALIZE_SCOPES`), so the caller applies
    :func:`normalize_global` explicitly.
    """
    x = np.asarray(sig, dtype=np.float32).ravel()
    if x.size == 0:
        raise ValueError("signal must be a non-empty 1-D array")
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN/Inf; ML preprocessing is undefined")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive, finite sample rate, got {fs!r}")

    def require_finite(stage: str, values: np.ndarray) -> np.ndarray:
        if not np.isfinite(values).all():
            raise ValueError(
                f"{stage} produced NaN/Inf; ML preprocessing stopped before a later "
                "stage could hide the invalid samples"
            )
        return values

    if config.notch_enabled and len(config.notch_freqs):
        # filtfilt pads by ~3 * ntaps (9 for the 2nd-order iirnotch); below that the
        # whole contract is undefined, so fail with one clear message.
        if x.size < 10:
            raise ValueError(
                "recording is too short for the ML notch contract "
                f"(need >= 10 samples, got {x.size})"
            )
        out: np.ndarray = notch_sequential(
            x, fs, freqs=config.notch_freqs, q=config.notch_q
        )
        require_finite("notch", out)
        if config.notch_float32:
            # The apo notebook's apply_notch_filter_harmonics ends with
            # ``.astype(np.float32)``, so its SavGol sees -- and accumulates in --
            # float32. dataLoader.ipynb has no such cast. See MLPreprocessConfig.
            out = np.asarray(out, dtype=np.float32)
    else:
        out = x

    method = config.baseline_method
    if method == "robust_median":
        out, _ = baseline_robust_median(
            out, fs, window_s=config.baseline_window_s, decim=config.baseline_decim
        )
    elif method == "masked_median":
        out, _ = baseline_masked_median(
            out,
            fs,
            rough_s=config.masked_rough_s,
            base_s=config.masked_base_s,
            decim=config.baseline_decim,
            mask_sigma=config.masked_mask_sigma,
            dilate_ms=config.masked_dilate_ms,
        )
    elif method == "savgol":
        out, _ = baseline_savgol(
            out, fs, window_s=config.baseline_window_s, poly=config.baseline_poly
        )
    # "none": leave the notched signal alone.
    require_finite(f"{method} baseline stage", np.asarray(out))

    out = despike_median(out, size=config.despike_size)
    return np.asarray(require_finite("despike", np.asarray(out)), dtype=np.float32)


def normalization_center(sig: np.ndarray) -> float:
    """``mean(x)`` -- the offset the APO notebook's ``normalize_signal`` subtracts.

    Measured over whatever array you hand it; the CALLER decides the span (whole
    recording vs analysed range), exactly as for :func:`normalization_scale`.
    Empty in -> ``0.0`` out, i.e. a no-op centre.

    DTYPE MATTERS. ``np.mean`` on a float32 array accumulates in float32 (numpy's
    pairwise summation), which is what the notebook gets. Widening the accumulator
    with ``dtype=np.float64`` changes the low bits over a 12.5 M-sample recording
    and breaks bit-exactness, so this deliberately does NOT do that. The result is
    widened to a Python float only at the end, which is lossless.
    """
    x = np.asarray(sig, dtype=np.float32).ravel()
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _centered(x: np.ndarray, center: float) -> np.ndarray:
    """``x - center`` in float32, or ``x`` untouched when ``center`` is 0.

    ``center`` is cast to float32 first so the subtraction is float32-on-float32
    under both numpy 1.x value-based casting and numpy 2.x NEP-50 promotion --
    the notebook's ``sig - np.mean(sig)`` on a float32 array.
    """
    if not center:
        return x
    return np.asarray(x - np.float32(center), dtype=np.float32)


def normalization_scale(sig: np.ndarray, center: float = 0.0) -> float:
    """``max|x - center| + 1e-8`` -- the divisor the notebooks use.

    ``center=0.0`` (the default, and what ``student_predict`` does) makes this the
    plain ``max|x| + 1e-8``. The apo notebook subtracts the mean BEFORE taking the
    maximum, so pass :func:`normalization_center` of the same span.

    Measured over whatever array you hand it; the CALLER decides the span (whole
    recording vs analysed range). An empty array yields the bare ``1e-8`` so the
    division is still safe.
    """
    x = np.asarray(sig, dtype=np.float32).ravel()
    if x.size == 0:
        return 1e-8
    return float(np.max(np.abs(_centered(x, float(center)))) + 1e-8)


def normalize_global(
    sig: np.ndarray, scale: float | None = None, center: float = 0.0
) -> np.ndarray:
    """``(sig - center) / scale``, float32.

    ``scale=None`` measures it on the CENTRED array, i.e.
    ``normalization_scale(sig, center)``.

    With the default ``center=0.0`` this is a pure rescale with NO mean
    subtraction -- exactly what ferritin training did. The apo notebook passes a
    non-zero centre; see :func:`normalization_center`.
    """
    x = _centered(np.asarray(sig, dtype=np.float32).ravel(), float(center))
    s = float(scale) if scale is not None else normalization_scale(x)
    if s == 0.0:
        s = 1e-8
    return (x / s).astype(np.float32)


def model_input_normalization(
    detr: np.ndarray,
    config: MLPreprocessConfig,
    *,
    scale: float | None = None,
    center: float | None = None,
) -> np.ndarray:
    """The MODEL INPUT, from the detrended signal -- THE ONE PLACE THE TWO KERNELS ARE
    CHOSEN BETWEEN.

    ``config.normalize_method``:

    * ``"maxabs"`` -> :func:`normalize_global`, i.e. ``(x - center) / scale``. ``scale``
      and ``center`` are the caller's whole-recording statistics when the analysed span
      is a sub-range (that is what :data:`NORMALIZE_SCOPES` exists for); ``None`` for
      either measures it on ``detr`` itself. ``center`` is forced to 0.0 unless
      :attr:`MLPreprocessConfig.normalize_center` is set, exactly as before.
    * ``"robust_sigma"`` -> :func:`~nanoscope.analysis.filters.robust_normalize`.

    THE SCOPE ARGUMENTS DO NOT APPLY TO THE ROBUST BRANCH, and pretending otherwise is
    the trap this function exists to close. That kernel is not affine: it re-estimates a
    trimmed median and MAD and then CLIPS, so there is no ``(center, scale)`` pair that
    reproduces it and no way to "apply the recording's statistics to a slice". A caller
    that needs recording scope for it must hand this function the WHOLE recording's
    detrended array and slice AFTERWARDS -- which is what
    :meth:`~nanoscope.analysis.ml_modules.MLDetect.run` does. Passing ``scale``/
    ``center`` alongside ``"robust_sigma"`` is refused rather than silently ignored.

    Returns float32, same length as ``detr``.
    """
    if config.normalize_method == "robust_sigma":
        if scale is not None or center:
            raise ValueError(
                "robust_sigma normalization estimates its own centre and scale and "
                "cannot be given a precomputed pair; normalize the whole span and "
                "slice afterwards (see MLDetect.run)."
            )
        return robust_normalize(
            np.asarray(detr, dtype=np.float32).ravel(),
            clip_sigma=float(config.normalize_clip_sigma),
        )
    c = float(center) if (config.normalize_center and center is not None) else (
        normalization_center(detr) if config.normalize_center else 0.0
    )
    s = float(scale) if scale is not None else normalization_scale(detr, c)
    return normalize_global(detr, s, c)


# ---------------------------------------------------------------------------
# Decode (StudentModel*.ipynb postprocess_predictions)
# ---------------------------------------------------------------------------
def smooth_probability(x: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """``smooth_signal`` -- median filter, kernel forced odd, no-op when the trace
    is shorter than the kernel."""
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1 or len(x) < kernel_size:
        return np.asarray(x)
    return medfilt(np.asarray(x), kernel_size)


def extract_regions_hysteresis(
    prob: np.ndarray, enter: float = 0.5, exit_thr: float = 0.3
) -> list[tuple[int, int]]:
    """``extract_regions_hysteresis`` -- a run of ``prob >= exit_thr`` becomes a
    region only if it ever reaches ``>= enter``. Inclusive ``(start, end)``."""
    prob = np.asarray(prob)
    above_exit = (prob >= exit_thr).astype(np.int8)
    above_enter = prob >= enter
    diff = np.diff(above_exit, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    regions: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if above_enter[s:e].any():
            regions.append((int(s), int(e - 1)))
    return regions


def merge_events_within_radius(
    regions: Iterable[tuple[int, int]], fs: float, radius_ms: float = 30.0
) -> list[tuple[int, int]]:
    """``merge_events_within_radius`` -- join regions separated by <= ``radius_ms``."""
    regions = list(regions)
    if not regions:
        return []
    merge_samples = int(float(fs) * float(radius_ms) / 1000.0)
    merged: list[tuple[int, int]] = []
    cs, ce = regions[0]
    for s, e in regions[1:]:
        if s - ce <= merge_samples:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def filter_short_events(
    regions: Iterable[tuple[int, int]], fs: float, min_ms: float = 2.0
) -> list[tuple[int, int]]:
    """``filter_short_events`` -- drop regions shorter than ``min_ms`` (inclusive
    extent, so ``e - s + 1`` samples)."""
    min_samples = int(float(fs) * float(min_ms) / 1000.0)
    return [(s, e) for s, e in regions if (e - s + 1) >= min_samples]


def trim_offsets_to_deep(
    regions: Iterable[tuple[int, int]], signal: np.ndarray, deep_level: float
) -> list[tuple[int, int]]:
    """``_trim_offsets_to_deep`` -- snap each region's offset to its last sample
    below ``deep_level``, removing shallow recovery tails. Onset unchanged."""
    sig = np.asarray(signal)
    out: list[tuple[int, int]] = []
    for s, e in regions:
        below = np.where(sig[s : e + 1] < deep_level)[0]
        e_trim = (s + int(below[-1])) if len(below) else e
        out.append((s, e_trim))
    return out


def snap_onsets_to_deep(
    regions: Iterable[tuple[int, int]],
    signal: np.ndarray,
    onset_level: float,
    max_lookback: int,
    gap_tol: int = 5,
) -> list[tuple[int, int]]:
    """``_snap_onsets_to_deep`` -- pull each region's ONSET earlier to the start of
    the deep excursion.

    Walks backward from the detected start while the current stays below
    ``onset_level``, tolerating up to ``gap_tol`` consecutive shallow blips,
    bounded by ``max_lookback`` samples and by the PREVIOUS region's end; then
    walks forward again so the onset lands on a deep sample. Offset unchanged.
    Corrects onsets detected late when the model's probability rises after the
    current has already dropped.
    """
    sig = np.asarray(signal)
    out: list[tuple[int, int]] = []
    prev_end = -1
    for s, e in regions:
        j, steps, gap = s, 0, 0
        while j - 1 > prev_end and steps < max_lookback:
            if sig[j - 1] < onset_level:
                j -= 1
                steps += 1
                gap = 0
            elif gap < gap_tol:
                j -= 1
                steps += 1
                gap += 1
            else:
                break
        while j < s and sig[j] >= onset_level:   # land onset on a deep sample
            j += 1
        out.append((int(j), int(e)))
        prev_end = e
    return out


def decode_ml_regions(
    prob: np.ndarray,
    fs: float,
    config: MLDecodeConfig,
    *,
    signal: np.ndarray | None = None,
    gate_signal: np.ndarray | None = None,
    raw: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """``postprocess_predictions`` + cell 15's amplitude gates -> INCLUSIVE regions.

    ``signal`` is the DETRENDED current (the notebooks pass ``current_detr``); it
    is only consulted by the onset snap / offset trim, whose deep levels are
    ``median(signal) - sigma * 1.4826 * MAD(signal)``. Without it both are skipped
    even when their sigmas are non-zero.

    WHERE THE AMPLITUDE GATES GO, AND WHY THERE. The notebook is the authority and
    it is unambiguous: ``postprocess_predictions`` (cell 4) ends with
    ``filter_short_events`` and returns; cells 8 and 15 then take the RETURNED
    regions and gate them. So the gates run **after** the min-duration filter, and
    they measure the **final merged extent** -- ``med - median(sig[s:e+1])`` over
    exactly the region the duration test passed. Putting them earlier would gate
    pre-merge fragments, each shallower and shorter than the event they belong to.

    The gates need the SIGNAL, which a probability-only decode does not have:

    * ``gate_signal`` -- the DETRENDED trace **in pA**, which is what the depths
      are measured on. Defaults to ``signal``, correct for every preset whose
      model input is an affine rescale of the detrended trace. It must be passed
      separately for the ``robust_sigma`` presets (hg / zn / cu), where the
      adapter hands ``signal`` an already-normalised, CLIPPED model input in
      sigma units -- a pA threshold read against that trace would be meaningless.
    * ``raw`` -- the RAW, un-detrended current in pA, for the time-local ``I0``.
      Only ``min_fracblock`` needs it, and nothing else can supply it: detrending
      removes precisely the open-pore level that ``I0`` is.

    Raises:
        ValueError: if a gate is switched on but the trace it needs was not
            supplied. It refuses rather than approximating -- a gate that
            silently does not run changes which events exist, and looks exactly
            like a recording that had no shallow ones.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive, finite sample rate, got {fs!r}")
    prob = np.asarray(prob, dtype=np.float32).ravel()
    if prob.size == 0:
        return []

    smoothed = smooth_probability(prob, config.smoothing_kernel)
    regions = extract_regions_hysteresis(
        smoothed, config.enter_threshold, config.exit_threshold
    )
    regions = merge_events_within_radius(regions, fs, config.merge_radius_ms)

    if signal is not None and (config.onset_sigma or config.offset_sigma):
        sig = np.asarray(signal)
        med = float(np.median(sig))
        noise = 1.4826 * float(np.median(np.abs(sig - med)))
        if config.onset_sigma:
            regions = snap_onsets_to_deep(
                regions,
                sig,
                med - float(config.onset_sigma) * noise,
                max_lookback=int(float(config.onset_max_ms) * fs / 1000.0),
                gap_tol=int(config.onset_gap_tol),
            )
        if config.offset_sigma:
            regions = trim_offsets_to_deep(
                regions, sig, med - float(config.offset_sigma) * noise
            )

    regions = filter_short_events(regions, fs, config.min_event_ms)
    regions = gate_regions_by_amplitude(
        regions, fs, config, gate_signal=(gate_signal if gate_signal is not None
                                          else signal), raw=raw
    )
    return [(int(s), int(e)) for s, e in regions]


def gate_regions_by_amplitude(
    regions: Iterable[tuple[int, int]],
    fs: float,
    config: MLDecodeConfig,
    *,
    gate_signal: np.ndarray | None,
    raw: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Cell 15's three sustained-depth gates, applied to decoded regions.

    The measurement is NOT reimplemented here. It is
    :func:`nanoscope.analysis.event_features.amplitude_gate`, which is itself
    built on :func:`~nanoscope.analysis.event_features.sustained_gate` -- the same
    function the Zn2 feature port and the subpopulation analysis gate with. There
    is one definition of "how deep is this event" in this codebase and this is a
    caller of it, not a copy: a second copy is exactly how the notebooks' own
    thresholds drifted apart (cell 1's comment: the summary ran ``enter=0.5``
    while the comparison ran ``enter=0.4``).

    Regions are INCLUSIVE ``[start, end]``; ``amplitude_gate`` takes HALF-OPEN
    windows, so ``end + 1`` is passed. Off-by-one here silently shortens every
    event by one sample and, on the shortest ones, changes the median it gates on.
    """
    regions = [(int(s), int(e)) for s, e in regions]
    if not regions:
        return []
    on = (
        float(config.min_depth_sigma) > 0
        or float(config.min_depth_pa) > 0
        or float(config.min_fracblock) > 0
    )
    if not on:
        return regions
    if gate_signal is None:
        raise ValueError(
            "an amplitude gate is on ("
            f"min_depth_sigma={config.min_depth_sigma:g}, "
            f"min_depth_pa={config.min_depth_pa:g}, "
            f"min_fracblock={config.min_fracblock:g}) but no gate_signal was "
            "supplied. These gates measure a DEPTH, in pA, on the detrended "
            "current -- a probability trace alone cannot answer them. Pass "
            "gate_signal=<detrended pA>, or set the gates to 0.0."
        )

    from .event_features import amplitude_gate  # local: keeps import order flat

    windows = np.asarray([(s, e + 1) for s, e in regions], dtype=np.int64)
    result = amplitude_gate(
        gate_signal,
        windows,
        min_depth_sigma=float(config.min_depth_sigma),
        min_depth_pa=float(config.min_depth_pa),
        min_fracblock=float(config.min_fracblock),
        raw=raw,
        fs=fs,
    )
    return [r for r, keep in zip(regions, result.keep) if bool(keep)]
