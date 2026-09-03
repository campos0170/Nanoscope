"""Signal conditioning for nanopore current traces (signal-first kernels).

Everything here is zero-phase (`sosfiltfilt` / symmetric Gaussian kernel) so that
event edges do not move in time. That choice is not free -- see `lowpass`.

Two scipy defaults are silently wrong for this application and are overridden here:
`bessel(..., norm='phase')` and `output='ba'`. Both are documented at their use site.

Ported from ``nanoscope.core.filters`` (the standalone package) and re-seamed onto
the target neo-based data model: every entry point now takes ``(data, fs, ...)``
numpy arrays rather than a ``Recording``. The provenance meta the old Recording
carried (``filter_kind``, ``filter_cutoff_hz``, ``filter_order``,
``filter_zero_phase``, ``filter_rise_s``) is stamped by the *caller*, not here --
concretely by :func:`nanoscope.analysis.detect.detect_events`, which owns the
low-pass and records all five.

`response` is the read-only companion to these kernels: it returns |H(f)| for a
stage AS APPLIED (squared for the zero-phase IIR paths), so a caller can show a
user what a parameter choice really does instead of asserting it in a label.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Callable

import numpy as np
from scipy import ndimage, signal

__all__ = [
    "lowpass", "highpass", "measure_rise_time", "notch", "savgol",
    "notch_settle_s", "notch_unsettled_s", "notch_frequencies",
    "running_mean", "median_filter", "noise_rms", "psd",
    "normalize", "robust_normalize",
    "response", "response_db", "response_crossing_hz", "rise_time_probe_samples",
    "savgol_window_samples", "running_mean_samples",
    "RESPONSE_STAGES", "KINDS", "HP_KINDS", "KINDS_WITHOUT_ORDER",
    "NORMALIZE_MODES", "MODEL_INPUT_MODE", "NORMALIZE_MODES_ALL",
    "MODEL_INPUT_CLIP_SIGMA",
]

_KINDS = ("bessel", "butter", "gaussian")

# The IIR kinds a high-pass can be built from. 'gaussian' is excluded: it is a
# low-pass kernel (a smoothing envelope), with no high-pass counterpart here.
_HP_KINDS = ("butter", "bessel")

# Public aliases: the GUI builds its type combos from these, so the list of kinds
# the user can pick can never drift from the list the kernels accept.
KINDS = _KINDS
HP_KINDS = _HP_KINDS

# 'gaussian' is a symmetric convolution kernel, not an IIR cascade -- it has no
# order. The GUI disables its order control by consulting this.
KINDS_WITHOUT_ORDER = ("gaussian",)

# The rescalings `normalize` can apply, in the order the GUI lists them. Public so
# the panel's mode combo can never drift from the set the kernel accepts.
#
# EVERY ONE OF THESE DESTROYS THE pA UNIT. That is not a caveat, it is the point of
# the stage, and it is why the name is exported next to the modes: a caller that
# offers these to a user owes that user a unit label that stops saying "pA".
NORMALIZE_MODES = ("zscore", "robust", "minmax")

#: The FOURTH rescale, kept out of `NORMALIZE_MODES` because it is not one of them.
#: `normalize` is a single affine map ``(x - c) / s`` with a `center` switch;
#: :func:`robust_normalize` is a three-stage trimmed re-estimation followed by a CLIP
#: (so it is not affine, `center` is meaningless for it, and its output is bounded to
#: [-1, 1] while the other three are unbounded) and it FALLS BACK on a degenerate
#: scale where `normalize` deliberately raises. Two opposite contracts cannot live
#: behind one ``mode=`` switch, so the caller dispatches on this name instead --
#: see :func:`robust_normalize` for the dispatch a panel/module owes it.
MODEL_INPUT_MODE = "model_input"

#: Every rescale a caller may offer, `MODEL_INPUT_MODE` included. The GUI combo is
#: built from THIS, so the list a user can pick from cannot drift from the set the
#: two kernels between them accept; the dispatch is one ``if`` on `MODEL_INPUT_MODE`.
NORMALIZE_MODES_ALL = (*NORMALIZE_MODES, MODEL_INPUT_MODE)

#: :func:`robust_normalize`'s clip, in robust sigmas. 30 is the notebooks' own value
#: in all six of them; it is BOTH the clip and the divisor, so the noise floor lands
#: at 1/30 = 0.033 and the model input stays inside [-1, 1].
MODEL_INPUT_CLIP_SIGMA = 30.0

# Amplitude ratios defining a 10-90% transition.
_LO, _HI = 0.1, 0.9

# |H| at the -3 dB point.
_HALF_POWER = 1.0 / np.sqrt(2.0)


def _validate_cutoff(fs: float, cutoff_hz: float) -> None:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive and finite, got {cutoff_hz}")
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz={cutoff_hz:g} Hz is at or above Nyquist ({nyq:g} Hz) for "
            f"fs={fs:g} Hz. There is nothing to filter; this is a data or units error."
        )
    if cutoff_hz > fs / 2.5:
        warnings.warn(
            f"cutoff_hz={cutoff_hz:g} Hz is above fs/2.5 ({fs / 2.5:g} Hz): the transition "
            f"band is truncated by Nyquist, so the filter barely attenuates anything and the "
            f"nominal order is meaningless. Sample faster or ask for a lower cutoff.",
            RuntimeWarning,
            stacklevel=3,
        )


def _design_sos(
    fs: float, cutoff_hz: float, order: int, kind: str, *, btype: str = "low"
) -> np.ndarray:
    """Second-order-sections for the IIR kinds. Not for kind='gaussian'.

    ``btype`` selects low- vs high-pass; it defaults to 'low' so every existing
    caller (and the direct-import tests) is unchanged, and `highpass` passes 'high'.

    norm='mag' is mandatory: scipy's default for Bessel is norm='phase', which
    normalises the group delay instead and leaves the -3 dB point somewhere other
    than `cutoff_hz` (increasingly far off as order rises). It is correct for both
    band types -- it pins |H| = 1/sqrt(2) at the cutoff either way. output='sos' is
    mandatory: the 'ba' polynomial form loses accuracy catastrophically at order 8
    with fs/cutoff ratios of 10-100, which is the normal operating point here.
    """
    if kind == "bessel":
        return signal.bessel(
            order, cutoff_hz, btype=btype, fs=fs, output="sos", norm="mag"
        )
    if kind == "butter":
        return signal.butter(order, cutoff_hz, btype=btype, fs=fs, output="sos")
    raise ValueError(f"_design_sos does not handle kind={kind!r}")


def _gaussian_sigma_samples(fs: float, cutoff_hz: float) -> float:
    """Kernel sigma (in samples) whose -3 dB amplitude point sits at cutoff_hz.

    H(f) = exp(-2 pi^2 sigma_t^2 f^2); |H(fc)| = 1/sqrt(2) => sigma_t = sqrt(ln 2)/(2 pi fc).
    """
    sigma_t = np.sqrt(np.log(2.0)) / (2.0 * np.pi * cutoff_hz)
    return float(sigma_t * fs)


def _filter_fn(
    fs: float, cutoff_hz: float, order: int, kind: str
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the single callable that `lowpass` and `measure_rise_time` both use.

    They must never diverge: the rise time is only meaningful if it was measured
    through the exact filter that touched the data.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    if kind != "gaussian":
        if int(order) != order or order < 1:
            raise ValueError(f"order must be a positive integer, got {order!r}")
        sos = _design_sos(fs, cutoff_hz, int(order), kind)

        def apply(x: np.ndarray) -> np.ndarray:
            return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))

        return apply

    sigma = _gaussian_sigma_samples(fs, cutoff_hz)

    def apply_gauss(x: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter1d(
            np.asarray(x, dtype=np.float64), sigma, mode="nearest"
        )

    return apply_gauss


def rise_time_probe_samples(fs: float, cutoff_hz: float) -> int:
    """Half-length (in samples) of the step `measure_rise_time` pushes through the filter.

    Exposed because the cost of that measurement is ``O(fs / cutoff_hz)`` and explodes at
    a low cutoff -- at fs=250 kHz a 1 Hz cutoff builds a 20 M-sample step (hundreds of MB)
    for a readout nobody asked to wait for. A live GUI readout consults this to decide
    whether the measurement is affordable *before* starting it, using the same formula the
    measurement itself uses so the two can never disagree.
    """
    return int(max(np.ceil(40.0 * fs / cutoff_hz), 512))


def savgol_window_samples(fs: float, window_s: float, polyorder: int) -> int:
    """`window_s` -> the ODD sample count `savgol` fits its polynomial over.

    Shared by :func:`savgol` (which then also clamps to the trace length) and
    :func:`response`, so the drawn response is the response of the window actually used
    -- including the clamp to ``polyorder + 2`` that silently widens a too-short window.
    """
    win = int(round(window_s * fs))
    win = max(win, int(polyorder) + 2)
    if win % 2 == 0:
        win += 1
    return win


def running_mean_samples(fs: float, window_s: float) -> int:
    """`window_s` -> the box length `running_mean` averages over (``1`` == identity).

    Shared with :func:`response` so the drawn curve shows the flat line that a window
    rounding to a single sample really is, rather than an idealised box.
    """
    return max(1, int(round(window_s * fs)))


def lowpass(
    data: np.ndarray, fs: float, cutoff_hz: float, *, order: int = 8, kind: str = "bessel"
) -> np.ndarray:
    """Zero-phase low-pass a current trace. Cutoff in Hz; units are carried unchanged.

    Returns a contiguous ``float32`` array the same length as ``data``.

    Bessel (the electrophysiology convention) has a maximally flat group delay, so
    it preserves pulse shape and does not ring on a step -- which matters when the
    quantity of interest *is* the shape of a sub-millisecond pulse. 'butter' gives a
    sharper roll-off at the cost of overshoot; 'gaussian' has no overshoot at all.

    Gotcha 1 -- filtfilt is acausal. Running the filter forwards and backwards means
    no time shift (edges stay put), but the price is that energy from an edge is
    smeared *symmetrically about it*: a blockade appears to begin slightly before it
    physically did. The onset is therefore biased early and the offset late, so
    dwell times of short events are biased LONG.

    Gotcha 2 -- filtfilt DOUBLES the effective order and squares the magnitude
    response. The two-pass response is -6 dB at `cutoff_hz`, not -3 dB, and the true
    10-90% rise time is far from the textbook 0.34/fc. Never assume it; call
    `measure_rise_time`, and stamp its result into the caller's provenance.

    Raises if cutoff_hz >= fs/2; warns above fs/2.5.
    """
    _validate_cutoff(fs, cutoff_hz)
    fn = _filter_fn(fs, cutoff_hz, order, kind)
    y = fn(data)
    return np.ascontiguousarray(y, dtype=np.float32)


def measure_rise_time(
    fs: float, cutoff_hz: float, *, order: int = 8, kind: str = "bessel"
) -> float:
    """10-90% rise time in **seconds**, measured by pushing a unit step through
    the exact callable `lowpass` uses.

    This number defines the resolution floor of the whole analysis: events shorter
    than ~2x this never reach their true depth (depth is attenuated toward zero) and
    their measured dwell is biased long. Every "we resolved a 50 us event" claim
    rests on it, so it is measured, never derived.

    NOTE WHICH DURATION THE FLOOR IS APPLIED TO. Because the same filter biases the
    measured dwell LONG, ``characterize`` does not test ``dwell_s`` against ``2 *
    rise``: it tests the width at half the event's own peak, which the smear leaves
    alone. Testing the measured dwell let the filter carry an event over the line
    exactly when it was destroying it -- 200 us events through a 2 kHz low-pass
    measured 545 us, cleared the 481 us boundary, and shipped 36.5 pA of a 100 pA
    depth unflagged.

    Do NOT substitute the textbook t_r ~= 0.34/fc. That is a single-pass, low-order
    approximation; `lowpass` uses zero-phase filtfilt, which doubles the effective
    order and moves the true rise time tens of percent away from it. The error is in
    the direction that flatters you.
    """
    _validate_cutoff(fs, cutoff_hz)
    fn = _filter_fn(fs, cutoff_hz, order, kind)

    # Long enough that both the pre-step baseline and the post-step plateau fully
    # settle, and that sosfiltfilt's edge padding never reaches the transition.
    n_side = rise_time_probe_samples(fs, cutoff_hz)
    step = np.concatenate([np.zeros(n_side), np.ones(n_side)])
    y = fn(step)

    lo_level = float(np.median(y[: n_side // 2]))
    hi_level = float(np.median(y[-n_side // 2 :]))
    amp = hi_level - lo_level
    if not np.isfinite(amp) or amp <= 0:
        raise RuntimeError(
            f"step response did not settle for kind={kind!r} order={order} "
            f"cutoff={cutoff_hz:g} Hz fs={fs:g} Hz; cannot measure rise time"
        )
    yn = (y - lo_level) / amp

    def _cross(level: float) -> float:
        """First upward crossing of `level`, in fractional samples."""
        above = np.flatnonzero(yn >= level)
        if above.size == 0:
            raise RuntimeError(f"step response never reached {level:g} of full scale")
        i = int(above[0])
        if i == 0:
            return 0.0
        y0, y1 = yn[i - 1], yn[i]
        return (i - 1) + (level - y0) / (y1 - y0)

    t10 = _cross(_LO)
    t90 = _cross(_HI)
    return float((t90 - t10) / fs)


def highpass(
    data: np.ndarray, fs: float, cutoff_hz: float, *, order: int = 4, kind: str = "butter"
) -> np.ndarray:
    """Zero-phase high-pass a current trace. Cutoff in Hz; units are carried unchanged.

    Returns a contiguous ``float32`` array the same length as ``data``. Butterworth by
    default (a maximally flat pass-band, so the open-pore level it leaves is not
    rippled); ``kind='bessel'`` is available for a maximally flat group delay. Both run
    zero-phase (``sosfiltfilt``), so an edge does NOT shift in time.

    USE WITH CARE -- A HIGH-PASS DISTORTS SQUARE TRANSLOCATION EVENTS. A blockade is (to
    first approximation) a rectangular pulse, and its flat top is exactly the low-
    frequency content a high-pass removes: the plateau DROOPS back toward the baseline
    during the event, and the trace goes BIPHASIC (an overshoot of opposite sign trails
    each edge). Depth read off a drooped plateau comes out shallow, and the overshoot can
    manufacture a phantom opposite-polarity excursion. This filter is for removing a DC
    offset or slow baseline drift, NOT for conditioning the events themselves -- to
    flatten drift while keeping event shape prefer `remove_baseline_wander` (a slow
    open-pore envelope subtraction), and keep any high-pass cutoff far below the event
    bandwidth. Zero-phase at least means the edges you do keep stay put.

    Raises if cutoff_hz is not in (0, fs/2); warns above fs/2.5.
    """
    if kind not in _HP_KINDS:
        raise ValueError(f"kind must be one of {_HP_KINDS}, got {kind!r}")
    if int(order) != order or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")
    _validate_cutoff(fs, cutoff_hz)
    sos = _design_sos(fs, cutoff_hz, int(order), kind, btype="high")
    y = signal.sosfiltfilt(sos, np.asarray(data, dtype=np.float64))
    return np.ascontiguousarray(y, dtype=np.float32)


#: How far the notch's own startup transient must have decayed before the filtered
#: samples are trustworthy: 1e-4 (-80 dB) of the hum the notch removed. The number is
#: chosen against the thing the transient competes with -- 1e-4 of a full 1 nA span is
#: 0.1 pA, an order of magnitude under any nanopore noise floor -- so the artefact can
#: never reach a detection threshold. See :func:`notch_settle_s` for why it exists.
_NOTCH_SETTLE_DECAY = 1e-4


def _notch_lines(
    fs: float,
    freq_hz: float,
    harmonics: int,
    freqs: Sequence[float] | None,
    *,
    warn: bool,
    stacklevel: int = 4,
) -> list[float]:
    """The notch lines to apply at ``fs``: either a harmonic series or an explicit list.

    ONE resolver for all four notch entry points, so the lines :func:`notch` filters
    with, the poles :func:`notch_settle_s` sizes the pad from, and the list
    :func:`notch_frequencies` reports to a user can never be three different sets.

    ``freqs=None`` is the original parameterisation -- a fundamental and a COUNT of
    CONTIGUOUS harmonics, ``[freq_hz * h for h in 1..harmonics]``. That form cannot
    express a line set with gaps, and the gaps are real: the owner's Welch census on
    the CB7/adamantyl recordings found peaks at 60, 120, 180, 300, 420 and 540 Hz and
    NOT at 240, 360 or 480, so the nearest contiguous series (``harmonics=9``) would
    notch three clean bands for nothing. Passing ``freqs`` expresses that set exactly.

    ``freqs`` WINS OUTRIGHT when it is given: ``freq_hz`` and ``harmonics`` are then
    not read at all, not even to seed the list. A caller that has both must decide
    which one it means -- silently merging them is how a stale fundamental left in a
    params dict starts adding a line nobody asked for.

    The set, not the order, decides the result: the biquads are stacked into one SOS
    cascade and a cascade is commutative to within the float error measured in
    ``tests/analysis/test_notch_explicit_lines.py`` (max 1.5e-3 pA over a 6-line Q=30
    set at 50 kHz). The caller's order is nevertheless preserved for REPORTING, so
    :func:`notch_frequencies` echoes the list the user typed.

    Duplicates are refused rather than de-duplicated: two identical biquads are a
    deeper, longer-ringing notch than the one line the user believes they asked for,
    and silently dropping one would equally hide a typo in a nine-line list.
    """
    if freqs is None:
        if not np.isfinite(freq_hz) or freq_hz <= 0:
            raise ValueError(f"freq_hz must be positive and finite, got {freq_hz}")
        if int(harmonics) != harmonics or harmonics < 1:
            raise ValueError(f"harmonics must be >= 1, got {harmonics}")
        wanted = [float(freq_hz) * h for h in range(1, int(harmonics) + 1)]
        asked = f"freq_hz={freq_hz:g} with {int(harmonics)} harmonic(s)"
    else:
        wanted = [float(f) for f in freqs]
        if not wanted:
            raise ValueError(
                "freqs is empty: an explicit notch line list must name at least one "
                "frequency. Pass freqs=None to use the freq_hz/harmonics series, or "
                "leave the notch stage disabled."
            )
        bad = [f for f in wanted if not np.isfinite(f) or f <= 0]
        if bad:
            raise ValueError(f"every notch frequency must be positive and finite, got {bad}")
        dupes = sorted({f for f in wanted if wanted.count(f) > 1})
        if dupes:
            raise ValueError(
                f"duplicate notch frequencies {dupes} Hz: the same line twice is a "
                "deeper, longer-ringing notch than the single line it looks like. "
                "List each frequency once."
            )
        asked = f"freqs={[float(f) for f in wanted]}"

    nyq = fs / 2.0
    keep = [f for f in wanted if f < nyq]
    if not keep:
        raise ValueError(
            f"no notch frequency below Nyquist ({nyq:g} Hz): {asked} is "
            f"unusable at fs={fs:g} Hz"
        )
    if warn and len(keep) < len(wanted):
        dropped = [f for f in wanted if f >= nyq]
        warnings.warn(
            f"skipping {len(dropped)} notch line(s) at or above Nyquist "
            f"({nyq:g} Hz) for fs={fs:g} Hz: {dropped}",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return keep


def _notch_sos(
    fs: float,
    freq_hz: float,
    q: float,
    harmonics: int,
    *,
    warn: bool,
    freqs: Sequence[float] | None = None,
    stacklevel: int = 4,
) -> np.ndarray:
    """The SOS cascade :func:`notch` filters with -- validation and Nyquist rule included.

    Shared with :func:`notch_settle_s` so the settling time is measured from the SAME
    poles that touch the data. A second copy of this design is how a "settled" claim
    starts describing a filter that was never applied.

    ALL THE LINES GO INTO ONE SOS ARRAY AND THE TRACE IS FILTERED ONCE. The reference
    notebooks instead call ``scipy.filtfilt`` once PER LINE, sequentially, ascending
    (``ml_preprocess.apply_notch_filters`` is that form, kept verbatim for the ML
    chain). THAT TOPOLOGY DIFFERENCE IS NOT MATERIAL AND IT WAS MEASURED, not assumed:
    on a 10 s / 50 kHz synthetic carrying the owner's own six-line set
    (60/120/180/300/420/540 Hz, Q=30) at 3-30 pA per line over 5 pA noise, stacked
    ``sosfiltfilt`` and six sequential ``filtfilt`` passes given the SAME pad differ by
    at most **8.8e-5 pA** peak (RMS 5.7e-6 pA) -- roughly ten float32 quanta at the
    100 pA scale, five orders of magnitude under the noise floor. Repeated at 1 s / 2 s
    durations and Q=10/30/100 the worst case is 1.5e-3 pA.

    WHAT DOES DIFFER BY 40 pA IS THE PAD, WHICH IS A SEPARATE AXIS. Run at scipy's
    DEFAULT ``filtfilt`` padlen (9 samples for a biquad) the same trace departs from
    the settled result by 20.9 pA over the first 0.1 s and 40.0 pA over the last 0.1 s,
    decaying to 5e-5 pA only after ~2 s at each end -- the notch's own ring-down
    (:func:`notch_settle_s`), six passes of it. So a caller chasing "the notebook's
    numbers" is chasing the notebook's PADDING, not its topology, and this module
    deliberately does not reproduce it: see :func:`notch`.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    keep = _notch_lines(
        fs, freq_hz, harmonics, freqs, warn=warn, stacklevel=stacklevel
    )
    return np.vstack([np.hstack(signal.iirnotch(f, q, fs=fs))[None, :] for f in keep])


def notch_settle_s(
    fs: float,
    freq_hz: float = 60.0,
    *,
    q: float = 30.0,
    harmonics: int = 3,
    freqs: Sequence[float] | None = None,
) -> float:
    """Seconds the notch needs to forget a wrong initial state, from its OWN poles.

    ``sosfiltfilt`` starts each pass assuming the signal was CONSTANT at the first
    sample for all prior time. For hum that assumption is wrong by the full hum
    amplitude, and a high-Q notch forgets a wrong state only as fast as its slowest
    pole decays: at Q=30 and 60 Hz the notch is 2 Hz wide and its ring-down time
    constant is 1/(pi*2 Hz) = 159 ms. Filtered with scipy's DEFAULT pad (21 samples
    for a 3-line cascade -- 0.1 ms at 200 kHz), the result therefore carries a
    one-sided head transient whose amplitude scales linearly with the hum removed:
    measured on a pure 300 pA / 50 Hz line at 100 kHz, max |residual| over the first
    0.1 s was 142.9 pA against 0.0003 pA at the tail. Detection reported that ringing
    as dozens of ordinary accepted events.

    This is the length of pad that makes the transient decay by
    :data:`_NOTCH_SETTLE_DECAY` before the real samples begin, so the artefact is
    gone rather than merely undocumented. Same trace, pad sized from here: max
    |residual| over the first 0.1 s = 0.015 pA.

    Measured, not assumed, in the same spirit as :func:`measure_rise_time`: the pole
    radius is read off the cascade :func:`notch` actually designs.

    ``freqs`` takes an EXPLICIT line list and then overrides ``freq_hz``/``harmonics``
    entirely (see :func:`_notch_lines`). It must be the same list the matching
    :func:`notch` call gets: the pad is only "the filter's own ring-down" if it is
    sized from the filter that ran.
    """
    sos = _notch_sos(
        fs, float(freq_hz), float(q), int(harmonics), warn=False, freqs=freqs
    )
    # Slowest section wins: the LOWEST line's notch is narrower in relative terms
    # than the ones above it, so it rings longest.
    radius = max(float(np.abs(np.roots(section[3:])).max()) for section in sos)
    if not (0.0 < radius < 1.0):  # pragma: no cover - an unstable/degenerate design
        raise RuntimeError(
            f"notch poles at radius {radius:g} do not decay; cannot size the settling "
            f"time for "
            f"{f'freqs={list(freqs)}' if freqs is not None else f'freq_hz={freq_hz:g}'} "
            f"q={q:g} fs={fs:g}"
        )
    tau_samples = -1.0 / np.log(radius)
    return float(tau_samples * np.log(1.0 / _NOTCH_SETTLE_DECAY) / float(fs))


def notch_unsettled_s(
    fs: float,
    n_samples: int,
    freq_hz: float = 60.0,
    *,
    q: float = 30.0,
    harmonics: int = 3,
    freqs: Sequence[float] | None = None,
) -> float:
    """Head duration a notch of this design CANNOT settle inside a trace this short.

    ``sosfiltfilt`` cannot pad with more samples than the trace holds, so a recording
    shorter than :func:`notch_settle_s` (1.5-1.8 s at the app's defaults) gets the
    longest pad that fits and starts with whatever transient is left. This returns how
    much of the head is still above the :data:`_NOTCH_SETTLE_DECAY` target -- ``0.0``
    when the pad fits, which is the ordinary case for a real recording.

    It exists so a CALLER can say so on its own surface: :class:`DetectSchmitt
    <nanoscope.analysis.modules.DetectSchmitt>` stamps it and puts it in
    ``res.warnings``, because a settling transient that lands in an events table is
    otherwise indistinguishable from a burst of real translocations.

    ``freqs`` is the explicit line list, overriding ``freq_hz``/``harmonics`` exactly
    as it does everywhere else; hand it the list the run actually used or this reports
    a settling time for a filter nobody applied.
    """
    need = notch_settle_s(fs, freq_hz, q=q, harmonics=harmonics, freqs=freqs)
    have = max(0, int(n_samples) - 1) / float(fs)
    return float(max(0.0, need - have))


def notch(
    data: np.ndarray,
    fs: float,
    freq_hz: float = 60.0,
    *,
    q: float = 30.0,
    harmonics: int = 3,
    freqs: Sequence[float] | None = None,
) -> np.ndarray:
    """Zero-phase notch out mains hum at `freq_hz` and its first `harmonics` multiples.

    Returns a contiguous ``float32`` array the same length as ``data``.

    Q=30 at 60 Hz is a 2 Hz-wide notch. Whether this is free depends on the event
    timescale, so check rather than assume: a ~100 us event has essentially no
    spectral content below ~1 kHz, so removing a 2 Hz sliver at 60 Hz leaves its
    shape untouched. That reasoning FAILS for slow events -- if your dwell times
    approach ~10 ms (i.e. ~100 Hz), the notch sits inside the event's own band and
    will distort depth and shape. Notch is for baseline hygiene, not for events that
    are themselves slow.

    Harmonics at or above Nyquist are skipped (a notch there is undefined).

    ``freqs`` REPLACES the (freq_hz, harmonics) series with an EXPLICIT list of lines,
    which is the only way to express a set with GAPS. The owner's CB7/adamantyl
    recordings need exactly that: their Welch census found 60, 120, 180, 300, 420 and
    540 Hz and nothing at 240, 360 or 480, so ``harmonics=9`` would notch three quiet
    bands to reach 540. When ``freqs`` is given, ``freq_hz`` and ``harmonics`` are
    ignored outright -- see :func:`_notch_lines` for why they are not merged.

    ONE STACKED SOS CASCADE, FILTERED ONCE -- not one ``filtfilt`` per line as the
    owner's notebooks do. Measured on a synthetic carrying that exact six-line set at
    Q=30 / 50 kHz, the two topologies given the same pad differ by at most 8.8e-5 pA
    (see :func:`_notch_sos`), so the option is not offered: it would be a second code
    path that provably changes nothing. What DOES differ is the pad, and this function
    keeps its own.

    THE PAD IS SIZED FROM THE NOTCH'S OWN RING-DOWN (:func:`notch_settle_s`), never
    left at scipy's default. The default pad is 21 samples for a 3-line cascade --
    0.1 ms at 200 kHz against a ~160 ms ring-down -- which left a head transient of
    up to 48% of the hum removed sitting in the conditioned trace, where detection
    read it as accepted events. A trace too short to hold that pad gets the longest
    pad that fits AND a warning naming the head that is still unsettled: see
    :func:`notch_unsettled_s`, which is the same number in a form a caller can put on
    a user surface.
    """
    # stacklevel: the warning is raised two frames down (_notch_sos -> _notch_lines),
    # so 4 lands on THIS function's caller -- the same frame the pre-change code
    # pointed at from one frame up.
    sos = _notch_sos(
        fs, float(freq_hz), float(q), int(harmonics), warn=True, freqs=freqs,
        stacklevel=4,
    )
    x = np.asarray(data, dtype=np.float64)

    # The pad is data (an odd extension), so it can never be longer than the trace.
    need = notch_settle_s(
        fs, float(freq_hz), q=float(q), harmonics=int(harmonics), freqs=freqs
    )
    need_samples = int(np.ceil(need * float(fs)))
    padlen = min(need_samples, max(0, x.size - 1))
    if padlen < need_samples:
        lines = notch_frequencies(fs, freq_hz, harmonics=harmonics, freqs=freqs)
        warnings.warn(
            f"trace is {x.size / float(fs):.4g} s, shorter than the "
            f"{need:.4g} s this notch ({len(lines)} line(s) at {lines} Hz, "
            f"q={q:g}) needs to settle; "
            f"the first {need - padlen / float(fs):.4g} s of the result still carry the "
            "filter's own startup transient, which scales with the hum removed",
            RuntimeWarning,
            stacklevel=2,
        )
    y = signal.sosfiltfilt(sos, x, padlen=padlen)
    return np.ascontiguousarray(y, dtype=np.float32)


def notch_frequencies(
    fs: float,
    freq_hz: float = 60.0,
    *,
    harmonics: int = 3,
    freqs: Sequence[float] | None = None,
) -> list[float]:
    """The lines `notch` will actually apply at `fs` (those strictly below Nyquist).

    Exposed so a caller (or a test) can record which lines were notched without
    re-deriving the Nyquist rule; `notch` itself returns only the filtered signal.

    ``freqs`` is the explicit line list; when given it overrides ``freq_hz`` /
    ``harmonics``, and the caller's ORDER is preserved so a provenance stamp echoes
    the list the user typed rather than a re-sorted version of it.

    Answers from the SAME resolver `notch` designs its cascade from, so the lines a
    user is told about and the lines that touched the data cannot be different sets.
    It does not warn (a report is not a run); `notch` warns for both of them.
    """
    return _notch_lines(fs, freq_hz, harmonics, freqs, warn=False)


def savgol(
    data: np.ndarray, fs: float, *, window_s: float, polyorder: int = 3
) -> np.ndarray:
    """Savitzky-Golay smooth: fit a low-order polynomial in a sliding window.

    Returns a contiguous ``float32`` array the same length as ``data``. Unlike a moving
    average -- a box low-pass that rounds off peaks and smears a step -- a Savitzky-Golay
    filter fits a degree-``polyorder`` polynomial to each window and takes its value at
    the centre, so it suppresses noise while PRESERVING the height and curvature of a
    peak far better: it keeps an event's depth and edges much more faithfully than an
    equivalent-width box. The window is symmetric, so it does not shift features in time.

    ``window_s`` is converted to a sample count, clamped to at least ``polyorder + 2``
    points (a window can host a degree-p polynomial only if it has more than p samples)
    and forced ODD so the window is centred. Keep the window shorter than the shortest
    event you care about, or the smoother rounds the event off along with the noise.

    Raises ValueError on a non-positive ``window_s`` / ``polyorder``, a non-positive
    rate, or a trace too short to host even the minimum window.
    """
    x = np.asarray(data, dtype=np.float64)
    if int(polyorder) != polyorder or polyorder < 1:
        raise ValueError(f"polyorder must be a positive integer, got {polyorder!r}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"window_s must be positive and finite, got {window_s}")
    polyorder = int(polyorder)

    # window_s -> samples, with more points than the polynomial has coefficients, and
    # ODD (a Savitzky-Golay window is centred on the sample it estimates).
    win = savgol_window_samples(fs, window_s, polyorder)
    # The window cannot exceed the trace (scipy's mode='interp'); shrink to the largest
    # odd length that fits. Only on a trace this short can polyorder < win then fail.
    if win > x.size:
        win = x.size if x.size % 2 == 1 else x.size - 1
    if polyorder >= win:
        raise ValueError(
            f"polyorder={polyorder} must be < window length, but the trace is only "
            f"{x.size} sample(s) long (window clamped to {win})."
        )
    y = signal.savgol_filter(x, win, polyorder)
    return np.ascontiguousarray(y, dtype=np.float32)


def running_mean(data: np.ndarray, fs: float, *, window_s: float) -> np.ndarray:
    """Centred (zero-phase) moving average over a ``window_s`` box.

    Returns a contiguous ``float32`` array the same length as ``data``. This is the plain
    box low-pass Savitzky-Golay is contrasted against: it averages ``N = round(window_s *
    fs)`` samples with a uniform kernel, applied CENTRED so features do not shift in time
    (unlike a trailing/causal moving average, which delays every edge by half a window).
    Edges are handled by nearest-sample extension (``mode='nearest'``), so the ends are not
    biased toward zero.

    It suppresses white noise by ~``sqrt(N)`` for the cost of rounding off peaks and
    smearing steps -- keep the window shorter than the shortest event you care about, or the
    average flattens the event along with the noise. When the height/curvature of a peak must
    survive, prefer :func:`savgol`; use this when a fast, unbiased noise floor is all you need.

    ``window_s`` is converted to a sample count ``N = max(1, round(window_s * fs))``. ``N ==
    1`` is a passthrough (a one-sample average is the identity). Raises ValueError on a
    non-positive / non-finite ``window_s`` or a non-positive rate.
    """
    x = np.asarray(data, dtype=np.float64)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"window_s must be positive and finite, got {window_s}")

    # window_s -> samples. N == 1 is the identity (a one-sample box averages nothing).
    n = running_mean_samples(fs, window_s)
    if n == 1:
        return np.ascontiguousarray(x, dtype=np.float32)

    # uniform_filter1d centres the box (its origin defaults to 0), so no time shift; the
    # nearest-sample edge mode keeps the ends from being dragged toward zero.
    y = ndimage.uniform_filter1d(x, n, mode="nearest")
    return np.ascontiguousarray(y, dtype=np.float32)


def normalize(
    data: np.ndarray, *, mode: str = "zscore", center: bool = True
) -> np.ndarray:
    """Rescale a whole trace onto a dimensionless scale (z-score / robust / min-max).

    Returns a contiguous ``float32`` array the same length as ``data``. Unlike every
    other kernel in this module this one takes NO ``fs``: it is a single affine map
    ``(x - c) / s`` with ``c`` and ``s`` estimated from the whole trace, so there is no
    window, no kernel and no time constant for a rate to convert.

    THE OUTPUT IS NOT IN pA, AND NOTHING HERE CAN GIVE THAT BACK. ``s`` carries the
    picoamps away with it: a 100 pA blockade in 10 pA noise comes out of ``'zscore'`` as
    a depth of 10 (sigma), and out of ``'minmax'`` as a depth of maybe 0.3 (of the
    trace's full span, which the ONE largest spike in the recording sets). Any depth,
    baseline, threshold or export measured downstream of this stage is in those units,
    not amperes -- and, for ``'minmax'``, in units that change when you cut the trace
    differently. This is why the stage exists (a model or a comparison across recordings
    may want it) and equally why a caller must say so out loud rather than let a "pA"
    label ride along unchanged.

    Modes, all with ``center=True`` subtracting the location estimate first:

    * ``'zscore'``  -- ``(x - mean) / std``. The textbook standardisation. Both moments
      are computed over the WHOLE trace including the events, so a recording with a high
      event rate inflates ``std`` and shrinks every depth in sigma; see :func:`noise_rms`
      for the same trap in the detection threshold.
    * ``'robust'``  -- ``(x - median) / (1.4826 * MAD)``, the event-resistant version of
      the above and the one to prefer when the trace is the app's usual contaminated
      kind: the divisor is :func:`noise_rms`, so "1.0" means "one baseline noise sigma"
      rather than "one sigma of noise-plus-events".
    * ``'minmax'``  -- ``(x - min) / (max - min)``, mapping the trace onto ``[0, 1]``.
      Two ORDER STATISTICS set the scale, so a single digitiser glitch rescales the
      entire recording; despike first (:func:`median_filter`) or prefer one of the
      other two.

    ``center`` means "put zero in the middle", and what that costs differs by mode. For
    ``'zscore'`` / ``'robust'`` ``center=False`` divides WITHOUT subtracting the
    location, which keeps the sign and the zero of the trace meaningful (useful when a
    downstream step cares that open pore is a positive current) at the cost of leaving
    the offset in. ``'minmax'`` always subtracts the minimum -- that subtraction is what
    the mode IS -- so there ``center=True`` shifts the finished ``[0, 1]`` onto
    ``[-0.5, 0.5]`` instead.

    Raises ValueError on an empty trace, an unknown ``mode``, a trace containing
    non-finite samples (the estimators would silently propagate the NaN over every
    sample), or a DEGENERATE scale -- a constant trace, where ``s == 0``: dividing by it
    would return an array of inf/NaN that looks like a filter result rather than the
    "there is nothing here to normalise" that it is.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot normalize an empty trace")
    if mode not in NORMALIZE_MODES:
        raise ValueError(
            f"mode must be one of {NORMALIZE_MODES}, got {mode!r}")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "cannot normalize a trace containing NaN/inf: the location and scale "
            "estimates would propagate it over every sample")

    if mode == "zscore":
        loc, scale = float(np.mean(x)), float(np.std(x))
        scale_name = "standard deviation"
    elif mode == "robust":
        # noise_rms IS the robust sigma (1.4826 * MAD) the detector thresholds on, so
        # "1.0" here and "1 sigma" there mean the same thing by construction.
        loc, scale = float(np.median(x)), float(noise_rms(x))
        scale_name = "robust sigma (1.4826 * MAD)"
    else:  # minmax
        lo, hi = float(np.min(x)), float(np.max(x))
        loc, scale = lo, hi - lo
        scale_name = "peak-to-peak span"

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"cannot normalize: the {scale_name} of this trace is {scale:g}. A constant "
            "(or effectively constant) trace has no scale to divide by."
        )

    if mode == "minmax":
        # The subtraction is the mode, not the centring: skipping it would not be a
        # min-max map at all. `center` moves the finished [0, 1] onto [-0.5, 0.5].
        y = (x - loc) / scale
        if center:
            y = y - 0.5
    else:
        y = (x - loc) / scale if center else x / scale
    return np.ascontiguousarray(y, dtype=np.float32)


def robust_normalize(
    data: np.ndarray, *, clip_sigma: float = MODEL_INPUT_CLIP_SIGMA
) -> np.ndarray:
    """The MODEL-INPUT rescale: trimmed robust sigma, clipped, divided by the clip.

    Returns a contiguous ``float32`` array the same length as ``data``, in units of
    the recording's own EVENT-EXCLUDED noise sigma divided by ``clip_sigma`` -- so the
    noise floor lands at ``1/clip_sigma`` (0.033 at the default 30) in every recording
    by construction, and the output is bounded to ``[-1, +1]``.

    THIS IS NOT A FOURTH MODE OF :func:`normalize`, AND THE SEPARATION IS DELIBERATE.
    That function is one affine map ``(x - c) / s`` with a ``center`` switch; this is a
    three-stage trimmed re-estimation followed by a CLIP, which is not affine, has no
    meaningful ``center``, and is bounded where the other three are unbounded. Their
    DEGENERATE-INPUT CONTRACTS are opposite too: `normalize` raises on a constant trace
    (a NaN/inf array that looks like a filter result is worse than a crash), while this
    one must fall back to a usable scale, because it sits in front of a model on a
    windowed inference path where one flat window would otherwise abort the run. Two
    opposite contracts behind one ``mode=`` keyword is how a caller ends up believing
    it got the other one. Dispatch on `MODEL_INPUT_MODE` instead.

    WHY THIS EXISTS AT ALL -- the notebooks' own reason, with their numbers. The model
    input used to be ``x / max|x|``. That divisor is a SINGLE EXTREME SAMPLE, and it
    ranged **8.8 to 42.6 pA** across the host-guest recordings, so ordinary noise
    reached the model at 0.049 for the pair (the training domain), 0.081 for the guest
    (1.7x) and 0.146 for the host (3.0x) -- i.e. host NOISE arrived at the amplitude
    the model had been taught to call an EVENT.
    That is the artefact over-detection. The same divisor also means one unusually deep
    sample scales every real event down and makes them easy to miss. In sigma units the
    noise floor is 1.0 in every recording, so a detector transfers between recordings
    instead of inheriting each file's extreme value.

    THE ALGORITHM, transcribed from the six notebooks that define it identically
    (``*_StudentModel`` / ``*_StudentInference`` in the Cu, Zn and host-guest
    pipelines). Every step matters and the order is load-bearing:

    1. ``m0 = median(x)``, ``s0 = 1.4826 * MAD(x - m0)`` -- a first robust pair over
       everything, events included.
    2. ``k = |x - m0| < 5*s0`` -> ``m1, s1`` re-estimated on ``x[k]``. Five sigmas is
       wide enough to keep the noise intact and narrow enough to drop deep blockades.
    3. ``q = |x - m1| < 3*s1`` -> ``med, mad`` re-estimated on ``x[q]``, **only if
       ``q.sum() > 100``**. That guard is the whole reason step 3 is safe: a 3-sigma
       trim on a short or event-dense window can leave a handful of samples whose MAD
       is a meaningless near-zero, and dividing by it would explode the window. Under
       101 surviving samples the stage keeps step 2's answer instead.
    4. ``z = (x - med) / mad``, then **clip to +-clip_sigma FIRST and divide by
       clip_sigma SECOND**. The order is not cosmetic: clipping after the divide would
       need a different bound, and dividing a clipped z by anything else would move the
       noise floor off ``1/clip_sigma`` -- which is the one number the model was
       trained against.

    ``max(s, 1e-12)`` guards each trim threshold, and a non-finite or non-positive
    final ``mad`` falls back to ``s0`` (or 1.0), exactly as the notebooks do: a flat
    window returns zeros rather than raising.

    THE ARITHMETIC RUNS IN float32, because the notebooks' does. ``np.asarray(signal,
    np.float32)`` is their first line, and every median/MAD after it is taken on that
    float32 view; computing in float64 and casting at the end would agree to about 1e-7
    but would not be the same bytes, and this function's contract is that it IS the
    same bytes. ``tests/analysis/test_robust_normalize.py`` pins that against a literal
    transcription of the notebook function rather than against this implementation.

    Raises ValueError on an empty trace, a non-finite sample (the medians would
    propagate the NaN over every sample -- the notebooks never pass one, and this is
    the module's standing rule), or a non-positive/non-finite ``clip_sigma``.
    """
    x = np.asarray(data, dtype=np.float32)
    if x.size == 0:
        raise ValueError("cannot normalize an empty trace")
    if not np.isfinite(clip_sigma) or clip_sigma <= 0:
        raise ValueError(f"clip_sigma must be positive and finite, got {clip_sigma}")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "cannot robust_normalize a trace containing NaN/inf: the median and MAD "
            "estimates would propagate it over every sample")

    # Pass 0 -- robust location/scale over everything, events included.
    m0 = float(np.median(x))
    s0 = 1.4826 * float(np.median(np.abs(x - m0)))
    # Pass 1 -- 5 sigma keeps the noise, drops the deep blockades.
    k = np.abs(x - m0) < 5 * max(s0, 1e-12)
    m1 = float(np.median(x[k])) if k.any() else m0
    s1 = 1.4826 * float(np.median(np.abs(x[k] - m1))) if k.any() else s0
    # Pass 2 -- a tighter 3 sigma trim, but ONLY with more than 100 survivors: below
    # that the MAD of the remainder is noise about noise, and it is the DIVISOR.
    q = np.abs(x - m1) < 3 * max(s1, 1e-12)
    med = float(np.median(x[q])) if q.sum() > 100 else m1
    mad = 1.4826 * float(np.median(np.abs(x[q] - med))) if q.sum() > 100 else s1
    if not np.isfinite(mad) or mad <= 0:
        mad = s0 if s0 > 0 else 1.0

    z = (x - med) / mad
    # CLIP THEN DIVIDE. See the docstring: this is the order the model was trained on.
    y = np.clip(z, -clip_sigma, clip_sigma) / clip_sigma
    return np.ascontiguousarray(y, dtype=np.float32)


def median_filter(data: np.ndarray, fs: float, *, window_s: float) -> np.ndarray:
    """Sliding median over a ``window_s`` window (an impulse/spike remover).

    Returns a contiguous ``float32`` array the same length as ``data``. A median is the tool
    for one-off contamination -- single-sample spikes, digitiser glitches, cosmic-ray-style
    outliers -- that a linear filter (moving average, Savitzky-Golay) would instead smear
    across its whole window. Crucially it PRESERVES a square edge: the median of a window
    straddling a step is the step, so a blockade's height and the location of its edges
    survive, while an isolated spike shorter than half the window is deleted outright.

    ``window_s`` is converted to a sample count and forced ODD (a median window is centred on
    the sample it replaces) and clamped to the trace length. A window of ``N`` rejects spikes
    up to ``(N - 1) / 2`` samples wide, so keep it a few samples -- much shorter than the
    shortest real event, or it will eat the events too.

    Raises ValueError on a non-positive / non-finite ``window_s``, a non-positive rate, or a
    trace too short to host even a length-1 window (an empty trace).
    """
    x = np.asarray(data, dtype=np.float64)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"window_s must be positive and finite, got {window_s}")
    if x.size == 0:
        raise ValueError("cannot median-filter an empty trace")

    # window_s -> samples, forced ODD (a median window is centred on its sample) and clamped
    # to the trace (medfilt needs the window no longer than the signal it slides over).
    win = int(round(window_s * fs))
    win = max(win, 1)
    if win % 2 == 0:
        win += 1
    if win > x.size:
        win = x.size if x.size % 2 == 1 else x.size - 1
        win = max(win, 1)
    y = signal.medfilt(x, kernel_size=win)
    return np.ascontiguousarray(y, dtype=np.float32)


def noise_rms(x: np.ndarray) -> float:
    """Robust baseline noise RMS in the units of `x` (pA), via 1.4826 * MAD.

    NOT np.std. The trace you measure noise on contains the very events you are
    about to detect against that noise, and std is not robust to them: every event
    inflates the estimate, which raises the detection threshold, which loses the
    smallest real events -- silently, and worst exactly when the event rate is high.
    A published-pipeline bug, not a hypothetical.

    1.4826 rescales the median absolute deviation to a Gaussian-equivalent sigma, so
    for clean Gaussian noise this agrees with std; it just stops agreeing when it
    should.

    Robust, not unbiased. Events are one-sided contamination, which drags the median
    slightly and leaves MAD biased HIGH by a few percent: measured against 12 pA
    Gaussian noise contaminated with 20-sigma blockades, MAD reads +1.3% at 1%
    contamination, +6.1% at 5%, +14.6% at 10% (std reads +123%, +347%, +508%). So
    at a plausible duty cycle MAD costs you a slightly conservative threshold while
    std costs you most of your events. Above ~20% duty cycle neither is usable --
    estimate noise from an event-free stretch instead.

    Note MAD is taken about the median, so a strong baseline DRIFT will still inflate
    it -- subtract a baseline first if the trace drifts.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("cannot estimate noise from an empty array")
    x = x.astype(np.float64, copy=False)
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite samples; noise estimate would be meaningless")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def psd(
    data: np.ndarray, fs: float, *, nperseg: int = 65536
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density: returns (f in Hz, Pxx in pA^2/Hz).

    Use it to see what you are fighting: mains lines and their harmonics, 1/f corner,
    and the dielectric noise ramp that sets the usable bandwidth. nperseg is clamped
    to the trace length; the frequency resolution is fs/nperseg, so a 65536-point
    segment at 250 kHz resolves ~3.8 Hz -- fine enough to distinguish a 60 Hz line.
    """
    if nperseg < 2:
        raise ValueError(f"nperseg must be >= 2, got {nperseg}")
    data = np.asarray(data)
    n = min(int(nperseg), data.size)
    f, pxx = signal.welch(
        data.astype(np.float64), fs=fs, nperseg=n, scaling="density"
    )
    return f, pxx


# ---------------------------------------------------------------------------
# Frequency response (pure; the GUI's response preview draws exactly this)
# ---------------------------------------------------------------------------

# The stages with a well-defined linear magnitude response. 'median' is absent ON
# PURPOSE: a sliding median is a NON-LINEAR rank filter, so it has no transfer
# function at all -- any |H(f)| drawn for it would be a fabrication. Callers must
# check membership and say so rather than plot a curve.
RESPONSE_STAGES = ("lowpass", "highpass", "notch", "running_mean", "savgol")

# Floor for the dB conversion, so a true zero (the Dirichlet nulls of a box average,
# the notch centre) plots as a finite trough instead of -inf.
_DB_FLOOR = -200.0


def _response_grid(
    fs: float, n: int, anchor_hz: float | None, f_min_hz: float | None
) -> np.ndarray:
    """Log-spaced frequencies from a sensible low end up to Nyquist.

    Log spacing is not cosmetic: a nanopore chain spans 60 Hz mains to 100 kHz
    bandwidth, and on a linear axis every interesting feature below 1 kHz collapses
    onto the origin. The low end is pinned two decades below whatever feature the
    stage is built around (its cutoff / line / 1-over-window) so that feature is never
    against the left edge.
    """
    nyq = fs / 2.0
    if f_min_hz is not None:
        # An explicit low end is honoured EXACTLY: a caller asking to evaluate from a
        # particular frequency (a probe, a zoom) must get that frequency, not a clamped
        # approximation of it.
        lo = float(max(min(float(f_min_hz), nyq), 1e-9))
    else:
        lo = nyq / 1e4
        if anchor_hz is not None and np.isfinite(anchor_hz) and anchor_hz > 0:
            lo = min(lo, float(anchor_hz) / 100.0)
        # Derived low ends only: keep at least a decade of axis, and stay off zero.
        lo = float(max(min(lo, nyq / 10.0), 1e-4))
    return np.logspace(np.log10(lo), np.log10(nyq), int(n))


def response(
    stage: str,
    fs: float,
    *,
    cutoff_hz: float | None = None,
    order: int = 8,
    kind: str = "bessel",
    freq_hz: float | None = None,
    q: float = 30.0,
    harmonics: int = 3,
    freqs: Sequence[float] | None = None,
    window_s: float | None = None,
    polyorder: int = 3,
    zero_phase: bool = True,
    n: int = 512,
    f_min_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude response |H(f)| of one filter stage, AS THE STAGE IS ACTUALLY APPLIED.

    Returns ``(f_hz, magnitude)`` -- linear amplitude ratio, not dB (call
    :func:`response_db`), on a log-spaced grid ending at Nyquist.

    THE POINT OF THIS FUNCTION IS THE SQUARING. Every IIR stage here runs through
    ``sosfiltfilt`` (forward then backward, so edges do not move), which applies the
    designed response TWICE: the magnitude is |H|^2 and the -3 dB design point is a
    -6 dB operating point. A "10 kHz 8-pole Bessel" therefore delivers roughly what a
    hardware 4-pole Bessel at ~7.1 kHz delivers, which is not what the label on the
    control says. With ``zero_phase=True`` (the default -- what the app does) this
    returns the doubled response, so that gap is visible on the plot instead of being
    a footnote in a docstring. ``zero_phase=False`` returns the single-pass design
    response, which is what a textbook or a scipy ``sosfreqz`` call gives you.

    The symmetric-kernel stages ('gaussian', 'running_mean', 'savgol') are applied ONCE
    as a centred convolution -- they are already zero-phase without a second pass -- so
    ``zero_phase`` does not affect them.

    Args:
        stage: One of :data:`RESPONSE_STAGES`. ``'median'`` is deliberately NOT
            supported (see that constant).
        fs: Sample rate in Hz.
        cutoff_hz: -3 dB design cutoff, for ``'lowpass'`` / ``'highpass'``.
        order: IIR order, for ``'lowpass'`` / ``'highpass'``. Ignored by
            ``kind='gaussian'``, which has no order.
        kind: ``'bessel'`` / ``'butter'`` / ``'gaussian'`` (low-pass) or
            ``'butter'`` / ``'bessel'`` (high-pass).
        freq_hz, q, harmonics: The notch line, its Q, and how many multiples.
        freqs: An EXPLICIT notch line list, overriding ``freq_hz``/``harmonics`` the
            same way it does in :func:`notch`. Given, the plotted response is the
            cascade the stage will really apply -- which for a set with gaps
            (60/120/180/300/420/540) is a visibly different comb from any contiguous
            series. The grid is still centred on the LOWEST line, the narrowest and
            therefore the hardest one to resolve.
        window_s, polyorder: For ``'running_mean'`` / ``'savgol'``. The window is
            converted to samples by the SAME helper the kernels use, so a window that
            rounds to one sample plots as the flat identity it really is.
        zero_phase: See above. Only meaningful for the IIR paths.
        n: Number of frequency points.
        f_min_hz: Override the grid's low end.

    Raises:
        ValueError: on an unknown/unsupported stage, a missing parameter for the stage,
            or a cutoff at/above Nyquist (the same rule the kernels enforce).
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if stage not in RESPONSE_STAGES:
        if stage == "median":
            raise ValueError(
                "a sliding median is a non-linear rank filter and has NO transfer "
                "function; there is no |H(f)| to draw for it."
            )
        raise ValueError(f"stage must be one of {RESPONSE_STAGES}, got {stage!r}")

    if stage in ("lowpass", "highpass"):
        if cutoff_hz is None:
            raise ValueError(f"stage={stage!r} needs cutoff_hz")
        _validate_cutoff(fs, float(cutoff_hz))
        f = _response_grid(fs, n, float(cutoff_hz), f_min_hz)
        if stage == "lowpass" and kind == "gaussian":
            # Applied once, symmetric: H(f) = exp(-2 pi^2 sigma_t^2 f^2), the closed
            # form _gaussian_sigma_samples is derived from (|H| = 1/sqrt2 at cutoff).
            sigma_t = _gaussian_sigma_samples(fs, float(cutoff_hz)) / fs
            mag = np.exp(-2.0 * np.pi**2 * sigma_t**2 * f**2)
            return f, mag
        kinds = _KINDS if stage == "lowpass" else _HP_KINDS
        if kind not in kinds:
            raise ValueError(f"kind must be one of {kinds}, got {kind!r}")
        if int(order) != order or order < 1:
            raise ValueError(f"order must be a positive integer, got {order!r}")
        sos = _design_sos(
            fs, float(cutoff_hz), int(order), kind,
            btype="low" if stage == "lowpass" else "high",
        )
    elif stage == "notch":
        if freq_hz is None and freqs is None:
            raise ValueError("stage='notch' needs freq_hz (or an explicit freqs list)")
        # `notch_frequencies` raises the "nothing below Nyquist" error itself, from the
        # same resolver `notch` designs its cascade from -- so the plot and the run
        # cannot disagree about which lines exist.
        keep = notch_frequencies(
            fs, 60.0 if freq_hz is None else float(freq_hz),
            harmonics=int(harmonics), freqs=freqs,
        )
        if q <= 0:
            raise ValueError(f"q must be positive, got {q}")
        f = _response_grid(fs, n, min(keep), f_min_hz)
        sos = np.vstack(
            [np.hstack(signal.iirnotch(line, q, fs=fs))[None, :] for line in keep]
        )
    elif stage == "running_mean":
        if window_s is None:
            raise ValueError("stage='running_mean' needs window_s")
        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError(f"window_s must be positive and finite, got {window_s}")
        n_box = running_mean_samples(fs, float(window_s))
        f = _response_grid(fs, n, 1.0 / float(window_s), f_min_hz)
        # Centred box of N samples, applied once: the Dirichlet kernel
        #     |H| = |sin(N w / 2) / (N sin(w / 2))|,   w = 2 pi f / fs
        # in closed form rather than via freqz, which would evaluate an N-tap polynomial
        # at every frequency -- O(N * n) work that reaches a third of a second at the
        # panel's widest window. |.| because the side lobes alternate sign (a phase flip,
        # not extra attenuation).
        w = np.pi * f / (fs / 2.0)
        den = n_box * np.sin(w / 2.0)
        mag = np.ones_like(f)
        nz = den != 0.0
        mag[nz] = np.abs(np.sin(n_box * w[nz] / 2.0) / den[nz])
        return f, mag
    else:  # savgol
        if window_s is None:
            raise ValueError("stage='savgol' needs window_s")
        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError(f"window_s must be positive and finite, got {window_s}")
        if int(polyorder) != polyorder or polyorder < 1:
            raise ValueError(f"polyorder must be a positive integer, got {polyorder!r}")
        win = savgol_window_samples(fs, float(window_s), int(polyorder))
        f = _response_grid(fs, n, 1.0 / float(window_s), f_min_hz)
        # The exact FIR kernel savgol_filter convolves with (use='conv'), applied once
        # and symmetric -- so the curve is the response of the window actually used,
        # clamp and odd-rounding included.
        #
        # Do NOT read the step overshoot off this curve. A Savitzky-Golay does put an
        # opposite-sign lobe on each edge of a square blockade, but that is a TIME-domain
        # effect of the kernel's negative tails; |H(f)| does not exceed 1 for any window
        # or polyorder this app can build (measured: max |H| == 1.0 at DC for every
        # window 5..59 x polyorder 2..6, the panel's whole reachable range). The caller's
        # pass-band-peaking note is therefore guarded on an actual measured peak rather
        # than asserted, and stays silent here -- which is correct, not a missing feature.
        coeffs = signal.savgol_coeffs(win, int(polyorder), use="conv")
        _, h = signal.freqz(coeffs, worN=f, fs=fs)
        return f, np.abs(h)

    _, h = signal.sosfreqz(sos, worN=f, fs=fs)
    mag = np.abs(h)
    return f, mag**2 if zero_phase else mag


def response_db(magnitude: np.ndarray) -> np.ndarray:
    """Linear amplitude ratio -> dB (20 log10), with a finite floor at a true zero."""
    mag = np.abs(np.asarray(magnitude, dtype=np.float64))
    out = np.full(mag.shape, _DB_FLOOR)
    nz = mag > 0
    np.log10(mag, out=out, where=nz)
    out[nz] *= 20.0
    return np.maximum(out, _DB_FLOOR)


def response_crossing_hz(
    f_hz: np.ndarray, magnitude: np.ndarray, level: float = _HALF_POWER
) -> float | None:
    """First frequency where ``magnitude`` crosses ``level``, log-interpolated.

    Defaults to the -3 dB amplitude point, so a caller can state the response's TRUE
    corner next to the nominal one -- for a zero-phase 8-pole Bessel those differ by
    ~29% and nothing in the app used to say so. Returns ``None`` when the curve never
    crosses (e.g. a high-pass whose grid stops short, or an identity stage).
    """
    f_hz = np.asarray(f_hz, dtype=np.float64)
    mag = np.asarray(magnitude, dtype=np.float64)
    if f_hz.size < 2 or f_hz.size != mag.size:
        return None
    above = mag >= level
    idx = np.flatnonzero(above[:-1] != above[1:])
    if idx.size == 0:
        return None
    i = int(idx[0])
    m0, m1 = mag[i], mag[i + 1]
    if m0 == m1:
        return float(f_hz[i])
    # Interpolate in log-f (the grid is geometric, so a linear interpolation there
    # would systematically favour the upper sample).
    t = (level - m0) / (m1 - m0)
    lf = np.log10(f_hz[i]) + t * (np.log10(f_hz[i + 1]) - np.log10(f_hz[i]))
    return float(10.0**lf)
