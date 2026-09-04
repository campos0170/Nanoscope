"""Power-spectral-density / current-noise analysis — the pore quality-control tool.

These are pure ``(signal, fs) -> numpy`` kernels with no Qt and no
:class:`~nanoscope.core.session.Session`; the GUI shell around them lives in
:mod:`nanoscope.gui.psd_panel`. They operate on the **calibrated current trace
directly** — PSD is raw-signal characterisation and needs no detection, so
nothing here touches the frozen 20-column ``events.SCHEMA``.

WHAT THE NOISE SPECTRUM SHOWS
-----------------------------
A nanopore current-noise power spectrum on log-log axes separates the three
physical noise regimes every pore-QC plot reads off:

* **1/f (flicker) noise** at low frequency — a ``S(f) = A · f^(-alpha)`` power
  law (``alpha`` near 1 for ideal flicker), dominant below the thermal knee.
  :func:`fit_flicker` recovers ``alpha`` and the magnitude at 1 Hz.
* a **mid-log-frequency median proxy** — :func:`noise_summary` reports the median
  level over a deterministic middle decade. It is useful for comparison but is not,
  by itself, proof that this band is a flat Johnson/shot-noise plateau; mains lines or
  a sloped region can fall inside it.
* **the high-frequency dielectric (``~f``) / capacitance (``~f^2``) rise** near
  Nyquist, where amplifier + membrane capacitance lift the spectrum.

UNITS — READ THIS
-----------------
Everything here is **unit-agnostic**: it works in whatever units the calibrated
current trace carries. If the caller passes a signal in **pA**, the PSD comes
back in **pA²/Hz**, :func:`band_rms` and ``total_rms`` in **pA**, the amplitude
spectral density (``sqrt(PSD)``) in **pA/√Hz**, and ``A1Hz`` (a PSD magnitude)
in **pA²/Hz**. Pass **nA** and every one of those reads in the nA-scaled unit
instead. No unit conversion or scale inference happens in this module — the
caller owns the unit label (exactly as :func:`nanoscope.analysis.measure.measure_selection`
does), so a pA-vs-nA channel flows through untouched and is labelled by the GUI.

The Welch estimate uses ``scaling="density"`` (a one-sided PSD in units²/Hz), so
integrating the PSD over ``[0, Nyquist]`` recovers the signal **variance** and
its square root the RMS — the Parseval identity the RMS readouts lean on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import signal as _sp_signal

from ._compat import trapezoid

__all__ = [
    "welch_psd",
    "fft_magnitude",
    "FFTSpectrum",
    "FFT_MAX_POINTS",
    "MAX_FFT_INPUT_SAMPLES",
    "spectrogram_psd",
    "Spectrogram",
    "default_spectrogram_nperseg",
    "TARGET_COLUMNS",
    "MAX_SPECTROGRAM_CELLS",
    "band_rms",
    "band_support",
    "fit_flicker",
    "noise_summary",
    "default_nperseg",
    "drop_dc",
    "asd_from_psd",
    "flicker_status",
    "DEFAULT_BANDS",
    "DEFAULT_WINDOW",
    "FLICKER_BAND_HZ",
    "FLICKER_MIN_POINTS",
    "FLICKER_MARGINAL_POINTS",
    "TARGET_SEGMENTS",
]

# Default band decomposition for :func:`noise_summary`, ``(label, f_lo, f_hi)`` in Hz.
# The top band runs to Nyquist (``inf`` is clipped there). Bands entirely above a
# low Nyquist collapse to a zero RMS, never NaN.
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<1 kHz", 0.0, 1.0e3),
    ("1-10 kHz", 1.0e3, 1.0e4),
    ("10 kHz-Nyquist", 1.0e4, np.inf),
)

DEFAULT_WINDOW = "hann"

# Fixed low-frequency band (Hz) the automatic 1/f fit in :func:`noise_summary`
# runs over, clipped to the spectrum's own ``[first positive bin, Nyquist]``. It
# is pinned to an ABSOLUTE range rather than the lowest FFT decade so the flicker
# exponent does not swing with the Welch segment length (see the note in
# :func:`noise_summary`). 1-100 Hz sits squarely in the flicker regime, below the
# thermal knee, for the sample rates this app targets.
FLICKER_BAND_HZ: tuple[float, float] = (1.0, 100.0)

# Welch averages over this many segments by default: the auto ``nperseg`` is the
# largest power of two that leaves at least this many non-overlapping blocks (so
# ~2x that many at the default 50% overlap), balancing variance against
# frequency resolution. Public because a caller that lets the user override
# ``nperseg`` needs to know what the automatic choice was aiming at before it can
# say "this segment is short for these parameters".
TARGET_SEGMENTS = 8
_TARGET_SEGMENTS = TARGET_SEGMENTS       # back-compatible private alias
_MIN_NPERSEG = 256
_MAX_NPERSEG = 1 << 16   # 65536 — cap the segment length on very long records

# HOW MANY FFT BINS A 1/f EXPONENT NEEDS — the one place that policy is written.
#
# Below :data:`FLICKER_MIN_POINTS`, :func:`noise_summary` reports the fit as
# unavailable rather than trusting a slope through a handful of bins. AT that
# floor the fit is still computed, and it is the *presentation* that used to lie:
# a straight line has two free parameters, so four points leave two residual
# degrees of freedom and an R² that is nearly guaranteed to look high — a real
# recording printed "α 1.207, R² 0.980" from exactly four bins with nothing on
# screen saying so. :data:`FLICKER_MARGINAL_POINTS` (twice the floor) is where a
# reader may start treating R² as evidence that the power law holds; between the
# two the fit is real but MARGINAL and every surface that prints α must say so.
#
# Both numbers, and the verdict derived from them, are exported so the GUI can
# not drift from the core: :func:`flicker_status` is the single arbiter.
FLICKER_MIN_POINTS = 4
FLICKER_MARGINAL_POINTS = 8
_FLICKER_MIN_POINTS = FLICKER_MIN_POINTS   # back-compatible private alias


def flicker_status(n_points: int) -> str:
    """How much weight the 1/f fit's ``n`` bins can carry: the single arbiter.

    Args:
        n_points: Number of FFT bins the log-log line was actually fitted through
            (``fit_flicker(...)["n"]``).

    Returns:
        ``"unavailable"`` (fewer than :data:`FLICKER_MIN_POINTS` bins — no exponent
        may be reported), ``"marginal"`` (at or above the floor but under
        :data:`FLICKER_MARGINAL_POINTS` — an exponent may be reported only
        alongside its bin count and a caution), or ``"ok"``.
    """
    n = int(n_points)
    if n < FLICKER_MIN_POINTS:
        return "unavailable"
    if n < FLICKER_MARGINAL_POINTS:
        return "marginal"
    return "ok"


# ================================================================================================
# helpers
# ================================================================================================

def default_nperseg(n: int) -> int:
    """A sensible Welch segment length for a signal of ``n`` samples.

    The largest power of two ``<= n / _TARGET_SEGMENTS`` (so Welch averages over
    at least ``_TARGET_SEGMENTS`` non-overlapping blocks — roughly twice that at
    the default half-overlap), floored at :data:`_MIN_NPERSEG`, capped at
    :data:`_MAX_NPERSEG`, and never larger than ``n`` itself. A short signal
    (``n <= _MIN_NPERSEG``) uses the whole record as one segment (a periodogram).
    """
    n = int(n)
    if n <= _MIN_NPERSEG:
        return max(n, 1)
    target = max(1, n // _TARGET_SEGMENTS)
    p = 1 << (target.bit_length() - 1)           # largest power of two <= target
    p = min(p, _MAX_NPERSEG)
    p = max(p, _MIN_NPERSEG)
    return min(p, n)


def drop_dc(freqs: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the ``f == 0`` (DC) bin so the spectrum can be drawn on a log axis.

    Returns views/copies of ``(freqs, psd)`` with every non-positive-frequency
    bin removed. Harmless if there is no DC bin (a two-sided or already-trimmed
    spectrum) — it simply keeps the positive-frequency bins.
    """
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    psd = np.asarray(psd, dtype=np.float64).ravel()
    keep = freqs > 0.0
    return freqs[keep], psd[keep]


def asd_from_psd(psd: np.ndarray) -> np.ndarray:
    """Amplitude spectral density ``sqrt(PSD)`` — units/√Hz from units²/Hz.

    Non-finite or (defensively) negative PSD values map to ``NaN`` rather than an
    ``sqrt`` warning, so a caller can plot the ASD without pre-cleaning.
    """
    psd = np.asarray(psd, dtype=np.float64)
    out = np.full(psd.shape, np.nan, dtype=np.float64)
    good = np.isfinite(psd) & (psd >= 0.0)
    out[good] = np.sqrt(psd[good])
    return out


# ================================================================================================
# Welch PSD
# ================================================================================================

def welch_psd(
    signal: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = DEFAULT_WINDOW,
    detrend: str | bool = "constant",
    onesided: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of a 1-D signal, in ``(signal-units)²/Hz``.

    A thin, guarded wrapper over :func:`scipy.signal.welch` with
    ``scaling="density"`` so the estimate is a one-sided PSD whose integral over
    ``[0, Nyquist]`` is the signal variance (Parseval).

    Args:
        signal: The 1-D calibrated current trace (or a segment of it). Coerced to
            ``float64`` and flattened.
        fs: Sample rate in Hz — finite and strictly positive.
        nperseg: Welch segment length. ``None`` picks :func:`default_nperseg`;
            an explicit value is clamped to at most ``len(signal)``.
        noverlap: Samples of overlap between segments. ``None`` lets scipy use
            ``nperseg // 2`` (50%); an explicit value is clamped below
            ``nperseg``.
        window: Window name passed to scipy (via ``scipy.signal.get_window``),
            default ``"hann"``.
        detrend: Per-segment detrending — ``"constant"`` (remove the mean,
            default), ``"linear"``, or ``False``. Removing the mean sends the DC
            bin to ~0, so the low-frequency RMS reflects fluctuation, not offset.
        onesided: Return a one-sided spectrum (real input). ``False`` gives the
            two-sided spectrum.

    Returns:
        ``(freqs, psd)`` — ``float64`` arrays of the same length. ``freqs`` is in
        Hz (``freqs[0] == 0`` is the DC bin for a one-sided spectrum; use
        :func:`drop_dc` before a log plot); ``psd`` is in ``(signal-units)²/Hz``.

    Raises:
        ValueError: on an empty / single-sample signal, or a non-finite,
            non-positive ``fs``.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError(
            f"welch_psd needs at least 2 samples to estimate a spectrum; got {n}")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"welch_psd needs a finite, positive sample rate; got fs={fs!r}")

    # Say it HERE, where the cause is still visible. Welch averages segment periodograms,
    # so one NaN/inf sample makes the WHOLE spectrum non-finite and every downstream number
    # unusable; by the time a caller is looking at the PSD there is nothing left to point at.
    # We do NOT interpolate or drop the bad samples: silently repairing a recording is worse
    # than reporting it, and the caller may legitimately want the poisoned spectrum in order
    # to see that it is poisoned.
    n_bad = int(np.count_nonzero(~np.isfinite(x)))
    if n_bad:
        warnings.warn(
            f"the signal contains {n_bad:,} non-finite sample(s) of {n:,}; every bin of the "
            f"resulting spectrum will be NaN and no RMS can be read from it.",
            RuntimeWarning, stacklevel=2)

    nps = default_nperseg(n) if nperseg is None else min(int(nperseg), n)
    nps = max(int(nps), 1)
    nov = None if noverlap is None else int(np.clip(int(noverlap), 0, nps - 1))

    freqs, psd = _sp_signal.welch(
        x,
        fs=float(fs),
        window=window,
        nperseg=nps,
        noverlap=nov,
        detrend=detrend,
        return_onesided=bool(onesided),
        scaling="density",
    )
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    # A LIVE signal whose spectrum reads all-zero has UNDERFLOWED, not gone quiet — and
    # this is the one place both the samples and the spectrum are in hand to say so.
    # Squaring any magnitude below ~1.5e-162 lands under float64's smallest subnormal
    # (~5e-324) and reads as exactly 0.0 inside the periodogram, so a 1e-310-scale
    # trace with real, measurable time-domain noise produces a spectrum of finite
    # zeros that every downstream integral will happily report as "Total RMS 0" — the
    # most convincing-looking QC result there is, and band_rms's own contract ("'no
    # noise in this band' and 'this band cannot be measured' are different answers")
    # forbids it. A genuinely CONSTANT signal (min == max) keeps its honest all-zero
    # spectrum with no warning; a varying one cannot produce one except by losing the
    # variation to representation, so that case warns like the non-finite case above.
    if n_bad == 0 and psd.size and not psd.any():
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax > xmin:
            span = xmax - xmin
            if span < 1e-150:      # even the full swing squares below float64's tiniest
                warnings.warn(
                    f"the signal varies (peak-to-peak {span:.4g}) but every one of "
                    f"{psd.size} PSD bins is exactly 0: fluctuations this small square to "
                    f"below float64's smallest representable number inside the Welch "
                    f"estimate, so its noise is real but unmeasurable from this spectrum — "
                    f"an RMS integrated from it is below-representable, not zero. Rescale "
                    f"the signal (e.g. a larger calibration gain) to measure it.",
                    RuntimeWarning, stacklevel=2)
            else:
                # The only other way a varying signal yields an identically-zero
                # spectrum: every analysed segment is constant once detrended (e.g. a
                # step that falls exactly between non-overlapping segments). The 0 is
                # then arithmetic on what Welch saw, not a measurement of the whole
                # signal, and the honest report names that.
                warnings.warn(
                    f"the signal varies (peak-to-peak {span:.4g}) but every one of "
                    f"{psd.size} PSD bins is exactly 0: each Welch segment is constant "
                    f"after detrending, so the variation falls entirely between segments "
                    f"and this spectrum measures none of it.",
                    RuntimeWarning, stacklevel=2)
    return (freqs, psd)


# ================================================================================================
# band-integrated RMS noise
# ================================================================================================

def band_support(freqs: np.ndarray, f_lo: float, f_hi: float) -> tuple[float, float] | None:
    """The band :func:`band_rms` will actually integrate: the request clipped to the
    spectrum's finite sampled support.

    Returns ``(lo, hi)`` with ``hi > lo``, or ``None`` when the clipped band is empty
    (degenerate, inverted, wholly above Nyquist, or there are no frequencies) — the
    case :func:`band_rms` scores as an honest ``0.0``.

    This is ONE arithmetic with two consumers: the integral, and any label that
    restates the band beside the integral's value. They must not be computed twice —
    the PSD window's "RMS in band" read-out used to round the edges independently of
    the integration and printed ``(0–0 Hz)`` beside a genuine 0–0.5 Hz figure (a band
    this module itself defines as holding nothing) and ``(0–2 Hz)`` at a 1.5 Hz
    Nyquist (a band that does not exist).
    """
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    finite = freqs[np.isfinite(freqs)]
    if finite.size == 0 or np.isnan(float(f_lo)) or np.isnan(float(f_hi)):
        return None
    lo = max(float(f_lo), float(np.min(finite)))
    hi = min(float(f_hi), float(np.max(finite)))
    if not (hi > lo):
        return None
    return (lo, hi)


def band_rms(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """RMS noise over ``[f_lo, f_hi]`` — ``sqrt(∫ PSD df)`` in signal units.

    Integrates the piecewise-linear PSD over the requested band (trapezoidal rule)
    and returns the square root, i.e. the standard deviation of the signal restricted
    to that frequency band. The PSD is interpolated at both requested edges before
    integration. Consequently a real band narrower than one native frequency bin still
    has a real, non-zero width; it is never mistaken for an empty band merely because
    neither edge happened to land on a Welch bin. For a full-band ``[0, Nyquist]``
    request this is the total RMS noise (Parseval); for a narrow band around a tone it
    is that tone's RMS.

    The band is clipped to the spectrum's finite sampled support by
    :func:`band_support`, which is also the function any caller that
    RESTATES the band beside this value must use. A band that is genuinely EMPTY —
    degenerate, inverted, or wholly above Nyquist — integrates to nothing and returns
    ``0.0``, so per-band summaries stay well-defined at any sample rate.

    A band that is NOT empty but whose PSD is not usable returns **NaN with a
    warning**, never ``0.0``. That distinction is the whole point of this function's
    contract:

        "no noise in this band" and "this band cannot be measured" are different
        answers, and only one of them may be printed as a number.

    Welch averages segment periodograms, so a single non-finite sample anywhere in
    the signal makes EVERY bin NaN. Filtering those bins out and reporting ``0.0``
    turned a recording whose true σ was 1.49 pA into a confident
    "Total RMS noise: 0 pA" — the most convincing-looking QC result a noise panel can
    produce, and indistinguishable from a perfect pore. Any non-finite PSD bin inside
    the requested band now makes the whole answer NaN: a partially poisoned band
    cannot be integrated honestly either, and quietly skipping the bad bins would
    understate the noise by exactly the part that is broken.

    Args:
        freqs: Frequency axis in Hz (ascending; ``freqs[-1]`` is the Nyquist).
        psd: PSD in ``(signal-units)²/Hz``, aligned to ``freqs``.
        f_lo: Low band edge in Hz.
        f_hi: High band edge in Hz.

    Returns:
        The band RMS in signal units (``float``); ``0.0`` for an empty/degenerate
        band; ``NaN`` (with a :class:`RuntimeWarning`) when the band holds bins that
        cannot be integrated.
    """
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    psd = np.asarray(psd, dtype=np.float64).ravel()
    if freqs.size != psd.size:
        raise ValueError(
            f"band_rms needs aligned frequency and PSD arrays; got {freqs.size} and "
            f"{psd.size} values"
        )
    if psd.size == 0:
        return 0.0

    # The clipped band comes from band_support, NEVER re-derived: it is also what a
    # caller prints beside this integral, and two computations of "the band" is how a
    # label and its number end up disagreeing.
    support = band_support(freqs, f_lo, f_hi)
    if support is None:
        return 0.0
    lo, hi = support

    finite_frequency = np.isfinite(freqs)
    f = freqs[finite_frequency]
    p = psd[finite_frequency]
    if f.size < 2:
        warnings.warn(
            f"the spectrum has fewer than two finite frequency samples, so the band RMS "
            f"over {lo:.4g}–{hi:.4g} Hz is undefined (NaN).",
            RuntimeWarning, stacklevel=2)
        return float("nan")
    if np.any(np.diff(f) <= 0.0):
        raise ValueError("band_rms needs a strictly increasing frequency axis")

    # Integrate the requested interval itself, not merely the native samples that happen
    # to fall inside it. The latter returns 0 for, say, 1–9 Hz on a 10 Hz grid even though
    # that band is eight hertz wide. Interpolated boundary values make the trapezoid exact
    # for the piecewise-linear spectrum represented by the sampled PSD and preserve the
    # old bit-for-bit arithmetic whenever both edges are native bins.
    interior = (f > lo) & (f < hi)
    f_band = np.concatenate(([lo], f[interior], [hi]))
    p_band = np.concatenate((
        ([np.interp(lo, f, p)]),
        p[interior],
        ([np.interp(hi, f, p)]),
    ))

    n_bad = int(np.count_nonzero(~np.isfinite(p_band)))
    if n_bad:
        warnings.warn(
            f"{n_bad} of {p_band.size} PSD support value(s) in {lo:.4g}–{hi:.4g} Hz are "
            f"non-finite, so "
            f"the band RMS is undefined (NaN). One non-finite sample in the signal poisons "
            f"every Welch bin — check the recording's load issues.",
            RuntimeWarning, stacklevel=2)
        return float("nan")

    var = float(trapezoid(p_band, f_band))
    if not np.isfinite(var):
        warnings.warn(
            f"the PSD integral over {lo:.4g}–{hi:.4g} Hz is not finite; the band RMS is "
            f"undefined (NaN).", RuntimeWarning, stacklevel=2)
        return float("nan")
    if var < 0.0:
        warnings.warn(
            f"the PSD integral over {lo:.4g}–{hi:.4g} Hz is negative ({var:.4g}); this is "
            f"not a power spectral density, so the band RMS is undefined (NaN).",
            RuntimeWarning, stacklevel=2)
        return float("nan")
    return float(np.sqrt(var))


# ================================================================================================
# 1/f (flicker) fit
# ================================================================================================

def fit_flicker(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    f_lo: float,
    f_hi: float,
) -> dict:
    """Fit the low-frequency 1/f (flicker) power law ``PSD ≈ A1Hz · f^(-alpha)``.

    A least-squares straight line through ``(log10 f, log10 PSD)`` over the band
    ``[f_lo, f_hi]`` — the classic log-log flicker fit. The line's slope is
    ``-alpha`` and its 1 Hz intercept is ``A1Hz``, so:

    * ``alpha`` is the flicker exponent, **positive** for a 1/f-like fall-off
      (``alpha ≈ 1`` for ideal flicker; ``0`` for a flat/white band).
    * ``A1Hz`` is the PSD magnitude extrapolated to 1 Hz, in ``(signal-units)²/Hz``.

    The fit itself is delegated to
    :func:`nanoscope.analysis.populations.linear_fit` (``log_log=True``), which
    drops non-finite and non-positive ``(f, PSD)`` pairs before fitting and
    reports ``R²`` in log space — the module is imported lazily so a plain PSD
    computation never pulls in the clustering stack.

    Args:
        freqs: Frequency axis in Hz.
        psd: PSD in ``(signal-units)²/Hz``, aligned to ``freqs``.
        f_lo: Low edge of the flicker band in Hz (clipped to ``> 0``; ``log10``
            needs a positive frequency).
        f_hi: High edge of the flicker band in Hz (clipped to Nyquist).

    Returns:
        ``{"alpha", "A1Hz", "r2", "n", "n_band", "status", "f_lo", "f_hi"}``. When
        the band holds fewer than two usable points or the frequencies are
        degenerate, ``alpha``, ``A1Hz`` and ``r2`` are ``NaN`` and ``n`` is ``0``
        — the fit is reported as unavailable rather than raised, so a caller can
        render it gracefully.

        ``n`` is the number of FFT bins the line was fitted through and ``n_band``
        how many bins lie in the requested band at all (they differ when bins are
        non-positive or non-finite). **``n`` travels with every reported exponent
        by contract**: an α is only as good as the number of bins under it, and
        R² alone cannot say whether that number is 4 or 400. ``status`` is
        :func:`flicker_status` of ``n`` — ``"ok"``, ``"marginal"`` or
        ``"unavailable"`` — so no caller has to re-derive the policy.

        The returned ``f_lo``/``f_hi`` are each edge AFTER its own clip, kept even
        when the fit is unavailable so a caller can explain the refusal. When they
        CROSS (``f_lo >= f_hi``) the clips emptied the band — Nyquist fell below the
        requested low edge, or the request was empty to begin with. A crossed pair
        names the two clips, not a band: no surface may print it as one ("the 1–0.5
        Hz flicker band" is a band that does not exist), and every consumer must
        check ``f_hi > f_lo`` before rendering the pair with a dash between them.
    """
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    psd = np.asarray(psd, dtype=np.float64).ravel()

    nyq = float(np.nanmax(freqs)) if freqs.size else 0.0
    lo = max(float(f_lo), 0.0)
    hi = min(float(f_hi), nyq)
    in_band = ((freqs >= lo) & (freqs <= hi) & (freqs > 0.0)) if freqs.size else np.zeros(0, bool)
    unavailable = {
        "alpha": float("nan"), "A1Hz": float("nan"), "r2": float("nan"),
        "n": 0, "n_band": int(np.count_nonzero(in_band)) if hi > lo else 0,
        "status": flicker_status(0), "f_lo": lo, "f_hi": hi,
    }
    if not (hi > lo):
        return unavailable

    band = in_band & (psd > 0.0) & np.isfinite(psd)
    fb = freqs[band]
    pb = psd[band]
    if fb.size < 2:
        return unavailable

    # Lazy import: the log-log line fit is the documented seam, but a bare PSD
    # computation should not drag in sklearn/pandas via populations.
    from .populations import linear_fit

    try:
        fit = linear_fit(fb, pb, log_log=True)
    except ValueError:
        return unavailable

    # linear_fit(log_log) returns the power law PSD = coefficient * f**exponent,
    # i.e. slope == exponent == -alpha and coefficient == A1Hz (the value at f=1).
    alpha = -float(fit.slope)
    a1hz = float(fit.meta.get("coefficient", 10.0 ** fit.intercept))
    return {
        "alpha": alpha,
        "A1Hz": a1hz,
        "r2": float(fit.r2),
        "n": int(fit.n),
        "n_band": int(fb.size),
        "status": flicker_status(int(fit.n)),
        "f_lo": lo,
        "f_hi": hi,
    }


# ================================================================================================
# whole-spectrum noise summary
# ================================================================================================

def _white_floor(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Median PSD over the geometric-mid decade — a robust thermal-floor proxy.

    The white/thermal plateau sits above the 1/f knee and below the dielectric
    rise, so the middle decade in log-frequency is a defensible place to read it.
    Falls back to the median over all positive-frequency bins if that decade is
    empty. Returns ``(level, (lo, hi))``.
    """
    pos = freqs > 0.0
    f = freqs[pos]
    p = psd[pos]
    finite = np.isfinite(f) & np.isfinite(p) & (p > 0.0)
    f, p = f[finite], p[finite]
    if f.size == 0:
        return float("nan"), (float("nan"), float("nan"))

    fmin, fmax = float(f.min()), float(f.max())
    if fmax <= fmin:
        return float(np.median(p)), (fmin, fmax)

    log_c = 0.5 * (np.log10(fmin) + np.log10(fmax))     # geometric centre
    lo = max(10.0 ** (log_c - 0.5), fmin)
    hi = min(10.0 ** (log_c + 0.5), fmax)
    band = (f >= lo) & (f <= hi)
    if int(np.count_nonzero(band)) == 0:
        return float(np.median(p)), (fmin, fmax)
    return float(np.median(p[band])), (float(lo), float(hi))


def noise_summary(freqs: np.ndarray, psd: np.ndarray, fs: float) -> dict:
    """A whole-spectrum noise report: total + per-band RMS, mid-log proxy, flicker.

    Bundles the quantities a pore-QC read-out shows, all derived from a single
    Welch PSD:

    * ``total_rms`` — :func:`band_rms` over ``[0, Nyquist]``: the total RMS noise
      (``sqrt(variance)``) in signal units.
    * ``bands`` — ``{label: rms}`` over :data:`DEFAULT_BANDS` (``<1 kHz``,
      ``1-10 kHz``, ``10 kHz-Nyquist``), each clipped to Nyquist. A band above
      Nyquist (low ``fs``) collapses to ``0.0`` — it is genuinely empty — so the
      RMS values add in quadrature to ``total_rms`` (within windowing leakage).
      A band that EXISTS but cannot be integrated (non-finite PSD bins) is ``NaN``,
      not ``0.0``: see :func:`band_rms`. Every consumer must therefore render these
      through an is-finite check rather than a bare format specifier.
    * ``white_floor`` — the median PSD over the mid decade, in ``(units)²/Hz``,
      with the decade used in ``white_floor_band``.
    * ``flicker`` — :func:`fit_flicker` over the fixed :data:`FLICKER_BAND_HZ`
      low-frequency band (clipped to ``[first positive bin, Nyquist]``), with the
      band used in ``flicker_band``. Reported unavailable (NaN) when that band holds
      fewer than :data:`FLICKER_MIN_POINTS` bins, so a coarse-resolution estimate
      never yields a spurious exponent. It carries its own ``n`` (bins fitted) and
      ``status`` (:func:`flicker_status`); a consumer that prints ``alpha`` without
      printing ``n`` is printing half the result, and a ``"marginal"`` status must
      reach the reader — an R² over four bins looks exactly like an R² over four
      hundred. The ``flicker_band`` edges are each clip on its own, so they can
      CROSS (``lo >= hi``) when the clip leaves nothing — a 0.5 Hz Nyquist under the
      1 Hz band low edge gives ``(1.0, 0.5)``, a 3815 Hz first bin over the 100 Hz
      top gives ``(3815, 100)``. A crossed pair means "no flicker band exists at
      this fs/resolution" and must never be rendered as a real band (see
      :func:`fit_flicker`).

    Args:
        freqs: Frequency axis in Hz from :func:`welch_psd`.
        psd: PSD in ``(signal-units)²/Hz``, aligned to ``freqs``.
        fs: Sample rate in Hz (used only to report Nyquist; the integration reads
            Nyquist off ``freqs``).

    Returns:
        A dict with ``total_rms``, ``bands``, ``white_floor``,
        ``white_floor_band``, ``flicker``, ``flicker_band``, ``nyquist_hz`` and
        ``df_hz`` (the frequency resolution).
    """
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    psd = np.asarray(psd, dtype=np.float64).ravel()

    nyq = float(np.nanmax(freqs)) if freqs.size else 0.0
    total_rms = band_rms(freqs, psd, 0.0, nyq)

    bands: dict[str, float] = {}
    for label, lo, hi in DEFAULT_BANDS:
        bands[label] = band_rms(freqs, psd, lo, hi)

    white_floor, wf_band = _white_floor(freqs, psd)

    # Frequency resolution = smallest positive-frequency step.
    pos = freqs[freqs > 0.0]
    f1 = float(pos.min()) if pos.size else 0.0
    df = f1
    # The flicker fit runs over the fixed FLICKER_BAND_HZ, clipped to the spectrum's
    # own [first positive bin, Nyquist]. Pinning it to an absolute Hz band — rather
    # than the lowest FFT decade [f1, 10*f1] — keeps the exponent stable across Welch
    # segment lengths: a longer nperseg just adds points to the same physical band
    # instead of collapsing the fit onto a handful of noisy low-resolution bins, which
    # could report a spurious (even negative) alpha the GUI would then draw as "1/f".
    if f1 > 0.0:
        fl_lo, fl_hi = max(FLICKER_BAND_HZ[0], f1), min(FLICKER_BAND_HZ[1], nyq)
    else:
        fl_lo, fl_hi = 0.0, 0.0
    flicker = fit_flicker(freqs, psd, f_lo=fl_lo, f_hi=fl_hi)
    if flicker["status"] == "unavailable":
        # too few low-frequency bins for a trustworthy exponent — report unavailable
        # rather than a slope through a couple of points. ``n``/``n_band`` survive the
        # blanking: the count is how a caller explains WHY there is no exponent, and
        # dropping it here is what left the GUI saying only "too few bins".
        flicker = {**flicker, "alpha": float("nan"),
                   "A1Hz": float("nan"), "r2": float("nan")}

    return {
        "total_rms": total_rms,
        "bands": bands,
        "white_floor": white_floor,
        "white_floor_band": wf_band,
        "flicker": flicker,
        "flicker_band": (fl_lo, fl_hi),
        "nyquist_hz": nyq,
        "df_hz": df,
    }


# ================================================================================================
# SINGLE-SHOT FFT — the unaveraged magnitude spectrum
# ================================================================================================
#
# WHY THIS SITS BESIDE welch_psd AND NOT INSTEAD OF IT.
# The Welch estimate above averages |X(f)|² over many segments, which is what makes it a
# usable NOISE measurement: the variance of a periodogram bin is ~100% of its own value no
# matter how long the record is (more samples buy finer bins, never a smoother curve), and
# averaging K segments divides that variance by K. A single-shot FFT of the whole span is
# the K = 1 case — the noisiest estimator of the family — and it looks MORE informative
# because it is spikier. It is here for the one job Welch cannot do: read a *coherent* line
# (mains at 50/60 Hz and its harmonics, a pump, a switching supply) at the full frequency
# resolution 1/T of the span, where a deterministic tone concentrates into one bin and does
# not average down. Every surface that draws it must say which of the two it is showing.
#
# SCALING: PEAK AMPLITUDE, NOT DENSITY. welch_psd returns a density in units²/Hz;
# fft_magnitude returns |X(f)| in the SIGNAL'S OWN UNITS, normalised so a pure sinusoid of
# amplitude A sitting on an exact bin reads exactly A. That is the number a user can check
# against the trace with a ruler ("the 60 Hz line is 3 pA peak"), and it is why the two
# functions are not interchangeable and are never plotted on the same axis.

#: Default cap on the number of points :func:`fft_magnitude` returns for drawing.
#: A 60 s span at 50 kHz has 1.5 M positive-frequency bins; a plot widget is ~1000 px
#: wide, so 20 000 points is already 20 per pixel. Above the cap the spectrum is reduced
#: to a LOG-SPACED PEAK ENVELOPE (see :func:`fft_magnitude`) rather than decimated:
#: uniform decimation of a 1.5 M-bin spectrum to 20 000 points throws away 74 of every 75
#: bins and a mains line lands in the discarded 74 about 99% of the time.
FFT_MAX_POINTS = 20_000

# The draw cap above reduces only *after* NumPy has built the full transform. Bound the
# real work as well: 16 M float samples already require several simultaneous 128 MB
# arrays (input, taper, product, complex FFT and magnitude). The GUI lets users analyse
# a selected segment, so refusing a larger whole-trace FFT is recoverable and preferable
# to macOS killing the application under memory pressure.
MAX_FFT_INPUT_SAMPLES = 16_000_000


@dataclass(frozen=True)
class FFTSpectrum:
    """One single-shot magnitude spectrum, plus what was done to it to fit a screen.

    ``freqs``/``mag`` are the drawable arrays (DC dropped — a magnitude spectrum is
    read on a log frequency axis and ``log10(0)`` has no place on it). ``n_bins`` is
    how many positive-frequency bins the transform actually produced, so a reader can
    see the difference between "this is every bin" and "this is an envelope of
    ``n_bins`` bins in ``freqs.size`` points".
    """
    freqs: np.ndarray          # Hz, strictly increasing, > 0
    mag: np.ndarray            # signal units, peak-amplitude normalised
    fs: float
    n_samples: int             # samples transformed
    n_bins: int                # positive-frequency bins before any reduction
    df_hz: float               # native bin spacing fs / n_samples
    window: str
    reduced: bool              # True == `mag` is a per-bin peak envelope, not raw bins
    n_nonfinite: int           # non-finite input samples (they poison every bin)

    @property
    def resolution_hz(self) -> float:
        """The frequency resolution of the TRANSFORM (``1/T``), not of the drawn points."""
        return self.df_hz


def fft_magnitude(
    signal: np.ndarray,
    fs: float,
    *,
    window: str = DEFAULT_WINDOW,
    detrend: bool = True,
    max_points: int = FFT_MAX_POINTS,
) -> FFTSpectrum:
    """Single-shot peak-amplitude spectrum ``|X(f)|`` of one span, in signal units.

    The normalisation is ``2 |X_k| / sum(w)`` for every bin except DC and (for an even
    ``n``) Nyquist, which take ``|X_k| / sum(w)``. Dividing by the window's SUM rather
    than by ``n`` is the coherent-gain correction. At the first one or two bins, the
    Blackman-family tapers have overlapping positive- and negative-frequency lobes, so
    ``2 |X_k| / sum(w)`` alone is phase-dependent. Those bins are solved as the exact
    two-equation real-sinusoid system before their magnitude is reported. Thus a
    bin-centred sinusoid reads its own peak amplitude, including immediately beside DC,
    under all seven tapers the GUI offers.

    Args:
        signal: 1-D span to transform (already sliced — this function never sees, and
            must never be handed, more of a recording than the user asked to look at).
        fs: Sample rate in Hz, finite and positive.
        window: Taper name (:func:`scipy.signal.get_window`). ``"boxcar"`` is the
            untapered transform; ``"hann"`` (the default) trades a wider main lobe for
            ~31 dB less leakage, which is what lets a small line be seen next to a big one.
        detrend: Subtract the mean first (default). The DC bin is dropped from the
            output either way; removing the pedestal also removes the leakage skirt a
            100 pA offset spreads across the low decade.
        max_points: Draw-side cap; ``0`` disables it and returns every bin. Above the
            cap the output is a log-spaced PEAK envelope: the frequency axis is split
            into ``max_points`` geometric bins and each keeps its MAXIMUM magnitude, so
            a mains spike survives the reduction (a mean would dilute it by the bin
            width and a decimation would usually miss it). The frequency reported for
            an envelope point is the frequency of the bin the peak was actually in, not
            the reduced bin's centre.

    Returns:
        :class:`FFTSpectrum`.

    Raises:
        ValueError: on fewer than 2 samples or a non-finite / non-positive ``fs``.
    """
    source = np.asarray(signal)
    n = int(source.size)
    if n < 2:
        raise ValueError(f"fft_magnitude needs at least 2 samples; got {n}")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fft_magnitude needs a finite, positive sample rate; got fs={fs!r}")
    if n > MAX_FFT_INPUT_SAMPLES:
        duration = n / float(fs)
        raise ValueError(
            f"the requested FFT has {n:,} samples ({duration:.4g} s), above the safe "
            f"{MAX_FFT_INPUT_SAMPLES:,}-sample working limit. Select a shorter Segment "
            "or Visible view; use Welch PSD when a whole-recording noise estimate is "
            "the goal."
        )
    x = np.asarray(source, dtype=np.float64).ravel()

    # One non-finite sample makes EVERY bin NaN (the transform is a full-support sum), the
    # same failure welch_psd warns about. Count it here so the caller can say so on screen
    # instead of drawing an empty axis and blaming the log toggle.
    n_bad = int(np.count_nonzero(~np.isfinite(x)))

    w = _sp_signal.get_window(window, n, fftbins=True)
    if detrend and n_bad == 0:
        x = x - x.mean()
    # ``inf * 0`` at a taper endpoint is expected to be invalid once a poisoned span
    # has already been counted; suppress that redundant NumPy warning and carry the
    # all-non-finite spectrum to the caller's explicit health message.
    with np.errstate(invalid="ignore"):
        spec = np.fft.rfft(x * w)
    scale = 2.0 / float(np.sum(w))
    mag = np.abs(spec) * scale
    _correct_low_bin_window_overlap(spec, mag, w, str(window))
    if mag.size:
        mag[0] *= 0.5                       # DC is not a two-sided pair
        if n % 2 == 0:
            mag[-1] *= 0.5                  # neither is the Nyquist bin of an even-n rfft
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    freqs, mag = drop_dc(freqs, mag)        # log frequency axis: DC has no place on it
    n_bins = int(freqs.size)
    reduced = bool(max_points and n_bins > int(max_points))
    if reduced:
        freqs, mag = _log_peak_envelope(freqs, mag, int(max_points))

    return FFTSpectrum(
        freqs=freqs, mag=mag, fs=float(fs), n_samples=int(n), n_bins=n_bins,
        df_hz=float(fs) / float(n), window=str(window), reduced=reduced,
        n_nonfinite=n_bad,
    )


def _correct_low_bin_window_overlap(
    spectrum: np.ndarray,
    magnitude: np.ndarray,
    taper: np.ndarray,
    window: str,
) -> None:
    """Remove near-DC mirror-lobe bias for the GUI's Blackman-family tapers.

    For a real bin-centred sinusoid, with complex positive-frequency amplitude ``a``,
    the windowed coefficient at bin ``k`` is::

        Y[k] = W[0] a + W[2k] conj(a)

    The familiar coherent-gain formula assumes ``W[2k] == 0``. That is true away from
    DC, and for Hann/Hamming/Bartlett/boxcar at every positive bin, but it is false in
    the first lobe of Blackman, Blackman-Harris and flattop. Solving the equation and
    its conjugate recovers ``2*abs(a)`` without phase bias. Only bins known to overlap
    for the periodic SciPy windows exposed by the UI are touched; the remainder retain
    the conventional, directly comparable coherent-gain spectrum.
    """
    correction_bins = {
        "blackman": 1,
        "blackmanharris": 1,
        "flattop": 2,
    }.get(window.lower(), 0)
    if correction_bins == 0 or spectrum.size <= 1:
        return

    n = int(taper.size)
    w0 = float(np.sum(taper))
    sample = np.arange(n, dtype=np.float64)
    last = min(correction_bins, spectrum.size - 1)
    for k in range(1, last + 1):
        # Directly evaluate only W[2] (and W[4] for flattop). This O(n) correction
        # avoids a second full FFT of the taper on million-sample recordings.
        phase = np.exp((-2j * np.pi * (2 * k) / n) * sample)
        w2k = complex(np.dot(taper, phase))
        determinant = w0 * w0 - abs(w2k) ** 2
        if determinant <= np.finfo(np.float64).eps * w0 * w0:
            continue
        yk = complex(spectrum[k])
        a = (w0 * yk - w2k * np.conjugate(yk)) / determinant
        magnitude[k] = 2.0 * abs(a)


def _log_peak_envelope(freqs: np.ndarray, values: np.ndarray,
                       n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an increasing spectrum to <= ``n_points`` log-spaced per-bin MAXIMA.

    Peak-preserving on purpose: the reason to look at an unaveraged spectrum at all is
    to find narrow lines, so the reduction that fits it on a screen must be the one that
    cannot lose them. Empty geometric bins are dropped (they contribute no point), which
    is why the result is "<= n_points" and why the low decade — where a geometric bin is
    narrower than the native ``df`` — passes through at full resolution, one bin per point.
    """
    f = np.asarray(freqs, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if f.size == 0 or n_points <= 0 or f.size <= n_points:
        return f, v
    edges = np.geomspace(f[0], f[-1], int(n_points) + 1)
    starts = np.searchsorted(f, edges[:-1], side="left")
    keep = np.empty(starts.shape, dtype=bool)
    keep[0] = True
    keep[1:] = starts[1:] > starts[:-1]      # drop geometric bins holding no native bin
    starts_u = starts[keep]
    # ``np.maximum`` propagates NaN. More importantly, the old peak-index search then
    # found no value equal to that NaN and returned its sentinel ``v.size`` as an index,
    # crashing only when a poisoned FFT was large enough to need screen reduction. Keep
    # a NaN for an all-invalid bin, but perform the reduction and index search on a safe
    # ``-inf`` projection so the UI can report ``n_nonfinite`` instead of crashing.
    finite = np.isfinite(v)
    safe = np.where(finite, v, -np.inf)
    out_safe = np.maximum.reduceat(safe, starts_u)
    valid_counts = np.add.reduceat(finite.astype(np.int64), starts_u)
    out_v = np.where(valid_counts > 0, out_safe, np.nan)
    # The frequency reported is the peak's OWN, not the bin's centre. A geometric centre is
    # off by up to half a bin, which at a screen-sized 400-point envelope is 0.55% — 59.67 Hz
    # printed for a 60 Hz line, measured, and telling 50 Hz mains from 60 Hz is most of why
    # anyone opens this view. Three extra passes over the positive bins buy the exact answer.
    counts = np.diff(np.append(starts_u, v.size))
    lin = np.arange(v.size)
    at_max = np.where(
        finite & (safe >= np.repeat(out_safe, counts)),
        lin,
        v.size,
    )
    peak_index = np.minimum.reduceat(at_max, starts_u)
    # Any frequency is a harmless carrier for an all-invalid envelope bin; choose its
    # first native bin deterministically while leaving the magnitude NaN.
    peak_index[valid_counts == 0] = starts_u[valid_counts == 0]
    out_f = f[peak_index]
    return out_f, out_v


# ================================================================================================
# SPECTROGRAM — the Welch average taken apart in time
# ================================================================================================
#
# WHY A NANOPORE USER NEEDS THIS AND THE WELCH PSD CANNOT SUBSTITUTE.
# welch_psd MEANS the per-segment periodograms. A mains change, a pump switching on, a
# bubble, or a stretch of high-frequency hash that starts 40 s into a 120 s recording is
# still in that mean — diluted by every quiet segment, showing up as a modest lift in one
# decade with nothing on screen saying WHEN. The spectrogram is the same computation with
# the mean not taken: one column per time block. Averaging its columns returns the Welch
# PSD bin for bin, which is the invariant
# :func:`test_the_spectrogram_columns_average_back_to_the_welch_psd` pins — so the two
# views can never quietly disagree about the same recording.

#: The spectrogram's own automatic segment length targets this many TIME COLUMNS, where
#: :data:`TARGET_SEGMENTS` targets 8 AVERAGES. The two autos answer different questions
#: and deliberately differ: a spectrogram with 8 columns has no time axis worth drawing.
#: 400 columns at 50% overlap gives ~800, comfortably more than the ~700 px a plot is wide.
TARGET_COLUMNS = 400

# SciPy materialises the complete (frequency x time) periodogram before our draw-side
# merge. At float64, 25 M cells is 191 MiB for the PSD grid alone, before input, window,
# FFT and rebinned output. Refuse before converting/calling SciPy so the limit protects
# the process rather than documenting a crash after it happens.
MAX_SPECTROGRAM_CELLS = 25_000_000


@dataclass(frozen=True)
class Spectrogram:
    """A PSD per time block: ``psd[i, j]`` is bin ``freqs[i]`` at time ``times[j]``.

    ``psd`` carries the SAME units as :func:`welch_psd` (``units²/Hz``, ``scaling=
    "density"``) because it is the same estimator before the average over ``j`` —
    ``psd.mean(axis=1)`` is the Welch PSD of the span for the same parameters.

    ``times`` are seconds from the START OF THE ANALYSED SPAN, not from the start of
    the recording and not on the instrument clock: the caller owns the offset, exactly
    as it owns the units (see the module header).
    """
    freqs: np.ndarray          # Hz, length n_f
    times: np.ndarray          # s from the span's first sample, length n_t
    psd: np.ndarray            # (n_f, n_t), units²/Hz
    fs: float
    nperseg: int
    noverlap: int
    n_samples: int
    columns_merged: int        # >1 == neighbouring columns were averaged to fit max_columns
    n_nonfinite: int
    source_columns: int = 0    # unmerged SciPy columns represented by the drawn grid

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.freqs.size), int(self.times.size))

    @property
    def time_resolution_s(self) -> float:
        """Approximate signal support contributing to one drawn time column.

        This is deliberately *not* ``block_duration * columns_merged``. Neighbouring
        spectrogram blocks overlap, so pooling ``g`` of them spans one full block plus
        only ``g - 1`` hops. The old product overstated an 8-way, 50%-overlap column by
        almost twofold (163.84 ms instead of 92.16 ms for 1024 samples at 50 kHz).

        ``columns_merged`` is the maximum/nominal group size. Uniform fractional
        rebinning can give edge columns fractional contributions and can touch one more
        source column at a fractional boundary, hence this is a *nominal* support rather
        than a strict upper bound; :attr:`time_step_s` is the exact spacing of the
        uniformly drawn column centres.
        """
        hop = max(self.nperseg - self.noverlap, 1)
        group = max(self.columns_merged, 1)
        return (self.nperseg + (group - 1) * hop) / self.fs

    @property
    def block_duration_s(self) -> float:
        """Support of each original (unmerged) spectral estimate."""
        return self.nperseg / self.fs

    @property
    def hop_s(self) -> float:
        """Spacing of adjacent original block centres."""
        return (self.nperseg - self.noverlap) / self.fs

    @property
    def time_step_s(self) -> float:
        """Exact spacing/width of the uniformly drawn time bins."""
        if self.times.size > 1:
            return float(np.median(np.diff(self.times)))
        represented = self.source_columns if self.source_columns > 0 else 1
        if represented == 1:
            # There is no neighbour from which to infer a hop. Paint the sole raw
            # estimate over the block it actually analysed, not half a block at the
            # usual 50% overlap.
            return self.block_duration_s
        return self.hop_s * represented

    @property
    def analyzed_samples(self) -> int:
        """Samples covered by complete source blocks before display rebinning.

        SciPy does not zero-pad a final partial block. Consequently a requested span can
        end with fewer than one hop of samples that are not represented by any source
        column. This property makes that coverage explicit instead of conflating the
        requested span with the interval actually analysed.
        """
        hop = max(self.nperseg - self.noverlap, 1)
        if self.n_samples < self.nperseg:
            return 0
        return min(
            self.n_samples,
            self.nperseg + ((self.n_samples - self.nperseg) // hop) * hop,
        )

    @property
    def uncovered_tail_samples(self) -> int:
        """Requested tail samples omitted because they do not fill a complete block."""
        return max(0, self.n_samples - self.analyzed_samples)


def _uniform_column_average(
    values: np.ndarray,
    times: np.ndarray,
    n_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average every input column into ``n_out`` equal-width time bins.

    A simple fixed-size ``reduceat`` leaves one short trailing group whenever the source
    count is not divisible by the merge factor. ``ImageItem`` has one affine time scale,
    so that short group was nevertheless painted as a full-width column and its centre
    was misplaced. Treating each source column as a unit-width interval and splitting
    boundary columns fractionally makes every output bin the same width, represents every
    source column exactly once, and preserves the all-column mean.
    """
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n_in = int(times.size)
    n_out = int(n_out)
    if values.ndim != 2 or values.shape[1] != n_in:
        raise ValueError("spectrogram values and time centres must be aligned")
    if not (1 <= n_out <= n_in):
        raise ValueError(f"output column count must be in [1, {n_in}]; got {n_out}")
    if n_out == n_in:
        return values, times

    edges = np.linspace(0.0, float(n_in), n_out + 1, dtype=np.float64)
    width = float(n_in) / float(n_out)
    out = np.empty((values.shape[0], n_out), dtype=np.float64)
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        first = int(np.floor(lo))
        last = min(int(np.ceil(hi)), n_in)
        weights = np.ones(last - first, dtype=np.float64)
        if weights.size == 1:
            weights[0] = hi - lo
        else:
            weights[0] = min(float(first + 1), hi) - lo
            weights[-1] = hi - max(float(last - 1), lo)
        out[:, j] = np.sum(values[:, first:last] * weights, axis=1) / width

    # SciPy centres are an arithmetic progression. Equal intervals in source-column
    # coordinates therefore map to exactly equal time intervals; deriving the centres
    # analytically avoids the tiny alternating jitter a weighted centre can acquire.
    hop = float(times[1] - times[0]) if n_in > 1 else 0.0
    centre_indices = 0.5 * (edges[:-1] + edges[1:]) - 0.5
    out_times = float(times[0]) + centre_indices * hop
    return out, out_times


def default_spectrogram_nperseg(n: int, target_columns: int = TARGET_COLUMNS) -> int:
    """Segment length that splits ``n`` samples into roughly ``target_columns`` blocks.

    The largest power of two ``<= n / target_columns``, clamped to the same
    ``[256, 65536]`` range :func:`default_nperseg` uses and never larger than ``n``.
    Distinct from :func:`default_nperseg` because the two are optimising opposite ends
    of the same trade: Welch wants LONG segments (few, well-averaged), a spectrogram
    wants SHORT ones (many, resolved in time).
    """
    n = int(n)
    if n <= _MIN_NPERSEG:
        return max(n, 1)
    target = max(1, n // max(int(target_columns), 1))
    p = 1 << (target.bit_length() - 1)
    p = min(max(p, _MIN_NPERSEG), _MAX_NPERSEG)
    return min(p, n)


def spectrogram_psd(
    signal: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = DEFAULT_WINDOW,
    detrend: str | bool = "constant",
    max_columns: int = 0,
) -> Spectrogram:
    """Welch's estimator with the time average not taken: PSD per block, in units²/Hz.

    Args:
        signal: The 1-D span (already sliced by the caller).
        fs: Sample rate in Hz.
        nperseg: Block length. ``None`` uses :func:`default_spectrogram_nperseg`.
        noverlap: Overlap in samples; ``None`` = 50%.
        window: Taper name, as :func:`welch_psd`.
        detrend: Per-block detrend, as :func:`welch_psd`. ``"constant"`` by default so a
            slow baseline drift does not paint a bright low-frequency stripe across
            every column.
        max_columns: ``0`` = no cap. Otherwise, when the block layout would produce more
            columns than this, neighbouring columns are AVERAGED into equal-width time
            bins (the largest/nominal merge factor is reported as ``columns_merged``).
            A source column cut by a bin boundary contributes fractionally to both bins.
            Averaging rather than dropping *complete source columns* is the whole point:
            every block SciPy produced contributes to the image. As in standard Welch
            segmentation, fewer than one hop of trailing samples that cannot fill a
            complete block may remain outside those columns; :attr:`Spectrogram.uncovered_tail_samples`
            reports that tail. Equal-width bins also prevent a short trailing group from
            being painted as though it covered a full group.

    Returns:
        :class:`Spectrogram`.

    Raises:
        ValueError: on fewer than 2 samples, a non-finite / non-positive ``fs``, or a
            request whose pre-merge periodogram would exceed the safe working budget.
    """
    source = np.asarray(signal)
    n = int(source.size)
    if n < 2:
        raise ValueError(f"spectrogram_psd needs at least 2 samples; got {n}")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"spectrogram_psd needs a finite, positive sample rate; got fs={fs!r}")

    nps = default_spectrogram_nperseg(n) if nperseg is None else min(int(nperseg), n)
    nps = max(int(nps), 2)
    nov = nps // 2 if noverlap is None else int(np.clip(int(noverlap), 0, nps - 1))
    hop = nps - nov
    source_columns_expected = 1 + (n - nps) // hop
    source_cells = (nps // 2 + 1) * source_columns_expected
    if source_cells > MAX_SPECTROGRAM_CELLS:
        grid_mib = source_cells * np.dtype(np.float64).itemsize / (1024.0**2)
        raise ValueError(
            f"the requested spectrogram would create {source_cells:,} pre-merge PSD "
            f"cells ({grid_mib:.0f} MiB for that grid alone), above the safe "
            f"{MAX_SPECTROGRAM_CELLS:,}-cell working limit. Select a shorter Segment "
            "or Visible view, or use a lower-rate copy. The on-screen column cap cannot "
            "reduce memory because SciPy builds the full grid first."
        )
    x = np.asarray(source, dtype=np.float64).ravel()
    n_bad = int(np.count_nonzero(~np.isfinite(x)))

    freqs, times, sxx = _sp_signal.spectrogram(
        x, fs=float(fs), window=window, nperseg=nps, noverlap=nov,
        detrend=detrend, return_onesided=True, scaling="density", mode="psd")
    freqs = np.asarray(freqs, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    sxx = np.asarray(sxx, dtype=np.float64)

    source_columns = int(times.size)
    merged = 1
    if max_columns and times.size > int(max_columns):
        merged = int(np.ceil(times.size / float(max_columns)))
        n_drawn = int(np.ceil(times.size / float(merged)))
        sxx, times = _uniform_column_average(sxx, times, n_drawn)
        # Equal-width bins are important twice: the affine image now puts every column at
        # its real time, and their ordinary mean remains the source-column mean even when
        # the source count did not divide evenly. Boundary source columns contribute
        # fractionally rather than being dropped or overweighted.

    return Spectrogram(
        freqs=freqs, times=times, psd=sxx, fs=float(fs), nperseg=int(nps),
        noverlap=int(nov), n_samples=int(n), columns_merged=int(merged),
        n_nonfinite=n_bad, source_columns=source_columns,
    )
