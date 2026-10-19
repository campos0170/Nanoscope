"""The Signal Conditioning window's previews: the chain on a padded window of the recording.

Pure and Qt-free. The window's one long-lived worker runs :func:`compute_preview` (and,
only when the user asks, :func:`compute_exact`) off the GUI thread; everything here can
also be called directly.

WINDOW ESTIMATE VERSUS EXACT. A preview conditions only the window on screen, padded on
each side by the chain's longest settling or window length (:func:`preview_pad_samples`).
For the LOCAL stages — notch, median, high-pass, running mean, Savitzky-Golay and the
low-pass — that padded window gives the same samples as conditioning the whole recording
and slicing it. The baseline-drift estimator and normalize use statistics of the WHOLE
recording, so on a window they are an estimate (:func:`window_is_exact`); the exact trace
needs the whole recording (:func:`compute_exact`), which the window runs only on demand.

Only :func:`~nanoscope.analysis.conditioning.conditioned_trace` builds a conditioned
array here, so the preview cannot drift from what detection runs.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from . import filters
from . import psd as psd_mod
from .baseline import ConstantBaselineWarning, collect_baseline_warnings
from .cancel import OperationCancelled, check_cancelled
from .conditioning import conditioned_trace, conditioning_params
from .conditioning_recipes import STAGE_TITLES, chain_signature
from .filter_views import ChainPhase, ChainStep, chain_phase, chain_step
from .modules import notch_freqs_param

__all__ = [
    "MAX_PREVIEW_PAD_SAMPLES",
    "MAX_PREVIEW_WINDOW_SAMPLES",
    "ConditionedTraceMemo",
    "ConditioningSource",
    "NotchMerge",
    "PreviewRequest",
    "PreviewResult",
    "SpectralLine",
    "SpectrumPair",
    "UnitsAfterChain",
    "clamp_window",
    "compute_exact",
    "compute_preview",
    "estimate_stage_titles",
    "find_spectral_lines",
    "line_is_placeable",
    "memo_key",
    "merge_notch_lines",
    "notch_covers",
    "pad_limited_stages",
    "preview_pad_samples",
    "spectrum_pair",
    "stage_pad_samples",
    "units_after_chain",
    "window_for_resolution_s",
    "window_is_exact",
]

#: The longest window a preview conditions (40 s at 50 kHz); a wider view is centred.
MAX_PREVIEW_WINDOW_SAMPLES = 2_000_000
#: The longest pad per side (the notch's own safety limit).
MAX_PREVIEW_PAD_SAMPLES = filters.MAX_NOTCH_PAD_SAMPLES
#: The line finder's low end (below it a "line" is drift, not hum).
LINE_SEARCH_MIN_HZ = 20.0


@dataclass(frozen=True, eq=False)
class ConditioningSource:
    """The selected channel as the previews read it: the calibrated samples (a reference)
    and where they sit on the main trace's clock."""

    doc_id: str
    channel: int
    label: str
    data: np.ndarray
    fs: float | None
    units: str
    #: Instrument-clock seconds at sample 0 (the main trace's x offset); None without a rate.
    x_offset_s: float | None
    cal_gain: float = 1.0
    cal_offset: float = 0.0

    @property
    def n(self) -> int:
        return int(self.data.size)

    @property
    def key(self) -> tuple:
        """What identifies these samples: document, channel, calibration, length."""
        return (self.doc_id, int(self.channel), float(self.cal_gain), float(self.cal_offset),
                self.n)


def _f(p: Mapping[str, Any], key: str, default: float) -> float:
    value = p.get(key)
    return float(default if value is None else value)


def stage_pad_samples(params: Mapping[str, Any] | None, fs: float) -> dict[str, int]:
    """Settling or window length (samples, uncapped) of every enabled local stage.

    notch: its own ring-down (:func:`filters.notch_settle_s`); low-/high-pass: the
    rise-time probe half-length, 40·fs/fc; median / running mean / Savitzky-Golay: two
    windows; baseline drift: two of its longest window (edge effects only — it stays an
    estimate). A stage whose parameters do not resolve contributes nothing: the kernel
    then raises its own error on the padded window.
    """
    p = dict(params or {})
    fs = float(fs)
    out: dict[str, int] = {}
    try:
        if p.get("notch_enabled"):
            settle = filters.notch_settle_s(
                fs, _f(p, "notch_freq_hz", 60.0), q=_f(p, "notch_q", 30.0),
                harmonics=int(p.get("notch_harmonics") or 3), freqs=notch_freqs_param(p))
            out["notch"] = int(math.ceil(settle * fs))
    except (TypeError, ValueError, RuntimeError):
        pass
    for key, cutoff_key, enabled in (("highpass", "hp_cutoff_hz", p.get("hp_enabled")),
                                     ("lowpass", "filter_cutoff_hz",
                                      p.get("filter_cutoff_hz") is not None)):
        if enabled:
            cutoff = _f(p, cutoff_key, 100.0)
            if cutoff > 0 and math.isfinite(cutoff):
                out[key] = filters.rise_time_probe_samples(fs, cutoff)
    if p.get("median_enabled"):
        out["median"] = 2 * max(1, int(round(_f(p, "median_window_s", 1e-4) * fs)))
    if p.get("rmean_enabled"):
        out["rmean"] = 2 * filters.running_mean_samples(fs, _f(p, "rmean_window_s", 5e-4))
    if p.get("savgol_enabled"):
        out["savgol"] = 2 * filters.savgol_window_samples(
            fs, _f(p, "savgol_window_s", 1e-3), int(p.get("savgol_polyorder") or 3))
    if p.get("detrend_enabled"):
        longest = max(_f(p, "detrend_window_s", 0.5), _f(p, "detrend_rough_s", 0.5),
                      _f(p, "detrend_base_s", 0.1))
        out["detrend"] = int(math.ceil(2.0 * longest * fs))
    return out


def _longest_pad(params: Mapping[str, Any] | None, fs: float) -> int:
    pads = stage_pad_samples(params, fs)
    return max(pads.values()) if pads else 0


def preview_pad_samples(params: Mapping[str, Any] | None, fs: float) -> int:
    """Pad per side: the chain's longest settling/window length, capped."""
    return int(min(_longest_pad(params, fs), MAX_PREVIEW_PAD_SAMPLES))


def pad_limited_stages(params: Mapping[str, Any] | None, fs: float
                       ) -> tuple[tuple[str, float], ...]:
    """``(title, seconds)`` of each enabled LOCAL stage that settles longer than the pad
    limit (a very slow high-pass or notch): its window preview is an estimate too."""
    return tuple((STAGE_TITLES.get(key, key), samples / float(fs))
                 for key, samples in stage_pad_samples(params, fs).items()
                 if key != "detrend" and samples > MAX_PREVIEW_PAD_SAMPLES)


def estimate_stage_titles(params: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The enabled stages that use whole-recording statistics (a window is an estimate)."""
    p = params or {}
    return tuple(title for key, title in (("detrend_enabled", "Baseline drift"),
                                          ("normalize_enabled", "Normalize"))
                 if p.get(key))


def window_is_exact(params: Mapping[str, Any] | None, fs: float) -> bool:
    """True when the padded window gives the whole-recording samples: no estimator stage
    and a pad that was not capped."""
    return (not estimate_stage_titles(params)
            and _longest_pad(params, fs) <= MAX_PREVIEW_PAD_SAMPLES)


def clamp_window(start: int, stop: int, n: int, *,
                 max_samples: int = MAX_PREVIEW_WINDOW_SAMPLES) -> tuple[int, int, bool]:
    """``[start, stop)`` inside ``[0, n)``, at most ``max_samples`` around its centre."""
    n = max(0, int(n))
    start = min(max(0, int(start)), n)
    stop = min(max(start, int(stop)), n)
    if stop - start <= max_samples:
        return start, stop, False
    mid = (start + stop) // 2
    a = max(0, min(mid - max_samples // 2, n - max_samples))
    return a, a + max_samples, True


@dataclass(frozen=True)
class UnitsAfterChain:
    """What the processed trace is measured in."""

    #: The processed axis label, e.g. 'processed (pA, DC removed by 100 Hz high-pass)'.
    label: str
    changes_units: bool
    removes_dc: bool
    #: The same unit in fewer words, for an axis too short for ``label``.
    short_label: str = ""
    #: The unit alone (the legend already says "processed"), for the shortest axis.
    unit_label: str = ""


_NORMALIZE_UNITS = {
    "zscore": ("σ, whole-trace z-score", "σ"),
    "robust": ("noise σ, 1.4826·MAD", "noise σ"),
    "minmax": ("fraction of the span, min–max", "fraction of span"),
}


def units_after_chain(params: Mapping[str, Any] | None, units: str) -> UnitsAfterChain:
    """The processed trace's real unit: normalize changes it, high-pass removes DC."""
    p = params or {}
    units = units or "units"
    removes_dc = bool(p.get("hp_enabled"))
    if p.get("normalize_enabled"):
        mode = str(p.get("normalize_mode") or "zscore")
        if mode == filters.MODEL_INPUT_MODE:
            unit = f"model input, clipped at ±{_f(p, 'normalize_clip_sigma', 30.0):g}σ"
            short = "model input"
        else:
            unit, short = _NORMALIZE_UNITS.get(mode, (mode, mode))
        return UnitsAfterChain(f"processed ({unit})", True, removes_dc, f"processed ({short})",
                               short)
    if removes_dc:
        return UnitsAfterChain(
            f"processed ({units}, DC removed by {_f(p, 'hp_cutoff_hz', 100.0):g} Hz high-pass)",
            False, True, f"processed ({units}, no DC)", f"{units}, no DC")
    return UnitsAfterChain(f"processed ({units})", False, False, f"processed ({units})", units)


# ---------------------------------------------------------------------------
# the spectrum and its lines
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SpectrumPair:
    """Welch PSDs of the raw and processed window, on one grid (DC bin dropped)."""

    f_hz: np.ndarray
    raw_psd: np.ndarray
    processed_psd: np.ndarray | None
    nperseg: int
    df_hz: float
    window_s: float
    #: True when normalize was left out of the processed curve (it only rescales).
    normalize_omitted: bool = False


def window_for_resolution_s(fs: float, df_hz: float = 1.0) -> float | None:
    """The shortest window whose default Welch segment resolves ``df_hz`` (None: never,
    because the segment length is capped)."""
    need = 1 << max(0, math.ceil(math.log2(float(fs) / float(df_hz))))
    if need > psd_mod._MAX_NPERSEG:
        return None
    return psd_mod.TARGET_SEGMENTS * need / float(fs)


def spectrum_pair(raw_window: np.ndarray, processed_window: np.ndarray | None, fs: float, *,
                  normalize_omitted: bool = False) -> SpectrumPair:
    """Both PSDs with :func:`psd.welch_psd` at the same ``nperseg``."""
    n = int(np.asarray(raw_window).size)
    nper = psd_mod.default_nperseg(n)
    f, raw = psd_mod.drop_dc(*psd_mod.welch_psd(raw_window, fs, nperseg=nper))
    proc = None
    if processed_window is not None and np.asarray(processed_window).size == n:
        _f2, proc = psd_mod.drop_dc(*psd_mod.welch_psd(processed_window, fs, nperseg=nper))
    return SpectrumPair(f, raw, proc, int(nper), float(fs) / float(nper), n / float(fs),
                        normalize_omitted)


@dataclass(frozen=True)
class SpectralLine:
    """A narrow line in a spectrum. ``freq_hz`` is where a notch would go: the mains
    harmonic itself when ``snapped``, else the measured peak to 0.1 Hz."""

    freq_hz: float
    prominence_db: float
    #: The mains fundamental it is a harmonic of (50 or 60 Hz), or None.
    mains_hz: float | None = None
    harmonic: int = 0
    #: The refined peak frequency as measured (None: ``freq_hz``).
    measured_hz: float | None = None
    #: ``freq_hz`` was moved onto ``mains_hz × harmonic`` (only when Δf ≤ 1 Hz).
    snapped: bool = False

    @property
    def peak_hz(self) -> float:
        return float(self.freq_hz if self.measured_hz is None else self.measured_hz)

    @property
    def text(self) -> str:
        tag = ""
        if self.mains_hz:
            tag = (f" · {'' if self.snapped else 'near '}mains "
                   f"{self.mains_hz:g}×{self.harmonic}")
        return f"{self.freq_hz:.1f} Hz · +{self.prominence_db:.1f} dB{tag}"

#: A mains harmonic is snapped onto only at this resolution or finer: at a coarser Δf a
#: line 5 Hz off a harmonic still lies within one bin of it.
SNAP_MAX_DF_HZ = 1.0
#: The refined peak's uncertainty, in bins (measured ≤ 0.03 on Welch/Hann spectra).
PEAK_UNCERTAINTY_BINS = 0.1


def _running_floor_db(f: np.ndarray, db: np.ndarray, df: float, floor_octaves: float,
                      min_bins: int) -> np.ndarray:
    """A running median of ``db`` about ``floor_octaves`` wide (and never narrower than
    ``min_bins`` bins), evaluated one octave band at a time so its width in Hz grows
    with frequency."""
    floor = np.empty_like(db)
    k = 2.0 ** (floor_octaves / 2.0) - 2.0 ** (-floor_octaves / 2.0)
    edge = float(f[0])
    while edge <= f[-1]:
        idx = np.flatnonzero((f >= edge) & (f < 2.0 * edge))
        if idx.size:
            centre = edge * math.sqrt(2.0)
            width = max(int(min_bins), int(round(centre * k / df)))
            width += 1 - width % 2
            half = width // 2
            a, b = max(0, int(idx[0]) - half), min(db.size, int(idx[-1]) + half + 1)
            med = (pd.Series(db[a:b]).rolling(width, center=True, min_periods=1)
                   .median().to_numpy())
            floor[idx] = med[idx - a]
        edge *= 2.0
    return floor


def find_spectral_lines(f_hz: np.ndarray, pxx: np.ndarray, *, f_min_hz: float = 20.0,
                        f_max_hz: float | None = None, min_prominence_db: float = 10.0,
                        floor_octaves: float = 0.25, min_floor_bins: int = 11,
                        max_lines: int = 24, mains_hz: Sequence[float] = (50.0, 60.0),
                        max_harmonic: int = 40) -> tuple[SpectralLine, ...]:
    """The strongest narrow lines: peaks at least ``min_prominence_db`` above a running
    median floor ``floor_octaves`` wide (at least ``min_floor_bins`` bins).

    Deterministic by construction: the floor is a pandas rolling median, never
    ``scipy.ndimage.median_filter`` (which returns different values for identical input
    in this environment). Each peak is refined by a parabola through its three dB bins. A
    refined peak within ``max(0.5 Hz, its own uncertainty)`` of a mains harmonic is
    tagged (when it fits both 50 and 60 Hz, the fundamental that explains more of the
    lines wins) and, only when Δf ≤ :data:`SNAP_MAX_DF_HZ`, snapped onto it. The
    ``max_lines`` most prominent are returned, sorted by frequency.
    """
    f = np.asarray(f_hz, dtype=np.float64).ravel()
    p = np.asarray(pxx, dtype=np.float64).ravel()
    keep = (f > 0) & np.isfinite(p) & (p > 0)
    f, p = f[keep], p[keep]
    if f.size < 8:
        return ()
    df = float(np.median(np.diff(f)))
    db = 10.0 * np.log10(p)
    prominence = db - _running_floor_db(f, db, df, float(floor_octaves), int(min_floor_bins))
    hi = float(f[-1] if f_max_hz is None else f_max_hz)
    peaks, _ = signal.find_peaks(prominence, height=float(min_prominence_db))
    peaks = [int(k) for k in peaks if f_min_hz <= f[k] <= hi]
    peaks = sorted(peaks, key=lambda k: (-prominence[k], f[k]))[: int(max_lines)]
    tolerance = max(0.5, PEAK_UNCERTAINTY_BINS * df)
    snap = df <= SNAP_MAX_DF_HZ
    found: list[tuple[float, float, list[tuple[float, int]]]] = []
    for k in peaks:
        fk = float(f[k])
        if 0 < k < f.size - 1:
            y0, y1, y2 = db[k - 1], db[k], db[k + 1]
            den = y0 - 2.0 * y1 + y2
            if den < 0:
                fk += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * df
        tags = []
        for f0 in mains_hz:
            h = int(round(fk / float(f0)))
            if 1 <= h <= max_harmonic and abs(fk - h * float(f0)) <= tolerance:
                tags.append((float(f0), h))
        found.append((fk, float(prominence[k]), tags))
    votes = {f0: sum(1 for *_x, tags in found if any(t[0] == f0 for t in tags))
             for f0 in mains_hz}
    lines = []
    for fk, prom, tags in found:
        if tags:
            f0, h = max(tags, key=lambda t: (votes[t[0]], -t[1]))
            lines.append(SpectralLine(f0 * h if snap else round(fk, 1), prom, f0, h,
                                      measured_hz=fk, snapped=snap))
        else:
            lines.append(SpectralLine(round(fk, 1), prom, measured_hz=fk))
    return tuple(sorted(lines, key=lambda line: line.freq_hz))


def line_is_placeable(line: SpectralLine, df_hz: float, q: float) -> bool:
    """A notch can be placed on ``line``: it was snapped onto a mains harmonic (at a fine
    enough Δf), or the resolution is finer than the notch's own width ``f/Q`` — which
    applies to a mains-tagged line too while Δf is coarse."""
    return line.snapped or (q > 0 and df_hz <= line.freq_hz / q)


def notch_covers(freq_hz: float, notch_hz: float, q: float) -> bool:
    """A notch at ``notch_hz`` (quality ``q``, run forward and backward as applied) takes
    at least 20 dB off a line at ``freq_hz``: within a third of its half-bandwidth.

    The applied amplitude response near the notch is δ²/(δ² + h²), h = f0/(2Q); it is
    ≤ 0.1 for |δ| ≤ h/3. The 0.05 Hz floor is half the 0.1 Hz the lines are rounded to."""
    if not (q > 0 and notch_hz > 0):
        return abs(float(freq_hz) - float(notch_hz)) <= 0.05
    return abs(float(freq_hz) - float(notch_hz)) <= max(0.05, float(notch_hz) / (6.0 * q))


@dataclass(frozen=True)
class NotchMerge:
    """The outcome of adding spectrum lines to the notch list."""

    lines: list[float] | None        # None: refused
    added: tuple[float, ...] = ()
    skipped: tuple[float, ...] = ()
    refusal: str = ""


def merge_notch_lines(existing: Sequence[float] | None, candidates: Sequence[float], *,
                      q: float, max_lines: int) -> NotchMerge:
    """Append ``candidates`` (rounded to 0.1 Hz) to ``existing``, skipping any a line
    already there removes (:func:`notch_covers`); refuse past ``max_lines``."""
    lines = [float(f) for f in (existing or [])]
    added: list[float] = []
    skipped: list[float] = []
    for raw in candidates:
        f = round(float(raw), 1)
        if not (f > 0 and math.isfinite(f)):
            continue
        if any(notch_covers(f, g, float(q)) for g in lines):
            skipped.append(f)
            continue
        lines.append(f)
        added.append(f)
    if len(lines) > int(max_lines):
        return NotchMerge(None, refusal=(
            f"Refused: the notch takes at most {int(max_lines)} lines (this would make "
            f"{len(lines)}) — nothing was added."))
    return NotchMerge(lines, tuple(added), tuple(skipped))


# ---------------------------------------------------------------------------
# the request, the result, the computation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PreviewRequest:
    """One preview, frozen on the GUI thread. ``raw`` is a REFERENCE (Session arrays are
    never written in place)."""

    token: int
    kind: str                                   # "window" | "exact"
    params: dict[str, Any]
    fs: float | None
    raw: np.ndarray | None = None
    start: int = 0
    stop: int = 0
    #: The chain signature; a request for another chain supersedes this one.
    signature: str = ""
    source_key: tuple | None = None
    #: An exact whole-recording ``(array, kernel warnings)`` to slice instead of conditioning.
    memo: tuple[np.ndarray, tuple[Warning, ...]] | None = None
    want_time: bool = True
    want_spectrum: bool = False
    response_view: str | None = None            # "phase" | "group_delay" | "step"
    response_stages: tuple[dict[str, Any], ...] = ()


@dataclass(eq=False)
class PreviewResult:
    """What a preview computed. Every array is a window of the recording."""

    token: int
    kind: str
    start: int = 0
    stop: int = 0
    pad: int = 0
    raw_window: np.ndarray | None = None
    processed_window: np.ndarray | None = None
    #: The processed window equals the whole-recording slice.
    exact: bool = False
    #: True when this window was sliced from a whole-recording (exact) pass.
    from_exact_pass: bool = False
    estimate_stages: tuple[str, ...] = ()
    elapsed_s: float = 0.0
    warnings: tuple[str, ...] = ()
    #: The kernel's own words when a stage degenerated (ConstantBaselineWarning).
    degenerate: str = ""
    error: str = ""
    #: The error/degenerate stage needed more samples than the padded window holds; the
    #: whole recording may be fine (Compute exact answers it).
    window_too_short: bool = False
    spectrum: SpectrumPair | None = None
    lines: tuple[SpectralLine, ...] = ()
    #: The band the lines were searched in (up to the low-pass cutoff when it is on).
    lines_band_hz: tuple[float, float] = (0.0, 0.0)
    phase: ChainPhase | None = None
    step: ChainStep | None = None
    response_error: str = ""
    full: tuple[np.ndarray, tuple[Warning, ...]] | None = None
    #: The request's chain signature and source, so a caller can tell what this is about.
    signature: str = ""
    source_key: tuple | None = None


def _condition(raw: np.ndarray, fs: float, params: Mapping[str, Any]
               ) -> tuple[np.ndarray | None, tuple[Warning, ...], str]:
    """``(array, kernel warnings, error)`` from :func:`conditioned_trace`."""
    try:
        with collect_baseline_warnings() as kernel_warns:
            out = np.asarray(conditioned_trace(raw, fs, params))
    except OperationCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - every kernel complaint becomes text
        return None, (), str(exc)
    return out, tuple(kernel_warns), ""


def _degenerate(warns: Sequence[Warning]) -> str:
    """The kernel's own words when a stage degenerated to a no-op (the main trace's rule)."""
    return " ".join(str(w) for w in warns if isinstance(w, ConstantBaselineWarning))


def compute_exact(req: PreviewRequest) -> PreviewResult:
    """The whole recording through the chain: the exact trace, with its warnings."""
    t0 = time.perf_counter()
    res = PreviewResult(req.token, "exact", 0, 0 if req.raw is None else int(req.raw.size),
                        signature=req.signature, source_key=req.source_key)
    if req.raw is None or not req.fs:
        res.error = "no recording with a known sample rate"
        return res
    arr, warns, error = _condition(req.raw, float(req.fs), conditioning_params(req.params))
    res.warnings = tuple(str(w) for w in warns)
    res.degenerate, res.error = _degenerate(warns), error
    if arr is not None:
        res.full = (arr, warns)
    res.exact = res.from_exact_pass = arr is not None
    res.elapsed_s = time.perf_counter() - t0
    return res


def compute_preview(req: PreviewRequest) -> PreviewResult:
    """The window of ``req`` through the chain, plus the products the request asks for.

    Never raises for a chain error (it is reported in ``.error``); raises
    :class:`~nanoscope.analysis.cancel.OperationCancelled` when the ambient cancel token
    is set.
    """
    t0 = time.perf_counter()
    params = conditioning_params(req.params)
    res = PreviewResult(req.token, "window", int(req.start), int(req.stop),
                        signature=req.signature, source_key=req.source_key)
    fs = float(req.fs) if req.fs else None
    have_data = req.raw is not None and fs is not None and (req.want_time or req.want_spectrum)
    if have_data:
        raw = req.raw
        n = int(raw.size)
        start, stop, _ = clamp_window(req.start, req.stop, n, max_samples=n)
        res.start, res.stop = start, stop
        res.raw_window = np.asarray(raw[start:stop])
        res.estimate_stages = estimate_stage_titles(params)
        if req.memo is not None:
            full, warns = req.memo
            res.warnings = tuple(str(w) for w in warns)
            res.degenerate = _degenerate(warns)
            if not res.degenerate:
                res.processed_window = np.asarray(full[start:stop])
            res.exact = res.from_exact_pass = True
        else:
            pad = preview_pad_samples(params, fs)
            a, b = max(0, start - pad), min(n, stop + pad)
            res.pad = pad
            arr, warns, error = _condition(raw[a:b], fs, params)
            res.warnings = tuple(str(w) for w in warns)
            res.degenerate, res.error = _degenerate(warns), error
            whole = a == 0 and b == n
            res.exact = whole or window_is_exact(params, fs)
            if arr is not None and not res.degenerate:
                res.processed_window = arr[start - a: stop - a]
            # Only the two complaints a SHORT input causes; any other error is the chain's.
            short = res.degenerate or "longer than the recording" in error
            if short and not whole and params.get("detrend_enabled"):
                res.window_too_short = True
        check_cancelled()
        if req.want_spectrum and res.raw_window is not None and res.raw_window.size >= 16:
            proc = res.processed_window
            omitted = bool(params.get("normalize_enabled"))
            if omitted:
                # Normalize only rescales: the "after" curve leaves it out, so both PSDs
                # share one unit.
                bare = dict(params, normalize_enabled=False)
                pad = preview_pad_samples(bare, fs)
                a, b = max(0, start - pad), min(n, stop + pad)
                arr, warns, _error = _condition(raw[a:b], fs, bare)
                proc = None if arr is None or _degenerate(warns) else arr[start - a: stop - a]
            res.spectrum = spectrum_pair(res.raw_window, proc, fs, normalize_omitted=omitted)
            check_cancelled()
            # Above an enabled low-pass a line is already gone; below it is what a notch is for.
            cutoff = params.get("filter_cutoff_hz")
            res.lines_band_hz = (LINE_SEARCH_MIN_HZ,
                                 float(cutoff) if cutoff is not None else fs / 2.0)
            res.lines = find_spectral_lines(res.spectrum.f_hz, res.spectrum.raw_psd,
                                            f_min_hz=LINE_SEARCH_MIN_HZ,
                                            f_max_hz=res.lines_band_hz[1])
    if req.response_view and fs is not None:
        check_cancelled()
        try:
            stages = [dict(s) for s in req.response_stages]
            if req.response_view == "step":
                res.step = chain_step(stages, fs)
            else:
                res.phase = chain_phase(stages, fs)
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            res.response_error = str(exc)
    res.elapsed_s = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# the memo of exact passes
# ---------------------------------------------------------------------------
def memo_key(source_key: tuple, params: Mapping[str, Any], fs: float | None) -> tuple:
    """``source_key`` (doc, channel, calibration, length) plus the chain's signature."""
    return (*tuple(source_key), chain_signature(conditioning_params(params), fs))


class ConditionedTraceMemo:
    """The last exact whole-recording passes, ``key -> (array, warnings)``. LRU; GUI thread
    only. The warnings travel with the array, so a slice of a degenerate pass is still
    reported as degenerate."""

    def __init__(self, capacity: int = 2) -> None:
        self._capacity = max(1, int(capacity))
        self._items: OrderedDict[tuple, tuple[np.ndarray, tuple[Warning, ...]]] = OrderedDict()

    def get(self, key: tuple) -> tuple[np.ndarray, tuple[Warning, ...]] | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: tuple, array: np.ndarray, warnings: Sequence[Warning] = ()) -> None:
        self._items[key] = (array, tuple(warnings))
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> list[tuple]:
        return list(self._items)

    def drop_doc(self, doc_id: str) -> None:
        for key in [k for k in self._items if k and k[0] == doc_id]:
            del self._items[key]

    def clear(self) -> None:
        self._items.clear()
