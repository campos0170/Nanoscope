"""Linear-response views of the signal chain: phase, group delay and the step response.

Pure and Qt-free. The Signal Conditioning window's Response tab draws these next to the
magnitude curve :func:`nanoscope.analysis.filters.combined_response` already gives.

Every stage NanoScope applies is zero-phase: the IIR stages run forward and backward
(``sosfiltfilt``, so their applied response is the real, non-negative ``|H|²``) and the
smoothers are centred kernels. So the APPLIED phase is 0 except where a smoother's real
response goes negative (a phase of π: a sign flip, not a delay) and the applied group
delay is 0. Each view therefore also gives a REFERENCE: the same IIR design run ONCE,
causally — what a hardware filter of that design does to the same signal.

The median, the baseline-drift estimator and normalize have no transfer function; they
are listed, never drawn (:func:`linear_stage_specs`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage, signal

from . import filters
from .conditioning_recipes import CHAIN_DEFAULTS, STAGE_TITLES

__all__ = [
    "ChainPhase",
    "ChainStep",
    "MAX_STEP_SIDE_SAMPLES",
    "chain_phase",
    "chain_step",
    "linear_stage_specs",
    "reference_sos",
    "step_span_s",
]

#: The omission notes, word for word as the Response tab's |H(f)| note gives them.
_OMITTED = {
    "median": "Median (nonlinear rank filter)",
    "detrend": "Baseline drift (signal-dependent estimator)",
    "normalize": "Normalize (data-dependent affine/nonlinear rescale)",
}

#: Samples on each side of the synthetic step (10 s at 50 kHz).
MAX_STEP_SIDE_SAMPLES = 500_000


def _get(p: Mapping[str, Any], key: str) -> Any:
    value = p.get(key)
    return CHAIN_DEFAULTS[key] if value is None and key != "notch_freqs" else value


def _stage_spec(key: str, p: Mapping[str, Any], lowpass_hz: float | None) -> dict[str, Any] | None:
    if key == "notch":
        freqs = p.get("notch_freqs")
        return {"stage": "notch", "freq_hz": float(_get(p, "notch_freq_hz")),
                "q": float(_get(p, "notch_q")), "harmonics": int(_get(p, "notch_harmonics")),
                "freqs": None if freqs is None else [float(f) for f in freqs]}
    if key == "highpass":
        return {"stage": "highpass", "cutoff_hz": float(_get(p, "hp_cutoff_hz")),
                "order": int(_get(p, "hp_order")), "kind": str(_get(p, "hp_kind"))}
    if key == "rmean":
        return {"stage": "running_mean", "window_s": float(_get(p, "rmean_window_s"))}
    if key == "savgol":
        return {"stage": "savgol", "window_s": float(_get(p, "savgol_window_s")),
                "polyorder": int(_get(p, "savgol_polyorder"))}
    if key == "lowpass":
        cutoff = p.get("filter_cutoff_hz")
        cutoff = lowpass_hz if cutoff is None else cutoff
        if cutoff is None:
            return None
        return {"stage": "lowpass", "cutoff_hz": float(cutoff),
                "order": int(_get(p, "filter_order")), "kind": str(_get(p, "filter_kind"))}
    return None


def linear_stage_specs(params: Mapping[str, Any] | None, *, only: str | None = None,
                       lowpass_design_hz: float | None = None
                       ) -> tuple[list[dict[str, Any]], list[str]]:
    """``(stages, omitted)``: the chain's linear stages as :func:`filters.combined_response`
    mappings, in execution order, and the enabled stages that have no linear response.

    The same split as ``DetectionPanel._active_linear_response_stages`` (the Magnitude
    view), from a params dict. ``only=<stage key>`` gives that one stage's DESIGN whether
    or not it is enabled (an off low-pass uses ``lowpass_design_hz``); a non-linear key
    gives ``([], [its note])``.
    """
    p = dict(params or {})
    if only is not None:
        if only in _OMITTED:
            return [], [_OMITTED[only]]
        spec = _stage_spec(only, p, lowpass_design_hz)
        return ([spec], []) if spec is not None else ([], [])
    stages: list[dict[str, Any]] = []
    omitted: list[str] = []
    enabled = {"notch": "notch_enabled", "median": "median_enabled", "highpass": "hp_enabled",
               "detrend": "detrend_enabled", "rmean": "rmean_enabled",
               "savgol": "savgol_enabled", "normalize": "normalize_enabled"}
    for key in ("notch", "median", "highpass", "detrend", "rmean", "savgol", "normalize"):
        if not p.get(enabled[key]):
            continue
        if key in _OMITTED:
            omitted.append(_OMITTED[key])
        else:
            stages.append(_stage_spec(key, p, None))
    if p.get("filter_cutoff_hz") is not None:
        stages.append(_stage_spec("lowpass", p, None))
    return stages, omitted


def _is_gaussian(spec: Mapping[str, Any]) -> bool:
    return spec["stage"] == "lowpass" and str(spec.get("kind", "bessel")) == "gaussian"


def _iir_sos(spec: Mapping[str, Any], fs: float) -> np.ndarray | None:
    """The SOS ``filters`` designs for one IIR stage (None for the centred kernels)."""
    stage = spec["stage"]
    if stage in ("lowpass", "highpass") and not _is_gaussian(spec):
        default_order, default_kind = (8, "bessel") if stage == "lowpass" else (4, "butter")
        return filters._design_sos(
            float(fs), float(spec["cutoff_hz"]), int(spec.get("order", default_order)),
            str(spec.get("kind", default_kind)), btype="low" if stage == "lowpass" else "high")
    if stage == "notch":
        lines = filters.notch_frequencies(
            float(fs), float(spec.get("freq_hz") or 60.0),
            harmonics=int(spec.get("harmonics", 3)), freqs=spec.get("freqs"))
        q = float(spec.get("q", 30.0))
        return np.vstack([np.hstack(signal.iirnotch(f, q, fs=fs))[None, :] for f in lines])
    return None


def reference_sos(stages: Sequence[Mapping[str, Any]], fs: float) -> np.ndarray | None:
    """Every IIR stage's single-pass design stacked into one cascade (None if there is none)."""
    parts = [s for s in (_iir_sos(spec, fs) for spec in stages) if s is not None]
    return np.vstack(parts) if parts else None


def _centred_real_response(spec: Mapping[str, Any], f: np.ndarray, fs: float
                           ) -> tuple[np.ndarray, float]:
    """``(real response, shift in samples)`` of a centred smoother as it is applied.

    An odd box or a Savitzky-Golay window is exactly centred (shift 0). An EVEN box —
    ``uniform_filter1d`` with an even size — averages ``[i - N/2, i + N/2 - 1]`` and so
    reads half a sample early; that shift is returned rather than hidden."""
    w = 2.0 * np.pi * np.asarray(f, dtype=np.float64) / float(fs)
    if spec["stage"] == "running_mean":
        n = filters.running_mean_samples(float(fs), float(spec["window_s"]))
        if n == 1:
            return np.ones_like(w), 0.0
        den = n * np.sin(w / 2.0)
        out = np.ones_like(w)
        nz = den != 0.0
        out[nz] = np.sin(n * w[nz] / 2.0) / den[nz]
        return out, (-0.5 if n % 2 == 0 else 0.0)
    if spec["stage"] == "savgol":
        win = filters.savgol_window_samples(float(fs), float(spec["window_s"]),
                                            int(spec.get("polyorder", 3)))
        coeffs = signal.savgol_coeffs(win, int(spec.get("polyorder", 3)), use="conv")
        _, h = signal.freqz(coeffs, worN=f, fs=fs)
        return np.real(h * np.exp(1j * w * (win - 1) / 2.0)), 0.0
    return np.ones_like(w), 0.0        # the Gaussian: real and positive


@dataclass(frozen=True, eq=False)
class ChainPhase:
    """Phase and group delay of a linear cascade, applied (zero-phase) and reference."""

    f_hz: np.ndarray
    applied_phase_rad: np.ndarray
    applied_group_delay_s: np.ndarray
    reference_phase_rad: np.ndarray
    reference_group_delay_s: np.ndarray
    #: True when at least one stage is an IIR design (the reference differs from 0).
    has_reference: bool


def chain_phase(stages: Sequence[Mapping[str, Any]], fs: float, *, n: int = 2048,
                f_min_hz: float | None = None) -> ChainPhase:
    """Phase and group delay of ``stages`` on :func:`filters.combined_response`'s grid.

    Validation is :func:`filters.combined_response`'s (it raises the same errors), so
    this view and the magnitude view refuse the same chains and share one grid.

    * applied phase: π where the product of the centred real responses is negative (a
      smoother's side lobe), else 0, minus any even box's half-sample shift; applied
      group delay: that shift (0 otherwise). IIR stages contribute ``|H|²`` ≥ 0.
    * reference: the single-pass causal response of the IIR designs (``sosfreqz``),
      unwrapped; group delay summed per section with :func:`scipy.signal.group_delay`.
      The centred kernels add nothing to it.
    """
    f, _mag = filters.combined_response(stages, fs, n=n, f_min_hz=f_min_hz)
    w = 2.0 * np.pi * f / float(fs)
    sign = np.ones_like(f)
    shift = 0.0
    for spec in stages:
        if spec["stage"] in ("running_mean", "savgol"):
            real, s = _centred_real_response(spec, f, fs)
            sign *= np.sign(real)
            shift += s
    applied = np.where(sign < 0, np.pi, 0.0) + w * shift
    applied = np.angle(np.exp(1j * applied))
    applied_gd = np.full(f.shape, -shift / float(fs))
    sos = reference_sos(stages, fs)
    if sos is None:
        ref_phase = np.zeros_like(f)
        ref_gd = np.zeros_like(f)
    else:
        _, h = signal.sosfreqz(sos, worN=f, fs=fs)
        ref_phase = np.unwrap(np.angle(h))
        ref_gd = _sos_group_delay(sos, w) / float(fs)
    return ChainPhase(f, applied, applied_gd, ref_phase, ref_gd, sos is not None)


def _sos_group_delay(sos: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Group delay in samples, per biquad in closed form: ``Re(z·P'/P)`` of numerator minus
    denominator, summed.

    Not :func:`scipy.signal.group_delay`: in this environment it returns different values
    for identical input (measured up to 0.04 samples between calls, through its complex
    ``polyval``). A notch zero ON the unit circle gives NaN at exactly that frequency."""
    z1 = np.exp(-1j * np.asarray(w, dtype=np.float64))
    z2 = z1 * z1
    out = np.zeros(z1.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for b0, b1, b2, a0, a1, a2 in np.asarray(sos, dtype=np.float64):
            num = b0 + b1 * z1 + b2 * z2
            den = a0 + a1 * z1 + a2 * z2
            out += np.real((b1 * z1 + 2.0 * b2 * z2) / num) - np.real(
                (a1 * z1 + 2.0 * a2 * z2) / den)
    out[~np.isfinite(out)] = np.nan
    return out


def step_span_s(stages: Sequence[Mapping[str, Any]], fs: float) -> float:
    """Half-width (s) of the step view: set by the edge-shaping stages (low-pass and the
    smoothers) when there are any, else by the slow ones (high-pass droop, notch ring)."""
    fast: list[float] = []
    slow: list[float] = []
    for spec in stages:
        stage = spec["stage"]
        if stage == "lowpass":
            fast.append(4.0 / float(spec["cutoff_hz"]))
        elif stage in ("running_mean", "savgol"):
            fast.append(2.0 * float(spec["window_s"]))
        elif stage == "highpass":
            slow.append(1.0 / float(spec["cutoff_hz"]))
        elif stage == "notch":
            lines = filters.notch_frequencies(
                float(fs), float(spec.get("freq_hz") or 60.0),
                harmonics=int(spec.get("harmonics", 3)), freqs=spec.get("freqs"))
            slow.append(2.0 * float(spec.get("q", 30.0)) / (np.pi * min(lines)))
    chosen = fast or slow
    span = max(chosen) if chosen else 1e-3
    return float(max(span, 16.0 / float(fs)))


def _apply_stage(spec: Mapping[str, Any], x: np.ndarray, fs: float) -> np.ndarray:
    """One linear stage exactly as the kernels apply it, in float64."""
    stage = spec["stage"]
    if _is_gaussian(spec):
        sigma = filters._gaussian_sigma_samples(float(fs), float(spec["cutoff_hz"]))
        return ndimage.gaussian_filter1d(x, sigma, mode="nearest")
    if stage in ("lowpass", "highpass"):
        return signal.sosfiltfilt(_iir_sos(spec, fs), x)
    if stage == "notch":
        lines = filters.notch_frequencies(
            float(fs), float(spec.get("freq_hz") or 60.0),
            harmonics=int(spec.get("harmonics", 3)), freqs=spec.get("freqs"))
        need = filters.notch_settle_s(float(fs), q=float(spec.get("q", 30.0)), freqs=lines)
        padlen = min(int(np.ceil(need * float(fs))), x.size - 1)
        return signal.sosfiltfilt(_iir_sos(spec, fs), x, padlen=padlen)
    if stage == "running_mean":
        n = filters.running_mean_samples(float(fs), float(spec["window_s"]))
        return x.copy() if n == 1 else ndimage.uniform_filter1d(x, n, mode="nearest")
    win = filters.savgol_window_samples(float(fs), float(spec["window_s"]),
                                        int(spec.get("polyorder", 3)))
    return signal.savgol_filter(x, win, int(spec.get("polyorder", 3)))


def _edge_metrics(t: np.ndarray, y: np.ndarray) -> tuple[float | None, float, float]:
    """``(10–90 % rise in s or None, overshoot, pre-shoot)`` of a unit step at t = 0."""
    above_hi = np.flatnonzero(y >= 0.9)
    above_lo = np.flatnonzero(y >= 0.1)
    rise = None
    if above_hi.size and above_lo.size:
        def cross(idx: np.ndarray, level: float) -> float:
            i = int(idx[0])
            if i == 0:
                return float(t[0])
            y0, y1 = y[i - 1], y[i]
            return float(t[i - 1] + (level - y0) / (y1 - y0) * (t[i] - t[i - 1]))

        rise = cross(above_hi, 0.9) - cross(above_lo, 0.1)
    after = y[t >= 0]
    before = y[t < 0]
    overshoot = float(max(0.0, after.max() - 1.0)) if after.size else 0.0
    preshoot = float(max(0.0, -before.min())) if before.size else 0.0
    return rise, overshoot, preshoot


@dataclass(frozen=True, eq=False)
class ChainStep:
    """A unit step (0 → 1 at t = 0) through the linear cascade."""

    t_s: np.ndarray
    applied: np.ndarray
    reference: np.ndarray | None
    applied_rise_s: float | None
    applied_overshoot: float
    applied_preshoot: float
    reference_rise_s: float | None
    reference_overshoot: float
    #: The high-pass removes DC: the plateau droops back to 0 and rise/overshoot do not apply.
    removes_dc: bool
    truncated: bool


def chain_step(stages: Sequence[Mapping[str, Any]], fs: float, *,
               span_s: float | None = None) -> ChainStep:
    """The step response of ``stages``: applied (zero-phase, as NanoScope filters) and the
    reference (each IIR design run once, causally, after the same centred smoothers).

    The probe is twice the shown span on each side with constant ends, so the kernels'
    own end handling (odd extension, nearest-sample, the notch's settle pad) leaves no
    transient in the shown part. Raises what :func:`filters.combined_response` raises.
    """
    filters.combined_response(stages, fs, n=2)          # the shared validation
    span = float(span_s) if span_s is not None else step_span_s(stages, fs)
    side = int(np.ceil(span * float(fs)))
    truncated = side > MAX_STEP_SIDE_SAMPLES
    side = int(min(max(side, 16), MAX_STEP_SIDE_SAMPLES))
    n = 2 * side
    x = np.concatenate([np.zeros(n), np.ones(n)])
    applied = x
    fir = x
    for spec in stages:
        applied = _apply_stage(spec, applied, fs)
        if _iir_sos(spec, fs) is None:
            fir = _apply_stage(spec, fir, fs)
    sos = reference_sos(stages, fs)
    reference = None if sos is None else signal.sosfilt(sos, fir)
    keep = slice(n - side, n + side)
    t = (np.arange(n - side, n + side, dtype=np.float64) - n) / float(fs)
    applied = applied[keep]
    removes_dc = any(spec["stage"] == "highpass" for spec in stages)
    a_rise, a_over, a_pre = (None, 0.0, 0.0) if removes_dc else _edge_metrics(t, applied)
    r_rise, r_over = None, 0.0
    if reference is not None:
        reference = reference[keep]
        if not removes_dc:
            r_rise, r_over, _ = _edge_metrics(t, reference)
    return ChainStep(t, applied, reference, a_rise, a_over, a_pre, r_rise, r_over,
                     removes_dc, truncated)


def stage_title(spec: Mapping[str, Any]) -> str:
    """'Low-pass' for a ``{'stage': 'lowpass', …}`` mapping."""
    key = {"running_mean": "rmean"}.get(str(spec["stage"]), str(spec["stage"]))
    return STAGE_TITLES.get(key, key)
