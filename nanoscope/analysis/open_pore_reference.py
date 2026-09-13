"""User-asserted open-pore references, not automatic open-pore identification.

Times are recording-relative seconds; currents retain their signed pA values.
The constructions never infer polarity, remove events, or alter the signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np

OPEN_PORE_REFERENCE_PARAM_KEY = "baseline_open_pore_reference"
MIN_REFERENCE_SAMPLES = 5


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class OpenPoreReference:
    """Versioned supplied-current or selected-interval reference recipe.

    ``constant`` uses the supplied signed current at every sample. ``intervals``
    pools selected samples for a constant median, or linearly interpolates each
    interval's median at its sampled time centre. Linear mode holds endpoint
    levels outside those centres; it does not extrapolate a drift slope.
    No construction verifies that selected levels really are open pore.
    """

    mode: str
    current_pA: float | None = None
    intervals_s: tuple[tuple[float, float], ...] = ()
    interpolation: str = "constant"
    notes: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or self.version != 1:
            raise ValueError("unsupported open-pore reference version; expected 1")
        if self.mode not in ("constant", "intervals"):
            raise ValueError("open-pore reference mode must be 'constant' or 'intervals'")
        if self.interpolation not in ("constant", "linear"):
            raise ValueError("reference interpolation must be 'constant' or 'linear'")
        if not isinstance(self.notes, str):
            raise ValueError("reference notes must be text")
        try:
            intervals = tuple(self.intervals_s)
        except TypeError as exc:
            raise ValueError("reference intervals must be start/end pairs") from exc
        canonical = []
        for interval in intervals:
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise ValueError("each reference interval must have a start and end")
            start = _number(interval[0], "reference interval start")
            end = _number(interval[1], "reference interval end")
            if start < 0 or end <= start:
                raise ValueError("reference intervals must satisfy 0 <= start < end")
            if canonical and start < canonical[-1][1]:
                raise ValueError("reference intervals must be ordered and non-overlapping")
            canonical.append((start, end))
        object.__setattr__(self, "intervals_s", tuple(canonical))
        if self.mode == "constant":
            object.__setattr__(self, "current_pA", _number(self.current_pA, "open-pore current (pA)"))
            if canonical or self.interpolation != "constant":
                raise ValueError("supplied-current mode cannot also use intervals or linear interpolation")
        else:
            if self.current_pA is not None:
                raise ValueError("selected-interval mode cannot also specify a supplied current")
            if not canonical:
                raise ValueError("select at least one open-pore interval")
            if self.interpolation == "linear" and len(canonical) < 2:
                raise ValueError("linear reference interpolation requires at least two intervals")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "OpenPoreReference") -> "OpenPoreReference":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("open-pore reference must be a recipe object")
        allowed = {"version", "mode", "current_pA", "intervals_s", "interpolation", "notes"}
        if set(value) - allowed:
            raise ValueError("unknown open-pore reference field(s): " + ", ".join(sorted(set(value) - allowed)))
        if "mode" not in value:
            raise ValueError("open-pore reference recipe requires a mode")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "mode": self.mode, "current_pA": self.current_pA,
            "intervals_s": [list(pair) for pair in self.intervals_s],
            "interpolation": self.interpolation, "notes": self.notes,
        }

    def sample_intervals(self, n_samples: int, fs: float) -> tuple[tuple[int, int], ...]:
        """Validate against this recording; return half-open sample spans.

        Five samples is a numerical minimum, not evidence of scientific adequacy.
        """
        fs = _number(fs, "sample rate (Hz)")
        if fs <= 0 or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("reference requires a nonempty recording and positive sample rate")
        duration = int(n_samples) / fs
        spans = []
        for start, end in self.intervals_s:
            if end > duration:
                raise ValueError(f"reference interval {start:g}–{end:g} s is outside this recording (0–{duration:g} s)")
            # Avoid losing an exact sample boundary to decimal multiplication.
            a = max(0, int(np.ceil(np.nextafter(start * fs, -np.inf))))
            b = min(int(n_samples), int(np.ceil(np.nextafter(end * fs, -np.inf))))
            if b - a < MIN_REFERENCE_SAMPLES:
                raise ValueError(f"each reference interval needs at least {MIN_REFERENCE_SAMPLES} samples; {start:g}–{end:g} s has {b - a}")
            if spans and a < spans[-1][1]:
                raise ValueError("reference intervals overlap on this recording's sample grid")
            spans.append((a, b))
        return tuple(spans)

    def describe(self) -> str:
        if self.mode == "constant":
            return f"user-supplied open-pore current, {self.current_pA:g} pA (constant)"
        return f"user-selected open-pore regions, {len(self.intervals_s)} intervals, {self.interpolation} reference"


def build_reference(
    signal: np.ndarray, fs: float, recipe: OpenPoreReference | Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct only from the recipe's evidence; never infer baseline/polarity."""
    recipe = OpenPoreReference.from_dict(recipe)
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("open-pore reference requires a nonempty, finite one-dimensional current trace")
    fs = _number(fs, "sample rate (Hz)")
    spans = recipe.sample_intervals(int(x.size), fs)
    details: dict[str, Any] = {
        "construction": "supplied_constant" if recipe.mode == "constant" else f"selected_intervals_{recipe.interpolation}",
        "user_assertion_not_verified": True,
        "event_polarity_used_for_construction": False,
        "interval_semantics": "recording-relative [start, end) seconds; samples at t=index/fs",
        "noise_policy": "detector's existing whole-residual noise estimate; no noise override",
        "sample_intervals": [list(span) for span in spans],
    }
    assumptions = [
        "This reference is supplied or selected by the user; NanoScope has not verified that it is the true open-pore current. Review the reference against this recording before trusting event depths."
    ]
    if recipe.mode == "constant":
        baseline = np.full(x.size, recipe.current_pA, dtype=np.float64)
        assumptions.append("The supplied signed open-pore current is held constant for the entire recording; drift or changes in pore conductance are not inferred.")
    else:
        levels = np.asarray([np.median(x[a:b]) for a, b in spans], dtype=np.float64)
        centres = np.asarray([(a + b - 1) / (2.0 * fs) for a, b in spans], dtype=np.float64)
        details.update({
            "interval_medians_pA": levels.tolist(), "interval_centres_s": centres.tolist(),
            "selected_samples": sum(b - a for a, b in spans),
            "selected_fraction": sum(b - a for a, b in spans) / x.size,
        })
        if recipe.interpolation == "constant":
            value = float(np.median(np.concatenate([x[a:b] for a, b in spans])))
            baseline = np.full(x.size, value, dtype=np.float64)
            details["pooled_median_pA"] = value
            assumptions.append("Selected samples are pooled with equal per-sample weight for one constant median reference. Unequal interval lengths have unequal weight; this reference does not track drift.")
        else:
            time = np.arange(x.size, dtype=np.float64) / fs
            baseline = np.interp(time, centres, levels)
            outside = (time < centres[0]) | (time > centres[-1])
            details.update({
                "endpoint_policy": "hold first/last interval median outside their sample centres",
                "left_hold_s": float(centres[0]),
                "right_hold_s": float((x.size - 1) / fs - centres[-1]),
                "held_endpoint_fraction": float(np.mean(outside)),
                "largest_centre_gap_s": float(np.max(np.diff(centres))),
            })
            assumptions.append("The reference linearly interpolates interval medians between their sample centres and holds the first/last levels outside them. Drift between selections and in held endpoint regions is assumed, not measured; long gaps have unsupported interpolation uncertainty.")
        assumptions.append("Interval medians resist isolated outliers but can be wrong if selected regions contain blockages, artifacts, or transitions. Five samples is only a numerical minimum, not a scientific adequacy guarantee.")
    if not np.isfinite(baseline).all():
        raise ValueError("reference construction produced nonfinite values")
    if np.any(baseline == 0):
        details["warnings"] = [
            "The user-supplied open-pore reference contains zero current. "
            "Fractional depth (dI/I0) is undefined for events whose reference is "
            "zero; review the reference and signed current before using those ratios."
        ]
    details["assumptions"] = assumptions
    return baseline, details
