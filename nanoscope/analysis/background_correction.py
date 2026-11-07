"""Explicit, non-mutating background comparisons for calibrated current traces.

This is separate from runtime conditioning and from spectral subtraction. Recipes
can describe manual anchors, the existing user open-pore reference, or a second
recording. Every result retains its full-resolution inputs and declared recipe;
calculating or exporting one never changes a recording or a detection.

Anchor times and region bounds are relative to the target recording. Regions are
half-open [start_s, end_s), cover every target sample, and never interpolate across
a region boundary. Positive ``time_shift_s`` moves a reference recording later on
the target timeline: background(t) = scale * reference(t - shift) + offset.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..core.atomic_write import staged_write_set
from .open_pore_reference import OpenPoreReference, build_reference

__all__ = [
    "AnchorRegion", "BackgroundRecipe", "BackgroundResult", "correct_background",
    "background_export_paths", "export_background",
]

_SOURCES = frozenset({"anchors", "intervals", "recording"})
_MODES = frozenset({"zero_residual", "retained_offset"})
_INTERPOLATIONS = frozenset({"constant", "linear", "pchip"})
_CURRENT_UNITS = frozenset({"A", "mA", "uA", "nA", "pA", "fA"})


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _units(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Background correction requires declared physical current units")
    value = value.strip().replace("µ", "u").replace("μ", "u")
    if value not in _CURRENT_UNITS:
        raise ValueError("Background correction requires declared physical current units (A, mA, uA, nA, pA or fA)")
    return value


def _plain(value: Any, label: str) -> Any:
    """Copy strict JSON metadata, without guessing how arbitrary objects serialize."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from exc


@dataclass(frozen=True)
class AnchorRegion:
    start_s: float
    end_s: float
    anchors: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        start = _number(self.start_s, "Region start")
        end = _number(self.end_s, "Region end")
        if start < 0 or end <= start:
            raise ValueError("Anchor regions must satisfy 0 <= start_s < end_s")
        try:
            raw_anchors = tuple(self.anchors)
        except TypeError as exc:
            raise ValueError("Each region needs time/current anchor pairs") from exc
        anchors = []
        for item in raw_anchors:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each anchor must be a [time_s, current] pair")
            t, current = _number(item[0], "Anchor time"), _number(item[1], "Anchor current")
            if not start <= t <= end:
                raise ValueError("Every anchor must lie inside its own region")
            if anchors and t <= anchors[-1][0]:
                raise ValueError("Anchor times must be strictly increasing with no duplicates")
            anchors.append((t, current))
        if not anchors:
            raise ValueError("Each region needs at least one anchor")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        object.__setattr__(self, "anchors", tuple(anchors))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "AnchorRegion") -> "AnchorRegion":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"start_s", "end_s", "anchors"}:
            raise ValueError("An anchor region requires exactly start_s, end_s and anchors")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {"start_s": self.start_s, "end_s": self.end_s,
                "anchors": [list(pair) for pair in self.anchors]}


@dataclass(frozen=True)
class BackgroundRecipe:
    source: str
    mode: str = "zero_residual"
    units: str = "pA"
    interpolation: str = "linear"
    edge_policy: str = "hold"
    regions: tuple[AnchorRegion, ...] = ()
    interval_reference: OpenPoreReference | None = None
    time_shift_s: float = 0.0
    scale: float = 1.0
    offset: float = 0.0
    reference_resampling: str = "none"
    version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version != 1:
            raise ValueError("Unsupported background recipe version; expected 1")
        if not isinstance(self.source, str) or self.source not in _SOURCES:
            raise ValueError("Background source must be anchors, intervals or recording")
        if not isinstance(self.mode, str) or self.mode not in _MODES:
            raise ValueError("Background mode must be zero_residual or retained_offset")
        if not isinstance(self.interpolation, str) or self.interpolation not in _INTERPOLATIONS:
            raise ValueError("Anchor interpolation must be constant, linear or pchip")
        if self.edge_policy not in ("hold", "refuse"):
            raise ValueError("Anchor edge policy must be hold or refuse")
        if self.reference_resampling not in ("none", "linear"):
            raise ValueError("Reference resampling must be none or linear")
        object.__setattr__(self, "units", _units(self.units))
        for name in ("time_shift_s", "scale", "offset"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        try:
            regions = tuple(AnchorRegion.from_dict(r) for r in self.regions)
        except TypeError as exc:
            raise ValueError("regions must be a list of anchor regions") from exc
        for previous, current in zip(regions, regions[1:]):
            if current.start_s != previous.end_s:
                raise ValueError("Anchor regions must be ordered, adjacent and non-overlapping; no gaps may be bridged")
        object.__setattr__(self, "regions", regions)
        if self.interval_reference is not None:
            object.__setattr__(self, "interval_reference", OpenPoreReference.from_dict(self.interval_reference))
        if self.source == "anchors":
            if not regions:
                raise ValueError("Manual background needs at least one anchor region")
            for region in regions:
                if self.interpolation == "constant" and len(region.anchors) != 1:
                    raise ValueError("Constant interpolation needs exactly one current anchor per region")
                if self.interpolation != "constant" and len(region.anchors) < 2:
                    raise ValueError("Linear and PCHIP interpolation need at least two anchors per region")
        if self.source == "intervals":
            if self.interval_reference is None:
                raise ValueError("The current open-pore reference recipe is required")
            if self.units != "pA":
                raise ValueError("The existing open-pore reference uses pA; explicitly calibrate to pA first")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "BackgroundRecipe") -> "BackgroundRecipe":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("Background recipe must be a JSON object")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown background recipe fields: " + ", ".join(sorted(str(k) for k in unknown)))
        if "source" not in value:
            raise ValueError("Background recipe requires a source")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "source": self.source, "mode": self.mode, "units": self.units,
            "interpolation": self.interpolation, "edge_policy": self.edge_policy,
            "regions": [r.to_dict() for r in self.regions],
            "interval_reference": None if self.interval_reference is None else self.interval_reference.to_dict(),
            "time_shift_s": self.time_shift_s, "scale": self.scale, "offset": self.offset,
            "reference_resampling": self.reference_resampling,
        }


@dataclass(frozen=True)
class BackgroundResult:
    recipe: BackgroundRecipe
    raw: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    time_s: np.ndarray
    warnings: tuple[str, ...]
    provenance: dict[str, Any]


def _array(value: Any, label: str) -> np.ndarray:
    original = np.asarray(value)
    if original.ndim != 1 or original.size < 2 or original.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be a real one-dimensional trace with at least two samples")
    result = np.array(original, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains nonfinite samples; repair or exclude them explicitly before correction")
    return result


def _rate(value: Any, label: str) -> float:
    result = _number(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be a positive sample rate in Hz")
    return result


def _times(n: int, fs: float, start: float, supplied: Any, label: str) -> np.ndarray:
    times = start + np.arange(n, dtype=np.float64) / fs
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{label} sample times are not representable at this rate and origin")
    if supplied is not None:
        actual = _array(supplied, f"{label} sample times")
        if actual.size != n or np.any(np.diff(actual) <= 0):
            raise ValueError(f"{label} sample times must be increasing and match the trace length")
        # A gap or nonuniform grid must not become an ordinary interpolation span.
        tolerance = max(1e-10 / fs, 8 * np.spacing(max(abs(start), abs(times[-1]), 1.0)))
        if not np.allclose(actual, times, rtol=0.0, atol=tolerance):
            raise ValueError(f"{label} sample times contain a gap or do not match the declared uniform rate/origin")
    return times


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes(order="C")).hexdigest()


def _anchor_baseline(t: np.ndarray, recipe: BackgroundRecipe) -> tuple[np.ndarray, list[str], dict]:
    baseline = np.empty(t.size, dtype=np.float64)
    assigned = np.zeros(t.size, dtype=bool)
    held_total = 0
    notes = ["Manual anchor currents are user assertions, not independently verified open-pore measurements."]
    details = []
    for region in recipe.regions:
        chosen = (t >= region.start_s) & (t < region.end_s)
        q = t[chosen]
        if not q.size:
            raise ValueError("Each anchor region must contain target samples")
        anchors = np.asarray(region.anchors, dtype=np.float64)
        at, av = anchors[:, 0], anchors[:, 1]
        if recipe.interpolation == "constant":
            baseline[chosen] = av[0]
            held = np.zeros(q.size, dtype=bool)
        else:
            held = (q < at[0]) | (q > at[-1])
            if held.any() and recipe.edge_policy == "refuse":
                raise ValueError("Target samples lie outside anchor support; add anchors or explicitly choose held endpoints")
            clipped = np.clip(q, at[0], at[-1])
            if recipe.interpolation == "linear":
                baseline[chosen] = np.interp(clipped, at, av)
            else:
                baseline[chosen] = PchipInterpolator(at, av, extrapolate=False)(clipped)
        assigned[chosen] = True
        held_count = int(held.sum())
        held_total += held_count
        details.append({"start_s": region.start_s, "end_s": region.end_s,
                        "n_samples": int(q.size), "held_endpoint_samples": held_count})
    if not assigned.all():
        raise ValueError("Anchor regions do not cover every target sample; extend explicit regions instead of bridging uncovered time")
    if held_total:
        notes.append(f"Held endpoint levels for {held_total} of {t.size} samples outside anchor support; these edges are assumed, not measured.")
    return baseline, notes, {"regions": details, "held_endpoint_samples": held_total,
                             "held_endpoint_fraction": held_total / t.size,
                             "boundary_policy": "independent half-open regions; no interpolation across discontinuities"}


def correct_background(
    raw: Any, fs: float, recipe: BackgroundRecipe | Mapping[str, Any], *, units: str,
    reference: Any = None, reference_fs: float | None = None,
    reference_units: str | None = None, time_start_s: float = 0.0,
    reference_time_start_s: float = 0.0, sample_times_s: Any = None,
    reference_sample_times_s: Any = None,
    source_metadata: Mapping[str, Any] | None = None,
    reference_metadata: Mapping[str, Any] | None = None,
) -> BackgroundResult:
    """Calculate one independent comparison, refusing uncovered or ambiguous data.

    Units must already agree, including their scale prefix; no implicit conversion
    occurs. ``reference_resampling='none'`` requires equal sample rates and a
    sample-aligned shift. ``'linear'`` explicitly permits resampling of a uniform
    reference grid, but never extends its domain or bridges supplied time gaps.
    Optional sample-time arrays validate the claimed uniform rate and origin.
    """
    recipe = BackgroundRecipe.from_dict(recipe)
    units = _units(units)
    if units != recipe.units:
        raise ValueError("Target units differ from the recipe units; recalibrate or recreate the recipe explicitly")
    x = _array(raw, "Target trace")
    fs = _rate(fs, "Target sample rate")
    start = _number(time_start_s, "Target time origin")
    time = _times(x.size, fs, start, sample_times_s, "Target")
    relative = np.arange(x.size, dtype=np.float64) / fs
    metadata = _plain(dict(source_metadata or {}), "Source metadata")
    reference_meta = _plain(dict(reference_metadata or {}), "Reference metadata")
    notes: list[str] = []
    details: dict[str, Any] = {}
    reference_stamp = None
    if recipe.source == "anchors":
        baseline, notes, details = _anchor_baseline(relative, recipe)
    elif recipe.source == "intervals":
        baseline, details = build_reference(x, fs, recipe.interval_reference)
        notes.extend(details.get("assumptions", []))
        notes.extend(details.get("warnings", []))
        # The existing interval contract always holds endpoints; its metadata states
        # that policy. This path does not re-interpret the original recipe.
    else:
        if reference is None or reference_fs is None or reference_units is None:
            raise ValueError("Reference recording, sample rate and physical units are required")
        if _units(reference_units) != units:
            raise ValueError("Target and reference units must match including scale; no implicit current-unit conversion is performed")
        ref = _array(reference, "Reference trace")
        rfs = _rate(reference_fs, "Reference sample rate")
        rstart = _number(reference_time_start_s, "Reference time origin")
        _times(ref.size, rfs, rstart, reference_sample_times_s, "Reference")
        # Subtract origins before adding relative times to retain sub-sample precision.
        indices = ((start - rstart - recipe.time_shift_s) + relative) * rfs
        tolerance = 1e-7  # sample-index roundoff only, not a license to extrapolate
        if not np.isfinite(indices).all() or indices[0] < -tolerance or indices[-1] > ref.size - 1 + tolerance:
            raise ValueError("Reference does not cover the entire shifted target span; partial overlap or extrapolation is refused")
        if recipe.reference_resampling == "none":
            if not np.isclose(fs, rfs, rtol=1e-12, atol=0):
                raise ValueError("Sample rates differ; explicitly choose linear reference resampling or provide a matching rate")
            nearest = np.rint(indices)
            if not np.allclose(indices, nearest, rtol=0, atol=tolerance):
                raise ValueError("The reference shift is not sample-aligned; explicitly choose linear resampling")
            aligned = ref[np.clip(nearest.astype(np.int64), 0, ref.size - 1)]
        else:
            aligned = np.interp(np.clip(indices, 0, ref.size - 1), np.arange(ref.size), ref)
            notes.append("Reference samples were explicitly linearly resampled; interpolation changes noise and correlation and is not anti-alias filtering.")
        with np.errstate(over="ignore", invalid="ignore"):
            baseline = recipe.scale * aligned + recipe.offset
        notes.append("Reference-recording subtraction assumes the declared alignment, scaling and offset are appropriate; independent noise is not automatically removed.")
        reference_stamp = {"n_samples": int(ref.size), "sample_rate_hz": rfs,
                           "time_start_s": rstart, "units": units,
                           "sha256_float64_le": _hash(ref), "metadata": reference_meta}
        details = {"reference_index_start": float(indices[0]), "reference_index_end": float(indices[-1]),
                   "alignment": "background(t) = scale * reference(t - time_shift_s) + offset",
                   "overlap_policy": "full target coverage required; no trimming or extrapolation",
                   "resampling": recipe.reference_resampling}
    baseline = np.array(baseline, dtype=np.float64, order="C", copy=True)
    if baseline.shape != x.shape or not np.isfinite(baseline).all():
        raise ValueError("The background calculation produced invalid or nonfinite values")
    retained = float(np.median(baseline)) if recipe.mode == "retained_offset" else 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        corrected = x - baseline + retained
    if not np.isfinite(corrected).all():
        raise ValueError("Background subtraction overflowed; choose a representable current scale")
    provenance = {
        "format": "nanoscope-background-correction", "version": 1,
        "recipe": recipe.to_dict(), "scope": "independent comparison/export; does not alter runtime detection or recordings",
        "source": {"n_samples": int(x.size), "sample_rate_hz": fs, "time_start_s": start,
                   "units": units, "sha256_float64_le": _hash(x), "metadata": metadata},
        "reference": reference_stamp, "details": _plain(details, "Background details"),
        "equation": "corrected = raw - background + retained_offset",
        "retained_offset": retained, "retained_offset_policy": "median background" if recipe.mode == "retained_offset" else "zero",
        "warnings": list(dict.fromkeys(notes)), "raw_input_modified": False,
        "baseline_sha256_float64_le": _hash(baseline),
        "corrected_sha256_float64_le": _hash(corrected),
        "time_sha256_float64_le": _hash(time),
    }
    # Own copies: neither editing the caller's source nor a later edit to that source
    # can silently change the completed comparison or its exported numbers.
    for array in (x, baseline, corrected, time):
        array.setflags(write=False)
    return BackgroundResult(recipe, x, baseline, corrected, time,
                            tuple(provenance["warnings"]), provenance)


def background_export_paths(path: str | Path) -> tuple[Path, Path]:
    data = Path(path)
    if not data.suffix:
        data = data.with_suffix(".csv")
    meta = data.with_suffix(".meta.json")
    if data == meta:
        raise ValueError("Choose a CSV filename distinct from its metadata sidecar")
    return data, meta


def export_background(path: str | Path, result: BackgroundResult) -> dict[str, Path]:
    """Atomically export every completed sample and the exact source/recipe record."""
    if not isinstance(result, BackgroundResult):
        raise ValueError("A completed BackgroundResult is required for export")
    data, meta = background_export_paths(path)
    provenance = _plain(result.provenance, "Result provenance")
    # Reject accidental modifications even if a caller deliberately re-enabled a
    # NumPy array's write flag or edited the public metadata after calculation.
    if (_hash(result.raw) != provenance["source"]["sha256_float64_le"]
            or _hash(result.baseline) != provenance["baseline_sha256_float64_le"]
            or _hash(result.corrected) != provenance["corrected_sha256_float64_le"]
            or _hash(result.time_s) != provenance["time_sha256_float64_le"]
            or result.recipe.to_dict() != provenance["recipe"]):
        raise ValueError("Completed background result changed; calculate it again before exporting")
    provenance["data_file"] = data.name
    provenance["columns"] = ["time_s", "raw_current", "background_current", "corrected_current"]
    provenance["current_units"] = result.recipe.units
    with staged_write_set() as group:
        staged_data = group.stage(data)
        with staged_data.open("w", encoding="utf-8", newline="") as stream:
            stream.write(",".join(provenance["columns"]) + "\n")
            # Keep memory bounded for long traces; output remains full resolution.
            for start in range(0, result.raw.size, 100_000):
                end = min(result.raw.size, start + 100_000)
                block = np.column_stack([a[start:end] for a in
                                         (result.time_s, result.raw, result.baseline, result.corrected)])
                np.savetxt(stream, block, delimiter=",", fmt="%.17g")
        staged_meta = group.stage(meta)
        staged_meta.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"data": data, "meta": meta}
