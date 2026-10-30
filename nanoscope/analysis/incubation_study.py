"""Headless, transactional study runner for the original Zn student notebook.

The 2026-09-11 notebook has three incompatible amplitude conventions and a
12-feature pooled clustering.  They are intentionally separate from the current
NanoScope 13-feature, per-scope clustering/GMM appendix. Prepared CSVs are the
scientific input; this runner neither regenerates their preprocessing nor trusts
pre-existing event catalogs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..core.atomic_write import staged_path
from . import event_features as EF
from . import subpop_cluster as SC
from . import subpop_gmm as SG
from . import subpop_report as SR
from .cancel import CancelFlag, check_cancelled, propagate_cancel
from .incubation_detection import analyze_recording, REFERENCE_NOTEBOOK_SHA256
from .events import empty_df as empty_event_frame

RECIPE_VERSION = "zn_student_incubation_v1"
BUNDLE_VERSION = 1
REFERENCE_TRAINING_TAG = "zn_30h"
NOTEBOOK_FEATURES = ("log_dwell", "depth_max_pA", "depth_sust_pA", "mean_pA", "std_pA",
                    "min_dev_pA", "skewness", "kurtosis", "area_pA_ms", "t_peak_frac", "f_sust", "snr")
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class StudyRecording:
    tag: str
    trace_path: Path | str
    metadata_path: Path | str
    incubation_h: float
    pore_id: str
    pore_diameter_nm: float
    concentration_pm: float = 100.0
    voltage_mv: float = -200.0


@dataclass(frozen=True)
class StudySpec:
    name: str
    recordings: tuple[StudyRecording, ...] | list[StudyRecording]
    model_path: Path | str
    mixture_tag: str = "zn_30h"
    recipe_version: str = RECIPE_VERSION
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "recordings", tuple(self.recordings))


@dataclass
class StudyResult:
    output_dir: Path
    summary: pd.DataFrame
    events: pd.DataFrame
    metadata: dict
    figure_paths: list[Path]

    @property
    def report_path(self) -> Path:
        return self.output_dir / "report.md"


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, pd.Index)):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _absolute(path: Path | str, base: Path | None = None) -> Path:
    value = Path(path).expanduser()
    return ((base or Path.cwd()) / value).resolve() if not value.is_absolute() else value.resolve()


def _manifest(spec: StudySpec, base: Path | None = None) -> dict:
    def path_value(value):
        path = _absolute(value)
        return os.path.relpath(path, base) if base is not None else str(path)
    rows = []
    for rec in spec.recordings:
        row = asdict(rec)
        row.update(trace_path=path_value(rec.trace_path), metadata_path=path_value(rec.metadata_path))
        rows.append(row)
    return dict(format="nanoscope-incubation-study", version=BUNDLE_VERSION,
                name=spec.name, recordings=rows, model_path=path_value(spec.model_path),
                mixture_tag=spec.mixture_tag, recipe_version=spec.recipe_version, seed=spec.seed)


def save_spec(spec: StudySpec, path: Path | str) -> Path:
    """Save an editable recipe atomically; relative paths are relative to the JSON."""
    path = _absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The user's own file, so through the staging door. A private mkstemp + os.replace
    # was atomic but overwrote a read-only recipe without a word, reset its mode to
    # 0600 and dropped its extended attributes. See nanoscope.core.atomic_write.
    with staged_path(path) as tmp:
        _write_json(tmp, _manifest(spec, path.parent))
    return path


def load_spec(path: Path | str) -> StudySpec:
    path = _absolute(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("format", "nanoscope-incubation-study") != "nanoscope-incubation-study":
        raise ValueError("Not a NanoScope incubation-study recipe")
    if obj.get("version", BUNDLE_VERSION) != BUNDLE_VERSION:
        raise ValueError("Unsupported incubation-study manifest version")
    records = []
    for row in obj["recordings"]:
        row = dict(row)
        row["trace_path"] = _absolute(row["trace_path"], path.parent)
        row["metadata_path"] = _absolute(row["metadata_path"], path.parent)
        records.append(StudyRecording(**row))
    return StudySpec(name=obj["name"], recordings=tuple(records),
                     model_path=_absolute(obj["model_path"], path.parent),
                     mixture_tag=obj.get("mixture_tag", "zn_30h"),
                     recipe_version=obj.get("recipe_version", RECIPE_VERSION), seed=obj.get("seed", 0))


def discover_recordings(folder: Path | str) -> list[StudyRecording]:
    """Discover prepared pairs only. Unknown scientific metadata stays unset (NaN).

    The four named Zn recordings have explicit, documented notebook defaults;
    pore_id defaults to the tag, identifying recordings rather than claiming that
    two diameter-matched recordings used the same physical pore.
    """
    folder = _absolute(folder)
    diameters = {"zn_2h": 16, "zn_10h": 18, "zn_18h": 18, "zn_30h": 20}
    records = []
    for metadata_path in sorted(folder.glob("*_metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"Metadata must be an object: {metadata_path.name}")
        tag = str(meta.get("tag") or metadata_path.name[:-len("_metadata.json")])
        candidates = [folder / f"{tag}.csv.gz", folder / f"{tag}.csv"]
        trace = next((p for p in candidates if p.is_file()), None)
        if trace is None:
            continue
        match = re.fullmatch(r"zn_(\d+(?:\.\d+)?)h", tag)
        hours = meta.get("incubation_h", float(match[1]) if match else float("nan"))
        records.append(StudyRecording(tag, trace, metadata_path, hours,
                        str(meta.get("pore_id") or tag),
                        meta.get("pore_nominal_nm", meta.get("pore_diameter_nm", diameters.get(tag, float("nan")))),
                        meta.get("concentration_pm", meta.get("conc_pM", 100.0 if tag in diameters else float("nan"))),
                        meta.get("bias_mV", meta.get("voltage_mv", -200.0 if tag in diameters else float("nan")))))
    return sorted(records, key=lambda r: (float(r.incubation_h) if np.isfinite(float(r.incubation_h)) else math.inf, r.tag))


def _validate_spec(spec: StudySpec) -> None:
    if not isinstance(spec.name, str) or not spec.name.strip():
        raise ValueError("Study name must be nonempty")
    if spec.recipe_version != RECIPE_VERSION:
        raise ValueError(f"Unsupported study recipe {spec.recipe_version!r}")
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, (int, np.integer)) or spec.seed < 0:
        raise ValueError("Study seed must be a nonnegative integer")
    if spec.seed != 0:
        raise ValueError("The reference recipe fixes seed 0; select seed 0 for this version")
    if not spec.recordings:
        raise ValueError("Study needs at least one recording")
    seen_tags, seen_traces, seen_metadata = set(), set(), set()
    for rec in spec.recordings:
        if not isinstance(rec.tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rec.tag):
            raise ValueError("Recording tags must be nonempty safe file identifiers")
        if rec.tag in seen_tags:
            raise ValueError(f"Duplicate recording tag: {rec.tag}")
        seen_tags.add(rec.tag)
        if not str(rec.pore_id).strip():
            raise ValueError(f"{rec.tag}: pore_id must be nonempty")
        for name, value, zero_ok in (("incubation_h", rec.incubation_h, True),
                                    ("pore_diameter_nm", rec.pore_diameter_nm, False),
                                    ("concentration_pm", rec.concentration_pm, False)):
            if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0 or (not zero_ok and float(value) == 0):
                raise ValueError(f"{rec.tag}: {name} must be finite and {'nonnegative' if zero_ok else 'positive'}")
        if isinstance(rec.voltage_mv, bool) or not np.isfinite(float(rec.voltage_mv)) or float(rec.voltage_mv) == 0:
            raise ValueError(f"{rec.tag}: voltage_mv must be finite and nonzero")
        for name, paths in (("trace_path", seen_traces), ("metadata_path", seen_metadata)):
            path = _absolute(getattr(rec, name))
            if not path.is_file():
                raise ValueError(f"{rec.tag}: {name} does not exist: {path}")
            identity = (path.stat().st_dev, path.stat().st_ino)
            if identity in paths:
                raise ValueError(f"The same {name} was assigned more than once")
            paths.add(identity)
    if spec.mixture_tag not in seen_tags:
        raise ValueError("mixture_tag must identify one study recording")
    if not _absolute(spec.model_path).is_file():
        raise ValueError("Student model file does not exist")


def _sha256(path: Path, cancel=None) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            check_cancelled(cancel)
            sha.update(data)
    return sha.hexdigest()


def block_bootstrap_rate(onsets, duration, n_blocks=20, n_boot=4000, seed=0) -> dict:
    """Original cell 1: resample contiguous time-block counts, not individual events."""
    onsets = np.asarray(onsets, float)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Bootstrap duration must be positive and finite")
    if onsets.ndim != 1 or not np.isfinite(onsets).all() or np.any((onsets < 0) | (onsets >= duration)):
        raise ValueError("Event onsets must fall within the recording's [0, duration) clock")
    if isinstance(n_blocks, bool) or int(n_blocks) != n_blocks or n_blocks < 2:
        raise ValueError("n_blocks must be an integer >= 2")
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    rng = np.random.default_rng(seed)
    counts = np.histogram(onsets, bins=np.linspace(0, duration, n_blocks + 1))[0].astype(float)
    draws = counts[rng.integers(0, n_blocks, (n_boot, n_blocks))].sum(1) / (n_blocks * (duration / n_blocks))
    return dict(rate=len(onsets) / duration, lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)),
                fano=float(counts.var(ddof=1) / counts.mean()) if counts.mean() > 0 else float("nan"),
                n_blocks=n_blocks, n_boot=n_boot, seed=seed)


def _condition(rec: StudyRecording) -> str:
    return f"{rec.incubation_h:g} h ({rec.pore_diameter_nm:g} nm) [{rec.tag}]"


def _comparison_tables(events: pd.DataFrame, summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    floors = np.round(np.arange(0.02, 0.201, 0.01), 3)
    sweep, sensitivity, distances, pairs = [], [], [], []
    for row in summary.itertuples():
        ev = events[events.tag == row.tag]
        for blocks in (10, 20, 40):
            sensitivity.append(dict(tag=row.tag, **block_bootstrap_rate(ev.onset_s, row.duration_s, n_blocks=blocks)))
        for definition, column in (("notebook_cell28_peak_global", "aggregation_fracblock_global"),
                                   ("sustained_local", "fracblock"), ("kde_sustained_global", "f_sust_global")):
            for floor in floors:
                count = int((ev[column] >= floor).sum())
                sweep.append(dict(tag=row.tag, condition=row.condition, definition=definition,
                                  floor=float(floor), n=count, rate=count / row.duration_s))
    sw = pd.DataFrame(sweep)
    eligible = summary[summary.n_events >= 50]
    common = float(eligible.gate_frac_of_I0.max()) if len(eligible) else float("nan")
    for a, b in itertools.combinations(summary.itertuples(), 2):
        fc = max(a.gate_frac_of_I0, b.gate_frac_of_I0)
        # Preserve cell 41's nearest-grid lookup and record the ACTUAL grid floor.
        grid_floor = float(floors[np.argmin(np.abs(floors - fc))]) if np.isfinite(fc) else float("nan")
        ea, eb = events[events.tag == a.tag], events[events.tag == b.tag]
        for definition, column in (("notebook_cell28_peak_global", "aggregation_fracblock_global"),
                                   ("sustained_local", "fracblock")):
            na, nb = int((ea[column] >= grid_floor).sum()), int((eb[column] >= grid_floor).sum())
            ra, rb = na / a.duration_s, nb / b.duration_s
            pairs.append(dict(tag_a=a.tag, tag_b=b.tag, definition=definition, native_common_floor=fc,
                              evaluated_floor=grid_floor, n_a=na, n_b=nb, rate_a=ra, rate_b=rb,
                              ratio_a_over_b=ra / rb if rb else float("nan"),
                              denominator_zero=rb == 0, fewer_than_10_in_an_arm=min(na, nb) < 10,
                              size_matched=a.pore_nm == b.pore_nm, same_pore_id=a.pore_id == b.pore_id))
        if a.n_events < 50 or b.n_events < 50 or not np.isfinite(common):
            continue
        aa, bb = ea[ea.aggregation_fracblock_global >= common], eb[eb.aggregation_fracblock_global >= common]
        for col in ("dwell_ms", "aggregation_fracblock_global"):
            valid = min(len(aa), len(bb)) >= 20
            distances.append(dict(tag_a=a.tag, tag_b=b.tag, variable=col, common_floor=common,
                                  n_a=len(aa), n_b=len(bb), status="computed" if valid else "fewer_than_20_above_floor",
                                  ks_distance=float(ks_2samp(aa[col], bb[col]).statistic) if valid else float("nan"),
                                  median_a=aa[col].median(), median_b=bb[col].median()))
    return {"gate_transfer_sweep": sw, "rate_block_sensitivity": pd.DataFrame(sensitivity),
            "gate_transfer_pairs": pd.DataFrame(pairs, columns=["tag_a", "tag_b", "definition", "native_common_floor", "evaluated_floor", "n_a", "n_b", "rate_a", "rate_b", "ratio_a_over_b", "denominator_zero", "fewer_than_10_in_an_arm", "size_matched", "same_pore_id"]),
            "distribution_distances": pd.DataFrame(distances, columns=["tag_a", "tag_b", "variable", "common_floor", "n_a", "n_b", "status", "ks_distance", "median_a", "median_b"])}


def _recording_features(analysis, rec: StudyRecording, accepted: pd.DataFrame, *,
                        events_down: bool = False) -> pd.DataFrame:
    # The fixed notebook recipe retains cell 51's original-sign convention.
    # Manual preparation has already oriented events downward; undoing that
    # orientation here reverses upward sources' depth, area and SNR features.
    sig = (analysis.detrended if events_down else
           analysis.detrended * int(analysis.provenance["event_polarity"]))
    windows = accepted[["start_idx", "end_idx"]].copy()
    windows["end_idx"] += 1  # detector inclusive -> feature kernel half-open
    scalars = EF.feature_scalars(sig, analysis.raw)
    feats = EF.event_features(sig, analysis.fs_hz, windows, scalars=scalars)
    for col in ("event_id", "tag", "onset_s", "end_s", "start_idx", "end_idx", "incubation_h", "pore_nm", "pore_id"):
        feats[col] = accepted.loc[feats.index, col]
    feats["duration_s"] = analysis.duration_s
    feats["scope"] = str(rec.incubation_h).removesuffix(".0")
    feats["recording"] = rec.tag
    # The current NanoScope appendix fixes the old onset-median tautology. Keep
    # the historical column too, but do not use it to claim time stationarity.
    feats["time_half"] = EF.time_half(feats.onset_s.to_numpy(), span=(0.0, analysis.duration_s))
    feats["time_half_legacy_onset_median"] = EF.time_half(feats.onset_s.to_numpy())
    return feats


def _original_clustering(features: pd.DataFrame, folder: Path, cancel=None) -> tuple[pd.DataFrame, dict]:
    columns = [c for c in features.columns if c != "entropy"]
    table = features[columns].copy()
    info = dict(recipe="current_original_notebook_cells51_53", feature_columns=NOTEBOOK_FEATURES,
                random_state=0, preprocessing="winsorize1_99_then_median_IQR; nonfinite_scaled_to0",
                amplitude="full_trace_median_minus_unoriented_detrended; global_float64_raw_I0",
                input_rows=len(table), status="computed")
    if len(table) < 20:
        info.update(status="insufficient_events", reason="Pooled clustering requires at least 20 measurable events (HDBSCAN cluster-size floor).")
        return table, info
    check_cancelled(cancel)
    projection = SC.scale_and_pca(table, columns=NOTEBOOK_FEATURES, random_state=0)
    check_cancelled(cancel)
    table["hdbscan"] = SC.hdbscan_labels(projection.scores)
    check_cancelled(cancel)
    table["kmeans3"] = SC.kmeans_labels(projection.scores, random_state=0)
    for i in range(projection.scores.shape[1]):
        table[f"PC{i+1}"] = projection.scores[:, i]
    SC.loading_table(projection).to_csv(folder / "loadings.csv", index=False)
    pd.DataFrame({"pc": np.arange(len(projection.explained_variance_ratio)) + 1,
                  "explained_variance_ratio": projection.explained_variance_ratio}).to_csv(folder / "variance.csv", index=False)
    # Record the complete transform, not just rounded human-readable loadings.
    X = features[list(NOTEBOOK_FEATURES)].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.nan)
    lo, hi = np.nanpercentile(X, [1, 99], axis=0)
    clipped = np.clip(X, lo, hi)
    _write_json(folder / "transform.json", dict(features=NOTEBOOK_FEATURES, winsor_lo=lo, winsor_hi=hi,
                center=np.nanmedian(clipped, axis=0), iqr=np.nanpercentile(clipped, 75, axis=0)-np.nanpercentile(clipped,25,axis=0),
                components=projection.components, explained_variance_ratio=projection.explained_variance_ratio,
                n_kept=projection.n_kept, pca_metadata=projection.meta))
    for label in ("hdbscan", "kmeans3"):
        ct = pd.crosstab(table[label], table.incubation_h)
        ct.to_csv(folder / f"{label}_composition_counts.csv")
        ct.div(ct.sum(axis=1), axis=0).to_csv(folder / f"{label}_composition_fractions.csv")
        table.groupby(label)[list(NOTEBOOK_FEATURES)].median().to_csv(folder / f"{label}_feature_medians.csv")
    info.update(n_pcs=projection.n_kept, pca_metadata=projection.meta,
                hdbscan_clusters=int(len(set(table.hdbscan) - {-1})),
                hdbscan_noise_fraction=float((table.hdbscan < 0).mean()))
    check_cancelled(cancel)
    refusal = SC.embedding_refusal(projection.scores, kind="t-SNE", perplexity=SC.tsne_perplexity(len(table)))
    if refusal:
        info["tsne_status"] = refusal
    else:
        coords = SC.tsne_embedding(projection.scores, random_state=0)
        table["tsne_x"], table["tsne_y"] = coords[:, 0], coords[:, 1]
        info["tsne_status"] = "computed; qualitative coordinates, not separation measurements"
    check_cancelled(cancel)
    return table, info


def _appendix_population_table(quant, duration_s: float) -> pd.DataFrame:
    """Name hard assignments and soft mixture-weight intervals independently."""
    table = quant.table.rename(columns={
        "fraction": "assigned_event_fraction", "rate_per_s": "assigned_event_rate_per_s",
        "frac_lo95": "mixture_weight_lo95", "frac_hi95": "mixture_weight_hi95",
    }).copy()
    components = table.subpop.to_numpy(int)
    # The kernel's human-readable table rounds these columns. Export the actual
    # estimates/intervals so the figure and data retain the same endpoints.
    table["assigned_event_fraction"] = table.n.to_numpy(float) / quant.n_events
    table["assigned_event_rate_per_s"] = table.n.to_numpy(float) / duration_s
    table["mixture_weight"] = quant.fit.weights[components]
    table["mixture_weight_lo95"] = quant.ci.lo[components]
    table["mixture_weight_hi95"] = quant.ci.hi[components]
    return table


def _appendix_weight_warnings(quant) -> None:
    """Assess weight intervals against weights, preserving unrelated caveats."""
    assigned = np.array([quant.fit.sizes.get(k, 0) / quant.n_events for k in range(quant.fit.k)])
    legacy = set(SG._ci_warnings(assigned, quant.ci.lo, quant.ci.hi))
    regenerated = tuple(message.replace("fraction", "mixture weight") for message in
                        SG._ci_warnings(quant.fit.weights, quant.ci.lo, quant.ci.hi))

    def replace_diagnostics(messages):
        # Exact matches only: convergence failures, lost replicates and any
        # independent calibration caveat must survive unchanged.
        return tuple(dict.fromkeys([message for message in messages if message not in legacy]
                                   + list(regenerated)))

    quant.ci.warnings = replace_diagnostics(quant.ci.warnings)
    quant.warnings = replace_diagnostics(quant.warnings)


def _appendix(features: pd.DataFrame, mixture, spec: StudySpec, folder: Path, progress=None, cancel=None):
    """Existing NanoScope 13-feature kernels, explicitly separate from cell 52."""
    table = features.copy()
    info = dict(recipe="nanoscope_13feature_quantitative_appendix_v1", feature_columns=EF.CLUST_FEATS,
                mixture_tag=spec.mixture_tag, random_state=0, bootstrap_seed=0,
                time_half="recording_clock_midpoint; legacy_onset_median retained separately",
                fraction_definitions=dict(
                    assigned_event_fraction="Hard-label count n / all measurable accepted events in the mixture recording",
                    assigned_event_rate_per_s="Hard-label count n / full recording duration",
                    mixture_weight="Fitted Gaussian mixture weight; soft membership",
                    mixture_weight_interval="Percentiles of event-bootstrap refitted mixture weights, matched by component means; interval-width and label-switching diagnostics compare fitted mixture weights; separate from hard-label fractions"),
                status="computed", warnings=[])
    if not len(table):
        info.update(status="insufficient_events", warnings=["No measurable accepted events."])
        table.to_csv(folder / "features_assignments.csv", index=False)
        _write_json(folder / "report.json", info)
        return table, None, info
    clusters = SC.cluster_scopes(table, scope_column="scope", random_state=0, cancel=cancel,
                                 progress=(lambda f,m: progress(.12*f,m)) if progress else None)
    labels = clusters.label_table()
    for col in labels:
        table[col] = labels[col]
    for scope, result in clusters.scopes.items():
        result.loadings.to_csv(folder / f"loadings_{scope}.csv", index=False)
        _write_json(folder / f"projection_{scope}.json", dict(components=result.pca.components,
                    explained_variance_ratio=result.pca.explained_variance_ratio, metadata=result.pca.meta,
                    n_hdbscan_clusters=result.n_hdbscan_clusters, hdbscan_noise_fraction=result.hdbscan_noise_fraction,
                    hdbscan_silhouette=result.hdbscan_silhouette, kmeans3_silhouette=result.kmeans3_silhouette))
    info["skipped_scopes"] = clusters.skipped
    sub = table[table.tag == spec.mixture_tag]
    quant = None
    if len(sub) < SR.MIN_GMM_EVENTS:
        info.update(status="mixture_insufficient_events", warnings=[f"Mixture has {len(sub)} events; requires at least {SR.MIN_GMM_EVENTS}."])
    else:
        check_cancelled(cancel)
        scope = str(sub.scope.iloc[0])
        name = f"hdbscan_{scope}"
        exploratory = labels.loc[sub.index, name] if name in labels else None
        quant = SG.quantify_subpopulations(sub, duration_s=mixture.duration_s,
                    exploratory_labels=exploratory, exploratory_name=name, random_state=0,
                    bootstrap_seed=0, cancel=cancel,
                    progress=(lambda f,m: progress(.12+.88*f,m)) if progress else None)
        _appendix_weight_warnings(quant)
        # The generic kernel inherited a voltage-series context. This study's
        # measured timing finding stands without claiming that acquisition design.
        quant.warnings = tuple(message.replace(
            "These recordings were also acquired as a voltage series, so a class that "
            "lives in one half of one recording is confounded with elapsed time.",
            "A class concentrated in one half of one recording can be confounded with elapsed time.",
        ) for message in quant.warnings)
        table.loc[sub.index, "subpop"] = quant.fit.labels
        table.loc[sub.index, "subpop_confidence"] = quant.fit.confidence
        for k in range(quant.fit.k):
            table.loc[sub.index, f"posterior_{k}"] = quant.fit.posteriors[:, k]
        _appendix_population_table(quant, mixture.duration_s).to_csv(folder / "population_quantification.csv", index=False)
        pd.DataFrame(quant.selection.grid, index=pd.Index(quant.selection.p_grid, name="n_pcs"),
                     columns=pd.Index(quant.selection.k_grid, name="k")).to_csv(folder / "cv_grid.csv")
        pd.DataFrame(quant.ci.samples, columns=[f"subpop_{k}" for k in range(quant.fit.k)]).to_csv(folder / "fraction_bootstrap.csv", index=False)
        pd.DataFrame({"k": list(quant.fit.bic or {}), "bic": list((quant.fit.bic or {}).values())}).to_csv(folder / "bic_ladder.csv", index=False)
        if quant.crosstab is not None:
            quant.crosstab.to_csv(folder / "exploratory_crosstab.csv")
        # Explicit half-open windows for waveform kernel (stored event table uses inclusive end).
        # SR's waveform helper constructs an EventSet internally, whose schema is
        # wider than its two consumed sample columns. Leave unrelated measurements
        # undefined rather than relabel clustering depths as canonical amplitudes.
        windows = empty_event_frame().reindex(sub.index)
        for col in ("event_id", "recording", "pore_id", "start_idx", "end_idx"):
            windows[col] = sub[col]
        windows["end_idx"] += 1
        windows["t_start_s"] = sub.onset_s
        windows["dwell_s"] = sub.dwell_ms / 1000
        windows["accepted"] = True
        windows["flag"] = ""
        waves = SR.subpopulation_waveforms(mixture.raw, mixture.fs_hz, windows, quant.fit.labels)
        waves.to_csv(folder / "population_median_waveforms.csv", index=False)
        _write_json(folder / "population_median_waveforms_metadata.json", dict(n_events=waves.attrs.get("n_events", {}),
                    axis="duration-normalized; 200 points; 25% padding; raw current, unchanged CSV sign"))
        info.update(claim=quant.claim(), selection=quant.selection, separation=quant.separation,
                    n_events=quant.n_events, bootstrap=dict(n_ok=quant.ci.n_ok, n_boot=quant.ci.n_boot, seed=quant.ci.seed,
                        diagnostic_estimand="fitted mixture weights", warnings=list(quant.ci.warnings)),
                    fit=dict(k=quant.fit.k, n_pcs=quant.fit.n_pcs, means=quant.fit.means,
                             weights=quant.fit.weights, meta=quant.fit.meta, bic=quant.fit.bic),
                    warnings=list(quant.warnings))
        continuum = clusters.scopes.get(scope)
        if continuum is not None and continuum.n_hdbscan_clusters == 0:
            info["warnings"].insert(0, "HDBSCAN finds no density cluster in this recording; mixture components may partition a continuum.")
        thin = SR._gmm_note(len(sub), quant.fit.n_pcs, quant.fit.k)
        if thin:
            info["warnings"].insert(0, thin)
    table.to_csv(folder / "features_assignments.csv", index=False)
    _write_json(folder / "report.json", info)
    return table, quant, info


def _source_identity(module_paths: dict[str, Path], cancel=None) -> dict:
    """PYZ archives have virtual __file__ paths; the embedded build is their identity."""
    from ..build_info import build_info
    import sys
    hashes, unavailable = {}, []
    for name, path in module_paths.items():
        if path.is_file():
            hashes[name] = _sha256(path, cancel)
        else:
            unavailable.append(name)
    status = ("complete_source_files" if not unavailable else
              "unavailable_in_frozen_archive" if getattr(sys, "frozen", False) else
              "some_source_files_unavailable")
    record = build_info()
    allowed = ("schema_version", "version", "build_id", "built_at_utc", "git_commit", "git_dirty",
               "source_sha256", "bundle_version", "environment")
    return dict(source_module_sha256=hashes, source_module_hash_status=status,
                unavailable_module_sources=unavailable,
                build_record={key: record.get(key) for key in allowed})


def _methods() -> dict:
    return dict(reference_notebook_sha256=REFERENCE_NOTEBOOK_SHA256, reference_notebook_cells=54,
        prepared_input="CSV samples and explicit preprocessing metadata are authoritative; no raw-ABF preprocessing or catalog cache reuse",
        training_domain="Expected training recording zn_30h from original recipe; not inferred from arbitrary model filenames or mixture selection",
        candidate_population="Frozen student candidates; inclusive indices; sustained depth >=4 event-excluded MAD gate",
        single_recording_summary="Cell8 ungated candidates: full detrended trace median minus event min, rounded0.1pA; dwell rounded0.001ms",
        local_fraction="fracblock=f_sust=depth_sust_pA/abs(5s smoothed 0.1s quiet raw block medians at onset)",
        aggregation_fraction="cell28 code uses depth_pA=PEAK alias / abs(median(float32 raw)); despite sustained prose labels",
        kde_fraction="cell43 fresh detector fallback uses sustained/global float32 raw I0; five-panel cell48 uses preserved local fracblock",
        comparison_gate_floor="4 * metadata.qc_mad_noise_pA / abs(median(float32 raw)); retained separately from measured detection quiet MAD",
        rate_interval="20 contiguous equal time blocks; 4000 count-bootstrap draws; seed0; sensitivity10/20/40 blocks",
        gate_transfer="cell41 floors0.02:0.01:0.20; nearest-grid evaluation of native common floor recorded explicitly",
        original_clustering="12 features; global full-trace baseline/MAD and global float64 raw I0; no polarity reorientation; pooled PCA/HDBSCAN/kmeans3/tSNE seed0",
        quantitative_appendix="Separate existing NanoScope13 features including entropy; pooled and per-incubation scopes; selected recording full-covariance GMM; current recording-clock time halves; hard-assigned event fractions/rates and fitted mixture weights exported separately; fraction-bootstrap intervals apply to mixture weights",
        seeds="Version1 fixes seed0 for the original notebook and existing appendix; alternate seeds require another recipe version",
        interpretation="Descriptive recordings, not replicated incubation effects. Different pores and detector transfer confound rates and distributions; mixture components are not identified molecular species.")


def _report(spec, summary, metadata, figures) -> str:
    lines = [f"# {spec.name}", "", "Completed incubation study from prepared recordings and one frozen student model.", "",
             "## Recording summary", "", "| Recording | Incubation (h) | Events | Rate /s [95% CI] | Median dwell (ms) | Median sustained/local fraction |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.tag} | {row.incubation_h:g} | {row.n_events} | {row.rate_per_s:.4g} [{row.lo:.4g}, {row.hi:.4g}] | {row.dwell_median_ms:.4g} | {row.sustained_local_fracblock_median:.4g} |")
    lines += ["", "[Full recording summary](summary.csv) · [Accepted events](events.csv)", "", "## Measurement definitions", ""]
    for name, method in metadata["methods"].items():
        lines.append(f"- **{name}**: {method}")
    lines += ["", "## Source fidelity and interpretation", "",
              f"Frozen student model SHA256: `{metadata['model_identity']['sha256']}`. Every recording's inference provenance was checked against this initial content hash.", "",
              "The current 54-cell original notebook uses 12 clustering features. Older saved notebook outputs and NanoScope's existing quantitative appendix use 13, including entropy. These are exported to separate folders; old saved labels are never treated as current results.", "",
              "The aggregation section's code divides the peak alias by global I0 even though some captions call it sustained. This bundle preserves the code's values under explicitly peak/global labels, alongside sustained/local comparisons. No cached audit catalog is silently selected.", "",
              "Each pore/recording is the experimental unit. Block-bootstrap intervals describe within-recording counting variation and do not establish reproducibility across pores. No event-level significance p-values are reported.", "",
              f"Original pooled clustering: {metadata['original_clustering']['status']}.", "",
              "## NanoScope quantitative appendix", "", metadata["quantitative_appendix"].get("claim", metadata["quantitative_appendix"]["status"]), "",
              "Assigned event fractions and rates use hard labels. Fitted mixture weights have separate bootstrap intervals; the table and figure identify both quantities explicitly.", ""]
    lines += ["Coverage follows the original unpadded sliding-window policy. Predictions are zero in the uncovered tail:", ""]
    for row in summary.itertuples():
        lines.append(f"- {row.tag}: {row.uncovered_tail_samples:,} samples ({row.uncovered_tail_s:.6g} s) uncovered at the end; rate denominator remains the full recording duration as in the notebook.")
    lines.append("")
    for warning in metadata["quantitative_appendix"].get("warnings", []):
        lines.append(f"- {warning}")
    lines += ["", "## Outputs", "", "`recipe.json` reopens the editable recipe; `metadata.json` carries input provenance, methods and output hashes. `summary.csv` and `events.csv` are the recording summary and accepted candidate measurements. Per-recording `catalogs/` retain rejected candidates. `comparison/` contains rate intervals, common-floor sweeps, pair ratios and descriptive distances. `original_clustering/` and `quantitative_appendix/` remain distinct.", "", "## Figure gallery", ""]
    for path in figures:
        lines += [f"### {path.stem.replace('_', ' ')}", "", f"![{path.stem}]({path.as_posix()})", ""]
    return "\n".join(lines)


@propagate_cancel
def run_study(spec: StudySpec, *, output_dir: Path | str, progress: Progress | None = None,
              cancel: CancelFlag | None = None) -> StudyResult:
    """Run all study stages and publish one complete, immutable-by-default bundle.

    Existing nonempty output directories are refused. A hidden sibling staging
    directory is removed on any failure/cancellation. A sibling exclusive lock
    prevents two cooperating runners from publishing to the same destination.
    """
    check_cancelled(cancel)
    _validate_spec(spec)
    original_output = Path(output_dir).expanduser()
    if original_output.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    output = _absolute(original_output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.study.lock"
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Another study owns this destination lock: {lock}") from exc
    os.close(lock_fd)
    staging = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
        for name in ("catalogs", "comparison", "original_clustering", "quantitative_appendix", "figures"):
            (staging / name).mkdir()
        _write_json(staging / "recipe.json", _manifest(spec))
        from ..plot import incubation_figures as IF
        frames, feature_parts, summaries, provenance, figure_paths = [], [], [], {}, []
        mixture = None
        # Per-recording detection verifies its own file snapshot. The study also
        # needs one shared content identity, otherwise replacing the model between
        # recordings could produce a valid-looking comparison of different models.
        model_path = _absolute(spec.model_path)
        model_identity = {"path": str(model_path), "sha256": _sha256(model_path, cancel)}
        for i, rec in enumerate(spec.recordings):
            check_cancelled(cancel)
            def recording_progress(frac, msg, i=i):
                if progress:
                    progress(0.48 * (i + frac) / len(spec.recordings), msg)
            analyzed = analyze_recording(rec.trace_path, rec.metadata_path, spec.model_path,
                                         progress=recording_progress, cancel=cancel)
            observed_model = analyzed.provenance.get("model")
            if not isinstance(observed_model, dict) or observed_model.get("sha256") != model_identity["sha256"]:
                raise ValueError(
                    f"{rec.tag}: student model identity changed or is missing; "
                    "every recording must use the same frozen model captured at study start"
                )
            if analyzed.tag != rec.tag:
                raise ValueError(f"Manifest tag {rec.tag!r} does not match metadata tag {analyzed.tag!r}")
            if not np.isfinite(analyzed.duration_s) or analyzed.duration_s <= 0 or not np.isclose(analyzed.duration_s, len(analyzed.detrended) / analyzed.fs_hz):
                raise ValueError(f"{rec.tag}: inconsistent analysis duration")
            catalog = analyzed.catalog.copy().reset_index(drop=True)
            catalog["event_id"] = [f"{rec.tag}:{j}" for j in range(len(catalog))]
            catalog["incubation_h"], catalog["pore_nm"], catalog["pore_id"] = rec.incubation_h, rec.pore_diameter_nm, rec.pore_id
            catalog["condition"] = _condition(rec)
            catalog["concentration_pm"], catalog["voltage_mv"] = rec.concentration_pm, rec.voltage_mv
            i0 = abs(float(np.median(np.asarray(analyzed.raw, np.float32))))
            if not np.isfinite(i0) or i0 <= 0:
                raise ValueError(f"{rec.tag}: global open-pore current is not positive and finite")
            catalog["aggregation_fracblock_global"] = catalog.depth_peak_pA / i0
            catalog["f_sust_global"] = catalog.depth_sust_pA / i0
            catalog.to_csv(staging / "catalogs" / f"events_catalog_{rec.tag}.csv", index=False)
            _write_json(staging / "catalogs" / f"{rec.tag}_provenance.json", analyzed.provenance)
            accepted = catalog[catalog.passes_gate].copy()
            frames.append(accepted)
            feats = _recording_features(analyzed, rec, accepted)
            feature_parts.append(feats)
            meta = analyzed.provenance["metadata"]
            qc_mad = float(meta.get("qc_mad_noise_pA", float("nan")))
            if not np.isfinite(qc_mad) or qc_mad <= 0:
                raise ValueError(f"{rec.tag}: original study comparisons require positive finite metadata qc_mad_noise_pA")
            bb = block_bootstrap_rate(accepted.onset_s.to_numpy(), analyzed.duration_s)
            row = dict(tag=rec.tag, condition=_condition(rec), incubation_h=rec.incubation_h,
                       pore_nm=rec.pore_diameter_nm, pore_id=rec.pore_id, conc_pM=rec.concentration_pm,
                       voltage_mv=rec.voltage_mv, in_domain=rec.tag == REFERENCE_TRAINING_TAG,
                       n_candidates=len(catalog), n_events=len(accepted), n_feature_events=len(feats),
                       uncovered_tail_samples=int(analyzed.provenance["inference"]["uncovered_tail_samples"]),
                       uncovered_tail_s=int(analyzed.provenance["inference"]["uncovered_tail_samples"]) / analyzed.fs_hz,
                       duration_s=analyzed.duration_s, rate_per_s=bb["rate"], lo=bb["lo"], hi=bb["hi"], fano=bb["fano"],
                       I0_pA=i0, noise_pA=qc_mad, detection_noise_mad_pA=analyzed.provenance["noise_mad_pA"],
                       gate_frac_of_I0=4.0 * qc_mad / i0,
                       measured_gate_frac_global=4.0 * analyzed.provenance["noise_mad_pA"] / i0,
                       depth_median_pA=accepted.depth_peak_pA.median(), depth_sust_median_pA=accepted.depth_sust_pA.median(),
                       fracblock_median=accepted.aggregation_fracblock_global.median(),
                       sustained_local_fracblock_median=accepted.fracblock.median(), dwell_median_ms=accepted.dwell_ms.median())
            summaries.append(row)
            provenance[rec.tag] = analyzed.provenance
            figure_paths.extend(IF.recording_figures(analyzed, accepted, staging / "figures", cancel=cancel))
            if rec.tag == spec.mixture_tag:
                mixture = analyzed
            del analyzed
        check_cancelled(cancel)
        events = pd.concat(frames, ignore_index=True)
        features = pd.concat(feature_parts, ignore_index=True)
        summary = pd.DataFrame(summaries)
        summary.to_csv(staging / "summary.csv", index=False)
        events.to_csv(staging / "events.csv", index=False)
        events.to_csv(staging / "comparison" / "incubation_events_local_sustained.csv", index=False)
        legacy = events.copy()
        legacy["fracblock"] = legacy.aggregation_fracblock_global
        legacy.to_csv(staging / "comparison" / "incubation_comparison_events_peak_global.csv", index=False)
        comparison = _comparison_tables(events, summary)
        for name, table in comparison.items():
            table.to_csv(staging / "comparison" / f"{name}.csv", index=False)
        if progress:
            progress(0.50, "Rendering recording comparisons")
        figure_paths.extend(IF.comparison_figures(events, summary, comparison, staging / "figures", cancel=cancel))
        if progress:
            progress(0.56, "Original notebook: pooled 12-feature clustering")
        original, original_info = _original_clustering(features, staging / "original_clustering", cancel=cancel)
        original.to_csv(staging / "original_clustering" / "features_assignments.csv", index=False)
        _write_json(staging / "original_clustering" / "report.json", original_info)
        figure_paths.extend(IF.clustering_figures(original, staging / "figures", prefix="original12", cancel=cancel))
        check_cancelled(cancel)
        def appendix_progress(frac, msg):
            if progress:
                progress(0.62 + 0.32 * frac, f"NanoScope quantitative appendix: {msg}")
        assignment, quant, appendix_info = _appendix(features, mixture, spec, staging / "quantitative_appendix", progress=appendix_progress, cancel=cancel)
        figure_paths.extend(IF.appendix_figures(assignment, quant, mixture, staging / "quantitative_appendix", staging / "figures", cancel=cancel))
        check_cancelled(cancel)
        relative_figures = [p.relative_to(staging) for p in figure_paths]
        versions = {}
        for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tensorflow", "tensorflow-macos", "keras", "nanoscope"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not_installed_as_distribution"
        from . import incubation_detection as detection_module
        from ..plot import incubation_galleries as gallery_module
        source_modules = [Path(__file__), Path(EF.__file__), Path(SC.__file__), Path(SG.__file__), Path(SR.__file__),
                          Path(detection_module.__file__), Path(IF.__file__), Path(gallery_module.__file__)]
        source_identity = _source_identity({p.relative_to(Path(__file__).parents[1]).as_posix(): p for p in source_modules}, cancel)
        metadata = dict(**source_identity, format="nanoscope-incubation-result", version=BUNDLE_VERSION, status="complete",
                        name=spec.name, created_utc=datetime.now(timezone.utc).isoformat(), recipe=_manifest(spec),
                        methods=_methods(), model_identity=model_identity,
                        input_provenance=provenance, package_versions=versions,
                        original_clustering=original_info, quantitative_appendix=appendix_info,
                        figure_paths=[p.as_posix() for p in relative_figures])
        metadata = _jsonable(metadata)
        (staging / "report.md").write_text(_report(spec, summary, metadata, relative_figures), encoding="utf-8")
        metadata["output_sha256"] = {p.relative_to(staging).as_posix(): _sha256(p, cancel) for p in sorted(staging.rglob("*")) if p.is_file()}
        _write_json(staging / "metadata.json", metadata)
        check_cancelled(cancel)
        if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
            raise FileExistsError("Output destination changed or became nonempty while the study ran")
        os.replace(staging, output)
        staging = None
        # Publication is the commit point. Do not report cancellation after it.
        if progress:
            progress(1.0, f"Completed study: {len(events):,} accepted events")
        return StudyResult(output, summary, events, metadata, [output / p for p in relative_figures])
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def load_study_result(output_dir: Path | str) -> StudyResult:
    """Reopen a complete bundle, verify table/figure hashes, without raw data/model."""
    output = _absolute(output_dir)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "nanoscope-incubation-result" or metadata.get("version") != BUNDLE_VERSION or metadata.get("status") != "complete":
        raise ValueError("Not a complete supported incubation-study result")
    inventory = metadata.get("output_sha256", {})
    if not {"summary.csv", "events.csv", "report.md", "recipe.json"}.issubset(inventory):
        raise ValueError("Study result is missing its required output inventory")
    for relative, digest in inventory.items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Study result is missing or changed: {relative}")
    figures = []
    for relative in metadata.get("figure_paths", []):
        path = (output / relative).resolve()
        if relative not in inventory or not path.is_relative_to(output):
            raise ValueError("Figure path is outside the verified study bundle")
        figures.append(path)
    return StudyResult(output, pd.read_csv(output / "summary.csv", float_precision="round_trip"),
                       pd.read_csv(output / "events.csv", float_precision="round_trip"), metadata, figures)
