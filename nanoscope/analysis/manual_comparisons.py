"""Independent analyses over explicitly selected, reviewed event results.

These operations do not load recordings, invoke a detector, or draw figures. Each
consumes the preceding committed result and records its own effective settings.
The incubation preset uses these same scientific definitions; callers can change
settings deliberately without changing upstream events.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import event_features as EF, subpop_cluster as SC, subpop_gmm as SG, subpop_report as SR
from .cancel import check_cancelled
from .events import empty_df as empty_event_frame
from .incubation_detection import RecordingAnalysis
from .incubation_study import (NOTEBOOK_FEATURES, StudyRecording, _appendix_population_table,
                              _appendix_weight_warnings, _condition, _recording_features,
                              _write_json, block_bootstrap_rate)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _features(columns):
    if not columns or len(set(columns)) != len(columns) or any(c not in EF.CLUST_FEATS for c in columns):
        raise ValueError("feature_columns must contain unique supported waveform features")
    return tuple(columns)


@dataclass(frozen=True)
class ComparisonSettings:
    n_blocks: int = 20
    n_boot: int = 4000
    seed: int = 0
    floor_start: float = .02
    floor_stop: float = .20
    floor_step: float = .01
    sensitivity_blocks: tuple[int, ...] = (10, 20, 40)

    def __post_init__(self):
        _integer(self.n_blocks, "n_blocks", 2)
        _integer(self.n_boot, "n_boot", 1)
        _integer(self.seed, "seed")
        for n in self.sensitivity_blocks:
            _integer(n, "sensitivity_blocks", 2)
        if len(set(self.sensitivity_blocks)) != len(self.sensitivity_blocks):
            raise ValueError("sensitivity_blocks must not contain duplicates")
        object.__setattr__(self, "sensitivity_blocks", tuple(self.sensitivity_blocks))
        if not all(np.isfinite(v) for v in (self.floor_start, self.floor_stop, self.floor_step)):
            raise ValueError("floor range must be finite")
        if not 0 <= self.floor_start <= self.floor_stop or self.floor_step <= 0:
            raise ValueError("floor range must be nonnegative and increasing, with positive step")
        if (self.floor_stop - self.floor_start) / self.floor_step > 10000:
            raise ValueError("floor range exceeds 10001 points")


@dataclass
class ComparisonResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    metadata: dict


@dataclass(frozen=True)
class ClusterSettings:
    seed: int = 0
    kmeans_k: int = 3
    feature_columns: tuple[str, ...] = NOTEBOOK_FEATURES
    pca_variance: float = .95
    max_pcs: int = 10
    hdbscan_min_samples: int = 10
    with_tsne: bool = True

    def __post_init__(self):
        for value, name, minimum in ((self.seed, "seed", 0), (self.kmeans_k, "kmeans_k", 1),
                                     (self.max_pcs, "max_pcs", 1), (self.hdbscan_min_samples, "hdbscan_min_samples", 1)):
            _integer(value, name, minimum)
        object.__setattr__(self, "feature_columns", _features(self.feature_columns))
        if not np.isfinite(self.pca_variance) or not 0 < self.pca_variance <= 1:
            raise ValueError("pca_variance must be in (0, 1]")
        if not isinstance(self.with_tsne, bool):
            raise ValueError("with_tsne must be boolean")


@dataclass(frozen=True)
class MixtureSettings:
    min_events: int = 0
    seed: int = 0
    n_boot: int = 200
    cv_folds: int = 5
    bootstrap_seed: int = 0
    p_grid: tuple[int, ...] | None = None
    k_grid: tuple[int, ...] = tuple(range(1, 9))
    feature_columns: tuple[str, ...] = EF.CLUST_FEATS

    def __post_init__(self):
        for value, name, minimum in ((self.seed, "seed", 0), (self.n_boot, "n_boot", 1),
                                     (self.cv_folds, "cv_folds", 2), (self.bootstrap_seed, "bootstrap_seed", 0)):
            _integer(value, name, minimum)
        _integer(self.min_events, "min_events", 0)
        columns = _features(self.feature_columns)
        if len(columns) < 2:
            raise ValueError("Mixture fitting needs at least two features")
        object.__setattr__(self, "feature_columns", columns)
        for name in ("p_grid", "k_grid"):
            values = getattr(self, name)
            if values is None and name == "p_grid":
                continue
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct positive integers")
            for value in values:
                _integer(value, name, 1)
                if name == "p_grid" and value > len(columns):
                    raise ValueError("p_grid cannot exceed the selected feature count")
            object.__setattr__(self, name, tuple(values))


def _notify(progress, fraction, message):
    if progress:
        progress(fraction, message)


def _inputs(analyses, recordings):
    analyses, recordings = list(analyses), list(recordings)
    if not analyses or len(analyses) != len(recordings):
        raise ValueError("Select matching recording analyses and conditions")
    if len({a.tag for a in analyses}) != len(analyses):
        raise ValueError("Recording analyses must have distinct source tags")
    for a, r in zip(analyses, recordings):
        if a.tag != r.tag:
            raise ValueError("Recording conditions do not match selected analysis identities")
        if not np.isfinite(a.fs_hz) or a.fs_hz <= 0 or not np.isfinite(a.duration_s) or a.duration_s <= 0:
            raise ValueError(f"{a.tag}: sample rate and duration must be positive and finite")
        if len(a.raw) != len(a.detrended) or not np.isclose(a.duration_s, len(a.detrended) / a.fs_hz):
            raise ValueError(f"{a.tag}: inconsistent recording clock")
    return list(zip(analyses, recordings))


def reviewed_catalog(analysis: RecordingAnalysis, recording: StudyRecording) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Keep catalog identity and user acceptance; derive explicit global fractions."""
    catalog = analysis.catalog.copy().reset_index(drop=True)
    selection = "accepted" if "accepted" in catalog else "passes_gate"
    if selection not in catalog:
        raise ValueError(f"{analysis.tag}: catalog needs an explicit acceptance decision")
    if catalog[selection].isna().any() or not catalog[selection].isin([True, False]).all():
        raise ValueError(f"{analysis.tag}: {selection} must contain boolean decisions")
    needed = {"start_idx", "end_idx", "onset_s", "end_s", "depth_peak_pA", "depth_sust_pA", "dwell_ms", "fracblock"}
    if not needed.issubset(catalog):
        raise ValueError(f"{analysis.tag}: missing measured columns {sorted(needed-set(catalog))}")
    bounds = catalog[["start_idx", "end_idx"]].to_numpy(float)
    if not np.isfinite(bounds).all() or np.any(bounds != np.floor(bounds)) or np.any(bounds < 0):
        raise ValueError(f"{analysis.tag}: candidate bounds must be finite sample indices")
    if len(bounds) and (np.any(bounds[:, 1] < bounds[:, 0]) or np.any(bounds[:, 1] >= len(analysis.detrended))):
        raise ValueError(f"{analysis.tag}: candidate bounds fall outside the recording")
    if not np.allclose(catalog.onset_s.to_numpy(float), bounds[:, 0] / analysis.fs_hz, rtol=0, atol=1e-12):
        raise ValueError(f"{analysis.tag}: event onset clock disagrees with sample indices")
    if "event_id" not in catalog:
        catalog["event_id"] = [f"{recording.tag}:{i}" for i in range(len(catalog))]
    if catalog.event_id.isna().any() or catalog.event_id.duplicated().any():
        raise ValueError(f"{analysis.tag}: event IDs must be present and unique within the source")
    catalog["tag"] = recording.tag
    catalog["incubation_h"], catalog["pore_nm"], catalog["pore_id"] = recording.incubation_h, recording.pore_diameter_nm, recording.pore_id
    catalog["condition"] = _condition(recording)
    catalog["concentration_pm"], catalog["voltage_mv"] = recording.concentration_pm, recording.voltage_mv
    i0 = abs(float(np.median(np.asarray(analysis.raw, np.float32))))
    if not np.isfinite(i0) or i0 <= 0:
        raise ValueError(f"{analysis.tag}: global open-pore current must be positive and finite")
    catalog["aggregation_fracblock_global"] = catalog.depth_peak_pA / i0
    catalog["f_sust_global"] = catalog.depth_sust_pA / i0
    return catalog, catalog.loc[catalog[selection].astype(bool)].copy(), i0


def _comparison_tables(events, summary, settings, cancel):
    # Decimal rounding preserves the reference .02:.01:.20 grid exactly.
    count = int(np.floor((settings.floor_stop-settings.floor_start) / settings.floor_step + 1e-10)) + 1
    floors = np.round(settings.floor_start + settings.floor_step * np.arange(count), 12)
    sweep, sensitivity, pairs, distances = [], [], [], []
    for row in summary.itertuples():
        check_cancelled(cancel)
        ev = events[events.tag == row.tag]
        for blocks in settings.sensitivity_blocks:
            check_cancelled(cancel)
            sensitivity.append(dict(tag=row.tag, **block_bootstrap_rate(ev.onset_s, row.duration_s,
                n_blocks=blocks, n_boot=settings.n_boot, seed=settings.seed)))
        for definition, column in (("notebook_cell28_peak_global", "aggregation_fracblock_global"),
                                   ("sustained_local", "fracblock"), ("kde_sustained_global", "f_sust_global")):
            for floor in floors:
                n = int((ev[column] >= floor).sum())
                sweep.append(dict(tag=row.tag, condition=row.condition, definition=definition,
                                  floor=float(floor), n=n, rate=n/row.duration_s))
    eligible = summary[summary.n_events >= 50]
    common = float(eligible.gate_frac_of_I0.max()) if len(eligible) else float("nan")
    for a, b in itertools.combinations(summary.itertuples(), 2):
        check_cancelled(cancel)
        native = max(a.gate_frac_of_I0, b.gate_frac_of_I0)
        grid = float(floors[np.argmin(np.abs(floors-native))])
        ea, eb = events[events.tag == a.tag], events[events.tag == b.tag]
        for definition, column in (("notebook_cell28_peak_global", "aggregation_fracblock_global"), ("sustained_local", "fracblock")):
            na, nb = int((ea[column] >= grid).sum()), int((eb[column] >= grid).sum())
            ra, rb = na/a.duration_s, nb/b.duration_s
            pairs.append(dict(tag_a=a.tag, tag_b=b.tag, definition=definition, native_common_floor=native,
                evaluated_floor=grid, n_a=na, n_b=nb, rate_a=ra, rate_b=rb, ratio_a_over_b=ra/rb if rb else float("nan"),
                denominator_zero=rb == 0, fewer_than_10_in_an_arm=min(na,nb) < 10,
                size_matched=a.pore_nm == b.pore_nm, same_pore_id=a.pore_id == b.pore_id))
        if min(a.n_events,b.n_events) < 50 or not np.isfinite(common):
            continue
        aa, bb = ea[ea.aggregation_fracblock_global >= common], eb[eb.aggregation_fracblock_global >= common]
        for column in ("dwell_ms", "aggregation_fracblock_global"):
            valid = min(len(aa),len(bb)) >= 20
            distances.append(dict(tag_a=a.tag, tag_b=b.tag, variable=column, common_floor=common,
                n_a=len(aa), n_b=len(bb), status="computed" if valid else "fewer_than_20_above_floor",
                ks_distance=float(ks_2samp(aa[column],bb[column]).statistic) if valid else float("nan"),
                median_a=aa[column].median(), median_b=bb[column].median()))
    return dict(gate_transfer_sweep=pd.DataFrame(sweep), rate_block_sensitivity=pd.DataFrame(sensitivity,
        columns=["tag","rate","lo","hi","fano","n_blocks","n_boot","seed"]),
        gate_transfer_pairs=pd.DataFrame(pairs, columns=["tag_a","tag_b","definition","native_common_floor","evaluated_floor","n_a","n_b","rate_a","rate_b","ratio_a_over_b","denominator_zero","fewer_than_10_in_an_arm","size_matched","same_pore_id"]),
        distribution_distances=pd.DataFrame(distances, columns=["tag_a","tag_b","variable","common_floor","n_a","n_b","status","ks_distance","median_a","median_b"]))


def compare_recordings(analyses, recordings, settings=ComparisonSettings(), *, folder, progress=None, cancel=None):
    selected = _inputs(analyses, recordings)
    frames, summaries, policies = [], [], {}
    for i, (a,r) in enumerate(selected):
        check_cancelled(cancel)
        catalog, accepted, i0 = reviewed_catalog(a,r)
        frames.append(accepted)
        policies[r.tag] = "accepted" if "accepted" in a.catalog else "passes_gate"
        metadata = a.provenance.get("metadata", {})
        qc = float(metadata.get("qc_mad_noise_pA", float("nan")))
        measured = float(a.provenance.get("noise_mad_pA", float("nan")))
        measurement = a.provenance.get("measurement_settings", a.provenance.get("contract", {}))
        sigma = float(measurement.get("min_depth_sigma", 4.))
        if not all(np.isfinite(v) and v > 0 for v in (qc, measured)) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"{r.tag}: positive QC/measured noise and nonnegative gate sigma are required")
        bb = block_bootstrap_rate(accepted.onset_s, a.duration_s, settings.n_blocks, settings.n_boot, settings.seed)
        tail = int(a.provenance.get("inference", {}).get("uncovered_tail_samples", 0))
        summaries.append(dict(tag=r.tag, condition=_condition(r), incubation_h=r.incubation_h,
            pore_nm=r.pore_diameter_nm, pore_id=r.pore_id, conc_pM=r.concentration_pm, voltage_mv=r.voltage_mv,
            n_candidates=len(catalog), n_events=len(accepted),
            n_feature_events=int(((accepted.end_idx-accepted.start_idx+1) >= EF.MIN_EVENT_SAMPLES).sum()),
            uncovered_tail_samples=tail, uncovered_tail_s=tail/a.fs_hz, duration_s=a.duration_s,
            rate_per_s=bb["rate"], lo=bb["lo"], hi=bb["hi"], fano=bb["fano"], I0_pA=i0,
            noise_pA=qc, detection_noise_mad_pA=measured, gate_frac_of_I0=sigma*qc/i0,
            measured_gate_frac_global=sigma*measured/i0, depth_median_pA=accepted.depth_peak_pA.median(),
            depth_sust_median_pA=accepted.depth_sust_pA.median(), fracblock_median=accepted.aggregation_fracblock_global.median(),
            sustained_local_fracblock_median=accepted.fracblock.median(), dwell_median_ms=accepted.dwell_ms.median()))
        _notify(progress, .5*(i+1)/len(selected), f"Computed rates for {r.tag}")
    events, summary = pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)
    events.attrs["selection_columns"] = dict(policies)
    summary.attrs["selection_columns"] = dict(policies)
    tables = _comparison_tables(events, summary, settings, cancel)
    info = dict(operation="condition_comparisons", settings=asdict(settings), selection_columns=policies,
        native_gate="gate sigma * recorded QC noise / absolute median float32 raw current",
        pair_floor="nearest sweep grid to each pair's maximum native gate; both values exported",
        distance_floor="exact maximum native gate among sources with >=50 accepted events; >=20 above-floor events per arm",
        input_tags=[a.tag for a,_ in selected], rate_interval="95% time-block count bootstrap; within-recording uncertainty")
    check_cancelled(cancel)
    out = Path(folder); out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out/"events.csv", index=False); summary.to_csv(out/"summary.csv", index=False)
    events.to_csv(out/"incubation_events_local_sustained.csv", index=False)
    legacy = events.copy(); legacy["fracblock"] = legacy.aggregation_fracblock_global
    legacy.to_csv(out/"incubation_comparison_events_peak_global.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out/f"{name}.csv", index=False)
    _write_json(out/"report.json", info)
    _notify(progress, 1., "Condition comparisons complete")
    return ComparisonResult(events, summary, tables, info)


def feature_orientation_issue(features, analyses):
    """Identify saved upward-source features from the pre-correction convention.

    Downward-only historical results are numerically compatible. The caller can
    preserve an affected checkpoint while asking for feature re-extraction.
    """
    if features is None or features.attrs.get("methods", {}).get("feature_orientation_version") == 2:
        return None
    tags = set(features["tag"]) if "tag" in features else None
    affected = sorted({a.tag for a in analyses if a is not None and
        a.provenance.get("event_polarity") == -1 and (tags is None or a.tag in tags)})
    if not affected:
        return None
    return ("Saved waveform features predate the upward-event orientation correction for " +
            ", ".join(affected) + ". Extract waveform features again, then rerun clustering or mixture fitting.")


def extract_features(analyses, recordings, *, progress=None, cancel=None):
    selected = _inputs(analyses, recordings)
    frames = []
    for i, (a,r) in enumerate(selected):
        check_cancelled(cancel)
        _, accepted, _ = reviewed_catalog(a,r)
        frames.append(_recording_features(a,r,accepted,events_down=True))
        _notify(progress, (i+1)/len(selected), f"Measured waveform features for {r.tag}")
    check_cancelled(cancel)
    result = pd.concat(frames, ignore_index=True)
    result.attrs["methods"] = dict(operation="waveform_features", feature_columns=EF.CLUST_FEATS,
        baseline="whole detrended trace median", noise="whole detrended trace MAD",
        i0="absolute whole raw float64 trace median", time_half="recording-clock midpoint",
        feature_orientation="events_down; retain manual preparation orientation",
        feature_orientation_version=2,
        amplitude="full_trace_median_minus_events_down_detrended; global_float64_raw_I0",
        source_event_polarities={a.tag: int(a.provenance["event_polarity"]) for a,_ in selected},
        selection_columns={a.tag: "accepted" if "accepted" in a.catalog else "passes_gate" for a,_ in selected})
    return result


def cluster_features(features, settings=ClusterSettings(), *, folder, progress=None, cancel=None):
    check_cancelled(cancel)
    columns = settings.feature_columns
    if not set(columns).issubset(features):
        raise ValueError("Selected feature columns have not been measured")
    # Derived clustering fields are replaced, not carried forward from a prior run.
    table = features.drop(columns=[c for c in features if c == "entropy" and c not in columns or
        c.startswith(("PC", "tsne_", "hdbscan", "kmeans"))], errors="ignore").copy()
    info = dict(operation="pooled_waveform_clustering", feature_columns=columns, settings=asdict(settings),
        random_state=settings.seed, preprocessing="winsorize1_99_then_median_IQR; nonfinite_scaled_to0",
        amplitude=features.attrs.get("methods", {}).get("amplitude", "input feature amplitudes; orientation not recorded"),
        feature_methods=deepcopy(features.attrs.get("methods", {})), input_rows=len(table), status="computed")
    out = Path(folder)
    if len(table) < max(20,settings.hdbscan_min_samples,settings.kmeans_k):
        info.update(status="insufficient_events", reason="Fewer events than selected cluster-size/sample requirements")
    else:
        _notify(progress,.05,"Projecting selected waveform features")
        projection = SC.scale_and_pca(table, columns=columns, var_target=settings.pca_variance,
                                     max_components=settings.max_pcs, random_state=settings.seed)
        check_cancelled(cancel)
        table["hdbscan"] = SC.hdbscan_labels(projection.scores, min_samples=settings.hdbscan_min_samples)
        check_cancelled(cancel)
        # A generic name records the actual k. kmeans3 exists only for the k=3 reference.
        label_name = f"kmeans{settings.kmeans_k}"
        table[label_name] = SC.kmeans_labels(projection.scores, k=settings.kmeans_k, random_state=settings.seed)
        for i in range(projection.scores.shape[1]):
            table[f"PC{i+1}"] = projection.scores[:,i]
        out.mkdir(parents=True, exist_ok=True)
        SC.loading_table(projection).to_csv(out/"loadings.csv", index=False)
        pd.DataFrame(dict(pc=np.arange(len(projection.explained_variance_ratio))+1,
            explained_variance_ratio=projection.explained_variance_ratio)).to_csv(out/"variance.csv", index=False)
        X = table[list(columns)].to_numpy(float); X = np.where(np.isfinite(X), X, np.nan)
        lo,hi = np.nanpercentile(X,[1,99],axis=0); clipped=np.clip(X,lo,hi)
        _write_json(out/"transform.json", dict(features=columns,winsor_lo=lo,winsor_hi=hi,
            center=np.nanmedian(clipped,axis=0),iqr=np.nanpercentile(clipped,75,axis=0)-np.nanpercentile(clipped,25,axis=0),
            components=projection.components,explained_variance_ratio=projection.explained_variance_ratio,
            n_kept=projection.n_kept,pca_metadata=projection.meta))
        for label in ("hdbscan",label_name):
            if "incubation_h" in table:
                ct=pd.crosstab(table[label],table.incubation_h)
                ct.to_csv(out/f"{label}_composition_counts.csv")
                ct.div(ct.sum(axis=1),axis=0).to_csv(out/f"{label}_composition_fractions.csv")
            table.groupby(label)[list(columns)].median().to_csv(out/f"{label}_feature_medians.csv")
        info.update(n_pcs=projection.n_kept,pca_metadata=projection.meta,kmeans_column=label_name,
            hdbscan_clusters=int(len(set(table.hdbscan)-{-1})), hdbscan_noise_fraction=float((table.hdbscan < 0).mean()))
        _notify(progress,.7,"Computed PCA and exploratory labels")
        check_cancelled(cancel)
        if settings.with_tsne:
            refusal=SC.embedding_refusal(projection.scores,kind="t-SNE",perplexity=SC.tsne_perplexity(len(table)))
            if refusal:
                info["tsne_status"]=refusal
            else:
                coords=SC.tsne_embedding(projection.scores,random_state=settings.seed)
                table["tsne_x"],table["tsne_y"]=coords[:,0],coords[:,1]
                info["tsne_status"]="computed; qualitative coordinates, not separation measurements"
        else:
            info["tsne_status"]="not requested"
    check_cancelled(cancel)
    out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"features_assignments.csv",index=False); _write_json(out/"report.json",info)
    _notify(progress,1.,"Pooled clustering complete")
    return table,info


def fit_mixture(features, analysis, settings=MixtureSettings(), *, folder, progress=None, cancel=None):
    """Fit the selected source and report compatible scoped exploratory comparisons."""
    check_cancelled(cancel)
    if not set((*settings.feature_columns,"scope","tag")).issubset(features):
        raise ValueError("Mixture fitting requires measured waveform features with source identities")
    if analysis.tag not in set(features.tag):
        raise ValueError("The mixture recording has no measured feature rows")
    if not np.isfinite(analysis.duration_s) or analysis.duration_s <= 0:
        raise ValueError("Mixture duration must be positive and finite")
    table=features.drop(columns=[c for c in features if c.startswith(("hdbscan","kmeans","posterior_")) or
        c in ("subpop","subpop_confidence")],errors="ignore").copy()
    out=Path(folder); out.mkdir(parents=True,exist_ok=True)
    info=dict(operation="waveform_mixture",feature_columns=settings.feature_columns,settings=asdict(settings),
        feature_methods=deepcopy(features.attrs.get("methods", {})),
        mixture_tag=analysis.tag,random_state=settings.seed,bootstrap_seed=settings.bootstrap_seed,
        time_half="recording_clock_midpoint; legacy_onset_median retained separately",status="computed",warnings=[],
        fraction_definitions=dict(assigned_event_fraction="Hard-label count / selected source's measurable accepted events",
        assigned_event_rate_per_s="Hard-label count / analyzed duration",mixture_weight="Fitted soft mixture weight",
        mixture_weight_interval="Event-bootstrap refitted mixture weights matched by component means, not hard-label fractions"))
    clusters=SC.cluster_scopes(table,columns=settings.feature_columns,scope_column="scope",random_state=settings.seed,
        cancel=cancel,progress=(lambda f,m:_notify(progress,.12*f,m)))
    labels=clusters.label_table()
    for col in labels:
        table[col]=labels[col]
    for scope,result in clusters.scopes.items():
        result.loadings.to_csv(out/f"loadings_{scope}.csv",index=False)
        _write_json(out/f"projection_{scope}.json",dict(components=result.pca.components,
            explained_variance_ratio=result.pca.explained_variance_ratio,metadata=result.pca.meta,
            n_hdbscan_clusters=result.n_hdbscan_clusters,hdbscan_noise_fraction=result.hdbscan_noise_fraction,
            hdbscan_silhouette=result.hdbscan_silhouette,kmeans3_silhouette=result.kmeans3_silhouette))
    info["skipped_scopes"]=clusters.skipped
    sub=table[table.tag==analysis.tag]; quant=None
    if len(sub)<max(SR.MIN_GMM_EVENTS,settings.min_events):
        info.update(status="mixture_insufficient_events",warnings=[f"Mixture has {len(sub)} events; requires at least {max(SR.MIN_GMM_EVENTS,settings.min_events)}."])
    else:
        scope=str(sub.scope.iloc[0]); name=f"hdbscan_{scope}"
        exploratory=labels.loc[sub.index,name] if name in labels else None
        check_cancelled(cancel)
        quant=SG.quantify_subpopulations(sub,duration_s=analysis.duration_s,features=settings.feature_columns,
            exploratory_labels=exploratory,exploratory_name=name,random_state=settings.seed,
            bootstrap_seed=settings.bootstrap_seed,n_boot=settings.n_boot,n_splits=settings.cv_folds,
            p_grid=settings.p_grid,k_grid=settings.k_grid,cancel=cancel,
            progress=(lambda f,m:_notify(progress,.12+.88*f,m)))
        _appendix_weight_warnings(quant)
        quant.warnings=tuple(m.replace("These recordings were also acquired as a voltage series, so a class that "
            "lives in one half of one recording is confounded with elapsed time.",
            "A class concentrated in one half of one recording can be confounded with elapsed time.") for m in quant.warnings)
        table.loc[sub.index,"subpop"]=quant.fit.labels
        table.loc[sub.index,"subpop_confidence"]=quant.fit.confidence
        for k in range(quant.fit.k):
            table.loc[sub.index,f"posterior_{k}"]=quant.fit.posteriors[:,k]
        _appendix_population_table(quant,analysis.duration_s).to_csv(out/"population_quantification.csv",index=False)
        pd.DataFrame(quant.selection.grid,index=pd.Index(quant.selection.p_grid,name="n_pcs"),
            columns=pd.Index(quant.selection.k_grid,name="k")).to_csv(out/"cv_grid.csv")
        pd.DataFrame(quant.ci.samples,columns=[f"subpop_{k}" for k in range(quant.fit.k)]).to_csv(out/"fraction_bootstrap.csv",index=False)
        pd.DataFrame(dict(k=list(quant.fit.bic or {}),bic=list((quant.fit.bic or {}).values()))).to_csv(out/"bic_ladder.csv",index=False)
        if quant.crosstab is not None:
            quant.crosstab.to_csv(out/"exploratory_crosstab.csv")
        windows=empty_event_frame().reindex(sub.index)
        for col in ("event_id","recording","pore_id","start_idx","end_idx"):
            windows[col]=sub[col]
        windows["end_idx"]+=1; windows["t_start_s"]=sub.onset_s; windows["dwell_s"]=sub.dwell_ms/1000
        windows["accepted"]=True; windows["flag"]=""
        waves=SR.subpopulation_waveforms(analysis.raw,analysis.fs_hz,windows,quant.fit.labels)
        waves.to_csv(out/"population_median_waveforms.csv",index=False)
        _write_json(out/"population_median_waveforms_metadata.json",dict(n_events=waves.attrs.get("n_events",{}),
            axis="duration-normalized; 200 points; 25% padding; raw current, unchanged input sign"))
        info.update(claim=quant.claim(),selection=quant.selection,separation=quant.separation,n_events=quant.n_events,
            bootstrap=dict(n_ok=quant.ci.n_ok,n_boot=quant.ci.n_boot,seed=quant.ci.seed,diagnostic_estimand="fitted mixture weights",warnings=list(quant.ci.warnings)),
            fit=dict(k=quant.fit.k,n_pcs=quant.fit.n_pcs,means=quant.fit.means,weights=quant.fit.weights,meta=quant.fit.meta,bic=quant.fit.bic),warnings=list(quant.warnings))
        continuum=clusters.scopes.get(scope)
        if continuum is not None and continuum.n_hdbscan_clusters==0:
            info["warnings"].insert(0,"HDBSCAN finds no density cluster in this recording; mixture components may partition a continuum.")
        thin=SR._gmm_note(len(sub),quant.fit.n_pcs,quant.fit.k)
        if thin:
            info["warnings"].insert(0,thin)
    check_cancelled(cancel)
    table.to_csv(out/"features_assignments.csv",index=False); _write_json(out/"report.json",info)
    _notify(progress,1.,"Mixture fitting complete")
    return table,quant,info
