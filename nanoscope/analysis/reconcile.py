"""Deterministic, overlap-aware reconciliation of event detector results.

Reconciliation is deliberately a *derived* result. The Schmitt, CUSUM and Keras
tables remain untouched, while this module builds a fourth :class:`AnalysisResult`
whose rows are the temporal union (or an agreement-filtered subset) of those tables.

The frozen event measurements are never averaged. A consensus boundary assembled
from several detectors would need to be measured again against one particular signal
and baseline contract; silently combining the bounds from one row with the depth from
another would create a physically impossible table. Instead, one complete contributor
row is selected deterministically and the other rows are recorded as provenance beside
it. A viable (not automatically rejected) measurement wins first; among viable rows, a
hand-drawn boundary wins.

Matching uses half-open sample spans and the overlap coefficient::

    intersection / min(duration_a, duration_b)

This treats a slightly wider detector span as the same event when it contains the
shorter one. Candidate links are considered from strongest to weakest and joined only
when doing so would not put two automatic rows from the same detector into one cluster,
and when every member of the resulting cluster independently meets the overlap rule.
This complete-linkage guard prevents a broad candidate from transitively bridging two
adjacent events. One manual and one automatic row from the same table *may* join: that
is the duplicate the reconciliation workflow exists to remove.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.analysis import AnalysisResult, OverlayHint, OverlayKind
from .events import (
    ANALYSED_START_KEY,
    ANALYSED_STOP_KEY,
    CLIPPED_COUNT_KEY,
    DERIVED_FLAGS,
    REVIEW_COL,
    REVIEW_DONE,
    SOURCE_COL,
    SOURCE_MANUAL,
    REVIEW_VALUES,
    EventSet,
    combine_flags,
    combine_reviews,
    combine_sources,
    empty_df,
    flag_tokens,
    is_automatic_rejection,
    n_clipped_boundary,
    n_filter_limited,
)
from .labels import label_conflicts
from .scoring import ADDED_COLUMN, added_mask

__all__ = [
    "CONSENSUS_MODULE_ID",
    "CONSENSUS_CONTRIBUTORS_COL",
    "CONSENSUS_CANONICAL_SOURCE_COL",
    "CONSENSUS_CANONICAL_MODULE_COL",
    "CONSENSUS_LABEL_CONFLICTS_COL",
    "CONSENSUS_CURATION_CONFLICT_COL",
    "CONSENSUS_SUPPORT_COL",
    "CONSENSUS_MEMBER_COUNT_COL",
    "MAX_PAIR_COMPARISONS",
    "MAX_MATCH_EDGES",
    "ReconcileConfig",
    "ReconcileOutcome",
    "clear_curation_conflicts",
    "reconciliation_source_fingerprint",
    "reconcile_detections",
]


# The public name says what the result *is* to callers; the stored module id says how it
# was made. Keep the constant in this module so the GUI, source discovery and project
# restore never invent competing spellings.
CONSENSUS_MODULE_ID = "detect.reconciled"
CONSENSUS_CONTRIBUTORS_COL = "consensus_contributors"
CONSENSUS_CANONICAL_SOURCE_COL = "consensus_canonical_source"
CONSENSUS_CANONICAL_MODULE_COL = "consensus_canonical_module"
CONSENSUS_LABEL_CONFLICTS_COL = "consensus_label_conflicts"
CONSENSUS_CURATION_CONFLICT_COL = "consensus_curation_conflict"
CONSENSUS_SUPPORT_COL = "consensus_support"
CONSENSUS_MEMBER_COUNT_COL = "consensus_members"

_VERSION = "1.0"
_CONTRIBUTOR_SEP = "|"

# A normal event stream is sparse: at any start sample only the corresponding rows from
# the other one or two detectors are active. A malformed table with thousands of spans
# all covering the same instant is qualitatively different -- every interval overlaps
# every other one and an edge materialisation becomes quadratic. Preview runs in the UI,
# so fail with an actionable message before such a table can consume gigabytes or look
# hung. The edge cap is lower because each ranked edge carries two stable tie-break keys;
# 100k still permits over three candidate links per row in a 30k-row reconciliation.
MAX_PAIR_COMPARISONS = 1_000_000
MAX_MATCH_EDGES = 100_000


def clear_curation_conflicts(df: pd.DataFrame, rows: Iterable[int]) -> int:
    """Resolve contradictory human votes for the explicitly curated positional rows.

    Returns the number of markers cleared so GUI callers can checkpoint a resolution even
    when the final boolean already matched the explicit Accept/Reject gesture. Ordinary
    detector frames carry no conflict column and are a no-op. This belongs in the analysis
    layer because every curation surface uses the same data rule; keeping it out of a GUI
    sibling also preserves ``panels.py``'s dependency boundary.
    """
    if CONSENSUS_CURATION_CONFLICT_COL not in df.columns or len(df) == 0:
        return 0
    positions = np.unique(np.asarray(list(rows), dtype=np.int64))
    positions = positions[(positions >= 0) & (positions < len(df))]
    if positions.size == 0:
        return 0
    col = int(df.columns.get_loc(CONSENSUS_CURATION_CONFLICT_COL))
    values = (
        df.iloc[positions, col].astype("string").fillna("").str.strip().to_numpy()
    )
    changed = positions[values != ""]
    if changed.size:
        df.iloc[changed, col] = ""
    return int(changed.size)


def reconciliation_source_fingerprint(result: AnalysisResult) -> str:
    """Stable digest of every source fact that can change a reconciled snapshot.

    Source tables are intentionally left editable after reconciliation. Persisting this
    digest lets the GUI say that the derived table predates a changed acceptance decision,
    label, boundary, measured feature, or replacement detector run. It is an audit/stale
    signal, never an instruction to destroy the separately curated derived table.
    """
    if not isinstance(result, AnalysisResult) or not isinstance(result.payload, EventSet):
        raise TypeError("source fingerprint requires an AnalysisResult with EventSet payload")
    digest = hashlib.sha256()
    header = {
        "module_id": str(result.module_id),
        "module_version": str(result.module_version),
        "params": dict(result.params or {}),
        "calibration_fingerprint": [float(v) for v in result.calibration_fingerprint],
        "baseline_fingerprint": _effective_baseline_fingerprint(result),
        "provenance": dict(result.payload.provenance or {}),
        "warnings": [str(warning) for warning in (result.warnings or ())],
    }
    digest.update(json.dumps(
        header, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8"))
    frame = result.payload.df
    digest.update(json.dumps(
        [(str(column), str(frame[column].dtype)) for column in frame.columns],
        separators=(",", ":"),
    ).encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame, index=True, categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
        digest.update(row_hashes.tobytes(order="C"))
    except (TypeError, ValueError):
        # Imported/plugin extras can hold objects pandas cannot hash (for example a
        # list-valued audit column). JSON is slower but deterministic and preserves the
        # fact that the table changed instead of disabling dependency tracking entirely.
        digest.update(frame.to_json(
            orient="split", date_format="iso", double_precision=15,
            default_handler=str,
        ).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class ReconcileConfig:
    """Rules for joining and retaining candidate events.

    ``min_overlap_fraction`` is the minimum overlap coefficient (not IoU). A value
    of 0.5 means that at least half of the shorter span must be shared.

    ``min_detector_support`` counts distinct selected detector tables that supplied
    an *automatic* row. One is a union, two is two-detector agreement, and the number
    of selected results is an intersection. A manual-only cluster can be retained
    independently with ``keep_manual`` because it represents the curator's positive
    annotation rather than a detector vote.
    """

    min_overlap_fraction: float = 0.5
    min_detector_support: int = 1
    keep_manual: bool = True

    def __post_init__(self) -> None:
        try:
            overlap = float(self.min_overlap_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_overlap_fraction must be a finite number in (0, 1]") from exc
        if not np.isfinite(overlap) or not 0.0 < overlap <= 1.0:
            raise ValueError("min_overlap_fraction must be a finite number in (0, 1]")
        support = self.min_detector_support
        if isinstance(support, bool) or not isinstance(support, Integral) or int(support) < 1:
            raise ValueError("min_detector_support must be an integer >= 1")
        if not isinstance(self.keep_manual, (bool, np.bool_)):
            raise ValueError("keep_manual must be boolean")
        object.__setattr__(self, "min_overlap_fraction", overlap)
        object.__setattr__(self, "min_detector_support", int(support))
        object.__setattr__(self, "keep_manual", bool(self.keep_manual))


@dataclass(frozen=True)
class ReconcileOutcome:
    """The ready-to-store result plus an audit-friendly reconciliation summary."""

    result: AnalysisResult
    input_modules: tuple[str, ...]
    n_input_rows: int
    n_clusters: int
    n_output_rows: int
    n_collapsed: int
    n_filtered: int
    n_manual_kept: int
    n_label_conflicts: int
    n_curation_conflicts: int


@dataclass(frozen=True)
class _Member:
    uid: int
    module_id: str
    row_position: int
    frame: pd.DataFrame
    start: int
    stop: int
    event_id: int
    added: bool
    accepted: bool
    auto_rejected: bool
    source: str
    review: str | None
    recording: str
    pore_id: str
    flag: str

    @property
    def duration(self) -> int:
        return self.stop - self.start

    @property
    def slot(self) -> tuple[str, str]:
        # A detector may legitimately carry one automatic row and one hand-added row
        # over the same event (the duplicate being repaired), but never two automatic
        # rows in a single cluster.
        return self.module_id, "manual" if self.added else "automatic"

    @property
    def stable_key(self) -> tuple[Any, ...]:
        return (self.module_id, self.added, self.start, self.stop,
                self.event_id, self.row_position)


@dataclass(frozen=True)
class _Bundle:
    """One output row as a canonical row reference plus column-wise overrides.

    Keeping this as values (rather than 10,000 one-row DataFrames) is the difference
    between a preview that blocks Qt for several seconds and one bulk slice per input.
    """

    canonical: _Member
    members: tuple[_Member, ...]
    support: int
    flag: str
    accepted: bool
    source: Any
    review: Any
    added: bool
    contributors: str
    label_conflict_keys: tuple[str, ...]
    curation_conflict: str


class _DisjointSet:
    """Union-find whose roots carry the source slots already represented."""

    def __init__(self, members: Sequence[_Member]) -> None:
        self.parent = list(range(len(members)))
        self.slots = [{m.slot} for m in members]
        self.items = [[idx] for idx in range(len(members))]

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def join(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b or self.slots[a].intersection(self.slots[b]):
            return False
        # Root choice is deterministic and independent of input sequence.
        if b < a:
            a, b = b, a
        self.parent[b] = a
        self.slots[a].update(self.slots[b])
        self.items[a].extend(self.items[b])
        self.items[b] = []
        return True


def reconcile_detections(
    results: Sequence[AnalysisResult],
    *,
    config: ReconcileConfig | None = None,
) -> ReconcileOutcome:
    """Reconcile detector results for one loaded document and channel.

    Inputs are not mutated. The returned ``outcome.result`` is an ordinary
    :class:`AnalysisResult` with an :class:`EventSet` payload, suitable for
    :class:`~nanoscope.core.analysis.ResultStore.put` and project persistence.

    Raises ``ValueError`` rather than guessing when the results do not identify the
    same document/channel/calibration, when a measurement contract conflicts, or when
    event spans are malformed.
    """

    cfg = config if config is not None else ReconcileConfig()
    if not isinstance(cfg, ReconcileConfig):
        raise TypeError("config must be a ReconcileConfig")
    ordered = _validate_results(results, cfg)
    members = _members(ordered)
    groups = _match_groups(members, cfg.min_overlap_fraction)

    kept: list[tuple[list[_Member], int, bool]] = []
    n_manual_kept = 0
    for group in groups:
        support = len({m.module_id for m in group if not m.added})
        has_manual = any(m.added for m in group)
        rescued = bool(cfg.keep_manual and has_manual and support < cfg.min_detector_support)
        if support >= cfg.min_detector_support or rescued:
            kept.append((group, support, rescued))
            n_manual_kept += int(rescued)

    bundles = [_consensus_bundle(group, support) for group, support, _ in kept]
    bundles.sort(key=lambda b: (
        b.canonical.start,
        b.canonical.stop,
        tuple(m.stable_key for m in b.members),
    ))
    frame, memberships = _assemble_frame(bundles)

    input_modules = tuple(r.module_id for r in ordered)
    n_input = len(members)
    n_clusters = len(groups)
    n_output = len(frame)
    n_filtered = n_clusters - n_output
    n_collapsed = n_input - n_clusters
    n_label_conflicts = sum(bool(bundle.label_conflict_keys) for bundle in bundles)
    n_curation_conflicts = sum(bool(bundle.curation_conflict) for bundle in bundles)
    provenance = _provenance(
        ordered,
        cfg,
        frame,
        memberships,
        n_input_rows=n_input,
        n_clusters=n_clusters,
        n_collapsed=n_collapsed,
        n_filtered=n_filtered,
        n_manual_kept=n_manual_kept,
        n_label_conflicts=n_label_conflicts,
        n_curation_conflicts=n_curation_conflicts,
    )
    events = EventSet(frame, provenance)
    spans = OverlayHint(
        OverlayKind.SPAN,
        starts=frame["start_idx"].to_numpy(dtype=np.int64),
        stops=frame["end_idx"].to_numpy(dtype=np.int64),
        label="events",
        name="Reconciled",
    )
    scalars: dict[str, float] = {"n_events": float(n_output)}
    try:
        scalars["rate_hz"] = float(events.rate_hz())
    except ValueError:
        pass

    baseline_values = [_effective_baseline_fingerprint(r) for r in ordered]
    baseline_fp = (
        baseline_values[0]
        if baseline_values and baseline_values[0] is not None
        and all(value == baseline_values[0] for value in baseline_values[1:])
        else None
    )
    warnings = [
        "Reconciled rows retain one complete contributor's measurements; viable rows "
        "are preferred before manual, human-ruling, and interval-medoid tie-breaks. "
        "Detector boundaries and measured features were not averaged."
    ]
    if any(value is None for value in baseline_values):
        warnings.append(
            "At least one input has no baseline fingerprint, so measurement-contract "
            "compatibility could not be fully verified."
        )
    chain_contract = provenance["reconciliation"]["measurement_contract"]
    if not chain_contract["polarity_fully_verified"]:
        warnings.append(
            "At least one input has no recorded detector polarity, so event-direction "
            "compatibility could not be fully verified; all known polarities agree."
        )
    if chain_contract["mixed_detection_signal_chains"]:
        warnings.append(
            "Canonical rows may come from different detector preprocessing chains. "
            "Use consensus_canonical_module/source and the reconciliation provenance "
            "before comparing measured features across rows."
        )
    if n_filtered:
        warnings.append(
            f"{n_filtered} candidate cluster(s) did not meet the requested detector "
            "support and were left out of the reconciled table."
        )
    if n_label_conflicts:
        warnings.append(
            f"Reconciliation initially found {n_label_conflicts} event(s) with "
            "conflicting user labels; "
            "review the consensus_label_conflicts column before training."
        )
    if n_curation_conflicts:
        warnings.append(
            f"Reconciliation initially found {n_curation_conflicts} event(s) with "
            "contradictory human accept/reject decisions. They are excluded from the "
            "reviewed training selection until a curator explicitly accepts or rejects "
            "the reconciled row and clears consensus_curation_conflict."
        )
    if not chain_contract["coverage_fully_verified"]:
        warnings.append(str(chain_contract["coverage_note"]))

    params = {
        "input_modules": list(input_modules),
        "min_overlap_fraction": cfg.min_overlap_fraction,
        "min_detector_support": cfg.min_detector_support,
        "keep_manual": cfg.keep_manual,
    }
    result = AnalysisResult(
        module_id=CONSENSUS_MODULE_ID,
        module_version=_VERSION,
        doc_id=ordered[0].doc_id,
        channel=int(ordered[0].channel),
        params=params,
        calibration_fingerprint=tuple(ordered[0].calibration_fingerprint),
        payload=events,
        overlays=(spans,),
        scalars=scalars,
        warnings=tuple(warnings),
        baseline_fingerprint=baseline_fp,
    )
    return ReconcileOutcome(
        result=result,
        input_modules=input_modules,
        n_input_rows=n_input,
        n_clusters=n_clusters,
        n_output_rows=n_output,
        n_collapsed=n_collapsed,
        n_filtered=n_filtered,
        n_manual_kept=n_manual_kept,
        n_label_conflicts=n_label_conflicts,
        n_curation_conflicts=n_curation_conflicts,
    )


def _validate_results(
    results: Sequence[AnalysisResult], cfg: ReconcileConfig
) -> list[AnalysisResult]:
    try:
        raw = list(results)
    except TypeError as exc:
        raise ValueError("results must contain at least one detector result") from exc
    if not raw:
        raise ValueError("select at least one detector result to reconcile")
    if not all(isinstance(r, AnalysisResult) for r in raw):
        raise ValueError("every selected item must be an AnalysisResult")
    module_ids = [str(r.module_id).strip() for r in raw]
    if any(not m for m in module_ids):
        raise ValueError("every selected result must have a non-empty module_id")
    if len(set(module_ids)) != len(module_ids):
        raise ValueError("selected detector module ids must be unique")
    if CONSENSUS_MODULE_ID in module_ids:
        raise ValueError("a reconciled result cannot be used as a reconciliation input")
    if cfg.min_detector_support > len(raw):
        raise ValueError(
            "min_detector_support cannot exceed the number of selected detector results"
        )
    if not all(isinstance(r.payload, EventSet) for r in raw):
        raise ValueError("every selected result must carry an EventSet payload")

    doc_ids = {r.doc_id for r in raw}
    channels = {int(r.channel) for r in raw}
    if len(doc_ids) != 1 or len(channels) != 1:
        raise ValueError("selected detections must belong to the same document and channel")

    fingerprints: list[tuple[float, float]] = []
    for r in raw:
        try:
            fp = tuple(float(v) for v in r.calibration_fingerprint)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "every selected result needs a two-value calibration fingerprint"
            ) from exc
        if len(fp) != 2 or not all(np.isfinite(v) for v in fp):
            raise ValueError(
                "every selected result needs a finite two-value calibration fingerprint"
            )
        fingerprints.append(fp)  # type: ignore[arg-type]
    if any(fp != fingerprints[0] for fp in fingerprints[1:]):
        raise ValueError("selected detections were measured under different calibrations")

    baseline = {
        value for value in (_effective_baseline_fingerprint(r) for r in raw)
        if value is not None
    }
    if len(baseline) > 1:
        raise ValueError("selected detections were measured under different baselines")

    _validate_common_provenance(raw, "fs", numeric=True, positive=True)
    _validate_common_provenance(raw, "units", numeric=False)
    _validate_polarities(raw)
    if cfg.min_detector_support > 1:
        spans = [_explicit_analysed_span(r) for r in raw]
        if any(span is None for span in spans) or len(set(spans)) != 1:
            raise ValueError(
                "detector-agreement reconciliation requires identical recorded analysed "
                "spans for every selected result. Use support 1 (union), or rerun the "
                "detectors over the same range before requiring agreement."
            )

    # Stable module ordering makes every downstream tie-break independent of checkbox
    # order in the GUI.
    ordered = sorted(raw, key=lambda r: str(r.module_id))
    for result in ordered:
        _validate_frame(result)

    pores = _known_row_values(ordered, "pore_id")
    recordings = _known_row_values(ordered, "recording")
    if len(pores) > 1:
        raise ValueError(f"selected detections disagree on pore_id: {sorted(pores)}")
    if len(recordings) > 1:
        raise ValueError(
            f"selected detections disagree on recording identity: {sorted(recordings)}"
        )
    return ordered


def _explicit_analysed_span(result: AnalysisResult) -> tuple[int, int] | None:
    """A trustworthy explicit coverage span, or ``None`` for legacy/partial metadata."""
    prov = result.payload.provenance
    lo, hi = prov.get(ANALYSED_START_KEY), prov.get(ANALYSED_STOP_KEY)
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if (not np.isfinite(lo_f) or not np.isfinite(hi_f)
            or lo_f != int(lo_f) or hi_f != int(hi_f) or int(hi_f) <= int(lo_f)):
        return None
    return int(lo_f), int(hi_f)


def _validate_common_provenance(
    results: Sequence[AnalysisResult], key: str, *, numeric: bool,
    positive: bool = False,
) -> None:
    if numeric:
        values = []
        for result in results:
            raw = result.payload.provenance.get(key)
            if raw is None or raw is pd.NA or (
                    isinstance(raw, str) and not raw.strip()):
                continue
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"selected detections carry an invalid {key!r} value")
            values.append(raw)
    else:
        values = [
            value
            for result in results
            if (value := _normalise_text(result.payload.provenance.get(key))) is not None
        ]
    if not values:
        return
    if numeric:
        try:
            numbers = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selected detections carry an invalid {key!r} value") from exc
        if not all(np.isfinite(v) for v in numbers):
            raise ValueError(f"selected detections carry a non-finite {key!r} value")
        if positive and not all(v > 0 for v in numbers):
            raise ValueError(f"selected detections carry a non-positive {key!r} value")
        if not all(np.isclose(v, numbers[0], rtol=1e-12, atol=0.0) for v in numbers[1:]):
            raise ValueError(f"selected detections disagree on {key}")
    elif len(set(values)) > 1:
        raise ValueError(f"selected detections disagree on {key}")


def _effective_baseline_fingerprint(result: AnalysisResult) -> str | None:
    """Return one baseline stamp, refusing contradictory copies in one result.

    Detector results carry the fingerprint both on :class:`AnalysisResult` (for live
    staleness) and in ``EventSet.provenance`` (for project/export round trips). A
    reconciler cannot choose between disagreeing copies without misrepresenting the
    measurement contract.
    """
    top = _normalise_text(result.baseline_fingerprint)
    nested = _normalise_text(result.payload.provenance.get("baseline_fingerprint"))
    if top is not None and nested is not None and top != nested:
        raise ValueError(
            f"{result.module_id} carries inconsistent baseline fingerprints between "
            "its result stamp and event provenance"
        )
    return top if top is not None else nested


def _polarity_value(result: AnalysisResult) -> int | None:
    """Return a detector polarity, preserving missing legacy metadata as unknown."""
    raw = result.payload.provenance.get("polarity")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(
            f"{result.module_id} carries invalid detector polarity {raw!r}; "
            "expected -1, +1, or missing legacy metadata"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{result.module_id} carries invalid detector polarity {raw!r}; "
            "expected -1, +1, or missing legacy metadata"
        ) from exc
    if not np.isfinite(value) or value not in (-1.0, 1.0):
        raise ValueError(
            f"{result.module_id} carries invalid detector polarity {raw!r}; "
            "expected -1, +1, or missing legacy metadata"
        )
    return int(value)


def _validate_polarities(results: Sequence[AnalysisResult]) -> None:
    known = {_polarity_value(result) for result in results}
    known.discard(None)
    if len(known) > 1:
        raise ValueError(
            "selected detections have opposite detector polarities (-1 and +1). "
            "They cannot describe the same physical event direction; rerun them with "
            "one polarity before reconciliation."
        )


def _known_row_values(results: Sequence[AnalysisResult], column: str) -> set[str]:
    out: set[str] = set()
    for result in results:
        df = result.payload.df
        for value in df[column].dropna().astype(str):
            text = value.strip()
            if text:
                out.add(text)
    return out


def _validate_frame(result: AnalysisResult) -> None:
    df = result.payload.df
    for pos, (start_raw, stop_raw) in enumerate(zip(df["start_idx"], df["end_idx"])):
        try:
            start_f, stop_f = float(start_raw), float(stop_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{result.module_id} row {pos} has non-numeric event bounds"
            ) from exc
        if (not np.isfinite(start_f) or not np.isfinite(stop_f)
                or start_f != int(start_f) or stop_f != int(stop_f)
                or int(start_f) < 0 or int(stop_f) <= int(start_f)):
            raise ValueError(
                f"{result.module_id} row {pos} has invalid half-open span "
                f"[{start_raw}, {stop_raw})"
            )


def _members(results: Sequence[AnalysisResult]) -> list[_Member]:
    out: list[_Member] = []
    uid = 0
    for result in results:
        df = result.payload.df
        added = added_mask(df)
        starts = df["start_idx"].to_numpy(dtype=np.int64)
        stops = df["end_idx"].to_numpy(dtype=np.int64)
        event_ids = df["event_id"].to_numpy(dtype=np.int64)
        accepted_values = df["accepted"].to_numpy(dtype=bool)
        flags = df["flag"].to_numpy(dtype=object)
        recordings = df["recording"].astype("string").to_numpy(dtype=object)
        pores = df["pore_id"].astype("string").to_numpy(dtype=object)
        sources = (df[SOURCE_COL].astype("string").to_numpy(dtype=object)
                   if SOURCE_COL in df.columns else np.full(len(df), pd.NA, dtype=object))
        reviews = (df[REVIEW_COL].astype("string").to_numpy(dtype=object)
                   if REVIEW_COL in df.columns else np.full(len(df), pd.NA, dtype=object))
        for pos in range(len(df)):
            is_added = bool(added[pos])
            accepted = bool(accepted_values[pos])
            flag = "" if _normalise_text(flags[pos]) is None else str(flags[pos])
            src = _normalise_text(sources[pos])
            if src is None:
                src = SOURCE_MANUAL if is_added else str(result.module_id)
            review_text = _normalise_text(reviews[pos])
            rev = review_text if review_text in REVIEW_VALUES else None
            out.append(_Member(
                uid=uid,
                module_id=str(result.module_id),
                row_position=pos,
                frame=df,
                start=int(starts[pos]),
                stop=int(stops[pos]),
                event_id=int(event_ids[pos]),
                added=is_added,
                accepted=accepted,
                auto_rejected=(not accepted and is_automatic_rejection(flag)),
                source=src,
                review=rev,
                recording=_normalise_text(recordings[pos]) or "",
                pore_id=_normalise_text(pores[pos]) or "",
                flag=flag,
            ))
            uid += 1
    return sorted(out, key=lambda m: (m.start, m.stop, m.stable_key))


def _normalise_text(value: Any) -> str | None:
    """Local scalar normaliser matching the public source/review readers."""
    if value is None or value is pd.NA:
        return None
    try:
        if not isinstance(value, str) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def _pair_allowed(a: _Member, b: _Member) -> bool:
    # Two automatic (or two manual) rows from one table are separate annotations; only
    # the detector/manual duplicate inside that table is eligible to converge.
    if a.module_id == b.module_id and a.added == b.added:
        return False
    if a.recording and b.recording and a.recording != b.recording:
        return False
    if a.pore_id and b.pore_id and a.pore_id != b.pore_id:
        return False
    return True


def _overlap(a: _Member, b: _Member) -> tuple[float, float]:
    intersection = min(a.stop, b.stop) - max(a.start, b.start)
    if intersection <= 0:
        return 0.0, 0.0
    coefficient = intersection / min(a.duration, b.duration)
    union = max(a.stop, b.stop) - min(a.start, b.start)
    return float(coefficient), float(intersection / union)


def _match_groups(members: Sequence[_Member], threshold: float) -> list[list[_Member]]:
    if not members:
        return []
    edges: list[tuple[Any, ...]] = []
    active: list[int] = []
    comparisons = 0
    for right, member in enumerate(members):
        active = [left for left in active if members[left].stop > member.start]
        for left in active:
            comparisons += 1
            if comparisons > MAX_PAIR_COMPARISONS:
                raise ValueError(
                    "reconciliation stopped after 1,000,000 simultaneous-span "
                    "comparisons. These event tables are unusually dense or contain "
                    "many mutually overlapping candidates; reconcile a shorter segment "
                    "or clean the pathological spans before previewing again."
                )
            other = members[left]
            if not _pair_allowed(other, member):
                continue
            coefficient, iou = _overlap(other, member)
            if coefficient + np.finfo(float).eps < threshold:
                continue
            if len(edges) >= MAX_MATCH_EDGES:
                raise ValueError(
                    "reconciliation found more than 100,000 eligible overlap links. "
                    "Increase the minimum temporal overlap, reconcile a shorter segment, "
                    "or clean the unusually dense event tables before previewing again."
                )
            centre_gap = abs((other.start + other.stop) - (member.start + member.stop))
            edges.append((
                -coefficient,
                -iou,
                centre_gap,
                other.stable_key,
                member.stable_key,
                left,
                right,
            ))
        active.append(right)

    dsu = _DisjointSet(members)
    for *_rank, left, right in sorted(edges):
        left_root, right_root = dsu.find(int(left)), dsu.find(int(right))
        if left_root == right_root:
            continue
        # Single-linkage would let [0,10), [0,20), [10,20) become one apparent
        # three-detector event even though the outer spans never overlap. Requiring
        # every cross-component pair to qualify keeps support scientifically literal.
        coherent = all(
            _pair_allowed(members[a], members[b])
            and _overlap(members[a], members[b])[0] + np.finfo(float).eps >= threshold
            for a in dsu.items[left_root]
            for b in dsu.items[right_root]
        )
        if not coherent:
            continue
        dsu.join(int(left), int(right))

    grouped: dict[int, list[_Member]] = {}
    for idx, member in enumerate(members):
        grouped.setdefault(dsu.find(idx), []).append(member)
    out = [sorted(group, key=lambda m: m.stable_key) for group in grouped.values()]
    out.sort(key=lambda g: (min(m.start for m in g), max(m.stop for m in g),
                            tuple(m.stable_key for m in g)))
    return out


def _iou_member(a: _Member, b: _Member) -> float:
    return _overlap(a, b)[1] if a.uid != b.uid else 1.0


def _canonical(group: Sequence[_Member]) -> _Member:
    def rank(member: _Member) -> tuple[Any, ...]:
        human_decision = member.review == REVIEW_DONE or (
            not member.accepted and not member.auto_rejected
        )
        similarity = sum(_iou_member(member, other) for other in group)
        return (
            # An automatically rejected row says its own measurement is not valid
            # (clipped, non-finite, or beyond the detector's dwell contract). It must
            # never donate frozen measurements to a row another detector can stand
            # behind, even when the invalid span was drawn manually.
            0 if not member.auto_rejected else 1,
            0 if member.added else 1,          # a curator's boundary wins
            0 if human_decision else 1,
            -similarity,                       # otherwise choose the interval medoid
            member.duration,
            member.stable_key,
        )

    return min(group, key=rank)


def _consensus_bundle(group: list[_Member], support: int) -> _Bundle:
    canonical = _canonical(group)

    # Derived flags describe the row's own measured span, so only the canonical row's
    # derived flags remain. Human labels/marks from every contributor are inherited;
    # conflicting label values remain visible through labels.parse_labels.
    canonical_derived = combine_flags(*[
        token for token in flag_tokens(canonical.flag) if token in DERIVED_FLAGS
    ])
    inherited = combine_flags(*[
        token
        for member in group
        for token in flag_tokens(member.flag)
        if token not in DERIVED_FLAGS
    ])
    combined_flag = combine_flags(canonical_derived, inherited)

    # An automatic self-rejection is not a curator vote. Contradictory human truth must
    # remain visibly unresolved: the common workflow "reject broad detector candidate,
    # then add the corrected span" contains both a negative and a positive decision.
    # Flattening that pair to a reviewed negative silently corrupts supervised labels.
    human_accepted = any(
        m.accepted and (m.review == REVIEW_DONE or m.added) for m in group
    )
    hand_rejected = any(not m.accepted and not m.auto_rejected for m in group)
    curation_conflict = "accepted|rejected" if human_accepted and hand_rejected else ""
    # Until the conflict is explicitly resolved on the derived row, fail closed as a
    # negative while the marker keeps it out of reviewed training exports.
    accepted = False if hand_rejected else any(m.accepted for m in group)

    sources = [m.source for m in group]
    reviews = [m.review if m.review is not None else pd.NA for m in group]
    combined_source = combine_sources(sources)
    # A detector duplicate being unreviewed does not erase a person's explicit ruling
    # on the physical event. Legacy hand rejections did not always receive a review
    # token, but ``accepted=False`` without an automatic reason is already the app's one
    # definition of a hand decision, so it is decisive too. With no human ruling at all,
    # retain combine_reviews' conservative all-members rule (including unknown).
    has_human_ruling = any(
        m.review == REVIEW_DONE or (not m.accepted and not m.auto_rejected)
        for m in group
    )
    combined_review = REVIEW_DONE if has_human_ruling else combine_reviews(reviews)

    # Relative to the reconciled detector, a row is missed only when *none* of the
    # selected detectors found it. This is the key duplicate/FN correction.
    contributor_tokens = sorted(set(sources))
    conflicts = tuple(sorted(label_conflicts(combined_flag)))
    return _Bundle(
        canonical=canonical,
        members=tuple(group),
        support=int(support),
        flag=combined_flag,
        accepted=bool(accepted),
        source=combined_source,
        review=combined_review,
        added=bool(all(m.added for m in group)),
        contributors=_CONTRIBUTOR_SEP.join(contributor_tokens),
        label_conflict_keys=conflicts,
        curation_conflict=curation_conflict,
    )


def _assemble_frame(
    bundles: Sequence[_Bundle]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not bundles:
        out = empty_df()
        out[SOURCE_COL] = pd.Series(dtype="string")
        out[REVIEW_COL] = pd.Series(dtype="string")
        out[ADDED_COLUMN] = pd.Series(dtype="bool")
        out[CONSENSUS_CONTRIBUTORS_COL] = pd.Series(dtype="string")
        out[CONSENSUS_CANONICAL_SOURCE_COL] = pd.Series(dtype="string")
        out[CONSENSUS_CANONICAL_MODULE_COL] = pd.Series(dtype="string")
        out[CONSENSUS_LABEL_CONFLICTS_COL] = pd.Series(dtype="string")
        out[CONSENSUS_CURATION_CONFLICT_COL] = pd.Series(dtype="string")
        out[CONSENSUS_SUPPORT_COL] = pd.Series(dtype="int64")
        out[CONSENSUS_MEMBER_COUNT_COL] = pd.Series(dtype="int64")
        return out, []

    # Take all canonical rows from each input frame in one operation. The former
    # one-row-frame-per-cluster path spent most of its runtime in 9 scalar pandas
    # assignments per event and then concatenated thousands of tiny managers.
    selections: dict[int, tuple[pd.DataFrame, list[int], list[int]]] = {}
    for out_position, bundle in enumerate(bundles):
        frame = bundle.canonical.frame
        key = id(frame)
        if key not in selections:
            selections[key] = (frame, [], [])
        selections[key][1].append(bundle.canonical.row_position)
        selections[key][2].append(out_position)
    chunks: list[pd.DataFrame] = []
    for frame, positions, output_positions in selections.values():
        chunk = frame.iloc[positions].copy()
        chunk.index = output_positions
        chunks.append(chunk)
    out = pd.concat(chunks, axis=0, sort=False).sort_index(kind="stable").reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=np.int64)
    out["flag"] = pd.array([bundle.flag for bundle in bundles], dtype="string")
    out["accepted"] = np.array([bundle.accepted for bundle in bundles], dtype=bool)
    out[SOURCE_COL] = pd.array([bundle.source for bundle in bundles], dtype="string")
    out[REVIEW_COL] = pd.array([bundle.review for bundle in bundles], dtype="string")
    out[ADDED_COLUMN] = np.array([bundle.added for bundle in bundles], dtype=bool)
    out[CONSENSUS_CONTRIBUTORS_COL] = pd.array(
        [bundle.contributors for bundle in bundles], dtype="string"
    )
    out[CONSENSUS_CANONICAL_SOURCE_COL] = pd.array(
        [bundle.canonical.source for bundle in bundles], dtype="string"
    )
    out[CONSENSUS_CANONICAL_MODULE_COL] = pd.array(
        [bundle.canonical.module_id for bundle in bundles], dtype="string"
    )
    out[CONSENSUS_LABEL_CONFLICTS_COL] = pd.array(
        [_CONTRIBUTOR_SEP.join(bundle.label_conflict_keys) for bundle in bundles],
        dtype="string",
    )
    out[CONSENSUS_CURATION_CONFLICT_COL] = pd.array(
        [bundle.curation_conflict for bundle in bundles], dtype="string"
    )
    out[CONSENSUS_SUPPORT_COL] = np.array([bundle.support for bundle in bundles], dtype=np.int64)
    out[CONSENSUS_MEMBER_COUNT_COL] = np.array(
        [len(bundle.members) for bundle in bundles], dtype=np.int64
    )
    # Bulk slicing/concatenation normally preserves these dtypes. Assert the public
    # frozen contract by coercing explicitly, including when input extras forced pandas
    # to find a common object dtype.
    from .events import SCHEMA
    for column, (dtype, _unit, _description) in SCHEMA.items():
        out[column] = out[column].astype(dtype)

    memberships: list[dict[str, Any]] = []
    for event_id, bundle in enumerate(bundles):
        members = bundle.members
        canonical = bundle.canonical
        refs = []
        for member in sorted(members, key=lambda m: m.stable_key):
            refs.append({
                "module_id": member.module_id,
                "row_position": member.row_position,
                "event_id": member.event_id,
                "start_idx": member.start,
                "end_idx": member.stop,
                "source": member.source,
                "accepted": member.accepted,
                "added": member.added,
                "review": member.review,
                "flag": member.flag,
                "canonical": member.uid == canonical.uid,
            })
        memberships.append({"event_id": event_id, "members": refs})
    return out, memberships


def _union_analysed_span(
    results: Sequence[AnalysisResult], fs: float | None
) -> tuple[float | None, tuple[int, int] | None, bool, str]:
    """Return only coverage that can be defended for the whole derived table.

    A maximum of input durations is not a union. If one result names a two-second span
    and a legacy result merely claims ten seconds, advertising two seconds inflates the
    event rate. Differing spans can be unioned exactly only when every input names its
    span and its analysed duration is nominal (no detector-specific excluded samples).
    """
    spans = [_explicit_analysed_span(result) for result in results]

    outside_declared_coverage = False
    for result, span in zip(results, spans):
        if span is None or result.payload.df.empty:
            continue
        starts = result.payload.df["start_idx"].to_numpy(dtype=np.int64)
        stops = result.payload.df["end_idx"].to_numpy(dtype=np.int64)
        if bool(np.any(starts < span[0]) or np.any(stops > span[1])):
            outside_declared_coverage = True
            break

    def known_duration(result: AnalysisResult) -> float | None:
        raw = result.payload.provenance.get("analyzed_duration_s")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    durations = [known_duration(result) for result in results]
    if outside_declared_coverage:
        return None, None, False, (
            "At least one hand-added or imported event lies outside its source detector's "
            "recorded analysed span. The extent of manual searching is unknown, so event "
            "rate and analysed span are omitted from the reconciled table."
        )
    if len(results) == 1:
        span = spans[0]
        duration = durations[0]
        if duration is None and span is not None and fs is not None and fs > 0:
            duration = (span[1] - span[0]) / fs
        exact = duration is not None
        note = (
            "Coverage is fully recorded for the reconciled table."
            if exact else
            "The source result does not record a trustworthy analysed duration; "
            "event rate is omitted from the reconciled table."
        )
        return duration, span, exact, note

    if any(span is None for span in spans):
        return None, None, False, (
            "At least one input lacks an explicit analysed sample span, so the combined "
            "coverage cannot be computed without guessing; event rate and analysed span "
            "are omitted from the reconciled table."
        )

    concrete_spans = [span for span in spans if span is not None]
    assert len(concrete_spans) == len(results)
    if all(span == concrete_spans[0] for span in concrete_spans[1:]):
        known = [value for value in durations if value is not None]
        if len(known) == len(results) and all(
                np.isclose(value, known[0], rtol=1e-9, atol=1e-12)
                for value in known[1:]):
            return known[0], concrete_spans[0], True, (
                "Coverage is fully recorded for the reconciled table."
            )
        return None, concrete_spans[0], False, (
            "Inputs name the same analysed sample span but do not all record the same "
            "valid analysed duration; event rate is omitted rather than guessed."
        )

    spans_sorted = sorted(concrete_spans)
    merged: list[list[int]] = []
    for lo, hi in spans_sorted:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    one_span = (merged[0][0], merged[0][1]) if len(merged) == 1 else None
    nominal_exact = fs is not None and fs > 0 and all(
        duration is not None
        and np.isclose(duration, (span[1] - span[0]) / fs,
                       rtol=1e-9, atol=1e-12)
        for span, duration in zip(concrete_spans, durations)
    )
    if nominal_exact:
        duration = sum(hi - lo for lo, hi in merged) / float(fs)
        return float(duration), one_span, True, (
            "Coverage is the exact union of every recorded analysed span."
        )
    return None, one_span, False, (
        "Inputs cover different sample spans and at least one has detector-specific or "
        "unknown valid duration; their exact union is unknowable, so event rate is "
        "omitted rather than guessed."
    )


def _common_text_provenance(results: Sequence[AnalysisResult], key: str) -> str | None:
    known = [
        value
        for result in results
        if (value := _normalise_text(result.payload.provenance.get(key))) is not None
    ]
    if not known:
        return None
    first = known[0]
    return first if all(value == first for value in known[1:]) else None


def _common_numeric_provenance(
    results: Sequence[AnalysisResult], key: str
) -> float | None:
    known: list[float] = []
    for result in results:
        raw = result.payload.provenance.get(key)
        if raw is None or raw is pd.NA or (
                isinstance(raw, str) and not raw.strip()):
            continue
        known.append(float(raw))
    if not known:
        return None
    return known[0] if all(
        np.isclose(value, known[0], rtol=1e-12, atol=0.0)
        for value in known[1:]
    ) else None


def _provenance(
    results: Sequence[AnalysisResult],
    cfg: ReconcileConfig,
    frame: pd.DataFrame,
    memberships: list[dict[str, Any]],
    **counts: int,
) -> dict[str, Any]:
    fs = _common_numeric_provenance(results, "fs")
    duration, span, coverage_verified, coverage_note = _union_analysed_span(results, fs)
    from .modules import detection_signal_key

    signal_keys = [detection_signal_key(r.payload.provenance) for r in results]
    common_signal_key = (
        signal_keys[0] if all(k == signal_keys[0] for k in signal_keys[1:]) else None
    )
    baseline_values = [_effective_baseline_fingerprint(r) for r in results]
    common_baseline = (
        baseline_values[0]
        if baseline_values and baseline_values[0] is not None
        and all(value == baseline_values[0] for value in baseline_values[1:])
        else None
    )
    input_polarities = {r.module_id: _polarity_value(r) for r in results}
    known_polarities = [p for p in input_polarities.values() if p is not None]
    common_polarity = known_polarities[0] if known_polarities else None
    polarity_fully_verified = all(p is not None for p in input_polarities.values())
    measurement_contract = {
        "row_measurements": "complete canonical contributor row; never averaged",
        "canonical_module_column": CONSENSUS_CANONICAL_MODULE_COL,
        "canonical_source_column": CONSENSUS_CANONICAL_SOURCE_COL,
        "mixed_detection_signal_chains": common_signal_key is None,
        "common_detection_signal_key": (list(common_signal_key)
                                         if common_signal_key is not None else None),
        "common_baseline_fingerprint": common_baseline,
        "input_polarities": input_polarities,
        "common_known_polarity": common_polarity,
        "polarity_fully_verified": polarity_fully_verified,
        "coverage_fully_verified": coverage_verified,
        "coverage_note": coverage_note,
        "note": (
            "Each output row may retain measurements from a different detector chain. "
            "Use the per-row canonical module/source and that input module's nested "
            "provenance before comparing bounds or measured features. No single "
            "conditioning contract is claimed for this derived table."
        ),
    }
    block: dict[str, Any] = {
        "version": 1,
        "algorithm": "constrained-overlap-coefficient",
        "overlap_metric": "intersection / min(duration_a, duration_b)",
        "cluster_coherence": "complete linkage: every member pair meets overlap threshold",
        "canonical_rule": (
            "viable (not automatically rejected), then manual, then explicit human "
            "ruling, then interval medoid, shortest duration, stable source/row order"
        ),
        "measurement_rule": "complete canonical row; no boundary or feature averaging",
        "measurement_contract": measurement_contract,
        "input_modules": [r.module_id for r in results],
        "input_rows": {r.module_id: int(len(r.payload.df)) for r in results},
        "input_result_fingerprints": {
            r.module_id: reconciliation_source_fingerprint(r) for r in results
        },
        "input_provenance": {r.module_id: dict(r.payload.provenance or {}) for r in results},
        "input_results": {
            r.module_id: {
                "module_version": str(r.module_version),
                "params": dict(r.params or {}),
                "calibration_fingerprint": [float(v) for v in r.calibration_fingerprint],
                "baseline_fingerprint": _effective_baseline_fingerprint(r),
                "warnings": [str(warning) for warning in (r.warnings or ())],
            }
            for r in results
        },
        "min_overlap_fraction": cfg.min_overlap_fraction,
        "min_detector_support": cfg.min_detector_support,
        "keep_manual": cfg.keep_manual,
        **{key: int(value) for key, value in counts.items()},
        "memberships": memberships,
    }
    out: dict[str, Any] = {
        "detector": "reconciled",
        "n_events": int(len(frame)),
        "n_filter_limited": n_filter_limited(frame),
        CLIPPED_COUNT_KEY: n_clipped_boundary(frame),
        "reconciliation": block,
    }
    for key in ("source", "recording", "units"):
        value = _common_text_provenance(results, key)
        if value is not None:
            out[key] = value
    if common_baseline is not None:
        out["baseline_fingerprint"] = common_baseline
    # A known polarity is safe to promote to the derived table only when every input
    # recorded it. One known plus one legacy unknown is compatible enough to reconcile,
    # but is not evidence for a table-wide conditioning contract.
    if polarity_fully_verified and common_polarity is not None:
        out["polarity"] = common_polarity
    if fs is not None:
        out["fs"] = fs
    if duration is not None:
        out["analyzed_duration_s"] = duration
    if span is not None:
        out[ANALYSED_START_KEY], out[ANALYSED_STOP_KEY] = span
    # Every reconciled input describes the same loaded recording. If an input
    # identifies it as baseline-subtracted with no physical I0, reconciliation
    # cannot manufacture that missing denominator for the derived table.
    if any(r.payload.provenance.get("fractional_blockage_reference_unavailable") is True
           for r in results):
        out["fractional_blockage_reference_unavailable"] = True
    return out
