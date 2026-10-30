"""Post-detection sub-event merge: stitch a shattered translocation back together.

A single translocation that briefly recrosses the exit threshold is reported by the
detector as several adjacent fragments. Thresholding alone cannot fix this without
also dropping the genuinely short events one wants to keep, so the fix belongs
*after* detection: merge nearby fragments by a time window, either over the whole
trace or over a user-selected region.

This is a **curation-style edit**, exactly like accept/reject or a hand-added missed
event. It never touches the frozen 20-column ``events.SCHEMA``: a merged run is
collapsed to one span ``[first.start_idx, last.end_idx]`` and pushed back through the
SAME :func:`nanoscope.analysis.characterize.characterize` pipeline the detector used,
so every derived column (median depth, ECD, dwell, SNR, levels, filter-limited flag)
stays mutually consistent. Non-merged events pass through byte-for-byte unchanged.

THE GAP RULE IS NOT RE-DERIVED HERE
-----------------------------------
Grouping uses :func:`nanoscope.analysis.detect.group_starts_by_gap` and
:func:`~nanoscope.analysis.detect.gap_window_samples` -- the very functions the
detector's own ``merge_gap_s`` uses. So "merge iff ``gap = next.start_idx -
cur.end_idx <= window``" means the same thing on both paths; there is deliberately
no second, subtly-different rule.

Two entry points:

* :func:`merge_by_gap` -- window-based. Every run of consecutive events each within
  the window collapses to one (transitive). Optional ``index_range`` limits which
  events are eligible (the region/inspector fixes), leaving the rest untouched.
* :func:`merge_region` -- selection-based. ALL events overlapping ``[t0_s, t1_s]``
  collapse to one, regardless of gap (an explicit user selection overrides the
  window). A no-op when fewer than two events overlap.

Both return ``(new_eventset, n_merged)`` where ``n_merged`` is how many source rows
were merged away (``len(before) - len(after)``, matching the detector's own
``n_merged`` bookkeeping). ``event_id`` is reassigned contiguously and ``provenance``
records the merge under ``provenance['merged']`` without disturbing
``analyzed_duration_s`` (so ``EventSet.rate_hz`` still works).

CURATION IS NOT DECIDED HERE EITHER
-----------------------------------
``accepted`` used to be the OR of the run's flags -- "a merged event is real if any
fragment was" -- and ``flag`` came back blank from the rebuild. Together those two
lines UN-REJECTED events and erased the reason: on the owner's -600 mV recording,
under a declared ``max_dwell_s`` of 2 ms, a 5 ms gap merge produced 240 accepted rows
longer than the limit (worst 24.88 ms, 12.4x) flowing into the table, both histograms,
the censored dwell fit -- whose ``t_max`` IS ``max_dwell_s`` -- and every export.

So ``flag`` and ``accepted`` are decided by
:func:`nanoscope.analysis.characterize.recharacterize_span`, the one function every
row-rebuild path in the app goes through: the automatic rules are re-judged against
the merged span, a curator's rejection survives, and ``accepted`` is combined with
AND. See :func:`nanoscope.analysis.events.inherited_curation`.

A MERGED EVENT MUST STILL BE ONE TRANSLOCATION
----------------------------------------------
The gap rule is transitive, and transitivity has no brake: every event within one
window of the next joins the same group, so a REGULARLY spaced train of clean,
well-separated events collapses into a single row the moment the window exceeds the
spacing. Reproduced: 40 isolated 0.4 ms / 300 pA events spaced 8 ms apart, merged at
a 10 ms window -- the very setting :mod:`nanoscope.gui.merge_panel`'s own text
recommends -- became ONE event of dwell 312.40 ms and depth 0.70 pA, accepted, with a
blank flag. Nothing downstream could see anything wrong with it: the depth is a median
over a span that is 96 % open pore, so the true 300 pA blockade is simply out-voted.

So grouping is necessary but not sufficient. A group is only merged when its
OCCUPANCY -- the fraction of the merged span that its fragments actually cover --
reaches :data:`MIN_MERGE_OCCUPANCY`; below that the group is split back into
singletons and counted as a refusal. The gap rule itself is NOT re-derived or
weakened by this (it is still the one shared :func:`group_starts_by_gap`); occupancy
is a second, physical question asked of the group the rule proposed.

And, like the gap rule, the floor is DEFINED ONCE and shared:
:data:`~nanoscope.analysis.detect.MIN_MERGE_OCCUPANCY` and
:func:`~nanoscope.analysis.detect.split_low_occupancy_groups` live in
:mod:`nanoscope.analysis.detect` so the detector's own ``merge_gap_s`` -- the same
grouping, run inside a detection -- is braked by the same floor, and are re-exported
from this module under their original names. See the note beside the import below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .baseline import derive_polarity
from .characterize import recharacterize_span
from .detect import (
    LOW_OCCUPANCY_TOKEN,
    MARGINAL_MERGE_OCCUPANCY,
    MIN_MERGE_OCCUPANCY,
    gap_window_samples,
    group_starts_by_gap,
    span_occupancy,
    split_low_occupancy_groups,
)
from .events import (
    REVIEW_COL,
    SOURCE_COL,
    EventSet,
    combine_reviews,
    combine_sources,
    reconcile_row_counts,
)
from .manual_events import LEGACY_IDS_COL, NOTE_COL, ORIGIN_COL, join_ids, join_notes

__all__ = [
    "MARGINAL_MERGE_OCCUPANCY",
    "MIN_MERGE_OCCUPANCY",
    "LOW_OCCUPANCY_TOKEN",
    "MergeOutcome",
    "merge_by_gap",
    "merge_region",
    "span_occupancy",
]

# A few ms of open-pore context each side of a merged span, so the re-characterization
# has flanks to read the local baseline / noise from and room to measure rise / fall.
_PAD_MS = 3.0

# THE OCCUPANCY FLOOR IS DEFINED IN `analysis.detect`, NOT HERE, and imported above.
# It began as this module's own constant, which left the Detection tab's own
# ``merge_gap_s`` -- the SAME grouping, one spinbox away -- with no floor at all, so the
# identical 312.40 ms / 0.70 pA row was still reachable at detection time. The floor
# belongs beside `group_starts_by_gap`, the gap rule it brakes, for the same reason that
# rule is not re-derived here: one definition, both paths. The direction of the move was
# forced -- `detect` cannot import `merge` (this module imports `detect`; the reverse is
# a cycle) -- and the names are kept in this module's ``__all__`` above, and re-exported
# unchanged, so ``from nanoscope.analysis.merge import MIN_MERGE_OCCUPANCY`` /
# ``span_occupancy`` (what the GUI and the tests already write) still resolves.
#
# ON THE 10-15 ms THE WALKTHROUGHS RECOMMEND. `gui.merge_panel` justifies its
# millisecond spinbox by noting that fragment gaps on this bench are millisecond-scale
# and the walkthroughs merge at 10-15 ms; the floor exists because a 10 ms merge on the
# owner's bench recording was pathological. Both are true and they are not about the
# same quantity. The WINDOW is a statement about how far apart the pieces of one
# translocation can fall (millisecond-scale here, hence the unit), and it is a necessary
# condition; OCCUPANCY is the sufficient one -- of everything that window sweeps up, only
# the groups that actually FILL their span are one translocation. 10-15 ms stays the
# right window: with the floor in place it stitches the shattered events it was chosen
# for and declines the 8 ms-spaced train it also happened to reach, which is exactly what
# made the recommendation safe to give in the first place.


class MergeOutcome(tuple):
    """The ``(new_eventset, n_merged)`` pair both entry points have always returned,
    carrying the occupancy verdict as attributes.

    It stays a 2-tuple ON PURPOSE. ``new_events, n = merge_by_gap(...)`` is what every
    caller and every existing test writes, and the refusal count is information the GUI
    wants but nothing else needs; growing the tuple to three would have broken all of
    them for the sake of one consumer. Callers that care read ``.n_refused`` (and
    ``.refused_occupancy`` for the actual fractions, so a status line can say HOW empty
    the span was); callers that do not are unchanged.

    ``n_refused`` is also what distinguishes "there was nothing to merge" from "there
    was something and it was declined": both return ``n_merged == 0``, and the second
    must not be reported to the user as the first.
    """

    # No ``__slots__``: CPython forbids a non-empty one on a variable-length subtype
    # (``TypeError: nonempty __slots__ not supported for subtype of 'tuple'``), and one
    # instance per merge is not worth working around it for.

    def __new__(
        cls,
        eventset: EventSet,
        n_merged: int,
        *,
        n_refused: int = 0,
        refused_occupancy: tuple[float, ...] = (),
    ) -> "MergeOutcome":
        obj = super().__new__(cls, (eventset, int(n_merged)))
        obj.n_refused = int(n_refused)
        obj.refused_occupancy = tuple(float(o) for o in refused_occupancy)
        return obj

    @property
    def eventset(self) -> EventSet:
        return self[0]

    @property
    def n_merged(self) -> int:
        return self[1]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def merge_by_gap(
    eventset: EventSet,
    signal: np.ndarray,
    fs: float,
    *,
    max_gap_s: float,
    index_range: tuple[int, int] | None = None,
) -> MergeOutcome:
    """Merge runs of events whose inter-event gap is within ``max_gap_s``.

    Events are sorted by ``start_idx``; a run of consecutive events, each separated
    from the next by ``gap = next.start_idx - cur.end_idx <= round(max_gap_s * fs)``,
    collapses to ONE event spanning ``[first.start_idx, last.end_idx]``, re-characterized
    over that full span (so ``dwell_s`` is the full span and depth / ECD are recomputed).
    The merge is transitive: a whole chain each within the window becomes one event.
    Non-merged events pass through unchanged.

    A grouped run is merged only if it is REALLY one translocation: a group whose
    occupancy (fragment coverage / merged span) is below :data:`MIN_MERGE_OCCUPANCY`
    is split back into its singletons and counted in the returned
    :class:`MergeOutcome`'s ``n_refused`` -- see the module docstring for the 40-event
    train that collapsed into one 312 ms / 0.70 pA row without it.

    Args:
        eventset: The detection to curate. Its ``.df`` (schema rows) and ``.provenance``
            drive polarity / noise / filter-rise for the re-characterization.
        signal: The calibrated current trace the detection ran on (1-D, same sample
            base as the events' indices). float32 in is fine.
        fs: Sample rate in Hz.
        max_gap_s: Merge window in seconds. ``gap <= round(max_gap_s * fs)`` merges.
        index_range: Optional ``(i0, i1)`` half-open sample window. When given, only
            events whose span intersects ``[i0, i1)`` are eligible to merge; the rest
            pass through untouched (and never merge across an ineligible neighbour).

    Returns:
        A :class:`MergeOutcome` -- the ``(new_eventset, n_merged)`` pair it has always
        been (schema-valid, ``event_id`` reassigned contiguously, ``n_merged =
        len(before) - len(after)``), plus ``n_refused`` / ``refused_occupancy`` for the
        groups declined by the occupancy floor. ``(eventset, 0)`` unchanged when nothing
        merges.
    """
    _validate_merge_input(eventset, signal, fs)
    if not np.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError(f"max_gap_s must be finite and >= 0, got {max_gap_s!r}")
    df = eventset.df
    n = len(df)
    if n < 2:
        return MergeOutcome(eventset, 0)

    order = _sort_order(df)
    starts = df["start_idx"].to_numpy(dtype=np.int64)[order]
    ends = df["end_idx"].to_numpy(dtype=np.int64)[order]

    is_group_start = group_starts_by_gap(
        starts, ends, gap_window_samples(max_gap_s, fs)
    )
    if index_range is not None:
        eligible = _intersects(starts, ends, int(index_range[0]), int(index_range[1]))
        # An ineligible event is always its own group (it passes through), and never
        # lets a run cross it: force a break AT every ineligible event and at the first
        # event after one.
        is_group_start = is_group_start.copy()
        is_group_start[~eligible] = True
        is_group_start[1:][~eligible[:-1]] = True

    is_group_start, refused, marginal = _apply_occupancy_floor(
        starts, ends, is_group_start
    )

    if is_group_start.all():
        # Every event is its own group -> nothing to merge. ``refused`` may still be
        # non-zero: the caller must be able to tell "nothing was close enough" from
        # "something was close enough and was declined".
        return MergeOutcome(
            eventset, 0, n_refused=len(refused), refused_occupancy=tuple(refused)
        )

    note: dict = {"rule": "gap", "window_s": float(max_gap_s)}
    _note_occupancy(note, refused, marginal)
    out, n_merged = _rebuild(eventset, signal, float(fs), order, is_group_start, note)
    return MergeOutcome(
        out, n_merged, n_refused=len(refused), refused_occupancy=tuple(refused)
    )


def merge_region(
    eventset: EventSet,
    signal: np.ndarray,
    fs: float,
    *,
    t0_s: float,
    t1_s: float,
) -> MergeOutcome:
    """Collapse EVERY event overlapping ``[t0_s, t1_s]`` into a single event.

    The explicit selection overrides the gap WINDOW entirely: all events whose span
    intersects the region are merged into one event spanning their minimum
    ``start_idx`` .. maximum ``end_idx``, re-characterized over that span, however far
    apart they are. Everything outside the region is untouched. A no-op (returns
    ``(eventset, 0)``) when fewer than two events overlap the region.

    What the selection does NOT override is the occupancy floor. A drag is a coarse
    gesture -- it selects a stretch of the trace, not a claim about the physics -- and
    the arithmetic it feeds is the same arithmetic as the gap path's: reproduced, two
    4 ms / 300 pA events 600 ms apart, dragged over together, merged into a single
    604 ms event of depth ~0.05 pA (0.017 % of the true blockade), accepted, with a
    blank flag. So a region whose events cover less than :data:`MIN_MERGE_OCCUPANCY` of
    it is REFUSED rather than merged; the GUI tells the user why, and a genuinely dense
    region merges exactly as before.

    Args:
        eventset: The detection to curate.
        signal: The calibrated current trace (1-D).
        fs: Sample rate in Hz.
        t0_s, t1_s: The region bounds in seconds from the start of the signal (order is
            normalised; ``sample = round(t * fs)``).

    Returns:
        A :class:`MergeOutcome` ``(new_eventset, n_merged)`` -- schema-valid,
        ``event_id`` reassigned, and the number of rows merged away. A refusal is
        ``(eventset, 0)`` with ``n_refused == 1``, which is how a caller tells it from
        the ``n_refused == 0`` no-op.
    """
    _validate_merge_input(eventset, signal, fs)
    if not np.isfinite(t0_s) or not np.isfinite(t1_s):
        raise ValueError("merge region bounds must be finite")
    df = eventset.df
    n = len(df)
    if n < 2:
        return MergeOutcome(eventset, 0)

    i0 = int(round(min(float(t0_s), float(t1_s)) * float(fs)))
    i1 = int(round(max(float(t0_s), float(t1_s)) * float(fs)))
    if i1 <= i0:
        return MergeOutcome(eventset, 0)

    order = _sort_order(df)
    starts = df["start_idx"].to_numpy(dtype=np.int64)[order]
    ends = df["end_idx"].to_numpy(dtype=np.int64)[order]

    overlap = _intersects(starts, ends, i0, i1)
    if int(overlap.sum()) < 2:
        return MergeOutcome(eventset, 0)   # nothing (or one event) in the region -> no-op

    # Sorted, non-overlapping spans intersecting an interval form one contiguous block;
    # make it a single group and leave every other event its own group.
    is_group_start = np.ones(n, dtype=bool)
    hit = np.flatnonzero(overlap)
    lo, hi = int(hit[0]), int(hit[-1]) + 1
    is_group_start[lo + 1 : hi] = False

    # The floor is measured on the EVENTS' own span, not on the dragged region: the user
    # is allowed to be generous with the drag (that is what a drag is), and merging is
    # about what sits between the first and last event, not about the empty trace either
    # side of them that the mouse also covered.
    occupancy = span_occupancy(starts[lo:hi], ends[lo:hi])
    if occupancy < MIN_MERGE_OCCUPANCY:
        return MergeOutcome(eventset, 0, n_refused=1, refused_occupancy=(occupancy,))

    note: dict = {"rule": "region", "region_s": (float(t0_s), float(t1_s))}
    marginal = ([(int(starts[lo]), int(ends[lo:hi].max()), occupancy)]
                if occupancy < MARGINAL_MERGE_OCCUPANCY else [])
    _note_occupancy(note, [], marginal)
    out, n_merged = _rebuild(eventset, signal, float(fs), order, is_group_start, note)
    return MergeOutcome(out, n_merged)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _validate_merge_input(eventset: EventSet, signal: np.ndarray, fs: float) -> None:
    """Refuse corrupt rows before a no-op or rebuild can silently retain them."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}")
    current = np.asarray(signal)
    if current.ndim != 1:
        raise ValueError("merge requires a 1-D signal")
    df = eventset.df
    starts = df["start_idx"].to_numpy(dtype=np.int64)
    ends = df["end_idx"].to_numpy(dtype=np.int64)
    invalid = (starts < 0) | (ends > current.size) | (ends <= starts)
    if np.any(invalid):
        i = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"event {i} has invalid span [{starts[i]}, {ends[i]}); "
            f"expected 0 <= start < end <= {current.size}"
        )


def _sort_order(df: pd.DataFrame) -> np.ndarray:
    """Positions that order the rows by ``start_idx`` (stable). Detection output is
    already sorted, but a curated set (a hand-added missed event appends at the end) may
    not be, so never assume it."""
    return np.argsort(df["start_idx"].to_numpy(dtype=np.int64), kind="stable")


def _intersects(starts: np.ndarray, ends: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Boolean mask: event span ``[start, end)`` intersects the half-open ``[i0, i1)``."""
    return (starts < i1) & (ends > i0)


def _apply_occupancy_floor(
    starts: np.ndarray, ends: np.ndarray, is_group_start: np.ndarray
) -> tuple[np.ndarray, list[float], list[tuple[int, int, float]]]:
    """Split every proposed group that is mostly open pore back into singletons.

    A thin alias for :func:`nanoscope.analysis.detect.split_low_occupancy_groups`, kept
    under this module's own name because that is what the merge paths and the comments
    around them call it. The BODY is shared with the detector's ``merge_gap_s`` on
    purpose: two copies of "is this group really one translocation?" would be the same
    mistake as two copies of the gap rule, and the whole reason this floor was needed at
    detect time is that the first copy of it lived only here.
    """
    return split_low_occupancy_groups(starts, ends, is_group_start)


def _note_occupancy(
    note: dict,
    refused: list[float],
    marginal: list[tuple[int, int, float]],
) -> None:
    """Record the occupancy verdicts under :data:`LOW_OCCUPANCY_TOKEN` in the merge note.

    The note becomes ``provenance['merged']``, which is what an export, a later session
    or a reviewer reads to find out what this table has had done to it. A merge that was
    declined and a merge that was kept at the edge of the floor are both facts about the
    table that no column can carry -- ``flag`` belongs to
    :func:`~nanoscope.analysis.characterize.recharacterize_span` and is about the EVENT,
    not about the decision -- so they are stated here, by name, rather than inferred
    later from a row count that does not add up. The key is omitted entirely when there
    is nothing to report, so a clean merge's provenance is unchanged.
    """
    if marginal:
        note[LOW_OCCUPANCY_TOKEN] = [
            {"start_idx": int(s), "end_idx": int(e), "occupancy": round(float(o), 4)}
            for s, e, o in marginal
        ]
    if refused:
        note["refused_low_occupancy"] = [round(float(o), 4) for o in refused]


def _rebuild(
    eventset: EventSet,
    signal: np.ndarray,
    fs: float,
    order: np.ndarray,
    is_group_start: np.ndarray,
    note: dict,
) -> tuple[EventSet, int]:
    """Assemble the merged EventSet from group boundaries over the start-sorted df.

    Singleton groups pass their original row through byte-for-byte; multi-event groups
    are re-characterized into one row. ``event_id`` is reassigned contiguously and the
    input's dtypes / extra columns (the scoring ``added`` marker, the detector's own
    bounds, ``source`` / ``review``) are preserved -- a singleton keeps its own cells
    untouched, and a merged row gets the combination :func:`_combined_provenance_cell`
    defines. A table WITHOUT those columns still merges and still comes back without
    them, which is what makes a project saved before they existed loadable."""
    df = eventset.df
    sorted_df = df.iloc[order].reset_index(drop=True)
    columns = list(df.columns)

    sig = np.asarray(signal)
    provenance = dict(eventset.provenance or {})
    polarity = _polarity(eventset.provenance, sig)
    units = str(provenance.get("units") or "pA")

    bounds = np.flatnonzero(is_group_start).tolist()
    edges = bounds + [len(sorted_df)]

    pieces: list[pd.DataFrame] = []
    for lo, hi in zip(bounds, edges[1:]):
        if hi - lo == 1:
            pieces.append(sorted_df.iloc[[lo]])
        else:
            pieces.append(
                _merged_row(
                    sorted_df.iloc[lo:hi], sig, fs, columns,
                    polarity=polarity, provenance=provenance, units=units,
                )
            )

    out = pd.concat(pieces, ignore_index=True)
    out["event_id"] = np.arange(len(out), dtype=np.int64)
    out = out.astype({c: df[c].dtype for c in columns})[columns]

    n_merged = len(df) - len(out)
    new_prov = dict(eventset.provenance or {})
    new_prov["merged"] = {**note, "n_before": int(len(df)), "n_after": int(len(out))}
    # Merging changes which rows touch an edge (two clipped fragments can become one
    # clipped row), how many rows there ARE, and which are filter-limited, so a
    # carried-over count would describe the table before the merge. This used to
    # re-read ``n_clipped_boundary`` alone and carry ``n_events`` forward -- which is
    # how "1 of 317 event(s)" stood over a 247-row table and the subpopulation report
    # counted 70 merges as 70 rows the user curated out (S7-DET4). Every row-derived
    # count is re-read from the rows through the one rule, so the dict handed to
    # ``EventSet`` is already true rather than true only by correction.
    new_prov = reconcile_row_counts(new_prov, out)
    return EventSet(out, new_prov), int(n_merged)


def _merged_row(
    run_df: pd.DataFrame,
    signal: np.ndarray,
    fs: float,
    columns: list[str],
    *,
    polarity: int,
    provenance: dict | None,
    units: str,
) -> pd.DataFrame:
    """One SCHEMA row for a run of >= 2 events, re-characterized over its full span.

    ``recording`` / ``pore_id`` copy the run's first row; any extra (non-schema)
    columns are carried as the OR of a boolean column, else the run's first value --
    except ``source`` / ``review`` (:func:`_combined_provenance_cell`) and what a
    hand-made row says about itself: ``note`` joins every fragment's note in time
    order, each once, and ``origin`` / ``legacy_measurement_ids`` list every
    fragment's ids once.

    ``flag`` and ``accepted`` are NOT decided here -- :func:`recharacterize_span`
    owns them, which is the whole point. This function used to OR ``accepted`` and
    let the rebuilt row's blank ``flag`` stand, so a fragment rejected for being
    longer than ``max_dwell_s`` came back accepted with its reason erased: measured
    on the owner's -600 mV recording under a declared 2 ms limit, a 5 ms gap merge
    produced 240 accepted rows over the limit, the worst 24.88 ms (12.4x)."""
    start = int(run_df["start_idx"].min())
    end = int(run_df["end_idx"].max())

    row = recharacterize_span(
        signal, fs, start, end,
        provenance=provenance,
        prior=run_df,
        polarity=polarity,
        recording=str(run_df["recording"].iloc[0]),
        pore_id=str(run_df["pore_id"].iloc[0]),
        units=units,
        pad_ms=_PAD_MS,
    )

    # Carry any columns beyond the frozen schema (e.g. scoring's ``added`` marker).
    for col in columns:
        if col in row.columns:
            continue
        series = run_df[col]
        if col in (SOURCE_COL, REVIEW_COL):
            row[col] = _combined_provenance_cell(col, series)
        elif col == NOTE_COL:
            # The first value is arbitrary for a measured column and a loss for these
            # three. A note is a person's words: every fragment's is kept, in time order
            # (the run is start-sorted), by the rule a re-run's carry-over uses.
            row[col] = join_notes(*series)
        elif col in (ORIGIN_COL, LEGACY_IDS_COL):
            # The ids are read back by other doors. Four converted measurements merged
            # into one event kept the first one's id, so the old list offered the other
            # three as unconverted; four merged copies kept one origin, so Copy hand-made
            # events to... called the other three overlaps instead of already copied.
            row[col] = join_ids(*series)
        elif pd.api.types.is_bool_dtype(series):
            row[col] = bool(series.to_numpy(dtype=bool).any())
        else:
            row[col] = series.iloc[0]
    # A whole-event/plateau amplitude belongs to its original span. Keep the
    # historical method/provenance but never publish the first fragment's value
    # as a measurement of the newly merged event.
    from .event_amplitude import invalidate_event_amplitudes

    invalidate_event_amplitudes(row, reason="events_merged")
    return row[columns]


def _combined_provenance_cell(col: str, series: pd.Series) -> Any:
    """``source`` / ``review`` for a row assembled from several -- NOT the first
    fragment's, and NOT re-decided here.

    The generic carry above takes ``series.iloc[0]`` for a non-boolean column, i.e.
    whichever fragment happened to start earliest. For a measured column that is
    merely arbitrary; for these two it is a false statement of provenance -- a
    Schmitt fragment stitched to a hand-added one would come back saying a detector
    authored the whole thing, or that a person did, depending only on which one is
    to the left. `events.combine_sources` / `events.combine_reviews` are the one
    definition of what a merge of two sources means, and this function is a two-line
    router to them:

    * ``source`` -- one distinct source survives as itself; two become ``mixed``.
    * ``review`` -- ``reviewed`` only when EVERY fragment was, and unknown as soon as
      one fragment is. A merged event nobody has looked at as a whole must not
      inherit "reviewed" from the one piece of it that somebody did.

    NO EXCEPTION FOR ``added``, AND THE RULE IT USED TO DUPLICATE HAS ONE HOME.
    This used to overwrite ``combine_sources``' answer with ``manual`` whenever the
    run's ``added`` column ORed True, so that `events.source_disagrees_with_added`
    -- whose test is one token wide, ``added & known & ~manual`` -- would stay quiet
    on a mixed merge. That is the tail wagging the dog twice over. A row stitched
    from a detector's fragment and a person's IS mixed; ``mixed`` is the token that
    vocabulary was invented for, and it is not a claim that no detector authored the
    row, so it does not contradict ``added``. Overwriting it also cost the surface
    the column was built for: ``source_kind`` read the row as ``manual``, dropping a
    detector-found event out of the ``conventional`` count in the export sidecar's
    ``by_kind`` -- the owner's conventional-vs-ML split.

    And the "added wins over source" rule now lives in exactly one place,
    `events.ensure_source_status`, instead of two. It still reaches a merged row:
    a run whose fragments were never stamped combines to ``pd.NA``, and the install
    door (``modules.reconcile_result`` -> ``ensure_source_status``) fills a missing
    source on an ``added`` row with ``manual`` there. What that rule fills is a
    BLANK; what it must not do -- and no longer does from here -- is overwrite a
    recorded, truthful one.
    """
    values = series.to_numpy(dtype=object)
    if col == REVIEW_COL:
        return combine_reviews(values)
    return combine_sources(values)


def _polarity(provenance: dict | None, signal: np.ndarray) -> int:
    """The detector's polarity (from provenance), else one measured from the trace."""
    if provenance is not None:
        p = provenance.get("polarity")
        if p in (-1, 1):
            return int(p)
    try:
        return int(derive_polarity(np.asarray(signal, dtype=np.float64)))
    except ValueError:
        return -1


# The noise scale, the filter-rise flag, the ``max_dwell_s`` limit and the
# boundary-clip determination for a merged span are all read out of the SAME
# provenance by :func:`~nanoscope.analysis.characterize.recharacterize_span`. They
# used to be four little extractor functions here, and a matching set in the Event
# Inspector, and a third set in the analysis panel's hand-added-event path -- which
# is how the same span got measured three subtly different ways.
