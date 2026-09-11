"""Score a detector against the user's curation, and export curated events for training.

This closes the ML loop. After a detector (Keras ML or Schmitt) runs and the user
curates the events -- accepting the real ones, rejecting the false ones, and
MANUALLY ADDING any event the detector missed -- this module turns the two boolean
columns those gestures write (``accepted`` and ``added``) into the standard
detection metrics and into a labelled training table.

THE TWO COLUMNS, AND WHAT THEY MEAN
-----------------------------------
Every event row already carries ``accepted`` (curation: does this survive?). We add
one more, ``added`` (default ``False``): ``True`` only for events the user inserted
by hand because the DETECTOR missed them. ``added`` events are always ``accepted``
(you would not hand-add an event only to reject it).

From those two flags, with the detector's output being "everything it emitted"
(``added == False``) and the truth being "everything the user kept"
(``accepted == True``):

    TP  matched  = accepted & ~added            the model found it, the user kept it
    FP  spurious = ~accepted & ~added & ~auto   the model found it, the USER rejected it
    FN  missed   = added                        the user added it; the model missed it

    precision = TP / (TP + FP)     of what the model flagged, how much was real
    recall    = TP / (TP + FN)     of the real events, how many the model found
    F1        = 2 P R / (P + R)    their harmonic mean

``auto`` is the third hand on the ``accepted`` column: the rows the DETECTOR itself
de-accepted (``clipped_by_boundary``, ``dwell>max_dwell_s``, ``non_finite`` -- read
through :func:`~nanoscope.analysis.events.is_automatic_rejection`, never re-listed
here). They are neither a detection to score nor a truth to keep: the detector
emitted them and then said, on the row, that they are not measurements of an event.
Counting them as the user's false positives is what put "truth (kept) 77, spurious
(FP) 240, precision 0.243" on screen for a table nobody had curated (S7-DET2); they
are reported separately as ``n_auto_rejected`` and excluded from ``n_detected``.

Every denominator is guarded: an undefined ratio (no detections, nothing real, both
zero) is reported as ``None`` -- ``n/a`` -- never as ``0.0`` or a raised exception,
because "the model flagged nothing so precision is undefined" is a true statement
and ``0.0`` would be a lie.

A frame with NO ``added`` column is treated as all-``False`` (nothing was hand-added
yet), so scoring works the moment a detection exists, before any curation -- and
``ScoreResult.n_user_gestures`` (``fp + fn``) is what says whether any has happened.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..core import clock
from ..core.atomic_write import staged_write_set
from .baseline import BASELINE_PARAM_KEYS
from .events import (
    REVIEW_COL,
    SOURCE_COL,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    event_status,
    has_source_status,
    is_automatic_rejection,
    source_kind,
)
from .remeasure import REMEASURED_KEY

__all__ = [
    "ADDED_COLUMN",
    "CLOCK_KEY",
    "CLOCK_BLOCK_KEY",
    "RECORDING_CLOCK_COLUMNS",
    "clock_stamp",
    "ScoreResult",
    "added_mask",
    "auto_rejected_mask",
    "ensure_added_column",
    "score_detection",
    "TRAINING_COLUMNS",
    "TRAINING_SELECTION_ACCEPTED",
    "TRAINING_SELECTION_REVIEWED",
    "TRAINING_SELECTION_ALL",
    "TRAINING_SELECTIONS",
    "build_training_frame",
    "training_provenance",
    "write_training_export",
    "training_export_paths",
    "BASELINE_CONTRACT_KEYS",
    "events_export_paths",
    "write_events_export",
    "CURATION_BLOCK_KEY",
    "curation_stamp",
]

#: The baseline block every export sidecar carries, in provenance spelling. Built
#: from `baseline.BASELINE_PARAM_KEYS` rather than re-spelled, so a new estimator
#: parameter reaches the exports the day it is added and cannot be forgotten here.
BASELINE_CONTRACT_KEYS: tuple[str, ...] = (
    *BASELINE_PARAM_KEYS,
    "baseline_open_pore_reference",
    "baseline_reference_applied",
    "baseline_reference_details",
    "baseline_estimator_bypassed",
    "baseline_polarity_role",
    "baseline_fingerprint",
    "baseline_param_source",
    "baseline_polarity",
    "baseline_mask_fraction",
    "baseline_provided",
    "baseline_fallback",
    # Not a ``baseline_`` key, and deliberately in here anyway: it is the ONE thing
    # that tells a downstream reader whether this CSV was measured by the estimator
    # it names at DETECTION time, or re-measured against it afterwards -- and, if
    # re-measured, which estimator it came from. Without it a migrated table and a
    # freshly detected one are indistinguishable in the sidecar, which is exactly
    # the question §4.2.6's release note exists because old exports cannot answer.
    REMEASURED_KEY,
)

# ================================================================================================
# THE CLOCK EVERY EXPORT THIS MODULE WRITES IS ON -- said once, in one place.
# ================================================================================================
#
# THE DEFECT THIS CLOSES
# ----------------------
# ``core.clock``'s rule is "One clock, or every control that shows or takes a time says
# which clock it is in", and the app keeps it everywhere a time is on SCREEN: the trace
# axis is labelled with its t0, the event table's ``t_start_s`` header renders
# ``t_start_s (t0=0)`` with a tooltip spelling out the conversion, and
# ``measure.MEASUREMENT_SCHEMA`` gives the hand-measurement export a ``t0_s`` COLUMN so a
# pooled table stays readable ("t_start_s + t0_s is the instant the trace axis showed").
#
# The files THIS module writes are the copies that leave the machine, and they said
# nothing at all. One session on the owner's quick apoferritin ABF produced, in one
# folder: measurements.csv carrying ``t0_s = 1154.99091``, events.csv carrying
# ``t_start_s = 0.2523`` with no origin anywhere in it or in its sidecar, and
# figs_trace.csv putting those same samples at x = 1154.997. Three files, one recording,
# two clocks, nothing on disk to reconcile them -- and on a WINDOWED load it is worse:
# two 8 s windows of one recording dated 57 matched events 6.000 s apart while their two
# sidecars were key-identical, because ``analysed_start_idx`` is window-local (0 in both)
# and no key names the window's origin.
#
# WHY THE SIDECAR AND NOT A COLUMN
# --------------------------------
# The 20-column event schema is frozen and ``write_events_export`` promises "no new
# column, no reordering" -- ``tests/analysis/test_baseline_seam.py`` pins that the writer
# "adds no column of its own", and ``TRAINING_COLUMNS`` is likewise a contract with the
# StudentModel notebook. The sidecar is the channel the frozen schema deliberately leaves
# open (it is already how ``baseline_fingerprint`` and ``frac_depth_note`` travel), so
# that is where the clock goes -- at the TOP LEVEL, next to ``fs``/``units``, because a
# reader looking for the units of a column looks in exactly one place.
#
# UNKNOWN IS SAID, NEVER GUESSED
# ------------------------------
# ``t0_s = None`` means the writing surface did not supply the origin. It is written out
# as an explicit null with a note that says so, following MEASUREMENT_SCHEMA's own
# convention for its ``t0_s`` column ("NaN when unknown"): a 0.0 filled in here would be
# indistinguishable from a recording that genuinely starts at 0, which is the same class
# of lie as a rebased axis.
#
# AND ``t0_s`` MEANS WHAT MEASUREMENT_SCHEMA SAYS IT MEANS
# --------------------------------------------------------
# Same spelling is not yet one voice. ``measure.MEASUREMENT_SCHEMA``'s ``t0_s`` is
# defined as ``clock.instrument_origin_s`` -- "for a segment, its parent recording's
# clock, not the segment's own rebased axis; a layer alignment nudge is a display shift
# and is never included" -- and ``gui/measure_panel`` stamps exactly that, naming
# ``clock.clock_offset_s`` in a comment as the WRONG function for this key. The two
# differ on the two document classes the user actually makes: on a SEGMENT of the
# owner's quick apo ABF ``instrument_origin_s`` is 1158.9909 s where ``clock_offset_s``
# is 0.0, and under a 2.5 s layer alignment nudge they differ by that nudge. A
# measurements.csv and an events.meta.json in ONE export folder disagreeing by 1158.99 s
# under one key name is the very defect this block exists to close, one level down.
#
# So the origin is taken HERE, from the document's ``meta``, through
# ``clock.instrument_origin_s`` -- the caller hands over the meta and never picks a
# function. A bare number is still accepted (a stored result reloaded from a project has
# no meta), and it is taken at face value as that same quantity.

#: The sidecar key naming the instrument-clock origin. Spelled exactly as
#: ``measure.MEASUREMENT_SCHEMA``'s column, AND meaning exactly what that column means:
#: ``t_start_s + t0_s`` is the row's instant on the instrument clock.
CLOCK_KEY = "t0_s"

#: The sidecar key holding the sentence + the columns it applies to.
CLOCK_BLOCK_KEY = "clock"

#: Columns in the tables this module writes that are INSTANTS on the recording clock.
#: ``dwell_s`` is deliberately absent: a duration is a LENGTH of time, on no clock, and
#: annotating it would be noise (the same call ``gui/measure_panel`` makes).
RECORDING_CLOCK_COLUMNS: tuple[str, ...] = ("t_start_s", "t_end_s")

_CLOCK_UNKNOWN_NOTE = (
    "Time seconds from the start of this recording: 0 is the first sample. The "
    "instrument-clock origin t0 of this recording was NOT supplied by the surface that "
    "wrote this file, so a time in this table cannot be placed on the trace axis, and "
    "two exports taken from two different windows of the same recording cannot be told "
    "apart by their times alone."
)


def _finite_or_none(value: Any) -> float | None:
    """``float(value)`` when it is a real number, else None. NaN/inf read as unknown."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _origin_and_offset(
    meta: Any, t0_s: Any, provenance: Mapping[str, Any] | None
) -> tuple[float | None, Any, bool]:
    """``(t0_s to stamp, the offset to write the note from, was a meta supplied)``.

    ONE resolution order for every writer in this module, so no two sidecars can be
    built from two different ideas of what ``t0_s`` is:

    1. ``meta`` -- the document itself. ``clock.instrument_origin_s`` is the stamped
       number (MEASUREMENT_SCHEMA's definition, verbatim: a segment's PARENT clock, no
       layer nudge) and ``clock.clock_offset_s`` is what the note is written from,
       because that is the :class:`~nanoscope.core.clock.ClockOffset` carrying the
       rebase — which is what makes the sentence in the file byte-identical to the one
       the on-screen table header shows for the same document.
    2. a bare ``t0_s`` number -- taken at face value as that same quantity. A
       :class:`~nanoscope.core.clock.ClockOffset` handed in here keeps its rebase
       origin (``float()`` used to strip it, so a rebased document's sidecar got the
       "t₀ is 0, both clocks read alike" sentence): the stamped number is then the
       distance to the instrument clock, ``clock.to_instrument_s(offset, 0.0)``.
    3. ``provenance["t0_s"]`` -- the channel for a caller that cannot change its call
       signature, or for a result reloaded from a project where the meta is gone.
    """
    if meta is not None:
        return (_finite_or_none(clock.instrument_origin_s(meta)),
                clock.clock_offset_s(meta), True)
    for candidate in (t0_s, None if provenance is None else provenance.get(CLOCK_KEY)):
        if candidate is None:
            continue
        rebased = clock.rebased_origin_s(candidate)
        if rebased is not None:
            # A ClockOffset: its float IS the on-screen distance and the rebase is the
            # rest of the way to the instrument clock. to_instrument_s does that sum.
            total = _finite_or_none(clock.to_instrument_s(candidate, 0.0))
            if total is not None:
                return total, candidate, False
            continue
        value = _finite_or_none(candidate)
        if value is not None:
            return value, value, False
    return None, None, False


def clock_stamp(
    df: pd.DataFrame | None = None,
    *,
    columns: Any = None,
    meta: Any = None,
    t0_s: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The clock block EVERY sidecar written by this module carries. THE one writer of it.

    Args:
        df: the table being written, used only to name which of its columns are
            instants on the recording clock (:data:`RECORDING_CLOCK_COLUMNS`).
        columns: the written column names, when the table itself is not in hand
            (``training_provenance`` builds its sidecar before the frame). Passing
            neither leaves the list at the full set.
        meta: the ``RecordingMeta`` of the document these rows came from — THE way to
            supply the origin, because it is the one that cannot be got wrong: the
            caller hands over the document and this function picks the quantity
            ``measure.MEASUREMENT_SCHEMA`` defines (``clock.instrument_origin_s``)
            instead of the caller picking between two functions that differ by
            1158.99 s on a segment. Takes precedence over ``t0_s``/``provenance``.
        t0_s: the same quantity as a bare number, for a caller with no meta in hand:
            the recording's instrument-clock origin, i.e. the experimental time of its
            first sample (``clock.instrument_origin_s``), NOT
            ``clock.clock_offset_s`` — see the module header. ``None``, or a
            non-finite value, means the caller does not know it; that is recorded as
            an explicit null with a note saying so, never filled in with 0.
        provenance: the detection's provenance, consulted for ``t0_s`` when neither
            of the above is given.

    Returns:
        ``{"t0_s": <float|None>, "clock": {...}}`` -- merge into the sidecar top level.
        The note is :func:`nanoscope.core.clock.field_note`'s own sentence, so the file
        and the on-screen column header describe the clock identically — including on a
        rebased cut, whose sentence is a different one and only ``meta`` (or a
        ``ClockOffset``) carries what it needs.
    """
    t0, note_offset, from_meta = _origin_and_offset(meta, t0_s, provenance)

    written = columns if columns is not None else (
        None if df is None else getattr(df, "columns", None))
    cols = list(RECORDING_CLOCK_COLUMNS)
    if written is not None:
        present = set(written)
        cols = [c for c in RECORDING_CLOCK_COLUMNS if c in present]

    if t0 is None:
        # Two different unknowns, and they have different causes, so they get different
        # sentences: a document with no sample rate HAS no clock (core.clock says so in
        # those words), while a missing meta/t0_s is a surface that did not say.
        note = (clock.field_note(clock.RECORDING, None) if from_meta
                else _CLOCK_UNKNOWN_NOTE)
    else:
        # The note comes from the OFFSET, not from the stamped number: on a rebased cut
        # they are different objects (0.0 carrying a 1158.99 s origin, vs 1158.99) and
        # only the offset makes core.clock produce the cut's own sentence — the one the
        # event-table header shows for that same document.
        note = clock.field_note(clock.RECORDING, note_offset)

    block: dict[str, Any] = {
        # WHICH clock, in core.clock's own vocabulary -- "recording" / "instrument".
        "columns_clock": clock.RECORDING,
        "columns": cols,
        CLOCK_KEY: t0,
        "t0_s_known": t0 is not None,
        # The instrument instant of THIS document's own sample 0 when the app rebased it
        # (a segment), else null. It is what makes the difference between ``t0_s`` and a
        # bare ``meta.t0_s`` machine-readable instead of only stated in the note.
        "rebased_from_s": clock.rebased_origin_s(note_offset),
        "note": note,
    }
    return {CLOCK_KEY: t0, CLOCK_BLOCK_KEY: block}


# The extra bool column, kept OUT of ``events.SCHEMA``/``REQUIRED`` on purpose: the
# schema is a subset contract, so an EventSet may carry extra columns, and a detector
# still produces the canonical 20 columns. The GUI (or a hand-added event) materialises
# this column; a frame without it scores as if nothing was hand-added.
ADDED_COLUMN = "added"


@dataclass(frozen=True)
class ScoreResult:
    """The detector's score against the user's curation of ONE detection.

    Counts are always defined; the three ratios are ``None`` when their denominator
    is zero (reported as ``n/a`` in the UI). ``matched_ids`` / ``missed_ids`` /
    ``spurious_ids`` are the ``event_id`` values behind TP / FN / FP, so the UI can
    point at exactly which events each bucket holds.
    """

    n_detected: int
    n_truth: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    matched_ids: list[int] = field(default_factory=list)
    missed_ids: list[int] = field(default_factory=list)
    spurious_ids: list[int] = field(default_factory=list)
    #: Rows the DETECTOR rejected on its own (``accepted=False`` with a flag that is
    #: fully explained by `events.AUTOMATIC_REJECTION_FLAGS` -- boundary-clipped, too
    #: long, non-finite). NOT in ``n_detected``, ``tp`` or ``fp``: the detector
    #: emitted them and then said itself that they are not measurements of an event,
    #: so they are neither a detection to score nor a truth to keep, and nobody
    #: curated them. Appended with a default so the positional shape above -- what
    #: every existing constructor and the training export's readers rely on -- is
    #: unchanged.
    n_auto_rejected: int = 0
    auto_rejected_ids: list[int] = field(default_factory=list)

    @property
    def n_user_gestures(self) -> int:
        """How many curation GESTURES the score rests on: hand rejections (``fp``) plus
        hand-added events (``fn``). Zero means nobody has curated this table yet and the
        three ratios follow from the construction, not from a measured disagreement."""
        return int(self.fp) + int(self.fn)

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view (numbers + id lists), for logging or a JSON sidecar."""
        return asdict(self)


def added_mask(df: pd.DataFrame) -> np.ndarray:
    """The boolean ``added`` mask for ``df`` -- all-``False`` when the column is absent.

    Centralises the "a missing ``added`` column means nothing was hand-added" rule so
    scoring and export never diverge on it.
    """
    n = len(df)
    if ADDED_COLUMN not in df.columns:
        return np.zeros(n, dtype=bool)
    return df[ADDED_COLUMN].to_numpy(dtype=bool)


def ensure_added_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a default-``False`` ``added`` column to ``df`` IN PLACE if it is missing.

    Returns the same frame for convenience. Used when a detection is first shown (so
    the column round-trips through save/load and exports) and before a hand-added
    event is appended. Idempotent: an existing column is left untouched.
    """
    if ADDED_COLUMN not in df.columns:
        df[ADDED_COLUMN] = np.zeros(len(df), dtype=bool)
    return df


def _ratio(num: int, den: int) -> float | None:
    """``num/den`` as a float, or ``None`` when ``den`` is zero (an undefined ratio)."""
    return (num / den) if den > 0 else None


def score_detection(df: pd.DataFrame) -> ScoreResult:
    """Score the detector behind ``df`` against the curation recorded in ``df``.

    ``df`` is a detection's per-event frame carrying at least ``accepted`` (and,
    optionally, ``added`` and ``event_id``). See the module docstring for the exact
    TP/FP/FN definitions. An empty frame scores all-zero with ``None`` ratios.
    """
    n = len(df)
    if n == 0 or "accepted" not in df.columns:
        return ScoreResult(0, 0, 0, 0, 0, None, None, None, [], [], [])

    accepted = df["accepted"].to_numpy(dtype=bool)
    added = added_mask(df)
    if "event_id" in df.columns:
        ids = df["event_id"].to_numpy(dtype=np.int64)
    else:
        ids = np.arange(n, dtype=np.int64)

    # A row the DETECTOR rejected is not a user's verdict. ``accepted=False`` is written
    # by two hands -- `characterize` (boundary-clipped, too long, non-finite) and the
    # curator -- and only the second is a gesture this score is about. Read through
    # `events.is_automatic_rejection`, the ONE definition of "the app rejected this
    # row, not a person" (the same one the undo recorder and the re-measure carry use),
    # so a fourth rejecting rule reaches this line with no second edit.
    auto = auto_rejected_mask(df)

    detected = ~added & ~auto         # the model's output it stands behind
    tp_mask = accepted & ~added       # matched:  model found it, user kept it
    fp_mask = ~accepted & ~added & ~auto   # spurious: model found it, USER rejected it
    fn_mask = added                   # missed:   user added it; model missed it

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    n_detected = int(detected.sum())
    n_truth = int(accepted.sum())

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0.0:
        f1: float | None = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return ScoreResult(
        n_detected=n_detected,
        n_truth=n_truth,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        matched_ids=[int(i) for i in ids[tp_mask]],
        missed_ids=[int(i) for i in ids[fn_mask]],
        spurious_ids=[int(i) for i in ids[fp_mask]],
        n_auto_rejected=int(auto.sum()),
        auto_rejected_ids=[int(i) for i in ids[auto]],
    )


def auto_rejected_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows the DETECTOR de-accepted on its own: ``accepted=False``, not hand-added,
    and a ``flag`` that `events.is_automatic_rejection` says is fully explained by the
    rules a rebuild re-applies (``clipped_by_boundary``, ``dwell>max_dwell_s``,
    ``non_finite``). All-``False`` for a frame with no ``flag`` column, because a row
    with no reason on it was rejected by a person (a bare Reject click).

    Sweep 7 measured what counting these as the user's false positives did: on the
    owner's -600 mV file with a 5 ms max dwell, "Detector score (vs your curation)"
    rendered truth (kept) 77, spurious (FP) 240, precision 0.243, F1 0.391 with zero
    curation gestures made -- the detector's honest self-flagging published as its
    own false-positive rate, and the tautology caption hidden by it (S7-DET2).
    """
    n = len(df)
    if n == 0 or "accepted" not in df.columns or "flag" not in df.columns:
        return np.zeros(n, dtype=bool)
    accepted = df["accepted"].to_numpy(dtype=bool)
    added = added_mask(df)
    flags = df["flag"].to_numpy(dtype=object)
    return np.array(
        [(not a) and (not h) and is_automatic_rejection(f)
         for a, h, f in zip(accepted, added, flags)],
        dtype=bool,
    )


# ================================================================================================
# training export -- the curated events as a regions/labels table for the StudentModel notebook.
# ================================================================================================

# One row per event, in this order. These are exactly the fields the StudentModel
# notebook's ``build_balanced_student_dataset`` consumes to cut labelled windows out
# of the source recording: the region (sample + time bounds), a coarse feature
# (depth), and the two labels (curation ``accepted`` + hand-added ``added``).
TRAINING_COLUMNS = [
    "recording",
    "pore_id",
    "start_idx",
    "end_idx",
    "t_start_s",
    "dwell_s",
    "depth_pA",
    "accepted",
    "added",
]

# Stable values accepted by the public training-export ``selection`` keyword and written
# verbatim to its sidecar. ``include_all`` remains the backwards-compatible boolean alias
# for the first/last choices; the middle choice is intentionally explicit because it is
# the only one that promises every row carries a human decision.
TRAINING_SELECTION_ACCEPTED = "accepted"
TRAINING_SELECTION_REVIEWED = "reviewed"
TRAINING_SELECTION_ALL = "all"
TRAINING_SELECTIONS = (
    TRAINING_SELECTION_ACCEPTED,
    TRAINING_SELECTION_REVIEWED,
    TRAINING_SELECTION_ALL,
)

# WHY `source` AND `review` ARE NOT IN THAT LIST. They are per-row facts now
# (`events.SOURCE_COL` / `events.REVIEW_COL`) and they ride in the events CSV, so the
# question "should the training table carry them too?" has to be answered rather than
# left to whoever adds a column next. It is answered NO, on four counts:
#
#  1. `source` WOULD BE A LEAKY FEATURE, and the leak is the exact one this loop exists
#     to escape. The owner's note is explicit -- "Final analyses should normally use all
#     accepted events, regardless of how they were detected" -- and a model handed a
#     column that says which detector found the row learns to reproduce that detector's
#     decisions instead of learning the signal. The whole point of training on the
#     CURATED set is that the label is the user's judgement, not the finder's identity.
#  2. `added` ALREADY CARRIES THE ONE SPLIT THE LABELS NEED (hand-added = the detector
#     missed it = FN), and `events.ensure_source_status` stamps `source` to AGREE with
#     it -- "added wins over source". A `source` column would be a SECOND writer of a
#     fact `added` already states, which is the drift this design refused everywhere
#     else (see `events.reconcile_row_counts`).
#  3. `review` is a fact about the CURATION PROCESS, not about the event. What a review
#     PRODUCES is `accepted`, and `accepted` is already the label.
#  4. TRAINING_COLUMNS is a frozen contract: the StudentModel notebook's
#     `build_balanced_student_dataset` consumes exactly this list, and
#     `tests/analysis/test_s3_clock_export_sidecar.py` asserts the written CSV's columns
#     equal it. Growing it moves an exported table, which this change may not do.
#
# The pair is NOT thereby lost to the owner: the counts he needs in order to know what
# mix his model was trained on (which detectors, how many rows a person had ruled on,
# how many were withheld) travel in the sidecar's `curation` block -- COUNTS, NOT
# COLUMNS, which is the additive door this schema leaves open. See
# :func:`training_provenance`.


def _resolve_training_selection(*, include_all: bool, selection: str | None) -> str:
    """Resolve the new selection vocabulary without moving the legacy boolean API.

    ``selection=None`` is exactly the historical behaviour: ``include_all=False`` means
    accepted rows and ``include_all=True`` means every row. Supplying two contradictory
    instructions is rejected rather than silently exporting a different label set than
    the caller requested.
    """
    if selection is None:
        return TRAINING_SELECTION_ALL if include_all else TRAINING_SELECTION_ACCEPTED
    selected = str(selection).strip().lower()
    if selected not in TRAINING_SELECTIONS:
        allowed = ", ".join(repr(value) for value in TRAINING_SELECTIONS)
        raise ValueError(f"training selection must be one of {allowed}; got {selection!r}")
    if include_all and selected != TRAINING_SELECTION_ALL:
        raise ValueError(
            f"include_all=True conflicts with selection={selected!r}; "
            f"use selection={TRAINING_SELECTION_ALL!r} or omit include_all")
    return selected


def _training_row_mask(
    df: pd.DataFrame,
    *,
    include_all: bool,
    selection: str | None = None,
) -> np.ndarray:
    """THE one definition of which rows a training export writes.

    Shared by :func:`build_training_frame` (which writes them) and
    :func:`write_training_export` (which describes them in the sidecar), so the count in
    the ``curation`` block cannot come to disagree with the table it sits beside.

    ``reviewed`` means a human decision of either sign: canonically derived status
    ``accepted`` OR ``rejected``. It therefore carries positive and negative supervised
    labels while excluding untouched accepted candidates, legacy rows whose review state
    is unknown, and detector-owned automatic rejections. In particular, a non-automatic
    rejected row is a human rejection even when an old project lacks a ``review`` token;
    :func:`event_status` is the single authority for that inference.
    """
    n = len(df)
    selected = _resolve_training_selection(include_all=include_all, selection=selection)
    if selected == TRAINING_SELECTION_ALL:
        return np.ones(n, dtype=bool)
    if selected == TRAINING_SELECTION_REVIEWED:
        # A frame with no accepted label cannot prove a human verdict. Fail closed for
        # this opt-in mode; the legacy accepted selection below retains its old fallback.
        if "accepted" not in df.columns:
            return np.zeros(n, dtype=bool)
        statuses = event_status(df)
        mask = statuses.isin((STATUS_ACCEPTED, STATUS_REJECTED)).to_numpy(dtype=bool)
        # Reconciliation preserves contradictory human accept/reject rulings in a
        # dedicated column. Such a row has been reviewed, but it does not yet carry one
        # resolved target label. Keep it out of the safe reviewed selection until an
        # explicit curation gesture clears the marker. The literal spelling avoids a
        # scoring -> reconcile import cycle (reconcile itself imports scoring).
        conflict_col = "consensus_curation_conflict"
        if conflict_col in df.columns:
            unresolved = (
                df[conflict_col].astype("string").fillna("").str.strip().ne("")
                .to_numpy(dtype=bool)
            )
            mask &= ~unresolved
        return mask
    if "accepted" not in df.columns:
        return np.ones(n, dtype=bool)
    return df["accepted"].to_numpy(dtype=bool)


def build_training_frame(
    df: pd.DataFrame,
    *,
    include_all: bool = False,
    selection: str | None = None,
) -> pd.DataFrame:
    """Reduce a detection frame to the labelled-region training table.

    Defaults to the ACCEPTED events -- the user's curated truth, which INCLUDES every
    hand-added event (they are accepted by construction). Pass ``include_all=True`` to
    keep the rejected (spurious) events too, so a model can also learn the negatives.
    These legacy choices are unchanged. ``selection='reviewed'`` instead keeps only
    human-reviewed decisions, of either sign, excluding unreviewed/unknown candidates and
    automatic detector rejections. ``selection='accepted'`` and ``selection='all'`` are
    the explicit spellings of the two legacy modes.

    Missing ``added`` is materialised as all-``False``; the columns come out in
    :data:`TRAINING_COLUMNS` order with a fresh ``RangeIndex``.
    """
    work = df.copy()
    ensure_added_column(work)
    # Select first. Parsing all rows once per vocabulary key was O(rows × keys), and
    # labels that occurred only on withheld rows created all-empty training columns.
    # The output schema should describe the rows actually exported.
    work = work[_training_row_mask(
        work, include_all=include_all, selection=selection)].copy()
    # The user's labels, one column per key. Without this the export carries only
    # accepted/added — "is this event real", never "what is it" — so a model trained from it
    # can only ever learn the detector's job over again. The keys come from the data rather
    # than from a fixed list, because the vocabulary is the user's (see analysis.labels).
    from .labels import label_column, label_keys
    # ``label_*`` is a GENERATED namespace. Never trust a same-named input column: it may
    # be stale after somebody changed the canonical token in ``flag`` or may name a key no
    # row carries any more. Dropping the entire namespace before regeneration prevents
    # both a direct collision and orphaned training features.
    stale_label_cols = [
        col for col in work.columns
        if isinstance(col, str) and col.startswith("label_")
    ]
    if stale_label_cols:
        work.drop(columns=stale_label_cols, inplace=True)
    if "flag" in work.columns:
        for key in label_keys(work["flag"].tolist()):
            col = f"label_{key}"
            work[col] = label_column(work["flag"], key)
    for col in TRAINING_COLUMNS:
        if col not in work.columns:
            work[col] = pd.Series([pd.NA] * len(work), index=work.index)
    label_cols = [c for c in work.columns if c.startswith("label_")]
    return work[TRAINING_COLUMNS + label_cols].reset_index(drop=True)


def training_provenance(
    result_provenance: dict[str, Any] | None,
    *,
    source_file: str = "",
    detector: str = "",
    include_all: bool = False,
    selection: str | None = None,
    n_rows: int | None = None,
    columns: Iterable[str] | None = None,
    meta: Any = None,
    t0_s: float | None = None,
    df: pd.DataFrame | None = None,
    source_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build the reproducibility sidecar for a training export.

    Carries the source file, the sample rate and polarity, the detector/model that
    produced the events, the resolved ``accepted`` / ``reviewed`` / ``all`` selection,
    and the detrend/filter contract the detection ran under (so the exact preprocessing
    that made these regions is recoverable). ``fs`` and the preprocessing flags are
    pulled from the detection's own provenance dict, coerced to JSON-safe builtins.

    ...and the CLOCK the table's ``t_start_s`` is on (:func:`clock_stamp`). A training
    table is cut back out of the source recording by sample index, so a consumer that
    reaches for the seconds column instead needs to know which clock it is on -- and a
    windowed load's regions are ``t0_s`` seconds early on the file's own time base.
    ``meta`` (the document's ``RecordingMeta``) is the way to supply the origin; ``t0_s``
    takes the same quantity as a bare number when no meta is in hand.

    ...and WHO FOUND THESE ROWS, and how many did not come. ``selection: accepted`` said
    the export was curated and ``n_rows`` said how many survived, and between them
    nothing said how many did NOT -- the same silence :func:`curation_stamp` was written
    for on the events export, in the file the owner actually feeds his models. Pass the
    exported rows as ``df`` (the EVENT rows, which still carry ``source``/``review`` --
    :func:`build_training_frame` drops them, deliberately: see the note under
    :data:`TRAINING_COLUMNS`) and the source table as ``source_df``, and the sidecar
    carries the ``curation`` block: ``by_source`` / ``by_kind`` / ``by_status`` over the
    rows written, ``withheld_by_*`` over the rows withheld. COUNTS, NOT source/review
    COLUMNS -- the table keeps :data:`TRAINING_COLUMNS` as its fixed leading contract and
    may append the user's dynamic ``label_*`` columns. Pass those actual written column
    names as ``columns`` so the sidecar describes the complete table schema; direct
    provenance callers default to the fixed contract.
    """
    prov = dict(result_provenance or {})
    selected = _resolve_training_selection(include_all=include_all, selection=selection)
    fs = prov.get("fs")
    written_columns = list(TRAINING_COLUMNS if columns is None else columns)
    meta: dict[str, Any] = {
        "format": "nanoscope-training",
        "version": 1,
        "source_file": str(source_file),
        "detector": str(detector),
        "fs": float(fs) if fs is not None else None,
        "polarity": prov.get("polarity"),
        "units": prov.get("units"),
        "selection": selected,
        "columns": written_columns,
        # Use the same complete schema as the sidecar's top-level declaration.
        **clock_stamp(columns=written_columns, meta=meta, t0_s=t0_s, provenance=prov),
    }
    if n_rows is not None:
        meta["n_rows"] = int(n_rows)
    if df is not None:
        # The additive door for source/review, taken. `curation_stamp` is THE one writer
        # of this block, so the training sidecar and the events sidecar cannot come to
        # describe the same curation differently.
        meta.update(curation_stamp(
            df, source_df=source_df, provenance=prov, selection=selected
        ))
        # TRAINING_COLUMNS intentionally omits event_id to preserve the model-input schema,
        # while reconciliation memberships are keyed by that id. Carry the selected ids in
        # exact table-row order so an external consumer can join each exported row back to
        # its full reconciliation audit without introducing an input feature.
        if "event_id" in df.columns:
            meta["exported_event_ids"] = [int(value) for value in df["event_id"]]
    # A reconciled table is otherwise impossible to trace back to the detector rows that
    # voted for each exported event: the frozen training columns intentionally omit source
    # and reconciliation-membership columns to avoid feature leakage. The versioned block
    # in EventSet provenance is therefore carried intact in the sidecar, where it remains
    # audit metadata rather than a model input.
    if "reconciliation" in prov:
        meta["reconciliation"] = prov["reconciliation"]
    # Session-wide label definitions are not detector provenance, but the GUI supplies a
    # copy under this additive key for training exports. Keeping key/value/hotkey/note in
    # the sidecar lets an external trainer interpret dynamic ``label_*`` columns without
    # reverse-engineering the project file; it remains metadata and never enters features.
    if "label_vocabulary" in prov:
        meta["label_vocabulary"] = prov["label_vocabulary"]
    # The detrend / filter contract, so the preprocessing that produced the regions is
    # reproducible. Only the keys the detection actually stamped are carried through.
    contract_keys = (
        "min_dwell_s", "max_dwell_s", "merge_gap_s", "filter_cutoff_hz", "filter_rise_s",
        "noise_rms_pA", "baseline_median_pA",
        # WHICH OPEN-PORE ESTIMATOR MEASURED THESE DEPTHS. Without these an exported
        # table is unreadable a year later: on a recording with the owner's event
        # statistics the same events yield depths 4.7x apart depending on the
        # estimator, and nothing in the 20 frozen columns says which one ran.
        # ``baseline_fingerprint`` is the one key that makes the answer checkable
        # rather than merely stated.
        *BASELINE_CONTRACT_KEYS,
        "detrend_enabled", "detrend_method", "detrend_window_s",
        "notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q",
        "median_enabled", "median_window_s", "hp_enabled", "hp_cutoff_hz",
        "hp_order", "hp_kind",
        "rmean_enabled", "rmean_window_s",
        "savgol_enabled", "savgol_window_s", "savgol_polyorder",
        # Dual-signal classical detection. The top-level baseline/noise block names
        # the trace that produced the exported row values; these fields retain the
        # high-pass localization contract and its independently measured diagnostics.
        "highpass_localization_only", "characterization_hp_omitted",
        "detection_signal_domain", "characterization_signal_domain",
        "detection_noise_rms_pA", "detection_noise_method",
        "measurement_noise_rms_pA", "threshold_noise_source",
        "detection_baseline", "detection_baseline_median_pA",
        "detection_baseline_warnings",
        "detection_polarity", "detection_polarity_source",
        "detection_polarity_measured", "detection_polarity_evidence",
        "candidate_signal_polarity_source",
        "candidate_signal_polarity_measured",
        "candidate_signal_polarity_evidence",
        "characterization_polarity_measured",
        "characterization_polarity_evidence",
        "polarity_domain_disagreement",
        "polarity_retry_after_highpass_mismatch",
        "candidate_initial_polarity", "characterization_resolved_polarity",
        "candidate_localization_attempts",
    )
    contract = {k: prov[k] for k in contract_keys if k in prov}
    meta["detrend_contract"] = contract
    # Coerce the whole sidecar to JSON-safe builtins in one pass: fs/polarity/units and
    # every contract value may be a numpy scalar carried straight from the detector.
    return _json_safe(meta)


def _json_safe(obj: Any) -> Any:
    """Coerce numpy scalars/arrays (and nested containers) to JSON-encodable builtins."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def training_export_paths(path: str | Path) -> tuple[Path, ...]:
    """Every file :func:`write_training_export` will write for ``path``, in write order.

    Two, not one: the table itself (with the suffix normalised exactly as the writer
    normalises it — a bare name becomes ``.csv``) AND the ``<stem>.meta.json``
    provenance sidecar. A GUI that confirms only the name the user typed silently
    clobbers that sidecar, which is the same defect as the figure export's siblings.

    The normalisation is shared with the writer (:func:`_training_data_path`) so the
    two can never disagree about which file is at stake.
    """
    data = _training_data_path(path)
    return (data, data.with_suffix(".meta.json"))


def _training_data_path(path: str | Path) -> Path:
    """The table path :func:`write_training_export` will actually write.

    ``.parquet`` selects parquet and is kept verbatim; anything else writes CSV, and a
    bare (suffixless) name gains ``.csv``. THE one place this rule lives.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return path
    return path.with_suffix(".csv") if path.suffix == "" else path


def events_export_paths(path: str | Path) -> tuple[Path, ...]:
    """Every file :func:`write_events_export` will write for ``path``, in write order.

    Two, not one: the CSV and its ``<stem>.meta.json`` provenance sidecar. Same
    contract, and the same reason, as :func:`training_export_paths`.
    """
    p = Path(path)
    data = p.with_suffix(".csv") if p.suffix == "" else p
    return (data, data.with_suffix(".meta.json"))


#: The sidecar key holding the curation block: what was withheld from the CSV beside it,
#: and how the rows it DOES hold break down by who found them and who has ruled on them.
#: Additive, like ``frac_depth_note`` and the clock block -- the 20-column schema is
#: frozen and ``provenance``/the sidecar are the channel it leaves open.
CURATION_BLOCK_KEY = "curation"

_WITHHELD_NOTE = (
    "A withheld row is a row of the source table with accepted=False -- rejected by "
    "hand during curation, or de-accepted by the detector itself (clipped by the "
    "boundary, over max_dwell_s, non-finite). This CSV holds the accepted rows only, so "
    "a count taken from it is a count of the CURATED dataset and not of the detection."
)

#: The same definition, for a file that withheld NOTHING. The sentence above ends with a
#: claim about THIS file ("holds the accepted rows only") that is false for a full export
#: -- ``build_training_frame(include_all=True)`` writes one, and so will
#: ``Analysis > Export events`` once the include-rejected option lands. One note per
#: case, so the sidecar never describes a table it is not sitting beside.
_NOTHING_WITHHELD_NOTE = (
    "A withheld row is a row of the source table with accepted=False -- rejected by "
    "hand during curation, or de-accepted by the detector itself (clipped by the "
    "boundary, over max_dwell_s, non-finite). Nothing was withheld here: this file holds "
    "every row of the source table, curated and rejected alike."
)

_REVIEWED_WITHHELD_NOTE = (
    "This training table holds only resolved human-reviewed decisions, including both "
    "accepted positives and rejected negatives. Withheld rows can be untouched or "
    "legacy-unknown candidates, automatic detector rejections, accepted rows nobody "
    "reviewed yet, or unresolved reconciliation accept/reject conflicts. Consult the "
    "withheld_by_status counts; accepted=False does not by itself mean a row was written."
)

_REVIEWED_NOTHING_WITHHELD_NOTE = (
    "This training table holds resolved human-reviewed decisions of both signs: accepted "
    "positives and rejected negatives. Nothing was withheld because every source row met "
    "that reviewed-selection contract."
)

#: How a withheld row is matched back to the table it was withheld from: EVERY one of
#: these that both frames carry, as one composite key. A POSITIONAL guess is deliberately
#: not an option -- ``events.accepted.df`` is a boolean-mask slice today, but a caller
#: that reset the index would make "rows 0..12 of 0..15" name the WRONG THREE ROWS.
#:
#: And ALL of them together rather than the best one available, because ``event_id`` is a
#: small integer that a re-numbered frame reproduces by accident: ids 0..12 of a renumbered
#: 13-row slice are all present in a 16-row table's 0..15, so an id-only match would report
#: the last three rows as withheld with total confidence. Requiring the bounds to agree too
#: makes the mismatch visible, and a wrong breakdown here is worse than none: this block is
#: the only thing on disk a reader can check the CSV against.
_WITHHELD_KEY_COLUMNS: tuple[str, ...] = ("event_id", "start_idx", "end_idx")


def _withheld_rows(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame | None:
    """The rows of ``source_df`` that did NOT reach ``df``, or ``None`` if unmatchable.

    ``None`` (rather than an empty frame) whenever the two frames cannot be shown to be
    the same table: no shared key column, a key that repeats within either frame, or a row
    in ``df`` that ``source_df`` does not hold. Each of those means a difference taken
    across them would be fiction, and this block exists to stop fiction.
    """
    if source_df is None or len(source_df) < len(df):
        return None
    key = [c for c in _WITHHELD_KEY_COLUMNS
           if c in df.columns and c in source_df.columns]
    if not key:
        return None
    src = pd.MultiIndex.from_frame(source_df[key].astype(object))
    kept = pd.MultiIndex.from_frame(df[key].astype(object))
    if src.has_duplicates or kept.has_duplicates:
        return None
    if len(kept.difference(src)):          # a row here that is not there: not our table
        return None
    return source_df.loc[~src.isin(kept)]


def _breakdown(values: "pd.Series") -> dict[str, int]:
    """``value_counts`` as a JSON-safe ``{label: n}``, with ``pd.NA`` named as such."""
    counts = values.astype("string").value_counts(dropna=False)
    return {("unrecorded" if pd.isna(k) else str(k)): int(v) for k, v in counts.items()}


def curation_stamp(
    df: pd.DataFrame,
    *,
    n_total: Any = None,
    source_df: pd.DataFrame | None = None,
    provenance: Mapping[str, Any] | None = None,
    selection: str | None = None,
) -> dict[str, Any]:
    """The curation block every events sidecar carries. THE one writer of it.

    TWO THINGS THE EXPORT USED TO LEAVE UNSAID, both measured on this tree.

    1. ``Analysis > Export events`` writes ``events.accepted.df`` -- the accepted rows
       only -- and the sidecar said ``n_rows: 113`` over a detection of 115 with nothing
       anywhere naming the two that did not come. A reader outside this app cannot tell a
       115-event recording exported after curation from a 113-event recording, and the
       difference is the whole of the owner's "user curation should determine the final
       dataset": the file is the CURATED dataset and must say so. ``n_withheld`` is that
       sentence as a number, and ``counted_from`` says where the total came from so a
       reader can weigh it.
    2. Which detector found these rows, and whether anyone has ruled on them, are now
       PER-ROW facts (`events.SOURCE_COL` / `events.REVIEW_COL`) and they travel in the
       CSV itself. The block summarises them so the sidecar describes the table beside
       it without a reader having to aggregate 115 rows by hand -- and, when the columns
       are ABSENT, says so explicitly rather than letting silence read as "one detector,
       all reviewed".

    THREE WAYS TO KNOW THE TOTAL, tried in this order, and the order is the design:

    * ``source_df`` -- the TABLE the export was taken from. Best, because it is not a
      claim about the table, it IS the table: the count cannot be stale, and the rows
      that did not come can be DESCRIBED rather than merely counted (see below).
    * ``n_total`` -- what the caller says the table held.
    * ``provenance['n_events']`` -- the detection's own count, correct for a normal
      export because every door that changes the frame's shape runs it back through
      `events.reconcile_row_counts`.

    A candidate SMALLER than ``len(df)`` is skipped and the next one is tried, rather
    than ending the search: a provenance smaller than the frame it describes is stale
    (`events.reconcile_row_counts` exists because that happened) and a negative
    "withheld" would be a fabricated number in the one key that exists to stop one -- but
    a caller that guesses badly must not thereby DELETE the disclosure that the fallback
    would have given, which is what a single-candidate rule did. None of the three means
    ``None``, i.e. "this file does not know" -- never 0, which would assert that nothing
    was withheld.

    WHAT WAS WITHHELD, not just how many. "3 rows did not come" does not distinguish a
    curator throwing away three artifacts from the detector clipping three at the
    recording boundary, and those are opposite facts about the dataset -- the first is
    curation, the second is the detector de-accepting its own output
    (`events.is_automatic_rejection`). With ``source_df`` in hand the block adds
    ``withheld_by_status`` and ``withheld_by_source``, the same two breakdowns it already
    gives for the rows that DID come, over the rows that did not. Without it both read
    ``None``: "this file does not know", never ``{}``.

    Returns ``{"curation": {...}}``, ready to merge into the sidecar top level.
    """
    n_rows = int(len(df))
    withheld = _withheld_rows(df, source_df) if source_df is not None else None

    def _as_total(value: Any) -> int | None:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate >= n_rows else None

    candidates: list[tuple[str, Any]] = []
    if source_df is not None:
        candidates.append(("source_table", len(source_df)))
    if n_total is not None:
        candidates.append(("caller", n_total))
    if provenance is not None:
        candidates.append(("provenance.n_events", provenance.get("n_events")))
    total: int | None = None
    counted_from: str | None = None
    for name, raw in candidates:
        total = _as_total(raw)
        if total is not None:
            counted_from = name
            break

    if selection == TRAINING_SELECTION_REVIEWED:
        note = (_REVIEWED_NOTHING_WITHHELD_NOTE
                if total == n_rows else _REVIEWED_WITHHELD_NOTE)
    else:
        note = _NOTHING_WITHHELD_NOTE if total == n_rows else _WITHHELD_NOTE

    block: dict[str, Any] = {
        "n_in_table": total,
        "n_withheld": None if total is None else int(total - n_rows),
        "note": note,
        "counted_from": counted_from,
        # WHICH description columns the CSV carries, so an absent breakdown below reads
        # as "the app that wrote this did not record it" and not as "there were none".
        "columns": [c for c in (SOURCE_COL, REVIEW_COL)
                    if c in df.columns],
        "by_source": None,
        "by_kind": None,
        "by_status": None,
        # The same two, over the rows that did NOT come. Only ever filled from a
        # ``source_df`` that matched row-for-row -- see :func:`_withheld_rows`.
        "withheld_by_status": None,
        "withheld_by_source": None,
    }
    if has_source_status(df) and n_rows:
        by_source = _breakdown(df[SOURCE_COL])
        by_kind: dict[str, int] = {}
        for value, n in df[SOURCE_COL].value_counts(dropna=False).items():
            kind = source_kind(value)
            by_kind[kind] = by_kind.get(kind, 0) + int(n)
        block["by_source"], block["by_kind"] = by_source, by_kind
    if withheld is not None:
        if has_source_status(withheld):
            block["withheld_by_source"] = _breakdown(withheld[SOURCE_COL])
        if "accepted" in withheld.columns:
            block["withheld_by_status"] = _breakdown(event_status(withheld))
    if "accepted" in df.columns and n_rows:
        # Computed even without the pair: `event_status` reads an unstamped frame as
        # all-"unknown" (plus the detector's own auto-rejections, which are visible in
        # ``flag``), and "unknown" is the honest description of a legacy export.
        status = event_status(df).astype("string").value_counts(dropna=False)
        block["by_status"] = {("unrecorded" if pd.isna(k) else str(k)): int(v)
                              for k, v in status.items()}
    return {CURATION_BLOCK_KEY: block}


def write_events_export(
    df: pd.DataFrame,
    path: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
    source_file: str = "",
    detector: str = "",
    meta: Any = None,
    t0_s: float | None = None,
    n_total: int | None = None,
    source_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write an events table as CSV, plus the provenance sidecar that describes it.

    ``Analysis > Export events...`` used to write a bare ``to_csv`` with nothing
    beside it. That is the one place the estimator's identity reaches a consumer
    outside this app, and without it the migration story has no anchor: the 20
    frozen columns carry ``baseline_pA`` and ``depth_pA`` but say nothing about
    WHICH open-pore estimator produced them, and on a recording with the owner's
    event statistics that choice moves ``depth_pA`` by 4.7x. A CSV that cannot
    name its baseline cannot be compared with one exported next month.

    The schema is untouched -- no new column, no reordering. The estimator travels
    beside the table, in ``<stem>.meta.json``, under
    :data:`BASELINE_CONTRACT_KEYS` including ``baseline_fingerprint``.

    ...and so does the CLOCK. ``t_start_s`` is on the RECORDING clock (0 = the first
    LOADED sample) while the trace the user read these events off -- and the
    ``figs_trace.csv`` this app writes into the same folder -- is on the instrument
    clock. Pass ``meta`` (the document's ``RecordingMeta``) and the sidecar carries the
    origin in the same spelling AND with the same meaning as
    ``measure.MEASUREMENT_SCHEMA``'s ``t0_s`` column, with the sentence the on-screen
    column header shows for that same document. ``t0_s=`` takes the number directly for
    a caller with no meta. Omit both and the sidecar says the origin is UNKNOWN rather
    than implying 0 -- see :func:`clock_stamp`. No new CSV column either way.

    ...and so does WHAT IS NOT IN THE TABLE. The caller writes the ACCEPTED rows (that
    is what ``Analysis > Export events`` hands over), so the CSV is the curated dataset
    and not the detection -- and the sidecar used to give a reader no way to tell those
    apart. :func:`curation_stamp` adds the ``curation`` block: how many rows the source
    table held, how many were withheld, and how the rows that DID come break down by
    ``source`` and by derived status. Hand the SOURCE TABLE as ``source_df`` (the frame
    the accepted rows were sliced out of) and the block also says WHAT the withheld rows
    were -- a curator's rejections and the detector's own auto-rejections broken apart,
    which "3 rows did not come" cannot do. ``n_total`` states the count alone for a
    caller that has only the number; without either, the block falls back to
    ``provenance['n_events']`` and names which of the three it used. Still no new CSV
    column on any of those paths.

    Returns ``{"data": <csv path>, "meta": <sidecar path>}``.

    THE TWO FILES LAND TOGETHER OR NOT AT ALL. The sidecar does not merely accompany the
    CSV, it *describes* it — which estimator, which fingerprint, how many rows — so a table
    from today beside a sidecar from last month is not a partial export, it is a export with
    a confidently wrong provenance record, and nothing on disk says so. See
    :class:`nanoscope.core.atomic_write.StagedSet`.
    """
    data_path, meta_path = events_export_paths(path)

    prov = dict(provenance or {})
    meta: dict[str, Any] = {
        "format": "nanoscope-events",
        "version": 1,
        "source_file": str(source_file),
        "detector": str(detector),
        "fs": float(prov["fs"]) if prov.get("fs") is not None else None,
        "polarity": prov.get("polarity"),
        "units": prov.get("units"),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        **clock_stamp(df, meta=meta, t0_s=t0_s, provenance=prov),
        **curation_stamp(df, n_total=n_total, source_df=source_df, provenance=prov),
        "provenance": prov,
    }
    with staged_write_set() as group:
        df.to_csv(group.stage(data_path), index=False)
        group.stage(meta_path).write_text(json.dumps(_json_safe(meta), indent=2))
    return {"data": data_path, "meta": meta_path}


def levels_export_paths(path: str | Path) -> tuple[Path, ...]:
    """Every file :func:`write_levels_export` will write. Same two-file contract as events."""
    p = Path(path)
    data = p.with_suffix(".csv") if p.suffix == "" else p
    return (data, data.with_suffix(".meta.json"))


def pivot_levels_wide(levels: pd.DataFrame) -> pd.DataFrame:
    """Long level table -> one row per event, ``(depth, dwell)`` pair per sub-state.

    The shape the user's own ``*Saved Data.txt`` files are already in, and the one
    ``core.document``'s EventTableDocument seam describes: "7 base columns plus one
    (amplitude, dwell) pair per additional substate". Produced HERE, at the export
    boundary, and never internally -- the column COUNT depends on the deepest event in
    the table, which is exactly the property that makes it unusable as a working schema
    and harmless as a file format.
    """
    if levels.empty:
        return pd.DataFrame(columns=["event_id", "n_levels"])
    wide = levels.pivot(index="event_id", columns="level_idx",
                        values=["depth_pA", "dwell_s"])
    wide.columns = [f"L{int(i)}_{name}" for name, i in wide.columns]
    order = sorted(wide.columns, key=lambda c: (int(c.split("_")[0][1:]), c.split("_", 1)[1]))
    wide = wide[order]
    n_lv = levels.groupby("event_id")["n_levels"].first()
    return pd.concat([n_lv.rename("n_levels"), wide], axis=1).reset_index()


def write_levels_export(
    levels: pd.DataFrame,
    path: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
    cusum_params: Mapping[str, Any] | None = None,
    source_file: str = "",
    detector: str = "",
    wide: bool = False,
    meta: Any = None,
    t0_s: float | None = None,
) -> dict[str, Path]:
    """Write a CUSUM level table as CSV, plus the sidecar that describes it.

    Same two-files-or-neither contract as :func:`write_events_export`, and the sidecar
    carries one thing that table does not need: ``cusum_params``. A level table without
    the (step_sigma, threshold_sigma, warmup, min_level) that produced it cannot be
    compared with one exported next month, for exactly the reason an event table needs
    its baseline estimator -- those four numbers decide how many levels an event HAS.

    ``wide=True`` writes the pivoted per-substate form instead; the sidecar records
    which layout was written, because the two are not interchangeable on read.

    The clock rides here too, for the same reason and through the same seam
    (:func:`clock_stamp`): a level's ``t_start_s`` is on the event table's clock by
    construction (``analysis.cusum``: "Levels are on the event table's clock, with no
    way to opt out"), which is the RECORDING clock, not the trace axis's. The wide
    layout has no time column at all, and the block then names none -- it still states
    the origin, which is what makes a wide table joinable back to the events CSV.
    """
    data_path, meta_path = levels_export_paths(path)
    out = pivot_levels_wide(levels) if wide else levels

    prov = dict(provenance or {})
    meta: dict[str, Any] = {
        "format": "nanoscope-levels",
        "version": 1,
        "layout": "wide" if wide else "long",
        "source_file": str(source_file),
        "detector": str(detector),
        "fs": float(prov["fs"]) if prov.get("fs") is not None else None,
        "polarity": prov.get("polarity"),
        "units": prov.get("units"),
        "n_rows": int(len(out)),
        "n_events": int(levels["event_id"].nunique()) if not levels.empty else 0,
        "columns": list(out.columns),
        "cusum_params": dict(cusum_params or {}),
        **clock_stamp(out, meta=meta, t0_s=t0_s, provenance=prov),
        "provenance": prov,
    }
    with staged_write_set() as group:
        out.to_csv(group.stage(data_path), index=False)
        group.stage(meta_path).write_text(json.dumps(_json_safe(meta), indent=2))
    return {"data": data_path, "meta": meta_path}


def write_training_export(
    df: pd.DataFrame,
    path: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
    source_file: str = "",
    detector: str = "",
    include_all: bool = False,
    selection: str | None = None,
    meta: Any = None,
    t0_s: float | None = None,
) -> dict[str, Path]:
    """Write the curated events of ``df`` as a training table, plus a provenance sidecar.

    ``path`` chooses the format by suffix: ``.parquet`` writes parquet, anything else
    (``.csv`` or bare) writes CSV. A ``<stem>.meta.json`` sidecar carrying
    :func:`training_provenance` is written next to it so the export is reproducible.
    ``selection`` accepts the stable values in :data:`TRAINING_SELECTIONS`; leaving it
    ``None`` retains the legacy ``include_all`` behaviour exactly.

    BOTH files are enumerated by :func:`training_export_paths`; a GUI caller must
    confirm that set rather than the single name it was handed — and both LAND together
    or not at all, for the reason given in :func:`write_events_export`.

    Returns ``{"data": <written table path>, "meta": <sidecar path>}``.
    """
    out = build_training_frame(
        df, include_all=include_all, selection=selection)
    path, meta_path = training_export_paths(path)

    # The rows the table above holds, as EVENT rows -- `build_training_frame` reduces to
    # TRAINING_COLUMNS and drops `source`/`review` on purpose, so the sidecar's curation
    # block is computed from the same selection over the original frame. One definition
    # of that selection, `_training_row_mask`, serves both.
    written = df[_training_row_mask(
        df, include_all=include_all, selection=selection)]
    meta = training_provenance(
        provenance, source_file=source_file, detector=detector,
        include_all=include_all, selection=selection,
        n_rows=len(out), columns=out.columns,
        meta=meta, t0_s=t0_s,
        df=written, source_df=df,
    )
    with staged_write_set() as group:
        staged_data = group.stage(path)
        if path.suffix.lower() == ".parquet":
            out.to_parquet(staged_data, engine="pyarrow", index=False)
        else:
            out.to_csv(staged_data, index=False)
        group.stage(meta_path).write_text(json.dumps(meta, indent=2))
    return {"data": path, "meta": meta_path}
