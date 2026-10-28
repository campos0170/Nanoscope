"""Re-measure an existing event table against a different open-pore estimator.

WHY THIS MODULE EXISTS
======================
The characterisation baseline moved. Every row the owner produced before this
change was measured by a 10 ms block median, which on his own recordings sits
*inside* the events (median local occupancy in +-5 ms is 34.5%, p90 50.8%), so
the six amplitude columns are understated by **4.7x** on the -600 mV apo file
(median depth 23.70 -> 110.17 pA over the same 1641 spans; the brief's 14.7x is a
different detection's statistic and does not reproduce -- see
``AFFECTED_COLUMNS``). Correcting the estimator is therefore not a refinement: it
silently multiplies a published number by five.

`nanoscope/core/analysis.py::AnalysisResult` already carries the precedent for
exactly this situation -- ``calibration_fingerprint`` exists because a deferred
gain question means stored depths WILL be revised, and the UI's contract there is
*mark it stale and offer a re-run*, never re-compute behind the user's back. The
baseline is the same shape of problem with a bigger factor, so it gets the same
treatment through ``AnalysisResult.baseline_fingerprint``:

    record which estimator produced a row -> mark a mismatch stale ->
    OFFER a re-measure -> never silently re-measure, never refuse to open.

This module is the third of those four. It is the engine only; the marking and
the offering live in the GUI.

WHAT A RE-MEASURE IS, AND WHAT IT IS NOT
========================================
It **re-measures**. It does **not** re-detect. Detection geometry
(``start_idx`` / ``end_idx`` / ``t_start_s`` / ``dwell_s``) is asserted unchanged
row for row -- a re-measure that moved an event boundary would be a re-detection
wearing a re-measure's name, and it would destroy the curation it is supposed to
preserve. Only the amplitude family moves:

    baseline_pA  depth_pA  frac_depth  ecd_pA_s  mean_dev_pA  max_dev_pA  snr
    rise_s  fall_s      (10%/90% of each event's OWN depth -- see REMEASURED_COLUMNS)

...plus the two DERIVED judgements that are functions of them
(``flag``/``accepted``'s ``zero_baseline`` and ``non_finite`` rules), which are
re-judged by :func:`~nanoscope.analysis.characterize.characterize` exactly as a
merge re-judges them.

Curation survives, through the machinery that already exists for it:
:func:`~nanoscope.analysis.events.inherited_curation`, reached by handing each
row to :func:`~nanoscope.analysis.characterize.recharacterize_span` as its own
``prior``. A hand rejection and a curator's own flag tokens carry over
(``accepted`` combined with AND, never OR); an automatic rejection is re-judged.

ONE DEFINITION, NOT TWO
=======================
:func:`compare_estimators` is a thin wrapper over :func:`remeasure_events`. That
is deliberate and structural. The report is what the owner reads in order to
decide whether to press the button; if it were computed by a second code path it
could disagree with the button, and the disagreement would be invisible -- a
report that says 1.0x for a change that is really 4.7x is worse than no report.
So the report is a BY-PRODUCT of the very rows the button would install, and the
two cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cancel import CancelPoller, check_cancelled
from .characterize import recharacterize_span, rise_fall_over_set
from .events import (
    AUTOMATIC_REJECTION_FLAGS,
    DERIVED_FLAGS,
    FLAG_SEP,
    REVIEW_COL,
    EventSet,
    flag_tokens,
    is_automatic_rejection,
    reconcile_row_counts,
    stale_rate_hz,
)

__all__ = [
    "AFFECTED_COLUMNS",
    "GEOMETRY_COLUMNS",
    "REMEASURED_COLUMNS",
    "MOVED_BASELINE_THRESHOLD_PA",
    "MATERIAL_RATIO_TOLERANCE",
    "RemeasureResult",
    "EstimatorComparison",
    "remeasure_events",
    "compare_estimators",
    "describe_stored_baseline",
    "CURATION_COLUMNS",
    "frozen_for_dispatch",
    "curation_snapshot",
    "merge_guard_snapshot",
    "assert_answer_describes",
    "merge_late_curation",
    "release_note",
]

#: The six columns the comparison report is about, in the order §4.1 lists them.
#:
#: §4.1 states the factors as 14.7x / 38x / 12.2x / 11.5x / 1.7x / 14.7x. Those do
#: not reproduce and are not what this app reports. Measured on the owner's apo
#: recording, 1641 events detected under the old estimator and re-measured against
#: the same spans, the factors are **4.65x / 16.4x / 4.72x / 4.57x / 3.18x /
#: 4.65x** (per-event median depth ratio 4.73x, p10 2.22, p90 7.02). The report
#: computes them from the table in front of it and always did; only the prose was
#: wrong, and the release note now carries the measured table.
AFFECTED_COLUMNS: tuple[str, ...] = (
    "depth_pA", "frac_depth", "ecd_pA_s", "mean_dev_pA", "max_dev_pA", "snr",
)

#: What a re-measure must NOT touch. Asserted row for row, not documented and
#: hoped for: a re-measure that moves a boundary is a re-detection, and the whole
#: promise of the offered button is that the user's curated event set survives it.
GEOMETRY_COLUMNS: tuple[str, ...] = ("start_idx", "end_idx", "t_start_s", "dwell_s")

#: Every column a re-measure legitimately rewrites, and therefore everything the
#: results table badges as stale. ``baseline_pA`` is the estimator's own output (it
#: moves by 0.36x on the apo file) and the six above are computed from it.
#:
#: ``rise_s``/``fall_s`` are here because they are measured against 10% and 90% OF
#: THAT EVENT'S OWN DEPTH, so a depth that moves 4.8x moves the levels the crossing
#: times are read at. On the owner's 1641-event apo table the re-measure moves
#: ``rise_s`` on 1243 rows and ``fall_s`` on 1236, and 1046/1022 of those become
#: NaN. That is the honest answer, not a loss: under the old diving baseline those
#: rows were ~0.4 ms fragments of a 17 ms blockade, and the "transition" measured
#: inside a fragment was the baseline climbing out of the event. With the open pore
#: at its true level the fragment is deep from its first sample to its last, never
#: crosses 10%, and NaN is what "not measurable in this window" means.
#:
#: They were left OUT of this list once, and the sentence that put them there was in
#: the shipped release note ("Unaffected: ... rise_s, fall_s"). They moved anyway --
#: on 812 and 920 rows even when NOTHING was measured differently -- because the
#: row-by-row rebuild lost the neighbour bound. That bug is fixed
#: (`_restore_set_wide_rise_fall`); this entry is the other half, and the two must
#: not be confused: what remains is a real consequence of a real change of depth.
REMEASURED_COLUMNS: tuple[str, ...] = (
    "baseline_pA", *AFFECTED_COLUMNS, "rise_s", "fall_s",
)

#: §4.2.7's yardstick: "the fraction of events whose stamped baseline moves more
#: than 10 pA". It is an absolute pA threshold rather than a ratio because the
#: question it answers is "did this row's open-pore reference land somewhere
#: else?", and near a zero-crossing a ratio is unbounded for a move of nothing.
MOVED_BASELINE_THRESHOLD_PA: float = 10.0

#: Provenance key recording that a table was re-measured, and from what.
REMEASURED_KEY = "remeasured"

#: How far the median depth ratio may sit from 1.0 before the report calls the
#: change MATERIAL. Its exact value is not load-bearing and deliberately so: measured
#: on synthetics, a recording where the block median genuinely lands on the open pore
#: comes out at 1.013 and one where it does not comes out at 1.73, and the owner's own
#: apo file at 4.73. The two regimes are two orders of magnitude apart in this
#: statistic, so anything between "a few percent" and "tens of percent" separates
#: them; 5% is chosen as comfortably above estimator jitter and far below any factor
#: that could change a conclusion.
MATERIAL_RATIO_TOLERANCE: float = 0.05


# ===========================================================================
# results
# ===========================================================================
@dataclass(frozen=True)
class EstimatorComparison:
    """Old vs new, per event and in aggregate. §4.2.7's report, as plain data.

    Every array is per-event and index-aligned with ``event_id``; the GUI renders
    them, the tests assert on them, and nothing here knows about Qt.

    The aggregate statistics are computed over the rows where BOTH depths are
    finite and the old depth is strictly positive -- a ratio against a zero or
    non-finite old depth is not a number the owner can act on, and silently
    folding those into a median would be the report lying in the safe-looking
    direction. ``n_ratio`` says how many rows the ratios actually describe.
    """

    n_events: int
    event_id: np.ndarray
    old_depth_pA: np.ndarray
    new_depth_pA: np.ndarray
    old_baseline_pA: np.ndarray
    new_baseline_pA: np.ndarray
    depth_ratio: np.ndarray            # new/old, NaN where old <= 0 or non-finite
    baseline_shift_pA: np.ndarray      # |new - old|
    new_frac_depth: np.ndarray = field(  # noqa: RUF009 - a per-event column, not a default
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False)
    #: How the table was measured before, and how it would be measured after.
    old_fingerprint: str | None = None
    new_fingerprint: str | None = None
    old_description: str = "unknown (this table predates the baseline stamp)"
    new_description: str = ""
    #: Soft advisories from the NEW baseline's health block, verbatim.
    warnings: tuple[str, ...] = ()

    # -- aggregates ---------------------------------------------------------
    @property
    def _ok(self) -> np.ndarray:
        return np.isfinite(self.depth_ratio)

    @property
    def n_ratio(self) -> int:
        """Rows the ratio statistics describe (old depth finite and > 0)."""
        return int(self._ok.sum())

    @property
    def median_ratio(self) -> float:
        """The headline number. ~1.0 on a clean file; 4.65x on the apo file."""
        return self._q(0.5)

    @property
    def ratio_p10(self) -> float:
        return self._q(0.10)

    @property
    def ratio_p90(self) -> float:
        return self._q(0.90)

    @property
    def median_old_depth_pA(self) -> float:
        return self._med(self.old_depth_pA)

    @property
    def median_new_depth_pA(self) -> float:
        return self._med(self.new_depth_pA)

    @property
    def n_baseline_moved(self) -> int:
        """Events whose stamped baseline moved more than
        `MOVED_BASELINE_THRESHOLD_PA`. A non-finite shift counts as moved: a row
        whose reference became unmeasurable did not stay put."""
        shift = self.baseline_shift_pA
        if shift.size == 0:
            return 0
        return int((~(shift <= MOVED_BASELINE_THRESHOLD_PA)).sum())

    @property
    def frac_baseline_moved(self) -> float:
        return (self.n_baseline_moved / self.n_events) if self.n_events else 0.0

    @property
    def n_frac_depth_gt1(self) -> int:
        """Rows whose corrected ``frac_depth`` exceeds 1 -- i.e. dI/I0 > 100%.

        Deliberately out of scope (§4.1), but the corrected baseline is what makes
        it visible, and this report is where the owner is looking when it becomes
        visible. Counting it here is the cheapest honest way to put it in front of
        him.

        WHAT IT IS NOT: a current-gain error. ``frac_depth`` is
        ``depth_pA / |baseline_pA|`` -- a RATIO -- so multiplying the whole channel
        by any gain factor leaves it exactly unchanged. An earlier version of the
        sentence in :meth:`summary` told the owner to look at his gain; no setting
        of the gain can move this number, and the docstring hedged where the
        sentence did not.

        WHAT IT IS: the current magnitude GROWS during the event. On his -600 mV
        apo file the open pore is -34.8 pA (89.0% of all samples lie in
        [-45, -25] pA and the 1 s block medians are flat at -33.6...-36.0 across
        all 59.9 s) and the event level is -145 pA. Those are CONDUCTIVE
        (current-enhancing) events, a category his own corpus names in its
        filenames, and dI/I0 > 1 is then a real observation rather than an
        artefact. The only instrument fault that could also produce it is an
        amplifier OFFSET, which shifts the open-pore level without touching the
        excursion; a gain cannot.
        """
        v = self.new_frac_depth
        if v.size == 0:
            return 0
        return int((np.isfinite(v) & (v > 1.0)).sum())

    @property
    def changed(self) -> bool:
        """Is this a change the owner needs to look at?

        True when any baseline moved past the threshold or the median depth ratio
        is more than 1% off unity. A comparison that is False here is the "clean
        file, re-run without anxiety" case §4.2.7 exists to identify.
        """
        if self.n_baseline_moved:
            return True
        r = self.median_ratio
        return bool(np.isfinite(r) and abs(r - 1.0) > MATERIAL_RATIO_TOLERANCE)

    def summary(self) -> str:
        """One paragraph, for a readout or a log. States the factor and the count."""
        if not self.n_events:
            return "No events to compare."
        pct = 100.0 * self.frac_baseline_moved
        head = (
            f"{self.n_events} events. Median depth {self.median_old_depth_pA:.3g} -> "
            f"{self.median_new_depth_pA:.3g} pA "
            f"({self.median_ratio:.3g}x; p10 {self.ratio_p10:.3g}x, "
            f"p90 {self.ratio_p90:.3g}x over {self.n_ratio} comparable rows). "
            f"{self.n_baseline_moved} of {self.n_events} events ({pct:.1f}%) have a "
            f"stamped baseline that moves more than "
            f"{MOVED_BASELINE_THRESHOLD_PA:g} pA."
        )
        if not self.changed:
            head += (
                " Nothing material changes on this recording -- re-measuring is safe."
            )
        if self.n_frac_depth_gt1:
            head += (
                f" {self.n_frac_depth_gt1} events come out with frac_depth > 1 "
                "(dI/I0 above 100%): the corrected baseline says the current "
                "magnitude GROWS during those events, i.e. they are conductive "
                "rather than blocking. frac_depth is a ratio, so this is not a "
                "current-gain question -- a gain cannot move it. Check the sign "
                "of the excursion against the open pore, and the amplifier "
                "offset, before reading these as blockades."
            )
        return head

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe aggregates (no per-event arrays), for provenance and logs."""
        return {
            "n_events": self.n_events,
            "n_ratio": self.n_ratio,
            "median_depth_ratio": self.median_ratio,
            "depth_ratio_p10": self.ratio_p10,
            "depth_ratio_p90": self.ratio_p90,
            "median_old_depth_pA": self.median_old_depth_pA,
            "median_new_depth_pA": self.median_new_depth_pA,
            "n_baseline_moved": self.n_baseline_moved,
            "frac_baseline_moved": self.frac_baseline_moved,
            "n_frac_depth_gt1": self.n_frac_depth_gt1,
            "moved_threshold_pA": MOVED_BASELINE_THRESHOLD_PA,
            "old_fingerprint": self.old_fingerprint,
            "new_fingerprint": self.new_fingerprint,
            "old_description": self.old_description,
            "new_description": self.new_description,
        }

    # -- internals ----------------------------------------------------------
    def _q(self, q: float) -> float:
        ok = self._ok
        if not ok.any():
            return float("nan")
        return float(np.quantile(self.depth_ratio[ok], q))

    @staticmethod
    def _med(vals: np.ndarray) -> float:
        finite = vals[np.isfinite(vals)]
        return float(np.median(finite)) if finite.size else float("nan")


@dataclass(frozen=True)
class RemeasureResult:
    """The re-measured table, and the report that describes what moved.

    ``events`` is what the button installs; ``comparison`` is what the report
    shows. They come out of the same pass, which is what makes the report a
    truthful preview of the button rather than a second opinion about it.
    """

    events: EventSet
    comparison: EstimatorComparison
    #: The full-length baseline the re-measure ran on, so a caller that wants to
    #: draw it (or hand it to a further rebuild) need not estimate it again.
    baseline: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    #: The flat ``baseline_*`` provenance block now stamped on ``events``.
    baseline_provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str | None:
        return self.baseline_provenance.get("baseline_fingerprint")


# ===========================================================================
# curation made WHILE the job runs
# ===========================================================================
#: The only two columns a curator writes THAT EVERY TABLE HAS. Everything else in the
#: row is measured. ``review`` is a third and is handled beside these rather than added
#: to the tuple: it is declared outside `events.SCHEMA` precisely so a project saved
#: before it existed still loads, so every read of it must be guarded on presence, and a
#: tuple that callers index blind (`curation_snapshot` does) cannot carry an optional
#: column. :data:`LATE_CURATION_COLUMNS` is the guarded set actually diffed.
CURATION_COLUMNS: tuple[str, ...] = ("accepted", "flag")


def _late_curation_columns(*frames: pd.DataFrame) -> tuple[str, ...]:
    """:data:`CURATION_COLUMNS` plus ``review`` when EVERY frame carries it.

    A curator can press "mark reviewed" during the 3.3 s a re-measure takes, exactly as
    he can press Accept -- and the answer being installed was measured from a frozen
    copy taken before the click, so without this the review is discarded by the install
    for the same reason a late accept used to be. All-or-nothing on presence because a
    diff needs the column on both sides: comparing a frame that has it against one that
    does not would read every row as changed.
    """
    if all(REVIEW_COL in f.columns for f in frames):
        return (*CURATION_COLUMNS, REVIEW_COL)
    return CURATION_COLUMNS


def _curated_differently(x: Any, y: Any) -> bool:
    """Did the curator change this cell? Answered for EVERY value the column's own
    declared dtype admits, missing values included.

    ``flag`` is declared ``"string"`` in :data:`~nanoscope.analysis.events.SCHEMA`, and
    the missing value of a pandas ``string`` column is ``pd.NA`` -- whose ``__bool__``
    raises. A bare ``x != y`` therefore does not merely give the wrong answer on a
    missing flag, it raises ``TypeError: boolean value of NA is ambiguous`` out of a Qt
    slot, so nothing is installed and the user is told the detection went away.

    Every other reader of the column already tolerates it (``flag_tokens(pd.NA) -> ()``,
    ``is_automatic_rejection(pd.NA) -> False``, ``inherited_curation`` returns ``''``),
    which is exactly why this one had to as well rather than the callers being asked to
    clean the frame first: a missing flag reaches this function from a CSV re-import with
    an empty cell, from a ``concat`` of a partial row, and through a project round trip.

    Two missing values are the SAME absence, so they are not a curation. A missing value
    against a real one IS a change -- someone wrote a flag, or cleared one.

    ``_isna`` collapses to a scalar deliberately: ``pd.isna`` of a list or an array
    returns an ARRAY, and testing that would replace one ambiguous-truth-value crash
    with another. A non-scalar in either curation column is not a missing value, so it
    falls through to the plain comparison, which is where such a cell belongs.
    """
    def _isna(v: Any) -> bool:
        if np.ndim(v) != 0:
            return False
        return bool(pd.isna(v))

    x_na, y_na = _isna(x), _isna(y)
    if x_na or y_na:
        return not (x_na and y_na)
    return bool(np.any(x != y)) if np.ndim(x) or np.ndim(y) else bool(x != y)


def frozen_for_dispatch(eventset: EventSet) -> EventSet:
    """``eventset``, detached from the frame the user is still editing.

    THE CONCURRENCY GUARANTEE, written once and applied at BOTH boundaries it has to
    hold at, because they are different boundaries rather than two call sites of one:

    * **The GUI's dispatch.** ``_launch_remeasure`` used to hand the worker
      ``res.payload`` itself -- the very ``DataFrame`` the results table, the Event
      Inspector and Undo/Redo curate IN PLACE -- and then keep it editable for the
      3-4 s the job takes. Freezing here is also what makes the outcome DEFINED
      rather than raced: after this call, every edit the user makes is unambiguously
      an edit made *during* the job, and the install has one rule for all of them.
    * **The engine.** :func:`remeasure_events` reads its argument row by row over
      those same seconds, and cannot know whether its caller froze anything. A
      function that reads a mutable argument for that long owns the copy. Without it
      ``_assert_geometry_held`` -- an assertion about the ENGINE -- was comparing its
      output against an input that had moved, and fired at the USER: measured, a mask
      drag 1.5 s into an 800-event job produced a modal reading "re-measure moved
      'start_idx'", discarded the whole job, and left the status bar reading
      "Re-measuring 800 events…" permanently.

    The second copy is a few hundred kB on the owner's largest table and is paid once
    per job, against a measurement that takes seconds. Making the engine trust its
    caller instead is how the first version of this was written.
    """
    return EventSet(eventset.df.copy(deep=True), dict(eventset.provenance or {}))


def curation_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """The curator-visible state of ``df``, keyed by ``event_id``.

    Taken before a re-measure is dispatched off-thread, so the install can tell an
    edit the user made WHILE it ran from one he made before it started.

    ``review`` joins the snapshot when the frame has it -- see
    :func:`_late_curation_columns`. Marking an event reviewed is a curation gesture and
    is lost by the install unless the "before" picture contains it.

    Optional amplitude fields also join when present. Unlike a review, a late
    amplitude measurement cannot be carried onto a changed baseline: the install
    detects that edit and refuses its stale answer.
    """
    cols = [c for c in _late_curation_columns(df)]
    from .event_amplitude import AMPLITUDE_COLUMNS

    cols.extend(c for c in AMPLITUDE_COLUMNS if c in df.columns)
    return df[["event_id", *cols]].copy(deep=True)


def merge_guard_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """The curation a MERGE cannot carry, keyed by ``event_id``: ``accepted``,
    ``flag`` and optional amplitude choices, never ``review``.

    :func:`curation_snapshot` answers "what will the re-measure install have to carry
    forward", and it widened to include ``review`` for exactly that reason. A second
    caller reads it for a DIFFERENT question -- "Merge all"'s install guard, which
    drops a finished merge when the user curated while it ran, because a merge
    DELETES rows and there is then no honest row to carry a decision onto. Those two
    questions have different answers and must not share one snapshot:

    * a widened snapshot differs whenever ``review`` changed, so a single "mark
      reviewed" click during a merge made the guard read "the user curated" and threw
      a completed merge away, reporting rows the user never edited as edited;
    * the thing the guard actually protects is the VERDICT -- an accept, a reject, a
      hand flag. That is what a merge would silently undo, and it is what this
      snapshot holds.

    ``review`` is deliberately outside it. A merge re-derives review from its
    fragments (:func:`~nanoscope.analysis.events.combine_reviews`: reviewed only when
    every fragment was), so a late review click is re-answered by the merge rather
    than overwritten by a stale copy of itself, and losing a click is a smaller,
    louder failure than losing the merge.

    Amplitude choices cannot describe the merged span automatically, so a new or
    revised choice during a pending merge must also make the install refuse.
    """
    cols = [c for c in CURATION_COLUMNS if c in df.columns]
    # A late amplitude edit is a curator's measurement, and cannot be mapped
    # honestly onto rows a pending merge is about to collapse.
    from .event_amplitude import AMPLITUDE_COLUMNS

    cols.extend(c for c in AMPLITUDE_COLUMNS if c in df.columns)
    return df[["event_id", *cols]].copy(deep=True)


def assert_answer_describes(out: pd.DataFrame, live: pd.DataFrame) -> None:
    """THE INSTALL'S PRECONDITION: ``out`` measured the rows ``live`` holds now.

    A re-measure is dispatched against one frame and installed against another,
    seconds later, and in between the world moves: an Undo, a Redo, an Event
    Inspector mask drag, a merge. Only the ROW COUNT and the ids used to be checked,
    so a boundary the user had just moved -- or just UNDONE -- was silently
    overwritten by the answer, under a status line reading "Detection geometry and
    curation unchanged". Measured on the owner's own file: an event dragged from
    (144689, 145934) to (144709, 145914), Re-measure pressed, Ctrl+Z during the
    4.15 s job; the table correctly returned to 24.900 ms and the install put
    24.100 ms back. On a synthetic the same sequence was a 100% dwell error and a
    2.0x depth error.

    So the check is what the sentence claims: the geometry the job ACTUALLY
    MEASURED, compared against the store's CURRENT geometry, column for column.
    ``out``'s geometry is the input's by construction --
    :func:`_assert_geometry_held` refuses to return a table where it is not -- so
    this is a comparison of dispatch-time against install-time, which is exactly the
    window the user's edit lands in.

    It is a precondition rather than a merge rule, and it lives here rather than in
    the GUI, because both things that can be wrong (the rows are different rows, the
    rows are the same rows in different places) have the same answer: **drop the
    answer and say so, never guess.** A caller that skipped it would install a
    measurement of spans the table no longer holds.

    Raises:
        ValueError: naming what moved, in the sentence the status bar shows.
    """
    out_ids = out["event_id"].to_numpy(dtype=np.int64)
    live_ids = live["event_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(out_ids, live_ids):
        raise ValueError(
            "the event table changed while the re-measure was running (rows were "
            "merged, added or re-detected), so this answer describes rows that no "
            "longer exist."
        )
    for col in GEOMETRY_COLUMNS:
        if col not in out.columns or col not in live.columns:
            continue
        a = out[col].to_numpy()
        b = live[col].to_numpy()
        if a.dtype.kind in "iu" and b.dtype.kind in "iu":
            same = np.array_equal(a, b)
        else:
            same = np.array_equal(
                a.astype(np.float64), b.astype(np.float64), equal_nan=True)
        if same:
            continue
        bad = int(np.flatnonzero(a != b)[0])
        raise ValueError(
            f"the detection geometry changed while the re-measure was running "
            f"({col} moved at row {bad}: {a[bad]!r} → {b[bad]!r}), so this answer "
            "measures spans the table no longer holds."
        )


def _hand_verdict(flag: Any, accepted: Any) -> bool:
    """The CURATOR's own acceptance of one row, with the app's rules factored out.

    ``True`` unless a PERSON rejected this row. A row the app rejected on its own
    re-derivable grounds reads ``True`` here, because the curator said nothing about
    it -- that is
    :func:`~nanoscope.analysis.events.is_automatic_rejection`'s question, asked once
    for all three readers of it.
    """
    if bool(accepted):
        return True
    return is_automatic_rejection(flag)


def merge_late_curation(
    out: pd.DataFrame, live: pd.DataFrame, before: pd.DataFrame
) -> pd.DataFrame:
    """``out`` with the curation DIFF between ``before`` and ``live`` applied to it.

    A re-measure of the owner's -600 mV file takes about 3.3 s off the GUI thread,
    and the event table stays live for every one of them. Without this, an
    accept/reject or a flag typed during those seconds is silently discarded by the
    install -- the exact failure §4.2.3 exists to prevent, on a three-second fuse
    instead of a project-reload one.

    A DIFF CARRY, NOT AN INHERITANCE
    ================================
    This used to go through :func:`~nanoscope.analysis.events.inherited_curation`,
    and that is the wrong tool here rather than the right tool used slightly wrong.
    ``inherited_curation`` is a REBUILD's rule -- union the flags, AND the
    acceptance -- and it is therefore MONOTONE in both columns: it can only add flag
    tokens and only remove acceptance. Every LOOSENING edit made during the job was
    silently discarded. Measured, on event 3 of a real detection:

        pre-job              live edit during the job     installed
        reject + 'noise'     accept, clear the flag       still rejected, still 'noise'
        bare hand reject     accept                       still rejected
        accepted + 'noise'   clear the flag               flag still there

    One realistic click: 40 rows hand-rejected earlier, "Accept all" pressed during
    a 3.3 s re-measure. Screen showed 361 accepted; the installed table held 321,
    and every accepted-gated number -- the summary, the exports, the fits -- reported
    321. The tightening half worked, and was the only half under test.

    So the carry is a per-column DIFF of ``live`` against ``before``, applied to
    ``out``. The invariant it implements: **installing must be indistinguishable
    from the user's click having landed one second after the job finished.**

    * ``flag`` -- the tokens the curator ADDED are added, the tokens he REMOVED are
      removed, and nothing else is touched. Derived tokens
      (:data:`~nanoscope.analysis.events.DERIVED_FLAGS`) are excluded from the diff
      IN BOTH DIRECTIONS: they are statements about the measurement, ``out`` has
      just re-derived them against the corrected baseline, and carrying either half
      of a stale one would resurrect a ``zero_baseline`` the correction just fixed.
    * ``accepted`` -- carried only when the CURATOR's verdict changed
      (:func:`_hand_verdict`, so an automatic rejection appearing in ``live`` is not
      mistaken for a person's), and combined with ``out``'s own automatic judgement:
      a late accept cannot un-reject a row the re-measure has just found clipped or
      non-finite, exactly as ``characterize`` would not. Contract 5 is untouched:
      ``inherited_curation`` is the rebuild path and still ANDs.

    * ``review`` -- carried as a plain diff, in both directions, WHEN THE TABLE HAS
      THE COLUMN (see :func:`_late_curation_columns`). "I have looked at this row" is
      a curation gesture like any other and is made with the same clicks during the
      same seconds; it needs none of ``accepted``'s automatic-verdict machinery
      because the app never reviews anything -- only a person does. Its diff carries
      ``review`` AND NOTHING ELSE: a review-only change writes the ``review`` cell
      and never enters the flag/accepted rewrite, so a description column cannot
      move a measured one. See the two masks below for what that cost when they
      were one.

    A row no carried column changed on is not written at all, and an untouched table
    returns ``out`` itself.

    Raises:
        ValueError: from :func:`assert_answer_describes` -- the rows moved or are
            not the same rows. The caller must drop the answer and say so.
    """
    assert_answer_describes(out, live)
    live_ids = live["event_id"].to_numpy(dtype=np.int64)
    before_ids = before["event_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(live_ids, before_ids):
        raise ValueError(
            "the event table changed while the re-measure was running (rows were "
            "merged, added or re-detected), so this answer describes rows that no "
            "longer exist."
        )

    # An amplitude edit made after dispatch used the old measurement context.
    # Carrying it into a new baseline (or overwriting it) would both be wrong.
    # Leave the live table intact and ask the user to run again instead.
    from .event_amplitude import AMPLITUDE_COLUMNS

    for col in AMPLITUDE_COLUMNS:
        if col not in live.columns and col not in before.columns:
            continue
        a = (live[col].to_numpy(dtype=object) if col in live.columns
             else np.full(len(live), pd.NA, dtype=object))
        b = (before[col].to_numpy(dtype=object) if col in before.columns
             else np.full(len(before), pd.NA, dtype=object))
        if any(_curated_differently(x, y) for x, y in zip(a, b)):
            raise ValueError(
                "an event amplitude was edited while the re-measure was running; "
                "the live amplitude curation was kept. Run the re-measure again "
                "to use the current table."
            )

    cols = _late_curation_columns(live, before, out)

    def _diff(col: str) -> np.ndarray:
        a = live[col].to_numpy(dtype=object)
        b = before[col].to_numpy(dtype=object)
        return np.array(
            [_curated_differently(x, y) for x, y in zip(a, b)], dtype=bool)

    # TWO MASKS, NOT ONE UNION, and this is the whole reason ``review`` is diffed
    # separately rather than appended to the loop that already existed.
    #
    # ``measured`` selects the rows whose ACCEPT/FLAG the curator touched, and it is
    # the only thing that may enter the rewrite below -- because that body writes
    # ``flag``, which is one of the frozen 20 measured columns, on every row it
    # visits. It does not write the cell back verbatim: it rebuilds the flag from
    # tokens, so a cell that merely round-trips through it still moves. Measured, on
    # a row nobody edited, with the ONLY gesture a review click elsewhere in the
    # table: ``pd.NA -> ''``, ``'short; noisy' -> 'short;noisy'``, ``'short;short'
    # -> 'short'``, ``'short;' -> 'short'``. A description column had moved a
    # measurement column.
    #
    # A fresh `characterize` output happens to hold canonical flags, so the damage
    # was LATENT on the engine's own answer and visible one non-canonical cell away:
    # a CSV re-import gives ``pd.NA`` on every unflagged row (60 of 61 measured on a
    # synthetic detection), and a hand-typed ``'noise; long tail'`` keeps its space.
    # The mask, not the rebuild's luck, is what has to make that impossible.
    #
    # ``review`` is a description of who has looked at the row and is carried on its
    # own mask, writing its own column and nothing else. So "I have looked at this"
    # can never reach the flag rebuild, and the invariant holds by construction
    # rather than by the rebuild happening to be idempotent -- which it is not.
    changed = np.zeros(len(live), dtype=bool)
    for col in CURATION_COLUMNS:
        changed |= _diff(col)
    review_changed = (_diff(REVIEW_COL) if REVIEW_COL in cols
                      else np.zeros(len(live), dtype=bool))
    if not (changed.any() or review_changed.any()):
        return out

    out = out.copy()
    pos = {int(e): i for i, e in enumerate(out["event_id"].to_numpy(dtype=np.int64))}
    flag_col = out.columns.get_loc("flag")
    accept_col = out.columns.get_loc("accepted")
    if review_changed.any():
        # A STRAIGHT DIFF CARRY, unlike ``accepted``. ``review`` records only whether
        # a person has ruled, so it has no automatic counterpart for the re-measure's
        # own judgement to win over -- there is no such thing as the app reviewing a
        # row. Marking reviewed and un-marking it both carry, in both directions, for
        # the reason the flag diff carries both: installing must be indistinguishable
        # from the click having landed one second after the job finished.
        #
        # Rebuilt as a whole column rather than written cell by cell: ``review`` is a
        # pandas ``string`` column that may hold ``pd.NA``, and ``iat`` on a nullable
        # extension array is where a stray object dtype creeps back in and the
        # ``astype`` at the end of the engine stops being a no-op.
        review_vals = out[REVIEW_COL].to_numpy(dtype=object)
        for i in np.flatnonzero(review_changed):
            row = pos.get(int(live_ids[i]))
            if row is None:
                continue
            review_vals[row] = live[REVIEW_COL].iloc[i]
        out[REVIEW_COL] = pd.array(review_vals, dtype="string")
    for i in np.flatnonzero(changed):
        row = pos.get(int(live_ids[i]))
        if row is None:
            continue

        was_flag, now_flag = before["flag"].iloc[i], live["flag"].iloc[i]
        was_tokens = set(flag_tokens(was_flag))
        now_tokens = set(flag_tokens(now_flag))

        # The curator's own tokens only, in both directions. `out`'s derived tokens
        # are the new measurement's and stay exactly as it left them.
        tokens = [
            t for t in flag_tokens(out["flag"].iloc[row])
            if t in now_tokens or t not in was_tokens or t in DERIVED_FLAGS
        ]
        for t in flag_tokens(now_flag):
            if t not in was_tokens and t not in DERIVED_FLAGS and t not in tokens:
                tokens.append(t)
        new_flag = FLAG_SEP.join(tokens)
        out.iat[row, flag_col] = new_flag

        hand_was = _hand_verdict(was_flag, before["accepted"].iloc[i])
        hand_now = _hand_verdict(now_flag, live["accepted"].iloc[i])
        if hand_now != hand_was:
            # The row's own automatic judgement still wins, so this cannot install an
            # accepted row that the re-measure has just flagged as unmeasurable --
            # `characterize` ANDs the same way, and a late click is not evidence
            # about a measurement made after it.
            auto_ok = not any(
                t in AUTOMATIC_REJECTION_FLAGS for t in flag_tokens(new_flag))
            out.iat[row, accept_col] = bool(auto_ok and hand_now)
    return out


# ===========================================================================
# the engine
# ===========================================================================
def remeasure_events(
    eventset: EventSet,
    signal: np.ndarray,
    fs: float,
    *,
    config: Any = None,
    polarity: int | None = None,
    param_source: str = "user",
    progress: Callable[[float, str], None] | None = None,
) -> RemeasureResult:
    """Re-measure every row of ``eventset`` against ``config``. No re-detection.

    Args:
        eventset: The curated table to re-measure. Its provenance supplies the
            noise scale, the analysed span, the filter rise and the polarity --
            all of which are properties of the DETECTION and must not change,
            because the only thing being revised is the open-pore reference.
        signal: The calibrated trace the detection ran on, FULL length. The
            baseline is estimated over the whole channel and then sliced per row
            (the mask threshold is a global MAD, so a per-row estimate is a
            different measurement -- 16.95 pA different on the owner's file).
        fs: Sample rate, Hz.
        config: The `~nanoscope.analysis.baseline.BaselineConfig` to measure
            against; ``None`` uses the app default.
        polarity: ``{-1, +1}``. ``None`` takes the detector's own polarity out of
            provenance, and only failing that measures one -- through
            `~nanoscope.analysis.baseline.measured_polarity`, never
            ``derive_polarity`` on the raw trace (which returns +1 on the owner's
            own drifting recordings and would point the one-sided mask the wrong
            way, moving the baseline by up to 312 pA).
        param_source: One of `~nanoscope.analysis.baseline.PARAM_SOURCES`; stamped
            so a reopened project can say how these parameters were chosen.
        progress: ``fn(fraction, message)``, called a bounded number of times.

    Returns:
        A :class:`RemeasureResult`. ``events`` carries the SAME ``event_id``s, the
        same geometry, the same extra (non-schema) columns, and a provenance whose
        ``baseline_*`` block has been replaced wholesale by the new estimator's.

    Raises:
        ValueError: the geometry moved (a bug, not a user error -- see
            `GEOMETRY_COLUMNS`), or the trace cannot carry the requested estimator.
        ~nanoscope.analysis.cancel.OperationCancelled: the ambient token was set.
    """
    # `DEFAULT_BASELINE`, never ``BaselineConfig()``: `baseline.py` owns the app's
    # only baseline-parameter literal, and an AST guard in
    # ``tests/analysis/test_baseline_seam.py`` fails the build if another module
    # constructs a config. That guard is the whole reason a fifth call site cannot
    # quietly acquire its own default, which is how this defect was built.
    from .baseline import (
        DEFAULT_BASELINE, characterisation_baseline, measured_polarity,
    )

    # FROZEN before the first row is read -- see `frozen_for_dispatch` for why the
    # engine owns a copy even when the caller already made one. What the user does
    # after this line is answered at the INSTALL, by `assert_answer_describes`, which
    # compares the geometry measured here against the geometry the store holds then.
    eventset = frozen_for_dispatch(eventset)
    df = eventset.df
    old_prov = dict(eventset.provenance or {})
    sig = np.asarray(signal)
    fs = float(fs)
    cfg = config if config is not None else DEFAULT_BASELINE

    from .baseline import BaselineConfig
    from .reference_context import reference_change_requires_detection, validate_open_pore_context
    previous_config = BaselineConfig.reproducible_from(old_prov)
    if reference_change_requires_detection(previous_config, cfg):
        raise ValueError(
            "Changing the user open-pore reference requires fresh detection to recalculate "
            "noise, Auto polarity and event boundaries. Press Detect with the new reference; "
            "Re-measure depths does not detect missed events. No existing rows were changed."
        )
    if getattr(cfg, "open_pore_reference", None) is not None:
        validate_open_pore_context({**old_prov, "baseline_config": cfg},
                                  units=old_prov.get("units"), module_id="detect.schmitt")

    _require_same_rate(old_prov, fs)

    if progress is not None:
        progress(0.0, "Estimating the open-pore baseline")
    check_cancelled()

    pol = _polarity_for_table(polarity, old_prov, sig, fs, cfg, measured_polarity)

    # ONE whole-recording estimate for the whole table, handed to every row as a
    # full-length array. Two reasons it is not left to `recharacterize_span` to
    # re-derive per row from provenance: the rows' provenance names the OLD
    # estimator (that is the entire point of this function), and the array is what
    # makes every row land on bit-identical numbers rather than on N estimates that
    # merely ought to agree.
    #
    # `on_nonfinite='interpolate'` for the same reason the per-row rebuild uses it:
    # a whole-recording estimate that REFUSES on one bad sample anywhere turns that
    # sample into a table-wide change of measurement (measured on the owner's file:
    # one inf at sample 1.9 M moved row 0's baseline from -27.16 to -31.13 pA).
    # Rows whose OWN pad contains the defect are still poisoned, by
    # `recharacterize_span`'s flank rule, which is where that judgement belongs.
    res = characterisation_baseline(
        sig, fs, config=cfg, polarity=pol, param_source=param_source,
        health=True, on_nonfinite="interpolate",
    )
    base = res.baseline
    new_prov_block = res.provenance()
    # THE NEW ESTIMATOR'S OWN ADVISORIES, stamped where `detect.py` stamps the
    # detector's (``provenance['baseline_warnings']``), so the one place that composes
    # a result's warnings -- `modules._build_warnings` -- finds them and the install can
    # rebuild the WHOLE list from the rows instead of taking this block's list as the
    # whole list. Sweep 7 measured the latter on the owner's -600 mV file at a 2 kHz
    # low-pass: four warnings before "Re-measure depths", one after -- the
    # filter-limited caveat about accepted rows still in the table was simply gone.
    new_prov_block["baseline_warnings"] = list(res.warnings)

    if not len(df):
        # An empty table still gets the new stamp: it was measured (of nothing) by
        # the new estimator, and marking it stale forever would be a lie.
        out = EventSet(df.copy(), _merged_provenance(old_prov, new_prov_block, 0, None))
        return RemeasureResult(
            events=out,
            comparison=_empty_comparison(old_prov, new_prov_block, cfg, res),
            baseline=base, baseline_provenance=new_prov_block,
        )

    columns = list(df.columns)
    order = np.argsort(df["start_idx"].to_numpy(dtype=np.int64), kind="stable")
    recording_col = df["recording"].to_numpy(dtype=object)
    pore_col = df["pore_id"].to_numpy(dtype=object)
    units = str(old_prov.get("units") or "pA")

    poller = CancelPoller()
    rows: list[pd.DataFrame] = []
    n = len(df)
    step = max(n // 20, 1)
    for k, i in enumerate(order):
        poller()
        src = df.iloc[[i]]
        row = recharacterize_span(
            sig, fs,
            int(src["start_idx"].iloc[0]), int(src["end_idx"].iloc[0]),
            provenance=old_prov,
            prior=src,                 # <- curation inheritance, the whole promise
            polarity=pol,
            baseline=base,             # <- 'provided': the detector's own numbers
            recording=str(recording_col[i]),
            pore_id=str(pore_col[i]),
            units=units,
        )
        # The row's IDENTITY is not a measurement. Reassigning it (as a merge does,
        # because a merge destroys rows) would break every reference the curation,
        # the plots and the Event Inspector hold.
        row = row.copy()
        row["event_id"] = int(src["event_id"].iloc[0])
        # ...and neither is its PROVENANCE. ``recharacterize_span`` returns the frozen
        # 20 columns and nothing else (`characterize` ends with ``df[list(SCHEMA)]``),
        # so every column outside them -- ``added``, the detector's own bounds, and
        # ``source`` / ``review`` -- is re-applied here from the row being replaced,
        # unchanged. A re-measure rewrites amplitudes; it does not re-detect, so who
        # found this event and whether a person has ruled on it are exactly what they
        # were. The one thing that CAN have moved is a review made while this job ran,
        # and that is `merge_late_curation`'s, not this loop's.
        for col in columns:
            if col not in row.columns:
                row[col] = src[col].to_numpy()[0]
        rows.append(row[columns])
        if progress is not None and (k % step == 0):
            progress(0.1 + 0.9 * (k / n), f"Re-measuring event {k + 1} of {n}")

    out_df = pd.concat(rows, ignore_index=True)
    # Back to the INPUT's row order, not start-order: a table the user has sorted
    # or hand-appended to must come back the shape it went in.
    out_df = out_df.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)
    out_df = out_df.astype({c: df[c].dtype for c in columns})[columns]

    _restore_set_wide_rise_fall(out_df, sig, base, fs, pol, order)

    # The optional amplitude is a separately curated measurement. A different
    # baseline definition invalidates it even when the scalar baseline happens
    # to coincide; unchanged definitions invalidate only rows actually changed.
    from .event_amplitude import AMPLITUDE_COLUMNS, invalidate_event_amplitudes

    if any(col in out_df.columns for col in AMPLITUDE_COLUMNS):
        if old_prov.get("baseline_fingerprint") != new_prov_block.get("baseline_fingerprint"):
            invalidate_event_amplitudes(out_df, reason="baseline_remeasured")
        else:
            changed_amplitude_context = np.zeros(len(df), dtype=bool)
            for col in REMEASURED_COLUMNS:
                if col in df.columns and col in out_df.columns:
                    changed_amplitude_context |= np.array([
                        _curated_differently(x, y)
                        for x, y in zip(df[col].to_numpy(), out_df[col].to_numpy())
                    ], dtype=bool)
            invalidate_event_amplitudes(
                out_df, rows=np.flatnonzero(changed_amplitude_context),
                reason="event_remeasured")

    _assert_geometry_held(df, out_df)

    comparison = _compare(df, out_df, old_prov, new_prov_block, cfg, res)
    # EVERY row-derived count re-read from the rows, not one of them. This used to
    # re-read ``n_clipped_boundary`` alone and carry ``n_filter_limited`` forward,
    # so after a re-measure provenance said 3 filter-limited over a table whose rows
    # flagged 2 -- stable across a second re-measure, and written into the exported
    # ``.meta.json`` (S7 MISSED-2). `reconcile_row_counts` is the one rule.
    new_prov = reconcile_row_counts(
        _merged_provenance(old_prov, new_prov_block, len(out_df), comparison), out_df)
    if progress is not None:
        progress(1.0, "Re-measured")
    return RemeasureResult(
        events=EventSet(out_df, new_prov), comparison=comparison,
        baseline=base, baseline_provenance=new_prov_block,
    )


def compare_estimators(
    eventset: EventSet,
    signal: np.ndarray,
    fs: float,
    *,
    config: Any = None,
    polarity: int | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> EstimatorComparison:
    """§4.2.7's report: what a re-measure against ``config`` would change.

    "Give him a way to see the size of the problem before committing." On a
    recording without the pathology this shows ~1.0x everywhere and no baseline
    past the 10 pA threshold; on the owner's -600 mV apo file it shows the real
    factor. Nothing is installed and nothing is mutated.

    It runs the re-measure and throws the rows away, which costs one extra pass
    and buys the guarantee that the report cannot describe a different computation
    from the button beside it. The whole-recording estimate is memoised on the
    trace object, so pressing the button afterwards does not pay for it twice.
    """
    return remeasure_events(
        eventset, signal, fs, config=config, polarity=polarity,
        param_source="user", progress=progress,
    ).comparison


# ===========================================================================
# internals
# ===========================================================================
def _polarity_for_table(
    polarity: int | None,
    provenance: Mapping[str, Any],
    signal: np.ndarray,
    fs: float,
    cfg: Any,
    measure: Callable[..., int],
) -> int:
    """``{-1, +1}`` for the whole table: caller, then the DETECTOR's, then measured.

    The detector's polarity is preferred over a fresh measurement because a
    re-measure must reproduce the frame the rows were detected in -- flipping it
    would not be a better baseline, it would be a different experiment.
    """
    if polarity in (-1, 1):
        return int(polarity)  # type: ignore[arg-type]
    p = provenance.get("polarity")
    if p in (-1, 1):
        return int(p)
    return int(measure(signal, fs, config=cfg))


def _require_same_rate(provenance: Mapping[str, Any], fs: float) -> None:
    """Refuse a re-measure at a rate the detection did not run at.

    Found by the geometry guard on the owner's own file, which is why it is a check
    of its own rather than a comment. ``t_start_s`` and ``dwell_s`` are
    ``start_idx / fs`` and ``(end - start) / fs``, so handing this function 50000.0
    for a recording whose header says 50000.001192 rewrites BOTH — and the geometry
    guard then fires with "re-measure moved 't_start_s'", which is true and names
    the wrong cause. The rate a table was detected at travels in its provenance;
    checking it here turns a confusing symptom into the actual sentence.

    The tolerance is relative and tight (1e-9): this is not about instrument
    accuracy, it is about the caller having read the rate off a different object
    from the one the detector used. The rule itself is not written here: it is
    :func:`~nanoscope.analysis.events.stale_rate_hz`, the same question the amber
    line and every in-place editing door ask (S8-DATA1), so this refusal -- the
    kernel's, worded for a caller of this function -- cannot disagree with them.
    """
    stated = stale_rate_hz(provenance, fs)
    if stated is not None:
        raise ValueError(
            f"these events were detected at {stated!r} Hz but this re-measure was "
            f"handed {fs!r} Hz. t_start_s and dwell_s are derived from the rate, so "
            "re-measuring at a different one would silently rewrite the geometry "
            "this function exists to preserve. Pass the recording's own rate."
        )


def _restore_set_wide_rise_fall(
    out_df: pd.DataFrame,
    sig: np.ndarray,
    base: np.ndarray,
    fs: float,
    pol: int,
    order: np.ndarray,
) -> None:
    """Recompute ``rise_s``/``fall_s`` for the WHOLE table, in place.

    The loop above rebuilds one row at a time, which is the only way to inherit
    one row's curation. Every column that is a reduction over ``[start, end)``
    survives that -- but ``rise_s`` and ``fall_s`` do not, because their search
    window is clipped to the NEIGHBOURING events (see
    :func:`~nanoscope.analysis.characterize.rise_fall_over_set`). Presented with
    one event, that bound degenerates to the padded window's edges and the
    +-64-sample search walks into whatever sits beside it.

    Measured on the owner's 1641-event apo detection, re-measured against its OWN
    stored estimator so that nothing else could move: ``baseline_pA``,
    ``depth_pA``, ``ecd_pA_s``, ``snr`` and ``dwell_s`` came back BIT-IDENTICAL
    while ``rise_s`` moved on 812 rows and ``fall_s`` on 920, with 331 and 341
    going from a finite value to NaN -- 20% of the table losing its rise time to
    an artefact of how it was rebuilt. Splitting those rows by neighbour distance
    is decisive: 812 of the 1399 rows with a neighbour inside the pad moved, and
    **0** of the 83 isolated rows. This one pass restores all 20 columns to
    bit-identity under a same-config re-measure, which is what
    ``test_a_same_config_remeasure_moves_no_column_at_all`` pins.

    It is done here rather than by teaching ``recharacterize_span`` about
    neighbours because a neighbour list is precisely the argument a sixth rebuild
    path could pass wrongly or forget. A whole-table rebuild has the whole table;
    it should use it.

    ``depth_pA`` from the REBUILT rows is what sets the 10%/90% levels, so a row
    the pad rule poisoned (depth NaN) is skipped and keeps its NaN, exactly as it
    did before.
    """
    if "rise_s" not in out_df.columns or "fall_s" not in out_df.columns:
        return
    starts = out_df["start_idx"].to_numpy(dtype=np.int64)[order]
    ends = out_df["end_idx"].to_numpy(dtype=np.int64)[order]
    depth = out_df["depth_pA"].to_numpy(dtype=np.float64)[order]
    rise, fall = rise_fall_over_set(
        np.asarray(sig, dtype=np.float64), np.asarray(base, dtype=np.float64),
        int(pol), starts, ends, depth, float(fs),
    )
    back = np.argsort(order, kind="stable")
    out_df["rise_s"] = rise[back]
    out_df["fall_s"] = fall[back]


def _assert_geometry_held(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Refuse to return a table whose geometry moved.

    This is an assertion about THIS FUNCTION, not about user input, and it is
    written as a raise rather than a comment because the promise the offered
    button makes -- "it re-measures, it does not re-detect" -- is exactly the
    promise a silent boundary shift would break, and the user would discover it as
    curation attached to the wrong events.
    """
    for col in GEOMETRY_COLUMNS:
        if col not in before.columns or col not in after.columns:
            continue
        a = before[col].to_numpy()
        b = after[col].to_numpy()
        if a.dtype.kind in "iu":
            same = np.array_equal(a, b)
        else:
            same = np.array_equal(a, b, equal_nan=True)
        if not same:
            bad = int(np.flatnonzero(a != b)[0]) if a.shape == b.shape else -1
            raise ValueError(
                f"re-measure moved {col!r} (first at row {bad}): a re-measure "
                "rewrites amplitudes only; moving an event boundary would be a "
                "re-detection, and the curation attached to these rows would no "
                "longer describe them."
            )
    if not np.array_equal(before["event_id"].to_numpy(), after["event_id"].to_numpy()):
        raise ValueError(
            "re-measure reassigned event_id; every curation reference, plot "
            "selection and Event Inspector target is keyed on it."
        )


def _compare(
    before: pd.DataFrame,
    after: pd.DataFrame,
    old_prov: Mapping[str, Any],
    new_block: Mapping[str, Any],
    cfg: Any,
    res: Any,
) -> EstimatorComparison:
    old_depth = before["depth_pA"].to_numpy(dtype=np.float64)
    new_depth = after["depth_pA"].to_numpy(dtype=np.float64)
    old_base = before["baseline_pA"].to_numpy(dtype=np.float64)
    new_base = after["baseline_pA"].to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = new_depth / old_depth
    # A ratio against a zero or non-finite old depth is not a number he can act
    # on; NaN it explicitly so the aggregates exclude it and `n_ratio` says so.
    ratio = np.where(
        np.isfinite(old_depth) & (old_depth > 0) & np.isfinite(new_depth),
        ratio, np.nan,
    )
    shift = np.abs(new_base - old_base)

    return EstimatorComparison(
        n_events=len(after),
        event_id=after["event_id"].to_numpy(dtype=np.int64),
        old_depth_pA=old_depth, new_depth_pA=new_depth,
        old_baseline_pA=old_base, new_baseline_pA=new_base,
        depth_ratio=ratio, baseline_shift_pA=shift,
        new_frac_depth=after["frac_depth"].to_numpy(dtype=np.float64),
        old_fingerprint=_str_or_none(old_prov.get("baseline_fingerprint")),
        new_fingerprint=_str_or_none(new_block.get("baseline_fingerprint")),
        old_description=describe_stored_baseline(old_prov),
        new_description=cfg.describe(),
        warnings=tuple(res.warnings),
    )


def _empty_comparison(
    old_prov: Mapping[str, Any], new_block: Mapping[str, Any], cfg: Any, res: Any
) -> EstimatorComparison:
    z = np.empty(0, dtype=np.float64)
    return EstimatorComparison(
        n_events=0, event_id=np.empty(0, dtype=np.int64),
        old_depth_pA=z, new_depth_pA=z, old_baseline_pA=z, new_baseline_pA=z,
        depth_ratio=z, baseline_shift_pA=z, new_frac_depth=z,
        old_fingerprint=_str_or_none(old_prov.get("baseline_fingerprint")),
        new_fingerprint=_str_or_none(new_block.get("baseline_fingerprint")),
        old_description=describe_stored_baseline(old_prov),
        new_description=cfg.describe(), warnings=tuple(res.warnings),
    )


def describe_stored_baseline(prov: Mapping[str, Any] | None) -> str:
    """How the stored rows were measured, in the owner's words, from provenance.

    A table with no ``baseline_method`` predates the stamp entirely, and ABSENCE IS
    THE SIGNAL: those are precisely the rows the 10 ms block median measured, which
    is what this whole change exists to replace. Saying so is more useful than
    "unknown", so the fallback names the estimator that was in force.
    """
    from .baseline import BaselineConfig

    try:
        cfg = BaselineConfig.reproducible_from(prov)
    except (TypeError, ValueError):
        return "invalid stored user reference (review the recipe before a fresh detection)"
    if cfg is not None:
        return cfg.describe()
    method = prov.get("baseline_method") if prov else None
    if method:
        window = prov.get("baseline_window_s")
        win = f" @ {float(window) * 1e3:g} ms" if window else ""
        return f"{method}{win}"
    return "block_median @ 10 ms (this table predates the baseline stamp)"


def _merged_provenance(
    old: Mapping[str, Any],
    new_block: Mapping[str, Any],
    n_events: int,
    comparison: EstimatorComparison | None,
) -> dict[str, Any]:
    """``old`` with its ENTIRE ``baseline_`` namespace replaced by ``new_block``.

    The namespace, not the key list: `BaselineResult.provenance` emits derived keys
    (``baseline_fingerprint``, ``baseline_health``, ``baseline_fallback``) beside
    the eight parameters, and updating only the parameters would leave a stale
    health block and a stale fallback reason describing an estimator that no
    longer measured anything here. The same lesson `modules._detect_kwargs`
    learned: make the namespace the boundary so the next derived key cannot
    reopen the hole.
    """
    out = {k: v for k, v in old.items() if not str(k).startswith("baseline_")}
    out.update(dict(new_block))
    record: dict[str, Any] = {
        "n_events": int(n_events),
        "from_fingerprint": _str_or_none(old.get("baseline_fingerprint")),
        "to_fingerprint": _str_or_none(new_block.get("baseline_fingerprint")),
        "from_description": describe_stored_baseline(old),
        "columns": list(REMEASURED_COLUMNS),
    }
    if comparison is not None:
        record["median_depth_ratio"] = comparison.median_ratio
        record["n_baseline_moved"] = comparison.n_baseline_moved
    out[REMEASURED_KEY] = record
    return out


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


# ===========================================================================
# the release note (§4.2.6)
# ===========================================================================
#: Previously exported CSVs cannot be fixed in place -- the app does not own them.
#: What it CAN do is make the next export self-describing (the ``.meta.json``
#: sidecar) and hand the owner a sentence to attach to the old ones. Kept here,
#: next to the engine, so the wording and the behaviour are revised together.
_RELEASE_NOTE = """\
NanoScope {version} -- open-pore baseline correction

CSVs exported before this version carry a BLOCK-MEDIAN baseline measured over a
fixed 10 ms window. On recordings whose events are long compared with half that
window -- which includes bursty solid-state nanopore data, where more than half
the samples in a 10 ms block can be inside an event -- that median lands INSIDE
the blockade rather than on the open pore, so the amplitude columns are
understated.

Affected columns: baseline_pA, depth_pA, frac_depth, ecd_pA_s, mean_dev_pA,
max_dev_pA, snr, rise_s, fall_s.

Unaffected: event_id, recording, pore_id, start_idx, end_idx, t_start_s, dwell_s,
n_levels, filter_limited, accepted, flag. Detection geometry and curation are
unchanged BY A RE-MEASURE -- but read the next section before comparing a
re-measured table with a fresh one.

Measured on a 22 nm pore, 1x PBS, -600 mV apo-ferritin recording (2,993,798
samples at 50 kHz), 1641 events detected at min_dwell_s = 1e-4 and re-measured
against the same spans:

    baseline_pA   -120.87  ->   -35.21 pA   (true open pore -34.8 pA)
    depth_pA        23.70  ->   110.17 pA   4.65x  (per-event median 4.73x)
    frac_depth      0.190  ->    3.116      16.4x
    ecd_pA_s        0.0106 ->    0.0501     4.72x
    mean_dev_pA     24.55  ->   112.24      4.57x
    max_dev_pA      41.18  ->   130.92      3.18x
    snr              5.76  ->    26.76      4.65x

1636 of 1641 events (99.7%) had their stamped open-pore reference move by more
than 10 pA.

rise_s and fall_s are measured at 10% and 90% OF EACH EVENT'S OWN DEPTH, so a
depth that moves 4.7x moves them too: 1243 and 1236 of the 1641 rows change, and
about 1000 of each become NaN. That is correct, not a loss -- see the next
section for why those rows had no measurable transition in the first place.

On recordings without this pathology the same columns move by ~1.0x. The app
cannot tell the two apart from an exported table alone, which is why this note
exists: re-measure the recording in NanoScope {version} or later and compare, or
use Analysis > Compare baseline estimators to see the factor for your own file
before deciding.

RE-MEASURING IS NOT THE SAME AS RE-DETECTING, AND ON THIS FILE THE DIFFERENCE IS
LARGER THAN THE DEPTH CORRECTION
-------------------------------------------------------------------------------
A re-measure keeps every event boundary exactly where it was, because that is
what protects your curation. But the old baseline did not only mis-stamp the
depths -- it also fed the detector's threshold, and a reference that dives into
each blockade rises and falls WITH the event, so a single long blockade was cut
into a string of short fragments. On the same apo recording, same detector
settings, detected fresh under each estimator:

    block median, 10 ms (before)   1641 events   median dwell   0.44 ms
    block median, 200 ms            327 events   median dwell  17.48 ms
    masked median, 0.5 / 0.1 s      315 events   median dwell  17.56 ms
    masked median, 2.0 / 0.5 s      275 events   median dwell  18.18 ms
        (2.0 / 0.5 s is what the Preprocess tab's probe proposes for this file)

Total blocked time goes 1.80 s -> 4.60 s. The event COUNT falls 5x and the median
dwell rises 40x, and neither of those is a mask or a merge artefact: the 200 ms
block median, which uses no event mask and no dilation at all, already gives 327
events at 17.48 ms.

So: a re-measured table has correct depths against dwell times that were measured
by the estimator being replaced. For a depth histogram that is fine. For a
dwell-time fit, a dwell-vs-depth scatter, an event rate, or ecd_pA_s -- all of
which integrate or bin over the span -- RE-DETECT the recording rather than
re-measure it, and expect a different number of events.

Exports from {version} onward write a <name>.meta.json sidecar naming the
estimator (baseline_method, its windows, and a baseline_fingerprint), so this
question is answerable from the files themselves from now on.
"""


def release_note(version: str = "this version") -> str:
    """The §4.2.6 note the owner can attach to CSVs exported before the change.

    Takes the version rather than reading it so a caller can name the build that
    actually shipped the change, and so the text is testable without importing the
    package metadata.
    """
    return _RELEASE_NOTE.format(version=version)
