"""THE UNDO DOOR — one snapshot stack over every mutation of every ``EventSet``,
and two siblings of the same shape at the foot of this file: one over the
hand-measurement table (:class:`MeasurementHistory`) and one over the analysis
settings (:class:`SettingsHistory`).

Why this file exists
====================
Before it, nothing in the application could be undone. ``grep -rn "QUndo"`` over
the package returned zero hits, and the measured consequences were these, on a
six-event curated table:

* **Re-running detection replaced the ``EventSet`` outright** — 2 hand rejections
  gone, ``dialogs shown: []``.
* **An Event-Inspector mask resize rewrote 14 columns** (bounds, ``dwell_s``,
  ``depth_pA``, ``snr``, …) and left ``dirty=False``: open a saved project, resize
  masks all afternoon, quit, and the title never grew a ``•``.
* **Closing a document dropped its results** with ``prompts: []`` — and for a
  SEGMENT, which is in-memory only and has no reopen path, the accept/reject
  decisions and hand-added events were simply unrecoverable.

Those are three symptoms of one hole, so this is one guarantee rather than three
patches: **every state an ``EventSet`` has ever been in, in this session, is
recoverable until the document closes.**

Why a snapshot stack and not ``QUndoStack``
===========================================
``QUndoStack`` is the obvious candidate and it loses here, for a structural
reason. ``QUndoStack.push()`` *performs* the command: every mutation site has to
build a ``QUndoCommand`` that knows how to do and un-do itself, BEFORE it mutates.
This application mutates an ``EventSet`` from eleven call sites. Six install a new
result (a finished detection, a batch detection, a merge from any of its three entry
points, a hand-added missed event, a re-measure, a project restore); the other **five
write the shared ``DataFrame`` in place** from inside a Qt slot on a widget — the
table's ``accepted`` checkbox, the Inspector's single accept/reject, Accept-all /
Reject-all, a mask drag, and the table's flag cell. Retro-fitting a command object
onto each is precisely the "fix applied at N call sites" shape this project has
watched come back three times: the twelfth mutation, written next month, would forget
to push and would be silently un-undoable.

So the door is inverted. This class keeps a **pristine deep copy** of every live
result and compares. A mutation does not have to announce itself to be captured —
announcing it only makes the *label* sharper, and even the label is derived from
what actually changed (:func:`describe_change`) rather than declared. A forgotten
announcement therefore costs a coarser menu string, never the data. That is the
only shape of this feature that a future call site cannot break.

The cost of the inversion is that ``checkpoint`` has to diff two frames. Measured
end to end on the owner's ``_0006_`` recording (317 events x 21 columns): **0.16 ms**
for a checkpoint that finds nothing changed, **1.4 ms** for one that records a step
(the deep copy dominates), and 19 ms for the diff alone on a hypothetical
100 000-event table. It is called on user-scale events — a click, a drag release, a
finished detection — never inside a per-event loop.

What is protected
=================
The snapshot is the whole :class:`~nanoscope.core.analysis.AnalysisResult`, not
just the frame, because a re-measure moves ``params``, ``baseline_fingerprint``
and ``warnings`` as well as the numbers. The frame is copied ``deep=True``, so it
carries the **non-schema** columns the frozen 20-column schema does not know about
— ``added`` (the hand-added-event marker the detector score reads) above all —
and every hand-written ``flag`` token.

Ownership and lifetime
======================
One :class:`EventHistory` per :class:`~nanoscope.core.session.Session`, i.e. per
window. Entries are keyed ``(doc_id, channel, module_id)`` and Undo acts on the
key the user is looking at. The alternative — one flat stack across every open
recording — loses: with four voltages open, Ctrl+Z would silently repair a table
on a recording that is not on screen, and the user would see nothing happen. A
per-key stack can always show its effect.

The four lifetime questions, and the answer to each:

* **A document closes.** Its entries are dropped (:meth:`drop_doc`) — a stack that
  outlived its recording could not be applied to anything, so Undo would name a
  step and then do nothing. That is exactly why
  :meth:`~nanoscope.gui.main_window.MainWindow._close_document` has to confirm
  first: closing is the one destructive action undo cannot cover.
* **A project opens** (or the session is otherwise reset). The whole stack goes
  (:meth:`clear`), for the same reason — nothing it holds is applicable any more.
  During the restore itself recording is suspended (:meth:`set_suspended`), so a
  reconstruction never becomes an offer to undo the project you just opened.
* **A project saves.** The stack SURVIVES. A save is not a state change to the
  tables, and losing the history at the moment the user finally has a safety net
  on disk is backwards. Undoing after a save legitimately re-dirties the session.
* **A segment whose parent closed.** Close-parent-only leaves the segment open as
  its own document, so its stack is untouched — the segment is exactly the case
  where the history is most precious, since the segment itself can never be
  rebuilt from disk.
"""

from __future__ import annotations

import copy
import weakref
from dataclasses import dataclass, fields as dc_fields, is_dataclass, replace
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from .analysis import AnalysisResult

__all__ = [
    "MAX_DEPTH",
    "MAX_BYTES",
    "EventKey",
    "describe_change",
    "EventHistory",
    "MeasurementHistory",
    "SettingsScope",
    "SettingsHistory",
]

#: Largest number of undoable steps kept **PER EVENT TABLE** — per ``EventKey``, i.e. the
#: unit Undo itself acts on. Measured on the owner's ``_0006_`` recording: 317 events x 21
#: columns is **125.8 KiB** per snapshot, so 30 steps cost 3.7 MiB there (measured RSS
#: across 20 real operations: **+1.9 MB**). It is more curation history than a session
#: realistically walks back through, and :data:`MAX_BYTES` binds first on any table big
#: enough for the depth to matter.
#:
#: IT USED TO BE A SESSION-WIDE BUDGET, and that was a data-loss bug, not a policy. ``_trim``
#: deleted ``self._undo[0]`` — the globally oldest entry, whatever key it belonged to — while
#: every READ (:meth:`EventHistory.can_undo`, :meth:`EventHistory.undo`) is per key. Measured
#: on the owner's quick ABF: *Merge all* took recording A from 115 events to 87 and Edit read
#: *"Undo Merge events (115 → 87)"*; 31 ordinary accept/reject clicks on a DIFFERENT recording
#: then pushed the shared stack to 30 and evicted A's only entry, after which Edit read
#: *"Undo — nothing to undo on this event table"* — the wording used for a table nobody has
#: ever edited — and the 28 merged-away rows were unrecoverable. That contradicted this
#: module's own headline guarantee (every state an ``EventSet`` has been in is recoverable
#: until the document closes), its own trim contract below ("the step the user is about to
#: undo is never the one that went"), and both user-facing documents, which pair the number
#: with "per event table" / "its recording" and declare closing a recording the ONE one-way
#: door. Depth is therefore per table; the SESSION-wide bound is :data:`MAX_BYTES`, which is
#: a resource ceiling and stays global — see there for why the two cannot be swapped.
MAX_DEPTH = 30

#: Hard ceiling on the bytes held by undo + redo together — whichever of the two
#: limits binds first. Measured: a 100 000-event table is **38.7 MiB** per snapshot,
#: so ``MAX_DEPTH`` alone would allow 1.1 GB; under this ceiling such a table keeps
#: 3 steps, and the owner's realistic worst case (~8 700 events at his 15 M-sample
#: ceiling, 3.4 MiB a step) keeps the full 30.
#:
#: Past the ceiling the OLDEST steps are dropped, so the history shortens from the
#: far end and the step the user is about to undo is never the one that went. An
#: entry is dropped WHOLE — depth is never traded for a partially restored table,
#: because half a table is worse than no undo at all.
#:
#: THIS ONE IS SESSION-WIDE AND HAS TO BE. Depth is a comfort budget and is per table
#: (:data:`MAX_DEPTH`); bytes are RAM, and RAM is shared, so a per-table byte budget would
#: multiply by the number of open detections and reintroduce the 1.1 GB this ceiling exists
#: to prevent. What was wrong before was not that it is global but that it evicted the
#: globally oldest entry — the small, cheap history of a recording that was not the reason
#: the ceiling bound. It now evicts the oldest entry of the key holding the MOST bytes (the
#: table actually spending the budget), and :meth:`EventHistory.evicted` remembers it so the
#: Edit menu can say a history was dropped rather than that the table was never edited.
#:
#: WHAT THIS CEILING DOES NOT COUNT: the pristine copy of each LIVE table
#: (:attr:`EventHistory._pristine`, one per open detection, measured at 11.1 KiB for six
#: synthetic recordings and ~3.4 MiB per open recording at the owner's realistic
#: worst case). That is bounded by the number of open tables rather than by session
#: length, and it is what the comparison is against, so trimming it would disable the
#: feature rather than shrink it — but ``nbytes()`` reports the stack only, and the
#: resident cost of the history is that number PLUS one copy of every live table.
MAX_BYTES = 128 * 1024 * 1024

#: ``(doc_id, channel, module_id)`` — the ResultStore's own key. An ML detection
#: and a Schmitt detection on the same channel are different tables and get
#: different stacks, which is right: undoing a Schmitt merge must not reach into
#: an ML result the user also has.
EventKey = tuple[str, int, str]

#: Columns that describe WHERE an event is. A rebuild that leaves these alone is a
#: re-measurement; one that moves them is a re-detection or a mask edit.
_GEOMETRY = ("start_idx", "end_idx")

#: Columns that carry the curator's decisions rather than a measurement.
_CURATION = ("accepted", "flag", "added")

#: Columns that carry a measurement of the event.
_MEASURED = (
    "dwell_s", "baseline_pA", "depth_pA", "frac_depth", "ecd_pA_s", "max_dev_pA",
    "mean_dev_pA", "rise_s", "fall_s", "snr", "n_levels", "filter_limited",
    "t_start_s",
)


# ================================================================================================
# frame comparison
# ================================================================================================
def _column_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality with NaN == NaN and ``pd.NA`` == ``pd.NA``.

    Plain ``a == b`` reports every NaN cell as a change, which would make a table
    holding one rejected ``non_finite`` row report a mutation on every checkpoint
    and push an undo step for a click that changed nothing.

    Numeric and boolean columns take the vectorised path; only the three string
    columns (``recording``, ``pore_id``, ``flag``) reach the object compare, and
    that is done in C first with a Python loop only over the cells it disagreed
    about. Measured on a 100 000-event frame: 19 ms, against 39 ms for the
    straight Python loop — and this runs on every curation click.

    THE BRANCH IS DECIDED BY WHAT NUMPY GOT, NEVER BY WHAT PANDAS CALLS IT, and that
    distinction is the whole of a data-loss bug. ``pd.Int64Dtype`` — a MASKED dtype —
    reports ``kind == 'i'``, so it used to take the integer branch, while
    ``Series.to_numpy()`` on it hands back ``float64`` with ``NaN`` in every null cell:
    the branch then compared ``NaN == NaN`` with a raw ``==`` and answered False, against
    this function's own first line. Measured on the -1000 mV recording (115 detected
    events): "Mark selection as missed event" appends a row with no detector bounds, so
    ``orig_start_idx``/``orig_end_idx`` are ``pd.NA`` on it and dtype ``Int64`` — and
    ``_frames_equal(pristine, live)`` then returned False on EVERY subsequent checkpoint
    for a table nobody had touched. Each of those pushed a phantom ``"Edit events"`` step
    whose "pre-edit" snapshot already contained the added row, and since
    :meth:`EventHistory._move` checkpoints before it moves, Ctrl+Z undid a freshly minted
    phantom instead of the add — 116 rows in, 116 rows out, forever, with the real
    ``"Add event"`` entry sitting one place further down and unreachable.

    A masked column whose nulls differ between the two frames materialises as ``float64``
    on one side and ``int64`` on the other; the dtypes then disagree and the object path
    below decides it, which is correct (``pd.NA`` is a singleton, so ``x is y`` answers
    it) and rare enough that its cost never lands on a curation click.
    """
    if a.dtype == b.dtype:
        av = a.to_numpy()
        bv = b.to_numpy()
        if av.dtype == bv.dtype:
            kind = av.dtype.kind
            if kind in ("f", "c"):
                with np.errstate(invalid="ignore"):
                    return np.asarray((av == bv) | (np.isnan(av) & np.isnan(bv)), dtype=bool)
            if kind in ("i", "u", "b", "M", "m"):
                return np.asarray(av == bv, dtype=bool)
    av = a.to_numpy(dtype=object, copy=False)
    bv = b.to_numpy(dtype=object, copy=False)
    # numpy compares object arrays element-wise in C. It is ~30x the Python loop
    # below on the three string columns, and it is only WRONG about NaN/NA cells --
    # which the loop then re-decides, on the handful of rows it flagged.
    try:
        fast = np.asarray(av == bv, dtype=bool)
    except (TypeError, ValueError):            # pd.NA in a cell: no boolean answer
        fast = None
    if fast is not None and fast.all():
        return fast
    out = np.empty(av.shape, dtype=bool)
    for i in range(av.size):
        if fast is not None and fast[i]:
            out[i] = True
            continue
        x, y = av[i], bv[i]
        if x is y:
            out[i] = True
            continue
        try:
            xn = isinstance(x, float) and np.isnan(x)
            yn = isinstance(y, float) and np.isnan(y)
        except TypeError:                      # pragma: no cover - exotic cell type
            xn = yn = False
        if xn or yn:
            out[i] = xn and yn
            continue
        try:
            out[i] = bool(x == y)
        except Exception:                      # noqa: BLE001 - a cell must never raise here
            out[i] = False
    return out


#: Columns that are OPTIONAL on a live frame, with the value their absence means.
#: ``added`` is materialised only when a detection is DISPLAYED
#: (``AnalysisPanel._show_result``), so a table produced by a batch detect over four
#: recordings carries it on the one the panel is showing and on none of the others —
#: while :func:`_snapshot` materialises it on every copy it takes. Comparing the
#: column LISTS then reported an untouched table as mutated on every single
#: ``checkpoint_all()``: measured with four recordings batch-detected, three phantom
#: "Edit events" steps per Edit-menu open, which re-dirtied a just-saved project and
#: evicted the user's real curation steps at :data:`MAX_DEPTH`. An absent ``added``
#: means "no hand-added events", so the two frames are reconciled HERE, at the one
#: comparison both sides pass through, rather than by teaching every producer to
#: materialise the column.
_OPTIONAL_COLUMNS: dict[str, Any] = {"added": False}


def _reconciled(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give ``a`` and ``b`` the same optional columns, defaults filled in.

    Shallow copies (``copy(deep=False)``) — the column added is a fresh array and the
    existing blocks are shared, so neither input frame is touched and nothing is
    duplicated.
    """
    for name, default in _OPTIONAL_COLUMNS.items():
        in_a, in_b = name in a.columns, name in b.columns
        if in_a == in_b:
            continue
        if in_a:
            b = b.copy(deep=False)
            b[name] = np.full(len(b), default)
        else:
            a = a.copy(deep=False)
            a[name] = np.full(len(a), default)
    return a, b


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Do these two event frames hold the same values, column set and row order?"""
    if len(a) != len(b):
        return False
    a, b = _reconciled(a, b)
    if list(a.columns) != list(b.columns):
        return False
    if len(a) == 0:
        return True
    for col in a.columns:
        if not bool(_column_equal(a[col], b[col]).all()):
            return False
    return True


def _aligned(before: pd.DataFrame, after: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows the two frames have in common, by ``event_id``, in a shared order.

    Row POSITION is not identity here: a merge renumbers, a hand-added event
    appends, a re-detection restarts at zero. ``event_id`` is what
    ``_write_measured_row`` deliberately preserves through an edit, so it is the
    only honest join key.
    """
    ids_a = before["event_id"].to_numpy(dtype=np.int64)
    ids_b = after["event_id"].to_numpy(dtype=np.int64)
    common = np.intersect1d(ids_a, ids_b)
    if common.size == 0:
        return before.iloc[:0], after.iloc[:0]
    try:
        pa = pd.Index(ids_a).get_indexer(common)
        pb = pd.Index(ids_b).get_indexer(common)
    except Exception:            # noqa: BLE001 - duplicated ids: no honest alignment exists
        return before.iloc[:0], after.iloc[:0]
    if (pa < 0).any() or (pb < 0).any():       # pragma: no cover - defensive
        return before.iloc[:0], after.iloc[:0]
    return before.iloc[pa], after.iloc[pb]


def _n_rows_differing(a: pd.DataFrame, b: pd.DataFrame, columns: Iterable[str]) -> int:
    """How many aligned rows differ in ANY of ``columns`` (missing columns skipped)."""
    if len(a) == 0:
        return 0
    diff = np.zeros(len(a), dtype=bool)
    for col in columns:
        if col in a.columns and col in b.columns:
            diff |= ~_column_equal(a[col], b[col])
    return int(diff.sum())


def _all_hand_made(df: pd.DataFrame) -> bool:
    """Is every row of ``df`` one a person drew (``source`` manual, ``added`` where unrecorded)?"""
    if not len(df):
        return False
    from ..analysis.manual_events import hand_made_mask
    return bool(hand_made_mask(df).all())


def _deleted_hand_made(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """How many rows ``after`` lost that were hand-made, when that is ALL it lost and the
    survivors kept their ids (a delete) -- 0 for a merge, which renumbers the survivors."""
    if "event_id" not in before.columns or "event_id" not in after.columns:
        return 0
    ids_b = before["event_id"].to_numpy()
    ids_a = after["event_id"].to_numpy()
    if not np.isin(ids_a, ids_b).all():
        return 0
    gone = ~np.isin(ids_b, ids_a)
    if not gone.any() or int(gone.sum()) != len(before) - len(after):
        return 0
    # The survivors must be the very same events. A merge renumbers its survivors, so its ids
    # can also be a subset of the old ones; only a delete leaves every survivor's span alone.
    where = pd.Index(ids_b).get_indexer(ids_a)
    for col in ("start_idx", "end_idx"):
        if col not in before.columns or col not in after.columns:
            return 0
        if not np.array_equal(before[col].to_numpy()[where], after[col].to_numpy()):
            return 0
    from ..analysis.manual_events import hand_made_mask
    return int(gone.sum()) if bool(hand_made_mask(before)[gone].all()) else 0


def describe_change(before: pd.DataFrame | None, after: pd.DataFrame) -> str:
    """Name the operation that turned ``before`` into ``after``.

    The label is DERIVED, not declared, and that is the point: a mutation path
    added later gets a correct menu string without knowing this module exists.
    ``"Edit events"`` is the honest fallback, never a lie about which button ran.

    The order of the tests matters and each one is a real distinction the app
    makes:

    * fewer rows -> a merge (the only thing that collapses rows);
    * more rows with the shared rows untouched -> a hand-added missed event;
    * the ``event_id`` set replaced, or most spans moved -> a re-detection;
    * a few spans moved -> an Event-Inspector mask resize;
    * spans identical but the numbers moved -> a re-measure. This is the app's own
      contract ("it re-measures, it does not re-detect, so the curated event set
      survives"), which is exactly what makes the two separable here;
    * only ``accepted``/``flag``/``added`` moved -> curation.
    """
    if before is None:
        return "Detection"
    n0, n1 = len(before), len(after)
    if n0 == 0 and n1 == 0:
        return "Edit events"
    if n1 < n0:
        gone = _deleted_hand_made(before, after)
        if gone:
            return "Delete event" if gone == 1 else f"Delete {gone} events"
        return "Merge events" if n0 - n1 == 1 else f"Merge events ({n0} → {n1})"

    ba, aa = _aligned(before, after)
    moved_geometry = _n_rows_differing(ba, aa, _GEOMETRY)

    if n1 > n0:
        added = n1 - n0
        # An empty table growing is a detector that found something this time, not the
        # user hand-adding N missed events one click at a time -- unless every row it grew by
        # is hand-made, which is how a Manual table starts.
        if n0 == 0 and _all_hand_made(after):
            return "Add event" if added == 1 else f"Add {added} events"
        if n0 and moved_geometry == 0 and len(ba) == n0:
            return "Add event" if added == 1 else f"Add {added} events"
        return "Re-run detection"

    # Same row count from here on.
    if len(ba) < n0 or (n0 and moved_geometry * 2 > n0):
        return "Re-run detection"
    if moved_geometry:
        return ("Resize event mask" if moved_geometry == 1
                else f"Resize {moved_geometry} event masks")
    if _n_rows_differing(ba, aa, _MEASURED):
        return "Re-measure depths"

    # Optional amplitude choices are curation, not replacements for depth_pA.
    # The first choice creates columns, so compare absence as an unrecorded cell.
    from ..analysis.event_amplitude import AMPLITUDE_COLUMNS

    amplitude_changed = np.zeros(len(ba), dtype=bool)
    for col in AMPLITUDE_COLUMNS:
        if col not in ba.columns and col not in aa.columns:
            continue
        left = ba[col] if col in ba.columns else pd.Series([pd.NA] * len(ba))
        right = aa[col] if col in aa.columns else pd.Series([pd.NA] * len(aa))
        left = left.astype(object).where(left.notna(), pd.NA)
        right = right.astype(object).where(right.notna(), pd.NA)
        amplitude_changed |= ~_column_equal(left, right)
    n_amplitudes = int(amplitude_changed.sum())
    if n_amplitudes:
        return ("Edit event amplitude" if n_amplitudes == 1
                else f"Edit {n_amplitudes} event amplitudes")

    changed = _n_rows_differing(ba, aa, _CURATION)
    if not changed and _n_rows_differing(ba, aa, ("note",)):
        return "Edit note"
    if changed:
        if "accepted" in aa.columns and changed == n0 and n0 > 1:
            values = aa["accepted"].to_numpy(dtype=bool)
            if values.all():
                return "Accept all events"
            if not values.any():
                return "Reject all events"
        return "Curate event" if changed == 1 else f"Curate {changed} events"
    return "Edit events"


# ================================================================================================
# the stack
# ================================================================================================
@dataclass(eq=False)
class _Entry:
    """One recoverable state of one event table.

    ``eq=False`` on purpose: the generated ``__eq__`` would compare the embedded
    ``DataFrame``, and ``list.remove`` would then raise
    ``"The truth value of a DataFrame is ambiguous"``. Entries are IDENTITIES, not
    values — two snapshots of the same table are two different steps.
    """

    key: EventKey
    result: AnalysisResult          # payload is a DEEP-copied EventSet
    label: str
    nbytes: int


def _snapshot(res: AnalysisResult) -> AnalysisResult:
    """A fully detached copy of ``res`` — the frame deep-copied, the provenance copied.

    The Event Inspector and the event table write the shared frame IN PLACE, so a
    reference here would silently track the very edits it exists to remember.
    """
    from ..analysis.events import EventSet
    from ..analysis.scoring import ensure_added_column

    events = res.payload
    df = events.df.copy(deep=True)
    # Provenance is copied DEEP for the same reason the frame is: `exclude_regions`,
    # `polarity_evidence` and `baseline_warnings` are mutable containers held by the
    # live result, and a snapshot sharing them would silently track edits to the very
    # description of the rows it exists to remember.
    try:
        prov = copy.deepcopy(dict(events.provenance or {}))
    except Exception:                          # noqa: BLE001 - a copy must never sink a checkpoint
        prov = dict(events.provenance or {})
    # The 'added' marker is materialised on the live frame the moment a detection is
    # DISPLAYED, so a snapshot taken before that point would differ from every later
    # one by a whole column and report a mutation that never happened.
    ensure_added_column(df)
    return replace(res, payload=EventSet(df, prov))


def _looks_like_remeasure(before: AnalysisResult, after: AnalysisResult) -> bool:
    """Did the open-pore estimator change and nothing else about how this ran?

    ``AnalysisPanel._remeasure_finished`` rebuilds ``params`` as "everything that
    does not start with ``baseline_``, plus the new baseline block", and restamps
    ``baseline_fingerprint``. A detection re-run leaves the baseline block alone
    and (when it differs at all) differs in the detector's own parameters.
    """
    if before.baseline_fingerprint == after.baseline_fingerprint:
        return False
    pb = dict(before.params or {})
    pa = dict(after.params or {})
    moved = {k for k in set(pb) | set(pa) if pb.get(k, _MISSING) != pa.get(k, _MISSING)}
    return bool(moved) and all(str(k).startswith("baseline_") for k in moved)


#: Sentinel for "this key was absent", so a key added by a re-measure counts as moved.
_MISSING = object()
#: The names a step that put a new run on its table goes by (:attr:`EventHistory.last_step_run`).
_RUN_LABELS = frozenset({"Re-run detection", "Re-measure depths"})


def _hand_curation(df: pd.DataFrame) -> dict[str, int]:
    """How much of this table is a PERSON's work rather than the detector's.

    The distinction is the one :func:`~nanoscope.analysis.events.inherited_curation`
    already owns and it is the only one worth quoting at the user: a rebuild
    re-derives an automatic rejection (``clipped_by_boundary``, ``non_finite``, …)
    from the new span, so losing it costs nothing, while a rejection a person made —
    a bare ``accepted=False``, or one carrying a reason outside
    :data:`~nanoscope.analysis.events.DERIVED_FLAGS` — cannot be recovered from the
    data. Counting the two together overstates what a re-detection is about to
    destroy, which is the fastest way to teach someone to ignore the warning.

    Parsed through the vocabulary's own :func:`~nanoscope.analysis.events.flag_tokens`,
    and parsed once per DISTINCT cell rather than once per row — a 100 000-event table
    holds a handful of distinct flag strings, and this runs on every recorded step.
    """
    from ..analysis.events import DERIVED_FLAGS, flag_tokens, is_automatic_rejection
    from ..analysis.manual_events import hand_made_mask

    out = {"n_hand_rejected": 0, "n_notes": 0, "n_added": 0,
           "n_amplitude_curated": 0}
    if len(df) == 0:
        return out
    # Hand-MADE rows are carried into a re-run, so they are nobody's loss to quote.
    mine = ~hand_made_mask(df)
    if "flag" in df.columns:
        cells = df["flag"].astype("string").fillna("")
        parsed = {v: set(flag_tokens(v)) for v in cells.unique()}
        has_note = {v: bool(t - DERIVED_FLAGS) for v, t in parsed.items()}
        # "Rejected for a reason the next run will re-derive" — the ONLY case a rebuild
        # is allowed to drop, per `inherited_curation`. No reason at all is a person.
        #
        # Asked THROUGH `is_automatic_rejection`, not re-derived from DERIVED_FLAGS
        # here, because this line and `inherited_curation` must agree by
        # construction: written as `bool(t) and not has_note[v]` they disagreed in
        # both directions. A row rejected with `zero_baseline` counted as 0 hand
        # rejections while the rebuild treated it as a person's (so the warning was
        # silent about a rejection that WAS destroyed); a row rejected
        # `clipped_by_boundary;molecule=apo` counted as 1 while the rebuild
        # re-derives it (so the warning over-counted one that costs nothing). Note
        # `has_note` is a different question again — "did a person write anything
        # here" — and a label IS a note, so it keeps its own answer.
        explained = {v: is_automatic_rejection(v) for v in parsed}
        notes = cells.map(has_note).to_numpy(dtype=bool)
        derived_only = cells.map(explained).to_numpy(dtype=bool)
    else:
        notes = np.zeros(len(df), dtype=bool)
        derived_only = np.zeros(len(df), dtype=bool)
    accepted = (df["accepted"].to_numpy(dtype=bool) if "accepted" in df.columns
                else np.ones(len(df), dtype=bool))
    out["n_notes"] = int((notes & mine).sum())
    out["n_hand_rejected"] = int((~accepted & ~derived_only & mine).sum())
    if "added" in df.columns:
        out["n_added"] = int((df["added"].to_numpy(dtype=bool) & mine).sum())
    if "amplitude_method" in df.columns:
        choices = df["amplitude_method"].astype("string").fillna("").str.strip()
        out["n_amplitude_curated"] = int((choices.ne("").to_numpy(dtype=bool) & mine).sum())
    return out


def _summarise(df: pd.DataFrame) -> dict[str, int]:
    """What a table held — the counts a warning about replacing it has to quote."""
    out = {"n_events": int(len(df)), "n_rejected": 0, "n_added": 0}
    if len(df) and "accepted" in df.columns:
        out["n_rejected"] = int((~df["accepted"].to_numpy(dtype=bool)).sum())
    out.update(_hand_curation(df))
    return out


def _curation_was_reset(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Did a replacement table arrive carrying NONE of the hand work the old one held?

    A merge, a mask resize and a hand-added event all rebuild rows and all carry hand
    curation across (``inherited_curation`` is the rule, and it is a frozen contract).
    A fresh detection cannot: it has never seen the old table. So "the previous table
    held hand curation and the new one holds none" is the signature of a re-detection
    — and it is available even when the two tables are otherwise identical, which is
    exactly the case (Detect pressed twice with the same settings) that ``describe_change``
    reads as a curation edit and that made C14's loss invisible.
    """
    b = _hand_curation(before)
    if not any(b.values()):
        return False
    a = _hand_curation(after)
    return not any(a.values())


def _is_carried_rerun(before: AnalysisResult, after: AnalysisResult) -> bool:
    """Is ``after`` a NEW run that carried the hand-made rows of ``before`` into itself?

    The carry-over stamps ``carried_hand_rows`` on the new run's provenance. A re-run is told
    from a later edit of that same table by the run stamp: ``detected_at`` is new on every run
    and copied intact by every edit."""
    pa = after.payload.provenance or {}
    pb = before.payload.provenance or {}
    return bool(pa.get("carried_hand_rows")) and pa.get("detected_at") != pb.get("detected_at")


def _new_run_stamp(before: AnalysisResult, after: AnalysisResult) -> bool:
    """Does ``after`` carry a run stamp ``before`` did not? ``detected_at`` is new on every run
    and copied intact by every edit (:func:`_is_carried_rerun`)."""
    stamp = (after.payload.provenance or {}).get("detected_at")
    return stamp is not None and stamp != (before.payload.provenance or {}).get("detected_at")


def _params_moved_outside_baseline(before: AnalysisResult, after: AnalysisResult) -> bool:
    """Did a setting that is NOT part of the open-pore block change?

    A merge and a hand-added event re-install with ``dataclasses.replace``, so their
    ``params`` are the previous result's, unchanged. A re-measure rebuilds ``params``
    as "everything that does not start with ``baseline_``, plus the new baseline
    block". Only a detection re-run can move a detector parameter — so a moved
    ``k_in``/``min_dwell_s`` names the operation with no help from the frame.
    """
    pb = dict(before.params or {})
    pa = dict(after.params or {})
    moved = {k for k in set(pb) | set(pa) if pb.get(k, _MISSING) != pa.get(k, _MISSING)}
    return any(not str(k).startswith("baseline_") for k in moved)


# ================================================================================================
# result comparison — the WHOLE result, never an enumeration of fields
# ================================================================================================
#
# ``checkpoint`` used to ask three questions: did ``payload.df`` move, did
# ``baseline_fingerprint`` move, did ``warnings`` move. Every other field of
# :class:`~nanoscope.core.analysis.AnalysisResult` was invisible, and ``provenance`` --
# which carries the analysed span and ``analyzed_duration_s``, the denominator of
# ``EventSet.rate_hz`` -- was the one that cost a number.
#
# Measured (C-3), 5 events at 50 kHz: detect over the whole file (span (0, 60000),
# 1.2 s, rate 4.1667 Hz), then re-run the detector over the VISIBLE VIEW, which
# "Visible view" being combo index 0 makes the default. The 21-column frame comes back
# byte-identical; the provenance moves to span (0, 45000), 0.9 s, rate 5.5556 Hz.
# ``checkpoint`` returned False, so no step was pushed AND ``_pristine`` was never
# refreshed. Reject one event, press Ctrl+Z -- and the undo aimed at one checkbox
# silently reinstated the previous detection's span and duration: 5.5556 -> 4.1667 Hz,
# a 33% error in a plotted, exported quantity, under a status line reading only
# "Undone: Curate event."
#
# So the comparison is over ``dataclasses.fields(AnalysisResult)``, not over a list
# somebody maintains. A field added to the result next month is compared the day it is
# added, without this file being edited -- which is the whole difference between a fix
# and the same fix again.
#
# THREE DIFFERENCES THIS DELIBERATELY DOES NOT SEE, all inside :func:`_column_equal`:
#
# * a dtype-only change (``int64`` -> ``float64`` holding the same values),
# * a pure index relabelling (values are compared positionally, via ``to_numpy``),
# * ``0.0`` vs ``-0.0``.
#
# Each is invisible on purpose. The question this comparison answers is "did the user's
# data change", and none of those three is a change to it -- while making them visible
# re-opens a failure this module already paid for once: ``_OPTIONAL_COLUMNS`` exists
# because a frame that differed only in an ABSENT column reported a mutation on every
# ``checkpoint_all()``, pushing phantom "Edit events" steps that re-dirtied a
# just-saved project and evicted the user's real curation at MAX_DEPTH. pandas dtype
# churn (a column round-tripping through parquet, ``added`` materialised late) is
# exactly that shape of noise. Value equality is the honest test.


def _values_equal(a: Any, b: Any) -> bool:
    """Deep, NaN-tolerant equality for the non-frame fields of an ``AnalysisResult``.

    ``params``, ``scalars``, ``provenance`` and ``overlays`` hold JSON-shaped data --
    scalars, strings, lists, dicts and the occasional numpy array (both
    ``core.project._json_safe`` and ``analysis.scoring._json_safe`` serialise them on
    exactly that assumption). Everything else is compared with ``==`` and, when that
    cannot produce a boolean answer, is treated as EQUAL.

    Treating the unanswerable as equal is the conservative direction and it is chosen
    deliberately: ``_snapshot`` deep-copies provenance, so a value whose ``__eq__`` is
    identity would compare unequal against its own copy on EVERY checkpoint and push a
    phantom step for a click that changed nothing. A missed difference costs a coarser
    undo label; a phantom difference costs the user's undo stack.
    """
    if a is b:
        return True
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return type(a) is type(b) and a == b
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)
                and _frames_equal(a, b))
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping)):
            return False
        if set(a) != set(b):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (np.ndarray, pd.Series)) or isinstance(b, (np.ndarray, pd.Series)):
        try:
            av, bv = np.asarray(a), np.asarray(b)
        except Exception:                      # noqa: BLE001 - a compare must never raise
            return True
        if av.shape != bv.shape:
            return False
        if av.dtype.kind in "fc" and bv.dtype.kind in "fc":
            return bool(np.array_equal(av, bv, equal_nan=True))
        try:
            return bool(np.array_equal(av, bv))
        except Exception:                      # noqa: BLE001
            return True
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (set, frozenset)) or isinstance(b, (set, frozenset)):
        return (isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset))
                and a == b)
    if type(a) is type(b) and type(a).__eq__ is object.__eq__:
        # Identity-only equality. `_snapshot` deep-copies provenance, so such a value can
        # NEVER compare equal to its own copy -- reporting it as changed would push a
        # phantom step on every checkpoint, forever, for a click that changed nothing.
        return True
    try:
        return bool(a == b)
    except Exception:                          # noqa: BLE001 - see the docstring
        return True


def _payloads_equal(a: Any, b: Any) -> bool:
    """Equality for the ``payload`` field: an ``EventSet`` is its frame AND its provenance."""
    from ..analysis.events import EventSet

    if isinstance(a, EventSet) or isinstance(b, EventSet):
        if not (isinstance(a, EventSet) and isinstance(b, EventSet)):
            return False
        return (_frames_equal(a.df, b.df)
                and _values_equal(dict(a.provenance or {}), dict(b.provenance or {})))
    return _values_equal(a, b)


def _results_equal(a: AnalysisResult, b: AnalysisResult) -> bool:
    """Are these two results the same in EVERY field the dataclass declares?

    Enumerated from ``dataclasses.fields`` on purpose — see the block comment above. The
    frozen 20-column EventSet schema means new information arrives as a provenance key or
    a flag token rather than a column, which is precisely the traffic an enumeration of
    field names would keep missing.
    """
    for f in dc_fields(AnalysisResult):
        av, bv = getattr(a, f.name, None), getattr(b, f.name, None)
        ok = _payloads_equal(av, bv) if f.name == "payload" else _values_equal(av, bv)
        if not ok:
            return False
    return True


def _sizeof(res: AnalysisResult) -> int:
    try:
        return int(res.payload.df.memory_usage(deep=True).sum())
    except Exception:                          # noqa: BLE001 - accounting must never raise
        return 0


class EventHistory(QObject):
    """Undo/redo over every ``EventSet`` in a session. See the module docstring.

    Lives on :class:`~nanoscope.core.session.Session`'s side of the app but is
    driven ENTIRELY by :class:`~nanoscope.gui.main_window.MainWindow`, which is
    the only object that already receives all four mutation announcements
    (``resultsChanged``, ``eventAcceptanceChanged``, ``curationBatchChanged``,
    ``eventEdited``). No panel knows this class exists — the sole-wirer rule holds
    for undo exactly as it holds for everything else.
    """

    #: The stack moved: an entry was pushed, undone, redone or dropped. MainWindow
    #: hangs the dirty flag and the Edit-menu text off THIS rather than off the
    #: individual signals, which is the real fix for "a mask resize never marked
    #: the project dirty": ``eventEdited`` reached nothing, and now every mutation
    #: — announced or not — reaches one signal.
    changed = Signal()

    #: ``(label, summary)`` for a step that was just recorded, where ``summary``
    #: counts what the REPLACED table held (``n_events`` / ``n_rejected`` /
    #: ``n_added``). Undo makes a destructive click recoverable; this is how the
    #: click also becomes VISIBLE, because a user who does not know 2 rejections
    #: just vanished has no reason to reach for Ctrl+Z.
    stepRecorded = Signal(str, dict)

    #: ``(doc_id, n_steps)`` — steps that were thrown away to stay under :data:`MAX_BYTES`.
    #: An eviction the user can NOTICE has to be said out loud: it is the one way a step
    #: leaves this stack that the user neither asked for nor can see coming, and the Edit
    #: menu's "nothing to undo on this event table" is otherwise indistinguishable from
    #: "you never edited this table". :meth:`evicted` is the same fact, asked per key.
    stepsEvicted = Signal(str, int)

    #: ``(doc_id, n_steps)`` — steps dropped from the far end of one table's stack to stay
    #: under :data:`MAX_DEPTH`. THE SAME LOSS AS :attr:`stepsEvicted`, AT THE OTHER
    #: CEILING, and it used to be the silent one: only the byte path counted and announced
    #: what it dropped, so a 50-gesture curation sprint let Ctrl+Z walk back exactly 30
    #: steps and then showed "nothing to undo on this event table" — the wording this
    #: module's own :meth:`evicted` contract reserves for a table whose history the app
    #: never touched — while 20 events stayed rejected. A separate signal rather than a
    #: reason argument on ``stepsEvicted`` because the two ceilings refuse by different
    #: names (a limit refuses by its own name): the byte handler talks about the memory
    #: ceiling, and MiB would be a lie here.
    stepsDepthDropped = Signal(str, int)

    def __init__(self, session, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._undo: list[_Entry] = []
        self._redo: list[_Entry] = []
        #: Steps dropped from each key to stay under :data:`MAX_BYTES`, ever. Never
        #: decremented: history that was thrown away stays thrown away, and this is the
        #: only record that it existed.
        self._evicted: dict[EventKey, int] = {}
        #: Steps dropped from each key to stay under :data:`MAX_DEPTH`, ever — the depth
        #: ceiling's own ledger, same never-decremented rule as :attr:`_evicted`, kept
        #: apart from it so the menu can name the ceiling that actually bound.
        self._depth_dropped: dict[EventKey, int] = {}
        #: The key of the last step PUSHED. MainWindow reads it to scope the cross-stack
        #: redo invalidation to the recording the step belongs to — see
        #: :meth:`clear_redo` and :meth:`~nanoscope.gui.main_window.MainWindow._note_step`.
        self.last_step_key: EventKey | None = None
        #: Did the last step PUSHED put a new run on its table (a detection or a re-measure)?
        #: MainWindow folds the form steps that led to a run into its press on this (design
        #: §3.6), never on the step's label.
        self.last_step_run = False
        #: The last COMMITTED state of each live table. Everything is measured
        #: against this; a key absent from it has never been seen.
        self._pristine: dict[EventKey, AnalysisResult] = {}
        #: A WEAK reference to the live ``EventSet`` each pristine state was taken from.
        #: The five in-place writers mutate that very object; the six installers replace
        #: it. Nothing else in the app distinguishes the two, and the distinction is what
        #: separates "the user curated" from "a detection replaced the table" when the
        #: two frames look alike. Weak so the history never keeps a discarded table alive.
        self._seen: dict[EventKey, weakref.ReferenceType] = {}
        self._suspended = False
        self._applying = False

    def _remember(self, key: EventKey, res: AnalysisResult) -> None:
        """Make ``res`` the committed state of ``key``: the snapshot AND the identity."""
        self._pristine[key] = _snapshot(res)
        try:
            self._seen[key] = weakref.ref(res.payload)
        except TypeError:                      # pragma: no cover - defensive
            self._seen.pop(key, None)

    def _was_replaced(self, key: EventKey, res: AnalysisResult) -> bool:
        """Is this a NEW ``EventSet`` object under ``key``, rather than the edited one?"""
        ref = self._seen.get(key)
        return ref is None or ref() is not res.payload

    # -- recording ------------------------------------------------------------------------------
    def set_suspended(self, on: bool) -> None:
        """While suspended, checkpoints REFRESH the pristine state but push nothing.

        A project restore rebuilds every table from disk. Those are reconstructions,
        not edits, and an undo stack that offered to walk back through them would
        offer to destroy the project the user just opened.
        """
        self._suspended = bool(on)

    def _live(self, key: EventKey) -> AnalysisResult | None:
        from ..analysis.events import EventSet

        doc_id, channel, module_id = key
        if not self._session.has_document(doc_id):
            return None
        res = self._session.results.get(doc_id, channel, module_id)
        if res is None or not isinstance(res.payload, EventSet):
            return None
        return res

    def checkpoint(self, key: EventKey, *, label: str | None = None) -> bool:
        """Commit whatever has happened to ``key`` since the last checkpoint.

        Returns True if a step was pushed. Calling it when nothing changed is free
        and pushes nothing — which is what lets MainWindow wire it to four
        redundant signals without inventing steps.
        """
        res = self._live(key)
        if res is None:
            return False
        prior = self._pristine.get(key)
        same_frame = prior is not None and _frames_equal(prior.payload.df, res.payload.df)
        # THE WHOLE RESULT, never three of its fields. A re-measure restamps
        # ``baseline_fingerprint`` and ``warnings``; a re-detection over the visible view
        # leaves the frame byte-identical and moves the analysed span and
        # ``analyzed_duration_s`` in PROVENANCE, which is the denominator of
        # ``EventSet.rate_hz``. The old three-field test saw the first and was blind to
        # the second, so it returned False here -- pushing no step AND leaving
        # ``_pristine`` (and ``_seen``) holding the previous run, which the next ordinary
        # curation click then quietly restored. See _results_equal.
        if prior is not None and _results_equal(prior, res):
            return False

        replaced = self._was_replaced(key, res)
        if prior is None or self._suspended or self._applying:
            # No previous state to return to (a first detection), or a
            # reconstruction: establish the baseline silently.
            self._remember(key, res)
            return False

        # ``describe_change`` sees only the two frames, and the two operations that
        # matter most are indistinguishable there. Pressing Detect a second time with
        # the SAME settings yields a byte-identical table whose accept/reject column has
        # been reset — which reads as "Curate 4 events", the exact opposite of what
        # happened, and (because MainWindow announces a replacement by its label) it is
        # what kept C14's loss invisible on the commonest re-detect. A re-measure and a
        # re-detection landing on the same spans are the same problem one level down.
        # The RESULT object separates all of them, so the last word belongs to it.
        detection = replaced and (_params_moved_outside_baseline(prior, res)
                                  or _is_carried_rerun(prior, res)
                                  or _curation_was_reset(prior.payload.df, res.payload.df)
                                  or (len(prior.payload.df) != len(res.payload.df)
                                      and not _values_equal(prior.params, res.params)))
        if label:
            text = label
        elif detection:
            text = "Re-run detection"
        elif same_frame:
            # The rows did not move at all: only a stamp or the provenance did. Naming a
            # detection here would be a lie about the most destructive button in the app
            # -- UNLESS the EventSet object itself was replaced, which no in-place editor
            # can do. A re-run that happens to land on the same rows is still a re-run,
            # and after C-3 it is the case that actually reaches this branch.
            if _looks_like_remeasure(prior, res):
                text = "Re-measure depths"
            elif replaced:
                text = "Re-run detection"
            else:
                text = "Edit events"
        else:
            text = describe_change(prior.payload.df, res.payload.df)
            if (text.startswith("Edit ") and "amplitude" in text
                    and replaced and _looks_like_remeasure(prior, res)):
                # A new baseline can leave every detector scalar identical while
                # invalidating optional choices; name the operation, not its symptom.
                text = "Re-measure depths"
            if text == "Re-measure depths" and not _looks_like_remeasure(prior, res):
                text = "Re-run detection"
        # A RUN, whatever the frames say: a Detect after a Baseline-only edit moves no detector
        # setting and can move a few spans, so the diff names it "Resize 3 event masks", and
        # the fold of design §3.6 keyed on the name left its setting for a second press
        # (S10-UNDO2). The run's own stamp says what it was, and a new stamp is a detection
        # even when only the baseline block moved; a re-measure keeps the stamp and is told
        # by its baseline block.
        stamped = replaced and _new_run_stamp(prior, res)
        remeasure = replaced and not stamped and _looks_like_remeasure(prior, res)
        run = detection or remeasure or stamped
        if not label:
            if stamped:
                text = "Re-run detection"
            elif run and text not in _RUN_LABELS:
                text = "Re-measure depths" if remeasure else "Re-run detection"
            run = run or text in _RUN_LABELS
        self._undo.append(_Entry(key, prior, text, _sizeof(prior)))
        self._redo = [e for e in self._redo if e.key != key]
        self._remember(key, res)
        self.last_step_key = key
        self.last_step_run = run
        self._trim()
        self.stepRecorded.emit(text, _summarise(prior.payload.df))
        self.changed.emit()
        return True

    def checkpoint_all(self) -> bool:
        """Commit every table that moved without announcing itself.

        This is the safety net that makes the design hold: a mutation path that
        forgets to tell MainWindow still lands on the stack the next time anything
        asks, it just lands with the derived label instead of a declared one. A
        missed door costs a word, never a row.

        The callers are the four moments at which "what happened last" has to be
        true: an undo or redo (:meth:`_move`), the Edit menu opening, a project
        SAVE and a QUIT. The last two are the ones that decide whether work is
        lost, so an unannounced edit reaches the dirty flag — and therefore the
        save prompt — through here.
        """
        pushed = False
        for key in list(self._pristine):
            pushed |= self.checkpoint(key)
        return pushed

    # -- querying -------------------------------------------------------------------------------
    def can_undo(self, key: EventKey | None) -> bool:
        return key is not None and any(e.key == key for e in self._undo)

    def can_redo(self, key: EventKey | None) -> bool:
        return key is not None and any(e.key == key for e in self._redo)

    def undo_label(self, key: EventKey | None) -> str:
        entry = self._last(self._undo, key)
        return entry.label if entry else ""

    def redo_label(self, key: EventKey | None) -> str:
        entry = self._last(self._redo, key)
        return entry.label if entry else ""

    def depth(self) -> int:
        """Undoable steps held, across every target."""
        return len(self._undo)

    def nbytes(self) -> int:
        """Bytes held by undo + redo together."""
        return sum(e.nbytes for e in self._undo) + sum(e.nbytes for e in self._redo)

    def has_any(self) -> bool:
        return bool(self._undo or self._redo)

    def siblings_with_history(self, key: EventKey | None) -> dict[str, int]:
        """Other detectors on the SAME (doc, channel) that hold undoable steps.

        ``{module_id: n_steps}``, never including ``key`` itself.

        A stack is per (recording, channel, DETECTOR) — deliberately, so an ML run and a
        Schmitt run on one channel do not undo into each other. The cost is that running
        the ML detector makes Edit ▸ Undo say "nothing to undo on this event table", which
        is true and reads as "your curation is gone": the Schmitt history is intact and
        one detector switch away, and nothing said so. This is what lets the menu say
        where it went instead of only that it is not here.
        """
        if key is None:
            return {}
        doc_id, channel, module_id = key
        out: dict[str, int] = {}
        for entry in self._undo:
            e_doc, e_ch, e_mod = entry.key
            if e_doc == doc_id and e_ch == channel and e_mod != module_id:
                out[e_mod] = out.get(e_mod, 0) + 1
        return out

    @staticmethod
    def _last(stack: list[_Entry], key: EventKey | None) -> _Entry | None:
        if key is None:
            return None
        for entry in reversed(stack):
            if entry.key == key:
                return entry
        return None

    # -- applying -------------------------------------------------------------------------------
    def undo(self, key: EventKey | None) -> str:
        """Restore ``key``'s previous state. Returns the label undone, or ``''``."""
        return self._move(self._undo, self._redo, key)

    def redo(self, key: EventKey | None) -> str:
        """Re-apply the state ``undo`` took away. Returns the label, or ``''``."""
        return self._move(self._redo, self._undo, key)

    def _move(self, source: list[_Entry], sink: list[_Entry], key: EventKey | None) -> str:
        # Anything edited in place and never announced is committed FIRST, so Ctrl+Z
        # after an unannounced edit undoes that edit rather than jumping past it.
        self.checkpoint_all()
        entry = self._last(source, key)
        if entry is None:
            return ""
        live = self._live(entry.key)
        if live is None:
            # The recording (or the result) went away under the stack. Drop the whole
            # branch rather than install a table onto a document that is not open.
            self._drop(lambda e: e.key == entry.key)
            self.changed.emit()
            return ""
        source.remove(entry)
        sink.append(_Entry(entry.key, _snapshot(live), entry.label, _sizeof(live)))

        self._applying = True
        try:
            # Through the SAME door every other installation uses: add_result emits
            # resultsChanged, which is what the table, the span overlays, the summary,
            # the score, the Event Inspector, the plots and the workflow rail all hang
            # off. An undo that repainted them itself would be a second, divergent
            # refresh path.
            self._session.add_result(_snapshot(entry.result))
            # Commit the object the SESSION now holds, not the one we handed it: the
            # committed identity has to be the live payload, or the next ordinary
            # curation click would look like a wholesale replacement.
            installed = self._live(entry.key)
            self._remember(entry.key, installed if installed is not None else entry.result)
        finally:
            self._applying = False
        self._trim()
        self.changed.emit()
        return entry.label

    def is_applying(self) -> bool:
        """True while an undo/redo is installing — MainWindow's status line uses it
        to keep an undo from announcing itself as a fresh destructive edit."""
        return self._applying

    # -- lifetime -------------------------------------------------------------------------------
    def drop_doc(self, doc_id: str) -> None:
        """Forget everything about a closed recording, including its pristine state."""
        before = (len(self._undo), len(self._redo), len(self._pristine))
        self._drop(lambda e: e.key[0] == doc_id)
        for key in [k for k in self._pristine if k[0] == doc_id]:
            del self._pristine[key]
            self._seen.pop(key, None)
        # The eviction record goes with the recording it describes: the menu would
        # otherwise report a dropped history for a key that no longer exists, and a
        # RE-opened file gets fresh doc_ids anyway (core/identity.py).
        for key in [k for k in self._evicted if k[0] == doc_id]:
            del self._evicted[key]
        for key in [k for k in self._depth_dropped if k[0] == doc_id]:
            del self._depth_dropped[key]
        if self.last_step_key is not None and self.last_step_key[0] == doc_id:
            self.last_step_key = None
        if before != (len(self._undo), len(self._redo), len(self._pristine)):
            self.changed.emit()

    def clear(self) -> None:
        """Forget everything (a project open, a session reset)."""
        had = self.has_any() or bool(self._pristine)
        self._undo.clear()
        self._redo.clear()
        self._pristine.clear()
        self._seen.clear()
        self._evicted.clear()
        self._depth_dropped.clear()
        self.last_step_key = None
        if had:
            self.changed.emit()

    def clear_redo(self, doc_ids: Iterable[str] | None = None) -> None:
        """End the redo branch of the recordings named by ``doc_ids``, silently.

        THE CROSS-STACK HALF OF "a new step ends the redo branch", and the scope is the
        RECORDING. :meth:`checkpoint` already prunes this stack's own key (``e.key != key``);
        what it cannot see is a step pushed on the OTHER stack — the hand-measurement table —
        which invalidates this one's branch too because there is one Edit ▸ Redo. Measured:
        curate 2 rejections → re-detect (wiped) → Ctrl+Z (restored) → add ONE hand
        measurement → Edit still reads *"Redo Re-run detection"*, press it, and the 2
        rejections are destroyed again, with the user's attention on the other table. The
        stale entry survives arbitrarily many legitimate undo/redo cycles on the other stack;
        it is never consumed or aged out, so hiding it (by deleting ``_redo_target``'s
        fallback) would not be ending it.

        ``doc_ids=None`` DROPS EVERYTHING, and that is what this method used to do
        unconditionally — which was a second data-loss bug of exactly the shape the first
        one had. That measurement is real but it is real WITHIN ONE RECORDING, while the
        wipe was session-wide: reject 2 events on recording A, Ctrl+Z (Edit reads
        *"Redo Curate event"*, enabled), reject one event on recording B, come back to A —
        *"Redo — nothing to redo on this event table"*, and the undone rejection is in no
        stack at all. Nothing in the app can put it back, nothing announced it, and it
        contradicts both promises the guide makes about history (it is per event table;
        closing a recording is the only one-way door). So the caller names the recordings
        its step actually affects, and ``None`` is reserved for the caller that genuinely
        cannot — where dropping the branch is the conservative answer, not the default one.
        An EMPTY collection means "this step ends no branch on this stack", which is the
        ordinary case for a step pushed on this very stack.

        SILENT: the caller is mid-checkpoint and emits ``changed`` itself an instant later.
        Emitting here too would mark the project dirty for the other stack's bookkeeping.
        """
        if doc_ids is None:
            self._redo.clear()
            return
        doomed = {str(d) for d in doc_ids}
        if not doomed:
            return
        self._redo = [e for e in self._redo if e.key[0] not in doomed]

    def evicted(self, key: EventKey | None) -> int:
        """Steps thrown away from ``key`` to stay under :data:`MAX_BYTES`, ever.

        Read by the Edit menu so that "nothing to undo on this event table" is never shown
        for a table whose history the app itself discarded — the two states look identical
        to the user and mean opposite things.
        """
        return int(self._evicted.get(key, 0)) if key is not None else 0

    def depth_dropped(self, key: EventKey | None) -> int:
        """Steps thrown away from ``key`` to stay under :data:`MAX_DEPTH`, ever.

        The depth ceiling's half of the :meth:`evicted` contract: the same never-edited
        wording must not cover a loss at EITHER ceiling, and this ceiling is the one an
        ordinary curation sprint reaches (31 checkbox clicks suffice). Separate from
        :meth:`evicted` so the menu can name the right ceiling.
        """
        return int(self._depth_dropped.get(key, 0)) if key is not None else 0

    def _drop(self, predicate) -> None:
        self._undo = [e for e in self._undo if not predicate(e)]
        self._redo = [e for e in self._redo if not predicate(e)]

    def _trim(self) -> None:
        """Enforce :data:`MAX_DEPTH` **per event table** and :data:`MAX_BYTES` for the session.

        The two bounds are different KINDS of bound and so are enforced differently:

        * ``MAX_DEPTH`` is how far back one table can be walked. It is per key, and a key
          over budget loses its OWN oldest step — the far end of its own history, which is
          the contract stated at :data:`MAX_BYTES`. Curation on one recording can no longer
          evict another recording's history (see :data:`MAX_DEPTH` for the measurement).
        * ``MAX_BYTES`` is RAM, which is shared, so it stays session-wide. The entry dropped
          is the oldest of the key holding the MOST bytes — the table that is actually
          spending the budget — rather than the globally oldest, which was how a 38 MiB
          table's third step used to evict a small recording's only one. Every such drop is
          counted in :attr:`_evicted` and announced through :attr:`stepsEvicted`, because it
          is a step leaving the stack for a reason the user neither asked for nor can see.
        """
        counts: dict[EventKey, int] = {}
        for entry in self._undo:
            counts[entry.key] = counts.get(entry.key, 0) + 1
        over = {k: n - MAX_DEPTH for k, n in counts.items() if n > MAX_DEPTH}
        if over:
            # Count and ANNOUNCE what the depth ceiling takes, exactly as `_evict` does
            # for the byte ceiling. Dropping without either is how a 31st curation click
            # silently began discarding steps while the menu kept the never-edited
            # wording — the outcome the MAX_DEPTH docstring itself calls "a data-loss
            # bug, not a policy" (there for the session-wide victim; the silence is the
            # same defect for the rightful one).
            dropped = dict(over)                     # per-key count before `over` is spent
            kept: list[_Entry] = []
            for entry in self._undo:                 # oldest first: the far end goes
                if over.get(entry.key, 0) > 0:
                    over[entry.key] -= 1
                    continue
                kept.append(entry)
            self._undo = kept
            for key, n in dropped.items():
                self._depth_dropped[key] = self._depth_dropped.get(key, 0) + n
                self.stepsDepthDropped.emit(key[0], self._depth_dropped[key])

        while self.nbytes() > MAX_BYTES and self._undo:
            self._evict(self._undo)
        while self.nbytes() > MAX_BYTES and self._redo:
            self._evict(self._redo)

    def _evict(self, stack: list[_Entry]) -> None:
        """Drop ONE entry for the byte ceiling: the oldest of the hungriest key, and say so."""
        weight: dict[EventKey, int] = {}
        for entry in stack:
            weight[entry.key] = weight.get(entry.key, 0) + entry.nbytes
        victim_key = max(weight, key=lambda k: (weight[k], k))
        index = next(i for i, e in enumerate(stack) if e.key == victim_key)
        del stack[index]
        self._evicted[victim_key] = self._evicted.get(victim_key, 0) + 1
        self.stepsEvicted.emit(victim_key[0], self._evicted[victim_key])

    # -- what a close would destroy -------------------------------------------------------------
    def stake_for_doc(self, doc_id: str) -> dict[str, Any]:
        """What closing ``doc_id`` would take with it, counted from the LIVE tables.

        Read from the session rather than from the stack, because the stack holds
        what a close would make unreachable while the tables hold what it would
        destroy — and the destructive number is the one a confirmation has to name.
        """
        from ..analysis.events import EventSet

        out = {"n_detections": 0, "n_events": 0, "n_accepted": 0,
               "n_rejected": 0, "n_added": 0}
        if not self._session.has_document(doc_id):
            return out
        for res in self._session.results.for_doc(doc_id):
            if not isinstance(res.payload, EventSet):
                continue
            df = res.payload.df
            out["n_detections"] += 1
            out["n_events"] += len(df)
            if len(df) and "accepted" in df.columns:
                acc = df["accepted"].to_numpy(dtype=bool)
                out["n_accepted"] += int(acc.sum())
                out["n_rejected"] += int((~acc).sum())
            if len(df) and "added" in df.columns:
                out["n_added"] += int(df["added"].to_numpy(dtype=bool).sum())
        return out


# ================================================================================================
# The hand-measurement table — the same rule, on the one table an EventSet cannot hold
# ================================================================================================

class MeasurementHistory(QObject):
    """Undo/redo over the MANUAL MEASUREMENT table. :class:`EventHistory`'s rule, applied
    to the one table that is not an ``EventSet``.

    Why a sibling class and not a sixth key on :class:`EventHistory`
    ---------------------------------------------------------------
    Everything that made the event stack right applies here unchanged — a **pristine deep
    copy that is compared**, so a mutation need not announce itself to be captured; a
    derived label; a bounded stack. What does NOT apply is the payload: every step of
    ``EventHistory`` goes through ``AnalysisResult``/``EventSet`` — the frozen 20-column
    schema, ``ensure_added_column``, ``describe_change``'s geometry/measured column groups,
    ``session.add_result`` as the install door. A measurement row shares none of that.
    Giving it a reserved key would have meant branching on that key in five places inside
    the module this project has already had to repair four defects in, which is the exact
    "one more special case" shape the structural rule here forbids. So the RULE is reused
    and the payload is not.

    What it closes
    --------------
    *Remove selected* was the last destructive click in the application that nothing could
    take back. It warned — "you drew these on the trace by hand, nothing in the app can
    re-derive them" — which was honest and no help at all to somebody who had just deleted
    the wrong four rows. It is now a step like any other, and the warning says so.

    Ownership
    ---------
    Driven entirely by :class:`~nanoscope.gui.main_window.MainWindow`, exactly like
    ``EventHistory``: the panel that owns the table knows nothing about this class. The
    frame is read and written through two callables the wirer supplies, so this module
    stays free of Qt widgets and of the measurement schema.
    """

    #: The stack moved. MainWindow hangs the dirty flag and the Edit-menu text off this,
    #: the same way it does for ``EventHistory.changed``.
    changed = Signal()

    #: A STEP was pushed — the label of it. ``EventHistory``'s sibling, and it exists for
    #: the same structural reason: MainWindow has to learn that this table just gained a
    #: step WHEREVER the push came from, not only from the one wrapper it wrote. Two pushes
    #: bypass that wrapper today (``_move``'s leading ``checkpoint()``, and the restore's
    #: adoption call), and a new step is what invalidates the other stack's redo branch —
    #: so the announcement has to come from the push itself. See
    #: :meth:`~nanoscope.gui.main_window.MainWindow._note_step`.
    stepRecorded = Signal(str)

    def __init__(self, get_df, set_df, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._get = get_df
        self._set = set_df
        #: ``(frame, label, docs)``. ``docs`` is the set of ``doc_key`` values whose rows the
        #: step moved, or ``None`` when the frame could not answer — see :func:`_docs_touched`.
        #: It travels with the entry into the redo stack, because it describes the STEP and
        #: the step is what a redo would re-apply.
        self._undo: list[tuple[pd.DataFrame, str, frozenset[str] | None]] = []
        self._redo: list[tuple[pd.DataFrame, str, frozenset[str] | None]] = []
        self._pristine: pd.DataFrame | None = None
        self._suspended = False
        self._applying = False
        #: Which recordings the last step PUSHED touched (``None`` = could not tell).
        #: MainWindow reads it to scope the cross-stack redo invalidation — see
        #: :meth:`~nanoscope.gui.main_window.MainWindow._note_step`.
        self.last_step_docs: frozenset[str] | None = None
        #: Monotonic tick stamped on every step pushed. MainWindow compares it with
        #: ``EventHistory``'s to decide which stack a plain Ctrl+Z should act on — see
        #: :meth:`~nanoscope.gui.main_window.MainWindow._undo_events`.
        self._seq = 0
        self.last_step_seq = 0

    # -- recording ------------------------------------------------------------------------------
    def set_suspended(self, on: bool) -> None:
        """While suspended, checkpoints refresh the pristine copy but push nothing — a
        project restore rebuilds this table from disk and those are reconstructions."""
        self._suspended = bool(on)

    def _read(self) -> pd.DataFrame | None:
        try:
            df = self._get()
        except Exception:                      # noqa: BLE001 - reading must never sink a save
            return None
        return df if isinstance(df, pd.DataFrame) else None

    def checkpoint(self, *, label: str | None = None) -> bool:
        """Commit whatever has happened to the table since the last checkpoint.

        Free when nothing changed, so it can be wired to every announcement AND called
        from the four "what happened last must be true" moments, without inventing steps.
        """
        df = self._read()
        if df is None:
            return False
        prior = self._pristine
        if prior is not None and _frames_equal(prior, df):
            return False
        if prior is None or self._suspended or self._applying:
            self._pristine = df.copy(deep=True)
            return False
        text = label or _describe_measurement_change(prior, df)
        docs = _docs_touched(prior, df)
        self._undo.append((prior, text, docs))
        self._redo.clear()
        self.last_step_docs = docs
        self._pristine = df.copy(deep=True)
        self._trim()
        self._seq += 1
        self.last_step_seq = self._seq
        self.stepRecorded.emit(text)
        self.changed.emit()
        return True

    # -- querying -------------------------------------------------------------------------------
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_label(self) -> str:
        return self._undo[-1][1] if self._undo else ""

    def redo_label(self) -> str:
        return self._redo[-1][1] if self._redo else ""

    def depth(self) -> int:
        return len(self._undo)

    def nbytes(self) -> int:
        return sum(int(df.memory_usage(deep=True).sum())
                   for df, _, _ in self._undo + self._redo)

    def is_applying(self) -> bool:
        return self._applying

    # -- applying -------------------------------------------------------------------------------
    def undo(self) -> str:
        return self._move(self._undo, self._redo)

    def redo(self) -> str:
        return self._move(self._redo, self._undo)

    def _move(self, source: list, sink: list) -> str:
        # Anything edited without announcing itself is committed FIRST, so Ctrl+Z after an
        # unannounced edit undoes that edit rather than jumping past it.
        self.checkpoint()
        if not source:
            return ""
        frame, label, docs = source.pop()
        live = self._read()
        if live is None:
            return ""
        sink.append((live.copy(deep=True), label, docs))
        self._applying = True
        try:
            self._set(frame.copy(deep=True))
            settled = self._read()
            self._pristine = (settled if settled is not None else frame).copy(deep=True)
        finally:
            self._applying = False
        self._trim()
        self.changed.emit()
        return label

    # -- lifetime -------------------------------------------------------------------------------
    def clear(self) -> None:
        """Forget everything — the session was reset (a project opened)."""
        self._undo.clear()
        self._redo.clear()
        self._pristine = None
        self.last_step_seq = 0
        self.last_step_docs = None

    def clear_redo(self, doc_keys: Iterable[str] | None = None) -> tuple[bool, ...]:
        """End this table's redo branch for the recordings named by ``doc_keys``, silently.

        :meth:`EventHistory.clear_redo`'s twin — read that one for why a step on the OTHER
        stack has to reach this list, and for why the scope is the RECORDING rather than the
        session. This table is one table for the whole session, but its rows name the
        recording they were measured on (``doc_key``; see
        :mod:`nanoscope.analysis.measure`), so a step on recording B does not end a redo
        branch that would only ever rewrite recording A's rows.

        ``None`` drops everything and is the conservative answer for a caller that cannot
        name the recordings; an entry whose own scope is unknown (a frame with no
        ``doc_key``, or one this module could not diff) is dropped by any scoped call for
        the same reason — an unknown scope is not a narrow one.

        Returns, for each redo step held before the call (oldest first), whether it is still
        held: the journal's entries for this table name no recording, so MainWindow matches
        them to the steps by position to drop exactly the ones ended here.
        """
        if doc_keys is None:
            kept = tuple(False for _ in self._redo)
            self._redo.clear()
            return kept
        doomed = {str(k) for k in doc_keys}
        if not doomed:
            return tuple(True for _ in self._redo)
        kept = tuple(e[2] is not None and not (e[2] & doomed) for e in self._redo)
        self._redo = [e for e, keep in zip(self._redo, kept) if keep]
        return kept

    def _trim(self) -> None:
        # One table, so ``MAX_DEPTH`` is already per-table here and the far end is this
        # table's own far end — the cross-key eviction ``EventHistory._trim`` had to grow
        # out of cannot arise on a stack with one key.
        while len(self._undo) > MAX_DEPTH:
            del self._undo[0]
        while self.nbytes() > MAX_BYTES and self._undo:
            del self._undo[0]
        while self.nbytes() > MAX_BYTES and self._redo:
            del self._redo[0]


#: Stand-in for a missing cell inside a row key. Python 3.10 made ``hash(nan)`` depend on
#: the object's identity, and ``pandas`` builds fresh float objects on every
#: ``itertuples``, so an UNCHANGED row holding one NaN would otherwise hash differently in
#: the two frames and read as changed — which, in :func:`_docs_touched`, means naming
#: recordings the step never touched and ending their redo branches for nothing.
_NA_KEY = "\x00<na>"


def _row_key(row: tuple) -> tuple:
    """A hashable, NaN-stable key for one measurement row. See :data:`_NA_KEY`."""
    out = []
    for v in row:
        try:
            out.append(_NA_KEY if pd.isna(v) else v)
        except (TypeError, ValueError):        # arrays and other non-scalars
            out.append(repr(v))
    return tuple(out)


def _docs_touched(before: pd.DataFrame, after: pd.DataFrame) -> frozenset[str] | None:
    """Which RECORDINGS a measurement step moved rows of — ``None`` when it cannot be told.

    The hand-measurement table is one table for the whole session, but every row names the
    recording it was measured on (``doc_key``, the persistent key — see
    :mod:`nanoscope.analysis.measure`). That is what lets the cross-stack redo invalidation
    be scoped to a recording instead of wiping every recording's branch, which was a
    data-loss bug in its own right (see :meth:`EventHistory.clear_redo`).

    Rows are compared as whole tuples and by MULTIPLICITY, so a step that only edits a cell
    reports both the row that left and the row that arrived — the same recording either way.
    ``None`` means "unknown", which every caller must treat as "could be any of them":
    a frame with no ``doc_key`` column (a table written before that column existed), a row
    whose ``doc_key`` is empty, or any comparison this function could not complete. Being
    unable to answer is never allowed to look like a narrow answer.
    """
    if "doc_key" not in before.columns or "doc_key" not in after.columns:
        return None
    try:
        col = list(after.columns)
        if list(before.columns) != col:
            return None
        pos = col.index("doc_key")
        counts: dict[tuple, int] = {}
        for frame, sign in ((before, 1), (after, -1)):
            for row in frame.itertuples(index=False, name=None):
                key = _row_key(row)
                counts[key] = counts.get(key, 0) + sign
        docs = {str(row[pos]) for row, n in counts.items() if n}
    except Exception:                          # noqa: BLE001 - unknown, never a narrow answer
        return None
    if any(not d or d == "nan" for d in docs):
        return None
    return frozenset(docs)


def _describe_measurement_change(before: pd.DataFrame, after: pd.DataFrame) -> str:
    """Name what turned ``before`` into ``after``. Derived, never declared — same
    principle as :func:`describe_change`, and a much smaller table of operations: this
    list is only ever added to, deleted from, restored wholesale, or annotated."""
    n0, n1 = len(before), len(after)
    if n1 < n0:
        gone = n0 - n1
        return ("Delete measurement" if gone == 1
                else f"Delete {gone} measurements")
    if n1 > n0:
        added = n1 - n0
        return "Add measurement" if added == 1 else f"Add {added} measurements"
    if n0 == 0:
        return "Edit measurements"
    return "Edit measurement"


# ================================================================================================
# The analysis settings — the same rule a third time, over values that are not tables
# ================================================================================================

#: A settings scope: a hashable tuple whose element ``[1]`` is the ``doc_id`` it belongs to.
#: MainWindow builds four shapes:
#:
#: * ``("form", doc_id, channel, module_id)`` — the Detection, ML and Baseline forms of one
#:   event table, ``module_id`` being ``None`` until that recording and channel has a table;
#: * ``("calibration", doc_id, channel)``;
#: * ``("time_base", doc_id)``;
#: * ``("acquisition", doc_id, channel)`` — the acquisition-bandwidth override.
#:
#: :class:`SettingsHistory` reads ``[1]``, to drop a closed recording's scopes, and nothing
#: else. A fifth kind of setting is therefore a new tuple in MainWindow, never an edit here.
SettingsScope = tuple

#: The label of a step nobody named. Only MainWindow's ``describe`` knows what a setting is
#: called; a missing or failing one costs the menu this wording, never the step.
_SETTINGS_LABEL = "Change settings"


def _settings_equal(a: Any, b: Any) -> bool:
    """Do two settings states hold the same values? Exact, with four allowances.

    Each allowance is a way the SAME setting can come back as a different Python value, and
    each would otherwise push a step for an edit nobody made — the phantom
    :func:`_column_equal` and ``_OPTIONAL_COLUMNS`` were both written against:

    * **NaN equals NaN.** Two reads produce two NaN objects and ``nan != nan``, so one NaN
      anywhere in a state would read as a change on every checkpoint.
    * **A list equals a tuple with equal items.** JSON has no tuple, so a value that last
      passed through a protocol or a project comes back in the other container.
    * **A numpy value is the Python value it holds:** ``np.float64(5.0)`` equals ``5.0``,
      and an array equals the nested list it holds.
    * **Containers compare item by item, to any depth:** dicts by key, lists and tuples by
      position, and a dataclass field by field (same type required; a ``compare=False``
      field is skipped, as the type's own ``==`` skips it), so a NaN inside a
      ``ChannelCalibration`` is no more a change than a NaN in a dict.

    Everything else is ``==``: ``5`` and ``5.0`` are one value, and ``None`` is not ``NaN``,
    ``0`` or ``""``.

    A COMPARISON THAT CANNOT ANSWER READS AS A CHANGE, the opposite of :func:`_values_equal`,
    because a wrong answer costs the opposite thing here. There the frame is compared on its
    own, and a stamp misread as unchanged costs a coarser label. Here the state IS the
    payload: an edit misread as unchanged leaves the pristine copy behind, the next Undo
    writes the older state over the edit, and no redo leads back to it. The price of this
    direction is a phantom step on every checkpoint for as long as such a value sits in a
    state, which is why every type a real state holds — the forms' dicts,
    :class:`~nanoscope.core.calibration.ChannelCalibration`,
    :class:`~nanoscope.core.sample_rate.SampleRateDeclaration`,
    :class:`~nanoscope.core.acquisition.AcquisitionBandwidth` — is answered above and never
    reaches it.
    """
    try:
        if a is b:
            return True
        if isinstance(a, np.ndarray):
            a = a.tolist()
        elif isinstance(a, np.generic):
            a = a.item()
        if isinstance(b, np.ndarray):
            b = b.tolist()
        elif isinstance(b, np.generic):
            b = b.item()
        if isinstance(a, float) and isinstance(b, float):
            return a == b or (a != a and b != b)
        if (is_dataclass(a) and not isinstance(a, type)) or (
                is_dataclass(b) and not isinstance(b, type)):
            return type(a) is type(b) and all(
                _settings_equal(getattr(a, f.name), getattr(b, f.name))
                for f in dc_fields(a) if f.compare)
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            return (isinstance(a, Mapping) and isinstance(b, Mapping) and set(a) == set(b)
                    and all(_settings_equal(a[k], b[k]) for k in a))
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
                    and len(a) == len(b)
                    and all(_settings_equal(x, y) for x, y in zip(a, b)))
        return bool(a == b)
    except Exception:                          # noqa: BLE001 - see the docstring's last paragraph
        return False


def _detached(state: Any) -> Any:
    """A deep copy of ``state``. A reader may hand out an object it goes on editing in place,
    and a history holding that very object would silently track the edits it exists to
    remember (``_snapshot``'s reason). A value that refuses to be copied is kept as it is: a
    shared value costs the isolation, a raised error would cost the step."""
    try:
        return copy.deepcopy(state)
    except Exception:                          # noqa: BLE001 - a copy must never sink a checkpoint
        return state


def _scope_doc(scope: Any) -> Any:
    """The recording a scope belongs to: its element ``[1]``. See :data:`SettingsScope`."""
    return scope[1] if isinstance(scope, tuple) and len(scope) > 1 else None


@dataclass(frozen=True, eq=False)
class _SettingsStep:
    """One settings change: the state before it, the state after it, and its name.

    BOTH ENDS ARE KEPT, where :class:`_Entry` keeps one state and snapshots the live table on
    every move. A settings state is small (see :class:`SettingsHistory`), so a step can stay
    one immutable fact that moves between the two stacks unchanged: undo writes ``before``,
    redo writes ``after``, and a redo re-applies exactly the change its label names.
    ``eq=False`` for :class:`_Entry`'s reason: steps are identities, and two equal changes
    are two steps.
    """

    before: Any
    after: Any
    label: str


class SettingsHistory(QObject):
    """Undo/redo over the ANALYSIS SETTINGS: the Detection, ML and Baseline forms of each
    event table, a channel's calibration, a recording's time base and a channel's
    acquisition-bandwidth override (``docs/EDITING_FOUNDATION_DESIGN.md`` §3).
    :class:`EventHistory`'s rule, applied a third time.

    Why a sibling class and not a key on :class:`EventHistory`
    ---------------------------------------------------------
    :class:`MeasurementHistory`'s answer, unchanged: the rule is reused and the payload is
    not. What carries over is a **pristine copy per scope that is compared**, so a setting
    need not announce itself to be captured, and wiring ``checkpoint`` to two signals for one
    edit cannot invent a step; silent adoption the first time a scope is seen, while
    suspended and while applying; a derived label; a depth ceiling that is counted and
    announced. What does not carry over is everything ``EventHistory`` does with an
    ``AnalysisResult``: the frame diff, :func:`describe_change`, ``session.add_result`` as
    the install door. A state here is a small plain value — a dict of form fields, a
    ``ChannelCalibration`` — read and written through two callables, so this class holds no
    widget, imports nothing from ``gui`` and knows no settings schema.

    Why one stack per scope
    -----------------------
    MainWindow's journal (``_undo_journal``) already decides the ORDER across stacks, and
    each settings entry in it names a scope (design §3.5); what it asks this class is only
    what one scope can do. And redo is per scope by the owner's decision 2: a new step ends
    its own scope's redo branch and no other — not another table's form, not a calibration,
    not an event table's branch. With a stack per scope, that is the only branch a push can
    reach.

    Why no byte budget
    ------------------
    Measured on the default forms, ``DetectionPanel.params()`` plus ``ml_params()`` plus
    ``BaselinePanel.params()`` is 98 scalars, 2.3 kB pickled; two of them compare in 15 µs,
    and a full 30-step history of one form holds 107 kB. A calibration, a declaration or an
    override is a handful of scalars. The Plots window measured a whole ``PlotConfig`` at
    1,446 bytes and keeps 200 of them with no byte ceiling (``_HISTORY_DEPTH`` in
    ``gui/plots_panel.py``). :data:`MAX_BYTES` exists because an event snapshot is a
    deep-copied frame, 38.7 MiB at 100 000 events; a whole form history is under a
    thousandth of that 128 MiB ceiling. :data:`MAX_DEPTH` per scope is the only bound.

    ``None`` is not a state
    -----------------------
    ``read(scope)`` returns ``None`` to say the scope cannot be read now, and
    :meth:`checkpoint` then does nothing. A setting whose own value can be ``None`` — the
    time base's "the file's own", a channel with no bandwidth override — is therefore read
    wrapped (``{"declaration": None}``), never bare: a bare ``None`` would make "nothing
    declared" a state this class can neither adopt nor return to.

    Ownership
    ---------
    Driven entirely by :class:`~nanoscope.gui.main_window.MainWindow`, like both siblings. It
    supplies ``read(scope)``, ``write(scope, state)`` and ``describe(scope, before, after)``,
    builds the scopes, checkpoints at the moments that commit an edit, and calls
    :meth:`rebase` as each scope appears and whenever a state arrives, :meth:`drop_doc` when
    a recording closes, :meth:`clear` when a project opens and :meth:`set_suspended` around a
    restore. No form and no part of the session knows this class exists.
    """

    #: The stacks moved: a step was pushed, undone, redone, re-keyed or dropped. MainWindow
    #: hangs the dirty flag and the Edit-menu text off this, as it does for both siblings.
    changed = Signal()

    #: ``(scope, label)`` — a step was PUSHED. Emitted by the push itself, wherever it came
    #: from (an undo's leading checkpoint, :meth:`checkpoint_all` at a save), for the reason
    #: :attr:`MeasurementHistory.stepRecorded` gives: MainWindow journals each step from
    #: here, so a push that bypassed its wrapper still takes its place in the order Ctrl+Z
    #: walks.
    stepRecorded = Signal(object, str)

    #: ``(scope, total)`` — steps dropped from the far end of one scope to stay under the
    #: depth, counted over the session for that scope. :attr:`EventHistory.stepsDepthDropped`'s
    #: twin, for its reason: a ceiling that drops history silently leaves the menu saying
    #: there is nothing to undo about settings the user did change.
    stepsDepthDropped = Signal(object, int)

    def __init__(self, read, write, describe=None, parent: QObject | None = None, *,
                 depth: int = MAX_DEPTH) -> None:
        super().__init__(parent)
        self._get = read
        self._set = write
        self._describe = describe
        self._depth = int(depth)
        #: Each scope's steps, oldest first.
        self._undo: dict[SettingsScope, list[_SettingsStep]] = {}
        self._redo: dict[SettingsScope, list[_SettingsStep]] = {}
        #: The last COMMITTED state of each scope, detached. Everything is measured against
        #: this; a scope absent from it has never been seen.
        self._pristine: dict[SettingsScope, Any] = {}
        #: Steps dropped from each scope to stay under the depth, ever — the same
        #: never-decremented ledger as :attr:`EventHistory._depth_dropped`.
        self._depth_dropped: dict[SettingsScope, int] = {}
        self._suspended = False
        self._applying = False
        #: The scope of the last step PUSHED — :attr:`EventHistory.last_step_key`'s twin.
        self.last_step_scope: SettingsScope | None = None

    # -- recording ------------------------------------------------------------------------------
    def set_suspended(self, on: bool) -> None:
        """While suspended, checkpoints adopt what they read and push nothing. A project
        restore rebuilds every setting from disk, and those are reconstructions, not edits."""
        self._suspended = bool(on)

    def _read(self, scope: SettingsScope) -> Any:
        try:
            return self._get(scope)
        except Exception:                      # noqa: BLE001 - reading must never sink a save
            return None

    def _label(self, scope: SettingsScope, before: Any, after: Any) -> str:
        if self._describe is None:
            return _SETTINGS_LABEL
        try:
            text = self._describe(scope, before, after)
        except Exception:                      # noqa: BLE001 - a label must never cost the step
            return _SETTINGS_LABEL
        return text if isinstance(text, str) and text else _SETTINGS_LABEL

    def checkpoint(self, scope: SettingsScope | None, *, label: str | None = None,
                   force: bool = False) -> bool:
        """Commit whatever has happened to ``scope`` since its last checkpoint.

        Returns True if a step was pushed. Free when nothing changed, so MainWindow can call
        it after the 700 ms pause, before a run or a switch, and from every signal that might
        mean an edit, without inventing steps. ``label`` names the step when the caller knows
        better than the diff ("Recall protocol ‘HG student’"); otherwise ``describe`` does.

        ``force`` pushes a step although the state compares equal: for a change the
        comparison leaves out on purpose, which the caller knows happened (a calibration
        commit that rewrote the sidecar and moved no number). Suspended or applying, it is
        adopted like any other.
        """
        if scope is None:
            return False
        state = self._read(scope)
        if state is None:
            return False
        prior = self._pristine.get(scope)
        if prior is None:
            # First sight: nothing to return to yet, so this is the baseline, not a step.
            self._pristine[scope] = _detached(state)
            return False
        if _settings_equal(prior, state) and not force:
            return False
        if self._suspended or self._applying:
            self._pristine[scope] = _detached(state)
            return False
        after = _detached(state)
        text = label or self._label(scope, prior, after)
        self._undo.setdefault(scope, []).append(_SettingsStep(prior, after, text))
        self._redo.pop(scope, None)
        self._pristine[scope] = after
        self._trim(scope)
        self.last_step_scope = scope
        self.stepRecorded.emit(scope, text)
        self.changed.emit()
        return True

    def checkpoint_all(self) -> bool:
        """:meth:`checkpoint` every scope seen so far. The safety net for the four moments at
        which "what happened last" has to be true — an undo, the Edit menu opening, a save
        and a quit — as :meth:`EventHistory.checkpoint_all` is."""
        pushed = False
        for scope in list(self._pristine):
            pushed |= self.checkpoint(scope)
        return pushed

    def rebase(self, scope: SettingsScope) -> None:
        """Adopt what ``scope`` holds now as committed, without a step.

        For a state that ARRIVED rather than was edited, and before anything can checkpoint
        the scope again:

        * **A scope appearing**: a recording's calibration, time base and override when it
          opens, a form when it first comes on screen. First sight is a baseline, and the
          session announces those three settings only AFTER they change (opening a
          recording emits ``documentOpened`` alone), so a scope first seen from
          ``calibrationChanged`` would take the first Apply as its baseline, and that Apply
          could never be undone.
        * **A switch, a project restore or a seed from a stored run**: an event undo or redo
          that installs a run re-seeds the form, so it is a seed too. The caller checkpoints
          the outgoing scope first, switches, then rebases the incoming one (design §3.9). A
          seed left unrebased is pushed as a step by the next checkpoint, :meth:`undo`'s
          leading one included, and §3.6's single press would then undo that phantom
          instead of the settings that led to the run.

        Does nothing when the scope cannot be read.
        """
        state = self._read(scope)
        if state is not None:
            self._pristine[scope] = _detached(state)

    def drop_redo_reached(self, scope: SettingsScope) -> int:
        """Drop the redo steps of ``scope`` that the state it holds now has already reached:
        the newest-to-redo step whose ``after`` is that state, and every step redone before
        it. Returns how many went. Silent, as :meth:`EventHistory.clear_redo` is.

        For a state that ARRIVED where an undone step led, which no step recorded: a run
        launched from a form lands and re-seeds it after Ctrl+Z took that form back, and
        the step's Redo would then name a change and write what the form already holds. A
        branch that does not pass through the state is left whole."""
        steps = self._redo.get(scope)
        current = self._pristine.get(scope)
        if not steps or current is None:
            return 0
        for i in range(len(steps) - 1, -1, -1):
            if _settings_equal(steps[i].after, current):
                dropped = len(steps) - i
                del steps[i:]
                if not steps:
                    del self._redo[scope]
                return dropped
        return 0

    def forget(self, scope: SettingsScope) -> None:
        """Drop ``scope``'s pristine copy and keep its steps: the next checkpoint adopts what
        the scope then holds, silently, as on first sight."""
        self._pristine.pop(scope, None)

    # -- querying -------------------------------------------------------------------------------
    def can_undo(self, scope: SettingsScope | None) -> bool:
        return bool(self._undo.get(scope))

    def can_redo(self, scope: SettingsScope | None) -> bool:
        return bool(self._redo.get(scope))

    def undo_label(self, scope: SettingsScope | None) -> str:
        steps = self._undo.get(scope)
        return steps[-1].label if steps else ""

    def redo_label(self, scope: SettingsScope | None) -> str:
        steps = self._redo.get(scope)
        return steps[-1].label if steps else ""

    def undo_depth(self, scope: SettingsScope | None) -> int:
        return len(self._undo.get(scope) or ())

    def redo_depth(self, scope: SettingsScope | None) -> int:
        return len(self._redo.get(scope) or ())

    def depth_dropped(self, scope: SettingsScope | None) -> int:
        """Steps dropped from ``scope`` to stay under the depth, ever. Read by the Edit menu
        so "nothing to undo" is never shown for settings whose history the ceiling took —
        :meth:`EventHistory.depth_dropped`'s contract."""
        return int(self._depth_dropped.get(scope, 0))

    def has_any(self) -> bool:
        return any(self._undo.values()) or any(self._redo.values())

    def scopes_with_history(self) -> tuple[SettingsScope, ...]:
        """Every scope holding an undo or a redo step — what MainWindow's fallback asks when
        its journal has run out (design §3.5)."""
        found = dict.fromkeys(s for s, steps in self._undo.items() if steps)
        found.update(dict.fromkeys(s for s, steps in self._redo.items() if steps))
        return tuple(found)

    def is_applying(self) -> bool:
        """True while :meth:`undo` or :meth:`redo` is writing, so MainWindow can keep an undo
        from announcing itself as a fresh edit."""
        return self._applying

    # -- applying -------------------------------------------------------------------------------
    def undo(self, scope: SettingsScope | None) -> str:
        """Write ``scope``'s state from before its newest step. Returns that step's label,
        or ``''``.

        An edit not yet committed (its 700 ms pause still running) becomes a step FIRST, as
        in :meth:`EventHistory._move`, so Ctrl+Z takes back that edit rather than jumping
        past it to the one before. The write runs with recording muted, and what the app
        settles on afterwards (a writer may clamp or normalise) becomes the pristine copy, so
        the settling is never recorded as an edit of its own. Muting reaches every scope but
        lasts only as long as the write: a writer that also moves ANOTHER scope (undoing a
        declaration clamps the form's low-pass to the slower Nyquist) checkpoints that scope
        before it returns, and the knock-on is adopted; caught later, by the 700 ms pause, it
        would be recorded as the user's edit. A write that raises puts the step back where it
        was and re-raises: nothing was undone, so nothing moves.
        """
        return self._move(scope, self._undo, self._redo, forward=False)

    def redo(self, scope: SettingsScope | None) -> str:
        """Write the state ``scope``'s newest undone step led to. :meth:`undo`'s mirror; it
        clears nothing, so the rest of the branch stays redoable."""
        return self._move(scope, self._redo, self._undo, forward=True)

    def _move(self, scope: SettingsScope | None, source: dict, sink: dict, *,
              forward: bool) -> str:
        self.checkpoint(scope)
        steps = source.get(scope)
        if not steps:
            return ""
        step = steps.pop()
        state = step.after if forward else step.before
        self._applying = True
        try:
            self._set(scope, _detached(state))
        except BaseException:
            steps.append(step)                 # nothing was undone, so nothing moves
            raise
        finally:
            self._applying = False
        settled = self._read(scope)
        self._pristine[scope] = _detached(settled) if settled is not None else state
        sink.setdefault(scope, []).append(step)
        self._trim(scope)
        self.changed.emit()
        return step.label

    # -- lifetime -------------------------------------------------------------------------------
    def move_scope(self, old: SettingsScope, new: SettingsScope) -> None:
        """Give ``old``'s steps, redo branch, pristine copy and dropped count to ``new``.

        For the settings a recording's form gathers before its first event table (design
        §3.4): the form is keyed ``(..., None)`` until a detection exists, and those settings
        produced that table, so they become its history. ``old``'s steps go BEFORE any
        ``new`` already holds, being older. A redo branch is the future of the newest step,
        so ``old``'s survives only into a scope with no steps of its own; where ``new`` has
        any, ``old``'s branch is dropped, never spliced. A pristine copy ``new`` already
        holds is kept, being the newer of the two.
        """
        if old == new:
            return
        own = bool(self._undo.get(new) or self._redo.get(new))
        moved = False
        steps = self._undo.pop(old, None)
        if steps:
            self._undo[new] = steps + self._undo.get(new, [])
            moved = True
        branch = self._redo.pop(old, None)
        if branch:
            if not own:
                self._redo[new] = branch
            moved = True
        dropped = self._depth_dropped.pop(old, 0)
        if dropped:
            self._depth_dropped[new] = self._depth_dropped.get(new, 0) + dropped
            moved = True
        prior = self._pristine.pop(old, None)
        if prior is not None:
            self._pristine.setdefault(new, prior)
        if self.last_step_scope == old:
            self.last_step_scope = new
        if moved:
            self._trim(new)
            self.changed.emit()

    def drop_doc(self, doc_id: str) -> None:
        """Forget every scope of a closed recording, with its pristine copy and dropped
        count. Steps that outlived their recording could not be written anywhere, so Undo
        would name a step and then do nothing — :meth:`EventHistory.drop_doc`'s reason."""
        before = (self._n_steps(), len(self._pristine))
        for table in (self._undo, self._redo, self._pristine, self._depth_dropped):
            for scope in [s for s in table if _scope_doc(s) == doc_id]:
                del table[scope]
        if _scope_doc(self.last_step_scope) == doc_id:
            self.last_step_scope = None
        if before != (self._n_steps(), len(self._pristine)):
            self.changed.emit()

    def clear(self) -> None:
        """Forget everything (a project opened, the session was reset)."""
        had = self.has_any() or bool(self._pristine)
        self._undo.clear()
        self._redo.clear()
        self._pristine.clear()
        self._depth_dropped.clear()
        self.last_step_scope = None
        if had:
            self.changed.emit()

    def _n_steps(self) -> int:
        return (sum(len(s) for s in self._undo.values())
                + sum(len(s) for s in self._redo.values()))

    def _trim(self, scope: SettingsScope | None) -> None:
        """Hold ``scope`` to the depth: its OWN oldest steps go, counted and announced, as in
        :meth:`EventHistory._trim`. One scope's sprint never costs another scope a step."""
        steps = self._undo.get(scope)
        excess = len(steps) - self._depth if steps else 0
        if excess <= 0:
            return
        del steps[:excess]
        self._depth_dropped[scope] = self._depth_dropped.get(scope, 0) + excess
        self.stepsDepthDropped.emit(scope, self._depth_dropped[scope])
