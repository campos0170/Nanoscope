"""The Event Inspector / Browser: step through detected events one at a time.

A standalone top-level window (opened from ``Analysis ▸ Event inspector…``) that
shows ONE event's measurement waveform at a time — the recorded conditioning of
the calibrated current around ``[start_idx, end_idx)``. The baseline and equivalent
amplitude level share that domain. Opt-in amplitude curation is stored separately
from the detector's median ``depth_pA``; browsing never changes either measurement.

It is fed the CURRENT detection by :class:`~nanoscope.gui.main_window.MainWindow`
via :meth:`set_events`, which installs a REFERENCE to the very same
:class:`~nanoscope.analysis.events.EventSet` ``.df`` the
:class:`~nanoscope.gui.panels.EventTable` edits — so an accept/reject here writes
the shared frame in place, exactly like a checkbox click in the table. Curation
therefore flows straight back to the table, the trace span overlays, the summary
and the exports (they all read that one ``accepted`` column). This window never
reaches into another panel; it talks only to :class:`~nanoscope.core.session.Session`
(to slice the calibrated channel + its time axis) and announces its decisions by
signal, which MainWindow routes to the AnalysisPanel.

Signals:
    eventSelected(int): the browsed event changed (navigation / show_event).
    acceptanceChanged(int, bool): the user curated an event (event_id, accepted).
    eventEdited(int): the user resized an event's mask (event_id); its bounds + stats
        were rewritten on the shared df, so the owner refreshes table + overlays + summary.
    amplitudeEdited(int): only optional amplitude columns changed (event_id, or -1
        for a batch); refresh/save/undo without changing masks or refitting states.
    curationBatchChanged(): Accept all / Reject all rewrote the whole ``accepted`` column;
        the owner resets the table + overlays + summary from the df.
    acceptanceRefused(int): an accept was refused (the event's id, or -1 for a
        whole-frame blocked accept) and NOTHING was written — the refusal is on the
        shared ledger only. No data signal will fire, so the owner repaints the other
        two surfaces' notes off this one.
    saveProjectRequested() / saveProjectAsRequested(): ask MainWindow to save the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..analysis.characterize import recharacterize_span
from ..analysis.event_amplitude import amplitude_is_current, invalidate_event_amplitudes
from ..analysis.events import (
    AUTOMATIC_REJECTION_FLAGS,
    flag_tokens,
    frac_depth_flag,
    frac_depth_note,
    mark_reviewed,
    original_bounds,
)
from ..analysis.modules import detection_signal, detection_signal_key
from ..analysis.labels import LabelVocabulary, format_labels, set_label
from ..analysis.reconcile import CONSENSUS_MODULE_ID, clear_curation_conflicts
from ..core.session import Session
from .overlays import STEP_LEVEL_HEX
from .trace_view import ChannelPane
from .event_amplitude_controls import EventAmplitudeControls

__all__ = ["EventInspectorWindow"]

# -- filter subsets (the QComboBox item texts, and the "browsed subset" contract) ----------------
_FILTER_ALL = "All"
_FILTER_ACCEPTED = "Accepted only"
_FILTER_REJECTED = "Rejected only"
_FILTERS = (_FILTER_ALL, _FILTER_ACCEPTED, _FILTER_REJECTED)

_RECONCILED_EDIT_TIP = (
    "The Reconciled table is curation-only. Switch to a source detector table, add "
    "or merge there, then reconcile again."
)

# -- padding around the event, so a square blockade is framed by open-pore context ---------------
_PAD_MS = 3.0                # a few ms of context each side (when the sample rate is known)

#: The context widths the "Context:" combo offers, as (label, ± milliseconds); ``None`` is the
#: WHOLE recording. :data:`_PAD_MS` is the default and, until this combo existed, was the only
#: value and unreachable from the UI -- so ±3 ms was a hard CEILING on how much open pore a
#: curator could ever see around an event (owner note: "even from the curate window the user
#: should be able to scroll right/left and inspect other parts of the event"). Panning inside a
#: 6 ms slice cannot answer "was the pore quiet before this?"; only loading more samples can.
_CONTEXT_CHOICES: tuple[tuple[str, float | None], ...] = (
    ("± 3 ms", _PAD_MS),
    ("± 30 ms", 30.0),
    ("± 300 ms", 300.0),
    ("Whole trace", None),
)

#: Level lines the readout will print before it truncates. The readout shares a fixed
#: column with the waveform plot, and CusumConfig.max_levels defaults to 32 — an
#: unbounded listing grows the window past the height of the plot it sits above.
_MAX_LEVEL_LINES = 8
#: Lines the Event box RESERVES for the staleness sentence, empty or not -- see
#: _FixedLinesNote. Three, measured: at this window's minimum width (1008 px; the box is
#: 976 px inside) the banner's two open-pore sentences (342 and 427 characters) wrap to
#: three lines and the calibration one to one. Fixed, so the sentence's arrival -- or
#: its clearing -- never changes the box's height, and the plot above keeps its geometry.
_STALE_NOTE_LINES = 3
_PAD_SAMPLES_FALLBACK = 200  # rate-unknown (legacy-TDMS): the axis is samples, so pad in samples

#: Samples of slack between the VIEW and the LOADED WINDOW before the view is treated as asking
#: for more data. A view range round-trips through the time axis and back to an index, and a
#: viewbox re-announces its range on every pane resize; without a couple of samples of slack the
#: window would creep a sample wider on each of those, forever, for no visible gain.
_WINDOW_SLACK = 2

# -- colours (kept local so the window does not couple to the plots-panel palette) ---------------
_CURVE_HEX = "#4C9BE6"       # the measurement-domain waveform
_REGION_HEX = "#FFB74D"      # the shaded event region (matches the trace span overlays)
_REGION_ALPHA = 55
_BASELINE_HEX = "#9AA2AB"    # the local open-pore baseline_pA line
_LEVEL_HEX = "#D55E00"       # the chosen amplitude's equivalent horizontal level
#: The FITTED sub-state levels. ONE name for the ink, shared with the main trace's step overlay
#: (:data:`~nanoscope.gui.overlays.STEP_LEVEL_HEX`): this used to be a locally spelled ``#009E73``
#: and the trace a different hex, which is how the two surfaces drifted apart about what a level
#: looks like — and the darker of the two is the one the owner could not read ("the sublevels …
#: cannot be read easily on the curate panel. Either the color has to be boldened/changed or
#: another way of representing the sublevels is needed").
_CUSUM_HEX = STEP_LEVEL_HEX
_CUSUM_CASING_HEX = "#10151b"  # near-black underlay, so a level reads as a line laid ON the trace
_CUSUM_LEVEL_WIDTH = 4         # matches overlays._STEP_LEVEL_WIDTH: one weight of line for a level
_CUSUM_CASING_WIDTH = 7        # always > the stroke: the casing is a border, not a second line
_MSG_HEX = "#8b9199"         # muted empty/invalid message ink (reads on the dark theme)

#: A level's label needs a segment WIDE ENOUGH ON SCREEN to sit over, and the measure of that
#: is PIXELS. It used to be ``max(xs.size // 40, 1)`` SAMPLES, i.e. 2.5% of whatever happened
#: to be loaded — so widening the context from ±3 ms to ±30 ms to look at the open pore took
#: the annotation off a level that had not moved and was drawn at exactly the same width in
#: time (measured: ``_cusum_extras`` went 3 → 1 on that one combo change). A rule about
#: legibility must not depend on how much context is loaded. 30 px is roughly two characters
#: of the label, which is the width below which the labels of adjacent levels start to
#: collide; it is the same question ``overlays._NAME_MIN_PX`` answers for the trace's span
#: names, now asked in the same unit.
_MIN_LABEL_PX = 30.0

#: Draw the faint "the event is HERE" band once the loaded window is this many times the
#: event's own length. At ± 3 ms around a 12 ms event the staircase fills the plot and a band
#: would be noise; at Whole trace the same staircase is a few pixels wide somewhere in two
#: million samples, and nothing on screen says where to look.
_EVENT_BAND_RATIO = 10.0
#: Ink for that band: the mask region's own colour at a fraction of its alpha, so it reads as
#: the same object seen from far away rather than as a second annotation.
_EVENT_BAND_ALPHA = 28
#: ...and the width below which a band is drawn as nothing at all. See `_draw_event_band`:
#: this is a pointer's floor, not a measurement.
_EVENT_BAND_MIN_PX = 3.0

#: What ``module_id`` the browsed table came from, in words, for the Revert control. The owner
#: asked to "revert to the model's start/end regions"; the ML detector and the Schmitt trigger
#: are distinguishable PER TABLE (this id) and never PER ROW, so the honest promise the control
#: can make is "whatever detector produced this table" — named here so the button, its tooltip
#: and the disabled-state sentence cannot describe it three ways.
_DETECTOR_NAMES = {
    "detect.schmitt": "the Schmitt detector",
    "detect.cusum": "the CUSUM detector",
    "detect.keras": "the ML detector",
    "detect.study": "the imported study candidate table",
    CONSENSUS_MODULE_ID: "the reconciled result",
}

#: Said when a row carries no recorded original. NOT backfilled from the current bounds — see
#: :func:`~nanoscope.analysis.events.original_bounds`.
_REVERT_TIP_UNKNOWN = (
    "Unavailable for this event: the detection does not record what {detector} originally "
    "produced. Detections saved before this app kept the original bounds carry only their "
    "CURRENT ones, and those may already have been edited — restoring them would claim the "
    "detector produced whatever the last edit left behind. Re-run detection to get a table "
    "that can be reverted.")

#: The never-ran empty state. It NAMES the per-event controls, because until a detection
#: exists they are hidden (see :meth:`EventInspectorWindow._sync_controls`) — two greyed
#: buttons under "No detection" taught the owner that this window had no per-event Accept
#: and Reject at all, which is the complaint that was filed against it twice.
_NO_DETECTION_MSG = ("No detection — run detection to browse events.\n"
                     "Accept / Reject and the context controls appear with the first event.")


def _region_qcolor() -> QtGui.QColor:
    c = QtGui.QColor(_REGION_HEX)
    c.setAlpha(_REGION_ALPHA)
    return c


def _fmt_num(value, suffix: str = "", prec: int = 4) -> str:
    """Format a scalar for the readout; non-finite / non-numeric renders as an em dash."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{f:.{prec}g}{suffix}"


def _canonical_filter(mode: str) -> str | None:
    """Map an exact combo label or a friendly form ("accepted", "rejected", "all")
    onto the canonical filter constant; unknown text returns None (ignored)."""
    if mode in _FILTERS:
        return mode
    m = str(mode).strip().lower()
    if "accept" in m:
        return _FILTER_ACCEPTED
    if "reject" in m:
        return _FILTER_REJECTED
    if m == "all":
        return _FILTER_ALL
    return None


# =================================================================================================
# THE DERIVED HALF OF ``accepted`` -- what a curation action has no standing over
# =================================================================================================
def _derived_accepted(flag: Any) -> bool:
    """``accepted`` as the SCHEMA derives it for a row carrying ``flag``.

    :func:`~nanoscope.analysis.characterize.characterize` owns the rule, and it is an AND,
    never an OR::

        accepted = ~(too_long | clipped | non_finite) & inherit_accepted

    The left half is a statement about the MEASUREMENT -- this row's numbers are not
    measurements of an event -- and the right half is the curator's own decision. A bulk
    "Accept all" is a curator's decision: it sets the right half, and has no standing over
    the left. This function reads the left half back off the row so the GUI can apply the
    same AND instead of writing ``True`` over both halves, which is how "Accept all" came
    to accept a row whose own flag says the detector could not measure it.

    It is read from ``flag`` and not re-derived from the samples because the flag column IS
    the record of that judgement: ``characterize`` writes the token and the ``False``
    together. :data:`~nanoscope.analysis.events.AUTOMATIC_REJECTION_FLAGS` is the ONE place
    that says which tokens are rejections, and naming it here rather than listing
    ``dwell>max_dwell_s`` / ``clipped_by_boundary`` / ``non_finite`` is the point: a fourth
    rejecting rule reaches this window with no second edit, while a derived token that
    rejects nothing (``zero_baseline`` -- a caveat about ``frac_depth``) and a curator's own
    token (``flagged``, ``molecule=apo``) never block an acceptance.

    Rejection is NOT guarded: ``accepted=False`` is always expressible, because a person may
    always reject an event the app was willing to keep.
    """
    return not any(t in AUTOMATIC_REJECTION_FLAGS for t in flag_tokens(flag))


def _refusal_note(kept: int, repaired: int, blocked: int, noun: str) -> str:
    """The sentence a bulk "accept" owes the user when the derived half refused rows.

    THREE OUTCOMES, THREE VERBS, because they are three different things and the user has
    to be able to tell them apart:

    * ``kept``      -- the row was already rejected and stays rejected. Nothing moved.
    * ``repaired``  -- the row was sitting ``accepted=True`` BESIDE a rejecting flag (the
      forbidden state, reachable from any table written before the derived-column fix) and
      this gesture put it back. Something moved, and "kept rejected" would be a lie about
      it: the user's accepted count went DOWN on a click that said "accept".
    * ``blocked``   -- the frame carries no ``flag`` column at all, so the derived half
      cannot be read and NOTHING was accepted. Fail closed: the guard turning itself off
      silently is the pre-fix broadcast wearing the fix's name.

    Returns ``""`` for a gesture with nothing to disclose, so the caller can test the
    string itself rather than remember a sentinel.
    """
    def _n(k: int) -> str:
        return f"{k} {noun}" + ("" if k == 1 else "s")

    if blocked > 0:
        return (f"{_n(blocked)} NOT accepted: this table has no 'flag' column, so the "
                "detector's own rejections cannot be read — refusing rather than guessing")
    parts: list[str] = []
    if kept > 0:
        parts.append(f"{_n(kept)} kept rejected")
    if repaired > 0:
        parts.append(f"{_n(repaired)} corrected back to rejected")
    if not parts:
        return ""
    return " and ".join(parts) + ": flagged by the detector as not measurable"


# =================================================================================================
# WHY THIS EVENT HAS NO SUB-STATE LEVELS -- ONE WORDING, BOTH SURFACES
# =================================================================================================
# F2. A mask drag re-fits the edited event's CUSUM levels
# (``MainWindow._refit_levels_for_event``), and there are five ways that ends in no levels
# at all: a span too short to resolve a boundary, a rejected row, a cold re-fit cache, a
# fit that raises, and a fit that comes back empty. Measured before this existed, on the
# five-event fixture in tests/gui/test_level_fit_reconcile.py:
#
#   * the INSPECTOR said nothing whatsoever. Drag event 0 down to 9 samples and the
#     staircase, the readout's whole CUSUM block and every plot item vanished together,
#     with `_batch_note` empty and the readout printing only the always-on `levels: 1`.
#     The one surface the user was looking at -- they had just dragged its mask -- was the
#     one surface that did not mention what had happened.
#   * the STATUS BAR named a recovery that measurably does not work. The rejected route
#     and the empty-fit route fall through with no message of their own, leaving
#     `_reconcile_level_fit`'s standing sentence: "...re-run Analysis > Fit sub-state
#     levels (CUSUM)... to measure the new spans." Performed on a rejected event, that
#     re-run returns 0 rows for it -- the fit runs `accepted_only=True`, so a rejected
#     event has no levels BY CONSTRUCTION and no number of re-runs will produce any. The
#     remedy that does work is to accept the event again (measured: accept + re-run -> 3
#     levels; accept + one more mask edit -> 3 levels).
#
# So the sentences live HERE, at module level, and both surfaces read them: MainWindow
# posts one to the status bar and hands the SAME string to
# :meth:`EventInspectorWindow.note_level_gap`, and this window derives one itself for the
# cases it can read off the frame it already holds. Two copies of this wording is how the
# window and the status bar would come to prescribe different remedies for one event.

def _refit_clause(standing_fit: bool) -> str:
    """How a declined event gets its levels back -- and it is a BRANCH, not one sentence.

    MEASURED on the five-event fixture. With the other four events still fitted, widening
    event 0's mask back re-fitted it to 7 levels on release: a mask edit is the whole
    remedy. With every other event rejected so event 0 carried the only group, the
    reconcile retired the WHOLE fit (``_invalidate_level_fit``), and
    ``_refit_levels_for_event`` then returns at its first guard (``fit is None``) -- so
    widening the mask back restored nothing at all, twice over. Telling that user "widen it
    again" prescribes a gesture that provably does nothing; they have to re-run the fit.
    """
    if standing_fit:
        return "edit the mask again and they are re-fitted on release"
    return ("re-run Analysis ▸ Fit sub-state levels (CUSUM)… — the last fitted group "
            "retired with this one, so a mask edit has no standing fit left to splice into")


def level_gap_too_short(event_id: int, *, n_samples: int, min_samples: int,
                        standing_fit: bool = True) -> str:
    """Below ``MIN_SAMPLES_FOR_LEVELS`` the fit does not fail, it returns one "level"
    spanning the whole event -- which says nothing and would sit in the export looking like
    a measurement. The remedy is the MASK, not the menu: measured, re-running the whole fit
    over a 9-sample event writes exactly that one meaningless row (``_fit_levels`` applies
    no minimum), so "re-run the fit" would be prescribing the very artefact the guard
    declines to invent."""
    tail = ("and they are re-fitted on release" if standing_fit else
            "and then re-run Analysis ▸ Fit sub-state levels (CUSUM)… — the last fitted "
            "group retired with this one, so a mask edit has no standing fit to splice into")
    return (f"Sub-state levels dropped for event {int(event_id)}: its new span is "
            f"{int(n_samples)} sample(s), below the {int(min_samples)} a level fit needs to "
            f"resolve a boundary. Widen the mask past {int(min_samples)} samples {tail}.")


def level_gap_rejected(event_id: int, *, standing_fit: bool = True) -> str:
    """THE SENTENCE F2-a IS ABOUT. The fit runs ``accepted_only=True``, so a rejected event
    has no levels by construction -- and the message this route used to fall back to told
    the user to re-run the fit, which measurably returns 0 rows for it however many times
    it is run. Naming the accept is naming something the user can actually do."""
    return (f"Sub-state levels dropped for event {int(event_id)}: it is rejected, and the "
            f"fit measures accepted events only — re-running Analysis ▸ Fit sub-state "
            f"levels (CUSUM)… leaves this event with none. Accept it again, then "
            f"{_refit_clause(standing_fit)}.")


def level_gap_cold(event_id: int) -> str:
    """A fit RESTORED FROM A PROJECT has no re-fit inputs behind it, and re-deriving them
    means a 644 ms open-pore baseline estimate on the GUI thread inside a mouse-release
    handler. Wording kept as it shipped: measured, the re-run it names does restore the
    levels (3 rows back on the fixture) and does make every later edit re-fit itself."""
    return (f"Sub-state levels dropped for event {int(event_id)}: re-fitting needs the "
            "open-pore baseline this fit was measured against, which is not in memory "
            "after reopening a project. Re-run Analysis ▸ Fit sub-state levels (CUSUM)… "
            "once and later edits will re-fit themselves.")


def level_gap_failed(event_id: int, *, detail: str) -> str:
    """The re-fit raised. ``detail`` is the exception text, kept verbatim -- it is the only
    record of what went wrong, and a message that hides it leaves the user with a
    disappearance and no name for it."""
    return (f"Sub-state levels dropped for event {int(event_id)}: they could not be "
            f"re-fitted over the new span ({detail}). Move the mask edge to try again, or "
            "re-run Analysis ▸ Fit sub-state levels (CUSUM)… to re-measure the whole "
            "detection.")


def level_gap_empty(event_id: int, *, standing_fit: bool = True) -> str:
    """The re-fit returned zero rows -- and "re-run the fit" is the one thing that cannot
    help, because a re-run measures the same samples against the same baseline with the
    same sigma and tau and returns the same nothing. Only the SPAN can change the answer."""
    tail = ("Move the mask edge — that is the only input a re-fit sees change"
            if standing_fit else
            "Move the mask edge, then re-run Analysis ▸ Fit sub-state levels (CUSUM)… — "
            "the last fitted group retired with this one")
    return (f"Sub-state levels dropped for event {int(event_id)}: the re-fit resolved no "
            f"levels over its new span. {tail}.")


def level_gap_unfitted(event_id: int) -> str:
    """A fit is standing, this event is in no group, and nothing announced why -- a merge,
    an undo or a re-measure moved its span and the drop is all that ran (those paths
    announce nothing to re-fit from, deliberately). Re-running the fit does cover it."""
    return (f"No sub-state levels for event {int(event_id)}: the standing fit does not "
            "cover it. Re-run Analysis ▸ Fit sub-state levels (CUSUM)… to measure it.")


# =================================================================================================
# THE REFUSAL LEDGER -- which events this window declined to accept, and how many still stand
# =================================================================================================
class _RefusalLedger:
    """The events this window has DECLINED to accept, on the frame it currently holds.

    A MEMBERSHIP RECORD THAT ONLY A NEW FRAME CLEARS. What it replaces was cleared at the
    top of every curation write, which made the disclosure a per-gesture artefact instead
    of a statement about the data. Measured in the wired app on the owner's -600 mV
    recording (317 events, ``max_dwell_s`` 26 ms, rows 0 / 4 / 260 unmeasurable):
    "Accept all" left ``314 of 317 accepted   ·   3 tiles kept rejected``, and then
    pressing R on ONE ORDINARY TILE left ``313 of 317 accepted`` with no clause at all --
    four rows unaccepted, three of them because the detector could not measure them, and
    nothing saying so. The IDENTICAL whole-frame reject made from the OTHER window left
    the note standing (that one arrives through ``refresh()``, which never cleared), so
    the app said two different things about one identical frame depending on which window
    the click landed in.

    THE RULE, in one place so it cannot be half-applied:

    * an ACCEPT re-decides the events IN ITS OWN SCOPE and no others. It is the only
      gesture that can create or retire a refusal, because a refusal is what an accept
      turned into.
    * a REJECT records nothing and forgets nothing. Rejection is never refused, so it has
      no refusal of its own to disclose, and the events an EARLIER accept was refused over
      are still refused. That is exactly what ``reflect_acceptance`` -- the same decision
      arriving from the events table -- has always done, and it is why the window's own
      gesture and the mirrored one now agree.
    * a NEW FRAME clears it, because the events it names no longer exist.

    THE COUNTS ARE NEVER STORED. :meth:`counts` re-reads the frame every time the line is
    rebuilt, so a row the events table has since accepted leaves the count on its own and
    takes nothing with it -- and comes BACK if it is rejected again, which a decrementing
    latch cannot do.

    IT REMEMBERS ``event_id``, NOT ROW POSITION. A row index survives an append and not a
    merge, and this record now outlives many more gestures than the per-gesture counter
    did. The window already re-finds the user's selection by ``event_id`` across a
    refilter for precisely that reason; this follows it rather than inventing a second
    policy.
    """

    __slots__ = ("_kept", "_repaired", "_blocked")

    def __init__(self) -> None:
        self._kept: set[int] = set()
        self._repaired: set[int] = set()
        # The fail-closed refusal is about the FRAME (no ``flag`` column at all) and not
        # about particular events, so it is a count of what one accept asked for and got
        # none of -- there are no event ids to remember when nothing vouches for any row.
        self._blocked: int = 0

    def clear(self) -> None:
        """Forget everything. A different frame is a different detection."""
        self._kept.clear()
        self._repaired.clear()
        self._blocked = 0

    def forget(self, event_ids) -> None:
        """This accept is about to re-decide ``event_ids``, so drop what an earlier one
        recorded about them. The frame-level refusal goes too: this gesture re-decides
        that as well, and on a frame with no ``flag`` column it reaches the same answer."""
        gone = {int(e) for e in event_ids}
        self._kept -= gone
        self._repaired -= gone
        self._blocked = 0

    def record(self, kept=(), repaired=()) -> None:
        """Two buckets, decided by where the row STARTED -- see :func:`_refusal_note`.
        WHICH events, never how many: the counts are recomputed on every rebuild of the
        line, and which bucket an event is in is history that cannot be re-derived."""
        self._kept.update(int(e) for e in kept)
        self._repaired.update(int(e) for e in repaired)

    def record_blocked(self, n: int) -> None:
        """``n`` rows asked for and none accepted: the frame carries no ``flag`` column."""
        self._blocked = int(n)

    def counts(self, df) -> tuple[int, int, int]:
        """``(kept, repaired, blocked)`` AS OF NOW -- recounted from ``df``, never stored.

        The frame-level refusal wins when it stands, because on a frame with no ``flag``
        column the other two questions cannot be asked at all.
        """
        if df is None or "flag" not in df.columns:
            return (0, 0, self._blocked)
        return (self.refused_in(df, self._kept), self.refused_in(df, self._repaired), 0)

    @staticmethod
    def refused_in(df, event_ids) -> int:
        """How many of ``event_ids`` are STILL refused -- rejected, AND rejected because
        the detector could not measure them. Both halves are re-read from the frame, so
        this is a question about NOW and not about the gesture that recorded them. An
        event that has left the frame entirely simply is not counted.

        VECTORISED, and counted PER DISTINCT FLAG STRING -- the same trick, for the same
        reason, as the gallery's accept-button promise. This runs on every rebuild of the
        status line, i.e. on every page turn, filter change, curation click and resize;
        asking :func:`_derived_accepted` once per row cost 833 ms on a 100 000-event
        detection with 90 % unmeasurable rows, which would have made an honest disclosure
        the slowest thing in the window.
        """
        if df is None or not len(event_ids) or "flag" not in df.columns:
            return 0
        if "event_id" not in df.columns:
            # A frame off the frozen schema has no identity to answer with. The window
            # cannot work at all without this column, but a REPAINT must not be the thing
            # that raises: the status line is rebuilt on every resize.
            return 0
        wanted = np.fromiter((int(e) for e in event_ids), dtype=np.int64,
                             count=len(event_ids))
        mask = np.isin(df["event_id"].to_numpy(), wanted)
        mask &= ~df["accepted"].to_numpy(dtype=bool)
        if not mask.any():
            return 0
        return int(sum(int(k) for value, k
                       in pd.Series(df["flag"].to_numpy()[mask])
                       .value_counts(dropna=False).items()
                       if not _derived_accepted(value)))

    @property
    def kept(self) -> set[int]:
        """The event ids in the ``kept rejected`` bucket. Read-only; the note counts them
        against the frame rather than trusting the size of this set."""
        return self._kept

    @property
    def repaired(self) -> set[int]:
        """The event ids in the ``corrected back to rejected`` bucket."""
        return self._repaired


class _FixedLinesNote(QtWidgets.QLabel):
    """A word-wrapping note that OWNS a fixed number of lines whether it says anything or not.

    The Event box's staleness sentence lives in one of these rather than in the mono
    readout under it, because the readout's height is its line count and the box sits in
    the same column as the waveform plot: spliced in as readout lines, the banner's
    ~340-character sentence took six lines, and one Preprocess knob on the MAIN window --
    or the "Re-measure depths" that clears it -- resized the plot under the pointer
    (measured at 1000x800: plot 531 -> 447 -> 531 px, readout 140 -> 224 px; at the
    default size the top-level window itself grew 640 -> 653 px). A disclosure must not
    move the surface it discloses on. With the height reserved here, the flip changes
    text and nothing else -- the pattern the subpopulation report's live note follows.

    Longer than the reserved lines at the current width, the sentence is shown from the
    head with its tail replaced by "…" and the whole of it in the tooltip: show the head
    and say what was withheld, as :meth:`EventInspectorWindow._level_lines` does. That
    case is a backstop, not the expectation -- at this window's minimum width the box is
    976 px wide and the banner's longest sentence (427 characters) wraps to exactly the
    three reserved lines (see ``_STALE_NOTE_LINES``).

    :meth:`text` returns the FULL sentence; what is painted is what fitted.
    """

    def __init__(self, lines: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("", parent)
        self._full = ""
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # The width is the box's, and a word-wrapping label must never set the window's
        # minimum width; the height is decided here, once.
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        # The measuring twin: the same font, wrap and interaction flags, so what it says
        # fits at a width is what this label paints at that width. A selectable QLabel
        # lays its text out through a text document rather than the bare font metrics,
        # so the same machinery is asked rather than a second one that can disagree.
        self._probe = QtWidgets.QLabel(self)
        self._probe.setWordWrap(True)
        self._probe.setTextFormat(Qt.PlainText)
        self._probe.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._probe.hide()
        self._probe.setText("\n".join(["x"] * max(int(lines), 1)))
        self.setFixedHeight(self._probe.sizeHint().height() + 4)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = str(text)
        self._fit()

    def text(self) -> str:  # noqa: N802 (Qt override)
        return self._full

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def _fits(self, text: str) -> bool:
        self._probe.setText(text)
        return self._probe.heightForWidth(self.width()) <= self.height()

    def _fit(self) -> None:
        shown, elided = self._full, False
        if shown and not self._fits(shown):
            words = shown.split()
            while len(words) > 1:
                words.pop()
                shown = " ".join(words) + " …"
                if self._fits(shown):
                    break
            else:
                shown = "…"
            elided = True
        if shown != QtWidgets.QLabel.text(self):
            QtWidgets.QLabel.setText(self, shown)
        self.setToolTip(self._full if elided else "")


class EventInspectorWindow(QtWidgets.QMainWindow):
    """Browse + curate detected events one at a time.

    Fed the current detection via :meth:`set_events` (a REFERENCE to the shared
    events ``.df``, never a copy), it renders the current event's measurement waveform,
    navigates within a filtered subset (All / Accepted only / Rejected only) and
    curates the shared ``accepted`` column in place — announcing every decision so
    the owner can mirror it onto the table, overlays, summary and exports.

    Public API (also used by tests):
        set_events(doc_id, channel, events_df, module_id, provenance)
        clear()
        show_event(event_id)
        current_event_id() -> int | None
        next_event() / prev_event()
        set_accepted_current(bool)
        reflect_acceptance(event_id, accepted)   # inbound mirror; never re-emits
        accept_all() / reject_all()              # batch-curate every event on the shared df
        set_filter(mode)
        event_count() -> int                     # size of the current filter subset
        set_stale_columns(columns, note)         # the events table's badge set + sentence
        stale_columns() / stale_note()           # ...as this window carries them
        set_project_save_state(path, unsaved)    # MainWindow's project-save truth

    The event region is MOVABLE: dragging either edge rewrites that event's
    ``[start_idx, end_idx]`` on the shared df and re-characterizes it (see
    :meth:`_recharacterize_row`), emitting ``eventEdited``. A programmatic
    ``show_event`` sets the region behind a ``_syncing`` fence so it never reads
    back as an edit.
    """

    eventSelected = Signal(int)              # the browsed event changed
    galleryRequested = Signal()              # "Gallery…" — open the contact sheet from here
    # "Show in trace" — the third side of the triangle. The panel reached the inspector and the
    # gallery, and the gallery reached the trace (eventActivated), but the INSPECTOR had no way
    # back to the main trace at all: the owner had to find the row in the events table by hand
    # (owner note: "there should be a direct connection between the curate window and the event
    # gallery … easier to access one to any of the other two"). A VIEW move, like the gallery's
    # own double-click, so MainWindow routes it to the same _goto_event_in_trace.
    traceRequested = Signal(int)             # show this event in the main trace (event_id)
    acceptanceChanged = Signal(int, bool)    # the user curated an event (id, accepted)
    eventEdited = Signal(int)                # the user resized an event's mask (event_id)
    amplitudeEdited = Signal(int)            # optional amplitude only; -1 means batch; NEVER refit
    curationBatchChanged = Signal()          # Accept all / Reject all rewrote 'accepted'
    mergeWithNextRequested = Signal(int)     # stitch this event to the following one(s) (event_id)
    labelsChanged = Signal()                 # a label was written on the shared frame
    # An accept the derived half refused, with NOTHING written (the A key / Accepted
    # checkbox on a row the detector flagged unmeasurable, already rejected). The same
    # contract as the events table's ``acceptanceRefused``: the refusal landed on the
    # SHARED ledger, and without a signal the other two surfaces' notes never repainted
    # (measured: this window said "1 event kept rejected" while the table label stayed
    # empty about the same record until some unrelated repaint).
    acceptanceRefused = Signal(int)          # refused event's id (-1: whole frame); no write
    # Project saving is intent only: MainWindow owns the path, dirty flag, validation and
    # serializer. These signals merely make saving reachable from the active inspector.
    saveProjectRequested = Signal()
    saveProjectAsRequested = Signal()

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("NanoScope — Event inspector")
        self.resize(780, 640)

        # Installed detection. _df is the SAME object the EventTable edits (shared truth).
        self._doc_id: str | None = None
        self._channel: int | None = None
        self._df: pd.DataFrame | None = None
        self._module_id: str | None = None
        self._units: str = ""                # the channel's own units, for the readout labels
        self._current_id: int | None = None  # event_id of the displayed event
        self._filter: str = _FILTER_ALL
        self._updating = False               # suppress control-change slots during bulk updates
        self._syncing = False                # fence programmatic region sets from the edit slot
        self._provenance: dict[str, Any] | None = None  # detector polarity / noise / filter rise
        #: The CUSUM level table (analysis.cusum.LEVEL_SCHEMA) for this detection, or None.
        #: Held SEPARATELY from _df rather than merged into it: the event frame is one row
        #: per event and frozen at 20 columns, and this is one row per (event, level).
        self._levels: pd.DataFrame | None = None
        #: WHY an event has no group in :attr:`_levels` — ``{event_id: (start, end,
        #: sentence)}``, the sentence MainWindow's re-fit handed over when it declined or
        #: failed (see :meth:`note_level_gap`), stamped with the bounds it was true of.
        #:
        #: NOT A LATCH, in three independent ways, because a false explanation standing
        #: over the picture is worse than the silence F2 replaced: :meth:`_level_gap_note`
        #: renders one only while the event genuinely has no rows; :meth:`set_levels`
        #: forgets the entry the moment those rows come back; and the STAMP retires it as
        #: soon as the span moves again. The stamp is what the single-event case needs —
        #: there, a further widen re-fits nothing (the whole fit has retired), so without it
        #: the window would go on saying "its new span is 9 sample(s)" over a 600-sample
        #: event that had just been dragged back out.
        self._level_gaps: dict[int, tuple[int | None, int | None, str]] = {}
        # -- THE LOADED WINDOW -------------------------------------------------------------
        #: The half-open sample range currently HELD BY THE CURVE, ``[_win_lo, _win_hi)``.
        #: It exists because the window and the VIEW used to be the same thing: the slice was
        #: taken once in _render_event and immediately framed edge to edge, so every loaded
        #: sample was already on screen before the first click and "pan" could only walk the
        #: view off the end of its own data -- half the pane blank after one press, all of it
        #: after two. Panning and zooming now move this range and RE-SLICE (see
        #: :meth:`_draw_window`), which is the only way a pan reveals a sample that was not
        #: already visible.
        self._win_lo: int | None = None
        self._win_hi: int | None = None
        #: The context the user asked for, in ± milliseconds, or None for the whole recording
        #: (the "Context:" combo). PERSISTS ACROSS NAVIGATION on purpose: a curator sets the
        #: context once and then steps, and re-framing to ±3 ms on every Prev/Next threw that
        #: setting away silently.
        self._pad_ms: float | None = _PAD_MS
        #: The per-side context the user last framed BY HAND (mouse zoom), in samples, or None.
        #: Same persistence rule as ``_pad_ms`` -- the WIDTH survives the navigation, the
        #: CENTRING does not (the next event is re-centred, as it always was). Cleared when the
        #: combo is used, because an explicit choice supersedes an implicit one.
        self._manual_pad: int | None = None
        #: Fence around our OWN setXRange calls, so the range handler cannot mistake this
        #: window's framing for the user's zoom (and cannot recurse through it).
        self._framing = False
        #: One-entry memo for :meth:`_conditioned` -- ``(key, conditioned_trace)``.
        self._cond_cache: tuple[tuple[Any, ...], np.ndarray] | None = None
        self._message_item: pg.TextItem | None = None
        #: The shared label palette (see analysis.labels). Handed in by MainWindow so the
        #: digit that means "apo-ferritin" in the gallery means it here too.
        self._vocab = LabelVocabulary()
        #: Columns the owner has marked as measured by a superseded estimator -- the events
        #: table's own badge set, pushed through :meth:`set_stale_columns` -- and the
        #: banner's sentence. Read by the readout and the plot title, which are the two
        #: places this window prints those numbers.
        self._stale_columns: tuple[str, ...] = ()
        self._stale_note: str = ""
        #: WHICH EVENTS this window has declined to accept on the frame it holds -- not how
        #: many, and not "the last gesture's".
        #: STATE, not text, for the same reason the gallery keeps it as state: this window is
        #: re-fed by MainWindow's surface refresher on its OWN ``curationBatchChanged``, so a
        #: label written after the write and before the emit is erased before it is read.
        #:
        #: Event IDENTITIES and not counts, because the counter is a COUNT OF
        #: CURRENTLY-REFUSED ROWS and not a latch -- see :attr:`_refused`. What clears it,
        #: and what does not, is stated once in :class:`_RefusalLedger`.
        self._refusals = _RefusalLedger()

        self._build_ui()
        self._install_project_save_controls()
        self._install_shortcuts()
        self._show_empty(_NO_DETECTION_MSG)

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # -- navigation + curation bars --------------------------------------------------------
        # TWO rows, not one (owner note, reported twice: "the curate window does not have a
        # reject button, only a reject all" / "I still only see Accept all / Reject all and not
        # the individual accept and reject"). The buttons were there and correctly wired the
        # whole time; the single non-wrapping QHBoxLayout that held them measured 1475 px of
        # minimum width on a 1512 px display, and Accept/Reject/Accepted were the LAST three
        # widgets in it — so any Dock, any display scaling, any window not flush with the left
        # edge pushed exactly those three off the right edge while leaving "Accept all" and
        # "Reject all" (earlier in the row) visible. A QHBoxLayout does not wrap and offers no
        # overflow affordance, so nothing said they existed. Splitting the row is the fix that
        # cannot regress: PER-EVENT gestures on row 1 beside the per-event navigation, WHOLE-
        # DETECTION gestures and the doors to the other surfaces on row 2.
        bar = QtWidgets.QHBoxLayout()
        bar2 = QtWidgets.QHBoxLayout()
        self._prev_btn = QtWidgets.QPushButton("◀ Prev")
        self._prev_btn.setToolTip("Previous event in the current filter (←).")
        self._prev_btn.clicked.connect(self.prev_event)
        self._next_btn = QtWidgets.QPushButton("Next ▶")
        self._next_btn.setToolTip("Next event in the current filter (→).")
        self._next_btn.clicked.connect(self.next_event)

        self._index_spin = QtWidgets.QSpinBox()
        self._index_spin.setPrefix("Event ")
        self._index_spin.setRange(0, 0)
        self._index_spin.setToolTip("Jump to the i-th event of the current filter subset.")
        self._index_spin.valueChanged.connect(self._on_spin_changed)
        self._count_label = QtWidgets.QLabel("of 0")

        self._filter_combo = QtWidgets.QComboBox()
        self._filter_combo.addItems(list(_FILTERS))
        self._filter_combo.setToolTip(
            "The browsed subset. Prev/Next step within it (curation moves events between "
            "the Accepted and Rejected subsets).")
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)

        self._accept_check = QtWidgets.QCheckBox("Accepted")
        self._accept_check.setToolTip("Curate this event (A = accept, R = reject, Space = toggle).")
        self._accept_check.toggled.connect(self._on_accept_toggled)

        # Explicit per-event buttons BESIDE the checkbox (owner note #6: "the curate window
        # has no reject button, only Reject all"). The checkbox and the A/R keys did this
        # already, but a state indicator is not an action affordance — the owner curated a
        # whole detection without finding either. Same handler as the keys, exactly: curate
        # the displayed event, do not navigate.
        self._accept_btn = QtWidgets.QPushButton("Accept")
        self._accept_btn.setToolTip("Accept this event (A). Does not advance — press → for the next.")
        self._accept_btn.clicked.connect(lambda: self.set_accepted_current(True))
        self._reject_btn = QtWidgets.QPushButton("Reject")
        self._reject_btn.setToolTip("Reject this event (R). Does not advance — press → for the next.")
        self._reject_btn.clicked.connect(lambda: self.set_accepted_current(False))

        # Revert THIS event's mask to the bounds the detector produced (owner note E1: "the
        # user needs the option to revert to the model's start/end regions"). It sits beside
        # Accept/Reject because it is the third per-event gesture that WRITES the shared frame,
        # and its enabled state is per-row (see _revert_state): a detection saved before the
        # original bounds were kept can answer for no row, and a row nobody has dragged has
        # nothing to revert.
        self._revert_btn = QtWidgets.QPushButton("Revert mask")
        self._revert_btn.setToolTip(_REVERT_TIP_UNKNOWN.format(detector="the detector"))
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self._revert_current)

        # Pan the padded window sideways (owner note #8: "scroll left/right to inspect
        # around the event"). Drag-pan is unavailable over the middle of the view — the
        # movable mask region owns that drag — so the buttons are the reliable gesture;
        # each shifts the view by half its width. A pure view move: nothing is edited,
        # and the next Prev/Next re-centres as always.
        self._pan_left_btn = QtWidgets.QPushButton("⟵")
        self._pan_left_btn.setToolTip(
            "Pan the view left by half a window to inspect the trace before this event. "
            "View only — the event mask does not move.")
        self._pan_left_btn.clicked.connect(lambda: self._pan_view(-0.5))
        self._pan_right_btn = QtWidgets.QPushButton("⟶")
        self._pan_right_btn.setToolTip(
            "Pan the view right by half a window to inspect the trace after this event. "
            "View only — the event mask does not move.")
        self._pan_right_btn.clicked.connect(lambda: self._pan_view(0.5))

        # HOW MUCH context to load around the event. _PAD_MS was unreachable from the UI, which
        # made ±3 ms a ceiling and not a default: no gesture in this window could show the
        # curator the second before the blockade. The choice persists across navigation (see
        # _pad_ms) because it is a working preference, not a per-event one.
        self._context_combo = QtWidgets.QComboBox()
        for _label, _ms in _CONTEXT_CHOICES:
            self._context_combo.addItem(_label, _ms)
        self._context_combo.setToolTip(
            "How much recording to load either side of the event. Kept while you step through "
            "events; the pan arrows and the mouse (drag / wheel) move within it and load more "
            "as needed.")
        self._context_combo.currentIndexChanged.connect(self._on_context_changed)
        # ...and ``activated`` as well, which fires when the user picks the entry that is
        # ALREADY current. Re-picking "± 3 ms" is how someone gets back to it after zooming
        # out with the wheel, and on currentIndexChanged alone that gesture did nothing.
        self._context_combo.activated.connect(self._on_context_changed)

        # The way BACK to the contact sheet (owner note #5: the surfaces should link to
        # each other, not only to the menu — the gallery lives under "Explore this
        # recording" and was hard to find from here).
        self._gallery_btn = QtWidgets.QPushButton("Gallery…")
        self._gallery_btn.setToolTip(
            "Open the Event gallery — every detected event as a grid of tiles "
            "(also under Analysis ▸ Event gallery…).")
        self._gallery_btn.clicked.connect(self.galleryRequested.emit)

        # ...and the way back to the MAIN TRACE, which did not exist at all: the panel opens the
        # inspector and the gallery, the gallery jumps to the trace, but from here the only way
        # to see the event in its recording was to hunt for its row in the events table. Emits
        # intent (traceRequested); MainWindow routes it to the same _goto_event_in_trace the
        # gallery's double-click uses.
        self._trace_btn = QtWidgets.QPushButton("Show in trace")
        self._trace_btn.setToolTip(
            "Centre the main trace on this event (T) — the same jump as double-clicking its "
            "tile in the gallery.")
        self._trace_btn.clicked.connect(self._on_show_in_trace)

        # Merge convenience: stitch THIS event to the following event(s) within the merge
        # window (a translocation the detector split). Emits intent; MainWindow routes it to
        # the AnalysisPanel's merge engine and advances the browser. Reversed by Edit > Undo,
        # which restores the exact table the merge replaced -- NOT by re-running detection,
        # which replaces the whole table and takes every accept/reject decision with it.
        self._merge_next_btn = QtWidgets.QPushButton("Merge with next")
        self._merge_next_btn.setToolTip(
            "Stitch this event to the following event(s) within the merge window into one "
            "event, then advance. Reversed by Edit ▸ Undo (⌘Z).")
        self._merge_next_btn.clicked.connect(self._on_merge_with_next)

        # Batch curation: accept or reject EVERY event at once (the whole df, not just the filtered
        # subset), so the reviewer can "accept all, then reject the few bad ones" or the reverse.
        self._accept_all_btn = QtWidgets.QPushButton("Accept all")
        self._accept_all_btn.setToolTip(
            "Mark every detected event accepted — the whole column, including events "
            "you have already judged. Events the detector flagged as unmeasurable "
            "(clipped by the boundary, over the dwell limit, measured from a non-finite "
            "sample) stay rejected: that is not a decision this button can reverse — the "
            "note under this bar counts them. Reversed by Edit ▸ Undo (⌘Z).")
        self._accept_all_btn.clicked.connect(self.accept_all)
        self._reject_all_btn = QtWidgets.QPushButton("Reject all")
        self._reject_all_btn.setToolTip(
            "Mark every detected event rejected — the whole column, including events "
            "you have already judged. Reversed by Edit ▸ Undo (⌘Z).")
        self._reject_all_btn.clicked.connect(self.reject_all)

        # Row 1 — THIS EVENT: where you are, how much of the recording you see around it, and
        # the two gestures that judge it. Accept/Reject sit beside the navigation they are used
        # with, never behind the batch buttons.
        bar.addWidget(self._prev_btn)
        bar.addWidget(self._index_spin)
        bar.addWidget(self._count_label)
        bar.addWidget(self._next_btn)
        bar.addSpacing(8)
        bar.addWidget(self._pan_left_btn)
        bar.addWidget(self._pan_right_btn)
        bar.addWidget(self._context_combo)
        bar.addSpacing(16)
        bar.addWidget(self._accept_btn)
        bar.addWidget(self._reject_btn)
        bar.addWidget(self._accept_check)
        bar.addSpacing(8)
        bar.addWidget(self._revert_btn)
        bar.addStretch(1)
        root.addLayout(bar)

        # Row 2 — THE WHOLE DETECTION, and the doors out of this window.
        bar2.addWidget(QtWidgets.QLabel("Filter:"))
        bar2.addWidget(self._filter_combo)
        bar2.addSpacing(12)
        bar2.addWidget(self._gallery_btn)
        bar2.addWidget(self._trace_btn)
        bar2.addStretch(1)
        bar2.addWidget(self._accept_all_btn)
        bar2.addWidget(self._reject_all_btn)
        bar2.addSpacing(12)
        bar2.addWidget(self._merge_next_btn)
        root.addLayout(bar2)

        # -- what the last curation gesture REFUSED ---------------------------------------
        # Its own widget, directly under the buttons and the checkbox that produce it, and
        # NOT part of the per-event readout: the readout is rewritten on every navigation,
        # and a disclosure that dies when the user presses "Next" is not a disclosure.
        # Written only by _sync_controls, from _refused -- see the note on that attribute.
        # It carries the PER-EVENT refusal too (the Accepted checkbox / the A key on a row
        # the detector could not measure), because a refusal is a refusal at any size.
        self._batch_note = QtWidgets.QLabel("")
        self._batch_note.setWordWrap(True)
        self._batch_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._batch_note.setStyleSheet(f"color: {_MSG_HEX};")
        self._batch_note.setVisible(False)
        root.addWidget(self._batch_note)

        # -- why the SUB-STATE LEVELS are not on the waveform (F2) -------------------------
        # ITS OWN WIDGET, beside the refusal note rather than inside it. The two say
        # different kinds of thing (one is about a curation gesture the window refused, the
        # other about a derived fit that has nothing to draw here), they clear on different
        # events, and _batch_note has exactly one writer by design — `_sync_controls`, from
        # `_refused`. A second source written into that same label would have made the
        # suite's `_batch_note.text() == ""` refusal assertions depend on whether a level
        # fit happened to be standing.
        #
        # ABOVE THE PLOT, not in the readout: this sentence is about the picture that just
        # went blank, and the readout is a fixed-height panel BELOW it that the eye reaches
        # second. The readout carries the same reason too (see :meth:`_level_lines`) so a
        # user reading the per-level breakdown finds it where the breakdown used to be.
        self._level_note = QtWidgets.QLabel("")
        self._level_note.setWordWrap(True)
        self._level_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._level_note.setStyleSheet(f"color: {_MSG_HEX};")
        self._level_note.setVisible(False)
        root.addWidget(self._level_note)

        self._amplitude = EventAmplitudeControls(self)
        root.addWidget(self._amplitude)

        # -- the waveform plot ----------------------------------------------------------------
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumSize(480, 300)
        self._plot: pg.PlotItem = self._plot_widget.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        # The MOUSE half of panning. The arrows load samples (see _pan_view); without this the
        # drag and the wheel would still run off into blank space exactly as they did before,
        # which is a worse inconsistency than having neither work. One handler serves both:
        # whatever moved the x range, the loaded window is widened to cover what is now framed.
        self._plot.getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)
        root.addWidget(self._plot_widget, 1)

        # Persistent plot items — created ONCE (four-step order via make_curve) and updated per
        # event; hidden in the empty state. The region is non-movable and ignoreBounds (an
        # annotation), and the two horizontal lines never drive the y-autorange either.
        self._curve = ChannelPane.make_curve(self._plot, _CURVE_HEX)
        # MOVABLE region: its two edges are draggable handles for resizing the event mask.
        # ignoreBounds keeps it out of the y-autorange (it is an annotation, not data), and every
        # PROGRAMMATIC set (in _render_event) is fenced by _syncing so it never fires a false edit.
        self._region = pg.LinearRegionItem(movable=True, brush=pg.mkBrush(_region_qcolor()))
        self._region.setZValue(-10)          # behind the waveform
        self._plot.addItem(self._region, ignoreBounds=True)
        self._region.sigRegionChangeFinished.connect(self._on_region_edited)
        self._baseline_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_BASELINE_HEX, width=1, style=Qt.DashLine),
            label="baseline", labelOpts={"color": _BASELINE_HEX, "position": 0.02,
                                         "anchors": [(0, 0), (0, 1)]})
        self._baseline_line.setZValue(10)
        self._plot.addItem(self._baseline_line, ignoreBounds=True)
        self._level_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(_LEVEL_HEX, width=1, style=Qt.DashLine),
            label="level", labelOpts={"color": _LEVEL_HEX, "position": 0.98,
                                      "anchors": [(1, 0), (1, 1)]})
        self._level_line.setZValue(10)
        self._plot.addItem(self._level_line, ignoreBounds=True)
        self._amplitude.install_plot_items()

        # -- per-event stat readout -----------------------------------------------------------
        readout_box = QtWidgets.QGroupBox("Event")
        rlay = QtWidgets.QVBoxLayout(readout_box)
        rlay.setContentsMargins(8, 6, 8, 8)
        # THE STALENESS SENTENCE, in a note that owns its lines -- see _FixedLinesNote.
        # First in the box, under its title, so it stands in one place whatever the mono
        # readout below it grows into (a labels line, a level table).
        self._stale_label = _FixedLinesNote(_STALE_NOTE_LINES)
        self._stale_label.setStyleSheet(f"color: {_MSG_HEX};")
        rlay.addWidget(self._stale_label)
        self._readout = QtWidgets.QLabel("")
        self._readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._readout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        mono = QtGui.QFont(self._readout.font())
        mono.setStyleHint(QtGui.QFont.Monospace)
        mono.setFamily("monospace")
        self._readout.setFont(mono)
        rlay.addWidget(self._readout)
        root.addWidget(readout_box)

        self.setCentralWidget(central)

    def _install_project_save_controls(self) -> None:
        """Install visible project state and standard Save / Save As gestures.

        The actions are scoped to this top-level window, so they remain available while
        the inspector is active without competing with the main window's actions. The
        compact split button exposes both pointer paths without rearranging either of the
        inspector's navigation/curation rows. Both actions emit intent; only MainWindow
        serializes the shared session.
        """
        self.act_save_project = QtGui.QAction("Save Project", self)
        self.act_save_project.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.act_save_project.setShortcutContext(Qt.WindowShortcut)
        self.act_save_project.setToolTip(
            "Save the complete analysis as a NanoScope project. Re-saves the current "
            "project in place once it has a path.")
        self.act_save_project.triggered.connect(
            lambda _checked=False: self.saveProjectRequested.emit())
        self.addAction(self.act_save_project)

        self.act_save_project_as = QtGui.QAction("Save Project As…", self)
        self.act_save_project_as.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.act_save_project_as.setShortcutContext(Qt.WindowShortcut)
        self.act_save_project_as.setToolTip(
            "Choose a new .nanoscope file for the complete analysis.")
        self.act_save_project_as.triggered.connect(
            lambda _checked=False: self.saveProjectAsRequested.emit())
        self.addAction(self.act_save_project_as)

        self._project_save_menu = QtWidgets.QMenu(self)
        self._project_save_menu.addAction(self.act_save_project)
        self._project_save_menu.addAction(self.act_save_project_as)

        self._project_save_button = QtWidgets.QToolButton(self)
        self._project_save_button.setDefaultAction(self.act_save_project)
        self._project_save_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._project_save_button.setMenu(self._project_save_menu)
        self._project_save_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._project_save_button.setToolTip(
            "Save the complete analysis project. Use the arrow for Save Project As…")
        self._project_save_button.setAccessibleName("Save project")

        self._project_save_label = QtWidgets.QLabel(self)
        self._project_save_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._project_save_label.setAccessibleName("Project save status")

        save_bar = self.statusBar()
        save_bar.setSizeGripEnabled(False)
        save_bar.addPermanentWidget(self._project_save_label, 1)
        save_bar.addPermanentWidget(self._project_save_button)
        self.set_project_save_state(None, False)

    def set_project_save_state(self, path: str | None, unsaved: bool) -> None:
        """Show MainWindow's authoritative project path and unsaved state.

        This is presentation only. The inspector neither infers project dirtiness from
        its own edits nor clears it after requesting a save, so cancellation, failure, and
        deliberately incomplete saves stay visibly unsaved until the owner says otherwise.
        """
        name = Path(str(path)).name if path else ""
        unsaved = bool(unsaved)
        if unsaved and name:
            text = f"Project: Unsaved changes — {name}"
            tip = (f"This session contains changes newer than {name}. Save the project to "
                   "keep them for the next launch.")
        elif unsaved:
            text = "Project: Unsaved changes — no project file yet"
            tip = ("This analysis has not been written to a project file. Save the project "
                   "to keep it for the next launch.")
        elif name:
            text = f"Project: Saved — {name}"
            tip = f"All project-backed changes are saved in {name}."
        else:
            text = "Project: No project file"
            tip = ("No project file has been chosen. Save Project creates one when there is "
                   "analysis to keep.")
        self._project_save_label.setText(text)
        self._project_save_label.setToolTip(tip)
        font = self._project_save_label.font()
        font.setBold(unsaved)
        self._project_save_label.setFont(font)

    def set_vocabulary(self, vocab: LabelVocabulary) -> None:
        """Install the shared palette and arm its safe explicit shortcuts."""
        self._vocab = vocab
        self._sync_label_shortcuts()
        row = self._row_for_id(self._current_id)
        if row is not None:
            self._refresh_readout(row)

    def set_stale_columns(self, columns: tuple[str, ...] = (), note: str = "") -> None:
        """Mark ``columns`` as measured by a superseded estimator (clear with ``()``).

        The same seam, the same tuple and the same sentence as the gallery's
        :meth:`~nanoscope.gui.event_gallery.EventGalleryWindow.set_stale_columns`: the
        owner pushes the events table's OWN badge set (``REMEASURED_COLUMNS`` on a
        baseline or calibration change) and the banner's text, so the three surfaces that
        print these numbers cannot tell three stories about one frame. Measured before it
        existed: one knob in the Preprocess tab badged nine table headers ⚠ and raised
        "Depth, frac_depth, ECD, mean/max deviation and SNR are stale", while this window
        went on printing "depth: 122.37 pA", "SNR: 21.4", "baseline: -61.079 pA" with no
        mark -- and its only ⚠ was the dI/I0 > 1 glyph, which means something else. The
        readout marks each affected line; the sentence stands in the Event box's
        fixed-height note (:class:`_FixedLinesNote` -- spliced into the readout as wrapped
        lines it resized the waveform plot under the pointer on every flip, 531 -> 447 px
        at 1000x800, and that plot is what this window is for); the plot title marks its
        depth and dI/I0. Kept across re-feeds, as the table's model keeps its own: the
        owner's staleness check runs on every re-feed and pushes the answer again.
        """
        cols = tuple(str(c) for c in columns)
        if cols == self._stale_columns and str(note) == self._stale_note:
            return
        self._stale_columns = cols
        self._stale_note = str(note)
        self._stale_label.setText(f"⚠ {self._stale_sentence()}" if cols else "")
        row = self._row_for_id(self._current_id)
        if row is not None and self._df is not None:
            self._refresh_readout(row)
            self._plot.setTitle(self._event_title(self._df.iloc[row]))
            self._render_event(row, preserve_view=True)

    def stale_columns(self) -> tuple[str, ...]:
        return self._stale_columns

    def stale_note(self) -> str:
        """The sentence the Event box's note carries ("" when nothing is marked): the
        owner's banner text, or the columns named by this window when none was given."""
        return self._stale_sentence() if self._stale_columns else ""

    def _stale(self, *columns: str) -> str:
        """" ⚠" when any of ``columns`` is marked stale, else "" -- the events-table header's
        own suffix, on the readout line that repeats the header's number."""
        return " ⚠" if any(c in self._stale_columns for c in columns) else ""

    def apply_hotkey(self, ch: str) -> None:
        """Label the event on screen. One event at a time is the inspector's whole job — the
        gallery is where a hundred get labelled at once."""
        entry = self._vocab.for_hotkey(ch)
        if entry is None or self._df is None:
            return
        row = self._row_for_id(self._current_id)
        if row is None:
            return
        col = self._df.columns.get_loc("flag")
        before = self._df.iat[row, col]
        after = set_label(before, entry.key, entry.value)
        label_changed = after != (before if isinstance(before, str) else "")
        if label_changed:
            self._df.iat[row, col] = after
        # Applying a structured label is a human inspection even when that exact value
        # was already present. Otherwise a labelled positive remains ``unreviewed`` and
        # silently disappears from the recommended training export.
        review_changed = mark_reviewed(
            self._df, rows=np.array([row], dtype=np.int64))
        if label_changed or review_changed:
            self._refresh_readout(row)
            self.labelsChanged.emit()

    def _install_shortcuts(self) -> None:
        """Keyboard on this window: ←/→ prev/next, A accept, R reject, Space toggle,
        T show the event in the main trace, 1–9 apply the corresponding label from the palette.

        THE CONTEXT IS ``WidgetWithChildrenShortcut``, NOT ``WindowShortcut``, AND THAT IS
        LOAD-BEARING. ``WindowShortcut`` asks only "is my window the active one?", which the
        events table's own A/R handler cannot outrank: measured at 1512×900 with this window
        open and ACTIVE while six rows were shift-selected in the events table, an R aimed at
        the TABLE was answered here instead — the table's six highlighted rows were untouched
        and event 7, which the user had selected nowhere, was silently rejected. A curation
        write to a row nothing on screen shows as chosen is the worst shape this app has.
        ``WidgetWithChildrenShortcut`` asks the question that actually decides who the key was
        meant for: does the focus sit inside this window? The gallery was narrowed for the
        same reason and carries the same note.

        The price, and why :meth:`_handle_own_key` exists below: Qt's shortcut map wants
        ``QApplication.focusWidget()`` to be a descendant of this window, and there are two
        ordinary states where it is not — a window shown but never activated has no focus
        widget at all, and a key delivered by ``QApplication.sendEvent`` rather than by the
        platform never reaches the shortcut map. The fallback restores exactly the reach the
        window-context shortcut had, without taking the events table's keys.
        """
        self._shortcuts: list[QtGui.QShortcut] = []

        def add(seq, slot) -> None:
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

        add(Qt.Key_Left, self.prev_event)
        add(Qt.Key_Right, self.next_event)
        add(Qt.Key_A, lambda: self.set_accepted_current(True))
        add(Qt.Key_R, lambda: self.set_accepted_current(False))
        add(Qt.Key_Space, self._toggle_accept_current)
        add(Qt.Key_T, self._on_show_in_trace)
        # 1-9 label the browsed event from the shared palette (see analysis.labels). Same
        # digits, same meanings, as the gallery — a curator who learns them once uses them
        # in both places. Explicit non-digit keys are rebuilt by
        # ``_sync_label_shortcuts`` whenever the vocabulary changes.
        #
        # Window-level is safe here for the same reason as in the gallery: a focused spin box
        # or line edit gets the key first, so the "Event N" box is still typeable. Verified
        # by driving both.
        for _n in range(1, 10):
            add(QtGui.QKeySequence(str(_n)), lambda ch=str(_n): self.apply_hotkey(ch))
        self._custom_label_shortcuts: list[QtGui.QShortcut] = []

    def _sync_label_shortcuts(self) -> None:
        """Rebuild explicit, non-digit label keys from the current vocabulary.

        Digits stay installed permanently because they are the automatic palette slots.
        ``LabelVocabulary.hotkeys`` is the single collision/reservation gate, so no
        duplicate or curation key can enter Qt's shortcut map here.
        """
        custom = getattr(self, "_custom_label_shortcuts", [])
        if custom:
            self._shortcuts = [shortcut for shortcut in self._shortcuts
                               if shortcut not in custom]
            for shortcut in custom:
                shortcut.setEnabled(False)
                shortcut.setParent(None)
                shortcut.deleteLater()
        self._custom_label_shortcuts = []
        for ch in self._vocab.hotkeys():
            if ch in "123456789":
                continue
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(ch), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda ch=ch: self.apply_hotkey(ch))
            self._shortcuts.append(shortcut)
            self._custom_label_shortcuts.append(shortcut)

    def _handle_own_key(self, event: QtGui.QKeyEvent) -> bool:
        """Run the shortcut THIS window registered for ``event``'s key; ``True`` if it was ours.

        THE SHORTCUTS STAY THE ONE REGISTRY. This does not re-list the keys — a second list is
        how a shortcut and its fallback drift into disagreeing about what R means — it asks the
        ``QShortcut`` objects ``_install_shortcuts`` already built which of them owns the
        combination, and activates that one. Disabled shortcuts are skipped.

        It exists because the narrowed ``WidgetWithChildrenShortcut`` context needs
        ``QApplication.focusWidget()`` to sit inside this window, and a window shown but never
        activated has no focus widget at all — the state every offscreen test and a
        just-opened window are in. Without this the narrowing would cost the inspector its own
        keyboard; with it, the inspector keeps every key it had and stops answering for the
        events table.
        """
        try:
            seq = QtGui.QKeySequence(event.keyCombination())
        except (TypeError, ValueError):  # pragma: no cover — key events carry one
            return False
        for sc in getattr(self, "_shortcuts", ()):
            if sc.isEnabled() and not sc.key().isEmpty() and sc.key() == seq:
                sc.activated.emit()
                return True
        return False

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 (Qt override)
        """Fall back to this window's own shortcuts when Qt's map could not reach them."""
        if self._handle_own_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    # -- data in ---------------------------------------------------------------------------------
    def set_events(self, doc_id: str, channel: int, events_df: pd.DataFrame | None,
                   module_id: str = "detect.schmitt",
                   provenance: dict[str, Any] | None = None,
                   refusals: "_RefusalLedger | None" = None) -> None:
        """Install the current detection. Stores the doc/channel and a REFERENCE to the SAME
        ``events_df`` object (never a copy), so an accept/reject here is shared with the table.
        Resets the filter to All and shows the first event (without emitting a selection — opening
        the browser must not yank the main trace).

        ``provenance`` is the detection's provenance dict (from the EventSet); a mask edit reads
        the detector's ``polarity`` / ``noise_rms_pA`` / ``filter_rise_s`` from it so a
        re-characterized event stays consistent with how the detector measured the rest. It is
        optional — absent, the edit measures those from the event's neighbourhood instead.

        ``refusals`` is the refusal ledger to adopt WITH the frame — in the wired app,
        MainWindow's per-detection-lineage ledger, shared with the gallery and the events
        table so all three surfaces disclose one record in one voice. Handing it in is the
        owner saying whether the standing record survives this re-feed: the same ledger
        arrives when the lineage did (an undo restored the detection, a merge edited it —
        measured, both used to erase the note while every counted row stayed refused; and
        a window OPENED after the gesture never showed it at all), a fresh one when a new
        detection replaced it. Without it (a standalone window) this method keeps its own
        rule: a genuinely new frame drops the window's private ledger."""
        if events_df is None:
            self.clear()
            return
        if refusals is not None:
            # Adopted before ANY early return: the same-frame re-feed below must also
            # leave every surface reading the one shared record.
            self._refusals = refusals
        same_frame = (events_df is self._df
                      and str(doc_id) == self._doc_id
                      and self._channel == int(channel))
        # A DIFFERENT detection invalidates the level fit. MainWindow calls this on every
        # selection change and the detection "may be a different detection", so without
        # this the levels fitted on recording A were printed under recording B's events --
        # matched by event_id, which both tables have and which means nothing across
        # recordings -- and the readout's disagreement line then actively argued for the
        # wrong number. A level fit belongs to exactly the frame it was fitted on.
        if not same_frame:
            self._levels = None
            # ...and every recorded reason WHY a level group was missing. Those sentences
            # name event ids, and an id means nothing across detections — carried over they
            # would explain the new frame's events with the old frame's edits.
            self._level_gaps.clear()
            # ...and so does a context framed by hand: it is a count of SAMPLES, which means a
            # different span of time on a recording sampled at another rate. The combo's ± ms
            # is rate-independent and stays, because it is a preference about looking.
            self._manual_pad = None
        self._doc_id = str(doc_id)
        self._channel = int(channel)
        self._df = events_df                 # shared frame — the single source of truth
        self._module_id = str(module_id)
        self._region.setMovable(self._module_id != CONSENSUS_MODULE_ID)
        self._provenance = dict(provenance) if provenance else None
        self._units = self._resolve_units()
        if (same_frame and self._current_id is not None
                and self._row_for_id(self._current_id) is not None):
            # Re-fed the SAME frame (e.g. a sibling fit result fired resultsChanged): keep the
            # user's filter + browsed event instead of yanking them back to the first event.
            self._render_and_sync(self._current_id, emit=False)
            return
        # A genuinely new (or empty) detection browses in All from the first event -- and a
        # note counting the PREVIOUS detection's unmeasurable rows is about events that no
        # longer exist. (The same-frame re-feed above deliberately keeps it: that path is
        # exactly the refresher this note has to survive.) When the owner handed a ledger
        # in, IT decided lineage and the private-ledger rule stands down.
        if refusals is None:
            self._clear_refusals()
        self._filter = _FILTER_ALL
        self._current_id = None
        if len(self._df) == 0:
            self._show_empty("Detection found no events.")
            return
        first = int(self._df["event_id"].iloc[0])
        self._render_and_sync(first, emit=False)

    def clear(self) -> None:
        """Empty the inspector (no detection). Safe to call at any time."""
        self._doc_id = None
        self._channel = None
        self._df = None
        self._module_id = None
        self._provenance = None
        self._levels = None
        self._level_gaps.clear()
        self._units = ""
        self._current_id = None
        self._filter = _FILTER_ALL
        # The hand-framed context is in SAMPLES of the recording that is going away; the combo
        # choice is in milliseconds and survives, because it is a preference about looking.
        self._manual_pad = None
        self._clear_refusals()
        self._show_empty(_NO_DETECTION_MSG)

    def set_levels(self, levels_df: pd.DataFrame | None) -> None:
        """Install (or drop) the CUSUM level table whose per-level breakdown the readout shows.

        Separate from :meth:`set_events` because the two have different lifetimes: a level
        fit is an on-demand derived result that can arrive long after the detection, be
        re-run with different parameters, or be dropped without touching the events. Passing
        ``None`` returns the readout to the ``n_levels`` count alone.
        """
        self._levels = levels_df if levels_df is not None and not levels_df.empty else None
        # A GROUP THAT CAME BACK ANSWERS ITS OWN "why is there none". Forgetting the
        # recorded reason here is what keeps the round trip clean: drag event 0 down to 9
        # samples (note: "below the 10 a level fit needs"), widen it back (7 levels
        # re-fitted), then MERGE it — the merge drops the group with no re-fit, and without
        # this the window would resurrect the sample-count sentence to explain a
        # disappearance that had nothing to do with it. `_level_gap_note` also refuses to
        # render over a live group, so the two rules fail closed in the same direction.
        if self._levels is not None and self._level_gaps:
            live = set(self._levels["event_id"].astype("int64").tolist())
            for eid in [k for k in self._level_gaps if int(k) in live]:
                self._level_gaps.pop(eid, None)
        if self._df is not None and self._current_id is not None:
            row = self._row_for_id(self._current_id)
            if row is not None:
                # RE-RENDER, not just re-word. The levels are DRAWN on the waveform as
                # well as listed in the readout, and refreshing only the text left the
                # plot showing no levels until the user happened to navigate away and
                # back -- i.e. the fit looked like it had not run on the very event they
                # were looking at when they ran it.
                self._render_event(row)
                self._refresh_readout(row)
        self._sync_level_note()

    def note_level_gap(self, event_id: int, reason: str) -> None:
        """Record (or, with an empty ``reason``, forget) WHY ``event_id`` has no levels.

        The caller is ``MainWindow._refit_levels_for_event``, which is the only object that
        knows which of the five routes a mask edit actually took — a cold cache and a fit
        that raised are not derivable from anything this window holds. It posts the same
        sentence to the status bar, so the two surfaces cannot prescribe different remedies
        for one event.

        Recording is not showing: :meth:`_level_gap_note` renders a reason only while the
        event genuinely has no rows in :attr:`_levels`. That is deliberate — a note is one
        signal handler away from outliving the disappearance it explains, and a stale "no
        levels" sentence sitting over a restored staircase is a worse bug than the silence
        it replaced.
        """
        eid = int(event_id)
        # A REASON CONTRADICTED BY THE TABLE IN HAND IS NOT RECORDED AT ALL. Same rule
        # `set_levels` applies when a group comes back, stated at the other door so the
        # dict never holds a sentence that could not be shown: whoever is telling this
        # window why there are no levels for an event that visibly has some is describing
        # an edit that has already been superseded.
        if reason and not self._has_levels(eid):
            # STAMPED with the bounds it is true of — the sentences quote sample counts and
            # spans, so a reason outlives its own accuracy the moment the mask moves again.
            self._level_gaps[eid] = (*self._bounds_of(eid), str(reason))
        else:
            self._level_gaps.pop(eid, None)
        row = self._row_for_id(self._current_id)
        if row is not None:
            self._refresh_readout(row)      # the CUSUM block carries the reason too
        self._sync_level_note()

    def level_note(self) -> str:
        """The sentence this window is CURRENTLY showing about missing levels ("" for none).

        Public for the same reason :meth:`refusal_note` is: a disclosure the suite can only
        reach through a private widget is one that can be silently stopped being rendered.
        """
        return self._level_note.text() if self._level_note.isVisible() else ""

    def _bounds_of(self, event_id: int) -> tuple[int | None, int | None]:
        """``(start_idx, end_idx)`` for one event, or ``(None, None)`` when it is not in the
        frame. ``None`` means "unstamped", and an unstamped reason never expires — better a
        sentence that outlives its span than one silently thrown away on a lookup failure."""
        row = self._row_for_id(event_id)
        if row is None or self._df is None:
            return (None, None)
        r = self._df.iloc[row]
        return (int(r["start_idx"]), int(r["end_idx"]))

    def _has_levels(self, event_id: int) -> bool:
        """Does the standing level table hold a group for ``event_id``?

        ONE predicate, read by everything that decides whether an absence needs
        explaining — the note, the readout block and the recorder. Written out because the
        two halves are easy to get subtly different: ``self._levels`` may be ``None``, and
        the join must be on int64 (the level table's ``event_id`` and the event frame's
        have arrived as different integer widths from a project restore).
        """
        if self._levels is None:
            return False
        return not self._levels[
            self._levels["event_id"].astype("int64") == int(event_id)].empty

    def _level_gap_note(self, event_id: "int | None") -> str:
        """The one sentence this window owes the user about a MISSING level group.

        Ordered by how much the speaker knows:

        1. **A live group** — nothing to explain, whatever anyone recorded. This is the
           guard that makes the round trip safe: widen the mask, the re-fit splices the
           group back, and the note is gone on the same repaint as the staircase returning.
        2. **A recorded reason** wins next: MainWindow watched the attempt. It is retired
           the moment the span it was stamped against moves, because these sentences quote
           sample counts — see :attr:`_level_gaps`. (That retirement PRUNES the entry, so
           this reader has one small write in it; it is idempotent and keeps the dict from
           accumulating sentences no surface could ever show.)
        3. **Derived**, for the groups nothing announced. A rejected row and a span below
           ``MIN_SAMPLES_FOR_LEVELS`` are both readable straight off the frame this window
           already holds, and they are the two cases where "re-run the fit" would be the
           wrong advice — so deriving them here is not a second copy of MainWindow's
           decision, it is this window answering from the data in its hands.
        4. **Silence** when no fit is standing at all — including after the whole fit has
           retired. A level fit is on demand, so most of the time there is no table and the
           window must look exactly as it did before the feature existed; "there was one a
           moment ago" is not distinguishable here from "there has never been one" without
           MainWindow saying so, and guessing would put this sentence under every event of
           every session that simply never ran the fit.
        """
        if event_id is None or self._df is None:
            return ""
        ev = int(event_id)
        if self._has_levels(ev):
            return ""
        told = self._level_gaps.get(ev)
        if told is not None:
            lo, hi, sentence = told
            if lo is None or (lo, hi) == self._bounds_of(ev):
                return sentence
            # THE SPAN MOVED UNDER THE SENTENCE. Measured on the single-event case: with
            # the whole fit retired, widening the mask back re-fits nothing, and the window
            # went on reporting "its new span is 9 sample(s)" over a 600-sample event. Drop
            # it and answer from what is true now.
            self._level_gaps.pop(ev, None)
        row = self._row_for_id(ev)
        if row is None:
            return ""
        r = self._df.iloc[row]
        if self._levels is not None:
            if "accepted" in self._df.columns and not bool(r["accepted"]):
                return level_gap_rejected(ev, standing_fit=True)
            from ..analysis.cusum import MIN_SAMPLES_FOR_LEVELS

            span = int(r["end_idx"]) - int(r["start_idx"])
            if span < MIN_SAMPLES_FOR_LEVELS:
                return level_gap_too_short(ev, n_samples=span,
                                           min_samples=MIN_SAMPLES_FOR_LEVELS,
                                           standing_fit=True)
            return level_gap_unfitted(ev)
        return ""

    def _sync_level_note(self) -> None:
        """Rebuild the missing-levels label FROM STATE, never written over from outside.

        Called from every path that changes what is on screen — a navigation, a re-feed, a
        new level table, a recorded reason — so the note follows the browsed event instead
        of being a per-gesture artefact of whichever edit happened last.
        """
        note = self._level_gap_note(self._current_id)
        self._level_note.setText(note)
        self._level_note.setVisible(bool(note))

    def _resolve_units(self) -> str:
        if (self._provenance or {}).get("normalize_enabled"):
            return "dimensionless"
        if self._doc_id is None or self._channel is None:
            return ""
        try:
            if self._session.has_document(self._doc_id):
                return self._session.calibrated(self._doc_id, self._channel).units or ""
        except Exception:  # noqa: BLE001 — a readout label must never crash the window
            return ""
        return ""

    # -- navigation ------------------------------------------------------------------------------
    def next_event(self) -> None:
        self._step(1)

    def prev_event(self) -> None:
        self._step(-1)

    def _step(self, delta: int) -> None:
        """Move within the current filter subset. When the displayed event is not in the subset
        (e.g. just curated out of an Accepted-only view), snap to the nearest subset member in the
        step direction by row order."""
        if self._df is None:
            return
        subset = self._subset_ids()
        if not subset:
            return
        cur = self._current_id
        if cur is not None and cur in subset:
            pos = subset.index(cur) + delta
            if 0 <= pos < len(subset):
                self.show_event(subset[pos])
            return
        cur_row = self._row_for_id(cur)
        if cur_row is None:
            self.show_event(subset[0])
            return
        rows = [(self._row_for_id(i), i) for i in subset]
        if delta > 0:
            ahead = [i for r, i in rows if r is not None and r > cur_row]
            self.show_event(ahead[0] if ahead else subset[-1])
        else:
            behind = [i for r, i in rows if r is not None and r < cur_row]
            self.show_event(behind[-1] if behind else subset[0])

    def show_event(self, event_id: int) -> None:
        """Display the event with this id and emit ``eventSelected``. A no-op (no re-emit) when it
        is already the displayed event — that early return is what keeps the table<->inspector
        selection sync from looping."""
        if self._df is None:
            return
        row = self._row_for_id(int(event_id))
        if row is None:
            return
        if self._current_id == int(event_id):
            return
        self._render_and_sync(int(event_id), emit=True)

    def current_event_id(self) -> int | None:
        return self._current_id

    def event_count(self) -> int:
        """Number of events in the current filter subset (the N in 'Event i of N')."""
        return len(self._subset_ids())

    def set_filter(self, mode: str) -> None:
        """Set the browsed subset. Keeps the current event if it is still a member; otherwise
        homes to the first event of the new subset (or shows an empty-subset note)."""
        canon = _canonical_filter(mode)
        if canon is None:
            return
        self._filter = canon
        self._updating = True
        self._filter_combo.setCurrentText(canon)
        self._updating = False
        if self._df is None:
            return
        subset = self._subset_ids()
        cur = self._current_id
        if cur is not None and cur in subset:
            row = self._row_for_id(cur)
            if row is not None:
                self._refresh_readout(row)
            self._sync_controls()
        elif subset:
            self._current_id = None
            self._render_and_sync(subset[0], emit=True)
        else:
            self._current_id = None
            self._show_empty(f"No events match “{canon}”.")

    # -- curation --------------------------------------------------------------------------------
    def set_accepted_current(self, accepted: bool) -> None:
        """Curate the displayed event: write the shared df's ``accepted`` in place and emit
        ``acceptanceChanged(event_id, WHAT WAS WRITTEN)``. Does NOT navigate (the just-curated
        event stays on screen; the filter subset for Prev/Next reflects the new state).

        ``accepted`` IS AN AND EVERYWHERE, NOT ONLY WHERE A BULK GESTURE WROTE IT. Both bulk
        surfaces were guarded and this one was not, so the A key / the Accepted checkbox put a
        row the detector could not measure straight into the accepted set -- one row at a
        time, past the same rule "Accept all" had just enforced over the whole frame. The
        guard is :meth:`_set_df_accepted`, the same one call this method already made, so
        there is no second copy of the rule; what changed is that the checkbox, the readout
        and the outgoing signal all now follow WHAT WAS WRITTEN rather than what was asked
        for. Rejection is still never blocked, and a refusal is disclosed on
        :attr:`_batch_note` exactly as a bulk refusal is -- refusing in silence would be the
        M6b defect one gesture smaller.
        """
        if self._df is None or self._current_id is None:
            return
        row = self._row_for_id(self._current_id)
        if row is None:
            return
        # AN ACCEPT RE-DECIDES THIS EVENT AND NO OTHER; A REJECT RE-DECIDES NOTHING. This
        # used to clear the whole record, which is the M6d defect at single-event size: R
        # on one ordinary event took down a disclosure about three OTHER events that were
        # still refused. See :class:`_RefusalLedger`.
        if bool(accepted):
            self._refusals.forget((int(self._current_id),))
        col = self._accepted_col()
        before = bool(self._df.iat[row, col])
        written = self._set_df_accepted(row, bool(accepted))
        if bool(accepted) and "flag" not in self._df.columns:
            # The accept was refused before a decision could be made.  Recording review
            # here would turn an invalid request into a resolved training label and emit
            # a data-change signal; in the wired app that signal also re-feeds the
            # inspector and wipes the missing-flag disclosure it just raised.
            self._sync_accept_check(written)
            self._refresh_readout(row)
            self._sync_controls()
            self.acceptanceRefused.emit(int(self._current_id))
            return
        # The click is a human ruling even when the detector already had the requested
        # boolean value.  In particular, Accept on a fresh accepted candidate must make
        # it eligible for the recommended reviewed-only training export.
        conflict_changed = clear_curation_conflicts(self._df, (row,))
        review_changed = mark_reviewed(
            self._df, rows=np.array([row], dtype=np.int64))
        self._sync_accept_check(written)
        self._refresh_readout(row)
        self._sync_controls()
        if written != before or review_changed or conflict_changed:
            # NOTHING WRITTEN, NO DATA SIGNAL -- the gallery's ``_set_rows_accepted`` has
            # said this since M6 and this path had not: a signal with no change behind it
            # reaches the dirty flag and the undo recorder, and (measured) it also sends
            # MainWindow's surface refresher back round to re-feed this window, which is
            # what silently erased the refusal note the refusal had just produced.
            self.acceptanceChanged.emit(int(self._current_id), written)
        if written == before and bool(accepted) and not written:
            # An accept was REFUSED and nothing changed: ``_set_df_accepted`` recorded it
            # on the shared ledger (kept — the row stays rejected — or blocked on a
            # flagless frame), and no data signal will fire. The ledger changed all the
            # same, so it is announced: windows announce ledger changes, not just written
            # rows. This is the dedicated refusal signal, NOT acceptanceChanged — the
            # dirty flag and the undo recorder rightly never hear about a no-write.
            self.acceptanceRefused.emit(int(self._current_id))

    def reflect_acceptance(self, event_id: int, accepted: bool) -> None:
        """Mirror an acceptance decision made ELSEWHERE (a table checkbox) onto this window's
        control + readout. The shared df is already written by the caller, so this only syncs the
        UI and NEVER re-emits ``acceptanceChanged`` (it is the inbound side of the sync).

        IT NO LONGER TOUCHES THE NOTE. It used to zero the whole counter whenever the
        mirrored verdict accepted a row the counter had counted -- right only when the counter
        held exactly one row, and on the owner's -600 mV recording it held four. The counter
        is recomputed from the rows (:attr:`_refused`), and the caller has already written the
        frame, so the accepted row leaves the count on its own and takes nothing with it.
        """
        self._sync_controls()   # a row's accepted state moved -> subset sizes may have shifted
        if self._current_id is None or int(event_id) != int(self._current_id):
            return
        self._sync_accept_check(bool(accepted))
        row = self._row_for_id(self._current_id)
        if row is not None:
            self._refresh_readout(row)

    def _toggle_accept_current(self) -> None:
        if self._df is None or self._current_id is None:
            return
        row = self._row_for_id(self._current_id)
        if row is None:
            return
        col = self._df.columns.get_loc("accepted")
        self.set_accepted_current(not bool(self._df.iat[row, col]))

    def _on_accept_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self.set_accepted_current(bool(checked))

    def _on_merge_with_next(self) -> None:
        """Announce that the displayed event should be merged with the following event(s). The
        owner (via MainWindow) runs the merge on the shared EventSet and advances the browser; a
        no-op forward is handled there. Guarded to a real displayed event."""
        if (self._df is None or self._current_id is None
                or self._module_id == CONSENSUS_MODULE_ID):
            return
        self.mergeWithNextRequested.emit(int(self._current_id))

    def _on_show_in_trace(self) -> None:
        """Announce that the displayed event should be shown in the MAIN trace ("Show in
        trace" / T). Intent only: this window has no business moving another surface's view,
        so MainWindow routes it to ``_goto_event_in_trace`` — the same jump the gallery's
        double-click makes. Guarded to a real displayed event."""
        if self._df is None or self._current_id is None:
            return
        self.traceRequested.emit(int(self._current_id))

    def _set_df_accepted(self, row: int, accepted: bool) -> bool:
        """Write ONE row's ``accepted`` cell in place (positional, like the table model), as
        ``derived AND requested``, and return the value that actually landed.

        THE PER-ROW HALF OF THE SAME AND. ``_curated_column`` applies it to the whole frame
        and :func:`_derived_accepted` states it once; this is that same function asked about
        a single row, not a fourth copy of the rule. Writing ``requested`` here was an OR in
        everything but name, exactly as it was at the bulk surfaces before M6: it could only
        ever turn a rejection into an acceptance.

        FAIL CLOSED on a frame with no ``flag`` column, for the reason
        :meth:`_set_all_accepted` does: nothing vouches for the row, so the accept is refused
        and the cell is left alone. Rejection is never blocked -- ``requested=False`` is the
        curator's own half going to False, which ANDs to False whatever the row says.

        The refusal is RECORDED, not announced: the caller renders it through the same
        :attr:`_refused` counter every other gesture uses.
        """
        col = self._df.columns.get_loc("accepted")
        target = bool(accepted)
        if target:
            if "flag" not in self._df.columns:
                self._refusals.record_blocked(1)
                return bool(self._df.iat[row, col])       # nothing written at all
            target = _derived_accepted(self._df.iat[row, self._df.columns.get_loc("flag")])
            if not target:
                # Same two verbs as the bulk path, decided the same way: on where the row
                # STARTED. A row sitting accepted beside a rejecting flag is repaired, and the
                # accepted count goes DOWN on a gesture that said "accept".
                eid = (int(self._df["event_id"].iloc[row]),)
                if bool(self._df.iat[row, col]):
                    self._refusals.record(repaired=eid)
                else:
                    self._refusals.record(kept=eid)
        self._df.iat[row, col] = target
        return target

    # -- batch curation --------------------------------------------------------------------------
    def accept_all(self) -> None:
        """Accept every event the detector was able to measure (the whole shared df, not just
        the filtered subset). Rows the detector flagged as unmeasurable stay rejected -- see
        :func:`_derived_accepted`."""
        self._set_all_accepted(True)

    def reject_all(self) -> None:
        """Mark EVERY event rejected (the whole shared df, not just the filtered subset)."""
        self._set_all_accepted(False)

    def _curated_column(self, requested: bool) -> np.ndarray:
        """The whole ``accepted`` column under this bulk decision: ``derived AND requested``.

        THE SAME AND ``characterize`` APPLIES, at the one place this window writes the column
        wholesale. It used to write ``requested`` over every row -- which is an OR in
        everything but name, because it could only ever turn a rejection into an acceptance --
        and a GUI action must not be able to express something the schema forbids.

        Rejecting is unconditional (``requested=False`` is the curator's half going to False,
        and False ANDs to False whatever the row says), so only the accepting direction reads
        the flags at all.

        A frame with no ``flag`` column has NO answer here, and ``np.ones`` was the wrong one:
        it is the pre-fix broadcast, restored by the guard itself, on any frame that has
        wandered off the frozen 20-column schema. :meth:`_set_all_accepted` now refuses the
        accept before reaching this method, so the accepting branch is only ever asked a
        question the frame can answer -- and if it is asked anyway, it fails CLOSED.
        """
        n = len(self._df)
        if not requested:
            return np.zeros(n, dtype=bool)
        if "flag" not in self._df.columns:
            return np.zeros(n, dtype=bool)   # fail closed: nothing vouches for these rows
        return np.array([_derived_accepted(f) for f in self._df["flag"]], dtype=bool)

    def _set_all_accepted(self, accepted: bool) -> None:
        """Rewrite ``accepted`` for every row of the shared df in place, resync this window's
        current-event control + readout + nav, and announce it via ``curationBatchChanged`` so the
        owner resets the table + overlays + summary. A no-op when there is no detection.

        The values written are :meth:`_curated_column`'s, not ``accepted`` broadcast: a row the
        detector rejected on its own re-derivable grounds stays rejected, and the checkbox below
        is resynced FROM THE FRAME rather than from the requested value -- a window that ticked
        the box while the row underneath it said ``False`` would be the same defect moved one
        layer up.
        """
        if self._df is None or len(self._df) == 0:
            return
        rows = np.arange(len(self._df), dtype=np.int64)
        # AN ACCEPT RE-DECIDES ITS OWN SCOPE, WHICH HERE IS THE WHOLE FRAME; A REJECT
        # RE-DECIDES NOTHING. "Reject all" clearing the record is the M6d defect at its
        # largest: the identical whole-frame reject arriving from the OTHER window leaves
        # the note standing, so the app said two different things about one frame depending
        # on where the click landed. See :class:`_RefusalLedger`.
        if bool(accepted):
            self._refusals.clear()
        if bool(accepted) and "flag" not in self._df.columns:
            # FAIL CLOSED, and write nothing: the derived half cannot be read, so no row can
            # be vouched for -- and newly REJECTING them would be a second, destructive
            # answer to a question the user did not ask. Rejection is still never blocked.
            self._refusals.record_blocked(len(self._df))
            self._sync_controls()
            # Nothing was written, but the SHARED ledger just changed -- announce the
            # refusal itself (windows announce ledger changes, not just written rows),
            # so the owner repaints the other surfaces' notes. -1: whole-frame, no id.
            self.acceptanceRefused.emit(-1)
            return
        # The button is a human ruling on the whole table, including rows whose
        # ``accepted`` value already matches the request.  Detector output starts
        # accepted, so restricting this to changed booleans made a fresh "Accept all"
        # disappear from reviewed-only training exports.  A wholly invalid, flagless
        # accept returned above because no ruling was possible.
        conflict_changed = clear_curation_conflicts(self._df, rows)
        review_changed = mark_reviewed(
            self._df, rows=rows)
        before = self._df["accepted"].to_numpy(dtype=bool).copy()
        # Whole column, in place on the shared frame -- but the schema's value for each row.
        self._df["accepted"] = self._curated_column(bool(accepted))
        changed = np.flatnonzero(
            before != self._df["accepted"].to_numpy(dtype=bool))
        if bool(accepted):
            after = self._df["accepted"].to_numpy(dtype=bool)
            forbidden = ~after                      # requested True, derived False
            # WHICH EVENTS, not how many: the counts are recomputed on every rebuild of the
            # note (see :attr:`_refused`), so what is remembered here is only which bucket
            # each refused event belongs to -- that part IS history and cannot be re-derived.
            # Recorded by ``event_id`` and not by position, so a later merge cannot make the
            # record point at somebody else's events.
            eid = self._df["event_id"].to_numpy()
            self._refusals.record(kept=eid[forbidden & ~before],
                                  repaired=eid[forbidden & before])
        if self._current_id is not None:
            row = self._row_for_id(self._current_id)
            if row is not None:
                self._sync_accept_check(bool(self._df.iat[row, self._accepted_col()]))
                self._refresh_readout(row)
        self._sync_controls()
        self.curationBatchChanged.emit()

    def refusal_note(self) -> str:
        """The disclosure the LAST curation gesture owes the user, or ``""`` -- bulk or
        single-event, because a refusal is a refusal at any size.

        The inspector had NO disclosure at all: "Accept all" on the owner's -600 mV
        recording moved 316 of 317 rows and said so nowhere, so an honest refusal and a
        broken button read identically. Rendered by :meth:`_sync_controls` from
        :attr:`_refused`, so MainWindow's surface refresher -- which re-feeds this window on
        the very ``curationBatchChanged`` the gesture emits -- rebuilds it rather than
        erasing it.

        NOT cleared by the next gesture. An accept re-decides its own scope, a reject
        re-decides nothing, and only a new frame drops the record -- see
        :class:`_RefusalLedger`.
        """
        return _refusal_note(*self._refused, noun="event")

    # -- the refusal counter ----------------------------------------------------------------------
    def _clear_refusals(self) -> None:
        """Drop the whole ledger. A NEW FRAME ONLY -- never a gesture.

        This used to be the first line of every curation write, which is the M6d defect:
        a reject re-decides no refusal, so clearing here made the note a per-gesture
        artefact that one ordinary R keypress took down while three rows were still
        refused. What an ACCEPT re-decides, it re-decides by scope
        (``_RefusalLedger.forget``), not by wiping the record.

        DETACHED, never ``.clear()``ed: the instance may be the owner's SHARED ledger
        (see :meth:`set_events`), read by the gallery and the events table as well, and
        whether that record dies is the owner's lineage decision -- this window may only
        stop holding it. For a private ledger the two are the same thing.
        """
        self._refusals = _RefusalLedger()

    def _still_refused(self, row: int) -> bool:
        """Is ``row`` STILL in the state the note describes -- rejected, and rejected because
        the detector could not measure it?

        Both halves are re-read from the frame, so this is a question about NOW and not about
        the gesture: a row the events table has since accepted answers ``False`` and drops out
        of the count, and it drops out ALONE.

        THE SAME counter :attr:`_refused` uses, asked about one row -- never a second
        implementation of the rule beside it.
        """
        df = self._df
        if df is None or not (0 <= row < len(df)):
            return False
        return _RefusalLedger.refused_in(df, (int(df["event_id"].iloc[row]),)) == 1

    @property
    def _refused_kept(self) -> list[int]:
        """The event ids in the ``kept rejected`` bucket -- READ-ONLY, and a membership list
        rather than a count. Empty after a new frame; see :class:`_RefusalLedger`."""
        return sorted(self._refusals.kept)

    @property
    def _refused_repaired(self) -> list[int]:
        """The event ids in the ``corrected back to rejected`` bucket. Read-only."""
        return sorted(self._refusals.repaired)

    @property
    def _refused(self) -> tuple[int, int, int]:
        """``(kept, repaired, blocked)`` AS OF NOW -- recounted, never remembered.

        THE INVARIANT this counter broke, twice: it is a count of currently-refused rows,
        not a latch AND NOT A PER-GESTURE ARTEFACT. The first fix stored the three counts at
        the moment of the gesture and had ``reflect_acceptance`` ZERO all three the first
        time the events table accepted any one of the counted rows -- so ticking one of four
        unmeasurable rows took the whole disclosure down while THREE rows were still refused.
        The second recounted, but still cleared the record at the top of this window's own
        write path, so the same disclosure died to any R keypress. Recounting from a ledger
        that only a new frame clears makes the arithmetic impossible to get wrong: one row
        leaves the count, the others stay in it, the count goes back UP when it is rejected
        again, and the same recount runs on every rebuild of the line rather than at one
        privileged moment.
        """
        return self._refusals.counts(self._df)

    def _accepted_col(self) -> int:
        return int(self._df.columns.get_loc("accepted"))

    def _neighbour_bounds(self, row: int, cur_start: int, cur_end: int) -> tuple[int, int]:
        """``(lo_limit, hi_limit)`` — the samples a mask drag on ``row`` may not cross.

        ``lo_limit`` is the largest ``end_idx`` among the events that begin BEFORE this one and
        ``hi_limit`` the smallest ``start_idx`` among those that begin after; either is the
        recording's own edge when there is no such neighbour. Computed from the frame rather
        than from ``row ± 1`` because the browsed order is a filter subset and the frame is not
        guaranteed sorted — a merge and a hand-added event both write rows out of position.

        The limits are never tighter than the event's OWN current bounds: on a frame that
        already overlaps (one written before this clamp existed) a limit inside the event would
        move an edge the user did not drag, which is an edit nobody asked for. The clamp's job
        is to stop an overlap being created, not to repair one silently.
        """
        no_neighbour = 1 << 62               # "no neighbour" -> the [0, n] clip still applies
        lo_limit, hi_limit = 0, no_neighbour
        if self._df is None or len(self._df) < 2:
            return lo_limit, hi_limit
        starts = self._df["start_idx"].to_numpy(dtype=np.int64)
        ends = self._df["end_idx"].to_numpy(dtype=np.int64)
        others = np.ones(starts.size, dtype=bool)
        others[row] = False
        before = others & (starts <= int(cur_start))
        after = others & (starts > int(cur_start))
        if before.any():
            lo_limit = min(int(ends[before].max()), int(cur_start))
        if after.any():
            hi_limit = max(int(starts[after].min()), int(cur_end))
        return lo_limit, hi_limit

    # -- mask editing (resize an event's [start, end]) -------------------------------------------
    def _on_region_edited(self) -> None:
        """A drag of the region's edge(s) finished: map both edges to sample indices and hand
        them to :meth:`_apply_edited_bounds`, which owns everything from the neighbour clamp
        onward (re-characterisation on the shared df, the redraw and ``eventEdited``). This
        method's own job is the x-to-sample mapping, and only that — Revert supplies the same
        two numbers from the detection instead of from the mouse and goes through the same
        writer. Programmatic region sets (from :meth:`_render_event`) are fenced off by
        ``_syncing`` so they never masquerade as a user edit."""
        if (self._syncing or self._df is None or self._current_id is None
                or self._module_id == CONSENSUS_MODULE_ID):
            return
        if self._doc_id is None or self._channel is None:
            return
        if not self._session.has_document(self._doc_id):
            return
        row = self._row_for_id(self._current_id)
        if row is None:
            return
        ch = self._session.calibrated(self._doc_id, self._channel)
        data = np.asarray(ch.data)
        n = int(data.size)
        if n == 0:
            return
        x0, x1 = self._region.getRegion()
        new_start = int(np.clip(self._x_to_index(min(float(x0), float(x1))), 0, n))
        new_end = int(np.clip(self._x_to_index(max(float(x0), float(x1))), 0, n))
        self._apply_edited_bounds(row, new_start, new_end, data)

    def _apply_edited_bounds(self, row: int, new_start: int, new_end: int,
                             data: np.ndarray) -> None:
        """Move event ``row`` to ``[new_start, new_end)``: clamp, re-measure, redraw, announce.

        *** THE ONE WRITE PATH FOR A CHANGE OF BOUNDS, AND IT HAS EXACTLY TWO CALLERS. ***
        The mask drag (:meth:`_on_region_edited`, which contributes the x-to-sample mapping)
        and Revert (:meth:`_revert_current`, which contributes the detector's own bounds).
        Everything after "which two samples" is identical for both and is here: the neighbour
        clamp, the degenerate-drag refusal, the re-characterisation through
        :meth:`_recharacterize_row`, the redraw and the ``eventEdited`` emission.

        It is factored rather than duplicated because a SECOND writer is precisely how the
        level reconcile and the characterisation got out of step before: ``_write_measured_row``
        is what calls :meth:`_reconcile_levels`, and ``eventEdited`` is what makes MainWindow
        re-fit the levels, checkpoint the undo step and refresh the table. A revert that wrote
        the two index columns itself would look correct on screen and leave the level table
        describing the edited span, the undo stack short one step and the events table stale.
        """
        if (self._df is None or self._current_id is None
                or self._module_id == CONSENSUS_MODULE_ID):
            return
        scol = self._df.columns.get_loc("start_idx")
        ecol = self._df.columns.get_loc("end_idx")
        cur_start = int(self._df.iat[row, scol])
        cur_end = int(self._df.iat[row, ecol])
        n = int(np.asarray(data).size)
        new_start = int(np.clip(int(new_start), 0, n))
        new_end = int(np.clip(int(new_end), 0, n))
        # NEIGHBOUR CLAMP. A drag was clipped to [0, n] and nothing else, so dragging an edge
        # across the adjacent event produced an OVERLAPPING event table — two rows claiming the
        # same samples. Nothing downstream is written for that: the trace draws the two shaded
        # spans on top of one another, and the merge grouping rule (which reasons about the gap
        # between one event's end and the next one's start) is handed a negative gap. The mask
        # stops at the neighbour's edge instead; a genuine over-split is joined with "Merge with
        # next", which is the gesture that exists for it.
        #
        # A REVERT IS CLAMPED TOO, and that is not a compromise. If a neighbour has since been
        # dragged over this event's original span, restoring it verbatim would recreate the
        # overlap the clamp exists to prevent — on a gesture the user reads as "undo my edit",
        # which is exactly when they would not be looking for it.
        lo_limit, hi_limit = self._neighbour_bounds(row, cur_start, cur_end)
        new_start = int(max(new_start, lo_limit))
        new_end = int(min(new_end, hi_limit))
        if new_end <= new_start:
            # A degenerate drag (edges crossed / collapsed): snap the region back, make no edit.
            self._render_event(row)
            return
        if new_start == cur_start and new_end == cur_end:
            return                          # rounded back onto the same samples — nothing to do
        fs = self._sample_rate()
        if fs is not None:
            # Re-measure on the trace the DETECTOR measured on, not the raw channel.
            try:
                conditioned = self._conditioned(data, float(fs))
            except Exception as exc:  # noqa: BLE001 — fail closed at the user-action boundary
                # No dataframe, fitted-level or amplitude write has happened. Put
                # the handles back without replaying the failing filter again;
                # the existing waveform and view remain available for browsing.
                xvals = self._session.x_axis(self._doc_id).values
                self._syncing = True
                try:
                    self._region.setRegion((
                        float(xvals[min(cur_start, len(xvals) - 1)]),
                        float(xvals[min(cur_end, len(xvals) - 1)]),
                    ))
                finally:
                    self._syncing = False
                self._refresh_readout(row)
                self._sync_revert_control()
                QtWidgets.QMessageBox.warning(
                    self, "Event bounds unchanged",
                    "The saved signal conditioning could not be reconstructed. "
                    "No event bounds, measurements or curation were changed. "
                    "Correct the saved analysis settings or run detection again.\n\n"
                    + str(exc),
                )
                return
            self._recharacterize_row(row, conditioned, float(fs), new_start, new_end)
        else:
            # Rate unknown (legacy TDMS): detection cannot have produced dwell-bearing events, but
            # keep the bounds honest anyway (dwell/depth stay whatever they were).
            self._df.iat[row, scol] = int(new_start)
            self._df.iat[row, ecol] = int(new_end)
        invalidate_event_amplitudes(self._df, [row], reason="event boundaries changed")
        self._render_event(row)             # redraw region / baseline / level at the new bounds
        self._refresh_readout(row)
        self._sync_revert_control()         # the row may now BE the detector's own span
        self.eventEdited.emit(int(self._current_id))

    # -- reverting to the detector's own bounds ---------------------------------------------------
    def _revert_current(self) -> None:
        """Put the browsed event back on the bounds the DETECTOR produced (owner note E1).

        The original span is read through :func:`~nanoscope.analysis.events.original_bounds`,
        the one reader, and the move is applied through :meth:`_apply_edited_bounds`, the one
        writer — so a revert is re-characterised, reconciles the level table, triggers the
        single-event level re-fit and lands on the undo stack exactly as a mask drag does. It
        is not a special kind of edit; it is an edit whose two numbers came from the detection
        instead of from the mouse.

        Silent no-op when the row has no recorded original (the control is disabled then) or
        when the event already sits on it — there is nothing to revert and writing anyway would
        put a pointless step on the undo stack.
        """
        if (self._df is None or self._current_id is None
                or self._module_id == CONSENSUS_MODULE_ID):
            return
        if self._doc_id is None or self._channel is None:
            return
        if not self._session.has_document(self._doc_id):
            return
        row = self._row_for_id(self._current_id)
        if row is None:
            return
        bounds = original_bounds(self._df.iloc[row])
        if bounds is None:
            return
        data = np.asarray(self._session.calibrated(self._doc_id, self._channel).data)
        if data.size == 0:
            return
        self._apply_edited_bounds(row, bounds[0], bounds[1], data)

    def _revert_state(self, row: int | None) -> tuple[bool, str]:
        """``(enabled, tooltip)`` for the Revert control on ``row``. The control's ONE rule.

        Three cases, and the disabled ones have to SAY WHICH, because they mean opposite
        things to the user:

        * no recorded original — this detection was saved before the app kept one. It is NOT
          backfilled from the current bounds: that would claim the detector produced whatever
          the last edit left behind. The tooltip says so, so a greyed button reads as "this
          table cannot answer" rather than "this build is broken".
        * an original that the event already sits on — nothing has been edited.
        * otherwise, enabled, naming the span it would restore.

        The label says "the detector's" rather than "the model's" (the owner's word) on
        purpose: ``module_id`` distinguishes the ML detector from the Schmitt trigger PER
        TABLE, so the honest promise is "whatever detector produced this table".
        """
        detector = _DETECTOR_NAMES.get(self._module_id or "", "the detector")
        if self._module_id == CONSENSUS_MODULE_ID:
            return False, _RECONCILED_EDIT_TIP
        if self._df is None or row is None:
            return False, _REVERT_TIP_UNKNOWN.format(detector=detector)
        bounds = original_bounds(self._df.iloc[row])
        if bounds is None:
            return False, _REVERT_TIP_UNKNOWN.format(detector=detector)
        scol = self._df.columns.get_loc("start_idx")
        ecol = self._df.columns.get_loc("end_idx")
        cur = (int(self._df.iat[row, scol]), int(self._df.iat[row, ecol]))
        if cur == bounds:
            return False, (
                f"This event is already on the bounds {detector} produced "
                f"(samples {bounds[0]}–{bounds[1]}); there is no edit to undo.")
        return True, (
            f"Put this event back on the bounds {detector} produced "
            f"(samples {bounds[0]}–{bounds[1]}, currently {cur[0]}–{cur[1]}). "
            "Re-measures and re-fits the sub-state levels exactly as a mask drag does, "
            "and is itself reversed by Edit ▸ Undo (⌘Z).")

    def _sync_revert_control(self) -> None:
        """Reconcile the Revert button with the browsed row. Called wherever the row or its
        bounds can have changed — navigation (:meth:`_sync_controls`) and an edit
        (:meth:`_apply_edited_bounds`)."""
        if not hasattr(self, "_revert_btn"):
            return
        row = self._row_for_id(self._current_id) if self._df is not None else None
        enabled, tip = self._revert_state(row)
        self._revert_btn.setEnabled(enabled)
        self._revert_btn.setToolTip(tip)

    def _recharacterize_row(self, row: int, data: np.ndarray, fs: float,
                            new_start: int, new_end: int) -> None:
        """Re-measure the single event now spanning ``[new_start, new_end)`` and write the recomputed
        SCHEMA columns back into df ``row`` — PRESERVING event_id / recording / pore_id.

        The measurement itself is :func:`~nanoscope.analysis.characterize.recharacterize_span`,
        the ONE row-rebuild path in the app (the merge and the hand-added-event path call the
        same function). This method used to be a private copy of it, and the copy blanked
        ``flag`` while keeping ``accepted=False`` — so resizing the mask of a boundary-clipped
        event left a REJECTED ROW WITH NO REASON: invisible to the ``clipped_by_boundary``
        filter and to the count that is supposed to find exactly those rows.

        The prior row is handed over as ``prior=``, so a curator's rejection survives the edit
        while the automatic rules (clipped / too long / zero baseline) are re-judged against the
        new bounds — which is the only way an edit can both keep a hand-reject and let an event
        dragged off the recording edge stop claiming it is clipped.

        ``data`` must be the CONDITIONED trace (see :meth:`_conditioned`), not the raw channel:
        the detector measured its rows on ``lowpass(preprocess_for_detection(raw))``, so
        re-measuring on the raw channel puts a different measurement in the same columns."""
        prior = self._df.iloc[row]
        recomputed = recharacterize_span(
            data, fs, new_start, new_end,
            provenance=self._provenance,
            prior=prior,
            units=self._units or "pA",
            pad_ms=_PAD_MS,      # identical to _pad_samples(): max(ev_len, fs*_PAD_MS/1000, 1)
        )
        self._write_measured_row(row, new_start, new_end, fs, recomputed.iloc[0])

    def _write_measured_row(self, row: int, new_start: int, new_end: int, fs: float,
                            crow: pd.Series) -> None:
        """Write the recomputed measurement columns into df ``row``. ``start_idx`` / ``end_idx`` /
        ``t_start_s`` are the ABSOLUTE new bounds (characterize measured them inside the local
        window); the rest are window-offset-independent. event_id / recording / pore_id are left
        untouched, so an event's IDENTITY survives an edit.

        ``accepted`` IS written, from the rebuilt row — it is not the detector's opinion being
        overwritten, it is the curator's own decision combined (with AND, never OR) with a fresh
        judgement of the new span. Leaving it alone is what let a rejection outlive the reason
        that produced it."""
        updates: dict[str, object] = {
            "start_idx": int(new_start),
            "end_idx": int(new_end),
            "t_start_s": float(new_start) / fs,
            "dwell_s": float(crow["dwell_s"]),
            "baseline_pA": float(crow["baseline_pA"]),
            "depth_pA": float(crow["depth_pA"]),
            "frac_depth": float(crow["frac_depth"]),
            "ecd_pA_s": float(crow["ecd_pA_s"]),
            "max_dev_pA": float(crow["max_dev_pA"]),
            "mean_dev_pA": float(crow["mean_dev_pA"]),
            "rise_s": float(crow["rise_s"]),
            "fall_s": float(crow["fall_s"]),
            "snr": float(crow["snr"]),
            "n_levels": int(crow["n_levels"]),
            "filter_limited": bool(crow["filter_limited"]),
            "accepted": bool(crow["accepted"]),
            "flag": str(crow["flag"]),
        }
        for name, value in updates.items():
            self._df.iat[row, self._df.columns.get_loc(name)] = value
        self._reconcile_levels()

    def _reconcile_levels(self) -> None:
        """Drop every CUSUM level group that no longer describes its event's span.

        Every measured column of an edited row is rebuilt from the NEW bounds, but the
        level table was fitted against the OLD ones and nothing recomputes it. Left in
        place it does not merely go slightly stale, it contradicts the row it sits under:
        dragging a 6000 us two-level event back to 2500 us leaves the readout printing two
        3000 us levels -- 3500 us of level inside an event that no longer contains them --
        while the disagreement line invites the user to trust the CUSUM count over the
        (correct) rebuilt one. Dropping is the honest option: a level fit is on demand, so
        the fix is to re-run it, and absence says that plainly where a wrong number would
        not.

        THE RULE IS NOT WRITTEN HERE. It is
        :func:`~nanoscope.analysis.cusum.reconcile_levels`, the same function
        :meth:`~nanoscope.gui.main_window.MainWindow._reconcile_level_fit` applies to the
        table that Export and the project file read. This window used to decide the
        question for itself, by event_id, on its own private copy -- so it drew no CUSUM
        line while the export wrote 6.0 ms of sub-state inside a 4.0 ms event (C-5). Two
        answers to one question is what produced that; there is now one answer, and this
        is a second caller of it, never a second copy.
        """
        from ..analysis.cusum import reconcile_levels

        if self._levels is None or self._df is None:
            return
        self._levels, _dropped = reconcile_levels(self._levels, self._df)

    # -- edit helpers ----------------------------------------------------------------------------
    def _x_to_index(self, x: float) -> int:
        """Map an x value in the axis unit onto a sample index: ``round((x - t0)*fs)`` when the
        rate is known, else ``round(x)`` (the axis IS samples on a rate-unknown recording)."""
        meta = self._session.document(self._doc_id).meta
        fs = meta.sample_rate_hz
        if fs is None:
            return int(round(float(x)))
        return int(round((float(x) - meta.t0_s) * fs))

    def _sample_rate(self) -> float | None:
        if self._doc_id is not None and self._session.has_document(self._doc_id):
            return self._session.document(self._doc_id).meta.sample_rate_hz
        return None

    # The polarity, the flank noise scale and the filter-rise time for a resized event
    # are all read out of ``self._provenance`` by
    # :func:`~nanoscope.analysis.characterize.recharacterize_span`. Three private
    # copies of that extraction used to live here; they are gone, because a private
    # copy is how this panel came to disagree with the merge about what a rebuilt row
    # means.

    def _conditioned(self, data: np.ndarray, fs: float) -> np.ndarray:
        """The trace the DETECTOR measured on: ``lowpass(preprocess_for_detection(raw))``.

        A mask resize re-measures the event; measuring it on the raw channel reports a
        different quantity under the same column names than every untouched row in the
        same table. ``detection_signal`` rebuilds the conditioning from the detection's
        own provenance (which stamps every enabled stage for exactly this reason), and a
        detection that recorded nothing is left alone.

        Cached, because the drag path pays this on every release: ~0.12 s for the user's
        3 M-sample recording. The key covers the document, the channel, the trace length,
        the CALIBRATION fingerprint (gain/offset are the only things that alter the array)
        and every provenance field :func:`detection_signal` reads, so a re-detection with
        different filtering, a calibration edit or a channel switch all invalidate it."""
        prov = self._provenance or {}
        cutoff = prov.get("filter_cutoff_hz")
        if cutoff is not None:
            try:
                cutoff = float(cutoff)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("The saved low-pass cutoff is not a valid frequency.") from exc
            if not np.isfinite(cutoff) or not (0 < cutoff < float(fs) / 2):
                raise ValueError("The saved low-pass cutoff must be positive and below Nyquist.")
        calib: tuple[float, float] | None = None
        if self._doc_id is not None and self._channel is not None:
            try:
                calib = self._session.calibration(self._doc_id, self._channel).fingerprint()
            except Exception:  # noqa: BLE001 — a missing calibration must not break an edit
                calib = None
        key = (self._doc_id, self._channel, int(data.size), float(fs), calib,
               detection_signal_key(prov))
        if self._cond_cache is not None and self._cond_cache[0] == key:
            return self._cond_cache[1]
        # Only successful reconstructions enter the cache. A replay failure must
        # reach the mask-edit boundary, never become a raw trace under the saved
        # filtered measurement's column names.
        out = detection_signal(data, fs, prov)
        self._cond_cache = (key, out)
        return out

    # -- subset / row helpers --------------------------------------------------------------------
    def _subset_ids(self) -> list[int]:
        """Event ids of the current filter subset, in df row order."""
        if self._df is None or len(self._df) == 0:
            return []
        ids = self._df["event_id"].to_numpy(dtype=np.int64)
        if self._filter == _FILTER_ALL:
            return [int(i) for i in ids]
        accepted = self._df["accepted"].to_numpy(dtype=bool)
        if self._filter == _FILTER_ACCEPTED:
            return [int(i) for i, a in zip(ids, accepted) if a]
        return [int(i) for i, a in zip(ids, accepted) if not a]

    def _row_for_id(self, event_id: int | None) -> int | None:
        """Positional row index of the event with this id (robust to any event_id values)."""
        if self._df is None or event_id is None:
            return None
        ids = self._df["event_id"].to_numpy(dtype=np.int64)
        matches = np.nonzero(ids == int(event_id))[0]
        return int(matches[0]) if matches.size else None

    # -- rendering -------------------------------------------------------------------------------
    def _render_and_sync(self, event_id: int, *, emit: bool) -> None:
        row = self._row_for_id(event_id)
        if row is None:
            return
        self._current_id = int(event_id)
        self._render_event(row)
        col = self._df.columns.get_loc("accepted")
        self._sync_accept_check(bool(self._df.iat[row, col]))
        self._refresh_readout(row)
        self._sync_controls()
        if emit:
            self.eventSelected.emit(int(event_id))

    def _render_event(self, row: int, *, preserve_view: bool = False) -> None:
        """Draw the measurement trace and explicitly named equivalent amplitude level."""
        if self._df is None or self._doc_id is None or self._channel is None:
            return
        if not self._session.has_document(self._doc_id):
            self._show_empty("The recording for this detection was closed.")
            return
        r = self._df.iloc[row]
        previous_view = self._plot.getViewBox().viewRange() if preserve_view else None
        start_idx = int(r["start_idx"])
        end_idx = int(r["end_idx"])

        ch = self._session.calibrated(self._doc_id, self._channel)
        data = np.asarray(ch.data)
        n = int(data.size)
        xvals = self._session.x_axis(self._doc_id).values
        if n == 0 or xvals.size == 0:
            self._show_empty("This channel has no samples to show.")
            return

        # Padded window [start - pad : end + pad], pad = max(event length, the chosen context),
        # clamped [0, n]. The CENTRING resets on every navigation — that is the point of
        # stepping through events — but the WIDTH is the user's standing choice (_pad_samples).
        ev_len = max(end_idx - start_idx, 1)
        pad = self._pad_samples(ev_len)
        lo = int(np.clip(start_idx - pad, 0, n))
        hi = int(np.clip(end_idx + pad, 0, n))
        if hi <= lo:
            hi = min(n, lo + 1)

        self._remove_message()
        xs = self._draw_window(lo, hi)
        if xs is None:
            return

        # The view may not leave the recording. Without this the mouse drag/wheel walked off
        # into unbounded blank space — the same failure the pan buttons had, and the reason a
        # user who tried either concluded the window "only shows this one event".
        #
        # FENCED, because installing a limit that excludes the CURRENT range makes the viewbox
        # clip it and announce the clip: the first render clamped the empty state's placeholder
        # 0..1 onto the recording and the handler read that as the user asking for the whole
        # trace as their context width (measured: every event afterwards was drawn from
        # sample 0).
        self._framing = True
        try:
            self._plot.getViewBox().setLimits(xMin=float(xvals[0]), xMax=float(xvals[-1]))
        finally:
            self._framing = False

        # Event region [start, end] (indices clamped for the x lookup; end_idx is half-open).
        xi0 = int(np.clip(start_idx, 0, n - 1))
        xi1 = int(np.clip(end_idx, 0, n - 1))
        x0, x1 = float(xvals[xi0]), float(xvals[xi1])
        if x1 < x0:
            x0, x1 = x1, x0
        self._syncing = True                 # this set is programmatic — not a user edit
        try:
            self._region.setRegion((x0, x1))
        finally:
            self._syncing = False
        self._region.setVisible(True)

        # The default line expresses the STORED median depth. Optional amplitude is
        # separate and never overwrites that number. Both are in the measurement domain.
        baseline = float(r["baseline_pA"])
        try:
            context = self._amplitude.context(r)
            self._amplitude._error = ""
            level = baseline + context.polarity * float(r["depth_pA"])
            label = "median-depth level"
            if amplitude_is_current(r) and not self._stale_columns:
                baseline = float(r["amplitude_baseline_pA"])
                level = float(r["amplitude_level_pA"])
                label = str(r["amplitude_method"]).replace("_", " ") + " level"
            valid = np.isfinite(baseline) and np.isfinite(level) and not self._stale_columns
            self._amplitude._updating = True
            try:
                self._baseline_line.setPos(baseline)
                self._level_line.setPos(level)
                self._baseline_line.label.setFormat("mean baseline reference")
                self._level_line.label.setFormat(label)
            finally:
                self._amplitude._updating = False
            self._set_line_visible(self._baseline_line, bool(valid))
            self._set_line_visible(self._level_line, bool(valid))
        except (ValueError, TypeError, OverflowError) as exc:
            self._amplitude._error = str(exc)
            self._set_line_visible(self._baseline_line, False)
            self._set_line_visible(self._level_line, False)
        self._amplitude.sync(r)

        # Frame the loaded window in x; y autoranges to the slice (annotations are ignoreBounds).
        self._plot.setTitle(self._event_title(r))
        if previous_view is None:
            self._plot.enableAutoRange(y=True)
            self._frame(float(xs[0]), float(xs[-1]))
        else:
            self._framing = True
            try:
                self._plot.setRange(xRange=previous_view[0], yRange=previous_view[1], padding=0)
            finally:
                self._framing = False
        if amplitude_is_current(r) and self._level_line.isVisible():
            # A typed manual level can lie outside the waveform's range. Keep x
            # and any in-view zoom unchanged, but do not hide the saved reference.
            view = self._plot.getViewBox()
            view.updateAutoRange()
            ymin, ymax = view.viewRange()[1]
            low = min(float(r["amplitude_level_pA"]), float(r["amplitude_baseline_pA"]))
            high = max(float(r["amplitude_level_pA"]), float(r["amplitude_baseline_pA"]))
            if low < ymin or high > ymax:
                span = max(ymax - ymin, high - low, np.finfo(float).eps)
                self._plot.setYRange(min(ymin, low - 0.05 * span),
                                     max(ymax, high + 0.05 * span), padding=0)

    def _draw_window(self, lo: int, hi: int) -> np.ndarray | None:
        """Load samples ``[lo, hi)`` into the curve and redraw everything keyed to the WINDOW.

        The single place the curve's data comes from, and the reason the window is state
        (:attr:`_win_lo`) rather than a local: panning and zooming re-slice through here, so a
        pan reveals samples that were never on screen instead of merely sliding the view off
        the end of one fixed slice.

        Returns the x values it drew (or None when there is nothing to draw). The CUSUM level
        segments are rebuilt here and NOT by the caller: they are positioned by
        ``lv.start_idx - win_start``, so drawing them against a stale ``lo`` puts every fitted
        level at the wrong time — off by exactly the distance panned.
        """
        if self._doc_id is None or self._channel is None:
            return None
        if not self._session.has_document(self._doc_id):
            return None
        try:
            data = self._amplitude.measurement_signal()
            self._plot.setLabel("left", "Measurement current", units=self._units)
        except (ValueError, TypeError, OverflowError):
            # Browsing remains available when an old recipe cannot be replayed, but
            # quantitative lines/editing are hidden by _render_event's explicit refusal.
            data = np.asarray(self._session.calibrated(self._doc_id, self._channel).data)
            self._plot.setLabel("left", "Raw channel — measurement unavailable")
        xvals = self._session.x_axis(self._doc_id).values
        n = int(min(data.size, xvals.size))
        lo = int(np.clip(lo, 0, n))
        hi = int(np.clip(hi, 0, n))
        if hi <= lo:
            return None
        xs = np.asarray(xvals[lo:hi], dtype=np.float64)
        ys = np.asarray(data[lo:hi], dtype=np.float64)
        self._win_lo, self._win_hi = lo, hi
        self._curve.setVisible(True)
        self._curve.setData(x=xs, y=ys)   # data LAST (four-step order)
        if self._current_id is not None:
            self._draw_cusum_levels(int(self._current_id), xs, lo)
        return xs

    def _frame(self, x0: float, x1: float, *, padding: float = 0.0) -> None:
        """``setXRange`` behind the :attr:`_framing` fence — OUR framing, never the user's zoom.

        The fence is what keeps :meth:`_on_x_range_changed` from recursing back through the
        range it was just told about.

        ``padding`` defaults to ZERO, where the old single-slice draw framed at 2%: the view
        must never be WIDER than the loaded window, or every unfenced range announcement (a
        pane resize is one) reads as the user asking for samples beyond it and the window
        creeps outward for the rest of the session. Flush against the axes is the price of
        "the view is always a sub-range of what is loaded".
        """
        self._framing = True
        try:
            self._plot.setXRange(float(x0), float(x1), padding=padding)
        finally:
            self._framing = False

    def _pad_samples(self, ev_len: int) -> int:
        """Per-side context around an event, in samples: the user's standing choice.

        A choice from the combo is floored by the event's OWN length, so a long blockade is
        never framed tighter than itself; a width the user framed by hand is taken literally,
        because it is already a statement about a picture they were looking at.

        ``_manual_pad`` (what the user last framed with the mouse) wins over the combo, and is
        cleared the moment the combo is used — an explicit choice supersedes an implicit one.
        Either way the WIDTH survives navigation while the centring resets, because a curator
        sets the context once and then steps; re-deriving ±3 ms on every Prev/Next threw the
        chosen context away without saying so.
        """
        if self._manual_pad is not None:
            return max(int(self._manual_pad), 1)
        if self._pad_ms is None:             # "Whole trace": every sample, the [0, n] clip does it
            return self._n_samples() or _PAD_SAMPLES_FALLBACK
        fs = None
        if self._doc_id is not None and self._session.has_document(self._doc_id):
            fs = self._session.document(self._doc_id).meta.sample_rate_hz
        few = int(round(fs * float(self._pad_ms) / 1000.0)) if fs else _PAD_SAMPLES_FALLBACK
        return max(int(ev_len), int(few), 1)

    def _n_samples(self) -> int:
        """Length of the browsed channel, or 0 when there is no document to ask."""
        if self._doc_id is None or self._channel is None:
            return 0
        if not self._session.has_document(self._doc_id):
            return 0
        return int(self._session.calibrated(self._doc_id, self._channel).data.size)

    # -- panning / zooming the loaded window -----------------------------------------------------
    def _on_context_changed(self, _index: int) -> None:
        """The "Context:" combo: adopt the width and re-render the browsed event around it."""
        if self._updating:
            return
        self._pad_ms = self._context_combo.currentData()
        self._manual_pad = None              # an explicit choice supersedes a hand-framed one
        row = self._row_for_id(self._current_id)
        if row is not None:
            self._render_event(row)

    def _on_x_range_changed(self, _vb, rng) -> None:
        """The user moved the x range with the MOUSE (drag-pan or wheel-zoom): widen the loaded
        window to cover what is now framed, and remember the width as the standing context.

        Without this the buttons would load data and the mouse would not — a worse
        inconsistency than today's "neither does", because the two gestures would disagree
        about what the same window contains. Our own framing is fenced off by
        :attr:`_framing`, so this runs only for a range change the user caused.
        """
        if self._framing or self._df is None or self._current_id is None:
            return
        if self._win_lo is None or self._win_hi is None:
            return                           # nothing loaded — there is no view to have moved
        n = self._n_samples()
        if n == 0:
            return
        row = self._row_for_id(self._current_id)
        if row is None:
            return
        lo_needed = int(np.clip(self._x_to_index(float(rng[0])), 0, n))
        hi_needed = int(np.clip(self._x_to_index(float(rng[1])) + 1, 0, n))
        if hi_needed <= lo_needed:
            return
        win_lo, win_hi = int(self._win_lo), int(self._win_hi)
        span, win_span = hi_needed - lo_needed, win_hi - win_lo
        # A ZOOM (the width changed) is the user choosing a context; a drag-pan and a window
        # RESIZE are not, and both of them also land here. Recording the width on all three
        # would keep re-adopting the width that is already loaded — harmless until an event of
        # a different length inherits it — so only a real change of width counts.
        if abs(span - win_span) > max(_WINDOW_SLACK, win_span // 100):
            ev_len = max(int(self._df.iat[row, self._df.columns.get_loc("end_idx")])
                         - int(self._df.iat[row, self._df.columns.get_loc("start_idx")]), 1)
            self._manual_pad = max(int(round((span - ev_len) / 2.0)), 1)
        if lo_needed < win_lo - _WINDOW_SLACK or hi_needed > win_hi + _WINDOW_SLACK:
            # Widen only: samples already loaded off-screen cost a slice, and the curve
            # downsamples + clips to the view (ChannelPane.make_curve), so they cost nothing
            # to draw. The window is rebuilt from scratch on the next navigation anyway.
            self._draw_window(min(win_lo, lo_needed), max(win_hi, hi_needed))

    def _event_title(self, r: pd.Series) -> str:
        # The waveform title is numerical artwork. Warnings remain in the Event
        # readout and its fixed-height staleness note, not in captured plots.
        units = f" {self._units}" if self._units else ""
        dwell_us = float(r["dwell_s"]) * 1e6 if np.isfinite(r["dwell_s"]) else float("nan")
        parts = [
            f"Event {int(r['event_id'])}",
            f"dwell {_fmt_num(dwell_us, ' µs', 4)}",
            f"median depth {_fmt_num(r['depth_pA'], units, 4)}",
            f"dI/I0 {_fmt_num(r['frac_depth'], '', 3)}",
        ]
        if amplitude_is_current(r):
            parts.append(f"amplitude {_fmt_num(r['amplitude_pA'], ' ' + str(r['amplitude_units']), 4)} "
                         f"({r['amplitude_method']})")
        return "   ·   ".join(parts)

    def _refresh_readout(self, row: int) -> None:
        if self._df is None:
            self._readout.setText("")
            return
        r = self._df.iloc[row]
        units = f" {self._units}" if self._units else ""
        subset = self._subset_ids()
        ev_id = int(r["event_id"])
        if ev_id in subset:
            where = f"{subset.index(ev_id) + 1} of {len(subset)} · {self._filter}"
        else:
            where = f"filtered out of {self._filter} ({len(subset)} shown)"
        _labels = format_labels(r.get("flag", "")) if hasattr(r, "get") else ""
        # THE STALE MARK, per line. Each readout line that repeats a badged events-table
        # column carries the header's own " ⚠"; the sentence that says what the glyph means
        # is the box's fixed-height note (_stale_label), NOT a readout line -- this label's
        # height is its line count, and every line it gains is height the plot above
        # loses. See set_stale_columns.
        stale = self._stale
        lines = [
            f"Event {ev_id}    [{where}]",
            f"accepted:        {'yes' if bool(r['accepted']) else 'no'}",
            f"dwell:           {_fmt_num(float(r['dwell_s']) * 1e6, ' µs', 5)}"
            f"   ({_fmt_num(r['dwell_s'], ' s', 4)}){stale('dwell_s')}",
            f"median depth:    {_fmt_num(r['depth_pA'], units, 5)}{stale('depth_pA')}",
            f"dI/I0:           {_fmt_num(r['frac_depth'], '', 4)}"
            f"{frac_depth_flag(r['frac_depth'])}{stale('frac_depth')}",
            f"SNR:             {_fmt_num(r['snr'], '', 3)}{stale('snr')}",
            f"rise / fall:     {_fmt_num(float(r['rise_s']) * 1e6, ' µs', 4)}"
            f" / {_fmt_num(float(r['fall_s']) * 1e6, ' µs', 4)}{stale('rise_s', 'fall_s')}",
            f"levels:          {int(r['n_levels'])}",
            f"filter-limited:  {'yes' if bool(r['filter_limited']) else 'no'}",
            *( [f"labels:          {_labels}"] if _labels else [] ),
            f"baseline:        {_fmt_num(r['baseline_pA'], units, 5)}{stale('baseline_pA')}",
        ]
        if amplitude_is_current(r):
            lines.append(f"amplitude:       {_fmt_num(r['amplitude_pA'], ' ' + str(r['amplitude_units']), 5)} "
                         f"({r['amplitude_method']}; separate from depth/dI/I0/SNR)"
                         + (" ⚠" if self._stale_columns else ""))
        lines.extend(self._level_lines(ev_id, units))
        self._readout.setText("\n".join(lines))

    def _stale_sentence(self) -> str:
        """The banner's sentence when the owner gave one, else the columns by name."""
        if self._stale_note:
            return self._stale_note
        return (", ".join(self._stale_columns)
                + " are stale — measured by an estimator that is no longer the configured "
                  "one; see the banner on the Analysis panel")

    def _draw_cusum_levels(self, event_id: int, xs: np.ndarray, win_start: int) -> None:
        """Draw the fitted sub-states as FLAT HORIZONTAL SEGMENTS across the waveform.

        A level fit whose only output is text is not a fit anyone can check. The readout
        listed L0/L1/L2 with amplitudes and dwells while the plot showed a single
        ``level`` line -- ``characterize``'s one in-event median -- so the picture said
        "one state" beside a table saying "three", and the eye believes the picture.

        One segment per level, drawn at that level's OWN mean current
        (``LEVEL_SCHEMA['level_pA']``, absolute so it sits on the trace without any
        polarity reasoning) and spanning exactly that level's samples. Where the segments
        step is where the fit says the molecule changed state, which is the thing to
        judge by eye against the data underneath.

        Cleared and rebuilt per event; ``ignoreBounds`` so the annotation never drags the
        y autorange the way a full-height item would.

        WHAT SURVIVES A WIDENED CONTEXT (owner, E2(b)). The segments and the risers are
        drawn whatever the window is; only the LABELS are gated, and the gate is in PIXELS
        (:data:`_MIN_LABEL_PX`) rather than in samples, so it stops depending on how much
        recording happens to be loaded around the event. Below the bar one label survives on
        the widest level, and once the window dwarfs the event a faint band
        (:meth:`_draw_event_band`) says where to look. The old rule — 2.5% of the loaded
        window — deleted every label the moment the user widened the context to see the open
        pore, on a fit that had not moved.
        """
        for item in [*getattr(self, "_cusum_items", []), *getattr(self, "_cusum_extras", [])]:
            try:
                self._plot.removeItem(item)
            except (RuntimeError, ValueError):
                pass
        #: The FLAT SEGMENTS, and only those: casing + stroke per fitted level. Kept apart
        #: from the risers and the labels below because "one horizontal item per level, at
        #: that level's own current" is the contract the suite reads this attribute for.
        self._cusum_items: list[pg.PlotDataItem] = []
        #: The legibility furniture drawn WITH them (step risers, per-level labels) — same
        #: lifetime, removed in the same sweep, not part of that contract.
        self._cusum_extras: list[pg.GraphicsObject] = []
        if self._levels is None or xs.size == 0:
            return
        rows = self._levels[self._levels["event_id"] == int(event_id)]
        if rows.empty:
            return

        # ``win_start`` is the padded window's FIRST SAMPLE INDEX, handed in by the
        # caller. It was briefly reconstructed as ``round(xs[0] * fs)`` instead, which is
        # the same number only when the x axis is seconds from zero -- it silently drew
        # nothing on a document with a t0 offset, and would also break on the
        # rate-unknown path where the axis IS samples. The caller already has the index;
        # deriving it back out of a float time was the mistake.
        # TWO-PASS CASING (owner, twice: "the sublevels ... cannot be read easily on the
        # curate panel. Either the color has to be boldened/changed or another way of
        # representing the sublevels is needed"): a wider dark underlay beneath a thick
        # stroke, so the segments read over the waveform instead of dissolving into it —
        # the same trick map labels use over roads. Three things carry the rest of that
        # answer, and all three are about the SAME picture rather than a second one:
        #   * the INK is now _CUSUM_HEX == overlays.STEP_LEVEL_HEX, one name for a level's
        #     colour across this window and the main trace, at full value rather than the
        #     0.62-value green that lost against a 1 px waveform of similar luminance;
        #   * the widths match the trace overlay's (_STEP_LEVEL_WIDTH / _STEP_CASING_WIDTH),
        #     so a level is the same weight of line wherever it is drawn;
        #   * a RISER is drawn at each state change and each segment is LABELLED with its own
        #     number, because "unmistakable" means the eye can tell which state is which and
        #     read its depth without crossing to the text panel — a staircase, not three
        #     unrelated dashes.
        casing = pg.mkPen(_CUSUM_CASING_HEX, width=_CUSUM_CASING_WIDTH)
        pen = pg.mkPen(_CUSUM_HEX, width=_CUSUM_LEVEL_WIDTH)
        riser = pg.mkPen(_CUSUM_HEX, width=1, style=Qt.DashLine)
        units = f" {self._units}" if self._units else ""
        # A label needs a segment wide enough to sit over — IN PIXELS. See _MIN_LABEL_PX for
        # what the sample-count version of this rule did to a widened context.
        plot_px = self._plot_width_px()
        # WHERE THE EVENT IS, when the window has grown far bigger than it. The levels tile
        # their parent exactly (cusum.reconcile_levels rests on the same property), so the
        # event's span is readable off the rows without going back to the frame.
        self._draw_event_band(rows, xs, win_start, plot_px)
        prev: tuple[int, float] | None = None         # (level's last sample in-window, its y)
        labelled = 0
        # (px, x0, x1, y, level_idx, depth) of the widest labellable segment — the fallback
        # the block after the loop uses so the annotation can never reach zero.
        widest: tuple[float, float, float, float, int, Any] | None = None
        for lv in rows.itertuples(index=False):
            lo = int(lv.start_idx) - int(win_start)
            hi = int(lv.end_idx) - int(win_start)
            lo = max(lo, 0)
            hi = min(hi, xs.size)
            if hi - lo < 1:
                continue
            y = float(lv.level_pA)
            if not np.isfinite(y):
                continue
            x0, x1 = float(xs[lo]), float(xs[min(hi, xs.size) - 1])
            for z, p in ((11, casing), (12, pen)):
                seg = pg.PlotDataItem([x0, x1], [y, y], pen=p)
                seg.setZValue(z)
                self._plot.addItem(seg, ignoreBounds=True)
                self._cusum_items.append(seg)
            # The vertical the fit asserts: WHERE the molecule changed state. Dashed and thin
            # so it never reads as data, and drawn only between levels that really do adjoin
            # (a level clipped away by the window edge must not be joined across the gap).
            if prev is not None and 0 <= lo - prev[0] <= 1:
                step = pg.PlotDataItem([x0, x0], [prev[1], y], pen=riser)
                step.setZValue(11)
                self._plot.addItem(step, ignoreBounds=True)
                self._cusum_extras.append(step)
            prev = (min(hi, xs.size) - 1, y)
            # PIXELS, not samples. ``_frame`` puts exactly [xs[0], xs[-1]] across the plot with
            # zero padding, so a segment covering (hi-lo) of the window's samples is drawn
            # (hi-lo)/(xs.size-1) of the plot's width — a number that does not move when the
            # user loads more context around the same event, which is the whole point.
            seg_px = plot_px * (hi - lo) / max(xs.size - 1, 1)
            # plot_px == 0 means the viewbox could not be measured (never laid out). LABEL
            # ANYWAY: the gate exists to stop labels colliding, and stripping annotation off a
            # plot whose width is unknown trades a cosmetic risk for a missing measurement.
            wide_enough = plot_px <= 0.0 or seg_px >= _MIN_LABEL_PX
            listable = int(lv.level_idx) < _MAX_LEVEL_LINES
            if listable and (widest is None or seg_px > widest[0]):
                widest = (seg_px, x0, x1, y, int(lv.level_idx), lv.depth_pA)
            if wide_enough and listable:
                self._add_level_label(x0, x1, y, int(lv.level_idx), lv.depth_pA, units)
                labelled += 1

        # AT LEAST ONE LABEL, ALWAYS — the second half of the owner's note, and the half a
        # pure width rule cannot deliver. At "Whole trace" the WHOLE EVENT is about 27 px
        # wide on a 942 px plot (measured, a 600-sample event in a 20 600-sample window), so
        # no individual level can clear a 30 px bar and a rule that only strips would take
        # the annotation to zero — the exact disappearance being complained about, moved from
        # ±30 ms out to ±300 ms. The band says WHERE the staircase is; one label on the
        # widest level says WHAT it is, which together are what "the labels vanished" was
        # asking for. Labels are dropped for CROWDING, and one label cannot crowd anything.
        if labelled == 0 and widest is not None:
            _px, x0, x1, y, idx, depth = widest
            self._add_level_label(x0, x1, y, idx, depth, units)

    def _add_level_label(self, x0: float, x1: float, y: float, level_idx: int,
                         depth: Any, units: str) -> None:
        """One ``L<n> <depth>`` label centred over a level segment. Two callers, one
        appearance — see the crowding rule in :meth:`_draw_cusum_levels`."""
        text = pg.TextItem(f"L{int(level_idx)}  {_fmt_num(depth, units, 3)}",
                           anchor=(0.5, 1.15), color=_CUSUM_HEX)
        text.setZValue(13)
        text.setPos(0.5 * (float(x0) + float(x1)), float(y))
        self._plot.addItem(text, ignoreBounds=True)
        self._cusum_extras.append(text)

    def _plot_width_px(self) -> float:
        """The plot's drawing width in SCREEN PIXELS, or ``0.0`` when it cannot be measured.

        Read off the viewbox rather than derived from its range, because this is called from
        :meth:`_draw_window` — which runs BEFORE :meth:`_frame` — so the viewbox's *range* at
        this moment still describes the previously browsed event, while its *width* is a
        property of the widget and is already correct. The framing that follows sets exactly
        ``[xs[0], xs[-1]]`` with zero padding, so a fraction of the loaded window is the same
        fraction of this width.

        ``0.0`` (a window that has never been laid out, a zero-width pane) means "unknown",
        and the caller labels everything rather than deciding legibility from a measurement
        it does not have.
        """
        try:
            width = float(self._plot.getViewBox().width())
        except (RuntimeError, ValueError, AttributeError, TypeError):
            return 0.0
        return width if np.isfinite(width) and width > 0.0 else 0.0

    def _draw_event_band(self, rows: pd.DataFrame, xs: np.ndarray, win_start: int,
                         plot_px: float) -> None:
        """Faint band over the event itself, once the loaded window dwarfs it.

        At ± 3 ms the staircase fills the plot and needs no help. At "Whole trace" — the
        context the user picks to see the open pore either side — a 600-sample event inside
        2 M samples is a hairline somewhere in the middle, and the fitted levels, which are
        the thing being judged, become impossible to FIND before they can be read. The band
        is the "it is here" marker; the exact span remains the draggable mask region and the
        readout, and this is drawn behind the waveform (``ignoreBounds``, z below the level
        segments) so it can never be mistaken for one of them.

        THE MINIMUM WIDTH IS DELIBERATE AND IS NOT A MEASUREMENT. Below about 3 px a band is
        drawn as nothing at all, which is the state this exists to end, so a sub-pixel event
        is widened symmetrically to that floor. It is a pointer, so a pointer that cannot be
        seen is worth less than one that is a pixel wider than its target; nothing reads a
        number off it.
        """
        if xs.size < 2:
            return
        try:
            ev_lo = int(rows["start_idx"].min()) - int(win_start)
            ev_hi = int(rows["end_idx"].max()) - int(win_start)
        except (KeyError, TypeError, ValueError):
            return
        ev_len = ev_hi - ev_lo
        if ev_len <= 0 or xs.size <= _EVENT_BAND_RATIO * ev_len:
            return
        lo = int(np.clip(ev_lo, 0, xs.size - 1))
        hi = int(np.clip(ev_hi, 0, xs.size - 1))
        x0, x1 = float(xs[lo]), float(xs[hi])
        if x1 < x0:
            x0, x1 = x1, x0
        per_px = (float(xs[-1]) - float(xs[0])) / plot_px if plot_px > 0.0 else 0.0
        floor = _EVENT_BAND_MIN_PX * per_px
        if per_px > 0.0 and (x1 - x0) < floor:
            mid = 0.5 * (x0 + x1)
            x0, x1 = mid - 0.5 * floor, mid + 0.5 * floor
        colour = _region_qcolor()
        colour.setAlpha(_EVENT_BAND_ALPHA)
        band = pg.LinearRegionItem(values=(x0, x1), movable=False, brush=pg.mkBrush(colour))
        band.setZValue(-9)                   # behind the waveform, above the mask region
        self._plot.addItem(band, ignoreBounds=True)
        self._cusum_extras.append(band)

    def _level_lines(self, event_id: int, units: str) -> list[str]:
        """The CUSUM per-level breakdown for one event, the reason there is none, or
        nothing at all.

        Nothing at all is still the common case and must stay silent: a level fit is
        on-demand, so most of the time there is no table and the readout should look
        exactly as it did before this feature existed. What is no longer silent is the
        case where a fit IS standing and this event has no group in it (F2).

        The count on the ``levels:`` line above is ``events.SCHEMA['n_levels']`` -- the
        always-on BIC count. This block is the CUSUM fit, which is a different estimator
        with different parameters and is allowed to disagree. When it does, SAY so
        rather than printing two numbers and leaving the user to notice: BIC over-splits
        on a filtered trace (see characterize._count_levels), so a disagreement is
        expected and is information about the trace, not a bug to hide.
        """
        # WHERE THE BREAKDOWN USED TO BE. A group that vanished takes ~5 lines out of the
        # middle of the readout, and printing nothing in their place is what let a mask drag
        # look like the feature breaking: the always-on ``levels:`` count above still said a
        # number, so the readout read as "1 level" rather than "no fit here". The note is
        # rendered only when there IS something to say — see :meth:`_level_gap_note`, which
        # stays silent for the common case of no fit at all.
        if not self._has_levels(int(event_id)):
            gap = self._level_gap_note(int(event_id))
            return ["", "CUSUM levels:    none", f"  {gap}"] if gap else []
        rows = self._levels[self._levels["event_id"] == int(event_id)]
        n_cusum = int(rows["n_levels"].iloc[0])
        out = ["", f"CUSUM levels:    {n_cusum}"]
        try:
            n_bic = int(self._df.iloc[self._row_for_id(int(event_id))]["n_levels"])  # type: ignore[arg-type]
        except (TypeError, ValueError, KeyError, IndexError):
            n_bic = n_cusum
        if n_bic != n_cusum:
            out.append(f"  (BIC count says {n_bic}; CUSUM is the fitted estimate)")
        # The readout is a fixed-height panel above the waveform plot, and this block is
        # the only part of it whose length depends on the DATA. max_levels defaults to 32,
        # so an unbounded listing can add ~30 lines, force the window several hundred
        # pixels taller and squeeze the waveform to its minimum -- on a browser whose
        # whole purpose is looking at the waveform. Show the head and say what was
        # withheld; the full set is in the exported level table.
        shown = rows.head(_MAX_LEVEL_LINES)
        for lv in shown.itertuples(index=False):
            step = ("" if not np.isfinite(lv.step_pA)
                    else f"   step {lv.step_pA:+.4g}{units}")
            out.append(
                f"  L{int(lv.level_idx)}: "
                f"{_fmt_num(lv.depth_pA, units, 4)}"
                f"   {_fmt_num(float(lv.dwell_s) * 1e6, ' µs', 4)}"
                f"   dI/I0 {_fmt_num(lv.frac_depth, '', 3)}"
                f"{frac_depth_flag(lv.frac_depth)}{step}"
            )
        if len(rows) > len(shown):
            out.append(f"  … {len(rows) - len(shown)} more (export the level table to see all)")
        return out

    # -- controls sync ---------------------------------------------------------------------------
    def _sync_controls(self) -> None:
        """Reconcile the nav widgets + accept checkbox with the current subset + displayed event."""
        subset = self._subset_ids()
        n = len(subset)
        cur = self._current_id
        in_subset = cur is not None and cur in subset
        pos = subset.index(cur) if in_subset else None
        self._updating = True
        try:
            self._filter_combo.setCurrentText(self._filter)
            if n == 0:
                self._index_spin.setRange(0, 0)
                self._index_spin.setValue(0)
                self._index_spin.setEnabled(False)
                self._count_label.setText("of 0")
            else:
                self._index_spin.setRange(1, n)
                self._index_spin.setEnabled(True)
                if pos is not None:
                    self._index_spin.setValue(pos + 1)
                self._count_label.setText(f"of {n}")
            self._prev_btn.setEnabled(n > 0 and (pos is None or pos > 0))
            self._next_btn.setEnabled(n > 0 and (pos is None or pos < n - 1))
            has_current = self._df is not None and cur is not None
            self._accept_check.setEnabled(has_current)
            self._accept_btn.setEnabled(has_current)
            self._reject_btn.setEnabled(has_current)
            self._pan_left_btn.setEnabled(has_current)
            self._pan_right_btn.setEnabled(has_current)
            self._context_combo.setEnabled(has_current)
            geometry_editable = has_current and self._module_id != CONSENSUS_MODULE_ID
            self._merge_next_btn.setEnabled(geometry_editable)
            self._merge_next_btn.setToolTip(
                _RECONCILED_EDIT_TIP if has_current and not geometry_editable
                else "Merge this event with the following event(s) inside the merge "
                     "window, then advance. Reversed by Edit ▸ Undo (⌘Z)."
            )
            self._trace_btn.setEnabled(has_current)
            # HIDDEN, not merely greyed, with no detection at all. A disabled control is a
            # promise that it applies here and is unavailable now; under "No detection — run
            # detection to browse events." that promise is false, and the owner read the two
            # dead buttons as this window not having per-event Accept/Reject (reported twice).
            # An event on screen shows them; the empty-state message names them meanwhile.
            for _w in (self._accept_btn, self._reject_btn, self._accept_check,
                       self._revert_btn):
                _w.setVisible(has_current)
        finally:
            self._updating = False
        # ENABLED separately from visible, and per row: a table with no recorded originals
        # shows the control (so the gesture is discoverable) and disables it with the sentence
        # that says why. See _revert_state.
        self._sync_revert_control()
        # Rebuilt WITH the controls, from state -- never written over their output.
        note = self.refusal_note()
        self._batch_note.setText(note)
        self._batch_note.setVisible(bool(note))
        # ...and the missing-levels sentence with them, so it follows the browsed event on
        # every navigation rather than sitting where the last mask edit left it.
        self._sync_level_note()
        row = self._row_for_id(self._current_id)
        self._amplitude.sync(self._df.iloc[row] if row is not None and self._df is not None else None)

    def _pan_view(self, frac: float) -> None:
        """Scroll ``frac`` of a window along the RECORDING (owner note #8). The mask region,
        the df and the readout are untouched, and the next navigation re-centres on its event
        as always. Button-driven rather than drag, because the movable mask region owns the
        drag over the middle of the view.

        It LOADS the samples it scrolls onto. Until it did, this was a view move over one fixed
        slice that was already framed edge to edge — so a press could only slide the pane off
        the end of its own data (half the plot blank after one press, all of it after two) and
        revealed not one sample the user had not already seen. The window state moves first,
        the framing follows it.
        """
        n = self._n_samples()
        lo, hi = self._win_lo, self._win_hi
        if n == 0 or lo is None or hi is None:
            # No window loaded (empty state): degrade to the old pure view move.
            (x0, x1), _y = self._plot.viewRange()
            dx = (float(x1) - float(x0)) * float(frac)
            self._frame(float(x0) + dx, float(x1) + dx)
            return
        step = int(float(frac) * (hi - lo))
        # Clamp the STEP, not the two edges: clipping each edge to [0, n] independently would
        # squeeze the window against the recording's start/end and silently change the context
        # width the user chose (and _pad_samples then inherits).
        step = max(step, -lo)
        step = min(step, n - hi)
        if step == 0:
            return
        xs = self._draw_window(lo + step, hi + step)
        if xs is None:
            return
        self._frame(float(xs[0]), float(xs[-1]))

    def _sync_accept_check(self, accepted: bool) -> None:
        self._updating = True
        self._accept_check.setChecked(bool(accepted))
        self._updating = False

    def _on_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        subset = self._subset_ids()
        pos = int(value) - 1
        if 0 <= pos < len(subset):
            self.show_event(subset[pos])

    def _on_filter_changed(self, text: str) -> None:
        if self._updating:
            return
        self.set_filter(text)

    # -- empty state -----------------------------------------------------------------------------
    def _show_empty(self, text: str) -> None:
        """Blank the waveform, hide the annotations and show a centred friendly note."""
        self._curve.setVisible(False)
        self._curve.setData(x=np.empty(0), y=np.empty(0))
        self._region.setVisible(False)
        self._set_line_visible(self._baseline_line, False)
        self._set_line_visible(self._level_line, False)
        for _it in [*getattr(self, '_cusum_items', []), *getattr(self, '_cusum_extras', [])]:
            try:
                self._plot.removeItem(_it)
            except (RuntimeError, ValueError):
                pass
        self._cusum_items = []
        self._cusum_extras = []
        self._plot.setTitle(None)
        self._remove_message()
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=_MSG_HEX)
        self._plot.addItem(item)
        # No window is loaded, and the 0..1 placeholder range is not in the recording's own
        # units — so the x limits a previous document installed have to go with it, or the
        # placeholder is clamped into a corner of a range it has nothing to do with.
        self._win_lo = self._win_hi = None
        self._plot.getViewBox().setLimits(xMin=None, xMax=None)
        self._frame(0, 1)
        self._plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)
        self._message_item = item
        self._readout.setText(text)
        self._sync_controls()
        # LAST, and unconditional: the empty state IS the message. A second sentence under
        # "The recording for this detection was closed." explaining a level fit that also
        # has nothing to draw is noise about the smaller of two problems.
        self._level_note.setVisible(False)

    def _remove_message(self) -> None:
        if self._message_item is not None:
            try:
                self._plot.removeItem(self._message_item)
            except (RuntimeError, ValueError):
                pass
            self._message_item = None

    @staticmethod
    def _set_line_visible(line: pg.InfiniteLine, on: bool) -> None:
        """Toggle an InfiniteLine AND its label (the label is a separate scene item)."""
        line.setVisible(on)
        label = getattr(line, "label", None)
        if label is not None:
            label.setVisible(on)

    # -- accessors (tests) -----------------------------------------------------------------------
    def plot_item(self) -> pg.PlotItem:
        return self._plot

    def current_filter(self) -> str:
        return self._filter
