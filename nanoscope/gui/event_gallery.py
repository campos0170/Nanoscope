"""The Event Gallery: a contact sheet of every detected event, for triage at scale.

A standalone top-level window (opened from ``Analysis ▸ Event gallery…``) that draws
EVERY event as a small waveform tile in a scrolling grid, so a detection with two
thousand events can be judged by eye in one pass instead of two thousand clicks. It is
the companion to :class:`~nanoscope.gui.event_inspector.EventInspectorWindow`, which
shows one event at a time in full detail: the inspector answers "what is THIS event?",
the gallery answers "what is in this detection, and which ones are wrong?".

Like the inspector, it is fed the CURRENT detection by
:class:`~nanoscope.gui.main_window.MainWindow` via :meth:`set_events`, which installs a
REFERENCE to the very same :class:`~nanoscope.analysis.events.EventSet` ``.df`` the
:class:`~nanoscope.gui.panels.EventTable` edits. Curation here writes that shared frame
in place and is announced by signal, so it flows back to the table, the span overlays,
the summary and the exports exactly as a checkbox click does.

WHY TILES ARE PAINTED, NOT PLOTTED
----------------------------------
Each tile is drawn by a :class:`QStyledItemDelegate` straight onto the view, from a
cached min/max envelope -- not by a pyqtgraph PlotWidget per event. A thousand
PlotWidgets is a thousand scene graphs, and the window would never open. A QListView in
IconMode paints only the tiles actually on screen, so the cost of the grid is set by the
size of the viewport and not by the size of the detection.

The envelope is min/max per pixel column rather than strided sampling. The padding here is
proportional to the event (see ``_PAD_FRACTION``), so the event itself always spans most
of the tile and could survive either method -- what does NOT survive stride-sampling is
structure INSIDE a long event. A 500 ms event on a 140 px tile puts ~358 samples in every
column; a 2 ms sub-state two-thirds of the way down is then one sample in three hundred.
Measured on exactly that shape (50,000-sample event, 200-sample deeper sub-state): the
min/max envelope bottoms out at 51.7 pA and shows the sub-state, stride-sampling bottoms
out at 76.4 pA and steps straight over it -- a tile 24 pA shallower than the event it
claims to draw. Sub-states are what the CUSUM fit exists to find, so a thumbnail that
hides them is worse than no thumbnail. Min/max also keeps the noise band honest, which is
what makes a marginal event look marginal at tile size.

PER-SUBPOPULATION GALLERIES (item 13)
------------------------------------
``MainWindow`` may hand the window an ``event_id -> subpopulation id`` map read off a
finished :class:`~nanoscope.analysis.subpop_report.SubpopReport`, plus that run's warnings
verbatim (:meth:`EventGalleryWindow.set_subpopulations`). That turns the same contact sheet
into the per-subpopulation galleries: a class filter, a chip of the class's own ink on every
tile, and -- because "the first 35 events of subpop 2" is a picture of the start of the
recording and not of subpop 2 -- a REPRESENTATIVE sample spaced through the class's dwell
distribution, which is what the owner's notebook draws.

Nothing about that assignment is written to the shared frame. It is a provisional mixture
posterior; a column would carry it into the exports and the project file as though it had
been measured. It is dropped the moment a different detection arrives, and the caveat pane
that appears with it cannot be collapsed.

Signals (deliberately the same contract as the inspector, so MainWindow wires both the
same way):
    eventSelected(int): the focused tile changed.
    acceptanceChanged(int, bool): one event was curated (event_id, accepted).
    curationBatchChanged(): a multi-tile or whole-detection curation rewrote ``accepted``;
        the owner resets the table + overlays + summary from the df.
    acceptanceRefused(): an accept gesture recorded refusals on the ledger while writing
        NOTHING (every asked-for row was already in its final state). No data signal will
        fire, so the owner repaints the other two surfaces' notes off this one.
    saveProjectRequested() / saveProjectAsRequested(): ask MainWindow to save the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..analysis.curation_view import (
    FILTER_ACCEPTED, FILTER_ALL, FILTER_REJECTED, FILTER_UNLABELLED, FILTER_UNREVIEWED,
    LABEL_FILTER_PREFIX, curation_progress, describe_progress, end_of_filter_note, filter_depends_on_labels_or_review,
    filter_display_candidates, filter_mask, filter_options, format_progress, is_filter,
    label_rows, no_unreviewed_note, row_unchanged, same_events, unlabel_rows,
    can_keep_place, landing_after_add, landing_after_refill, landing_after_removal,
)
from ..analysis.events import (
    AUTOMATIC_REJECTION_FLAGS, FRAC_DEPTH_ENHANCEMENT, flag_tokens, mark_reviewed,
    reviewed_mask,
)
from ..analysis.labels import LabelVocabulary, label_column
from ..analysis.reconcile import clear_curation_conflicts
from ..core.session import Session
from ..plot.style import LEVEL_INK
from .categorical import NOISE_HEX, category_colour
from .curation_keys import CurationKeyGuard

__all__ = ["TRACE_CALIBRATED", "TRACE_DETRENDED", "EventGalleryWindow"]

# -- filter subsets. The inspector's, by the same names, because they are the same question
# -- asked of the same frame: both windows read them from analysis.curation_view.
_FILTER_ALL = FILTER_ALL
_FILTER_ACCEPTED = FILTER_ACCEPTED
_FILTER_REJECTED = FILTER_REJECTED
_FILTERS = (_FILTER_ALL, _FILTER_ACCEPTED, _FILTER_REJECTED)

#: The Event inspector's key, deliberately: one "Advance after decision" for both windows.
#: Spelled twice because this window may not import its sibling.
_ADVANCE_SETTINGS_KEY = "curation/advance_after_decision"
#: The progress line's floor in the status bar (the inspector's number): room for "End of
#: Unreviewed · 868 earlier (⌘U)" and the reviewed count, however long the project label is.
_PROGRESS_MIN_W = 330

# -- sort orders. "Detection order" is the default because it is the only one in which a
# -- tile's neighbours are its neighbours IN TIME, which is what makes a burst of false
# -- positives from one disturbance visible as a run rather than scattered through the grid.
_SORT_ORDER = "Detection order"
_SORT_DWELL = "Dwell (longest first)"
_SORT_DEPTH = "Depth (deepest first)"
_SORT_FRAC = "dI/I0 (deepest first)"
_SORTS = (_SORT_ORDER, _SORT_DWELL, _SORT_DEPTH, _SORT_FRAC)
#: The frame column each sort reads (and captions) -- ONE table, read by the model's sort,
#: the caption and the stale mark, so the three cannot name different columns.
_SORT_COLUMNS = {_SORT_DWELL: "dwell_s", _SORT_DEPTH: "depth_pA", _SORT_FRAC: "frac_depth"}

# -- item 13: the per-subpopulation galleries ------------------------------------------------
# The gallery is fed an ``event_id -> subpopulation id`` map by MainWindow, read off a finished
# SubpopReport. It is a FILTER and a chip of ink, never a column: nothing here is written to the
# shared events frame, because a provisional mixture assignment that reached the frame would
# reach the exports, the project file and every other surface as though it had been measured.
_SUBPOP_ALL = "All events"
#: The id every event the report does not name is drawn as -- ``categorical``'s "in no class"
#: sentinel. On the owner's zn_30h that is the ~90 % of detected regions the >=4 sigma gate threw
#: away, and they are the reason this entry is offered rather than hidden: a gallery of the
#: 1076 gated events that never mentions the 9928 dropped ones is a picture of a filter, not of
#: a detection.
_SUBPOP_NONE = -1

#: "Representative" in the owner's sense, which is NOT "the first N". A page of the first 35
#: events of a subpopulation is a picture of the start of the recording; the notebook takes
#: examples spaced evenly through the class's own DWELL distribution, so the sheet spans the
#: class from its shortest event to its longest. That is the whole difference between a sample
#: and a prefix, and on zn_30h two of the four subpopulations differ by a factor of 18 in median
#: dwell -- a prefix of either says nothing about its spread.
_REPRESENTATIVE_TIP = (
    "Show a sample spaced evenly through this subset's DWELL distribution -- shortest to "
    "longest -- instead of the first tiles in the current sort. One page IS the sample, so "
    "turning the page is off; change 'Per page' to sample more or fewer.\n\n"
    "Events whose dwell could not be measured are not on a dwell ladder and are left out of "
    "the sample; the count is in the status line, and unchecking this reaches them.")

# -- which trace the tiles are drawn from ----------------------------------------------------
# NAMED ON THE SURFACE, because the honest answer is not the one the label suggests. The owner's
# notebook calls these "raw-event galleries" and draws them from ``current_detr_pA`` -- the
# DETRENDED trace, the one the detector and all thirteen features were measured on, not the
# recording. Both are legitimate pictures of the same event and they differ visibly wherever the
# open pore drifts, so the gallery offers both and always says which one is on screen rather
# than letting "raw" mean whichever it happens to be.
#: Public because ``main_window`` selects one by name when it opens the gallery on a
#: report; a private constant would have meant either a magic string in the caller or
#: a second spelling of it, which is how two surfaces come to disagree about a mode.
TRACE_CALIBRATED = "Calibrated (as recorded)"
TRACE_DETRENDED = "Detrended (as detected)"
_TRACE_TIP = (
    "Which trace the tiles are drawn from.\n\n"
    "'Calibrated' is the recording itself, including any open-pore drift.\n"
    "'Detrended' is the array the detector and the subpopulation features were measured on -- "
    "and is what the owner's notebook galleries actually show, despite calling them raw. It is "
    "offered only after a subpopulation run has handed one over.")

#: The caveat that is about THIS PICTURE rather than about the run, and therefore cannot come
#: from the report's own warnings. Numbered first in the pane, ahead of them.
_TILE_CAVEAT = (
    "The chip on each tile is the MIXTURE's assignment, not a measurement: an event is in "
    "subpop 2 because a full-covariance Gaussian mixture, at the K a cross-validation "
    "preferred, gave it the highest posterior — often not by much. Filtering to one "
    "subpopulation therefore produces a grid that LOOKS homogeneous whether or not the class "
    "is real, which is what the caveats below are about. Nothing here is written to the events "
    "table, so no export leaves this window carrying it.")

#: Context each side of the event, as a fraction of its own length with a floor in samples.
#: PROPORTIONAL, and with no upper clamp: a fixed cap in milliseconds made long events fill
#: ~75% of the tile, leaving a sliver of open pore too short to judge the blockade against —
#: the tile showed the event but not the thing that makes it an event. Half the event length
#: each side puts the blockade in the middle 50% of the tile at every dwell.
_PAD_FRACTION = 0.5
_PAD_MIN_SAMPLES = 25

#: Tiles per page. PAGES, not a scrollbar: a page is a fixed frame the eye can sweep and
#: then dismiss, and "I have looked at everything" is a statement you can actually make
#: about page 7 of 42. An endless scroll gives no such landmark -- there is no position in
#: it that means finished, and no way to tell a re-read from new ground.
#:
#: "Fit" is the default and computes the count from the viewport, so a page is exactly one
#: screenful and turning the page is the ONLY way to move. A fixed count cannot promise
#: that: 200 tiles in a small window overflows, and with no scrollbar those events would be
#: silently unreachable. The scrollbar therefore stays AsNeeded as a safety valve for the
#: explicit sizes -- under "Fit" it never appears, which is the point of "Fit".
#:
#: THE NUMBERS ARE LITERAL, AND DELIBERATELY SO. Measured on a 1512 px display: choosing 48
#: puts 48 tiles on the page and about 32 inside the viewport (96 and 200 both show 36), so
#: the page scrolls -- and on macOS the overlay scrollbar is invisible until it is scrolled,
#: which is what makes that read as "the count is wrong" (owner note #12). Capping the number
#: to the screenful was tried and rejected: "Per page" is ALSO the size of the representative
#: subpopulation sample (see :data:`_REPRESENTATIVE_TIP` -- "one page IS the sample"), so a
#: cap would quietly turn "sample 24 events spaced through the dwell distribution" into 12
#: because the window was small. Silently shrinking a sample is a worse failure than a page
#: that scrolls, so the number stays literal and the SHORTFALL IS DISCLOSED instead: the
#: status line says the page is taller than the window (see :meth:`_page_overflows`) and the
#: tooltip says a number can overflow while "Fit" cannot.
_FIT = "Fit"
_PAGE_SIZES = (_FIT, "24", "48", "96", "200")

#: What "Fit" means BEFORE there is a viewport to fit to. A window that has never been shown
#: has no laid-out viewport, and the honest arithmetic over an unrealised one lands on a page
#: of ONE TILE -- which is not a page, and which a headless caller (the batch-curation tests
#: drive a gallery that is never shown) has no way to notice. So Fit falls back to a plain
#: count until the first viewport resize replaces it with the real screenful.
#:
#: This is the only page-size default in the module, and it is REACHED. Its predecessor,
#: ``_DEFAULT_PAGE_SIZE``, sat beside ``_PAGE_SIZES`` reading like the gallery's default page
#: size while the combo has always been set to "Fit" -- a number nothing applied.
_FIT_FALLBACK_PAGE_SIZE = 48

#: Lines the footer status line RESERVES, whether the line needs them or not -- see
#: :class:`_StatusLine`. Three, measured: at this window's default width (940 px, which
#: leaves the label ~640 px beside the two "…all shown" buttons) the ordinary line -- the
#: range, the accepted count, the trace and the gesture hints -- wraps to three lines,
#: 45 px, and that is the footer the grid above had always been laid out against at that
#: size; the same three lines at 1000 and 1100 px. Fixed, so nothing the line says can
#: move the grid: the banner's staleness sentence (342-427 characters) took the footer to
#: six lines and "Fit" re-cut the page under the user. What a wider line cannot fit is
#: elided at the end of the third line with the whole line in the tooltip, and the
#: clauses are ordered so the static gesture hints go before any count or disclosure.
_STATUS_LINES = 3

# -- tile geometry -------------------------------------------------------------------------------
_TILE_SIZES = {"Small": (96, 72), "Medium": (140, 104), "Large": (200, 148)}
_DEFAULT_TILE = "Medium"
_CAPTION_H = 15          # px reserved under the waveform for the id/dwell caption
_TILE_PAD = 4

# -- colours. Local, like the inspector's, so the gallery does not couple to another
# -- panel's palette; the waveform/region hues MATCH the inspector so a tile and the
# -- full-size view of the same event are recognisably the same picture.
_CURVE_HEX = "#4C9BE6"
_REGION_HEX = "#FFB74D"
#: MUST EQUAL ``overlays._DEFAULT_SPAN_ALPHA``. Same hue as the trace's span overlays and
#: half its alpha (30 against 60) meant the gallery's mask was literally half the shade of
#: the main panel's over the same samples -- the two surfaces drawing one event's mask two
#: different strengths, which is the "the mask on the gallery does not match the mask on the
#: main panel" the owner reported. The indices always agreed; only the ink did.
#:
#: Duplicated here rather than imported because ``overlays`` spells it privately and this
#: window deliberately owns its own palette (see the block comment above). The right home for
#: it is ``plot.style``, imported by both; until one constant serves both surfaces, this
#: comment is the coupling and a change to either alpha has to be made twice.
_REGION_ALPHA = 60
_TILE_BG_HEX = "#20242b"
_TILE_BG_SEL_HEX = "#2d3a4d"
_BORDER_HEX = "#333941"
_BORDER_SEL_HEX = "#4C9BE6"
_CAPTION_HEX = "#98a0aa"
_REJECTED_HEX = "#c1554a"    # the strike-through + border on a rejected tile
_REJECTED_DIM = 0.34         # opacity a rejected tile's waveform is drawn at
_LABEL_HEX = "#8FD6C0"       # the label tag on a tile (Okabe-Ito green, lightened)
_MANUAL_MARK_HEX = "#F0D58A"  # the "M" on a hand-made tile (Okabe-Ito yellow, lightened)

#: The fitted CUSUM sub-state levels, in the ink ``overlays`` draws them in on the main
#: trace -- so a level is one colour on all three event surfaces.
#:
#: EQUALS ``overlays.STEP_LEVEL_HEX`` by construction now: both are ``plot.style.LEVEL_INK``,
#: which is the spelling's one home. Imported from ``plot.style`` and not from ``overlays``
#: because this window is deliberately not allowed to reach a sibling GUI module (the guard:
#: ``test_subpop_galleries.test_the_gallery_still_reaches_no_sibling_panel``, which allows
#: only the shared ``categorical`` leaf among GUI imports — ``plot.style`` sits below the
#: GUI layer and is fair game, exactly why the value moved there).
_LEVEL_HEX = LEVEL_INK
_LEVEL_WIDTH = 2

#: MEDIUM AND LARGE TILES ONLY, and that was rendered rather than guessed. On a 3-level
#: event drawn offscreen at all three sizes: Medium (140x104 -> 89 px of waveform) and Large
#: (200x148 -> 133 px) put the bars clearly on the sub-states they annotate with the min/max
#: band still readable either side of each one. Small (96x72 -> 57 px) does not: at that
#: height the whole blockade is ~20 px of envelope, the band itself is 4-6 px thick, and a
#: 2 px bar drawn across it COVERS it -- the tile goes green where it was blue and the
#: noise band disappears under its own annotation. That band is the thing the module
#: docstring argues the min/max envelope exists to keep honest ("what makes a marginal event
#: look marginal at tile size"), so hiding it to show a level the user can open the tile to
#: see is the wrong trade at that size.
#:
#: The floor is on the WAVEFORM height rather than the tile name so it keeps meaning
#: something if the sizes are ever retuned.
_LEVEL_MIN_TILE_H = 80

#: Envelopes held in memory. Each is two float32 arrays of at most the tile width, so
#: 4000 tiles at 200 px is ~6 MB -- bounded, and enough that scrolling back over a
#: detection never recomputes. Beyond it the oldest go, which costs one slice + one
#: reduction to redraw.
_ENVELOPE_CACHE_MAX = 4000

#: Role carrying the df ROW INDEX of the event a tile draws. The model is a permutation
#: (filter + sort) over the frame, so the row is what every lookup needs; event_id is
#: read from the frame, never assumed equal to it.
_ROW_ROLE = int(Qt.UserRole) + 1


def _cmd_key() -> str:
    """The NAME of the modifier ``Qt.ControlModifier`` actually is on this machine: ``⌘`` on
    macOS, ``Ctrl`` elsewhere.

    The status line and the batch tooltips have to name a key the owner can find. Qt maps
    ``Qt::ControlModifier`` to Command on macOS (Ctrl-click there is the context menu, not
    an add-to-selection), so the shipped "Ctrl/Shift-click" of the "Accept selected" tooltip
    named a gesture that opens a menu on the only display this is used on. Asked of
    QKeySequence rather than of ``sys.platform`` so it is the same answer Qt paints into the
    menus beside it.
    """
    txt = QtGui.QKeySequence(QtCore.Qt.Modifier.CTRL | Qt.Key_A).toString(
        QtGui.QKeySequence.SequenceFormat.NativeText)
    return txt[:-1].rstrip("+") or "Ctrl"


def _fmt_dwell(dwell_s: float) -> str:
    """A dwell in the largest unit that keeps it readable, for a caption with ~9 chars."""
    if not np.isfinite(dwell_s):
        return "--"
    us = dwell_s * 1e6
    if us < 1000.0:
        return f"{us:.0f}µs"
    ms = us / 1000.0
    return f"{ms:.2f}ms" if ms < 100.0 else f"{ms:.0f}ms"


# =================================================================================================
# THE DERIVED HALF OF ``accepted`` -- what a curation action has no standing over
# =================================================================================================
def _digit_span(digits: tuple[str, ...]) -> str:
    """``("1","2","3") -> "1–3"``; a gap or a single key is listed ("1, 3", "1")."""
    if not digits:
        return ""
    if len(digits) == 1:
        return digits[0]
    ints = [int(d) for d in digits]
    if ints == list(range(ints[0], ints[0] + len(ints))):
        return f"{digits[0]}–{digits[-1]}"
    return ", ".join(digits)


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


#: What "Accept all shown" acts on, said once so the plain and the reduced tooltip cannot
#: drift apart.
_ACCEPT_ALL_TIP = (
    "Mark every tile in the CURRENT filter accepted — not the whole detection "
    "unless the filter is All. Tiles the detector flagged as unmeasurable "
    "(clipped by the boundary, over the dwell limit, measured from a non-finite "
    "sample) stay rejected; the status line counts them.")


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


class _EnvelopeSource:
    """Per-event min/max waveform envelopes, sliced from the calibrated channel and cached.

    Holds the session and the current (doc, channel) rather than a copy of the signal: the
    channel array is the recording, and this window must not double the memory a 600 MB
    file already costs.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._doc_id: str | None = None
        self._channel: int | None = None
        self._cache: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
        self._order: list[tuple[int, int, int, int]] = []
        #: An alternative full-length trace to slice instead of the calibrated channel -- the
        #: detrended array a subpopulation run already built. Held as a REFERENCE to the caller's
        #: array and never copied: it is a second full-length float array for the recording, and
        #: on a 600 MB file that is the whole reason it is handed in from outside rather than
        #: rebuilt here. Dropped with the target, so closing the detection releases it.
        self._override: np.ndarray | None = None
        #: Trace length, remembered by the last _build — so window_bounds() can clip the
        #: padded window the way _build clipped it WITHOUT re-fetching the calibrated
        #: array per tile paint. 0 until something builds (paint always builds the
        #: envelope before asking for bounds, so the fallback below is theoretical).
        self._n: int = 0

    def set_target(self, doc_id: str | None, channel: int | None) -> None:
        if (doc_id, channel) != (self._doc_id, self._channel):
            self._doc_id, self._channel = doc_id, channel
            self._override = None      # it belonged to the previous detection
            self._n = 0                # a different trace is a different length
            self.clear()

    def set_override(self, data: np.ndarray | None) -> None:
        """Install (or drop) the alternative trace. Invalidates every cached envelope."""
        if data is self._override:
            return
        self._override = None if data is None else np.asarray(data)
        self.clear()

    def has_override(self) -> bool:
        return self._override is not None

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()

    def invalidate(self, start_idx: int, end_idx: int) -> None:
        """Forget one event's envelope (its bounds were edited, so its picture changed)."""
        dead = [k for k in self._cache if k[0] == int(start_idx) and k[1] == int(end_idx)]
        for k in dead:
            self._cache.pop(k, None)
            try:
                self._order.remove(k)
            except ValueError:
                pass

    def window_bounds(self, start_idx: int, end_idx: int) -> tuple[int, int]:
        """The padded sample window ``_build`` renders for this event — SAME pad, SAME
        clip — so a caller can place other events' spans on the same x scale the tile's
        waveform is drawn on. Uses the trace length the last build remembered; before
        any build (theoretical — paint builds the envelope first) it clips against the
        event's own end, which degrades to an unpadded right edge, never a wrong one."""
        ev_len = max(int(end_idx) - int(start_idx), 1)
        pad = self._pad(ev_len)
        n = self._n if self._n > 0 else int(end_idx)
        lo_i = int(np.clip(int(start_idx) - pad, 0, n))
        hi_i = int(np.clip(int(end_idx) + pad, 0, n))
        if hi_i <= lo_i:
            hi_i = min(n, lo_i + 1)
        return lo_i, hi_i

    def envelope(self, start_idx: int, end_idx: int, width: int
                 ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """``(lo, hi, event_frac)`` for a tile ``width`` px wide, or None if unavailable.

        ``lo``/``hi`` are the per-column minimum and maximum of the padded window, already
        normalised into ``[0, 1]`` with 0 at the TOP of the tile, so the delegate only has
        to scale by the tile rect. ``event_frac`` is ``(x0, x1)``-style span of the event
        itself within that window, as a fraction of the width, so the shaded region lands
        on the same samples the inspector shades.
        """
        if self._doc_id is None or self._channel is None:
            return None
        key = (int(start_idx), int(end_idx), int(width), 0)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        built = self._build(int(start_idx), int(end_idx), int(width))
        if built is None:
            return None
        self._cache[key] = built
        self._order.append(key)
        while len(self._order) > _ENVELOPE_CACHE_MAX:
            self._cache.pop(self._order.pop(0), None)
        return built

    def _pad(self, ev_len: int) -> int:
        return max(_PAD_MIN_SAMPLES, int(round(ev_len * _PAD_FRACTION)))

    def _build(self, start_idx: int, end_idx: int, width: int
               ) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not self._session.has_document(self._doc_id):
            return None
        if self._override is not None:
            data = self._override
        else:
            try:
                ch = self._session.calibrated(self._doc_id, self._channel)
            except Exception:  # noqa: BLE001 -- a closed/!changed channel must not break painting
                return None
            data = np.asarray(ch.data)
        n = int(data.size)
        if n == 0:
            return None
        self._n = n

        ev_len = max(int(end_idx) - int(start_idx), 1)
        pad = self._pad(ev_len)
        lo_i = int(np.clip(start_idx - pad, 0, n))
        hi_i = int(np.clip(end_idx + pad, 0, n))
        if hi_i <= lo_i:
            hi_i = min(n, lo_i + 1)
        ys = np.asarray(data[lo_i:hi_i], dtype=np.float64)
        if ys.size == 0:
            return None

        cols = max(int(width), 8)
        if ys.size <= cols:
            lo = hi = ys.astype(np.float32)
        else:
            # Min/max per column. Trim to a whole number of columns, then reduce -- so a
            # blockade narrower than one column still sets that column's minimum and is
            # drawn one pixel wide, instead of being stepped over. See the module docstring.
            per = ys.size // cols
            usable = ys[: per * cols].reshape(cols, per)
            lo = usable.min(axis=1).astype(np.float32)
            hi = usable.max(axis=1).astype(np.float32)
            tail = ys[per * cols:]
            if tail.size:
                lo[-1] = min(float(lo[-1]), float(tail.min()))
                hi[-1] = max(float(hi[-1]), float(tail.max()))

        # Returned in DATA UNITS, not normalised. Normalisation is the caller's, because it
        # depends on whether tiles share one scale (see EventGalleryWindow.envelope_for) --
        # and keeping it out of here means switching scale mode costs nothing, instead of
        # invalidating every cached envelope in the window.
        win = max(hi_i - lo_i, 1)
        ev0 = (int(np.clip(start_idx, lo_i, hi_i)) - lo_i) / win
        ev1 = (int(np.clip(end_idx, lo_i, hi_i)) - lo_i) / win
        return (np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32),
                (float(ev0), float(ev1)))


class _TileModel(QtCore.QAbstractListModel):
    """The filtered + sorted permutation of the shared events frame, one row per tile.

    Holds df ROW INDICES, never a copy of the frame: the frame is shared truth that the
    table and the inspector also write, and a gallery working from a snapshot would keep
    showing accepted-state that had already changed elsewhere.
    """

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._rows: list[int] = []          # the whole filtered+sorted subset
        self._filter: str = _FILTER_ALL
        self._sort: str = _SORT_ORDER
        self._page: int = 0
        #: THE PAGE SIZE UNTIL A VIEWPORT SAYS OTHERWISE, and that is a longer window than it
        #: looks: ``page_size_box`` is built already reading "Fit" (it is the first entry), so
        #: setting it to "Fit" changes nothing and ``_on_page_size_changed`` never fires at
        #: build time. This number is therefore what a gallery that has not been SHOWN pages
        #: by -- which is how the batch-curation tests drive one. Same constant Fit falls
        #: back to for the same reason, so the two cannot drift.
        self._page_size: int = _FIT_FALLBACK_PAGE_SIZE
        #: df row -> subpopulation id, resolved once per install from the event_id map. Held as
        #: an ARRAY aligned with the frame rather than a dict keyed by event_id, so filtering is
        #: one comparison over the frame instead of a lookup per row.
        self._classes: np.ndarray | None = None
        self._subpop: int | None = None     # None = every class; else the class id shown
        self._representative = False
        #: Rows the representative sample could not place, because a dwell ladder has no rung
        #: for an unmeasurable dwell. Reported, never silently dropped -- see the status line.
        self._no_dwell = 0

    # -- contents ------------------------------------------------------------------------
    def set_frame(self, df: pd.DataFrame | None) -> None:
        self.beginResetModel()
        self._df = df
        self._rebuild()
        self.endResetModel()

    def set_filter(self, mode: str) -> None:
        if not is_filter(mode) or mode == self._filter:
            return
        self.beginResetModel()
        self._filter = mode
        self._rebuild()
        self.endResetModel()

    def set_sort(self, mode: str) -> None:
        if mode not in _SORTS or mode == self._sort:
            return
        self.beginResetModel()
        self._sort = mode
        self._rebuild()
        self.endResetModel()

    def set_classes(self, classes: np.ndarray | None) -> None:
        """Install the per-row subpopulation ids (aligned with the frame), or ``None``."""
        self.beginResetModel()
        self._classes = classes
        if classes is None:
            self._subpop = None
        self._rebuild()
        self.endResetModel()

    def classes(self) -> np.ndarray | None:
        return self._classes

    def set_subpop(self, class_id: int | None) -> None:
        """Show one subpopulation (``None`` = all). A no-op when no classes are installed."""
        if self._classes is None:
            class_id = None
        if class_id == self._subpop:
            return
        self.beginResetModel()
        self._subpop = class_id
        self._rebuild()
        self.endResetModel()

    def subpop(self) -> int | None:
        return self._subpop

    def set_representative(self, on: bool) -> None:
        if bool(on) == self._representative:
            return
        self.beginResetModel()
        self._representative = bool(on)
        self._rebuild()
        self.endResetModel()

    def representative(self) -> bool:
        return self._representative

    def no_dwell_count(self) -> int:
        """How many rows the current representative sample left out. 0 when it is off."""
        return int(self._no_dwell)

    def subset_size(self) -> int:
        """The filtered subset BEFORE the representative sample thins it.

        The two numbers are different questions -- "how many events are in subpop 2" and "how
        many of them are on screen" -- and a status line that printed one under both names is
        how a 35-tile sample gets quoted as a class of 35.
        """
        df = self._df
        if df is None or len(df) == 0:
            return 0
        return int(self._filtered_rows(df).size)

    def refilter(self) -> None:
        """Re-apply the filter after curation moved events between the subsets."""
        self.beginResetModel()
        self._rebuild()
        self.endResetModel()

    def _filtered_rows(self, df: pd.DataFrame) -> np.ndarray:
        """The Show + subpopulation subset, unsorted and unsampled."""
        keep = np.flatnonzero(filter_mask(df, self._filter))
        if self._subpop is not None and self._classes is not None and keep.size:
            keep = keep[self._classes[keep] == int(self._subpop)]
        return keep

    def _sample(self, df: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
        """Thin ``keep`` to one page of examples spaced through its DWELL distribution.

        Sorted by dwell, then ``page_size`` positions taken at even spacing from the shortest
        to the longest -- so the sheet always contains this subset's extremes and a regular
        walk between them, whatever the page size. It is a QUANTILE ladder in the order
        statistics, not in dwell VALUE: even spacing in value on a distribution spanning a
        factor of 18 would put most of the sample in an empty tail.

        Rows whose dwell is not finite have no place on that ladder. They are left out and
        COUNTED (:meth:`no_dwell_count`), never quietly folded in at one end, because a tile
        that got into a dwell-spaced sample without a dwell would be read as having one.
        """
        d = df["dwell_s"].to_numpy(dtype=np.float64)[keep]
        finite = np.isfinite(d)
        self._no_dwell = int((~finite).sum())
        ladder = keep[finite]
        if ladder.size == 0:
            return ladder
        ladder = ladder[np.argsort(d[finite], kind="stable")]
        n = int(self._page_size)
        if n >= ladder.size:
            return ladder
        pos = np.unique(np.rint(np.linspace(0.0, ladder.size - 1, n)).astype(np.int64))
        return ladder[pos]

    def _rebuild(self) -> None:
        df = self._df
        if df is None or len(df) == 0:
            self._rows = []
            self._no_dwell = 0
            return
        keep = self._filtered_rows(df)
        self._no_dwell = 0
        if self._representative and keep.size:
            keep = self._sample(df, keep)

        if self._sort != _SORT_ORDER and keep.size:
            col = _SORT_COLUMNS[self._sort]
            vals = df[col].to_numpy(dtype=np.float64)[keep]
            # Descending, and NaN LAST rather than wherever argsort puts it: a row whose
            # depth could not be measured is exactly the row worth reaching, so it must
            # land at a predictable end of the grid instead of the middle.
            order = np.argsort(np.where(np.isfinite(vals), -vals, np.inf), kind="stable")
            keep = keep[order]
        self._rows = [int(i) for i in keep]
        self._clamp_page()

    # -- Qt model ------------------------------------------------------------------------
    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: B008
        """Tiles on the CURRENT PAGE. The model is the page, not the subset — so the view
        lays out one page's worth and there is nothing below the fold to scroll to."""
        if parent.isValid():
            return 0
        return len(self._page_rows())

    def data(self, index: QtCore.QModelIndex, role: int = int(Qt.DisplayRole)) -> Any:
        """The df row a tile draws — indexed into THE CURRENT PAGE, never the whole subset.

        This read ``self._rows[index.row()]`` when pagination landed, and every page after
        the first therefore painted page 1's waveforms and captions while the selection,
        the curation and the labels — which all go through :meth:`df_row` — acted on the
        page's real events. You looked at event 0, pressed R, and rejected event 10.

        It survived the pagination tests because those assert on ``df_row`` and
        ``page_df_rows``, both of which were right. Only the delegate reads this role, so
        only the PICTURE was wrong. :meth:`df_row` is now the single implementation.
        """
        row = self.df_row(index)
        return row if (row is not None and role == _ROW_ROLE) else None

    # -- pagination ----------------------------------------------------------------------
    def _page_rows(self) -> list[int]:
        lo = self._page * self._page_size
        return self._rows[lo:lo + self._page_size]

    def page(self) -> int:
        return self._page

    def page_count(self) -> int:
        if not self._rows:
            return 1
        return (len(self._rows) + self._page_size - 1) // self._page_size

    def page_size(self) -> int:
        return self._page_size

    def set_page(self, page: int) -> bool:
        page = int(max(0, min(int(page), self.page_count() - 1)))
        if page == self._page:
            return False
        self.beginResetModel()
        self._page = page
        self.endResetModel()
        return True

    def set_page_size(self, size: int) -> None:
        size = int(max(1, size))
        if size == self._page_size:
            return
        # Keep the FIRST TILE currently on screen on screen. Re-paginating around a fixed
        # page NUMBER moves the user somewhere else in the detection for no reason they
        # asked for; anchoring on the tile they were looking at does not.
        anchor = self._page * self._page_size
        self.beginResetModel()
        self._page_size = size
        self._page = anchor // size
        if self._representative:
            # The sample IS one page, so its size is the page size: changing one changes the
            # other, and rebuilding here is what makes "Per page" the sample-size control the
            # representative tooltip says it is.
            self._rebuild()
        self._clamp_page()
        self.endResetModel()

    def _clamp_page(self) -> None:
        self._page = max(0, min(self._page, self.page_count() - 1))

    def page_of_df_row(self, df_row: int) -> int | None:
        try:
            return self._rows.index(int(df_row)) // self._page_size
        except ValueError:
            return None

    # -- lookups the window needs --------------------------------------------------------
    def df_row(self, index: QtCore.QModelIndex) -> int | None:
        rows = self._page_rows()
        if not index.isValid() or not (0 <= index.row() < len(rows)):
            return None
        return rows[index.row()]

    def index_for_df_row(self, df_row: int) -> QtCore.QModelIndex:
        """A model index for ``df_row`` IF it is on the current page; invalid otherwise.

        The caller is responsible for turning to the right page first (see
        EventGalleryWindow.show_event) — a silent page turn inside a lookup would move the
        grid under a user who only asked where something was.
        """
        try:
            return self.index(self._page_rows().index(int(df_row)), 0)
        except ValueError:
            return QtCore.QModelIndex()

    def df_rows(self) -> list[int]:
        """Every row in the filtered subset, across ALL pages."""
        return list(self._rows)

    def page_df_rows(self) -> list[int]:
        return list(self._page_rows())


class _TileDelegate(QtWidgets.QStyledItemDelegate):
    """Paints one event tile: waveform envelope, shaded event span, caption, curation state."""

    def __init__(self, window: EventGalleryWindow) -> None:
        super().__init__(window)
        self._w = window

    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: ARG002
        w, h = self._w.tile_size()
        return QtCore.QSize(w + 2 * _TILE_PAD, h + _CAPTION_H + 2 * _TILE_PAD)

    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        df = self._w.frame()
        row = index.data(_ROW_ROLE)
        if df is None or row is None or not (0 <= int(row) < len(df)):
            return
        r = df.iloc[int(row)]
        accepted = bool(r["accepted"])
        selected = bool(option.state & QtWidgets.QStyle.State_Selected)

        rect = option.rect.adjusted(_TILE_PAD, _TILE_PAD, -_TILE_PAD, -_TILE_PAD)
        wave = QtCore.QRect(rect.left(), rect.top(), rect.width(), rect.height() - _CAPTION_H)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        painter.fillRect(rect, QtGui.QColor(_TILE_BG_SEL_HEX if selected else _TILE_BG_HEX))

        env = self._w.envelope_for(int(row), wave.width())
        if env is not None:
            lo_n, hi_n, (ev0, ev1) = env
            # The event span, shaded behind the trace, in the same hue and at the same alpha
            # as the trace-view span overlays -- so a tile reads as the same object the main
            # window shades.
            #
            # AND ONLY WHEN IT IS ACCEPTED. ``analysis_panel`` filters the starts it hands the
            # span renderer by ``accepted`` and draws NOTHING for a rejected event; this fill
            # used to run unconditionally, above the ``_REJECTED_DIM`` that dims the waveform,
            # so an accepted tile and a rejected one painted the identical shade (measured:
            # centre pixel 0xff3a362f on both) while the main panel showed a mask on one and
            # none on the other. That is the mask mismatch the owner reported -- the sample
            # indices always agreed, the shading rule did not.
            #
            # A rejected event gets an OUTLINE in the rejection ink instead of nothing at all.
            # "Nothing" is what the trace does, but the trace has a time axis and the rest of
            # the recording around it; a tile is a 2x-event window with no axis, so with no
            # mark at all there is no way to see WHERE the span the detector proposed lay --
            # which is the one thing needed to judge whether to put it back. The outline is
            # unmistakably not the accepted shading, and it is the same red as the tile's
            # border and caption, so it reads as part of "this one is rejected".
            x0 = wave.left() + int(round(ev0 * wave.width()))
            x1 = wave.left() + int(round(ev1 * wave.width()))
            if x1 <= x0:
                x1 = x0 + 1
            own = QtCore.QRect(x0, wave.top(), x1 - x0, wave.height())
            if accepted:
                span = QtGui.QColor(_REGION_HEX)
                span.setAlpha(_REGION_ALPHA)
                painter.fillRect(own, span)
            else:
                pen = QtGui.QPen(QtGui.QColor(_REJECTED_HEX))
                pen.setWidth(1)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(own.adjusted(0, 0, -1, -1))

            # ...and every OTHER accepted event inside the padded window, at half
            # strength. The window shows the neighbouring waveform, and the trace view
            # shades every accepted span in it — a sibling fragment drawn bare here
            # read as "not detected" while the main panel said otherwise (owner note
            # #4). Half alpha keeps this tile's own event the unmistakably darker one.
            faint = QtGui.QColor(_REGION_HEX)
            faint.setAlpha(_REGION_ALPHA // 2)
            for f0, f1 in self._w.neighbour_spans(int(row)):
                n0 = wave.left() + int(round(f0 * wave.width()))
                n1 = wave.left() + int(round(f1 * wave.width()))
                if n1 <= n0:
                    n1 = n0 + 1
                painter.fillRect(QtCore.QRect(n0, wave.top(), n1 - n0, wave.height()),
                                 faint)

            pen = QtGui.QPen(QtGui.QColor(_CURVE_HEX))
            pen.setWidth(1)
            painter.setPen(pen)
            if not accepted:
                painter.setOpacity(_REJECTED_DIM)
            self._paint_envelope(painter, wave, lo_n, hi_n)
            painter.setOpacity(1.0)

            # The fitted CUSUM sub-states, ON TOP of the waveform, when a fit is installed
            # and the tile is big enough to separate them (see _LEVEL_MIN_TILE_H). Until
            # this existed the gallery was the only one of the three event surfaces that
            # never showed a level -- so the grid that exists to find the events worth
            # opening could not show the very structure the fit was run to find.
            #
            # Drawn at full opacity even on a rejected tile: the dim is about the
            # WAVEFORM's standing, and a level bar is an annotation, not signal. Dimming it
            # to 0.34 over a 0.34 waveform simply removed it.
            self._paint_levels(painter, wave, self._w.level_bars_for(
                int(row), wave.width(), wave.height()))
        else:
            painter.setPen(QtGui.QColor(_CAPTION_HEX))
            painter.drawText(wave, int(Qt.AlignCenter), "no signal")

        # Caption: the id always, plus what the current sort is ABOUT, so sorting by dwell
        # produces a grid whose captions can be read down as a sequence.
        painter.setPen(QtGui.QColor(_REJECTED_HEX if not accepted else _CAPTION_HEX))
        cap = QtCore.QRect(rect.left(), rect.bottom() - _CAPTION_H + 1,
                           rect.width(), _CAPTION_H)
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 2.0))
        painter.setFont(font)
        painter.drawText(cap, int(Qt.AlignCenter), self._w.caption_for(r))

        # The label, if the row carries one: a small tag in the top-left of the waveform.
        # Drawn INSIDE the plot rather than in the caption because the caption already
        # carries the id and the sort quantity, and a label the user cannot see while
        # sweeping the grid is a label they will apply twice.
        text = self._w.label_of(int(row))
        if text:
            painter.setFont(font)
            fm = painter.fontMetrics()
            tw = min(fm.horizontalAdvance(text) + 8, wave.width() - 6)
            tag = QtCore.QRect(wave.left() + 3, wave.top() + 3, tw, fm.height() + 2)
            bg = QtGui.QColor(_LABEL_HEX)
            bg.setAlpha(210)
            painter.fillRect(tag, bg)
            painter.setPen(QtGui.QColor("#10151b"))
            painter.drawText(tag, int(Qt.AlignCenter), text)

        # The subpopulation chip: the class's own ink, in the top-RIGHT of the waveform.
        #
        # A CHIP AND NOT THE BORDER. The border already carries curation (rejected is red,
        # selected is blue) and the caption already carries the sort quantity; painting the class
        # into either would have made one mark mean two things, and the one it would have
        # displaced -- "this event is rejected" -- is the one the gallery exists to show. The
        # hue is ``gui.categorical``'s, so subpop 2 is the same colour here, in the report
        # window's t-SNE and waveform figures, and on the trace.
        chip = self._w.subpop_of(int(row))
        if chip is not None:
            cid, text = chip
            painter.setFont(font)
            fm = painter.fontMetrics()
            cw = min(fm.horizontalAdvance(text) + 8, wave.width() - 6)
            box = QtCore.QRect(wave.right() - cw - 3, wave.top() + 3, cw, fm.height() + 2)
            bg = QtGui.QColor(category_colour(cid))
            bg.setAlpha(225)
            painter.fillRect(box, bg)
            painter.setPen(QtGui.QColor("#10151b"))
            painter.drawText(box, int(Qt.AlignCenter), text)

        # A hand-made event wears an "M" in the bottom-left corner of its waveform (the top
        # corners carry the label and the subpopulation chip, the border carries curation and
        # the caption the sort quantity), so a grid of mixed tiles shows which ones a person drew.
        if self._w.is_hand_made(int(row)):
            painter.setFont(font)
            fm = painter.fontMetrics()
            side = fm.height() + 2
            box = QtCore.QRect(wave.left() + 3, wave.bottom() - side - 2, side, side)
            painter.fillRect(box, QtGui.QColor(_MANUAL_MARK_HEX))
            painter.setPen(QtGui.QColor("#10151b"))
            painter.drawText(box, int(Qt.AlignCenter), "M")

        border = QtGui.QColor(_BORDER_SEL_HEX if selected
                              else (_REJECTED_HEX if not accepted else _BORDER_HEX))
        pen = QtGui.QPen(border)
        pen.setWidth(2 if selected else 1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.restore()

    @staticmethod
    def _paint_levels(painter: QtGui.QPainter, wave: QtCore.QRect,
                      bars: list[tuple[float, float, float]]) -> None:
        """The fitted sub-state levels as flat horizontal bars, one per level.

        The trace's :func:`~nanoscope.gui.overlays.render_step` cases each bar in a dark
        underlay because a coloured line dissolves into the waveform it annotates; at tile
        size a casing is 3 px of a 89 px picture and swallows the sub-state instead, so
        here the separation comes from the ink alone -- :data:`_LEVEL_HEX` is a saturated
        green against a blue envelope and reads at 2 px without one. A bar is always at
        least 1 px long, so the shortest resolved level is still a mark rather than nothing.
        """
        if not bars or wave.height() <= 0 or wave.width() <= 0:
            return
        pen = QtGui.QPen(QtGui.QColor(_LEVEL_HEX))
        pen.setWidth(_LEVEL_WIDTH)
        pen.setCapStyle(Qt.FlatCap)
        painter.setPen(pen)
        h = wave.height() - 1
        lines = []
        for f0, f1, y_n in bars:
            x0 = wave.left() + int(round(f0 * wave.width()))
            x1 = wave.left() + int(round(f1 * wave.width()))
            if x1 <= x0:
                x1 = x0 + 1
            y = wave.top() + int(round(y_n * h))
            lines.append(QtCore.QLine(x0, y, x1, y))
        painter.drawLines(lines)

    @staticmethod
    def _paint_envelope(painter: QtGui.QPainter, wave: QtCore.QRect,
                        lo_n: np.ndarray, hi_n: np.ndarray) -> None:
        """One vertical segment per column, from the column's max to its min.

        Drawing the min/max band as segments (rather than joining sample points) is what
        makes a one-column blockade visible: the segment spans the excursion even when it
        is narrower than a pixel.
        """
        n = int(lo_n.size)
        if n == 0 or wave.height() <= 0:
            return
        h = wave.height() - 1
        top = wave.top()
        left = wave.left()
        step = wave.width() / float(n)
        lines = []
        prev_lo = prev_hi = None
        for i in range(n):
            x = left + int(round(i * step))
            y_hi = top + int(round(float(hi_n[i]) * h))
            y_lo = top + int(round(float(lo_n[i]) * h))
            lines.append(QtCore.QLine(x, y_hi, x, max(y_lo, y_hi + 1)))
            # Join to the previous column so a fast edge reads as a continuous stroke
            # rather than two detached bars.
            if prev_lo is not None and (y_hi > prev_lo or y_lo < prev_hi):
                px = left + int(round((i - 1) * step))
                lines.append(QtCore.QLine(px, prev_lo if y_hi > prev_lo else prev_hi,
                                          x, y_hi if y_hi > prev_lo else y_lo))
            prev_lo, prev_hi = y_lo, y_hi
        painter.drawLines(lines)


class _StatusLine(QtWidgets.QLabel):
    """The footer's status line: a word-wrapping label that OWNS a fixed number of lines.

    WHY THE HEIGHT IS FIXED. This label shares the window's QVBoxLayout with the tile grid,
    and the grid's "Fit" page is one screenful re-derived from the viewport on every
    viewport resize (:meth:`EventGalleryWindow.eventFilter`). As a free-wrapping label its
    height was its text's height, so any clause that carried the line over a wrap boundary
    took a line's worth of viewport from the grid and re-fitted the page. Measured on the
    owner's -1000 mV recording (110 events, window 1000x700, page size Fit): the staleness
    sentence pushed by one Preprocess knob on the MAIN window grew the footer 45 -> 90 px,
    the page went 24 -> 18 tiles and 5 -> 7 pages, and with "Representative sample" ticked
    the dwell ladder was re-cut and 18 of the 24 events on screen were replaced -- the thing
    the page-size tooltip says a sample must never do; the knob turned back re-cut it again.
    That sentence was the longest clause, not the only one: the refusal, off-page,
    missed-event and hotkey notes all land on this line and a "Representative sample" tick
    adds a clause of its own, so shortening the sentence would have closed one door of
    five. With the height decided here, once, what the line says changes the text and
    nothing else.

    WHY IT STILL WRAPS. A label that cannot wrap reports its whole line as its minimum
    width: measured, this dozen-clause line demanded 1879 px on its own -- more than the
    two toolbar rows together -- and made the window's minimumSizeHint 1432 px on a
    1512 px display. Wrapping inside a fixed height keeps the window openable AND the grid
    still; the horizontal policy is Ignored so the label never sets a minimum width.

    LONGER THAN ITS LINES, THE LINE IS SHOWN FROM THE HEAD with the tail replaced by " …"
    and the whole of it in the tooltip; :meth:`text` returns the FULL line, so whatever
    reads the line (every assertion on it does) sees no cut, and :meth:`clipped` says
    whether the glass does. What goes first is the gesture hint at the end, never a count
    or a disclosure (see the clause order in ``_sync_controls``). The same pattern as the
    inspector's ``_FixedLinesNote``, written here rather than imported because this window
    may not reach a sibling GUI module (the guard is named at :data:`_LEVEL_HEX`).

    PLAIN TEXT, EXPLICITLY. This line interpolates strings the app did not author -- the
    trace name, the subpopulation report's source, the label vocabulary -- and a QLabel
    left at Qt::AutoText runs mightBeRichText() over whatever it is handed and PARSES IT
    AS HTML. A '<' followed by a recognised tag takes the rest of the name with it on the
    way to the screen ('Ch <A>' paints as 'Ch'), which is the defect
    baseline_panel._check_html was written for one panel over. Declaring the format is the
    whole fix and it cannot go stale the way escaping every interpolation would.
    """

    def __init__(self, lines: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("", parent)
        self._full = ""
        self._clipped = False
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        # The measuring twin: the same font, wrap and format, so what fits it at a width
        # is what this label paints at that width -- the same layout machinery is asked,
        # not a second one (bare font metrics) that can disagree with it.
        self._probe = QtWidgets.QLabel(self)
        self._probe.setWordWrap(True)
        self._probe.setTextFormat(Qt.PlainText)
        self._probe.hide()
        self._probe.setText("\n".join(["x"] * max(int(lines), 1)))
        self.setFixedHeight(self._probe.sizeHint().height() + 4)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = str(text)
        self._fit()

    def text(self) -> str:  # noqa: N802 (Qt override)
        """The WHOLE line, whether or not the glass had room for it."""
        return self._full

    def clipped(self) -> bool:
        """Whether the glass shows less than :meth:`text` (the rest is in the tooltip)."""
        return self._clipped

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def _fits(self, text: str) -> bool:
        self._probe.setText(text)
        return self._probe.heightForWidth(self.width()) <= self.height()

    @staticmethod
    def _head(full: str, cut: int) -> str:
        """The line cut before character ``cut`` (a word boundary), the separator it may
        end on dropped, and the ellipsis that says the rest was withheld."""
        return full[:cut].rstrip(" ·") + " …"

    def _fit(self) -> None:
        full = self._full
        shown, clipped = full, False
        if full and not self._fits(full):
            # Cut at a word boundary found by BISECTION, not by popping words off the tail:
            # the wrapped height is monotone in how much of the line is kept, and a linear
            # pop costs one text layout per word over a line that runs to a thousand
            # characters with the staleness sentence on it, rebuilt on every page turn.
            cuts = [i for i in range(1, len(full)) if full[i] == " " and full[i - 1] != " "]
            best = -1
            lo, hi = 0, len(cuts) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if self._fits(self._head(full, cuts[mid])):
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            shown = self._head(full, cuts[best]) if best >= 0 else "…"
            clipped = True
        if shown != QtWidgets.QLabel.text(self):
            QtWidgets.QLabel.setText(self, shown)
        self._clipped = clipped
        self.setToolTip(full if clipped else "")


class _ProgressLine(_StatusLine):
    """The curation progress line in the status bar: one reserved line, cut from the tail,
    whose tooltip always carries the full line and the detail behind it."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(1, parent)
        self._detail = ""
        # Its whole text is its hint: the window shares the bar by content (_share_status_bar).
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    def set_line(self, text: str, detail: str = "") -> None:
        self._detail = str(detail)
        changed = str(text) != self._full
        self.setText(text)
        if changed:
            self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self._full) + 8,
                            self.minimumHeight())

    def _fit(self) -> None:
        super()._fit()
        detail = getattr(self, "_detail", "")
        self.setToolTip("\n\n".join(s for s in (self._full, detail) if s))


#: Item data: an item's shorter display forms, longest first (see _ElidingCombo).
_SHORT_TEXTS_ROLE = int(Qt.ItemDataRole.UserRole) + 1


class _ElidingCombo(QtWidgets.QComboBox):
    """A fixed-width combo whose closed face shows the longest of the current item's
    ``_SHORT_TEXTS_ROLE`` forms that fits, so a label filter keeps its VALUE in view
    rather than its prefix. Spelled identically in both curation windows, which may not
    import each other."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt override)
        painter = QtWidgets.QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.Text))
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_ComboBox, opt)
        opt.currentText = self._face(opt)
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_ComboBoxLabel, opt)

    def shown_text(self) -> str:
        """What the closed face paints."""
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        return self._face(opt)

    def _face(self, opt: QtWidgets.QStyleOptionComboBox) -> str:
        forms = (self.itemData(self.currentIndex(), _SHORT_TEXTS_ROLE)
                 if self.currentIndex() >= 0 else None)
        if not forms:
            return opt.currentText
        field = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox, opt,
            QtWidgets.QStyle.SubControl.SC_ComboBoxEditField, self)
        fm = self.fontMetrics()
        fit = next((t for t in forms if fm.horizontalAdvance(t) <= field.width()), None)
        # Nothing fits: cut the shortest form in the middle, so both ends of the value stay.
        return fit if fit is not None else fm.elidedText(forms[-1], Qt.ElideMiddle,
                                                         field.width())


class _ElidedLabel(QtWidgets.QLabel):
    """A one-line label that gives up width before its neighbours do: it paints as much
    of its text as fits, ending in "…", while :meth:`text` returns all of it. Spelled
    identically in both curation windows, which may not import each other."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setTextFormat(Qt.PlainText)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = str(text)
        self.updateGeometry()
        self._fit()

    def text(self) -> str:  # noqa: N802 (Qt override)
        return self._full

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        m = self.contentsMargins()
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self._full) + 4
                            + m.left() + m.right(), super().sizeHint().height())

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        return QtCore.QSize(min(self.sizeHint().width(),
                                self.fontMetrics().horizontalAdvance("x" * 14)),
                            super().minimumSizeHint().height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 (Qt override)
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange:
            self.updateGeometry()
            self._fit()

    def _fit(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideRight,
                                              max(self.contentsRect().width() - 4, 0))
        if shown != QtWidgets.QLabel.text(self):
            QtWidgets.QLabel.setText(self, shown)


def _widest_count(fm: QtGui.QFontMetrics, n: int) -> str:
    """The widest number with as many digits as ``n`` in this font. Digits are
    proportional ("1" is narrower), so a width sized for ``n`` can clip a smaller count."""
    lead = max("123456789", key=fm.horizontalAdvance)
    rest = max("0123456789", key=fm.horizontalAdvance)
    return lead + rest * (len(str(max(int(n), 0))) - 1)


def _status_share(avail: int, want_p: int, want_l: int, keep_l: int,
                  floor_p: int) -> tuple[int, int]:
    """``(progress floor, label width)`` for ``avail`` px shared by the progress line and
    the project label, granted in turn: the progress line up to ``floor_p``, the label up
    to ``keep_l``, the progress line up to all of its text, the label up to all of its.
    What is left over goes to the progress line."""
    floor = min(want_p, floor_p)
    label = min(want_l, keep_l, max(avail - floor, 0))
    progress = min(want_p, max(avail - label, floor))
    return floor, min(want_l, max(avail - progress, label))


def _native_key(seq: str) -> str:
    """``"Ctrl+U"`` as this platform prints it (``⌘U`` on macOS)."""
    return QtGui.QKeySequence(seq).toString(QtGui.QKeySequence.SequenceFormat.NativeText)


def _read_advance_setting() -> bool:
    """The shared "Advance after decision" choice; any unreadable store means off."""
    try:
        value = QtCore.QSettings().value(_ADVANCE_SETTINGS_KEY, False)
    except Exception:  # noqa: BLE001 — a broken settings store is the default, never a crash
        return False
    return value is True or str(value).strip().lower() in ("true", "1")


def _write_advance_setting(on: bool) -> None:
    try:
        QtCore.QSettings().setValue(_ADVANCE_SETTINGS_KEY, bool(on))
    except Exception:  # noqa: BLE001
        pass


class _TileView(QtWidgets.QListView):
    """The tile grid. :meth:`select_only` moves the cursor as a plain click does -- the
    current tile, the only selected one, and the anchor a Shift-click extends from --
    whatever modifier is down. A bare ``setCurrentIndex`` reads the live modifiers: under
    the inspector's ⌘Z or ⌘U (relayed here) it toggled the tile INTO the selection."""

    _forced = False

    def selectionCommand(self, index, event=None):  # noqa: N802 (Qt override)
        if self._forced:
            return (QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows)
        return super().selectionCommand(index, event)

    def select_only(self, index: QtCore.QModelIndex) -> None:
        self._forced = True
        try:
            self.setCurrentIndex(index)
        finally:
            self._forced = False


class EventGalleryWindow(QtWidgets.QMainWindow):
    """Scan a whole detection as a grid of waveform tiles and curate it in bulk.

    Public API (also used by tests):
        set_events(doc_id, channel, events_df, module_id, provenance)
        set_levels(levels_df)                    # the CUSUM fit, drawn on the tiles
        clear()
        show_event(event_id)                     # turn to its page + focus it (inbound mirror)
        current_event_id() -> int | None
        inspect_current()                        # focused tile -> the Event Inspector
        selected_event_ids() -> list[int]
        select_all_shown() / clear_selection() / select_inverse()   # the batch gestures
        set_undo_state(enabled, label="")        # MainWindow's Edit ▸ Undo, mirrored here
        set_accepted_selected(bool)              # curate every selected tile
        accept_all() / reject_all()              # curate the whole CURRENT SUBSET
        reflect_acceptance(event_id, accepted)   # inbound mirror; never re-emits
        set_stale_columns(columns, note="")      # the events table's badge set, mirrored
        set_project_save_state(path, unsaved)    # MainWindow's project-save truth
        armed_digits() -> tuple[str, ...]        # which of 1–9 the palette arms
        set_filter(mode) / set_sort(mode) / set_tile_size(name)
        set_page(n) / page() / page_count() / set_page_size(n)
        tile_count() -> int                      # tiles on the current page
        event_count() -> int                     # events in the current filter subset
    """

    eventSelected = Signal(int)
    # Go to this event on the MAIN window's trace. Raised by the context menu's "Go to event
    # in trace" only: it used to be the double-click, and a gesture whose whole effect
    # happens in a window BEHIND this one reads as a gesture that did nothing.
    eventActivated = Signal(int)
    # Open this event in the Event Inspector. Three gestures raise it -- double-click, the
    # "Inspector…" button and the context menu -- because it is the surface the owner calls
    # "the curate", and it is a window of its own, so unlike the trace behind this one the
    # user can SEE the gesture land.
    inspectRequested = Signal(int)
    #: The tile menu's "Delete hand-made event": the ``event_id``. Intent only.
    deleteHandMadeRequested = Signal(int)
    acceptanceChanged = Signal(int, bool)
    curationBatchChanged = Signal()
    labelsChanged = Signal()                 # a label was written on the shared frame
    # An accept that recorded refusals but WROTE nothing. The written paths repaint every
    # surface through their data signals; this is the no-write path's equivalent — the
    # same contract as the events table's ``acceptanceRefused``, without which a refusal
    # recorded here reached the shared ledger but no other surface's note (measured:
    # "Accept all" right after detection said "3 tiles kept rejected" here while the
    # events table's label and an open inspector said nothing about the same frame).
    acceptanceRefused = Signal()
    # "Undo that" — raised by this window's Undo button and by its ⌘Z. INTENT, not the undo:
    # there is ONE undo stack in this application and it lives on MainWindow (core/history.py),
    # and a second one here would be a second answer to "what was the last step?". Measured
    # before this signal existed, on the owner's -1000 mV recording: Ctrl+A in the gallery
    # selected 45 tiles, R rejected them (110 accepted -> 65), MainWindow's Edit menu read
    # "Undo Curate 45 events" enabled — and QTest.keyClick(view, Key_Z, Ctrl) and
    # keyClick(gallery, Key_Z, Ctrl) both left it at 65, because the Undo QAction is a
    # WindowShortcut on a window that is now BEHIND this one and this window has no menu bar
    # (``menuBar().actions() == []``). Only clicking back to the main window undid it.
    # MainWindow connects this to that same action (see set_undo_state for the state coming
    # back the other way), so the gesture reaches the one stack from the window the batch
    # gesture happened in.
    undoRequested = Signal()
    # The user ticked/unticked "Advance after decision"; MainWindow mirrors it on the inspector.
    advanceAfterDecisionChanged = Signal(bool)
    # Project saving is intent only: MainWindow owns the path, dirty flag, validation and
    # serializer. These signals merely make saving reachable from the active gallery.
    saveProjectRequested = Signal()
    saveProjectAsRequested = Signal()

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("NanoScope — Event gallery")
        self.resize(940, 700)

        self._doc_id: str | None = None
        self._channel: int | None = None
        self._df: pd.DataFrame | None = None
        #: Did a DETECTION put that ``None`` there? ``_df`` collapses an empty frame to
        #: ``None`` so every row-indexed path below can ask one question -- but "a detection
        #: ran and found nothing" and "no detection has ever been handed to this window" are
        #: two different states, and for a null control, a conservative threshold or a quiet
        #: pore the first one is a RESULT. Sharing the never-ran sentence told the user to
        #: repeat the run they had just finished and withheld its answer, while the Event
        #: inspector, on the identical result, said "Detection found no events." Re-derived
        #: on every :meth:`set_events` / :meth:`clear`, never left over from a previous one.
        self._ran_empty: bool = False
        self._module_id: str | None = None
        self._units: str = ""
        self._tile_name: str = _DEFAULT_TILE
        self._updating = False        # suppress control slots during programmatic changes
        #: One depth scale for every tile (see envelope_for). Default ON: the grid is for
        #: comparing events, and a per-tile scale makes every event look the same depth.
        self._shared_scale = True
        self._shared_span_cache: float | None = None
        #: The tiles the LAST selection could not be kept whole through a frame edit, by
        #: event_id — because a curation write (this window's or another surface's) moved
        #: them out of the filter's subset, onto another page, or off a re-cut representative
        #: sample. A curation gesture no longer narrows the selection (see
        #: :meth:`_set_rows_accepted`); where a reset genuinely cannot keep a tile, the
        #: status line says how many and where each went, and :meth:`_offpage_note`
        #: re-derives that from the live subset so the sentence disappears of its own
        #: accord when the filter (or the page) brings them back. Replaced by this window's
        #: own write, extended by an inbound reset, and retired by a new frame or by a
        #: selection the user makes by hand (:meth:`_on_selection_changed`).
        self._offpage_ids: tuple[int, ...] = ()
        #: The event the last INBOUND :meth:`show_event` could not reach, by event_id,
        #: because the current subset (the Show filter, a subpopulation class, the
        #: representative sample) excludes it. State, like ``_offpage_ids``, so the status
        #: line can rebuild the sentence; re-derived by :meth:`_missed_note`, which prints
        #: it only while the event is still excluded and no tile is focused.
        self._missed_id: int | None = None
        #: What the last digit press did NOT do -- an unarmed key, or an armed one pressed
        #: over an empty selection. Cleared by the next gesture that writes something.
        self._hotkey_note: str = ""
        #: Columns the owner has marked as measured by a superseded estimator (the events
        #: table's badge set, pushed through :meth:`set_stale_columns`), and its sentence.
        self._stale_columns: tuple[str, ...] = ()
        self._stale_note_text: str = ""
        self._fit_pages = True        # page size follows the window; see _PAGE_SIZES
        #: The label palette. Owned by MainWindow and handed in, so the gallery, the
        #: inspector and the project all read one vocabulary rather than three.
        self._vocab = LabelVocabulary()
        #: Dynamic non-digit label actions. Digits are permanent automatic slots; these
        #: are rebuilt from the safe vocabulary mapping whenever the palette changes.
        self._custom_label_actions: list[QtGui.QAction] = []
        self._label_key: str = ""     # which key the tiles show, "" = the first in use
        #: event_id -> subpopulation id, from a finished report. None = no report installed, in
        #: which case the subpopulation row of the toolbar is hidden entirely rather than shown
        #: empty: an inert control is a claim that there is something to control.
        self._subpops: dict[int, int] | None = None
        self._subpop_source: str = ""
        self._subpop_caveats: tuple[str, ...] = ()
        self._trace_name: str = TRACE_CALIBRATED
        #: WHICH EVENTS this window has declined to accept on the frame it holds -- not how
        #: many, and not "the last gesture's".
        #: STATE, not text: :meth:`_sync_controls` rebuilds the status line from scratch and
        #: is re-entered by every listener of ``curationBatchChanged`` (MainWindow's surface
        #: refresher calls ``refresh()``), so a note appended to the line AFTER the rebuild
        #: survived only until the next repaint -- i.e. was never read. See _refusal_note.
        #:
        #: Event IDENTITIES and not counts, because the counter is a COUNT OF
        #: CURRENTLY-REFUSED ROWS and not a latch: the table's checkbox can accept one of
        #: them at any time, and a stored ``4`` cannot answer "and how many now?". See
        #: :class:`_RefusalLedger` for what does and does not clear it -- and for the
        #: measured defect that came of clearing it in this window's own write path.
        self._refusals = _RefusalLedger()
        #: The detrended array, held only while it is offered. A REFERENCE to MainWindow's
        #: array; see _EnvelopeSource.set_override for why it is never copied.
        self._pending_detrended: np.ndarray | None = None
        #: The CUSUM level table (analysis.cusum.LEVEL_SCHEMA) for this detection, or None --
        #: the inspector's ``_levels`` by the same name and the same rule. Dropped with the
        #: frame for the reason the subpopulation map is: levels fitted on one detection,
        #: printed under another detection's ids, are a picture of the wrong event.
        self._levels: pd.DataFrame | None = None
        #: What this window believes Edit ▸ Undo would do, as ``(enabled, label)``. A LOWER
        #: BOUND kept by this window's own writes and OVERWRITTEN by :meth:`set_undo_state`
        #: when MainWindow pushes the real one: every curation this window performs pushes an
        #: undo step on the shared stack (``acceptanceChanged`` and ``curationBatchChanged``
        #: are both checkpointed there), so "this window has just written" is a fact, not a
        #: guess -- but "there is nothing to undo" is NOT knowable from here, which is why the
        #: affordance starts disabled and never re-derives itself downwards.
        self._undo_enabled = False
        self._undo_label = ""
        #: The selection size the status line currently prints, so a selection signal that
        #: does not move the number does not rebuild the line. See _on_selection_changed.
        self._shown_sel_count = 0
        # -- curation throughput: the inspector's state by the same names ----------------
        self._advancing = False          # fence: the cursor move an advance itself makes
        self._advanced_from: int | None = None
        self._note: tuple[str, str] = ("", "")   # (short, full) on the progress line
        self._progress_state = None
        self._filter_items: tuple[str, ...] = ()
        #: ``(before, after, cursor)`` event ids around a cursor tile that a write took out
        #: of the subset: the cursor waits, unselected, and ←/→ land on before / after.
        self._parked: tuple[int | None, int | None, int] | None = None

        self._envelopes = _EnvelopeSource(session)
        self._model = _TileModel(self)
        self._build_ui()
        self._install_project_save_controls()
        self._install_curation_status()
        # The keys the window binds go to the control that USES them -- Space to a focused
        # button, any bare key to an open dropdown or menu -- not to the tiles behind it
        # (gui/curation_keys.py). Built last: it hooks every dropdown and menu made above.
        self._key_guard = CurationKeyGuard(self)
        self._sync_controls()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # TWO ROWS, NOT ONE. Measured: as a single row this bar's
        # ``minimumSizeHint().width()`` was 1432 px at .AppleSystemUIFont 13 pt, with
        # "Accept selected" / "Reject selected" last in it -- so on the owner's 1512 px
        # display, any window narrower than the whole screen (and every window at all once
        # the "Inspector…" button below joined the row) clipped the per-event controls off
        # the right-hand end. A QHBoxLayout does not wrap: the widgets do not move down,
        # they cease to be reachable. Splitting by WHAT THE CONTROL ACTS ON -- the whole
        # grid above, the selected tiles below -- is the division that survives a narrow
        # window, because the two halves shrink independently.
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QtWidgets.QLabel("Show"))
        self.filter_box = _ElidingCombo()
        # A fixed width, so label filters arriving with a palette never shift the row.
        self.filter_box.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.filter_box.setMinimumContentsLength(11)
        self._rebuild_filter_items()
        self.filter_box.setToolTip(
            "Which events the grid holds. Unreviewed: events no person has ruled on yet "
            "(accept, reject or a label), including events with no review record. "
            "Unlabelled: no key=value label. Labelled: events carrying that label (a merge "
            "conflict counts under each value).")
        self.filter_box.currentTextChanged.connect(self._on_filter_changed)
        bar.addWidget(self.filter_box)

        bar.addWidget(QtWidgets.QLabel("Sort"))
        self.sort_box = QtWidgets.QComboBox()
        self.sort_box.addItems(_SORTS)
        self.sort_box.currentTextChanged.connect(self._on_sort_changed)
        bar.addWidget(self.sort_box)

        bar.addWidget(QtWidgets.QLabel("Tiles"))
        self.size_box = QtWidgets.QComboBox()
        self.size_box.addItems(list(_TILE_SIZES))
        self.size_box.setCurrentText(_DEFAULT_TILE)
        self.size_box.currentTextChanged.connect(self._on_size_changed)
        bar.addWidget(self.size_box)

        self.shared_scale_box = QtWidgets.QCheckBox("Same depth scale")
        self.shared_scale_box.setChecked(True)
        self.shared_scale_box.setToolTip(
            "Draw every tile against one depth scale for the whole detection, so a deep "
            "event looks deeper than a shallow one. Uncheck to scale each tile to itself, "
            "which is better for comparing event SHAPE when the depths are similar.")
        self.shared_scale_box.toggled.connect(self._on_shared_scale_toggled)
        bar.addWidget(self.shared_scale_box)

        bar.addStretch(1)
        root.addLayout(bar)

        # -- row two: everything that acts on the SELECTED tiles ------------------------------
        sel_bar = QtWidgets.QHBoxLayout()
        sel_bar.setSpacing(8)

        # -- THE BATCH GESTURES, AS A CONTROL AND NOT ONLY AS KEYS ---------------------------
        # ONE menu button rather than three push buttons, for two measured reasons. (1) WIDTH:
        # this row's ``minimumSizeHint().width()`` is the budget the two-row split above was
        # made to protect, and three more buttons ("Select all" / "Select none" / "Select
        # inverse", ~95 px each) spend it; the button is ~80 px and holds all three.
        # (2) BROWSABILITY: this window has NO MENU BAR (``menuBar().actions() == []``), so
        # its sixteen shortcuts are unlabelled QActions with nowhere to be read -- the audit's
        # finding for why the owner believed the gallery curates one event at a time. A menu
        # is the one widget that shows a gesture's NAME and its KEY together, so these three
        # actions are created once and shown in both places: the window (where the shortcut
        # fires) and this menu (where it can be found).
        self.select_menu = QtWidgets.QMenu(self)
        self.act_select_all = self.select_menu.addAction("All on this page")
        self.act_select_all.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)
        self.act_select_all.setToolTip(
            "Select every tile on this page, ready for A / R / a digit or the "
            "\"…selected\" buttons.")
        self.act_select_all.triggered.connect(self.select_all_shown)
        self.act_select_none = self.select_menu.addAction("None")
        self.act_select_none.setShortcut(QtGui.QKeySequence(Qt.Key_Escape))
        self.act_select_none.setToolTip(
            "Clear the selection, leaving every tile as it is. The keyboard cursor stays "
            "where it is, so the arrow keys carry on from the same tile.")
        self.act_select_none.triggered.connect(self.clear_selection)
        # NOT "invert". ``Space`` already means "invert the ACCEPTANCE of the selected
        # tiles" in this window (see _toggle_selected) -- a destructive gesture -- and
        # shipping a harmless one under the same word, one keystroke away, is how a user
        # rejects a page they meant to re-select. "Inverse" is the selection; "invert" is
        # still the acceptance.
        self.act_select_inverse = self.select_menu.addAction("Inverse")
        self.act_select_inverse.setShortcut(QtGui.QKeySequence("Ctrl+I"))
        self.act_select_inverse.setToolTip(
            "Select the tiles on this page that are NOT selected, and deselect the ones "
            "that are — the way to keep a handful and curate all the rest.")
        self.select_menu.setToolTipsVisible(True)
        for _act in (self.act_select_all, self.act_select_none, self.act_select_inverse):
            # WidgetWithChildren, not Window -- see _handle_own_key. ⌘A is the events
            # table's own "select every row" too, and a shortcut this window resolves
            # against the ACTIVE window would answer for it.
            _act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            self.addAction(_act)          # the SAME action: one gesture, one implementation
        self.act_select_inverse.triggered.connect(self.select_inverse)

        self.select_btn = QtWidgets.QToolButton()
        self.select_btn.setText("Select")
        self.select_btn.setMenu(self.select_menu)
        self.select_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.select_btn.setToolTip(
            "Select tiles in bulk. Every curation control on this row acts on the whole "
            "selection, so this is how one gesture curates many events.")
        sel_bar.addWidget(self.select_btn)

        sel_bar.addWidget(QtWidgets.QLabel("Label"))
        self.label_box = QtWidgets.QComboBox()
        self.label_box.setMinimumWidth(150)
        self.label_box.setToolTip(
            "The label the number keys and \"Apply\" write onto the selected tiles. "
            "Edit the list with \"Labels…\".")
        sel_bar.addWidget(self.label_box)
        self.apply_label_btn = QtWidgets.QPushButton("Apply")
        self.apply_label_btn.setToolTip(
            "Write the chosen label onto every selected tile. 1–9 do the same for the first "
            "nine labels in the list.")
        self.apply_label_btn.clicked.connect(self.apply_selected_label)
        sel_bar.addWidget(self.apply_label_btn)
        self.clear_label_btn = QtWidgets.QPushButton("Clear")
        self.clear_label_btn.setToolTip(
            "Remove that label's KEY from every selected tile — the other labels and every "
            "flag on the row are left alone.")
        self.clear_label_btn.clicked.connect(self.clear_selected_label)
        sel_bar.addWidget(self.clear_label_btn)

        sel_bar.addStretch(1)

        # The way OUT of the grid, on the bar rather than only in the context menu (owner
        # notes #2/#4: the three event surfaces should link to each other). It emits the
        # same ``inspectRequested`` the menu does -- MainWindow already routes that -- so
        # the button adds a road to the inspector, not a second implementation of one.
        self.inspect_btn = QtWidgets.QPushButton("Inspector…")
        self.inspect_btn.setToolTip(
            "Open the focused tile's event in the Event Inspector, where its bounds, its "
            "sub-state levels and its flags can be edited. Double-clicking a tile does the "
            "same; right-click a tile for the trace instead.")
        self.inspect_btn.clicked.connect(self.inspect_current)
        sel_bar.addWidget(self.inspect_btn)

        self.accept_selected_btn = QtWidgets.QPushButton("Accept selected")
        self.accept_selected_btn.setToolTip(
            f"Mark every selected tile accepted (drag a box, shift-click for a run, or "
            f"{_cmd_key()}-click to add one — see \"Select\").")
        self.accept_selected_btn.clicked.connect(lambda: self.set_accepted_selected(True))
        sel_bar.addWidget(self.accept_selected_btn)

        self.reject_selected_btn = QtWidgets.QPushButton("Reject selected")
        self.reject_selected_btn.setToolTip("Mark every selected tile rejected.")
        self.reject_selected_btn.clicked.connect(lambda: self.set_accepted_selected(False))
        sel_bar.addWidget(self.reject_selected_btn)

        root.addLayout(sel_bar)

        # -- the subpopulation row. Hidden until a report is installed. ----------------------
        self._subpop_bar = QtWidgets.QWidget()
        sub = QtWidgets.QHBoxLayout(self._subpop_bar)
        sub.setContentsMargins(0, 0, 0, 0)
        sub.setSpacing(8)
        sub.addWidget(QtWidgets.QLabel("Subpopulation"))
        self.subpop_box = QtWidgets.QComboBox()
        self.subpop_box.setMinimumWidth(210)
        self.subpop_box.setToolTip(
            "Show only the events one subpopulation of the last report was assigned to. The "
            "assignment is the mixture's, it is provisional, and the caveats below say why — "
            "it is not written to the events table and never reaches an export from here.")
        self.subpop_box.currentIndexChanged.connect(self._on_subpop_changed)
        sub.addWidget(self.subpop_box)

        self.representative_box = QtWidgets.QCheckBox("Representative sample")
        self.representative_box.setToolTip(_REPRESENTATIVE_TIP)
        self.representative_box.toggled.connect(self._on_representative_toggled)
        sub.addWidget(self.representative_box)

        sub.addWidget(QtWidgets.QLabel("Trace"))
        self.trace_box = QtWidgets.QComboBox()
        self.trace_box.setToolTip(_TRACE_TIP)
        self.trace_box.addItems([TRACE_CALIBRATED])
        self.trace_box.currentTextChanged.connect(self._on_trace_changed)
        sub.addWidget(self.trace_box)
        sub.addStretch(1)
        root.addWidget(self._subpop_bar)
        self._subpop_bar.setVisible(False)

        # -- the caveats. A PANE, not a tooltip and not a tab. --------------------------------
        # Same decision as the report window's caveat strip, for the same reason: the moment
        # this window can draw four tidy classes of tiles it can also be read as evidence that
        # there are four classes, and the reasons that may not be so have to be on the same
        # screen. It appears with the subpopulation row and cannot be put away.
        self._caveat_box = QtWidgets.QGroupBox("Caveats")
        cav = QtWidgets.QVBoxLayout(self._caveat_box)
        cav.setContentsMargins(6, 4, 6, 6)
        self.caveat_text = QtWidgets.QPlainTextEdit()
        self.caveat_text.setReadOnly(True)
        self.caveat_text.setMaximumHeight(96)
        self.caveat_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        cav.addWidget(self.caveat_text)
        root.addWidget(self._caveat_box)
        self._caveat_box.setVisible(False)

        self.view = _TileView()
        self.view.setModel(self._model)
        self.view.setItemDelegate(_TileDelegate(self))
        self.view.setViewMode(QtWidgets.QListView.IconMode)
        self.view.setResizeMode(QtWidgets.QListView.Adjust)
        self.view.setMovement(QtWidgets.QListView.Static)
        self.view.setWrapping(True)
        self.view.setSpacing(2)
        # UniformItemSizes + Batched layout are what let the view lay out ten thousand
        # tiles without measuring each one; without them the model reset itself is the
        # thing that stalls, before a single tile is painted.
        self.view.setUniformItemSizes(True)
        self.view.setLayoutMode(QtWidgets.QListView.Batched)
        self.view.setBatchSize(256)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.view.setSelectionRectVisible(True)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.selectionModel().currentChanged.connect(self._on_current_changed)
        # A selection the USER makes retires the off-page record; see _on_selection_changed.
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.view.doubleClicked.connect(self._on_double_clicked)
        # Right-click: the direct road to the OTHER surfaces (owner notes #2/#5 — the
        # gallery, the inspector and the main panel should link to each other, not only
        # through the Analysis menu). A context menu rather than more toolbar buttons:
        # the actions are per-tile, and the bar is already full of per-page ones.
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._on_context_menu)
        # THE VIEWPORT'S OWN RESIZE IS WHAT "FIT" HAS TO BE COMPUTED FROM. See eventFilter.
        self.view.viewport().installEventFilter(self)
        root.addWidget(self.view, 1)

        pager = QtWidgets.QHBoxLayout()
        pager.setSpacing(6)
        self.first_btn = QtWidgets.QToolButton()
        self.first_btn.setText("|◀")
        self.first_btn.setToolTip("First page (Home)")
        self.first_btn.clicked.connect(lambda: self.set_page(0))
        pager.addWidget(self.first_btn)
        self.prev_btn = QtWidgets.QToolButton()
        self.prev_btn.setText("◀")
        self.prev_btn.setToolTip("Previous page (PgUp, or ←)")
        self.prev_btn.clicked.connect(lambda: self.set_page(self._model.page() - 1))
        pager.addWidget(self.prev_btn)

        self.page_spin = QtWidgets.QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(1)
        self.page_spin.setToolTip("Jump to a page.")
        self.page_spin.valueChanged.connect(self._on_page_spin)
        pager.addWidget(self.page_spin)
        self.page_label = QtWidgets.QLabel("of 1")
        self.page_label.setStyleSheet(f"color: {_CAPTION_HEX};")
        pager.addWidget(self.page_label)

        self.next_btn = QtWidgets.QToolButton()
        self.next_btn.setText("▶")
        self.next_btn.setToolTip("Next page (PgDn, or →)")
        self.next_btn.clicked.connect(lambda: self.set_page(self._model.page() + 1))
        pager.addWidget(self.next_btn)
        self.last_btn = QtWidgets.QToolButton()
        self.last_btn.setText("▶|")
        self.last_btn.setToolTip("Last page (End)")
        self.last_btn.clicked.connect(lambda: self.set_page(self._model.page_count() - 1))
        pager.addWidget(self.last_btn)

        pager.addSpacing(12)
        pager.addWidget(QtWidgets.QLabel("Per page"))
        self.page_size_box = QtWidgets.QComboBox()
        self.page_size_box.addItems(list(_PAGE_SIZES))
        self.page_size_box.setToolTip(
            "Tiles per page. \"Fit\" sizes the page to the window, so a page is exactly one "
            "screenful and turning the page is the only way to move.\n\n"
            "A number is taken literally, and a page of 96 in a small window holds tiles "
            "below the fold — the status line says so, and the page scrolls to them. It is "
            "literal because under a representative subpopulation sample this control IS the "
            "sample size, and a sample quietly cut to fit a window is not a sample.")
        self.page_size_box.setCurrentText(_FIT)
        self.page_size_box.currentTextChanged.connect(self._on_page_size_changed)
        pager.addWidget(self.page_size_box)
        pager.addStretch(1)
        # Right-aligned, so a page count that changes width never slides it.
        self._advance_check = QtWidgets.QCheckBox("Advance after decision")
        self._advance_check.setObjectName("advanceAfterDecision")
        # Tab only. It was NoFocus so that Space could not hit it, which left it no keyboard
        # route at all; Space now belongs to a focused button (gui/curation_keys.py), and a
        # click still leaves the keyboard on the tiles.
        self._advance_check.setFocusPolicy(Qt.TabFocus)
        self._advance_check.setChecked(_read_advance_setting())
        self._advance_check.setToolTip(
            "A keyboard decision on ONE selected tile — A, R, Space or a label key — moves "
            "the selection to the next tile of the Show subset once it is written, turning "
            "the page when needed. A selection of several tiles never moves, and neither do "
            "the buttons. At the end of the subset the selection stays and the status bar "
            "says so. Shared with the Event inspector and remembered between sessions; off "
            "by default.")
        self._advance_check.toggled.connect(self._on_advance_toggled)
        pager.addWidget(self._advance_check)
        root.addLayout(pager)

        foot = QtWidgets.QHBoxLayout()
        # A FIXED NUMBER OF LINES, WHATEVER THE LINE SAYS. It was a plain wrapping QLabel,
        # and its height -- its text's height -- was the one input to the grid's "Fit" page
        # that nothing about the window controlled: a longer line took rows from the grid
        # above and re-cut the page (and, under a representative sample, the sample).
        # Plain text and word wrap are decided inside the class, with the measurements.
        self.status = _StatusLine(_STATUS_LINES)
        foot.addWidget(self.status, 1)
        # THE WAY BACK, IN THE WINDOW THE MISTAKE IS MADE IN. One R over a swept page curates
        # every tile on it, and until now the only road back was the MAIN window's Edit menu,
        # behind this one, unnamed by anything on this screen. The button emits intent; the
        # stack stays MainWindow's (see undoRequested). Disabled until there is something it
        # could undo -- a control that claims a step that does not exist is the same lie as a
        # shortcut that does nothing.
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(lambda _=False: self._request_undo())
        foot.addWidget(self.undo_btn)
        self.accept_all_btn = QtWidgets.QPushButton("Accept all shown")
        self.accept_all_btn.setToolTip(_ACCEPT_ALL_TIP)
        self.accept_all_btn.clicked.connect(self.accept_all)
        foot.addWidget(self.accept_all_btn)
        self.reject_all_btn = QtWidgets.QPushButton("Reject all shown")
        self.reject_all_btn.setToolTip("Mark every tile in the current filter rejected.")
        self.reject_all_btn.clicked.connect(self.reject_all)
        foot.addWidget(self.reject_all_btn)
        root.addLayout(foot)

        self.setCentralWidget(central)

        # ⌘Z / Ctrl+Z, at the STANDARD sequence and the same WindowShortcut context
        # MainWindow's own Edit ▸ Undo uses -- the two never compete, because only one of the
        # two windows is active. Safe next to this window's page spin box for the reason that
        # action's docstring records: Qt sends a ShortcutOverride first and QLineEdit accepts
        # it for the editing keys, so ⌘Z in the page field still undoes the typing.
        self.act_undo = QtGui.QAction("Undo", self)
        self.act_undo.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.act_undo.setShortcutContext(Qt.WindowShortcut)
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(lambda _=False: self._request_undo())
        self.addAction(self.act_undo)
        self._sync_undo_affordance()

        # THIS WINDOW'S KEYS ARE RESOLVED AGAINST THIS WINDOW'S OWN WIDGETS, NOT AGAINST
        # WHICHEVER WINDOW QT CALLS ACTIVE. ``Qt.WindowShortcut`` asks only "is my window
        # the active one", and the answer is a lie whenever the focus and the activation
        # disagree -- which is exactly the state a second curation window creates. Measured
        # in the wired app on the owner's -1000 mV recording, gallery open, six rows
        # selected in the events table and 'r' sent to the TABLE's view: the gallery's
        # window-context R took the key, ``EventTable._handle_curation_key`` was never
        # called ("handler calls: []"), and one unrelated event -- the gallery's own current
        # tile -- was rejected instead of the six the user had highlighted. That is a silent
        # write to curation, on a project where curation decides the final dataset.
        # ``WidgetWithChildrenShortcut`` asks the question that actually decides who the key
        # is for: Qt walks up from ``QApplication.focusWidget()`` and stops at the first
        # window, so the shortcut matches when (and only when) the focused widget lives in
        # here. With the same probe after the change: "handler calls: [82]", rows 10-15 all
        # rejected, and nothing in the gallery moved.
        #
        # This does NOT weaken the gallery's own A/R -- see _handle_own_key for the other
        # half, which covers the one state the narrower context would otherwise cost.
        # A / R / Space / the label keys go through the decide_* doors (which move on only
        # with "Advance after decision" on) and never auto-repeat: a held R must not curate
        # a run of tiles nobody looked at.
        for seq, slot, repeat in (
                (QtGui.QKeySequence(Qt.Key_A), lambda: self.decide_selected(True), False),
                (QtGui.QKeySequence(Qt.Key_R), lambda: self.decide_selected(False), False),
                (QtGui.QKeySequence(Qt.Key_Space), self.decide_toggle_selected, False),
                (QtGui.QKeySequence(Qt.Key_PageDown),
                 lambda: self.set_page(self._model.page() + 1), True),
                (QtGui.QKeySequence(Qt.Key_PageUp),
                 lambda: self.set_page(self._model.page() - 1), True),
                (QtGui.QKeySequence(Qt.Key_Home), lambda: self.set_page(0), True),
                (QtGui.QKeySequence(Qt.Key_End),
                 lambda: self.set_page(self._model.page_count() - 1), True),
                (QtGui.QKeySequence("Ctrl+U"), lambda: self.first_unreviewed(), False)):
            act = QtGui.QAction(self)
            act.setShortcut(seq)
            act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            act.setAutoRepeat(repeat)
            act.triggered.connect(slot)
            self.addAction(act)
        # 1-9 write the palette's first nine labels onto the selection. This is the whole
        # point of labelling in the gallery: sweep a box over a run of tiles, press a digit.
        #
        # Window-level, and that is safe: Qt hands a key press to a focused QAbstractSpinBox
        # or QLineEdit BEFORE a window shortcut sees it, so typing a page number into the
        # pager still works. Measured both ways — window shortcut and an event filter on the
        # view — and the spin box reads "2" either way; the event filter only cost the
        # digits their reach after a toolbar click.
        for n in range(1, 10):
            act = QtGui.QAction(self)
            act.setShortcut(QtGui.QKeySequence(str(n)))
            act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            act.setAutoRepeat(False)
            act.triggered.connect(lambda _=False, ch=str(n): self.decide_label(ch))
            self.addAction(act)
        # AND THE SAME KEYS WHEN NO WIDGET IN HERE HOLDS THE FOCUS. See _handle_own_key:
        # narrowing the context above is what stops this window answering for the events
        # table, and this filter is what keeps the narrowing from costing this window its
        # own keys. On the VIEW, because QListView's type-to-search accepts a bare letter
        # or digit and would swallow A / R / 1-9 before they could reach this window.
        self.view.installEventFilter(self)

    def _install_project_save_controls(self) -> None:
        """Install project Save / Save As for this top-level window.

        The gallery deliberately has no menu bar. A compact status-bar control keeps the
        project state visible, gives pointer users a direct Save button plus a Save-As
        drop-down, and the window-scoped actions make the platform-standard shortcuts work
        while this window is active. Both actions emit intent; only MainWindow serializes.
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

        self._project_save_label = _ElidedLabel(self)
        self._project_save_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._project_save_label.setAccessibleName("Project save status")

        save_bar = self.statusBar()
        save_bar.setSizeGripEnabled(False)
        # No stretch: it takes the width its text needs and no more; the progress line
        # beside it takes the rest (see _install_curation_status).
        save_bar.addPermanentWidget(self._project_save_label, 0)
        save_bar.addPermanentWidget(self._project_save_button)
        self.set_project_save_state(None, False)

    def _install_curation_status(self) -> None:
        """First unreviewed + the progress line at the left of the status bar -- the
        inspector's container by the same object names. In the status bar, not the
        footer: the footer's height is the Fit page's input and must not change."""
        self._curation_status = QtWidgets.QWidget(self)
        self._curation_status.setObjectName("curationStatus")
        lay = QtWidgets.QHBoxLayout(self._curation_status)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._first_unreviewed_btn = QtWidgets.QToolButton(self._curation_status)
        self._first_unreviewed_btn.setText("First unreviewed")
        self._first_unreviewed_btn.setObjectName("firstUnreviewedButton")
        self._first_unreviewed_btn.setFocusPolicy(Qt.NoFocus)
        self._first_unreviewed_btn.setToolTip(
            f"Select the first tile of the Show subset, in the current sort, that no person "
            f"has ruled on yet ({_native_key('Ctrl+U')}), turning to its page. Events with "
            "no review record count as unreviewed.")
        self._first_unreviewed_btn.clicked.connect(lambda _=False: self.first_unreviewed())
        lay.addWidget(self._first_unreviewed_btn)
        self._progress = _ProgressLine(self._curation_status)
        self._progress.setObjectName("curationProgress")
        # A floor; past it the line and the project label share the bar by content
        # (_share_status_bar, the inspector's rule).
        self._progress.setMinimumWidth(_PROGRESS_MIN_W)
        lay.addWidget(self._progress, 1)
        self.statusBar().insertPermanentWidget(0, self._curation_status, 1)
        self._render_progress()

    def set_project_save_state(self, path: str | None, unsaved: bool) -> None:
        """Show MainWindow's authoritative project path and unsaved state.

        This is presentation only. The child neither infers dirty state from its own edits
        nor clears it after requesting a save; cancellation, failure, and deliberately
        incomplete saves therefore remain visibly unsaved until the owner says otherwise.
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
        # The save state stays readable however narrow the bar gets; the name gives way.
        floor = QtGui.QFontMetrics(font).horizontalAdvance(text.split(" — ")[0] + " …")
        self._project_save_label.setMinimumWidth(
            min(floor + 4, self._project_save_label.sizeHint().width()))
        self._share_status_bar()

    # -- contents --------------------------------------------------------------------------------
    def set_events(self, doc_id: str, channel: int, events_df: pd.DataFrame | None,
                   module_id: str | None = None,
                   provenance: dict[str, Any] | None = None,
                   refusals: "_RefusalLedger | None" = None,
                   keep_place: bool = False, quiet: bool = False) -> None:
        """Install the CURRENT detection's shared frame (a reference, never a copy).

        ``keep_place`` is MainWindow saying this re-feed edits the table on screen (an undo or
        redo, a hand-made add, a Delete). When the new frame holds the very same events, or
        those and more, or those less some (``can_keep_place``: each kept ``event_id`` at its
        ``start_idx``), the selection and the cursor are put back -- quietly, emitting nothing
        -- instead of the cursor landing on tile 0, which announced a selection and moved the
        trace, the table and the inspector with it. When the cursor's tile is one of those
        gone, the tile that took its place is selected, else the one before it. When the grid
        held no tile (the Show filter held none, or the table none at all) and the new frame
        fills it, the first tile is selected, as quietly. Anything else gets today's reset.

        ``quiet`` is MainWindow saying this re-feed is a VIEW switch (the Results dock's
        Table selector put another stored table on screen): a different frame, so the
        cursor still starts on tile 0, but nothing is announced -- the trace, the table and
        the inspector stay where the curator left them, as they do for an edit the app made
        itself (the same-lineage branch below).

        ``refusals`` is the refusal ledger to adopt WITH the frame — in the wired app,
        MainWindow's per-detection-lineage ledger, shared with the inspector and the
        events table so all three surfaces disclose one record in one voice. Handing it
        in is also the owner saying whether the standing record survives this re-feed:
        the same ledger arrives when the lineage did (an undo restored the detection, a
        merge edited it — measured, both used to erase the note here while every counted
        row stayed refused), and a fresh one when a new detection replaced it. Without it
        (a standalone window) this method keeps its own rule: a different frame is a
        different detection, and the window's private ledger is dropped."""
        keep = (bool(keep_place) and self._df is not None and events_df is not self._df
                and doc_id == self._doc_id and channel == self._channel
                and can_keep_place(self._df, events_df))
        # The owner handing back the ledger this window already holds says it is the same
        # detection, edited by the app itself (a hand-add, a merge, their undo).
        same_lineage = (self._df is not None and refusals is not None
                        and refusals is self._refusals
                        and doc_id == self._doc_id and channel == self._channel)
        keep_sel = ([int(self._df["event_id"].iloc[r]) for r in self._selected_rows()]
                    if keep else [])
        keep_cur = self._current_event_id() if keep else None
        # The tiles in the order they were drawn: where to land if the cursor's tile is among
        # those the new frame has lost.
        drawn = (self._df["event_id"].to_numpy(dtype=np.int64)[self._model.df_rows()]
                 if keep else None)
        # NO TILE TO KEEP: the Show filter held none (a Delete took its last event), or the
        # table held none at all, which this window keeps as no frame -- the same table on
        # screen, as nothing else is left to compare. A re-feed that keeps the place and fills
        # the grid again selects its first tile (``landing_after_refill``); the filter is the
        # model's and survives any frame. The restored tile used to be left unselected (the
        # undo of the Delete), or, with no frame left, tile 0 announced as a click.
        refill = bool(keep_place) and (
            (keep and not len(drawn))
            or (self._df is None and self._ran_empty and events_df is not None
                and doc_id == self._doc_id and channel == self._channel
                and module_id == self._module_id))
        self._doc_id = doc_id
        self._channel = channel
        self._module_id = module_id
        self._df = events_df if (events_df is not None and len(events_df)) else None
        # A FRAME ARRIVED. Even an empty one: a detection ran here, and the status line has
        # to be able to say so rather than deny the run (see :attr:`_ran_empty`).
        self._ran_empty = events_df is not None and not len(events_df)
        self._units = self._resolve_units()
        self._envelopes.set_target(doc_id, channel)
        self._shared_span_cache = None
        # A DIFFERENT FRAME IS A DIFFERENT DETECTION: ids the last gesture could not keep
        # selected point at events that are not here any more (see :attr:`_offpage_ids`).
        self._offpage_ids = ()
        self._missed_id = None
        self._parked = None
        self._hotkey_note = ""
        self._model.set_frame(self._df)
        # A DIFFERENT FRAME IS A DIFFERENT DETECTION, so any installed subpopulation map is
        # about events that no longer exist under those ids. Dropped here rather than
        # re-indexed: re-indexing by event_id would quietly re-attach a mixture fitted on one
        # detection to the output of another. MainWindow re-installs it when it still applies.
        self.set_subpopulations(None)
        # Same rule, same reason, for the CUSUM fit (the inspector drops its own on a re-feed
        # for exactly this: levels fitted on recording A printed under recording B's events).
        self.set_levels(None)
        if refusals is not None:
            # The owner's shared, lineage-scoped ledger: adopt it as-is. Refusals this
            # window recorded earlier on the same lineage are IN it already, and the note
            # below is rebuilt from it — which is how the note survives the re-feeds an
            # undo and a merge provoke, and how a window opened AFTER the gesture still
            # shows the standing disclosure.
            self._refusals = refusals
        else:
            # A DIFFERENT FRAME IS A DIFFERENT DETECTION, so a note counting the previous
            # one's unmeasurable rows describes events that are no longer on screen -- and
            # the ids it counted now point at somebody else's events.
            self._clear_refusals()
        self._advanced_from = None
        self._note = ("", "")
        self._fix_counter_widths()
        self._sync_progress()
        self._sync_controls()
        if refill:
            land = landing_after_refill(
                self._df["event_id"].to_numpy(dtype=np.int64)[self._model.df_rows()]
                if self._df is not None else ())
            if land is not None:
                self.show_event(land)          # quietly, as every landing here is
                return
        if keep:
            if keep_cur is not None and self._row_for_id(keep_cur) is None:
                # THE CURSOR'S TILE WENT WITH THE ROWS (a Delete from this menu, the table's or
                # the inspector's; the undo of the add it stood on). The Show filter stays, and
                # the next tile after it in the grid's order -- the one that took its place --
                # is shown, else the one before it, and none when the subset is empty. Through
                # show_event, quietly, as an add's landing is: it turns the page only when rows
                # removed before that tile moved it onto an earlier one. It used to be the next
                # event in TABLE order, which Hand-made only does not hold.
                now = (self._df["event_id"].to_numpy(dtype=np.int64)[self._model.df_rows()]
                       if self._df is not None else ())
                land = landing_after_removal(drawn, keep_cur, now)
                if land is not None:
                    self.show_event(land)
                    return
                keep_cur = None
            self._restore_selection(keep_sel, keep_cur)
            self._sync_controls()
            return
        if self._model.rowCount() and not self.view.currentIndex().isValid():
            # An edit the app made itself is not a gesture: announcing tile 0 would move the
            # trace, the table and the inspector off the place the curator left them.
            prev = self._updating
            self._updating = prev or same_lineage or quiet
            try:
                self.view.select_only(self._model.index(0, 0))
            finally:
                self._updating = prev
            if same_lineage or quiet:
                self._sync_controls()

    def clear(self) -> None:
        self._doc_id = self._channel = self._df = self._module_id = None
        # The detection went away; there is no run left to report the emptiness OF.
        self._ran_empty = False
        self._envelopes.set_target(None, None)
        self._shared_span_cache = None
        self._offpage_ids = ()
        self._missed_id = None
        self._parked = None
        self._hotkey_note = ""
        self._model.set_frame(None)
        self.set_subpopulations(None)
        self.set_levels(None)
        self._clear_refusals()
        self._advanced_from = None
        self._note = ("", "")
        self._fix_counter_widths()
        self._sync_progress()
        self._sync_controls()

    def _fix_counter_widths(self) -> None:
        """Size the page box, its "of N" and the footer's "…all N" buttons once per frame,
        for the largest numbers this frame can put in them: a subset shrinking as decisions
        land (Unreviewed) must not slide Undo or the pager under the pointer."""
        n = len(self._df) if self._df is not None else 0

        def fix(widget, texts) -> None:
            keep = widget.text()
            if self._df is None:
                # No frame: _sync_controls leaves the old count on the (disabled) button.
                texts = (*texts, keep)
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16777215)
            width = 0
            for text in texts:
                widget.setText(text)
                width = max(width, widget.sizeHint().width())
            widget.setText(keep)
            widget.setFixedWidth(width)

        big = _widest_count(self.accept_all_btn.fontMetrics(), n)
        fix(self.accept_all_btn, (f"Accept all {big}", f"Accept {big} of {big}"))
        fix(self.reject_all_btn, (f"Reject all {big}",))
        fix(self.page_label, (f"of {_widest_count(self.page_label.fontMetrics(), max(n, 1))}",))
        spin = self.page_spin
        prev, self._updating = self._updating, True
        try:
            lo, hi, value = spin.minimum(), spin.maximum(), spin.value()
            spin.setMinimumWidth(0)
            spin.setMaximumWidth(16777215)
            spin.setRange(1, int(_widest_count(spin.fontMetrics(), max(n, 1))))
            width = spin.sizeHint().width()
            spin.setRange(lo, hi)
            spin.setValue(value)
        finally:
            self._updating = prev
        spin.setFixedWidth(width)

    def refresh(self) -> None:
        """Re-read the shared frame after somebody else edited it (bounds, curation, merge).

        THE SELECTION SURVIVES. ``refilter()`` is a model reset and a reset clears the
        selection — which is why ``_after_label_write`` had to avoid it, and why this
        method could not be used as the "somebody else edited the frame" door it is named
        for: a curator holding five tiles selected lost them the moment anything else in
        the app announced an edit. The rows are re-found by ``event_id`` rather than by
        position because a refilter is exactly the moment positions move (the
        Accepted/Rejected subsets are recomputed from ``accepted``); an event that has
        left the current filter subset simply does not come back, which is correct.

        The snapshot/restore itself lives in :meth:`_refilter_keeping_selection`, which is
        THE model reset for a frame edit -- this window's own write and the inbound mirror
        (:meth:`reflect_acceptance`) go through the same one, for the reason recorded there.

        THE UNIT IS RE-RESOLVED HERE TOO. A calibration edit reaches this window through
        this door -- MainWindow re-feeds the SAME frame as a ``refresh()``, where the
        inspector's re-feed is its ``set_events``, which re-resolves -- and the depth
        caption printed the unit it was first fed with: after gain 0.001 -> nA on a copy
        of the owner's 15 s -1000 mV recording the tile read "91 · 139pA ⚠" while the
        inspector read "depth: 138.85 nA ⚠" for the same event, one stale number under
        two units on the two curation windows at once. One resolver
        (:meth:`_resolve_units`), called on both doors, so the two windows say one thing.
        """
        self._envelopes.clear()
        self._shared_span_cache = None
        self._units = self._resolve_units()
        self._advanced_from = None
        self._note = ("", "")            # another surface wrote: its counts may be stale
        self._sync_progress()
        self._refilter_keeping_selection()
        self._sync_controls()

    def _current_event_id(self) -> int | None:
        row = self._model.df_row(self.view.currentIndex())
        if row is None or self._df is None or not (0 <= row < len(self._df)):
            return None
        return int(self._df["event_id"].iloc[row])

    def _park_cursor(self, order: list[int], gone_id: int) -> None:
        """The cursor's tile just left the subset (a decision under Unreviewed, an R under
        Accepted only). Put the cursor -- not the selection -- on the tile that took its
        place in ``order`` (the subset before the write), else on the last one before it,
        on this page. A first ← / → then lands next to the decided tile, as the inspector's
        step does, instead of on the page's first tile; nothing is selected, so no key
        curates a tile nobody chose."""
        row = self._row_for_id(gone_id)
        slot = {r: i for i, r in enumerate(self._model.page_df_rows())}
        sm = self.view.selectionModel()
        if row is None or not slot or sm is None or row not in order:
            return
        pos = order.index(row)
        after = next((r for r in order[pos + 1:] if r in slot), None)
        before = next((r for r in reversed(order[:pos]) if r in slot), None)
        target = after if after is not None else before
        if target is None:
            return
        prev, self._updating = self._updating, True
        try:
            sm.setCurrentIndex(self._model.index(slot[target], 0),
                               QtCore.QItemSelectionModel.NoUpdate)
        finally:
            self._updating = prev
        eid = self._df["event_id"]
        self._parked = (None if before is None else int(eid.iloc[before]),
                        None if after is None else int(eid.iloc[after]),
                        int(eid.iloc[target]))

    def _step_from_parked(self, forward: bool) -> bool:
        """← / → straight after :meth:`_park_cursor`: select the tile on that side of the
        one that left. ``False`` when there is none (the arrow then does what it does)."""
        parked, self._parked = self._parked, None
        target = parked[1 if forward else 0] if parked else None
        row = self._row_for_id(target) if target is not None else None
        idx = self._model.index_for_df_row(row) if row is not None else QtCore.QModelIndex()
        if not idx.isValid():
            return False
        self._select_and_announce(idx)
        self.view.scrollTo(idx)
        return True

    def _select_and_announce(self, idx: QtCore.QModelIndex) -> None:
        """Make ``idx`` the current and only selected tile, announced as a click is -- also
        when the cursor was already parked on it (no ``currentChanged`` fires then)."""
        self._parked = None
        was_current = self.view.currentIndex() == idx
        self.view.select_only(idx)
        if was_current:
            self._on_current_changed(idx, idx)

    def _restore_selection(self, event_ids: list[int], current: int | None) -> None:
        """Re-select ``event_ids`` (those still on the current page) and put the cursor
        back, after a model reset -- ON THIS PAGE, and WITHOUT selecting the cursor's tile.

        THE CURSOR COMES BACK WITH ``NoUpdate``, NEVER THROUGH ``view.setCurrentIndex``. On
        an ExtendedSelection view the latter is ClearAndSelect, and the reset used to reach
        for it whenever a cursor existed: click tile 3, Ctrl-click it off (holds ``[]``,
        cursor on 3), tick ONE unrelated checkbox in the events table -> tile 3 was SELECTED
        again, and "Reject selected" rejected the tile the user had just deselected
        (measured on the owner's -1000 mV recording: rejected ``[3]``). The cursor goes back
        to its own tile when that is still on this page -- selected or not, exactly as the
        user left it -- and otherwise to the first tile put back, so the keys have somewhere
        to act from; a tile it cannot find on this page is not followed to another one
        (see :meth:`_refilter_keeping_selection`).
        """
        if self._df is None:
            return
        sm = self.view.selectionModel()
        if sm is None:
            return
        self._updating = True
        try:
            first = None
            for ev in event_ids:
                row = self._row_for_id(ev)
                if row is None:
                    continue
                idx = self._model.index_for_df_row(row)
                if not idx.isValid():
                    continue
                sm.select(idx, QtCore.QItemSelectionModel.Select)
                if first is None:
                    first = idx
            target = QtCore.QModelIndex()
            if current is not None:
                row = self._row_for_id(current)
                if row is not None:
                    target = self._model.index_for_df_row(row)
            if not target.isValid() and first is not None:
                target = first
            if target.isValid():
                # NoUpdate: move the keyboard cursor WITHOUT touching the selection.
                # `Current | Select` reads as an extend-to-here range operation and
                # measurably dropped one of five restored tiles.
                sm.setCurrentIndex(target, QtCore.QItemSelectionModel.NoUpdate)
        finally:
            self._updating = False

    def _refilter_keeping_selection(self) -> tuple[list[int], list[int]]:
        """THE model reset for "the frame changed under me" -- and the selection survives it.

        Returns ``(selected_before, selected_after)`` by event_id, so a caller that wants
        to name the tiles the subset would not give back (:meth:`_set_rows_accepted`) can.

        WHY THERE IS EXACTLY ONE. ``refilter()`` is a model reset and a reset clears the
        selection. Three callers used to reset the model on a frame edit: :meth:`refresh`
        ("THE SELECTION SURVIVES" -- snapshot, refilter, restore), :meth:`_set_rows_accepted`
        (the same, for this window's own write) and :meth:`reflect_acceptance`, which called
        the bare ``refilter()``. MainWindow connects BOTH ``refresh`` and
        ``reflect_acceptance`` to ``AnalysisPanel.eventAcceptanceChanged``, the second one
        later (when the gallery is lazily built), so on every curation made on ANOTHER
        surface -- the table's checkbox, the inspector's A/R -- ``refresh`` restored the
        sweep and ``reflect_acceptance`` wiped it a moment later. Measured on the owner's
        110-event -1000 mV recording under Show = All: click tile 3, shift-click tile 8, tick
        ONE unrelated event's checkbox in the events table -> the gallery held ``[]`` with
        "Reject selected" still enabled, and the click rejected nothing and said nothing.
        A rule written into two callers was missing from the third; now there is one.

        The subpopulation branch is the same reset by another name: the frame was edited in
        place, so a row may have been ADDED (a hand-added missed event) and the per-row class
        array is one shorter than the frame. Re-derived from the same event_id map rather
        than re-installed -- the map is still about these events, and a hand-added event
        simply is not in it, which is exactly what ``_SUBPOP_NONE`` means.

        Tiles the new subset cannot give back are added to :attr:`_offpage_ids` so the
        status line admits the shrink (see :meth:`_offpage_note`); a reset that keeps every
        tile adds nothing.

        THE PAGE STAYS. The reset used to follow the cursor's tile to whatever page it now
        sat on (``_focus_event``, the page turn :meth:`show_event` makes on purpose) before
        putting the selection back -- and :meth:`_restore_selection` can only put back what
        is on the CURRENT page. Under "Accepted only", cursor on the FIRST tile of page 2
        and five tiles swept, the events table rejecting an event EARLIER in the subset
        shifted every page-2 tile back one slot, the cursor's tile landed at the end of
        page 1, the page turned to 1 under the pointer, and the other four -- still
        accepted, still on the page the user had been looking at -- came back as
        "4 selected tiles left this page" (measured on the owner's -1000 mV recording:
        selection ``[24]`` of ``[24..28]``). An edit made on another surface turns no page
        here, and neither does this window's own write: the selection is restored on the
        page the user is on, and a tile that shifted off it is named, with where it went,
        by :meth:`_offpage_note`. (The page can still move when the subset shrank below
        the page's start -- ``_TileModel._clamp_page`` -- which is a page that no longer
        exists, not a page turned.)
        """
        selected = [int(self._df["event_id"].iloc[r]) for r in self._selected_rows()] \
            if self._df is not None else []
        current = self._current_event_id()
        order = self._model.df_rows() if current is not None else []
        parked, self._parked = self._parked, None
        if self._subpops is not None:
            self._model.set_classes(self._row_classes())
            self._rebuild_subpop_box()
        else:
            self._model.refilter()
        self._restore_selection(selected, current)
        if current is not None and not self.view.currentIndex().isValid():
            self._park_cursor(order, current)
        elif parked is not None and not selected and current == parked[2]:
            self._parked = parked            # an echo of the same write: still waiting
        back = [int(self._df["event_id"].iloc[r]) for r in self._selected_rows()] \
            if self._df is not None else []
        gone = [ev for ev in selected if ev not in set(back)]
        if gone:
            self._offpage_ids = tuple(dict.fromkeys((*self._offpage_ids, *gone)))
        return selected, back

    def _focus_event(self, event_id: int) -> bool:
        """Put the cursor on ``event_id``'s tile, turning to its page first. ``True`` when the
        tile is in the current subset and is now current; ``False``, with nothing moved, when
        the subset excludes it. Mechanics only -- :meth:`show_event` is the door that also
        SAYS so when this returns ``False``.

        :meth:`show_event`'s alone. The page turn is what the inbound mirror is FOR (the
        table moved to an event; the gallery follows), and it is what a frame edit must never
        do -- so the selection-keeping reset does not call this; see
        :meth:`_refilter_keeping_selection`. The move is ClearAndSelect on purpose -- the
        mirror selects the tile the other surface chose -- and stated, never read from the
        modifiers: the inspector's ⌘Z / ⌘U arrive here with ⌘ still down."""
        idx = self._index_on_its_page(event_id)
        if not idx.isValid():
            return False
        self._updating = True
        try:
            self.view.select_only(idx)
        finally:
            self._updating = False
        return True

    def _index_on_its_page(self, event_id: int) -> QtCore.QModelIndex:
        """``event_id``'s tile index, turning (quietly) to its page first; invalid when the
        current subset excludes it. The page turn :meth:`_focus_event` and the advance share."""
        row = self._row_for_id(event_id)
        page = self._model.page_of_df_row(row) if row is not None else None
        if page is None:
            return QtCore.QModelIndex()
        if page != self._model.page():
            self._updating = True
            try:
                self._model.set_page(page)
            finally:
                self._updating = False
        return self._model.index_for_df_row(row)

    def _resolve_units(self) -> str:
        if self._doc_id is None or self._channel is None:
            return ""
        try:
            return self._session.calibrated(self._doc_id, self._channel).units or ""
        except Exception:  # noqa: BLE001 -- units are cosmetic; never break the gallery
            return ""

    # -- item 13: subpopulations, representative sampling, and which trace ----------------------
    def set_levels(self, levels_df: pd.DataFrame | None) -> None:
        """Install (or drop) the CUSUM level table this detection was fitted to.

        The same name and the same contract as
        :meth:`~nanoscope.gui.event_inspector.EventInspectorWindow.set_levels`, so
        ``MainWindow`` hands the fit to both windows the same way and neither can be showing
        levels the other has dropped. A REFERENCE, like the events frame: ``MainWindow``
        reconciles the table when a span is edited (``_reconcile_level_fit``) and re-installs
        the reconciled frame on every surface that holds one.

        Levels are drawn only on the tile sizes that have room for them; see
        :meth:`level_bars_for`.
        """
        self._levels = (levels_df if levels_df is not None and not levels_df.empty else None)
        if self.isVisible():
            self.view.viewport().update()

    def level_bars_for(self, df_row: int, wave_w: int, wave_h: int
                       ) -> list[tuple[float, float, float]]:
        """``(f0, f1, y_n)`` per fitted sub-state of ``df_row`` — x as fractions of the
        tile's padded window, y normalised into the tile with 0 at the TOP.

        Empty when no level fit is installed, when this event has none, when the fit
        resolved a single level (a flat event annotated with one flat line says nothing the
        waveform did not), or when the tile is too small to draw them legibly — see
        :data:`_LEVEL_MIN_TILE_H`. On the same axis as the waveform by construction: the
        window bounds come from the envelope source and the y scale from
        :meth:`_tile_scale`, which are the two the tile itself was drawn with.
        """
        if (self._levels is None or self._df is None
                or not (0 <= df_row < len(self._df))
                or wave_h < _LEVEL_MIN_TILE_H):
            return []
        r = self._df.iloc[df_row]
        rows = self._levels[self._levels["event_id"] == int(r["event_id"])]
        if len(rows) < 2:
            return []
        built = self._envelopes.envelope(int(r["start_idx"]), int(r["end_idx"]), wave_w)
        if built is None:
            return []
        lo, hi, _ev = built
        top, full = self._tile_scale(r, lo, hi)
        lo_i, hi_i = self._envelopes.window_bounds(int(r["start_idx"]), int(r["end_idx"]))
        win = max(hi_i - lo_i, 1)
        starts = rows["start_idx"].to_numpy(dtype=np.int64)
        stops = rows["end_idx"].to_numpy(dtype=np.int64)
        values = rows["level_pA"].to_numpy(dtype=np.float64)
        f0 = np.clip((starts - lo_i) / win, 0.0, 1.0)
        f1 = np.clip((stops - lo_i) / win, 0.0, 1.0)
        y = (top - values) / full
        keep = np.isfinite(y) & (y >= 0.0) & (y <= 1.0)
        return list(zip(f0[keep].tolist(), f1[keep].tolist(), y[keep].tolist()))

    def set_subpopulations(self, assignment: dict[int, int] | None, *,
                           source: str = "", caveats: tuple[str, ...] = ()) -> None:
        """Install (or drop) an ``event_id -> subpopulation id`` map from a finished report.

        The gallery is handed the MAP and the report's own WARNINGS; it never sees the report,
        never imports the analysis layer and computes nothing -- ``MainWindow`` reads both off
        the :class:`~nanoscope.analysis.subpop_report.SubpopReport` and hands them over, which
        is the same shape every other cross-surface hand-off in this app has.

        Args:
            assignment: event_id -> class id, or ``None`` to drop the colouring entirely.
            source: What produced it, for the status line (e.g. the recording name).
            caveats: The run's warnings, VERBATIM. They are rendered as a pane that cannot be
                collapsed, for the reason argued in ``subpop_window``: this window can now
                draw four tidy classes, and a picture of four tidy classes without the
                reasons they may not be four things is the figure this pipeline should not
                produce.
        """
        self._subpops = None if assignment is None else {
            int(k): int(v) for k, v in dict(assignment).items()}
        self._subpop_source = str(source)
        self._subpop_caveats = tuple(str(c) for c in caveats)
        self._model.set_classes(self._row_classes())
        self._rebuild_subpop_box()
        self._subpop_bar.setVisible(self._subpops is not None)
        self._caveat_box.setVisible(self._subpops is not None)
        self.caveat_text.setPlainText(self._caveat_text())
        self._caveat_box.setTitle(
            f"Caveats — {len(self._subpop_caveats)} from the run, and one about these tiles")
        if self._subpops is None:
            self._updating = True
            try:
                self.representative_box.setChecked(False)
            finally:
                self._updating = False
            self._model.set_representative(False)
        self._sync_controls()
        self.view.viewport().update()

    def subpopulations(self) -> dict[int, int] | None:
        """The installed map, or ``None``."""
        return None if self._subpops is None else dict(self._subpops)

    def subpop_caveats(self) -> str:
        """Exactly what the caveat pane shows."""
        return self.caveat_text.toPlainText()

    def _caveat_text(self) -> str:
        if self._subpops is None:
            return ""
        lines = [f"1. {_TILE_CAVEAT}", ""]
        for i, w in enumerate(self._subpop_caveats, start=2):
            lines.append(f"{i}. {w}")
            lines.append("")
        if not self._subpop_caveats:
            lines.append("The run itself raised none.")
        return "\n".join(lines).rstrip() + "\n"

    def _row_classes(self) -> np.ndarray | None:
        """The subpopulation id per df ROW, or ``None`` when nothing is installed.

        Every event the map does not name becomes :data:`_SUBPOP_NONE`, which is
        ``categorical``'s "in no class" grey. On a real run that is most of the detection --
        the >=4 sigma gate keeps roughly a tenth of it -- and it is a class of its own in the
        combo, not an omission.
        """
        if self._subpops is None or self._df is None or not len(self._df):
            return None
        ids = self._df["event_id"].to_numpy(dtype=np.int64)
        return np.array([self._subpops.get(int(e), _SUBPOP_NONE) for e in ids],
                        dtype=np.int64)

    def _rebuild_subpop_box(self) -> None:
        """Re-fill the combo with the classes actually present, with their counts.

        Ascending by id with the unassigned entry LAST -- the same order
        :func:`~nanoscope.gui.categorical.category_counts` puts a legend in, so this combo, the
        report window's legend and the trace's overlays enumerate the classes identically.
        """
        self._updating = True
        try:
            previous = self.subpop_box.currentData()
            self.subpop_box.clear()
            classes = self._model.classes()
            if classes is None:
                return
            self.subpop_box.addItem(f"{_SUBPOP_ALL} · {classes.size}", None)
            present = sorted({int(c) for c in classes.tolist()}, key=lambda c: (c < 0, c))
            for cid in present:
                n = int(np.count_nonzero(classes == cid))
                text = (f"subpop {cid} · {n}" if cid >= 0
                        else f"not in the report · {n}")
                self.subpop_box.addItem(text, cid)
            if previous is not None:
                idx = self.subpop_box.findData(previous)
                if idx >= 0:
                    self.subpop_box.setCurrentIndex(idx)
        finally:
            self._updating = False

    def set_subpop_filter(self, class_id: int | None) -> None:
        """Show one class (``None`` = all).

        Raises:
            ValueError: naming a class this gallery does not carry. A silent no-op would leave
                the caller reading "subpop 3" over a grid of every event there is.
        """
        if class_id is not None and self.subpop_box.findData(int(class_id)) < 0:
            raise ValueError(
                f"no subpopulation {class_id!r} in this gallery; have "
                f"{[self.subpop_box.itemData(i) for i in range(self.subpop_box.count())]}")
        idx = self.subpop_box.findData(None if class_id is None else int(class_id))
        if idx >= 0:
            self.subpop_box.setCurrentIndex(idx)

    def subpop_filter(self) -> int | None:
        return self._model.subpop()

    def is_hand_made(self, df_row: int) -> bool:
        """Did a person draw this event? (``source`` reads manual, ``added`` where unrecorded.)"""
        df = self._df
        if df is None or not (0 <= df_row < len(df)):
            return False
        if "source" in df.columns:
            value = df["source"].iloc[df_row]
            if not pd.isna(value):
                return str(value).strip() == "manual"
        return bool(df["added"].iloc[df_row]) if "added" in df.columns else False

    def subpop_of(self, df_row: int) -> tuple[int, str] | None:
        """``(class_id, chip_text)`` for a tile, or ``None`` when no report is installed."""
        classes = self._model.classes()
        if classes is None or not (0 <= df_row < classes.size):
            return None
        cid = int(classes[df_row])
        return (cid, f"s{cid}" if cid >= 0 else "s–")

    def set_representative(self, on: bool) -> None:
        self.representative_box.setChecked(bool(on))

    def is_representative(self) -> bool:
        return self._model.representative()

    def set_detrended_signal(self, data: np.ndarray | None) -> None:
        """Offer (or withdraw) the detrended trace the tiles can be drawn from.

        The array is the one the subpopulation run already built from the detection's own
        provenance; it is NOT rebuilt here, because rebuilding it in the GUI would be a second
        implementation of ``modules.detection_signal`` and the two would drift. Withdrawing it
        while it is on screen falls back to the calibrated trace, and says so.
        """
        self._envelopes.set_override(None)
        offered = data is not None
        self._pending_detrended = data
        self._updating = True
        try:
            want = self.trace_box.currentText()
            self.trace_box.clear()
            self.trace_box.addItem(TRACE_CALIBRATED)
            if offered:
                self.trace_box.addItem(TRACE_DETRENDED)
            if want == TRACE_DETRENDED and offered:
                self.trace_box.setCurrentText(TRACE_DETRENDED)
        finally:
            self._updating = False
        self._apply_trace(self.trace_box.currentText())

    def set_trace(self, name: str) -> None:
        """Draw the tiles from ``name``.

        Raises:
            ValueError: naming a trace this gallery is not holding.
        """
        options = [self.trace_box.itemText(i) for i in range(self.trace_box.count())]
        if name not in options:
            raise ValueError(f"no trace {name!r} in this gallery; have {options}")
        self.trace_box.setCurrentText(name)

    def trace_name(self) -> str:
        """Which trace the tiles on screen are drawn from, by name."""
        return self._trace_name

    def _apply_trace(self, name: str) -> None:
        self._trace_name = name if name else TRACE_CALIBRATED
        data = getattr(self, "_pending_detrended", None)
        self._envelopes.set_override(data if self._trace_name == TRACE_DETRENDED else None)
        self._shared_span_cache = None
        self._sync_controls()
        self.view.viewport().update()

    def _on_trace_changed(self, text: str) -> None:
        if self._updating:
            return
        self._apply_trace(text)

    def _on_subpop_changed(self, *_a) -> None:
        if self._updating:
            return
        keep_id = self.current_event_id()
        self._model.set_subpop(self.subpop_box.currentData())
        if keep_id is not None:
            self.show_event(keep_id)
        self._sync_controls()
        self.view.viewport().update()

    def _on_representative_toggled(self, on: bool) -> None:
        if self._updating:
            return
        self._model.set_representative(bool(on))
        self._sync_controls()
        self.view.viewport().update()

    # -- things the delegate asks for -------------------------------------------------------------
    def frame(self) -> pd.DataFrame | None:
        return self._df

    def tile_size(self) -> tuple[int, int]:
        return _TILE_SIZES.get(self._tile_name, _TILE_SIZES[_DEFAULT_TILE])

    def neighbour_spans(self, df_row: int) -> list[tuple[float, float]]:
        """ACCEPTED events other than ``df_row`` that intersect its tile's padded
        window, as ``(f0, f1)`` fractions of that window — the same axis
        ``envelope_for``'s ``(ev0, ev1)`` is on.

        The tile's padded window shows the neighbouring waveform but used to shade
        only its own event, so a fragmented detection read differently in the gallery
        than on the trace: the sibling dip sat bare in the tile while the main panel
        shaded it (owner note #4, noticed on the fractured −500 mV events). Accepted
        only, because that is exactly the set the trace shades.

        A vector scan over the shared frame per call, not a prebuilt index: two
        zero-copy column views and four array ops on ~1e4 rows per painted tile,
        and the frame is shared truth that curation edits in place — an index would
        need the invalidation story the envelope cache has, for less than it costs.
        """
        if self._df is None or not (0 <= df_row < len(self._df)):
            return []
        r = self._df.iloc[df_row]
        lo_i, hi_i = self._envelopes.window_bounds(int(r["start_idx"]), int(r["end_idx"]))
        win = max(hi_i - lo_i, 1)
        starts = self._df["start_idx"].to_numpy(dtype=np.int64)
        ends = self._df["end_idx"].to_numpy(dtype=np.int64)
        hit = (self._df["accepted"].to_numpy(dtype=bool)
               & (ends > lo_i) & (starts < hi_i))
        hit[df_row] = False
        if not hit.any():
            return []
        f0 = np.clip((starts[hit] - lo_i) / win, 0.0, 1.0)
        f1 = np.clip((ends[hit] - lo_i) / win, 0.0, 1.0)
        return list(zip(f0.tolist(), f1.tolist()))

    def envelope_for(self, df_row: int, width: int):
        """``(lo, hi, (ev0, ev1))`` normalised into the tile, 0 at the TOP.

        THE SCALE IS THE POINT OF THIS METHOD. Normalising each tile to its own extremes —
        the obvious thing, and what a thumbnail usually wants — made a 15 pA event and a
        45 pA event draw as the same picture, so sorting the grid by depth produced a wall
        of identical tiles and the surface could not answer the question it was sorted to
        answer. Under the default SHARED scale every tile is drawn against one depth span
        for the whole detection, referenced to its OWN baseline so a drifting open pore does
        not shift the trace off the tile: deep events look deep, marginal ones look
        marginal, and the grid can be read down a column.

        Per-tile scaling stays available (it is the better choice for judging SHAPE when
        depths are similar), which is why the span is applied here and not baked into the
        cached envelope.
        """
        if self._df is None or not (0 <= df_row < len(self._df)):
            return None
        r = self._df.iloc[df_row]
        built = self._envelopes.envelope(int(r["start_idx"]), int(r["end_idx"]), width)
        if built is None:
            return None
        lo, hi, ev = built
        top, full = self._tile_scale(r, lo, hi)
        # 0 at the top of the tile: Qt's y grows downward, and doing the flip here keeps the
        # delegate free of sign conventions. Clipped, because a shared span is deliberately
        # NOT the extremes of every event -- an outlier draws at the rim rather than
        # flattening the other nine hundred tiles.
        lo_n = np.clip((top - lo) / full, 0.0, 1.0).astype(np.float32)
        hi_n = np.clip((top - hi) / full, 0.0, 1.0).astype(np.float32)
        return lo_n, hi_n, ev

    def _tile_scale(self, r: pd.Series, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
        """``(top, full)`` in DATA UNITS for one tile: the current at the tile's top edge,
        and the current span the tile's height covers.

        Split out of :meth:`envelope_for` so anything else drawn ON the waveform lands on
        the same axis as the waveform. :meth:`level_bars_for` is the caller that made it
        necessary -- a CUSUM level normalised against a second, separately-derived scale is
        a line that sits a few pixels off the trace it annotates, which at tile size is the
        difference between "this is the fitted level" and "this is some other line".
        """
        if self._shared_scale:
            top = float(r["baseline_pA"])
            if not np.isfinite(top):
                top = float(np.max(hi))
            span = self._shared_span()
            # Half a span of headroom above the baseline, so open-pore noise (and any
            # upward excursion) has somewhere to go instead of being clipped to the rim.
            top = top + 0.25 * span
            full = 1.25 * span
        else:
            top = float(np.max(hi))
            full = top - float(np.min(lo))

        if not np.isfinite(full) or full <= 0:
            full = 1.0
        return top, full

    def _shared_span(self) -> float:
        """The depth all tiles are drawn against: a high percentile of the detection's own
        depths, not the maximum.

        The maximum is one event, and letting the single deepest blockade in a recording set
        the scale for the other nine hundred is how a shared axis becomes useless. The 98th
        percentile keeps the bulk of the distribution using most of the tile height and lets
        the rare monster clip.
        """
        if self._shared_span_cache is not None:
            return self._shared_span_cache
        span = 1.0
        if self._df is not None and len(self._df):
            d = np.abs(self._df["depth_pA"].to_numpy(dtype=np.float64))
            d = d[np.isfinite(d)]
            if d.size:
                span = float(np.percentile(d, 98.0))
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        self._shared_span_cache = span
        return span

    def caption_for(self, r: pd.Series) -> str:
        """The tile's caption: the id, then the quantity the current sort is about.

        THE DEPTH CARRIES ITS UNIT, like the dwell beside it ("33.88ms") -- it was the one
        caption of the three without one ("86 · 139" under a sort by depth), on a window
        that had already resolved the channel's units (:meth:`_resolve_units`), and on a
        channel calibrated to nA a bare number is read as pA. The unit is the resolved one
        and never the schema column's suffix: ``depth_pA`` is the frozen column NAME, and
        the events-table header prints it over nA values too.

        Captions are numerical artwork. Staleness and above-one fractional-depth
        warnings remain in the footer, outside the waveform tiles.
        """
        eid = int(r["event_id"])
        sort = self.sort_box.currentText() if hasattr(self, "sort_box") else _SORT_ORDER
        if sort == _SORT_DEPTH:
            return f"{eid} · {r['depth_pA']:.3g}{self._units}"
        if sort == _SORT_FRAC:
            return f"{eid} · {r['frac_depth']:.3g}"
        return f"{eid} · {_fmt_dwell(float(r['dwell_s']))}"

    # -- selection --------------------------------------------------------------------------------
    def show_event(self, event_id: int) -> None:
        """Focus the tile for ``event_id``, TURNING TO ITS PAGE first (inbound mirror; emits
        nothing back).

        Without the page turn this silently did nothing whenever the event was on another
        page — the table and the inspector would move to an event and the gallery would sit
        on an unrelated page looking authoritative.

        AND WHEN THE SUBSET EXCLUDES IT, NOTHING STAYS CURRENT. The page turn fixed the miss
        for pages only: under "Accepted only" with a rejected event, under a subpopulation
        class, or under the representative sample (which the report window's "Event
        gallery…" button turns on), ``page_of_df_row`` is ``None`` and this returned with
        the PREVIOUS tile still current, still drawn with the selection ring, the status
        line unchanged. Measured on the owner's -1000 mV recording: the user clicks event 0
        in the events table, the trace, the table and the inspector move to it, the gallery
        stays on event 12 -- and R rejects 12. So the miss now clears the cursor and the
        selection (A / R / Space and the "…selected" buttons act on nothing) and the status
        line says which event is filtered out of what, in the inspector's words
        ("filtered out of Accepted only (100 shown)"), with the control that reaches it.
        The subset is NOT widened for the caller: a filter the user set is a decision, and
        a gallery that silently switched itself to "All" to follow a table click would be
        the other kind of surprise.
        """
        if self._row_for_id(event_id) is None:
            return
        if self._current_event_id() == int(event_id):
            # ALREADY THERE: A MIRROR THAT MOVES NOTHING MUST TOUCH NOTHING. This is the
            # echo of this window's OWN ``eventSelected`` -- MainWindow relays it to
            # AnalysisPanel.select_event, whose table selection comes straight back here --
            # and, because Qt6 emits ``currentChanged`` BEFORE it applies the mouse press's
            # own selection command, the echo lands INSIDE the click, between the cursor
            # move and the selection change. ``_focus_event``'s ``setCurrentIndex`` is
            # ClearAndSelect on an ExtendedSelection view, so it pre-selected the very tile
            # Qt was about to Toggle, and the toggle then took it straight back off:
            # ⌘/Ctrl-click -- the gesture the "Accept selected" tooltip advertises --
            # NETTED ZERO TILES in the wired app. Measured on the owner's -1000 mV
            # recording, 110 events, gallery at 1200x860: click tile 0 -> [0];
            # ⌘-click tile 6 -> [] (the trace read "_on_current_changed(row=6) sel=[0]" ->
            # "_focus_event exit sel=[6]" -> "_on_selection_changed sel=[]"); ⌘-click tile
            # 11 -> [] again. Cutting the relay made the same three clicks read
            # [0] / [0,6] / [0,6,11], which is what said the round trip was the writer.
            #
            # THE GUARD IS THE QUESTION THE MIRROR IS FOR, not a flag about who is calling:
            # "show this event" over an event already shown is a no-op by definition,
            # whoever asks and whenever it arrives -- so it also holds if the relay is ever
            # queued rather than direct, which a ``_relaying`` latch set around the emit
            # would not. Nothing about the page, the cursor or the selection changes; the
            # missed-event note does, because a tile IS focused (see _missed_note).
            if self._missed_id is not None:
                self._missed_id = None
            self._sync_controls()
            return
        if self._focus_event(int(event_id)):
            self._missed_id = None
            self._sync_controls()
            return
        self._missed_id = int(event_id)
        sm = self.view.selectionModel()
        if sm is not None:
            self._updating = True
            try:
                sm.clearSelection()
                sm.clearCurrentIndex()
            finally:
                self._updating = False
        self._sync_controls()
        self.view.viewport().update()

    def land_on_added_event(self, event_id: int) -> None:
        """A hand-made event, ``event_id``, joined the table this window shows: focus its tile,
        quietly, when the grid's list holds it -- the Show filter, a subpopulation class, the
        representative sample (``curation_view.landing_after_add``). When the list hides it,
        the cursor and the selection stay where they were. The add used to call
        :meth:`show_event` whatever the list held, and its miss clears both ("Event 7 is
        filtered out of Detected only"), so after an add A, R and Space acted on nothing."""
        if self._df is None:
            return
        drawn = self._df["event_id"].to_numpy(dtype=np.int64)[self._model.df_rows()]
        land = landing_after_add(drawn, int(event_id))
        if land is not None:
            self.show_event(land)

    def _missed_note(self) -> str:
        """"Event 12 is filtered out of Accepted only (100 shown) — …", or "".

        RE-DERIVED, NEVER REMEMBERED -- the same rule as :meth:`_offpage_note`. The id is
        remembered; whether it is still excluded, and whether a tile has since been focused,
        is asked of the model and the view every time the line is built, so the sentence
        goes away by itself when the user clicks a tile or widens the subset to reach it.
        """
        if self._missed_id is None or self._df is None:
            return ""
        if self.view.currentIndex().isValid():
            return ""
        row = self._row_for_id(self._missed_id)
        if row is None or self._model.page_of_df_row(row) is not None:
            return ""
        subset, reach = self._excluding_subset(row)
        return (f"Event {self._missed_id} is filtered out of {subset} "
                f"({len(self._model.df_rows())} shown) — nothing is selected here; {reach}")

    def _excluding_subset(self, row: int) -> tuple[str, str]:
        """Which control keeps df ``row`` out of the subset, named the way the status line
        names it, and the gesture that reaches it. Asked in the order the model applies
        them (:meth:`_TileModel._filtered_rows`, then the sample)."""
        mode = self.filter_box.currentText()
        if is_filter(mode) and not bool(filter_mask(self._df.iloc[[row]], mode)[0]):
            return mode, "set Show to All to reach it"
        classes = self._model.classes()
        cid = self._model.subpop()
        if classes is not None and cid is not None and int(classes[row]) != int(cid):
            name = f"subpop {cid} only" if cid >= 0 else "not in the report only"
            return name, "set Subpopulation to every subpopulation to reach it"
        if self._model.representative():
            return "the representative sample", "untick Representative sample to reach it"
        return "the current subset", "change Show to reach it"

    def current_event_id(self) -> int | None:
        row = self._model.df_row(self.view.currentIndex())
        if row is None or self._df is None:
            return None
        return int(self._df.iloc[row]["event_id"])

    def selected_event_ids(self) -> list[int]:
        if self._df is None:
            return []
        rows = [self._model.df_row(i) for i in self.view.selectionModel().selectedIndexes()]
        return [int(self._df.iloc[r]["event_id"]) for r in rows if r is not None]

    # -- the batch selection gestures ------------------------------------------------------------
    # THE PAGE IS THE SCOPE, for all three. ``Ctrl+A`` on the view has always meant "every row
    # the model has", which is this page and not the 110-event subset behind it (measured on
    # the owner's -1000 mV recording: 45 of 45 tiles, page size Fit), and the curation buttons
    # these feed act on the selection. So "all", "none" and "inverse" are all statements about
    # what is on screen -- the whole subset already has its own two controls, "Accept all N"
    # and "Reject all N" in the footer, which say their count out loud precisely because they
    # reach past the page.
    def select_all_shown(self) -> None:
        """Select every tile on the current page (⌘A / Ctrl+A, and Select ▸ All on this page).

        ``view.selectAll()`` and not a hand-built QItemSelection: the view's own ⌘A does
        exactly this, and two implementations of one gesture is how the menu item and the key
        come to disagree.
        """
        self.view.selectAll()

    def clear_selection(self) -> None:
        """Drop the selection, leaving the cursor and every tile's acceptance alone
        (Escape, and Select ▸ None).

        THE CURSOR STAYS. ``clearSelection`` and not ``clearCurrentIndex``: the keyboard
        cursor is where the arrow keys and Shift-arrow carry on from, and a deselect-all that
        also threw it away turned "I picked the wrong block" into "and now start from the
        top". A / R / Space / the digits all read the SELECTION (:meth:`_selected_rows`), so
        an empty selection is already the guarantee that they write nothing.
        """
        sm = self.view.selectionModel()
        if sm is not None:
            sm.clearSelection()

    def select_inverse(self) -> None:
        """Select this page's UNselected tiles and deselect its selected ones (⌘I / Ctrl+I,
        and Select ▸ Inverse).

        Named "inverse" and never "invert": ``Space`` in this window inverts the ACCEPTANCE of
        the selection (:meth:`_toggle_selected`), which writes the frame, and two gestures one
        keystroke apart must not share a verb -- one of them curates.

        Built as ONE QItemSelection and applied ONCE, rather than toggling row by row: each
        ``select`` call emits ``selectionChanged``, and 200 of them (the largest page) is 200
        status-line rebuilds and 200 repaints for one gesture.
        """
        sm = self.view.selectionModel()
        n = self._model.rowCount()
        if sm is None or n == 0:
            return
        inverse = QtCore.QItemSelection()
        for row in range(n):
            idx = self._model.index(row, 0)
            if not sm.isSelected(idx):
                inverse.select(idx, idx)
        sm.select(inverse, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def tile_count(self) -> int:
        """Tiles on the CURRENT PAGE. See :meth:`event_count` for the whole subset."""
        return int(self._model.rowCount())

    def event_count(self) -> int:
        """Events in the current filter subset, across all pages — the same question, and
        the same name, as EventInspectorWindow.event_count()."""
        return len(self._model.df_rows())

    def _row_for_id(self, event_id: int | None) -> int | None:
        if self._df is None or event_id is None:
            return None
        ids = self._df["event_id"].to_numpy(dtype=np.int64)
        hit = np.nonzero(ids == int(event_id))[0]
        return int(hit[0]) if hit.size else None

    def _on_selection_changed(self, *_a) -> None:
        """The user changed the selection by hand: the tiles the LAST selection could not
        keep are about a selection that no longer exists.

        The off-page record (:attr:`_offpage_ids`) is extended by every inbound reset
        (:meth:`_refilter_keeping_selection`) and replaced by this window's own write, and
        nothing retired it when the user moved on: sweep six tiles under "Accepted only",
        have the events table reject one of them, click ONE new tile, have the table reject
        that too -- and the line counted "2 selected tiles left this page" over a selection
        that had been one tile. Programmatic selection changes (the restore itself,
        :meth:`show_event`'s mirror) run under ``_updating`` and leave the record alone; a
        click, a sweep, a Ctrl-click are the user's, and they start a new record.

        HEARD ON BOTH SIGNALS. A click reaches this window as ``currentChanged`` first and
        ``selectionChanged`` second -- and in the wired app the first one is relayed
        (``eventSelected`` -> the table -> :meth:`show_event`), whose mirror selects the
        clicked tile under ``_updating`` before the click's own selection change lands, so
        that second signal never fires (the selection is already what the click asked
        for). Measured: the record survived the click. ``currentChanged`` is therefore
        where a click, a Shift-click and an arrow key retire it
        (:meth:`_on_current_changed`); this slot is for the gestures that change the
        selection WITHOUT moving the cursor -- a Ctrl-click toggle, a rubber band.
        """
        if self._updating:
            return
        if self._parked is not None:
            cur = self.view.currentIndex()
            row = self._model.df_row(cur)
            if row is not None and self._selected_rows() == [row]:
                # A click on the parked tile: no currentChanged fires, so announce it here.
                self._on_current_changed(cur, cur)
        retired = self._retire_offpage_record()
        # AND THE COUNT ON THE LINE FOLLOWS THE SELECTION. Rebuilt only when the NUMBER
        # moved, not on every signal: a rubber band emits one of these per mouse-move, and
        # ``_sync_controls`` rebuilds the whole line and re-fits it (a bisection over text
        # layouts). Held as an int rather than read back from the line so the comparison
        # cannot drift from what was printed.
        n_sel = len(self._selected_rows())
        if retired or n_sel != self._shown_sel_count:
            self._shown_sel_count = n_sel
            self._sync_controls()

    def _retire_offpage_record(self) -> bool:
        """Drop :attr:`_offpage_ids`; ``True`` when there was one to drop."""
        if not self._offpage_ids:
            return False
        self._offpage_ids = ()
        return True

    def _on_current_changed(self, current: QtCore.QModelIndex, _prev) -> None:
        if self._updating:
            return
        self._parked = None
        if not self._advancing:
            self._advanced_from = None
        if self._note != ("", ""):
            self._note = ("", "")
            self._render_progress()
        row = self._model.df_row(current)
        if row is None or self._df is None:
            return
        # The user moved to a tile by hand: the record of what the LAST selection could
        # not keep is over (see _on_selection_changed for why it is retired here too).
        if self._retire_offpage_record():
            self._sync_controls()
        if self._missed_id is not None:
            # The user focused a tile: the line describes THAT tile now, not the event
            # another surface asked for. Retired here, on this window's own gesture, and not
            # left to the relay below to come back and do it -- the events table's
            # ``select_event`` emits nothing when the row is already its current one, so
            # the round trip does not always return (measured: a click on tile 0 straight
            # after the table had selected event 0 left the miss sentence standing).
            self._missed_id = None
            self._sync_controls()
        self.eventSelected.emit(int(self._df.iloc[row]["event_id"]))

    def _on_double_clicked(self, index: QtCore.QModelIndex) -> None:
        """Double-click OPENS the event: the Event Inspector, on this event.

        It used to toggle accept/reject, which put a destructive action on the gesture
        people use to open things. It was then made to zoom the MAIN WINDOW's trace -- and
        that is what the owner reported as "when an event is clicked on the gallery it
        should open the curate, or take to the main panel": the gallery is a separate
        top-level window sitting OVER the main one, so a gesture whose whole effect happens
        behind it reads as a gesture that did nothing. Nothing moved on screen.

        The inspector is the surface the owner calls "the curate" -- it is where one event's
        bounds, levels and flags are edited -- and it opens as its own window, in front, so
        the gesture visibly lands. The trace is still one right-click away ("Go to event in
        trace"), and a single click still selects and zooms it behind, unchanged.
        """
        row = self._model.df_row(index)
        if row is None or self._df is None:
            return
        self.inspectRequested.emit(int(self._df.iloc[row]["event_id"]))

    def inspect_current(self) -> None:
        """Open the FOCUSED tile's event in the Event Inspector (the "Inspector…" button).

        The same signal, and therefore the same MainWindow route, as the double-click and
        the context menu. Silent with nothing focused rather than opening the inspector on
        an arbitrary event: "open this one" needs a this.
        """
        event_id = self.current_event_id()
        if event_id is not None:
            self.inspectRequested.emit(int(event_id))

    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        """Per-tile context menu: open this event on the other two surfaces, or curate it.

        Emits intent (the same signals the buttons/keys use); MainWindow routes. The menu
        is built per invocation — its actions depend on which tile is under the cursor,
        and a menu that outlives the model it indexes into is a stale-row bug waiting."""
        index = self.view.indexAt(pos)
        row = self._model.df_row(index) if index.isValid() else None
        if row is None or self._df is None:
            return
        event_id = int(self._df.iloc[row]["event_id"])
        menu = QtWidgets.QMenu(self.view)
        act_inspect = menu.addAction("Open in Event Inspector")
        act_goto = menu.addAction("Go to event in trace")
        menu.addSeparator()
        act_accept = menu.addAction("Accept")
        act_reject = menu.addAction("Reject")
        act_delete = None
        if ("source" in self._df.columns
                and str(self._df["source"].iloc[row]) == "manual"):
            menu.addSeparator()
            act_delete = menu.addAction("Delete hand-made event")
        chosen = self._exec_menu(menu, self.view.viewport().mapToGlobal(pos))
        if act_delete is not None and chosen is act_delete:
            self.deleteHandMadeRequested.emit(event_id)
        elif chosen is act_inspect:
            self.inspectRequested.emit(event_id)
        elif chosen is act_goto:
            self.eventActivated.emit(event_id)
        elif chosen is act_accept:
            self._set_rows_accepted([row], True)
        elif chosen is act_reject:
            self._set_rows_accepted([row], False)

    @staticmethod
    def _exec_menu(menu: QtWidgets.QMenu, global_pos: QtCore.QPoint):
        """The one blocking line, behind a Python seam. QMenu.exec is a Shiboken
        built-in that cannot be reliably monkeypatched, and offscreen it blocks a
        test forever — so the tests patch THIS instead and the handler stays real."""
        return menu.exec(global_pos)

    # -- curation ---------------------------------------------------------------------------------
    def set_accepted_selected(self, accepted: bool) -> None:
        """Curate every SELECTED tile. One tile emits the per-event signal; several emit the
        batch signal, because the owner's per-event path repaints the table row by row."""
        rows = [self._model.df_row(i) for i in self.view.selectionModel().selectedIndexes()]
        self._set_rows_accepted([r for r in rows if r is not None], accepted)

    def accept_all(self) -> None:
        """Accept every tile the CURRENT filter shows and the detector could measure; rows
        flagged by a rejecting rule stay rejected (:func:`_derived_accepted`)."""
        self._set_rows_accepted(self._model.df_rows(), True)

    def reject_all(self) -> None:
        self._set_rows_accepted(self._model.df_rows(), False)

    def _toggle_selected(self) -> None:
        """Space: flip the selection. Mixed selections all become REJECTED -- the gallery is
        used to cull, so the destructive-looking direction is the one the user means when
        they have swept up a mixed block of bad detections."""
        rows = [self._model.df_row(i) for i in self.view.selectionModel().selectedIndexes()]
        rows = [r for r in rows if r is not None]
        if not rows or self._df is None:
            return
        any_accepted = any(bool(self._df.iloc[r]["accepted"]) for r in rows)
        self._set_rows_accepted(rows, not any_accepted)

    def _set_rows_accepted(self, rows: list[int], accepted: bool) -> None:
        """THE one place this window writes ``accepted``, for every curation gesture it has:
        the two "…selected" buttons, A / R, Space (invert) and both "…all shown" buttons.

        What lands in the cell is ``derived AND requested``, not ``requested`` -- the same AND
        :func:`~nanoscope.analysis.characterize.characterize` applies (see
        :func:`_derived_accepted`). Writing ``requested`` was an OR in everything but name: it
        could only ever turn a rejection into an acceptance, so one click on "Accept all shown"
        accepted rows whose own ``flag`` says the detector could not measure them -- with the
        tile beside it still painted, and captioned, rejected.

        The guard sits HERE and not in the four callers because a rule copied into four gestures
        is a rule that will be missing from the fifth.

        FAIL CLOSED. A frame with no ``flag`` column cannot answer the derived half, and the
        guard used to answer it ``True`` -- i.e. broadcast ``requested`` over every row, which
        is precisely the leak this method exists to close, reachable by handing the window a
        frame off the frozen schema. An accept is now REFUSED outright on such a frame (a
        reject still lands: rejection is never blocked), and :meth:`_sync_controls` says so.
        """
        if self._df is None or not rows:
            return
        col = self._df.columns.get_loc("accepted")
        flag_col = (self._df.columns.get_loc("flag")
                    if "flag" in self._df.columns else None)
        # AN ACCEPT RE-DECIDES ITS OWN SCOPE; A REJECT RE-DECIDES NOTHING. Clearing here
        # unconditionally is the M6d defect: measured in the wired app, "Accept all" said
        # "3 tiles kept rejected" and pressing R on ONE ORDINARY TILE took the clause away
        # while all three rows were still refused. See :class:`_RefusalLedger`; the two
        # bulk paths and the per-row path all go through it, so the rule cannot be missing
        # from one of them.
        # Read inside the accept branch and nowhere else: a REJECT must keep working on a
        # frame that has wandered off the frozen schema, exactly as it did before, and it
        # has no refusal to record anyway.
        rows_arr = np.asarray(rows, dtype=np.intp)
        event_id = self._df["event_id"].to_numpy() if bool(accepted) else None
        if bool(accepted):
            self._refusals.forget(event_id[rows_arr])
        if bool(accepted) and flag_col is None:
            # This is a REFUSAL, not a human ruling.  In particular, do not stamp
            # ``reviewed`` (or clear a reconciliation conflict) before failing closed:
            # either write would emit ``curationBatchChanged`` and MainWindow would
            # replace this off-schema frame with its stored detection, erasing the
            # missing-flag disclosure on the surface that just raised it.  More
            # importantly, a request the app cannot decide is not a resolved training
            # label.  Valid rows refused by an automatic flag still pass the guard below
            # and are reviewed normally.
            self._refusals.record_blocked(len(rows))
            self._note = ("", "")
            self._sync_controls()
            self._render_progress()
            # Nothing was written, but the SHARED ledger just changed: announce the
            # refusal itself so the owner can repaint the other surfaces' notes.
            self.acceptanceRefused.emit()
            return
        # A curation gesture reviews its whole target scope, even when ``accepted``
        # already has the requested value.  Fresh detector rows normally start accepted,
        # so without this an explicit "Accept all shown" exported zero reviewed truths.
        conflict_changed = clear_curation_conflicts(self._df, rows_arr)
        review_changed = mark_reviewed(self._df, rows=rows_arr)
        self._note = ("", "")
        # ONE VECTORISED PASS, NOT ONE ``iat`` PER ROW. The per-row loop this replaces
        # cost 2.5-4.0 s of frozen GUI per "Reject all"/"Accept all" at 100,000 events
        # (5 pandas scalar round-trips per row); the decision it made is unchanged and
        # still per-row — ``derived AND requested`` — only the fan-out is batched:
        # O(rows) numpy instead of O(rows) Python/pandas calls, one frame write for the
        # whole gesture.
        before = self._df.iloc[:, col].to_numpy().astype(bool)[rows_arr]
        if bool(accepted):
            target = self._derived_accepted_rows(self._df, flag_col, rows_arr)
            refused = ~target
            # Which of the two it is depends on where the row STARTED: a row already
            # rejected is kept, a row sitting accepted beside a rejecting flag is the
            # forbidden state being repaired -- and the count goes DOWN. Which BUCKET
            # a row is in is history and is remembered; whether it is still refused at
            # all is re-read later, from the frame, by ``_RefusalLedger.refused_in``.
            kept = rows_arr[refused & ~before]
            repaired = rows_arr[refused & before]
            # Only the accept records; a reject has no refusal of its own to disclose.
            # An empty () is spelled out — ``[]`` is not an integer index array.
            self._refusals.record(kept=event_id[kept] if kept.size else (),
                                  repaired=event_id[repaired] if repaired.size else ())
        else:
            target = np.zeros(rows_arr.size, dtype=bool)
            kept = repaired = rows_arr[:0]
        changed_mask = before != target
        changed = rows_arr[changed_mask]
        if changed.size:
            # In place on the SHARED frame — the same cells the ``iat`` writes hit,
            # in the same (caller) row order, written once.
            self._df.iloc[changed, col] = target[changed_mask]
        if not changed.size:
            # The acceptance column may be unchanged while the review column moved. That
            # is still persistent curation, so it gets the same one batch announcement and
            # Undo step as a boolean-changing gesture.
            if ((review_changed or conflict_changed)
                    and filter_depends_on_labels_or_review(self.filter_box.currentText())):
                # Under Unreviewed (or a label filter) the stamp alone moves a tile out.
                self._refilter_keeping_selection()
            self._sync_progress()
            self._sync_controls()
            if review_changed or conflict_changed:
                self.set_undo_state(True, self._undo_label)
                self.view.viewport().update()
                self.curationBatchChanged.emit()
            if kept.size or repaired.size:
                # ...but the gesture DID record refusals on the shared ledger — the most
                # ordinary case being "Accept all" straight after detection, when every
                # measurable row is already accepted. Windows announce ledger changes,
                # not just written rows: without this the other two surfaces stayed
                # silent (or stale) about the very record this window just disclosed.
                self.acceptanceRefused.emit()
            return
        # The subset can change under a non-All filter (a rejected tile leaves "Accepted
        # only"), so the model is rebuilt before anything is announced -- a listener that
        # calls back in must not find the view describing the previous subset.
        #
        # THE SELECTION SURVIVES THIS WINDOW'S OWN WRITE TOO -- the same rule :meth:`refresh`
        # states for somebody else's edit, and the reason is identical: ``refilter()`` is a
        # model reset and a reset clears the selection. Restoring only the CURRENT tile (all
        # this used to do, via ``show_event``) turned every curation gesture into a silent
        # narrowing: sweep six tiles, press R, all six are rejected and ONE stays selected,
        # so the A you press when you realise it was the wrong block puts back one tile and
        # leaves five rejected. Nothing said the scope had shrunk. Snapshotted by
        # ``event_id`` and not by row because a refilter is exactly the moment rows move.
        # ``event_id`` and not by row because a refilter is exactly the moment rows move --
        # and through the ONE reset every frame edit uses (:meth:`_refilter_keeping_selection`).
        selected, back = self._refilter_keeping_selection()
        # ...and where the filter really did take tiles off the page (rejecting under
        # "Accepted only" is the ordinary case), the ones that could NOT come back are named
        # rather than quietly dropped -- the difference between a selection that shrank for
        # a reason the user can see and one that shrank silently. Remembered as ids, and the
        # note itself is re-derived from the live subset on every rebuild of the status
        # line, so switching the filter back to "All" makes it go away by itself instead of
        # leaving a stale count standing. See :meth:`_offpage_note`. THIS gesture's ids
        # replace the standing tuple: a gesture that keeps every tile leaves it empty.
        self._offpage_ids = tuple(ev for ev in selected if ev not in set(back))
        self._hotkey_note = ""
        self._sync_progress()
        self._sync_controls()
        self.view.viewport().update()

        # A WRITE HAPPENED HERE, SO A STEP EXISTS. Both signals below are checkpointed on
        # MainWindow's stack (``acceptanceChanged`` -> the panel's per-event step,
        # ``curationBatchChanged`` -> _checkpoint_gallery_curation), so this window can arm its
        # own Undo without asking: it is the LOWER bound of what Edit ▸ Undo can do, and
        # :meth:`set_undo_state` replaces it with the real state and the real label the moment
        # MainWindow pushes one. Armed only when a row actually moved -- a refused accept
        # writes nothing and has nothing to walk back.
        if changed.size or review_changed or conflict_changed:
            self.set_undo_state(True, self._undo_label)
        if changed.size == 1 and not (review_changed and rows_arr.size > 1):
            # WHAT WAS WRITTEN, never what was asked for. The owner mirrors this straight onto
            # the table's checkbox, so announcing the request would put ``True`` on a row this
            # method just declined to accept -- the leak closed above, re-opened one signal
            # later and one panel over.
            row = int(changed[0])
            self.acceptanceChanged.emit(int(self._df.iloc[row]["event_id"]),
                                        bool(self._df.iat[row, col]))
        else:
            self.curationBatchChanged.emit()

    @staticmethod
    def _derived_accepted_rows(df: pd.DataFrame, flag_col: int,
                               rows: np.ndarray) -> np.ndarray:
        """:func:`_derived_accepted` over many rows in ONE pass.

        The RULE stays the single per-value function — it is evaluated once per UNIQUE
        flag string and broadcast back over the rows, so a fifth rejecting token added
        to ``AUTOMATIC_REJECTION_FLAGS`` reaches this path with no second edit, exactly
        as it reaches the per-row path. Unique flag values are a handful while rows are
        1e5, which is the whole speed-up.
        """
        flags = df.iloc[:, flag_col].to_numpy()[rows]
        codes, uniques = pd.factorize(flags, use_na_sentinel=False)
        per_unique = np.fromiter((_derived_accepted(u) for u in uniques),
                                 dtype=bool, count=len(uniques))
        return per_unique[codes]

    def refusal_note(self) -> str:
        """The disclosure the LAST curation gesture owes the user, or ``""``.

        Read by :meth:`_sync_controls`, which is the only writer of the status line -- so
        the note is rebuilt WITH the line rather than written over its output. It used to be
        appended after the rebuild, which cost it its life: ``curationBatchChanged`` is wired
        (through ``MainWindow.SURFACE_MUTATION_SIGNALS``) to a refresher that calls
        :meth:`refresh`, which rebuilds the line -- and even the no-change path lost it to
        the relayout the click itself provokes. Measured in the wired app on the owner's
        -600 mV recording: "Accept all 317" left "316 of 317 accepted" with no clause at
        all, both immediately after the click and after draining the event loop.

        NOT cleared by the next gesture. An accept re-decides its own scope
        (:meth:`_set_rows_accepted`), a reject re-decides nothing, and only a new frame
        (:meth:`set_events` / :meth:`clear`) drops the record -- see :class:`_RefusalLedger`
        for the measured defect that came of doing otherwise. Never cleared by a repaint.
        In the wired app "a new frame" means a new detection LINEAGE: MainWindow re-hands
        the SAME shared ledger through :meth:`set_events` when an undo or a merge re-fed
        the same detection, so those re-feeds no longer silence a standing disclosure.
        """
        return _refusal_note(*self._refused, noun="tile")

    # -- the refusal counter ----------------------------------------------------------------------
    def _clear_refusals(self) -> None:
        """Drop the whole ledger. A NEW FRAME ONLY -- never a gesture.

        This used to be the first line of every curation write, which is the M6d defect:
        a reject re-decides no refusal, so clearing here made the note a per-gesture
        artefact that one ordinary R keypress took down while three rows were still
        refused. What an ACCEPT re-decides, it re-decides by scope
        (``_RefusalLedger.forget``), not by wiping the record.

        DETACHED, never ``.clear()``ed: the instance may be the owner's SHARED ledger
        (see :meth:`set_events`), read by the inspector and the events table as well, and
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

    # -- labelling -------------------------------------------------------------------------------
    def set_vocabulary(self, vocab: LabelVocabulary) -> None:
        """Install the shared palette, combo and safe explicit shortcuts."""
        self._vocab = vocab
        self._sync_label_shortcuts()
        self._updating = True
        try:
            current = self.label_box.currentText()
            self.label_box.clear()
            for e in vocab.entries:
                # THE shared assignment, never a second guess at it. See LabelVocabulary.hotkeys.
                hot = vocab.hotkey_for(e)
                self.label_box.addItem(f"{e.token}" + (f"   [{hot}]" if hot else ""), e.token)
            if current:
                idx = self.label_box.findText(current)
                if idx >= 0:
                    self.label_box.setCurrentIndex(idx)
        finally:
            self._updating = False
        has = bool(vocab.entries)
        for w in (self.label_box, self.apply_label_btn, self.clear_label_btn):
            w.setEnabled(has)
        # The status line's digit hint is derived from the palette (see _label_hint), so a
        # new palette rebuilds the line; a note about a digit the OLD palette did not arm
        # is about a palette that is gone.
        self._hotkey_note = ""
        self._sync_progress()
        if hasattr(self, "status"):
            self._sync_controls()
        self.view.viewport().update()

    def _sync_label_shortcuts(self) -> None:
        """Rebuild explicit, non-digit label actions from the current vocabulary."""
        for action in self._custom_label_actions:
            self.removeAction(action)
            action.setEnabled(False)
            action.deleteLater()
        self._custom_label_actions = []
        for ch in self._vocab.hotkeys():
            if ch in "123456789":
                continue
            action = QtGui.QAction(self)
            action.setShortcut(QtGui.QKeySequence(ch))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.setAutoRepeat(False)
            action.triggered.connect(
                lambda _checked=False, ch=ch: self.decide_label(ch))
            self.addAction(action)
            self._custom_label_actions.append(action)

    def vocabulary(self) -> LabelVocabulary:
        return self._vocab

    def armed_digits(self) -> tuple[str, ...]:
        """The digits 1–9 that currently write a label, ascending -- THE shared assignment
        (:meth:`LabelVocabulary.hotkeys`), never a second count of the palette."""
        return tuple(sorted(ch for ch in self._vocab.hotkeys() if ch in "123456789"))

    def _label_hint(self) -> tuple[str, str]:
        """``(gesture clause, palette clause)`` for the status line.

        THE HINT NAMES THE KEYS THAT ARE ARMED. The line used to end "1–9 to label"
        unconditionally, on a session whose LabelVocabulary is ``entries=[]`` -- so on every
        session that had not built a palette yet, the one line that enumerates this window's
        gestures advertised nine that wrote nothing and said nothing; with a two-entry
        palette, seven of the nine were silent (measured: flag unchanged, status unchanged).
        Nothing on any surface named Analysis ▸ Event labels… as the place that arms them.
        """
        armed = self.armed_digits()
        # "…acts on EVERY selected tile" is the half of this sentence the owner never had.
        # The line was singular throughout -- "click to select, R to reject … double-click
        # opens IT" -- and the audit traced his "the gallery rejects one event at a time" to
        # exactly that: shift-click, drag-a-box, ⌘A and the two "…selected" buttons were named
        # NOWHERE on screen except inside one button's tooltip, and this window has no menu
        # bar to browse (``menuBar().actions() == []``). The keys are unchanged; what changed
        # is that the line says what they act on. See :meth:`_batch_hint` for the gestures
        # that BUILD the selection this clause spends.
        tail = " — each acts on every selected tile"
        if not armed:
            return ("click to select, R to reject" + tail,
                    "no labels defined — define some under Analysis ▸ Event labels… to arm "
                    "the digit keys")
        return f"click to select, R to reject, {_digit_span(armed)} to label" + tail, ""

    def _batch_hint(self) -> str:
        """The clause that names the BATCH gestures -- how a selection is built.

        AHEAD OF THE PER-TILE HINT IN THE CLAUSE ORDER, because the status line is cut from
        the tail (:class:`_StatusLine`) and this is the sentence the audit called the highest
        value-per-hour change in the window: the one that tells a user who believes curation
        is one-at-a-time that it is not. The modifier is asked of Qt (:func:`_cmd_key`), so on
        the owner's Mac it reads ⌘ -- a literal "Ctrl-click" there is the CONTEXT MENU.
        """
        cmd = _cmd_key()
        return (f"select in bulk: drag a box or shift-click for a run, {cmd}-click to add or "
                f"drop one, {cmd}A all on the page, Esc none, {cmd}I inverse")

    def _selected_rows(self) -> list[int]:
        rows = [self._model.df_row(i) for i in self.view.selectionModel().selectedIndexes()]
        return [r for r in rows if r is not None]

    def apply_hotkey(self, ch: str) -> None:
        """A digit press: look the character up in the palette and write it.

        A digit that writes nothing SAYS SO on the status line -- an unarmed key (the
        palette has no entry on it) and an armed key pressed over an empty selection were
        both silent no-ops, indistinguishable from a label that landed on a tile the user
        was not looking at. The sentence is state (:attr:`_hotkey_note`), rebuilt with the
        line, and cleared by the next gesture that writes.
        """
        entry = self._vocab.for_hotkey(ch)
        if entry is None:
            armed = self.armed_digits()
            self._hotkey_note = (
                f"{ch} is not armed — "
                + ("no labels defined; define some under Analysis ▸ Event labels… to arm "
                   "the digit keys"
                   if not armed else
                   f"the palette arms {_digit_span(armed)} only; Analysis ▸ Event labels… "
                   "adds more"))
            self._sync_controls()
            return
        if not self._selected_rows():
            self._hotkey_note = (f"{ch} labels the SELECTED tiles and none is selected — "
                                 f"click or drag over tiles, then press {ch}")
            self._sync_controls()
            return
        self._hotkey_note = ""
        self.apply_label(entry.key, entry.value)

    def apply_selected_label(self) -> None:
        token = self.label_box.currentData()
        if token:
            key, _, value = str(token).partition("=")
            self.apply_label(key, value)

    def clear_selected_label(self) -> None:
        token = self.label_box.currentData()
        if not token or self._df is None:
            return
        rows = self._selected_rows()
        if not rows:
            return
        label_changed, review_changed = unlabel_rows(
            self._df, rows, str(token).partition("=")[0])
        if label_changed or review_changed:
            self._after_label_write(len(rows))

    def apply_label(self, key: str, value: str) -> int:
        """Write ``key=value`` onto every selected tile. Returns how many rows changed.

        Writes the SHARED events frame, so the label reaches the table, the exports and the
        project by the same route curation does — and lands in the ``flag`` column, so it
        survives a later merge or boundary edit without any code here.
        """
        if self._df is None:
            return 0
        rows = self._selected_rows()
        if not rows:
            return 0
        changed, review_changed = label_rows(self._df, rows, key, value)
        if changed or review_changed:
            self._after_label_write(len(rows))
        return changed

    def _after_label_write(self, n_selected: int) -> None:
        """Repaint and announce — WITHOUT resetting the model.

        This called refilter() first, copied from the curation path, and that was wrong in a
        way that made multi-key labelling impossible: refilter() is a model reset, a reset
        clears the selection, so the tiles you had just labelled were deselected and the next
        digit wrote nothing. Curation needs the reset because the Accepted/Rejected filters
        are computed from `accepted`; a label never changes `accepted`, so the subset cannot
        have moved.

        EXCEPT UNDER A LABEL OR REVIEW FILTER (Unreviewed, Unlabelled, Labelled: k=v), where
        the write itself moves tiles in or out. Then the selection-keeping reset runs: the
        tiles still in the subset stay selected, and the ones that left are named on the
        status line ("filtered out of Unlabelled"). Under All / Accepted / Rejected nothing
        is reset, exactly as before.
        """
        self._hotkey_note = ""
        self._note = ("", "")
        # A label write is checkpointed too (``labelsChanged`` -> _checkpoint_labels), so the
        # same lower bound holds: see _set_rows_accepted for why this window may arm its own
        # Undo but may never disarm it.
        if n_selected:
            self.set_undo_state(True, self._undo_label)
        if filter_depends_on_labels_or_review(self.filter_box.currentText()):
            self._refilter_keeping_selection()
        self._sync_progress()
        self._sync_controls()
        self.view.viewport().update()
        self.labelsChanged.emit()

    def label_of(self, df_row: int) -> str:
        """The label text a tile shows: the chosen key's value, else the first label on the
        row. Empty when the row carries none."""
        if self._df is None or not (0 <= df_row < len(self._df)):
            return ""
        from ..analysis.labels import parse_labels
        labels = parse_labels(self._df.iloc[df_row].get("flag", ""))
        if not labels:
            return ""
        if self._label_key and self._label_key in labels:
            return labels[self._label_key]
        return next(iter(labels.values()))

    # -- Advance after decision: the keyboard doors that may move on ------------------------
    def advance_after_decision(self) -> bool:
        """The shared setting, re-read so a change made in the inspector holds here too."""
        on = _read_advance_setting()
        if self._advance_check.isChecked() != on:
            was = self._advance_check.blockSignals(True)
            self._advance_check.setChecked(on)
            self._advance_check.blockSignals(was)
        return on

    def set_advance_after_decision(self, on: bool) -> None:
        self.advance_after_decision()
        self._advance_check.setChecked(bool(on))

    def _on_advance_toggled(self, on: bool) -> None:
        _write_advance_setting(bool(on))
        if self._note != ("", ""):
            self._note = ("", "")
            self._render_progress()
        self.advanceAfterDecisionChanged.emit(bool(on))

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self.advance_after_decision()        # the inspector may have changed the shared choice
        self._share_status_bar()

    def _advance_target(self) -> tuple[int, list[int], int] | None:
        """``(decided event_id, subset order, its position)`` when a decision may advance:
        Advance on, and the selection is exactly the current tile. A sweep never moves."""
        if self._df is None or not self.advance_after_decision():
            return None
        row = self._model.df_row(self.view.currentIndex())
        if row is None or self._selected_rows() != [row]:
            return None
        order = self._model.df_rows()
        try:
            pos = order.index(row)
        except ValueError:
            return None
        return int(self._df["event_id"].iloc[row]), order, pos

    def _decision_landed(self, event_id: int, accepted: bool | None) -> bool:
        """An accept that the derived half refused did not land; anything else did."""
        row = self._row_for_id(event_id)
        if row is None:
            return False
        return not (accepted and not bool(self._df.iloc[row]["accepted"]))

    def decide_selected(self, accepted: bool) -> bool:
        """A / R: :meth:`set_accepted_selected`, then -- for a one-tile selection with
        Advance on, once the decision landed -- the next tile. ``True`` when it moved."""
        target = self._advance_target()
        self.set_accepted_selected(accepted)
        if target is None:
            return False
        if not self._decision_landed(target[0], bool(accepted)):
            return self._after_refusal(target)
        return self._advance_to(target)

    def decide_toggle_selected(self) -> bool:
        """Space, as :meth:`decide_selected`."""
        target = self._advance_target()
        requested = None
        if target is not None:
            requested = not bool(self._df.iloc[self._row_for_id(target[0])]["accepted"])
        self._toggle_selected()
        if target is None:
            return False
        if not self._decision_landed(target[0], requested):
            return self._after_refusal(target)
        return self._advance_to(target)

    def _after_refusal(self, target: tuple[int, list[int], int]) -> bool:
        """A refused accept does not move on: the tile stays current, with the refusal on
        the status line. Unless the review stamp the refusal still makes took the tile out
        of the subset (Unreviewed): then nothing would be selected and the next key would
        act on nothing, so the selection goes where an advance would have. ``False``
        either way -- the decision did not land."""
        row = self._row_for_id(target[0])
        if not self._selected_rows() and row not in set(self._model.df_rows()):
            self._advance_to(target, end_note=False)
        return False

    def decide_label(self, ch: str) -> bool:
        """A label key: :meth:`apply_hotkey`, then advance as :meth:`decide_selected`."""
        target = self._advance_target()
        self.apply_hotkey(ch)
        if (target is None or self._vocab.for_hotkey(ch) is None
                or self._df is None or "flag" not in self._df.columns):
            return False
        return self._advance_to(target)

    def _advance_to(self, target: tuple[int, list[int], int], *, end_note: bool = True) -> bool:
        """Select the first tile after the decided one, in the order the subset had when
        the key was pressed, that is still in the subset -- or stay and say so."""
        decided, order, pos = target
        alive = set(self._model.df_rows())
        for row in order[pos + 1:]:
            if row in alive:
                moved = self._move_cursor_to(int(self._df["event_id"].iloc[row]))
                if moved:
                    self._advanced_from = int(decided)
                return moved
        if not end_note:
            return False
        unreviewed = ~reviewed_mask(self._df)
        mode = self.filter_box.currentText()
        earlier = sum(1 for r in order[:pos] if r in alive)
        self._note = end_of_filter_note(
            mode, noun="tile", n_earlier=earlier if mode == FILTER_UNREVIEWED else 0,
            n_unreviewed=int(unreviewed.sum()), first_key=_native_key("Ctrl+U"))
        self._render_progress()
        return False

    def _move_cursor_to(self, event_id: int) -> bool:
        """Make ``event_id``'s tile the current AND only selected tile, turning to its page
        first, and announce it as a click would (:meth:`_TileView.select_only`, which no
        held or latched modifier can turn into an extension)."""
        idx = self._index_on_its_page(event_id)
        sm = self.view.selectionModel()
        if not idx.isValid() or sm is None:
            return False
        self._advancing = True
        try:
            self._select_and_announce(idx)
        finally:
            self._advancing = False
        self.view.scrollTo(idx)
        self._sync_controls()
        self.view.viewport().update()
        return self.current_event_id() == int(event_id)

    def first_unreviewed(self) -> int | None:
        """Select the first tile of the Show subset, in the current sort, that no person has
        ruled on (⌘U / the status-bar button); ``None``, with a note, when there is none."""
        if self._df is None:
            return None
        unreviewed = ~reviewed_mask(self._df)
        for row in self._model.df_rows():
            if unreviewed[row]:
                eid = int(self._df["event_id"].iloc[row])
                return eid if self._move_cursor_to(eid) else None
        self._note = no_unreviewed_note(self.filter_box.currentText(), noun="tile",
                                        n_unreviewed=int(unreviewed.sum()), chooser="Show")
        self._render_progress()
        return None

    def _request_undo(self) -> None:
        """This window's Undo / ⌘Z: MainWindow's undo, then back to the tile the last
        advance moved away from -- only when the same events came back AND the undo
        changed that event (the inspector's rule; see its ``_request_undo``)."""
        back, before = self._advanced_from, self._df
        self._advanced_from = None
        self.undoRequested.emit()
        after = self._df
        if back is None or after is before or not same_events(before, after):
            return
        row = self._row_for_id(back)
        if (row is None or back == self.current_event_id()
                or row_unchanged(before, after, row)):
            return
        self._move_cursor_to(back)

    # -- the progress line ------------------------------------------------------------------
    def progress_text(self) -> str:
        return self._progress.text() if hasattr(self, "_progress") else ""

    def filter_options(self) -> tuple[str, ...]:
        return self._filter_items

    def _sync_progress(self) -> None:
        """Recount curation progress for the frame and refresh what reads it (the Show
        items, the line). On writes and re-feeds only, never on a page turn."""
        self._progress_state = (curation_progress(self._df, self._vocab)
                                if self._df is not None else None)
        self._rebuild_filter_items()
        self._render_progress()

    def _render_progress(self) -> None:
        if not hasattr(self, "_progress"):
            return
        short, full = self._note
        p = self._progress_state
        if p is None:
            self._progress.set_line(short, full)
            self._share_status_bar()
            return
        line = (f"{short} · " if short else "") + format_progress(p)
        self._progress.set_line(line, "\n\n".join(s for s in (full, describe_progress(p)) if s))
        self._share_status_bar()

    def _share_status_bar(self) -> None:
        """Size the progress line's floor and the project label by what each says (see
        :func:`_status_share`). The pair's total width is the bar's to give, so this moves
        neither First unreviewed nor Save."""
        prog = getattr(self, "_progress", None)
        label = getattr(self, "_project_save_label", None)
        if prog is None or label is None:
            return
        fixed = (self._first_unreviewed_btn.sizeHint().width()
                 + self._curation_status.layout().spacing())
        avail = self._curation_status.width() + label.width() - fixed
        # The label's floor is its save state; the file name is the first thing to go.
        floor, width = _status_share(avail, prog.sizeHint().width(), label.sizeHint().width(),
                                     label.minimumWidth(), _PROGRESS_MIN_W)
        prog.setMinimumWidth(floor)
        label.setMaximumWidth(max(width, label.minimumWidth()))

    def _rebuild_filter_items(self, extra: str | None = None) -> None:
        """Refill the Show combo when the offered filters change, keeping the one shown
        (and ``extra``, about to be chosen). Signals are blocked: a rebuild is not a choice."""
        if not hasattr(self, "filter_box"):
            return
        current = self._model._filter
        opts = filter_options(self._progress_state, self._vocab, extra or current)
        if current not in opts and is_filter(current):
            opts = (*opts, current)
        if opts == self._filter_items:
            return
        box = self.filter_box
        blocker = QtCore.QSignalBlocker(box)
        try:
            box.clear()
            for text in opts:
                if text == FILTER_UNLABELLED and box.count():
                    box.insertSeparator(box.count())
                box.addItem(text)
                box.setItemData(box.count() - 1, filter_display_candidates(text),
                                _SHORT_TEXTS_ROLE)
            box.setCurrentText(current)
        finally:
            blocker.unblock()
        self._filter_items = opts
        fm = box.fontMetrics()
        box.view().setMinimumWidth(max(fm.horizontalAdvance(t) for t in opts) + 32)

    def reflect_acceptance(self, event_id: int, accepted: bool) -> None:
        """Somebody else curated this event; repaint to match. Never re-emits.

        IT NO LONGER TOUCHES THE NOTE AT ALL. It used to zero the whole counter whenever the
        mirrored verdict accepted a row the counter had counted -- correct only when the
        counter held exactly one row. On the owner's -600 mV recording it held four, one tick
        in the events table erased all four, and three rows stayed refused with nothing on
        either surface saying so. The counter is now recomputed from the rows themselves
        (:attr:`_refused`), so the write two lines up is the whole of the update: the accepted
        row stops being refused because the FRAME says it is accepted, and the other three go
        on being counted. Nothing here has to know that, which is the point -- ordering tricks
        in this method are what produced the defect.

        AND IT KEEPS THE SELECTION. This called the bare ``refilter()`` -- a model reset with
        no snapshot and no restore -- and MainWindow connects it to the same
        ``eventAcceptanceChanged`` as :meth:`refresh`, after it; so every curation made on
        the table or the inspector had ``refresh`` put the user's sweep back and this method
        wipe it a moment later, leaving "Reject selected" enabled over nothing. See
        :meth:`_refilter_keeping_selection` for the measurement; there is now one reset.
        """
        row = self._row_for_id(event_id)
        if row is None or self._df is None:
            return
        col = self._df.columns.get_loc("accepted")
        if bool(self._df.iat[row, col]) != bool(accepted):
            self._df.iat[row, col] = bool(accepted)
        self._advanced_from = None
        self._note = ("", "")
        self._sync_progress()
        self._refilter_keeping_selection()
        self._sync_controls()
        self.view.viewport().update()

    # -- staleness (the events table's badge set, mirrored) ---------------------------------------
    # -- Edit ▸ Undo, reached from the window the batch gesture happens in -----------------------
    def set_undo_state(self, enabled: bool, label: str = "") -> None:
        """MainWindow's Edit ▸ Undo, mirrored onto this window's button (``label`` is that
        action's own text, e.g. "Undo Curate 45 events").

        THE STACK IS NOT COPIED, ONLY ITS STATE. There is one undo in this application
        (core/history.py) and the gesture goes back to it through :attr:`undoRequested`; this
        is the return leg, so the button can be disabled when there is nothing to undo and can
        say WHAT it would undo when there is. MainWindow pushes it (``_act_undo.changed`` ->
        here); until it does, the button is the lower bound this window can prove on its own
        — see :attr:`_undo_enabled`.
        """
        self._undo_enabled = bool(enabled)
        self._undo_label = str(label or "")
        self._sync_undo_affordance()

    def _sync_undo_affordance(self) -> None:
        """One writer for the Undo button and the ⌘Z action: both read :attr:`_undo_enabled`.

        The KEY is gated with the button on purpose. A shortcut that fires into a disabled
        stack is the defect this window already had -- ⌘Z did nothing here and said nothing --
        and "nothing happens, silently" is not improved by moving which object it happens on.
        """
        label = self._undo_label or "Undo"
        self.undo_btn.setEnabled(self._undo_enabled)
        # THE TEXT NEVER CHANGES, only the tooltip. "Undo Curate 45 events" on the button
        # would widen it by ~110 px and slide "Accept all shown" and "Reject all shown" along
        # the footer -- under the pointer that is about to click one of them, every time a
        # curation lands. What the step IS belongs in the tooltip, where reading it costs a
        # hover and nothing moves.
        seq = self.act_undo.shortcut().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
        self.undo_btn.setToolTip(
            f"{label} ({seq}). Curation in this window is undone from the main window's "
            f"Edit ▸ Undo; this is the same step, without leaving the gallery."
            if self._undo_enabled else
            "Nothing to undo yet. Curate some tiles and this walks the last step back.")
        self.act_undo.setEnabled(self._undo_enabled)

    def set_stale_columns(self, columns: tuple[str, ...] = (), note: str = "") -> None:
        """Mark ``columns`` as measured by a superseded estimator (clear with ``()``).

        The seam the owner pushes the events table's OWN badge set through -- the same
        tuple ``EventTable.set_stale_columns`` was given (``REMEASURED_COLUMNS`` on a
        baseline or calibration change) and the banner's sentence -- so three surfaces
        cannot tell three stories about one frame. Measured before it existed: after one
        knob in the Preprocess tab the table badged nine headers ⚠ and raised "Depth,
        frac_depth, ECD, mean/max deviation and SNR are stale", while this window went on
        sorting and captioning by ``depth_pA`` with no mark anywhere -- and it is a separate
        top-level window sitting OVER the banner. The captions carry the glyph
        (:meth:`caption_for`); the status line carries the sentence (:meth:`_stale_line`).
        Kept across re-feeds, exactly as the table's model keeps its own: the owner's
        staleness check runs on every re-feed and pushes the answer again.
        """
        cols = tuple(str(c) for c in columns)
        if cols == self._stale_columns and str(note) == self._stale_note_text:
            return
        self._stale_columns = cols
        self._stale_note_text = str(note)
        if hasattr(self, "status"):
            self._sync_controls()
            self.view.viewport().update()

    def stale_columns(self) -> tuple[str, ...]:
        return self._stale_columns

    def _stale_line(self) -> str:
        """The staleness sentence for the status line, naming the sort column when the
        grid is ordered by a badged number; "" when nothing is marked."""
        if not self._stale_columns:
            return ""
        note = self._stale_note_text or (
            ", ".join(self._stale_columns)
            + " are stale — measured by an estimator that is no longer the configured one; "
              "see the banner on the Analysis panel")
        sort = self.sort_box.currentText() if hasattr(self, "sort_box") else _SORT_ORDER
        col = _SORT_COLUMNS.get(sort)
        if col in self._stale_columns:
            return f"⚠ sorted and captioned by stale {col}: {note}"
        return f"⚠ {note}"

    def _fraction_note(self) -> str:
        """Disclose above-one values on this page without marking the tile artwork."""
        if self._df is None or "frac_depth" not in self._df:
            return ""
        rows = self._model.page_df_rows()
        if not rows:
            return ""
        values = self._df.iloc[rows]["frac_depth"].to_numpy(dtype=np.float64)
        count = int(np.count_nonzero(np.isfinite(values) & (values > FRAC_DEPTH_ENHANCEMENT)))
        if not count:
            return ""
        return (f"⚠ {count} event(s) on this page have dI/I0 > 1: not a blockade fraction. "
                "Check polarity, the baseline reference and amplifier calibration before interpretation.")

    # -- controls ---------------------------------------------------------------------------------
    def set_filter(self, mode: str) -> None:
        if is_filter(mode):
            # A non-editable combo ignores text it does not hold: offer the filter first.
            if mode not in self._filter_items:
                self._rebuild_filter_items(extra=mode)
            self.filter_box.setCurrentText(mode)

    def set_sort(self, mode: str) -> None:
        if mode in _SORTS:
            self.sort_box.setCurrentText(mode)

    def set_tile_size(self, name: str) -> None:
        if name in _TILE_SIZES:
            self.size_box.setCurrentText(name)

    def _on_filter_changed(self, text: str) -> None:
        if not is_filter(text):
            return
        self._model.set_filter(text)
        if self._note != ("", ""):
            self._note = ("", "")
            self._render_progress()
        self._sync_controls()

    def _on_sort_changed(self, text: str) -> None:
        self._model.set_sort(text)
        self._sync_controls()
        self.view.viewport().update()      # captions follow the sort

    # -- pages ------------------------------------------------------------------------------------
    def set_page(self, page: int) -> None:
        if self._model.set_page(int(page)):
            self._sync_controls()

    def page(self) -> int:
        return self._model.page()

    def page_count(self) -> int:
        return self._model.page_count()

    def set_page_size(self, size) -> None:
        """``"Fit"`` or a tile count."""
        if str(size) in _PAGE_SIZES:
            self.page_size_box.setCurrentText(str(size))

    def _on_page_spin(self, value: int) -> None:
        if self._updating:
            return
        self.set_page(int(value) - 1)

    def _on_page_size_changed(self, text: str) -> None:
        self._fit_pages = (text == _FIT)
        if self._fit_pages:
            self._apply_fit_page_size()
        else:
            # LITERAL, never capped to the window -- see _PAGE_SIZES for the measurement
            # and for why capping it was rejected. The overflow it can produce is disclosed
            # by the status line rather than prevented.
            try:
                self._model.set_page_size(int(text))
            except ValueError:
                return
        self._sync_controls()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Recompute "Fit" on the VIEWPORT's own resize — the only authoritative source.

        ``_apply_fit_page_size`` reads ``self.view.viewport().size()``, and until this
        filter existed the only two things that called it were the page-size combo (during
        ``_build_ui``, before the window has ever been laid out) and this window's
        ``resizeEvent`` (which fires BEFORE the central widget's layout has given the view
        its share of the new size). Both therefore measured a stale viewport. Measured on
        first show with 300 events: the viewport ended up 1471x588, a grid of 9 columns by 4
        rows = 36 tiles -- and "Fit" had chosen 12, so the status line read "1–12 of 300" on
        a screen with room for three times that, and nudging the window by one pixel
        corrected it to 32. A page that is a third of a screenful is precisely the "number
        per window does not match the number displayed" the owner reported.

        The viewport's resize arrives AFTER the layout has settled, every time, including
        the one that follows the first show — so this is the event that knows the answer,
        and there is nothing left to defer with a zero-timer.
        """
        if obj is self.view and event.type() == QtCore.QEvent.KeyPress:
            # THE KEYS, CAUGHT ON THE VIEW -- the widget that actually has the focus.
            # Reached only when the shortcut map did NOT match (no focus widget in here
            # yet, or a non-spontaneous synthetic event), so a key is handled once and
            # never twice. Returning True also stops QListView's type-to-search from
            # eating a / r / 1-9 to jump to a tile whose caption starts with that letter.
            # Same seam, same reason, as EventTable.eventFilter.
            if self._handle_own_key(event):
                return True
            if (self._parked is not None and event.key() in (Qt.Key_Left, Qt.Key_Right)
                    and not (event.modifiers() & ~Qt.KeypadModifier)
                    and self._step_from_parked(event.key() == Qt.Key_Right)):
                return True
        if obj is self.view.viewport() and event.type() == QtCore.QEvent.Resize:
            # ``status`` is the last thing _build_ui creates: a viewport resize delivered
            # while the toolbar is still being assembled has nothing to sync.
            if hasattr(self, "status"):
                if self._fit_pages:
                    self._apply_fit_page_size()
                # Resync either way, for the same reason resizeEvent does: with a fixed page
                # size a resize is what makes the page start (or stop) overflowing.
                self._sync_controls()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 (Qt override)
        """The same keys if they reach the window itself — no widget in here focused, or a
        synthesised event. See :meth:`_handle_own_key`."""
        if not self._handle_own_key(event):
            super().keyPressEvent(event)

    def _handle_own_key(self, event: QtGui.QKeyEvent) -> bool:
        """Run the action THIS window registered for ``event``'s key; ``True`` if it was ours.

        THE ACTIONS STAY THE ONE REGISTRY. This does not re-list the keys -- a second list
        is how a shortcut and its fallback drift into disagreeing about what R means -- it
        asks the QActions already on this window which of them owns the combination, and
        triggers that one. Disabled actions are skipped, so ⌘Z here is as dead as the Undo
        button next to it when there is nothing to undo.

        WHY THERE IS A FALLBACK AT ALL. The shortcuts are
        ``Qt.WidgetWithChildrenShortcut`` so they cannot answer for the events table (see
        the loop that builds them); the price is that Qt's shortcut map wants
        ``QApplication.focusWidget()`` to be a descendant of this window, and there are two
        states where it is not: a window shown offscreen or not yet activated has no focus
        widget at all, and a key delivered with ``QApplication.sendEvent`` rather than by the
        platform never reaches the shortcut map. Measured with the gallery shown but never
        activated: with the narrower context alone, R on the tile view did nothing (0 events
        curated); with this filter, R rejected the whole selection, which is the behaviour
        the window-context shortcut had. So the narrowing costs the events table nothing and
        costs this window nothing either.
        """
        try:
            seq = QtGui.QKeySequence(event.keyCombination())
        except (TypeError, ValueError):  # pragma: no cover -- defensive, key events carry one
            return False
        for act in self.actions():
            if not act.isEnabled():
                continue
            shortcut = act.shortcut()
            if not shortcut.isEmpty() and shortcut == seq:
                if event.isAutoRepeat() and not act.autoRepeat():
                    return True              # a held decision key: consumed, not repeated
                act.trigger()
                return True
        return False

    def _grid_cells(self) -> tuple[int, int, int, int]:
        """``(cell_w, cell_h, cols, viewport_h)`` — the one place the layout arithmetic
        lives, shared by the Fit page size and the overflow hint so the two can never
        disagree about what a screenful is."""
        tw, th = self.tile_size()
        cell_w = tw + 2 * _TILE_PAD + 2 * self.view.spacing()
        cell_h = th + _CAPTION_H + 2 * _TILE_PAD + 2 * self.view.spacing()
        vp = self.view.viewport().size()
        cols = max(1, vp.width() // max(cell_w, 1))
        return cell_w, cell_h, cols, vp.height()

    def _apply_fit_page_size(self) -> None:
        """One screenful, from the viewport and the tile size.

        An UNSHOWN window has no viewport to measure -- the arithmetic over an unrealised
        one lands on a single tile -- so until the first viewport resize this is a plain
        count instead (see :data:`_FIT_FALLBACK_PAGE_SIZE`).
        """
        if not self.view.isVisible():
            self._model.set_page_size(_FIT_FALLBACK_PAGE_SIZE)
            return
        _cw, cell_h, cols, vp_h = self._grid_cells()
        rows = max(1, vp_h // max(cell_h, 1))
        self._model.set_page_size(int(cols * rows))

    def _page_overflows(self) -> bool:
        """Whether the current page needs more vertical room than the viewport has —
        i.e. whether it scrolls (owner note #12's hint)."""
        n = self._model.rowCount()
        if n == 0:
            return False
        _cw, cell_h, cols, vp_h = self._grid_cells()
        rows_needed = -(-n // cols)          # ceil
        return rows_needed * cell_h > vp_h

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_fit_pages", False):
            self._apply_fit_page_size()
        # ALWAYS resync, not only under Fit: with a fixed page size a resize is what
        # makes the page overflow (or stop overflowing) the window, and the status
        # line's "scroll for the rest" hint must track that (owner note #12).
        if hasattr(self, "_model"):
            self._sync_controls()
        self._share_status_bar()

    def set_shared_scale(self, on: bool) -> None:
        self.shared_scale_box.setChecked(bool(on))

    def _on_shared_scale_toggled(self, on: bool) -> None:
        # No envelope invalidation: they are cached in DATA UNITS precisely so this costs
        # a repaint and not a recompute of every tile in the grid.
        self._shared_scale = bool(on)
        self.view.viewport().update()

    def _on_size_changed(self, name: str) -> None:
        if name not in _TILE_SIZES:
            return
        self._tile_name = name
        if self._fit_pages:
            self._apply_fit_page_size()
        # The envelopes are cached PER TILE WIDTH, so a size change is a different picture,
        # not a scaled one -- recomputing is what keeps a large tile from being a blurred
        # small one.
        self._envelopes.clear()
        self._model.refilter()
        self.view.viewport().update()

    def _sync_controls(self) -> None:
        n_pages = self._model.page_count()
        page = self._model.page()
        self._updating = True
        try:
            self.page_spin.setMaximum(max(1, n_pages))
            self.page_spin.setValue(page + 1)
        finally:
            self._updating = False
        self.page_label.setText(f"of {n_pages}")
        self.first_btn.setEnabled(page > 0)
        self.prev_btn.setEnabled(page > 0)
        self.next_btn.setEnabled(page < n_pages - 1)
        self.last_btn.setEnabled(page < n_pages - 1)

        n = self._model.rowCount()
        has = n > 0
        for w in (self.accept_selected_btn, self.reject_selected_btn,
                  self.accept_all_btn, self.reject_all_btn, self.inspect_btn):
            w.setEnabled(has)
        if hasattr(self, "_first_unreviewed_btn"):
            self._first_unreviewed_btn.setEnabled(self._df is not None)
        if self._df is None:
            # TWO STATES, TWO SENTENCES. A completed run that found nothing is a finding --
            # the null control is SUPPOSED to produce it -- and it must not be reported as
            # "you have not run detection yet", which withholds the answer and asks for the
            # run again. The wording is the Event inspector's, deliberately: the two windows
            # are handed the same result and must not describe it in different words.
            self.status.setText(
                ("No manual events yet — bracket an event on the trace and press Ctrl+M."
                 if self._module_id == "detect.manual" else
                 "Detection found no events — nothing to show in the gallery.")
                if self._ran_empty else
                "No detection — run detection to fill the gallery.")
            return
        n_all = len(self._df)
        n_acc = int(self._df["accepted"].to_numpy(dtype=bool).sum())
        shown_rows = self._model.df_rows()
        subset = len(shown_rows)
        shown = self.filter_box.currentText() if hasattr(self, "filter_box") else _FILTER_ALL
        lo = page * self._model.page_size() + 1 if subset else 0
        # "0 of 0", never "0–-1 of 0": with an empty subset there is no range to print, and
        # ``lo + n - 1`` with n = 0 printed a negative upper bound as the first clause of the
        # line this window states its counts on (measured: reject everything, then Show =
        # Accepted only).
        span = f"{lo}–{lo + n - 1}" if n else "0"
        # A label filter keeps its token's case: "Labelled: molecule=ApoF" is not "apof".
        shown_name = (shown.lower() if not shown.startswith(LABEL_FILTER_PREFIX)
                      else LABEL_FILTER_PREFIX.lower() + shown[len(LABEL_FILTER_PREFIX):])
        bits = [f"{span} of {subset} ({shown_name})",
                f"{n_acc} of {n_all} accepted"]
        # HOW MANY TILES THE NEXT A / R / DIGIT WOULD WRITE. The audit assumed this count was
        # already here ("the selection count is already in the status line so the feedback
        # path exists") -- it was not; the only clause that said "selected" was the off-page
        # note, which appears when a selection is LOST. So the one number that tells a user
        # a batch gesture landed at all was missing from the one line that is always on
        # screen, and Escape had nothing to visibly undo. Printed only when there IS a
        # selection: "0 selected" on a line this long is noise, and its absence is exactly
        # what deselect-all is meant to produce.
        n_sel = len(self._selected_rows())
        self._shown_sel_count = n_sel
        if n_sel:
            bits.append(f"{n_sel} selected")
        # A FIXED page size can hold more tiles than one screenful, and macOS overlay
        # scrollbars are invisible until scrolled — so a page of 96 over a window that
        # shows 40 read as "the count is wrong" (owner note #12). Say the page scrolls,
        # in the same line that states the count the visible tiles fail to match.
        # From GEOMETRY, not the scrollbar: this line is rebuilt in the same event-loop
        # turn as the page-size change, before the view has relaid out and the
        # scrollbar knows its new maximum.
        if self._page_overflows():
            bits.append("page taller than the window — scroll for the rest")
        bits.extend(self._subset_notes(subset))
        # WHICH TRACE, ALWAYS. Not only when it is the unusual one: a picture is not
        # self-describing, and "these are raw events" is precisely the thing the owner's own
        # notebook label got wrong.
        bits.append(f"drawn from: {self._trace_name.lower()}")
        # THE REFUSAL, REBUILT WITH THE LINE. Placed before the two how-to hints so it is not
        # the last thing on a long line, and read from state so every rebuild -- this
        # gesture's, the surface refresher's, a resize's -- puts it back. See refusal_note.
        note = self.refusal_note()
        if note:
            bits.append(note)
        # THE SELECTION THAT COULD NOT BE KEPT, in the same place and by the same rule: read
        # from state, re-derived from the live subset, rebuilt with the line. A curation
        # gesture keeps its selection (see _set_rows_accepted); when the filter really does
        # take tiles off the page, this is where that is admitted.
        offpage = self._offpage_note()
        if offpage:
            bits.append(offpage)
        # THE EVENT ANOTHER SURFACE ASKED FOR AND THIS SUBSET EXCLUDES -- same rule, same
        # place: state, re-derived, rebuilt with the line. See show_event.
        missed = self._missed_note()
        if missed:
            bits.append(missed)
        # THE BADGE THE EVENTS TABLE CARRIES, carried here too. See set_stale_columns.
        stale = self._stale_line()
        if stale:
            bits.append(stale)
        fraction_note = self._fraction_note()
        if fraction_note:
            bits.append(fraction_note)
        # THE DIGIT THAT WROTE NOTHING. See apply_hotkey.
        if self._hotkey_note:
            bits.append(self._hotkey_note)
        # THE GESTURES, ALL OF THEM. The line used to advertise click / R / 1–9 and
        # "double-click opens it in the trace" and never once mentioned the right-click
        # menu, which is where "go to it on the trace" and the per-tile accept/reject
        # actually live — a whole road out of this window that nothing on screen named.
        # The digits named are the ones the palette ARMS (see _label_hint).
        gestures, palette = self._label_hint()
        bits.append(self._batch_hint())
        bits.append(gestures)
        if palette:
            bits.append(palette)
        bits.append("double-click opens it in the Event Inspector")
        bits.append("right-click for the inspector, the trace, or accept/reject")
        self.status.setText("   ·   ".join(bits))
        # The bulk buttons act on ``df_rows()``, i.e. on exactly the tiles the current
        # controls produce -- so under a representative sample they curate the SAMPLE and say
        # so, rather than reaching the 387 events behind it that nobody has looked at. That is
        # why the count here is ``subset`` (what is drawn) and not ``subset_size()`` (what it
        # was drawn from); the status line above prints both, under different names.
        #
        # THE ACCEPT BUTTON PROMISES WHAT IT WILL DO, not how many tiles it can see. Measured
        # in the wired app on the owner's -600 mV recording, it read "Accept all 317" and
        # accepted 313: the four rows the detector could not measure stay rejected, and the
        # button was the last surface still claiming otherwise. The disclosure on the status
        # line only exists AFTER the click; this is the one the user reads BEFORE it.
        # Rejection is never blocked, so "Reject all N" was already true and is left alone.
        self._sync_accept_all_button(shown_rows, subset)
        self.reject_all_btn.setText(f"Reject all {subset}")

    def _sync_accept_all_button(self, shown_rows: list[int], subset: int) -> None:
        """Put the accept button's promise on the button: ``Accept all N`` when it really
        would accept them all, ``Accept K of N`` when the detector's own rejections make
        K < N -- with the tooltip that explains the shortfall, set here beside the text so
        the two cannot disagree.

        Both forms name ``subset`` -- the tiles the current filter draws -- so the button
        still says WHICH tiles it acts on; the difference is that it no longer says it will
        accept ones it is about to refuse. A frame with no ``flag`` column would accept
        nothing at all (:meth:`_set_rows_accepted` fails closed), and ``Accept 0 of N`` is
        that promise stated rather than discovered by clicking.

        COUNTED PER DISTINCT FLAG STRING, not per row. ``_sync_controls`` runs on every page
        turn, filter change, curation click and resize, and asking :func:`_derived_accepted`
        once per row cost 7 ms on a 1 196-event detection and 1.1 SECONDS on 100 000 events
        -- i.e. this label would have been the slowest thing in the window. A flag column
        holds a handful of distinct strings whatever its length, so ``value_counts`` moves
        the per-row work into pandas' hash table and leaves the RULE where it belongs: still
        exactly one call of :func:`_derived_accepted` per distinct value, no second
        implementation of it, and no cache to invalidate when a mask edit rewrites a flag.
        """
        if self._df is None or not shown_rows:
            self.accept_all_btn.setToolTip(_ACCEPT_ALL_TIP)
            self.accept_all_btn.setText(f"Accept all {subset}")
            return
        if "flag" not in self._df.columns:
            acceptable = 0
        else:
            flags = self._df["flag"].to_numpy()
            if len(shown_rows) < len(flags):
                flags = flags[np.asarray(shown_rows, dtype=np.intp)]
            acceptable = int(sum(int(n) for value, n
                                 in pd.Series(flags).value_counts(dropna=False).items()
                                 if _derived_accepted(value)))
        if acceptable == subset:
            self.accept_all_btn.setToolTip(_ACCEPT_ALL_TIP)
            self.accept_all_btn.setText(f"Accept all {subset}")
            return
        self.accept_all_btn.setToolTip(
            f"{subset - acceptable} of the {subset} tiles shown are flagged by the detector "
            "as not measurable (clipped by the boundary, over the dwell limit, measured from "
            f"a non-finite sample). They stay rejected, so this accepts {acceptable}.\n\n"
            + _ACCEPT_ALL_TIP)
        self.accept_all_btn.setText(f"Accept {acceptable} of {subset}")

    def _offpage_note(self) -> str:
        """"…3 selected tiles left this page — filtered out of Accepted only — no longer
        selected", or "" when they are all reachable again.

        RE-DERIVED, NEVER REMEMBERED. The ids are remembered; whether they are still off
        the page is asked of the model every time the status line is built, so the sentence
        vanishes by itself when the filter is put back to "All" (or the page is turned to
        where they went) instead of standing there counting a state that has been undone.
        A gesture that keeps every tile leaves the tuple empty and prints nothing.

        IT SAYS WHERE EACH TILE WENT, AND CLAIMS NOTHING ELSE. The sentence used to end
        "still curated, but no longer selected" -- written for this window's own R under
        "Accepted only", where the tiles leave BECAUSE they were curated. The same record is
        fed by the inbound reset (:meth:`_refilter_keeping_selection`), where a tile can
        leave for a reason nobody curated: the events table rejects an event EARLIER in
        the subset and a swept tile shifts onto the previous page; under the representative
        sample any curation elsewhere re-cuts the dwell ladder and a swept, still-accepted
        tile falls off it. Measured on the owner's -1000 mV recording: "1 selected tile
        left this page — still curated" about event 38, still accepted, untouched by
        anyone. The reason is now asked of the model per tile, in the model's own terms --
        the control that excludes it (:meth:`_excluding_subset`, the words the missed-event
        note uses) or the page it is on now -- so the line says only what the model knows.
        """
        if not self._offpage_ids or self._df is None:
            return ""
        reasons: dict[str, int] = {}
        for ev in self._offpage_ids:
            row = self._row_for_id(ev)
            if row is None:
                why = "no longer in the events table"
            elif self._model.index_for_df_row(row).isValid():
                continue                                    # reachable again: not gone
            else:
                page = self._model.page_of_df_row(row)
                if page is not None:
                    why = f"now on page {page + 1}"
                else:
                    subset, _reach = self._excluding_subset(row)
                    why = ("no longer in the representative sample"
                           if subset == "the representative sample"
                           else f"filtered out of {subset}")
            reasons[why] = reasons.get(why, 0) + 1
        gone = sum(reasons.values())
        if not gone:
            return ""
        where = (next(iter(reasons)) if len(reasons) == 1
                 else ", ".join(f"{n} {why}" for why, n in reasons.items()))
        return (f"{gone} selected {'tile' if gone == 1 else 'tiles'} left this page — "
                f"{where} — no longer selected")

    def _subset_notes(self, shown: int) -> list[str]:
        """What the subpopulation filter and the representative sample did to the subset."""
        out: list[str] = []
        if self._subpops is None:
            return out
        cid = self._model.subpop()
        out.append("every subpopulation" if cid is None
                   else (f"subpop {cid} only" if cid >= 0 else "not in the report only"))
        if self._model.representative():
            total = self._model.subset_size()
            note = f"representative sample: {shown} of {total}, spaced by dwell"
            missed = self._model.no_dwell_count()
            if missed:
                # SAID, never silently dropped. A dwell ladder has no rung for an
                # unmeasurable dwell, and the events that have none are exactly the ones
                # somebody needs to look at.
                note += f" ({missed} with no dwell left out — untick to reach them)"
            out.append(note)
        if self._subpop_source:
            out.append(f"from: {self._subpop_source}")
        return out
