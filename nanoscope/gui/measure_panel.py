"""The manual event-measurement panel: cursor tool + live stats + a saveable table.

This is the GUI face of :mod:`nanoscope.analysis.measure` (the pure compute core,
Phase 1) driven by the :class:`~nanoscope.gui.trace_view.TraceView` Measure region
(the LinearRegionItem cursor tool, Phase 2). The workflow it wires:

1. **Measure event (cursors)** toggles the movable region on the current-channel
   pane (``TraceView.set_measure_active``). The two edges ARE the boundary cursors;
   dragging either edge (or the body) moves the selection.
2. Every region move fires ``TraceView.sigMeasureChanged`` → the panel pulls the
   calibrated channel + fs + units from :class:`~nanoscope.core.session.Session`
   and recomputes :func:`~nanoscope.analysis.measure.measure_selection` LIVE,
   filling a rich readout (dwell, amplitude, dI/I0, moments, skew, KURTOSIS,
   area/ECD, SNR, …).
3. The **amplitude method** combo (median/mean/min/max/mode) recomputes live.
4. **Add to list** (or ``Ctrl+M``) stamps the current measurement — a running
   ``measurement_id``, the recording + channel names AND the durable identity pair
   ``doc_key``/``channel_index`` — into a :class:`_MeasurementsModel` over
   :data:`~nanoscope.analysis.measure.MEASUREMENT_SCHEMA`; rows can be labelled,
   sorted, multi-selected, copied as TSV and removed. **Export CSV…** writes the
   whole table via :func:`write_measurements_with_basis` — ``MEASUREMENT_SCHEMA``'s
   32 columns exactly as
   :func:`~nanoscope.analysis.measure.write_measurements_csv` writes them, plus the
   additive ``signal_basis`` stamp as a 33rd, last column.

LIVE IS BOUNDED; SAVED IS EXACT (why step 2 and step 4 do not compute the same thing)
--------------------------------------------------------------------------------------
``measure_selection`` is O(n) over the selection, and step 2 runs on the GUI thread. Its
measured cost is ~8 ms at 200 k samples, ~80 ms at 2 M and ~1.1 s at 15 M — and this app's
recordings ARE 15 M samples. Wired naively (one synchronous full recompute per
``sigRegionChanged``, which fires per drag FRAME) the readout freezes the whole app for
about a second per frame the moment the selection gets big, and a very large manually
resized selection can otherwise become a trap rather than a shortcut. Two mechanisms fix that, and one
rule keeps it honest:

* **Coalesce.** A drag frame does not recompute; it (re)starts a single-shot
  ``_RECOMPUTE_DEBOUNCE_MS`` timer, so a continuous drag costs one recompute per idle
  moment rather than one per frame, and a superseded pending recompute is simply dropped.
  ``TraceView.sigMeasureFinished`` (mouse released, or the region placed programmatically)
  cancels the timer and settles immediately, so releasing the mouse never feels laggy.
* **Bound.** The live recompute always passes ``max_samples=_LIVE_MAX_SAMPLES``, so above
  that a preview is computed over a strided subsample. This applies to the settle path
  too, deliberately: programmatic placement settles immediately, so the eager path is
  exactly where the unbounded cost would still bite.
* **THE RULE: nothing approximate is ever saved.** :meth:`add_current` re-measures the
  selection with ``max_samples=None`` before stamping a row, so a saved measurement is
  computed over EVERY sample of its span whatever the readout happened to be showing. The
  live dict (:meth:`current_measurement`) may be a preview and says so; the stored row
  cannot be. *Mark selection as missed event* needs nothing here — its owner
  (``AnalysisPanel.add_current_selection_as_event``) reads the cursor bounds off the shared
  TraceView and characterizes the raw samples itself, so it never sees a preview at all.

Because a preview can be visibly incomplete, it is LABELLED: an "approx" note under the
readout gives the sampling ratio and the statistics that were blanked, and it clears the
instant an exact value is in hand (including right after *Add to list*, which refreshes the
readout with the exact numbers it just stored).

WHICH SIGNAL IS BEING MEASURED — RAW OR PROCESSED (the *measure on* row)
-------------------------------------------------------------------------
This tool used to measure ``session.calibrated(doc, ch).data`` — the raw channel —
unconditionally, INCLUDING while the orange processed curve drawn from the very same
Preprocess form sat on the plot in front of the user (measured, audit 2026-09-02). The
fix is not "measure whatever is drawn": a number whose basis depends on a checkbox two
tabs away is as unreadable as a number that ignores it. So the basis is an explicit
choice, and three things follow from it:

* the *measure on* combo picks ``Raw signal`` (the default, and byte-for-byte the
  behaviour that shipped) or ``Processed signal`` —
  :func:`~nanoscope.analysis.conditioning.conditioned_trace`, the same function, and the
  same bytes, that draws the preview curve. The Processed item is GREYED, never removed,
  when no chain is configured, so the row never changes height;
* the basis line under the combo says which signal the readout describes and names the
  chain, and it is always two lines tall whatever it says;
* every saved row stamps it (``signal_basis``), so a pooled table of 30 pA rows some of
  which were low-passed and normalized stays legible. The chain parameters themselves
  live with the export (below) — the row records WHICH signal, the sidecar records HOW.

A chosen basis that becomes unavailable (the chain is switched off, a cutoff goes above
Nyquist) blanks the readout and prints the reason in the refusal line. It does NOT
quietly fall back to the other signal — the whole point of the control is that the
number's basis is never in doubt.

The chain parameters arrive from the owner through :meth:`set_conditioning`, like the
target and the detection-present flag: this panel talks to Session and its TraceView, and
never to a sibling panel.

AND WHEN NOBODY PUSHES ONE, THE PANEL FINDS THE CHAIN ITSELF. That door had no caller for
the whole of its first build, which made the combo above a control that toggled nothing
(audit, 2026-09-02), and even with a caller it fires on the owner's refreshes rather than
on the keystroke that edits the form. So :meth:`_stored_detection_chain` recovers the
chain a detection on THIS channel ran under — ``AnalysisResult.params`` out of the result
store Session owns, fed to the same pure ``conditioned_trace`` — whenever the pushed form
has nothing enabled. It is not a second definition of the chain and it is not a guess: it
is the array the events in the table beside this one were measured on, and the basis line
says so in those words rather than claiming the Preprocess form.

GETTING THE PROCESSED SAMPLES OUT (:meth:`export_processed_trace`)
------------------------------------------------------------------
The same one-way door, at the other end: before this, nothing in ``gui/`` wrote a signal
to disk at all. The export is CSV (``time_s`` + the conditioned current) plus a
``.meta.json`` sidecar holding the exact parameters, the chain in words, the rate, the
units, the sample range and the recording's clock origin — enough to rebuild the array.
The cursor selection when there is one, the whole recording otherwise, and the file says
which.

The door is the **Export processed trace…** button under the table, in this panel,
because this is where the *measure on* choice is made. (It shipped for one round with no
button, no menu item and no shortcut anywhere in the app — a feature only a test could
open. MainWindow's Analysis menu is the second door; this one is the near one.) The
button never greys: :meth:`~MeasurePanel.processed_export_gate` returns the SENTENCE that
says what is missing, the tooltip carries it, and a click with nothing to export opens it.

The current column is named after the unit the samples ACTUALLY have
(:func:`processed_units_label`), which is not always the channel's: ``normalize`` in the
chain leaves z-scores, and a column headed ``current_pA`` over z-scores is the one kind
of thing an export must never write.

A SAVED ROW SAYS WHAT IT WAS MEASURED UNDER (calibration + clock)
-----------------------------------------------------------------
*Add to list* stamps each row with the provenance that gives its numbers meaning: the
channel's ``units`` label, the calibration fingerprint (``cal_gain``/``cal_offset`` —
the same ``ChannelCalibration.fingerprint()`` detections stamp), and the recording's
instrument-clock origin ``t0_s`` (``t_start_s`` is on the RECORDING clock while the
trace axis is not — see :mod:`nanoscope.core.clock`; the ``t_start_s`` header wears the
same ``(t₀=0)`` badge and tooltip the event table's does). When a later calibration
edit supersedes a row's stamp the row is MARKED — amber cells, a ⚠ on its row header,
a tooltip saying to re-measure — and never silently rescaled: the saved numbers are the
record of what was measured, exactly as the analysis panel treats a stale detection.
Rows saved before the stamp columns existed read as *unknown*, not as matching and not
as stale. The stamps ride the CSV export, so pooled pA/nA rows stay distinguishable
downstream.

``t0_s`` IS THE EXPERIMENT'S CLOCK, NOT THE SCREEN'S — the two questions this table used
to answer with one number:

* the row stamps :func:`~nanoscope.core.clock.instrument_origin_s`, the experimental
  time of its recording's first sample. On a SEGMENT that is the parent recording's
  origin (1158.99 s on the owner's quick cut), not the 0 this app rebased the segment's
  own axis to — stamping the 0 put a parent row and a row from a segment *of that
  parent* 19 minutes apart under a column that calls itself the instrument-clock origin;
* the ``t_start_s`` header's clock note is about the axis the trace is DRAWN through,
  which additionally carries the layer's alignment nudge. So the header says which of
  the two each sentence lands on, and names the nudge when there is one, instead of
  promising an arithmetic that only holds while the two coincide.

A ROW'S MARK HAS THREE STATES, not two: checked-and-current (unmarked), superseded (⚠,
amber, re-measure), and *no verdict* — the row's recording is not open, or it predates
the stamps, so nothing can be compared. The third one wears a ``?`` and says so, because
"nobody can check this" drawn identically to "checked and current" is the same silence
the ⚠ exists to break. A row already marked ⚠ when its recording is closed KEEPS the
mark and its tooltip explains that the check can no longer be repeated: the numbers did
not change when the document closed, so neither does what was true about them.

RETURNING TO A SAVED MEASUREMENT (the reason a row is not a dead end)
--------------------------------------------------------------------
A hand measurement is worthless if you cannot get back to it: zoom out to check the
whole recording and the excursion you measured is one pixel among millions. So a saved
row is a LIVE handle on its span:

* selecting a row makes it *current* — it is highlighted on the trace;
* **double-clicking it** (or *Go to measurement*) re-arms the cursors on that exact
  ``[start_idx, end_idx)`` and frames it with symmetric padding via the shared
  :meth:`~nanoscope.gui.trace_view.TraceView.zoom_to_span` — the SAME padded-zoom the
  detected-event table dives in with;
* if the row belongs to a DIFFERENT open document/channel, the panel emits
  :attr:`navigateRequested` and lets its owner switch the app selection first (it never
  reaches into the Files dock itself — same routing pattern as
  :attr:`markMissedRequested`);
* if the row's recording is not open at all, the status line says so.

Identity is the row's ``doc_key`` (``core.segment.persistent_key``) + ``channel_index``,
not the display name, so navigation is unambiguous when a segment and its parent share a
name. Rows saved before those columns existed degrade to display-name matching.

Every saved measurement of the CURRENT channel is also drawn on the trace as a subtle
span (toggleable), with the current row marked distinctly, so "have I already measured
this stretch?" is answerable at a glance.

Like every other inspector panel this talks to Session (for data) and to the ONE
shared TraceView it was handed, and to NOTHING else in ``gui/``. Its owner
(:class:`~nanoscope.gui.analysis_panel.AnalysisPanel`) pushes the current target
in via :meth:`set_selection` and the last-selected auto-event via
:meth:`set_selected_event`; the panel never reaches back into another panel.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import (
    QAbstractTableModel,
    QEventLoop,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)

from ..analysis.conditioning import (
    BASIS_PROCESSED,
    BASIS_RAW,
    BASIS_UNKNOWN,
    SIGNAL_BASIS_COLUMN,
    SIGNAL_BASIS_TIP,
    basis_label,
    chain_description,
    conditioned_trace,
    conditioning_enabled,
    ensure_signal_basis_column,
    processed_trace_export_paths,
    write_processed_trace,
)
from ..analysis.events import frac_depth_note
from ..analysis.measure import (
    AMPLITUDE_METHODS,
    CAL_STAMP_STALE,
    CAL_STAMP_UNKNOWN,
    LIVE_PREVIEW_MAX_SAMPLES,
    MEASUREMENT_DIAGNOSTIC_KEYS,
    UNKNOWN_CHANNEL_INDEX,
    NonFiniteSelection,
    append_measurement,
    calibration_stamp_state,
    coerce_measurements,
    empty_measurements_df,
    measure_selection,
)
from ..core import clock
from ..core.analysis import OverlayHint, OverlayKind
from ..core.atomic_write import staged_path
from ..core.document import Document
from ..core.segment import persistent_key
from ..core.session import Session
from ..core.sources import DETECTION_MODULE_IDS
from ..core.types import ChannelKind
from .export_paths import ask_export_path
from .trace_view import MEASURE_MAX_VIEW_FRACTION, TraceView

__all__ = ["MeasurePanel"]

# The free-text annotation column — the ONLY editable cell in the saved table, and the one
# column whose double-click means "rename", not "take me there".
_LABEL_COL = "label"

# Saved measurements drawn on the trace. Muted violet for the whole set (distinct from the
# detector's orange spans and from the blue measure region), a brighter one for the row the
# user currently has selected.
_MARK_COLOR = "#9E7BB5"
_CURRENT_MARK_COLOR = "#C77DFF"

# Raw (uncast, comparable) cell value, used as the proxy's sort role. Sorting on the
# DISPLAY string would order dwell 1e-05 before 9e-06 and "10" before "2".
_SORT_ROLE = Qt.UserRole + 1

# Drag coalescing. A LinearRegionItem drag emits sigRegionChanged per mouse-move frame
# (~60/s); this is how long the edges must sit still before the readout recomputes. Long
# enough that a continuous drag collapses to a handful of recomputes instead of one per
# frame, short enough that pausing mid-drag still reads live. The mouse-release path does
# not wait it out at all (sigMeasureFinished settles at once), so this only ever delays the
# readout DURING motion.
_RECOMPUTE_DEBOUNCE_MS = 60

# Cost bound for the LIVE readout: above this the preview strides. See
# analysis.measure.LIVE_PREVIEW_MAX_SAMPLES for the measured latency it is chosen from.
# Saved rows ignore it entirely — add_current re-measures with max_samples=None.
_LIVE_MAX_SAMPLES = LIVE_PREVIEW_MAX_SAMPLES

# Above this many samples an EXACT (re)measurement is slow enough to deserve a wait cursor:
# ~80 ms at 2 M on the reference machine, growing linearly to ~1.1 s at 15 M.
_BUSY_CURSOR_SAMPLES = 2_000_000

# The approximate-preview note. Amber, not red: a preview is a normal working state, not an
# error — the numbers are good, they are just not the final ones.
_APPROX_COLOR = "#C98A20"

# The refusal note. Red, unlike the amber preview note: this one IS an error state — the
# readout is blank because the data under the cursors is not measurable, and until this
# line existed the panel simply showed "—" with no way for the user to find out why.
_REFUSAL_COLOR = "#D05A50"

# Rows measured under a calibration that has since changed. The SAME amber the event
# table paints its stale columns in (gui/panels.py), so "this number was measured under a
# superseded configuration" looks the same on both tables — two surfaces, one vocabulary.
_STALE_ROW_COLOR = "#ffd479"

# Saved-table columns that are an INSTANT in seconds and so are on a clock — the manual
# table's counterpart of gui/panels.py's _CLOCK_COLUMNS, and the same rule: `dwell_s` is
# a LENGTH of time, on no clock, and annotating it would be noise. `t0_s` is the clock
# STAMP itself, not an instant on one.
_CLOCK_COLUMNS = frozenset({"t_start_s"})

# What the stale-row marking means and what to do about it. One sentence per rule of the
# analysis panel's stale banner: the numbers are the record of what was measured — never
# rescaled — and only a re-measurement produces numbers under the current calibration.
_STALE_ROW_TIP = (
    "Measured under a calibration that has since changed. The row's units / cal_gain / "
    "cal_offset columns hold the gain, offset and units it was measured under; the "
    "Calibration panel holds the current ones. Saved numbers are the record of what was "
    "measured and are never rescaled — re-measure the span (double-click the row to "
    "return to it, then Add to list) for a row under the current calibration."
)

# THE THREE STATES A SAVED ROW'S CALIBRATION STAMP CAN BE IN, and why there are three
# rather than two. `calibration_stamp_state` has always returned three answers — match,
# stale, unknown — and the panel's own rule is that a stamp which CANNOT be compared
# "reads as unknown, never as matches". The model rendered only two, so "nobody can check
# this" was drawn exactly like "checked, and current". Closing a recording is where that
# bit: the ⚠ and the amber and the re-measure tooltip all vanished a second after the user
# was shown them, with nothing saying the app had merely stopped being able to look.
_ROW_STALE = "stale"          # checked against the live calibration, and superseded
_ROW_LAPSED = "lapsed"        # WAS superseded when last checked; no longer checkable
_ROW_UNCHECKED = "unchecked"  # never checkable: recording closed, or no stamp at all

# A mark that outlived the ability to refresh it. Kept, and honest about why: the row's
# numbers did not change when the recording closed, so neither does its warning — but the
# app must not imply it is still watching.
_LAPSED_ROW_TIP = (
    "Measured under a calibration that had already changed when this row was last "
    "checked — and its recording is no longer open, so the check cannot be repeated. The "
    "mark is kept because the row is: the saved numbers are unchanged, and so is what "
    "was true about them. Reopen the recording to have it checked again, and re-measure "
    "the span for a row under the current calibration."
)

# Never checked, and not claiming to have been. Deliberately NOT amber and NOT a ⚠: this
# is not an alarm about the numbers, it is the absence of a verdict, and painting it like
# a warning would be the opposite error to the one it fixes.
_UNCHECKED_ROW_TIP = (
    "This row's calibration stamp has not been compared with anything: its recording is "
    "not open (or the row was saved before the stamp columns existed), so the app cannot "
    "tell whether the calibration it was measured under is still the current one. It is "
    "NOT known to match. Reopen the recording to have it checked; the units / cal_gain / "
    "cal_offset stamps travel in the CSV export either way."
)

# Header tooltips for the provenance stamp columns, so the table explains its own schema
# where the user is looking. `t_start_s` gets its (clock-dependent) tooltip in
# `headerData` instead; these three are static.
_STAMP_COLUMN_TIPS = {
    "units": "The channel's units label when this row was measured — rows from "
             "different recordings (or different calibrations) can legitimately mix "
             "pA and nA in one table, and this column is what tells them apart.",
    "cal_gain": "Calibration gain in force when this row was measured. NaN = unknown "
                "(a row saved before this column existed). Together with cal_offset "
                "and units it is the row's calibration stamp; a row whose stamp no "
                "longer matches the live calibration is marked ⚠.",
    "cal_offset": "Calibration offset in force when this row was measured. NaN = "
                  "unknown (a row saved before this column existed).",
    SIGNAL_BASIS_COLUMN: SIGNAL_BASIS_TIP,
    "t0_s": "Instrument-clock origin of the row's recording — the experimental time of "
            "its first sample: t_start_s + t0_s is this row's start ON THE INSTRUMENT "
            "CLOCK. For a segment that is the parent recording's clock (the origin the "
            "Files dock and the segment's notes show), not the segment's own axis, which "
            "this app rebased to 0. A layer alignment nudge is a display shift and is "
            "never included. NaN = unknown (a row saved before this column existed, or a "
            "recording with no time base).",
}


def _cast_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``df`` to the MEASUREMENT_SCHEMA columns/dtypes, KEEPING the basis stamp.

    Mostly the pure :func:`~nanoscope.analysis.measure.coerce_measurements`: a missing
    column is filled with its dtype-appropriate default (NOT ``pd.NA``, which an
    int64/float64 cast rejects), extra columns are dropped, and an empty/None frame yields
    the canonical empty table. This is what makes a project saved before ``doc_key`` /
    ``channel_index`` / ``label`` existed still restore.

    THE ONE COLUMN THAT SURVIVES THE DROP is
    :data:`~nanoscope.analysis.conditioning.SIGNAL_BASIS_COLUMN`, re-attached here after
    the coercion. It is an ADDITIVE column declared outside ``MEASUREMENT_SCHEMA`` (the
    ``scoring.ADDED_COLUMN`` pattern), so ``coerce_measurements`` — which knows only the
    schema — treats it as an extra and drops it; letting that happen would erase, on the
    next project restore, the one fact that says whether a row's numbers describe the raw
    channel or the processed trace. A frame that never had the column gets ``''``
    ("not recorded"), which is exactly what an old project must load as: those rows were
    saved by a build where no other basis was reachable, but nobody wrote that down, and
    a stamp the app invents is not a record.
    """
    out = coerce_measurements(df)
    if (df is not None and len(df) and len(out) == len(df)
            and SIGNAL_BASIS_COLUMN in getattr(df, "columns", ())):
        out[SIGNAL_BASIS_COLUMN] = pd.Series(
            [BASIS_UNKNOWN if pd.isna(v) else str(v)
             for v in df[SIGNAL_BASIS_COLUMN].to_numpy()],
            index=out.index, dtype="string")
    return ensure_signal_basis_column(out)


def write_measurements_with_basis(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the measurement table to CSV: the schema columns, THEN the basis stamp.

    :func:`~nanoscope.analysis.measure.write_measurements_csv` writes exactly
    ``MEASUREMENT_SCHEMA`` — 32 columns, in their frozen order — because that is all it
    knows; ``coerce_measurements`` drops everything else as an extra. That is right for
    the schema and wrong for the export: the basis stamp is reachable in the GUI, rides
    the project (parquet keeps the frame verbatim) and survives a restore
    (:func:`_cast_measurements`), and then the ONE artefact the user takes to their
    analysis — the CSV — dropped it, leaving a pooled table of 30 pA rows in which
    "raw channel" and "processed trace" are again indistinguishable. That ambiguity is
    the whole reason the column exists.

    So the CSV is written here, from :func:`_cast_measurements`: the 32 schema columns
    unchanged, in the same order, with the same values and the same dtypes, and
    :data:`~nanoscope.analysis.conditioning.SIGNAL_BASIS_COLUMN` APPENDED as a 33rd, LAST
    column — the additive out-of-schema pattern ``scoring.ADDED_COLUMN`` already rides the
    event export with. Nothing a reader indexed by name moved, and a reader that indexed
    by position still finds the first 32 fields where they were. Rows written before the
    column existed export as ``''`` = "not recorded", which is not a claim that they were
    raw.

    Staged, like every other write over user data (``core.atomic_write``): a truncated
    half-file with a plausible header is worse than no file, because it opens.
    """
    p = Path(path)
    with staged_path(p) as staged:
        _cast_measurements(df).to_csv(staged, index=False)
    return p


#: The Preprocess flag for the one stage that takes the channel's units WITH it.
#: Notch, medians, the drift detrend, running mean, SavGol and the low-pass are all
#: linear in the samples and leave a picoamp a picoamp; ``normalize`` rescales the trace
#: (z-score / 0..1), after which a 100 pA open pore reads ~0.29 and there are no units
#: left to name.
_NORMALIZE_FLAG = "normalize_enabled"

#: What :func:`processed_units_label` says instead of a unit when the numbers no longer
#: have one. Spelled with no comma on purpose: ``write_processed_trace`` builds the CSV
#: header as ``"current_" + units.replace(" ", "_")``, so a comma anywhere in this string
#: would split a two-column header into three and the file would not parse.
_DIMENSIONLESS = "normalized (dimensionless)"


def processed_units_label(units: str | None, params: Mapping[str, Any] | None) -> str:
    """The unit the CONDITIONED samples actually carry — the label the export may write.

    ``write_processed_trace`` names its second column after this string
    (``current_pA``) and copies it into the sidecar's ``units``. Handed the channel's
    label verbatim it therefore writes ``current_pA`` over a normalized trace, which is
    false in the one way an export cannot be: the numbers are z-scores, a reader who
    pools them with real picoamps gets silent nonsense, and nothing in the file says so.
    (:mod:`~nanoscope.analysis.conditioning` anticipates exactly this — "after that stage
    the numbers are no longer picoamps and the export must not pretend otherwise".)

    So: with ``normalize`` in the chain the label carries the word ``dimensionless``,
    keeping the channel's own label in front of it (``"pA normalized (dimensionless)"``)
    because what the samples were BEFORE the stage is still worth recording; with no
    declared units it degrades to ``"normalized (dimensionless)"`` rather than inventing
    one. Without ``normalize`` the channel's label is returned unchanged — every other
    stage is linear, so a filtered picoamp is still a picoamp, and a raw-units export
    written by this build is byte-identical to one written before it.

    Commas are stripped and whitespace collapsed for the header reason in
    :data:`_DIMENSIONLESS`: units are free text on the Calibration tab, and a label typed
    as ``"pA, corrected"`` would otherwise write a three-field header over two-column
    rows.
    """
    label = " ".join(str(units or "").replace(",", " ").split())
    if not (params or {}).get(_NORMALIZE_FLAG):
        return label
    return f"{label} {_DIMENSIONLESS}" if label else _DIMENSIONLESS



# ---------------------------------------------------------------------------
# the manual-measurement table model
# ---------------------------------------------------------------------------
class _MeasurementsModel(QAbstractTableModel):
    """A :class:`QAbstractTableModel` over a MEASUREMENT_SCHEMA frame.

    The DataFrame is the source of truth (exportable verbatim via
    :func:`write_measurements_with_basis`). Rows are appended through
    :func:`append_measurement` so the on-screen table and the CSV are the same
    schema-typed object; removals renumber the row index but never the
    ``measurement_id`` (that stays the running session id the panel stamped).

    Read-only EXCEPT the ``label`` column, which is free text the user types to say why a
    row exists ("the weird double-dip", "clean square for the figure"). It round-trips
    through the CSV export and the project like every other column.

    Beyond ``Qt.DisplayRole`` (a formatted string) each cell also serves ``_SORT_ROLE``:
    the RAW value, so a sorted view orders numbers numerically. Appends use
    ``beginInsertRows`` rather than a full model reset so the user's row selection — and
    therefore the on-trace marker — survives measuring the next event.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # The basis stamp rides every frame this model ever holds, empty one included, so
        # the column exists before the first row does: a table whose columns appear only
        # after the first PROCESSED measurement would shift under the user's pointer, and
        # `column_index` would answer -1 to anything that asked early.
        self._df: pd.DataFrame = ensure_signal_basis_column(empty_measurements_df())
        self._columns: list[str] = list(self._df.columns)
        #: ``measurement_id``s of rows whose calibration stamp no longer matches the live
        #: calibration of their channel — pushed in by the panel (the model cannot see
        #: Session), keyed on the stable id so the marking survives sorting and removals.
        #: Empty is the normal case. Same pushed-state shape as the event-table model's
        #: ``set_stale_columns``, but per ROW: a manual row is one measurement under one
        #: calibration, so staleness is a property of the row, not of a column.
        self._stale_ids: frozenset[int] = frozenset()
        self._stale_tip: str = _STALE_ROW_TIP
        #: ``measurement_id`` -> one of the ``_ROW_*`` states, for rows that are NOT
        #: "checked and current". Absent means checked and current — the only state that
        #: earns a blank row header. Three states, not two: see the ``_ROW_*`` block.
        self._row_states: dict[int, str] = {}
        #: ``instrument_seconds - recording_seconds`` for the CURRENT recording, or None.
        #: `t_start_s` is on the recording clock while the trace axis is not, and the
        #: header is where that stops being a trap — the same fix, pushed the same way,
        #: as the event table's (gui/panels.py, core/clock.py). This table pools rows
        #: from many recordings, so the header note states the CONVENTION and each row's
        #: own `t0_s` column carries its recording's actual t₀.
        self._clock_offset_s: float | None = None
        #: The CURRENT recording's instrument-clock origin — what `add_current` stamps
        #: into `t0_s`, i.e. `_clock_offset_s` WITHOUT the layer alignment nudge (and
        #: the cut's real origin on a rebased segment, where the drawn offset is 0).
        #: Held separately because the header has to state both halves and they are not
        #: the same number: the drawn axis is `t0 + nudge`, a saved row is `t0`. Saying
        #: only one of them is how one tooltip came to give two conversions.
        self._clock_origin_s: float | None = None
        #: This layer's alignment nudge (the Files-dock Δt spin), in seconds. Pushed
        #: rather than derived as `offset - origin`: on a rebased segment those two
        #: differ by the CUT origin as well, and subtracting them would report a
        #: 1158.99 s "alignment shift" nobody made.
        self._clock_nudge_s: float = 0.0

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def set_stale_rows(self, ids, tooltip: str = _STALE_ROW_TIP) -> None:
        """Mark the rows with these ``measurement_id``s as measured under a superseded
        calibration (or clear with ``()``). The honest half of "never silently rescale":
        the numbers on screen stay exactly what was measured, and the table says which
        of them the current calibration no longer describes.

        The two-state door, kept because it is what "these rows are stale" reads like at
        a call site; it is :meth:`set_row_states` with every other row declared checked.
        """
        self.set_row_states({int(i): _ROW_STALE for i in ids}, stale_tip=tooltip)

    def set_row_states(self, states, *, stale_tip: str = _STALE_ROW_TIP) -> None:
        """Push the per-row calibration-stamp verdicts: ``{measurement_id: _ROW_*}``.

        A row absent from ``states`` is CHECKED AND CURRENT and wears nothing. Every
        other row wears its state's marker, including the two that mean "no verdict" —
        which is the whole point: an un-judgeable row must not render like a judged one.
        """
        wanted = {int(k): str(v) for k, v in dict(states).items()}
        if wanted == self._row_states and stale_tip == self._stale_tip:
            return
        self._row_states = wanted
        self._stale_ids = frozenset(k for k, v in wanted.items()
                                    if v in (_ROW_STALE, _ROW_LAPSED))
        self._stale_tip = str(stale_tip)
        self.beginResetModel()   # row headers + every cell of those rows repaint
        self.endResetModel()

    @property
    def stale_ids(self) -> frozenset[int]:
        """Rows wearing the "measured under a superseded calibration" ⚠ — including the
        ones whose recording has since closed, whose mark is kept rather than dropped."""
        return self._stale_ids

    @property
    def row_states(self) -> dict[int, str]:
        """``measurement_id`` -> ``_ROW_*`` for every row that is not checked-and-current."""
        return dict(self._row_states)

    def set_clock_offset(self, offset_s: float | None,
                         origin_s: float | None = None,
                         nudge_s: float = 0.0) -> None:
        """The current recording's clock story, so the time header can name it — the same
        door the event table's model has, plus the half a pooled table needs.

        ``offset_s`` is what the trace is DRAWN through (``t₀ + layer nudge``, a
        :class:`~nanoscope.core.clock.ClockOffset`); ``origin_s`` is the recording's
        instrument-clock origin, the number :meth:`MeasurePanel.add_current` stamps into
        the row's ``t0_s``. They differ by the alignment nudge, and on a rebased segment
        they differ by the whole cut origin — so the header states both rather than one
        sentence that is true of only one of them.

        Equality here is by CLOCK STATEMENT, not by number: ``ClockOffset`` compares its
        rebase origin too, so switching from a genuine t₀ = 0 file to a segment of a
        t₀ = 1154.99 s recording (offset 0.0 both times) still repaints the header.
        """
        if (offset_s == self._clock_offset_s and origin_s == self._clock_origin_s
                and float(nudge_s) == self._clock_nudge_s):
            return
        self._clock_offset_s = offset_s
        self._clock_origin_s = origin_s
        self._clock_nudge_s = float(nudge_s)
        self.headerDataChanged.emit(Qt.Horizontal, 0, max(len(self._columns) - 1, 0))

    def _nudge_note(self) -> str:
        """The sentence that reconciles the header's two clock statements, or ``""``.

        The trace is drawn at ``t₀ + nudge`` while a saved row stamps ``t₀``. When the
        Files-dock Δt spin has moved this layer the two are apart, and a reader following
        ``t_start_s + t0_s`` to "where it is on screen" lands exactly one nudge away. That
        is not a number to hide behind a general caveat: it is on screen, it is the
        difference, and it is persisted in the project.
        """
        nudge = float(self._clock_nudge_s)
        if self._clock_offset_s is None or clock.clocks_agree(nudge):
            return ""
        return (f"This layer is currently shifted {nudge:+.4f} s along the time axis for "
                f"alignment, so the trace DRAWS it that far from its instrument clock. "
                f"The nudge is a display alignment and is not stamped into any saved "
                f"row — add it as well to go from a row to a position on screen.")

    def _row_state(self, row: int) -> str:
        """The ``_ROW_*`` state of a VIEW-independent source row, or ``""`` when the row
        is checked and current (or carries no id to key on)."""
        if not self._row_states or not (0 <= row < len(self._df)):
            return ""
        col = self._columns.index("measurement_id") if "measurement_id" in self._columns else -1
        if col < 0:
            return ""
        value = self._df.iat[row, col]
        if pd.isna(value):
            return ""
        return self._row_states.get(int(value), "")

    def _row_is_stale(self, row: int) -> bool:
        """Wearing the superseded-calibration ⚠ (checked-and-stale, or a kept mark whose
        recording has since closed). The amber-and-⚠ predicate, not "has a state"."""
        return self._row_state(row) in (_ROW_STALE, _ROW_LAPSED)

    def _row_marker(self, row: int) -> str:
        """The non-colour glyph for a row's state — colour alone is the first thing lost
        to a colour-blind reader, a screenshot or a printout."""
        state = self._row_state(row)
        if state == _ROW_STALE:
            return "⚠"
        if state == _ROW_LAPSED:
            return "⚠"          # the same alarm; its tooltip is the one that differs
        if state == _ROW_UNCHECKED:
            return "?"          # not an alarm: an absent verdict
        return ""

    def _row_tip(self, row: int) -> str:
        state = self._row_state(row)
        if state == _ROW_STALE:
            return self._stale_tip
        if state == _ROW_LAPSED:
            return _LAPSED_ROW_TIP
        if state == _ROW_UNCHECKED:
            return _UNCHECKED_ROW_TIP
        return ""

    def column_index(self, name: str) -> int:
        """Column number of a schema column, or -1. Never hard-code a position."""
        return self._columns.index(name) if name in self._columns else -1

    def row_dict(self, row: int) -> dict[str, Any] | None:
        """One source row as a plain dict (schema keys -> python values), or None."""
        if not (0 <= row < len(self._df)):
            return None
        return {c: self._df.iat[row, i] for i, c in enumerate(self._columns)}

    # -- Qt model protocol ------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation != Qt.Horizontal:
            # The ROW header is where a stale row gets its non-colour marker: a glyph
            # in text, because colour alone is the first thing lost to a colour-blind
            # reader, a screenshot or a printout — the same reasoning as the event
            # table's header ⚠ (gui/panels.py), applied per row.
            if role == Qt.DisplayRole:
                marker = self._row_marker(section)
                return f"{section + 1} {marker}" if marker else section + 1
            if role == Qt.ToolTipRole:
                return self._row_tip(section) or None
            return None
        if not (0 <= section < len(self._columns)):
            return None
        name = self._columns[section]
        if role == Qt.DisplayRole:
            # `t_start_s` is on the RECORDING clock while the trace axis beside the
            # table is on the instrument clock; the same "(t₀=0)" badge the event
            # table wears, under the same condition (the two clocks actually differ
            # on the recording being shown). See core/clock.py.
            if name in _CLOCK_COLUMNS and self._clock_offset_s is not None \
                    and not clock.clocks_agree(self._clock_offset_s):
                return f"{name} (t₀=0)"
            return name
        if role == Qt.ToolTipRole:
            if name in _CLOCK_COLUMNS:
                # The per-row truth first (this table pools recordings with different
                # t₀), then the event table's field_note for the recording currently
                # on screen — the same sentence, so the two tables cannot disagree.
                #
                # THE TWO STATEMENTS NAME DIFFERENT CLOCKS ON PURPOSE. field_note is
                # about the DRAWN axis (t₀ + the layer's alignment nudge); the pooled
                # sentence is about the INSTRUMENT clock, which is what a saved row's
                # t0_s holds. They used to both claim to reach "the trace axis", which
                # made one tooltip give two conversions the moment a layer was nudged —
                # and made the t0_s column's stated arithmetic wrong by exactly the
                # nudge. Now each says which clock it lands on, and `_nudge_note` says
                # out loud when the two are apart.
                pooled = ("Each row's own t0_s column holds its recording's "
                          "instrument-clock origin, so t_start_s + t0_s is that row's "
                          "start ON THE INSTRUMENT CLOCK — rows from different "
                          "recordings have different t₀.")
                parts = [pooled, self._nudge_note()]
                if self._clock_offset_s is not None:
                    parts.insert(0, clock.field_note(clock.RECORDING,
                                                     self._clock_offset_s))
                return "\n\n".join(p for p in parts if p)
            return _STAMP_COLUMN_TIPS.get(name)
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == _SORT_ROLE:
            # Raw + comparable. NA sorts last-ish rather than raising against a number.
            if pd.isna(value):
                return None
            if isinstance(value, (int, np.integer, float, np.floating)):
                return float(value)
            return str(value)
        # A stale row is VISIBLY a stale row in every cell: amber foreground (the event
        # table's stale colour) plus the tooltip naming the state and the remedy. The
        # non-colour marker is the ⚠ on the row header — see headerData.
        if role == Qt.ForegroundRole and self._row_is_stale(index.row()):
            return QtGui.QBrush(QtGui.QColor(_STALE_ROW_COLOR))
        if role == Qt.ToolTipRole:
            # Every marked state explains itself here, including "?": a glyph the user
            # cannot ask about is only marginally better than no glyph. Colour stays
            # exclusive to the ⚠ states — an absent verdict is not an alarm.
            tip = self._row_tip(index.row())
            if tip:
                return tip
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        if pd.isna(value):
            return ""
        if role == Qt.EditRole and self._columns[index.column()] == _LABEL_COL:
            return str(value)          # edit the text itself, never a formatted view of it
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    def flags(self, index: QModelIndex):
        base = super().flags(index)
        if index.isValid() and self._columns[index.column()] == _LABEL_COL:
            return base | Qt.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:  # noqa: N802
        """Only the annotation is editable; every other column is a measurement."""
        if (not index.isValid() or role != Qt.EditRole
                or self._columns[index.column()] != _LABEL_COL):
            return False
        self._df.iat[index.row(), index.column()] = str(value)
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        return True

    # -- mutation ----------------------------------------------------------
    def append(self, row: dict[str, Any]) -> int:
        """Append one measurement; returns its SOURCE row number.

        ``beginInsertRows`` (not a reset) so views keep their selection/scroll.

        The basis stamp is written AFTER ``append_measurement``, not through it:
        ``append_measurement`` builds its row from ``MEASUREMENT_SCHEMA`` and therefore
        drops any key outside it, which is precisely the behaviour that keeps the frozen
        schema frozen. So the additive column is re-established (idempotently) and the
        new row's value written into it here — one writer, in the one method every append
        goes through. The key is also left in ``row`` on purpose: the day
        ``signal_basis`` is declared in ``MEASUREMENT_SCHEMA`` itself (one line — see this
        panel's ``needs_wiring`` note) ``append_measurement`` carries it, this assignment
        writes the same string over the same cell, and nothing else has to change.
        """
        at = len(self._df)
        self.beginInsertRows(QModelIndex(), at, at)
        # The EXISTING rows' stamps have to be carried over by hand as well:
        # `append_measurement` re-coerces the frame it is given, which drops the additive
        # column off every row already in the table. Measured before this line existed:
        # row 0 saved 'processed', then a second measurement was added, and row 0 read ''.
        prior = ([str(v) for v in self._df[SIGNAL_BASIS_COLUMN].fillna(BASIS_UNKNOWN)]
                 if SIGNAL_BASIS_COLUMN in self._df.columns else [BASIS_UNKNOWN] * at)
        stamps = prior + [str(row.get(SIGNAL_BASIS_COLUMN) or BASIS_UNKNOWN)]
        out = append_measurement(self._df, row)
        out[SIGNAL_BASIS_COLUMN] = pd.Series(stamps, index=out.index, dtype="string")
        self._df = out
        self._columns = list(self._df.columns)
        self.endInsertRows()
        return at

    def replace(self, source_row: int, row: dict[str, Any]) -> bool:
        """Replace one SOURCE row without changing its position or identity.

        A saved manual measurement is an editable record, not an append-only note: after
        returning to its cursors and refining their bounds, the user can deliberately
        replace its computed values in place. Coercing the one incoming row through the
        same schema door as a project restore is important here: a replacement must retain
        the frozen measurement dtypes *and* the additive signal-basis stamp rather than
        leaving a mixed-object DataFrame that later exports differently.

        ``dataChanged`` deliberately covers every column and the raw sort role. The proxy
        may currently be sorted by a measurement value which has just changed, and the
        panel's existing connection turns this one mutation into the normal
        ``measurementsChanged`` announcement (dirty state + undo checkpoint).
        """
        if not (0 <= int(source_row) < len(self._df)):
            return False
        replacement = _cast_measurements(pd.DataFrame([row]))
        if len(replacement) != 1:
            return False
        row_index = self._df.index[int(source_row)]
        for column in self._columns:
            self._df.at[row_index, column] = replacement.at[0, column]
        left = self.index(int(source_row), 0)
        right = self.index(int(source_row), max(len(self._columns) - 1, 0))
        self.dataChanged.emit(
            left, right,
            [Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole, Qt.ForegroundRole, _SORT_ROLE])
        return True

    def remove_rows(self, rows) -> int:
        """Drop every SOURCE row in ``rows`` (any order, duplicates fine). Returns the
        count actually removed. Deleting back-to-front keeps the earlier indices valid."""
        wanted = sorted({int(r) for r in rows if 0 <= int(r) < len(self._df)}, reverse=True)
        if not wanted:
            return 0
        self.beginResetModel()
        self._df = self._df.drop(self._df.index[wanted]).reset_index(drop=True)
        self.endResetModel()
        return len(wanted)

    def set_df(self, df: pd.DataFrame) -> None:
        """Replace the whole table (project restore). Coerced to the schema so a table
        read back from parquet lands with the same dtypes/columns as a live one — and so a
        table written before a column existed still loads (see :func:`_cast_measurements`)."""
        self.beginResetModel()
        self._df = _cast_measurements(df)
        self._columns = list(self._df.columns)
        self.endResetModel()


# ---------------------------------------------------------------------------
# sorting + "this recording only"
# ---------------------------------------------------------------------------
class _MeasurementsProxy(QSortFilterProxyModel):
    """Sorts on the RAW value and can narrow the pooled table to the current recording.

    The saved table is one global log across every recording, segment and channel measured
    this session. When you are working on one recording that is noise; ``set_scoped`` hides
    every row that is not this document's. Filtering matches on the durable ``doc_key``
    when both sides carry one and degrades to the display name for rows saved before the
    column existed.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSortRole(_SORT_ROLE)
        self._scoped = False
        self._doc_key = ""
        self._recording = ""

    def set_target(self, doc_key: str, recording: str) -> None:
        self._doc_key = str(doc_key or "")
        self._recording = str(recording or "")
        self.invalidate()   # (not invalidateRowsFilter: PySide6 marks it deprecated)

    def set_scoped(self, on: bool) -> None:
        self._scoped = bool(on)
        self.invalidate()   # (not invalidateRowsFilter: PySide6 marks it deprecated)

    def filterAcceptsRow(self, source_row: int, parent: QModelIndex) -> bool:  # noqa: N802
        if not self._scoped:
            return True
        model = self.sourceModel()
        if model is None:
            return True
        row = model.row_dict(source_row)
        if row is None:
            return True
        if not self._doc_key and not self._recording:
            return False                      # scoped with nothing selected -> show nothing
        key = "" if pd.isna(row.get("doc_key")) else str(row.get("doc_key") or "")
        if key and self._doc_key:
            return key == self._doc_key
        name = "" if pd.isna(row.get("recording")) else str(row.get("recording") or "")
        return name == self._recording


# ---------------------------------------------------------------------------
# MeasurePanel
# ---------------------------------------------------------------------------
class MeasurePanel(QtWidgets.QWidget):
    """Hand-select an event with cursors; measure it live; save the measurements.

    Emits:
        measureActiveChanged(bool): the Measure toggle flipped (any cause) — lets
            MainWindow keep its ``Analysis ▸ Measure event`` menu check in sync.
        markMissedRequested(): the user asked to mark the current cursor selection as
            an event the DETECTOR missed. The owner (AnalysisPanel) characterizes the
            bracketed span and appends it to the active detection as an ``added`` event
            — the panel itself never touches the detection, only announces the request.
        measurementsChanged(): the saved-measurements table gained, lost, or replaced a row through
            a user edit (Add to list / Update selected / Remove selected / a label edit) — lets MainWindow
            treat manual measurements as unsaved work. NOT emitted by
            ``set_measurements_df`` (a bulk project restore), so reopening a project never
            looks dirty.
        navigateRequested(doc_id, channel): the user asked to return to a saved measurement
            that belongs to a DIFFERENT open document/channel than the one selected. The
            panel cannot select a document itself (that is the Files dock's job and no panel
            imports another panel), so it announces the target and its owner routes the
            selection — the same shape as ``markMissedRequested``. Once the app selection
            lands back on this panel through ``set_selection`` the panel finishes the jump
            (cursors + zoom) itself.
    """

    measureActiveChanged = Signal(bool)
    markMissedRequested = Signal()
    measurementsChanged = Signal()
    navigateRequested = Signal(str, int)

    # (row label, MEASUREMENT_SCHEMA key). Order = the live readout's rows.
    _READOUT_ROWS: tuple[tuple[str, str], ...] = (
        ("n samples", "n_samples"),
        ("dwell", "dwell_s"),
        ("amplitude", "amplitude"),
        ("dI/I₀ (frac)", "frac_amplitude"),
        ("level", "level"),
        ("baseline I₀", "baseline"),
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
        ("variance", "var"),
        ("min", "min"),
        ("max", "max"),
        ("peak-to-peak", "ptp"),
        ("rms", "rms"),
        ("skewness", "skewness"),
        ("kurtosis (excess)", "kurtosis"),
        ("area / ECD", "area_ecd"),
        ("SNR", "snr"),
    )
    # Schema keys carrying the channel's own signal units (get the unit suffix).
    _SIGNAL_KEYS = frozenset(
        {"amplitude", "level", "baseline", "mean", "median", "std",
         "min", "max", "ptp", "rms"}
    )

    def __init__(self, session: Session, trace_view: TraceView,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._trace_view = trace_view

        self._target: tuple[str, int] | None = None   # (doc_id, channel) — the CURRENT channel
        self._selected_event: tuple[str, int, int, int] | None = None  # doc,ch,start,end
        self._last_stats: dict[str, Any] | None = None
        # Why the last measurement attempt produced nothing, when the reason is a defect
        # in the DATA rather than an ordinary gesture. See `_measure` / `refusal_note`.
        self._refusal: str | None = None
        self._next_id: int = 0                          # running measurement_id
        #: The LAST calibration-stamp verdict each row got while its recording was open
        #: (``measurement_id`` -> ``CAL_STAMP_*``). Session-lived and never persisted: it
        #: is not a finding about the data, it is the memory of a check that was possible
        #: once. It exists so that closing a recording downgrades a ⚠ to "marked, and no
        #: longer re-checkable" instead of erasing it — see `_refresh_stale_rows`.
        self._last_stamp_state: dict[int, str] = {}
        # Whether the active (doc, channel) has a stored detection — pushed in by the
        # owner (AnalysisPanel). Gates the "Mark selection as missed event" button, which
        # only makes sense when there is a detector output to add the missed event to.
        self._detection_present: bool = False
        self._missed_gate_reason: str = ""
        # The LIVE Preprocess-form parameters, pushed in by the owner (AnalysisPanel)
        # through `set_conditioning`. None / an empty dict means "no processing is
        # configured" FROM THE FORM, and the panel then asks `_stored_detection_chain`
        # instead; with nothing there either it behaves exactly as it did before the basis
        # choice existed: raw, and the Processed option greyed with a reason. The panel
        # never reaches for the form itself — a panel in this app does not import a
        # sibling panel, and the fallback goes through Session, which is not one.
        self._conditioning: dict[str, Any] | None = None
        # The identity of the chain the panel is ACTUALLY using (pushed, or the fallback
        # recovered from the stored detection — see `_active_conditioning`). It is what
        # `_conditioning_changed` compares against to decide whether a re-measure would
        # move a number, and it therefore has to track the EFFECTIVE chain rather than the
        # pushed one: the fallback changes under this panel with no push at all.
        self._effective_key: tuple[Any, ...] = ()
        # Memo of the conditioned trace: (key, array). Conditioning the whole recording
        # costs 16 ms (low-pass only) to 85 ms (notch + detrend + SavGol + low-pass) on
        # the owner's 749 k-sample -1000 mV file and ~4x that on his 3.0 M-sample one,
        # measured — which is nothing once, and a stutter on every debounced drag frame.
        # The key carries everything that can change the array underneath a cached one:
        # target, length, rate, CALIBRATION fingerprint (the conditioning runs on the
        # calibrated samples) and the parameter dict itself.
        self._cond_cache: tuple[tuple[Any, ...], np.ndarray] | None = None
        # On-trace markers for the saved rows of the CURRENT channel: one batched SPAN
        # overlay for the set, one MARKER for the row the user has selected.
        self._span_overlay_id: str | None = None
        self._marker_overlay_id: str | None = None
        self._current_measurement_id: int | None = None   # the selected row's stable id

        # Drag coalescing: sigMeasureChanged only ARMS this; the recompute happens when the
        # edges have been still for _RECOMPUTE_DEBOUNCE_MS. Single-shot + restart-on-arm is
        # what drops a superseded pending recompute — there is never more than one queued.
        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(_RECOMPUTE_DEBOUNCE_MS)
        self._recompute_timer.timeout.connect(self._recompute)

        self._build_ui()

        # The live wires: the region moves → schedule a recompute; the region SETTLES (mouse
        # released, or placed programmatically) → recompute now. Double-click placement
        # arrives as a data-x request; this panel resolves it against its authoritative target.
        self._trace_view.sigMeasureChanged.connect(self._on_measure_changed)
        self._trace_view.sigMeasureFinished.connect(self._on_measure_finished)
        self._trace_view.sigMeasureActiveChanged.connect(self._on_trace_measure_active_changed)
        self._trace_view.sigMeasurePlacementRequested.connect(
            self._on_trace_measure_placement_requested)
        # A layer toggle / view-mode change REBUILDS the TraceView, destroying every overlay
        # with its plot. Session is the panel's only other permitted correspondent, so redraw
        # from it. TraceView subscribes to the same signal first (it is constructed first), so
        # by the time this runs the new panes exist.
        self._session.layersChanged.connect(self._refresh_markers)
        # TraceView rebuilds first on this signal. Re-evaluate target visibility after its panes
        # exist so hiding/showing the selected channel updates every Measure control immediately.
        self._session.layersChanged.connect(self._update_enabled)
        # A calibration edit (gain/offset/units Apply) changes the numbers under the cursors
        # WITHOUT the region moving. The trace curve repaints from this very signal
        # (TraceView subscribes to it too), so the readout must re-measure from it as well —
        # otherwise the trace and the Live statistics disagree on screen, and add_current()
        # would save the stale dict (the exact-commit path hands a non-preview _last_stats
        # back unchanged).
        self._session.calibrationChanged.connect(self._on_calibration_changed)

        self._update_enabled()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        title = QtWidgets.QLabel("Manual measurement (cursors)")
        title.setStyleSheet("font-weight:600; font-size:13px;")
        root.addWidget(title)

        # -- the cursor toggle + snap conveniences ------------------------
        self.action_widget = QtWidgets.QWidget()
        tools = QtWidgets.QGridLayout(self.action_widget)
        tools.setContentsMargins(0, 0, 0, 0)
        self._toggle = QtWidgets.QPushButton("Measure event (cursors)")
        self._toggle.setCheckable(True)
        self._toggle.setToolTip(
            "Show a movable region on the current-channel trace; its two edges are "
            "the boundary cursors. Double-click the trace to place it, drag the shaded "
            "body to move it, or drag either circular handle to resize it.")
        self._toggle.toggled.connect(self._on_toggle)
        self._snap_view_btn = QtWidgets.QPushButton("Snap to view")
        self._snap_view_btn.setToolTip(
            "Place a mask over the middle 20% of the visible view, then drag its circular "
            "edge handles to fit the event. Double-clicking the trace places it at the click.")
        self._snap_view_btn.clicked.connect(self._snap_to_view)
        self._snap_event_btn = QtWidgets.QPushButton("Snap to selected event")
        self._snap_event_btn.setToolTip("Set the cursors to the event last selected in the event table.")
        self._snap_event_btn.clicked.connect(self._snap_to_event)
        self._clear_mask_btn = QtWidgets.QPushButton("Clear mask")
        self._clear_mask_btn.setToolTip(
            "Remove the live cursor mask. This never deletes a saved measurement or event.")
        self._clear_mask_btn.setEnabled(False)
        self._clear_mask_btn.clicked.connect(self.clear_mask)
        tools.addWidget(self._toggle, 0, 0, 1, 2)
        tools.addWidget(self._snap_view_btn, 1, 0)
        tools.addWidget(self._snap_event_btn, 1, 1)
        tools.addWidget(self._clear_mask_btn, 2, 0, 1, 2)
        tools.setColumnStretch(0, 1)
        tools.setColumnStretch(1, 1)
        root.addWidget(self.action_widget)

        # -- mark the current cursor selection as an event the DETECTOR missed --
        # A distinct action (own row) because it feeds the DETECTION event set, not the
        # manual-measurement list below: it appends the bracketed span to the active
        # detection as an ``added`` event so recall (and the training export) count it.
        self._mark_missed_btn = QtWidgets.QPushButton("Mark selection as missed event")
        self._mark_missed_tip = (
            "Add the current cursor selection to the active detection as an event the "
            "detector MISSED. It is characterized like a detected event, marked "
            "accepted + added, and counts as a miss (FN) in the detector score. "
            "Ctrl+Shift+M works while this tab is hidden.")
        self._mark_missed_btn.setToolTip(self._mark_missed_tip)
        self._mark_missed_btn.setEnabled(False)
        self._mark_missed_btn.clicked.connect(self._on_mark_missed)
        root.addWidget(self._mark_missed_btn)

        # -- amplitude method ---------------------------------------------
        method_row = QtWidgets.QFormLayout()
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(list(AMPLITUDE_METHODS))
        self._method_combo.setToolTip(
            "How the in-selection level is defined for the amplitude:\n"
            "median/mean — central level; min/max — the deepest/peak sample; "
            "mode — the histogram-peak (flat-blockade) level.")
        # A combo change is one discrete gesture, not a stream: settle at once (dropping any
        # frame the debounce still had pending, which would recompute the same thing).
        self._method_combo.currentTextChanged.connect(
            lambda *_a: (self._stop_pending_recompute(), self._recompute()))
        method_row.addRow("amplitude method", self._method_combo)

        # -- WHICH SIGNAL the numbers describe -----------------------------
        # An EXPLICIT choice, never a silent switch. With the processed curve on the plot
        # this tool used to measure the raw calibrated channel and say nothing (audit,
        # 2026-09-02), so the fix cannot be "measure whatever is drawn" either: a reading
        # whose basis depends on a checkbox two tabs away is exactly as unreadable as a
        # reading that ignores it. The user picks; the row records the pick; this panel
        # says which one it is, on the line under the numbers.
        self._basis_combo = QtWidgets.QComboBox()
        self._basis_combo.addItem("Raw signal", BASIS_RAW)
        self._basis_combo.addItem("Processed signal", BASIS_PROCESSED)
        self._basis_combo.setToolTip(
            "Which signal to measure:\n"
            "Raw — the calibrated channel exactly as loaded (the app's behaviour "
            "before this control existed, and still the default).\n"
            "Processed — the conditioned trace: the Preprocess tab's filter chain, the "
            "same array the on-trace preview curve draws. Available once a chain is "
            "configured; every saved row records which of the two it was measured on.")
        # Same "one discrete gesture, settle at once" wiring as the amplitude method: a
        # basis change invalidates nothing that a queued preview could still use.
        self._basis_combo.currentIndexChanged.connect(
            lambda *_a: (self._stop_pending_recompute(),
                         self._render_basis_note(), self._recompute()))
        method_row.addRow("measure on", self._basis_combo)
        root.addLayout(method_row)

        # The basis line: ALWAYS present, ALWAYS two lines tall, whatever it says. It is
        # the sentence under which every number in the readout below should be read, and
        # a label that appears and disappears with the basis would shove the whole
        # readout up and down under the pointer — the failure this panel's layout rules
        # exist to prevent. So the height is reserved for two lines up front and the full
        # text (a long chain can exceed them) is always in the tooltip.
        self._basis_note = QtWidgets.QLabel("")
        self._basis_note.setWordWrap(True)
        self._basis_note.setStyleSheet("color:#8b9199; font-size:11px;")
        self._basis_note.setFixedHeight(2 * self._basis_note.fontMetrics().height() + 4)
        self._basis_note.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                       QtWidgets.QSizePolicy.Fixed)
        root.addWidget(self._basis_note)
        self._sync_basis_controls()

        # -- live readout of every stat -----------------------------------
        readout_box = QtWidgets.QGroupBox("Live statistics")
        rlay = QtWidgets.QVBoxLayout(readout_box)
        rlay.setContentsMargins(6, 6, 6, 6)
        self._readout_table = QtWidgets.QTableWidget(len(self._READOUT_ROWS), 1)
        self._readout_table.setHorizontalHeaderLabels(["value"])
        self._readout_table.setVerticalHeaderLabels([lbl for lbl, _ in self._READOUT_ROWS])
        self._readout_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._readout_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._readout_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._readout_table.setFocusPolicy(Qt.NoFocus)
        self._readout_table.verticalHeader().setDefaultSectionSize(20)
        for i in range(len(self._READOUT_ROWS)):
            self._readout_table.setItem(i, 0, QtWidgets.QTableWidgetItem("—"))
        rlay.addWidget(self._readout_table)
        # The honesty line. Empty (and hidden) whenever the readout above is exact; when the
        # selection is too big to measure in full at drag speed it says so, in the same
        # breath as promising that the SAVED row will not be. Silent approximation would be
        # worse than the freeze this whole mechanism exists to avoid.
        self._approx_label = QtWidgets.QLabel("")
        self._approx_label.setWordWrap(True)
        self._approx_label.setStyleSheet(f"color:{_APPROX_COLOR}; font-size:11px;")
        self._approx_label.setVisible(False)
        rlay.addWidget(self._approx_label)
        # The refusal line, in the same slot and by the same rules. `measure_selection`
        # refuses a selection (or an open-pore window) carrying NaN/Inf rather than
        # printing a robust number measured on poisoned data — the right call, but the
        # panel used to swallow the exception and leave a table of "—", which reads as
        # "nothing selected". The reason belongs where the user is looking, not in a log.
        self._refusal_label = QtWidgets.QLabel("")
        self._refusal_label.setWordWrap(True)
        self._refusal_label.setStyleSheet(f"color:{_REFUSAL_COLOR}; font-size:11px;")
        self._refusal_label.setVisible(False)
        rlay.addWidget(self._refusal_label)
        root.addWidget(readout_box)

        # -- add / remove / export / go-to --------------------------------
        # A GRID, not one long row: the un-wrappable 3-button row is exactly what used to
        # floor the whole Inspector dock's width, so the new actions go on a SECOND line
        # instead of widening it further.
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setContentsMargins(0, 0, 0, 0)
        self._add_button = QtWidgets.QPushButton("Add to list")
        self._add_button.setToolTip(
            "Append the current measurement to the table below.  (Ctrl+M)")
        self._add_button.setEnabled(False)
        self._add_button.clicked.connect(self.add_current)
        self._update_button = QtWidgets.QPushButton("Update selected")
        self._update_tip = (
            "Replace one selected saved measurement with the current cursor span and "
            "statistics. Its measurement ID and label are retained. Select a row, use "
            "Go to measurement, adjust the mask, then update it.")
        self._update_button.setToolTip(self._update_tip)
        self._update_button.setEnabled(False)
        self._update_button.clicked.connect(self.update_selected)
        self._remove_button = QtWidgets.QPushButton("Remove selected")
        self._remove_button.setToolTip(
            "Delete every selected row. Shift/Ctrl-click to select more than one.")
        self._remove_button.clicked.connect(self.remove_selected)
        self._export_button = QtWidgets.QPushButton("Export CSV…")
        self._export_button.setToolTip("Write every listed measurement to a CSV for extraction.")
        self._export_button.clicked.connect(lambda: self.export_csv())
        self._goto_button = QtWidgets.QPushButton("Go to measurement")
        self._goto_button.setToolTip(
            "Put the cursors back on the selected row's span and zoom the trace to it "
            "(double-clicking the row does the same). Switches recording/channel if needed.")
        self._goto_button.setEnabled(False)
        self._goto_button.clicked.connect(self.go_to_selected)
        self._scope_check = QtWidgets.QCheckBox("This recording only")
        self._scope_check.setToolTip(
            "Hide rows measured on other recordings. The table itself still holds them "
            "(and Export CSV still writes them all).")
        self._scope_check.toggled.connect(self._on_scope_toggled)
        self._markers_check = QtWidgets.QCheckBox("Show on trace")
        self._markers_check.setChecked(True)
        self._markers_check.setToolTip(
            "Shade every saved measurement of this channel on the trace, so you can see "
            "at a glance where you have already measured. The selected row is highlighted.")
        self._markers_check.toggled.connect(self._on_markers_toggled)
        # THE DOOR TO THE PROCESSED-TRACE EXPORT (see `export_processed_trace`). It was
        # written with no way in: no button, no menu item, no shortcut anywhere in the app
        # reached it, so the samples the user was looking at could still only be got out by
        # a test. It lives HERE, next to the measurements export, because this panel is
        # where the *measure on* choice is made — the user who cares about the processed
        # array is the user who just measured on it.
        #
        # A THIRD GRID ROW, SPANNING THE THREE EXISTING COLUMNS: the row it would otherwise
        # have to squeeze into is the one whose un-wrappable width already floored the
        # Inspector dock (see the grid comment above), and a fourth column would widen that
        # floor by a whole button. Spanning costs vertical space the stretching table
        # absorbs and no horizontal space at all.
        #
        # ALWAYS ENABLED, never greyed: `processed_export_gate` returns a SENTENCE, and a
        # click with no chain configured opens it as a dialog that says which control to
        # go and use. A dead grey button with a tooltip nobody hovers is the failure this
        # app already fixed once for the detection actions.
        self._export_processed_button = QtWidgets.QPushButton("Export processed trace…")
        self._export_processed_button.clicked.connect(lambda: self.export_processed_trace())
        btn_grid.addWidget(self._add_button, 0, 0)
        btn_grid.addWidget(self._remove_button, 0, 1)
        btn_grid.addWidget(self._export_button, 2, 0)
        btn_grid.addWidget(self._goto_button, 1, 0)
        btn_grid.addWidget(self._update_button, 1, 1)
        btn_grid.addWidget(self._scope_check, 3, 0)
        btn_grid.addWidget(self._markers_check, 3, 1)
        btn_grid.addWidget(self._export_processed_button, 2, 1)
        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)
        root.addLayout(btn_grid)
        self._sync_processed_export()

        # -- the saved-measurements table ---------------------------------
        # Sorted + filtered THROUGH A PROXY, so every row number that leaves this widget must
        # be mapped back to the source (see _selected_source_rows) — a view row is not a
        # measurement.
        self._model = _MeasurementsModel(self)
        self._proxy = _MeasurementsProxy(self)
        self._proxy.setSourceModel(self._model)
        self._table_view = QtWidgets.QTableView()
        self._table_view.setModel(self._proxy)
        self._table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Only the label cell is editable (the model's flags say so); a double-click anywhere
        # else is the NAVIGATE gesture, handled in _on_row_activated.
        self._table_view.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setSortingEnabled(True)
        # Explicit initial sort: QHeaderView's default sort ORDER is Descending, so enabling
        # sorting alone silently flips the log newest-first. A lab log reads in the order you
        # measured, so start on measurement_id ascending.
        self._table_view.sortByColumn(self._model.column_index("measurement_id"),
                                      Qt.AscendingOrder)
        self._table_view.doubleClicked.connect(self._on_row_activated)
        self._table_view.selectionModel().selectionChanged.connect(
            lambda *_a: self._on_row_selection_changed())
        # A typed annotation is unsaved work exactly like an added row.
        self._model.dataChanged.connect(lambda *_a: self.measurementsChanged.emit())
        root.addWidget(self._table_view, 1)

        # Keyboard: Ctrl+C copies the selected rows as TSV for a lab notebook, scoped to the
        # table so it never fights the Event Inspector's own window keys (←/→/A/R/Space) or
        # the event table's j/k/a/r/f.
        #
        # CTRL+M IS NOT HERE. It used to be a Qt.WindowShortcut on this panel, which Qt
        # delivers only while the panel is VISIBLE — and this panel is one page of a
        # four-tab stack inside a dock that itself sits behind an outer tab, so the key was
        # dead whenever the Manual-measurement tab was not raised (measured: fires 1/1 with
        # the tab shown, 0/1 hidden) and dead again whenever the Inspector dock was on
        # Calibration or Files. That contradicts the deliberate design that the cursors stay
        # LIVE when the tab is hidden — the whole loop is drag on the trace, press Ctrl+M,
        # drag again, without ever looking at this panel — while the Add button's tooltip
        # went on advertising it. It is now a QAction on MainWindow's Analysis menu, which
        # is window-wide by construction and cannot be hidden by a tab.
        self._copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Copy, self._table_view)
        self._copy_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._copy_shortcut.activated.connect(self.copy_selected_to_clipboard)

        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color:#8b9199;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

    # -- selection (pushed in by the owner) ------------------------------------------------------
    def set_selection(self, doc_id: str, channel: int) -> None:
        """Repoint at (doc_id, channel) — the AnalysisPanel's resolved CURRENT channel.

        A change of target clears any live selection (the region belongs to the old
        channel's pane), so the measurement always describes the channel on screen."""
        new = (doc_id, channel)
        if new == self._target:
            self._sync_target_scope()   # same target, but the layer/name may have changed
            return
        if self._toggle.isChecked():
            self._toggle.setChecked(False)   # deactivates the OLD target's region
        self._stop_pending_recompute()       # a queued frame would measure the OLD channel
        self._target = new
        self._selected_event = None
        self._refusal = None       # it described the OLD channel's selection
        self._cond_cache = None    # ...and the memo held the OLD channel's samples
        self._install_stats(None)
        self._sync_target_scope()
        # A new target can have a different stored chain (`_stored_detection_chain`) as
        # well as a different rate, so this is the full chain re-read, not just a re-sync.
        self._conditioning_changed()
        self._update_enabled()
        self._refresh_markers()

    def clear(self) -> None:
        """Drop the target entirely (no channel selected / document closed)."""
        if self._toggle.isChecked():
            self._toggle.setChecked(False)
        self._stop_pending_recompute()
        self._target = None
        self._selected_event = None
        self._refusal = None       # no channel, no selection, nothing to refuse
        self._cond_cache = None    # the conditioned samples belonged to that channel
        self._install_stats(None)
        self._sync_target_scope()
        self._conditioning_changed()   # no target, so no chain either
        self._update_enabled()
        self._refresh_markers()

    def _sync_target_scope(self) -> None:
        """Tell the proxy which recording is current, so 'This recording only' follows the
        selection — and tell the model which clock offset is on screen, so the
        ``t_start_s`` header can say which clock it is on (the same push the event table
        gets from its owner; see core/clock.py)."""
        doc = self._target_document()
        if doc is None:
            self._proxy.set_target("", "")
            self._model.set_clock_offset(None, None, 0.0)
        else:
            self._proxy.set_target(persistent_key(doc), doc.display_name)
            # BOTH halves of the clock story, from the two sources that disagree: the
            # offset the trace is DRAWN through (t₀ + this layer's alignment nudge) and
            # the instrument-clock origin a saved row stamps. The header needs both to
            # avoid promising an arithmetic that only holds when they coincide.
            self._model.set_clock_offset(
                self._trace_view.clock_offset_for(self._target[0], self._target[1]),
                clock.instrument_origin_s(doc.meta),
                self._trace_view.layer_time_offset(self._target[0], self._target[1]))
        # A new target can also change which rows are judgeable (their document may just
        # have (re)opened on this selection change), so re-derive the stale marks.
        self._refresh_stale_rows()

    def _target_document(self) -> Document | None:
        if self._target is None or not self._session.has_document(self._target[0]):
            return None
        return self._session.document(self._target[0])

    def set_selected_event(self, doc_id: str, channel: int,
                           start_idx: int, end_idx: int) -> None:
        """Remember the event last selected in the event table so *Snap to selected
        event* can jump the cursors to it. Pushed by the owner (AnalysisPanel), which
        owns event-row selection; the panel itself never queries another panel."""
        self._selected_event = (doc_id, channel, int(start_idx), int(end_idx))
        self._sync_snaps()

    def set_detection_present(self, present: bool, reason: str = "") -> None:
        """Tell the panel whether the active (doc, channel) has a stored detection.

        Pushed by the owner (:class:`~nanoscope.gui.analysis_panel.AnalysisPanel`) on
        every result / selection change; it is the gate (together with a live cursor
        selection) for the *Mark selection as missed event* button."""
        self._detection_present = bool(present)
        self._missed_gate_reason = "" if present else str(reason)
        self._sync_mark_missed()
        # A detection arriving (or being dropped) is also a CHAIN arriving or leaving —
        # see :meth:`_stored_detection_chain`. This push already runs on every result /
        # selection change, so it is the hook that keeps the fallback chain live without
        # this panel ever asking a sibling panel anything.
        self._conditioning_changed()

    # -- the measurement basis (raw vs processed) ------------------------------------------------
    def set_conditioning(self, params: Mapping[str, Any] | None) -> None:
        """Push in the LIVE Preprocess-form parameters — the chain the preview draws.

        The owner (:class:`~nanoscope.gui.analysis_panel.AnalysisPanel`) calls this
        whenever the detection form changes, exactly as it pushes the current target and
        the detection-present flag: this panel talks to Session and its TraceView and to
        no sibling panel, so the parameters have to arrive from outside.

        ``None`` or an empty dict means "no chain configured" — the panel then falls back
        to :meth:`_stored_detection_chain`, and with nothing there either the *Processed
        signal* option is greyed with a reason and the panel measures raw, which is the
        shipped default and the pre-existing behaviour.

        Re-measures only when it would change a number: a chain edit while the basis is
        *Raw* changes the note under the readout and nothing else, and re-running a
        whole-recording measurement for it would be pure latency.

        AN UNCHANGED PUSH IS A NO-OP, and callers depend on it: ``MainWindow`` re-asks the
        form and pushes on every measure-action refresh and on every Analysis-menu open,
        precisely so the export gate can never describe a chain the user edited three
        gestures ago. Throwing the conditioned-trace memo away on those would re-condition
        the whole recording (16-85 ms on the owner's 749 k-sample file, ~4x that at 3 M)
        the next time a processed number is asked for, for a chain that did not move."""
        new = dict(params) if params else None
        if self._params_key(new) != self._params_key(self._conditioning):
            self._conditioning = new
            self._cond_cache = None         # the array this described is gone
        self._conditioning_changed()

    def _stored_detection_chain(self) -> dict[str, Any] | None:
        """The chain THIS TARGET'S stored detection ran under — the in-app fallback.

        WHY THIS EXISTS. ``set_conditioning`` is a door, and until the owner pushes
        through it the basis combo is a control that toggles nothing: the *Processed*
        item stays greyed, the export gate stays shut, and the feature is reachable and
        inert (audit, 2026-09-02). This recovers a chain WITHOUT reaching for a sibling
        panel, through the only two correspondents this panel is allowed: Session — which
        owns the :class:`~nanoscope.core.analysis.ResultStore` — and
        :mod:`~nanoscope.analysis.conditioning`, whose ``conditioned_trace`` is a pure
        function of ``(raw, fs, params)``.

        ``AnalysisResult.params`` for a detection IS the Preprocess/Detection form as it
        stood when that run was launched: the same ``filter_cutoff_hz`` /
        ``filter_kind`` / ``filter_order`` and the same ``*_enabled`` stage flags
        ``conditioned_trace`` reads. Measuring on it is not an approximation of the live
        form — it is the array the events in the table beside this one were measured on,
        so a hand measurement taken this way is directly comparable to a detected row
        rather than a second quantity under the same column names.

        Only a chain that CONDITIONS anything is returned: a Schmitt run with no filter
        stage enabled leaves the panel exactly where it was, on raw, with the *Processed*
        item greyed and saying so. Detectors are tried in
        :data:`~nanoscope.core.sources.DETECTION_MODULE_IDS` order and the first
        conditioning chain wins; they are alternatives on one channel, not a pipeline.
        """
        if self._target is None or not self._session.has_document(self._target[0]):
            return None
        doc_id, channel = self._target
        for module_id in DETECTION_MODULE_IDS:
            try:
                res = self._session.results.get(doc_id, int(channel), module_id)
            except Exception:  # noqa: BLE001 — a store that cannot answer is "no chain"
                res = None
            if res is None:
                continue
            params = dict(res.params or {})
            if conditioning_enabled(params):
                return params
        return None

    def _active_conditioning(self) -> dict[str, Any] | None:
        """THE chain this panel is working from: the pushed one, else the stored run's.

        One reader, so the note under the readout, the greying of the *Processed* item,
        the array ``_signal_for_basis`` measures, the export gate and the sidecar's
        ``params`` can never disagree about which chain is in force.

        A push that CONDITIONS SOMETHING always wins — it is the live form, and the
        preview curve on the trace is drawing it. The fallback speaks in the one state
        where the push says nothing: no chain pushed at all, or a pushed form with every
        stage off. That second case is not the panel overruling the user, because the
        fallback is not a guess — it is the chain a detection on THIS channel actually ran
        under, it is still on offer rather than in force (the combo item merely stops
        being grey), and :meth:`_render_basis_note` names it as the stored run, so the
        sentence under the readout never claims the Preprocess form when it means the
        detection. Without this the feature would stay inert for the whole stretch between
        a form edit and the next push, which is where the audit found it.
        """
        if conditioning_enabled(self._conditioning):
            return self._conditioning
        return self._stored_detection_chain() or self._conditioning

    def _chain_is_from_stored_detection(self) -> bool:
        """Whether the chain in force came from the stored detection, not from the form.

        Asked only by the note: two chains that condition the same way are still two
        different claims about WHY these numbers are what they are."""
        return (not conditioning_enabled(self._conditioning)
                and self._stored_detection_chain() is not None)

    def _conditioning_changed(self) -> None:
        """The chain may have moved (a push, a new target, a detection landing).

        Re-syncs every surface unconditionally — those are cheap and their text depends on
        more than the chain — and re-measures ONLY when the EFFECTIVE chain actually
        changed and the numbers on screen are being taken from it. A chain edit while the
        basis is *Raw* moves a sentence and nothing else, and re-running a whole-recording
        measurement for it would be pure latency."""
        key = self._params_key(self._active_conditioning())
        moved = key != self._effective_key
        self._effective_key = key
        self._sync_basis_controls()
        if moved and self.measurement_basis() == BASIS_PROCESSED:
            self._stop_pending_recompute()
            self._recompute()

    def conditioning_params(self) -> dict[str, Any] | None:
        """The chain in force (a copy), or None. Public for the export path and for tests
        that need to assert what the panel is conditioning with. This is the EFFECTIVE
        chain — what was pushed, or what :meth:`_stored_detection_chain` recovered — not
        the push alone, because it is the answer to "what would Processed measure?"."""
        active = self._active_conditioning()
        return None if active is None else dict(active)

    def measurement_basis(self) -> str:
        """The basis the user has CHOSEN: ``'raw'`` or ``'processed'``.

        What the combo says, not what the last measurement managed to do — those differ
        exactly when the chosen basis is unavailable, and in that case the panel refuses
        out loud (blank readout + a red line) rather than quietly measuring the other
        one. A silent fallback is the defect this control was added to remove."""
        data = self._basis_combo.currentData()
        return str(data) if data else BASIS_RAW

    def set_measurement_basis(self, basis: str) -> None:
        """Select ``'raw'`` or ``'processed'`` programmatically (menu action, tests)."""
        idx = self._basis_combo.findData(str(basis))
        if idx >= 0 and idx != self._basis_combo.currentIndex():
            self._basis_combo.setCurrentIndex(idx)

    def basis_note(self) -> str:
        """The sentence currently under the readout naming the basis. Reads the TEXT, for
        the same reason ``approx_note`` does: a widget in an unshown window reports itself
        invisible, so asking Qt would make this lie headlessly and in a collapsed dock."""
        return self._basis_note.text()

    def processed_basis_available(self) -> bool:
        """Whether *Processed* can be measured right now: a chain is configured AND the
        target has a sample rate (every stage is designed in Hz/seconds)."""
        if not conditioning_enabled(self._active_conditioning()):
            return False
        return self._target_sample_rate() is not None

    def _target_sample_rate(self) -> float | None:
        """The current target's sample rate, or None (no target / legacy no-rate file)."""
        if self._target is None or not self._session.has_document(self._target[0]):
            return None
        try:
            fs = self._session.calibrated(*self._target).sample_rate_hz
        except (KeyError, ValueError, IndexError):
            return None
        return None if fs is None else float(fs)

    @staticmethod
    def _params_key(params: Mapping[str, Any] | None) -> tuple[Any, ...]:
        """A hashable, comparable identity for a parameter dict.

        ``repr`` per value rather than the value itself: the form's dicts carry lists
        (the explicit notch line list) and numpy scalars, neither of which is reliably
        hashable or ``==``-comparable to a plain Python twin. This key only ever has to
        answer "is this the same form state as last time"."""
        if not params:
            return ()
        return tuple(sorted((str(k), repr(v)) for k, v in params.items()))

    def _sync_basis_controls(self) -> None:
        """Grey the *Processed* option when there is no processed trace, and re-say why.

        The item is DISABLED, never removed: a combo that grows and shrinks a row moves
        every control under it between two identical-looking app states. A selection that
        has become unavailable is left standing (and refused out loud when measured) for
        the same reason — silently re-selecting *Raw* would change the meaning of the
        numbers with no gesture from the user."""
        model = self._basis_combo.model()
        item = model.item(self._basis_combo.findData(BASIS_PROCESSED)) \
            if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(self.processed_basis_available())
        self._render_basis_note()
        # The same chain change opens or closes the export's gate; its tooltip is the one
        # surface that has to re-say why.
        self._sync_processed_export()

    def _render_basis_note(self) -> None:
        """Write the basis line: which signal the numbers below describe, and its chain."""
        params = self._active_conditioning()
        basis = self.measurement_basis()
        chain = chain_description(params)
        # WHOSE chain it is. Two chains can condition identically and still be two
        # different claims about why these numbers are what they are, so the sentence
        # names the source whenever it is the stored run rather than the live form.
        whose = (" — the chain the stored detection on this channel ran under"
                 if self._chain_is_from_stored_detection() else "")
        if basis != BASIS_PROCESSED:
            text = ("Measuring the RAW channel — the calibrated samples as loaded"
                    + (f". A chain IS available ({chain}{whose}); switch to “Processed "
                       f"signal” to measure it instead." if chain else "."))
        elif not self.processed_basis_available():
            text = ("Processed is selected but there is no processed trace: "
                    + ("this recording has no sample rate, and every stage is designed "
                       "in Hz and seconds."
                       if conditioning_enabled(params)
                       else "no filter stage is enabled on the Preprocess tab, and no "
                            "stored detection on this channel ran under one."))
        else:
            text = f"Measuring the PROCESSED trace: {chain}{whose}."
            if (params or {}).get("normalize_enabled"):
                # The one stage that takes the picoamps with it. modules.py says so of
                # the detection path; a hand measurement is no different, and the row's
                # `units` stamp still spells the channel's label.
                text += (" Normalize is in this chain, so these amplitudes are on a "
                         "dimensionless scale, not in the channel's units.")
        self._basis_note.setText(text)
        self._basis_note.setToolTip(text)   # the two reserved lines can clip a long chain

    def _signal_for_basis(self, doc_id: str, channel: int, chan: Any) -> np.ndarray | None:
        """The array to measure for the CHOSEN basis, or None (refusing out loud).

        Raw is the calibrated channel, exactly as this panel has always measured it —
        same object, same numbers, so nothing about a raw measurement moved. Processed is
        :func:`~nanoscope.analysis.conditioning.conditioned_trace`, the same function (and
        the same bytes: verified by hash on the owner's own recording) that draws the
        preview curve.

        Refuses rather than substitutes. A chain that cannot be built — a cutoff above
        Nyquist, a detrend window longer than the trace, a recording with no rate — leaves
        the readout blank with the kernel's own message under it, because a number
        silently taken from the other basis is indistinguishable from the one the user
        asked for."""
        if self.measurement_basis() != BASIS_PROCESSED:
            return np.asarray(chan.data)
        params = self._active_conditioning()
        fs = chan.sample_rate_hz
        if not conditioning_enabled(params) or fs is None:
            self._refusal = (
                "No processed trace to measure: "
                + ("this recording has no sample rate, so the filter stages (all designed "
                   "in Hz and seconds) cannot run."
                   if conditioning_enabled(params)
                   else "no filter stage is enabled on the Preprocess tab. Enable one, "
                        "or switch “measure on” back to the raw signal."))
            return None
        raw = np.asarray(chan.data)
        cal = self._session.calibration(doc_id, channel)
        key = (str(doc_id), int(channel), int(raw.size), float(fs),
               tuple(cal.fingerprint()), self._params_key(params))
        cached = self._cond_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            data = conditioned_trace(raw, float(fs), params)
        except Exception as exc:  # noqa: BLE001 — the kernels' refusals are user-facing
            self._cond_cache = None
            self._refusal = (f"The processing chain could not be applied to this "
                             f"recording: {exc}")
            return None
        self._cond_cache = (key, data)
        return data

    def _on_mark_missed(self) -> None:
        """Announce that the current cursor selection should be added to the active
        detection as a missed event. Guarded by the same condition that enables the
        button; the owner reads the cursor bounds off the shared TraceView and does the
        characterization + append.

        Nothing about the live preview reaches the event: the owner characterizes the RAW
        samples of the bracketed span itself, so a hand-added event is exact by construction
        however approximate the readout was. It is also unbounded, so a big span gets the
        same wait cursor an exact re-measurement does — the routing is a direct connection,
        so the work happens inside this emit."""
        if not self._can_mark_missed():
            return
        n = int((self._last_stats or {}).get("n_samples", 0))
        with self._busy(n, f"Characterizing all {n:,} samples…"):
            self.markMissedRequested.emit()

    def _can_mark_missed(self) -> bool:
        """A detection exists, the cursors are active on this target, and there is a
        valid live selection to bracket."""
        return (self._detection_present and self.is_active()
                and self.can_measure() and self._last_stats is not None)

    def _sync_mark_missed(self) -> None:
        self._mark_missed_btn.setEnabled(self._can_mark_missed())
        self._mark_missed_btn.setToolTip(
            self._mark_missed_tip
            + (("\n\n" + self._missed_gate_reason)
               if not self._detection_present and self._missed_gate_reason else "")
        )

    # -- public state ----------------------------------------------------------------------------
    def has_open_target(self) -> bool:
        """Whether the resolved target recording is still open, drawn or not."""
        if self._target is None:
            return False
        return self._session.has_document(self._target[0])

    def can_measure(self) -> bool:
        """Whether the live target has a visible pane on which cursors can exist."""
        if not self.has_open_target() or self._target is None:
            return False
        return self._trace_view.visible_x_range_for(*self._target) is not None

    def is_active(self) -> bool:
        return self._toggle.isChecked()

    def set_active(self, on: bool) -> None:
        """Programmatically flip the Measure toggle (MainWindow's menu action)."""
        on = bool(on)
        if on and not self.can_measure():
            return
        if self._toggle.isChecked() != on:
            self._toggle.setChecked(on)

    def clear_mask(self) -> bool:
        """Remove the transient cursor mask without touching anything saved.

        A trace double-click intentionally creates the same live cursor selection as the
        Measure tool.  Clearing that selection must be a separate, reversible-in-spirit
        operation from deleting a table row: it turns the tool off, drops only the
        ``LinearRegionItem``, and leaves saved manual measurements, detector events,
        labels, and project history unchanged.
        """
        if not self.is_active():
            self._set_status("There is no live annotation mask to clear.")
            return False
        self.set_active(False)
        self._set_status(
            "Cleared the live annotation mask. Saved measurements and events were not changed.")
        return True

    def set_method(self, method: str) -> None:
        """Choose the amplitude method by name (recomputes live)."""
        if method in AMPLITUDE_METHODS:
            self._method_combo.setCurrentText(method)

    def current_measurement(self) -> dict[str, Any] | None:
        """The last computed stat dict (as returned by ``measure_selection``), or None.

        This is what the READOUT is showing, so on a selection above the live bound it is a
        bounded preview: ``approximate`` is True, ``stride``/``n_measured`` say by how much,
        and the keys in ``suppressed`` are NaN. Never persist it — call
        :meth:`exact_measurement` (or just :meth:`add_current`, which does it for you)."""
        return self._last_stats

    def readout_value(self, key: str) -> str:
        """The displayed text for a MEASUREMENT_SCHEMA key in the live readout."""
        for row, (_, k) in enumerate(self._READOUT_ROWS):
            if k == key:
                return self._readout_table.item(row, 0).text()
        return ""

    @property
    def measurements_df(self) -> pd.DataFrame:
        """The saved-measurements table as a schema-typed DataFrame (what CSV export writes)."""
        return self._model.df

    def set_measurements_df(self, df: pd.DataFrame) -> None:
        """Replace the whole manual-measurement table (project restore).

        Coerced to the schema and installed as-is, so a table read back from a
        ``.nanoscope`` project shows exactly what was saved. The running
        ``measurement_id`` counter is advanced past the restored rows so a later
        *Add to list* never reuses an id."""
        self._model.set_df(df)
        ids = self._model.df["measurement_id"]
        self._next_id = int(ids.max()) + 1 if len(ids) else 0
        self._current_measurement_id = None
        self._refresh_stale_rows()   # restored rows may have been measured under old calibrations
        self._fit_columns()
        self._refresh_markers()
        self._sync_goto()
        n = len(self._model.df)
        self._set_status(f"Restored {n} measurement(s)." if n else "")

    # -- toggle + snapping -----------------------------------------------------------------------
    def _on_toggle(self, checked: bool) -> None:
        checked = bool(checked)
        if checked and not self.can_measure():
            self._toggle.setChecked(False)   # nothing to measure on; revert
            return
        if self._target is not None and self._session.has_document(self._target[0]):
            doc_id, channel = self._target
            # set_measure_active(True) seeds the region and emits sigMeasureChanged,
            # which drives the first _recompute; (False) tears it down.
            self._trace_view.set_measure_active(doc_id, channel, checked)
        if not checked:
            self._stop_pending_recompute()   # the region is gone; nothing left to measure
            self._refusal = None             # ... and so is the selection it described
            self._install_stats(None)
        self._clear_mask_btn.setEnabled(checked)
        self._sync_snaps()
        self._sync_mark_missed()
        self._sync_update_button()
        self.measureActiveChanged.emit(checked)

    def _on_trace_measure_active_changed(self, active: bool) -> None:
        """Mirror a TraceView-side teardown (notably hiding the target layer) onto this owner."""
        if not active and self._toggle.isChecked():
            self._toggle.setChecked(False)
        self._clear_mask_btn.setEnabled(bool(active and self._toggle.isChecked()))
        self._sync_update_button()

    def _snap_to_view(self) -> None:
        self._place_mask_in_view()

    def _on_trace_measure_placement_requested(self, x: float) -> None:
        """Place the working annotation mask where the user double-clicked the trace."""
        self._place_mask_in_view(float(x))

    def _place_mask_in_view(self, centre_x: float | None = None) -> bool:
        """Place a visible mask no wider than 20% of the current x window.

        ``centre_x`` is the trace double-click location; ``None`` is the Snap-to-view
        button and uses the middle of this recording's visible overlap. The mask is
        clamped wholly inside both that overlap and the target recording. This cap is
        intentionally placement-only: either handle can still be dragged beyond it.
        """
        if not self.can_measure() or self._target is None:
            return False
        rng = self._trace_view.visible_x_range_for(*self._target)
        extent = self._trace_view.measure_cursor_extent(*self._target)
        if rng is None or extent is None:
            self._set_status(
                "This recording has no visible sample span for the annotation mask.")
            return False

        vlo, vhi = sorted((float(rng[0]), float(rng[1])))
        e0, e1 = sorted((float(extent[0]), float(extent[1])))
        overlap0, overlap1 = max(vlo, e0), min(vhi, e1)
        view_span = vhi - vlo
        overlap_span = overlap1 - overlap0
        if (not np.isfinite([vlo, vhi, e0, e1]).all()
                or view_span <= 0.0 or overlap_span <= 0.0):
            self._set_status(
                "The visible view does not overlap this recording. Pan to where this "
                "recording is drawn, then place the mask again.")
            return False

        clicked = centre_x is not None
        if clicked:
            centre = float(centre_x)
            if not np.isfinite(centre) or centre < overlap0 or centre > overlap1:
                self._set_status(
                    "That point is outside the visible part of the selected recording. "
                    "Double-click where its trace is drawn.")
                return False
        else:
            centre = 0.5 * (overlap0 + overlap1)

        width = min(MEASURE_MAX_VIEW_FRACTION * view_span, overlap_span)
        if not np.isfinite(width) or width <= 0.0:
            return False
        lo = centre - 0.5 * width
        lo = min(max(lo, overlap0), overlap1 - width)
        hi = lo + width

        self.set_active(True)
        self._trace_view.set_measure_region(lo, hi)
        if clicked:
            self._set_status(
                "Placed a Measure mask at the double-click (maximum 20% of the view). "
                "Drag the shaded body to move it or a circular handle to resize it.")
        elif overlap0 != vlo or overlap1 != vhi:
            self._set_status(
                "Placed a Measure mask in the visible part of this recording "
                "(maximum 20% of the view).")
        else:
            self._set_status(
                "Placed a Measure mask over the middle 20% of the visible view. "
                "Drag its handles to fit the event.")
        return True

    def _snap_to_event(self) -> None:
        if not self.can_measure() or self._selected_event is None:
            return
        doc_id, channel, start, end = self._selected_event
        if (doc_id, channel) != self._target:
            return
        self.set_active(True)
        self._place_cursors(doc_id, channel, start, end)

    def _n_samples(self, doc_id: str) -> int:
        """Loaded sample count of ``doc_id`` (0 when it is gone).

        Read from the metadata, never by materialising the array — it is the SAME ``n`` the
        shared x-axis (and therefore both index<->x mappings) is built from."""
        if not self._session.has_document(doc_id):
            return 0
        return int(self._session.document(doc_id).meta.n_samples)

    def _x_span(self, doc_id: str, channel: int,
                start: int, end: int) -> tuple[float, float] | None:
        """The data-x range of the half-open sample span [start, end) on (doc_id, channel).

        Both edges map through ``TraceView.x_for_index``, i.e. the SAME layer time offset
        ``measure_bounds()`` maps back through — so placing the region here and reading the
        bounds straight back returns the indices passed in. (Going via
        ``session.x_axis(doc_id)`` instead, at offset 0, lands the cursors
        ``time_offset_s`` away on any time-aligned layer and silently measures the wrong
        span.)

        Two edge cases the plain round-trip gets WRONG, because ``x_for_index`` CLAMPS its
        index into ``[0, n-1]`` while ``measure_bounds`` clamps its index into ``[0, n]``:

        * **the half-open end AT the trace end.** Dragging the right cursor off the end of
          the trace is an ordinary gesture and saves ``end_idx == n``; asking for the x of
          sample ``n`` gives back the x of sample ``n-1``, so the restored cursors would sit
          one sample short of what the row says. The end edge is therefore placed ONE SAMPLE
          PAST the last x, which is exactly what maps back to ``n``.
        * **a span outside the loaded data** — a project saved over the whole recording and
          reopened at a shorter sample range, or a row carried onto a trimmed segment. The
          clamp would quietly land both cursors on the last sample and the caller would
          report success on a measurement it is NOT showing. Out of range returns None so the
          caller can say so.
        """
        start = int(start)
        end = int(end)
        n = self._n_samples(doc_id)
        if n <= 0 or not (0 <= start < n) or not (start < end <= n):
            return None
        x0 = self._trace_view.x_for_index(doc_id, channel, start)
        dx = self._trace_view.sample_dx(doc_id, channel)
        if end < n:
            x1 = self._trace_view.x_for_index(doc_id, channel, end)
        else:
            last = self._trace_view.x_for_index(doc_id, channel, n - 1)
            x1 = None if last is None else last + dx
        if x0 is None or x1 is None:
            return None
        if x1 <= x0:                      # degenerate/1-sample span: give it a visible width
            x1 = x0 + max(dx, 0.0)
            if x1 <= x0:
                return None
        return (x0, x1)

    def _place_cursors(self, doc_id: str, channel: int,
                       start: int, end: int) -> tuple[float, float] | None:
        """Put the region on [start, end) of (doc_id, channel). Returns the x range placed."""
        span = self._x_span(doc_id, channel, start, end)
        if span is not None:
            self._trace_view.set_measure_region(*span)
        return span

    # -- live recompute --------------------------------------------------------------------------
    def _on_measure_changed(self, _start: int, _stop: int) -> None:
        """A drag FRAME. Arm the debounce and return — never measure here.

        ``QTimer.start()`` on a running single-shot timer restarts it, so a stream of frames
        leaves exactly one pending recompute and the superseded ones simply never happen."""
        self._recompute_timer.start()

    def _on_measure_finished(self, _start: int, _stop: int) -> None:
        """The selection came to rest. Drop any pending frame and settle immediately.

        This is what keeps the readout feeling instant on mouse-release and on every
        programmatic placement (snap / go-to). It is still the BOUNDED path: a user can
        resize the mask across a very large span, so an unbounded eager recompute here
        would reintroduce the very freeze the debounce was added to remove."""
        self._recompute_timer.stop()
        self._recompute()

    def _on_calibration_changed(self, doc_id: str, channel: int) -> None:
        """The calibration of (doc_id, channel) changed.

        TWO things here can now disagree with it, and they are refreshed separately:

        * the SAVED table may hold rows measured on (doc_id, channel) under the
          calibration that just died — any target, not only the current one — so the
          stale marks are re-derived on every edit;
        * the LIVE readout, which describes the current target alone: ``_recompute``
          pulls the calibrated channel + units fresh, so re-measuring is all it takes to
          agree with the repainted trace again. A settle, not a debounce arm: Apply is
          one discrete gesture like an amplitude-method change, and a pending drag frame
          would only re-measure under the dead calibration. With the tool off (or no
          valid selection) this installs None over None — a no-op."""
        self._refresh_stale_rows()
        # The conditioned memo was built FROM the calibrated samples, so it died with the
        # calibration. Its key carries the fingerprint and would have caught this anyway;
        # dropping it here means the stale array is not even held until the next compare.
        if self._cond_cache is not None and self._cond_cache[0][:2] == (str(doc_id), int(channel)):
            self._cond_cache = None
        if self._target != (doc_id, int(channel)):
            return
        self._stop_pending_recompute()
        self._recompute()

    def _recompute(self) -> None:
        """Re-measure the current selection for the LIVE readout (bounded, may be a preview).

        Pulls the calibrated channel + fs + units from Session. Baseline, polarity and noise
        are left to ``measure_selection`` (it reads them from just outside the selection —
        the open-pore reference), so the panel passes only the geometry + method + the cost
        bound. Anything that PERSISTS the result must go through ``_exact_stats`` instead."""
        stats: dict[str, Any] | None = None
        if self.is_active() and self.can_measure():
            bounds = self._trace_view.measure_bounds()
            if bounds is not None:
                stats = self._measure(bounds[0], bounds[1],
                                      self._method_combo.currentText(),
                                      max_samples=_LIVE_MAX_SAMPLES)
        self._install_stats(stats)

    def _measure(self, start: int, stop: int, method: str,
                 *, max_samples: int | None) -> dict[str, Any] | None:
        """One ``measure_selection`` call on the current target, or None if it cannot run.

        The single seam both the live path and the exact-commit path go through, so they can
        never drift apart on anything but ``max_samples``. A collapsed / <2-sample selection
        (or a channel that went away mid-gesture) is a user gesture, not a crash."""
        # Cleared BEFORE the guard, not after it: the note describes the selection
        # under the cursors, so a target that went away mid-gesture must not leave a
        # red "your samples are not numbers" line standing under an empty readout.
        self._refusal = None
        if self._target is None or not self.can_measure():
            return None
        doc_id, channel = self._target
        try:
            chan = self._session.calibrated(doc_id, channel)
            # WHICH ARRAY — the one question this method used to answer implicitly, and
            # wrongly whenever the processed curve was on screen (audit, 2026-09-02).
            # None means the chosen basis is unavailable and `_signal_for_basis` has
            # already written the reason into `_refusal`: blank readout, visible cause.
            data = self._signal_for_basis(doc_id, channel, chan)
            if data is None:
                return None
            return measure_selection(
                data,
                chan.sample_rate_hz,
                start,
                stop,
                amplitude_method=method,
                units=chan.units or "",
                max_samples=max_samples,
            )
        except NonFiniteSelection as exc:
            # The ONE refusal the user has to be told about: it means the samples under
            # the cursors are not numbers, and no gesture of theirs will fix it. Every
            # other ValueError here is a gesture (a collapsed drag, a 1-sample
            # selection, a channel that went away mid-drag) and stays silent — a banner
            # on every collapsed drag would train the user to ignore this one.
            self._refusal = str(exc)
            return None
        except (ValueError, TypeError, KeyError):
            return None

    def _install_stats(self, stats: dict[str, Any] | None) -> None:
        """Adopt ``stats`` as the current measurement and refresh everything that shows it."""
        self._last_stats = stats
        self._render_readout(stats)
        self._render_approx_note(stats)
        self._render_refusal_note()
        self._add_button.setEnabled(stats is not None)
        self._sync_mark_missed()
        self._sync_update_button()

    def _stop_pending_recompute(self) -> None:
        """Cancel a queued recompute — the thing it would have measured is gone."""
        self._recompute_timer.stop()

    # -- exact measurement (everything that persists) ---------------------------------------------
    def _exact_stats(self) -> dict[str, Any] | None:
        """The current selection measured over EVERY sample — the only thing worth saving.

        Re-measures from the geometry and method the live readout is *currently showing*
        (``_last_stats``' own ``start_idx``/``end_idx``/``amplitude_method``), not from a
        fresh ``measure_bounds()``: the row must describe the span the user was looking at
        when they clicked, and reading the region again would be one more chance for the two
        to disagree. Returns ``_last_stats`` unchanged when it is already exact — the common
        case (any selection under the live bound), which therefore costs nothing at all and
        behaves exactly as it did before the bound existed.

        Slow by construction on a huge span (~1.1 s at 15 M samples). That is the correct
        trade — an approximated saved number is not a measurement — so it is done under a
        wait cursor rather than hidden or skipped."""
        stats = self._last_stats
        if stats is None:
            return None
        if not stats.get("approximate", False):
            return stats
        start, stop = int(stats["start_idx"]), int(stats["end_idx"])
        method = str(stats["amplitude_method"])
        self._stop_pending_recompute()   # a queued preview would only be thrown away
        with self._busy(stop - start, f"Measuring all {stop - start:,} samples exactly…"):
            return self._measure(start, stop, method, max_samples=None)

    @contextmanager
    def _busy(self, n_samples: int, message: str):
        """Wait cursor + status while an unbounded measurement of `n_samples` runs.

        Nothing at all below `_BUSY_CURSOR_SAMPLES`, where the work is a few ms and a
        flickering cursor would be the only thing anyone noticed.

        The repaint is pumped with ``ExcludeUserInputEvents`` deliberately: the cursor and
        the status line have to reach the screen before a ~1 s computation starts, but a
        plain ``processEvents`` would also deliver the second click of an impatient
        double-click straight back into the handler that is already running it, and save the
        row twice. Timers still fire, which is harmless — the geometry this measurement uses
        was read before the pump, so a preview landing mid-flight cannot change it."""
        busy = n_samples >= _BUSY_CURSOR_SAMPLES
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(Qt.WaitCursor))
            self._set_status(message)
            QtWidgets.QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        try:
            yield
        finally:
            if busy:
                QtWidgets.QApplication.restoreOverrideCursor()

    def exact_measurement(self) -> dict[str, Any] | None:
        """The current selection measured over every sample, or None. Public for callers
        (and tests) that need the committed truth rather than the live preview."""
        return self._exact_stats()

    def is_preview(self) -> bool:
        """Whether the live readout is currently a bounded (strided) estimate."""
        return bool(self._last_stats is not None
                    and self._last_stats.get("approximate", False))

    def approx_note(self) -> str:
        """The approximate-preview note; '' exactly when the readout is exact.

        Reads the TEXT, not ``isVisible()``: a child of a window that has not been shown
        reports itself invisible, so asking Qt would make this lie in any headless context
        (and in a collapsed dock). ``_render_approx_note`` clears the text and the visibility
        together, so the two can never disagree."""
        return self._approx_label.text()

    # -- readout rendering -----------------------------------------------------------------------
    def _render_readout(self, stats: dict[str, Any] | None) -> None:
        units = self._units()
        suppressed = set(stats.get("suppressed", ())) if stats else set()
        for row, (_, key) in enumerate(self._READOUT_ROWS):
            item = self._readout_table.item(row, 0)
            if stats is None:
                item.setText("—")
                item.setToolTip("")
                continue
            text = self._fmt_stat(key, stats.get(key), units)
            # dI/I0 > 1 is an ENHANCEMENT, not a 200% blockade. Same threshold and same
            # sentence as the detection Summary and the CSV sidecar
            # (`analysis.events.frac_depth_note`) -- this read-out used to print the
            # ratio bare, on the one surface a user watches live (report High-5).
            enh = frac_depth_note(stats.get(key)) if key == "frac_amplitude" else ""
            item.setText(text + (" ⚠>1" if enh else ""))
            # A blanked statistic reads as "—" like any undefined one, so say WHY: it is
            # withheld, not missing, and it will be there the moment the row is saved.
            item.setToolTip(
                "Not shown while the readout is a subsampled preview — this statistic "
                "depends on the extreme samples, which subsampling systematically misses. "
                "The saved measurement includes it, computed over every sample."
                if key in suppressed else enh)

    def _render_approx_note(self, stats: dict[str, Any] | None) -> None:
        """Show/clear the "this is a preview" line under the readout.

        Spells out the sampling ratio (so the user can judge it), exactly what was withheld,
        and the promise that saving re-measures in full. Hidden — not merely emptied —
        whenever the readout is exact, so the panel does not carry a permanent caveat."""
        if stats is None or not stats.get("approximate", False):
            self._approx_label.setVisible(False)
            self._approx_label.setText("")
            return
        stride = int(stats.get("stride", 1))
        measured = int(stats.get("n_measured", 0))
        total = int(stats.get("n_samples", 0))
        text = (f"≈ preview: 1:{stride} sampling ({measured:,} of {total:,} samples). "
                f"Saving re-measures every sample.")
        suppressed = tuple(stats.get("suppressed", ()))
        if suppressed:
            text += f"  Not estimated: {', '.join(suppressed)}."
        self._approx_label.setText(text)
        self._approx_label.setVisible(True)

    def _render_refusal_note(self) -> None:
        """Show/clear the "this selection cannot be measured" line under the readout.

        Driven off ``self._refusal``, which ``_measure`` sets from
        :class:`~nanoscope.analysis.measure.NonFiniteSelection` and clears on every
        attempt — so the note describes the CURRENT cursors and disappears by itself the
        moment the user drags off the poisoned span. Text and visibility are set
        together, for the same reason ``_render_approx_note`` does it.
        """
        text = self._refusal or ""
        self._refusal_label.setText(text)
        self._refusal_label.setVisible(bool(text))

    def refusal_note(self) -> str:
        """Why the readout is blank, or '' when it is not blank for a reason.

        Reads the TEXT rather than ``isVisible()``, exactly as ``approx_note`` does: a
        child of a window that was never shown reports itself invisible, which would
        make this lie in every headless test.
        """
        return self._refusal_label.text()

    def _fmt_stat(self, key: str, value: Any, units: str) -> str:
        if value is None:
            return "—"
        if key == "n_samples":
            return str(int(value))
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(v):
            return "—"
        suffix = f" {units}" if units else ""
        if key == "dwell_s":
            return self._fmt_time(v)
        if key == "area_ecd":
            return f"{v:.4g}{(' ' + units + '·s') if units else ' ·s'}"
        if key == "var":
            return f"{v:.4g}{(' ' + units + '²') if units else ''}"
        if key in self._SIGNAL_KEYS:
            return f"{v:.4g}{suffix}"
        # dimensionless: frac_amplitude, skewness, kurtosis, snr
        return f"{v:.4g}"

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds * 1e6:.4g} µs"
        if seconds < 1.0:
            return f"{seconds * 1e3:.4g} ms"
        return f"{seconds:.4g} s"

    def _units(self) -> str:
        """The channel's effective units label, without materialising the array."""
        if self._target is None:
            return ""
        doc_id, channel = self._target
        if not self._session.has_document(doc_id):
            return ""
        info = self._session.document(doc_id).meta.channels[channel]
        cal = self._session.calibration(doc_id, channel)
        return cal.target_units or info.raw_units or ""

    # -- the saved table -------------------------------------------------------------------------
    def _saved_row_from_exact(self, exact: dict[str, Any], *, label: str,
                              measurement_id: int) -> dict[str, Any] | None:
        """Stamp one exact cursor measurement as a durable saved-table row.

        Both *Add to list* and *Update selected* use this one writer.  That prevents an
        in-place update from quietly keeping stale calibration, clock, signal-basis, or
        target metadata while changing only its start/end indices.
        """
        if self._target is None:
            return None
        doc_id, channel = self._target
        if not self._session.has_document(doc_id):
            return None
        doc = self._session.document(doc_id)
        row = dict(exact)
        for key in MEASUREMENT_DIAGNOSTIC_KEYS:
            row.pop(key, None)          # preview diagnostics: not schema columns
        row["recording"] = doc.display_name
        row["doc_key"] = persistent_key(doc)        # path (file) or segment uid — THE identity
        row["channel"] = doc.meta.channels[channel].name
        row["channel_index"] = int(channel)         # the stable channel key; the name is not
        row["label"] = str(label)
        # WHICH SIGNAL these numbers describe. Stamped from the basis the measurement was
        # actually taken on — `_exact_stats` re-measures through the same `_measure`, so a
        # row can only exist if that basis was reachable. A pooled table of 30 pA rows
        # some of which were low-passed and normalized and some of which were not is
        # permanently ambiguous without this column; see analysis/conditioning.py.
        row[SIGNAL_BASIS_COLUMN] = self.measurement_basis()
        # The provenance stamps (see analysis.measure's module docstring). `units` is
        # already in `exact` — the measurement echoes the calibrated channel's label —
        # so only the calibration fingerprint and the recording's clock origin are
        # stamped here. The calibration is read fresh from Session, the same source
        # `_measure` read the calibrated samples from, so the stamp and the numbers
        # cannot describe two different calibrations.
        cal = self._session.calibration(doc_id, channel)
        row["cal_gain"], row["cal_offset"] = cal.fingerprint()
        # WHERE THIS ROW SAT IN THE EXPERIMENT: `clock.instrument_origin_s`, the
        # instrument-clock instant of the recording's first sample. Not
        # `clock.clock_offset_s`, which answers the different question "how far apart are
        # the two clocks ON SCREEN" and is therefore 0 on a SEGMENT — whose axis this app
        # rebased to 0 while the Files dock, the segment's name and its notes all print
        # its real origin (1158.99 s). Stamping that 0 put a parent row and a row from a
        # segment OF THAT PARENT 19 minutes apart under a column that calls itself the
        # row's instrument-clock origin, which is exactly what this column was added to
        # prevent. It also excludes the layer alignment nudge, deliberately and as before:
        # the row records what was measured, and the nudge is a display alignment the user
        # can change afterwards (the header tooltip says so when one is in force). None
        # (no sample rate → no clock, see core/clock.py) becomes the schema's NaN default.
        row["t0_s"] = clock.instrument_origin_s(doc.meta)
        row["measurement_id"] = int(measurement_id)
        return row

    def add_current(self) -> bool:
        """Append the current measurement, computed over EVERY sample of its span.

        The live readout may be a bounded preview (see the module docstring); a saved row
        may not be, so this re-measures with ``max_samples=None`` first and stamps THOSE
        numbers. When the readout was already exact — any selection under the live bound,
        i.e. essentially every hand-measured event — ``_exact_stats`` hands the same dict
        straight back and nothing is recomputed. The refreshed exact values then replace the
        preview in the readout, so what the user sees after saving is what was saved.

        Stamps BOTH the human labels (``recording`` / ``channel``) and the durable identity
        (``doc_key`` = :func:`~nanoscope.core.segment.persistent_key`, ``channel_index``)
        that *Go to measurement* resolves against — a display name is not an identity (a
        segment can carry its parent's name verbatim)."""
        if self._last_stats is None or self._target is None:
            return False
        exact = self._exact_stats()
        if exact is None:
            self._set_status("Could not re-measure the selection exactly; nothing was added.")
            return False
        if exact is not self._last_stats:
            self._install_stats(exact)   # the readout now shows what is about to be stored
        row = self._saved_row_from_exact(
            exact, label="", measurement_id=self._next_id)
        if row is None:
            return False
        self._next_id += 1
        source_row = self._model.append(row)
        self._refresh_stale_rows()   # the new row is fresh by construction; older ones may not be
        self._fit_columns()
        self._select_source_row(source_row)
        self._refresh_markers()
        self._set_status(f"Added measurement #{row['measurement_id']} "
                         f"({len(self._model.df)} in list), measured on "
                         f"{basis_label(row[SIGNAL_BASIS_COLUMN])}.")
        self.measurementsChanged.emit()
        return True

    def _selected_update_candidate(self) -> tuple[int, dict[str, Any]] | None:
        """The one saved row the live cursors may deliberately replace, if any.

        A table can be sorted, filtered, and multi-selected, so the answer must be a
        source row and must be exactly one row.  The cursor selection also has to be on
        that row's open recording/channel; otherwise an innocent selection change could
        overwrite a row from another trace with unrelated samples.
        """
        rows = self._selected_source_rows()
        if len(rows) != 1 or self._target is None or not self.is_active() \
                or self._last_stats is None:
            return None
        if self._trace_view.measure_target() != self._target \
                or self._trace_view.measure_bounds() is None:
            return None
        source_row = rows[0]
        row = self._model.row_dict(source_row)
        if row is None or self._resolve_row_target(row) != self._target:
            return None
        mid = row.get("measurement_id")
        if mid is None or pd.isna(mid):
            return None
        return source_row, row

    def update_selected(self) -> bool:
        """Replace one saved measurement with the current cursor measurement in place.

        The row's durable ID and human annotation are kept; its bounds, statistics,
        calibration/basis stamps, and target metadata are remeasured from the live
        selection.  ``_MeasurementsModel.replace`` emits ``dataChanged``, so this flows
        through the ordinary dirty-state, project-save, and measurement-history paths.
        """
        candidate = self._selected_update_candidate()
        if candidate is None:
            self._set_status(
                "Select one saved measurement on this recording, return to it, then adjust "
                "the live mask before updating it.")
            self._sync_update_button()
            return False
        source_row, previous = candidate
        exact = self._exact_stats()
        if exact is None:
            self._set_status("Could not re-measure the selection exactly; nothing was updated.")
            return False
        if exact is not self._last_stats:
            self._install_stats(exact)
        measurement_id = int(previous["measurement_id"])
        replacement = self._saved_row_from_exact(
            exact, label=self._row_text(previous, "label"), measurement_id=measurement_id)
        if replacement is None or not self._model.replace(source_row, replacement):
            self._set_status("Could not update the saved measurement.")
            self._sync_update_button()
            return False
        self._refresh_stale_rows()
        self._select_source_row(source_row)
        self._refresh_markers()
        self._sync_update_button()
        self._set_status(
            f"Updated measurement #{measurement_id} in place, measured on "
            f"{basis_label(replacement[SIGNAL_BASIS_COLUMN])}.")
        return True

    def _confirm_remove(self, n: int) -> bool:
        """Ask before deleting hand measurements. An INJECTABLE seam, like MainWindow's
        close confirmations, so the suite never faces a live modal.

        The confirmation STAYS now that Edit ▸ Undo reaches this table
        (:class:`~nanoscope.core.history.MeasurementHistory`), because the two do different
        jobs: undo makes the click recoverable, and the confirmation is what makes the user
        aware there is anything to recover. What has changed is the sentence — it used to
        end "and Edit ▸ Undo does not reach this list", which was true and no help at all
        to somebody who had just deleted the wrong four rows. It now names the remedy.
        """
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setText(f"Delete {n} manual measurement{'' if n == 1 else 's'}?")
        box.setInformativeText(
            "You drew these on the trace by hand — nothing in the app can re-derive them. "
            "Edit ▸ Undo (Ctrl+Z) puts them back, and any project you have already saved "
            "still holds them.")
        box.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        box.button(QtWidgets.QMessageBox.Yes).setText(f"Delete {n}")
        return box.exec() == QtWidgets.QMessageBox.Yes

    def remove_selected(self) -> None:
        """Delete EVERY selected row (the table is multi-select), mapped back to the source
        rows — deleting by view row is the classic sorted-table corruption."""
        rows = self._selected_source_rows()
        if not rows:
            return
        if not self._confirm_remove(len(rows)):
            self._set_status("Nothing was deleted.")
            return
        removed = self._model.remove_rows(rows)
        if not removed:
            return
        self._current_measurement_id = None
        self._refresh_markers()
        self._sync_goto()
        self._set_status(f"Removed {removed} measurement(s) "
                         f"({len(self._model.df)} in list).")
        self.measurementsChanged.emit()

    # -- view row <-> source row -----------------------------------------------------------------
    def _selected_source_rows(self) -> list[int]:
        """The SOURCE rows behind the current selection, de-duplicated and ordered.

        The view is sorted + filtered through ``_MeasurementsProxy``, so view row N is not
        measurement N — every row number that leaves the view goes through ``mapToSource``.
        Deliberately reads the SELECTION, never ``currentIndex()``: the current index stays
        valid after the visual selection is gone, which is how you delete a row you did not
        think was selected."""
        sel = self._table_view.selectionModel()
        if sel is None:
            return []
        rows = {self._proxy.mapToSource(i).row() for i in sel.selectedRows() if i.isValid()}
        return sorted(r for r in rows if r >= 0)

    def _selected_source_rows_in_view_order(self) -> list[int]:
        """The same source rows, ordered as they appear ON SCREEN.

        ``_selected_source_rows`` sorts by source row, which is right for a bulk delete and
        WRONG for anything the user reads back: under a sorted view the on-screen order is
        not the source order, so a clipboard paste (or "the first selected row") must follow
        the view. Same mapToSource discipline — the view row is still never a measurement."""
        sel = self._table_view.selectionModel()
        if sel is None:
            return []
        out: list[int] = []
        seen: set[int] = set()
        for index in sorted((i for i in sel.selectedRows() if i.isValid()),
                            key=lambda i: i.row()):
            source_row = self._proxy.mapToSource(index).row()
            if source_row >= 0 and source_row not in seen:
                seen.add(source_row)
                out.append(source_row)
        return out

    def _select_source_row(self, source_row: int) -> None:
        """Select (and scroll to) a source row through the proxy — no-op when the filter
        currently hides it.

        ``QAbstractItemView.selectRow`` is NOT "select this row": it asks
        ``selectionCommand``, which consults ``QGuiApplication.keyboardModifiers()``, so
        with Ctrl held it EXTENDS the selection instead of replacing it. This method is
        called from ``add_current`` and from Go to, and both mean "this row is now the
        one" — after which ``_on_row_selection_changed`` takes the TOPMOST selected row
        as current, so an extended selection silently leaves the previous measurement
        current and the wrong marker highlighted.

        Reachable by hand (Ctrl+M is the add shortcut, and the modifier is still down
        when the row lands) and reproducible in the suite: a test that sends a synthetic
        Ctrl+M leaves ``keyboardModifiers()`` at ControlModifier for the whole process,
        which is what made six tests in ``test_measure_navigation.py`` depend on which
        file pytest collected first. Stating the command removes both.
        """
        src = self._model.index(source_row, 0)
        view_index = self._proxy.mapFromSource(src)
        if not view_index.isValid():
            return
        sel = self._table_view.selectionModel()
        if sel is not None:
            sel.select(view_index,
                       QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            sel.setCurrentIndex(view_index, QItemSelectionModel.NoUpdate)
        self._table_view.scrollTo(view_index)

    def _fit_columns(self) -> None:
        """Size columns to their content once, then leave the user's sizing alone.

        ``resizeColumnsToContents`` on every add re-widened the ``recording`` column to the
        longest filename each time and destroyed any manual sizing, so it runs only on a
        bulk change and the interactive mode takes over afterwards."""
        self._table_view.resizeColumnsToContents()
        header = self._table_view.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        # A long filename must not eat the whole viewport before dwell/amplitude are reached.
        col = self._model.column_index("recording")
        if col >= 0 and self._table_view.columnWidth(col) > 220:
            self._table_view.setColumnWidth(col, 220)

    # -- returning to a saved measurement ---------------------------------------------------------
    def _on_row_selection_changed(self) -> None:
        """A row became current: remember its id (so the marker follows it) and re-mark.

        "Current" is the TOPMOST SELECTED ROW ON SCREEN, so the highlighted marker is the one
        the eye lands on rather than whichever row happens to be earliest in the source."""
        rows = self._selected_source_rows_in_view_order()
        row = self._model.row_dict(rows[0]) if rows else None
        self._current_measurement_id = int(row["measurement_id"]) if row is not None else None
        self._sync_goto()
        self._refresh_markers()

    def _on_row_activated(self, index) -> None:
        """Double-click. On the annotation column that means EDIT (the view is already
        opening an editor); anywhere else it means "take me back to this measurement"."""
        if not index.isValid():
            return
        source = self._proxy.mapToSource(index)
        if source.column() == self._model.column_index(_LABEL_COL):
            return
        self.navigate_to_row(source.row())

    def go_to_selected(self) -> bool:
        """*Go to measurement* — the discoverable twin of the double-click. With more than
        one row selected it goes to the topmost one ON SCREEN (the same row whose marker is
        highlighted), not to whichever was measured first."""
        rows = self._selected_source_rows_in_view_order()
        if not rows:
            self._set_status("Select a saved measurement first.")
            return False
        return self.navigate_to_row(rows[0])

    def navigate_to_row(self, source_row: int) -> bool:
        """Return to the saved measurement in SOURCE row ``source_row``.

        1. resolve the row's document + channel from its durable identity (degrading to the
           display name for rows saved before ``doc_key`` existed);
        2. if that is not the current target, ANNOUNCE it (``navigateRequested``) and let the
           owner switch the app selection — the panel never selects a document itself;
        3. frame the span with symmetric padding through the shared
           ``TraceView.zoom_to_span``, THEN re-arm the cursors on the exact
           ``[start_idx, end_idx)``.

        Zoom before arming, deliberately: arming seeds the region across ~20% of the CURRENT
        view and recomputes it synchronously, so arming first on a fully zoomed-out 15 M-sample
        recording would measure millions of samples for nothing. Zoomed in first, the seed is
        a few thousand.

        Returns True when the trace ended up on the measurement. A closed recording, or a
        target the app declined to switch to, is reported in the status line — never a
        silent no-op."""
        row = self._model.row_dict(source_row)
        if row is None:
            return False
        target = self._resolve_row_target(row)
        if target is None:
            self._set_status(
                f"“{self._row_recording(row) or 'that recording'}” is not open — "
                f"reopen it to return to this measurement.")
            return False
        doc_id, channel = target
        # Resolve the span BEFORE anything is switched: a row whose indices fall outside the
        # data now loaded (a project reopened at a shorter sample range) must not first drag
        # the whole app selection onto that recording and only then admit it cannot show it.
        span = self._x_span(doc_id, channel, int(row["start_idx"]), int(row["end_idx"]))
        if span is None:
            self._set_status("This measurement's span is no longer inside the loaded data.")
            return False
        if (doc_id, channel) != self._target:
            # Cross-document / cross-channel: the owner owns document selection.
            self.navigateRequested.emit(doc_id, channel)
            if self._target != (doc_id, channel):
                self._set_status(
                    f"Could not switch to “{self._row_recording(row)}” "
                    f"channel {channel} to show this measurement.")
                return False
        x0, x1 = span
        self._current_measurement_id = int(row["measurement_id"])
        self._trace_view.zoom_to_span(
            x0, x1, min_pad=self._trace_view.sample_dx(doc_id, channel) * 8)
        self.set_active(True)
        self._trace_view.set_measure_region(x0, x1)
        self._refresh_markers()
        self._select_source_row(source_row)
        self._set_status(
            f"Showing measurement #{int(row['measurement_id'])} "
            f"({self._row_recording(row)} · {self._row_text(row, 'channel')}).")
        return True

    @staticmethod
    def _row_text(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        return "" if value is None or pd.isna(value) else str(value)

    def _row_recording(self, row: dict[str, Any]) -> str:
        return self._row_text(row, "recording")

    def _resolve_row_target(self, row: dict[str, Any]) -> tuple[str, int] | None:
        """(doc_id, channel) for a saved row, or None when its recording is not open."""
        doc = self._document_for_row(row)
        if doc is None:
            return None
        return (doc.doc_id, self._channel_for_row(doc, row))

    def _document_for_row(self, row: dict[str, Any]) -> Document | None:
        """Resolve the row's document. ``doc_key`` (the persistent key) is AUTHORITATIVE —
        it is what disambiguates a segment from the parent whose name it embeds. Only a row
        without one falls back to matching the display name, and then a tie prefers the
        currently-selected document."""
        key = self._row_text(row, "doc_key")
        if key:
            for doc in self._session.documents():
                if persistent_key(doc) == key:
                    return doc
            return None
        name = self._row_recording(row)
        if not name:
            return None
        matches = [d for d in self._session.documents() if d.display_name == name]
        if not matches:
            return None
        if len(matches) > 1 and self._target is not None:
            for doc in matches:
                if doc.doc_id == self._target[0]:
                    return doc
        return matches[0]

    def _channel_for_row(self, doc: Document, row: dict[str, Any]) -> int:
        """The channel a row was measured on: its recorded INDEX when usable, else the
        channel of that name, else the document's first live current channel."""
        raw = row.get("channel_index")
        idx = UNKNOWN_CHANNEL_INDEX if raw is None or pd.isna(raw) else int(raw)
        if 0 <= idx < len(doc.meta.channels):
            return idx
        name = self._row_text(row, "channel")
        for info in doc.meta.channels:
            if info.name == name:
                return info.index
        for info in doc.meta.channels:
            if info.kind is ChannelKind.CURRENT and not info.is_dead:
                return info.index
        return 0

    def _sync_goto(self) -> None:
        self._goto_button.setEnabled(bool(self._selected_source_rows()))
        self._sync_update_button()

    def _sync_update_button(self) -> None:
        """Enable the explicit overwrite only for a safe, single live target."""
        button = getattr(self, "_update_button", None)
        if button is None:
            return
        enabled = self._selected_update_candidate() is not None
        button.setEnabled(enabled)
        if enabled:
            button.setToolTip(self._update_tip)
            return
        button.setToolTip(
            self._update_tip + "\n\nRequires exactly one selected row and an active mask on "
            "that row's open recording/channel.")

    # -- calibration staleness of saved rows -----------------------------------------------------
    def _refresh_stale_rows(self) -> None:
        """Re-derive which saved rows were measured under a superseded calibration.

        Each row's ``units``/``cal_gain``/``cal_offset`` stamp is compared with the LIVE
        calibration of the channel it was measured on — the manual table's counterpart
        of the analysis panel's ``_check_staleness`` (same rule: mark, never rescale).
        Resolution goes through the same ``doc_key``/``channel_index`` identity that
        navigation uses. A row whose recording is not open, or whose stamp predates the
        stamp columns, CANNOT be compared: per the fingerprint rule it reads as
        "unknown", never as "matches" — and never as "stale" either, which would claim
        knowledge nobody has. The stamps still travel in the CSV either way.

        AND "UNKNOWN" IS NOW A STATE ON THE SURFACE, not only in this loop. It reaches
        the model as ``_ROW_UNCHECKED`` (a ``?`` and a tooltip, no alarm colour), and a
        row that was found STALE while its recording was open keeps its ⚠ as
        ``_ROW_LAPSED`` when the recording closes — the verdict is remembered in
        ``_last_stamp_state`` precisely so that a warning the user was shown is not
        erased by an event that changed nothing about the row."""
        states: dict[int, str] = {}
        seen: set[int] = set()
        for i in range(self._model.rowCount()):
            row = self._model.row_dict(i)
            if row is None:
                continue
            mid_raw = row.get("measurement_id")
            if mid_raw is None or pd.isna(mid_raw):
                continue
            mid = int(mid_raw)
            seen.add(mid)
            doc = self._document_for_row(row)
            if doc is None:
                # NOT JUDGEABLE — and that is a third state, not the absence of one. If
                # the last verdict this row got was "superseded", the mark stays and says
                # it can no longer be refreshed; otherwise it wears "?" for "nobody has
                # checked this". Dropping the marking outright is what made closing a
                # recording erase a warning the user had just been shown, leaving the row
                # drawn exactly like one that was checked and matched.
                states[mid] = (_ROW_LAPSED
                               if self._last_stamp_state.get(mid) == CAL_STAMP_STALE
                               else _ROW_UNCHECKED)
                continue
            ch = self._channel_for_row(doc, row)
            live = self._session.calibration(doc.doc_id, ch)
            live_gain, live_offset = live.fingerprint()
            live_units = (live.target_units
                          or doc.meta.channels[ch].raw_units or "")
            state = calibration_stamp_state(
                row.get("cal_gain"), row.get("cal_offset"), row.get("units"),
                live_gain, live_offset, live_units)
            # Remember the verdict WHILE it can be made, so that when the recording is
            # closed the panel can say which of the two un-judgeable cases this row is in
            # rather than silently forgetting it was ever warned about.
            self._last_stamp_state[mid] = state
            if state == CAL_STAMP_STALE:
                states[mid] = _ROW_STALE
            elif state == CAL_STAMP_UNKNOWN:
                states[mid] = _ROW_UNCHECKED
        # Forget rows that no longer exist (deleted, or a project restore replaced the
        # table): a remembered verdict for an id the table has reused would be a lie.
        self._last_stamp_state = {k: v for k, v in self._last_stamp_state.items()
                                  if k in seen}
        self._model.set_row_states(states)

    # -- on-trace markers -------------------------------------------------------------------------
    def _on_markers_toggled(self, _on: bool) -> None:
        self._refresh_markers()

    def markers_visible(self) -> bool:
        return self._markers_check.isChecked()

    def _clear_markers(self) -> None:
        for attr in ("_span_overlay_id", "_marker_overlay_id"):
            overlay_id = getattr(self, attr)
            if overlay_id is not None:
                self._trace_view.remove_overlay(overlay_id)
                setattr(self, attr, None)

    def _rows_for_target(self) -> list[dict[str, Any]]:
        """Every saved row belonging to the CURRENT (doc, channel), in table order.

        Matched on ``doc_key`` when the row has one, on the display name when it does not
        (a row saved before the column existed) — the same rule the navigation resolver and
        the 'this recording only' filter use."""
        doc = self._target_document()
        if doc is None or self._target is None:
            return []
        channel = self._target[1]
        key = persistent_key(doc)
        name = doc.display_name
        out: list[dict[str, Any]] = []
        for i in range(self._model.rowCount()):
            row = self._model.row_dict(i)
            if row is None:
                continue
            row_key = self._row_text(row, "doc_key")
            if (row_key != key) if row_key else (self._row_recording(row) != name):
                continue
            if self._channel_for_row(doc, row) == channel:
                out.append(row)
        return out

    def _refresh_markers(self) -> None:
        """Redraw the saved-measurement markers for the current channel.

        ONE batched SPAN overlay for the whole set plus ONE MARKER for the selected row —
        the same two renderers (and the same anti-``10k-LinearRegionItem`` design) the
        detection overlays use. Both are added with ``ignoreBounds=True`` by their renderers:
        a full-pane-height item left in autorange ratchets the y-axis wider on every pass and
        crushes the trace to a flat line."""
        self._clear_markers()
        if not self._markers_check.isChecked() or self._target is None:
            return
        doc_id, channel = self._target
        if self._trace_view.pane_for_layer(f"{doc_id}#{channel}") is None:
            return                       # channel not layered; nothing to draw onto
        rows = self._rows_for_target()
        if not rows:
            return
        starts = np.array([int(r["start_idx"]) for r in rows], dtype=np.int64)
        stops = np.array([int(r["end_idx"]) for r in rows], dtype=np.int64)
        current = next((r for r in rows
                        if self._current_measurement_id is not None
                        and int(r["measurement_id"]) == self._current_measurement_id), None)
        try:
            self._span_overlay_id = self._trace_view.add_overlay(
                doc_id, channel,
                OverlayHint(OverlayKind.SPAN, starts=starts, stops=stops,
                            color=_MARK_COLOR, label="saved measurements"))
            if current is not None:
                self._marker_overlay_id = self._trace_view.add_overlay(
                    doc_id, channel,
                    OverlayHint(OverlayKind.MARKER,
                                starts=np.array([int(current["start_idx"])], dtype=np.int64),
                                stops=np.array([int(current["end_idx"])], dtype=np.int64),
                                color=_CURRENT_MARK_COLOR, label="selected measurement"))
        except NotImplementedError:
            # No overlay renderer registered (an embedding that skipped
            # overlays.register_all_renderers). The markers are DECORATION — losing them
            # must never cost the user a measurement — so switch them off and say so.
            self._clear_markers()
            self._markers_check.setChecked(False)
            self._set_status("On-trace markers are unavailable (no overlay renderer).")

    # -- clipboard --------------------------------------------------------------------------------
    def copy_selected_to_clipboard(self) -> str:
        """Copy the selected rows (header + values, TSV) — one number into a lab notebook
        must not require exporting the whole pooled CSV. Returns the text copied.

        Emitted in the order the rows are DISPLAYED, not the order they were measured in:
        the user sorted the table for a reason, and a paste that silently reshuffles is a
        paste you have to re-check by hand."""
        rows = self._selected_source_rows_in_view_order()
        if not rows:
            return ""
        df = self._model.df
        header = "\t".join(df.columns)
        lines = [header]
        for r in rows:
            row = self._model.row_dict(r) or {}
            lines.append("\t".join(
                "" if (v := row.get(c)) is None or pd.isna(v) else str(v)
                for c in df.columns))
        text = "\n".join(lines)
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._set_status(f"Copied {len(rows)} measurement(s) to the clipboard.")
        return text

    def _on_scope_toggled(self, on: bool) -> None:
        self._proxy.set_scoped(bool(on))
        self._sync_goto()

    def export_csv(self, path: str | None = None) -> Path | None:
        """Write every listed measurement to CSV. With ``path`` None, ask via a dialog.

        33 columns: ``MEASUREMENT_SCHEMA``'s 32, unchanged and in their frozen order, then
        the additive ``signal_basis`` stamp last — see :func:`write_measurements_with_basis`
        for why an export that drops it hands back exactly the ambiguity the column exists
        to remove."""
        if len(self._model.df) == 0:
            if path is None:
                QtWidgets.QMessageBox.information(
                    self, "Export measurements",
                    "No measurements to export yet — use “Add to list” first.")
            return None
        if path is None:
            chosen = ask_export_path(
                self, title="Export measurements", suggestion="measurements.csv",
                filters="CSV (*.csv);;All files (*)")   # writes exactly this file
            if chosen is None:
                return None
            path = str(chosen)
        try:
            out = write_measurements_with_basis(self._model.df, path)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            # Same report MainWindow's exports give in the identical spot ("Could not
            # write the export" + the real cause — a PermissionError names the folder).
            # This used to escape the Qt slot: traceback on the console, no dialog, no
            # status change, no file — and a user free to close the app believing the
            # hand-measured table was saved.
            QtWidgets.QMessageBox.warning(
                self, "Export measurements", f"Could not write the export:\n{exc}")
            self._set_status("Export failed — no CSV was written.")
            return None
        self._set_status(f"Wrote {len(self._model.df)} measurement(s) to {out.name}.")
        return out

    # -- getting the PROCESSED SAMPLES out ---------------------------------------------------
    #: How each non-current effective kind is named in the export refusal. Spelled here
    #: rather than from the enum value so the sentence reads like the Calibration tab's
    #: own combo and not like a field name.
    _KIND_PHRASE: Mapping[str, str] = MappingProxyType({
        ChannelKind.VOLTAGE.value: "a VOLTAGE channel",
        ChannelKind.RAW_VOLTAGE.value: "raw ADC volts",
        ChannelKind.CONDUCTANCE.value: "a CONDUCTANCE channel",
        ChannelKind.UNKNOWN.value: "of UNDECLARED kind",
    })

    def _would_write_header(self) -> str:
        """The CSV's second column header this export would produce for the target now.

        Built the same way :func:`~nanoscope.analysis.conditioning.write_processed_trace`
        builds it, from the same :func:`processed_units_label`, so the refusal can quote
        the exact false string rather than describe it — ``current_mV`` is an argument and
        "the units may be wrong" is not. Cheap: :meth:`_units` reads the label off the
        calibration and never materialises the array.
        """
        units = processed_units_label(self._units(), self._active_conditioning())
        return "current_" + units.replace(" ", "_") if units else "current"

    def _non_current_export_refusal(self) -> str:
        """``''`` when the target may be exported as a processed CURRENT trace.

        The kind is :meth:`~nanoscope.core.session.Session.effective_kind` — the user's
        Calibration-tab override first, the file header only as a fallback — because
        that is the one resolver every other surface in this app asks, and a door that
        asked ``meta.channels[ch].kind`` instead would refuse a channel the user had just
        corrected to *current* (the exact contradiction ``effective_kind`` exists to end).

        A DEAD current channel is NOT refused here: "this channel has no signal" is a
        fact about the data that ``meta.issues`` already reports, and the samples it
        exports are honestly labelled. This gate is only about the label being true.
        """
        if self._target is None or not self._session.has_document(self._target[0]):
            return ""
        doc_id, channel = self._target
        try:
            kind = self._session.effective_kind(doc_id, int(channel))
        except (KeyError, ValueError, IndexError):
            return ""
        if kind is ChannelKind.CURRENT:
            return ""
        phrase = self._KIND_PHRASE.get(kind.value, f"a {kind.value} channel")
        try:
            name = self._session.document(doc_id).meta.channels[int(channel)].name
        except (KeyError, ValueError, IndexError, TypeError):
            name = f"channel {channel}"
        # KEPT SHORT ON PURPOSE. This sentence is also what `_set_status` prints, and that
        # label is the last, word-wrapped widget in the panel: every extra line it takes
        # comes off the measurements table above it. It stays inside the length of the
        # gate sentences that were already here; the full reasoning is in the button
        # tooltip and this method’s docstring, neither of which is on the layout.
        return (f"“{name}” is {phrase}, not a current trace — this export would head its "
                f"column “{self._would_write_header()}”. Pick the current channel, or set "
                f"“Treat this channel as: current” on the Calibration tab.")

    def processed_export_gate(self) -> str:
        """``''`` when *Export processed trace…* can run; otherwise the reason it cannot.

        A gate string, not a bool, because the caller that needs it most is a menu item:
        an action greyed with no explanation is the failure this app already fixed once
        for the detection actions. Returned sentences are complete and name the fix.

        IT REFUSES A CHANNEL THAT IS NOT A CURRENT. ``write_processed_trace`` builds its
        second column header as ``"current_" + units`` — that is the file's contract and
        it is not this panel's to change — so pointed at the owner's IN14 (a command
        voltage, ``mV``) this door wrote a column headed ``current_mV`` and a sidecar
        whose ``units`` said ``mV`` under ``"export": "processed_trace"``. Every field in
        that file is individually true and the file as a whole is a lie: a reader pooling
        it with real exports gets millivolts in a picoamp column. The header cannot be
        made honest from here, so the honest move is the refusal, and the refusal names
        both ways out (pick the current channel, or say on the Calibration tab that this
        one IS the current — the same ``effective_kind`` override every other surface in
        the app already honours).
        """
        if self._target is None or not self._session.has_document(self._target[0]):
            return "Open a recording and select a channel first."
        kind_refusal = self._non_current_export_refusal()
        if kind_refusal:
            return kind_refusal
        if not conditioning_enabled(self._active_conditioning()):
            return ("No filter stage is enabled on the Preprocess tab and no stored "
                    "detection on this channel ran under one, so there is no processed "
                    "trace to export — the raw channel is already on disk.")
        if self._target_sample_rate() is None:
            return ("This recording has no sample rate, so the filter stages (all "
                    "designed in Hz and seconds) cannot run.")
        return ""

    def export_processed_trace(self, path: str | None = None, *,
                               whole: bool = False) -> dict[str, Path] | None:
        """Write the CONDITIONED samples to CSV + a ``.meta.json`` provenance sidecar.

        The audit's one-way door: until now the processed trace could be SEEN and nothing
        else — no ``to_csv``/``savetxt``/``np.save`` anywhere in ``gui/`` — so a user
        could filter, detrend and normalize, watch it happen, and then not get the numbers
        out. This is the door.

        WHAT IS WRITTEN: ``time_s`` (from the recording's first sample, so a slice still
        says where it came from) and the conditioned current in the channel's units, one
        row per sample, plus a sidecar carrying the exact parameter dict, the chain in
        words, the rate, the units, the sample range and the recording's instrument-clock
        origin — enough to rebuild the array with
        :func:`~nanoscope.analysis.conditioning.conditioned_trace` and to put the time
        base back on the experiment's clock.

        WHAT IS EXPORTED: the cursor selection when the Measure tool is bracketing one,
        otherwise the whole recording (``whole=True`` forces the whole one). Both are
        named in the sidecar's ``span`` and in the status line, because "processed trace"
        alone would be ambiguous — and because a whole 15 M-sample recording is a ~250 MB
        CSV, which is a thing to ask for rather than to receive by accident.

        THE CHAIN RUNS ON THE WHOLE RECORDING AND IS SLICED AFTERWARDS, never the other
        way round: every stage here is a filter with memory (filtfilt, block medians,
        SavGol windows), so conditioning a 2-second cut out of the middle would produce
        different numbers at its edges from the ones on screen. The exported samples are
        the same array the preview draws and the Measure tool measures.

        Returns ``{"data": …, "meta": …}``, or None (gate refused / dialog cancelled /
        write failed — each of which is reported where the user is looking)."""
        gate = self.processed_export_gate()
        if gate:
            if path is None:
                QtWidgets.QMessageBox.information(self, "Export processed trace", gate)
            self._set_status(f"Nothing exported — {gate[0].lower()}{gate[1:]}")
            return None
        doc_id, channel = self._target                      # gate proved both are live
        doc = self._session.document(doc_id)
        chan = self._session.calibrated(doc_id, channel)
        raw = np.asarray(chan.data)
        fs = float(chan.sample_rate_hz)
        params = self._active_conditioning() or {}

        bounds = self._trace_view.measure_bounds() if self.is_active() else None
        if whole or bounds is None or int(bounds[1]) - int(bounds[0]) < 2:
            start, stop, span = 0, int(raw.size), "whole recording"
        else:
            start, stop = int(bounds[0]), int(bounds[1])
            span = "cursor selection"

        try:
            with self._busy(int(raw.size), "Applying the processing chain…"):
                data = conditioned_trace(raw, fs, params)
        except Exception as exc:  # noqa: BLE001 — a kernel refusal is a message, not a crash
            QtWidgets.QMessageBox.warning(
                self, "Export processed trace",
                f"The processing chain could not be applied to this recording:\n{exc}")
            self._set_status("Export failed — the processing chain did not run.")
            return None

        if path is None:
            chosen = ask_export_path(
                self, title="Export processed trace",
                suggestion=f"{Path(doc.display_name).stem}_processed.csv",
                filters="CSV (*.csv);;All files (*)",
                # TWO files land, so the overwrite question has to be asked about both.
                outputs=processed_trace_export_paths)
            if chosen is None:
                return None
            path = str(chosen)
        # THE COLUMN IS NAMED AFTER THE UNIT THE SAMPLES ACTUALLY HAVE. `chan.units` is
        # what they had BEFORE the chain, and with `normalize` in it that is no longer
        # true: the header would read `current_pA` over z-scores. See
        # `processed_units_label`.
        units = processed_units_label(chan.units, params)
        # THE CALIBRATION THE CHAIN RAN UNDER, so the sidecar is a RECIPE and not just a
        # description. `conditioned_trace` runs on the CALIBRATED channel, never on the
        # file, so `params` alone cannot rebuild these samples from `doc_key`; the writer
        # has carried `cal_gain`/`cal_offset` for exactly this and this caller was passing
        # neither, so every sidecar this door had ever written said `null` for both and a
        # reader's only recourse was to assume gain 1. On an UNCALIBRATED channel that
        # assumption happens to be right and the file still could not say so — "not
        # recorded" and "identity" are different facts, and now (1.0, 0.0) states the
        # second one. `fingerprint()` is the same pair `MEASUREMENT_SCHEMA` stamps on a
        # measured row, so an exported trace and a row measured on it agree by
        # construction.
        cal_gain, cal_offset = self._session.calibration(doc_id, channel).fingerprint()
        try:
            out = write_processed_trace(
                path, data[start:stop], fs,
                units=units, start_idx=start,
                t0_s=clock.instrument_origin_s(doc.meta), params=params,
                recording=doc.display_name, doc_key=persistent_key(doc),
                channel=doc.meta.channels[channel].name, channel_index=int(channel),
                span=span, cal_gain=cal_gain, cal_offset=cal_offset)
        except Exception as exc:  # noqa: BLE001 — same report every other export gives
            QtWidgets.QMessageBox.warning(
                self, "Export processed trace", f"Could not write the export:\n{exc}")
            self._set_status("Export failed — no file was written.")
            return None
        # The status line names the unit too, because the one place a user can catch a
        # mislabelled export before it reaches their analysis is here, while they still
        # remember which chain was on. "no declared units" is a state, not a unit: the
        # column is then just `current` and the file claims nothing.
        self._set_status(
            f"Wrote {stop - start:,} processed samples ({span}) "
            f"{('in ' + units) if units else 'with no declared units'} "
            f"to {out['data'].name}, with the chain in {out['meta'].name}.")
        return out

    def _sync_processed_export(self) -> None:
        """Keep the *Export processed trace…* tooltip describing what a click does NOW.

        The button never greys (see its construction), so the tooltip is where the current
        state is said: the gate's own sentence when the export cannot run — the same words
        the click would put in a dialog — and what will actually be written when it can."""
        # The basis row is built BEFORE the button row and syncs itself on the way past,
        # so this can legitimately run once with no button to describe yet; the button's
        # own construction calls it again.
        if getattr(self, "_export_processed_button", None) is None:
            return
        gate = self.processed_export_gate()
        self._export_processed_button.setToolTip(
            f"Not available yet — {gate[0].lower()}{gate[1:]}" if gate else
            "Write the conditioned samples — the same array the preview curve draws — to "
            "CSV, plus a .meta.json sidecar carrying the chain, the sample rate, the "
            "units and the sample range. The cursor selection when the cursors bracket "
            "one, otherwise the whole recording; the sidecar and the status line say "
            "which.")

    # -- enablement ------------------------------------------------------------------------------
    def _update_enabled(self) -> None:
        ok = self.can_measure()
        self._toggle.setEnabled(ok)
        self._sync_processed_export()   # a new target can open or close the gate
        if not ok and self._toggle.isChecked():
            self._toggle.setChecked(False)
        self._sync_snaps()
        self._sync_mark_missed()
        self._clear_mask_btn.setEnabled(bool(ok and self.is_active()))
        self._sync_update_button()

    def _sync_snaps(self) -> None:
        ok = self.can_measure()
        self._snap_view_btn.setEnabled(ok)
        self._snap_event_btn.setEnabled(
            ok and self._selected_event is not None
            and self._selected_event[:2] == self._target)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)
