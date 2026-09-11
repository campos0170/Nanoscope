from __future__ import annotations

import os
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.reconcile import CONSENSUS_CURATION_CONFLICT_COL, CONSENSUS_MODULE_ID
from ..core.history import MAX_BYTES, MAX_DEPTH, EventHistory, MeasurementHistory
from ..core.identity import same_source, source_of
from ..core.layer import ColourRegistry, ViewMode
from ..core.analysis_workspace import (
    ANALYSIS_PANEL_ARTIFACT_ID,
    DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
    COMPARISON_FIGURES_ARTIFACT_ID,
    EVENT_PLOTS_ARTIFACT_ID,
    IV_ARTIFACT_ID,
    OWNED_ARTIFACT_IDS,
    PLOTS_ARTIFACT_ID,
    PSD_ARTIFACT_ID,
    SUBPOPULATION_ARTIFACT_ID,
    VOLTAGE_DEPENDENCE_ARTIFACT_ID,
    artifact_source_recipe,
    capture_analysis_workspace,
    resolve_result_recipe,
    result_recipe,
    restore_analysis_workspace,
)
from ..core.project import AnalyticalWorkspace
from ..core.project_guard import (
    ProjectDiskStamp,
    ProjectSaveConflict,
    guarded_project_write,
    locked_project_read,
    project_changed,
    project_disk_stamp,
    project_save_lock,
)
from ..core.session import Session
from ..core.types import SOFT_SAMPLE_CEILING, ChannelKind, SampleRange
from .analysis_launcher import (
    SURFACE_EVENT_PLOTS,
    SURFACE_FIGURES,
    SURFACE_GALLERY,
    SURFACE_IV,
    SURFACE_ORDER,
    SURFACE_POPULATIONS,
    SURFACE_PSD,
    SURFACE_VDEP,
    AnalysisLauncher,
)
from .analysis_explorer import (
    AnalysisExplorer,
    PAGE_DISTRIBUTIONS,
    PAGE_DISTRIBUTION_SUMMARIES,
    PAGE_COMPARISON_FIGURES,
    PAGE_EVENT_WAVEFORMS,
    PAGE_IV,
    PAGE_NOISE,
    PAGE_SUBPOPULATIONS,
    PAGE_VOLTAGE_SERIES,
)
from .analysis_panel import AnalysisPanel
from .analysis_catalog import ANALYSIS_LABELS, PAGE_HOME
from .analysis_worker import AnalysisController, BatchDetectController
from ..core import clock
from .calibration_panel import CalibrationPanel
from .export_paths import ask_export_path
from .file_dock import FileDock
from .file_info_panel import FileInfoPanel
from .navigator import Navigator
from .pan_scrollbar import PanScrollBar
from .reconcile_dialog import ReconcileDialog, can_reconcile_sources
from .status_chip import CalibrationChip
from .trace_view import TraceView, effective_kind
from .worker_thread import (
    JOB_ANALYSIS,
    blocking_job,
    drain_worker_threads,
    refusal_text,
)
from .workers import LoadController
from .workflow_rail import WorkflowStrip
from .workflow_status import Stage, compute_status

__all__ = ["MainWindow"]

_OPEN_FILTER = ("Recordings (*.abf *.h5 *.hdf5 *.tdms *.csv *.txt);;"
                "Axon ABF (*.abf);;HDF5 (*.h5 *.hdf5);;TDMS (*.tdms);;"
                "Text trace (*.csv *.txt);;All files (*)")
_MAX_RECENT = 10

#: How long :meth:`MainWindow._teardown` will wait for a detached worker on quit.
#: Every kernel polls the cooperative cancel token, so a job that CAN stop stops in
#: milliseconds and never sees this number; it only ever bites on an uninterruptible
#: C-level call (``ml_detect._import_tf`` measured at 4.1 s). 1000 ms was too short to
#: cover any of those and long enough to be felt, which is the worst of both — 2500 ms
#: covers the realistic ones, and ``gui.app.main`` handles the rest without a crash.
_TEARDOWN_DRAIN_MS = 2500

#: Why the ML actions are off on a machine without TensorFlow — VERBATIM the sentence
#: the DetectionPanel's gate label shows (panels.DetectionPanel._refresh_ml_enabled),
#: because the menu and the panel are two surfaces for the same fact and may never
#: disagree about it, down to the wording.
_ML_TF_MISSING = ("TensorFlow is not installed, so ML detection is unavailable. "
                  "Install tensorflow (2.15) to enable it.")

# The three answers to the unsaved-work close prompt (see MainWindow.closeEvent /
# _confirm_close_choice). Kept as plain strings so a test can select one without a modal.
_CLOSE_SAVE = "save"
_CLOSE_DISCARD = "discard"
_CLOSE_CANCEL = "cancel"

# The startup recovery prompt's three deliberate outcomes.  Plain strings keep the UI
# decision injectable: the integration suite can exercise every branch without opening a
# native modal, exactly as the close guard does above.
_RECOVERY_RECOVER = "recover"
_RECOVERY_DISCARD = "discard"
_RECOVERY_SHOW = "show"


@dataclass(frozen=True)
class _RecoverySnapshot:
    """One immutable hand-off from the GUI thread to the recovery writer."""

    revision: int
    state: Any
    source_project_path: str | None
    covered_explicit_save_revision: int


def _write_recovery_snapshot(store, snapshot: _RecoverySnapshot,
                             _revision: int | None = None):
    """Qt-free worker callable; it closes over the store, never a MainWindow."""
    return store.write(
        snapshot.state,
        revision=int(snapshot.revision),
        source_project_path=snapshot.source_project_path,
        covered_explicit_save_revision=int(snapshot.covered_explicit_save_revision),
    )

#: How a detector's ``module_id`` is spelled in a sentence the user reads. Used by the
#: Edit menu to say WHICH other table holds the undo history it does not have here.
_DETECTOR_LABELS = {
    "detect.schmitt": "the threshold detector",
    "detect.cusum": "the CUSUM detector",
    "detect.keras": "the ML detector",
    CONSENSUS_MODULE_ID: "the reconciled event table",
}

_RECONCILED_TRACE_ANALYSIS_REFUSAL = (
    "The Reconciled table can contain rows from different detector preprocessing "
    "chains, so one waveform cannot be reconstructed honestly for every row. Switch "
    "to a source detector table and run this analysis there."
)

# ================================================================================================
# THE SURFACE-STALENESS REGISTRY — every signal that changes what a surface window draws
# ================================================================================================
#
# WHY THIS IS A TABLE AND NOT FIVE ``.connect`` LINES. ``_refresh_surface_windows`` was
# wired to four signals — ``resultsChanged``, ``documentClosed``, ``measurementsChanged``
# and the measurement stack's ``changed`` — while SIX others changed what those windows
# draw. Measured on the owner's ``_0006_`` recording, Populations open on a dwell
# histogram with "use accepted events only" and an exponential fit: rejecting 159 events
# left tau at 0.01429 s / n = 316; the Refresh button turned it into 0.008773 s / n = 158.
# tau overstated 63%, n wrong by 100%, no staleness marking anywhere, and the exported PNG
# byte-identical to the pre-curation figure. The data was never stale —
# ``discover_plot_sources`` passes the frame BY REFERENCE — only the drawn scene and the
# cached fit were.
#
# The four connections were not wrong; the SHAPE was. A hand-maintained list of
# connections is only ever as complete as the last person to remember it, and
# ``labelsChanged`` (added with ``analysis/labels.py`` this week) is the proof: it joined
# the app and the wiring did not follow. So the set is declared ONCE, here, the wiring is
# derived from it (:meth:`MainWindow._wire_surface_refresh`), and
# ``tests/gui/test_surface_staleness.py`` fails the day any signal declared on any of
# these classes is in NEITHER table. A seventh signal cannot be added silently: it has to
# be classified, in writing, before the suite is green again.
#
# Keys are ROLES, not attribute paths, because two of these objects are built lazily when
# the user first opens their window (see _open_event_inspector / _open_event_gallery) and
# one is owned by the panel — the wiring happens in three places, off one table.
SURFACE_MUTATION_SIGNALS: dict[str, tuple[str, ...]] = {
    # A detection appears, is replaced, merged or cleared; a recording opens or closes
    # (Populations reads across EVERY open recording, and the I-V / V-dep windows gain
    # and lose a point with the document); a gain is edited, which re-scales the
    # calibrated signal that PSD, I-V and V-dep all measure.
    "session": ("resultsChanged", "documentOpened", "documentClosed", "calibrationChanged"),
    # One event's accept/reject, and one event's curator flag — both re-decide which rows
    # an "accepted only" histogram, density map or ensemble average is built from.
    "analysis_panel": ("eventAcceptanceChanged", "eventFlagChanged"),
    # The manual-measurement table: an add, a delete or an annotation edit...
    "measure_panel": ("measurementsChanged",),
    # ...and its undo stack, which installs snapshots through the deliberately silent
    # ``set_measurements_df`` door and so never reaches ``measurementsChanged``.
    "measure_history": ("changed",),
    # Accept-all / Reject-all rewrite the whole accepted column; a mask drag moves a span
    # and every measured column with it; a label write changes the ``flag`` token set that
    # a per-label split is grouped by.
    "event_inspector": ("curationBatchChanged", "eventEdited", "amplitudeEdited", "labelsChanged"),
    "event_gallery": ("curationBatchChanged", "labelsChanged"),
}

#: The other side of the ledger: every signal these same objects declare that does NOT
#: reach the refresher, and the reason. Being on this list is a CLAIM, and the guard test
#: exists so the claim has to be made out loud rather than by omission.
SURFACE_NEUTRAL_SIGNALS: dict[str, str] = {
    # -- selection and navigation: they move the view, never the numbers -----------------
    "analysis_panel.eventSelected": "selection only — no surface window plots the selection",
    "event_inspector.eventSelected": "selection only; relayed to the panel's table + trace",
    "event_gallery.eventSelected": "selection only; relayed to the panel's table + trace",
    "event_gallery.eventActivated": "double-click 'go to this event in the trace' — a view move",
    "event_gallery.inspectRequested": "'Open in Event Inspector' — the tile's double-click, its "
                                      "context menu and the Inspector… button all raise it; "
                                      "opens a window and selects, never the numbers",
    "event_inspector.galleryRequested": "the 'Gallery…' button — opens a window, a view move",
    "event_inspector.traceRequested": "the 'Show in trace' button / T — centres the main trace "
                                      "on the browsed event; a view move, never the numbers",
    "measure_panel.navigateRequested": "'go to measurement' on another document — a view move",
    "measure_panel.markMissedRequested": "a REQUEST to add an event; the add itself lands "
                                         "on resultsChanged",
    "measure_panel.measureActiveChanged": "the measure tool's on/off state, not its table",
    "analysis_panel.willReplaceEvents": "a detection run has STARTED and the table it will "
                                        "overwrite has to be recoverable — wired to "
                                        "``_checkpoint_everything`` (undo), never to the "
                                        "refresher: at the instant it fires no frame has "
                                        "moved yet, and the replacement itself arrives on "
                                        "``resultsChanged``, which is wired above",
    "event_gallery.undoRequested": "⌘Z / the Undo button INSIDE the gallery — a request to "
                                   "trigger MainWindow's own Edit ▸ Undo, which is the one "
                                   "undo in this application. Whatever that undo restores "
                                   "reaches every surface on the history's own signals; the "
                                   "request itself moves nothing",
    "event_gallery.saveProjectRequested": "a request to run MainWindow's authoritative "
                                           "project save; emitting it changes no analysis",
    "event_gallery.saveProjectAsRequested": "a request to run MainWindow's authoritative "
                                             "Save As flow; emitting it changes no analysis",
    "event_inspector.saveProjectRequested": "a request to run MainWindow's authoritative "
                                             "project save; emitting it changes no analysis",
    "event_inspector.saveProjectAsRequested": "a request to run MainWindow's authoritative "
                                               "Save As flow; emitting it changes no analysis",
    "event_inspector.mergeWithNextRequested": "a REQUEST; the merge itself lands on "
                                              "resultsChanged",
    "measure_history.stepRecorded": "a step landed on the undo stack — bookkeeping for "
                                    "Edit ▸ Undo/Redo (see _note_step); the TABLE change "
                                    "that caused it reaches the refresher through "
                                    "``changed``, which is wired above",
    # -- status / chrome ----------------------------------------------------------------
    "analysis_panel.baselineJobStatus": "progress text for the baseline probe",
    "analysis_panel.runRefused": "the run gate declining a start — nothing ran, nothing moved",
    "analysis_panel.stalenessChanged": "the shown result was re-judged for staleness — wired "
                                       "separately to _push_stale_columns (S7-POP1), which "
                                       "pushes the table's badge tuple + the banner sentence "
                                       "onto the gallery and inspector; no frame moved",
    "analysis_panel.mergeFinished": "the OUTCOME of an off-thread 'Merge all', for the "
                                    "status line. When it installed a merged table it did "
                                    "so through Session.add_result, so the frame change "
                                    "reaches every surface on ``resultsChanged``, which is "
                                    "wired above; the other outcomes (started, busy, "
                                    "dropped, failed, noop) changed no frame at all",
    # -- relayed: they DO change the frame, and they reach the refresher through a signal
    #    that is already in the table above. Wiring them again would be harmless (the
    #    refresher is idempotent) but would hide where the guarantee actually comes from.
    "event_inspector.acceptanceChanged": "relayed — AnalysisPanel.set_event_accepted goes "
                                         "through the table model, which emits "
                                         "eventAcceptanceChanged",
    "event_gallery.acceptanceChanged": "relayed — same door as the inspector's",
    # -- refusals: nothing was written, so no surface window's DATA moved. The three
    #    refusal notes read the shared ledger, and these are wired to the repaint-only
    #    hop (_on_gallery_acceptance_refused / _on_inspector_acceptance_refused,
    #    mirroring the table's own acceptanceRefused hop) — never to the full refresher.
    "event_inspector.acceptanceRefused": "no-write refusal — repaint-only hop for the "
                                         "three refusal notes; no frame changed",
    "event_gallery.acceptanceRefused": "no-write refusal — same repaint-only hop",
    # -- session signals about the VIEW, not the data ------------------------------------
    "session.cursorMoved": "the trace cursor",
    "session.viewModeChanged": "stacked/overlaid trace layout",
    "session.layersChanged": "which traces are drawn in the main viewer",
}

#: The two tables Edit ▸ Undo can act on. There is ONE Undo in this application and there
#: will not be a second: a user who has just deleted four measurement rows reaches for
#: Ctrl+Z, not for a different Ctrl+Z. The two stacks are separate because their payloads
#: are (``core/history.py``); which one a press acts on is decided by RECENCY, through the
#: journal in :meth:`MainWindow._note_step`.
_UNDO_EVENTS = "events"
_UNDO_MEASUREMENTS = "measurements"

# The two extra answers to the "this recording has segments cut from it" close prompt (see
# _confirm_close_with_segments); _CLOSE_CANCEL above is the third. Same plain-string discipline.
_CLOSE_SEGMENTS_TOO = "segments_too"
_CLOSE_PARENT_ONLY = "parent_only"

# Opening widths for the two docks on a FIRST launch (see _apply_default_dock_widths). They
# are deliberate defaults, NOT floors — both docks stay draggable well below and above these,
# and the moment the user settles on her own widths the saved windowState takes over.
# Left column: a little above the file manager's natural 293 px, so the tree's name column
# starts with room to grow rather than at its minimum. (It cannot start much wider without
# taking from the viewer; the tree's Δt column claims a fixed ~148 px whatever the dock's
# width, which is what really caps how much of a recording name is visible.)
_FILE_DOCK_DEFAULT_WIDTH = 320
# Right column: the Inspector's own children are all scroll areas now, so their combined
# sizeHint is an accident of whichever control stack happens to be widest rather than a
# considered default. This is the considered one — and it is the FLOOR of the opening
# width, not the width itself: see _apply_default_dock_widths, which opens the dock wide
# enough to SHOW its widest page (capped) rather than wide enough to hide half of it.
#
# WHY THIS STOPPED BEING ONE NUMBER. 460 opened the dock at a viewport of 420 px against
# an Analysis control column whose natural width is 836, so the page came up with a
# horizontal scrollbar and everything past x=420 — the protocol Save as…/Rename/Delete
# buttons, the ML preset buttons, Null control…, Recovery map…, Merge all — was off the
# right-hand edge of a 1512 px screen with no menu item and no shortcut behind it. A
# scrollbar is not a door: five features were reported missing that were only off-screen.
_INSPECTOR_DEFAULT_WIDTH = 460
# ...and the ceiling. The Inspector may not open so wide that the signal viewer — the thing
# this application is for — stops being the widest column. On the 1512 px display this is
# measured against, the three columns and their two separators are
# 320 (file) + 12 (separators) + inspector + central = 1512, so an Inspector of 560 leaves
# the viewer 620: wider than the Inspector, and wider than it than the file manager.
# MEASURED with a fresh profile at the width the dock actually opens at (502, below): the
# Analysis control column lays out in its 462 px viewport with NO horizontal scrollbar, and
# every control the audit found off-screen (protocol Save as…/Rename/Delete, the ML preset
# pair, Detect / Sweep threshold… / Null control… / Recovery map…, Merge all) sits whole
# inside the window.
# Above this the dock stays freely draggable — nothing here is a minimum or a maximum SIZE,
# both docks still drag from ~137 px to the whole window — it is only where they OPEN.
#
# THIS IS A CEILING, NOT THE WIDTH. The dock used to open AT it, because the derivation
# below asked each page for its *natural* width (the Analysis control column's sizeHint is
# 646) and every page wants more than the cap, so the cap always won. Whether a control is
# reachable is decided by the page's MINIMUM, not by its natural width, and the minimums are
# much smaller. Asking for the minimum instead (see _inspector_default_width) opens the dock
# at 502 px on this tree and hands the other 58 px back to the signal viewer — 678 px at
# 1512, the widest column by 176 px. The ceiling now only bites if a page's MINIMUM grows
# past it, which is the case it was written for.
_INSPECTOR_MAX_DEFAULT_WIDTH = 560
#: Frame + vertical scrollbar between a dock's own width and the viewport its page is
#: laid out in — measured (dock 460 -> viewport 420) and stable across the three pages.
_INSPECTOR_CHROME_WIDTH = 40
#: ...and one further nesting level of the same, for a page that lives inside a SECOND
#: scroll area rather than directly in the dock's own. Measured at dock 560, as
#: dock width minus viewport width: 6 px for a page sitting straight in the dock,
#: 40 for the Analysis control column (one scroll area in), 64 for the Manual-measurement
#: panel (two in). A nested page hides its minimum from its parent — that is why the dock
#: cannot simply ask the tab widget — so the deepest allowance is the one to reserve.
_INSPECTOR_NESTED_CHROME_WIDTH = 24
#: One dock separator. Two of them stand between the three columns: measured at 1512, where
#: 320 (file) + 620 (viewer) + 560 (inspector) = 1500 leaves exactly 12 px.
_DOCK_SEPARATOR_WIDTH = 6


def _with_project_suffix(path: str) -> str:
    """A chosen project path with the mandatory ``.nanoscope`` suffix.

    Shared by the overwrite check and the write, so the file the user is warned about
    is the file that gets replaced. Keeping the two spellings apart is how "Save As"
    over an existing ``study.nanoscope``, typed as ``study``, could destroy it with no
    warning at all.
    """
    return path if path.lower().endswith(".nanoscope") else path + ".nanoscope"


def _restorable_bytes(settings: QtCore.QSettings, key: str, ignored: list[str]) -> object | None:
    """A preferences value ``restoreGeometry``/``restoreState`` can actually be handed.

    *** THE PREFERENCES STORE IS EXTERNAL INPUT, LIKE ANY OTHER FILE ON DISK. ***

    ``QSettings.value`` returns whatever the backing store decodes to, and neither the
    plist (the shipping ``NativeFormat`` backend) nor an .ini can promise a QByteArray:
    a store the app wrote itself and that was later TRUNCATED — a crash or a full disk
    mid-quit, a half-synced or partially restored ``~/Library`` — hands back a plain
    ``str``, and so does ``defaults write … windowState -string x`` or any plist editor.
    Both Qt calls raise ``TypeError`` on that, inside ``MainWindow.__init__``, which
    ``gui.app.main`` does not guard — so the app did not start AT ALL, on every launch,
    with no message and no in-app way back: the user had to know to delete
    ``~/Library/Preferences/com.nanoscope-local.NanoScope.plist`` by hand.

    A damaged preference costs the user their saved layout. It must never cost them the
    program. So a value of the wrong kind is IGNORED and its key recorded in ``ignored``,
    which :meth:`MainWindow._restore_settings` says out loud rather than swallowing —
    the same reasoning ``recentFiles``/``recentProjects`` two lines below already got,
    and the same policy ``_open_project`` states for the other externally-mutable file
    ("a corrupt archive must not crash the app").
    """
    raw = settings.value(key)
    if raw is None:
        return None
    if isinstance(raw, (QtCore.QByteArray, bytes, bytearray, memoryview)):
        return raw
    ignored.append(key)
    return None


def _settings_store_damaged(s: QtCore.QSettings) -> bool:
    """Did the preferences store fail to parse? Asked so it can be SAID (see
    :meth:`MainWindow._restore_settings`) — a store that drops keys wholesale makes the
    app come up factory-fresh, and that is a change the user sees.

    ``status()`` alone is not a reliable witness: Qt parses a settings file once per
    process and serves every later ``QSettings`` from a cache that has forgotten the
    parse error (measured: first instance ``FormatError``, second ``NoError``, same
    damaged file). So when ``status()`` says nothing, the file is copied to a fresh path
    — which cannot be served from the cache — and parsed again; THAT instance's status
    is the file's real health. Best-effort in the honest direction: a copy that cannot
    be made (unreadable file, full temp disk) reports "not damaged", because a disclosure
    built on a guess is the false alarm that teaches the user to ignore the real one.
    """
    if s.status() != QtCore.QSettings.NoError:
        return True
    path = Path(s.fileName())
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / path.name
            shutil.copyfile(path, copy)
            fresh = QtCore.QSettings(str(copy), s.format())
            damaged = fresh.status() != QtCore.QSettings.NoError
            del fresh
        return damaged
    except OSError:
        return False


def _on_screen(window: QtWidgets.QWidget | None) -> bool:
    """Is this auxiliary window built AND actually on screen?

    The predicate :meth:`MainWindow._refresh_surface_windows` re-feeds by. ``is not None``
    only answers "was it ever built": closing one of these windows hides it, it does not
    destroy it, so the two questions come apart the moment a user closes a window — which
    is exactly when re-feeding it stops buying anything and starts costing latency on
    every curation keystroke. ``isVisible()`` is False for a hidden or closed window and
    True for a minimized one (which the user CAN restore without passing through an
    opener), so it is the right question on both ends.
    """
    return window is not None and window.isVisible()


def _string_setting(settings: QtCore.QSettings, key: str, fallback: str,
                    ignored: list[str]) -> str:
    """A preferences value that is supposed to be a plain string, or ``fallback``.

    Same reasoning as :func:`_restorable_bytes`, one severity down: a ``lastDir`` that
    comes back as a list cannot crash anything, but ``str(["/a"])`` is not a directory
    and the next Open dialog would start in a place that does not exist, with nobody
    ever told why. Wrong kind -> the default, and the key is named.

    CONTENT is checked too, not only kind. Store damage does not always break the file's
    grammar: a truncated-then-scribbled ini parsed back a ``lastDir`` of
    ``'/Users/somebody/heal \\x00\\xff[broken'`` with ``status() == NoError`` — a str,
    happily adopted, silently pointing the next Open dialog at gibberish. This app never
    writes a control character into these keys (they hold real filesystem paths), so one
    coming BACK is the same statement as a wrong type: a value this app did not write.
    """
    raw = settings.value(key, fallback)
    if isinstance(raw, str) and not any(ord(ch) < 0x20 or ch == "\x7f" for ch in raw):
        return raw
    if raw is not None:
        ignored.append(key)
    return fallback


def _stored_baseline_fingerprint(provenance: object) -> str | None:
    """The open-pore estimator stamp a restored result carries, or ``None``.

    ``None`` for a table that never had one -- a pre-change project, or a detector
    handed a baseline array by its caller, which can name no estimator at all.
    Absence is the signal the staleness check reads, so a malformed value must
    become absence rather than a stamp that can never match anything.
    """
    if not isinstance(provenance, Mapping):
        return None
    fp = provenance.get("baseline_fingerprint")
    return str(fp) if isinstance(fp, str) and fp else None


class _InfoBar(QtWidgets.QFrame):
    """A thin, dismissible, non-blocking banner (used for the duplicate-representation advisory)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet(
            "QFrame{background:#2a2f36; border:1px solid #3a4048; border-radius:4px;}"
            "QLabel{border:none;}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 6, 6)
        self._label = QtWidgets.QLabel()
        self._label.setWordWrap(True)
        lay.addWidget(self._label, 1)
        close = QtWidgets.QToolButton()
        close.setText("✕")
        close.setAutoRaise(True)
        close.clicked.connect(self.hide)
        lay.addWidget(close)

    def show_message(self, text: str) -> None:
        self._label.setText(text)
        self.setVisible(True)


class _RangeDialog(QtWidgets.QDialog):
    """Pick a [start, stop) sample window for an over-ceiling file."""

    def __init__(self, n_samples: int, rate_hz: float | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load a sample range")
        form = QtWidgets.QFormLayout(self)
        self._start = QtWidgets.QSpinBox()
        self._stop = QtWidgets.QSpinBox()
        for sb in (self._start, self._stop):
            sb.setRange(0, n_samples)
            sb.setGroupSeparatorShown(True)
        self._start.setValue(0)
        self._stop.setValue(min(SOFT_SAMPLE_CEILING, n_samples))
        form.addRow("Start sample", self._start)
        form.addRow("Stop sample", self._stop)
        self._hint = QtWidgets.QLabel()
        self._hint.setStyleSheet("color:#8b9199;")
        form.addRow(self._hint)
        self._rate = rate_hz
        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._start.valueChanged.connect(self._update_hint)
        self._stop.valueChanged.connect(self._update_hint)
        self._update_hint()
        form.addRow(self._buttons)

    def _update_hint(self) -> None:
        """Restate the chosen window — and GATE OK on it being a window at all.

        The two spin boxes are independent, so an empty range is not a perverse input: it
        is what filling them in the natural order produces. The dialog opens at
        ``start=0, stop=SOFT_SAMPLE_CEILING``; a user who wants samples 30,000,000–45,000,000
        types the start first, at which point stop is still 15,000,000 and the range is
        backwards. OK stayed enabled, ``range()`` returned None, and the caller took the
        same bare ``_finish_active()`` branch as Cancel — the file simply did not open, with
        no document, no status line and no dialog, on a recording the user had committed to
        opening through two prompts. Indistinguishable from a click that missed.

        Disabling OK is the fix that needs no words: the hint already reads "0 samples · 0 s",
        and now the button agrees with it. ``range()`` therefore cannot be None while the
        dialog is accepted, which is what makes the caller's Cancel branch honest again.
        """
        n = max(0, self._stop.value() - self._start.value())
        if self._rate:
            self._hint.setText(f"{n:,} samples  ·  {n / self._rate:.3g} s")
        else:
            self._hint.setText(f"{n:,} samples")
        ok = self._buttons.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setEnabled(n > 0)
        ok.setToolTip("" if n > 0 else
                      "Stop sample has to be past Start sample — this range holds no "
                      "samples, so there would be nothing to load.")

    def range(self) -> SampleRange | None:
        start, stop = self._start.value(), self._stop.value()
        return SampleRange(start, stop) if stop > start else None


class MainWindow(QtWidgets.QMainWindow):
    """The application shell. Constructs Session and every panel, and wires each panel to Session
    (and to this window), never to one another. This is the ONLY gui/ module permitted to import
    other panels.
    """

    projectRestored = QtCore.Signal()   # emitted once an Open Project restore fully settles
    # One authoritative answer for every editing window. The empty path means no project
    # has been named yet; ``unsaved`` is the SAME visible predicate used by the title, not
    # the raw dirty bit (an empty, disposable session must never claim it has work to save).
    projectSaveStateChanged = QtCore.Signal(str, bool)

    def __init__(self, parent=None, *, recovery_root=None,
                 recovery_enabled: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("NanoScope")
        self.resize(1440, 880)
        self.setAcceptDrops(True)
        #: True while the OPENING dock widths are still provisional. ``resize`` above is a
        #: REQUEST, not a width: on a 1280 px display the window comes up 1280 wide. Every
        #: layout number derived before the window is shown is therefore derived against
        #: 1440 whatever the screen — which is how the Inspector's ceiling came to be
        #: unreachable from the app's own launch path. See :meth:`showEvent`.
        self._opening_layout_pending = False

        self._session = Session()
        # THE UNDO DOOR. One stack over every EventSet mutation in the session (see
        # core/history.py for why it is a snapshot stack and not a QUndoStack). It is
        # driven only from here: MainWindow is the one object that already receives all
        # four mutation announcements, so no panel needs to know undo exists.
        self._history = EventHistory(self._session, self)
        # ...and the SAME rule over the hand-measurement table, which is not an EventSet
        # and so cannot live in the stack above (core/history.py explains why it is a
        # sibling rather than a sixth key). Deleting a hand measurement was the last
        # destructive click in the application that nothing could take back.
        self._measure_history = MeasurementHistory(
            lambda: self._analysis_panel.measure_panel.measurements_df,
            lambda df: self._analysis_panel.measure_panel.set_measurements_df(df),
            self)
        #: Which stack each undoable step landed on, oldest first — the only thing that
        #: decides which one a plain Ctrl+Z acts on. See :meth:`_note_step`.
        self._undo_journal: list[str] = []
        self._redo_journal: list[str] = []
        self._controller = LoadController(self)
        self._analysis_controller = AnalysisController(self._session, self)
        # Batch detection runs on its OWN controller (never the panel's), so a
        # whole-series run never collides with an interactive single detection.
        self._batch_controller = BatchDetectController(self._session, self)
        self._batch_skipped = 0                  # recordings skipped this batch (no current/rate)
        #: The progress-bar label of the analysis/batch job currently running, exactly as
        #: its started-slot painted it. The bar is SHARED chrome: a load that finishes
        #: mid-run has to hand it back to the surviving job, and a sweep started without
        #: progress ticks will never repaint it on its own — see _release_load_chrome.
        self._busy_job_format: str = ""
        self._current_selection: tuple[str, int] | None = None
        self._recent: list[str] = []
        self._recent_projects: list[str] = []
        self._last_dir: str = str(Path.home())
        self._last_project_dir: str = str(Path.home())
        self._project_path: str | None = None   # the .nanoscope currently being edited
        self._dirty: bool = False   # unsaved analysis a project save would persist (see closeEvent)
        # Stage 2 crash recovery needs a REVISION, not another interpretation of the dirty
        # Boolean.  Every mutation that reaches `_mark_dirty` advances this clock.  A save
        # and a recovery snapshot each remember the exact revision they covered, so a late
        # edit cannot be retired merely because an older write happened to finish.
        self._session_revision: int = 0
        self._explicit_saved_revision: int = 0
        self._recovery_revision: int = 0
        self._recovery_enabled = bool(recovery_enabled)
        self._recovery_root = recovery_root
        self._recovery_store = None
        self._recovery_controller = None
        self._recovered_candidate = None
        self._recovery_suggested_project_path: str | None = None
        self._startup_recovery_checked = False
        self._restore_is_recovery = False
        # The top-level window that initiated the current synchronous project-save flow.
        # Gallery/Inspector emit intent only; MainWindow still owns every chooser, warning,
        # conflict check and write. Remembering the initiator keeps native macOS sheets in
        # front of the window where the user pressed Save.
        self._project_save_requester: QtWidgets.QWidget | None = None
        self._plots_window = None   # the single, lazily-built PlotsWindow
        self._distribution_summary_window = None
        self._comparison_figures_window = None
        self._iv_window = None   # the single, lazily-built IVWindow
        self._vdep_window = None   # the single, lazily-built VDepWindow
        # One top-level home for the analytical surfaces above/below.  The individual
        # attributes remain the authoritative page objects for compatibility and for
        # the v11 workspace adapters; the Explorer only organises their presentation.
        self._analysis_explorer: AnalysisExplorer | None = None
        self._first_analysis_guide = None
        self._figure_library = None
        self._event_inspector = None   # the single, lazily-built EventInspectorWindow
        self._event_gallery = None     # the single, lazily-built EventGalleryWindow
        #: ``(accept, n_shown, n_events)`` while one of the gallery's "all shown" buttons is
        #: mid-click, else None — the one thing its ``curationBatchChanged`` cannot say.
        #: See :meth:`_arm_gallery_whole_frame`.
        self._gallery_whole_frame: tuple[bool, int, int] | None = None
        #: ONE refusal ledger per detection LINEAGE — (doc, channel, module, lineage
        #: token) -> (token, weak ref to the frame, id->start_idx anchors, ledger,
        #: retired start_idx->bucket). The record of which events an
        #: accept was refused over is a fact about the DETECTION, not about the window the
        #: click landed in, so it lives here and is handed to the gallery, the inspector
        #: AND the events table (see :meth:`_refusal_ledger_for`): all three disclose one
        #: record, a window opened after the gesture shows the standing note, and the
        #: re-feeds an undo or a merge provoke no longer silence it. It dies with the
        #: lineage — the recording closing — never with a re-feed. The token is IN the
        #: key: with one slot per (doc, channel, module), resolving an OLD lineage's
        #: restored frame (Edit ▸ Undo past a re-detection) evicted the NEW lineage's
        #: record, so the redo brought the frame back with its rows still refused and
        #: the standing disclosure silently gone (measured). Bounded exactly as before:
        #: pruned on documentClosed (see :meth:`_drop_refusals_for_doc`).
        self._event_refusals: dict[tuple[str, int, str, str | None],
                                   tuple[Any, Any, Any, Any, Any]] = {}
        self._subpop_window = None     # the single, lazily-built SubpopReportWindow
        #: The detection a subpopulation run is in flight FOR. Writing it declares the run
        #: pending (``_subpop_pending`` is derived from it) and says what the answer will
        #: be about, because those are one fact — see the property below.
        self._subpop_target = None
        # Exact persistent source/result identity captured while an asynchronous report's
        # source is unquestionably live.  The report may land after that recording closes;
        # a launch-local doc_id is never an acceptable fallback in a project artifact.
        self._subpop_source_recipe_pending: dict[str, Any] | None = None
        #: event_id -> subpopulation id from the last finished report, once it has been pushed
        #: onto the trace and the gallery, plus what it is ABOUT. Held here and not on either
        #: surface because MainWindow is the only thing that has both the report and the
        #: detection, and therefore the only thing that can tell when the map has gone stale.
        self._subpop_assignment: dict[int, int] | None = None
        self._subpop_source: str = ""
        self._subpop_caveats: tuple[str, ...] = ()
        #: The finished report's OWN sentence about what it found — ``SubpopReport.headline()``
        #: verbatim, taken at adoption because the report object does not outlive it. Every
        #: surface here that speaks about the assignment repeats this; none of them counts
        #: classes for itself (see :meth:`_colour_trace_by_subpops`).
        self._subpop_headline: str = ""
        #: The events frame the assignment was built against, by IDENTITY. A re-detect, a
        #: merge or a re-measure makes a NEW frame, and the same event_id in it is a
        #: different event; re-attaching the map by id would silently describe one
        #: detection's events with another detection's mixture. The same identity test the
        #: gallery refresher already uses to tell "the frame was edited" from "there is a
        #: new frame". ONE new frame does not mean a new detection, and is re-bound rather
        #: than refused: Edit ▸ Undo/Redo of a plain curation installs a SNAPSHOT COPY with
        #: the same ids, spans and depths and only ``accepted``/``flag`` moved — the map
        #: survived the forward curation and is exactly as valid on the copy (X6; see
        #: :meth:`_is_curation_only_change`, which asks ``describe_change``, the same
        #: author that names the step in the Edit menu).
        self._subpop_frame = None
        #: The detrended array the last run was measured on, offered to the gallery as the
        #: trace the owner's notebook actually draws. One extra full-length float array for
        #: the recording; dropped with the assignment.
        self._subpop_signal = None
        #: The same array while the run is still in flight. Handed over on the terminal
        #: signal, never before: a run that is cancelled or fails must not leave a
        #: full-length copy of the recording alive behind it.
        self._subpop_signal_pending = None
        #: WHICH detection the assignment was adopted on — ``(doc_id, channel, module_id)``,
        #: the key the ResultStore files that result under. With the frame identity above
        #: this is what lets the map be KEPT while the user looks at another recording and
        #: re-offered when the same frame comes back: sweep 7 measured that merely
        #: clicking another file row (three recordings open, the owner's own project)
        #: threw a finished mixture away, because the frame anchor was nulled together
        #: with the map and the "same frame" branch could never fire again. Retired only
        #: when THIS detection's own frame is replaced by something that is not the same
        #: events — a re-detect, a merge, a mask edit, a re-measure, an undo past the
        #: detection, a project open — or its recording closes; the same rule the CUSUM
        #: level fit follows in _sync_export_action. The module id is IN the key because
        #: the store keeps one result per detector: running CUSUM (or the ML detector, or
        #: Schmitt over an ML result) on the same channel leaves THIS detector's stored
        #: result untouched, so nothing is retired and the map is parked "away" with a
        #: sentence that names both detectors (C1 — the previous wording told the user to
        #: re-select a recording that was already selected).
        self._subpop_key: tuple[str, int, str] | None = None
        #: The rows the run was HANDED (``report.n_fired``: the events accepted at run
        #: time), so the report window can say when the accepted set has moved under a
        #: standing report (S7-POP4). Curation edits the shared frame in place and the
        #: map deliberately survives it; this is the number that lets the window state
        #: that its fractions describe the run's set and not the current one.
        self._subpop_n_fired: int | None = None
        #: The SET of event_ids the run was HANDED — the set behind ``_subpop_n_fired``,
        #: not just its size. The curation note (S7-POP4) compares the accepted set NOW
        #: with this one, so a swap that keeps the count (reject one event, accept another
        #: back) is still disclosed: measured on 0004, the count read equal on exactly that
        #: swap and cleared the one line that exists to say the run's fractions describe a
        #: different set (C2). ``None`` when the frame carried no ``event_id``/``accepted``
        #: column, and then the note falls back to the count so it is never falsely silent.
        self._subpop_accepted_ids: frozenset[int] | None = None
        #: ...and the same set while the run is still in flight, taken at LAUNCH off the
        #: very ``EventSet`` the job was handed. It is snapshotted there and not rebuilt at
        #: adoption because ``n_fired`` — the other half of the same claim — counts the rows
        #: as they were at launch: a curation made DURING the run (the status line invites
        #: it; nothing gates the File dock) otherwise landed inside the adoption-time
        #: snapshot and the note compared the current set with itself. Measured on 0004:
        #: reject one event while the mixture flies and the report lands "n_fired 110", 109
        #: accepted, standing line "" (C2 close; the re-verifier's probe_R1/R3).
        self._subpop_accepted_ids_pending: frozenset[int] | None = None
        #: The Event gallery's chosen trace ("Detrended (as detected)") and its
        #: representative-sample flag, while the held map is on it. Leaving the recording
        #: resets both — ``set_events`` drops the map and the sample with it, and
        #: withdrawing the detrended trace falls the box back to the calibrated trace — and
        #: the re-offer on return used to leave both reset, silently changing what the
        #: tiles were drawn from (C4). Written ONLY by
        #: :meth:`_remember_gallery_subpop_view` (which refuses a gallery that has already
        #: been reset) and re-applied by :meth:`_restore_gallery_subpop_view` when the map
        #: arrives on a gallery that does not have it.
        self._subpop_gallery_trace: str | None = None
        self._subpop_gallery_representative: bool = False
        #: Why there is NO assignment although the report fitted a mixture — "" while an
        #: assignment stands or when the report found nothing to colour with. Set when
        #: the map is retired (and by a refused adoption), read by the two report-window
        #: buttons so they say the TRUE reason rather than "fitted no mixture" (S7-POP3).
        self._subpop_absent_reason: str = ""
        #: The label palette, owned HERE and handed to every surface that writes labels, so
        #: the digit that means "apo-ferritin" in the gallery means it in the inspector too.
        #: Persisted with the project (see _gather_project_state).
        from ..analysis.labels import LabelVocabulary as _LV
        self._label_vocab = _LV()
        #: The CUSUM level fit, and the events frame it was fitted on. Both None until the
        #: user explicitly runs Analysis ▸ Fit sub-state levels; the frame is kept purely
        #: as an IDENTITY anchor, so a fit can be dropped the moment the detection under it
        #: is replaced (see _invalidate_level_fit).
        self._level_fit = None
        self._level_fit_frame = None
        #: What a SINGLE-EVENT re-fit needs, cached beside the fit by `_fit_levels`. See
        #: `_refit_levels_for_event`: a mask drag retires the edited event's levels, and
        #: putting them back means re-running the same estimator over the new span. The
        #: expensive input is the open-pore BASELINE TRACK (measured 644 ms for 2 M samples),
        #: which is why it is held rather than re-derived on the GUI thread; the conditioned
        #: signal is NOT held, because rebuilding it costs 2.1 ms on the same recording and a
        #: second full-length float64 copy of every open recording is not worth 2 ms.
        #:
        #: ``None`` is the COLD case and is load-bearing: a fit restored from a project file
        #: is adopted into `_level_fit` with no inputs behind it, and re-deriving them would
        #: put a 644 ms baseline estimate inside a mouse-release handler. Cold means "keep
        #: today's drop, and say so".
        self._level_fit_inputs = None
        self._level_fit_pending = None
        self._event_plot_window = None   # the single, lazily-built EventPlotWindow
        self._psd_window = None   # the single, lazily-built PSDWindow
        # Stage 3 analytical state belongs to the PROJECT, not to whichever optional
        # top-level widget happens to be alive.  Widgets are deliberately lazy and may
        # blank themselves on close; this cache is therefore the durable in-memory copy.
        self._analytical_workspace = AnalyticalWorkspace()
        self._figure_project_epoch = 0  # Lifecycle identity, unchanged by ordinary checkpoints.
        # Built-in artifacts adopted from a project are installed only when their surface
        # is opened.  Keeping the ids (rather than opening the windows) is what prevents a
        # v11 project from producing a wall of plots at load time.
        self._workspace_pending: set[str] = set()
        # A restored numerical snapshot is frozen until the user explicitly edits/runs/
        # refreshes that surface.  Session navigation must not silently recompute it.
        self._workspace_frozen: set[str] = set()
        # Adapter restore writes ordinary Qt controls.  Even user-only signals are not a
        # promise across every Qt widget, so this guard makes installing a project a
        # reconstruction rather than a new dirty revision.
        self._workspace_installing = False
        self._workspace_navigating = False
        self._workspace_default_analysis = None
        self._workflow_status = None   # last computed PipelineStatus (drives the strip routing)

        # Open pipeline state: opening several files is SEQUENTIAL through the one controller.
        self._open_queue: list[Path] = []
        self._active_open: Path | None = None

        # Project-restore state: reopening a .nanoscope drives the SAME sequential open
        # pipeline; per-doc calibration/layers/results are applied as each doc lands and
        # the whole thing is finalised (measurements + view mode) when the queue drains.
        self._restoring: bool = False
        self._restore_state = None                              # core.project.ProjectState
        self._restore_ranges: dict[str, SampleRange | None] = {}
        self._restore_missing: list[str] = []          # source FILES that moved/vanished on disk
        # ...and, for the ones that failed at the READER rather than the pre-flight
        # existence check, the reader's own sentence. `_restore_missing` alone cannot
        # tell the two apart, and the banner used to call both "could not be found":
        # a file whose saved window selects nothing in a re-exported shorter recording,
        # a file replaced by a CPYA-variant ABF, and a file whose permissions deny
        # reading are three different problems with three different fixes, and all
        # three were present on disk the whole time. path -> the refusal it gave.
        self._restore_refusals: dict[str, str] = {}
        self._restore_lost_segments: list[str] = []    # segments with no parent to re-derive from
        self._restore_bad_calibrations: list[str] = []  # stored gain/offset a channel cannot take
        self._restore_sources: dict[str, Any] = {}     # path -> saved SourceFingerprint | None
        self._restore_notes: list[str] = []            # ready-written sentences for the banner
        self._restore_channel_moved: list[str] = []    # records re-attached by channel identity
        self._restore_channel_lost: list[str] = []     # ...and ones whose channel is gone
        #: Recordings whose records were re-attached by FILE identity because the project's
        #: stored path spelling is no longer how the file resolves (a moved folder with a
        #: symlink left behind). See :meth:`_apply_restore_for_doc`'s ``_belongs``.
        self._restore_path_moved: list[str] = []
        #: Recordings this restore never attempted because the user pressed Cancel — the
        #: eighth way a restore can fall short, and the only one that is not a failure of
        #: the project or the disk. See :meth:`_cancel_load`.
        self._restore_cancelled: list[str] = []
        #: ``id()`` of every record from the opened project that a restore call ACTUALLY
        #: installed into the live session. The carry is the complement of this set — see
        #: :meth:`_capture_unrestorable`, which is why this is a stamp at the install and
        #: not a list of remembered failure reasons.
        self._restored_records: set[int] = set()
        # Everything in the OPENED project that could not be loaded, kept verbatim so the
        # next save writes it back out. See _capture_unrestorable.
        self._carried_over: dict[str, list] | None = None
        # (project path, sentence) already shown for a save. A symlinked project is a STANDING
        # arrangement, so its note is true on every Ctrl+S — and a modal that fires on every
        # save is the thing that teaches a user to dismiss modals unread, which is how the
        # warning that matters gets missed. Said once per file per session; a note that has
        # not been said before (a hard link created since) still interrupts.
        self._disclosed_file_notes: set[tuple[str, str]] = set()
        # What the current project file LOOKED LIKE ON DISK when this window last read or
        # wrote it — resolved path + size + mtime + SHA-256, or None before any project is
        # held. The digest matters: preserving size and restoring the old mtime must not
        # disguise changed curation. This is the concurrent-writer guard's whole memory:
        # multi-instance is a
        # supported shape (packaging/nanoscope_app.py keeps its log lock alive precisely so
        # a later instance can run), so two windows on one .nanoscope were silent
        # last-writer-wins — window A saved a rejection, window B pressed Save, and A's
        # committed curation left the file with zero dialogs while both status bars claimed
        # their save was the file. See _project_changed_on_disk / _write_project.
        self._project_disk_stamp: ProjectDiskStamp | None = None

        self._build_central()
        self._build_docks()
        self._build_actions_and_menus()
        self._build_status_bar()
        self._wire()
        self._restore_settings()
        self._init_recovery()

    # -- construction ----------------------------------------------------------------------------
    def _build_central(self) -> None:
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Top strip, full width: the workflow breadcrumb (non-closable chrome, wired in _wire).
        # It used to be a fixed 172 px column down the left-hand side; six one-word stages do
        # not need a column, and the one they had now belongs to the file manager.
        self._workflow_strip = WorkflowStrip()
        lay.addWidget(self._workflow_strip)

        # Directly under it: the analysis launcher — the ANALYZE stage expanded into the six
        # surfaces it actually means. Before this row, the density map, ensemble, noise
        # spectrum, I-V fit and voltage-dependence series had NO representation anywhere in the
        # visible layout; they lived only in a flat menu that on macOS is detached from the
        # window, and the tooltips written to explain them were set on QActions, where a native
        # menu never renders them. Its buttons stay visible (greyed) when their surface is not
        # available yet, which is the whole point: a feature you cannot use is still a feature
        # you should know exists. Wired in _wire (the actions do not exist yet at this point).
        self._analysis_launcher = AnalysisLauncher()
        lay.addWidget(self._analysis_launcher)

        self._info_bar = _InfoBar()
        lay.addWidget(self._info_bar)

        self._trace_view = TraceView(self._session)
        lay.addWidget(self._trace_view, 1)

        # Directly under the trace, ABOVE the navigator, so it reads as belonging to the plot it
        # pans rather than to the overview map. Hidden until the user zooms in; its height is
        # reserved permanently so crossing that threshold never resizes the trace (see the
        # module docstring). Bound like the Navigator: two-way, guarded, view-is-truth.
        self._pan_scrollbar = PanScrollBar()
        self._pan_scrollbar.bind(self._session, self._trace_view)
        lay.addWidget(self._pan_scrollbar)

        self._navigator = Navigator()
        self._navigator.bind(self._session, self._trace_view)
        lay.addWidget(self._navigator)

        self.setCentralWidget(central)

    def _build_docks(self) -> None:
        # The LEFT column is the file manager, and only the file manager. It is a real
        # QDockWidget, so the user drags its inner edge to resize it and closes it (View ▸
        # Panels, and its own ✕) to collapse it — the space the old fixed-width workflow rail
        # used to occupy is now hers to spend on recordings, segments and channels.
        self._file_dock = FileDock(self._session)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._file_dock)

        self._calibration_panel = CalibrationPanel(self._session)
        self._file_info_panel = FileInfoPanel(self._session)
        # AnalysisPanel meets the live viewer; it is handed the SAME TraceView the
        # central widget shows, so its overlays/preview/zoom land on the real panes.
        self._analysis_panel = AnalysisPanel(
            self._session, self._analysis_controller, self._trace_view)
        tabs = QtWidgets.QTabWidget()
        # INVARIANT for every Inspector tab: nothing in here may impose a width floor. A
        # child's minimum width propagates through the tab widget into
        # QDockWidget::minimumSize(), which is exactly the bound QDockAreaLayout refuses to
        # drag a separator past — one un-shrinkable control stack and the user can no longer
        # make the signal viewer wider. So each page is a widgetResizable QScrollArea with an
        # explicit zero minimum: content keeps its natural width and scrolls horizontally
        # instead of pushing back. (The Analysis page wraps its own sub-panels the same way;
        # see analysis_panel._scroll_wrap.)
        cal_scroll = QtWidgets.QScrollArea()
        cal_scroll.setWidgetResizable(True)
        cal_scroll.setMinimumWidth(0)
        cal_scroll.setWidget(self._calibration_panel)
        # Kept so the chip / strip can reveal the Calibration tab by widget (reorder-proof).
        self._cal_scroll = cal_scroll
        info_scroll = QtWidgets.QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setMinimumWidth(0)
        info_scroll.setWidget(self._file_info_panel)
        tabs.addTab(cal_scroll, "Calibration")
        tabs.addTab(info_scroll, "File Info")
        # The analysis panel owns its own scrolling (inner scroll areas + splitter) so the
        # event table keeps its native scrollbar; add it directly, not wrapped.
        tabs.addTab(self._analysis_panel, "Analysis")
        # Let the tab bar scroll rather than widen the dock once the three labels stop fitting.
        tabs.setUsesScrollButtons(True)
        tabs.setMinimumWidth(0)
        # THE TAB THE APPLICATION OPENS ON. It used to be tab 0 (Calibration) by nothing
        # more than insertion order, which put the whole detector — every control the app
        # exists to drive, and the event table — behind a tab that is only visited when a
        # gain is wrong. Measured on a fresh profile at 1512x900: at launch not one
        # detection control was laid out at all, so "is the Detect button reachable" had
        # the answer "it does not have a geometry yet". Analysis is the page the work
        # happens on; Calibration keeps its own door (its tab, the calibration chip, and
        # _reveal_calibration_panel, which is what the workflow strip triggers when the
        # gain really is the next thing to fix).
        #
        # Set HERE, at construction, before the window is shown and before anything is
        # connected: choosing the opening tab is not a reveal, and nothing may reselect a
        # tab under a user who is already aiming at one.
        tabs.setCurrentWidget(self._analysis_panel)
        self._inspector_tabs = tabs

        self._inspector = QtWidgets.QDockWidget("Inspector", self)
        self._inspector.setObjectName("InspectorDock")
        self._inspector.setWidget(tabs)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._inspector)

    def _build_actions_and_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        act_open = file_menu.addAction("&Open…")
        act_open.setShortcut(QtGui.QKeySequence.Open)
        act_open.triggered.connect(self._open_dialog)

        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        act_open_project = file_menu.addAction("Open &Project…")
        act_open_project.setToolTip(
            "Reopen a saved .nanoscope project: its documents, calibrations, layers, "
            "detections (with curation) and manual measurements are all restored.")
        act_open_project.triggered.connect(self._open_project_dialog)

        self._recent_projects_menu = file_menu.addMenu("Open Recent Pro&ject")
        self._rebuild_recent_projects_menu()

        act_save_project = file_menu.addAction("&Save Project…")
        act_save_project.setShortcut(QtGui.QKeySequence.Save)   # Ctrl+S
        act_save_project.setToolTip(
            "Save the whole analysis to a .nanoscope project file so nothing is lost on "
            "relaunch. Re-saves the current project in place once it has a path.")
        act_save_project.triggered.connect(self._save_project)

        act_save_project_as = file_menu.addAction("Save Project &As…")
        act_save_project_as.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        act_save_project_as.triggered.connect(self._save_project_as)

        file_menu.addSeparator()
        # Create a derived recording (segment) from the current on-trace Measure selection. Gated
        # (in _sync_create_segment_action) on an active recording + a non-empty region; the twin
        # of the Files-dock "Segment" button, both routed here so MainWindow stays the sole wirer.
        self._act_create_segment = file_menu.addAction("Create &segment from selection…")
        self._act_create_segment.setToolTip(
            "Crop the current on-trace selection (turn on Measure and drag a region) into a new "
            "derived recording in the Files list, analysed independently of the parent.")
        self._act_create_segment.setEnabled(False)
        self._act_create_segment.triggered.connect(self._create_segment_from_selection)

        file_menu.addSeparator()
        act_close = file_menu.addAction("&Close document")
        act_close.setShortcut(QtGui.QKeySequence.Close)
        act_close.triggered.connect(self._close_current)

        file_menu.addSeparator()
        act_quit = file_menu.addAction("&Quit")
        act_quit.setShortcut(QtGui.QKeySequence.Quit)
        act_quit.triggered.connect(self.close)

        # ``&Edit`` exists for exactly one thing: walking back an event-table mutation.
        # Its two actions are the visible half of core/history.py — everything else in
        # this application is either non-destructive or guarded by a confirmation.
        edit_menu = menubar.addMenu("&Edit")
        self._act_undo = edit_menu.addAction("Undo")
        # The STANDARD sequence (Cmd+Z on macOS, Ctrl+Z elsewhere), at the default
        # WindowShortcut context. That is safe next to the detection form's spin boxes
        # and line edits: Qt sends a ShortcutOverride first and QLineEdit accepts it for
        # the editing keys, so typing Cmd+Z in a parameter field still undoes the typing
        # rather than rolling back the event table (verified).
        self._act_undo.setShortcut(QtGui.QKeySequence.Undo)
        self._act_undo.setEnabled(False)
        self._act_undo.triggered.connect(self._undo_events)
        self._act_redo = edit_menu.addAction("Redo")
        self._act_redo.setShortcuts([QtGui.QKeySequence.Redo,
                                     QtGui.QKeySequence("Ctrl+Shift+Z")])
        self._act_redo.setEnabled(False)
        self._act_redo.triggered.connect(self._redo_events)
        # The menu text is the ONLY place this can explain itself: a disabled QAction
        # never receives a click, and a tooltip on a native macOS menu item never
        # renders. So the reason goes IN the label (see _sync_undo_actions), and it is
        # recomputed on every open in case a table was edited without announcing it.
        edit_menu.aboutToShow.connect(self._on_edit_menu_about_to_show)
        self._edit_menu = edit_menu

        view_menu = menubar.addMenu("&View")
        self._act_overlay = view_menu.addAction("&Overlay")
        self._act_stacked = view_menu.addAction("&Stacked")
        mode_group = QtGui.QActionGroup(self)
        for act, mode in ((self._act_overlay, ViewMode.OVERLAY),
                          (self._act_stacked, ViewMode.STACKED)):
            act.setCheckable(True)
            mode_group.addAction(act)
            act.triggered.connect(lambda _checked=False, m=mode: self._choose_view_mode(m))
        self._sync_view_actions()

        view_menu.addSeparator()
        act_back = view_menu.addAction("Zoom &Back")
        act_back.setShortcuts([QtGui.QKeySequence("Ctrl+Left"), QtGui.QKeySequence("Alt+Left")])
        act_back.setToolTip("Return to the previous view — reject an event and keep scanning.")
        act_back.triggered.connect(self._trace_view.view_back)

        act_forward = view_menu.addAction("Zoom &Forward")
        act_forward.setShortcuts([QtGui.QKeySequence("Ctrl+Right"), QtGui.QKeySequence("Alt+Right")])
        act_forward.triggered.connect(self._trace_view.view_forward)

        act_reset = view_menu.addAction("&Reset zoom")
        act_reset.setShortcut(QtGui.QKeySequence("Ctrl+0"))
        act_reset.setToolTip(
            "Zoom all the way back out to the full trace. The Full button below the plot "
            "does the same; Home also works while the trace itself has focus.")
        act_reset.triggered.connect(self._trace_view.reset_zoom)
        # HOME IS SCOPED TO THE TRACE, NOT TO THE WINDOW.
        #
        # A QAction shortcut defaults to Qt::WindowShortcut, and a window shortcut outruns
        # the focused widget unless that widget accepts the ShortcutOverride Qt sends first.
        # QLineEdit and QAbstractSpinBox do — which is the reasoning the Ctrl+Z comment
        # above records, and event_gallery's for the label digits — but QAbstractItemView
        # has NO ShortcutOverride handler at all, so a window-wide Home was taken from every
        # list and table in the window unconditionally. Measured: with the File Manager tree
        # focused on its last row, Home left the row where it was and instead snapped the
        # trace from a 0.044 s window back to the full 2.2 s, silently; a bare QTreeWidget
        # given the identical keystroke moves to the first row. The tree's own Home
        # navigation was perfectly good — it simply never received the key.
        #
        # So the binding lives on the TRACE VIEW with WidgetWithChildrenShortcut: it fires
        # when the thing Home would reset is the thing that has focus, and every item view
        # keeps the standard "go to the top" key. Ctrl+0 and the visible Full button still
        # reset the zoom without taking that key away from tables.
        self._home_reset_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Home"), self._trace_view)
        self._home_reset_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._home_reset_shortcut.activated.connect(act_reset.trigger)

        # Event-scale raw-trace scanning. These keys belong to the TRACE only: a window-wide
        # Shift+Arrow would steal selection/range gestures from tables, editors and the separate
        # Event Inspector. The visible buttons in the Scan strip are the beginner-facing door;
        # these focus-scoped twins make a long manual-labeling pass a keyboard loop.
        self._annotation_navigation_shortcuts: list[QtGui.QShortcut] = []
        for sequence, slot in (
                ("Shift+Left", self._pan_scrollbar.previous_view),
                ("Shift+Right", self._pan_scrollbar.next_view),
                ("Shift+Up", self._pan_scrollbar.zoom_in),
                ("Shift+Down", self._pan_scrollbar.zoom_out)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self._trace_view)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
            self._annotation_navigation_shortcuts.append(shortcut)

        # The pan bar's one-click sibling: keep the zoom, move the selection to the middle. Same
        # selection every other tool uses (the Measure region), same widget doing the panning, so
        # it can never disagree with the scrollbar the user could have dragged instead.
        self._act_centre_selection = view_menu.addAction("&Centre selection in view")
        self._act_centre_selection.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        self._act_centre_selection.setToolTip(
            "Slide the view (without changing the zoom) so the Measure selection sits in the "
            "middle — the keyboard twin of dragging the pan bar under the trace.")
        self._act_centre_selection.setEnabled(False)
        self._act_centre_selection.triggered.connect(self._centre_selection_in_view)

        self._act_boxzoom = view_menu.addAction("Drag draws &zoom box")
        self._act_boxzoom.setCheckable(True)
        self._act_boxzoom.setChecked(True)   # default: drag a box around an event to dive in
        self._act_boxzoom.setToolTip(
            "On: left-drag frames a region and zooms into it (dive inside an event). "
            "Off: left-drag pans. The mouse wheel moves the signal vertically; use −/+ "
            "in the Scan strip to change the time scale.")
        self._act_boxzoom.toggled.connect(self._trace_view.set_drag_mode)

        view_menu.addSeparator()
        panels = view_menu.addMenu("&Panels")
        panels.addAction(self._file_dock.toggleViewAction())
        panels.addAction(self._inspector.toggleViewAction())

        # ``&Analysis`` used to be a flat ~14-item list in which the five plot surfaces were
        # items 10-14 — a dumping ground you had to read end-to-end to find anything. It is now
        # sectioned along the way the work actually runs: DETECT (make events) · CURATE (judge
        # them) · EXPLORE (look at this recording) · COMPARE (look across recordings) · EXPORT
        # (take it away). Every action, shortcut and behaviour is unchanged — this is ordering
        # and separators only.
        analysis_menu = menubar.addMenu("&Analysis")
        self._analysis_menu = analysis_menu

        analysis_menu.addSection("Detect")
        self._act_run_detection = analysis_menu.addAction("&Run detection")
        self._act_run_detection.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        self._act_run_detection.setToolTip(
            "Detect translocation events on the selected current channel "
            "(Schmitt trigger). Requires a known sample rate.")
        self._act_run_detection.setEnabled(False)
        self._act_run_detection.triggered.connect(self._run_detection)

        # Batch detection: the SAME Schmitt params over every open recording at once,
        # off-thread with progress + cancel. Schmitt-only on purpose — running a Keras
        # model over many full recordings is far too slow to batch.
        self._act_batch_detect = analysis_menu.addAction(
            "&Batch detect (all open recordings)…")
        self._act_batch_detect.setToolTip(
            "Run the current Schmitt detection parameters on every open recording that "
            "has a current channel and a known sample rate, in the background. Detect "
            "the whole -1000..-200 mV series in one pass, then plot the voltage "
            "dependence. (ML detection is not offered here — it is too slow to batch "
            "over many full recordings.)")
        self._act_batch_detect.setEnabled(False)
        self._act_batch_detect.triggered.connect(self._batch_detect)

        # ML detector: load a runtime .keras model, then run it on the selected
        # current channel. Its events land in the SAME table / overlays / inspector /
        # plots / exports as the Schmitt ones.
        self._act_load_ml_model = analysis_menu.addAction("&Load ML model…")
        self._act_load_ml_model.setToolTip(
            "Load a .keras event-detector model to use for ML detection.")
        self._act_load_ml_model.triggered.connect(self._load_ml_model)

        self._act_run_ml_detection = analysis_menu.addAction("Run M&L detection…")
        self._act_run_ml_detection.setToolTip(
            "Run the loaded Keras ML detector on the selected current channel over the "
            "chosen range (visible view or whole recording). Inference is slow, so it "
            "runs in the background with progress and cancel.")
        self._act_run_ml_detection.setEnabled(False)
        self._act_run_ml_detection.triggered.connect(self._run_ml_detection)

        analysis_menu.addSection("Curate")
        self._act_event_inspector = analysis_menu.addAction("&Event inspector…")
        self._act_event_inspector.setToolTip(
            "Step through the detected events one at a time: see each event's raw "
            "waveform with its baseline/level/boundaries and accept/reject it. Curation "
            "flows back to the table, the trace, the summary and the exports.")
        # THE TWO CURATION WINDOWS GET KEYS. Detect had Ctrl+D and Add measurement had
        # Ctrl+M; the two windows a curator actually lives in — one event at a time, and
        # all of them at once — had none, and on macOS the menu that holds them is at the
        # top of the screen rather than on the window. Ctrl+I / Ctrl+G are both free here
        # and in every child window (which are separate top-level windows with their own
        # shortcut scopes, so neither key is taken from anything).
        self._act_event_inspector.setShortcut(QtGui.QKeySequence("Ctrl+I"))
        self._act_event_inspector.setEnabled(False)
        self._act_event_inspector.triggered.connect(self._open_event_inspector)

        self._act_measure = analysis_menu.addAction("&Measure event")
        self._act_measure.setCheckable(True)
        self._act_measure.setToolTip(
            "Hand-select an event with cursors and measure its dwell, amplitude and "
            "full statistics (incl. kurtosis) live; add measurements to a saveable table.")
        self._act_measure.setEnabled(False)
        self._act_measure.toggled.connect(self._on_measure_toggled)

        # Ctrl+M lives HERE, not on the MeasurePanel. A Qt.WindowShortcut is delivered only
        # while its parent widget is visible, and that panel is one page of a four-tab stack
        # inside a dock behind an outer tab — so the key died the moment the user raised any
        # other tab, which is most of the workflow, since the cursors deliberately stay live
        # when the tab is hidden. A QAction on the menu bar is window-wide by construction.
        # The whole loop this serves is: drag a region on the trace, press Ctrl+M, drag the
        # next one — without ever looking at the panel.
        self._act_add_measurement = analysis_menu.addAction("Add measurement to &list")
        self._act_add_measurement.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        self._act_add_measurement.setToolTip(
            "Append the region currently selected on the trace to the measurement table. "
            "Works from any tab — turn on Measure event and drag a region first.")
        self._act_add_measurement.setEnabled(False)
        self._act_add_measurement.triggered.connect(self._add_measurement_to_list)

        # §4.2.7. The evidence the owner needs BEFORE he commits to re-measuring a
        # table he has already curated and exported: on a clean recording the report
        # shows ~1.0x everywhere and the decision is free, and on his own -600 mV apo
        # file it shows the real factor. Nothing is installed by opening it.
        self._act_compare_baseline = analysis_menu.addAction(
            "&Compare baseline estimators…")
        self._compare_baseline_tooltip = (
            "Show what re-measuring this detection against the configured open-pore "
            "estimator would change: old vs new depth per event, the ratio "
            "distribution, and how many events have a stamped baseline that moves "
            "more than 10 pA. Nothing is re-measured until you say so.")
        self._act_compare_baseline.setToolTip(self._compare_baseline_tooltip)
        self._act_compare_baseline.setEnabled(False)
        self._act_compare_baseline.triggered.connect(self._compare_baseline_estimators)

        analysis_menu.addSection("Explore this recording")
        self._act_plots = analysis_menu.addAction(ANALYSIS_LABELS[PAGE_DISTRIBUTIONS] + "…")
        self._act_plots.setToolTip(
            "Explore the detected events and saved manual measurements in a "
            "Analysis Explorer: dwell/amplitude histograms with fitted "
            "overlays, scatter of any two parameters, GMM/k-means clustering "
            "and PCA of a chosen feature set.")
        self._act_plots.setEnabled(False)
        self._act_plots.triggered.connect(self._open_plots)

        self._act_distribution_summaries = analysis_menu.addAction(ANALYSIS_LABELS[PAGE_DISTRIBUTION_SUMMARIES] + "…")
        self._act_distribution_summaries.setToolTip(
            "Optional ECDF, empirical survival, box and violin plots of event measurements. "
            "Choose sources and explicitly generate a summary; existing analyses are unchanged.")
        self._act_distribution_summaries.triggered.connect(self._open_distribution_summaries)
        self._act_comparison_figures = analysis_menu.addAction(ANALYSIS_LABELS[PAGE_COMPARISON_FIGURES] + "…")
        self._act_comparison_figures.setToolTip(
            "Generate faceted comparisons, pair-plot matrices and individual-event overlays "
            "from existing event measurements. Opening this page does not run analysis.")
        self._act_comparison_figures.triggered.connect(self._open_comparison_figures)

        self._act_event_plots = analysis_menu.addAction(ANALYSIS_LABELS[PAGE_EVENT_WAVEFORMS] + "…")
        self._act_event_plots.setToolTip(
            "The signature translocation plots for the current detection: the 2-D "
            "dwell-vs-blockage density map (with ECD / charge-deficit as a metric) and "
            "the event-ensemble overlay with its mean and median waveform. Needs at "
            "least one accepted event.")
        self._act_event_plots.setEnabled(False)
        self._act_event_plots.triggered.connect(self._open_event_plots)

        # Beside "Event plots", not up in *Curate* with the Event inspector, and the
        # grouping is real rather than a concession: these two are the app's "all of your
        # events at once" windows — one reduces them to a density map / ensemble average,
        # the other lays them out as a contact sheet. It curates, so *Curate* was where it
        # was first put; that placement pushed the five plot surfaces to menu item 10,
        # which is the exact burial `test_the_five_plot_surfaces_are_no_longer_buried_at_
        # the_end` exists to prevent. Same trade `_act_fit_levels` already lost, resolved
        # the same way: move the action, never widen the invariant.
        self._act_event_gallery = analysis_menu.addAction("Event &gallery…")
        self._act_event_gallery.setToolTip(
            "See every detected event at once as a grid of waveform tiles: sort by dwell "
            "or depth, sweep a box over the bad ones and reject them together. The "
            "inspector is for one event; this is for a thousand.")
        self._act_event_gallery.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        self._act_event_gallery.setEnabled(False)
        self._act_event_gallery.triggered.connect(self._open_event_gallery)

        # Reconciliation acts on the event tables from several detectors. It is placed
        # beside their all-events gallery, after the first plot surfaces: inserting it in
        # Curate would bury those surfaces past the menu-position limit the launcher/menu
        # regression suite enforces. The output is a fourth, derived EventSet; sources are
        # preserved and remain selectable above the shared table.
        self._act_reconcile = analysis_menu.addAction("&Reconcile detector results…")
        self._act_reconcile.setToolTip(
            "Combine overlapping candidates from detector results—or manual and automatic "
            "candidates in one table—into a separate reconciled event table. Choose union, "
            "agreement or intersection; preview the counts first. Source detections and "
            "their curation are preserved."
        )
        self._reconcile_tooltip = self._act_reconcile.toolTip()
        self._act_reconcile.setEnabled(False)
        self._act_reconcile.triggered.connect(self._reconcile_detections)

        # Beside the gallery because the gallery is where labelling happens. NOT earlier in
        # the menu: anything ahead of `_act_plots` pushes the five plot surfaces to item 10,
        # which `test_the_five_plot_surfaces_are_no_longer_buried_at_the_end` forbids.
        self._act_labels = analysis_menu.addAction("Event &labels…")
        self._act_labels.setToolTip(
            "Define the labels you want to apply to events — a molecule, an event class, a "
            "sub-state count, anything. The list becomes the 1–9 keys in the Event gallery "
            "and the Event inspector.")
        self._act_labels.triggered.connect(self._edit_labels)

        # Window-wide because the Measure page is normally hidden during a trace-first labeling
        # pass. Distinct from Ctrl+M: this writes a reviewed, hand-added row into the selected
        # SOURCE detector table (and therefore into supervised-learning exports), whereas Ctrl+M
        # appends statistics to the separate manual-measurement notebook.
        self._act_mark_missed = analysis_menu.addAction("Mark selection as &missed event")
        self._act_mark_missed.setShortcut(QtGui.QKeySequence("Ctrl+Shift+M"))
        self._act_mark_missed.setToolTip(
            "Add the current Measure selection to the active source detector as a reviewed "
            "event it missed. Works from any tab. (Ctrl+Shift+M)")
        self._act_mark_missed.setEnabled(False)
        self._act_mark_missed.triggered.connect(self._mark_selection_as_missed)

        self._act_psd = analysis_menu.addAction(ANALYSIS_LABELS[PAGE_NOISE] + "…")
        self._act_psd.setToolTip(
            "Open the current-noise power spectrum for a recording's current channel: "
            "the log-log PSD that separates 1/f (flicker), a mid-log-frequency median "
            "proxy and "
            "the high-frequency dielectric/capacitance rise, plus band-integrated RMS "
            "noise. Pore quality control on the RAW calibrated trace — no detection needed.")
        self._act_psd.setEnabled(False)
        self._act_psd.triggered.connect(self._open_psd)

        # THE CLOCK-SAFE WAY TO ANALYSE A STRETCH YOU PICKED OFF THE TRACE.
        #
        # The trace x-axis is on the INSTRUMENT clock (t₀ = 10.25 / 61.30 / 0 s on the
        # owner's three apoferritin recordings) and the PSD window's Start/End are in
        # seconds FROM THE START OF THE RECORDING. Both are labelled "s". Reading a
        # quiet window off the trace and typing it into the panel therefore analysed a
        # different stretch: measured on the -500 mV file, total RMS 22.66 pA / α 1.822
        # instead of 3.99 pA / α 0.985 — a 5.7x error in a headline noise number, and
        # nothing on either surface said anything was wrong.
        #
        # Labelling both surfaces (nanoscope.core.clock) stops the user from being misled.
        # This stops the user from having to do the arithmetic at all, which is the
        # actual fix: the selection travels as SAMPLE INDICES — on neither clock — and
        # the conversion happens once, here, in code. MainWindow is the sole wirer, so
        # here is where a selection made on one surface is handed to another.
        self._act_psd_selection = analysis_menu.addAction(
            "Noise spectrum of the &measured region")
        self._act_psd_selection.setToolTip(
            "Open the power spectrum over exactly the region the Measure tool is "
            "bracketing on the trace.\n\n"
            "Use this rather than reading times off the trace and typing them in: the "
            "trace x-axis is on the instrument clock (this recording may start at a "
            "t₀ that is not 0) while the PSD Start/End are seconds from the start of "
            "the recording. The selection is carried as sample indices, so the two "
            "clocks cannot be confused.")
        self._act_psd_selection.setEnabled(False)
        self._act_psd_selection.triggered.connect(self._open_psd_for_selection)

        # Sub-state level fitting. USER-INITIATED, ALWAYS. It is deliberately not part of
        # detection, not an AnalysisModule, and not triggered by selection or by results
        # changing: it is an added capability with its own parameters, and a second
        # opinion on n_levels that arrives without being asked for is indistinguishable
        # from the app changing its mind about the data. Running it is a decision.
        self._act_fit_levels = analysis_menu.addAction("Fit sub-state &levels (CUSUM)…")
        self._fit_levels_tooltip = (
            "Fit the sub-states INSIDE each accepted event with CUSUM, giving each level "
            "its own amplitude and dwell. Runs only when you choose it; the event table's "
            "own n_levels column is never touched.")
        self._act_fit_levels.setToolTip(self._fit_levels_tooltip)
        self._act_fit_levels.setEnabled(False)
        self._act_fit_levels.triggered.connect(self._fit_levels)

        # Beside the CUSUM fit for the same reason: an added capability, run only when
        # chosen, that answers a question about the events already on screen rather than
        # changing them. It writes nothing back — no schema column, no curation, no
        # label — so a run costs the user nothing but the minute it takes.
        self._act_subpop = analysis_menu.addAction("Generate " + ANALYSIS_LABELS[PAGE_SUBPOPULATIONS] + "…")
        self._subpop_tooltip = (
            "Ask whether the accepted events are ONE population or several. Measures 13 "
            "waveform features per event on the trace the detector used, gates on ≥4σ "
            "SUSTAINED depth, clusters (PCA + HDBSCAN + k-means) and then fits a Gaussian "
            "mixture whose component count is chosen by cross-validated held-out "
            "likelihood — not by eye and not by BIC, which overfits here. Produces a "
            "readable, exportable report with every caveat attached. About a minute at "
            "a thousand events; it asks first when there are enough for that to become "
            "many minutes.")
        self._act_subpop.setToolTip(self._subpop_tooltip)
        self._act_subpop.setEnabled(False)
        self._act_subpop.triggered.connect(self._run_subpopulations)

        # A saved report is a first-class analytical result, not a side effect of the
        # window that originally displayed it.  Reopening it must never mean rerunning
        # the mixture (and a project open must never pop the report window up by itself).
        self._act_show_saved_subpop = analysis_menu.addAction(
            "Open saved Subpopulations…")
        self._act_show_saved_subpop.setToolTip(
            "Open the subpopulation report stored in this project without rerunning "
            "PCA, clustering, the mixture, or bootstrap analysis.")
        self._act_show_saved_subpop.setEnabled(False)
        self._act_show_saved_subpop.triggered.connect(self._show_saved_subpop_report)

        analysis_menu.addSection("Compare recordings")
        self._act_iv = analysis_menu.addAction(ANALYSIS_LABELS[PAGE_IV] + "…")
        self._act_iv.setToolTip(
            "Gather (voltage, open-pore current) from every open recording, fit I vs V, "
            "and report the pore CONDUCTANCE (slope, in nS) with the leak and R². The "
            "classic pore-characterisation plot across your -1000..-200 mV runs.")
        self._act_iv.setEnabled(False)
        self._act_iv.triggered.connect(self._open_iv)

        self._act_vdep = analysis_menu.addAction("&Voltage dependence…")
        self._act_vdep.setToolTip(
            "Plot how an event metric (capture rate, dwell, depth, dI/I₀, count) varies "
            "with the applied voltage across your detected recordings, and fit the trend "
            "(linear or the A·exp(V/V₀) capture-rate model). The classic nanopore "
            "voltage series — batch-detect first, then open this.")
        self._act_vdep.setEnabled(False)
        self._act_vdep.triggered.connect(self._open_vdep)

        analysis_menu.addSection("Export")
        self._act_export_events = analysis_menu.addAction("Export &events…")
        self._act_export_events.setToolTip(
            "Export the current detection's events to CSV (the accepted/curated-in "
            "events), for extraction into your own tools.")
        self._act_export_events.setEnabled(False)
        self._act_export_events.triggered.connect(self._export_events)

        # THE PROCESSED TRACE'S ONLY BROWSABLE DOOR. `MeasurePanel.export_processed_trace`
        # has worked since the measure-processed round and could be reached from exactly one
        # place: a button on the Manual-measurement tab, behind the Inspector's tab bar. A
        # button is a thing you find once you already know it exists; a menu is where a user
        # looks to find out WHAT this app can write. The panel button stays — it is the
        # nearer hand when the user is already standing in the Preprocess/Measure column,
        # which is the argument the measure_panel round made and it is right — but "the
        # processing should be visible rather than occurring entirely in the background"
        # is not answered by a control only the author can find.
        self._act_export_processed = analysis_menu.addAction("Export &processed trace…")
        self._act_export_processed.setToolTip(
            "Write the conditioned samples — the Preprocess chain in force, which is the "
            "form's while a stage is enabled on it and otherwise the chain the stored "
            "detection on this channel ran under — to a CSV plus a .meta.json sidecar "
            "carrying that chain. The cursor selection when there is one, otherwise the "
            "whole recording.")
        # Gated HONESTLY rather than hopefully, on `MeasurePanel.processed_export_gate`,
        # which refuses in four ways: no live recording/channel; a channel that is not a
        # current (and is not declared one on the Calibration tab); no chain in force — no
        # stage enabled on the Preprocess form AND no stored detection on this channel that
        # ran under one; and a recording with no sample rate, which every stage needs since
        # they are designed in Hz and seconds. `_sync_processed_export_action` writes the
        # gate's own sentence onto the action, so a hover over the greyed item says which of
        # the four it is. Enabled without them, this would write a CSV of the raw channel
        # and call it processed.
        self._act_export_processed.setEnabled(False)
        self._act_export_processed.triggered.connect(self._export_processed_trace)

        # Training export: the curated events as a regions/labels table for the
        # StudentModel notebook — one row per event with accepted + added labels, plus
        # a provenance sidecar so the preprocessing that made the regions is recoverable.
        self._act_export_training = analysis_menu.addAction("Export events for &training…")
        self._act_export_training.setToolTip(
            "Export the current detection as labelled training data (regions + "
            "accepted/added labels + a provenance sidecar) to improve the model. "
            "Defaults to human-reviewed decisions of both signs.")
        self._act_export_training.setEnabled(False)
        self._act_export_training.triggered.connect(self._export_training)

        self._act_export_levels = analysis_menu.addAction("Export sub-state &levels…")
        self._act_export_levels.setToolTip(
            "Export the fitted sub-state levels (one row per event level, plus a "
            "provenance sidecar naming the CUSUM parameters). Enabled once a level fit "
            "has been run.")
        self._act_export_levels.setEnabled(False)
        self._act_export_levels.triggered.connect(self._export_levels)

        self._act_export_density = analysis_menu.addAction("Export &density figure…")
        self._act_export_density.setToolTip(
            "Overlay every open recording's detected events as 2-D KDE contours "
            "(dwell vs amplitude) with marginal densities — the comparison figure for "
            "several molecules or several voltages on one pair of axes.\n\n"
            "Amplitude can be depth in pA or fractional blockade ΔI/I₀, which is "
            "gain-independent and therefore comparable across recordings.")
        self._act_export_density.setEnabled(False)
        # Drop QAction.triggered's `checked` bool -- see _export_density's docstring.
        self._act_export_density.triggered.connect(lambda *_a: self._export_density())

        self._act_density_style = analysis_menu.addAction("Density figure st&yle…")
        self._act_density_style.setToolTip(
            "Contour levels, line width, font scale and the legend (size, position, "
            "heading, whether it carries counts). Overlaying several recordings puts the "
            "legend on top of the contours it names, so it is adjustable.")
        self._act_density_style.triggered.connect(self._edit_density_style)

        self._act_export_figure = analysis_menu.addAction("Generate " + ANALYSIS_LABELS["report_figures"] + "…")
        self._act_export_figure.setToolTip(
            "Generate a standard report (dwell/depth histograms, dwell-vs-depth "
            "scatter, trace preview) for the current detection result, each with a "
            "source-data CSV sidecar. This does not export your styled Explorer canvas; "
            "use Export current figure in the Explorer for that.")
        self._act_export_figure.setEnabled(False)
        self._act_export_figure.triggered.connect(self._export_figures)

        self._act_figure_library = analysis_menu.addAction("Figure &Library…")
        self._act_figure_library.setToolTip(
            "Named figure snapshots saved in this project, separate from the Event Gallery.")
        self._act_figure_library.triggered.connect(self._open_figure_library)

        analysis_menu.addSeparator()
        self._act_analysis_home = analysis_menu.addAction(ANALYSIS_LABELS[PAGE_HOME] + "…")
        self._act_analysis_home.setToolTip("Browse every analysis and its requirements without running analysis.")
        self._act_analysis_home.triggered.connect(self._open_analysis_home)
        self._act_first_analysis_guide = analysis_menu.addAction(ANALYSIS_LABELS["guided_first_analysis"] + "…")
        self._act_first_analysis_guide.setToolTip("A self-paced walkthrough with optional synthetic practice data; scientific settings are not changed automatically.")
        self._act_first_analysis_guide.triggered.connect(self._open_first_analysis_guide)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self._act_first_analysis_guide)
        help_menu.addAction(self._act_analysis_home)
        help_menu.addSeparator()
        self._act_about = help_menu.addAction("About NanoScope…")
        self._act_about.setToolTip("Identify this running build and copy local, path-free diagnostics.")
        self._act_about.triggered.connect(self._open_build_info)

        # Every launcher surface's gate (Plots, density/ensemble, PSD, I-V, V-dependence and
        # the figure export) is re-derived through the ONE shared path, so opening the menu and
        # looking at the always-visible launcher can never report different realities.
        analysis_menu.aboutToShow.connect(self._sync_analysis_surfaces)
        # Batch detect is live once >=1 open recording has a current channel + known rate.
        analysis_menu.aboutToShow.connect(self._sync_batch_detect_action)
        # Run-ML enablement depends on a model being loaded (from the panel or the
        # menu), which can change any time; re-derive it whenever the menu opens.
        analysis_menu.aboutToShow.connect(self._sync_ml_detection_action)
        # ...and Run detection likewise (S7-CAL1 / CAL2). Its gate follows the session's
        # EFFECTIVE channel kind, which a Calibration-panel override moves without any
        # selection change; until this line the action kept its pre-override state until
        # the next click on a file row (measured: the panel's Detect button and this
        # action disagreeing about one recording).
        analysis_menu.aboutToShow.connect(self._sync_detection_action)
        # ...and Export processed trace…, whose gate moves with the PREPROCESS FORM — a
        # stage ticked on the Preprocess tab, with no selection change and no run, is what
        # turns this item live. There is no signal in this window for "the chain changed"
        # (the form belongs to AnalysisPanel), and opening the menu is the only moment the
        # answer has to be right, so it is re-derived here. Deriving it on a timer instead
        # would be a greyed item that ungreys itself while the pointer is over it.
        analysis_menu.aboutToShow.connect(self._sync_processed_export_action)
        analysis_menu.aboutToShow.connect(self._sync_reconcile_action)

        # Last, because it walks the finished menu tree: make the sentences readable.
        self._install_menu_hover_help()

    # -- menu help: the sentences written onto QActions, made readable ----------------------------
    def _install_menu_hover_help(self) -> None:
        """Push each menu action's written sentence into the STATUS BAR while it is hovered.

        WHY. Thirty-three of this menu bar's actions carry an explanatory tooltip that no
        user could read. Measured on the wired window at 1512x900
        (verify-reachable/p10_menu_tooltips.py): File toolTipsVisible=False with 5 written
        tooltips, View False with 4, Analysis False with 24, and not one action carried a
        statusTip, so the status bar stayed empty on hover as well. QMenu paints an
        action's tooltip only when ``toolTipsVisible()`` is True (Qt's default is False) —
        and on macOS the menu bar is NATIVE, where Qt paints no action tooltip whatever
        that flag says, which is why ``_sync_undo_actions`` already writes Undo's reason
        into the LABEL. The cost was not decorative: the app's only browsable sentence
        naming batch curation is Analysis > Event gallery...'s "sweep a box over the bad
        ones and reject them together", and it had never once been readable.

        WHAT IS DONE. Every menu's ``hovered`` signal — which Qt emits for native macOS
        items too, from the Cocoa delegate's highlight callback — shows the action's
        sentence in the status bar, withdrawn again when the menu closes. Measured at the
        owner's 1512x900: the gallery sentence is 197 characters / 1142 px in the bar's
        font; the message area is 1342 px with nothing open and 1001 px once a recording
        fills the rate label, which paints its first 170 characters — through "...reject
        them together." — so the clause naming batch curation is on screen whole and only
        the closing restatement is clipped. ``setToolTipsVisible(True)``
        goes on as well (as event_gallery.py:1540 already does for its Select menu) so the
        same sentence renders as a tooltip on the non-native platforms where Qt honours
        it. Nothing is added to any layout and no widget is created, so no row's
        ``minimumSizeHint().width()`` moves and nothing under the pointer is resized.
        """
        # The message this window last pushed from a menu hover, so the withdrawal on close
        # can never clear somebody else's status line (a triggered action's own report is
        # emitted AFTER aboutToHide, so it survives).
        self._menu_help_message: str = ""

        # ``findChildren`` and NOT ``action.menu()``: walking the tree through QAction.menu()
        # gives the returned QMenu PYTHON ownership in this PySide6, so the wrapper going out
        # of scope DELETES the live menu and every QAction under it. Measured — one such walk
        # over the finished menu bar left `_act_plots` and its six siblings as dead C++
        # objects and `_wire` raised "Internal C++ object (QAction) already deleted" before
        # the window could show. findChildren is recursive and takes no ownership, so it
        # reaches the three submenus (Open Recent, Open Recent Project, Panels) safely.
        # Seven menus are wired here. An eighth, untitled one appears under the menu bar
        # later — Qt's own overflow ("extension") popup, built lazily; it holds copies of the
        # four top-level titles, which carry no sentences, so nothing is lost by missing it.
        for menu in self.menuBar().findChildren(QtWidgets.QMenu):
            menu.setToolTipsVisible(True)
            menu.hovered.connect(self._show_menu_action_help)
            menu.aboutToHide.connect(self._queue_clear_menu_action_help)

    @staticmethod
    def _menu_action_help(action: QtGui.QAction) -> str:
        """The sentence an action was written to explain itself with, or "" if it has none.

        A statusTip wins over a tooltip (it is the one written FOR this line), and a tooltip
        that merely repeats the label — Qt's default for an action with none — is not a
        sentence, so it is not shown."""
        text = " ".join(((action.statusTip() or "") or (action.toolTip() or "")).split())
        if not text:
            return ""
        # ASK QT WHAT ITS OWN FALLBACK WOULD BE, rather than reconstructing it. Qt derives an
        # unwritten toolTip from the label by stripping the mnemonic AND a trailing ellipsis,
        # so comparing against `action.text()` with the ellipsis intact never matched for the
        # actions whose labels end in one: measured on a real window, "&Open…" pushed the bare
        # word "Open" into the status bar and wiped whatever report was standing there, and
        # "Save Project &As…" did the same. Neither carries a written sentence. A fresh QAction
        # with the same text answers the question exactly, in every Qt version, without this
        # code having to know the rule.
        if text == QtGui.QAction(action.text()).toolTip():
            return ""
        return text

    def _show_menu_action_help(self, action: QtGui.QAction) -> None:
        """A menu action is hovered: show its sentence, or clear a stale one if it has none."""
        # A popup can deliver queued ``hovered`` signals after ``aboutToHide``.  Accepting
        # one would resurrect the sentence that the hide handler has just withdrawn and,
        # because an already-hidden QMenu emits no second ``aboutToHide`` when close() is
        # called, leave it in the status bar indefinitely.  The native menu path still
        # reaches this slot while its QMenu is visible; direct callers have no sender and
        # retain the public helper behaviour.
        source = self.sender()
        if isinstance(source, QtWidgets.QMenu) and not source.isVisible():
            return
        help_text = self._menu_action_help(action)
        if not help_text:
            self._clear_menu_action_help()
            return
        self._menu_help_message = help_text
        # No timeout: the sentence stands as long as the item is highlighted, and the menu's
        # aboutToHide withdraws it. A timeout would blank a sentence mid-read.
        self.statusBar().showMessage(help_text)

    def _clear_menu_action_help(self) -> None:
        """Withdraw the hover sentence — but only while it is still the message on screen."""
        held, self._menu_help_message = self._menu_help_message, ""
        if held and self.statusBar().currentMessage() == held:
            self.statusBar().clearMessage()

    def _queue_clear_menu_action_help(self) -> None:
        """Withdraw menu help after Qt has finished closing the native/popup menu.

        Some platform plugins emit one final ``hovered`` notification *after*
        ``aboutToHide``.  Clearing synchronously therefore lets that late notification
        put the just-closed action's sentence straight back.  A zero-delay callback runs
        after the close event and withdraws whichever menu sentence survived it.
        :meth:`_clear_menu_action_help` still compares the status bar text before
        clearing, so a command's own newer report is never erased.
        """
        QtCore.QTimer.singleShot(0, self._clear_menu_action_help)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._cursor_label = QtWidgets.QLabel("")
        self._cursor_label.setMinimumWidth(320)
        bar.addWidget(self._cursor_label)

        self._rate_label = QtWidgets.QLabel("")
        bar.addPermanentWidget(self._rate_label)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setMaximumWidth(240)
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        bar.addPermanentWidget(self._progress)

        self._cancel_btn = QtWidgets.QToolButton()
        self._cancel_btn.setText("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_load)
        bar.addPermanentWidget(self._cancel_btn)

        # The calibration chip: a permanent bottom-right indicator that the current gain is
        # (un)calibrated — so every non-physical magnitude the app shows is flagged, not silent.
        self._calibration_chip = CalibrationChip()
        bar.addPermanentWidget(self._calibration_chip)

    # -- workflow status (calibration chip + top strip) ------------------------------------------
    def _workflow_target(self) -> tuple[str | None, int | None]:
        """The (doc_id, channel) the chip + strip describe — the active selection while it points
        at a still-open document, else nothing open/selected."""
        sel = self._current_selection
        if sel is not None and self._session.has_document(sel[0]):
            return sel
        return None, None

    def _refresh_workflow_status(self) -> None:
        """Recompute the pipeline status for the active target and push it to the chip + strip.
        The SINGLE point that drives both features; connected to every state-changing signal."""
        doc_id, channel = self._workflow_target()
        status = compute_status(self._session, doc_id, channel)
        self._workflow_status = status
        self._calibration_chip.set_status(status.calibration)
        self._workflow_strip.set_status(status)
        guide = self._first_analysis_guide
        if guide is not None and guide.isVisible():
            guide.refresh_context()

    def _on_stage_activated(self, stage_value: str) -> None:
        """A workflow-strip stage was clicked. Route an AVAILABLE stage to its existing reveal
        action; show a locked stage's reason in the status bar (every chip emits, available or
        not)."""
        stage = Stage(stage_value)
        status = self._workflow_status
        if status is not None and not status.stage(stage).available:
            self.statusBar().showMessage(status.stage(stage).reason, 6000)
            return
        self._reveal_stage(stage)

    def _reveal_stage(self, stage: Stage) -> None:
        """Map a stage to the EXISTING reveal/open action for it — never a new code path."""
        if stage is Stage.OPEN:
            if self._session.documents():
                self._reveal_file_dock()
            else:
                self._open_dialog()
        elif stage is Stage.CALIBRATE:
            self._reveal_calibration_panel()
        elif stage is Stage.FILTER:
            self._reveal_detection_panel()
            self._analysis_panel.detection_panel.show_signal_chain()
        elif stage is Stage.DETECT:
            self._reveal_detection_panel()
            self._analysis_panel.detection_panel.show_detection_tab()
        elif stage is Stage.CURATE:
            # The chip opens the INSPECTOR, because one event at a time is where curation
            # starts and because the chip has always meant that door. The gallery is the
            # other half of the same stage and the chip is the only thing that ever pointed
            # at either — so it now names it, rather than opening a second window nobody
            # asked for. (The gallery's own always-visible door is the ▤ launcher button
            # beside this row, and Ctrl+G.)
            self._open_event_inspector()
            if self._act_event_gallery.isEnabled():
                self.statusBar().showMessage(
                    "Curating one event at a time. To judge them all at once — sort by "
                    "dwell or depth, sweep a box over the bad ones — use ▤ Event gallery "
                    "on the row above the trace, or Ctrl+G.", 8000)
        elif stage is Stage.ANALYZE:
            self._open_analysis_home()

    def _reveal_calibration_panel(self) -> None:
        """Raise the Inspector dock and select its Calibration tab (the chip's click target)."""
        self._inspector.show()
        self._inspector.raise_()
        self._inspector_tabs.setCurrentWidget(self._cal_scroll)

    def _reveal_detection_panel(self) -> None:
        """Raise the Inspector dock and select its Analysis tab (the DetectionPanel lives there)."""
        self._inspector.show()
        self._inspector.raise_()
        self._inspector_tabs.setCurrentWidget(self._analysis_panel)

    def _reveal_measure_panel(self) -> None:
        """Raise the Inspector's Analysis tab AND its inner Manual measurement tab.

        Same two-step reveal the workflow strip performs for Preprocess / Detection — the
        outer tab, then the inner one — so there is one way to bring an Analysis sub-tab
        forward, not three."""
        self._reveal_detection_panel()
        self._analysis_panel.detection_panel.show_measure_tab()

    def _reveal_file_dock(self) -> None:
        """Raise the file dock and focus the active document, if any."""
        self._file_dock.show()
        self._file_dock.raise_()
        if self._current_selection is not None \
                and self._session.has_document(self._current_selection[0]):
            self._file_dock.focus_document(self._current_selection[0])

    def _wire(self) -> None:
        self._controller.probed.connect(self._on_probed)
        self._controller.loaded.connect(self._on_loaded)
        self._controller.failed.connect(self._on_failed)
        self._controller.progressed.connect(self._on_progress)
        self._controller.cancelled.connect(self._on_cancelled)

        # Analysis controller: the AnalysisPanel handles the RESULTS (it stores them and
        # redraws overlays); MainWindow only drives the shared status-bar chrome and
        # surfaces failures, exactly as it does for the load controller.
        self._analysis_controller.started.connect(self._on_analysis_started)
        self._analysis_controller.progressed.connect(self._on_progress)
        self._analysis_controller.finished.connect(self._on_analysis_finished)
        self._analysis_controller.failed.connect(self._on_analysis_failed)
        self._analysis_controller.cancelled.connect(self._on_analysis_cancelled)
        # The subpopulation report rides the SAME controller (one analysis at a time in
        # the whole app, and the run gate enforces it), so it listens on the same three
        # terminal signals and gates itself on `_subpop_pending` + the payload type.
        # `_on_analysis_finished` still clears the progress chrome; this adds the window.
        self._analysis_controller.finished.connect(self._on_subpop_finished)
        self._analysis_controller.failed.connect(self._on_subpop_stopped)
        self._analysis_controller.cancelled.connect(self._on_subpop_stopped)
        # Sub-state fitting returns a distinct outcome, never an AnalysisResult,
        # so only its frozen-source validation slot can install the fitted table.
        self._analysis_controller.finished.connect(self._on_level_fit_finished)
        self._analysis_controller.failed.connect(self._on_level_fit_stopped)
        self._analysis_controller.cancelled.connect(self._on_level_fit_stopped)

        # Batch detection: its own controller (see __init__). MainWindow installs the
        # per-recording results on the main thread and drives the status-bar chrome,
        # reusing the same progress bar + failure box as a single run.
        self._batch_controller.started.connect(self._on_batch_started)
        self._batch_controller.progressed.connect(self._on_progress)
        self._batch_controller.finished.connect(self._on_batch_finished)
        self._batch_controller.failed.connect(self._on_analysis_failed)
        self._batch_controller.cancelled.connect(self._on_batch_cancelled)

        # A start the run gate would not admit. Both controllers ask ONE gate
        # (:mod:`nanoscope.gui.worker_thread`) and both report through one slot, so the
        # user gets the same sentence whichever direction the collision came from.
        self._analysis_controller.refused.connect(self._on_run_refused)
        self._batch_controller.refused.connect(self._on_run_refused)

        self._file_dock.selectionChanged.connect(self._on_selection)
        # The dock's right-click compare affordances. It announces the intent; the policy for
        # what "shown" means (THE SHOWN-RECORDING RULE) stays here, in one place.
        self._file_dock.showOnlyRequested.connect(self._on_show_only_requested)
        self._file_dock.addToViewRequested.connect(self._add_document_to_view)
        self._session.cursorMoved.connect(self._on_cursor)
        self._session.viewModeChanged.connect(self._sync_view_actions)
        self._session.layersChanged.connect(self._sync_measure_action)
        self._session.documentClosed.connect(self._on_document_closed)
        # A finished detection (or a cleared/closed result) changes what can be
        # exported — and whether there is anything to plot.
        self._session.resultsChanged.connect(lambda *_: self._sync_export_action())
        self._session.resultsChanged.connect(lambda *_: self._sync_plots_action())
        # Every standalone surface window is re-fed from ONE method, off EVERY signal that
        # changes what the session contains. A result appearing/vanishing was already
        # covered; a recording being CLOSED was not, and that is how the Plots window came
        # to sit there showing 5 rows of a detection the session no longer held, with a
        # working Fit and a working Export CSV. See :meth:`_refresh_surface_windows` — the
        # windows individually are half-live the moment they are wired individually.
        #
        # The manual-measurement table is the THIRD source, and it was missed for the same
        # reason the second one was: `_plots_sources` lists it ("plus the manual-measurement
        # table") and nothing re-fed the window when it changed. Measured: three
        # measurements, Plots open, delete one -> session 2 rows, window still 3, still
        # holding the deleted row's t_start, with a live Fit and a live Export CSV — C22's
        # symptom on the other table.
        #
        # It takes TWO connections because the frame has two writers and neither is a
        # superset of the other. `measurementsChanged` is the panel announcing an add, a
        # delete or an annotation edit; an UNDO does not go through it at all — the stack
        # installs its snapshot with `set_measurements_df`, which is deliberately silent
        # (a project restore uses the same door and must not look like an edit). So the
        # stack's own `changed` is wired too. Both land on the same idempotent refresher,
        # which is why wiring a signal twice cannot do any harm beyond one repeat.
        #
        # ...and that is why the list is no longer written here. The four connections this
        # block used to spell out were four of TEN, and the six that were missing are the
        # six a curator presses most (see SURFACE_MUTATION_SIGNALS at the top of this
        # file, which now drives every one of them). The two lazily-built windows wire
        # themselves from the same table when they are first opened.
        self._wire_surface_refresh("session", self._session)
        self._wire_surface_refresh("analysis_panel", self._analysis_panel)
        # The events table records its per-row refusals (an accept of a row the detector
        # could not measure, refused by the same derived-AND the gallery and inspector
        # apply) on THE ONE ledger this window keeps per detection lineage — so all three
        # surfaces disclose one record in one voice (see _event_refusals). Handed as a
        # RESOLVER, not an instance: the right ledger changes with the current detection
        # and only this window knows its lineage. Reaches the panel's ``_table`` directly:
        # the table is the panel's widget, but the ledger is cross-surface wiring, which
        # is this window's job as the sole wirer (the panel exposes no table accessor).
        self._analysis_panel._table.set_refusals_source(self._current_refusal_ledger)
        # A refusal WRITES nothing, so no data signal will repaint the other two surfaces;
        # this repaint-only hop keeps their notes current the moment the table refuses.
        self._analysis_panel._table.acceptanceRefused.connect(
            self._on_table_acceptance_refused)
        # ...and the table's new right-click, to the two doors this window already owns.
        # Same reach-through and the same justification: cross-surface routing is the sole
        # wirer's job, and the table announces intent without importing either window.
        self._analysis_panel._table.inspectRequested.connect(self._open_inspector_at_event)
        self._analysis_panel._table.galleryRequested.connect(self._open_gallery_at_event)
        self._wire_surface_refresh("measure_panel", self._analysis_panel.measure_panel)
        self._wire_surface_refresh("measure_history", self._measure_history)
        # Keep the Measure-event menu check in step with the panel's own toggle
        # (the user can flip it from either place, or it can self-disable on a switch).
        self._analysis_panel.measure_panel.measureActiveChanged.connect(
            self._on_measure_active_changed)
        # "Go to measurement" on a row belonging to a DIFFERENT document/channel. The panel
        # cannot select a document (no panel imports another panel), so it announces the
        # target and MainWindow — the sole wirer — drives the Files dock, exactly as it does
        # for markMissedRequested. Routing is synchronous, so by the time the emit returns the
        # panel has been re-pointed and can finish the jump (cursors + zoom) itself.
        self._analysis_panel.measure_panel.navigateRequested.connect(
            self._on_measurement_navigate_requested)
        # Post-detection sub-event merge. The controls live in the AnalysisPanel (near the
        # summary/table), but MainWindow is the sole wirer: it routes their intent onto the
        # panel's merge engine (which mutates the active detection's EventSet through the
        # existing curation refresh) and reports the outcome in the status bar.
        self._analysis_panel.merge_controls.mergeAllRequested.connect(
            self._on_merge_all_requested)
        # 'Merge all' runs off-thread, so its outcome arrives here rather than as the
        # return value of the click. Same sink, same sentences.
        self._analysis_panel.mergeFinished.connect(self._post_merge_status)
        self._analysis_panel.merge_controls.mergeRegionRequested.connect(
            self._on_merge_region_requested)

        # Workflow status feeds the calibration chip + the workflow strip. Recompute for the
        # ACTIVE target on every event that can change it: a document opens/closes, the
        # selection changes (via _on_selection), a detection is stored/cleared
        # (resultsChanged), a gain is edited (calibrationChanged) and an event is curated
        # (eventAcceptanceChanged — table AND inspector single toggles; batch curation is
        # wired in _open_event_inspector).
        self._session.documentOpened.connect(lambda *_: self._refresh_workflow_status())
        self._session.documentClosed.connect(lambda *_: self._refresh_workflow_status())
        self._session.resultsChanged.connect(lambda *_: self._refresh_workflow_status())
        self._session.calibrationChanged.connect(lambda *_: self._refresh_workflow_status())
        self._analysis_panel.eventAcceptanceChanged.connect(
            lambda *_: self._refresh_workflow_status())
        # A calibration edit can retype a channel (a "treat as current" override), which
        # moves the panel's Detect / Run-ML gates in place; the two menu actions mirror
        # those gates and must move with them, not on the next selection (S7-CAL1/CAL2).
        self._session.calibrationChanged.connect(
            lambda *_: (self._sync_detection_action(), self._sync_ml_detection_action()))

        # Unsaved-work tracking. The SAME state-changing signals that refresh the strip also
        # mark the session dirty — each is a mutation a project save would persist: a document
        # opens/closes, a detection is stored/cleared or merged (resultsChanged), a gain is
        # edited (calibrationChanged), an event is curated (eventAcceptanceChanged; batch
        # Accept/Reject-all arrives on curationBatchChanged, wired in _open_event_inspector)
        # and a manual measurement is added/removed (measurementsChanged). Cleared on a
        # successful save and after a restore.
        self._session.documentOpened.connect(lambda *_: self._mark_dirty_for_document_set())
        self._session.documentClosed.connect(lambda *_: self._mark_dirty_for_document_set())
        self._session.resultsChanged.connect(lambda *_: self._mark_dirty())
        self._session.calibrationChanged.connect(lambda *_: self._mark_dirty())
        self._session.sourceAppearanceChanged.connect(lambda *_: self._mark_dirty())
        # Overlay/Stacked is part of ProjectState. Unlike ``layersChanged`` this signal is
        # not emitted by opening, closing or merely selecting a recording: it represents a
        # deliberate view-mode edit (and restore is already protected by ``_restoring``).
        self._session.viewModeChanged.connect(lambda *_: self._mark_dirty())
        self._analysis_panel.eventAcceptanceChanged.connect(lambda *_: self._mark_dirty())
        self._analysis_panel.eventFlagChanged.connect(lambda *_: self._mark_dirty())
        self._analysis_panel.measure_panel.measurementsChanged.connect(
            lambda *_: self._mark_dirty())
        # A layer's view state — its tick, its Δt, its colour — is written into the project
        # file, so losing one silently is the same class of loss as an unsaved detection, and
        # it now arms the quit guard. It hangs off the DOCK's edit announcement rather than
        # ``Session.layersChanged`` on purpose: that signal cannot tell an edit from the layer
        # churn of opening, closing or switching recordings, so wiring the guard to it would
        # rebuild the "open a file, look at it, quit, answer a modal" nag that B3 and C29
        # removed. Edits are work; navigation is not.
        self._file_dock.viewStateEdited.connect(lambda *_: self._mark_dirty())
        # ...and mask edits reach the dirty flag through the HISTORY, not through
        # their own signals. That is the fix for the measured hole: a mask resize fired
        # ``eventEdited`` alone, ``_mark_dirty`` was connected to ``curationBatchChanged``
        # alone, and a resize rewrote 14 columns — bounds, dwell_s, depth_pA, snr — with
        # the title never growing a ``•``. The history sees every EventSet mutation,
        # announced or not, so hanging the flag off it covers the ones nobody remembered
        # to wire, including the ones written next year.
        self._history.changed.connect(self._mark_dirty)
        self._history.changed.connect(self._sync_undo_actions)
        self._history.stepRecorded.connect(self._on_step_recorded)
        self._history.stepRecorded.connect(lambda *_: self._note_step(_UNDO_EVENTS))
        # An eviction is the one way a step leaves the stack that the user neither asked
        # for nor can see coming, so it is said out loud where every other loss is said.
        self._history.stepsEvicted.connect(self._on_steps_evicted)
        # ...and the DEPTH ceiling's losses are the same kind of leaving, at the other
        # ceiling — they used to be the silent half (see _on_steps_depth_dropped).
        self._history.stepsDepthDropped.connect(self._on_steps_depth_dropped)
        # The measurement table's stack, off exactly the same shape of wiring: its own
        # mutation announcement drives the checkpoint, and the stack's `changed` drives the
        # dirty flag and the menu text. `measurementsChanged` is emitted by an add, a
        # delete AND an in-place annotation edit, and the checkpoint is free when nothing
        # moved — so a path that forgets to announce still lands on the stack at the next
        # Edit-menu open, save or quit (`_checkpoint_everything`).
        self._analysis_panel.measure_panel.measurementsChanged.connect(
            lambda *_: self._checkpoint_measurements())
        self._measure_history.changed.connect(self._mark_dirty)
        self._measure_history.changed.connect(self._sync_undo_actions)
        # ...and the JOURNAL hangs off the push itself, exactly as the event stack's does.
        # It used to hang off `_checkpoint_measurements`, which two real pushes never pass
        # through, and an unjournalled step is a step that did not invalidate the other
        # stack's redo branch. See `_note_step`.
        self._measure_history.stepRecorded.connect(
            lambda *_: self._note_step(_UNDO_MEASUREMENTS))
        # Adopt the (empty) table as the starting point NOW, so the very first Add is a
        # step with somewhere to go back to rather than the silent baseline-setting one.
        self._checkpoint_measurements()

        # Every announcement of an EventSet mutation, routed to the ONE recorder.
        # Redundancy is deliberate and free: a checkpoint that finds nothing changed
        # pushes nothing, so wiring a signal twice cannot invent an undo step.
        self._session.resultsChanged.connect(self._checkpoint_results)
        self._analysis_panel.eventAcceptanceChanged.connect(
            lambda *_: self._checkpoint_current())
        self._analysis_panel.eventFlagChanged.connect(lambda *_: self._checkpoint_current())
        # ...and the one announcement that comes BEFORE its mutation rather than after it:
        # a detection run has started and is going to overwrite a stored table.
        #
        # ``_checkpoint_everything`` and not ``_checkpoint_current``, because "commit
        # everything the user can lose work in" is exactly the question a replacement
        # asks, and the run may land on a table other than the one on screen.
        #
        # WHY IT CANNOT WAIT FOR ``resultsChanged``: ``checkpoint`` pushes the state it
        # last COMMITTED, so an in-place curation edit that announced nothing (the class
        # ``checkpoint_all`` exists to catch) is jumped straight past by the checkpoint
        # the arriving result triggers. Measured on the -1000 mV recording before this
        # line existed: commit at 110 accepted, reject 20 rows in place without
        # announcing (90 accepted), re-run — Edit ▸ Undo read "Undo Re-run detection",
        # and pressing it restored 110 accepted, i.e. the 20 rejections were gone and the
        # menu had said it was walking the run back. Free when nothing moved, by the same
        # contract that lets the four redundant signals above drive a checkpoint each.
        self._analysis_panel.willReplaceEvents.connect(self._checkpoint_everything)
        # A closed recording's history can no longer be applied to anything.
        self._session.documentClosed.connect(self._history.drop_doc)
        # ...and neither can a closed recording's refusal record: its lineage is gone.
        self._session.documentClosed.connect(self._drop_refusals_for_doc)
        self._session.resultsChanged.connect(lambda *_: self._sync_undo_actions())

        # Segment-from-selection: the Files-dock "Segment" button and the File-menu action both
        # route to one handler, gated by the live Measure region. MainWindow is the sole wirer —
        # it reads the on-trace selection off the shared TraceView and drives Session.create_segment.
        self._file_dock.createSegmentRequested.connect(self._create_segment_from_selection)
        # Closing a recording is routed through MainWindow for the same reason: closing a parent
        # that has segments cut from it is a data-loss decision (_close_document owns the guard),
        # and the Files-dock button and File ▸ Close document must not disagree about it.
        self._file_dock.closeDocumentRequested.connect(self._close_document)
        self._trace_view.sigMeasureChanged.connect(lambda *_: self._sync_create_segment_action())
        self._analysis_panel.measure_panel.measureActiveChanged.connect(
            lambda *_: self._sync_create_segment_action())
        self._session.documentOpened.connect(lambda *_: self._sync_create_segment_action())
        self._session.documentClosed.connect(lambda *_: self._sync_create_segment_action())

        self._calibration_chip.clicked.connect(self._reveal_calibration_panel)
        self._workflow_strip.stageActivated.connect(self._on_stage_activated)

        # The open-pore re-measure / comparison speaks through ONE channel, whether it
        # was asked for from the Analysis menu or from the stale banner's own button.
        # The panel owns the decision and the message; MainWindow owns the status bar.
        self._analysis_panel.baselineJobStatus.connect(self._on_baseline_job_status)

        # A run the panel would not start, or an answer it had to throw away, reaches the
        # status bar through here — MainWindow owns the status bar, the panel owns the
        # decision. Both cases used to be completely silent: a "Visible view" run whose
        # target is not on screen (which quietly detected over the WHOLE recording instead),
        # and a detection returning for a load of the recording that no longer exists.
        self._analysis_panel.runRefused.connect(
            lambda msg: self.statusBar().showMessage(msg, 12000))
        # S7-POP1: the moment the events table's badge set is (re)written — a Preprocess
        # knob, a gain edit, a re-feed — the SAME tuple and the banner's sentence go onto
        # the two curation windows, which are separate top-level windows sitting OVER the
        # banner. Neither the knob path nor the calibration path reaches a surface refresh,
        # so this is the only door that catches them (see _push_stale_columns).
        self._analysis_panel.stalenessChanged.connect(self._push_stale_columns)

        # The analysis launcher. Its buttons carry the surfaces' OWN explanatory copy — the
        # tooltips that were written for the QActions and that a native macOS menu never
        # renders — so the prose finally reaches the user instead of dying in the menu. The
        # launcher itself decides nothing: it announces a click and MainWindow, the sole wirer,
        # either triggers that surface's existing action or posts its gate reason.
        self._analysis_launcher.set_tooltips(
            {sid: act.toolTip() for sid, act in self._surface_actions().items()})
        self._analysis_launcher.surfaceActivated.connect(self._on_surface_activated)
        self._analysis_launcher.homeRequested.connect(self._open_analysis_home)
        # Every signal that can change a surface's gate. The launcher is ALWAYS visible, so
        # unlike the menu (which could re-derive on aboutToShow) it has to track the session
        # live: a document opens/closes, a detection is stored/cleared, an event is curated, a
        # manual measurement is saved. Selection changes are pushed from _on_selection.
        self._session.documentOpened.connect(lambda *_: self._sync_analysis_surfaces())
        self._session.documentClosed.connect(lambda *_: self._sync_analysis_surfaces())
        self._session.resultsChanged.connect(lambda *_: self._sync_analysis_surfaces())
        self._analysis_panel.eventAcceptanceChanged.connect(
            lambda *_: self._sync_analysis_surfaces())
        self._analysis_panel.measure_panel.measurementsChanged.connect(
            lambda *_: self._sync_analysis_surfaces())

        # Stage 3: the analysis form is always alive, so wire its durable controls now.
        # Optional surface windows are wired at construction by
        # ``_watch_workspace_surface`` instead; creating them here would defeat lazy load.
        self._wire_analytical_workspace_controls()

        self._refresh_workflow_status()   # initial paint (empty session -> Open is current)
        self._sync_create_segment_action()   # initial gate (nothing selected -> disabled)
        self._sync_analysis_surfaces()    # initial gates (empty session -> all six locked)

    # -- open pipeline ---------------------------------------------------------------------------
    def _already_open(self, source) -> str | None:
        """*** THE ONE PLACE THE OPEN PIPELINE ASKS "IS THIS FILE ALREADY OPEN?" ***

        Returns the live ``doc_id`` loaded from ``source``, or None.

        Since a ``doc_id`` carries a per-load token (:mod:`nanoscope.core.identity`) this
        is a DIFFERENT question from ``has_document()`` and has to be asked of the SOURCE.
        Asking the old way would have quietly turned "one load per file" into "load it a
        second time" and doubled the 180 MB. The pipeline asks it at three moments — the
        request, the moment the queue is drained, and the moment a load lands — and three
        spellings of one question is how one of them ends up being the wrong question.
        """
        return self._session.doc_id_for_source(str(source))

    def _adopt_already_open(self, doc_id: str) -> None:
        """...and the ONE definition of what "yes" means.

        A restore MUST still apply this document's saved calibration / layers / results —
        silently skipping it is what discarded the project's work and then presented the
        stale in-memory state as "the project". And an open SHOWS what it opened, whether
        or not anything had to be loaded, or File ▸ Open on something already open would
        be the one open that does not display what you opened. ``_show_document`` is a
        no-op during a restore.
        """
        if self._restoring:
            self._apply_restore_for_doc(doc_id)
        self._show_document(doc_id)
        self._file_dock.focus_document(doc_id)

    def open_paths(self, paths) -> None:
        """Open recordings — THE one door every open gesture comes through.

        File ▸ Open…, the Recent files menu and a drag-and-drop onto the window all end
        here, so anything true of "opening a file" is stated here once.

        A ``.nanoscope`` is not a recording, but it is not a stranger either: it is the
        format this very app writes for File ▸ Save Project and reads for File ▸ Open
        Project. Sent on to the reader dispatch it fell out of the bottom as *"not a
        recognised recording … its first bytes (b'PK\\x03\\x04') match no known format"* —
        the app calling its OWN save format unknown, with the one action that opens it
        never named. So it is routed to the project opener instead of being described as
        a foreign file. (File ▸ Open's "All files (*)" filter reaches this too, not just
        the drop.)

        A project REPLACES the whole session, so it cannot be combined with anything: two
        projects, or a project alongside recordings, is a gesture with two incompatible
        meanings and no defensible order. That is refused BY NAME with nothing opened,
        rather than picking one and dropping the rest silently.
        """
        paths = [Path(p) for p in paths]
        projects = [p for p in paths if p.suffix.lower() == ".nanoscope"]
        if projects:
            others = [p for p in paths if p not in projects]
            if others:
                QtWidgets.QMessageBox.warning(
                    self, "Open",
                    f"“{projects[0].name}” is a NanoScope project, and opening a project "
                    f"replaces the whole session — so it cannot be opened in the same "
                    f"gesture as {len(others)} recording(s) "
                    f"({', '.join(p.name for p in others)}).\n\nNothing was opened. Open "
                    f"the project on its own, or the recording(s) without it.")
                return
            if len(projects) > 1:
                QtWidgets.QMessageBox.warning(
                    self, "Open",
                    f"{len(projects)} NanoScope projects were opened at once "
                    f"({', '.join(p.name for p in projects)}), and only one project can be "
                    f"open at a time — each replaces the session.\n\nNothing was opened. "
                    f"Open one of them.")
                return
            # A file named .nanoscope that is not one still refuses BY NAME: `_open_project`
            # reports the archive's own error ("Could not open the project: …") rather than
            # describing it as a recording whose bytes match no reader.
            self._open_project(str(projects[0]))
            return
        for p in paths:
            path = Path(p)
            doc_id = self._already_open(path.resolve())
            if doc_id is not None:
                self._adopt_already_open(doc_id)   # one load per file; never double the RAM
                continue
            # ...and the same question about the files that are not open YET. `path not in
            # queue` is the string compare `_already_open` used to be, and it is wrong in
            # exactly the same way: two case-variant spellings of one file, dropped into the
            # window together, are one recording and must be queued once. Asked through the
            # one function that answers it (core/identity.same_source), so the queue and the
            # session cannot end up with two different ideas of "the same file".
            if (not any(same_source(q.resolve(), path.resolve()) for q in self._open_queue)
                    and not (self._active_open is not None
                             and same_source(self._active_open.resolve(), path.resolve()))):
                self._open_queue.append(path)
        self._maybe_start_next()

    def _maybe_start_next(self) -> None:
        if self._active_open is not None or not self._open_queue:
            return
        path = self._open_queue.pop(0)
        doc_id = self._already_open(path.resolve())
        if doc_id is not None:
            # It became open while it sat in the queue (a duplicate request, or a restore
            # that reached it first).
            self._adopt_already_open(doc_id)
            self._maybe_start_next()
            return
        self._active_open = path
        self._controller.probe(path)

    def _finish_active(self) -> None:
        self._active_open = None
        self._maybe_start_next()
        # A restore is done once the open queue has fully drained (no doc in flight and
        # none waiting). Finalise measurements + view mode + selection exactly once.
        if self._restoring and self._active_open is None and not self._open_queue:
            self._finalize_restore()

    @QtCore.Slot(object)
    def _on_probed(self, probe) -> None:
        path = self._active_open
        if path is None:
            return
        if self._restoring:
            # Restore reopens each document at its SAVED range; never nag about the
            # ceiling — the user already chose the range when the project was made.
            self._begin_load(path, self._restore_ranges.get(str(path.resolve())))
            return
        if not probe.exceeds_ceiling:
            self._begin_load(path, None)
            return

        meta = probe.meta
        dur = f"{meta.duration_s:.1f} s" if meta.duration_s is not None else "unknown duration"
        ratio = meta.n_samples / SOFT_SAMPLE_CEILING
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Large file")
        box.setText(f"{path.name} has {meta.n_samples:,} samples ({dur}) — "
                    f"{ratio:.1f}× the {SOFT_SAMPLE_CEILING:,}-sample working ceiling.")
        box.setInformativeText("How would you like to load it?")
        b_first = box.addButton(f"Load first {SOFT_SAMPLE_CEILING:,}", QtWidgets.QMessageBox.AcceptRole)
        b_range = box.addButton("Load a range…", QtWidgets.QMessageBox.ActionRole)
        b_all = box.addButton("Load all anyway", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()

        if clicked is b_first:
            self._begin_load(path, SampleRange(0, SOFT_SAMPLE_CEILING))
        elif clicked is b_range:
            dlg = _RangeDialog(meta.n_samples, meta.sample_rate_hz, self)
            accepted = dlg.exec() == QtWidgets.QDialog.Accepted
            chosen = dlg.range() if accepted else None
            if chosen is not None:
                self._begin_load(path, chosen)
            else:
                # ABANDONING AN OPEN IS NOT THE SAME EVENT AS CANCELLING ONE, so it does
                # not get to look identical. `_RangeDialog` now disables OK on an empty
                # range, so this can only be reached by a Cancel — but if a future change
                # ever lets an accepted dialog hand back no range again, the user hears
                # about it here instead of watching a committed open evaporate in silence.
                if accepted:
                    self.statusBar().showMessage(
                        f"{path.name} was not opened: the sample range you confirmed holds "
                        f"no samples (Stop is not past Start). Open it again and give a "
                        f"range with samples in it.", 12000)
                self._finish_active()
        elif clicked is b_all:
            self._begin_load(path, None)
        else:
            self._finish_active()

    def _begin_load(self, path: Path, sample_range: SampleRange | None) -> None:
        # Start FIRST, then show the bar: _set_progress_visible only enables Cancel while
        # a job is actually in flight, so offering it before the controller is busy would
        # show a dead button.
        self._controller.start(path, sample_range)
        self._set_progress_visible(True)
        self._progress.setValue(0)
        self._progress.setFormat(f"Opening {path.name}…")

    @QtCore.Slot(object)
    def _on_loaded(self, doc) -> None:
        self._release_load_chrome()
        # A freshly loaded document carries a brand-new load token, so it is never already
        # in the session under its own id. The question that CAN be true is whether this
        # FILE is open (a second queue entry for the same path, a re-open racing a load): in
        # that case adopt the load that is already there rather than adding a second copy —
        # and never keep `doc_id` pointing at a document the session does not hold, which is
        # what a bare `has_document(doc.doc_id)` would now do. Same question, same asker
        # (`_already_open`); the "yes" handling differs here only because the rest of this
        # method already shows and focuses the document on both branches.
        existing = self._already_open(doc.source_key)
        fresh = existing is None
        if fresh:
            doc_id = self._session.add_document(doc)
        else:
            doc_id = existing

        if self._restoring:
            # Restore path: apply THIS doc's saved calibrations + layers + results instead of
            # the default-layer policy — UNCONDITIONALLY, never gated on the document having
            # been freshly added. Gating it meant an already-open document kept its stale
            # in-memory state while the project's saved state was thrown away. Selection and
            # finalisation are deferred to _finalize_restore so the project settles as one.
            self._apply_restore_for_doc(doc_id)
            self._update_rate_label(doc_id)
            self._finish_active()
            return

        if fresh:
            self._create_default_layers(doc)
            dups = self._session.duplicates_of(doc_id)
            if dups:
                names = ", ".join(self._session.document(d).display_name for d in dups)
                self._info_bar.show_message(
                    f"“{doc.display_name}” may be another representation of: {names}. "
                    f"Both are kept open.")
            self._add_recent(doc.meta.path)
        # Opening a recording SHOWS that recording (THE SHOWN-RECORDING RULE). Explicit rather
        # than left to the focus_document -> selectionChanged -> _on_selection round trip, which
        # is silent whenever the dock's current row already happens to be this document.
        self._show_document(doc_id)
        self._announce_switch(doc_id)
        self._file_dock.focus_document(doc_id)
        self._update_rate_label(doc_id)
        self._finish_active()

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self._release_load_chrome()
        if self._restoring:
            # Never abort a whole restore over one unreadable source: record it and
            # move on. Its results are simply not installed (the doc never opened).
            if self._active_open is not None:
                self._restore_missing.append(str(self._active_open))
                # KEEP THE READER'S SENTENCE. The reader has already worked out what is
                # actually wrong and said it in words the user can act on; discarding it
                # here is what made every restore-time refusal read "could not be found"
                # for a file sitting right there on disk.
                if message:
                    self._restore_refusals[str(self._active_open)] = message
            self._finish_active()
            return
        path = self._active_open
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Could not open file")
        box.setText(message)
        if path is not None:
            box.setInformativeText(str(path))
        box.exec()
        self._finish_active()

    @QtCore.Slot()
    def _on_cancelled(self) -> None:
        self._release_load_chrome()
        self._finish_active()

    @QtCore.Slot(float, str)
    def _on_progress(self, frac: float, msg: str) -> None:
        self._set_progress_visible(True)
        self._progress.setValue(int(frac * 100))
        self._progress.setFormat(msg if msg else "%p%")

    def _cancel_load(self) -> None:
        """The Cancel button. Stops whichever background job is live — a load, a single
        analysis run, or a batch detection — and SAYS WHAT ACTUALLY HAPPENED.

        The controllers' ``abort()`` emits nothing, so the UI is reset here. Two things
        this must not do, both of which it used to:

        * It must not claim the work stopped when it did not. ``abort_stopped()``
          reports whether the worker thread was actually joined; a run that is still
          winding down is DETACHED (never terminated — that is the deadlock this whole
          change exists to remove) and the status bar says so, because "Cancelled." over
          a still-pegged CPU is a lie the user will act on.
        * It must not leave partial state behind. The detached worker's result is
          dropped by each controller's sender guard, and the analysis panel's record of
          the in-flight job is cleared here — abort() never reaches its cancelled slot.
        """
        stopped = self._controller.abort_stopped()
        stopped &= self._analysis_controller.abort_stopped()
        stopped &= self._batch_controller.abort_stopped()
        self._analysis_panel.forget_pending_job()
        self._level_fit_pending = None
        self._subpop_signal_pending = None
        self._subpop_accepted_ids_pending = None
        self._subpop_source_recipe_pending = None
        # ...and the same for the subpopulation run, for the same reason: ``abort()``
        # emits nothing, so a cancelled report would otherwise leave its claim standing.
        self._subpop_pending = False
        self._batch_skipped = 0
        self._set_progress_visible(False)
        # A multi-file open is ONE gesture to the user, so Cancel abandons ALL of it —
        # the load in flight AND the files still waiting their turn. ``_finish_active``
        # below pops the open queue and starts the next load inside this very slot, so
        # without this the status bar said "Cancelled." at the exact moment the next
        # queued file began loading, and it then opened as if Cancel had never been
        # pressed; the only way to stop the rest of the queue was one Cancel per file,
        # racing each load start. The drop is disclosed with its count — the queued
        # files' work did not merely stop, it never started.
        dropped = len(self._open_queue)
        # A RESTORE DRIVES THIS SAME QUEUE (_restore_documents -> open_paths), and dropping
        # it there ends the restore: `_finish_active` finds nothing in flight and nothing
        # waiting, and runs `_finalize_restore`. Which is acceptable — Cancel means stop —
        # but a restore that installed 1 of 3 recordings may not then report itself
        # finished. Measured before this: the info bar stayed EMPTY, the status bar read
        # "Project restored." (overwriting this very line inside the same slot, so it was
        # never painted), projectRestored was emitted and `_mark_clean` stripped the • from
        # the title — a truncated session presenting itself as a whole, saved project.
        # So the cancelled recordings are recorded as what they are: the eighth restore
        # shortfall, named in the banner beside the other seven and disclosed in the one
        # sentence `_finalize_restore` ends with.
        if self._restoring:
            if self._active_open is not None:
                self._restore_cancelled.append(str(self._active_open))
            self._restore_cancelled.extend(str(p) for p in self._open_queue)
        self._open_queue.clear()
        msg = ("Cancelled." if stopped else
               "Cancelling — the run is finishing in the background. Its result is discarded.")
        if dropped:
            msg += f" {dropped} more queued file(s) were not opened."
        self.statusBar().showMessage(msg, 6000)
        self._finish_active()

    def _set_progress_visible(self, visible: bool) -> None:
        self._progress.setVisible(visible)
        # Cancel is offered only while something is genuinely stoppable. Showing an
        # enabled Cancel next to a progress bar nothing is driving would be the same
        # class of lie as reporting a cancel that did not happen.
        self._cancel_btn.setVisible(visible)
        self._cancel_btn.setEnabled(visible and self._has_cancellable_job())

    def _release_load_chrome(self) -> None:
        """A finishing load takes the progress chrome down ONLY if nothing cancellable
        is left running.

        The bar and Cancel are ONE shared set of chrome across the load, analysis and
        batch controllers. Opening a file is not blocked by an analysis, so a load can
        land while a threshold sweep is still running — and a sweep started without
        progress ticks never repaints the bar, so an unconditional hide here stripped
        the still-running, still-stoppable sweep of its progress chrome and its ONLY
        Cancel control for the rest of the run (chaos-001). Cancel is offered exactly
        while something is genuinely stoppable, and the sweep is. When a job survives
        the load, hand the chrome back to it: re-show, and repaint the label its
        started-slot put up — the bar otherwise keeps claiming "Opening <file>…" about
        a load that just ended. Any later progress tick overwrites this repaint."""
        if not self._has_cancellable_job():
            self._set_progress_visible(False)
            return
        self._set_progress_visible(True)
        self._progress.setValue(0)
        self._progress.setFormat(self._busy_job_format)

    @QtCore.Slot(str)
    def _on_run_refused(self, message: str) -> None:
        """A background job the run gate would not admit — say so where the user is looking.

        A status-bar line, not a modal: the click was reasonable, nothing was lost, and
        the remedy (Cancel, or wait) is one control away. It is the SAME surface a
        refused "Visible view" run already uses, so the app has one voice for
        "that did not run, and here is why"."""
        self.statusBar().showMessage(message, 12000)

    def _has_cancellable_job(self) -> bool:
        """Is any background job in flight? Every one of them polls the cooperative
        cancel token (:mod:`nanoscope.analysis.cancel`), so any of them can be stopped."""
        return bool(self._controller.is_busy()
                    or self._analysis_controller.is_busy()
                    or self._batch_controller.is_busy())

    # -- analysis controller chrome --------------------------------------------------------------
    @QtCore.Slot(str)
    def _on_analysis_started(self, module_id: str) -> None:
        if module_id != "fit.levels":
            # A different job on the shared controller replaces any detached fit.
            self._level_fit_pending = None
        self._set_progress_visible(True)
        self._progress.setValue(0)
        # Remembered so a load finishing mid-run can repaint it (_release_load_chrome).
        self._busy_job_format = f"Analysing ({module_id})…"
        self._progress.setFormat(self._busy_job_format)

    @QtCore.Slot(object)
    def _on_analysis_finished(self, _result: object) -> None:
        # The AnalysisPanel installs/handles the result; MainWindow just clears the bar.
        self._set_progress_visible(False)

    @QtCore.Slot(str)
    def _on_analysis_failed(self, message: str) -> None:
        self._set_progress_visible(False)
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Analysis failed")
        box.setText(message)
        box.exec()

    @QtCore.Slot()
    def _on_analysis_cancelled(self) -> None:
        self._set_progress_visible(False)

    # -- batch detection -------------------------------------------------------------------------
    def _detectable_targets(self) -> list[tuple[str, int, str]]:
        """Every open recording that CAN be Schmitt-detected: a live CURRENT channel and
        a known sample rate. Returns ``(doc_id, channel, display_name)`` per recording,
        the channel being the document's first live current channel."""
        out: list[tuple[str, int, str]] = []
        for doc in self._session.documents():
            meta = doc.meta
            if meta.sample_rate_hz is None:
                continue
            # effective_kind, not the header: a channel corrected to pA in the
            # Calibration panel must count as current HERE too, or it ranks as
            # current in the panes while this picker skips its recording entirely.
            cur = next((c for c in meta.channels
                        if effective_kind(self._session, doc.doc_id, c.index)
                        is ChannelKind.CURRENT and not c.is_dead), None)
            if cur is None:
                continue
            out.append((doc.doc_id, cur.index, doc.display_name))
        return out

    def _sync_batch_detect_action(self) -> None:
        """Batch detect is live once >=1 open recording could be detected."""
        self._act_batch_detect.setEnabled(bool(self._detectable_targets()))

    def _batch_detect(self) -> None:
        """Analysis ▸ Batch detect… — run the current Schmitt params over every open
        recording that has a current channel + known rate, off-thread.

        Recordings already carrying a ``detect.schmitt`` result prompt a choice:
        re-run all, or only detect the missing ones (default). Recordings with no
        current channel / no rate are not targets and are counted as skipped for the
        summary. Delegates the run to :class:`BatchDetectController`; the results are
        installed on the main thread in :meth:`_on_batch_finished`."""
        if (self._analysis_panel.baseline_panel.reference_mode() != "automatic"
                or self._analysis_panel.baseline_panel.reference_error()):
            self.statusBar().showMessage(
                "Batch detect refused: this open-pore reference belongs to the selected "
                "recording and channel. Run Detect there, or explicitly choose Automatic reference.",
                12000)
            return
        # Asked BEFORE the "re-run all / only missing" question, not after: putting a
        # dialog in front of a run that is going to be refused anyway asks the user to
        # make a decision that cannot be honoured. The controller refuses it again on
        # its own account — that is the invariant; this is only the courtesy of not
        # wasting a click on the way there.
        blocker = blocking_job(JOB_ANALYSIS, self._batch_controller)
        if blocker is not None:
            self._on_run_refused(
                refusal_text(blocker, BatchDetectController.JOB_LABEL))
            return
        targets = self._detectable_targets()
        n_docs = sum(1 for _ in self._session.documents())
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "Batch detect",
                "No open recording has a current channel with a known sample rate.")
            return

        already = {doc_id for (doc_id, ch, _name) in targets
                   if self._session.results.get(doc_id, ch, "detect.schmitt") is not None}
        run_targets = targets
        if already:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("Batch detect")
            box.setText(f"{len(already)} of {len(targets)} recording(s) already have a "
                        f"Schmitt detection.")
            # NAME THE CURATION AT STAKE, the way the four single-run doors do. This dialog
            # counted RECORDINGS and never said that some of those tables hold events the user
            # rejected or added by hand — the one sentence
            # ``AnalysisPanel.confirm_replacing_curation`` exists to say. The count comes from
            # ``hand_curation_at_stake`` rather than being re-derived here, so batch and the
            # single-run prompt can never quote different numbers for the same table; it
            # already excludes the rejections a re-run would re-derive on its own
            # (clipped_by_boundary, non_finite …), which are the detector's, not the user's.
            totals = {
                "n_hand_rejected": 0,
                "n_added": 0,
                "n_notes": 0,
                "n_reviewed": 0,
                "n_amplitude_curated": 0,
            }
            docs_with_stake = 0
            for (doc_id, ch, _name) in targets:
                res = self._session.results.get(doc_id, ch, "detect.schmitt")
                if res is None or not hasattr(res.payload, "df"):
                    continue
                work = self._analysis_panel.hand_work_at_stake(res.payload.df)
                if any(work.values()):
                    for key in totals:
                        totals[key] += work[key]
                    docs_with_stake += 1
            tail = ""
            if docs_with_stake:
                details = []
                for key, singular, plural in (
                    ("n_hand_rejected", "hand rejection", "hand rejections"),
                    ("n_added", "hand-added event", "hand-added events"),
                    ("n_notes", "event with labels/notes", "events with labels/notes"),
                    ("n_reviewed", "reviewed event", "reviewed events"),
                    ("n_amplitude_curated", "amplitude choice", "amplitude choices"),
                ):
                    n = totals[key]
                    if n:
                        details.append(f"{n} {singular if n == 1 else plural}")
                tail = (
                    f"\n\n{docs_with_stake} of them carry human curation "
                    f"({', '.join(details)}). Re-running replaces those tables and the "
                    "new rows carry none of that curation."
                )
            box.setInformativeText(
                "Re-run all of them, or only the ones that are missing one?" + tail)
            b_missing = box.addButton("Only missing", QtWidgets.QMessageBox.AcceptRole)
            b_all = box.addButton("Re-run all", QtWidgets.QMessageBox.DestructiveRole)
            box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            box.setDefaultButton(b_missing)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_all:
                run_targets = targets
            elif clicked is b_missing:
                run_targets = [t for t in targets if t[0] not in already]
            else:
                return

        if not run_targets:
            self.statusBar().showMessage(
                "Batch detect: nothing to do — every recording already has a detection.", 6000)
            return

        # `batch_detection_params`, not `detection_params`: the batch applies ONE
        # form to every recording, and the form's ``baseline_param_source`` is a
        # claim about the recording it was filled in for. Sent as is, ``'proposed'``
        # ("measured on this recording, and confirmed") landed on every file in the
        # batch and silenced the never-measured advisory on all of them (S7-DET1);
        # the batch variant names the recording the windows belong to, and every
        # other run is stamped ``'batch'`` inside the detector.
        params = self._analysis_panel.batch_detection_params()
        self._batch_skipped = n_docs - len(targets)   # recordings without a current channel/rate
        self._batch_controller.start([(d, c) for (d, c, _n) in run_targets], params)

    @QtCore.Slot(int)
    def _on_batch_started(self, n: int) -> None:
        self._set_progress_visible(True)
        self._progress.setValue(0)
        # Remembered so a load finishing mid-run can repaint it (_release_load_chrome).
        self._busy_job_format = f"Batch detecting {n} recording(s)…"
        self._progress.setFormat(self._busy_job_format)

    @QtCore.Slot(object)
    def _on_batch_finished(self, outcome: object) -> None:
        """Install every batch result on the MAIN THREAD, then summarise. The
        resulting resultsChanged refreshes the panel + re-gates the plot actions."""
        from ..analysis.events import EventSet

        self._set_progress_visible(False)
        results = getattr(outcome, "results", ())
        errors = getattr(outcome, "errors", ())
        total_events = 0
        n_installed = 0
        n_stale = 0
        for res in results:
            # Same rule as the interactive detection sink: a result whose document is no
            # longer open is a result about nothing (close_document already dropped the store
            # entry). Since a doc_id names the LOAD (core.identity), this now also catches the
            # twin the batch controller shares with the single-run path — a recording closed
            # and REOPENED while the pass ran, whose new load the answer was never computed on.
            if not self._session.has_document(res.doc_id):
                n_stale += 1
                continue
            self._session.add_result(res)
            n_installed += 1
            if isinstance(res.payload, EventSet):
                total_events += len(res.payload.df)

        parts = [f"Batch detect: {n_installed} recording(s), {total_events} event(s)."]
        if n_stale:
            parts.append(f"Dropped {n_stale} whose recording was closed (or closed and "
                         f"reopened) while it ran.")
        if self._batch_skipped:
            parts.append(f"Skipped {self._batch_skipped} without a current channel/known rate.")
        if errors:
            parts.append(f"{len(errors)} failed: "
                         + "; ".join(f"{lbl} ({msg})" for lbl, msg in errors[:3]))
        self._batch_skipped = 0
        self.statusBar().showMessage(" ".join(parts), 12000)
        self._sync_export_action()
        self._sync_plots_action()

    @QtCore.Slot()
    def _on_batch_cancelled(self) -> None:
        self._set_progress_visible(False)
        self._batch_skipped = 0
        self.statusBar().showMessage("Batch detect cancelled.", 6000)

    # -- layer + selection policy ----------------------------------------------------------------
    #
    # *** THE SHOWN-RECORDING RULE. *** Opening a recording, or picking one in the File Manager,
    # SHOWS THAT RECORDING: its layers become visible and every other document's go invisible.
    # Nothing is closed and no layer is destroyed, so every other recording stays open, stays
    # listed, stays analysable, and comes back — with its colours and its alignment intact — in
    # one click. Comparing two runs is still one tick of a channel checkbox, but it is now
    # something the user ASKS for rather than what a second File ▸ Open silently does to them.
    #
    # Displayed is expressed as layer VISIBILITY, never as layer membership: membership is the
    # authoritative include/exclude decision for a channel (see core/layer.py) and survives being
    # looked away from. See Session.show_only.
    #
    # The one exception is a project RESTORE, which must reproduce the saved layer set EXACTLY —
    # including a deliberately built multi-recording overlay. Every entry point below is gated on
    # ``self._restoring`` for that reason.

    def _create_default_layers(self, doc) -> None:
        """Add a Layer for every non-dead channel of ``doc``. Dead channels are never
        auto-layered but are always listed in the dock and can be ticked on.

        This used to layer only channels "whose kind is already layered" for the 2nd and later
        documents, so that a second voltage run overlaid the first comparably. That rule existed
        to serve the automatic pile-on, which is exactly the behaviour being removed here: a
        newly opened recording is now SHOWN ON ITS OWN, so it wants its own full channel set —
        otherwise opening a 3-channel ABF after a current-only CSV would display the ABF with its
        voltage channel silently missing.
        """
        for info in doc.meta.channels:
            if not info.is_dead:
                self._session.add_layer(doc.doc_id, info.index)

    def _show_document(self, doc_id: str) -> None:
        """Make ``doc_id`` the shown recording (THE SHOWN-RECORDING RULE, above).

        Creates the default layers first if this document has none at all — it has either never
        been displayed, or the user unticked every one of its channels — so that "show me this"
        always has something to show.

        A no-op during a project restore: the restore's own layer set (which may legitimately be
        a multi-recording overlay) is the saved one, and must not be collapsed to one document by
        the reconstruction's own opens and selections.
        """
        if self._restoring or not self._session.has_document(doc_id):
            return
        if not any(lay.doc_id == doc_id for lay in self._session.layers()):
            self._create_default_layers(self._session.document(doc_id))
        self._session.show_only(doc_id)

    def _ensure_target_drawn(self, doc_id: str) -> bool:
        """*** THE APP NEVER ANALYSES A RECORDING IT IS NOT DRAWING. ***

        Make sure the recording the selection points at has at least one VISIBLE layer,
        additively — nothing is ever hidden here, so a deliberately built multi-recording
        overlay survives. Returns True when the view had to change.

        This exists as one guarantee rather than as a fix to the path that exposed it,
        because that path is the third of its kind. A project restore lands its selection on
        "the first document with a detection" while ``_show_document`` is a no-op during the
        restore (correctly — the saved layer set may be a deliberate overlay that must not be
        collapsed), so the app drew ``short_b`` and analysed ``long_a``. What that costs, all
        measured: "Detect over the visible view" resolved to samples (0, 4000) of 40000 —
        the first 10 % of a recording nobody was looking at — because the visible x window
        came off whichever pane was first; the Measure tool's region has no pane to attach
        to, so ``measure_bounds()`` was None and *Add to list* was a silent no-op; and every
        cursor readout described a different recording from the one the panels named. The
        same shape produced C8 and the measure-panel offset defect before this.

        DOCUMENT level, not channel level. A channel the user has explicitly unticked stays
        unticked — ``Layer.excluded`` is the authoritative include/exclude decision for a
        channel (``core/layer.py``) and merely selecting its row is not a request to
        overrule it. What is not negotiable is that the RECORDING is somewhere on screen; a
        gate that then refuses "visible view" for an unticked channel can at least say so
        (:meth:`AnalysisPanel._visible_sample_range`), which an app drawing nothing of the
        recording cannot.
        """
        if not self._session.has_document(doc_id):
            return False
        layers = [lay for lay in self._session.layers() if lay.doc_id == doc_id]
        if any(lay.visible for lay in layers):
            return False
        if not layers:
            self._create_default_layers(self._session.document(doc_id))
            layers = [lay for lay in self._session.layers() if lay.doc_id == doc_id]
        wanted = [lay for lay in layers if not lay.excluded]
        if not wanted:
            # EVERY channel of this recording was unticked. Selecting the recording is a
            # request to look at it, and an app that draws nothing cannot explain why —
            # so the exclusion is lifted rather than silently overruled, which keeps the
            # tick boxes agreeing with the trace. (Before ``excluded`` existed this
            # happened anyway, by the back door: unticking the last channel deleted the
            # last layer and the branch above rebuilt the defaults.)
            for lay in layers:
                self._session.set_layer_excluded(lay.layer_id, False)
            return bool(layers)
        for lay in wanted:
            self._session.set_layer_visible(lay.layer_id, True)
        return bool(wanted)

    @QtCore.Slot(str)
    def _on_show_only_requested(self, doc_id: str) -> None:
        """Right-click ▸ *Show only this recording* — show it AND retarget onto it.

        Showing without retargeting is a wrong-target bug, not a cosmetic one: the context
        menu can be raised on a recording that is not the current row, so displaying it while
        the selection stayed put left Detect / Measure / Calibrate / the Inspector / the
        workflow strip all pointed at the recording that had just gone invisible. The next
        Detect then ran on a recording that was not on screen and reported its event count as
        if it were the one being looked at.

        Routed through the dock's own selection so the entire ``_on_selection`` chain runs
        exactly as it does for a click. ``_show_document`` is still called afterwards for the
        case where that row is ALREADY current — Qt emits no ``selectionChanged`` then, and
        the show would otherwise never happen. When the selection did move, the second call
        is a no-op (``Session.show_only`` reports no change and emits nothing).
        """
        if self._restoring or not self._session.has_document(doc_id):
            return
        self._file_dock.focus_document(doc_id)
        self._show_document(doc_id)

    def _add_document_to_view(self, doc_id: str) -> None:
        """Draw ``doc_id`` ALONGSIDE whatever is already shown — the explicit opt-in to the
        comparison the app used to perform uninvited. Every non-dead channel of the recording is
        layered (if it is not already) and made visible; NOTHING else is hidden.

        "Add to view" is an explicit request for this recording, so it clears any exclusion
        on its channels — this is the one place where overruling an untick is what the user
        just asked for, and the tick boxes follow within the same signal."""
        if self._restoring or not self._session.has_document(doc_id):
            return
        doc = self._session.document(doc_id)
        for info in doc.meta.channels:
            if info.is_dead and not self._session.has_layer(doc_id, info.index):
                continue
            layer = self._session.add_layer(doc_id, info.index)
            self._session.set_layer_excluded(layer.layer_id, False)
        self.statusBar().showMessage(
            f"Added “{doc.display_name}” to the view. Overlay/Stacked (File Manager) decide "
            f"whether comparable channels share a pane.", 8000)

    def _announce_switch(self, doc_id: str) -> None:
        """Say, once and where the user is already looking, that the others are still open and
        how to compare them. Opening a second recording is the exact moment the old pile-on
        behaviour was doing the comparing for you, so it is the moment to teach the opt-in."""
        others = [d for d in self._session.documents() if d.doc_id != doc_id]
        if not others:
            return
        name = self._session.document(doc_id).display_name
        self.statusBar().showMessage(
            f"Showing “{name}”. {len(others)} other recording(s) stay open — tick a channel in "
            f"the File Manager to draw it alongside for comparison.", 10_000)

    def _ensure_something_shown(self) -> None:
        """After a close, never leave recordings listed but the viewer blank. If nothing is
        drawn any more and documents remain, show one of them."""
        if self._restoring:
            return
        if any(lay.visible for lay in self._session.layers()):
            return
        docs = self._session.documents()
        if docs:
            self._show_document(docs[0].doc_id)

    def _on_selection(self, doc_id: str, channel: int) -> None:
        """Retarget the UI while suppressing analysis-form echoes from navigation."""
        navigating = self._workspace_navigating
        self._workspace_navigating = True
        try:
            self._retarget_selection(doc_id, channel)
        finally:
            self._workspace_navigating = navigating

    def _retarget_selection(self, doc_id: str, channel: int) -> None:
        """Point the whole window at ``(doc_id, channel)``. ALL OF IT, OR THE USER IS TOLD.

        Picking a recording DISPLAYS it — it does not merely retarget the analysis panels.
        (Ticking a channel checkbox does not come through here: Qt routes a click on a check
        indicator to the item delegate and leaves the current item alone, so an overlay tick
        never doubles as a request to switch away from what it is being overlaid on.)

        Sixteen consumers, run through :meth:`_retarget` so that ONE OF THEM RAISING
        CANNOT LEAVE THE WINDOW SPLIT. Before this, a raise simply abandoned the rest of
        the list: a zero-sample recording made the calibration panel raise at step 4, so
        the title bar, the calibration panel and ``self._current_selection`` were on the
        empty recording while the analysis panel, the Inspector, the plots and every gate
        were still on the previous one — and pressing Detect ran on the recording the user
        was no longer looking at (C20, measured). The empty recording itself is now
        refused at the reader, but the split was never really about empty recordings: it
        is what ANY exception in ANY of sixteen consumers does, and PySide6 already
        swallows an exception that escapes a slot, so the old behaviour was to print a
        traceback nobody reads and carry on half-pointed. Now the traceback is still
        printed, every remaining consumer is still retargeted, and the user is told which
        part of the window did not follow.
        """
        failures: list[str] = []
        step = self._retarget_step(failures)

        step("view", lambda: self._show_document(doc_id))
        step("view", lambda: self._ensure_target_drawn(doc_id))
        self._current_selection = (doc_id, channel)
        step("Calibration", lambda: self._calibration_panel.set_selection(doc_id, channel))
        step("File info", lambda: self._file_info_panel.set_document(doc_id))
        # AnalysisPanel.set_selection repoints its owned MeasurePanel too (which clears
        # the cursor region when the doc/channel changes), so the region never lingers
        # on a pane the user has navigated away from.
        step("Analysis", lambda: self._set_analysis_selection_for_navigation(
            doc_id, channel))
        annotation_target = self._analysis_panel.current_target()
        if annotation_target is None or annotation_target[0] != doc_id:
            # None means this recording has no resolvable current channel. A different document
            # means the guarded Analysis step raised and left its old target standing; never let
            # that stale recording leak into the rest of this retarget chain.
            annotation_target = (doc_id, channel)
        # AFTER Analysis: repointing its MeasurePanel clears an old recording's live region. Scan
        # deliberately prefers a live measure_target over this fallback, so setting the fallback
        # first would briefly (and, without another x change, permanently) cache the old extent.
        # Use Analysis's RESOLVED current channel, not the raw row: selecting an ABF's command
        # voltage is still an analysis of its current, and aligned channels can have different x
        # extents. All three surfaces must describe the same annotation target.
        target_doc, target_channel = annotation_target
        step("Navigator", lambda: self._navigator.set_target(target_doc, target_channel))
        step("Scan", lambda: self._pan_scrollbar.set_target(target_doc, target_channel))
        step("rate", lambda: self._update_rate_label(doc_id))
        step("Detect", self._sync_detection_action)
        step("Run ML", self._sync_ml_detection_action)
        step("Measure", self._sync_measure_action)
        step("Export", self._sync_export_action)
        # THE FOUR SURFACE-WINDOW FEEDS FOLLOW THE `_refresh_surface_windows` RULE: a
        # window the user has CLOSED is not re-fed. These four used to be unconditional —
        # each refresher's own guard answers only "was it ever built", and close() merely
        # hides these windows — so once the PSD or Event-plots window had EVER been opened,
        # every recording switch for the rest of the session paid its full synchronous
        # recompute (a Welch PSD; the whole calibrated array) with nothing on screen to
        # show for it (measured: +86-304 ms per switch, 12/12 feeds landing on
        # isVisible()==False windows). Nothing is lost by skipping: every opener refreshes
        # immediately before show(). This guard covers every door into _on_selection —
        # the File Manager's selectionChanged (the click-a-recording path), the
        # measurement-table jump (_on_measurement_navigate_requested's direct call), and
        # the project restore's final focus (_select_after_restore) — they all pass
        # through here.
        step("Event Inspector",
             lambda: self._refresh_event_inspector()            # may be a different detection
             if _on_screen(self._event_inspector) else None)
        step("Event Gallery",
             lambda: self._refresh_event_gallery()
             if _on_screen(self._event_gallery) else None)
        step("Subpopulation views", self._refresh_subpop_views)  # ...which may now be stale
        step("Event plots",
             lambda: self._refresh_event_plots()                # ...as may density / ensemble
             if (_on_screen(self._event_plot_window)
                 and EVENT_PLOTS_ARTIFACT_ID not in self._workspace_frozen)
             else None)
        step("Noise spectrum",
             lambda: self._point_psd_at_selection()             # ...and the PSD window follows
             if (_on_screen(self._psd_window)
                 and PSD_ARTIFACT_ID not in self._workspace_frozen)
             else None)
        step("workflow", self._refresh_workflow_status)         # ...and the chip + strip track
        step("Create segment", self._sync_create_segment_action)  # ...and its gate follows
        step("analysis surfaces", self._sync_analysis_surfaces)   # ...and the six surface gates
        step("Analysis Explorer", self._refresh_analysis_explorer_header)
        step("Undo", self._sync_undo_actions)   # ...and Edit ▸ Undo names THIS table's last step

        if failures:
            self.statusBar().showMessage(
                "Switched recording, but " + ", ".join(sorted(set(failures)))
                + " could not follow. Everything else is pointed at the recording you "
                  "picked; re-select it, or reopen the file, before trusting those panels.",
                12000)

    @staticmethod
    def _retarget_step(failures: list[str]):
        """Build the runner :meth:`_on_selection` puts every consumer through.

        Deliberately catches ``Exception`` and carries on. That reads as swallowing, and
        it would be if the alternative were the exception reaching somebody — it is not:
        this runs inside a Qt slot, where PySide6 already prints the traceback and returns
        as if nothing happened. So the choice is not "raise or swallow", it is "half a
        retarget and silence" or "a whole retarget and a sentence". The traceback still
        goes to stderr for whoever is debugging."""
        import traceback

        def step(what: str, fn) -> None:
            try:
                fn()
            except Exception:            # noqa: BLE001 — see the docstring
                failures.append(what)
                traceback.print_exc()

        return step

    def _sync_detection_action(self) -> None:
        """Mirror the AnalysisPanel's own runnable state — it resolves the selection to the
        document's CURRENT channel (so clicking voltage does not disable detection), and this action
        must agree with the panel's Detect button rather than re-deriving from the raw selection."""
        self._act_run_detection.setEnabled(self._analysis_panel.can_run())

    def _run_detection(self) -> None:
        """The Analysis ▸ Run detection action — delegate to the panel so params and
        gating live in one place.

        THE MENU'S DOOR, NOT THE PROGRAMMATIC ONE. ``trigger_detection_from_menu`` is
        ``trigger_detection`` plus the panel's own curation prompt
        (``confirm_replacing_curation``), and this action is a person's click exactly as
        the Detect button is — the two must not disagree about whether re-running over ten
        hand rejections asks first. Measured on the -1000 mV recording before this line
        changed: 10 hand rejects (100 accepted of 110), Analysis ▸ Run detection ->
        ``confirm_replacing_curation called: False``, a run started, curation gone. The
        stale banner's Re-run, the scripts and the suite keep ``trigger_detection``, which
        is deliberately unprompted; see both docstrings on AnalysisPanel.
        """
        self._analysis_panel.trigger_detection_from_menu()

    def _sync_ml_detection_action(self) -> None:
        """Mirror the AnalysisPanel's ML runnable state — a CURRENT channel with a
        known rate, TensorFlow available AND a model loaded — so the menu action and
        the panel's Run-ML button never disagree.

        Load ML model… rides the SAME TensorFlow gate as the panel's Load-model
        button (DetectionPanel._refresh_ml_enabled): with TensorFlow absent there is
        nothing a loaded model could ever run on, so the menu offering the load
        while the panel refused it was two surfaces disagreeing about one fact — and
        the menu path accepted a model with no TensorFlow message anywhere (DET-01).
        Disabled WITH its reason: the tooltip carries the panel gate label's exact
        sentence. Re-derived here because this sync already runs everywhere the ML
        gates can change (the Analysis menu's aboutToShow, every selection retarget)."""
        from ..analysis import ml_detect

        tf_ok = ml_detect.TENSORFLOW_AVAILABLE
        self._act_load_ml_model.setEnabled(tf_ok)
        self._act_load_ml_model.setToolTip(
            "Load a .keras event-detector model to use for ML detection."
            if tf_ok else _ML_TF_MISSING)
        self._act_run_ml_detection.setEnabled(self._analysis_panel.can_run_ml())

    def _run_ml_detection(self) -> None:
        """The Analysis ▸ Run ML detection action — delegate to the panel so the model,
        range and thresholds live in one place."""
        self._analysis_panel.trigger_ml_detection()

    def _load_ml_model(self) -> None:
        """The Analysis ▸ Load ML model action — delegate to the panel's file dialog so
        the loaded-model state has a single owner; then re-gate the Run-ML action.

        Refuses, with the gate's reason, if it fires while TensorFlow is absent — the
        action is disabled by :meth:`_sync_ml_detection_action` before every menu
        show, but a trigger that beats the first resync must not silently accept a
        model that can never run (DET-01)."""
        from ..analysis import ml_detect

        if not ml_detect.TENSORFLOW_AVAILABLE:
            self.statusBar().showMessage(_ML_TF_MISSING, 10000)
            self._sync_ml_detection_action()   # the action was stale to get here; fix it
            return
        self._analysis_panel.load_ml_model()
        self._sync_ml_detection_action()

    # -- measure event -----------------------------------------------------------------------
    def _sync_measure_action(self) -> None:
        """Enable Measure event whenever there is a DRAWN live channel to select on. The
        measurement is a hand gesture, not a detector, so it never waits on a run.

        *Add measurement to list* (Ctrl+M) and *Mark missed* ride the broader OPEN-target gate,
        deliberately not the narrower "is a region drawn right now". Qt
        never delivers a keystroke to a disabled action, so gating it on the region would
        leave the user pressing a dead key with no way to learn why. Their handlers name a hidden
        channel explicitly, while the toggle itself cannot promise cursors it cannot draw."""
        panel = self._analysis_panel.measure_panel
        self._act_measure.setEnabled(panel.can_measure())
        self._act_add_measurement.setEnabled(panel.has_open_target())
        self._act_mark_missed.setEnabled(panel.has_open_target())
        # The processed-trace export rides the same refreshes (a selection retarget is one
        # of the two things that can move its gate); the other one, the Preprocess form, is
        # covered by the menu's own aboutToShow.
        self._sync_processed_export_action()

    def _sync_processed_export_action(self) -> None:
        """Gate *Export processed trace…* on the panel's own sentence, and SHOW the sentence.

        ``MeasurePanel.processed_export_gate`` returns "" when the export can run and one of
        four complete sentences naming the fix when it cannot — verbatim, the first and the
        third: "Open a recording and select a channel first." and "No filter stage is enabled
        on the Preprocess tab and no stored detection on this channel ran under one, so there
        is no processed trace to export — the raw channel is already on disk."; the other two
        refuse a channel that is not a current and a recording with no sample rate. That
        sentence
        goes on the action as its statusTip, which :meth:`_menu_action_help` prefers over
        the tooltip, so hovering a GREYED item says why it is grey — the failure mode this
        menu bar already fixed once for the detection actions. Cleared to "" when the export
        is live, so the hover falls back to the tooltip describing what a click writes.

        THE FORM IS PUSHED AT THE MOMENT THE QUESTION IS ASKED, so the panel is never
        judging a Preprocess form the user has since edited. ``MeasurePanel`` talks to Session
        and its TraceView and to no sibling panel, so those parameters have to arrive from
        outside — and both public halves of that seam already exist
        (``AnalysisPanel.detection_params`` out, ``set_conditioning`` in);
        ``set_conditioning`` is a no-op when the parameters have not moved, so re-asking
        costs nothing.

        WHAT THE GATE THEN JUDGES IS NOT ALWAYS THAT FORM. ``MeasurePanel._active_conditioning``
        uses the pushed form while any stage is enabled on it, and otherwise falls back to the
        chain a stored detection on this channel ran under — so with every stage off and a
        detection standing, this item is ENABLED and the export writes that stored chain. That
        is the panel's rule rather than this window's, and it is the same one the *Processed*
        measurement basis and the note under its readout already follow; pushing the form here
        is what keeps the live half of it current, not a claim that the form is the whole of
        it."""
        panel = self._analysis_panel.measure_panel
        panel.set_conditioning(self._analysis_panel.detection_params())
        gate = panel.processed_export_gate()
        self._act_export_processed.setEnabled(not gate)
        self._act_export_processed.setStatusTip(gate)

    def _export_processed_trace(self) -> None:
        """Analysis ▸ Export processed trace… — the conditioned samples, through the panel.

        The panel owns the write (it holds the chain, the target and the cursor bounds) and
        reports into its own status label; that label lives on the Manual-measurement tab,
        which is exactly where a user who came through the MENU is not looking. So the
        window repeats the result on the status bar and drains the export's identity notes
        the way every other export in this window does. Nothing is revealed, raised or
        resized: a menu item that rearranged the Inspector to show its own confirmation
        would be moving furniture under the pointer to say "done"."""
        out = self._analysis_panel.measure_panel.export_processed_trace()
        if out is None:
            # Gate refused (the panel put the reason in a dialog), dialog cancelled, or the
            # write failed and was reported. Re-derive: the gate is the one of those three
            # that leaves the action's own state stale.
            self._sync_processed_export_action()
            return
        self.statusBar().showMessage(
            f"Wrote the processed trace to {out['data'].name} "
            f"(+ the chain in {out['meta'].name}).", 8000)
        self._disclose_identity_notes("Export processed trace", list(out.values()))

    def _add_measurement_to_list(self) -> None:
        """Analysis ▸ Add measurement to list (Ctrl+M) — stamp the current cursor region.

        Reports the FIRST real blocker rather than failing silently, which is what the
        panel's own shortcut did (``add_current`` returns False and says nothing when there
        is no live measurement)."""
        panel = self._analysis_panel.measure_panel
        if not panel.has_open_target():
            self.statusBar().showMessage(
                "Nothing to measure — select a recording first.", 8000)
            return
        if not panel.can_measure():
            self.statusBar().showMessage(
                "The selected current channel is not shown. Tick that channel in the File "
                "Manager, then turn on Measure and press Ctrl+M again.", 10000)
            return
        if not panel.is_active():
            self.statusBar().showMessage(
                "Measure is off. Turn on Analysis ▸ Measure event, then drag a region on "
                "the trace and press Ctrl+M.", 10000)
            return
        if not panel.add_current():
            self.statusBar().showMessage(
                "No region is selected on the trace — drag one across the event you want "
                "to measure, then press Ctrl+M.", 10000)

    def _mark_selection_as_missed(self) -> None:
        """Ctrl+Shift+M: promote the live cursor span into the selected source detector table."""
        panel = self._analysis_panel.measure_panel
        if not panel.has_open_target():
            self.statusBar().showMessage(
                "Nothing to mark — open and select a recording first.", 8000)
            return
        if not panel.can_measure():
            self.statusBar().showMessage(
                "The selected current channel is not shown. Tick that channel in the File "
                "Manager, then bracket the missed event and press Ctrl+Shift+M again.", 12000)
            return
        if not panel.is_active() or self._trace_view.measure_bounds() is None:
            self.statusBar().showMessage(
                "Turn on Measure event and bracket the missed event first, then press "
                "Ctrl+Shift+M.", 10000)
            return
        current = self._analysis_panel.current_detection()
        if current is None:
            self.statusBar().showMessage(
                "There is no source detector table to add this event to. Run a detector, or "
                "select an existing Schmitt/CUSUM/ML result, then mark the missed event.", 12000)
            return
        if current.module_id == CONSENSUS_MODULE_ID:
            self.statusBar().showMessage(
                "The Reconciled table is derived and cannot be edited directly. Select a source "
                "detector table, add the missed event there, then reconcile again.", 12000)
            return
        event_id = self._analysis_panel.add_current_selection_as_event()
        if event_id is None:
            self.statusBar().showMessage(
                "The selection could not be added. Check that its cursors are on the selected "
                "recording and span at least one sample.", 10000)
            return
        self.statusBar().showMessage(
            f"Added missed event {event_id} as a reviewed training label. Use Shift+Right "
            "or the Scan ▶ button to continue at the same zoom.", 10000)

    def _on_measure_toggled(self, checked: bool) -> None:
        """Menu ▸ Measure event drives the panel's own toggle (one source of truth), and
        REVEALS the tool it just turned on.

        The cursors appear on the trace either way, but the readout, the Add-to-list button
        and the saved-measurement table live on the Manual measurement tab — a menu item that
        silently arms a tool the user cannot see is the reason this panel was hard to find in
        the first place. Reveal only on the way IN: switching the Inspector back to some other
        tab as the user turns the tool OFF would be the app moving furniture for no reason."""
        self._analysis_panel.measure_panel.set_active(bool(checked))
        if checked:
            self._reveal_measure_panel()

    def _on_measure_active_changed(self, active: bool) -> None:
        """Mirror the panel's toggle back onto the menu check without re-triggering it."""
        was = self._act_measure.blockSignals(True)
        self._act_measure.setChecked(bool(active))
        self._act_measure.blockSignals(was)

    def _on_measurement_navigate_requested(self, doc_id: str, channel: int) -> None:
        """Route "return to that saved measurement, it's on another recording/channel".

        Selects the exact CHANNEL row in the Files dock so the app selection matches what the
        measurement was taken on (its own selectionChanged then flows through ``_on_selection``
        and back into the measure panel). Falls back to the document row — and to calling
        ``_on_selection`` directly — when the tree has no such row (e.g. an unlayered channel),
        so the caller is never left half-navigated."""
        if not self._session.has_document(doc_id):
            return
        channel = int(channel)
        if self._file_dock.focus_channel(doc_id, channel):
            if self._current_selection != (doc_id, channel):
                self._on_selection(doc_id, channel)
            return
        self._file_dock.focus_document(doc_id)
        self._on_selection(doc_id, channel)

    # -- create segment from selection ---------------------------------------------------------
    def _can_create_segment(self) -> bool:
        """True when a segment could be cut right now: an active recording IS selected AND the
        on-trace Measure region is a non-empty span. Reuses the SAME LinearRegionItem selection
        the Measure + Merge tools use — no second selection mechanism."""
        sel = self._current_selection
        if sel is None or not self._session.has_document(sel[0]):
            return False
        bounds = self._trace_view.measure_bounds()
        return bounds is not None and (bounds[1] - bounds[0]) >= 2

    def _sync_create_segment_action(self) -> None:
        """Gate the Create-segment menu action AND the Files-dock button together, so they never
        disagree. Connected to every signal that can change the answer: selection, the Measure
        region moving, the Measure tool toggling, and documents opening/closing."""
        ok = self._can_create_segment()
        self._act_create_segment.setEnabled(ok)
        self._file_dock.set_segment_enabled(ok)
        # Centring needs a selection but NOT a whole segment's worth of it (a one-sample cursor
        # pair is a perfectly good thing to centre on), so it is gated separately, on the same
        # signals.
        self._act_centre_selection.setEnabled(self._trace_view.measure_bounds() is not None)
        # ...and the clock-safe hand-off of that same selection to the PSD window. Gated
        # on the SAME signals for the same reason: an action that is live when there is
        # nothing selected can only answer the click with an apology.
        self._act_psd_selection.setEnabled(
            self._trace_view.measure_bounds() is not None
            and self._trace_view.measure_target() is not None
            and self._has_psd_recording())

    def _centre_selection_in_view(self) -> None:
        """Pan so the middle of the Measure selection is the middle of the view. Zoom untouched."""
        region = self._trace_view.measure_x_range()
        if region is None:
            self.statusBar().showMessage(
                "Turn on Measure (Analysis ▸ Measure event) and drag a region on the trace to "
                "choose what to centre.", 8000)
            return
        x0, x1 = region
        # Navigation lives in plotted x, so keep the live edges. Going x -> nearest sample index ->
        # x shifts sub-sample cursor positions, and stop_idx==n maps back to the last sample rather
        # than the legal half-open edge one sample later.
        self._pan_scrollbar.centre_on(0.5 * (x0 + x1))

    def _default_segment_name(self, parent_id: str, i0: int, i1: int) -> str:
        """The prefilled name for a new segment: ``‹parent short name› [t0–t1 s]`` (or a sample
        window when the recording declares no rate). The range in the name is why the Files list
        needs no extra range column for segments."""
        doc = self._session.document(parent_id)
        short = doc.display_name
        if len(short) > 32:
            short = short[:31] + "…"
        xax = self._session.x_axis(parent_id)
        vals = xax.values
        n = vals.size
        if n == 0:
            return f"{short} [segment]"
        lo = float(vals[min(i0, n - 1)])
        hi = float(vals[min(i1, n - 1)])
        if xax.is_time:
            return f"{short} [{lo:.4g}–{hi:.4g} s]"
        return f"{short} [{int(lo)}–{int(hi)} samp]"

    def _ask_segment_name(self, default: str) -> str | None:
        """Prompt for the new segment's name (prefilled, cancellable). Returns the chosen name, or
        None if cancelled. A single injectable seam so the suite never faces a live modal."""
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Create segment", "Name for the new segment:",
            QtWidgets.QLineEdit.Normal, default)
        return text if ok else None

    def _create_segment_from_selection(self) -> None:
        """Crop the current on-trace Measure selection into a new derived recording, then select
        it in the Files list. Reads ``[i0, i1)`` off the shared TraceView, prompts for a name,
        and delegates to :meth:`Session.create_segment`; the new document opens like any other
        (its own results/selection).

        It marks the session dirty EXPLICITLY. A segment is not an opened file — it is a
        derived recording the user built by choosing a sample window on the trace, and a
        project stores it as that spec. Nothing on disk holds it, so it is exactly the kind of
        work the quit guard exists for. It used to ride on ``documentOpened``, which no longer
        marks dirty by itself (see :meth:`_mark_dirty_for_document_set`)."""
        if not self._can_create_segment():
            self.statusBar().showMessage(
                "Turn on Measure (Analysis ▸ Measure event) and drag a region on the trace to "
                "choose what to segment.", 8000)
            return
        parent_id = self._current_selection[0]
        i0, i1 = self._trace_view.measure_bounds()
        default = self._default_segment_name(parent_id, i0, i1)
        name = self._ask_segment_name(default)
        if name is None:
            return   # cancelled
        name = name.strip() or default
        try:
            seg_id = self._session.create_segment(parent_id, i0, i1, name=name)
        except (ValueError, TypeError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Create segment", f"Could not create the segment:\n{exc}")
            return
        # Layer it like a freshly opened recording so its trace + detection overlays are viewable,
        # SHOW it (a segment you just cut is what you want on screen — same rule as an open),
        # then select it (which repoints every panel to the segment).
        self._create_default_layers(self._session.document(seg_id))
        self._show_document(seg_id)
        self._file_dock.focus_document(seg_id)
        self._mark_dirty()   # a hand-cut segment is work; only a project file holds it
        self.statusBar().showMessage(
            f"Created segment “{name}” ({i1 - i0:,} samples).", 8000)

    def _sync_export_action(self) -> None:
        """Enable the export actions AND the Event inspector only when the current selection has a
        detection result (the same resultsChanged/selection gate the Plots action uses).

        The figure export and the event gallery are two of the launcher's seven surfaces, so
        they go through the shared gate path with the rest of them rather than being set twice
        from two predicates."""
        det = self._analysis_panel.current_detection()
        has_detection = det is not None
        # A level fit belongs to ONE event table, so it is dropped the moment the
        # detection under it is replaced. This lives HERE, in the gate that always runs,
        # and not beside the Event Inspector refresh: that one returns early when the
        # inspector has never been opened, which left a stale fit alive and exportable
        # under a different recording's provenance for any user who had not opened the
        # browser.
        #
        # *** STALENESS IS ABOUT THE FIT'S OWN DETECTION, NOT ABOUT WHERE THE USER IS
        # LOOKING. *** This used to lead with ``det is None``, i.e. "the CURRENT selection
        # has no detection" — so clicking a raw recording (or a voltage channel) in the
        # Files panel for one moment permanently discarded a perfectly valid fit, out of
        # this window's hands AND out of the ResultStore, i.e. out of the next project
        # save. Selecting somewhere else is not a statement about the levels. The two
        # conditions left ARE: the recording they were measured on is gone, or the
        # detection now on screen IS that recording and channel and carries a different
        # frame (a re-run, a merge, a restored snapshot) — which is exactly the case
        # `test_a_level_table_retired_by_a_re_detection_is_not_persisted_at_all` pins.
        fit = self._level_fit
        # The active table can change because the user selected ANOTHER DETECTOR on the
        # same channel.  That is navigation, not replacement: ResultStore still owns the
        # original detector frame and its level fit must remain parked beside it.  A real
        # re-detection replaces that frame in the store, so identity over all same-target
        # EventSets distinguishes the two without guessing from the current selection.
        anchor_still_stored = bool(fit is not None and any(
            getattr(getattr(result, "payload", None), "df", None)
            is self._level_fit_frame
            for result in self._session.results.for_doc(fit.doc_id)
            if int(getattr(result, "channel", -1)) == int(fit.channel)
        ))
        if fit is not None and (
                not self._session.has_document(fit.doc_id)
                or not anchor_still_stored):
            self._invalidate_level_fit()
        # ...and the mirror of that: ADOPT a level table this window does not yet hold but
        # the session does. That is how a RESTORED fit reaches the UI. Persisting it (v6)
        # only put it in the store; without this the user reopened a project, found
        # "Export sub-state levels…" greyed out, and was told to run the fit again — the
        # exact cost the persistence exists to avoid, moved one layer up.
        #
        # Adoption is safe precisely BECAUSE _invalidate_level_fit now drops the retired
        # entry from the store: anything still there was fitted on the detection that is
        # still current, so re-anchoring it to that detection's df is a statement of fact,
        # not an assumption. It fires only when this window holds nothing (after a restore,
        # or after switching away and back) — a live fit is already in hand.
        #
        # A LIVE fit reaching a surface that dropped its own copy is NOT this branch's job
        # any more: ``set_events`` nulls the window's levels on every frame change, and
        # ``_refresh_event_inspector`` / ``_refresh_event_gallery`` now re-push through
        # :meth:`_levels_for_detection` on every re-feed. Widening the condition here as
        # well would give that guarantee two owners for no extra coverage.
        if (self._level_fit is None and det is not None
                and det.module_id != CONSENSUS_MODULE_ID):
            stored = self._session.results.get(det.doc_id, det.channel, "fit.levels")
            if stored is not None and getattr(stored.payload, "empty", True) is False:
                self._level_fit = stored
                self._level_fit_frame = getattr(det.payload, "df", None)
                self._push_level_fit(stored.payload)
        self._act_export_levels.setEnabled(self._level_fit is not None)
        self._act_export_events.setEnabled(has_detection)
        self._act_export_density.setEnabled(len(self._density_groups()) >= 1)
        self._act_export_training.setEnabled(has_detection)
        self._act_event_inspector.setEnabled(has_detection)
        can_remeasure = has_detection and det.module_id != CONSENSUS_MODULE_ID
        self._act_compare_baseline.setEnabled(can_remeasure)
        self._act_compare_baseline.setToolTip(
            self._compare_baseline_tooltip
            + ("\n\nThe Reconciled table can contain rows from different detector "
               "conditioning. Switch to a source detector table, re-measure there, "
               "then reconcile again."
               if has_detection and not can_remeasure else "")
        )
        can_reconstruct_trace = (
            has_detection and det.module_id != CONSENSUS_MODULE_ID)
        trace_refusal = (
            f"\n\n{_RECONCILED_TRACE_ANALYSIS_REFUSAL}"
            if has_detection and not can_reconstruct_trace else "")
        self._act_fit_levels.setEnabled(can_reconstruct_trace)
        self._act_fit_levels.setToolTip(self._fit_levels_tooltip + trace_refusal)
        self._act_subpop.setEnabled(can_reconstruct_trace)
        self._act_subpop.setToolTip(self._subpop_tooltip + trace_refusal)
        # The gallery's ``setEnabled(has_detection)`` used to live on the line above, which
        # made this the SECOND owner of its enablement once it became a launcher surface —
        # and a second owner is how the greyed button and the live menu item come to disagree.
        # Both gates go through the one shared path instead.
        self._apply_surface_gate(SURFACE_FIGURES)
        self._apply_surface_gate(SURFACE_GALLERY)
        self._sync_reconcile_action()

    def _reconcile_sources(self) -> tuple:
        """Compatible source detector results on the Analysis panel's resolved target.

        The derived reconciliation itself is deliberately excluded: feeding it back as a
        detector vote would double-count all of its contributors. The pure core function
        validates the deeper calibration/baseline contracts when Preview is pressed.
        """
        target = self._analysis_panel.current_target()
        if target is None or not self._session.has_document(target[0]):
            return ()
        from ..analysis.events import EventSet
        from ..core.sources import DETECTION_MODULE_IDS

        doc_id, channel = target
        out = []
        for module_id in DETECTION_MODULE_IDS:
            if module_id == CONSENSUS_MODULE_ID:
                continue
            res = self._session.results.get(doc_id, channel, module_id)
            if res is not None and isinstance(res.payload, EventSet):
                out.append(res)
        return tuple(out)

    def _sync_reconcile_action(self) -> None:
        sources = self._reconcile_sources()
        available = can_reconcile_sources(sources)
        self._act_reconcile.setEnabled(available)
        if len(sources) >= 2:
            detail = f"\n\n{len(sources)} source detector results are available here."
        elif available:
            detail = (
                "\n\nThis detector table contains both hand-added and automatic "
                "candidates, so they can be converged directly."
            )
        elif self._analysis_panel.current_target() is None:
            detail = "\n\nOpen a recording and select its current channel first."
        else:
            detail = (
                f"\n\n{len(sources)} source detector result is available here. Run a "
                "second detector, or add a missed event to the source table so manual "
                "and automatic candidates can be converged."
            )
        self._act_reconcile.setToolTip(self._reconcile_tooltip + detail)

    def _make_reconcile_dialog(self, sources: tuple) -> ReconcileDialog:
        """Factory seam kept tiny so GUI tests never need to face a modal dialog."""
        return ReconcileDialog(sources, self)

    def _reconcile_detections(self) -> None:
        """Preview and install a non-destructive derived detector table."""
        sources = self._reconcile_sources()
        if not can_reconcile_sources(sources):
            self._sync_reconcile_action()
            self.statusBar().showMessage(
                "Run a second detector, or add a missed event to the source table before "
                "reconciling.",
                8000,
            )
            return

        dialog = self._make_reconcile_dialog(sources)
        if dialog.exec() != QtWidgets.QDialog.Accepted or dialog.outcome is None:
            return
        outcome = dialog.outcome
        result = outcome.result
        prior = self._session.results.get(
            result.doc_id, int(result.channel), CONSENSUS_MODULE_ID)
        if prior is not None and not self._analysis_panel.confirm_replacing_curation(
                CONSENSUS_MODULE_ID):
            return

        # Commit any in-place edits before the replacement arrives. Session.add_result's
        # resultsChanged checkpoint then records the prior reconciled table as an Undo step.
        # On first creation there is no prior state, which is harmless: no source was edited
        # and there is nothing destructive to undo.
        self._checkpoint_everything()
        self._session.add_result(result)
        conflict_note = (
            f" {outcome.n_label_conflicts:,} unresolved label conflict(s) need review "
            "before training."
            if outcome.n_label_conflicts else ""
        )
        curation_conflict_note = (
            f" {outcome.n_curation_conflicts:,} contradictory accept/reject decision(s) "
            "need an explicit curator verdict before training."
            if outcome.n_curation_conflicts else ""
        )
        self.statusBar().showMessage(
            f"Reconciled {outcome.n_input_rows:,} input candidates into "
            f"{outcome.n_output_rows:,} candidates: {outcome.n_collapsed:,} rows "
            f"converged, {outcome.n_filtered:,} groups filtered. Source tables unchanged."
            f"{curation_conflict_note}{conflict_note}",
            12000,
        )

    # ================================================================================================
    # analysis surfaces: ONE gate per surface, feeding BOTH the menu action and the launcher
    # ================================================================================================
    #
    # Each surface's gate returns ``(available, reason)`` from a single evaluation of the SAME
    # predicate that used to drive ``setEnabled`` alone. Everything the user can see about that
    # surface — the menu item's enablement, the launcher button's greyed/live state, its tooltip
    # and the message a locked click posts — is derived from that one tuple, so they cannot drift
    # into disagreeing about reality. Writing the reason was most of the work here: before this,
    # only one of the seven gates stated its condition anywhere the user could read it.

    def _surface_actions(self) -> dict:
        """Surface id -> the EXISTING ``&Analysis`` QAction that opens it.

        The launcher never duplicates an open/gating code path: it triggers these.
        """
        return {
            SURFACE_POPULATIONS: self._act_plots,
            SURFACE_EVENT_PLOTS: self._act_event_plots,
            SURFACE_PSD: self._act_psd,
            SURFACE_IV: self._act_iv,
            SURFACE_VDEP: self._act_vdep,
            SURFACE_FIGURES: self._act_export_figure,
            SURFACE_GALLERY: self._act_event_gallery,
        }

    def _surface_gates(self) -> dict:
        """Surface id -> the callable returning that surface's ``(available, reason)``."""
        return {
            SURFACE_POPULATIONS: self._gate_populations,
            SURFACE_EVENT_PLOTS: self._gate_event_plots,
            SURFACE_PSD: self._gate_psd,
            SURFACE_IV: self._gate_iv,
            SURFACE_VDEP: self._gate_vdep,
            SURFACE_FIGURES: self._gate_figures,
            SURFACE_GALLERY: self._gate_gallery,
        }

    def _apply_surface_gate(self, sid: str) -> None:
        """Evaluate ``sid``'s gate ONCE and push the result to both places it shows up."""
        available, reason = self._surface_gates()[sid]()
        self._surface_actions()[sid].setEnabled(available)
        self._analysis_launcher.set_gate(sid, available, reason)

    def _sync_analysis_surfaces(self) -> None:
        """Re-derive every analysis surface's gate. Connected to every signal that can change
        one (a document opens/closes, a detection is stored/cleared, an event is curated, a
        measurement is saved, the selection moves) AND to the Analysis menu's aboutToShow."""
        for sid in SURFACE_ORDER:
            self._apply_surface_gate(sid)
        self._sync_analysis_open_state()
        if self._analysis_explorer is not None:
            self._analysis_explorer.home().refresh_availability()
        if self._first_analysis_guide is not None and self._first_analysis_guide.isVisible():
            self._first_analysis_guide.refresh_context()

    def _gate_populations(self) -> tuple[bool, str]:
        """Populations (the Plots window): needs SOME plottable per-event table — any open
        recording's detection, or the saved manual measurements."""
        if self._plots_sources():
            return True, ""
        if self._workspace_has_artifact(PLOTS_ARTIFACT_ID):
            # A frozen project result remains a readable/exportable result even when its
            # source recording or event table cannot currently be resolved (AP-16).
            return True, ""
        if not self._session.documents():
            # Naming the REAL blocker. "Run detection first" with nothing open sends the user
            # to a detector that is itself disabled for want of a recording.
            return False, ("Open a recording first — Distributions plots per-event "
                           "tables (a detection's events, or a saved manual measurement).")
        return False, ("Run detection first — Distributions plots per-event tables "
                       "(a saved manual measurement counts too).")

    def _gate_event_plots(self) -> tuple[bool, str]:
        """Density map / ensemble: needs the ACTIVE detection to have >=1 accepted event —
        there is nothing to bin or overlay otherwise. Each way to fail is a different problem
        with a different fix (nothing open / nothing selected / no detection / nothing
        accepted), so each gets its own sentence naming the blocker that is actually first."""
        from ..analysis.events import EventSet

        if self._workspace_has_artifact(EVENT_PLOTS_ARTIFACT_ID):
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the density map and the ensemble are "
                           "built from a detection's events.")
        if self._workflow_target()[0] is None:
            return False, ("Select a recording in the Files panel first — the density map and "
                           "the ensemble are built from the selected recording's events.")
        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return False, ("Run detection on the selected recording first — the density map "
                           "and the ensemble are built from its detected events.")
        if res.payload.accepted.df.shape[0] < 1:
            return False, ("No accepted events yet — accept at least one in the Event "
                           "inspector (or tick it in the event table) to fill the density "
                           "map and the ensemble.")
        return True, ""

    def _gate_psd(self) -> tuple[bool, str]:
        """Power spectrum: needs >=1 open recording with a live current channel and a known
        sample rate. It reads the RAW calibrated trace, so it never waits on a detection."""
        if self._has_psd_recording():
            return True, ""
        if self._workspace_has_artifact(PSD_ARTIFACT_ID):
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the noise spectrum needs a current "
                           "channel with a known sample rate (no detection required).")
        return False, ("No open recording has a current channel with a known sample rate — "
                       "the noise spectrum needs both.")

    def _gate_iv(self) -> tuple[bool, str]:
        """I-V / conductance: needs >=1 open recording with a live current channel AND a known
        applied voltage (a live voltage channel, or a voltage recorded in the file's metadata)."""
        if self._has_iv_data():
            return True, ""
        if self._workspace_has_artifact(IV_ARTIFACT_ID):
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the I-V fit needs a current channel and "
                           "a known applied voltage.")
        return False, ("No open recording has both a current channel and a known applied "
                       "voltage (a voltage channel, or a voltage in the file's metadata).")

    def _gate_vdep(self) -> tuple[bool, str]:
        """Voltage dependence: needs >=2 recordings that each have a known applied voltage AND
        a stored detection — one point is not a trend. The reason carries the live count, from
        the very number the predicate thresholds, so it can never overstate progress."""
        n = self._vdep_ready_count()
        if n >= 2:
            return True, ""
        if self._workspace_has_artifact(VOLTAGE_DEPENDENCE_ARTIFACT_ID):
            return True, ""
        return False, (f"Needs at least 2 detected recordings with a known applied voltage — "
                       f"{n} so far. Open the voltage series and run Analysis ▸ Batch detect.")

    def _gate_figures(self) -> tuple[bool, str]:
        """Export figures: needs a detection on the CURRENT selection — the figures are drawn
        from its events (the same gate the other two exports use). Like the density map, it
        names the blocker that is actually first rather than always blaming detection."""
        if self._analysis_panel.current_detection() is not None:
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the figures are drawn from a detection's "
                           "events.")
        if self._workflow_target()[0] is None:
            return False, ("Select a recording in the Files panel first — the figures are "
                           "drawn from the selected recording's events.")
        return False, ("Run detection on the selected recording first — the figures are drawn "
                       "from its events.")

    def _gate_gallery(self) -> tuple[bool, str]:
        """Event gallery: needs a detection on the CURRENT selection — the contact sheet is a
        tile per detected event.

        This is the SAME ``current_detection() is not None`` predicate ``_sync_export_action``
        used to apply straight to ``_act_event_gallery.setEnabled``; that line is gone, so the
        menu item, the launcher button and the reason a locked click posts now all come from
        this one evaluation. Unlike the density map it does NOT ask for an accepted event —
        the gallery is where a curator goes to decide which events those are.
        """
        if self._analysis_panel.current_detection() is not None:
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the gallery lays out a detection's events "
                           "as a grid of waveform tiles.")
        if self._workflow_target()[0] is None:
            return False, ("Select a recording in the Files panel first — the gallery lays out "
                           "the selected recording's detected events.")
        return False, ("Run detection on the selected recording first — the gallery lays out "
                       "its detected events as a grid of waveform tiles.")

    def _on_surface_activated(self, sid: str) -> None:
        """A launcher button was clicked. An AVAILABLE surface triggers its EXISTING menu action
        (never a parallel open path); a LOCKED one posts its gate reason to the status bar and
        opens nothing — exactly what a locked workflow-strip chip does."""
        available, reason = self._surface_gates()[sid]()
        if not available:
            self.statusBar().showMessage(reason, 6000)
            return
        self._surface_actions()[sid].trigger()
        self._sync_analysis_open_state()

    # ============================================================================================
    # Stage 3 analytical workspace — project-owned, lazily installed
    # ============================================================================================

    def _set_analysis_selection_for_navigation(self, doc_id: str, channel: int) -> None:
        """Retarget the live form without turning navigation into a project edit.

        ``AnalysisPanel.set_selection`` legitimately seeds controls from the stored result
        on the new target, and those public setters emit the same aggregate signals as a
        person's edits.  The File Manager selection itself is navigation (AP-13), so guard
        only this synchronous seed.  Detector/result combo changes made by the user still
        arrive outside the guard and remain durable workspace edits.
        """
        installing = self._workspace_installing
        self._workspace_installing = True
        try:
            self._analysis_panel.set_selection(doc_id, channel)
        finally:
            self._workspace_installing = installing

    def _wire_analytical_workspace_controls(self) -> None:
        """Route every always-live analysis-form edit into the v11 workspace cache.

        These connections intentionally live in MainWindow.  The child panels describe
        analysis and emit intent; only MainWindow owns project dirty/recovery revisions.
        Programmatic replay during restore is suppressed by ``_workspace_installing`` and
        ``_restoring``, so opening a project cannot manufacture an unsaved revision.
        """
        changed = lambda *_: self._on_workspace_changed(ANALYSIS_PANEL_ARTIFACT_ID)
        detection = self._analysis_panel.detection_panel
        baseline = self._analysis_panel.baseline_panel
        measure = self._analysis_panel.measure_panel
        stats = getattr(self._analysis_panel, "_stats", None)

        detection.paramsChanged.connect(changed)
        detection.mlModelChanged.connect(changed)
        detection.previewFilterToggled.connect(changed)
        baseline.paramsChanged.connect(changed)
        baseline.showBaselineToggled.connect(changed)
        baseline.subtractBaselineToggled.connect(changed)
        measure.measureActiveChanged.connect(changed)

        # The remaining fields are deliberately exposed through the analysis adapter but
        # do not have one aggregate project-state signal of their own.
        for widget, signal_name in (
                (getattr(detection, "mode_tabs", None), "currentChanged"),
                (getattr(measure, "_method_combo", None), "currentTextChanged"),
                (getattr(measure, "_basis_combo", None), "currentIndexChanged"),
                (getattr(measure, "_scope_check", None), "clicked"),
                (getattr(measure, "_markers_check", None), "clicked"),
                (getattr(self._analysis_panel.merge_controls, "window_ms", None),
                 "valueChanged"),
                (getattr(stats, "n_components_dwell", None), "valueChanged"),
                (getattr(stats, "n_components_depth", None), "valueChanged"),
                (getattr(stats, "use_efficiency", None), "clicked"),
                (getattr(stats, "use_filter_limited", None), "clicked")):
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(changed)

        # Keep one immutable constructor-default recipe so opening a v1-v10 project (which
        # correctly has no workspace block) cannot inherit the outgoing project's draft.
        defaults = capture_analysis_workspace(analysis_panel=self._analysis_panel)
        self._workspace_default_analysis = next(
            (item for item in defaults.artifacts
             if item.artifact_id == ANALYSIS_PANEL_ARTIFACT_ID), None)

    @staticmethod
    def _workspace_kwarg(artifact_id: str) -> str | None:
        return {
            PLOTS_ARTIFACT_ID: "plots",
            DISTRIBUTION_SUMMARIES_ARTIFACT_ID: "distribution_summaries",
            COMPARISON_FIGURES_ARTIFACT_ID: "comparison_figures",
            SUBPOPULATION_ARTIFACT_ID: "subpopulation",
            PSD_ARTIFACT_ID: "psd",
            EVENT_PLOTS_ARTIFACT_ID: "event_plots",
            IV_ARTIFACT_ID: "iv",
            VOLTAGE_DEPENDENCE_ARTIFACT_ID: "voltage_dependence",
            ANALYSIS_PANEL_ARTIFACT_ID: "analysis_panel",
        }.get(str(artifact_id))

    def _workspace_window_id(self, window) -> str | None:
        for artifact_id, candidate in (
                (PLOTS_ARTIFACT_ID, self._plots_window),
                (DISTRIBUTION_SUMMARIES_ARTIFACT_ID, self._distribution_summary_window),
                (COMPARISON_FIGURES_ARTIFACT_ID, self._comparison_figures_window),
                (SUBPOPULATION_ARTIFACT_ID, self._subpop_window),
                (PSD_ARTIFACT_ID, self._psd_window),
                (EVENT_PLOTS_ARTIFACT_ID, self._event_plot_window),
                (IV_ARTIFACT_ID, self._iv_window),
                (VOLTAGE_DEPENDENCE_ARTIFACT_ID, self._vdep_window)):
            if window is candidate:
                return artifact_id
        return None

    def _workspace_surface(self, artifact_id: str):
        return {
            PLOTS_ARTIFACT_ID: self._plots_window,
            DISTRIBUTION_SUMMARIES_ARTIFACT_ID: self._distribution_summary_window,
            COMPARISON_FIGURES_ARTIFACT_ID: self._comparison_figures_window,
            SUBPOPULATION_ARTIFACT_ID: self._subpop_window,
            PSD_ARTIFACT_ID: self._psd_window,
            EVENT_PLOTS_ARTIFACT_ID: self._event_plot_window,
            IV_ARTIFACT_ID: self._iv_window,
            VOLTAGE_DEPENDENCE_ARTIFACT_ID: self._vdep_window,
            ANALYSIS_PANEL_ARTIFACT_ID: self._analysis_panel,
        }.get(str(artifact_id))

    def _workspace_artifact(self, artifact_id: str):
        workspace = self._analytical_workspace
        return next((item for item in workspace.artifacts
                     if item.artifact_id == artifact_id), None)

    def _workspace_has_artifact(self, artifact_id: str) -> bool:
        return self._workspace_artifact(artifact_id) is not None

    def _drop_workspace_artifact(self, artifact_id: str) -> None:
        """Forget one invalidated built-in result without touching opaque siblings.

        A live surface can cease to have a result when one of its scientific inputs is
        edited (for example, re-declaring the only current channel as voltage empties the
        PSD window).  Keeping its last checkpoint would make the corresponding launcher
        gate claim that a readable saved result still exists.  This is deliberately a
        single-artifact operation: plugin blocks, ordering, workspace metadata and every
        other built-in result survive byte-for-byte.
        """
        workspace = self._analytical_workspace
        kept = [item for item in workspace.artifacts
                if item.artifact_id != str(artifact_id)]
        if len(kept) == len(workspace.artifacts):
            return
        self._analytical_workspace = AnalyticalWorkspace(
            version=workspace.version,
            artifacts=kept,
            extra=workspace.extra,
        )
        self._workspace_pending.discard(str(artifact_id))
        self._workspace_frozen.discard(str(artifact_id))
        self._sync_saved_subpop_action()

    def _sync_saved_subpop_action(self) -> None:
        action = getattr(self, "_act_show_saved_subpop", None)
        if action is not None:
            action.setEnabled(self._workspace_has_artifact(SUBPOPULATION_ARTIFACT_ID))

    def _watch_workspace_surface(self, artifact_id: str, window) -> None:
        """Wire one lazily-created surface without making it a project writer."""
        signal = getattr(window, "workspaceChanged", None)
        if signal is not None:
            signal.connect(
                lambda *_args, _id=artifact_id: self._on_workspace_changed(_id))
        checkpoint = getattr(window, "workspaceCheckpointRequested", None)
        if checkpoint is not None:
            checkpoint.connect(
                lambda *_args, _id=artifact_id: self._checkpoint_workspace_surface(_id))

        # Subpopulation view state predates the common signal.  Its two choices and
        # splitter sizes are still durable; user-only combo signals avoid treating adapter
        # replay as an edit, and the install guard covers splitter replay.
        if artifact_id == SUBPOPULATION_ARTIFACT_ID:
            view = window.embedding_view()
            for combo in (getattr(view, "_embed_combo", None),
                          getattr(view, "_colour_combo", None)):
                activated = getattr(combo, "activated", None)
                if activated is not None:
                    activated.connect(
                        lambda *_args: self._on_workspace_changed(
                            SUBPOPULATION_ARTIFACT_ID))
            window.splitter().splitterMoved.connect(
                lambda *_: self._on_workspace_changed(SUBPOPULATION_ARTIFACT_ID))
            window.figures_splitter().splitterMoved.connect(
                lambda *_: self._on_workspace_changed(SUBPOPULATION_ARTIFACT_ID))

    def _checkpoint_workspace_surface(
            self, artifact_id: str, *, capture_frozen: bool = False) -> bool:
        """Synchronously copy one surface into the project-owned cache.

        ``PlotsWindow.closeEvent`` reaches this seam before deliberately resetting itself.
        Closing a window is not project data, so the checkpoint itself is silent; every
        durable control/result already reaches :meth:`_on_workspace_changed`.  A failed
        checkpoint is different: the cache may now be incomplete, so keep the project
        visibly dirty and let Save Project retry/report it.
        """
        if self._workspace_installing or self._workspace_navigating or self._restoring:
            return False
        # An untouched restored snapshot already lives in the cache.  Re-reading its
        # widget at close could restamp it against a ResultStore that changed underneath
        # while frozen; the later Hide event merely makes that same artifact pending for
        # the next lazy install.
        if artifact_id in self._workspace_frozen and not capture_frozen:
            return True
        surface = self._workspace_surface(artifact_id)
        keyword = self._workspace_kwarg(artifact_id)
        if surface is None or keyword is None:
            return False
        try:
            self._analytical_workspace = capture_analysis_workspace(
                self._analytical_workspace, **{keyword: surface})
        except Exception as exc:  # noqa: BLE001 - the save path retries and reports fully
            self._mark_dirty()
            self.statusBar().showMessage(
                f"Could not checkpoint this analytical view yet: {exc}. "
                "Save Project will try again.", 12000)
            return False
        self._workspace_pending.discard(artifact_id)
        # Keep an installed project snapshot insulated from session-navigation refreshes
        # when this was merely a durable style/config edit.  Explicit Refresh routes
        # through ``_replace_workspace_surface`` and clears the bit *before* computing;
        # generic ``workspaceChanged`` cannot tell a title colour from a numerical run and
        # therefore must take the lossless/conservative side of that boundary.
        self._sync_saved_subpop_action()
        return True

    def _on_workspace_changed(self, artifact_id: str) -> None:
        """Checkpoint one durable edit/result and advance the recovery revision."""
        if not self._checkpoint_workspace_surface(artifact_id, capture_frozen=True):
            return
        self._mark_dirty()

    def _capture_analytical_workspace(self) -> AnalyticalWorkspace:
        """Capture safe live surfaces for Save/Recovery, preserving every carried block.

        Hidden PlotsWindow is intentionally excluded: its close contract blanked the
        widget after ``_on_workspace_changed`` cached the real result, and gathering every
        instantiated window here would overwrite that cache with defaults.  A pending
        block is likewise excluded until successfully installed or explicitly edited, so
        an unavailable input cannot be converted into an empty artifact by a later save.
        A frozen restored surface is already represented by that authoritative artifact;
        leaving it out prevents a changed live ResultStore from silently restamping an old
        snapshot merely because the user pressed Save while its window was visible.
        """
        supplied: dict[str, Any] = {}
        if ANALYSIS_PANEL_ARTIFACT_ID not in self._workspace_pending:
            supplied["analysis_panel"] = self._analysis_panel
        for artifact_id in (
                PLOTS_ARTIFACT_ID, SUBPOPULATION_ARTIFACT_ID, PSD_ARTIFACT_ID,
                DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
                COMPARISON_FIGURES_ARTIFACT_ID,
                EVENT_PLOTS_ARTIFACT_ID, IV_ARTIFACT_ID,
                VOLTAGE_DEPENDENCE_ARTIFACT_ID):
            window = self._workspace_surface(artifact_id)
            # A live PSD whose last eligible current channel disappeared has no result
            # to save.  Its stale checkpoint is removed by ``_refresh_surface_windows``;
            # do not recreate an empty artifact merely because the still-visible window
            # is included in a later Save/Recovery gather.  Frozen saved snapshots take
            # the earlier exclusion branch and remain readable without their source.
            if (artifact_id == PSD_ARTIFACT_ID and window is not None
                    and not window.has_data()):
                continue
            if (artifact_id not in self._workspace_pending
                    and artifact_id not in self._workspace_frozen
                    and _on_screen(window)):
                supplied[self._workspace_kwarg(artifact_id)] = window
        self._analytical_workspace = capture_analysis_workspace(
            self._analytical_workspace, **supplied)
        self._sync_saved_subpop_action()
        return self._analytical_workspace

    def _restore_workspace_surface(self, artifact_id: str, surface) -> bool:
        """Install one pending artifact into an already-built/bound surface, once."""
        if artifact_id not in self._workspace_pending:
            return False
        keyword = self._workspace_kwarg(artifact_id)
        if keyword is None or self._workspace_artifact(artifact_id) is None:
            self._workspace_pending.discard(artifact_id)
            return False
        self._workspace_installing = True
        try:
            report = restore_analysis_workspace(
                self._analytical_workspace, **{keyword: surface})
        finally:
            self._workspace_installing = False
        restored = artifact_id in report.restored
        if restored:
            self._workspace_pending.discard(artifact_id)
            self._workspace_frozen.add(artifact_id)
        if report.warnings:
            sentence = " ".join(str(value) for value in report.warnings)
            if self._restoring:
                self._restore_notes.append(sentence)
            else:
                self.statusBar().showMessage(
                    f"Saved analytical view restored with a caveat: {sentence}", 15000)
        return restored

    def _replace_workspace_surface(self, artifact_id: str, refresh) -> None:
        """Explicit Refresh: leave frozen mode, recompute, then checkpoint the new result."""
        self._workspace_pending.discard(artifact_id)
        self._workspace_frozen.discard(artifact_id)
        refresh()
        self._on_workspace_changed(artifact_id)

    def _reset_analytical_workspace(self) -> None:
        """Drop the outgoing project's analytical state without emitting an edit.

        Plots and the subpopulation report predate the project-owned workspace and have a
        useful established contract: if the user leaves either visible while opening a
        project, that same window stays visible and is re-fed at the end of restore.  Keep
        those two *widget shells* but scrub every source, result and identity sidecar.
        Numerical inspectors are rebuilt lazily because their private arrays have no
        comparable blank-state contract.  The caller has already completed the outgoing
        project's Save/Discard decision, so none of this checkpoints or marks dirty.
        """
        self._figure_project_epoch += 1
        self._level_fit_pending = None
        self._workspace_installing = True
        try:
            if self._plots_window is not None:
                abort = getattr(self._plots_window, "_abort_note_job", None)
                if callable(abort):
                    abort()
                self._plots_window.reset_state()
                for sidecar in ("_nanoscope_unresolved_plot_sources",
                                "_nanoscope_unresolved_plot_results"):
                    if hasattr(self._plots_window, sidecar):
                        delattr(self._plots_window, sidecar)
            if self._subpop_window is not None:
                self._subpop_window.set_report(None)
                # These are constructor defaults.  Splitter positions are durable view
                # state, so an incoming project without a saved report must not inherit
                # the outgoing project's arrangement.
                self._subpop_window.splitter().setSizes([620, 800])
                self._subpop_window.figures_splitter().setSizes([400, 320, 190])
                for sidecar in ("_nanoscope_workspace_source",
                                "_nanoscope_workspace_source_result",
                                "_nanoscope_workspace_accepted_event_ids"):
                    if hasattr(self._subpop_window, sidecar):
                        delattr(self._subpop_window, sidecar)
            reset_pages = {
                "_distribution_summary_window": PAGE_DISTRIBUTION_SUMMARIES,
                "_comparison_figures_window": PAGE_COMPARISON_FIGURES,
                "_psd_window": PAGE_NOISE,
                "_event_plot_window": PAGE_EVENT_WAVEFORMS,
                "_iv_window": PAGE_IV,
                "_vdep_window": PAGE_VOLTAGE_SERIES,
            }
            for name, page_id in reset_pages.items():
                window = getattr(self, name, None)
                if window is None:
                    continue
                if self._analysis_explorer is not None:
                    self._analysis_explorer.remove_page(page_id)
                try:
                    window.hide()
                    window.deleteLater()
                except RuntimeError:
                    pass
                setattr(self, name, None)
            # If the outgoing project had only disposable numerical pages open, do not
            # leave an empty Analysis Explorer floating over the newly reset workspace.
            # The shell object is retained and will reopen with the next lazy page.
            if (self._analysis_explorer is not None
                    and self._analysis_explorer.page_count() == 0):
                self._analysis_explorer.close()
            self._drop_subpop_assignment()
            self._subpop_pending = False
            self._subpop_source_recipe_pending = None
            self._subpop_signal_pending = None
            self._subpop_accepted_ids_pending = None
            self._analytical_workspace = AnalyticalWorkspace()
            if self._figure_library is not None:
                self._figure_library.refresh()
            self._workspace_pending.clear()
            self._workspace_frozen.clear()

            # An old project has no analysis-panel artifact.  Replay the constructor
            # defaults now so its absence means EMPTY, not "whatever the last project
            # happened to leave in the always-live panel".
            default = self._workspace_default_analysis
            if default is not None:
                restore_analysis_workspace(
                    AnalyticalWorkspace(artifacts=[default]),
                    analysis_panel=self._analysis_panel)
        finally:
            self._workspace_installing = False
        self._sync_saved_subpop_action()
        self._sync_analysis_open_state()

    # -- "this window is already open" indication ------------------------------------------------
    def _analysis_home_availability(self, route: str) -> tuple[bool, str]:
        """Read-only navigation gates; inspecting the catalog never constructs a result."""
        gates = {
            PAGE_DISTRIBUTIONS: self._gate_populations,
            PAGE_EVENT_WAVEFORMS: self._gate_event_plots,
            PAGE_NOISE: self._gate_psd,
            PAGE_IV: self._gate_iv,
            PAGE_VOLTAGE_SERIES: self._gate_vdep,
            "event_gallery": self._gate_gallery,
            "report_figures": self._gate_figures,
        }
        if route in gates:
            return gates[route]()
        if route in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES):
            return True, "Open the controls; choose eligible source tables before generating a figure."
        if route == PAGE_SUBPOPULATIONS:
            return True, "Open an existing report or an empty view. Generate explicitly to run an analysis."
        if route in (PAGE_HOME, "figure_library", "guided_first_analysis"):
            return True, ""
        return False, "Unknown analysis destination."

    def _launch_analysis_route(self, route: str) -> None:
        available, reason = self._analysis_home_availability(route)
        if not available:
            self.statusBar().showMessage(reason, 8000)
            return
        actions = {
            PAGE_HOME: self._open_analysis_home,
            "guided_first_analysis": self._open_first_analysis_guide,
            PAGE_DISTRIBUTIONS: self._open_plots,
            PAGE_DISTRIBUTION_SUMMARIES: self._open_distribution_summaries,
            PAGE_COMPARISON_FIGURES: self._open_comparison_figures,
            PAGE_EVENT_WAVEFORMS: self._open_event_plots,
            PAGE_NOISE: self._open_psd,
            PAGE_IV: self._open_iv,
            PAGE_VOLTAGE_SERIES: self._open_vdep,
            PAGE_SUBPOPULATIONS: self._open_subpopulations_view,
            "event_gallery": self._open_event_gallery,
            "figure_library": self._open_figure_library,
            "report_figures": self._export_figures,
        }
        actions[route]()

    def _open_subpopulations_view(self) -> None:
        """Opening a destination is not permission to run a mixture analysis."""
        if self._workspace_has_artifact(SUBPOPULATION_ARTIFACT_ID):
            self._show_saved_subpop_report()
        else:
            self._show_analysis_page(PAGE_SUBPOPULATIONS, self._ensure_subpop_window(),
                                     refresh=self._run_subpopulations)

    def _open_analysis_home(self) -> None:
        self._ensure_analysis_explorer().show_home()

    def _first_analysis_context(self):
        return self._workflow_target(), self._analysis_explorer_context()

    def _first_analysis_availability(self, route: str) -> tuple[bool, str]:
        if route in ("open_recording", "practice", "analysis_current", "save_project"):
            return True, ""
        stages = {"calibration": Stage.CALIBRATE, "baseline": Stage.FILTER,
                  "detection": Stage.DETECT, "essential": Stage.DETECT,
                  "advanced": Stage.DETECT, "event_inspector": Stage.CURATE}
        if route in stages:
            status = compute_status(self._session, *self._workflow_target()).stage(stages[route])
            return status.available, status.reason
        return self._analysis_home_availability(route)

    def _open_first_analysis_guide(self) -> None:
        from .first_analysis_guide import FirstAnalysisGuide
        if self._first_analysis_guide is None:
            self._first_analysis_guide = FirstAnalysisGuide(
                self, context_provider=self._first_analysis_context,
                availability_provider=self._first_analysis_availability)
            self._first_analysis_guide.actionRequested.connect(self._first_analysis_action)
            self._first_analysis_guide.controlLevelRequested.connect(
                lambda level: self._analysis_panel.detection_panel.set_control_level(level, persist=True))
        self._first_analysis_guide.refresh_context()
        self._first_analysis_guide.show()
        self._first_analysis_guide.raise_()
        self._first_analysis_guide.activateWindow()

    def _open_build_info(self) -> None:
        from .build_info_dialog import show_build_info
        show_build_info(self)

    def _first_analysis_action(self, route: str) -> None:
        available, reason = self._first_analysis_availability(route)
        if not available:
            self.statusBar().showMessage(reason, 8000)
            return
        if route == "open_recording":
            self._open_dialog()
        elif route == "practice":
            self._create_practice_recording()
        elif route == "calibration":
            self._reveal_calibration_panel()
        elif route in ("baseline", "detection", "essential", "advanced"):
            self._reveal_detection_panel()
            panel = self._analysis_panel.detection_panel
            if route == "baseline":
                panel.show_open_pore_baseline()
            else:
                panel.show_detection_tab()
                if route in ("essential", "advanced"):
                    panel.set_control_level(route, persist=False)
        elif route == "event_inspector":
            self._open_event_inspector()
        elif route == "analysis_current":
            if self._analysis_explorer is None:
                self._open_analysis_home()
            else:
                self._analysis_explorer.show()
                self._analysis_explorer.raise_()
        elif route == "save_project":
            self._save_project_from(self._first_analysis_guide or self, save_as=False)
        else:
            self._launch_analysis_route(route)

    def _create_practice_recording(self) -> None:
        from .first_analysis_guide import practice_path, write_practice_recording
        parent = self._first_analysis_guide or self
        answer = QtWidgets.QMessageBox.question(
            parent, "Create synthetic practice recording",
            "Create synthetic learning data and add it alongside your open recordings? "
            "This is not experimental data. No analysis settings will be changed. "
            "The saved filename will include SYNTHETIC.",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer != QtWidgets.QMessageBox.StandardButton.Ok:
            return
        chosen = ask_export_path(parent, title="Save synthetic practice recording",
            suggestion="SYNTHETIC-NanoScope-practice.csv", filters="CSV files (*.csv)",
            outputs=lambda path: [practice_path(path)])
        if chosen is None:
            return
        target = practice_path(chosen)
        if any(doc.meta.path.resolve() == target.resolve() for doc in self._session.documents()):
            QtWidgets.QMessageBox.warning(parent, "Recording is already open",
                "Choose a new filename. A practice recording cannot replace a file currently open in this project.")
            return
        try:
            target = write_practice_recording(target)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(parent, "Practice recording could not be saved", str(exc))
            return
        self.open_paths([target])

    def _ensure_analysis_explorer(self) -> AnalysisExplorer:
        """Build the single analytical shell without constructing any analysis page.

        Pages remain lazy because several of their constructors perform real numerical
        work.  The Explorer is presentation only; every existing page attribute and
        workspace adapter remains the source of truth.
        """
        if self._analysis_explorer is None:
            self._analysis_explorer = AnalysisExplorer(
                self,
                context_provider=self._analysis_explorer_context,
                status_provider=self._analysis_explorer_status,
                availability_provider=self._analysis_home_availability,
            )
            self._analysis_explorer.visibilityChanged.connect(
                lambda _visible: self._sync_analysis_open_state())
            self._analysis_explorer.saveFigureRequested.connect(self._save_analysis_figure)
            self._analysis_explorer.saveBundleRequested.connect(self._export_analysis_bundle)
            self._analysis_explorer.figureLibraryRequested.connect(self._open_figure_library)
            self._analysis_explorer.launchRequested.connect(self._launch_analysis_route)
        return self._analysis_explorer

    def _analysis_explorer_context(self) -> str:
        """Short recording context shared by every Explorer page."""
        documents = list(self._session.documents())
        count = len(documents)
        if (self._current_selection is not None
                and self._session.has_document(self._current_selection[0])):
            doc_id, channel = self._current_selection
            name = self._session.document(doc_id).display_name
            suffix = f" · {count} recording{'s' if count != 1 else ''} open"
            return f"{name} · channel {int(channel) + 1}{suffix}"
        if count:
            return f"{count} recording{'s' if count != 1 else ''} open · none selected"
        return "No recordings open"

    # -- project Figure Library (not the Event Gallery) ------------------------------------
    def _figure_entries(self):
        from ..core.figure_library import figures
        return figures(self._analytical_workspace)

    def _figure_status(self, entry):
        from ..core.figure_library import figure_status, source_state
        try:
            live = source_state(self, entry.payload.get("source_recipes", []))
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            live = None
        return figure_status(entry, live)

    def _append_figure(self, entry) -> None:
        self._analytical_workspace.artifacts.append(entry)

    def _open_figure_library(self) -> None:
        from .figure_library import FigureLibrary
        if self._figure_library is None:
            self._figure_library = FigureLibrary(
                self, entries=self._figure_entries, status=self._figure_status,
                append=self._append_figure, project_token=lambda: self._figure_project_epoch)
            self._figure_library.changed.connect(self._mark_dirty)
        self._figure_library.show()
        self._figure_library.raise_()
        self._figure_library.activateWindow()

    def _figure_page(self, page_id: str):
        artifact_id = self._analysis_page_artifact(page_id)
        page = self._workspace_surface(artifact_id) if artifact_id else None
        if page is None:
            raise ValueError("Open an analytical figure before saving it.")
        return artifact_id, page

    def _figure_capture_options(self, page_id: str, *, ask_name: bool):
        """Name/residual inclusion only: no analysis controls or live view mutation."""
        _artifact_id, page = self._figure_page(page_id)
        residual_getter = getattr(page, "figure_residual_target", None)
        residual = residual_getter() if callable(residual_getter) else None
        if not ask_name and residual is None:
            return "Figure", False
        dialog = QtWidgets.QDialog(self._analysis_explorer or self)
        dialog.setWindowTitle("Save figure snapshot" if ask_name else "Figure bundle")
        layout = QtWidgets.QVBoxLayout(dialog)
        name = QtWidgets.QLineEdit(str(page_id).replace("_", " ").title())
        if ask_name:
            name.setMaxLength(200)
            layout.addWidget(QtWidgets.QLabel("Figure name:"))
            layout.addWidget(name)
        include = QtWidgets.QCheckBox("Include the displayed residual panel")
        include.setToolTip("Copies the existing residuals without fitting or recomputing them.")
        include.setVisible(residual is not None)
        layout.addWidget(include)
        note = QtWidgets.QLabel(
            "Saves the visible picture and exact plotted numbers. Pending analysis changes "
            "are recorded separately; this does not apply them.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if ask_name:
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setEnabled(bool(name.text().strip()))
            name.textChanged.connect(lambda text: buttons.button(
                QtWidgets.QDialogButtonBox.StandardButton.Save).setEnabled(bool(text.strip())))
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return name.text().strip(), include.isChecked()

    def _capture_figure(self, page_id: str, *, include_residual: bool = False):
        """Copy one visible result. No checkpoint, navigation, fitting or replotting."""
        from ..core.figure_library import source_recipes, source_state
        from .figure_bundle import capture_figure_snapshot, encode_figure_table

        artifact_id, page = self._figure_page(page_id)
        keyword = self._workspace_kwarg(artifact_id)
        captured = capture_analysis_workspace(**{keyword: page})
        analysis = next(a for a in captured.artifacts if a.artifact_id == artifact_id)
        bindings = getattr(page, "figure_source_recipes", None)
        if callable(bindings):
            analysis.payload["figure_sources"] = bindings()
        getter = getattr(page, "figure_export_target", None)
        ordered = getattr(page, "figure_export_items", None)
        residual_getter = getattr(page, "figure_residual_target", None)
        residual = residual_getter() if callable(residual_getter) else None
        components = ["current figure"]
        report_companion = False
        if callable(ordered):
            items = list(ordered())
            if not items or len(items) > 2:
                raise ValueError("This figure does not expose a supported frozen canvas.")
            target = items[0]
            if len(items) == 2:
                residual = items[1]
                include_residual = True
                components = ["embedding", "waveform overlay"]
                report_companion = True
        elif callable(getter):
            target = getter()
        else:
            raise ValueError("This view does not yet expose a figure export target.")
        data = page._plot_data()
        if data.empty:
            raise ValueError("There are no plotted numerical values to save.")
        _label, capture_status = self._analysis_explorer_status(page_id, page)
        recipes = source_recipes(analysis)
        live = source_state(self, recipes) if capture_status == "current" else None
        names = []
        for recipe in recipes:
            label = str(recipe.get("source_key") or "")
            channel = recipe.get("channel")
            if channel is not None:
                label += f" · channel {int(channel) + 1}"
            if label and label not in names:
                names.append(label)
        source_label = "; ".join(names) or "Saved analytical source (see provenance)"
        resources = {}
        table_names = {}
        for index, (key, frame) in enumerate(analysis.tables.items()):
            filename = f"analysis_tables/table_{index}.parquet"
            resources[filename] = encode_figure_table(frame)
            table_names[key] = filename
        blob_names = {}
        for index, (key, value) in enumerate(analysis.blobs.items()):
            extension = ".parquet" if str(key).endswith(".parquet") else ".bin"
            filename = f"analysis_blobs/blob_{index}{extension}"
            resources[filename] = bytes(value)
            blob_names[key] = filename
        if not include_residual:
            # Optional exact Cartesian replay enables true shared axes on detached
            # copies. Unsupported graphics remain usable as frozen composition panels.
            from .figure_composition import capture_cartesian_model
            model = capture_cartesian_model(target)
            if model is not None:
                resources["rendering/cartesian.json"] = model
        applied_getter = getattr(page, "figure_applied_config", None)
        applied = applied_getter() if callable(applied_getter) else None
        if applied is not None:
            # The plot API provides JSON-shaped applied settings, never live objects.
            applied_config = applied
        else:
            applied_config = None
        details_getter = getattr(page, "details_text", None)
        figure_details = str(details_getter()) if callable(details_getter) else ""
        snapshot = capture_figure_snapshot(
            target, data=data,
            config={"analysis_kind": analysis.kind, "schema_version": analysis.schema_version,
                    "workspace_state": analysis.payload,
                    "applied_settings": applied_config,
                    "state_contract": "workspace_state includes current controls; data and saved picture describe the visible result",
                    "tables": table_names, "blobs": blob_names},
            provenance={"source": source_label, "source_recipes": recipes,
                        "capture_status": capture_status, "components": components,
                        "details": figure_details,
                        "analysis_recomputed": False, "source_state": live},
            residual_item=residual, include_residual=include_residual,
            additional_resources=resources)
        if report_companion:
            # Reuse the two-canvas renderer, but never call a waveform a residual
            # in the durable manifest. Its true panel identities are in provenance.
            from dataclasses import replace
            snapshot = replace(snapshot, includes_residual=False)
        return analysis, snapshot, source_label, live, capture_status

    def _save_analysis_figure(self, page_id: str, name: str | None = None, *,
                              include_residual: bool = False):
        from ..core.figure_library import make_figure
        try:
            if name is None:
                selected = self._figure_capture_options(page_id, ask_name=True)
                if selected is None:
                    return None
                name, include_residual = selected
            analysis, snapshot, source_label, live, status = self._capture_figure(
                page_id, include_residual=include_residual)
            metadata, blobs = snapshot.to_storage()
            entry = make_figure(
                name=name, page_id=page_id, source_label=source_label, analysis=analysis,
                snapshot_metadata=metadata, snapshot_blobs=blobs, live_state=live,
                capture_status=status)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self._analysis_explorer or self,
                                          "Save figure snapshot", str(exc))
            return None
        self._append_figure(entry)
        self._mark_dirty()
        self._open_figure_library()
        self._figure_library.refresh(entry.artifact_id)
        self.statusBar().showMessage("Figure saved in this project. Save Project to persist it to disk.", 10000)
        return entry

    def _export_analysis_bundle(self, page_id: str, path: str | Path | None = None, *,
                                include_residual: bool = False):
        from .export_paths import ask_export_path
        from .figure_bundle import write_figure_bundle
        from ..core.atomic_write import take_identity_notes
        parent = self._analysis_explorer or self
        try:
            if path is None:
                selected = self._figure_capture_options(page_id, ask_name=False)
                if selected is None:
                    return None
                _name, include_residual = selected
                path = ask_export_path(
                    parent, title="Save figure bundle", suggestion=f"{page_id}_figure.zip",
                    filters="Figure bundle (*.zip)",
                    outputs=lambda chosen: [chosen if chosen.suffix else chosen.with_suffix(".zip")])
                if path is None:
                    return None
            out = Path(path)
            if not out.suffix:
                out = out.with_suffix(".zip")
            if out.suffix.lower() != ".zip":
                raise ValueError("Choose a .zip filename for the figure bundle.")
            _analysis, snapshot, _source, _live, _status = self._capture_figure(
                page_id, include_residual=include_residual)
            write_figure_bundle(out, snapshot)
            notes = take_identity_notes(out)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(parent, "Save figure bundle", str(exc))
            return None
        if notes:
            QtWidgets.QMessageBox.information(parent, "Save figure bundle", "\n\n".join(notes))
        self.statusBar().showMessage(f"Saved figure bundle → {out}", 10000)
        return out

    @staticmethod
    def _analysis_page_artifact(page_id: str) -> str | None:
        return {
            PAGE_DISTRIBUTIONS: PLOTS_ARTIFACT_ID,
            PAGE_DISTRIBUTION_SUMMARIES: DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
            PAGE_COMPARISON_FIGURES: COMPARISON_FIGURES_ARTIFACT_ID,
            PAGE_EVENT_WAVEFORMS: EVENT_PLOTS_ARTIFACT_ID,
            PAGE_NOISE: PSD_ARTIFACT_ID,
            PAGE_IV: IV_ARTIFACT_ID,
            PAGE_VOLTAGE_SERIES: VOLTAGE_DEPENDENCE_ARTIFACT_ID,
            PAGE_SUBPOPULATIONS: SUBPOPULATION_ARTIFACT_ID,
        }.get(page_id)

    def _analysis_explorer_status(
            self, page_id: str, page: QtWidgets.QWidget) -> tuple[str, str]:
        """Return the visible page's freshness label without recomputing anything."""
        if page_id in (PAGE_DISTRIBUTIONS, PAGE_DISTRIBUTION_SUMMARIES,
                       PAGE_COMPARISON_FIGURES) and bool(getattr(page, "_stale", False)):
            return "STALE", "stale"
        artifact_id = self._analysis_page_artifact(page_id)
        if artifact_id in self._workspace_frozen:
            return "SAVED SNAPSHOT", "snapshot"
        if artifact_id in self._workspace_pending:
            return "SAVED SNAPSHOT", "snapshot"

        if page_id == PAGE_DISTRIBUTIONS:
            has_data = bool(page.sources())
        elif page_id in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES):
            has_data = page.has_data()
        elif page_id in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE):
            has_data = bool(page.has_data())
        elif page_id in (PAGE_IV, PAGE_VOLTAGE_SERIES):
            has_data = bool(page.points())
        elif page_id == PAGE_SUBPOPULATIONS:
            has_data = page.report() is not None
        else:  # pragma: no cover - AnalysisExplorer rejects unknown page ids
            has_data = False
        return ("CURRENT", "current") if has_data else ("NO DATA", "empty")

    def _show_analysis_page(
            self, page_id: str, page: QtWidgets.QWidget, *, refresh=None) -> None:
        """Put an existing analytical widget in the one Explorer and focus it."""
        explorer = self._ensure_analysis_explorer()
        explorer.show_page(
            page_id, page, refresh=refresh,
            activate=lambda _id=page_id: self._activate_analysis_page(_id),
        )
        self._sync_analysis_open_state()

    def _activate_analysis_page(self, page_id: str) -> None:
        """Refresh a live page reached directly through the Explorer's tab bar.

        This is deliberately different from the shared explicit Refresh button: a tab
        switch follows ordinary live session navigation and never replaces a frozen
        project snapshot.  The callback runs synchronously from ``currentChanged``, before
        Qt returns to the event loop to paint the newly-selected page.
        """
        if self._workspace_installing or self._workspace_navigating or self._restoring:
            return
        artifact_id = self._analysis_page_artifact(page_id)
        if artifact_id in self._workspace_frozen or artifact_id in self._workspace_pending:
            self._refresh_analysis_explorer_header()
            return
        if page_id == PAGE_DISTRIBUTIONS and self._plots_window is not None:
            self._refresh_plots()
        elif page_id == PAGE_DISTRIBUTION_SUMMARIES and self._distribution_summary_window is not None:
            self._offer_distribution_sources()
        elif page_id == PAGE_COMPARISON_FIGURES and self._comparison_figures_window is not None:
            self._offer_comparison_sources()
        elif page_id == PAGE_EVENT_WAVEFORMS and self._event_plot_window is not None:
            self._refresh_event_plots()
        elif page_id == PAGE_NOISE and self._psd_window is not None:
            self._point_psd_at_selection()
        elif page_id == PAGE_IV and self._iv_window is not None:
            self._iv_window.refresh()
        elif page_id == PAGE_VOLTAGE_SERIES and self._vdep_window is not None:
            self._vdep_window.refresh()
        elif page_id == PAGE_SUBPOPULATIONS and self._subpop_window is not None:
            self._sync_subpop_window()
        self._refresh_analysis_explorer_header()

    def _refresh_analysis_explorer_header(self) -> None:
        explorer = self._analysis_explorer
        if explorer is not None:
            explorer.refresh_header()

    def _surface_windows(self) -> dict:
        """Surface id -> its single lazily-built window (``None`` until first opened).

        The figure export has no window (it is a file dialog), so it is absent here and simply
        never lights up."""
        return {
            SURFACE_POPULATIONS: self._plots_window,
            SURFACE_EVENT_PLOTS: self._event_plot_window,
            SURFACE_PSD: self._psd_window,
            SURFACE_IV: self._iv_window,
            SURFACE_VDEP: self._vdep_window,
            SURFACE_GALLERY: self._event_gallery,
        }

    def _sync_analysis_open_state(self) -> None:
        """Tint the launcher buttons whose window is currently on screen (these windows are
        single instances that re-focus rather than re-open, so 'open' is worth saying)."""
        windows = self._surface_windows()
        explorer = self._analysis_explorer
        explorer_pages = {
            SURFACE_POPULATIONS: PAGE_DISTRIBUTIONS,
            SURFACE_EVENT_PLOTS: PAGE_EVENT_WAVEFORMS,
            SURFACE_PSD: PAGE_NOISE,
            SURFACE_IV: PAGE_IV,
            SURFACE_VDEP: PAGE_VOLTAGE_SERIES,
        }
        for sid in SURFACE_ORDER:
            win = windows.get(sid)
            page_id = explorer_pages.get(sid)
            in_open_explorer = bool(
                explorer is not None and explorer.isVisible() and page_id is not None
                and explorer.page(page_id) is win
                and win is not None
                and not bool(win.property("nanoscopeAnalysisExplorerClosing"))
            )
            self._analysis_launcher.set_open(
                sid, in_open_explorer or (win is not None and win.isVisible()))

    def _watch_surface_window(self, win) -> None:
        """Track a just-built surface window's show/hide so the launcher's open indication stays
        honest when the user closes the window directly (``close()`` ends in a hide event)."""
        win.installEventFilter(self)

    def _wire_surface_refresh(self, role: str, source) -> None:
        """Connect every signal :data:`SURFACE_MUTATION_SIGNALS` lists under ``role`` to the
        one refresher.

        The only door through which a signal reaches ``_refresh_surface_windows``. Wiring
        from the table rather than by hand is what makes the registry load-bearing instead
        of documentation: a name added to the table is wired everywhere the role is wired,
        and a signal that is in NEITHER table fails ``test_surface_staleness.py`` before it
        can reach a user.

        An unknown role, or a name the object does not carry, is a programming error and
        says so — silently skipping it would restore exactly the failure mode this replaces.
        """
        try:
            names = SURFACE_MUTATION_SIGNALS[role]
        except KeyError:                       # pragma: no cover - defended by the guard test
            raise KeyError(
                f"{role!r} is not a role in SURFACE_MUTATION_SIGNALS "
                f"(have: {sorted(SURFACE_MUTATION_SIGNALS)})") from None
        for name in names:
            signal = getattr(source, name, None)
            if signal is None or not hasattr(signal, "connect"):   # pragma: no cover
                raise AttributeError(
                    f"SURFACE_MUTATION_SIGNALS[{role!r}] names {name!r}, which "
                    f"{type(source).__name__} does not declare as a signal")
            signal.connect(lambda *_: self._refresh_surface_windows())

    def _refresh_surface_windows(self) -> None:
        """*** NO STANDALONE WINDOW OUTLIVES THE DATA IT IS SHOWING. ***

        Re-feed every open surface window from the CURRENT session. One method, wired to
        every signal that can change what the session contains, because the alternative —
        each window wired to the signals somebody remembered — is what was measured:
        ``_refresh_plots`` was connected to ``resultsChanged`` and to nothing else, so
        closing the recording left the Plots window holding 5 rows of a detection the
        session no longer had, with a working Fit and a working Export CSV writing numbers
        attributed to a recording that is not open. ``_on_document_closed`` did refresh
        three of the five windows, but only when the closed document happened to be the
        selected one — and the Plots window, which reads across ALL open recordings, was
        not among the three either way.

        A window the user has CLOSED is not re-fed. Each refresher's own guard is "has this
        window ever been built?", which is not the same question: `close()` on a surface
        window only hides it (WA_DeleteOnClose is set nowhere in nanoscope/gui, deliberately
        — see :meth:`closeEvent`), so once a window existed it was refreshed forever, with
        nothing on screen to show for it. Those refreshes are real compute, not repaints:
        :meth:`_point_psd_at_selection` runs a full synchronous Welch PSD, `_refresh_plots`
        re-pulls every open recording's event table and `_refresh_event_plots` the whole
        calibrated array. Measured over one accept/reject with five surfaces opened once
        and closed: 5.5 ms -> 43.3 ms, and 14.1 ms -> 218.7 ms across a 314-operation
        session — paid on EVERY keystroke of the app's most repetitive task.

        NOTHING IS LOST BY THAT, and that is a property of the openers rather than a hope:
        `_open_plots`, `_open_event_plots`, `_open_event_gallery`, `_open_event_inspector`,
        `_open_psd`, `_open_iv` and `_open_vdep` every one call their own refresher
        IMMEDIATELY BEFORE `show()`, so a window reopened after a skipped refresh is fed the
        current session on the way back up. The freshness guarantee at the top of this
        docstring is about what the user can SEE, and a hidden window shows nothing.

        The two entries that are NOT windows stay unconditional: `_reconcile_level_fit`
        owns a derived table that the export and the project file read, and
        `_refresh_subpop_views` owns app-wide colouring. Both are about state that outlives
        any window, and both measured 0.0 ms.

        Each refresher is already a no-op when its window has never been built, and each
        already re-derives its contents from the session rather than patching them, so this
        is cheap and idempotent. Every window that reads the session belongs here; a sixth
        one added later is one line, in the one place a reviewer will look.

        WHICH SIGNALS REACH HERE is :data:`SURFACE_MUTATION_SIGNALS`, at the top of this
        file, and nothing else — see :meth:`_wire_surface_refresh`. This method used to be
        connected to four of the ten, and the six that were missing were the six a curator
        presses: rejecting 159 events on the owner's ``_0006_`` recording left the
        Populations window drawing tau = 0.01429 s over n = 316 when the truth was
        0.008773 s over n = 158.

        The CUSUM level fit is reconciled first for the same reason the windows are
        re-fed: it is a derived table that describes spans, and a mask drag moves them.
        """
        self._reconcile_level_fit()         # ...and no DERIVED table outlives it either
        if (_on_screen(self._distribution_summary_window)
                and DISTRIBUTION_SUMMARIES_ARTIFACT_ID not in self._workspace_frozen
                and DISTRIBUTION_SUMMARIES_ARTIFACT_ID not in self._workspace_pending):
            self._offer_distribution_sources()
        if (_on_screen(self._comparison_figures_window)
                and COMPARISON_FIGURES_ARTIFACT_ID not in self._workspace_frozen
                and COMPARISON_FIGURES_ARTIFACT_ID not in self._workspace_pending):
            self._offer_comparison_sources()
        if _on_screen(self._plots_window):
            if (PLOTS_ARTIFACT_ID in self._workspace_pending
                    and not self._restoring):
                # A retained window first binds the incoming project's durable source
                # identities, then installs its self-contained numerical/config snapshot.
                # Restoring earlier (on the first documentOpened) would resolve against a
                # half-built session and freeze the result before later sources arrive.
                self._refresh_plots()
                self._restore_workspace_surface(
                    PLOTS_ARTIFACT_ID, self._plots_window)
            elif PLOTS_ARTIFACT_ID not in self._workspace_frozen:
                self._refresh_plots()       # every open recording's event table
        if (_on_screen(self._event_plot_window)
                and EVENT_PLOTS_ARTIFACT_ID not in self._workspace_frozen):
            self._refresh_event_plots()     # the active detection (density / ensemble)
        if _on_screen(self._event_inspector):
            self._refresh_event_inspector() # the active detection's shared frame
        if _on_screen(self._event_gallery):
            self._refresh_event_gallery()
        self._refresh_subpop_views()        # ...and the report's colouring, if it still applies
        if (_on_screen(self._subpop_window)
                and SUBPOPULATION_ARTIFACT_ID in self._workspace_pending
                and not self._restoring):
            restored = self._restore_workspace_surface(
                SUBPOPULATION_ARTIFACT_ID, self._subpop_window)
            if restored:
                # Opening a project creates new Result objects.  Even identical event ids
                # therefore cannot silently re-adopt an outgoing/in-file assignment onto
                # this load.  Keep the report readable and require the deliberate saved-
                # report resolver (or a re-run) to attach its colouring.  A report which
                # fitted no mixture never had an assignment and remains the honest NONE.
                artifact = self._workspace_artifact(SUBPOPULATION_ARTIFACT_ID)
                saved_state = (str(artifact.payload.get("assignment_state") or "none")
                               if artifact is not None
                               and isinstance(artifact.payload, Mapping) else "none")
                self._subpop_absent_reason = (
                    "This report's assignment was retired when the project was opened; "
                    "the restored event frame is a new load, so event ids alone cannot "
                    "reattach it. Re-run Analysis ▸ Generate Subpopulations to colour "
                    "the current detection."
                    if saved_state != self._SUBPOP_NONE else "")
                self._sync_subpop_window()
        if (_on_screen(self._psd_window)
                and PSD_ARTIFACT_ID not in self._workspace_frozen):
            self._point_psd_at_selection()  # the selected recording's spectrum
            if not self._psd_window.has_data():
                # This is a LIVE result invalidation, not an unavailable source for a
                # frozen saved snapshot (the latter never enters this branch).  Retire the
                # stale checkpoint so the menu and the visibly empty window agree.
                self._drop_workspace_artifact(PSD_ARTIFACT_ID)
        if (_on_screen(self._iv_window)
                and IV_ARTIFACT_ID not in self._workspace_frozen):
            self._iv_window.refresh()       # every open recording's I-V point
        if (_on_screen(self._vdep_window)
                and VOLTAGE_DEPENDENCE_ARTIFACT_ID not in self._workspace_frozen):
            self._vdep_window.refresh()     # every open recording's voltage point
        self._refresh_analysis_explorer_header()

    def _reconcile_level_fit(self) -> None:
        """Drop the sub-state levels of every event whose span no longer matches them.

        *** THE LEVEL TABLE HAS ONE OWNER, AND THIS IS IT. ***

        Measured (C-5): fit CUSUM levels, then drag one event's mask in the Inspector from
        6.000 ms to 4.000 ms. The Inspector dropped that event's levels from its own
        private ``self._levels`` — so the window drew no CUSUM line — while THIS window's
        ``_level_fit`` and ``Session.results['fit.levels']`` kept them, Export stayed
        enabled, and the export wrote level 0 starting 0.800 ms before the event now starts
        and level 1 ending 1.200 ms after it ends: 6.0 ms of sub-state inside a 4.0 ms
        event, a 50% overstatement. It then round-tripped through the project file and came
        back exportable. The only signal the user had pointed the wrong way.

        The fix is NOT another connection from the Inspector. Staleness here is DERIVABLE:
        :func:`~nanoscope.analysis.cusum.reconcile_levels` compares the span each level
        group tiles against the span its parent event now has, so a mask drag, a merge, an
        undo, a re-measure and a project restore are all caught by the same rule, including
        the paths written after this one. The Inspector applies that same function to its
        own view (one rule, one implementation) — but the export and the project file read
        THIS table, so this is where the guarantee has to hold.

        Idempotent, and a no-op in the overwhelmingly common case: when nothing has moved,
        ``reconcile_levels`` returns the very frame it was given.

        DROPPING IS NOT THE WHOLE STORY ANY MORE, AND THIS METHOD IS STILL ONLY THE DROP.
        The owner read the deletion as the feature breaking ("when I change the window
        manually, the sublevels disappear"), and :meth:`_refit_levels_for_event` is the
        answer: on a mask edit it re-measures that one event and splices its group back.
        It calls THIS method first, always, so the invariant above holds between the two
        steps and a re-fit that declines or fails leaves exactly the behaviour described
        here. Every other span-mover — a merge, an undo, a re-measure, a restore — still
        gets the drop alone, because they announce nothing to re-fit from.
        """
        from dataclasses import replace

        from ..analysis.cusum import levels_summary, reconcile_levels
        from ..analysis.events import EventSet

        fit = getattr(self, "_level_fit", None)
        if fit is None or getattr(fit, "payload", None) is None:
            return
        det = self._analysis_panel.current_detection()
        events = det.payload.df if (det is not None
                                    and isinstance(det.payload, EventSet)) else None
        # A fit that does not belong to the detection on screen is _sync_export_action's
        # question, not this one; answering it here would drop a fit merely because the
        # user selected another recording for a moment.
        if events is None or det.doc_id != fit.doc_id or int(det.channel) != int(fit.channel):
            return
        kept, dropped = reconcile_levels(fit.payload, events)
        if kept is fit.payload:
            # IDENTITY, not `not dropped`: on the inputs reconcile_levels refuses to
            # certify it returns kept=None while naming no ids, and reading that as
            # "nothing changed" is what left this window exporting a table the
            # Inspector had already discarded.
            return
        if kept is None:
            # Nothing survives: this is exactly _invalidate_level_fit's job — clear the
            # window's hands, the menu, the Inspector AND the store entry (a retired fit
            # left in the store is what project save writes into the file).
            self._invalidate_level_fit()
        else:
            self._level_fit = replace(fit, payload=kept, scalars=levels_summary(kept))
            # Straight into the store, not through add_result: this is a correction to a
            # table already there, and resultsChanged would re-enter this method.
            self._session.results.put(self._level_fit)
            self._push_level_fit(kept)
            self._act_export_levels.setEnabled(True)
        self.statusBar().showMessage(
            (f"Sub-state levels dropped for {len(dropped)} event(s) whose span changed — "
             if dropped else
             "Sub-state levels dropped: the table no longer states which spans it "
             "describes — ")
            + "re-run Analysis ▸ Fit sub-state levels (CUSUM)… to measure the new spans.",
            8000)

    def _refit_levels_for_event(self, event_id: int) -> None:
        """Re-fit ONE edited event's sub-state levels over its NEW span and splice them back.

        *** THIS IS THE SECOND HALF OF THE MASK EDIT, AND IT USED NOT TO EXIST. ***

        Measured: with a level fit standing, event 0 spanning (5000, 5600) had 14 CUSUM plot
        items in the Inspector and 32 level rows in both the Inspector and the result store.
        Drag the mask to (5000, 5300) and it had 0 plot items and 25 rows — the group was
        DELETED from the table, the export and the project file, and nothing put anything
        back. That deletion is correct and deliberate (:meth:`_reconcile_level_fit`: levels
        that no longer tile their parent describe a span that does not exist), but on its own
        it reads to the user as "changing the window destroys the sublevels", which is exactly
        the note this answers. The other half is to MEASURE THE NEW SPAN.

        THE DROP RUNS FIRST, ALWAYS, AND THE SPLICE IS SECOND. Not an ordering convenience:
        the invariant every reader of the level table rests on is "a group tiles its parent
        exactly", and reconciling first means that between the two steps the table holds
        strictly fewer groups rather than a wrong one. If anything below refuses or raises —
        a cold cache, a rejected row, a span too short, a fit that throws — the outcome is
        today's behaviour (dropped, and said so), never a stale group left standing.

        THE RE-FIT IS THE SAME FIT, not a fresh one. ``sigma`` and ``tau`` come from the
        cached inputs, which came from the original result's own ``params`` — so the new group
        is segmented at the significance scale its neighbours were, instead of at whatever a
        second whole-recording estimate would have produced from a table that has since
        changed. It is also what makes it cheap: measured on a 60 k-sample synthetic, one
        event costs 3.4 ms with ``tau`` handed in against 26.3 ms without, because without it
        the call re-walks every event's residual in the recording.

        Guards, each of which leaves the drop standing:
          * no fit, or a fit belonging to another recording/channel;
          * the event is not in the current table, or is REJECTED (the fit ran
            ``accepted_only=True``, so a rejected event has no levels by construction and
            inventing some here would put rows in the export that a re-run would not);
          * a span shorter than :data:`~nanoscope.analysis.cusum.MIN_SAMPLES_FOR_LEVELS`,
            where the fit cannot resolve a boundary and would return one "level" that is
            just a copy of the event;
          * a COLD cache — the fit was restored from a project file, so the baseline track
            behind it is not in memory. Re-deriving it here would run a 644 ms open-pore
            estimate inside a mouse-release handler. The status bar says which, and one
            re-run of the menu item makes every later edit re-fit.

        F2 — EVERY ONE OF THOSE NOW SAYS SO IN THE INSPECTOR, AND SAYS SOMETHING TRUE.
        Two things were wrong with how they ended. The Inspector — the window whose mask
        was just dragged and whose staircase just vanished — said nothing at all; and two
        of the routes (a rejected row, a re-fit that came back empty) returned in silence,
        leaving `_reconcile_level_fit`'s generic "re-run the fit" as the only sentence on
        screen. MEASURED: perform that re-run on a rejected event and it returns 0 level
        rows for it, because the fit is `accepted_only=True` — the prescribed recovery
        could not work, ever. Each route now builds its own sentence (the builders live in
        :mod:`~nanoscope.gui.event_inspector`, next to the surface that renders them) and
        posts it through :meth:`_report_level_gap` to the status bar AND the Inspector,
        naming a remedy that was checked against the app: accept the event, widen the mask,
        or re-run the fit — whichever of those actually puts the levels back from here.
        """
        import numpy as np
        from dataclasses import replace

        import pandas as pd

        from ..analysis.cusum import MIN_SAMPLES_FOR_LEVELS, fit_event_levels, levels_summary
        from ..analysis.events import EventSet
        from ..analysis.modules import detection_signal
        # The sentences themselves live with the window that has to SHOW them (F2): this
        # method posts one to the status bar and hands the same string to the Inspector, so
        # there is one wording per route rather than one per surface.
        from .event_inspector import (
            level_gap_cold,
            level_gap_empty,
            level_gap_failed,
            level_gap_rejected,
            level_gap_too_short,
        )

        fit = getattr(self, "_level_fit", None)
        if fit is None or getattr(fit, "payload", None) is None:
            return
        det = self._analysis_panel.current_detection()
        if det is None or not isinstance(det.payload, EventSet):
            return
        if det.doc_id != fit.doc_id or int(det.channel) != int(fit.channel):
            return
        events = det.payload.df
        rows = events[events["event_id"].astype("int64") == int(event_id)]
        if rows.empty:
            return

        cache = getattr(self, "_level_fit_inputs", None)
        cold = (cache is None
                or cache.get("doc_id") != fit.doc_id
                or int(cache.get("channel", -1)) != int(fit.channel))
        # THE DROP, FIRST AND UNCONDITIONALLY. Idempotent, and a no-op when nothing moved --
        # so calling it here as well as from `_sync_analysis_surfaces` costs nothing and makes
        # this method's guarantee independent of the order Qt happens to run the two slots in.
        self._reconcile_level_fit()
        # WHETHER ANY GROUP IS STILL STANDING, read AFTER the drop, because it decides which
        # remedy each sentence below may name. With a fit still in hand a mask edit re-fits
        # on release; with the whole fit retired (`_invalidate_level_fit`, the single-event
        # case) this method returns at its `fit is None` guard and a mask edit does nothing
        # at all -- measured, twice, on the five-event fixture with the other four rejected.
        standing = getattr(self, "_level_fit", None) is not None

        if cold:
            self._report_level_gap(event_id, level_gap_cold(int(event_id)))
            return

        row = rows.iloc[0]
        if "accepted" in rows.columns and not bool(row["accepted"]):
            # F2-a. This route used to return in silence, leaving `_reconcile_level_fit`'s
            # standing sentence -- "re-run Analysis ▸ Fit sub-state levels (CUSUM)… to
            # measure the new spans" -- as the only thing the user saw. MEASURED: perform
            # that re-run on a rejected event and it comes back with 0 level rows, because
            # the fit runs `accepted_only=True` and always will. The gesture that does work
            # is the accept, so that is what the sentence names now.
            self._report_level_gap(
                event_id, level_gap_rejected(int(event_id), standing_fit=standing))
            return
        start, end = int(row["start_idx"]), int(row["end_idx"])
        if end - start < MIN_SAMPLES_FOR_LEVELS:
            self._report_level_gap(
                event_id,
                level_gap_too_short(int(event_id), n_samples=end - start,
                                    min_samples=MIN_SAMPLES_FOR_LEVELS,
                                    standing_fit=standing),
                msecs=8000)
            return

        try:
            chan = self._session.calibrated(fit.doc_id, int(fit.channel))
            # REBUILT, not cached: 2.1 ms for 2 M samples against a second full-length float64
            # copy of every open recording. The BASELINE is the one that had to be kept.
            signal = np.asarray(
                detection_signal(np.asarray(chan.data), float(cache["fs"]),
                                 cache["provenance"]),
                dtype=np.float64)
            fresh = fit_event_levels(
                signal, float(cache["fs"]), rows.iloc[[0]],
                baseline=cache["baseline"],
                polarity=int(cache["polarity"]),
                sigma=cache["sigma"], tau=cache["tau"],
                provenance=cache["provenance"],
                config=cache["config"],
                # The row's own `accepted` was checked above; asking the fit to filter again
                # on a one-row frame only adds a way for it to come back empty.
                accepted_only=False,
            )
        except Exception as exc:  # noqa: BLE001 — a failed re-fit degrades to the drop
            self._report_level_gap(
                event_id, level_gap_failed(int(event_id), detail=str(exc)))
            return
        if fresh.empty:
            # The literal "a re-fit that yields zero levels" case, and it used to return in
            # silence too — leaving the reconcile's "re-run the fit" standing over it. A
            # re-run cannot help here BY CONSTRUCTION: it measures the same samples against
            # the same cached baseline with the same sigma and tau, so it resolves the same
            # nothing. The span is the only input a second attempt sees differently.
            self._report_level_gap(
                event_id, level_gap_empty(int(event_id), standing_fit=standing))
            return

        # SPLICE. `self._level_fit` may have been cleared entirely by the reconcile above (the
        # edited event was the only one carrying levels), so the surviving rows are read from
        # whatever is in hand NOW while the result's identity comes from the fit that existed
        # BEFORE — same module_id, doc, channel and both staleness stamps.
        current = getattr(self, "_level_fit", None)
        kept = getattr(current, "payload", None) if current is not None else None
        if kept is not None and not kept.empty:
            survivors = kept[kept["event_id"].astype("int64") != int(event_id)]
            merged = pd.concat([survivors, fresh], ignore_index=True)
        else:
            merged = fresh.reset_index(drop=True)
        merged = merged.sort_values(["event_id", "level_idx"], kind="stable") \
                       .reset_index(drop=True)

        result = replace(fit, payload=merged, scalars=levels_summary(merged))
        self._level_fit = result
        self._level_fit_frame = events
        self._level_fit_inputs = cache      # the reconcile may have cleared it; it is still true
        # Straight into the store, as `_reconcile_level_fit` does and for the same reason: this
        # is a correction to a table already there, and `add_result` would re-enter the
        # surface refresh that just called this.
        self._session.results.put(result)
        self._push_level_fit(merged)
        self._act_export_levels.setEnabled(True)
        self._analysis_panel.redraw_overlays()
        # THE OTHER HALF OF THE ROUND TRIP. `_push_level_fit` above already forgets the
        # recorded reason for any event whose rows came back (see `set_levels`), and the
        # Inspector refuses to render one over a live group anyway — this is the third
        # guard, at the one place that KNOWS the re-fit succeeded, because a "no levels"
        # sentence surviving over a restored staircase is a worse defect than the silence
        # F2 set out to fix.
        self._report_level_gap(
            event_id,
            f"Event {int(event_id)} re-fitted over its new span: {int(len(fresh))} "
            f"sub-state level(s) across {end - start} sample(s).",
            msecs=6000, gap=False)

    def _report_level_gap(self, event_id: int, note: str, *, msecs: int = 10000,
                          gap: bool = True) -> None:
        """Say ONCE why an event has no sub-state levels, on BOTH surfaces that can show it.

        The status bar is where every one of these sentences used to end, and it is not
        where the user is looking: they just dragged the Event Inspector's mask, and the
        Inspector's staircase is what vanished. Measured before this — drag event 0 down to
        9 samples and the window's readout, its refusal label and its plot said nothing
        whatsoever about the disappearance, while a status bar behind another window
        carried the whole explanation for eight seconds.

        ONE STRING, both surfaces. The two cannot then prescribe different remedies for the
        same event, which is the failure mode a second wording invites — and it is exactly
        the failure F2-a found between the status bar and the fit's own accepted-only rule.

        ``gap=False`` reports a SUCCESS through the same door: the status bar gets the
        sentence and the Inspector is told to forget whatever reason it was holding for
        that event.
        """
        self.statusBar().showMessage(note, msecs)
        insp = self._event_inspector
        if insp is not None:
            insp.note_level_gap(int(event_id), note if gap else "")

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() in (QtCore.QEvent.Show, QtCore.QEvent.Hide):
            if any(obj is w for w in self._surface_windows().values() if w is not None):
                self._sync_analysis_open_state()
            # A hidden analytical widget is not necessarily the project-owned copy.
            # PlotsWindow has already emitted/cached its live result and then blanked itself
            # by the time this Hide arrives, so it always needs a lazy reinstall.  A surface
            # carrying a restored/frozen snapshot needs the same protection: closing and
            # reopening it must not become an implicit Refresh against a newly-selected
            # recording.  Ordinary live PSD/Event/I-V/VDep windows do *not* reset on close,
            # however; checkpoint their latest programmatic refresh and let their existing
            # opener follow the current session when shown again (the Stage-5 contract).
            if event.type() == QtCore.QEvent.Hide and not self._workspace_installing:
                # A page hidden because another Analysis Explorer tab came forward was
                # not closed.  In particular, PlotsWindow has NOT run its closeEvent and
                # has not reset itself, so applying the old standalone-window rule here
                # would falsely mark an intact page as pending restoration.  An actual
                # page/shell close is flagged by AnalysisExplorer before the Hide event.
                tab_switch = (
                    bool(obj.property("nanoscopeAnalysisExplorerPage"))
                    and not bool(obj.property("nanoscopeAnalysisExplorerClosing"))
                )
                if tab_switch:
                    artifact_id = self._workspace_window_id(obj)
                    if artifact_id and self._workspace_has_artifact(artifact_id):
                        # Preserve the latest live result just as a standalone Hide did,
                        # but do not mark the intact page pending: unlike closeEvent, a
                        # tab switch has neither reset nor discarded its widget state.
                        self._checkpoint_workspace_surface(artifact_id)
                    return super().eventFilter(obj, event)
                artifact_id = self._workspace_window_id(obj)
                if artifact_id and self._workspace_has_artifact(artifact_id):
                    if (artifact_id == PLOTS_ARTIFACT_ID
                            or artifact_id in self._workspace_frozen):
                        self._workspace_pending.add(artifact_id)
                    else:
                        self._checkpoint_workspace_surface(artifact_id)
        return super().eventFilter(obj, event)

    # -- plots window --------------------------------------------------------------------------
    def _plots_sources(self) -> list:
        """Snapshot EVERY per-event table the Plots window may overlay.

        One :class:`~nanoscope.core.sources.PlotSource` per open ``(document,
        channel)`` that has a stored detection — segments included, since a segment
        is an ordinary document and "this segment vs that segment" is half the
        reason segments exist — plus the manual-measurement table. This used to
        offer exactly two frames (the CURRENT detection and the manual table), which
        made a two-pore or two-segment comparison structurally impossible.

        The rule itself lives in the pure :mod:`nanoscope.core.sources` helper so it
        is testable without a window; MainWindow only supplies the session lookups.
        Each source carries its OWN provenance and its own calibration fingerprint:
        a per-source dwell fit must resolve ``t_min`` from *its* detector's
        ``min_dwell_s``, and absolute pA is only comparable across recordings that
        share a real calibration.
        """
        from ..core.sources import discover_plot_sources
        from ..core.source_style import display_sources

        meas = self._analysis_panel.measure_panel.measurements_df
        sources = discover_plot_sources(
            self._session.documents(),
            self._session.results.get,
            manual_df=meas if len(meas) else None,
            # The same persistent identity map the trace view reads. Without this the
            # saved colour could survive in the project yet every publication plot would
            # still assign colour by discovery position.
            colours=self._session.colours,
            # THE TRIPLE, NOT THE SELECTION. ``_current_selection`` is (doc_id, channel),
            # and since discovery started offering one source per DETECTOR a channel no
            # longer names a table: the two-element form makes discovery guess, and it
            # guesses the channel's first detection. Measured on the -1000 mV recording
            # with Schmitt (110) and CUSUM (250) both stored and the panel showing CUSUM:
            # the source flagged ``current`` was Schmitt's. ``active_detection_key`` is
            # the same resolver Edit ▸ Undo uses, so "the table on screen" has one answer
            # in this window. It falls back to the selection when there is no detection
            # at all, which is what a discovery with no stored result wants anyway.
            current=self.active_detection_key() or self._current_selection,
            calibration_fingerprint=lambda d, c: self._session.calibration(d, c).fingerprint(),
        )
        return display_sources(sources, labels=self._session.source_labels)

    def _sync_plots_action(self) -> None:
        """The Plots action is live whenever there is SOME plottable data — any open
        recording's detection OR any saved manual measurements. Routed through the shared
        surface gate so the action and the launcher button read the same predicate once."""
        self._apply_surface_gate(SURFACE_POPULATIONS)

    def _open_distribution_summaries(self) -> None:
        """Optional new statistics are lazy and run only through explicit Generate."""
        from .distribution_summary_panel import DistributionSummaryWindow

        if self._distribution_summary_window is None:
            page = self._distribution_summary_window = DistributionSummaryWindow(self._session, self)
            page.refreshRequested.connect(self._prepare_distribution_summary)
            self._watch_surface_window(page)
            self._watch_workspace_surface(DISTRIBUTION_SUMMARIES_ARTIFACT_ID, page)
        page = self._distribution_summary_window
        if DISTRIBUTION_SUMMARIES_ARTIFACT_ID not in self._workspace_frozen:
            self._offer_distribution_sources()
        self._restore_workspace_surface(DISTRIBUTION_SUMMARIES_ARTIFACT_ID, page)
        self._show_analysis_page(
            PAGE_DISTRIBUTION_SUMMARIES, page, refresh=self._generate_distribution_summary)

    def _generate_distribution_summary(self) -> None:
        page = self._distribution_summary_window
        if page is None:
            return
        self._prepare_distribution_summary()
        try:
            page.analyze()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(page, "Generate distribution summary", str(exc))

    def _prepare_distribution_summary(self) -> None:
        """Generate intent leaves frozen mode before binding today's event tables."""
        self._workspace_pending.discard(DISTRIBUTION_SUMMARIES_ARTIFACT_ID)
        self._workspace_frozen.discard(DISTRIBUTION_SUMMARIES_ARTIFACT_ID)
        page = self._distribution_summary_window
        if page is not None and page.has_data():
            # A failed explicit generation must not relabel an old, frozen result
            # CURRENT just because today's offered bindings happened to be unchanged.
            page._stale = True
            page._update_status()
        self._offer_distribution_sources()
        self._refresh_analysis_explorer_header()

    def _offer_distribution_sources(self) -> None:
        """Passive rebinding is not a project edit and must not overwrite pending state."""
        if self._distribution_summary_window is None:
            return
        installing = self._workspace_installing
        self._workspace_installing = True
        try:
            self._distribution_summary_window.set_sources(self._plots_sources())
        finally:
            self._workspace_installing = installing

    def _open_comparison_figures(self) -> None:
        """Open the optional publication comparisons without regenerating a result."""
        from .comparison_figures_panel import ComparisonFiguresWindow

        if self._comparison_figures_window is None:
            page = self._comparison_figures_window = ComparisonFiguresWindow(self._session, self)
            page.refreshRequested.connect(self._prepare_comparison_figures)
            self._watch_surface_window(page)
            self._watch_workspace_surface(COMPARISON_FIGURES_ARTIFACT_ID, page)
        page = self._comparison_figures_window
        if COMPARISON_FIGURES_ARTIFACT_ID not in self._workspace_frozen:
            self._offer_comparison_sources()
        self._restore_workspace_surface(COMPARISON_FIGURES_ARTIFACT_ID, page)
        self._show_analysis_page(
            PAGE_COMPARISON_FIGURES, page, refresh=self._generate_comparison_figures)

    def _generate_comparison_figures(self) -> None:
        page = self._comparison_figures_window
        if page is None:
            return
        self._prepare_comparison_figures()
        try:
            page.analyze()
        except (TypeError, ValueError, OverflowError, ImportError, RuntimeError) as exc:
            QtWidgets.QMessageBox.warning(page, "Generate comparison figures", str(exc))

    def _prepare_comparison_figures(self) -> None:
        """Only explicit generation replaces a restored figure with today's inputs."""
        self._workspace_pending.discard(COMPARISON_FIGURES_ARTIFACT_ID)
        self._workspace_frozen.discard(COMPARISON_FIGURES_ARTIFACT_ID)
        page = self._comparison_figures_window
        if page is not None and page.has_data():
            # Failed generation must leave the previous picture visibly stale.
            page._stale = True
            page._update_status()
        self._offer_comparison_sources()
        self._refresh_analysis_explorer_header()

    def _offer_comparison_sources(self) -> None:
        """Offer current bindings silently, preserving pending saved figure state."""
        if self._comparison_figures_window is None:
            return
        installing = self._workspace_installing
        self._workspace_installing = True
        try:
            self._comparison_figures_window.set_sources(self._plots_sources(), analyze=False)
        finally:
            self._workspace_installing = installing

    def _open_plots(self) -> None:
        """Open the Distributions page of the single Analysis Explorer."""
        from .plots_panel import PlotsWindow

        if _on_screen(self._plots_window):
            self._show_analysis_page(
                PAGE_DISTRIBUTIONS, self._plots_window,
                refresh=lambda: self._replace_workspace_surface(
                    PLOTS_ARTIFACT_ID, self._refresh_plots))
            return
        if self._plots_window is None:
            self._plots_window = PlotsWindow(self._session, self)
            self._plots_window.refreshRequested.connect(
                lambda: self._replace_workspace_surface(
                    PLOTS_ARTIFACT_ID, self._refresh_plots))
            self._plots_window.eventRowsSelected.connect(
                self._on_plot_event_rows_selected)
            self._watch_surface_window(self._plots_window)
            self._watch_workspace_surface(PLOTS_ARTIFACT_ID, self._plots_window)
        self._refresh_plots()
        self._restore_workspace_surface(PLOTS_ARTIFACT_ID, self._plots_window)
        self._show_analysis_page(
            PAGE_DISTRIBUTIONS, self._plots_window,
            refresh=lambda: self._replace_workspace_surface(
                PLOTS_ARTIFACT_ID, self._refresh_plots))

    def _refresh_plots(self) -> None:
        """(Re-)push the current snapshot into the open PlotsWindow — the target of
        both the initial open and the window's own Refresh button. Each source's
        detection provenance travels WITH its frame so a per-source dwell fit uses
        that detector's own min_dwell_s, exactly as the Fits panel does.

        The ACTIVE DETECTION is pushed first and separately, because the window has to
        answer "which of these is the table the user is looking at" before it decides
        which row to tick — and a source list can carry two detections of one channel.
        ``set_active_detection`` before ``set_data``, in that order, so the tick is
        decided from this answer and never from the stale one (see
        ``PlotsWindow._active_source_key``, which otherwise has to reach back in here)."""
        if self._plots_window is not None:
            self._plots_window.set_active_detection(self.active_detection_key())
            self._plots_window.set_data(self._plots_sources())

    @QtCore.Slot(str, object)
    def _on_plot_event_rows_selected(self, source_key: str, row_ids: object) -> None:
        """Link clicked Scatter/PCA points back to their event table and trace span.

        ``PlotsWindow`` emits detector ``event_id`` values (not dataframe labels or plot
        positions). Cross the window boundary with that stable identity already used by
        the table, Gallery and Inspector. No curation is changed by this navigation.
        """
        from ..analysis.events import EventSet

        plots = self._plots_window
        if plots is None:
            return
        source = next((item for item in plots.sources()
                       if item.key == str(source_key)), None)
        if source is None:
            self.statusBar().showMessage(
                "That plotted source is no longer available. Refresh the plot and try again.",
                8000)
            return
        if (source.kind != "detection" or source.doc_id is None
                or source.channel is None or not source.module_id):
            self.statusBar().showMessage(
                "Manual-measurement points are not rows in a detector event table.", 7000)
            return
        if not self._session.has_document(source.doc_id):
            self.statusBar().showMessage(
                "That point belongs to a recording that is no longer open. Refresh the plot.",
                8000)
            return
        result = self._session.results.get(
            source.doc_id, int(source.channel), str(source.module_id))
        if result is None or not isinstance(result.payload, EventSet):
            self.statusBar().showMessage(
                "That detector result is no longer loaded. Refresh the plot before selecting it.",
                8000)
            return
        # A detector rerun replaces the EventSet but normally starts event_id at zero
        # again.  Matching only those integers could therefore turn a click on a frozen
        # saved plot into navigation to a different, newly detected event. PlotSource is
        # fed the live EventSet frame by reference, so object identity is the exact and
        # inexpensive generation guard; ordinary in-place curation keeps that identity.
        if source.df is not result.payload.df:
            self.statusBar().showMessage(
                "This plot belongs to an earlier detector run. Refresh it before selecting "
                "an event.", 9000)
            return

        try:
            wanted = tuple(row_ids or ())
        except TypeError:
            wanted = ()
        if not wanted:
            # Additive clicking can toggle the final point off.  Clear only when the
            # table is showing this same source; never erase another detector's selection.
            active = self._analysis_panel.current_detection()
            if (active is result
                    or (active is not None and active.doc_id == result.doc_id
                        and int(active.channel) == int(result.channel)
                        and active.module_id == result.module_id)):
                self._analysis_panel._table.view.clearSelection()
            return

        frame = source.df
        if "event_id" not in frame.columns:
            self.statusBar().showMessage(
                "These plotted rows have no event identifiers to link back to the trace.",
                8000)
            return
        plotted_ids = set(int(value) for value in frame["event_id"].tolist())
        event_ids: list[int] = []
        selected_ids: set[int] = set()
        for value in wanted:
            try:
                event_id = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if event_id in plotted_ids and event_id not in selected_ids:
                event_ids.append(event_id)
                selected_ids.add(event_id)
        if not event_ids:
            self.statusBar().showMessage(
                "Those plotted events are no longer present. Refresh the plot and try again.",
                8000)
            return
        if "event_id" not in result.payload.df.columns:
            self.statusBar().showMessage(
                "The live detector table has no event identifiers to receive this selection.",
                8000)
            return
        live_ids = set(int(value) for value in
                       result.payload.df["event_id"].tolist())
        event_ids = list(dict.fromkeys(value for value in event_ids if value in live_ids))
        if not event_ids:
            self.statusBar().showMessage(
                "The selected events were replaced after this plot was drawn. Refresh the plot.",
                8000)
            return

        target = (source.doc_id, int(source.channel))
        if self._current_selection != target:
            if not self._file_dock.focus_channel(*target):
                self._on_selection(*target)

        # Selecting the source recording can legitimately choose another stored detector
        # by default.  Move the existing result selector onto the source that was plotted;
        # this is its ordinary, non-mutating view-switch path.
        selector = self._analysis_panel._result_selector
        result_index = selector.findData(str(source.module_id))
        if result_index < 0:
            self.statusBar().showMessage(
                "The plotted detector is no longer available in the event table. Refresh the plot.",
                8000)
            return
        if selector.currentIndex() != result_index:
            selector.setCurrentIndex(result_index)
        else:
            self._analysis_panel._show_result_choice(result_index)

        # Focus the last clicked/extended event on the trace, then restore the complete
        # plot selection in the table.  Calling these in the opposite order would let the
        # single-event focus collapse an intentional multi-point selection.
        focus_id = int(event_ids[-1])
        self._analysis_panel.select_event(focus_id)
        self._analysis_panel._table._restore_selection(event_ids, focus_id)
        noun = "event" if len(event_ids) == 1 else "events"
        self.statusBar().showMessage(
            f"Selected {len(event_ids):,} {noun} from the plot; the event table and trace "
            f"are focused on event {focus_id}.", 7000)

    # -- event-plots window (density / ensemble) -------------------------------------------------
    def _event_plots_ready(self) -> bool:
        """True when the active detection has at least one ACCEPTED event — the gate for
        the density / ensemble window (there is nothing to bin or overlay otherwise).

        Reads the answer off :meth:`_gate_event_plots` rather than re-deriving it, so the
        boolean and the reason the user is shown can never come apart."""
        return self._gate_event_plots()[0]

    def _sync_event_plots_action(self) -> None:
        self._apply_surface_gate(SURFACE_EVENT_PLOTS)

    def _open_event_plots(self) -> None:
        """Open the Event waveforms page of the single Analysis Explorer."""
        from .event_plot_panel import EventPlotWindow

        if _on_screen(self._event_plot_window):
            self._show_analysis_page(
                PAGE_EVENT_WAVEFORMS, self._event_plot_window,
                refresh=lambda: self._replace_workspace_surface(
                    EVENT_PLOTS_ARTIFACT_ID, self._refresh_event_plots))
            return
        pending = EVENT_PLOTS_ARTIFACT_ID in self._workspace_pending
        if self._event_plot_window is None:
            self._event_plot_window = EventPlotWindow(self._session, self)
            self._event_plot_window.refreshRequested.connect(
                lambda: self._replace_workspace_surface(
                    EVENT_PLOTS_ARTIFACT_ID, self._refresh_event_plots))
            self._watch_surface_window(self._event_plot_window)
            self._watch_workspace_surface(
                EVENT_PLOTS_ARTIFACT_ID, self._event_plot_window)
        if pending:
            self._bind_event_plots_for_restore()
        else:
            self._refresh_event_plots()
        self._restore_workspace_surface(
            EVENT_PLOTS_ARTIFACT_ID, self._event_plot_window)
        if not pending and self._event_plot_window.has_data():
            # Opening this surface performs the deterministic density/ensemble analysis.
            # Its completed numerical view is now project work even before another knob
            # is touched, so checkpoint it and advance recovery like every other result.
            self._on_workspace_changed(EVENT_PLOTS_ARTIFACT_ID)
        self._show_analysis_page(
            PAGE_EVENT_WAVEFORMS, self._event_plot_window,
            refresh=lambda: self._replace_workspace_surface(
                EVENT_PLOTS_ARTIFACT_ID, self._refresh_event_plots))

    def _refresh_event_plots(self) -> None:
        """(Re-)push the active detection (doc/channel + EventSet) into the open EventPlotWindow,
        or clear it when the selection has no detection. A no-op until the window is built."""
        self._feed_event_plots(defer_analysis=False)

    def _bind_event_plots_for_restore(self) -> None:
        """Bind live source identity before restoring frozen arrays, running no kernel."""
        self._feed_event_plots(defer_analysis=True)

    def _feed_event_plots(self, *, defer_analysis: bool) -> None:
        """Install the current event source with either compute or bind-only semantics."""
        if self._event_plot_window is None:
            return
        from ..analysis.events import EventSet

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            if defer_analysis:
                self._event_plot_window.set_source(
                    None, None, None, defer_analysis=True)
            else:
                self._event_plot_window.set_source(None, None, None)
            return
        if defer_analysis:
            self._event_plot_window.set_source(
                res.doc_id, res.channel, res.payload, defer_analysis=True)
        else:
            self._event_plot_window.set_source(res.doc_id, res.channel, res.payload)

    # -- I-V / conductance window ----------------------------------------------------------------
    def _has_iv_data(self) -> bool:
        """True when >=1 open recording could contribute an I-V point — a live current channel plus
        a known voltage (a live voltage channel or a nominal_voltage_mV). Mirrors the conditions in
        analysis.iv.iv_point, but reads META ONLY so gating the menu never computes a median."""
        for doc in self._session.documents():
            meta = doc.meta
            # effective_kind on both tests: the Calibration-panel override moves a
            # channel's kind for the panes and the overlay grouping, so the menu gates
            # must follow it — a corrected recording that draws as current must not
            # stay greyed out here. Still meta+calibration only; nothing is computed.
            has_current = any(effective_kind(self._session, doc.doc_id, c.index)
                              is ChannelKind.CURRENT and not c.is_dead
                              for c in meta.channels)
            has_voltage = (any(effective_kind(self._session, doc.doc_id, c.index)
                               is ChannelKind.VOLTAGE and not c.is_dead
                               for c in meta.channels)
                           or meta.nominal_voltage_mV is not None)
            if has_current and has_voltage:
                return True
        return False

    def _sync_iv_action(self) -> None:
        self._apply_surface_gate(SURFACE_IV)

    def _open_iv(self) -> None:
        """Open the I-V page of the single Analysis Explorer."""
        from .iv_panel import IVWindow

        if _on_screen(self._iv_window):
            self._show_analysis_page(
                PAGE_IV, self._iv_window,
                refresh=lambda: self._replace_workspace_surface(
                    IV_ARTIFACT_ID, self._iv_window.refresh))
            return
        pending = IV_ARTIFACT_ID in self._workspace_pending
        if self._iv_window is None:
            self._iv_window = (
                IVWindow(self._session, self, defer_initial_refresh=True)
                if pending else IVWindow(self._session, self))
            self._watch_surface_window(self._iv_window)
            self._watch_workspace_surface(IV_ARTIFACT_ID, self._iv_window)
            created = True
        else:
            created = False
        if pending:
            self._restore_workspace_surface(IV_ARTIFACT_ID, self._iv_window)
        elif not created:
            self._iv_window.refresh()
        if not pending and self._iv_window.points():
            self._on_workspace_changed(IV_ARTIFACT_ID)
        self._show_analysis_page(
            PAGE_IV, self._iv_window,
            refresh=lambda: self._replace_workspace_surface(
                IV_ARTIFACT_ID, self._iv_window.refresh))

    # -- voltage-dependence window ---------------------------------------------------------------
    def _vdep_ready_count(self) -> int:
        """How many open recordings could place on the voltage-dependence plot: a live
        current channel + a known voltage (like I-V) AND a stored Schmitt detection.
        Reads META + the ResultStore only, so gating never computes a metric."""
        from ..analysis.events import EventSet

        n = 0
        for doc in self._session.documents():
            meta = doc.meta
            # effective_kind on both tests: the Calibration-panel override moves a
            # channel's kind for the panes and the overlay grouping, so the menu gates
            # must follow it — a corrected recording that draws as current must not
            # stay greyed out here. Still meta+calibration only; nothing is computed.
            has_current = any(effective_kind(self._session, doc.doc_id, c.index)
                              is ChannelKind.CURRENT and not c.is_dead
                              for c in meta.channels)
            has_voltage = (any(effective_kind(self._session, doc.doc_id, c.index)
                               is ChannelKind.VOLTAGE and not c.is_dead
                               for c in meta.channels)
                           or meta.nominal_voltage_mV is not None)
            if not (has_current and has_voltage):
                continue
            has_detection = any(
                r.module_id == "detect.schmitt" and isinstance(r.payload, EventSet)
                for r in self._session.results.for_doc(doc.doc_id))
            if has_detection:
                n += 1
        return n

    def _sync_vdep_action(self) -> None:
        """Voltage dependence is live once >=2 recordings have BOTH a voltage AND a
        detection (a single point is not a trend)."""
        self._apply_surface_gate(SURFACE_VDEP)

    def _open_vdep(self) -> None:
        """Open the Voltage series page of the single Analysis Explorer."""
        from .voltage_dep_panel import VDepWindow

        if _on_screen(self._vdep_window):
            self._show_analysis_page(
                PAGE_VOLTAGE_SERIES, self._vdep_window,
                refresh=lambda: self._replace_workspace_surface(
                    VOLTAGE_DEPENDENCE_ARTIFACT_ID, self._vdep_window.refresh))
            return
        pending = VOLTAGE_DEPENDENCE_ARTIFACT_ID in self._workspace_pending
        if self._vdep_window is None:
            self._vdep_window = (
                VDepWindow(self._session, self, defer_initial_refresh=True)
                if pending else VDepWindow(self._session, self))
            self._watch_surface_window(self._vdep_window)
            self._watch_workspace_surface(
                VOLTAGE_DEPENDENCE_ARTIFACT_ID, self._vdep_window)
            created = True
        else:
            created = False
        if pending:
            self._restore_workspace_surface(
                VOLTAGE_DEPENDENCE_ARTIFACT_ID, self._vdep_window)
        elif not created:
            self._vdep_window.refresh()
        if not pending and self._vdep_window.points():
            self._on_workspace_changed(VOLTAGE_DEPENDENCE_ARTIFACT_ID)
        self._show_analysis_page(
            PAGE_VOLTAGE_SERIES, self._vdep_window,
            refresh=lambda: self._replace_workspace_surface(
                VOLTAGE_DEPENDENCE_ARTIFACT_ID, self._vdep_window.refresh))

    # -- power-spectrum / noise window -----------------------------------------------------------
    def _has_psd_recording(self) -> bool:
        """True when >=1 open recording could yield a noise spectrum — a live current
        channel with a known sample rate. Reads META only, so gating never computes a
        PSD. PSD is raw-signal QC, so this never looks at detections."""
        for doc in self._session.documents():
            meta = doc.meta
            if meta.sample_rate_hz is None:
                continue
            if any(effective_kind(self._session, doc.doc_id, c.index)
                   is ChannelKind.CURRENT and not c.is_dead for c in meta.channels):
                return True
        return False

    def _sync_psd_action(self) -> None:
        self._apply_surface_gate(SURFACE_PSD)

    def _open_psd(self) -> None:
        """Open the Noise page of the single Analysis Explorer."""
        from .psd_panel import PSDWindow

        if _on_screen(self._psd_window):
            self._show_analysis_page(
                PAGE_NOISE, self._psd_window,
                refresh=lambda: self._replace_workspace_surface(
                    PSD_ARTIFACT_ID, self._point_psd_at_selection))
            return
        pending = PSD_ARTIFACT_ID in self._workspace_pending
        if self._psd_window is None:
            self._psd_window = (
                PSDWindow(self._session, self, defer_initial_refresh=True)
                if pending else PSDWindow(self._session, self))
            self._watch_surface_window(self._psd_window)
            self._watch_workspace_surface(PSD_ARTIFACT_ID, self._psd_window)
        if pending:
            self._restore_workspace_surface(PSD_ARTIFACT_ID, self._psd_window)
        else:
            self._point_psd_at_selection()
            if self._psd_window.has_data():
                self._on_workspace_changed(PSD_ARTIFACT_ID)
        self._show_analysis_page(
            PAGE_NOISE, self._psd_window,
            refresh=lambda: self._replace_workspace_surface(
                PSD_ARTIFACT_ID, self._point_psd_at_selection))

    def _open_psd_for_selection(self) -> None:
        """Analysis ▸ Power spectrum of the measured region — the CLOCK-SAFE transfer.

        The trace and the PSD panel are on two different time origins, both spelled
        "s" (see :mod:`nanoscope.core.clock` and the action's own tooltip). This carries
        the user's on-trace selection across as SAMPLE INDICES and does the one
        conversion — ``index / fs``, into seconds from the start of that recording,
        which is the clock ``PSDWindow.set_segment`` takes — here, in code.

        Both halves have to be right, and the second is the easy one to forget: the
        indices come from :meth:`TraceView.measure_bounds`, and the recording they are
        indices INTO comes from :meth:`TraceView.measure_target`. The Measure region
        does not have to be on the app's currently selected recording, and dividing one
        recording's indices by another's sample rate is exactly the class of error this
        action exists to remove.
        """
        from .psd_panel import PSDWindow

        target = self._trace_view.measure_target()
        bounds = self._trace_view.measure_bounds()
        if target is None or bounds is None:
            self.statusBar().showMessage(
                "Turn on the Measure tool and drag a region on the trace first — this "
                "opens the spectrum of exactly that region.", 8000)
            return
        doc_id, channel = target
        if not self._session.has_document(doc_id):
            self.statusBar().showMessage(
                "That selection is on a recording that is no longer open.", 8000)
            return
        meta = self._session.document(doc_id).meta
        fs = meta.sample_rate_hz
        if not fs:
            self.statusBar().showMessage(
                "That recording declares no sample rate, so a window in seconds has no "
                "meaning for it.", 8000)
            return
        # ...and it must be a recording the PSD window can actually point AT. ``set_target``
        # ignores a recording with no live current channel and leaves the combo where it
        # was — so without this the segment would be applied to a DIFFERENT recording, and
        # the status line below would confidently describe the wrong one. The whole action
        # exists to stop a window being analysed on the wrong footing.
        if not any(effective_kind(self._session, doc_id, c.index)
                   is ChannelKind.CURRENT and not c.is_dead for c in meta.channels):
            self.statusBar().showMessage(
                "That selection is on a recording with no live current channel — there is "
                "no noise spectrum to take.", 8000)
            return
        i0, i1 = int(bounds[0]), int(bounds[1])
        start_s, end_s = i0 / float(fs), i1 / float(fs)

        # This menu item is an explicit request for a NEW spectrum, unlike merely opening
        # a saved PSD surface.  It therefore retires the frozen snapshot before any kernel
        # is allowed to run and checkpoints the replacement below.
        self._workspace_pending.discard(PSD_ARTIFACT_ID)
        self._workspace_frozen.discard(PSD_ARTIFACT_ID)
        if self._psd_window is None:
            self._psd_window = PSDWindow(self._session, self)
            self._watch_surface_window(self._psd_window)
            self._watch_workspace_surface(PSD_ARTIFACT_ID, self._psd_window)
        self._psd_window.set_target(doc_id, channel)
        self._psd_window.set_segment(start_s, end_s)
        self._on_workspace_changed(PSD_ARTIFACT_ID)
        self._show_analysis_page(
            PAGE_NOISE, self._psd_window,
            refresh=lambda: self._replace_workspace_surface(
                PSD_ARTIFACT_ID, self._point_psd_at_selection))
        # The status line states BOTH clocks, so the number the user just saw on the
        # trace and the number now in the Start box are visible together — the one
        # place they were never shown side by side is where the 5.7x error lived.
        offset = self._trace_view.clock_offset_for(doc_id, channel)
        if clock.clocks_agree(offset) or offset is None:
            where = f"{start_s:.4f}–{end_s:.4f} s"
        else:
            where = (f"{start_s:.4f}–{end_s:.4f} s into the recording "
                     f"(= {start_s + offset:.4f}–{end_s + offset:.4f} s on the trace)")
        self.statusBar().showMessage(
            f"Noise spectrum over {where}, {i1 - i0} samples.", 8000)

    def _point_psd_at_selection(self) -> None:
        """Point the open PSD window at the current selection's recording, else just
        refresh its recording list. A no-op until the window is built."""
        if self._psd_window is None:
            return
        if self._current_selection is not None \
                and self._session.has_document(self._current_selection[0]):
            self._psd_window.set_target(self._current_selection[0], self._current_selection[1])
        else:
            self._psd_window.refresh()

    # -- event inspector -----------------------------------------------------------------------
    def _open_event_inspector(self) -> None:
        """Analysis ▸ Event inspector… — open (or re-focus) the single EventInspectorWindow, fed
        the current detection's SHARED events df. MainWindow stays the only panel-importer: it
        builds the window, wires the two-way selection + acceptance sync to the AnalysisPanel
        (which owns the table + overlays), and (re-)feeds its contents. The df is shared, so an
        accept/reject in the inspector writes the very frame the table edits."""
        from .event_inspector import EventInspectorWindow

        if self._event_inspector is None:
            insp = EventInspectorWindow(self._session, self)
            # Selection sync (both ways) — the panel owns the trace zoom, marker + table-row highlight.
            insp.eventSelected.connect(self._analysis_panel.select_event)
            self._analysis_panel.eventSelected.connect(insp.show_event)
            # "Gallery…" — the direct road back to the contact sheet (owner note #5).
            insp.galleryRequested.connect(self._open_event_gallery)
            # "Show in trace" / T — the same slot the gallery's double-click uses, so both
            # curation windows have ONE way of saying "put the trace on this event", and
            # both get the child-lowering that makes it visible on macOS.
            insp.traceRequested.connect(self._goto_event_in_trace)
            # Acceptance sync (both ways) on the SHARED events df.
            insp.acceptanceChanged.connect(self._analysis_panel.set_event_accepted)
            self._analysis_panel.eventAcceptanceChanged.connect(insp.reflect_acceptance)
            # Mask edits + batch curation write the shared df; the panel (owner of the table +
            # overlays) refreshes the affected row / whole table, the span overlays and the summary.
            insp.eventEdited.connect(self._analysis_panel.refresh_after_edit)
            insp.curationBatchChanged.connect(self._analysis_panel.resync_curation)
            # Both of these write the SHARED frame in place, so neither reaches
            # ``add_result``: they are announced to the undo recorder here, and through
            # it to the dirty flag. ``eventEdited`` used to reach the dirty flag through
            # nothing at all.
            insp.eventEdited.connect(lambda *_: self._checkpoint_current())
            # ...and the OTHER HALF of a mask edit. `_reconcile_level_fit` (reached through
            # `_sync_analysis_surfaces`) deletes the edited event's sub-state levels, because
            # they describe a span it no longer has; this measures the new span and splices
            # the group back, so "changing the window destroys the sublevels" stops being
            # true. It re-runs the drop itself before splicing, so it does not depend on
            # which of these two slots Qt happens to call first.
            insp.eventEdited.connect(self._refit_levels_for_event)
            # Optional amplitude curation does NOT change boundaries or trigger a
            # fitted-state recomputation. Repaint the new columns and checkpoint once.
            insp.amplitudeEdited.connect(lambda *_: self._analysis_panel.resync_curation())
            insp.amplitudeEdited.connect(lambda *_: self._checkpoint_current())
            insp.curationBatchChanged.connect(self._checkpoint_batch_curation)
            # Batch curation (Accept all / Reject all) changes the accepted count without a
            # per-event signal, so the chip + strip must also refresh off it — and it is
            # unsaved work like any single-event curation.
            insp.curationBatchChanged.connect(self._refresh_workflow_status)
            # ...and so must the density / ensemble gate, whose condition IS the accepted count.
            insp.curationBatchChanged.connect(self._sync_analysis_surfaces)
            insp.curationBatchChanged.connect(self._mark_dirty)
            # "Merge with next" stitches the browsed event to its forward run; the panel merges
            # on the shared EventSet (existing refresh) and MainWindow advances the browser.
            insp.mergeWithNextRequested.connect(self._on_merge_with_next)
            insp.labelsChanged.connect(self._analysis_panel.resync_curation)
            insp.labelsChanged.connect(self._mark_dirty)
            insp.labelsChanged.connect(self._checkpoint_labels)
            # Project saving belongs to MainWindow even when the active editing surface is
            # this top-level child. The Inspector owns only the shortcut/button and emits
            # intent; this window keeps the one project path, dirty bit, conflict guard and
            # serializer. Passing the initiator through the wrapper also makes native save
            # sheets and error dialogs belong to the window where the gesture happened.
            insp.saveProjectRequested.connect(
                lambda w=insp: self._save_project_from(w, save_as=False))
            insp.saveProjectAsRequested.connect(
                lambda w=insp: self._save_project_from(w, save_as=True))
            self.projectSaveStateChanged.connect(insp.set_project_save_state)
            insp.set_project_save_state(
                self._project_path or "", self._visible_project_dirty())
            # A refusal WRITES nothing, so no data signal will repaint the other two
            # surfaces' notes — the mirror of the table's acceptanceRefused hop.
            insp.acceptanceRefused.connect(self._on_inspector_acceptance_refused)
            # ...and every one of the three, from the ONE table, to the ONE refresher. A
            # curation done in this window used to leave the Populations and Event Plots
            # windows drawing the pre-curation fit (see SURFACE_MUTATION_SIGNALS).
            self._wire_surface_refresh("event_inspector", insp)
            self._event_inspector = insp
        self._event_inspector.set_vocabulary(self._label_vocab)
        self._refresh_event_inspector()
        self._event_inspector.show()
        self._event_inspector.raise_()
        self._event_inspector.activateWindow()

    def _edit_labels(self) -> None:
        """Analysis ▸ Event labels… — edit the palette.

        Deliberately a plain table of ``key``, ``value``, ``hotkey``: the vocabulary is the
        USER'S, and any fixed list this app shipped would be wrong for the next bench. A
        molecule, an event class and a hand-read level count are all just keys.
        """
        from ..analysis.labels import (
            LabelVocabulary,
            validate_label,
            validate_label_hotkey,
        )

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Event labels")
        dlg.resize(560, 380)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(
            "Each row is one label. <b>Key</b> groups them (a row may carry one value per "
            "key); <b>value</b> is what gets written. Leave <b>hotkey</b> blank to use "
            "1–9 by position. An explicit hotkey is one character; A, R, T and Space "
            "are reserved for curation controls.<br>Examples: "
            "<code>molecule / apo-ferritin</code>, "
            "<code>type / collision</code>, <code>levels / 3</code>."))
        table = QtWidgets.QTableWidget(0, 4)
        table.setObjectName("labelVocabularyTable")
        table.setHorizontalHeaderLabels(["Key", "Value", "Hotkey", "Note"])
        table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(table, 1)

        def _row(key="", value="", hotkey="", note=""):
            r = table.rowCount()
            table.insertRow(r)
            for c, v in enumerate((key, value, hotkey, note)):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(v))

        for e in self._label_vocab.entries:
            _row(e.key, e.value, e.hotkey, e.note)
        if not self._label_vocab.entries:
            # The explanation above supplies examples; starter rows must be genuinely
            # blank. Prefilling key-only rows made the untouched examples invalid, so a
            # newcomer who completed just one row closed onto a warning and lost the edit.
            for _ in range(3):
                _row()

        hotkey_error = QtWidgets.QLabel("")
        hotkey_error.setObjectName("labelHotkeyError")
        hotkey_error.setWordWrap(True)
        hotkey_error.setStyleSheet("color:#ff8a80;")
        hotkey_error.setVisible(False)
        lay.addWidget(hotkey_error)

        def _validation_problems() -> dict[tuple[int, int], str]:
            """Invalid cell -> explanation; completely blank starter rows are ignored."""
            problems: dict[tuple[int, int], str] = {}
            claimed: dict[str, list[int]] = {}
            for row in range(table.rowCount()):
                raw_cells = [
                    table.item(row, col).text() if table.item(row, col) is not None else ""
                    for col in range(table.columnCount())
                ]
                if not any(value.strip() for value in raw_cells):
                    continue
                try:
                    validate_label(raw_cells[0], raw_cells[1])
                except ValueError as exc:
                    message = str(exc)
                    col = 0 if "key" in message else 1
                    problems[(row, col)] = message
                try:
                    hotkey = validate_label_hotkey(raw_cells[2])
                except ValueError as exc:
                    problems[(row, 2)] = str(exc)
                    continue
                if hotkey:
                    claimed.setdefault(hotkey, []).append(row)
            for hotkey, rows in claimed.items():
                if len(rows) > 1:
                    where = ", ".join(str(row + 1) for row in rows)
                    for row in rows:
                        problems[(row, 2)] = (
                            f"hotkey {hotkey!r} is duplicated in rows {where}"
                        )
            return problems

        btns = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add row")
        rm = QtWidgets.QPushButton("Remove row")
        btns.addWidget(add)
        btns.addWidget(rm)
        btns.addStretch(1)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok
                                         | QtWidgets.QDialogButtonBox.Cancel)
        box.rejected.connect(dlg.reject)
        btns.addWidget(box)
        lay.addLayout(btns)

        def _refresh_label_validation(*_args) -> bool:
            problems = _validation_problems()
            table.blockSignals(True)
            try:
                for row in range(table.rowCount()):
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        if item is None:
                            item = QtWidgets.QTableWidgetItem("")
                            table.setItem(row, col, item)
                        message = problems.get((row, col), "")
                        item.setBackground(
                            QtGui.QBrush(QtGui.QColor("#5a2626"))
                            if message else QtGui.QBrush()
                        )
                        item.setToolTip(
                            message
                            or ("One printable character, or blank for an automatic 1–9 key."
                                if col == 2 else "")
                        )
            finally:
                table.blockSignals(False)
            ok = box.button(QtWidgets.QDialogButtonBox.Ok)
            ok.setEnabled(not problems)
            hotkey_error.setVisible(bool(problems))
            hotkey_error.setText(
                "Fix the highlighted label cells before saving: "
                + "; ".join(
                    f"row {row + 1}: {message}"
                    for (row, _col), message in sorted(problems.items())
                )
                if problems else ""
            )
            return not problems

        def _accept_if_valid() -> None:
            # Recheck at the acceptance boundary as well as on cellChanged. The dialog
            # cannot close first and discover an invalid key/value afterwards, which was
            # the edit-discarding first-use trap this gate replaces.
            if _refresh_label_validation():
                dlg.accept()

        def _add_row() -> None:
            _row()
            _refresh_label_validation()

        def _remove_row() -> None:
            if table.currentRow() >= 0:
                table.removeRow(table.currentRow())
                _refresh_label_validation()

        add.clicked.connect(_add_row)
        rm.clicked.connect(_remove_row)
        box.accepted.connect(_accept_if_valid)
        table.cellChanged.connect(_refresh_label_validation)
        _refresh_label_validation()

        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        vocab, bad = LabelVocabulary(), []
        claimed: dict[str, int] = {}
        for r in range(table.rowCount()):
            def cell(c, *, strip=True):
                it = table.item(r, c)
                text = it.text() if it else ""
                return text.strip() if strip else text
            key, value = cell(0), cell(1)
            if not key and not value:
                continue                      # a blank row is not an error, it is a blank row
            try:
                hotkey = validate_label_hotkey(cell(2, strip=False))
                if hotkey and hotkey in claimed:
                    raise ValueError(
                        f"hotkey {hotkey!r} duplicates row {claimed[hotkey] + 1}"
                    )
                if hotkey:
                    claimed[hotkey] = r
                vocab.add(key, value, hotkey, cell(3))
            except ValueError as exc:
                bad.append(f"row {r + 1}: {exc}")
        if bad:
            QtWidgets.QMessageBox.warning(
                self, "Event labels",
                "The palette was not changed:\n\n" + "\n".join(bad)
                + "\n\nKeys and values cannot be empty or contain ';', '=' or '|'. "
                "Hotkeys must be unique single characters; A, R, T and Space are reserved.")
            return
        changed = vocab.to_json() != self._label_vocab.to_json()
        self._label_vocab = vocab
        self._push_vocabulary()
        # The palette is saved WITH the project, so editing it is unsaved work like any
        # other. Without this, defining a vocabulary and quitting threw it away silently —
        # no prompt, because nothing had marked the session dirty.
        if changed:
            self._mark_dirty()
        self.statusBar().showMessage(
            f"{len(vocab.entries)} label(s) defined." if vocab.entries
            else "No labels defined.", 6000)

    def _push_vocabulary(self) -> None:
        """Hand the current palette to every window that writes labels. One vocabulary, one
        set of digits — the gallery and the inspector must never disagree about what 3 means."""
        if self._event_gallery is not None:
            self._event_gallery.set_vocabulary(self._label_vocab)
        if self._event_inspector is not None:
            self._event_inspector.set_vocabulary(self._label_vocab)

    def _open_event_gallery(self) -> None:
        """Analysis ▸ Event gallery… — open (or re-focus) the single EventGalleryWindow.

        Wired exactly like the inspector, and deliberately so: the gallery installs the SAME
        shared events df and announces curation through the same three signals, so both
        windows are two views of one frame. They are also kept in step with each other —
        clicking a tile drives the inspector to that event and vice versa, through the
        AnalysisPanel that already brokers selection for the table and the trace.
        """
        from .event_gallery import EventGalleryWindow

        if self._event_gallery is None:
            gal = EventGalleryWindow(self._session, self)
            gal.eventSelected.connect(self._analysis_panel.select_event)
            # ...and a word about it, because everything ``select_event`` moves is on the
            # window the gallery is standing in front of (see _note_gallery_selection).
            # Connected AFTER the relay so the sentence describes work already done.
            gal.eventSelected.connect(self._note_gallery_selection)
            self._analysis_panel.eventSelected.connect(gal.show_event)
            # Double-click on a tile: go to the event IN THE TRACE. select_event already
            # zooms and marks it, so the part this adds is bringing the main window forward
            # — the gallery is a separate window and is, by construction, in front of the
            # thing the user just asked to look at.
            gal.eventActivated.connect(self._goto_event_in_trace)
            # Context menu "Open in Event Inspector": the direct road between the two
            # curation surfaces (owner notes #2/#5). Open (or focus) the inspector, then
            # select the event THROUGH the panel — the same relay a table-row click uses,
            # so the inspector, the table and the trace all land on it together.
            gal.inspectRequested.connect(self._open_inspector_at_event)
            gal.acceptanceChanged.connect(self._analysis_panel.set_event_accepted)
            self._analysis_panel.eventAcceptanceChanged.connect(gal.reflect_acceptance)
            # Bulk curation writes the shared frame in place, so — as with the inspector's
            # Accept all — it reaches the table, the overlays, the summary, the surface
            # gates and the dirty flag by announcement, never through add_result.
            gal.curationBatchChanged.connect(self._analysis_panel.resync_curation)
            # NOT ``_checkpoint_batch_curation``: that hint is the inspector's, because only
            # the inspector's signal means a whole-frame gesture. See both docstrings.
            gal.curationBatchChanged.connect(self._checkpoint_gallery_curation)
            # WHICH GESTURE, since the signal will not say. ``pressed`` is emitted before
            # ``clicked`` and the window's own handler hangs off ``clicked``, so this arms
            # BEFORE the rows move; our ``clicked`` connection was made after the window's
            # and so runs AFTER the batch signal has been delivered and consumed, which is
            # what keeps the marker from outliving its own click (a gesture that moves no
            # row emits no batch signal at all). Every other curation path in the gallery —
            # the "…selected" buttons, A / R, Space — leaves it disarmed, and is therefore
            # named by the derived label, which is the honest answer for a subset gesture.
            for _btn, _accept in ((gal.accept_all_btn, True), (gal.reject_all_btn, False)):
                _btn.pressed.connect(
                    lambda accept=_accept: self._arm_gallery_whole_frame(accept))
                _btn.clicked.connect(self._disarm_gallery_whole_frame)
            gal.curationBatchChanged.connect(self._refresh_workflow_status)
            gal.curationBatchChanged.connect(self._sync_analysis_surfaces)
            gal.curationBatchChanged.connect(self._mark_dirty)
            gal.labelsChanged.connect(self._analysis_panel.resync_curation)
            gal.labelsChanged.connect(self._mark_dirty)
            gal.labelsChanged.connect(self._checkpoint_labels)
            # Same single-writer save route as the Inspector. In particular, this does
            # NOT add a menu bar to the Gallery: its own window-scoped actions and visible
            # control emit these two requests, and MainWindow does the write.
            gal.saveProjectRequested.connect(
                lambda w=gal: self._save_project_from(w, save_as=False))
            gal.saveProjectAsRequested.connect(
                lambda w=gal: self._save_project_from(w, save_as=True))
            self.projectSaveStateChanged.connect(gal.set_project_save_state)
            gal.set_project_save_state(
                self._project_path or "", self._visible_project_dirty())
            # A refusal WRITES nothing, so no data signal will repaint the other two
            # surfaces' notes — the mirror of the table's acceptanceRefused hop.
            gal.acceptanceRefused.connect(self._on_gallery_acceptance_refused)
            # ⌘Z IN THE WINDOW THE MISTAKE WAS MADE IN, on the ONE undo stack.
            #
            # The gallery is a top-level QMainWindow with no menu bar
            # (``gal.menuBar().actions() == []``), and Edit ▸ Undo is a WindowShortcut on
            # THIS window — which is behind it. Measured before these two lines, wired app,
            # -1000 mV recording: Ctrl+A selected 35 tiles, R took accepted 110 -> 75, this
            # window's Edit menu read "Undo Curate 35 events" ENABLED, and Ctrl+Z on the
            # gallery's view, Ctrl+Z on the gallery window and a click on its own Undo
            # button all left it at 75. Only clicking back here undid it.
            #
            # The gallery emits INTENT and never keeps a stack: a second undo stack is a
            # second answer to "what was the last step?", and this one would be answering
            # for a table three other surfaces are writing. ``trigger`` and not
            # ``_undo_events`` directly, so a disabled action stays disabled from over
            # there too — the gesture goes through the same door the menu item does,
            # including its RECENCY arbitration between the two tables (see _note_step).
            gal.undoRequested.connect(self._act_undo.trigger)
            # ...and the state comes back the other way, off the action's OWN ``changed``
            # rather than a call sited next to each of the ten places that re-sync it: the
            # button can then be disabled when there is nothing to undo and can say WHAT it
            # would undo (the action's text, e.g. "Undo Curate 35 events") whenever it can.
            # One writer, and it cannot fall behind. The seed call below is needed because
            # ``changed`` has not fired yet on a gallery opened after the stack already had
            # steps on it, and the window's own lower-bound guess is not this action's
            # answer — measured: freshly opened over a 35-tile curation, the button was
            # enabled and captioned "Undo" while the menu said "Undo Curate 35 events".
            self._act_undo.changed.connect(
                lambda g=gal: g.set_undo_state(self._act_undo.isEnabled(),
                                               self._act_undo.text()))
            gal.set_undo_state(self._act_undo.isEnabled(), self._act_undo.text())
            self._wire_surface_refresh("event_gallery", gal)
            self._event_gallery = gal
            # The gallery is a launcher surface now, so its show/hide has to reach the row's
            # "this window is open" tint like every other surface window's does.
            self._watch_surface_window(gal)
        self._event_gallery.set_vocabulary(self._label_vocab)
        self._refresh_event_gallery()
        # A gallery opened AFTER a subpopulation run still gets that run: the assignment
        # lives here, not on the window, precisely so that opening order does not decide
        # what the user sees. _refresh_event_gallery has just installed the frame, and
        # set_events drops any stale map, so this is the point at which it is safe to offer.
        self._refresh_subpop_views()
        self._event_gallery.show()
        self._event_gallery.raise_()
        self._event_gallery.activateWindow()

    def _goto_event_in_trace(self, event_id: int) -> None:
        """Select ``event_id`` and bring the MAIN TRACE out from behind the curation windows.

        Reached from the gallery's double-click and from the Event Inspector's "Show in
        trace" / T — in both cases the user is asking to look at the trace, and in both
        cases the window they asked from is on top of it.

        *** LOWER THE CHILD, DO NOT JUST RAISE THE PARENT. *** Both windows are built as
        ``X(self._session, self)`` — a QMainWindow with a transient parent — and on macOS a
        transient child stays above its parent whatever the parent does, so ``raise_()``
        alone moved nothing the user could see and "go to this event in the trace" looked
        like a dead gesture. Lowering the child works where raising the parent does not.
        Un-parenting is NOT the fix: seven windows use that same construction and
        ``_close_holders`` discovers holders through the Qt object tree as well as through
        this window's attributes, so un-parenting one of them would make it the sole
        exception at shutdown.
        """
        self._analysis_panel.select_event(int(event_id))
        for win in (self._event_gallery, self._event_inspector):
            if win is not None and win.isVisible():
                win.lower()
        self.raise_()
        self.activateWindow()

    def _note_gallery_selection(self, event_id: int) -> None:
        """A PLAIN CLICK on a tile also did something — say so.

        The click already selects the event everywhere (the ``select_event`` relay wired
        beside this one zooms the trace, moves the table row and moves the marker), but all
        three of those are on the window the gallery is sitting in front of, so from the
        user's chair a single click changed nothing at all. This is the one line that names
        what happened and what the next gesture is.
        """
        self.statusBar().showMessage(
            f"Event {int(event_id)} selected — the trace, the table and the marker moved. "
            f"Double-click the tile to bring the trace forward.", 6000)

    def _open_inspector_at_event(self, event_id: int) -> None:
        """Gallery context menu → the Event Inspector, ON that event (owner notes #2/#5).

        Open (or focus) the inspector first, THEN select through the panel: the panel's
        ``eventSelected`` relay is what lands the inspector on the event, and it only
        reaches an inspector that exists. Selecting through the panel — not calling
        ``show_event`` directly — also lands the table row and the trace zoom, so all
        three surfaces agree about which event is being looked at.

        *** AND THEN SAY IT TO THE INSPECTOR ITSELF. *** The panel relay alone is not
        enough, because ``select_event`` is a no-op when the requested row is ALREADY the
        table's current row — which it always is for a gesture on the FOCUSED tile: the
        gallery's own ``eventSelected`` put it there the moment the tile took focus. So a
        freshly opened inspector received no selection and sat on event 0 while the user
        had asked for event 5 (measured). ``show_event`` after the relay costs nothing when
        the relay did land (the inspector is already there) and is the whole gesture when
        it did not."""
        self._open_event_inspector()
        self._analysis_panel.select_event(int(event_id))
        if self._event_inspector is not None:
            self._event_inspector.show_event(int(event_id))

    def _open_gallery_at_event(self, event_id: int) -> None:
        """The events table's right-click ▸ "Open in Event gallery" — the gallery, focused on
        that event.

        The mirror of :meth:`_open_inspector_at_event`, and it needs the same second step
        for the same reason: ``select_event`` is a no-op when the row is already current
        (a right-click makes it current), so the panel relay alone would leave a freshly
        opened gallery on tile 0.
        """
        self._open_event_gallery()
        self._analysis_panel.select_event(int(event_id))
        if self._event_gallery is not None:
            self._event_gallery.show_event(int(event_id))

    def _refresh_event_gallery(self) -> None:
        """(Re-)feed the open EventGalleryWindow the current detection's shared df.

        RE-FEEDING THE SAME FRAME IS A RE-READ, NOT AN INSTALL. ``set_events`` rebuilds the
        model and puts the cursor back on tile 0, which is correct for a different
        detection and destructive for the same one: the gallery labels a MULTI-TILE
        selection, so a refresh in the middle of that work threw the selection away and the
        curator's next digit wrote one label instead of five. The window already carries the
        cheap door for this (``refresh()`` — "re-read the shared frame after somebody else
        edited it"), and the frame's own identity says which door to use. The Inspector
        makes the same distinction inside its own ``set_events``.

        This matters more now than it did: the refresher is reached from every signal in
        :data:`SURFACE_MUTATION_SIGNALS`, including the gallery's own ``labelsChanged``.
        """
        if self._event_gallery is None:
            return
        from ..analysis.events import EventSet

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            self._event_gallery.clear()
            self._push_stale_columns()
            return
        # Same rule as the inspector's, and for the same reason: ``set_events`` drops the
        # gallery's level table, so every re-feed has to re-establish it. Pushed on the
        # cheap ``refresh()`` path too — a re-read cannot install what it never had, and
        # the gallery was the surface whose CUSUM bars nothing installed at all.
        levels = self._levels_for_detection(res)
        if self._event_gallery.frame() is res.payload.df:
            self._event_gallery.refresh()
            self._event_gallery.set_levels(levels)
            self._push_stale_columns()
            return
        # A NEW FRAME IS ABOUT TO REPLACE THE MAP'S: this is the last instant at which the
        # gallery still says what the curator was drawing the held subpopulation map with,
        # because ``set_events`` drops the map and with it the representative sample. Taken
        # here so the round trip away and back can put the view back (C4); it reads nothing
        # unless the gallery is carrying the map right now.
        self._remember_gallery_subpop_view()
        # The refusal ledger travels WITH the frame, chosen by LINEAGE: the same standing
        # record when this is the same detection re-fed (an undo's restored snapshot, a
        # merge's rewritten frame — both are new df objects), a fresh one when a new
        # detection replaced it. This is what keeps the "kept rejected" note alive across
        # exactly the re-feeds that used to erase it while the rows stayed refused.
        self._event_gallery.set_events(
            res.doc_id, res.channel, res.payload.df, res.module_id,
            dict(res.payload.provenance or {}),
            refusals=self._refusal_ledger_for(res))
        self._event_gallery.set_levels(levels)
        # ...and the staleness badges, on EVERY branch: the table's judgement is about
        # the frame just fed, and a window fed a frame without its badges repeats a
        # superseded depth with no mark (S7-POP1).
        self._push_stale_columns()

    # -- the staleness badges, on every surface that prints a measured column ------------------
    def _push_stale_columns(self) -> None:
        """S7-POP1: the events table's stale badge set + the banner's sentence, on both
        curation windows — the SAME tuple the table was given.

        The table derives the set (``REMEASURED_COLUMNS`` on a baseline or calibration
        change, ``()`` when clean; see ``AnalysisPanel._judge_staleness``) and this pushes
        exactly that, so three surfaces cannot tell three stories about one frame.
        Measured before it existed: one Preprocess knob after a detection badged nine table
        headers ⚠ and raised the banner, while the gallery went on sorting and captioning
        by ``depth_pA`` unmarked and the inspector printed "depth: 122.37 pA" bare — both
        separate top-level windows sitting over the banner. Reached from every re-feed
        (``_refresh_event_gallery`` / ``_refresh_event_inspector``) and, for the two paths
        that reach no re-feed — the knob and the gain edit — from
        ``AnalysisPanel.stalenessChanged``. Idempotent: both seams no-op on an unchanged
        tuple + sentence.
        """
        cols = tuple(self._analysis_panel._table.stale_columns)
        note = self._analysis_panel._stale.message() if cols else ""
        for win in (self._event_gallery, self._event_inspector):
            if win is not None:
                win.set_stale_columns(cols, note)

    # -- the fitted sub-state levels, on every surface that draws them --------------------------
    def _push_level_fit(self, payload) -> None:
        """Hand a level table (or ``None``) to EVERY curation surface that draws one.

        The Inspector prints the per-level breakdown in its readout and draws the steps on
        the waveform; the gallery draws them as bars on each tile. They take the same
        ``set_levels`` frame by design, so there is one call site instead of two that can
        drift — the gallery's copy was simply never installed by anything before this.
        """
        for win in (self._event_inspector, self._event_gallery):
            if win is not None:
                win.set_levels(payload)

    def _levels_for_detection(self, res) -> Any:
        """The fitted levels to draw under ``res``'s events — ``None`` unless the standing fit
        was measured on exactly that recording and channel.

        *** THE RE-FEED IS WHERE THE LEVELS GO MISSING. *** Both curation windows null their
        level table inside ``set_events`` whenever the frame differs (see
        ``EventInspectorWindow.set_events``: a fit belongs to the frame it was measured on),
        and until now nothing put it back. So fitting the levels and THEN opening the
        Inspector gave a window with no CUSUM steps and an ``n_levels``-only readout — the
        fit looked like it had not run — and the same was true of any surface that was
        re-fed after switching recording and back.

        The guard is the one ``_fit_levels`` itself applies, and it is not optional: levels
        from recording A printed under recording B's events would be matched by ``event_id``,
        which both tables have and which means nothing across recordings. That hazard is the
        entire reason ``set_events`` resets in the first place, so re-establishing has to
        answer it.
        """
        fit = self._level_fit
        if fit is None or res is None:
            return None
        if fit.doc_id != res.doc_id or int(fit.channel) != int(res.channel):
            return None
        payload = getattr(fit, "payload", None)
        if payload is None or getattr(payload, "empty", True):
            return None
        return payload

    def _refresh_event_inspector(self) -> None:
        """(Re-)feed the open EventInspectorWindow the current detection's shared df, or clear it
        when the selection has no detection. A no-op until the window has been built."""
        if self._event_inspector is None:
            return
        from ..analysis.events import EventSet

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            self._event_inspector.clear()
            self._push_stale_columns()
            return
        # The ledger travels with the frame, chosen by lineage — see _refresh_event_gallery.
        self._event_inspector.set_events(res.doc_id, res.channel, res.payload.df, res.module_id,
                                         res.payload.provenance,
                                         refusals=self._refusal_ledger_for(res))
        # ...and the levels are re-established on EVERY re-feed, not only on the one that
        # happens to follow a fit (see _levels_for_detection)...
        self._event_inspector.set_levels(self._levels_for_detection(res))
        # ...and so are the staleness badges (S7-POP1; see _push_stale_columns).
        self._push_stale_columns()

    # -- the shared refusal ledger (one record per detection lineage) ---------------------------
    def _refusal_ledger_for(self, res) -> Any:
        """The refusal ledger for ``res``'s detection LINEAGE — the same instance for as
        long as the lineage stands, a fresh (empty) one the moment it is replaced.

        WHY LINEAGE AND NOT THE FRAME OBJECT. The windows used to drop their records
        whenever a different ``df`` object arrived, and three ordinary paths hand them
        one: Edit ▸ Undo installs a deep-copied snapshot, a merge installs a rewritten
        frame, and a window opened late is fed the frame it never saw. All three left
        rows refused with the disclosure gone (measured; see the windows' ``set_events``).
        What actually identifies "the same detection, re-fed" is the run stamp the
        detector wrote into provenance — ``detected_at``, copied intact by the merge and
        by the undo snapshot, and NEW on every re-run — so that stamp is the token.

        A payload with no stamp (a frame off the Schmitt path) falls back to frame-object
        identity, which is exactly the windows' old private rule: without a run stamp
        this window cannot tell a re-fed frame from a new detection, and guessing
        "same" would let one detection's refusals stand over another's events.

        THE TOKEN IS IN THE CACHE KEY, so every lineage this session has seen keeps its
        own record side by side. With one slot per (doc, channel, module), resolving an
        OLD lineage's frame — which Edit ▸ Undo past a re-detection does as a matter of
        course — evicted the newer lineage's record, and the redo then restored the
        newer frame with its rows still refused, its curation exact, and the standing
        "N kept rejected" disclosure silently gone on all three surfaces (measured; a
        single "Undo Re-run detection" was enough). Entries are pruned when the
        recording closes (:meth:`_drop_refusals_for_doc`), exactly as before.
        """
        from .event_gallery import _RefusalLedger

        prov = getattr(res.payload, "provenance", None) or {}
        token = prov.get("detected_at")
        key = (str(res.doc_id), int(res.channel), str(res.module_id),
               None if token is None else str(token))
        df = res.payload.df
        entry = self._event_refusals.get(key)
        if entry is not None:
            stored_token, frame_ref, anchors, ledger, retired = entry
            # With the token in the key, a hit IS the lineage — except the no-stamp
            # slot (token None), which still falls back to frame identity: without a
            # run stamp a different frame cannot be told from a new detection.
            same = token is not None or frame_ref() is df
            if same:
                old_df = frame_ref()
                if old_df is not df:
                    # SAME LINEAGE, DIFFERENT FRAME — an undo's snapshot or a merge. A
                    # merge renumbers ``event_id`` wholesale (``np.arange`` over the
                    # survivors), so the ids the ledger remembers can now name OTHER
                    # events. Measured on the owner's -600 mV recording: rows 0/4/260
                    # refused, one early merge, and the surviving note counted 2 while
                    # three rows stayed refused — the id 4 it remembered had become an
                    # ordinary event's. Re-point the record before anyone recounts it.
                    #
                    # Anchors are read off the OLD frame while it is still alive (the
                    # windows hold it until this very re-feed replaces it), because the
                    # STORED anchors can miss late refusals: the no-change bulk accept
                    # records refusals but emits nothing, so nothing resolved this
                    # ledger between the record and the frame swap. The stored map is
                    # only the fallback for an old frame already collected.
                    if old_df is not None:
                        anchors = self._refusal_anchors(ledger, old_df)
                    retired.update(self._remap_refusals(ledger, anchors, df))
                    # ...and the OTHER direction of the same swap: a refusal the merge
                    # retired comes back when the frame that holds its row does — Edit ▸
                    # Undo of the merge restores the row refused, and a disclosure that
                    # stayed down would undercount the standing refusals indefinitely
                    # (measured: 3 rows refused, note stuck at 2). The retire is recorded
                    # by anchor, so only a frame that again HAS that start resurrects it.
                    self._restore_retired_refusals(ledger, retired, df)
                # Anchors are re-taken on EVERY resolve (a mask edit moves start_idx in
                # place, and refusals recorded since the last resolve have none yet).
                self._event_refusals[key] = (
                    token, weakref.ref(df), self._refusal_anchors(ledger, df), ledger,
                    retired)
                return ledger
        ledger = _RefusalLedger()
        # The weak ref never keeps a replaced detection's frame alive just to compare it.
        self._event_refusals[key] = (token, weakref.ref(df), {}, ledger, {})
        return ledger

    @staticmethod
    def _refusal_anchors(ledger, df) -> dict[int, int]:
        """``event_id -> start_idx`` for every event the ledger remembers, read off the
        CURRENT frame. ``start_idx`` is the identity that survives a merge's renumbering:
        an unmerged event keeps its exact start, and a merged run starts where its head
        fragment did — while a refused (unmeasurable) row is precisely the row a merge
        leaves alone. Only the remembered handful is anchored, never the whole frame."""
        import numpy as np

        remembered = set(ledger.kept) | set(ledger.repaired)
        if not remembered or "event_id" not in df.columns or "start_idx" not in df.columns:
            return {}
        ids = df["event_id"].to_numpy()
        mask = np.isin(ids, np.fromiter((int(e) for e in remembered), dtype=np.int64,
                                        count=len(remembered)))
        starts = df["start_idx"].to_numpy()[mask]
        return {int(i): int(s) for i, s in zip(ids[mask], starts)}

    @staticmethod
    def _remap_refusals(ledger, anchors: dict[int, int], new_df) -> dict[int, str]:
        """Re-point the ledger's remembered ``event_id``s at the same EVENTS in
        ``new_df``, by their anchored ``start_idx``. An event whose start is gone was
        merged away — its refusal retires with it, exactly as one that left the frame. An
        id with no anchor (recorded since the last resolve) is kept as-is: with nothing
        to re-point by, dropping it would silently retire a real refusal. Uses only the
        ledger's public scope API (``forget`` + ``record``), which is also why the
        buckets survive the trip — which bucket an event is in is history.

        Returns the refusals this remap RETIRED, as ``start_idx -> bucket``. A retire is
        right for the merge and wrong the moment the merge is undone (the restored frame
        holds the refused row again), so the caller keeps the retired anchors alongside
        the ledger and :meth:`_restore_retired_refusals` resurrects each one on the
        first re-fed frame that has its start back. Empty when nothing retired."""
        retired: dict[int, str] = {}
        old_kept, old_repaired = set(ledger.kept), set(ledger.repaired)
        if not (old_kept or old_repaired):
            return retired
        if "event_id" not in new_df.columns or "start_idx" not in new_df.columns:
            return retired
        by_start = {int(s): int(i) for s, i in zip(new_df["start_idx"].to_numpy(),
                                                   new_df["event_id"].to_numpy())}

        def _remapped(ids, bucket: str) -> set[int]:
            out: set[int] = set()
            for e in ids:
                if e not in anchors:
                    out.add(int(e))                     # unanchored: keep as-is
                elif anchors[e] in by_start:
                    out.add(by_start[anchors[e]])       # the same event, renumbered
                else:
                    # Merged away — retired with the event, remembered by its anchor so
                    # an undo that brings the row back brings the refusal back with it.
                    retired[int(anchors[e])] = bucket
            return out

        new_kept = _remapped(old_kept, "kept")
        new_repaired = _remapped(old_repaired, "repaired")
        if new_kept == old_kept and new_repaired == old_repaired:
            return retired
        ledger.forget(old_kept | old_repaired)
        ledger.record(kept=new_kept, repaired=new_repaired)
        return retired

    @staticmethod
    def _restore_retired_refusals(ledger, retired: dict[int, str], new_df) -> None:
        """Resurrect every retired refusal whose anchored ``start_idx`` exists again in
        ``new_df`` — the flip side of the retire in :meth:`_remap_refusals`.

        The retire is one-way on the ledger (``forget``), which was right for the merge
        and irreversible for its undo: the undo installs the pre-merge frame under the
        SAME lineage token, the frame again holds the row the accept was refused over,
        and the note kept saying one fewer indefinitely — against the ledger's own
        contract ("a count of currently-refused rows, not a latch ... goes back UP") and
        USER_GUIDE 9b's "An undo restores the table exactly". Re-recording into the
        remembered bucket keeps history intact; a resurrection onto a row that is no
        longer refused adds nothing to the note, because the count is re-read from the
        frame (``_RefusalLedger.counts``), never from the record. Restored entries leave
        ``retired`` (in place) so a redo of the merge simply retires them again."""
        if not retired:
            return
        if "event_id" not in new_df.columns or "start_idx" not in new_df.columns:
            return
        by_start = {int(s): int(i) for s, i in zip(new_df["start_idx"].to_numpy(),
                                                   new_df["event_id"].to_numpy())}
        for start, bucket in list(retired.items()):
            if start in by_start:
                ledger.record(**{bucket: (by_start[start],)})
                del retired[start]

    def _current_refusal_ledger(self) -> Any:
        """The active detection's shared refusal ledger, or None when there is none.
        Handed to the events table as its resolver (see the wiring in ``__init__``)."""
        from ..analysis.events import EventSet

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return None
        return self._refusal_ledger_for(res)

    def _drop_refusals_for_doc(self, doc_id: str) -> None:
        """A closed recording's refusal records name events nothing can show any more."""
        for key in [k for k in self._event_refusals if k[0] == str(doc_id)]:
            del self._event_refusals[key]

    def _on_table_acceptance_refused(self, _event_id: int) -> None:
        """The events table refused an accept. Nothing was written, so nothing else will
        announce it — repaint the two windows' notes from the shared ledger so the three
        surfaces say the same thing at the same moment (the table's own label is already
        up). Both refreshers are same-frame no-op re-reads here: cheap and idempotent."""
        self._refresh_event_gallery()
        self._refresh_event_inspector()

    def _on_gallery_acceptance_refused(self) -> None:
        """The gallery refused an accept with NOTHING written (its ``acceptanceRefused``).
        The reverse of :meth:`_on_table_acceptance_refused`, and it was missing: the
        refusal landed on the SHARED ledger but only the gesturing window rebuilt its
        note, so the events table's label and an open inspector stayed silent — or kept
        displaying a stale count — about the same record.

        Repaint-only, and — exactly like the table's hop — it repaints the OTHER two
        surfaces: the gesturing window's own note is already up, and re-feeding it from
        the session here would overwrite whatever frame it is disclosing about. The
        table's note sync is reached directly (``_table`` is the panel's widget, but the
        ledger is cross-surface wiring — the same justification as ``set_refusals_source``
        in ``__init__``; a model reset via ``refresh_all`` would throw away the user's
        row selection for a gesture that wrote nothing). The inspector refresher is a
        same-frame re-read that rebuilds its note and emits no curation signal, so this
        hop cannot loop.
        """
        self._analysis_panel._table._sync_refusal_note()
        self._refresh_event_inspector()

    def _on_inspector_acceptance_refused(self, _event_id: int) -> None:
        """The inspector refused an accept with NOTHING written — the same repaint-only
        hop as :meth:`_on_gallery_acceptance_refused`, pointed at the other two
        surfaces (see there for why the gesturing window is left alone)."""
        self._analysis_panel._table._sync_refusal_note()
        self._refresh_event_gallery()

    # -- post-detection sub-event merge (MainWindow is the sole wirer) --------------------------
    def _on_merge_all_requested(self, window_s: float) -> None:
        """'Merge all' — collapse every run of fragments within the window across the whole active
        detection, then report the before/after counts in the status bar."""
        report = self._analysis_panel.merge_all_subevents(float(window_s))
        self._post_merge_status(report, window_s=float(window_s))

    def _on_merge_region_requested(self) -> None:
        """'Merge events in selection' — collapse every event overlapping the on-trace Measure
        selection into one. Nudges the user to select a region first when there is none."""
        report = self._analysis_panel.merge_selection_subevents()
        self._post_merge_status(report, region=True)

    def _on_merge_with_next(self, event_id: int) -> None:
        """Event Inspector 'Merge with next' — stitch the browsed event to its forward run within
        the panel's merge window, then advance the browser to the following event."""
        window_s = self._analysis_panel.merge_window_s()
        report = self._analysis_panel.merge_event_with_next(int(event_id), window_s)
        if (report.status == "ok" and report.focus_id >= 0
                and self._event_inspector is not None):
            self._event_inspector.show_event(int(report.focus_id))
        self._post_merge_status(report, window_s=window_s)

    def _post_merge_status(self, report, *, window_s: float | None = None,
                           region: bool = False) -> None:
        """Turn a :class:`~nanoscope.gui.analysis_panel.MergeReport` into an honest status line.

        ``window_s`` may be None for a report that arrived LATE: 'Merge all' runs on
        the analysis controller (3.3-8.7 s on the owner's recordings), so its outcome
        comes back through ``mergeFinished`` long after the click, and the report
        carries the window it was asked for. The caller's value still wins when given,
        so the two synchronous entry points are unchanged.
        """
        if window_s is None:
            window_s = getattr(report, "window_s", None)
        if report.status == "started":
            # THE MISSING SENTENCE. The status-bar progress and the Cancel button are
            # already up (AnalysisController.started drives both); this says what they
            # are for. A long job with a progress bar and no words is still a window
            # that has stopped answering.
            msg = (f"Merging {report.n_before} events… (Cancel in the status bar stops "
                   f"it; nothing is changed until it finishes.)")
        elif report.status == "busy":
            msg = "Another analysis is still running — wait for it to finish, or cancel it."
        elif report.status == "dropped":
            msg = ("The merge was discarded: the event table changed while it ran, so its "
                   "answer describes rows that are no longer there. Nothing was changed — "
                   "press Merge all again.")
        elif report.status == "failed":
            msg = "The merge could not finish. The event table is unchanged."
        elif report.status == "no_detection":
            msg = "Run detection first — there are no events to merge."
        elif report.status == "reconciled":
            msg = (
                "The Reconciled table is curation-only. Switch to a source detector "
                "table, add or merge there, then reconcile again."
            )
        elif report.status == "no_selection":
            msg = ("Select a region on the trace first — turn on the Measure tool and drag, "
                   "then Merge events in selection.")
        elif report.status == "low_occupancy":
            # NOT a no-op: runs WERE within the window / under the drag, and every one of
            # them was declined because the span they would make is mostly open pore
            # (analysis.merge's occupancy floor). Reporting that as "nothing to merge" is
            # precisely what made the merge look broken — the user asked twice, widened the
            # window, and got the same sentence about a set of events that plainly WERE
            # inside it.
            n = getattr(report, "n_refused_low_occupancy", 0)
            if region:
                msg = ("That selection is mostly open pore between its events, so merging "
                       "it would make one long, shallow event rather than one "
                       "translocation. Nothing was changed — drag over the fragments of a "
                       "single event instead.")
            else:
                where = f"{window_s * 1e3:g} ms" if window_s is not None else "the window"
                msg = (f"Nothing merged: {n} run(s) within {where} are mostly open pore "
                       f"between their events, so merging them would make one long, "
                       f"shallow event. Try a smaller window.")
        elif report.status == "noop":
            if region:
                msg = "Fewer than two events overlap the selection — nothing to merge."
            else:
                # The spinbox speaks milliseconds now (a µs field cost the first
                # hands-on user a 1000x under-merge); this sentence speaks the same
                # unit as the control it points back at.
                where = f"{window_s * 1e3:g} ms" if window_s is not None else "the window"
                msg = f"No sub-events within {where} to merge."
        else:  # "ok"
            merged = report.n_before - report.n_after
            msg = (f"Merged {merged} fragment(s): {report.n_before} → "
                   f"{report.n_after} events.")
            # A PARTIAL refusal has to be said out loud too. Without this the user reads
            # "Merged 12 fragment(s)", counts the events that did not move, and concludes
            # the merge is unreliable — when in fact the floor declined exactly the runs
            # whose merged span would have been mostly open pore.
            n_ref = getattr(report, "n_refused_low_occupancy", 0)
            if n_ref:
                msg += (f" {n_ref} further run(s) were left alone: they are mostly open "
                        f"pore between their events.")
        self.statusBar().showMessage(msg, 8000)

    def _on_baseline_job_status(self, _token: str, message: str) -> None:
        """Post whatever the panel said about a baseline re-measure / comparison.

        The panel composes the sentence because the panel is what knows the estimator,
        the event count and the reason. This does not second-guess it: a window that
        rewrote the message would be the second version of the story that
        ``baselineJobStatus`` exists to prevent."""
        if message:
            self.statusBar().showMessage(message, 8000)

    def _compare_baseline_estimators(self) -> None:
        """Analysis ▸ Compare baseline estimators… — delegate to the panel.

        The window owns the menu, the panel owns the detection, the trace and the
        controller; this posts the honest reason when the panel refuses, exactly as
        the merge actions do. It never reaches into the panel's internals and it
        never re-measures anything itself.
        """
        # The panel says what happened through ``baselineJobStatus`` — the SAME
        # channel the stale banner's own button uses — so a refusal reads identically
        # whichever surface asked for it, and this action never composes a second
        # version of the sentence.
        self._analysis_panel.compare_baseline_estimators()

    def _export_source(self, doc_id: str) -> "tuple[str, object | None]":
        """``(the FILE these rows came from, that document's meta)`` for an export.

        ONE ASK, ONE DOCUMENT. Every scoring export answers two questions about the
        recording behind its rows — which file to name in ``source_file``, and which
        clock its ``t_start_s`` column is on — and they must be answered about the
        SAME document or the sidecar names one recording and stamps another's origin.
        So they are asked together, once, here.

        ``source_of``, not the raw doc_id: the fallback is only reached once the
        recording has been closed, and what belongs in an export's provenance is the
        FILE it came from, not this session's private per-load token (core.identity).

        The meta is the whole ``RecordingMeta``, handed to
        :func:`~nanoscope.analysis.scoring.clock_stamp` unopened on purpose — that seam
        picks ``clock.instrument_origin_s`` (MEASUREMENT_SCHEMA's definition of
        ``t0_s``: a segment's PARENT clock, no layer nudge), which is a choice this
        window must not make for it, because the wrong one of the two clock functions
        differs by 1158.99 s on a cut of the owner's own quick ABF. ``None`` once the
        recording is closed — an honest "the origin is not knowable from here", which
        the sidecar writes as an explicit null rather than a fabricated 0.
        """
        if self._session.has_document(doc_id):
            meta = self._session.document(doc_id).meta
            return str(meta.path), meta
        return source_of(doc_id), None

    def _export_events(self) -> None:
        """Analysis ▸ Export events… — write the current detection's events to a CSV,
        honouring curation (the accepted/curated-in events). This is the plain
        'save these detections for extraction' path, distinct from the figure export."""
        from ..analysis.events import EventSet
        from ..analysis.scoring import events_export_paths, write_events_export

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, "Export events",
                "Run detection on a current channel first — there are no events to export.")
            return
        events = res.payload

        # TWO files, so the confirmation covers both: the CSV and its
        # `<stem>.meta.json` provenance sidecar. The sidecar is the only place the
        # open-pore estimator's identity (`baseline_fingerprint` and every
        # parameter) reaches a consumer outside this app — the 20-column schema is
        # frozen and must not grow one. Without it, two CSVs exported months apart
        # are not comparable and nothing in either says why.
        path = ask_export_path(
            self, title="Export events", suggestion="events.csv",
            filters="CSV (*.csv);;All files (*)",
            outputs=events_export_paths)
        if path is None:
            return
        accepted = events.accepted.df
        source_file, doc_meta = self._export_source(res.doc_id)
        try:
            out = write_events_export(
                accepted, str(path),
                provenance=self._export_provenance(events.provenance, accepted),
                source_file=source_file, detector=res.module_id, meta=doc_meta,
                # How many rows the table this export came FROM had, so the sidecar's
                # `curation.n_withheld` is the exact count rather than the one inferred
                # from `provenance['n_events']` (which a merge or a hand-add can leave
                # stale -- see events.reconcile_row_counts). The frame is in hand here;
                # the sidecar was guessing from a number written before curation.
                n_total=len(events.df))
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export events", f"Could not write the export:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Wrote {len(accepted)} accepted event(s) of {len(events.df)} to "
            f"{out['data'].name} (+ provenance {out['meta'].name}).", 8000)
        self._disclose_identity_notes("Export events", list(out.values()))

    @staticmethod
    def _export_provenance(provenance: dict | None, df) -> dict:
        """The detection's provenance plus the note a ``dI/I0`` above 1 must travel with.

        THE OTHER HALF OF THE SUMMARY'S WARNING. ``frac_depth`` is
        ``depth_pA / |baseline_pA|`` and reads as a blockade fraction — a number
        between 0 and 1. On the owner's apoferritin recordings it is 1.81–2.57,
        because the current MAGNITUDE grows during an event, and the Summary table now
        says so with a glyph and a sentence. A CSV leaving the app said nothing at all,
        and the CSV is the copy a co-author actually gets: a column of 2.36 with a
        header spelled ``frac_depth`` is exactly how "2.36" becomes "a 236 % blockade"
        in someone else's notes, with no screen anywhere to contradict it.

        An ADDITIVE PROVENANCE KEY, which is the channel the frozen 20-column schema
        leaves open for precisely this (a new column is forbidden; ``provenance`` and
        the ``flag`` token set are not). It rides in ``<stem>.meta.json`` beside
        ``baseline_fingerprint``, which is where a reader already has to look to find
        out what the depths were measured against.

        The sentence comes from :meth:`nanoscope.gui.panels.StatsPanel.frac_depth_note`
        — the one owner of this wording — so the file and the screen cannot come to
        describe the same number differently. Keyed on the MEDIAN of the rows actually
        being written, so a curated export is judged on what it contains rather than on
        what was detected.
        """
        import numpy as np

        from .panels import StatsPanel

        prov = dict(provenance or {})
        # An EMPTY export (everything rejected) has no median to take, and asking numpy
        # for one is a "Mean of empty slice" warning on a path the user reached on
        # purpose. No rows, no claim about them.
        if df is None or not len(df):
            return prov
        try:
            median = float(np.nanmedian(df["frac_depth"].to_numpy()))
        except (KeyError, TypeError, ValueError):
            return prov
        note = StatsPanel.frac_depth_note(median)
        if note:
            prov["frac_depth_note"] = note
            prov["frac_depth_median"] = median
        return prov

    def _fit_levels(self) -> None:
        """Analysis ▸ Fit sub-state levels (CUSUM)… — resolve the levels inside each
        accepted event, show them in the Event Inspector and enable the level export.

        The two inputs that decide whether the answer means anything are the SIGNAL and
        the BASELINE, and both come from the same doors every other re-measurement in
        this app uses:

        * ``modules.detection_signal`` rebuilds ``lowpass(preprocess_for_detection(raw))``
          -- the array the detector actually measured on. Fitting levels on the raw
          channel instead would put every transition a filter rise time away from the
          edge the event table records (measured at 2.6% of frac_depth at the app
          default, further with a slower filter).
        * ``baseline.characterisation_baseline`` is THE door to the open-pore estimator,
          re-run with the detection's OWN recorded parameters, so a level depth and an
          event depth are differences from the same reference.

        THE FIT ITSELF STAYS DELIBERATE: running it is a menu click, re-running it is how you
        change the parameters, and a stale fit is dropped rather than silently refreshed (see
        :meth:`_invalidate_level_fit`).

        What IS remembered is what a re-fit of ONE event would need — the baseline track and
        the two estimators — in ``_level_fit_inputs``. That is not a cache of the answer; it
        is a cache of the QUESTION, and it exists so that a mask drag can put back the group
        the reconcile just retired (:meth:`_refit_levels_for_event`) using the same estimator
        the rest of the table was fitted with, without re-running a 644 ms open-pore estimate
        on the GUI thread. It is dropped with the fit it belongs to.
        """
        from copy import deepcopy
        from ..analysis.events import EventSet
        from .analysis_worker import FrozenChannelSource
        from .level_fit_job import run_level_fit, source_signature

        if self._analysis_controller.is_busy():
            self._on_run_refused("Another analysis is running. Wait or cancel it before fitting sub-state levels.")
            return
        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, "Fit sub-state levels",
                "Run detection on a current channel first — there are no events to fit.")
            return
        if res.module_id == CONSENSUS_MODULE_ID:
            QtWidgets.QMessageBox.information(
                self, "Fit sub-state levels", _RECONCILED_TRACE_ANALYSIS_REFUSAL)
            return
        events = res.payload
        prov = dict(events.provenance or {})
        accepted = events.accepted.df
        if accepted.empty:
            QtWidgets.QMessageBox.information(
                self, "Fit sub-state levels",
                "Every event in this detection is curated out, so there is nothing to fit.")
            return

        doc = self._session.document(res.doc_id)
        source = FrozenChannelSource.capture(
            doc, res.channel, self._session.calibration(res.doc_id, res.channel))
        fs = source.meta.sample_rate_hz
        if not fs:
            QtWidgets.QMessageBox.information(
                self, "Fit sub-state levels",
                "This recording has no known sample rate, so a level dwell time would be "
                "fictional. Set the rate first.")
            return

        # Copy the small mutable inputs; loaded samples are immutable and shared
        # through a read-only view. No whole-recording filter or calibration runs
        # on Qt here. The job sees no Session, window, or changing event frame.
        token = object()
        frame_snapshot = events.df.copy(deep=True)
        accepted_snapshot = accepted.copy(deep=True)
        provenance = deepcopy(prov)
        pending = {
            "token": token, "source": res, "document": doc,
            "frame": frame_snapshot, "provenance": source_signature(prov),
            "params": source_signature(dict(res.params or {})),
            "calibration": source.calibration,
        }

        def job(report):
            return run_level_fit(
                source, accepted_snapshot, provenance, doc_id=res.doc_id,
                channel=res.channel, token=token, report=report)

        if not self._analysis_controller.run_callable(
                job, label="fit.levels", wants_progress=True):
            return
        self._analysis_panel.forget_pending_job()
        self._level_fit_pending = pending
        self.statusBar().showMessage(
            "Fitting sub-state levels in the background. Cancel is safe; existing levels remain unchanged until the fit is validated.", 0)

    def _on_level_fit_stopped(self, *_args) -> None:
        self._level_fit_pending = None

    @QtCore.Slot(object)
    def _on_level_fit_finished(self, outcome: object) -> None:
        from .level_fit_job import LevelFitOutcome, source_signature
        if not isinstance(outcome, LevelFitOutcome):
            return
        pending, self._level_fit_pending = self._level_fit_pending, None
        if pending is None or outcome.token is not pending["token"]:
            return
        res = pending["source"]
        valid = self._session.has_document(res.doc_id)
        if valid:
            valid = (
                self._session.document(res.doc_id) is pending["document"]
                and self._session.calibration(res.doc_id, res.channel) == pending["calibration"]
                and self._session.results.get(res.doc_id, res.channel, res.module_id) is res
                and res.payload.df.equals(pending["frame"])
                and source_signature(dict(res.payload.provenance or {})) == pending["provenance"]
                and source_signature(dict(res.params or {})) == pending["params"]
            )
        if not valid:
            self.statusBar().showMessage(
                "Sub-state fit discarded: its recording, calibration or source event table changed while it ran. Existing analysis is unchanged; run the fit again.", 12000)
            return
        result = outcome.result
        self._level_fit = result
        self._level_fit_frame = res.payload.df
        self._level_fit_inputs = outcome.inputs
        self._session.add_result(result)
        # Both curation surfaces, not just the browser: the gallery draws the same levels
        # as bars on its tiles (see _push_level_fit).
        current = self._analysis_panel.current_detection()
        self._push_level_fit(result.payload if current is res else None)
        self._act_export_levels.setEnabled(not result.payload.empty)
        # ...and onto the MAIN TRACE, not just the inspector (owner note #18): the panel
        # reads the fit back out of the store and draws the STEP overlay. Explicit call
        # because resultsChanged only redraws for detection modules.
        self._analysis_panel.redraw_overlays()

        s = result.scalars
        self.statusBar().showMessage(
            f"Fitted {int(s['n_levels_total'])} level(s) across "
            f"{int(s['n_events_fitted'])} event(s); "
            f"{s['frac_multilevel']:.0%} are multi-level "
            f"(mean {s['mean_levels_per_event']:.2f} levels/event).", 10000)

    # ================================================================================================
    # Analysis ▸ Event subpopulations (GMM)… — the ported Zn2-insulin pipeline
    # ================================================================================================

    @property
    def _subpop_pending(self) -> bool:
        """Is a subpopulation report on its way? — DERIVED, never stored separately.

        The claim "a report is coming" and the fact "it is about THIS detection" are one
        fact, so they are one attribute (:attr:`_subpop_target`). They were two, and the
        landing slot then read the pending flag but re-read the SELECTION for the target:
        click another recording during the minute the mixture takes and one recording's
        event ids were installed on another's frame. Deriving the flag from the target
        makes that state unrepresentable — a pending run always knows what it is about,
        and clearing the claim clears the target with it.

        The claim itself is still needed because the analysis controller is SHARED with
        the AnalysisPanel's own jobs: the report's finished/failed/cancelled slots must be
        able to tell "mine" from "the panel's", and a payload type check alone would
        silently adopt whatever a future job happens to return.
        """
        return self._subpop_target is not None

    @_subpop_pending.setter
    def _subpop_pending(self, pending: bool) -> None:
        """``False`` drops the claim (and the target with it); ``True`` stamps the
        detection that is current right now, which is the launch target for every caller
        that does not know a more exact answer. :meth:`_run_subpopulations` DOES know one
        and writes :attr:`_subpop_target` itself."""
        self._subpop_target = (
            self._analysis_panel.current_detection() if pending else None)

    def _run_subpopulations(self) -> None:
        """Run the whole subpopulation pipeline on the current detection, off-thread.

        Everything the worker needs is resolved HERE, on the main thread, and handed
        over as plain arrays — the same seam :class:`AnalysisController` keeps for a
        module: the job never sees the Session.

        Three of those resolutions are load-bearing and none of them is a default:

        * **The trace** is ``modules.detection_signal``, not the raw channel. The
          thirteen features and the ≥4σ gate are statistics of the array the detector
          measured on; computing them on the raw channel would report a different
          measurement under the same column names, exactly as the CUSUM fit above
          would.
        * **The scalars** (``i0_pA`` and both baselines) are properties of the WHOLE
          recording, so the raw channel goes over too. Deriving them from the event
          rows instead — which is what a caller with only a table can do — silently
          produces a different feature table.
        * **The polarity** is negated. This app's ``+1`` means excursions run ABOVE the
          open-pore level; ``event_features``' ``+1`` means below. See
          :func:`~nanoscope.analysis.subpop_report.polarity_for_features`: feeding the
          app's sign through unchanged mirrors the trace, and on a genuinely ``-1``
          recording the result is a plausible-looking report of the baseline noise.

        ``events.accepted`` is what goes in, so the report and the screen are about the
        same rows. The gate then removes more of them, and says how many.
        """
        import numpy as np

        from ..analysis.events import EventSet
        from ..analysis.modules import detection_signal
        from ..analysis.subpop_report import (
            SubpopConfig,
            polarity_for_features,
            run_subpopulations,
            slow_gmm_note,
        )

        title = ANALYSIS_LABELS[PAGE_SUBPOPULATIONS]
        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, title,
                "Run detection on a current channel first — there are no events to group.")
            return
        if res.module_id == CONSENSUS_MODULE_ID:
            QtWidgets.QMessageBox.information(
                self, title, _RECONCILED_TRACE_ANALYSIS_REFUSAL)
            return
        events = res.payload
        prov = dict(events.provenance or {})
        accepted = events.accepted
        if not len(accepted.df):
            QtWidgets.QMessageBox.information(
                self, title,
                "Every event in this detection is curated out, so there is nothing to group.")
            return

        chan = self._session.calibrated(res.doc_id, res.channel)
        fs = chan.sample_rate_hz
        if not fs:
            QtWidgets.QMessageBox.information(
                self, title,
                "This recording has no known sample rate, so every dwell-based feature "
                "would be fictional. Set the rate first.")
            return

        raw = np.asarray(chan.data)
        try:
            signal = detection_signal(raw, float(fs), prov)
            source_recipe = result_recipe(self, res)
        except Exception as exc:  # noqa: BLE001 — reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, title,
                f"Could not prepare the detection and its persistent source identity:\n{exc}")
            return

        # The analysed duration, never the recorded one and never the last onset: the
        # rate column's denominator has to be the span the events were looked for in.
        duration = prov.get("analyzed_duration_s")
        duration = float(duration) if duration else None
        name = (self._session.document(res.doc_id).display_name
                if self._session.has_document(res.doc_id) else res.doc_id)
        config = SubpopConfig(polarity=polarity_for_features(prov.get("polarity")))

        # The gate runs INSIDE the job, so the exact mixture n is not known here;
        # the accepted count is its upper bound, which is the right side to warn
        # on. Asking first, rather than starting a run whose status bar promises a
        # minute and then takes half an hour.
        n_accepted = len(accepted.df)
        slow = slow_gmm_note(n_accepted)
        if slow and QtWidgets.QMessageBox.question(
                self, title,
                f"{n_accepted} accepted event(s) — at most that many reach the "
                f"mixture stage after the ≥4σ gate.\n\n{slow}\n\nStart it anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
        ) is not QtWidgets.QMessageBox.StandardButton.Yes:
            return

        def job(report):
            return run_subpopulations(
                signal, raw, float(fs), accepted,
                recording=str(name),
                duration_s=duration,
                config=config,
                progress=report,
            )

        # A pending job the panel is still holding would otherwise consume THIS result:
        # its ``_on_controller_finished`` clears ``_pending`` on the first thing that
        # finishes, whatever that is. ``abort()`` deliberately emits nothing, so the
        # panel cannot learn about the abort below on its own.
        self._analysis_panel.forget_pending_job()
        if not self._analysis_controller.run_callable(
                job, label="subpopulations", wants_progress=True):
            return          # refused by the run gate; it said so itself
        # THE DETECTION THE RUN WAS LAUNCHED ON — and, by writing it, the declaration that
        # a report is on its way (``_subpop_pending`` is derived from it; they are one
        # fact). A mixture takes a minute or many, the status line invites the user to keep
        # working, and nothing gates the File dock while it runs — so whatever is SELECTED
        # when the answer lands is not the question the report answers. Recorded the way
        # AnalysisPanel._on_controller_finished records ``_pending_sel``, for the same
        # stated reason ("Its answer belongs to that recording, not to the current one").
        # ``res``, not ``current_detection()`` again: the slow-run confirmation above is a
        # modal that spins the event loop, and the job was built from THIS result.
        self._subpop_target = res
        self._subpop_source_recipe_pending = source_recipe
        self._subpop_signal_pending = signal
        # THE ROWS THE JOB WAS HANDED, as a set of ids and HERE — the same instant, and the
        # same object, ``report.n_fired`` will count. ``accepted`` is a ``.copy()`` (see
        # ``EventSet.accepted``), so this is a snapshot and not a view of a frame curation
        # goes on editing in place while the mixture runs.
        _launched_on = accepted.df
        self._subpop_accepted_ids_pending = (
            frozenset(int(e) for e in _launched_on["event_id"])
            if "event_id" in _launched_on.columns else None)
        self._set_progress_visible(True)
        self.statusBar().showMessage(
            f"Grouping {n_accepted} accepted event(s) into subpopulations — "
            + ("a cross-validated mixture search takes about a minute."
               if not slow else
               "the cross-validated mixture search will take many minutes at this "
               "event count; Stop is safe."), 0)

    @QtCore.Slot(object)
    def _on_subpop_finished(self, payload: object) -> None:
        """Install a finished subpopulation report (main thread) and show it.

        Gated on BOTH the pending flag and the payload type. The controller is shared
        with the AnalysisPanel's own callable jobs, and a slot that adopted anything
        arriving on ``finished`` would open a window over somebody else's answer.

        THE STATUS BAR IS NOT A SECOND AUTHOR OF THE CLAIM. It used to build its own
        sentence out of ``quantification.fit.k`` and render, on the owner's -700 mV apo
        run, *"Subpopulations: 985 of 986 region(s) passed the gate; 7 subpopulation(s)
        at n_PCs=11; 10 caveat(s)."* — a class count with no statistic anywhere near it,
        over a partition the report itself calls NO SUBSTANTIAL STRUCTURE. That was the
        second of the two strings a reader carries away, and the report window's header
        was only the first. Both now repeat :meth:`SubpopReport.headline` verbatim, which
        is the single author of the claim, welded in the analysis layer to the separation
        statistic and the continuum verdict it has to be honest about. This surface adds
        the label and the timeout and nothing else — so a future change to HOW the claim
        is qualified reaches the status bar without anyone remembering to come here.
        """
        from ..analysis.subpop_report import SubpopReport

        if not self._subpop_pending or not isinstance(payload, SubpopReport):
            return
        # Consume the claim AND the target in one move: everything below is about the
        # detection this run was launched on, and nothing may re-derive it from the
        # selection (see :meth:`_adopt_subpop_assignment`).
        target, self._subpop_target = self._subpop_target, None
        source_recipe, self._subpop_source_recipe_pending = (
            self._subpop_source_recipe_pending, None)
        # Keep the launch-time accepted set independently of whether live colouring can
        # still attach.  The recording/result may have closed or been replaced while the
        # worker ran, but the self-contained report still needs to remember exactly which
        # rows it analysed (AP-07/AP-16).
        launch_accepted_ids = self._subpop_accepted_ids_pending
        self._set_progress_visible(False)
        note = self._adopt_subpop_assignment(payload, target)
        try:
            self._open_subpop_window(
                payload, source_result=target, source_recipe=source_recipe,
                accepted_event_ids=launch_accepted_ids)
        finally:
            # Successful adoption normally consumes both fields.  Refused adoption used
            # to retain the full conditioned trace (potentially hundreds of MB) and the
            # accepted-id set indefinitely; the report has now captured the durable part.
            self._subpop_signal_pending = None
            self._subpop_accepted_ids_pending = None
        # The headline is still the SINGLE author of what the run found (above). ``note`` is
        # a different fact — whether the report's per-event colouring reached the trace and
        # the gallery at all — and it is said HERE because this is where the run's one
        # sentence is written; a colouring that quietly did not happen is exactly the kind
        # of absence the user would otherwise read as "the mixture found nothing to colour".
        self.statusBar().showMessage(
            f"Subpopulations: {payload.headline()}" + (f"  {note}" if note else ""), 12000)

    @QtCore.Slot()
    def _on_subpop_stopped(self) -> None:
        """Clear the pending flag on a failure or a cancel.

        The failure BOX is MainWindow's existing ``_on_analysis_failed``; this only
        drops the claim that a report is on its way, so the next job's answer cannot
        be adopted by :meth:`_on_subpop_finished`."""
        self._subpop_pending = False
        self._subpop_source_recipe_pending = None
        self._subpop_signal_pending = None    # a full-length array; do not outlive the run
        self._subpop_accepted_ids_pending = None   # ...and neither may the run's own set

    def _ensure_subpop_window(self):
        """Build the report view without installing a report or showing a window."""
        from .subpop_window import SubpopReportWindow

        if self._subpop_window is None:
            win = SubpopReportWindow(self)
            win.exportRequested.connect(self._export_subpop_report)
            win.analyzeRequested.connect(self._run_subpopulations)
            # The report window announces "show me these events"; the routing is here,
            # because it is the only object holding the report AND both surfaces.
            win.galleryRequested.connect(self._open_gallery_on_subpops)
            win.traceRequested.connect(self._colour_trace_by_subpops)
            self._subpop_window = win
            self._watch_surface_window(win)
            self._watch_workspace_surface(SUBPOPULATION_ARTIFACT_ID, win)
        return self._subpop_window

    def _open_subpop_window(
            self, report, *, source_result=None, source_recipe=None,
            accepted_event_ids=None) -> None:
        """Open a newly completed report and checkpoint it as a durable result."""
        self._workspace_pending.discard(SUBPOPULATION_ARTIFACT_ID)
        self._workspace_frozen.discard(SUBPOPULATION_ARTIFACT_ID)
        self._ensure_subpop_window()
        # The adapter turns this live result into a persistent source/result recipe.  It
        # never serializes the object itself; keeping it on the widget only bridges the
        # synchronous capture immediately below.
        self._subpop_window._nanoscope_workspace_source_result = source_result
        self._subpop_window._nanoscope_workspace_source = source_recipe
        self._subpop_window._nanoscope_workspace_accepted_event_ids = (
            accepted_event_ids)
        self._subpop_window.set_report(
            report, subtitle=f"{report.meta.get('recording') or '(unnamed)'}")
        # ``set_report`` resets the window's live state (a new document, no claim about
        # any assignment); the truth about THIS report's assignment is this window's to
        # push, and it is pushed right away so the two entry buttons never open live over
        # a colouring that was refused or has already gone (S7-POP3).
        self._sync_subpop_window()
        self._on_workspace_changed(SUBPOPULATION_ARTIFACT_ID)
        self._show_analysis_page(
            PAGE_SUBPOPULATIONS, self._subpop_window,
            refresh=self._run_subpopulations)

    def _show_saved_subpop_report(self) -> None:
        """Show the project-owned report snapshot without dispatching an analysis job."""
        if not self._workspace_has_artifact(SUBPOPULATION_ARTIFACT_ID):
            self.statusBar().showMessage(
                "This project does not contain a saved subpopulation report.", 8000)
            return
        if _on_screen(self._subpop_window):
            self._show_analysis_page(
                PAGE_SUBPOPULATIONS, self._subpop_window,
                refresh=self._run_subpopulations)
            return
        win = self._ensure_subpop_window()
        # A window destroyed/reset during a session transition must install from the cache;
        # a hidden one is marked pending by eventFilter.  The explicit add here also covers
        # tests/support callers which adopted a workspace without driving a Hide event.
        self._workspace_pending.add(SUBPOPULATION_ARTIFACT_ID)
        restored = self._restore_workspace_surface(SUBPOPULATION_ARTIFACT_ID, win)
        if restored and win.report() is not None:
            artifact = self._workspace_artifact(SUBPOPULATION_ARTIFACT_ID)
            recipe = artifact_source_recipe(artifact) if artifact is not None else None
            target, warning = resolve_result_recipe(self, recipe or {})
            if target is None:
                self._retire_subpop_assignment(
                    "This saved report remains viewable, but its event colouring was "
                    f"not attached: {warning or 'its source-result lineage is unavailable'}. "
                    "Re-run Analysis ▸ Generate Subpopulations to create a current assignment.")
            else:
                # The adapter keeps the accepted ids beside the report.  Feed them through
                # the existing one-writer adoption path so the live curation note compares
                # against the exact rows the saved run received, not today's accepted count.
                saved_ids = getattr(
                    win, "_nanoscope_workspace_accepted_event_ids", None)
                self._subpop_accepted_ids_pending = (
                    None if saved_ids is None
                    else frozenset(int(value) for value in saved_ids))
                self._subpop_signal_pending = None  # raw/detrended recordings are not embedded
                win._nanoscope_workspace_source_result = target
                note = self._adopt_subpop_assignment(
                    win.report(), target, require_current_selection=False)
                self._subpop_accepted_ids_pending = None
                if note:
                    self.statusBar().showMessage(note, 10000)
                elif warning:
                    self.statusBar().showMessage(
                        f"Saved subpopulation assignment restored with a caveat: {warning}",
                        12000)
                self._sync_subpop_window()
        self._show_analysis_page(
            PAGE_SUBPOPULATIONS, win, refresh=self._run_subpopulations)

    # -- item 13: the report's events, on the trace and in the gallery --------------------------
    def _adopt_subpop_assignment(
            self, report, target, *, require_current_selection: bool = True) -> str:
        """Read the per-event subpopulation ids off a finished report and push them out.

        ``target`` is the detection the run was LAUNCHED on, handed in by the landing slot
        rather than looked up here — see below.

        THE REPORT IS READ ONCE, HERE, AND NOTHING IS RECOMPUTED. ``report.event_table()`` is
        the report's own join of its feature table with the mixture's labels -- the same table
        the export writes -- so the ids the trace and the gallery colour by are, by
        construction, the ids the report printed. Deriving them here from
        ``quantification.fit.labels`` and ``features['event_id']`` would be a second
        implementation of that join living in a window.

        Absent when the mixture was refused: ``event_table`` has no ``subpop`` column at all,
        which is exactly the case the report prints as "NO MIXTURE". The colouring is then
        dropped rather than faked from the exploratory labels -- HDBSCAN's clusters are a
        different question with a different answer (on zn_30h, two clusters and 229 noise
        points where the mixture reports four classes) and are not what "subpop 2" means.
        """
        import numpy as np
        import pandas as pd

        from ..analysis.events import EventSet

        self._drop_subpop_assignment()
        table = report.event_table()
        if "subpop" not in table.columns or "event_id" not in table.columns:
            return ""      # "NO MIXTURE" — the headline already says it; nothing to add
        ids = table["event_id"].to_numpy()
        labels = table["subpop"].to_numpy()
        assignment = {int(e): int(v) for e, v in zip(ids, labels, strict=True)
                      if not pd.isna(v)}
        if not assignment:
            return ""
        # *** THE MAP IS ADOPTED ONTO THE DETECTION IT WAS COMPUTED FROM, OR NOT AT ALL. ***
        #
        # This used to re-read `current_detection()` — whatever happened to be SELECTED when
        # the report landed — and install the report's event ids onto that frame. Click
        # another recording in the File dock during the minute the mixture takes and 120
        # event ids computed on A were attached to B's 40-row frame: 40 of B's events cut
        # into "subpop 0" / "subpop 1" / "not in the report" by event-id NUMBER collision,
        # while the gallery caption and the trace status line still named A. A fabricated
        # per-event class assignment on a recording that was never analysed, attributed in
        # writing to a different file. `_refresh_subpop_views` could not clean it up either,
        # because the frame it compares against had already been overwritten with B's.
        #
        # `_subpop_target` is the detection the run was LAUNCHED on (see
        # :meth:`_run_subpopulations`). It is still valid only if it is still the stored
        # result for its recording — a re-detect or a merge replaces the object, and then
        # the ids describe events that no longer exist under those ids. Anything else and
        # the colouring is dropped WITH A WORD, never silently re-homed.
        if target is None or not isinstance(target.payload, EventSet):
            return ""
        # Every refusal below is ALSO recorded as the reason there is no assignment, so
        # the report window's two buttons can repeat it instead of the only other sentence
        # they used to know ("fitted no mixture" — false for every one of these).
        if not self._session.has_document(target.doc_id):
            self._subpop_absent_reason = (
                f"Nothing was coloured: “{Path(source_of(target.doc_id)).name}” was "
                f"closed while the run was in flight.")
            return self._subpop_absent_reason
        live = self._session.results.get(
            target.doc_id, target.channel, target.module_id)
        if live is not target:
            self._subpop_absent_reason = (
                "Nothing was coloured: the detection this was computed from has been "
                "replaced since the run started, so its event ids no longer name the "
                "same events.")
            return self._subpop_absent_reason
        # ...and the colouring only goes ON SCREEN if the screen is showing that detection.
        # The trace spans and the gallery tiles colour by matching the CURRENT frame's
        # event ids against this map, so pushing it while another recording is selected is
        # the same id-collision, one step later. Same rule the sibling surface keeps.
        if (require_current_selection
                and self._analysis_panel.current_detection() is not target):
            self._subpop_absent_reason = (
                f"Nothing was coloured on the trace or in the gallery: you moved off "
                f"“{Path(source_of(target.doc_id)).name}” while the run was in flight. "
                f"Select it again and re-run to colour its events.")
            return self._subpop_absent_reason
        res = target
        self._subpop_assignment = assignment
        self._subpop_frame = res.payload.df
        # The store key the frame lives under, so a later navigation can ask "is the
        # detection this was fitted on still the stored one?" without re-deriving it from
        # whatever is selected then (see _refresh_subpop_views).
        self._subpop_key = (str(res.doc_id), int(res.channel), str(res.module_id))
        self._subpop_n_fired = int(report.n_fired)
        # ...and the accepted SET behind that count, so the curation note can compare sets
        # and not counts (C2). CONSUMED FROM THE LAUNCH, never rebuilt from the frame here:
        # the frame is edited in place by curation and this is minutes later, so a rebuild
        # would snapshot the rows as they are NOW and call them the rows the run was handed
        # — which is how an in-flight curation used to silence the very line that exists to
        # disclose it (measured: n_fired 110, 109 accepted, note "").
        #
        # And it is installed only if it can be SQUARED with the count the report printed.
        # The two halves are one claim; a set whose size is not ``n_fired`` does not
        # describe the rows the report describes, whatever it does describe, so it is
        # dropped and the note falls back to the count — which still fires on that
        # disagreement. Never a set the line would compare against and read equal.
        _run_ids = self._subpop_accepted_ids_pending
        self._subpop_accepted_ids_pending = None
        self._subpop_accepted_ids = (
            _run_ids if _run_ids is not None and len(_run_ids) == int(report.n_fired)
            else None)
        self._subpop_source = str(report.meta.get("recording", "") or "")
        self._subpop_caveats = tuple(report.warnings)
        # THE REPORT IS READ ONCE, HERE (see above), and its own sentence about what it
        # found is part of what is read: ``_colour_trace_by_subpops`` speaks about this
        # assignment long after the report object has gone, and without this it had no
        # way to say what the classes are worth except by counting them itself — which
        # is exactly the claim ``headline()`` exists to be the only author of.
        self._subpop_headline = report.headline()
        self._subpop_signal = self._subpop_signal_pending
        self._subpop_signal_pending = None
        if self._subpop_signal is not None:
            # A trace of the wrong length would slice tiles at the wrong samples. It cannot
            # normally happen (the run and the channel are the same recording), so it is
            # dropped rather than clipped: a silently shifted picture is worse than none.
            # The same for a channel that has gone away under us -- the assignment is still
            # good, only the alternative trace is not, and losing the whole colouring
            # because a cosmetic extra could not be validated would be the wrong trade.
            try:
                n_channel = len(self._session.calibrated(res.doc_id, res.channel).data)
            except Exception:  # noqa: BLE001 -- cosmetic; never breaks the adoption
                n_channel = -1
            if np.asarray(self._subpop_signal).size != n_channel:
                self._subpop_signal = None
        self._push_subpop_assignment()
        return ""

    def _subpop_installed(self) -> bool:
        """Is the held assignment about the detection ON SCREEN? — frame identity, the
        one test the sibling surfaces already use to tell "the frame was edited" from
        "there is a new frame". False while another recording (or another detector's
        result) is selected; the map is then HELD, not dropped."""
        from ..analysis.events import EventSet

        if self._subpop_assignment is None:
            return False
        res = self._analysis_panel.current_detection()
        df = res.payload.df if (res is not None and isinstance(res.payload, EventSet)) else None
        return df is not None and df is self._subpop_frame

    def _push_subpop_assignment(self) -> None:
        """Hand the assignment to the trace spans and the gallery — or its ABSENCE, which
        is what they get whenever the detection on screen is not the one it was fitted
        on. Both surfaces colour by matching the CURRENT frame's event ids against the
        map, so a held map pushed onto another recording's frame is the id-collision
        `_adopt_subpop_assignment` refuses, one step later. Idempotent on both surfaces."""
        from .event_gallery import TRACE_DETRENDED

        shown = self._subpop_assignment if self._subpop_installed() else None
        if self._analysis_panel.subpop_labels() != shown:
            self._analysis_panel.set_subpop_labels(shown)
        if self._event_gallery is not None:
            gallery = self._event_gallery
            # The detrended trace is offered with the map and withdrawn with it; offering
            # it again re-applies the tile trace, so only the transitions are pushed.
            signal = self._subpop_signal if shown is not None else None
            offered = gallery.trace_box.findText(TRACE_DETRENDED) >= 0
            had_map = gallery.subpopulations() is not None
            # C4: the map is leaving this gallery — take the user's view of it with it,
            # while the widgets still carry it. A push is only the SECOND place the view
            # can be lost (the re-feed in ``_refresh_event_gallery`` is the first and
            # usually gets there earlier), so the remembering is one method used by both
            # and it refuses to read a gallery that has already been reset.
            if shown is None and had_map:
                self._remember_gallery_subpop_view()
            if (signal is not None) != offered:
                gallery.set_detrended_signal(signal)
            if gallery.subpopulations() != shown:
                gallery.set_subpopulations(
                    shown,
                    source=self._subpop_source if shown is not None else "",
                    caveats=self._subpop_caveats if shown is not None else ())
            # C4: and put that view back the moment the map arrives on a gallery that did
            # not have it — the return leg of the round trip. A fresh adoption remembers
            # nothing, so the report window's "Event gallery…" button keeps choosing the
            # trace and the sample itself (see :meth:`_open_gallery_on_subpops`).
            if shown is not None and not had_map:
                self._restore_gallery_subpop_view()

    def _remember_gallery_subpop_view(self) -> None:
        """Remember what the Event gallery is DRAWING the held map with, for the return.

        The one writer of :attr:`_subpop_gallery_trace` /
        :attr:`_subpop_gallery_representative`. Both are the CURATOR'S choices — the report
        window's "Event gallery…" turns on the detrended trace and the representative
        sample, and the curator may change either — and leaving the recording resets them:
        ``set_events`` drops the map (and with it the representative sample) and
        ``set_detrended_signal(None)`` falls the trace box back to "Calibrated (as
        recorded)". Measured on the owner's project before this existed: away to 0005 and
        back to 0006 returned the chips and the classes with the tiles drawn from the
        calibrated trace and the sample off, and the status line stated the new state as
        if it had always been so (C4).

        READ ONLY WHILE THE GALLERY STILL CARRIES THE MAP. The re-feed runs first
        (``_refresh_event_gallery`` before ``_refresh_subpop_views``) and has already put
        the representative box back to False by the time the push withdraws, so a capture
        taken then would remember exactly the reset it exists to undo — measured: the
        trace was remembered correctly and the sample came back off.
        """
        gallery = self._event_gallery
        if (gallery is None or self._subpop_assignment is None
                or gallery.subpopulations() is None):
            return
        self._subpop_gallery_trace = gallery.trace_name()
        self._subpop_gallery_representative = gallery.is_representative()

    def _restore_gallery_subpop_view(self) -> None:
        """Put the remembered trace + representative sample back (see the method above).

        Idempotent and defensive: a trace the gallery is not offering is not forced, and
        neither widget is touched when it already reads the remembered value, so the
        restore cannot bounce the tiles a second time.
        """
        gallery = self._event_gallery
        if gallery is None:
            return
        name = self._subpop_gallery_trace
        if name and gallery.trace_name() != name and gallery.trace_box.findText(name) >= 0:
            gallery.set_trace(name)
        if self._subpop_gallery_representative and not gallery.is_representative():
            gallery.set_representative(True)

    def _drop_subpop_assignment(self) -> None:
        """Forget the assignment everywhere. Idempotent."""
        self._subpop_assignment = None
        self._subpop_frame = None
        self._subpop_key = None
        self._subpop_n_fired = None
        self._subpop_accepted_ids = None
        self._subpop_gallery_trace = None
        self._subpop_gallery_representative = False
        self._subpop_absent_reason = ""
        self._subpop_source = ""
        self._subpop_caveats = ()
        self._subpop_headline = ""
        self._subpop_signal = None
        self._push_subpop_assignment()

    def _retire_subpop_assignment(self, why: str) -> None:
        """Drop the assignment because the detection it describes is gone, and SAY SO —
        on the status bar now, and in the report window for as long as that report
        stands (its two buttons then repeat this reason; see :meth:`_subpop_state`)."""
        self._drop_subpop_assignment()
        self._subpop_absent_reason = why
        self._sync_subpop_window()
        self.statusBar().showMessage(why, 10000)

    def _refresh_subpop_views(self) -> None:
        """Keep the colouring for as long as the detection it describes still stands.

        Reached from :data:`SURFACE_MUTATION_SIGNALS` and from the retarget, i.e. from every
        edit and every navigation. Curation, labelling and a mask drag all write the SAME
        frame in place, so the map survives them and the colours track the edit. A re-detect,
        a merge, a re-measure or a project open REPLACES the stored frame, and then the map
        is about events that no longer exist under those ids -- so it goes, WITH ITS REASON,
        rather than colouring one detection with another's mixture.

        TWO THINGS THIS DOES NOT RETIRE, both measured as false retirements in sweep 7's
        fix batch. (1) An Edit ▸ Undo/Redo of a plain CURATION: it installs a snapshot copy
        of the same events with one accept flag moved, which the forward edit had already
        left the map installed through, so the map is re-bound to the copy (X6; see
        :meth:`_is_curation_only_change`). (2) Another DETECTOR's run on this channel: it
        is filed under its own module id and never reaches this frame, so nothing here
        fires at all and the map is simply not on screen -- :meth:`_subpop_state` calls
        that "away" and says which detector is showing instead (C1).

        *** NAVIGATION IS NOT A STATEMENT ABOUT THE MIXTURE. *** This used to compare the
        held frame against whatever detection was on screen and drop the map on any
        mismatch -- and nulled the frame anchor with it, so the "same frame" branch below
        could never fire again. Sweep 7 measured the cost on the owner's own project:
        click 0005's row, click straight back to 0006, and a finished 236-event mixture
        was gone -- assignment None, gallery combo empty, both report buttons still live
        and giving a false reason -- although the frame that came back was the SAME
        OBJECT (``k3_frame_identity``: id 13820809488 before and after). The question is
        now asked of the ResultStore, under the key the map was adopted on: is the stored
        detection for (doc, channel, module) still the frame this map describes? While
        it is, the map is held -- pushed onto the surfaces when that frame is on screen,
        withdrawn from them (not forgotten) while another recording is -- exactly as the
        CUSUM level fit is retired only when ITS detection is replaced and re-adopted
        "after switching away and back" (see _sync_export_action).
        """
        from ..analysis.events import EventSet

        if self._subpop_assignment is None:
            return
        key = self._subpop_key
        if key is None or not self._session.has_document(key[0]):
            self._retire_subpop_assignment(
                "This report's assignment was retired when the recording it was fitted "
                "on was closed — re-run Analysis ▸ Generate Subpopulations on the recording "
                "you want coloured.")
            return
        stored = self._session.results.get(*key)
        df = (stored.payload.df
              if stored is not None and isinstance(stored.payload, EventSet) else None)
        if df is not self._subpop_frame:
            # X6: an Edit ▸ Undo/Redo of a plain curation installs a SNAPSHOT COPY of the
            # frame (core.history.EventHistory._move), a new object with the same event
            # ids, the same spans and the same depths — only ``accepted``/``flag`` moved.
            # Frame identity breaks, but the map is event_id → class and is exactly as
            # valid on that copy as on the frame it survived the forward curation on, so
            # it is RE-BOUND to the new object rather than retired. ``describe_change`` is
            # the same author the Edit menu uses to name the step ("Undo Curate event"),
            # so "a curation was undone" and "the map may stay" come from one place and
            # cannot drift. Anything it does not NAME a curation — a merge, a re-detect
            # (even one that returned the same table), a mask resize, a re-measure — still
            # retires the map: those change what the ids mean, and colouring one detection
            # with another's mixture is the hazard this guard exists for.
            if df is not None and self._is_curation_only_change(self._subpop_frame, df):
                self._subpop_frame = df
            else:
                # The wording names what actually retires the map, and nothing else (C1):
                # curation and its undo do not (above), and neither does another
                # DETECTOR's run on this channel — that result is filed under its own
                # module id, leaves this one's frame alone, and parks the map "away"
                # (see :meth:`_subpop_state`). An ML run reaches this line only when the
                # map was fitted on the ML detector's own table and a new one replaced it,
                # which is the same fact as "re-detected".
                self._retire_subpop_assignment(
                    "This report's assignment was retired when the detection changed (a "
                    "re-detect, a merge, a mask edit, a re-measure, or an undo past the "
                    "detection) — re-run Analysis ▸ Generate Subpopulations to colour the "
                    "events on screen.")
                return
        # The detection stands. On screen: offer the map (the gallery may simply have
        # been opened since). Off screen: withdraw it from the surfaces and keep it.
        self._push_subpop_assignment()
        self._sync_subpop_window()

    @staticmethod
    def _is_curation_only_change(before, after) -> bool:
        """Did ``after`` differ from ``before`` by CURATION and nothing else?

        The test :meth:`_refresh_subpop_views` uses to keep a subpopulation map across an
        Edit ▸ Undo/Redo of a plain curation: the undo installs a fresh snapshot frame, so
        frame identity breaks, but the event ids, spans and depths are unchanged and only
        ``accepted``/``flag`` moved — the map is still about exactly those events, and the
        map survived the forward edit by design. Answered by ``describe_change``, the app's
        own name for the step ("Undo Curate event" is the Edit menu's own label for the
        gesture), so "a curation was undone" and "the map may stay" come from one place and
        cannot drift.

        NAMED CURATION ONLY. ``describe_change``'s no-change fallback ("Edit events") is
        NOT exempted, deliberately: a detector that re-ran and happened to return the same
        table is a new detection, and the wave-1 rule — a stored result the user replaced
        retires the map — is the one an identical re-detect must keep obeying. The gate is
        "this frame differs from the anchored one only in the curator's own columns", not
        "these frames look alike".
        """
        from ..core.history import describe_change

        label = describe_change(before, after)
        return (label.startswith("Curate ")
                or label in ("Accept all events", "Reject all events"))

    #: Short human name for each detector, for the report window's "held" sentence when a
    #: DIFFERENT detector's result is on screen. The three the app runs; anything else
    #: reads back its own module id rather than a guess.
    _DETECTOR_DISPLAY_NAMES = {
        "detect.schmitt": "Schmitt",
        "detect.cusum": "CUSUM",
        "detect.keras": "ML",
        CONSENSUS_MODULE_ID: "Reconciled",
    }

    @classmethod
    def _detector_display_name(cls, module_id) -> str:
        """The short name a "held" sentence names a detector by (see :meth:`_subpop_state`)."""
        return cls._DETECTOR_DISPLAY_NAMES.get(str(module_id), str(module_id))

    #: The four things the report window can be told about its assignment. Spelled here
    #: as the window spells them (``subpop_window.ASSIGNMENT_*``; a test pins the two
    #: spellings equal) because this file imports that window lazily, on first use.
    _SUBPOP_NONE, _SUBPOP_INSTALLED, _SUBPOP_AWAY, _SUBPOP_RETIRED = (
        "none", "installed", "away", "retired")

    def _subpop_state(self) -> tuple[str, str]:
        """``(state, reason)`` — the ONE resolver of what the report's assignment is doing,
        read by the two report-window buttons, by their gate and by the window's standing
        line, so they cannot disagree about it.

        ``installed``: the map is on the trace and the gallery (reason ""). ``away``: the
        map is held but the detection on screen is another one. ``retired``: the map is
        gone for the reason given (the detection changed, the recording closed, or the
        adoption was refused). ``none``: the report fitted no mixture, so there was never
        anything to colour with — the one case the old sentence was true for.
        """
        if self._subpop_assignment is not None:
            if self._subpop_installed():
                return self._SUBPOP_INSTALLED, ""
            key = self._subpop_key
            name = self._subpop_source or (
                Path(source_of(key[0])).name if key is not None else "its recording")
            where = f"“{name}”" + (f" (channel {key[1]})" if key is not None else "")
            # C1/C-SUB-2: there are TWO ways a map is held-but-not-shown, and they need
            # different words. When the recording+channel it was fitted on IS the one on
            # screen, only the DETECTOR differs: its result was fitted on (say) Schmitt and
            # a CUSUM or ML run is now active on the same channel. The store keeps one
            # result per detector, so this map's own frame is untouched and nothing was
            # retired — but telling the user to "select that recording again in the File
            # Manager" is a dead end, because the recording is already selected and
            # re-selecting shows the same other-detector result. That case gets its own
            # sentence, naming the recovery that actually works.
            res = self._analysis_panel.current_detection()
            on_same_target = (
                key is not None and res is not None
                and str(res.doc_id) == key[0] and int(res.channel) == key[1])
            if on_same_target:
                return self._SUBPOP_AWAY, (
                    f"This report describes the {self._detector_display_name(key[2])} "
                    f"detection of {where}; the "
                    f"{self._detector_display_name(res.module_id)} detection is on screen "
                    f"now, so its colouring is held, not shown. Re-run Analysis ▸ Generate "
                    f"Subpopulations to colour the detection on screen; nothing was retired.")
            # A DIFFERENT recording is on screen: the map comes back the moment its own
            # recording is selected again. The dock the user picks it in is titled "File
            # Manager" (X-SURF-12: there is no widget called "Files").
            return self._SUBPOP_AWAY, (
                f"This report describes {where}, which is not the detection on screen — "
                f"its colouring is held, not shown, until you select that recording again "
                f"in the File Manager; nothing was retired.")
        if self._subpop_absent_reason:
            return self._SUBPOP_RETIRED, self._subpop_absent_reason
        return self._SUBPOP_NONE, (
            "That report fitted no mixture, so there are no subpopulations to colour "
            "the trace with — see the report text.")

    def _subpop_curation_note(self) -> str:
        """S7-POP4: the standing sentence for the report window when the ACCEPTED SET has
        moved under a standing report, "" while it has not.

        The map survives curation by design (the frame is edited in place; see
        :meth:`_refresh_subpop_views`), and the report's fractions, rates and CIs stay
        the run's — measured on the owner's 0006: reject 66 of class 0 and the header,
        the population table and the gallery combo all stood byte-identical while a
        re-run of the same analysis returned three classes instead of two. Nothing here
        recomputes anything; it states which set the numbers describe and how far the
        accepted set is from it now.

        The comparison is of the accepted SET, not its size (C2): rejecting one event and
        accepting another back leaves the count equal while the set is no longer the run's
        (measured on 0004: event 3 accepted now was not in the run, event 7 in the run is
        rejected now), and a count read equal on exactly that swap and cleared the one line
        that exists to catch it. The COUNT is still the fallback — for a frame with no
        ``event_id`` column, and for a run whose set was never captured — because a line
        that is only sometimes right is still better here than one that is silently absent.

        BOTH HALVES ARE THE LAUNCH'S. ``_subpop_accepted_ids`` is snapshotted in
        :meth:`_run_subpopulations` off the ``EventSet`` the job is handed, which is the
        instant ``n_fired`` counts; it was rebuilt from the live frame at adoption, and
        because a mixture takes a minute or many and the app invites the user to keep
        curating while it runs, an in-flight rejection landed inside the snapshot and the
        line compared the current set with itself — measured: the report says 110 accepted,
        109 are, and this returned "" (C2 close).
        """
        frame = self._subpop_frame
        n_run = self._subpop_n_fired
        run_ids = self._subpop_accepted_ids
        if frame is None or n_run is None or "accepted" not in frame.columns:
            return ""
        acc = frame["accepted"].to_numpy().astype(bool)
        if "event_id" in frame.columns and run_ids is not None:
            now_ids = frozenset(int(e) for e in frame.loc[acc, "event_id"])
            n_now, unchanged = len(now_ids), now_ids == run_ids
        else:
            n_now = int(acc.sum())
            unchanged = n_now == n_run
        if unchanged:
            return ""
        return (f"accepted now: {n_now} of {len(frame)}; this run's fractions describe "
                f"the {n_run} accepted at run time — re-run to update")

    def _sync_subpop_window(self) -> None:
        """Push the assignment's state + reason and the curation note onto the report
        window, which gates its two entry buttons on the state and prints both lines."""
        win = self._subpop_window
        if win is None:
            return
        state, reason = self._subpop_state()
        win.set_assignment_state(state, reason)
        win.set_curation_note(self._subpop_curation_note())

    def _open_gallery_on_subpops(self) -> None:
        """Report window ▸ "Event gallery…" — open the gallery already carrying the report.

        Only when the assignment is INSTALLED. Any other state posts the true reason and
        opens nothing: a plain gallery with no chips and no sample, opened from a button
        that promised the report's classes, is the silent failure S7-POP3 measured.
        """
        from .event_gallery import TRACE_DETRENDED

        state, reason = self._subpop_state()
        if state != self._SUBPOP_INSTALLED:
            self.statusBar().showMessage(reason, 10000)
            return
        self._open_event_gallery()
        self._push_subpop_assignment()
        if self._subpop_signal is not None:
            self._event_gallery.set_trace(TRACE_DETRENDED)
        self._event_gallery.set_representative(True)

    def _colour_trace_by_subpops(self) -> None:
        """Report window ▸ "Colour the trace" — shade the spans by class and come forward.

        The refusal names the TRUE state (see :meth:`_subpop_state`). It used to print
        "That report fitted no mixture" for every case where the map was absent — after
        a merge, on the recording the report was fitted on, two lines under a header
        saying "1 subpopulation(s)" (S7-POP3).
        """
        state, reason = self._subpop_state()
        if state != self._SUBPOP_INSTALLED:
            self.statusBar().showMessage(reason, 10000)
            return
        self._push_subpop_assignment()
        self.raise_()
        self.activateWindow()
        # The class count is NOT recomputed here. ``{v for v in ... if v >= 0}`` was a
        # second author of the same claim — it printed "coloured by 7 subpopulation(s)"
        # with nothing beside it, on the same run the report calls NO SUBSTANTIAL
        # STRUCTURE. What the run found is said once, by the report, and repeated:
        # this surface only says what has been coloured.
        # The headline goes LAST, and nothing is appended after it. It is another module's
        # sentence and this surface must not assume how it ends — an embedded quotation
        # needed a "." after it, which would read as ".." the day ``headline()`` acquires
        # its own full stop. Found in this fix's own verify pass.
        self.statusBar().showMessage(
            f"Trace spans coloured by subpopulation for "
            f"{len(self._subpop_assignment)} gated event(s); every other detected event keeps "
            f"the grey that means 'in no class'. Zoom in to read the names. The assignment is "
            f"the mixture's and is provisional — the report's caveats say why. "
            f"What the report found: {self._subpop_headline}", 15000)

    def _export_subpop_report(self) -> None:
        """The report window's Export… — the text, the sidecar and every table.

        Several files, so the confirmation covers all of them: the report text is the
        document, the ``.meta.json`` carries the settings and EVERY caveat (a CSV of
        fractions with the label-switching warning stripped off is precisely the
        artefact this pipeline must not emit), and one CSV per table.

        X5: the window shows a standing sentence when the accepted set has moved under this
        report (S7-POP4), and the file is written to be read WITHOUT the app beside it — so
        that same disclosure has to travel into it. It is appended as one more caveat, which
        is exactly where the text's CAVEATS block and the ``.meta.json`` ``warnings`` list
        already carry the reasons the fractions may not describe what is accepted now. The
        numbers are untouched: only the caveat list grows.
        """
        import dataclasses

        from ..analysis.subpop_report import report_paths, write_report

        title = "Export subpopulation report"
        report = None if self._subpop_window is None else self._subpop_window.report()
        if report is None:
            QtWidgets.QMessageBox.information(
                self, title, "There is no subpopulation report to export yet.")
            return
        curation_note = self._subpop_curation_note()
        if curation_note:
            report = dataclasses.replace(
                report,
                warnings=tuple(report.warnings) + (
                    f"CURATION SINCE THE RUN: {curation_note}",))
        names = tuple(report.tables())
        path = ask_export_path(
            self, title=title, suggestion="subpopulations",
            filters="Text report (*.txt);;All files (*)",
            outputs=lambda p: report_paths(p, names))
        if path is None:
            return
        try:
            written = write_report(report, path)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, title, f"Could not write the export:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Wrote {written['text'].name} + {len(written) - 1} sidecar file(s) to "
            f"{written['text'].parent}.", 10000)

    def _invalidate_level_fit(self) -> None:
        """Drop the stored level fit. Called when the selection's detection changes.

        A level table belongs to the exact event table it was fitted on. Carried across a
        detection change it would be exported under another recording's provenance and
        matched to another recording's event_ids -- which both tables have and which mean
        nothing across recordings. Absence is recoverable in one menu click; a level table
        silently describing the wrong recording is not.
        """
        stale = self._level_fit
        self._level_fit = None
        self._level_fit_frame = None
        # ...and the re-fit inputs with it. They describe the retired fit's recording, its
        # baseline estimate and its two estimators; kept alive past the fit they belong to they
        # would let a later re-fit splice rows measured against ANOTHER recording's open pore
        # into a table matched by event_id, which is the hazard `_invalidate_level_fit` exists
        # for. `_refit_levels_for_event` restores them only when it is itself the caller.
        self._level_fit_inputs = None
        if hasattr(self, "_act_export_levels"):
            self._act_export_levels.setEnabled(False)
        self._push_level_fit(None)
        # ...and out of the RESULT STORE, not just out of this window's hands. A retired
        # level table left in the store was inert for as long as nothing read it — but
        # project save reads the store, so it was written into the file and restored
        # against whatever detection came back, its event_ids naming different events. A
        # fit this method has just declared invalid must not survive anywhere.
        if stale is not None:
            self._session.results.drop(stale.doc_id, stale.channel, stale.module_id)
            # The store drop emits nothing, so tell the panel to take the STEP overlay
            # off the trace now — not on whatever curation edit happens to come next.
            self._analysis_panel.redraw_overlays()

    def _export_levels(self) -> None:
        """Analysis ▸ Export sub-state levels… — the fitted level table plus its sidecar.

        Long layout (one row per event level). The sidecar carries the CUSUM parameters
        for the same reason the events sidecar carries the baseline estimator: those
        numbers decide how many levels an event HAS, and a table that cannot name them
        cannot be compared with one exported next month.
        """
        from ..analysis.scoring import levels_export_paths, write_levels_export

        # PASS THROUGH THE RECONCILER, do not read around it. Every path that moves a
        # span today emits one of SURFACE_MUTATION_SIGNALS and so reaches
        # _reconcile_level_fit before this action can run — but the guarantee "the
        # export never writes a level table describing spans the events no longer
        # have" then rests on the refresher having fired, not on the export. It is
        # idempotent and a no-op when nothing moved (identity return), so this costs
        # nothing and makes the export a passer-through instead of a second reader.
        self._reconcile_level_fit()

        result = getattr(self, "_level_fit", None)
        if result is None or result.payload is None or result.payload.empty:
            QtWidgets.QMessageBox.information(
                self, "Export sub-state levels",
                "Run Analysis ▸ Fit sub-state levels (CUSUM)… first — there are no "
                "fitted levels to export.")
            return

        path = ask_export_path(
            self, title="Export sub-state levels", suggestion="levels.csv",
            filters="CSV (*.csv);;All files (*)",
            outputs=levels_export_paths)
        if path is None:
            return
        source_file, doc_meta = self._export_source(result.doc_id)
        det = self._analysis_panel.current_detection()
        prov = {}
        if det is not None and getattr(det.payload, "provenance", None):
            prov = dict(det.payload.provenance)
        try:
            out = write_levels_export(
                result.payload, str(path), provenance=prov,
                cusum_params=dict(result.params),
                source_file=source_file,
                detector=det.module_id if det is not None else "",
                meta=doc_meta)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export sub-state levels", f"Could not write the export:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Wrote {len(result.payload)} level row(s) to {out['data'].name} "
            f"(+ provenance {out['meta'].name}).", 8000)

    def _density_groups(self) -> "dict[str, object]":
        """Every open recording that HAS a detection, label -> accepted EventSet.

        Keyed by the recording's own name because that is what the legend must say: the
        figure exists to put several molecules (or several voltages) side by side, and a
        legend reading "Detected events" three times would defeat it. Ordered by descending
        n so the densest group is drawn first and the sparse ones stay visible on top.
        """
        from ..analysis.events import EventSet
        from ..analysis.iv import applied_voltage_mV

        def _common(strings: list[str]) -> int:
            """Length of the longest common prefix. Local so this needs no new import."""
            if not strings:
                return 0
            first, n = strings[0], len(strings[0])
            for other in strings[1:]:
                n = min(n, len(other))
                while n and first[:n] != other[:n]:
                    n -= 1
            return n

        out = []
        for res in self._session.results.all():
            if not isinstance(res.payload, EventSet):
                continue
            acc = res.payload.accepted
            if len(acc.df) < 5:      # a KDE of four points is a decoration
                continue
            stem = (Path(self._session.document(res.doc_id).meta.path).stem
                    if self._session.has_document(res.doc_id) else source_of(res.doc_id))
            # The SHARED resolver, not a local copy. This block used to look only for a
            # VOLTAGE channel, so a current-only recording — every CSV export the owner has —
            # never got a voltage label and fell through to the filename stub, printing
            # "6_Process_-6" in the legend of a three-voltage figure. iv.applied_voltage_mV
            # already prefers the measured channel and falls back to the filename/header
            # nominal, which is exactly the answer the Calibration panel shows.
            volts = None
            if self._session.has_document(res.doc_id):
                try:
                    got = applied_voltage_mV(self._session, res.doc_id)
                    volts = None if got is None else float(got[0])
                except Exception:      # noqa: BLE001 - a label must never break the figure
                    volts = None
            out.append((len(acc.df), stem, volts, acc))
        out.sort(key=lambda t: -t[0])

        # LABEL BY WHAT DISTINGUISHES THEM. A legend is the whole point of this figure and
        # a recording filename is ~60 characters of shared boilerplate -- three of them
        # ran off both sides of the plot and buried the contours. When every recording
        # reports a DIFFERENT applied voltage, that is the axis of the comparison and the
        # label writes itself; otherwise fall back to the filename with the common prefix
        # and suffix stripped, which leaves exactly the part that differs.
        vs = [v for _n, _s, v, _e in out]
        if len(out) > 1 and all(v is not None for v in vs) and len({round(v) for v in vs}) == len(vs):
            return {f"{v:.0f} mV": ev for _n, _s, v, ev in out}
        stems = [s_ for _n, s_, _v, _e in out]
        head = _common(stems) if len(stems) > 1 else 0
        tail = _common([s_[::-1] for s_ in stems]) if len(stems) > 1 else 0
        labels = []
        for s_ in stems:
            core = s_[head: len(s_) - tail] if (head or tail) else s_
            labels.append((core or s_)[:32])
        return {lab: ev for lab, (_n, _s, _v, ev) in zip(labels, out)}

    #: Amplitude measures the density figure can use, as (column, axis label).
    #:
    #: ``frac_depth`` is dI/I0 -- the blockade as a FRACTION of that event's own open-pore
    #: current -- and it is the honest axis whenever the current gain is not calibrated,
    #: which the status bar says outright for most of this owner's recordings. A ratio
    #: divides the gain out, so two recordings taken at different gains are comparable in
    #: dI/I0 and are NOT comparable in pA. It is also the axis that survives a calibration
    #: being revised afterwards.
    #:
    #: ``depth_pA`` stays the default because it is what a reader expects on a blockade
    #: axis and what the event table's headline column reports.
    _DENSITY_Y = {
        "depth (pA)": ("depth_pA", "Amplitude (pA)"),
        "fractional blockade (dI/I0)": ("frac_depth", "Fractional blockade  ΔI/I₀"),
    }

    #: Density-figure style, remembered for the session. Defaults are deliberately modest:
    #: this figure overlays several populations, so its legend sits ON TOP of the contours
    #: it names, and a legend is an index rather than content.
    _DENSITY_STYLE: dict[str, object] = {
        "levels": 8, "line_width": 0.7, "legend_loc": "upper right",
        "legend_fontsize": 6.0, "legend_title": "molecule", "show_counts": True,
        "font_scale": 1.0,
    }

    _DENSITY_LEGEND_LOCS = ("upper right", "upper left", "lower right", "lower left",
                            "center right", "outside right", "none")

    def _edit_density_style(self) -> bool:
        """Small form for the density figure's styling. True when the user accepted.

        Kept as a dialog rather than a menu of presets because the useful knob depends on
        the data: three overlaid recordings need a smaller legend, a single one does not,
        and a figure going into a paper wants the legend off the data entirely.
        """
        st = dict(self._DENSITY_STYLE)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Density figure style")
        form = QtWidgets.QFormLayout(dlg)

        levels = QtWidgets.QSpinBox(); levels.setRange(2, 20); levels.setValue(int(st["levels"]))
        form.addRow("Contour levels", levels)
        lw = QtWidgets.QDoubleSpinBox(); lw.setRange(0.1, 5.0); lw.setSingleStep(0.1)
        lw.setValue(float(st["line_width"])); form.addRow("Line width", lw)
        loc = QtWidgets.QComboBox(); loc.addItems(self._DENSITY_LEGEND_LOCS)
        loc.setCurrentText(str(st["legend_loc"]))
        loc.setToolTip("“outside right” lifts the legend clear of the contours entirely; "
                       "“none” removes it.")
        form.addRow("Legend position", loc)
        lfs = QtWidgets.QDoubleSpinBox(); lfs.setRange(3.0, 20.0); lfs.setSingleStep(0.5)
        lfs.setValue(float(st["legend_fontsize"])); form.addRow("Legend font size", lfs)
        ltitle = QtWidgets.QLineEdit(str(st["legend_title"] or ""))
        ltitle.setPlaceholderText("(no heading)"); form.addRow("Legend heading", ltitle)
        counts = QtWidgets.QCheckBox("show (n = …) in the legend")
        counts.setChecked(bool(st["show_counts"]))
        counts.setToolTip("A legend is sized by its widest entry, and the counts are the "
                          "longest part of one. They stay in the exported source CSV.")
        form.addRow("", counts)
        fs = QtWidgets.QDoubleSpinBox(); fs.setRange(0.5, 3.0); fs.setSingleStep(0.1)
        fs.setValue(float(st["font_scale"])); form.addRow("Font scale", fs)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return False
        self._DENSITY_STYLE = {
            "levels": levels.value(), "line_width": lw.value(),
            "legend_loc": loc.currentText(), "legend_fontsize": lfs.value(),
            "legend_title": ltitle.text().strip() or None,
            "show_counts": counts.isChecked(), "font_scale": fs.value(),
        }
        return True

    def _export_density(self, *, ycol: str | None = None) -> None:
        """Analysis ▸ Export density figure… — overlaid KDE contours, one per recording.

        ``ycol`` is KEYWORD-ONLY, and that is load-bearing rather than tidiness.
        ``QAction.triggered`` carries a ``checked`` bool, which Qt passes as the first
        POSITIONAL argument to whatever it is connected to -- so as a positional parameter
        ``ycol`` arrived as ``False`` from the menu, the amplitude prompt was skipped, and
        the export died on "events frame has no column False". Keyword-only makes that
        mis-binding impossible to write; the connection below also drops the bool.
        """
        from ..plot.figures import figure_output_paths, joint_density, save_figure

        groups = self._density_groups()
        if not groups:
            QtWidgets.QMessageBox.information(
                self, "Export density figure",
                "Run detection on at least one recording first — a density plot needs "
                "detected events.")
            return
        ylabel = "Amplitude (pA)"
        if ycol is None:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("Amplitude measure")
            box.setText("Which amplitude should the density use?")
            box.setInformativeText(
                "depth (pA) is the absolute blockade.\n\n"
                "Fractional blockade ΔI/I₀ divides by each event's own open-pore current, "
                "so it is independent of the current gain — the comparable choice when the "
                "gain is uncalibrated or differs between recordings.")
            b_pa = box.addButton("depth (pA)", QtWidgets.QMessageBox.AcceptRole)
            b_fr = box.addButton("fractional ΔI/I₀", QtWidgets.QMessageBox.AcceptRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(b_pa)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_fr:
                ycol, ylabel = self._DENSITY_Y["fractional blockade (dI/I0)"]
            elif clicked is b_pa:
                ycol, ylabel = self._DENSITY_Y["depth (pA)"]
            else:
                return
        else:
            ylabel = next((lab for col, lab in self._DENSITY_Y.values() if col == ycol),
                          "Amplitude (pA)")
        path = ask_export_path(
            self, title="Export density figure", suggestion="density.pdf",
            filters="PDF (*.pdf);;PNG (*.png);;All files (*)",
            outputs=lambda p: figure_output_paths(Path(p)))
        if path is None:
            return
        try:
            fig, _ax = joint_density(
                groups, ycol=ycol, ylabel=ylabel,
                title=f"Event density — {len(groups)} recording(s)",
                **self._DENSITY_STYLE)
            out = save_figure(fig, str(path))
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export density figure", f"Could not write the figure:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Wrote a density figure over {len(groups)} recording(s) to "
            f"{out['image'].name if 'image' in out else Path(path).name} "
            f"(+ source data).", 8000)

    def _export_training(self) -> None:
        """Write reviewed decisions (recommended), accepted candidates, or all rows.

        Reviewed is the supervised-learning-safe default: both human positives and human
        negatives, with untouched candidates and automatic rejects withheld. Accepted/all
        remain explicit escape hatches and are described without calling them curated truth.
        """
        import numpy as np

        from ..analysis.events import EventSet, STATUS_ACCEPTED, STATUS_REJECTED, event_status
        from ..analysis.labels import label_conflicts
        from ..analysis.scoring import (
            TRAINING_SELECTION_ACCEPTED,
            TRAINING_SELECTION_ALL,
            TRAINING_SELECTION_REVIEWED,
            build_training_frame,
            training_export_paths,
            write_training_export,
        )

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, "Export events for training",
                "Run detection on a current channel first — there are no events to export.")
            return

        df = res.payload.df
        n_reviewed = len(build_training_frame(
            df, selection=TRAINING_SELECTION_REVIEWED))
        n_accepted = len(build_training_frame(
            df, selection=TRAINING_SELECTION_ACCEPTED))
        n_all = len(df)
        n_unresolved_curation = 0
        if CONSENSUS_CURATION_CONFLICT_COL in df.columns:
            n_unresolved_curation = int(
                df[CONSENSUS_CURATION_CONFLICT_COL]
                .astype("string").fillna("").str.strip().ne("").sum()
            )
        reviewed_exclusion_note = (
            f"\n{n_unresolved_curation:,} unresolved reconciliation decision"
            f"{'' if n_unresolved_curation == 1 else 's'} excluded from Reviewed until "
            "you explicitly Accept or Reject the row."
            if n_unresolved_curation else ""
        )
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle("Export events for training")
        box.setText("Which events should the training table contain?")
        box.setInformativeText(
            f"Reviewed decisions ({n_reviewed:,}) — recommended: human positives and "
            f"human negatives only.{reviewed_exclusion_note}\n\n"
            f"Accepted candidates ({n_accepted:,}) — includes accepted rows nobody has "
            "reviewed yet.\n\n"
            f"All raw candidates ({n_all:,}) — also includes automatic detector rejects."
        )
        b_reviewed = box.addButton(
            "Reviewed decisions (recommended)", QtWidgets.QMessageBox.AcceptRole)
        b_acc = box.addButton("Accepted candidates", QtWidgets.QMessageBox.ActionRole)
        b_all = box.addButton("All raw candidates", QtWidgets.QMessageBox.ActionRole)
        b_cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setEscapeButton(b_cancel)
        if n_reviewed:
            box.setDefaultButton(b_reviewed)
        else:
            b_reviewed.setEnabled(False)
            b_reviewed.setToolTip(
                (f"{n_unresolved_curation:,} reviewed decision(s) remain contradictory; "
                 "explicitly Accept or Reject them before using Reviewed export."
                 if n_unresolved_curation else
                 "No event in this table records a human review yet. Review both accepted "
                 "and rejected examples before exporting supervised-learning truth.")
            )
            # With no reviewed truth, Return must not quietly downgrade the export to
            # unreviewed accepted candidates. That remains available as an explicit click.
            box.setDefaultButton(b_cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_reviewed:
            selection = TRAINING_SELECTION_REVIEWED
        elif clicked is b_acc:
            selection = TRAINING_SELECTION_ACCEPTED
        elif clicked is b_all:
            selection = TRAINING_SELECTION_ALL
        else:
            return

        # ``|`` in a structured label is reserved for an unresolved merge/reconciliation
        # conflict (for example ``molecule=apo|holo``), not a legitimate combined class.
        # Warn over exactly the rows the chosen export mode will write, defaulting to
        # Cancel so ambiguous labels cannot silently become supervised-learning truth.
        if selection == TRAINING_SELECTION_REVIEWED:
            selected_mask = event_status(df).isin(
                (STATUS_ACCEPTED, STATUS_REJECTED)).to_numpy(dtype=bool)
            if CONSENSUS_CURATION_CONFLICT_COL in df.columns:
                unresolved_curation = (
                    df[CONSENSUS_CURATION_CONFLICT_COL]
                    .astype("string").fillna("").str.strip().ne("")
                    .to_numpy(dtype=bool)
                )
                selected_mask &= ~unresolved_curation
        elif selection == TRAINING_SELECTION_ACCEPTED:
            selected_mask = (df["accepted"].to_numpy(dtype=bool)
                             if "accepted" in df.columns
                             else np.ones(len(df), dtype=bool))
        else:
            selected_mask = np.ones(len(df), dtype=bool)
        n_conflicts = 0
        if "flag" in df.columns and selected_mask.any():
            n_conflicts = sum(
                bool(label_conflicts(flag))
                for flag in df.loc[selected_mask, "flag"].to_numpy(dtype=object)
            )
        if n_conflicts:
            conflict = QtWidgets.QMessageBox(self)
            conflict.setIcon(QtWidgets.QMessageBox.Warning)
            conflict.setWindowTitle("Unresolved label conflicts")
            conflict.setText(
                f"{n_conflicts:,} selected event(s) have unresolved label conflicts."
            )
            conflict.setInformativeText(
                "A vertical bar (|) is the conflict marker, not a combined class. "
                "Resolve these labels in the Event Gallery or Event Inspector before "
                "training, or explicitly export them for external cleanup."
            )
            b_export_conflicts = conflict.addButton(
                "Export with conflicts", QtWidgets.QMessageBox.DestructiveRole)
            b_cancel_conflicts = conflict.addButton(
                "Cancel", QtWidgets.QMessageBox.RejectRole)
            conflict.setDefaultButton(b_cancel_conflicts)
            conflict.setEscapeButton(b_cancel_conflicts)
            conflict.exec()
            if conflict.clickedButton() is not b_export_conflicts:
                return

        # Reconciliation keeps contradictory human rulings visible rather than silently
        # choosing one.  The row remains reviewed but is rejected until a curator makes a
        # fresh explicit decision, so reviewed-only selection alone cannot make it safe.
        n_curation_conflicts = 0
        if CONSENSUS_CURATION_CONFLICT_COL in df.columns and selected_mask.any():
            markers = (
                df.loc[selected_mask, CONSENSUS_CURATION_CONFLICT_COL]
                .astype("string").fillna("").str.strip()
            )
            n_curation_conflicts = int(markers.ne("").sum())
        if n_curation_conflicts:
            conflict = QtWidgets.QMessageBox(self)
            conflict.setIcon(QtWidgets.QMessageBox.Warning)
            conflict.setWindowTitle("Contradictory curation decisions")
            conflict.setText(
                f"{n_curation_conflicts:,} selected event(s) combine contradictory "
                "human accept/reject decisions."
            )
            conflict.setInformativeText(
                "Explicitly Accept or Reject these rows in the event table, Event Gallery, "
                "or Event Inspector before training, or explicitly export them for external "
                "resolution."
            )
            b_export_conflicts = conflict.addButton(
                "Export with curation conflicts", QtWidgets.QMessageBox.DestructiveRole)
            b_cancel_conflicts = conflict.addButton(
                "Cancel", QtWidgets.QMessageBox.RejectRole)
            conflict.setDefaultButton(b_cancel_conflicts)
            conflict.setEscapeButton(b_cancel_conflicts)
            conflict.exec()
            if conflict.clickedButton() is not b_export_conflicts:
                return

        # C18: this command writes a `<stem>.meta.json` provenance sidecar next to the
        # table, so the confirmation has to cover BOTH. training_export_paths is the
        # writer's own enumeration, including its bare-name -> .csv normalisation.
        chosen = ask_export_path(
            self, title="Export events for training", suggestion="events_training.csv",
            filters="CSV (*.csv);;Parquet (*.parquet);;All files (*)",
            outputs=training_export_paths)
        if chosen is None:
            return
        path = str(chosen)

        source_file, doc_meta = self._export_source(res.doc_id)
        export_provenance = dict(res.payload.provenance or {})
        # The vocabulary belongs to the project/session rather than to one detector result.
        # Supply a copy only at this GUI export seam so the sidecar explains each dynamic
        # label column without mutating the stored detector provenance.
        export_provenance["label_vocabulary"] = self._label_vocab.to_json()
        try:
            out = write_training_export(
                res.payload.df, path, provenance=export_provenance,
                source_file=source_file, detector=res.module_id,
                selection=selection, meta=doc_meta)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export events for training",
                f"Could not write the training export:\n{exc}")
            return
        n_sel = len(build_training_frame(df, selection=selection))
        selection_label = {
            TRAINING_SELECTION_REVIEWED: "reviewed decision",
            TRAINING_SELECTION_ACCEPTED: "accepted candidate",
            TRAINING_SELECTION_ALL: "raw candidate",
        }[selection]
        self.statusBar().showMessage(
            f"Wrote {n_sel} {selection_label}{'' if n_sel == 1 else 's'} to "
            f"{out['data'].name} "
            f"(+ provenance {out['meta'].name}).", 8000)

    def _export_figures(self) -> None:
        """Analysis ▸ Export figure… — write the current detection's publication figures
        (each as PDF + PNG + a source-data CSV) via nanoscope.plot.figures.

        The event-statistics figures are data-model-agnostic (they eat the EventSet); the
        trace preview is threaded from the CalibratedChannel + XAxis so its y-label carries
        the channel's own units instead of a hard-coded pA (PORT_PLAN R9)."""
        from ..analysis.events import EventSet
        from ..plot import figures

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, "Export figure",
                "Run detection on a current channel first — there is no analysis to export.")
            return
        events = res.payload

        # THE fit the user computed, not a fresh one. figures.dwell_histogram(fit=True)
        # refits from scratch at n_components=1 — so a user who set "dwell components = 2",
        # read tau1/tau2 in the Fits panel and then exported would get a figure annotated
        # with a DIFFERENT, one-component tau. _resolve_fit accepts a FitResult, and the
        # session is already holding the right one; pass it. With no stored fit we keep
        # today's behaviour and say so in the status line.
        dwell_fit, depth_fit = self._stored_fit(res, "fit.dwell"), self._stored_fit(res, "fit.depth")

        # Each figure is best-effort: a fit can fail on sparse data, but the histogram
        # itself must still export. We never let one figure sink the whole export.
        builders = [
            ("dwell", lambda: figures.dwell_histogram(events, fit=dwell_fit[0] or True)),
            ("depth", lambda: figures.depth_histogram(events, fit=depth_fit[0] or True)),
            ("scatter", lambda: figures.scatter_dwell_depth(events)),
        ]
        # The trace preview needs the raw signal + its time axis.
        try:
            ch = self._session.calibrated(res.doc_id, res.channel)
            xax = self._session.x_axis(res.doc_id)
            if xax.is_time:
                builders.append(
                    ("trace", lambda: figures.trace_preview(ch, xax, events=events)))
        except Exception:
            pass

        # The builders are settled BEFORE the dialog because the dialog has to be able to
        # say what will be written. Naming `figures.pdf` — a file this command never writes
        # — while overwriting `{stem}_{name}.{pdf,png,csv}` for every builder was C11.
        base = ask_export_path(
            self, title="Generate standard report figures — choose a base name",
            suggestion="figures",
            filters="PDF (*.pdf);;PNG (*.png);;All files (*)",
            outputs=lambda p: self._figure_export_paths(p, [n for n, _ in builders]))
        if base is None:
            return
        # *** THE FIGURES DESCRIBE THE DETECTION THAT IS CURRENT WHEN THEY ARE WRITTEN. ***
        #
        # Everything above — `res`, `events`, both stored fits, the calibrated channel and
        # the x-axis — was resolved BEFORE `ask_export_path`, and that dialog is a real
        # modal: it spins the event loop, so a detection can land inside it. Nothing gates
        # Detect (or a batch pass) during an export, and `AnalysisController` delivers
        # `finished` queued on the GUI thread, so this is not exotic. Measured: the screen
        # went from 8 events to 2 while the dialog was open, and twelve files were written
        # from the 8-event frame under a status line that said only "Wrote 4 figure(s)".
        # The figure source CSVs carry no provenance columns at all, so nothing on disk or
        # on screen could tell the two runs apart.
        #
        # Re-read, and REFUSE if it moved rather than silently exporting either one: the
        # user consented to a named list of files for the analysis they were looking at,
        # and both "write the old one" and "quietly write the new one" break that. Pressing
        # the action again exports what is on screen now, with the same names.
        # `_stored_fit` validates the fits against `res`, so it cannot catch this — `res`
        # itself is the thing that went stale.
        # The closed-recording case is asked FIRST, because it is also a "the detection
        # moved" case (`current_detection` returns None for a document that is gone) and
        # it has a different, more useful sentence.
        if not self._session.has_document(res.doc_id):
            QtWidgets.QMessageBox.warning(
                self, "Export figure",
                f"“{Path(source_of(res.doc_id)).name}” was closed while the file dialog "
                f"was open, so there is no recording behind these figures any more.\n\n"
                f"Nothing was written.")
            return
        if self._analysis_panel.current_detection() is not res:
            QtWidgets.QMessageBox.warning(
                self, "Export figure",
                "The detection on screen changed while the file dialog was open, so these "
                "figures would have described an analysis you are no longer looking at.\n\n"
                "Nothing was written. Press Generate standard report figures… again to export the detection "
                "that is current now.")
            return
        stem = base.with_suffix("")

        # ONE staged set across the WHOLE export, not one per figure. The user consented to
        # a list of twelve files; three of them landing beside nine survivors of a previous
        # export, under one `figures_*` stem, with no marker and a status line that clears
        # after twelve seconds, is a folder of mixed vintages that is indistinguishable from
        # a consistent one. Everything is staged first and committed together, so the folder
        # goes from one whole export to the next.
        from ..core.atomic_write import StagedSet

        group = StagedSet()
        written: list[str] = []
        # The PATHS as well as the display names: the identity disclosure below is
        # scoped to the files this export actually wrote, so that a pending note about
        # some other file is never printed under this export's title.
        written_paths: list[Path] = []
        failed: list[str] = []
        #: figure name -> the real error, for the figures that BUILT but failed while
        #: being rendered/saved. Build failures have their own retry-without-fit path
        #: and wording; these are disclosed with their cause so the user chases the one
        #: bad label instead of a disk problem that does not exist.
        save_errors: dict[str, str] = {}
        commit_error: Exception | None = None
        write_error: OSError | None = None
        for name, build in builders:
            try:
                fig, _ = build()
            except Exception:
                # Retry the histograms without the fit overlay so the figure still ships.
                if name in ("dwell", "depth"):
                    try:
                        fn = figures.dwell_histogram if name == "dwell" else figures.depth_histogram
                        fig, _ = fn(events, fit=False)
                    except Exception:
                        failed.append(name)
                        continue
                else:
                    failed.append(name)
                    continue
            try:
                out = figures.save_figure(fig, f"{stem}_{name}.pdf", group=group)
                written.append(Path(out["pdf"]).name)
                written_paths.extend(Path(v) for v in out.values())
            except OSError as exc:
                # The FOLDER, not the figure: a permission or disk error at staging
                # hits every figure the same way, so pressing on would only collect
                # the same error again — and the "no figures could be exported from
                # the current detection result" summary would then blame the user's
                # data for a directory they cannot write (EXP-01). Stop and report
                # it below as what it is, the way _export_events already does.
                write_error = exc
                break
            except Exception as exc:
                # THIS figure failed while being rendered/saved (fig.savefig raises at
                # DRAW time — e.g. invalid mathtext in a title carried from the data
                # file), not the folder: save_figure has withdrawn the family from the
                # shared staged set, so the healthy figures still commit below. The
                # figure is disclosed BY NAME with its real error — before this, the
                # orphaned staged entry sank the whole export at commit and the dialog
                # blamed the environment ("could not be put in place") in a perfectly
                # writable folder.
                failed.append(name)
                save_errors[name] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        if write_error is not None:
            # Environmental, not per-figure. Nothing was committed (everything was
            # still staged), so the folder is exactly as it was — say so, and name
            # the real cause with the same sentence the events export uses.
            group.abandon()
            QtWidgets.QMessageBox.warning(
                self, "Export figures",
                f"Could not write the export:\n{write_error}\n\n"
                f"Nothing was written — every file already in the folder is exactly "
                f"as it was.")
            return

        if written:
            try:
                group.commit()
            except Exception as exc:      # the commit is where a full disk finally bites
                commit_error = exc
                written = []
                failed = [name for name, _ in builders]
        else:
            group.abandon()

        if commit_error is not None:
            QtWidgets.QMessageBox.critical(
                self, "Export figures",
                f"Nothing was written.\n\nThe figures were prepared but could not be put in "
                f"place next to {stem.name} ({commit_error}). Every file that was already "
                f"there is exactly as it was.")
            return
        if not written:
            msg = "No figures could be exported from the current detection result."
            if save_errors:
                # ...and when a figure built but failed to render, say WHICH and WHY —
                # "no figures could be exported" alone points the user at their data.
                msg += "\n\nFailed while rendering:\n" + "\n".join(
                    f"{stem.name}_{n}: {err}" for n, err in save_errors.items())
            QtWidgets.QMessageBox.warning(self, "Export figure", msg)
            # ...and the leftovers, on THIS path too. The partial-failure path below
            # reports every consented output an export could not replace, with its
            # vintage — the whole point being that a file of that name from a PREVIOUS
            # export now sits under this export's stem and reads as part of this one.
            # Returning here skipped that, so 3-of-4 figures failing got the dated
            # leftover dialog and 4-of-4 — strictly worse, every consented file left
            # unreplaced — got nothing. The user learns least in the case where the
            # most is stale. `failed` is every builder on this branch, which is exactly
            # the set whose names may still be on disk.
            self._report_unreplaced_figures(stem, failed, errors=save_errors)
            return

        msg = f"Wrote {len(written)} figure(s) next to {stem.name} (PDF + PNG + CSV each)."
        msg += "\n" + self._fit_provenance_note(dwell_fit, depth_fit)
        self.statusBar().showMessage(msg, 12000)
        # A figure that could not be built is not merely "skipped": if a file of that name
        # is already there it is now a survivor of an EARLIER export sitting under this
        # export's stem. Name those files, with the date they are from, in a dialog — a
        # status line the user may not be looking at is not a disclosure.
        if failed:
            self._report_unreplaced_figures(stem, failed, errors=save_errors)
        self._disclose_identity_notes("Export figures", written_paths)

    def _report_unreplaced_figures(self, stem: Path, failed: list[str],
                                   errors: dict[str, str] | None = None) -> None:
        """Name every consented output this export could not replace, and its vintage.

        Two different situations, and the user needs to be told them apart:

        * the figure could not be built and NOTHING of that name is on disk — a plain
          "not exported", which is what the old status line said;
        * the figure could not be built and a file of that name IS on disk — it is from a
          previous export, it now sits under this export's stem, and every tool that reads
          the folder (including the user, next month) will take it for part of this one.

        Only the second needs a dialog, and it gets one, with each file's modification date.

        ``errors`` carries the real cause for the figures that BUILT but failed while
        being rendered/saved (name -> "ExcType: message"). Those are named WITH their
        error — "could not be built" alone sent the user to their data for what was one
        bad label in one figure's title.
        """
        from ..plot import figures

        # Each failed figure, with its real error when one was captured.
        def _named(name: str) -> str:
            return f"{name} — {errors[name]}" if errors and name in errors else name

        stale: list[str] = []
        for name in failed:
            for p in figures.figure_output_paths(f"{stem}_{name}.pdf"):
                try:
                    when = QtCore.QDateTime.fromSecsSinceEpoch(
                        int(p.stat().st_mtime)).toString("dd MMM yyyy HH:mm")
                except OSError:
                    continue
                stale.append(f"{p.name}   ({when})")
        if not stale:
            self.statusBar().showMessage(
                f"{len(failed)} figure(s) could not be built and were not exported: "
                f"{', '.join(_named(n) for n in failed)}.", 12000)
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Export figures")
        box.setText(f"{len(stale)} file(s) under this name are left over from an earlier "
                    f"export.")
        box.setInformativeText(
            f"These figures could not be built from the current detection "
            f"({', '.join(_named(n) for n in failed)}), so the files of that name were "
            f"NOT replaced and still "
            f"hold the previous export:\n\n" + "\n".join(stale) +
            f"\n\nDelete them, or export to a different base name, before treating "
            f"{stem.name}_* as one set.")
        box.exec()

    @staticmethod
    def _figure_export_paths(base: Path, names: list[str]) -> list[Path]:
        """Every file the figure export may write for base name ``base``.

        The same ``f"{stem}_{name}.pdf"`` the write loop hands to ``figures.save_figure``,
        expanded through ``figures.figure_output_paths`` — so the set the user confirms is
        derived from the writer's own declaration rather than a second guess at it.
        """
        from ..plot import figures

        stem = Path(base).with_suffix("")
        return [p for name in names
                for p in figures.figure_output_paths(f"{stem}_{name}.pdf")]

    def _stored_fit(self, res, module_id: str):
        """The FitResult already computed for this selection under ``module_id``.

        Returns ``(fit, reason)`` — ``fit`` is None when the export must fall back
        to refitting, and ``reason`` says why, for the status line. Two guards, and
        both matter because a figure is where a number stops being reviewable:

        * calibration fingerprint — a fit taken against a different gain/offset
          describes different depths;
        * sample size — re-running detection (or curating events) leaves the old
          fit in the store, and stamping a fit of yesterday's events onto today's
          histogram is worse than a documented refit.
        """
        try:
            stored = self._session.results.get(res.doc_id, res.channel, module_id)
        except Exception:
            return None, "no saved fit"
        if stored is None or not hasattr(stored.payload, "predict"):
            return None, "no saved fit"
        if tuple(stored.calibration_fingerprint) != tuple(res.calibration_fingerprint):
            return None, "saved fit predates a calibration change"
        n_now = int(res.payload.df["accepted"].to_numpy(dtype=bool).sum())
        n_fit = stored.payload.meta.get("n_input_events")
        if n_fit is not None and int(n_fit) != n_now:
            return None, "saved fit is stale (the accepted events changed since it ran)"
        return stored.payload, ""

    @staticmethod
    def _fit_provenance_note(dwell, depth) -> str:
        """Say, in the status line, WHICH fit each histogram was annotated with.

        The number on an exported figure is the one that gets published; the user
        has to be able to tell at a glance whether it is the fit they read in the
        panel or a default the exporter invented."""
        parts = []
        for label, (fit, reason) in (("dwell", dwell), ("depth", depth)):
            if fit is None:
                parts.append(f"{label}: {reason} — the figure was refit fresh at k=1")
            else:
                k = int(fit.meta.get("n_components", 1) or 1)
                parts.append(f"{label}: used your saved fit (k={k})")
        return "Fits — " + "; ".join(parts) + "."

    def _close_current(self) -> None:
        if self._current_selection is not None:
            self._close_document(self._current_selection[0])

    # -- closing a recording that has segments cut from it ----------------------------------------
    def _dependent_segments(self, doc_id: str) -> list:
        """Every OPEN segment that would be orphaned by closing ``doc_id`` — the segments cut
        from it, plus (transitively) the segments cut from those.

        A segment owns a COPY of its samples, so it survives its parent in memory perfectly
        well. What it cannot survive is a project round-trip: a segment is persisted as a SPEC
        (parent key + window) and re-derived from its parent on load, so with the parent gone
        there is nothing to re-derive it from."""
        from ..core.segment import SegmentDocument, persistent_key

        session = self._session
        if not session.has_document(doc_id):
            return []
        doomed = {persistent_key(session.document(doc_id))}
        found: list = []
        seen: set[str] = set()   # by doc_id: a SegmentDocument holds arrays, so never compare by ==
        # Repeat to a fixpoint so a segment cut from a segment is caught too.
        changed = True
        while changed:
            changed = False
            for doc in session.documents():
                if not isinstance(doc, SegmentDocument) or doc.doc_id in seen:
                    continue
                if doc.origin.parent_key in doomed:
                    found.append(doc)
                    seen.add(doc.doc_id)
                    doomed.add(doc.origin.segment_uid)
                    changed = True
        return found

    def _confirm_close_with_segments(self, name: str, n: int) -> str:
        """Ask what to do about closing a recording that has ``n`` segment(s) cut from it.

        Returns ``_CLOSE_SEGMENTS_TOO`` / ``_CLOSE_PARENT_ONLY`` / ``_CLOSE_CANCEL``. Injectable
        (like :meth:`_confirm_close_choice`) so the suite never faces a live modal."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Close recording")
        box.setText(f"{n} segment(s) were cut from “{name}”.")
        box.setInformativeText(
            f"A segment is saved in a project as a crop of its parent, so it can only be "
            f"restored while that parent recording is open. Closing “{name}” on its own would "
            f"leave those segments — and any detections on them — unsaveable.")
        both = box.addButton(f"Close it and its {n} segment(s)",
                             QtWidgets.QMessageBox.DestructiveRole)
        parent_only = box.addButton("Close only this recording",
                                    QtWidgets.QMessageBox.ActionRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is both:
            return _CLOSE_SEGMENTS_TOO
        if clicked is parent_only:
            return _CLOSE_PARENT_ONLY
        return _CLOSE_CANCEL

    @QtCore.Slot(str)
    def _close_document(self, doc_id: str) -> None:
        """THE close-a-recording path — the Files dock's Close button and File ▸ Close document
        both land here, so the segment guard can never be bypassed by using the other one.

        Closing a recording that has segments cut from it silently made those segments
        unsaveable: the next project save wrote them out as specs whose parent no longer
        existed, reported success, and dropped them (and their detections) on reload. So the
        user is told, and chooses.

        There are TWO questions here and they are deliberately separate, because they have
        different answers. The first decides WHAT closes (this recording, or it and its
        segments). The second — :meth:`_confirm_close_discarding_events` — asks about the
        detected events on whatever that turned out to be, and it is the last guard in the
        application: every other destructive action is now recoverable through Edit ▸ Undo,
        and a close is the one that takes the undo stack with it."""
        if not self._session.has_document(doc_id):
            return
        dependents = self._dependent_segments(doc_id)
        doomed = [doc_id]
        if dependents:
            name = self._session.document(doc_id).display_name
            choice = self._confirm_close_with_segments(name, len(dependents))
            if choice == _CLOSE_CANCEL:
                return
            if choice == _CLOSE_SEGMENTS_TOO:
                doomed.extend(seg.doc_id for seg in dependents)
        # Undo cannot reach past this point: a closed recording takes its results AND its
        # undo history with it, because neither can be applied to a document that is no
        # longer open. So the LAST guard in the app has to be a confirmation, and it has
        # to name the events and the curation rather than the button.
        if not self._confirm_close_discarding_events(doomed):
            return
        for seg_id in doomed[1:]:
            self._session.close_document(seg_id)
        self._session.close_document(doc_id)

    # -- closing a recording that carries detected + curated events -------------------------------
    def _describe_close_stake(self, doc_ids: list[str]) -> tuple[str, bool]:
        """A sentence naming the event work that closing ``doc_ids`` would destroy, plus
        whether any of them is a SEGMENT.

        Returns ``("", False)`` when nothing would be lost — closing a recording you only
        looked at must not raise a dialog, and the mirror of that mistake (nagging on an
        ordinary open) is one this window has already paid for once.
        """
        from ..core.segment import SegmentDocument

        totals = {"n_detections": 0, "n_events": 0, "n_accepted": 0,
                  "n_rejected": 0, "n_added": 0}
        any_segment = False
        for did in doc_ids:
            if not self._session.has_document(did):
                continue
            stake = self._history.stake_for_doc(did)
            if stake["n_detections"] and isinstance(self._session.document(did),
                                                    SegmentDocument):
                any_segment = True
            for k in totals:
                totals[k] += int(stake[k])
        if not totals["n_detections"]:
            return "", False

        bits = [f"{totals['n_accepted']} accepted", f"{totals['n_rejected']} rejected"]
        if totals["n_added"]:
            bits.append(f"{totals['n_added']} hand-added")
        where = ("this recording" if len(doc_ids) == 1
                 else f"these {len(doc_ids)} recordings")
        detections = ("1 detection" if totals["n_detections"] == 1
                      else f"{totals['n_detections']} detections")
        return (f"Closing {where} discards {detections}: "
                f"{totals['n_events']} events ({', '.join(bits)})."), any_segment

    def _confirm_close_discarding_events(self, doc_ids: list[str]) -> bool:
        """Ask before closing recordings that carry detected events. True = go ahead.

        Injectable (like :meth:`_confirm_close_choice`) so the suite never faces a live
        modal. The wording names the events and the curation — what is actually at stake —
        rather than the button, and it says the one thing that decides how bad this is: a
        SEGMENT is in-memory only, so there is no file to reopen and nothing to detect
        again.
        """
        text, any_segment = self._describe_close_stake(doc_ids)
        if not text:
            return True
        if any_segment:
            detail = ("A segment exists only in this session — it is a crop held in memory, "
                      "not a file on disk — so there is no way to reopen it and no way to "
                      "detect these events again. Undo cannot reach past a close.\n\n"
                      "Save the project first if you want to keep them.")
        else:
            detail = ("The recording itself is untouched, but the detection and every "
                      "accept/reject decision on it are held only in this session, and "
                      "undo cannot reach past a close.\n\n"
                      "Save the project first if you want to keep them.")
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Close recording")
        box.setText(text)
        box.setInformativeText(detail)
        discard = box.addButton("Close and discard the events",
                                QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Keep it open", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is discard

    def _on_document_closed(self, doc_id: str) -> None:
        # Closing the SHOWN recording would otherwise leave the viewer blank while the File
        # Manager still lists the recordings that are open — the mirror image of the pile-on
        # complaint. Fall back to one of the survivors.
        self._ensure_something_shown()
        if self._current_selection is not None and self._current_selection[0] == doc_id:
            self._current_selection = None
            self._rate_label.setText("")
            self._sync_detection_action()
            self._sync_ml_detection_action()
            self._sync_measure_action()
            self._sync_export_action()
            self._sync_plots_action()
            self._sync_create_segment_action()
        # The windows are re-fed unconditionally, by the session-wide refresher wired to
        # this same signal — a recording that was not the SELECTED one is still plotted in
        # the Plots and voltage-dependence windows, which is what the old "only if it was
        # the selection" branch missed.
        self._sync_undo_actions()   # the closed doc's history went with it

    # -- status readouts -------------------------------------------------------------------------
    def _on_cursor(self, x: float) -> None:
        """The status-bar ``t = … y = …`` — and the ``y`` describes THE CHANNEL THE USER IS ON.

        It used to take the first visible layer in layer order, with no reference to the
        selection and no preference for kind. ABF channel order puts the command voltage
        BEFORE the current (the navigator already prefers the current-kind layer for exactly
        this reason), so on the owner's recordings the readout was always the command
        voltage: measured on the quick ABF with IN15 selected and the current trace on
        screen, the cursor at sample 200 000 sat on -168.457 pA and the status bar read
        *"t = 1158.99 s     y = -999.878 mV"*, naming no channel — the same in stacked mode,
        where the cursor is demonstrably in the current pane. It is the same rule the window
        already holds for analysis ("THE APP NEVER ANALYSES A RECORDING IT IS NOT DRAWING"):
        a number on screen describes the thing the user is working on, or says what it does
        describe.

        The selection wins when it is drawn. When it is not — nothing selected, or the
        selected channel hidden behind an overlay of other recordings — the first visible
        layer is still the only honest candidate, and then the readout NAMES it rather than
        letting the number pass for the selection's.
        """
        layers = [lay for lay in self._session.layers()
                  if lay.visible and self._session.has_document(lay.doc_id)]
        layer = None
        sel = self._current_selection
        if sel is not None:
            layer = next((lay for lay in layers
                          if (lay.doc_id, lay.channel) == (sel[0], sel[1])), None)
        borrowed = layer is None            # ...then say whose number this is
        if layer is None:
            layer = next(iter(layers), None)
        if layer is None:
            self._cursor_label.setText("")
            return
        xax = self._session.x_axis(layer.doc_id, layer.time_offset_s)
        idx = self._session.time_to_index(layer.doc_id, layer.channel, x, layer.time_offset_s)
        if idx is None:
            self._cursor_label.setText(f"t = {x:.6g} {xax.unit}")
            return
        chan = self._session.calibrated(layer.doc_id, layer.channel)
        units = f" {chan.units}" if chan.units else ""
        whose = f"  ({chan.info.name})" if borrowed else ""
        self._cursor_label.setText(
            f"t = {xax.values[idx]:.6g} {xax.unit}     "
            f"y = {float(chan.data[idx]):.6g}{units}{whose}")

    def _update_rate_label(self, doc_id: str | None = None) -> None:
        if doc_id is None and self._current_selection is not None:
            doc_id = self._current_selection[0]
        if doc_id is None or not self._session.has_document(doc_id):
            self._rate_label.setText("")
            return
        meta = self._session.document(doc_id).meta
        if meta.sample_rate_hz is None:
            self._rate_label.setText("rate: unknown · x-axis in samples")
        else:
            self._rate_label.setText(
                f"{meta.sample_rate_hz:,.6g} Hz · {meta.rate_provenance.value}")

    def _choose_view_mode(self, mode: ViewMode) -> None:
        """View ▸ Overlay / Stacked — set the layout, and say so when the choice cannot show.

        *** "I SWITCHED TO STACKED AND THEN OVERLAY DID NOT COME BACK." *** The round trip is
        functionally correct; what is missing is that the two modes are IDENTICAL while
        exactly one recording is shown, because Overlay's rule is "group comparable channels
        ACROSS recordings" and there is nothing to group across. And one recording is the
        DEFAULT: ``_show_document`` calls ``Session.show_only``. So the user's Overlay click
        left the trace visibly unchanged, with nothing on screen connecting that to the
        multi-recording view she had never built.

        The control is deliberately NOT disabled — the mode really is set, it is remembered,
        and it decides the layout the moment a second recording joins the view. What is added
        is the one sentence naming the missing ingredient and the gesture that supplies it.
        """
        before = len(self._trace_view.panes())
        self._session.set_view_mode(mode)
        if (mode is ViewMode.OVERLAY
                and len(self._trace_view.panes()) == before
                and len(self._session.shown_doc_ids()) < 2):
            self.statusBar().showMessage(
                "Overlay groups comparable channels ACROSS recordings, and only one "
                "recording is shown — so it lays out exactly like Stacked. Right-click a "
                "recording in the Files dock and choose “Add to view (compare)” to bring a "
                "second one in.", 10000)

    def _sync_view_actions(self) -> None:
        self._act_overlay.setChecked(self._session.view_mode is ViewMode.OVERLAY)
        self._act_stacked.setChecked(self._session.view_mode is ViewMode.STACKED)

    # -- open dialog + recent --------------------------------------------------------------------
    def _open_dialog(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open recording(s)", self._last_dir, _OPEN_FILTER)
        if files:
            self._last_dir = str(Path(files[0]).parent)
            self.open_paths(files)

    def _add_recent(self, path: Path) -> None:
        p = str(path.resolve())
        if p in self._recent:
            self._recent.remove(p)
        self._recent.insert(0, p)
        del self._recent[_MAX_RECENT:]
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        if not self._recent:
            empty = self._recent_menu.addAction("(none)")
            empty.setEnabled(False)
            return
        for p in self._recent:
            act = self._recent_menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(lambda _checked=False, path=p: self.open_paths([path]))
        self._recent_menu.addSeparator()
        clear = self._recent_menu.addAction("Clear recent")
        clear.triggered.connect(self._clear_recent)

    def _clear_recent(self) -> None:
        self._recent = []
        self._rebuild_recent_menu()

    # -- crash-recovery autosave ---------------------------------------------------------------
    def _init_recovery(self) -> None:
        """Create the Stage 2 recovery store and idle/deadline scheduler.

        Recovery is deliberately optional at this seam.  A read-only or otherwise broken
        application-support directory must cost the user crash recovery, never the ability
        to launch NanoScope and save a real project.  The controller reports later write
        failures through the status bar while leaving the user project untouched.
        """
        if not self._recovery_enabled:
            return
        try:
            from ..core.recovery import RecoveryStore
            from .recovery_controller import RecoveryController

            store = RecoveryStore(root=self._recovery_root)
            controller = RecoveryController(
                self._capture_recovery_snapshot,
                lambda snapshot, revision, _store=store:
                    _write_recovery_snapshot(_store, snapshot, revision),
                parent=self,
            )
            controller.saved.connect(self._on_recovery_saved)
            controller.failed.connect(self._on_recovery_failed)
            controller.staleSaved.connect(self._on_stale_recovery_saved)
        except Exception as exc:  # noqa: BLE001 - recovery may not prevent application launch
            self._recovery_store = None
            self._recovery_controller = None
            self.statusBar().showMessage(
                f"Crash recovery is unavailable: {exc}. Save projects manually.", 12000)
            return
        self._recovery_store = store
        self._recovery_controller = controller

    def _capture_recovery_snapshot(self, requested_revision: int):
        """Return an immutable snapshot for a recovery worker, on the GUI thread.

        The live project records intentionally share pandas frames with the editing
        surfaces.  They are safe for the synchronous explicit save, but handing those
        references to a worker would let a curation click mutate a table while Parquet is
        serialising it.  ``clone_project_state`` is therefore part of the thread boundary,
        not an implementation detail of the writer.
        """
        if not self._visible_project_dirty() or self._restoring:
            return None
        # Catch an in-place edit whose author forgot to emit its panel signal.  This can
        # advance the revision; the snapshot records the post-checkpoint value rather than
        # the timer's older request.
        self._checkpoint_everything()
        if not self._visible_project_dirty():
            return None
        from ..core.recovery import clone_project_state

        state = clone_project_state(self._gather_project_state())
        return _RecoverySnapshot(
            revision=int(self._session_revision),
            state=state,
            source_project_path=(self._project_path or
                                 self._recovery_suggested_project_path),
            covered_explicit_save_revision=int(self._explicit_saved_revision),
        )

    def _persist_recovery_snapshot(self, snapshot: _RecoverySnapshot,
                                   revision: int | None = None):
        """Direct test/support seam for writing an already-detached snapshot."""
        store = self._recovery_store
        if store is None:
            return None
        return _write_recovery_snapshot(store, snapshot, revision)

    @QtCore.Slot(int, object)
    def _on_recovery_saved(self, revision: int, candidate) -> None:
        self._recovery_revision = max(self._recovery_revision, int(revision))
        # A real project save may have committed while this older worker was finishing.
        # Re-apply the retirement fence after the landing so that completion order cannot
        # resurrect a revision the user already saved explicitly.
        if int(revision) <= self._explicit_saved_revision:
            self._retire_recovery_through(self._explicit_saved_revision)

    @QtCore.Slot(int, str)
    def _on_recovery_failed(self, _revision: int, message: str) -> None:
        if self._visible_project_dirty():
            self.statusBar().showMessage(
                f"Could not update the crash-recovery copy: {message}. "
                "Save the project manually.", 12000)

    @QtCore.Slot(int, int, object)
    def _on_stale_recovery_saved(self, _generation: int, _revision: int,
                                 candidate) -> None:
        """Remove an outgoing-session write that landed after ``reset_session``."""
        if candidate is None or self._recovery_store is None:
            return
        try:
            # Root revisions never reset inside this process.  A revision fence removes
            # both old generations but still lets the new logical session write revision
            # N+1.  An all-revisions discard tombstone would disable recovery for the rest
            # of this app launch because the store deliberately keeps one UUID.
            self._recovery_store.retire_through(
                int(getattr(candidate, "revision", _revision)),
                session_id=getattr(candidate, "session_id", None),
            )
        except Exception:
            # This is already a stale fallback for work the user saved/discarded before
            # changing sessions.  The store's retirement tombstone is the primary guard;
            # a cleanup failure must not disturb the new live session.
            pass

    @staticmethod
    def _candidate_value(candidate, name: str, default=None):
        value = getattr(candidate, name, default)
        return default if value is None else value

    def _choose_recovery_action(self, candidate) -> str | None:
        """Offer Recover / Discard / Show in Finder for one validated candidate."""
        source = self._candidate_value(candidate, "source_project_path", None)
        revision = self._candidate_value(candidate, "revision", "?")
        when = self._candidate_value(candidate, "timestamp", None)
        label = Path(source).name if source else "an unnamed analysis"
        detail = f"NanoScope found unsaved work from {label} (revision {revision})"
        if when:
            try:
                from datetime import datetime
                dt = datetime.fromtimestamp(float(when)).astimezone()
                shown_when = (f"{dt:%b} {dt.day}, {dt:%Y} at "
                              f"{dt.strftime('%I:%M:%S %p').lstrip('0')}")
            except (ValueError, TypeError, OSError):
                shown_when = str(when)
            detail += f", captured {shown_when}"
        detail += "."

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Recover unsaved NanoScope work")
        box.setText("NanoScope did not finish its previous session cleanly.")
        box.setInformativeText(
            detail + "\n\nRecover opens that snapshot as unsaved work. Discard removes "
            "both recovery generations permanently.")
        recover = box.addButton("Recover", QtWidgets.QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
        show = box.addButton("Show in Finder", QtWidgets.QMessageBox.ActionRole)
        not_now = box.addButton("Not now", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(recover)
        box.setEscapeButton(not_now)
        box.exec()
        clicked = box.clickedButton()
        if clicked is recover:
            return _RECOVERY_RECOVER
        if clicked is discard:
            return _RECOVERY_DISCARD
        if clicked is show:
            return _RECOVERY_SHOW
        return None

    def _candidate_path(self, candidate) -> Path | None:
        for name in ("path", "envelope_path", "payload_path", "directory"):
            value = getattr(candidate, name, None)
            if value:
                return Path(value)
        return None

    def _show_recovery_in_finder(self, candidate) -> None:
        path = self._candidate_path(candidate)
        if path is None:
            return
        target = path if path.is_dir() else path.parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))

    def _offer_startup_recovery(self) -> None:
        """Inspect completed prior-session generations once, after the window is visible."""
        if self._startup_recovery_checked:
            return
        self._startup_recovery_checked = True
        store = self._recovery_store
        if store is None:
            return
        try:
            candidates = store.list_candidates()
        except Exception as exc:  # noqa: BLE001 - discovery cannot prevent normal use
            self.statusBar().showMessage(
                f"Could not inspect crash-recovery copies: {exc}.", 12000)
            return
        while candidates:
            candidate = candidates[0]
            action = self._choose_recovery_action(candidate)
            while action == _RECOVERY_SHOW:
                self._show_recovery_in_finder(candidate)
                action = self._choose_recovery_action(candidate)
            if action == _RECOVERY_RECOVER:
                self._recover_candidate(candidate)
                return
            if action != _RECOVERY_DISCARD:
                return
            self._discard_recovery(candidate)
            # More than one crashed process can leave valid work.  Discard is a decision
            # about the candidate just named, not permission to hide every older session.
            try:
                candidates = store.list_candidates()
            except Exception:
                return

    def _safe_recovery_target(self, value) -> str | None:
        if not value:
            return None
        try:
            path = Path(str(value)).expanduser()
        except (TypeError, ValueError, OSError):
            return None
        if not path.is_absolute() or path.suffix.lower() != ".nanoscope":
            return None
        return str(path)

    def _recover_candidate(self, candidate) -> bool:
        store = self._recovery_store
        if store is None:
            return False
        try:
            recovered = store.load(candidate)
        except Exception as exc:  # noqa: BLE001 - a bad generation falls back in the store
            QtWidgets.QMessageBox.warning(
                self, "Recover unsaved work",
                f"Could not recover the saved snapshot:\n{exc}")
            return False

        state = getattr(recovered, "state", None)
        if state is None:
            state = getattr(recovered, "project", None)
        if state is None:
            QtWidgets.QMessageBox.warning(
                self, "Recover unsaved work",
                "The recovery copy did not contain a NanoScope project snapshot.")
            return False

        self._clear_session()
        self._recovered_candidate = candidate
        source = getattr(recovered, "source_project_path", None)
        if source is None:
            source = getattr(candidate, "source_project_path", None)
        self._recovery_suggested_project_path = self._safe_recovery_target(source)
        revision = int(getattr(recovered, "revision",
                               getattr(candidate, "revision", 0)) or 0)
        covered = int(getattr(
            recovered, "covered_explicit_save_revision",
            getattr(candidate, "covered_explicit_save_revision", 0)) or 0)
        self._session_revision = max(self._session_revision, revision)
        self._explicit_saved_revision = max(self._explicit_saved_revision, covered)
        self._restore_is_recovery = True
        self._begin_restore(state)
        return True

    def _discard_recovery(self, candidate=None) -> None:
        """Permanently retire the surfaced lineage and this session's queued copies."""
        controller = self._recovery_controller
        if controller is not None:
            reset = getattr(controller, "reset_session", None)
            if callable(reset):
                reset()
        store = self._recovery_store
        if store is not None:
            try:
                if candidate is not None:
                    if getattr(candidate, "session_id", None) == store.session_id:
                        store.retire_through(int(getattr(candidate, "revision", 0)))
                    else:
                        # A surfaced prior-process lineage will never be reused; its
                        # permanent discard tombstone makes the user's choice definitive.
                        store.discard(candidate)
                # This process DOES continue after Open Project.  Fence its outgoing
                # revisions, do not tombstone every future revision in the launch.
                store.retire_through(int(self._session_revision))
            except Exception as exc:  # noqa: BLE001 - report, but a close must remain possible
                self.statusBar().showMessage(
                    f"Could not discard the crash-recovery copy: {exc}.", 12000)
        if candidate is None or candidate is self._recovered_candidate:
            self._recovered_candidate = None

    def _retire_recovery_through(self, revision: int) -> None:
        controller = self._recovery_controller
        if controller is not None:
            fence = getattr(controller, "retire_through", None)
            if callable(fence):
                fence(int(revision))
        store = self._recovery_store
        if store is None:
            return
        try:
            candidate = self._recovered_candidate
            store.retire_after_save(
                int(revision), recovered_candidate=candidate)
            if candidate is not None:
                self._recovered_candidate = None
        except Exception as exc:  # noqa: BLE001 - real project save already succeeded
            self.statusBar().showMessage(
                f"The project was saved, but its old recovery copy could not be removed: "
                f"{exc}.", 12000)

    # -- project save/open -----------------------------------------------------------------------
    def _gather_project_state(self):
        """Snapshot the whole Session into a plain :class:`ProjectState` for saving.

        File-backed documents are stored as REFERENCES (resolved path == doc_id, plus the
        loaded SampleRange); SEGMENTS are stored as SPECS (uid + parent key + window +
        name) and re-derived from their parent on load, never as signal. Every channel's
        calibration, every layer's view state, every detection/ML EventSet (curated df +
        provenance + module_id + doc/channel), the DERIVED fits (v6: fit.dwell / fit.depth /
        fit.levels — see FitRec) and the manual measurements table come along. Raw signal
        does NOT: file signal reopens from disk.

        What the live session does NOT know is also written: records from the opened project
        whose recording could not be loaded are carried through verbatim rather than dropped
        (see :meth:`_capture_unrestorable`).

        Calibrations, layers and results are keyed by each document's PERSISTENT KEY (a
        file's path, a segment's uid) — never the volatile in-memory doc_id — so they
        re-attach after the parents reopen and the segments are re-derived."""
        import pandas as pd

        from ..analysis.events import EventSet
        from ..analysis.stats import FitResult, fit_to_json
        from ..core.project import (
            CalibrationRec, DocumentRef, FitRec, LayerRec, ProjectState, ResultRec, SegmentRec,
        )
        from ..core.calibration import channel_identity
        from ..core.identity import fingerprint_source
        from ..core.segment import SegmentDocument, persistent_key

        session = self._session
        documents: list[DocumentRef] = []
        segments: list[SegmentRec] = []
        calibrations: list[CalibrationRec] = []
        pkey: dict[str, str] = {}   # live doc_id -> persistent key (identity across a round-trip)
        # live doc_id -> {channel index: ChannelInfo}. v7: every record files the IDENTITY of
        # its channel (name + kind) alongside the index, because an index is a position and a
        # re-export that reorders the signal columns moves it. See CalibrationRec/ResultRec
        # and `calibration.resolve_channel`.
        chinfo: dict[str, dict[int, Any]] = {}

        for doc in session.documents():
            doc_id = doc.doc_id
            key = persistent_key(doc)
            pkey[doc_id] = key
            chinfo[doc_id] = {int(i.index): i for i in doc.meta.channels}
            if isinstance(doc, SegmentDocument):
                s = doc.spec
                segments.append(SegmentRec(
                    segment_uid=s.segment_uid, parent_key=s.parent_key,
                    i0=int(s.i0), i1=int(s.i1), name=s.name, t0_abs_s=float(s.t0_abs_s)))
            else:
                lr = doc.meta.loaded_range
                # `key` (== source_key == the resolved path), never `doc_id`: a doc_id
                # carries a per-load token and is meaningless in the next session.
                #
                # The SOURCE FINGERPRINT is what makes the reference checkable. n_samples
                # comes from the DOCUMENT (what was actually analysed), the rest from the
                # file; the digest samples 3x256 KiB rather than hashing whole, which is
                # 1.5 ms against ~1.4 s on the owner's 606 MB recording — a cost every save
                # and every open would otherwise pay.
                documents.append(DocumentRef(
                    path=key,
                    loaded_range=(int(lr.start), int(lr.stop)) if lr is not None else None,
                    source=fingerprint_source(doc.source_key, doc.meta.n_samples)))
            for info in doc.meta.channels:
                # The record's OWN translation, never a field list spelled out here: a
                # calibration field that exists on the value object and not in the record is
                # not merely "not persisted", it is silently BLANKED on the next ordinary
                # Apply — which is how `note` used to destroy the user's sidecar. A test
                # compares the two field lists; see CalibrationRec.
                calibrations.append(CalibrationRec.from_calibration(
                    key, info.index, session.calibration(doc_id, info.index), info))

        def _identity(live_doc_id: str, channel: int) -> dict[str, str | None]:
            """``channel_name`` / ``channel_kind`` for a record, or None when the channel is
            not on a live document (nothing to be honest about, so claim nothing)."""
            info = chinfo.get(live_doc_id, {}).get(int(channel))
            if info is None:
                return {"channel_name": None, "channel_kind": None}
            name, kind = channel_identity(info)
            return {"channel_name": name, "channel_kind": kind}

        layers = [LayerRec(doc_id=pkey.get(lay.doc_id, lay.doc_id), channel=lay.channel,
                           color=lay.color, visible=lay.visible,
                           time_offset_s=lay.time_offset_s, excluded=lay.excluded,
                           **_identity(lay.doc_id, lay.channel))
                  for lay in session.layers()]

        results: list[ResultRec] = []
        for res in session.results.all():
            if isinstance(res.payload, EventSet):
                results.append(ResultRec(
                    doc_id=pkey.get(res.doc_id, res.doc_id), channel=res.channel,
                    module_id=res.module_id, events=res.payload.df,
                    provenance=dict(res.payload.provenance or {}),
                    # The settings the run used, and the gain/offset it was computed under.
                    # Dropping either was a silent lie on reload: an empty params made
                    # "Re-run" re-detect with the PANEL'S defaults (wiping every curation),
                    # and a re-derived fingerprint made a genuinely stale result look fresh.
                    params=dict(res.params or {}),
                    calibration_fingerprint=res.calibration_fingerprint,
                    module_version=str(res.module_version or ""),
                    baseline_fingerprint=res.baseline_fingerprint,
                    # And the caveats those numbers were produced under — the
                    # unprobed-baseline advisory above all. See ResultRec.
                    warnings=tuple(res.warnings or ()),
                    **_identity(res.doc_id, res.channel)))

        # v6 -- DERIVED results (fit.dwell / fit.depth / fit.levels). The loop above
        # collects only EventSet payloads, which is why every fit a user ran was silently
        # dropped on save: the numbers were re-derivable from the events, but a minute of
        # bootstrap vanishing without a word is not a re-derivation, it is a loss. See
        # FitRec. A payload this build does not recognise is skipped rather than guessed at.
        fits: list[FitRec] = []
        for res in self._session.results.all():
            if isinstance(res.payload, EventSet):
                continue
            table = curve = None
            if isinstance(res.payload, pd.DataFrame):
                table = res.payload
            elif isinstance(res.payload, FitResult):
                curve = fit_to_json(res.payload)
            else:
                continue
            fits.append(FitRec(
                doc_id=pkey.get(res.doc_id, res.doc_id), channel=res.channel,
                module_id=res.module_id, table=table, curve=curve,
                params=dict(res.params or {}),
                calibration_fingerprint=res.calibration_fingerprint,
                baseline_fingerprint=res.baseline_fingerprint,
                warnings=tuple(res.warnings or ()),
                **_identity(res.doc_id, res.channel)))

        meas = self._analysis_panel.measure_panel.measurements_df
        measurements = meas if (meas is not None and len(meas)) else None

        # Capture after every legacy table has settled.  The cache merge preserves
        # unknown/plugin artifacts and pending unresolved built-ins byte-for-byte; only
        # live, successfully installed surfaces replace the stable id they own.
        workspace = self._capture_analytical_workspace()
        state = ProjectState(
            documents=documents, segments=segments, calibrations=calibrations, layers=layers,
            view_mode=self._session.view_mode.value, results=results,
            measurements=measurements, fits=fits,
            label_vocabulary=self._label_vocab.to_json(),
            colours=self._session.colours.to_json(),
            source_labels=self._session.source_labels.to_json(), workspace=workspace)

        # ...and everything the OPENED project held that the restore could not install,
        # written back verbatim. The live session is the wrong authority on what the project
        # CONTAINS: it holds what LOADED. See _capture_unrestorable.
        #
        # Merged BY FIELD NAME off the state itself, not by six hand-written lines: the
        # carry is computed by walking ProjectState's fields, so writing it back must walk
        # them too or a record type added to one side would be collected and then silently
        # dropped on its way to the file — which is the same defect one layer along.
        for name, extra in (self._carried_over or {}).items():
            current = getattr(state, name, None)
            if isinstance(current, list):
                current.extend(extra)
        return state

    def _save_project_from(self, requester: QtWidgets.QWidget, *, save_as: bool) -> bool:
        """Run the authoritative save flow for a secondary editing window.

        Gallery and Inspector are independent top-level windows on macOS, so a dialog
        parented to ``self`` can open behind the window that owns the user's attention.
        This short-lived pointer changes only dialog ownership; the project path, dirty
        state, conflict stamp, snapshot and write all remain MainWindow's.
        """
        previous = self._project_save_requester
        self._project_save_requester = requester
        try:
            return self._save_project_as() if save_as else self._save_project()
        finally:
            self._project_save_requester = previous

    def _project_save_parent(self) -> QtWidgets.QWidget:
        """The window in front of which the current save UI must appear."""
        return self._project_save_requester or self

    def _save_project(self) -> bool:
        """File ▸ Save Project (Ctrl+S). Re-saves the current project in place once it
        has a path; otherwise falls through to Save As. Returns True when a file was
        actually written (so the close guard can tell a real save from a cancel/failure)."""
        if self._project_path:
            return self._write_project(self._project_path)
        return self._save_project_as()

    def _save_project_as(self) -> bool:
        """File ▸ Save Project As… — always prompts for a .nanoscope path. Returns True when
        a file was written, False if the dialog was cancelled or the save failed."""
        recovery_target = self._recovery_suggested_project_path
        start = (str(Path(self._project_path).parent) if self._project_path else
                 str(Path(recovery_target).parent) if recovery_target else
                 self._last_project_dir)
        # A recovered snapshot never silently ADOPTS the old project's path: its disk
        # revision may have changed since the crashed process last saw it.  We do retain
        # that path as the chooser's suggestion, so the user can deliberately replace it
        # through Save As (and its existing overwrite confirmation) in one step.
        suggestion = (self._project_path or recovery_target or
                      str(Path(start) / "analysis.nanoscope"))
        # `outputs` applies the SAME .nanoscope normalisation the write does, so the
        # overwrite prompt is about the file that will actually be replaced — typing
        # "study" over an existing study.nanoscope has to warn, not sail past.
        chosen = ask_export_path(
            self._project_save_parent(), title="Save Project", suggestion=suggestion,
            filters="NanoScope project (*.nanoscope);;All files (*)",
            outputs=lambda p: (Path(_with_project_suffix(str(p))),))
        if chosen is None:
            return False
        return self._write_project(_with_project_suffix(str(chosen)))

    @staticmethod
    def _unrestorable_segments(state) -> list:
        """The :class:`SegmentRec`s in ``state`` that a reload could NOT re-derive.

        A segment is persisted as a spec and rebuilt from its parent on load, so it is
        restorable only if its ``parent_key`` names a document that is ALSO in this project —
        a saved file reference, or another restorable segment. Anything else is a spec pointing
        at nothing: it will be dropped on reload, along with every result keyed under it.
        Computed by fixpoint so a chain of nested segments resolves in either save order."""
        restorable = {d.path for d in state.documents}
        pending = list(getattr(state, "segments", ()) or ())
        progressed = True
        while pending and progressed:
            progressed = False
            still = []
            for s in pending:
                if s.parent_key in restorable:
                    restorable.add(s.segment_uid)
                    progressed = True
                else:
                    still.append(s)
            pending = still
        return pending

    def _confirm_save_dropping_segments(self, n_segments: int, n_results: int) -> bool:
        """Ask before writing a project that cannot carry ``n_segments`` segment(s). Returns
        True to write it anyway. Injectable so the suite never faces a live modal."""
        box = QtWidgets.QMessageBox(self._project_save_parent())
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Save Project")
        box.setText(f"{n_segments} segment(s) cannot be saved in this project.")
        box.setInformativeText(
            "A segment is stored as a crop of its parent recording and rebuilt from it on "
            "load, and their parent recording is no longer open — so they, and "
            f"{n_results} result(s) on them, would be missing when this project is reopened.\n\n"
            "Reopen the parent recording(s) and save again to keep them.")
        save_anyway = box.addButton("Save without them", QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is save_anyway

    # -- the concurrent-writer guard --------------------------------------------------------------
    @staticmethod
    def _disk_stamp_of(path: str) -> ProjectDiskStamp | None:
        """Content-aware identity of the project at ``path``, or None when absent.

        Size and nanosecond mtime are retained for diagnosis and backward familiarity;
        SHA-256 is the deciding fourth field, so a same-size rewrite that restores the old
        timestamp cannot pass for the revision this window loaded.
        """
        return project_disk_stamp(path)

    def _project_changed_under_us(self, path: str) -> bool:
        """Has the project file changed on disk since this window last read or wrote it?

        True only when all three hold: this window HAS a stamp (it read or wrote a project),
        ``path`` is that same file, and its metadata or content digest no longer matches. A
        file that is simply GONE is not a conflict — there is no newer save there to destroy,
        and the write will recreate it. A different target path is the Save-As overwrite
        prompt's business, not this guard's.
        """
        stamp = self._project_disk_stamp
        if stamp is not None and os.path.realpath(str(path)) != stamp.real_path:
            return False
        return project_changed(stamp, path)

    def _confirm_overwrite_newer_project(self, path: str) -> bool:
        """Ask before replacing a project file that changed under this window. Returns True
        to overwrite it anyway. Injectable so the suite never faces a live modal.

        THE QUESTION IS THE FIX for silent last-writer-wins: two windows on one
        ``.nanoscope`` (a supported shape — the packaged log lock exists so a later
        instance can run) used to revert each other's committed curation with zero
        dialogs, both status bars claiming their save was the file. The atomic layer
        holds (no interleaving, no corruption); what was missing was the honesty. The
        default is Cancel: the destructive answer is never the accident."""
        box = QtWidgets.QMessageBox(self._project_save_parent())
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Save Project")
        box.setText(
            f"{Path(path).name} has changed on disk since this window last read or "
            f"saved it.")
        box.setInformativeText(
            "Another NanoScope window — or another program — has written this project "
            "in the meantime. Saving now replaces that newer file with this window's "
            "state, and any work it holds that this window does not will be lost.\n\n"
            "To keep both, cancel and use File ▸ Save Project As… to save this window's "
            "work under another name, or reopen the project to see the newer version.")
        overwrite = box.addButton("Overwrite the newer file",
                                  QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is overwrite

    def _confirm_overwrite_existing_project(self, path: str) -> bool:
        """Ask before Save As replaces a project this window does not own.

        The native save dialog normally asks too, but its check is outside NanoScope's
        inter-process lock. Two windows can both choose a missing name and one can create
        it before the other's commit. This second, revision-specific question is what
        prevents that newly-created project from being overwritten silently.
        """
        box = QtWidgets.QMessageBox(self._project_save_parent())
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Save Project As")
        box.setText(f"{Path(path).name} now exists.")
        box.setInformativeText(
            "Another NanoScope window — or another program — may have saved this project "
            "after you chose the name. Replacing it will lose any work that file holds.\n\n"
            "To keep both, cancel and choose a different project name.")
        overwrite = box.addButton("Replace the existing file",
                                  QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is overwrite

    def _write_project(self, path: str) -> bool:
        """Serialize the session to ``path`` and adopt it as the current project. Returns
        True when a file was written; on failure the error is reported and False returned
        (the window stays open with the work untouched). A successful write clears the
        unsaved-changes flag and repaints the title.

        A save NEVER reports success while dropping work. A segment whose parent recording is
        closed cannot be re-derived on load, so it (and every result on it) would silently
        vanish — the old code wrote the file, said "Saved project (1 result(s))" and cleared the
        dirty flag while the reload came back empty. Now the user is asked first, and if she
        saves anyway the status line names exactly what was left behind.

        A save NEVER silently reverts somebody else's, either. The file's byte digest is
        compared against the stamp taken at this window's last load/save of it, and a
        mismatch — another window's Ctrl+S, most likely — raises
        :meth:`_confirm_overwrite_newer_project` before anything is replaced. The final
        comparison and atomic replacement share one inter-process lock. Confirmation
        approves one exact conflicting revision; a third write while the question is open
        raises the question again rather than becoming a lost update."""
        from ..core.project import save_project

        # Approval is local to this save attempt and NEVER changes the window's remembered
        # revision. Only a successful commit below may do that. The stamp names the exact
        # conflicting bytes the user agreed to replace; if another writer lands after the
        # dialog, guarded_project_write raises with that third revision and we ask again.
        held_stamp = self._project_disk_stamp
        same_held_target = (
            held_stamp is not None
            and os.path.realpath(str(path)) == held_stamp.real_path
        )
        expected_stamp = held_stamp if same_held_target else None
        # ``None`` used to mean both "there was no baseline" and "the baseline was
        # absent". The distinction matters when two first-time Save As operations choose
        # the same missing name: once one creates it, the other must conflict.
        expected_absent = not same_held_target
        confirm_conflict = (
            self._confirm_overwrite_newer_project
            if same_held_target else self._confirm_overwrite_existing_project
        )
        approved_stamp: ProjectDiskStamp | None = None
        approved_dropped: tuple[tuple[str, ...], int] | None = None
        while True:
            # Preserve the old user-visible order: if a conflict already exists, ask before
            # checkpointing or presenting any unrelated "save without segments" question.
            # This is only a preflight; correctness comes from the second comparison inside
            # guarded_project_write's lock immediately before the atomic write.
            try:
                with project_save_lock(path):
                    current_stamp = self._disk_stamp_of(path)
            except Exception as exc:  # noqa: BLE001 — a save failure is reported, never crashes
                QtWidgets.QMessageBox.warning(
                    self._project_save_parent(), "Save Project",
                    f"Could not save the project:\n{exc}")
                return False
            if project_changed(expected_stamp, path, current_stamp,
                               expected_absent=expected_absent) \
                    and current_stamp != approved_stamp:
                if not confirm_conflict(path):
                    return False
                approved_stamp = current_stamp
                continue

            # Commit anything edited in place that never announced itself, BEFORE the state
            # is gathered. core/history.py promises that a forgotten announcement costs a
            # menu word and never a row, and this is one of the two moments that promise is
            # about: what goes into the file and what the stack can put back must agree.
            # A conflict dialog runs a nested Qt event loop, so a successful confirmation
            # loops back here and gathers afresh rather than saving a pre-dialog snapshot.
            self._checkpoint_everything()
            write_revision = self._session_revision
            try:
                state = self._gather_project_state()
            except Exception as exc:  # noqa: BLE001 — a save failure is reported, never crashes
                QtWidgets.QMessageBox.warning(
                    self._project_save_parent(), "Save Project",
                    f"Could not save the project:\n{exc}")
                return False

            dropped = self._unrestorable_segments(state)
            n_dropped_results = 0
            if dropped:
                lost_keys = {s.segment_uid for s in dropped}
                # state.fits counts too. A fit on an unrestorable segment is written into the
                # file and can never be re-attached (its segment cannot be re-derived), so it
                # is dropped work — and this dialog exists for the sole purpose of naming the
                # work being left behind. Counting only EventSets under-reported by exactly the
                # artefacts v6 made persistable: a bootstrap dwell fit and a CUSUM level table
                # went unmentioned while the user was told "1 result(s)".
                n_dropped_results = sum(1 for r in state.results if r.doc_id in lost_keys)
                n_dropped_results += sum(1 for f in state.fits if f.doc_id in lost_keys)
                dropped_signature = (
                    tuple(sorted(str(s.segment_uid) for s in dropped)), n_dropped_results)
                if dropped_signature != approved_dropped:
                    if not self._confirm_save_dropping_segments(
                            len(dropped), n_dropped_results):
                        return False
                    approved_dropped = dropped_signature

            # `notes` collects what the write changed about the FILE ITSELF — a symlink
            # followed to its destination, or other hard-linked names that now hold the
            # previous contents. An atomic replace installs a new inode, and that is the one
            # thing it cannot preserve; the alternative is writing in place, which is the
            # destructive path this save exists to avoid. Empty on an ordinary save.
            file_notes: list[str] = []
            try:
                _, committed_stamp = guarded_project_write(
                    path,
                    expected_stamp,
                    approved_stamp,
                    lambda: save_project(path, state, file_notes),
                    expected_absent=expected_absent,
                )
            except ProjectSaveConflict as conflict:
                # The target changed after preflight/gather. Ask outside the lock so another
                # window is never frozen behind a human-facing modal, then loop to re-gather
                # state. Approval applies only to the revision carried by this exception.
                if not confirm_conflict(path):
                    return False
                approved_stamp = conflict.current_stamp
                continue
            except Exception as exc:  # noqa: BLE001 — a save failure is reported, never crashes
                QtWidgets.QMessageBox.warning(
                    self._project_save_parent(), "Save Project",
                    f"Could not save the project:\n{exc}")
                return False
            break

        self._project_path = path
        self._recovery_suggested_project_path = None
        # Use the stamp captured BEFORE the lock was released. Re-reading here would let a
        # fast third writer land in between and be mistaken for this window's own revision.
        self._project_disk_stamp = committed_stamp
        self._last_project_dir = str(Path(path).parent)
        self._add_recent_project(path)
        self._explicit_saved_revision = max(
            self._explicit_saved_revision, int(write_revision))
        if dropped:
            # A SAVE THAT DROPPED WORK DOES NOT REPORT SUCCESS. The user was warned and
            # chose "Save without them", so the file is written — but the segments and
            # their results are STILL LIVE in this session and STILL unrecoverable (a
            # segment has no reopen path), and the file on disk does not contain them.
            # Clearing the flag here retired the quit guard on the one session that most
            # needed it: the title lost its `•` and quitting asked nothing, so the work
            # the modal had just named went out with the window. Staying dirty means
            # reopening the parent recording and pressing Save again still costs one
            # click, and closing without doing so still asks.
            self._update_title()
        else:
            # Retire only snapshots no newer than the state that actually reached disk.
            # An edit arriving through a nested Qt loop after the snapshot was gathered
            # remains dirty and its newer recovery generation remains live.
            self._retire_recovery_through(write_revision)
            if self._session_revision == write_revision:
                self._mark_clean()   # the written revision is still the live revision
            else:
                self._update_title()
        kept = len(state.results) + len(state.fits) - n_dropped_results
        msg = f"Saved project ({kept} result(s)) to {Path(path).name}."
        carried = self._carried_over or {}
        n_carried = len(carried.get("documents", []))
        if n_carried:
            # Say it, every time. The recording is not on screen, so without this line a
            # save looks like it wrote only what is visible — which is precisely the
            # impression under which the old behaviour destroyed the reference.
            # "not open in this session", not "could not be opened": the restore has more
            # than one way of leaving a recording behind, and one of them is the user
            # pressing Cancel — those were never attempted, so naming them as unopenable
            # blamed the disk for something the app was told to do.
            msg += (f" {n_carried} recording(s) that are not open in this session were kept "
                    f"in the project, not dropped.")
        # The same sentence for everything ELSE the restore could not install — a segment
        # whose window no longer fits, a record naming a channel that is gone, a table this
        # build could not read. Counting only documents was how the enumerated carry looked
        # correct while erasing the other six exits: the status line said "0 result(s)" and
        # nothing said the four curated events had just left the file.
        n_other = sum(len(v) for k, v in carried.items() if k != "documents")
        if n_other:
            msg += (f" {n_other} record(s) this project holds but could not be re-attached "
                    f"were kept in it, not dropped.")
        if dropped:
            msg += (f" {len(dropped)} segment(s) and {n_dropped_results} result(s) were NOT "
                    f"saved — their parent recording is closed.")
        self.statusBar().showMessage(msg, 12000 if dropped else 8000)
        fresh = [n for n in file_notes
                 if (str(path), n) not in self._disclosed_file_notes]
        self._disclosed_file_notes.update((str(path), n) for n in file_notes)
        if fresh:
            QtWidgets.QMessageBox.information(
                self._project_save_parent(), "Save Project",
                f"The project was saved.\n\n" + "\n\n".join(fresh))
        return True

    def _disclose_identity_notes(self, title: str, paths) -> None:
        """Tell the user what a finished write changed about the FILE, not its contents.

        THE PROJECT SAVE WAS THE ONLY SURFACE THAT DID THIS, and only because it happens
        to pass a ``notes`` list into :func:`save_project`. Every other write in the app —
        the events CSV and its sidecar, each exported figure, every ``Save image…`` —
        goes through the same atomic replace, breaks hard links and follows symlinks in
        exactly the same way, and said nothing. So a researcher could hard-link an
        exported ``events.csv`` into a manuscript folder, re-export after curating two
        events, and be left with a manuscript copy silently holding the OLD table, while
        the project save in the same session disclosed the identical situation.

        :func:`atomic_write.take_identity_notes` is the seam that makes this cheap: the
        ledger is filled by the write itself, so a surface only has to ask for the notes
        belonging to the files IT wrote — anything pending for another file stays pending,
        and draining means a note is never shown twice. Scoped by ``paths`` for that
        reason; never call it unscoped from an export.
        """
        from ..core import atomic_write

        notes = atomic_write.take_identity_notes(paths)
        fresh = [n for n in notes if (title, n) not in self._disclosed_file_notes]
        self._disclosed_file_notes.update((title, n) for n in notes)
        if fresh:
            QtWidgets.QMessageBox.information(self, title, "\n\n".join(fresh))

    def _open_project_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", self._last_project_dir,
            "NanoScope project (*.nanoscope);;All files (*)")
        if path:
            self._open_project(path)

    def _open_project(self, path: str) -> None:
        """Load a .nanoscope and drive the async restore. A load-time error (not a
        project, corrupt, unsupported version) is reported and aborts before any
        Session mutation.

        Opening a project REPLACES the session; it never merges into it. Two things follow,
        and both used to be missing:

        * unsaved work in the current session is guarded FIRST (Save / Discard / Cancel), so
          the open can never silently discard analysis the user has not written anywhere;
        * the session is then emptied, so the restore starts from nothing. Restoring on top of
          the live session skipped every document that happened to be open — presenting stale
          in-memory calibration/detections AS the project and then marking the whole thing
          clean, so the next Ctrl+S overwrote the good file with the stale state. It also
          merged two projects into one whenever the second was opened over the first.
        """
        from ..core.project import ProjectError, load_project

        try:
            # Read and stamp ONE revision. Without the shared project lock another
            # NanoScope process can commit after load_project returns but before the
            # stamp is taken; this window would then hold the old state while claiming
            # the new revision and could silently overwrite it on its next save.
            state, loaded_stamp = locked_project_read(
                path, lambda: load_project(path))
        except ProjectError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Open Project", f"Could not open the project:\n{exc}")
            return
        except Exception as exc:  # noqa: BLE001 — a corrupt archive must not crash the app
            QtWidgets.QMessageBox.warning(
                self, "Open Project", f"Could not open the project:\n{exc}")
            return

        # The file is readable; only now may the current session be disturbed.
        if not self._offer_to_save_unsaved_work():
            return   # Cancel, or a Save the user cancelled / that failed — nothing is touched.
        self._clear_session()

        self._project_path = path
        # This is the stamp captured under the SAME lock as the read above. A writer that
        # lands after the lock is released must remain different and raise a later save
        # conflict; re-reading here would incorrectly bless that newer revision.
        self._project_disk_stamp = loaded_stamp
        self._last_project_dir = str(Path(path).parent)
        self._add_recent_project(path)
        # The opened project is clean by definition; clear now (before the async restore,
        # whose own mutations are guarded) so the title never flashes a stale marker.
        self._mark_clean()
        self._begin_restore(state)

    def _offer_to_save_unsaved_work(self) -> bool:
        """Guard an action that will DISCARD the current session (today: Open Project).

        Returns True when it is safe to proceed — nothing to lose, the user chose Discard, or
        the user chose Save AND a file was actually written. Returns False when the user
        cancelled or the save did not happen, in which case the caller must leave the session
        exactly as it found it. Routes through the same :meth:`_confirm_close_choice` /
        :meth:`_run_save` seams the close guard uses, so there is ONE unsaved-work dialog.
        """
        if not (self._dirty and self._has_work_to_lose()):
            return True
        choice = self._confirm_close_choice()
        if choice == _CLOSE_CANCEL:
            return False
        if choice == _CLOSE_SAVE:
            return bool(self._run_save())
        # An explicit Discard means the recovery copy must not offer the same work again
        # on the next launch.  Retire it before replacing the session.
        self._discard_recovery(self._recovered_candidate)
        return True

    def _clear_session(self) -> None:
        """Empty the session so a project restore starts from nothing.

        Stops every job in flight (and drops anything queued behind it) so neither a stray
        document nor a late detection can land in the middle of the incoming restore, closes
        every open document — which drops its layers, calibrations and results through the
        normal Session teardown — and clears the manual measurements.

        Two deliberate omissions. It is NOT a dirty-flag operation: the caller has already dealt
        with unsaved work and sets the flag it wants afterwards. And it closes documents
        straight through the Session rather than via :meth:`_close_document`, so the
        parent-has-segments guard does not re-ask per recording — the caller asked once, about
        the whole session.
        """
        recovery_controller = self._recovery_controller
        if recovery_controller is not None:
            reset = getattr(recovery_controller, "reset_session", None)
            if callable(reset):
                reset()
        # All three emit nothing; we reset the UI ourselves — and we use what they
        # report. A run that did NOT stop is detached: its answer is discarded, but its
        # thread is still burning cores behind the project the user is about to open,
        # and the app has been silent about that. Unlike the quit path there is somebody
        # here to tell, so tell them.
        stopped = self._controller.abort_stopped()
        stopped &= self._analysis_controller.abort_stopped()
        stopped &= self._batch_controller.abort_stopped()
        if not stopped:
            self.statusBar().showMessage(
                "A background run would not stop in time. Its answer is discarded, but it "
                "is still finishing in the background — the new project may open slowly.",
                8000)
        self._open_queue.clear()
        self._active_open = None
        self._reset_analytical_workspace()
        # The outgoing project's unloadable records belong to the outgoing project.
        self._carried_over = None
        self._batch_skipped = 0
        self._set_progress_visible(False)
        for doc in self._session.documents():
            self._session.close_document(doc.doc_id)
        self._analysis_panel.measure_panel.set_measurements_df(None)
        # The stack is per-session state: with every document closed there is nothing left
        # for any of its entries to be applied to, and an entry that outlived its recording
        # would be an undo the user could see offered and never see happen.
        self._history.clear()
        self._measure_history.clear()
        self._undo_journal.clear()
        self._redo_journal.clear()
        # The whole-recording baseline memo is per-trace session state like everything else
        # here, and it had no caller anywhere in `nanoscope/`. It is keyed on a trace's
        # object identity through a weakref, so this is delayed RELEASE rather than a stale
        # hit (95.8 MB of float64 baselines outlived the owner's closed recording) — but a
        # memo that survives a session reset is a wrong number waiting for an id() the
        # allocator hands back, and one line here costs nothing.
        from ..analysis.baseline import clear_baseline_cache
        clear_baseline_cache()
        self._current_selection = None
        # Colour identity is project state, not an application preference. The Session
        # object itself is reused across Open Project, so its registry must not carry the
        # outgoing project's identities into the incoming one.
        self._session.colours = ColourRegistry()
        from ..core.source_style import SourceLabelRegistry
        self._session.source_labels = SourceLabelRegistry()
        self._project_path = None
        self._project_disk_stamp = None   # no project held -> nothing to conflict with
        self._recovery_suggested_project_path = None
        self._recovered_candidate = None
        self._restore_is_recovery = False
        self._info_bar.hide()

    # -- async restore ---------------------------------------------------------------------------
    def _begin_restore(self, state) -> None:
        """Enter restore mode and reopen every referenced document through the SAME
        sequential open pipeline, each at its saved range. Missing sources are skipped
        up front (recorded for a warning); per-doc state is applied as each lands.

        A source that EXISTS is not the same statement as a source that is the one the
        project was saved from, and the difference is not academic: re-exporting from
        Clampex over the same filename used to reattach 5 saved events to a 5,000-sample
        flat trace — all 5 past the end of it, event 0 claiming a 30 pA blockade in a signal
        with a 1.1 pA standard deviation — in complete silence. Each ref's saved fingerprint
        is carried through to :meth:`_apply_restore_for_doc`, which compares it against the
        file and the reopened document (see :mod:`nanoscope.core.identity`). The project
        still opens either way: this warns and explains, and never refuses.
        """
        self._restoring = True
        self._history.set_suspended(True)   # a reconstruction is not an undoable edit
        self._measure_history.set_suspended(True)
        self._restore_state = state
        workspace = getattr(state, "workspace", None)
        self._analytical_workspace = (
            workspace if isinstance(workspace, AnalyticalWorkspace)
            else AnalyticalWorkspace())
        self._workspace_pending = {
            item.artifact_id for item in self._analytical_workspace.artifacts
            if item.artifact_id in OWNED_ARTIFACT_IDS
        }
        self._workspace_frozen.clear()
        self._sync_saved_subpop_action()
        # Install the registry BEFORE the first document creates a layer. add_layer asks it
        # for the persistent recording/channel key, so every trace and plot starts in the
        # saved ink instead of flashing or retaining a positional colour.
        self._session.colours = ColourRegistry.from_json(
            getattr(state, "colours", None))
        from ..core.source_style import SourceLabelRegistry
        self._session.source_labels = SourceLabelRegistry.from_json(getattr(state, "source_labels", None))
        self._restore_ranges = {}
        self._restore_sources = {}
        self._restore_missing = []
        self._restore_refusals = {}
        self._restore_lost_segments = []
        self._restore_bad_calibrations = []
        self._restore_notes = []
        self._restore_channel_moved = []
        self._restore_channel_lost = []
        self._restore_path_moved = []
        self._restore_cancelled = []
        self._restored_records = set()
        self._info_bar.hide()

        to_open: list[str] = []
        for d in state.documents:
            p = Path(d.path)
            if not p.exists():
                self._restore_missing.append(d.path)
                continue
            rng = SampleRange(d.loaded_range[0], d.loaded_range[1]) if d.loaded_range else None
            self._restore_ranges[str(p.resolve())] = rng
            self._restore_sources[str(p.resolve())] = getattr(d, "source", None)
            to_open.append(d.path)

        if to_open:
            self.open_paths(to_open)
        # Nothing to open (every source missing, or an empty project): finalise now so the
        # measurements/view/warning still land and projectRestored still fires.
        if self._active_open is None and not self._open_queue:
            self._finalize_restore()

    def _rederive_segments(self, state) -> None:
        """Re-derive every saved SEGMENT from its (now-open) parent, then attach its saved
        calibrations / layers / results by persistent key.

        Parents are re-opened first (file docs, through the normal pipeline); a segment is
        rebuilt with :meth:`Session.create_segment` passing the SAVED uid, so it lands on the
        same identity its records were keyed under. A fixpoint pass handles nesting (a segment
        cut from another segment): each round derives every segment whose parent is now
        available, until no more progress is made. Segments whose parent never appeared (a
        missing source, or a broken chain) are recorded SEPARATELY from missing source files:
        they are not files, they were never on disk, and running them through the
        "source file(s) could not be found" banner (which then took ``Path(...).name`` of a
        sentence) told the user a straightforwardly false story about what went wrong."""
        specs = list(getattr(state, "segments", ()) or ())
        if not specs:
            return
        from ..core.segment import persistent_key, spec_window_refusal

        # persistent key -> live doc_id for everything currently open (parents so far).
        live: dict[str, str] = {persistent_key(d): d.doc_id for d in self._session.documents()}

        pending = specs
        progressed = True
        while pending and progressed:
            progressed = False
            still: list = []
            for spec in pending:
                parent_live = live.get(spec.parent_key)
                if parent_live is None or not self._session.has_document(parent_live):
                    still.append(spec)          # parent not (yet) available — retry next round
                    continue
                # THE PRECONDITION. A saved SegmentRec is an ASSERTION about a particular
                # stretch of a particular recording, not a request to crop; the builder
                # clamps because a live rubber-band drag genuinely is a request. Measured
                # on the sibling path: a "cut" of [40 000, 60 000) whose parent was
                # re-exported at 45 000 samples came back as [40 000, 45 000) — a QUARTER
                # of the analysed window — with meta.issues == (), a note stating only the
                # new window, all 4 saved events reattached and 3 of them lying entirely
                # past the end of the segment they were attached to, their dwell/depth/ECD
                # measured on data no longer on disk. On a project written without a source
                # fingerprint (an older project, which MIN_READ_VERSION = 1 guarantees
                # still loads) there was no banner at all. It refuses and is reported
                # instead; the spec and its results are then carried by
                # `_capture_unrestorable`, so the next save cannot rewrite i1 to 45 000 and
                # cannot drop the four curated events.
                parent_doc = self._session.document(parent_live)
                why = spec_window_refusal(
                    int(parent_doc.meta.n_samples), spec.i0, spec.i1)
                if why is not None:
                    self._restore_lost_segments.append(f"“{spec.name}” ({why})")
                    progressed = True           # consumed (as a refusal) — do not retry
                    continue
                try:
                    seg_id = self._session.create_segment(
                        parent_live, spec.i0, spec.i1, name=spec.name,
                        segment_uid=spec.segment_uid)
                except (ValueError, TypeError, KeyError) as exc:
                    self._restore_lost_segments.append(f"“{spec.name}” ({exc})")
                    progressed = True           # consumed (as a failure) — do not retry
                    continue
                live[spec.segment_uid] = seg_id
                self._restored_records.add(id(spec))    # INSTALLED — see _capture_unrestorable
                self._apply_restore_for_doc(seg_id, match_key=spec.segment_uid)
                progressed = True
            pending = still
        for spec in pending:   # parents never showed up (missing file, or a broken parent chain)
            self._restore_lost_segments.append(
                f"“{spec.name}” (its parent recording is not in this project)")

    def _apply_restore_for_doc(self, doc_id: str, match_key: str | None = None) -> None:
        """Apply the saved calibrations, layers and results for ONE just-landed document.

        Records are matched by the document's PERSISTENT KEY but applied to its LIVE
        ``doc_id``. The key is DERIVED from the document (``segment.persistent_key``: its
        source path for a file, its uid for a segment) and never defaulted to ``doc_id``,
        because a ``doc_id`` now carries this session's per-load token
        (:mod:`nanoscope.core.identity`) and can therefore never equal anything a previous
        session wrote down. ``match_key`` is still accepted for the segment path, which
        knows its uid before the document exists.

        Order matters: calibrations first (so a restored result's calibration fingerprint
        is computed against the SAME gain/offset it had when saved), then the layers, then
        the results.

        WHICH CHANNEL a record belongs to is decided by ``calibration.resolve_channel``,
        never by its stored index alone. An index is a position: re-export a recording with
        its signal columns in the other order and the index-only restore put a
        ``gain=0.5, offset=-1.0, 'pA'`` calibration on the VOLTAGE channel — the -600 mV
        trace read -301.0 pA — left the real current channel at gain 1.0, reattached an
        11-event detection to the voltage trace, and, because the result's saved
        fingerprint matched the misplaced live calibration, positively CERTIFIED those
        events as current-valid. A v7 record names its channel, so it is re-attached by
        identity when the order moved (said out loud) and NOT applied when the channel is
        gone; a pre-v7 record carries no identity and resolves by index exactly as before,
        silently.

        EVERY INSTALL STAMPS ``_restored_records``. That set, not a list of remembered
        failure reasons, is what :meth:`_capture_unrestorable` subtracts from the project
        to decide what the next save must carry — so a record this method skips for ANY
        reason, including one written next year, is kept rather than erased."""
        from ..analysis.events import EventSet
        from ..analysis.stats import fit_from_json
        from ..core.analysis import ANALYSIS_REGISTRY, AnalysisResult
        from ..core.calibration import resolve_channel
        from ..core.segment import persistent_key

        state = self._restore_state
        if state is None or not self._session.has_document(doc_id):
            return
        doc = self._session.document(doc_id)
        key = match_key if match_key is not None else persistent_key(doc)
        channels = doc.meta.channels
        where = doc.display_name

        def _belongs(stored_key: object) -> bool:
            """Is this stored record about the recording that just landed?

            *** RECORDS ARE MATCHED BY RECORDING IDENTITY, NEVER BY PATH SPELLING. ***

            The project stores the path as it resolved AT SAVE TIME; ``key`` is how the same
            file resolves NOW. Those two strings stop being equal the moment the path picks
            up a symlink between sessions — move raw data to a NAS or an archive folder and
            leave a stub behind so every old path still works, which is exactly why people
            leave one. The plain ``==`` this replaces then matched nothing, and every
            calibration, layer and detection saved for that recording was ``continue``d in
            silence while the banner said "The rest of the project was restored."

            :func:`~nanoscope.core.identity.same_source` is THE answer to "are these two
            source keys the same recording?" — it is what ``open_paths``/``_already_open``
            already use so two spellings of one file are one recording, and it asks the
            FILESYSTEM (``os.path.samefile``) rather than transforming strings. It costs
            nothing in the common case: identical strings short-circuit before any syscall.

            A match that needed the filesystem is a match under a DIFFERENT spelling than the
            project holds, and that is said out loud (``_restore_path_moved``) — the same
            treatment a record re-attached by channel name gets. Two surfaces must not
            disagree about which path this recording lives at.
            """
            stored = str(stored_key)
            if stored == key:
                return True
            if not same_source(stored, key):
                return False
            said = f"“{where}” (saved as {stored})"
            if said not in self._restore_path_moved:
                self._restore_path_moved.append(said)
            return True

        def _channel_for(rec, what: str) -> int | None:
            """The live channel index ``rec`` belongs to, or None (and say why, once)."""
            idx, moved = resolve_channel(
                channels, rec.channel,
                getattr(rec, "channel_name", None), getattr(rec, "channel_kind", None))
            if moved is not None:
                sink = (self._restore_channel_moved if idx is not None
                        else self._restore_channel_lost)
                # Once per sentence: a document that was already open reaches this method by
                # a second route (see _note_source_change's own POP for the same reason).
                said = f"“{where}” {what} ({moved})"
                if said not in sink:
                    sink.append(said)
            return idx

        for d in getattr(state, "documents", ()) or ():
            if _belongs(d.path):
                self._restored_records.add(id(d))

        for c in state.calibrations:
            if not _belongs(c.doc_id):
                continue
            ch = _channel_for(c, "calibration")
            if ch is None:
                continue
            # A project is a file on disk and can be edited or damaged, so a non-finite or
            # zero gain can arrive here. ChannelCalibration refuses it (that is the point);
            # the restore keeps the channel's seeded default and says so, rather than aborting
            # the whole reopen over one bad number.
            #
            # `to_calibration` is handed the channel's CURRENT calibration — the sidecar's,
            # loaded moments ago by Session.add_document — because a pre-v5 project carries
            # no `note` and must not be read as asserting there is none. See CalibrationRec.
            try:
                cal = c.to_calibration(self._session.calibration(doc_id, ch))
            except ValueError:
                # display_name, not Path(doc_id).name: a doc_id now carries a per-load token
                # (core.identity) and a segment's path is synthetic, so neither is something
                # to show a user.
                self._restore_bad_calibrations.append(
                    f"{self._session.document(doc_id).display_name} ch{ch}")
                continue
            # write_sidecar=False: a RESTORE is reconstruction, not a calibration edit. Letting
            # it fall through to save_sidecar rewrote (or, at default values, DELETED) the
            # user's <datafile>.npcal.json next to their raw data — destroying its `note`, which
            # a project cannot carry. Restoring a project must never touch anything outside it.
            self._session.set_calibration(doc_id, ch, cal, write_sidecar=False)
            self._restored_records.add(id(c))

        for lay in state.layers:
            if not _belongs(lay.doc_id):
                continue
            ch = _channel_for(lay, "layer")
            if ch is None:
                continue
            layer = self._session.add_layer(doc_id, ch)
            self._session.set_layer_color(layer.layer_id, lay.color, update_registry=False)
            # `excluded` first: it drives visibility, and a project that predates it
            # restores False, leaving `visible` exactly as it was written.
            self._session.set_layer_excluded(layer.layer_id, lay.excluded)
            self._session.set_layer_visible(layer.layer_id, lay.visible and not lay.excluded)
            self._session.set_layer_time_offset(layer.layer_id, lay.time_offset_s)
            self._restored_records.add(id(lay))

        for r in state.results:
            if not _belongs(r.doc_id):
                continue
            ch = _channel_for(r, "detection")
            if ch is None:
                continue
            try:
                events = EventSet(r.events, dict(r.provenance or {}))
            except ValueError:
                continue   # a malformed table is skipped, never aborts the restore
            # The fingerprint the result was COMPUTED under, restored from the project —
            # never re-read from the live calibration. That re-derivation silently declared
            # every reopened result fresh: a detection run at gain 1 and then re-gained to 2
            # came back with the "calibration changed" banner GONE while every depth on
            # screen was still the gain-1 number. A pre-v3 project stored no stamp, so it
            # falls back to the (just-restored) calibration, which is what it meant.
            fp = r.calibration_fingerprint
            if fp is None:
                fp = self._session.calibration(doc_id, ch).fingerprint()
            module = ANALYSIS_REGISTRY.get(r.module_id)
            version = str(r.module_version or (
                getattr(module, "version", "") if module is not None else ""))
            # overlays=() — the SPAN is rebuilt from the (curated) df by the AnalysisPanel;
            # the stored EventSet IS the result, so no detection re-runs.
            # params: the settings the run used, so a re-run reproduces THIS detection
            # instead of whatever the panel happens to be showing.
            # THE OTHER STALENESS STAMP, restored the same way and for the same
            # reason. It travels inside the EventSet's own provenance (the sole
            # writer is `BaselineResult.provenance`, and a project persists the
            # provenance verbatim), so no project VERSION bump is needed and a v1/v2
            # project simply has no key -- which is exactly the "absence IS the
            # signal" case: those rows were measured by the 10 ms block median.
            #
            # Dropping it on restore is the same defect C6 fixed for the calibration
            # stamp, one level worse: not re-deriving it from the live config but
            # losing it entirely, so EVERY reopened project read as "no estimator
            # named" -- indistinguishable from a genuinely pre-change table, and the
            # staleness banner could never clear for a project that was in fact
            # freshly measured.
            res = AnalysisResult(
                module_id=r.module_id, module_version=version, doc_id=doc_id,
                channel=ch, params=dict(r.params or {}),
                calibration_fingerprint=tuple(float(v) for v in fp),
                payload=events, overlays=(), scalars={},
                warnings=tuple(r.warnings or ()),
                baseline_fingerprint=(
                    r.baseline_fingerprint
                    if r.baseline_fingerprint is not None
                    else _stored_baseline_fingerprint(r.provenance)))
            self._session.add_result(res)
            self._restored_records.add(id(r))

        # v6 -- the derived fits. Both staleness stamps come back from the FILE, never from
        # the live calibration/baseline, for the reason the block above spells out: a
        # re-derived stamp makes a stale fit read as fresh. `curve` is rebuilt through
        # stats.fit_from_json, which reconstructs the density with the same function the
        # fitter used, so a reopened fit draws the curve its own tau/AIC describe.
        for fr in getattr(state, "fits", ()):
            if fr.doc_id != key:
                continue
            ch = _channel_for(fr, "fit")
            if ch is None:
                continue
            if fr.table is not None:
                payload: object = fr.table
            elif fr.curve is not None:
                try:
                    payload = fit_from_json(fr.curve)
                except (KeyError, TypeError, ValueError):
                    continue   # a malformed fit is skipped, never aborts the restore
            else:
                continue
            fp = fr.calibration_fingerprint
            if fp is None:
                fp = self._session.calibration(doc_id, ch).fingerprint()
            self._session.add_result(AnalysisResult(
                module_id=fr.module_id, module_version="", doc_id=doc_id,
                channel=ch, params=dict(fr.params or {}),
                calibration_fingerprint=tuple(float(v) for v in fp),
                payload=payload, overlays=(), scalars={},
                warnings=tuple(fr.warnings or ()),
                baseline_fingerprint=fr.baseline_fingerprint))
            self._restored_records.add(id(fr))

        self._note_source_change(doc_id)

    #: "This document was never part of the restore (or its sentence is already said)" —
    #: distinct from ``None``, which means "part of the restore, but the project carries
    #: no fingerprint". Conflating the two is exactly how the unstamped case went silent.
    _NO_RESTORE_SOURCE = object()

    def _note_source_change(self, doc_id: str) -> None:
        """Say, once per reopened FILE, when the events no longer fit the recording — and,
        when a fingerprint proves it, that it is not the file the project was saved from.

        THE POINT IS THE CONSEQUENCE, not the metadata. "The file changed" is a sentence a
        user clicks past; "5 of your 5 saved events now lie past the end of this recording"
        is one they act on, and it is the thing that was actually silent. So the note names
        what differs AND counts the restored rows that can no longer be true.

        COMPOSED BY :func:`~nanoscope.core.identity.describe_restored_source` — the seam
        built (and pinned by ``tests/core/test_s4_identity_restored_source.py``) precisely
        so the two halves are gated separately: the "this file changed" half needs
        EVIDENCE (a saved fingerprint; silence still means "no evidence", never "verified
        identical" — a false alarm on every old project is how a real one gets ignored),
        while the events-past-the-end half is a comparison between the project's own saved
        indices and the sample count now in hand, which needs no stamp at all. This method
        used to gate BOTH on the stamp: it early-returned for a stamp-less project
        (every project written before manifest v4 — MIN_READ_VERSION is frozen at 1 so
        they all still open — and any project whose source could not be read at save
        time), so a re-exported, shorter recording came back with its whole event table
        reattached, rows lying past the end of the file, and not one word anywhere.
        """
        from ..core.identity import describe_restored_source, fingerprint_source
        from ..core.segment import SegmentDocument

        if not self._session.has_document(doc_id):
            return
        doc = self._session.document(doc_id)
        if isinstance(doc, SegmentDocument):
            return   # a segment is re-derived from its parent; it has no file of its own
        # POP, not get: one sentence per file however many times the restore touches it
        # (a document that was already open reaches _apply_restore_for_doc by a second route).
        # The sentinel keeps "already said / not restored" apart from "restored, unstamped":
        # only the former is silence.
        saved = self._restore_sources.pop(doc.source_key, self._NO_RESTORE_SOURCE)
        if saved is self._NO_RESTORE_SOURCE:
            return
        # With no stamp there is nothing a re-read could establish (the seam's contract
        # says callers may skip it); with one, the re-read is the evidence.
        now = fingerprint_source(doc.source_key) if saved is not None else None

        n = int(doc.meta.n_samples)
        rows = past = 0
        for res in self._session.results.for_doc(doc_id):
            df = getattr(getattr(res, "payload", None), "df", None)
            if df is None or "end_idx" not in df:
                continue
            ends = df["end_idx"].to_numpy()
            rows += int(len(df))
            past += int((ends > n).sum())
        note = describe_restored_source(
            doc.display_name, saved, now,
            loaded_n_samples=n, restored_events=rows, events_past_end=past)
        if note:
            self._restore_notes.append(note)

    def _capture_unrestorable(self, state) -> None:
        """Remember every record in the OPENED project that the restore did NOT install, so
        that the next save writes it back out instead of dropping it.

        COMPUTED AS A DIFF — the project minus what was actually installed — and that is
        the whole fix, not a tidier spelling of the old one. It used to be
        ``lost = set(self._restore_missing)``: ONE enumerated reason (the file was not on
        disk), while the restore has at least seven exits. Measured on two of them:

        * A segment whose saved window no longer fits its parent. The banner names it; the
          carry did not. Ctrl+S then wrote ``segments=[]``, ``results=[]`` and reported
          *"Saved project (0 result(s))"* — the segment's definition and four curated
          events gone from the file permanently, unrecoverable even by restoring the
          original recording from backup, because the manifest no longer describes them.
        * A record naming a channel the reopened file no longer has. Both the calibration
          and an 11-event detection were ``continue``d in SILENCE, and the next save
          dropped both while the banner mentioned only the file's byte size.

        The others (a malformed EventSet, a non-finite stored calibration, unparseable fit
        JSON, and the four channel guards separately) were the same shape, and enumerating
        reasons is what has come back on this project three rounds running. So the question
        asked here is not "why might a record be missing" but "which records are missing",
        and the answer needs no maintenance: :meth:`_apply_restore_for_doc` and
        :meth:`_rederive_segments` stamp ``_restored_records`` at the line that installs,
        and everything else in the project is carried. A skip path written next year is
        covered on the day it is written; so is a record TYPE nobody here remembered,
        because the fields are walked off the dataclass rather than listed.

        The old transitive fixpoint for nested segments is gone with it, and its
        disappearance is a check on the shape: a segment whose parent never loaded is never
        installed, so it — and its results, and anything nested under it — is carried by
        construction rather than by a rule that had to know about nesting.

        MEASURED BEFORE: a project referencing two recordings, one of whose files was not
        readable at open time, restored one document and warned about the other in the info
        bar — correctly. Then Ctrl+S rewrote the file from the LIVE session, which contains
        only what loaded: documents 2 -> 1, results 2 -> 1, status line "Saved project
        (1 result(s))", no dialog. The second recording's reference, its calibration, its
        layer state and its whole curated detection were gone from the project, and the
        commonest cause of the original failure — an unmounted drive, a network volume, a
        file moved for a moment — is exactly the case where nothing was actually wrong with
        the work. Recovery had been "put the file back and reopen"; after that save there
        was nothing left to reopen.

        A save may not decide that unreadable means unwanted. What could not be loaded is
        carried verbatim: it was read from this very file, so writing it back is lossless,
        and if the recording returns the project restores whole.

        Segments ride along too, and transitively: a segment whose parent could not be
        loaded could not be re-derived either, so it and anything nested under it are kept
        as specs. Carrying the parent reference is also what keeps them RESTORABLE — the
        save-time drop check resolves a segment against the documents in the project, and
        the parent is now still there.
        """
        from dataclasses import fields as _dc_fields, is_dataclass

        installed = self._restored_records
        carried: dict[str, list] = {}
        for f in _dc_fields(state):
            values = getattr(state, f.name, None)
            if not isinstance(values, list):
                continue
            # `is_dataclass` is the definition of "a project RECORD", not a shortcut: every
            # persisted record is one, and the only other list on a ProjectState is
            # `label_vocabulary`, which is a plain value the restore adopts WHOLESALE
            # (LabelVocabulary.from_json) rather than record by record. Carrying its entries
            # would duplicate them on the next save.
            missing = [v for v in values if is_dataclass(v) and id(v) not in installed]
            if missing:
                carried[f.name] = missing
        self._carried_over = carried or None

    def _finalize_restore(self) -> None:
        """Runs once the open queue drains: re-derive segments from their now-open parents,
        restore measurements + view mode, select a document (preferring one with a detection
        so the table + span overlays repopulate), surface any missing-source warning, then
        leave restore mode and announce."""
        state = self._restore_state
        recovery_restore = bool(self._restore_is_recovery)
        # Re-derive segments (and attach their calibrations/layers/results) BEFORE tearing down
        # restore mode: create_segment emits documentOpened, and _mark_dirty is a no-op only while
        # self._restoring is still True — reconstruction must never look like unsaved work.
        if state is not None:
            from ..analysis.labels import LabelVocabulary
            self._label_vocab = LabelVocabulary.from_json(
                getattr(state, "label_vocabulary", None))
            # BOTH windows, not just the gallery: an inspector left open across an Open
            # Project kept the previous project's palette, so its digits wrote labels from a
            # vocabulary the new project has never heard of.
            self._push_vocabulary()
            self._rederive_segments(state)
            # BEFORE the state is dropped: what could not be loaded must outlive the restore,
            # or the next save writes a project with it silently missing.
            self._capture_unrestorable(state)
        self._restore_state = None
        self._restore_ranges = {}

        if state is not None:
            # UNCONDITIONAL: the incoming project's measurement table replaces whatever is on
            # screen, even when it is empty. Restoring only non-empty tables left the PREVIOUS
            # project's hand measurements in the list, re-attributed to this project (and
            # written into it by the next save) — and they would now also draw their markers
            # onto a recording they were never measured on. set_measurements_df coerces None
            # to the canonical empty table and does not emit measurementsChanged, so a fresh
            # restore still looks clean.
            self._analysis_panel.measure_panel.set_measurements_df(state.measurements)
            try:
                self._session.set_view_mode(ViewMode(state.view_mode))
            except ValueError:
                pass
            self._select_after_restore(state)
            # The panel recipe is applied only after documents/results and the final target
            # exist.  Its setters can emit ordinary widget signals, so this remains inside
            # both restore guards and can neither launch a job nor create a phantom revision.
            self._restore_workspace_surface(
                ANALYSIS_PANEL_ARTIFACT_ID, self._analysis_panel)

        # Restore mode ends only HERE — after the restore's own selection has been made.
        # ``_select_after_restore`` drives the File Manager, and selecting a recording now SHOWS
        # it (hiding every other layer). Clearing the flag before that point would have the
        # reconstruction's last act collapse a deliberately saved multi-recording overlay down
        # to whichever document the restore happened to land on. Everything else the flag guards
        # (the open pipeline, the dirty marker) is equally right to stay suppressed until the
        # restored project is fully on screen.
        self._restoring = False
        if self._figure_library is not None:
            self._figure_library.schedule_refresh()
        # A restore rebuilds every table from disk. Those are reconstructions, not edits:
        # an undo stack that offered to walk back through them would offer to destroy the
        # project the user just opened. The suspension is lifted here, with the restored
        # state adopted as each table's starting point.
        self._history.set_suspended(False)
        self._measure_history.set_suspended(False)
        self._measure_history.checkpoint()   # adopt the restored table as the starting point
        self._sync_undo_actions()

        self._sync_export_action()
        self._sync_plots_action()
        self._sync_saved_subpop_action()
        # ...and the standalone windows, which a restore changes more completely than any
        # signal does. The per-signal wiring does NOT cover this: `_clear_session` fires
        # `documentClosed` at the START, when the outgoing project's tables are still the
        # live ones, and everything the incoming project installs afterwards either arrives
        # through a silent door (`set_measurements_df`) or is simply not a document.
        # Measured on two real projects that carry only hand measurements — the case
        # hunt2 predicted would be masked by any project that happens to carry a detection:
        # open A (3 measurements) with Plots open, then open B (1) -> the session held 1
        # and the window still plotted A's 3, with a live Fit and a live Export CSV.
        self._refresh_surface_windows()

        # Two DIFFERENT failures, reported as two different sentences. A missing source file is
        # a file that moved on disk; an unrebuilt segment never was a file — it is a crop whose
        # parent is not in this project. Folding the second into the first (and then taking
        # Path(...).name of its sentence) described the wrong problem and suggested the wrong fix.
        # Sentences already written by the restore itself (a source that is not the file the
        # project was saved from; a target that had to be brought on screen). They come
        # FIRST: a changed source recontextualises everything else in the banner.
        notes: list[str] = list(self._restore_notes)
        # FIRST, and in its own words: the other seven shortfalls are things that went
        # wrong, this one is something the user did, and calling it "could not be opened"
        # (which is what a later Save's own note used to call it) would be false — those
        # recordings were never attempted.
        if self._restore_cancelled:
            names = ", ".join(Path(p).name for p in self._restore_cancelled)
            notes.append(
                f"You cancelled this restore, so {len(self._restore_cancelled)} of the "
                f"project's recording(s) were not loaded, along with any of their results: "
                f"{names}. They are kept in the project, not dropped — reopen the project to "
                f"load them.")
        if self._restore_missing:
            # TWO SENTENCES, because these are two different problems. A file the
            # pre-flight check could not find is gone from that path; a file the READER
            # refused was found, opened and rejected for a reason it already stated —
            # a window that selects nothing, an unsupported ABF variant, permissions.
            # Folding the second into "could not be found" sent the user looking for a
            # file that never moved, and threw away the one sentence that says what to do.
            gone = [p for p in self._restore_missing if p not in self._restore_refusals]
            refused = [p for p in self._restore_missing if p in self._restore_refusals]
            if gone:
                names = ", ".join(Path(p).name for p in gone)
                notes.append(
                    f"{len(gone)} source file(s) could not be found and were "
                    f"skipped, along with any of their results: {names}.")
            if refused:
                notes.append(
                    f"{len(refused)} source file(s) were found but could not be read, and "
                    f"were skipped along with any of their results. Each one's reason, as "
                    f"the reader gave it: " + "; ".join(
                        f"{Path(p).name} — {self._restore_refusals[p]}" for p in refused))
        if self._restore_lost_segments:
            notes.append(
                f"{len(self._restore_lost_segments)} segment(s) could not be rebuilt, along "
                f"with any of their results: {', '.join(self._restore_lost_segments)}. A "
                f"segment is stored as a crop of its parent recording, so its parent has to "
                f"be part of the project AND still hold the samples the crop was taken "
                f"from. They are kept in the project, not dropped: re-export or restore the "
                f"parent as it was and they come back.")
        if self._restore_bad_calibrations:
            notes.append(
                f"{len(self._restore_bad_calibrations)} stored calibration(s) held a "
                f"non-finite gain or offset and were ignored; those channels keep their "
                f"defaults: {', '.join(self._restore_bad_calibrations)}.")
        # The channel-identity pair. A MOVE is not a failure — the record's own channel is
        # still here, just at another index — so it is reported plainly; a LOSS is, and it
        # says which record and what it was looking for. Neither sentence can fire for a
        # project written before v7: those records carry no identity, resolve by index and
        # say nothing, which is what keeps a real warning worth reading.
        # The PATH sibling of the channel-moved sentence, and reported the same way: the
        # records were all applied, so this is not a shortfall — it is the fact that the
        # project's stored path and the live one have come apart, which the user is the
        # only one who can settle (by saving the project at its new spelling).
        if self._restore_path_moved:
            notes.append(
                f"{len(self._restore_path_moved)} recording(s) are now reached by a "
                f"different path than the one stored in the project — a moved folder with a "
                f"symlink left behind, typically. Their calibrations, layers and detections "
                f"were re-attached by file identity, not by path: "
                f"{', '.join(self._restore_path_moved)}. Save the project to record the "
                f"path they are at now.")
        if self._restore_channel_moved:
            notes.append(
                f"{len(self._restore_channel_moved)} record(s) were re-attached by channel "
                f"name after this recording's channels moved: "
                f"{', '.join(self._restore_channel_moved)}.")
        if self._restore_channel_lost:
            notes.append(
                f"{len(self._restore_channel_lost)} record(s) name a channel this recording "
                f"no longer has and were NOT applied — they are kept in the project, not "
                f"dropped: {', '.join(self._restore_channel_lost)}.")
        if notes:
            self._info_bar.show_message(" ".join(notes) + " The rest of the project was restored.")
        cancelled = list(self._restore_cancelled)
        self._restore_cancelled = []
        self._restore_missing = []
        self._restore_refusals = {}
        self._restore_lost_segments = []
        self._restore_bad_calibrations = []
        self._restore_channel_moved = []
        self._restore_channel_lost = []
        self._restore_path_moved = []
        self._restore_sources = {}
        self._restore_notes = []
        self._restore_is_recovery = False
        if cancelled:
            # NOT `_mark_clean`, and NOT "Project restored." A cancelled restore left the
            # session holding less than the file does, so the two are not in agreement and
            # the • is the honest marker of that — the recordings are still IN the project
            # (`_capture_unrestorable` carried their refs), so a save writes them back, but
            # until then what is on screen is not what was opened. The count is said here as
            # well as in the banner because this is the line that used to claim the opposite,
            # and the banner's message is the one the user may already have dismissed.
            self._mark_dirty()
            self.statusBar().showMessage(
                f"Project restore cancelled — {len(cancelled)} of its recording(s) were not "
                f"loaded. They are still in the project.", 12000)
        elif recovery_restore:
            # Recovery is a copy of work that was NEVER explicitly saved.  Loading it
            # through the ordinary project pipeline must not inherit that pipeline's
            # final `_mark_clean()` or adopt the recovery file as a project path.
            self._dirty = True
            self._update_title()
            self.statusBar().showMessage(
                "Recovered unsaved work. Save the project explicitly to keep it.", 12000)
        else:
            self._mark_clean()   # a freshly restored project holds no unsaved changes
            self._explicit_saved_revision = max(
                self._explicit_saved_revision, self._session_revision)
            self.statusBar().showMessage("Project restored.", 6000)
        self.projectRestored.emit()

    def _select_after_restore(self, state) -> None:
        """Pick the (doc, channel) to land on after a restore: a restored detection if
        there is one (so its events + spans show immediately), else a document's default
        channel.

        PREFERRING ONE THAT IS ALREADY DRAWN is the whole change here, and it is what keeps
        the saved view intact. The restored layer set is the user's — it may be a deliberate
        multi-recording overlay — so the honest first move is to analyse something the
        project chose to show, not to redecorate. Only when NO drawn recording carries a
        detection does the target have to be brought on screen, and then it is added to the
        view (never swapped in for what is there) and said out loud, because the project's
        saved view genuinely did not include it.

        ``r.doc_id`` here is the LIVE per-load id ``_apply_restore_for_doc`` stamped onto the
        restored result, so ``has_document`` is asking about this very load — see
        :mod:`nanoscope.core.identity`.
        """
        from .file_dock import _default_channel

        shown = self._session.shown_doc_ids()
        live = [r for r in self._session.results.all()
                if self._session.has_document(r.doc_id) and getattr(r.payload, "df", None)
                is not None]
        target: tuple[str, int] | None = None
        has_detection = False
        for r in live:                                 # a detection ON SOMETHING DRAWN
            if r.doc_id in shown:
                target, has_detection = (r.doc_id, r.channel), True
                break
        if target is None and live:                    # ...else any restored detection
            target, has_detection = (live[0].doc_id, live[0].channel), True
        if target is None:                             # ...else a drawn document, else any
            docs = self._session.documents()
            doc = next((d for d in docs if d.doc_id in shown), docs[0] if docs else None)
            if doc is not None:
                target = (doc.doc_id, _default_channel(doc))
        if target is None:
            return
        self._file_dock.focus_document(target[0])
        self._on_selection(*target)      # _ensure_target_drawn runs inside
        if target[0] not in shown and self._session.has_document(target[0]):
            what = ("carries this project's detection but was not"
                    if has_detection else "was not")
            self._restore_notes.append(
                f"“{self._session.document(target[0]).display_name}” {what} among the "
                f"recordings this project was saved showing, so it has been added to the "
                f"view — the app never analyses a recording it is not drawing.")

    # -- recent projects (QSettings, mirroring recent files) -------------------------------------
    def _add_recent_project(self, path: str) -> None:
        p = str(Path(path).resolve())
        if p in self._recent_projects:
            self._recent_projects.remove(p)
        self._recent_projects.insert(0, p)
        del self._recent_projects[_MAX_RECENT:]
        self._rebuild_recent_projects_menu()

    def _rebuild_recent_projects_menu(self) -> None:
        self._recent_projects_menu.clear()
        if not self._recent_projects:
            empty = self._recent_projects_menu.addAction("(none)")
            empty.setEnabled(False)
            return
        for p in self._recent_projects:
            act = self._recent_projects_menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(lambda _checked=False, path=p: self._open_project(path))
        self._recent_projects_menu.addSeparator()
        clear = self._recent_projects_menu.addAction("Clear recent projects")
        clear.triggered.connect(self._clear_recent_projects)

    def _clear_recent_projects(self) -> None:
        self._recent_projects = []
        self._rebuild_recent_projects_menu()

    # -- drag & drop -----------------------------------------------------------------------------
    @staticmethod
    def _drop_split(mime: QtCore.QMimeData) -> tuple[list[str], list[str]]:
        """A dragged payload split into (local files we can open, everything else).

        ``hasUrls()`` is scheme-BLIND — a link dragged out of Safari or Mail, or an item
        from an unmounted share, arrives as a perfectly good url. ``isLocalFile()`` is the
        question that actually matters, and it is asked HERE, once, so the drag and the
        drop cannot disagree about what this window can accept.
        """
        local: list[str] = []
        foreign: list[str] = []
        for u in mime.urls():
            (local if u.isLocalFile() else foreign).append(
                u.toLocalFile() if u.isLocalFile() else u.toString())
        return local, foreign

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        """Offer the window as a drop target ONLY for what it could actually open.

        Accepting every url and then discarding the non-local ones is how the cursor came
        to say "drop here" over a browser link that this window has no way to read.
        """
        local, _foreign = self._drop_split(event.mimeData())
        if local:
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        """Open the local files that were dropped — and NAME anything that was not.

        A drop that silently discards half of what the user dragged is the one wordless
        refusal this app had: every other refusal path (``filesystem_refusal``,
        ``foreign_format_redirect``, the "not a recognised recording" fallthrough) states
        its condition. The mixed drop is the sharp case — one recording opens, the drag
        reads as a success, and the item that went nowhere is never mentioned.
        """
        local, foreign = self._drop_split(event.mimeData())
        if foreign:
            self.statusBar().showMessage(
                f"{len(foreign)} dropped item(s) are not files on this computer and were "
                f"not opened — NanoScope reads recordings off a local (or mounted) disk, "
                f"not from a link: {', '.join(foreign)}. "
                + ("The local file(s) dropped with them are opening."
                   if local else "Download or mount it first, then drop the file."),
                12000)
        if local:
            self.open_paths(local)
        if local or foreign:
            event.acceptProposedAction()

    # -- settings + teardown ---------------------------------------------------------------------
    def _restore_settings(self) -> None:
        """Re-apply the saved layout and lists — treating the store as external input.

        EVERY value read here is validated before it is used, because the preferences
        store is a file on disk that the app is not the only writer of (see
        :func:`_restorable_bytes`). A value of the wrong kind is dropped, the key it came
        under is remembered, and the info bar names them: a preference the app cannot
        stand behind is worth one default window, not a launch failure and not a silent
        substitution.
        """
        s = QtCore.QSettings()
        # THE STORE ITSELF can be the casualty, not just one value in it. A corrupted
        # preferences file (crash mid-quit, half-synced ~/Library) makes QSettings drop
        # keys wholesale with a FormatError — and this method used to never ask, so the
        # app came up looking factory-fresh (layout gone, recent files gone) with nothing
        # anywhere saying why, then quit and wrote a clean file over the evidence. That
        # silent fallback is exactly what `_restorable_bytes` forbids for a single value
        # ("said out loud rather than swallowing"), one level up.
        store_broken = _settings_store_damaged(s)
        #: Preference keys whose stored value was not something the app could use. Said
        #: out loud at the end of this method — a dropped preference is a change the user
        #: will SEE (their layout is gone), so it is never dropped silently.
        ignored: list[str] = []
        geo = _restorable_bytes(s, "geometry", ignored)
        if geo is not None and not self.restoreGeometry(geo):
            # Right kind, wrong contents — an empty or foreign-version byte array. Qt says
            # so by returning False rather than raising, and that answer is worth the same
            # sentence as a wrong-typed value: either way the saved geometry is not applied.
            ignored.append("geometry")
        state = _restorable_bytes(s, "windowState", ignored)
        if state is None or not self.restoreState(state):
            if state is not None:
                ignored.append("windowState")
            # No usable dock layout, for either reason: fall back to the designed widths
            # rather than to whatever the docks' sizeHints happen to add up to.
            #
            # PROVISIONALLY. This runs from ``__init__``, where ``self.width()`` is still
            # the 1440 the constructor asked for rather than a width any screen has agreed
            # to, so the width-dependent half of the derivation (the Inspector's ceiling)
            # cannot be honest yet. :meth:`showEvent` re-derives it against the width the
            # window actually got. Applied here as well, because a window that is never
            # shown — most of the headless tests — still needs the designed widths.
            self._opening_layout_pending = True
            self._apply_default_dock_widths()
        raw = s.value("recentFiles", [])
        if isinstance(raw, str):
            raw = [raw] if raw else []
        self._recent = [str(x) for x in (raw or []) if Path(x).exists()]
        raw_proj = s.value("recentProjects", [])
        if isinstance(raw_proj, str):
            raw_proj = [raw_proj] if raw_proj else []
        self._recent_projects = [str(x) for x in (raw_proj or []) if Path(x).exists()]
        # The two remembered directories: a str is what was written, and anything else
        # (a list from a damaged store) would become a nonsense `str([...])` path that
        # the next Open dialog would silently start in. Fall back to home instead.
        self._last_dir = _string_setting(s, "lastDir", str(Path.home()), ignored)
        self._last_project_dir = _string_setting(
            s, "lastProjectDir", str(Path.home()), ignored)
        self._rebuild_recent_menu()
        self._rebuild_recent_projects_menu()
        messages: list[str] = []
        if store_broken:
            messages.append(
                f"The saved-preferences file is damaged and could not be fully read "
                f"({s.fileName()}), so anything it could not give up — window layout, "
                f"recent files, remembered folders — starts at its default. Nothing else "
                f"is affected, and the next quit writes a healthy file over it.")
        if ignored:
            messages.append(
                f"{len(ignored)} saved preference(s) held a value this app did not write "
                f"and could not use, so they were ignored and their defaults used instead: "
                f"{', '.join(ignored)}. Nothing else was affected — the next quit writes "
                f"healthy values over them.")
        if messages:
            # One info bar, one message: the second show_message would replace the first,
            # and a disclosure that overwrites a disclosure is a silent drop with a spare
            # step.
            self._info_bar.show_message(" ".join(messages))

    def _inspector_width_ceiling(self, layout_width: int) -> int:
        """The widest the Inspector may OPEN and still leave the viewer the widest column.

        THE RULE, as arithmetic rather than as a property of one laptop. The three columns
        and the two separators between them fill the window::

            file (320) + separator + viewer + separator + inspector == layout_width

        so the viewer is wider than the Inspector exactly while the Inspector is under half
        of what is left after the file manager — that is the number returned here, taken one
        pixel short of the tie. It was previously a flat 560, chosen because 1512 - 12 - 320
        - 560 = 620 leaves the viewer widest **on a 1512 px display and nowhere narrower**:
        at 1440 the same 560 made the Inspector the widest column (viewer 548), and at 1280
        it was 560 against 388. A signal viewer that stops being the widest column as soon as
        the window is not exactly the developer's is not a rule, it is a coincidence.

        Measured AS THIS METHOD RETURNS IT, cap included: 1512 -> 560, 1440 -> 553,
        1280 -> 473, 1152 -> 409. Only the widest of those is capped — half the room at 1512
        is 589 — and that cap is what stops a big display opening a 900 px control column.

        BELOW 1337 px the two things this method arbitrates — the Inspector showing its own
        pages, and the viewer staying the widest column — cannot both be had, and the clamp
        in :meth:`_inspector_default_width` resolves it towards the floor: the Inspector
        opens at its designed minimum and takes a scrollbar. A scrollbar on a control page
        whose controls are all still whole is a door with a step in front of it; a viewer
        squeezed under the Inspector has no door at all.

        1337 IS MEASURED, and it is the width at which this ceiling first reaches the width
        the pages need (502 on this tree): ``room = 1337 - 12 - 320 = 1005``, ``(1005-1)//2
        == 502``; at 1336 it is 501. This line used to say "roughly 1252", which is a
        different boundary — the width at which the ceiling drops below the FLOOR (460) —
        and stating it here left an 84 px band, 1253 to 1336, where the docstring claimed
        both things could be had and the clamp silently gave neither: the Inspector opened
        at the ceiling (460–501, measured 483 at 1300 and 501 at 1336), which is under the
        502 its pages need, so it took a horizontal scrollbar ANYWAY — the Manual-measurement
        panel clipped by 19 px at 1300 — while still spending up to 41 px of the trace on the
        difference. The clamp now resolves the whole band the way this paragraph says it does.
        """
        room = layout_width - 2 * _DOCK_SEPARATOR_WIDTH - _FILE_DOCK_DEFAULT_WIDTH
        return min(_INSPECTOR_MAX_DEFAULT_WIDTH, (room - 1) // 2)

    def _inspector_default_width(self) -> int:
        """How wide the Inspector should OPEN: the narrowest width that shows its pages.

        The pages are all widgetResizable scroll areas with a zero minimum (see
        ``_build_docks``), which is what lets the user drag this dock down to the tab bar —
        and is also why the dock cannot ask for a width. So the ASKING happens here, once,
        from the same number the scroll area would otherwise turn into a horizontal
        scrollbar.

        THE NUMBER TO ASK FOR IS THE PAGE'S MINIMUM, NOT ITS NATURAL WIDTH. This method used
        to read ``sizeHint()`` — the width at which a page is comfortable, 646 px for the
        Analysis control column — which is more than any of these pages needs and more than
        the ceiling allows, so the ceiling became the width and the derivation was theatre.
        ``minimumSizeHint()`` is the width below which the page's own controls start being
        clipped, which is what "reachable" actually means; the widest of those across every
        page, plus the chrome between the dock's edge and the deepest of their viewports, is
        the smallest dock that hides nothing. MEASURED on this tree at 1512x900: 502 px, and
        a 2 px sweep down from there confirms it is the knee — at 500 the Manual-measurement
        panel takes a horizontal scrollbar and clips "Snap to selected event" and "Mark
        selection as missed event"; at 472 the Preprocess column clips "Back to default"; at
        460, the pre-round width, both. Everything above 502 is width taken off the trace for
        nothing, which is why the viewer now opens at 678 rather than 620.

        Clamped at both ends, and both ends are load-bearing:

        * never below ``_INSPECTOR_DEFAULT_WIDTH`` — a page that happens to be narrow today
          must not open a dock so thin the next page looks broken in it;
        * never above :meth:`_inspector_width_ceiling` — the signal viewer stays the widest
          column, at every window width and not just at 1512. A panel that grows a long combo
          item can no longer take the viewer's width with it; it gets a scrollbar instead,
          which is the trade the ceiling exists to make.

        AND WHEN THE CEILING CANNOT REACH ``wanted``, THE FLOOR TAKES IT, not the ceiling.
        Below a 1337 px window (measured; see :meth:`_inspector_width_ceiling`) no width
        satisfies both rules, and a width BETWEEN the two — the ceiling, 460–501 — satisfies
        neither: the pages clip exactly as they do at the floor (the Manual-measurement panel
        by 19 px at a 1300 px window, by 1 px at 1336) and the extra pixels come off the trace
        for a scrollbar the user gets either way. So the answer below the boundary is one
        answer, the one the ceiling's docstring states: the designed floor, with the scrollbar,
        and every pixel the Inspector cannot use handed to the viewer (measured at a 1300 px
        window: Inspector 483 -> 460, viewer 485 -> 508). Above the boundary nothing changes.

        This is a change to the width the dock OPENS at, and only that. It is applied by
        ``_apply_default_dock_widths`` on a first launch — once from ``__init__`` and again
        from :meth:`showEvent`, which is where the window's width stops being the 1440 the
        constructor asked for and becomes the one the screen gave it. The boundary is never
        re-imposed after that opening turn, because a separator that slides because the
        window got 3 px narrower is furniture moving under the pointer, which is a worse bug
        than the scrollbar it would be chasing.

        Read before the window is shown, which is where a first launch applies it:
        ``minimumSizeHint()`` on these pages is layout arithmetic, not painted geometry, and
        measures the same (438 px for the widest, the Manual-measurement panel) before
        ``show()`` as after it.
        """
        widest = 0
        for area in self._inspector_tabs.findChildren(QtWidgets.QScrollArea):
            page = area.widget()
            if page is not None:
                widest = max(widest, page.minimumSizeHint().width())
        wanted = widest + _INSPECTOR_CHROME_WIDTH + _INSPECTOR_NESTED_CHROME_WIDTH
        ceiling = self._inspector_width_ceiling(self.width())
        if wanted > ceiling:
            # The window is too narrow for both rules. One answer for the whole of that
            # range — the floor — rather than a ceiling that clips like the floor and costs
            # like the width that would not. (Under a ~1253 px window the ceiling is already
            # below the floor and this is what the old `max` returned anyway; the change is
            # the band above it.)
            return _INSPECTOR_DEFAULT_WIDTH
        return max(_INSPECTOR_DEFAULT_WIDTH, wanted)

    def _apply_default_dock_widths(self) -> None:
        """Open the two docks at their designed widths — first launch only.

        Neither dock has a meaningful natural width any more: the file manager's tree will
        happily live in 293 px and truncate every recording name, and the Inspector's pages
        are all scroll areas whose combined sizeHint means nothing on its own. So the opening
        layout is stated here rather than inferred: a stated width for the left column, and
        for the right one the content-derived width above. This runs ONLY when there is no
        saved windowState; once the user has dragged the separators, ``_save_settings``
        persists her choice and ``restoreState`` above wins on every later launch — this
        method is never the thing that fights a remembered layout. These are not minimums:
        both docks remain draggable narrower and wider than this.
        """
        self.resizeDocks(
            [self._file_dock, self._inspector],
            [_FILE_DOCK_DEFAULT_WIDTH, self._inspector_default_width()],
            QtCore.Qt.Horizontal)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        """Re-derive the opening dock widths against the width the window ACTUALLY got.

        THE CEILING WAS UNREACHABLE FROM THE LAUNCH PATH. ``_apply_default_dock_widths``
        runs from ``_restore_settings``, which runs from ``__init__``, where ``self.width()``
        is the 1440 of the constructor's ``resize`` — a request the window has not yet been
        granted. So :meth:`_inspector_width_ceiling` was only ever asked about 1440, where it
        answers 553 and never bites, and the clamp under it could not fire however narrow the
        display was. Measured on a fresh profile, launched at 1280x800 BEFORE this method
        existed: the Inspector opened at 502 against a ceiling of 473 and left the signal
        viewer 446 px — the Inspector was the widest column, which is the single thing that
        ceiling exists to prevent. With the derivation moved here it opens at its floor (460)
        and the viewer keeps 488.

        THE OPENING LAYOUT STAYS PROVISIONAL FOR ONE MORE TURN of the event loop, because a
        platform that shrinks a window to fit its screen does it as the window is mapped, and
        that resize can arrive after this event rather than before it (:meth:`resizeEvent`
        re-applies while the flag is up). :meth:`_finish_opening_layout` drops the flag at the
        end of that turn — before a user could have touched a separator — and nothing here
        moves furniture under the pointer afterwards, which is the rule
        :meth:`_inspector_default_width` states.
        """
        super().showEvent(event)
        if self._opening_layout_pending:
            self._apply_default_dock_widths()
            # Bound to `self` so the callback dies with the window rather than firing into a
            # deleted C++ object at teardown.
            QtCore.QTimer.singleShot(0, self, self._finish_opening_layout)
        if not self._startup_recovery_checked:
            # A startup modal shown from __init__ can appear behind the not-yet-mapped main
            # window on macOS.  Offer only after the first show event; the method itself is
            # idempotent in case the window is hidden and shown again before this turn runs.
            QtCore.QTimer.singleShot(0, self, self._offer_startup_recovery)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        """Only interesting while the opening layout is still settling — see :meth:`showEvent`.

        This is the window manager having its say about the size the constructor asked for.
        Once :meth:`_finish_opening_layout` has run, a window resize moves no separator: the
        docks hold their absolute widths, which is Qt's own behaviour and the behaviour
        ``tests/gui/test_lay2_inspector_width.py`` pins.
        """
        super().resizeEvent(event)
        if self._opening_layout_pending and self.isVisible():
            self._apply_default_dock_widths()

    def _finish_opening_layout(self) -> None:
        """The layout is the user's from here: the opening widths are never re-imposed."""
        self._opening_layout_pending = False

    def _save_settings(self) -> None:
        s = QtCore.QSettings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        s.setValue("recentFiles", self._recent)
        s.setValue("recentProjects", self._recent_projects)
        s.setValue("lastDir", self._last_dir)
        s.setValue("lastProjectDir", self._last_project_dir)
        # Flush to disk HERE rather than relying on QSettings' own destructor. The quit
        # path may end in ``exit_without_destroying_live_threads`` (a worker that would
        # not stop), which runs no destructors at all — so an unsynced setting would be
        # silently lost exactly on the quits that are already going least well.
        s.sync()

    # -- unsaved-work guard (dirty tracking + close confirmation) --------------------------------
    # -- undo / redo over the event tables (core/history.py is the recorder) ----------------------
    def _note_step(self, kind: str) -> None:
        """Record WHICH of the two tables just gained a step.

        There is one Edit ▸ Undo and two stacks (the event tables, and the hand-measurement
        table — ``core/history.py`` explains why they cannot be one). This journal is what
        decides which a press acts on, and the rule is the only one a user would guess:
        **the most recent thing that happened is the first thing undone.** A new step
        invalidates the redo branch, exactly as it does inside each stack.
        """
        self._undo_journal.append(kind)
        self._redo_journal.clear()
        # ...AND THE BRANCH ITSELF, on BOTH stacks — BUT ONLY FOR THE RECORDING THE STEP
        # BELONGS TO. Clearing only the journal left the entries alive inside whichever
        # stack the journal was not naming, and `_redo_target`'s fallback reaches straight
        # past an empty journal into them: curate 2 rejections -> re-detect -> Ctrl+Z -> add
        # one hand measurement, and Edit still offered "Redo Re-run detection", which
        # destroyed the 2 rejections again while the user's attention was on the other
        # table. Each stack already invalidates its OWN branch on a push (EventHistory per
        # key, MeasurementHistory wholesale — it is one table), so what is left for this
        # door is exactly the CROSS-stack half.
        #
        # The unscoped pair this used to be — `self._history.clear_redo()` and
        # `self._measure_history.clear_redo()` — made that half session-wide, which was the
        # same data-loss bug one boundary out: reject 2 events on recording A, Ctrl+Z, then
        # reject one event on recording B, and A's redo branch was gone from every stack
        # with nothing said. The measurement that justified the cross-stack rule was taken
        # WITHIN one recording, and that is the scope it holds at: a step names the
        # recordings it touched and ends their branches, nobody else's. `None` (unknown)
        # still drops everything, because an unknown scope is not a narrow one.
        doc_ids, doc_keys = self._step_scope(kind)
        self._history.clear_redo(doc_ids)
        self._measure_history.clear_redo(doc_keys)
        # Bounded, and the bound is a HEURISTIC one: this journal only decides WHICH OF THE
        # TWO STACKS a press acts on, never which entry, and past the bound `_undo_target`
        # falls back to asking the stacks directly (event table first) — the degradation the
        # method already documents. It is not a bound on recoverable work. It was described
        # as "entries beyond them are dead anyway" while MAX_DEPTH capped the whole session;
        # depth is per event table now, so a session with several detections can hold more
        # steps than this, and the fallback is what carries them.
        if len(self._undo_journal) > 2 * MAX_DEPTH:
            del self._undo_journal[0]

    def _step_scope(self, kind: str):
        """``(doc_ids, doc_keys)`` — whose redo branch on the OTHER stack a step just ended.

        One tuple, read two ways, because the two stacks name recordings differently: the
        event stack is keyed by ``doc_id`` (this load, this session) and the measurement
        table by ``doc_key`` (the persistent key). Returns ``None`` for a stack whose scope
        could not be determined, which every reader treats as "all of them" — see
        :meth:`nanoscope.core.history.EventHistory.clear_redo`.

        An empty set is the ordinary answer for the stack the step was pushed ON: that
        stack has already pruned its own branch, so this door owes it nothing.
        """
        from ..core.segment import persistent_key

        if kind == _UNDO_MEASUREMENTS:
            # A measurement step ends the EVENT branch of the recordings whose rows moved.
            docs = self._measure_history.last_step_docs
            if docs is None:
                return None, frozenset()
            return frozenset(self._doc_ids_for_keys(docs)), frozenset()
        key = self._history.last_step_key
        if key is None:
            return frozenset(), None
        doc_id = key[0]
        if not self._session.has_document(doc_id):
            return frozenset(), None
        try:
            return frozenset(), frozenset({persistent_key(self._session.document(doc_id))})
        except Exception:                      # noqa: BLE001 - unknown, never a narrow answer
            return frozenset(), None

    def _doc_ids_for_keys(self, doc_keys) -> set[str]:
        """The open documents whose persistent key is one of ``doc_keys``.

        The measurement table records a recording by its PERSISTENT key (it outlives the
        session); the event stack is keyed by the per-load ``doc_id``. This is the one
        place the two are joined, so the join cannot drift between call sites.
        """
        from ..core.segment import persistent_key

        wanted = {str(k) for k in doc_keys}
        out: set[str] = set()
        for doc in self._session.documents():
            try:
                if persistent_key(doc) in wanted:
                    out.add(doc.doc_id)
            except Exception:                  # noqa: BLE001 - a scope query must never raise
                continue
        return out

    def _checkpoint_measurements(self) -> bool:
        """Commit whatever has happened to the measurement table.

        The journalling is NOT done here any more: it hangs off
        ``MeasurementHistory.stepRecorded``, so a push that never came through this wrapper
        — ``_move``'s leading ``checkpoint()``, the restore's adoption call — journals too.
        A step that skipped the journal also skipped the redo invalidation, which is half
        of what made the cross-stack redo branch reachable."""
        return self._measure_history.checkpoint()

    def _can_undo(self, kind: str) -> bool:
        if kind == _UNDO_MEASUREMENTS:
            return self._measure_history.can_undo()
        return self._history.can_undo(self._undo_key())

    def _can_redo(self, kind: str) -> bool:
        if kind == _UNDO_MEASUREMENTS:
            return self._measure_history.can_redo()
        return self._history.can_redo(self._undo_key())

    def _undo_target(self) -> str | None:
        """Which table the next Undo acts on: the most recent step that CAN be undone.

        "Can be undone" matters as much as "most recent": the event stack is per
        (recording, channel, detector), so its newest step may belong to a table that is
        not on screen, and offering it would repaint nothing. The journal is scanned newest
        first and the first live candidate wins; if the journal has run out (it is bounded)
        the stacks are asked directly, event table first.
        """
        for kind in reversed(self._undo_journal):
            if self._can_undo(kind):
                return kind
        for kind in (_UNDO_EVENTS, _UNDO_MEASUREMENTS):
            if self._can_undo(kind):
                return kind
        return None

    def _redo_target(self) -> str | None:
        for kind in reversed(self._redo_journal):
            if self._can_redo(kind):
                return kind
        for kind in (_UNDO_EVENTS, _UNDO_MEASUREMENTS):
            if self._can_redo(kind):
                return kind
        return None

    @staticmethod
    def _take_last(journal: list[str], kind: str) -> None:
        """Remove the newest journal entry naming ``kind`` (it has just been applied)."""
        for i in range(len(journal) - 1, -1, -1):
            if journal[i] == kind:
                del journal[i]
                return

    def _checkpoint_everything(self) -> None:
        """Commit BOTH tables. The four moments at which "what happened last" has to be
        true — an undo/redo, the Edit menu opening, a project save and a quit — are about
        every table the user can lose work in, not only the event ones."""
        self._history.checkpoint_all()
        self._checkpoint_measurements()

    def active_detection_key(self):
        """The ``(doc_id, channel, module_id)`` of the detection ON SCREEN, or None.

        PUBLIC because it is the answer to a question more than one surface window asks,
        and every window that asks it separately is a window that can disagree with the
        Analysis panel about which table it is drawing. ``ResultStore`` is keyed
        ``(doc_id, channel, module_id)`` and genuinely holds a channel's detectors side by
        side, so the ``(doc_id, channel)`` selection stopped identifying a table the day
        discovery began offering one source per detector: measured on the -1000 mV
        recording with Schmitt (110 events) and CUSUM (250) both stored and the panel
        showing CUSUM, ``_plots_sources`` flagged Schmitt's source as the current one.

        Read from the panel's own ``current_detection`` rather than from the selection,
        so an ML result and a Schmitt result on the same channel are distinguishable —
        which is also why Edit ▸ Undo keys its stacks off this (see :meth:`_undo_key`,
        which is this method under the name the undo code reads it by).
        """
        res = self._analysis_panel.current_detection()
        if res is None:
            return None
        from ..analysis.events import EventSet

        if not isinstance(res.payload, EventSet):
            return None
        return (res.doc_id, int(res.channel), res.module_id)

    def _undo_key(self):
        """The ``(doc_id, channel, module_id)`` Edit ▸ Undo acts on — the detection the
        user is actually looking at — or None.

        The SAME key the rest of the window resolves "the table on screen" to
        (:meth:`active_detection_key`); kept as its own name because the undo code reads
        it a dozen times and "the undo key" is what it is asking for. One resolver: a
        second spelling of this is how the Plots window came to draw one detector while
        Undo walked another one's stack back."""
        return self.active_detection_key()

    def _checkpoint_current(self) -> None:
        """Commit whatever just happened to the detection on screen."""
        key = self._undo_key()
        if key is not None:
            self._history.checkpoint(key)

    @staticmethod
    def _any_measurable(df) -> bool:
        """Could a whole-frame accept have accepted ANY row of this frame?

        The schema's own AND: ``accepted = ~(too_long | clipped | non_finite) & the
        curator's decision``. Read through
        :data:`~nanoscope.analysis.events.AUTOMATIC_REJECTION_FLAGS`, the ONE place that
        says which tokens are automatic rejections, rather than copied — a fourth rejecting
        rule then reaches this label with no second edit. A frame with no ``flag`` column
        cannot answer, and fails CLOSED (the same direction the curation surfaces fail).
        """
        from ..analysis.events import AUTOMATIC_REJECTION_FLAGS, flag_tokens

        if "flag" not in getattr(df, "columns", ()):
            return False
        return any(not any(t in AUTOMATIC_REJECTION_FLAGS for t in flag_tokens(f))
                   for f in df["flag"])

    def _arm_gallery_whole_frame(self, accept: bool) -> None:
        """The Event Gallery's "Accept/Reject all shown" was pressed — remember the GESTURE.

        The scope is captured HERE, before the rows move, because it is a fact about the
        press and not about the result: how many events the gallery's current filter was
        showing, and how many the detection holds. "All shown" is filter-scoped by design
        (its own tooltip: "not the whole detection unless the filter is All"), so a press
        under *Accepted only* is a gesture over 3 of 10 rows and must never be recorded as
        a whole-frame batch — the button itself already says "Reject all 3".
        """
        res = self._analysis_panel.current_detection()
        gal = self._event_gallery
        if res is None or gal is None:
            self._gallery_whole_frame = None
            return
        try:
            self._gallery_whole_frame = (bool(accept), int(gal.event_count()),
                                         int(len(res.payload.df)))
        except Exception:                      # noqa: BLE001 - a label hint may never raise
            self._gallery_whole_frame = None

    def _disarm_gallery_whole_frame(self) -> None:
        """...and the marker dies with the click that armed it. See the wiring comment."""
        self._gallery_whole_frame = None

    def _checkpoint_gallery_curation(self) -> None:
        """A multi-row curation gesture in the Event Gallery — named by what the USER DID.

        The gallery routes EVERY gesture that wrote more than one row into
        ``curationBatchChanged`` (``event_gallery.py``: "a multi-tile or whole-detection
        curation rewrote ``accepted``"), so the signal alone does not say which gesture ran
        and nothing here may pretend it does. Only the marker armed by the two "all shown"
        buttons does, and only when that press covered the whole detection; everything else
        — the "…selected" buttons, A / R, Space, a filter-scoped "all shown" — is named by
        ``describe_change``, which reads the diff and is already right ("Curate 2 events").

        THE MEASURED DEFECT was labelling by the resulting STATE. Rejecting the last 2
        accepted tiles of 11 leaves ``accepted`` uniformly False, so Edit ▸ Undo read
        *"Undo Reject all events"* — offering to walk back a whole-frame reject the user
        never pressed, while the undo restores exactly those 2 rows. Two surfaces then
        disagreed about one gesture: the gallery's own button read "Reject all 3" and its
        status line "3 of 10 accepted", while the undo stack recorded "all".
        """
        marker, self._gallery_whole_frame = self._gallery_whole_frame, None
        key = self._undo_key()
        if key is None:
            return
        label = None
        if marker is not None:
            accept, subset_n, frame_n = marker
            if subset_n and subset_n == frame_n:
                label = "Accept all events" if accept else "Reject all events"
        self._history.checkpoint(key, label=label)

    def _checkpoint_batch_curation(self) -> None:
        """The Event INSPECTOR's Accept-all / Reject-all, which the diff alone cannot name.

        A LABEL HINT, and the only one in the application. ``describe_change`` sees a
        batch that flipped the two remaining rejections as "Curate 2 events", which is
        true but not what the user pressed. It may say so HERE and only here, because the
        inspector emits ``curationBatchChanged`` from exactly one place
        (``EventInspectorWindow._set_all_accepted``) and that place is always the whole
        frame. If this connection were ever dropped the step is still recorded — with the
        derived label. That is the whole shape of the design: a forgotten door costs a
        word, never a row.

        THE DIRECTION IS READ FROM THE GESTURE'S EFFECT, NOT FROM "is everything accepted".
        A whole-frame Accept-all can never leave the column uniformly True on a frame that
        holds any unmeasurable row — those stay rejected by design, and the owner's −600 mV
        run has three — so the old ``accepted.all()`` test never fired there and the
        documented *"Undo Accept all events"* was unreachable on every real recording. Only
        an accept can leave any row accepted after a whole-frame gesture, so that is the
        question asked; the one frame where the two gestures are genuinely indistinguishable
        (nothing on it is measurable, so both write all-False) gets no hint at all rather
        than the wrong one.
        """
        key = self._undo_key()
        if key is None:
            return
        res = self._analysis_panel.current_detection()
        df = res.payload.df
        accepted = df["accepted"].to_numpy(dtype=bool)
        if not len(accepted):
            label = None
        elif accepted.any():
            label = "Accept all events"
        elif self._any_measurable(df):
            label = "Reject all events"
        else:                    # nothing on this frame could be accepted: cannot tell them apart
            label = None
        self._history.checkpoint(key, label=label)

    def _checkpoint_labels(self) -> None:
        """The undo hint for a LABEL write.

        Labelling used to share _checkpoint_batch_curation, which names the step by reading
        the accepted column — so labelling a fully-accepted detection recorded "Accept all
        events", an operation the user did not perform and whose name says nothing about
        what Ctrl+Z would undo. Labels do not touch `accepted`, so they need their own word.
        """
        key = self._undo_key()
        if key is not None:
            self._history.checkpoint(key, label="Label events")

    @QtCore.Slot(str, str)
    def _checkpoint_results(self, doc_id: str, module_id: str) -> None:
        """``resultsChanged`` carries no channel, so commit every EventSet-bearing result
        of this (doc, module). A batch detect emits one of these per recording; each
        lands on its own stack."""
        from ..analysis.events import EventSet

        if not self._session.has_document(doc_id):
            return
        for res in self._session.results.for_doc(doc_id):
            if res.module_id == module_id and isinstance(res.payload, EventSet):
                self._history.checkpoint((res.doc_id, int(res.channel), res.module_id))

    def _on_step_recorded(self, label: str, summary: dict) -> None:
        """Say out loud when a click just replaced curated events.

        The measured defect was not only that re-running detection destroyed 2 hand
        rejections — it was that it did so with ``dialogs shown: []`` and nothing in the
        status bar, so there was no moment at which the user could have known to act. A
        modal here would be wrong (re-detect is now the RECOMMENDED move after the
        baseline change, and it is fully undoable), but silence is what made the loss
        invisible. So it is a status line, and it names the remedy.
        """
        if self._restoring or self._history.is_applying():
            return
        # The counts are the HAND ones (core/history.py: _hand_curation). A table whose
        # only rejection is an automatic `clipped_by_boundary` loses nothing a re-run
        # cannot re-derive, so warning about it is how a user learns to ignore the
        # warning that matters.
        bits = []
        if summary.get("n_hand_rejected"):
            bits.append(f"{summary['n_hand_rejected']} rejected by hand")
        if summary.get("n_notes"):
            bits.append(f"{summary['n_notes']} with your own flag")
        if summary.get("n_added"):
            bits.append(f"{summary['n_added']} hand-added")
        if not bits or label not in ("Re-run detection", "Detection"):
            return
        held = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]
        self.statusBar().showMessage(
            f"Detection replaced the previous table ({summary.get('n_events', 0)} events, "
            f"{held}). Your curation is not carried over — "
            f"Edit ▸ Undo puts it back.", 15000)

    @QtCore.Slot(str, int)
    def _on_steps_evicted(self, doc_id: str, total: int) -> None:
        """Say that undo history was thrown away to stay under the memory ceiling.

        The stack is bounded (``core/history.py``: one events table can be 1.1 GB), so
        SOMETHING has to go when the ceiling binds — but the app's own rule is that a loss
        the user can notice is never silent, and this one is invisible everywhere else: the
        Edit menu's "nothing to undo on this event table" is the same sentence it shows for
        a table nobody has ever edited. The menu says it too (:meth:`_sync_undo_actions`),
        so the two surfaces cannot disagree about it.
        """
        name = (self._session.document(doc_id).display_name
                if self._session.has_document(doc_id) else Path(source_of(doc_id)).name)
        # The same plural fix as :meth:`_on_steps_depth_dropped` below, in this toast's
        # own tense: the two ceilings refuse by their own names but must not also speak
        # differently ABOUT the loss, and leaving one of the pair saying "step(s)" would
        # have put the tic back on screen for whichever ceiling bound first.
        lost = ("its oldest step was dropped and cannot be undone." if total == 1
                else f"its {total} oldest steps were dropped and cannot be undone.")
        self.statusBar().showMessage(
            f"Undo history for “{name}” is at the {MAX_BYTES // (1024 * 1024)} MiB memory "
            f"ceiling — {lost}", 12000)

    @QtCore.Slot(str, int)
    def _on_steps_depth_dropped(self, doc_id: str, total: int) -> None:
        """Say that undo history was thrown away at the DEPTH ceiling — the same loss as
        :meth:`_on_steps_evicted`, refused by its own name.

        This was the silent half: only the byte path counted and announced its drops, so
        an ordinary curation sprint past the 30-step cap began discarding the oldest
        steps with nothing anywhere — no status clause while it happened, and at the undo
        floor the menu's never-edited wording (see :meth:`_sync_undo_actions`, which now
        keys on ``depth_dropped`` too). The app's own rule is that a loss the user can
        notice is never silent, and 20 events left rejected after a full undo walk-back
        is a loss the user notices.
        """
        name = (self._session.document(doc_id).display_name
                if self._session.has_document(doc_id) else Path(source_of(doc_id)).name)
        # PLURALISED RATHER THAN PARENTHESISED, here and in the Plots window's standing
        # line (`plots_panel._sync_config_buttons`), which borrows this exact clause. The
        # old spelling read "1 of its oldest step(s) HAVE been dropped" at the first drop
        # — and the first drop is the one this toast is most likely to be read on, since
        # it is fired at the instant the ceiling starts biting. `_sync_undo_actions` 60
        # lines below has always got it right ("step" plus a conditional s), so the
        # window was disagreeing with itself between the status bar and the Edit menu.
        # Both branches are written out as whole sentences rather than assembled from a
        # verb and a plural, because the Plots window carries the same two sentences
        # verbatim and ``test_the_two_windows_speak_one_dialect`` pins the two files
        # together by reading the clause out of `inspect.getsource`.
        lost = ("its oldest step has been dropped and cannot be undone." if total == 1
                else f"its {total} oldest steps have been dropped and cannot be undone.")
        self.statusBar().showMessage(
            f"Undo history for “{name}” is at its {MAX_DEPTH}-step ceiling — {lost}",
            12000)

    def _on_edit_menu_about_to_show(self) -> None:
        """Re-derive the labels as the menu opens — and commit any table that was edited
        in place without announcing it, so what the menu offers to undo is what actually
        happened last."""
        self._checkpoint_everything()
        self._sync_undo_actions()

    def _sync_undo_actions(self) -> None:
        """Put what will be undone INTO the menu text.

        A disabled QAction never receives a click, so it cannot explain itself on
        demand, and a tooltip on a native macOS menu item never renders. Both of this
        project's UI lessons therefore point the same way: the label carries the reason.
        The labels, in the order the branches below test for them: "Undo <what>" (there
        is a step and it names itself); a bare "Undo" (there is one and it does not);
        "Undo — nothing on this table; the threshold detector (3 steps) on this channel"
        (the history is real and one detector switch away); "Undo — this table's oldest
        history was dropped at the 30-step ceiling (4 steps lost)" (it existed and a
        ceiling took it — see the comment at that branch, which names each ceiling);
        "Undo — nothing to undo on this event table" (there IS history in this session,
        but not for this recording); and a bare "Undo" when there is none at all. Every
        count in them is pluralised where it is built, so those are real labels.

        THE DETECTOR-SWITCH LABEL EXISTS FOR ONE CASE. Running the ML detector on a
        curated channel used to land on "nothing to undo on this event table": honest,
        and indistinguishable from "your curation is gone". The threshold detector's
        stack was intact the whole time — under the same recording and channel, keyed by
        the other detector — and nothing pointed at it."""
        key = self._undo_key()
        siblings = self._history.siblings_with_history(key)
        undo_target, redo_target = self._undo_target(), self._redo_target()
        for act, target, verb in ((self._act_undo, undo_target, "Undo"),
                                  (self._act_redo, redo_target, "Redo")):
            if target == _UNDO_MEASUREMENTS:
                label = (self._measure_history.undo_label() if verb == "Undo"
                         else self._measure_history.redo_label())
            elif target == _UNDO_EVENTS:
                label = (self._history.undo_label(key) if verb == "Undo"
                         else self._history.redo_label(key))
            else:
                label = ""
            act.setEnabled(target is not None)
            if target is not None and label:
                act.setText(f"{verb} {label}")
            elif target is not None:
                act.setText(verb)
            elif siblings:
                where = ", ".join(
                    f"{_DETECTOR_LABELS.get(mid, mid)} ({n} step{'s' if n != 1 else ''})"
                    for mid, n in sorted(siblings.items()))
                act.setText(
                    f"{verb} — nothing on this table; {where} on this channel")
            elif self._history.evicted(key) or self._history.depth_dropped(key):
                # NOT the same statement as "nothing to undo", and the old wording made the
                # two indistinguishable: this table HAD history and the app threw it away —
                # to stay under the memory ceiling, or past the 30-step depth ceiling (the
                # one an ordinary curation sprint reaches, and the one that used to keep
                # the never-edited wording). A user told their table is untouched has no
                # reason to look for the work; one told the history was dropped does.
                # Each ceiling is named by its own name, and both when both bound.
                lost = []
                n_mem = self._history.evicted(key)
                if n_mem:
                    lost.append(f"the memory ceiling ({n_mem} step"
                                f"{'s' if n_mem != 1 else ''} lost)")
                n_depth = self._history.depth_dropped(key)
                if n_depth:
                    lost.append(f"the {MAX_DEPTH}-step ceiling ({n_depth} step"
                                f"{'s' if n_depth != 1 else ''} lost)")
                act.setText(
                    f"{verb} — this table's oldest history was dropped at "
                    + " and ".join(lost))
            elif self._history.has_any():
                act.setText(f"{verb} — nothing to {verb.lower()} on this event table")
            else:
                act.setText(verb)

    def _undo_events(self) -> None:
        """Edit ▸ Undo — on whichever of the two tables moved most recently."""
        target = self._undo_target()
        if target is None:
            self._sync_undo_actions()
            return
        if target == _UNDO_MEASUREMENTS:
            label = self._measure_history.undo()
        else:
            label = self._history.undo(self._undo_key())
        if label:
            self._take_last(self._undo_journal, target)
            self._redo_journal.append(target)
            self.statusBar().showMessage(f"Undone: {label}.", 8000)
        self._sync_undo_actions()

    def _redo_events(self) -> None:
        target = self._redo_target()
        if target is None:
            self._sync_undo_actions()
            return
        if target == _UNDO_MEASUREMENTS:
            label = self._measure_history.redo()
        else:
            label = self._history.redo(self._undo_key())
        if label:
            self._take_last(self._redo_journal, target)
            self._undo_journal.append(target)
            self.statusBar().showMessage(f"Redone: {label}.", 8000)
        self._sync_undo_actions()

    def _mark_dirty(self) -> None:
        """Flag that the session holds unsaved analysis a project save would persist, and
        repaint the title marker. A no-op during a project restore — those mutations are
        reconstruction, not user edits, and the restore clears the flag once it settles."""
        if self._restoring:
            return
        self._session_revision += 1
        self._dirty = True
        if self._figure_library is not None:
            self._figure_library.schedule_refresh()
        self._update_title()
        controller = self._recovery_controller
        if controller is not None:
            controller.note_dirty(self._session_revision)

    def _mark_dirty_for_document_set(self) -> None:
        """``documentOpened`` / ``documentClosed``'s entry into the dirty set — CONDITIONAL.

        Opening a recording and touching nothing is not unsaved work. Nothing was created:
        the session holds exactly what File ▸ Open would rebuild in a second from a path the
        app already lists under Recent. Treating it as a mutation armed the quit guard on the
        most ordinary action in the program — open a file, look at it, quit — and answered a
        Save/Discard/Cancel modal at the user about a document, which is the one thing a
        project stores as a mere reference.

        CLOSING one is the mirror of that action and gets the mirror rule. It used to be
        unconditional, which reached the same nag by the opposite door: open two recordings,
        close one, and the title grew its ``•`` and quitting prompted — over a session whose
        remaining content is one file path. (Closing a recording that carries results is
        guarded separately, by a confirmation that names what is discarded, and what it
        discards is gone rather than unsaved — a Save prompt afterwards would offer to write
        a project that could not contain it anyway.)

        Either IS a change once a project is open, because then there is a file on disk that
        claims to describe this session and no longer does: the recording just added would be
        missing the next time it is opened, and the one just closed would come back. So the
        flag is set only when ``_project_path`` is set. Everything else in the dirty set
        (detections, calibration, curation, measurements) is work the user cannot get back by
        reopening a file, and is tracked unconditionally.
        """
        if self._project_path:
            self._mark_dirty()

    def _mark_clean(self) -> None:
        """Clear the unsaved-changes flag (a successful save, or a settled restore) and drop
        the title marker."""
        self._dirty = False
        self._update_title()

    def _has_work_to_lose(self) -> bool:
        """True when the session holds anything a project save would persist.

        "Any documents open" is NOT that question. Manual measurements and the label
        vocabulary live outside Session documents and can be the only irreplaceable work.
        A named project is also work even after its last document is removed: that removal
        is precisely the unsaved edit, and letting the now-empty session answer "nothing to
        lose" would close without writing the deletion or asking about it.
        """
        if self._project_path or self._session.documents():
            return True
        if self._label_vocab.entries:
            return True
        meas = self._analysis_panel.measure_panel.measurements_df
        if meas is not None and len(meas) > 0:
            return True
        # An analytical recipe/result can be the only irreplaceable content in an unnamed
        # session (for example a carried self-contained subpopulation or PSD snapshot whose
        # source drive is currently absent).  Workspace-only edits therefore receive the
        # same Save / Discard / Cancel close guard as event tables and measurements.
        return bool(self._analytical_workspace.artifacts)

    def _visible_project_dirty(self) -> bool:
        """The user-facing unsaved predicate shared by every editing window.

        ``_dirty`` is intentionally a little broader than the title marker: internal
        bookkeeping can mark an otherwise empty, unnamed session, but there is no project
        work to save in that state. Publishing the raw bit would let the Gallery say
        "Unsaved changes" while the main title truthfully showed no marker.
        """
        return bool(self._dirty and self._has_work_to_lose())

    def _update_title(self) -> None:
        """Compose the window title: ``NanoScope`` plus the project file name once one is
        set, with a leading ``• `` while there is unsaved work worth keeping. Publish that
        exact same answer to every open editing window."""
        if self._project_path:
            base = f"NanoScope — {Path(self._project_path).name}"
        elif self._recovery_suggested_project_path:
            base = (f"NanoScope — Recovered "
                    f"{Path(self._recovery_suggested_project_path).name}")
        else:
            base = "NanoScope"
        show_marker = self._visible_project_dirty()
        self.setWindowTitle(f"• {base}" if show_marker else base)
        self.projectSaveStateChanged.emit(self._project_path or "", show_marker)

    def _confirm_close_choice(self) -> str:
        """Ask Save / Discard / Cancel about unsaved work, returning one of
        ``_CLOSE_SAVE`` / ``_CLOSE_DISCARD`` / ``_CLOSE_CANCEL`` (Cancel is also the answer
        for Esc or the window's close box). Overridable so a test can choose without a live
        modal.

        Shared by every action that would discard the session: closing the window, and opening
        a project over it. The wording is deliberately neutral about which, so both callers can
        use the ONE seam (and the suite has ONE thing to override)."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Unsaved work")
        box.setText("You have unsaved analysis.")
        box.setInformativeText("Save your project first?")
        save_btn = box.addButton("Save", QtWidgets.QMessageBox.AcceptRole)
        discard_btn = box.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(save_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is discard_btn:
            return _CLOSE_DISCARD
        if clicked is save_btn:
            return _CLOSE_SAVE
        return _CLOSE_CANCEL

    def _run_save(self) -> bool:
        """Run the SAME project-save flow Ctrl+S uses and report whether a file was written
        (False if the Save-As dialog was cancelled or the save failed). Overridable so a test
        can simulate the outcome without a dialog."""
        return bool(self._save_project())

    def _teardown(self) -> int:
        """Abort any background jobs and persist window settings — the shared close path for
        both an unguarded close and a confirmed one. RETURNS the number of worker threads
        that were still running when the drain gave up (0 in a healthy quit).

        The drain is the one place worth spending real time on a worker: on quit we would
        like no live QThread left, and every job polls the cooperative cancel token so a
        couple of seconds is generous. It stays BOUNDED — a worker that will not stop
        delays the quit, it never prevents it, and it is never terminated.

        THE RETURN VALUE IS THE POINT. It used to be discarded, and that discard was the
        whole of the SIGABRT on quit: a worker still running when the drain expired stayed
        in ``worker_thread._DETACHED``, that global was cleared at interpreter shutdown,
        the C++ ``QThread`` was deleted while its OS thread ran, and Qt called ``qFatal``
        (measured: exit 134, 4/4 runs). Two things now use it — the budget below was
        raised because a cooperative job stops in milliseconds, so a longer wait only ever
        bites on a genuinely uninterruptible C call (a TensorFlow import is 4.1 s), and
        ``gui.app.main`` reads the count back to guarantee the process leaves cleanly
        either way. Settings are saved BEFORE any of that, and synced, so nothing depends
        on how the process ends.

        WHAT IS STOPPED IS DISCOVERED, NOT LISTED — see :meth:`_owned_background_holders`.
        This method used to name ``_controller``, ``_analysis_controller`` and
        ``_batch_controller``, which is a list somebody has to remember to extend; the
        Plots window's fit-note / bootstrap-CI worker was added without being added here,
        and measured through a real ``MainWindow`` it survived the quit — ``_teardown()``
        returned 0 and ``detached_count()`` was 0 while the worker's QThread was still
        running, so ``gui.app.main``'s final drain had nothing to drain and the live
        thread went into interpreter shutdown.

        THE COUNT IS MEASURED FROM THE THREADS, NOT FROM ANYBODY'S INTENT. It used to be
        ``drain_detached()``, which can only see threads that reached
        ``retire_worker_thread`` — so a holder nobody asked to stop, and a job started
        with ``JOB_NONE`` (the Plots window's fit-note batch is one, and it is absent
        from the run gate's registry by design), were both counted as stopped. Measured
        through a real ``MainWindow`` on a plain child widget holding a live worker:
        ``teardown() returned: 0    detached_count(): 0`` with ``isRunning() == True``.
        :func:`~nanoscope.gui.worker_thread.drain_worker_threads` waits on the union of
        every unstopped thread and parks whatever survives, which is what keeps
        ``gui.app.main``'s final ``drain_detached()`` able to see it."""
        self._stop_owned_background_work()
        self._save_settings()
        return drain_worker_threads(_TEARDOWN_DRAIN_MS)

    # -- teardown: ownership, not a list -----------------------------------------------------
    #
    # The two protocols by which anything in this tree stops background work, and the two
    # KINDS of holder that speak them. They are not alternatives and a holder is asked
    # through every one it implements: a controller stops its worker in ``abort()``, and a
    # standalone window stops its own in ``closeEvent`` — ``PlotsWindow`` is the one that
    # does today, and it is precisely the one the old hand-written list missed. Neither
    # name is a list of OBJECTS; they are the vocabulary an owned object uses to say "I may
    # be holding a worker", which is what makes a holder added tomorrow stop on quit
    # without an edit here.
    #
    # ``close`` is deliberately only ever asked of a WIDGET THAT IS A WINDOW. Half the Qt
    # object tree has a method spelled ``close`` that has nothing to do with a worker (a
    # QIODevice, a model's backing store), and a dock or a tab page has no ``closeEvent``
    # Qt will ever deliver on a quit — closing one would be a state change, not a stop.
    #
    # THE WINDOW FILTER BELONGS TO ``close`` AND TO NOTHING ELSE. It was first written as a
    # filter on the HOLDER — a widget was admitted only if ``isWindow()`` — and that
    # excluded, on the day it was written, every panel in this tree: an ``AnalysisPanel``,
    # a ``PlotsPanel``, a dock's contents are all non-window child widgets. Measured on a
    # plain child widget of a real ``MainWindow`` holding a running worker and an
    # ``abort()`` that retires it:
    #
    #     isWindow(): False   has abort(): True
    #     discovered as a holder: False    (21 holders found)
    #     teardown() returned: 0    detached_count(): 0
    #     panel.abort() was called: False
    #     its QThread STILL RUNNING: True
    #     QThread: Destroyed while thread '' is still running
    #
    # which is the by-name list's failure exactly, re-created by a filter instead of by an
    # enumeration. ``abort()`` is not a window's protocol and never was — it is what an
    # object that OWNS A WORKER answers to, and owning one is the only qualification that
    # has ever mattered. So the filter moved off the holder and onto the one protocol whose
    # reason it actually is.
    _STOP_ABORT = "abort"
    _STOP_CLOSE = "close"

    def _stop_protocols(self, holder: object) -> list[str]:
        """Which stop protocols ``holder`` speaks — the ONE place the two are told apart.

        ``abort`` is asked of anything that implements it, widget or not, window or not.
        ``close`` is asked only of a widget that is a window, for the reason above. A
        holder that speaks both is asked through both."""
        protocols: list[str] = []
        try:
            if callable(getattr(holder, self._STOP_ABORT, None)):
                protocols.append(self._STOP_ABORT)
            if (isinstance(holder, QtWidgets.QWidget)
                    and (holder.isWindow()
                         or bool(holder.property("nanoscopeAnalysisExplorerPage")))):
                protocols.append(self._STOP_CLOSE)
        except RuntimeError:      # the C++ object is already gone — nothing to stop
            return []
        return protocols

    def _owned_background_holders(self) -> list[object]:
        """Everything this window OWNS that could be holding a long-lived worker.

        DISCOVERED FROM OWNERSHIP. Two sources, both of which are what "this window owns
        it" actually means in this codebase and neither of which anyone has to maintain:

        * this window's own attributes — every controller (``LoadController``,
          ``AnalysisController``, ``BatchDetectController``) and every lazily-built
          standalone window (``_plots_window``, ``_psd_window``, ``_subpop_window``,
          the event gallery …) is one, because ``main_window.py`` is the sole wirer and
          therefore the only place any of them is constructed and kept;
        * the Qt object tree beneath this window — a surface window is built as
          ``PlotsWindow(self._session, self)``, i.e. parented HERE, so it is reachable
          even if a future one is held in a list or a dict instead of an attribute.

        Deduplicated by identity (the two sources overlap heavily and both find the
        surface windows), and ``self`` is excluded: this window's own ``close`` is the
        caller, not a thing to be stopped.

        ADMITTED BY PROTOCOL, NOT BY KIND. A holder is anything that speaks at least one
        of the two stop protocols (see :meth:`_stop_protocols`): anything with ``abort()``
        — widget or not, window or not — and any widget that is a window, which is the
        ``close()`` route. The ``isWindow()`` test used to sit HERE instead, gating
        admission itself, and it silently re-created the very defect the discovery
        replaced: **every panel in this tree is a non-window child widget**, so a panel
        holding a worker was not a holder at all, its ``abort()`` was never called, and
        the quit reported itself clean over a live QThread (the measurement is beside
        ``_STOP_ABORT`` above). A window is a place a worker can be CLOSED from; it was
        never the definition of a thing that owns one.

        NOT NARROWED FURTHER, DELIBERATELY. Measured on a real window with the Plots
        window open: 1664 QObjects in the tree, 36 holders, 6.0 ms — of which the three
        controllers and the Plots window are the live ones and the rest are Qt's own
        hidden popups (8 QMenu, 24 QFrame). Closing a hidden popup is a no-op, and the
        one thing that could make it not a no-op — ``WA_DeleteOnClose``, which turns
        ``close()`` into a delete — is set nowhere in ``nanoscope/gui`` and is guarded
        against in ``test_fup_wiring_and_suite_green.py``. The tempting narrowing (only
        widgets that OVERRIDE ``closeEvent``) would have cost the ``hideEvent`` route,
        which ``close()`` also runs, for six milliseconds — the wrong trade for a method
        whose entire purpose is to stop being clever about which holders exist.
        """
        holders: list[object] = []
        seen: set[int] = {id(self)}
        for obj in list(vars(self).values()) + self.findChildren(QtCore.QObject):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if self._stop_protocols(obj):
                holders.append(obj)
        return holders

    def _stop_owned_background_work(self) -> None:
        """Ask every owned holder to stop, through every protocol it implements —
        :meth:`_stop_protocols` is the one place that decides which those are, so this
        loop and :meth:`_owned_background_holders` can never disagree about whether an
        object is a holder and about what it would be asked.

        Nothing here is allowed to prevent the quit: a holder whose C++ object has already
        gone (``RuntimeError``) or that refuses for a reason of its own must not keep the
        other holders from being asked, and must not keep ``_save_settings`` below from
        running. The bounded drain in :meth:`_teardown` is what actually decides whether
        the process leaves with a live thread, and it runs either way.

        The no-``terminate()`` / no-unbounded-``wait()`` rule is inherited rather than
        restated: every ``abort()`` and every window ``closeEvent`` in this tree retires
        its thread through :func:`nanoscope.gui.worker_thread.retire_worker_thread`, which
        is the one place that rule lives.
        """
        for holder in self._owned_background_holders():
            for name in self._stop_protocols(holder):
                stop = getattr(holder, name, None)
                if not callable(stop):
                    continue
                try:
                    stop()
                except RuntimeError:      # the C++ object is already gone
                    break
                except Exception:         # noqa: BLE001 — a quit is never blocked by one holder
                    continue

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Guard unsaved analysis on quit/close. With unsaved work AND something a save would
        persist (see :meth:`_has_work_to_lose` — including a named project, a standalone label
        vocabulary, open documents or manual measurements), prompt Save / Discard / Cancel;
        with nothing to lose, close straight away.

        Decision table (once the prompt is shown):

        * Save    → run the project save; a written file closes, a cancelled/failed save
                    keeps the window open (the work is never lost).
        * Discard → close, dropping the work — the explicit choice.
        * Cancel  → keep the window open (also the Esc / close-box answer).

        The choice and the save both route through :meth:`_confirm_close_choice` and
        :meth:`_run_save` so the suite never faces a live modal."""
        # The other moment core/history.py's safety net is about: an edit that reached no
        # signal has not set the dirty flag either, so without this the quit prompt would
        # never be shown for it and the work would go silently. Committing it here sets
        # `_dirty` through `history.changed` in time for the test below.
        self._checkpoint_everything()
        if self._dirty and self._has_work_to_lose():
            choice = self._confirm_close_choice()
            if choice == _CLOSE_CANCEL:
                event.ignore()
                return
            if choice == _CLOSE_SAVE and not self._run_save():
                event.ignore()   # Save-As cancelled or the save failed — never lose the work
                return
            if choice == _CLOSE_DISCARD:
                self._discard_recovery(self._recovered_candidate)
            # _CLOSE_DISCARD, or a save that wrote a file: fall through and close.
        self._teardown()
        super().closeEvent(event)
