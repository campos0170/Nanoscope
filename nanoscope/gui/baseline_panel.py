"""Open-pore baseline controls: the app PROPOSES, the user CONFIRMS.

WHY THERE IS A CONFIRMATION CLICK
=================================
The characterisation baseline decides every ``depth_pA`` in the table, and no global
default is correct: the owner's apo recording needs a 2 s rough window (it holds a
521 ms partial clog, and a 500 ms window sitting inside a 521 ms state *is* that
state), while a gating pore in the same corpus needs 30 ms. So the app measures the
recording and offers parameters -- and stops, because two things measured on this
data say it must not decide alone:

* the obvious noise estimator reads 7x low on these already-low-passed exports, and
  its usable substitute over-reads 3-4x on a gating pore;
* the reference's own QC triple is monotone in window length, so ANY search over it
  selects the shortest window there is -- exactly the defect this change removes.

`nanoscope.analysis.baseline_probe` carries the measurements and the reasoning; this
widget shows them and takes the click.

WHAT THIS WIDGET IS
===================
A pure view. It holds parameter VALUES, renders a proposal and a health block, and
emits intent. It never measures: no estimator call, no probe call, nothing that
touches a trace -- the probe STAGE alone measured 343 ms on a 3 M-sample recording
and 1,073 ms on 9.1 M (medians of three presses, 2026-09-01; the first press of a
file is colder), and the whole "Measure this recording" job -- conditioning + probe +
health pass -- 2 s on 3 M samples and 6 s on 9 M, which on the GUI thread is a
visible freeze (see ``baseline_probe.probe_and_propose`` for the bench). The owning
:class:`~nanoscope.gui.analysis_panel.AnalysisPanel` runs the probe off-thread on the
shared controller, exactly as it runs a threshold sweep, and pushes the answer back
in through :meth:`set_proposal`. An AST guard in ``tests/gui/test_baseline_panel.py``
fails the build if this module ever imports a sibling gui panel or calls a measuring
function.

THE UX RULES THIS CODEBASE ALREADY PAID FOR
===========================================
* **"Locked" is styling, never** ``setEnabled(False)``. Qt does not deliver a click to
  a disabled widget, so a disabled button cannot explain itself. Every button here
  stays enabled and answers "why not?" in :attr:`gate_label` -- the same rule the
  workflow strip and the analysis launcher follow.
* **A gate reports the FIRST REAL blocker**, not a downstream one: "select a
  recording" before "run a probe first", because telling a user to press a button
  that is not reachable yet is worse than saying nothing.
* **Tooltips go on widgets.** (Never on a native macOS menu item, where they never
  render.)
* **A CONDEMNED PROPOSAL IS REFUSED ONCE, THEN MARKED FOREVER.** The probe can offer
  windows the estimator's own checks then condemn -- on the owner's -700 mV apo
  recording it proposes 0.1 / 0.03 s and ``BaselineHealth.ok`` comes back False -- and
  Apply used to take that click plain, unmarked, in one press. It now refuses the
  first press with the reason, arms itself (``Use these values anyway``, in the
  warning colour), and, once accepted, says in :attr:`source_label` for as long as
  the form holds those numbers that they were accepted over a failed check. See
  :meth:`BaselinePanel._recompute_condemnation` for why this is a two-step and not a
  refusal.
* **The inspector page owns scrolling.** Actions use a two-row grid, so individual
  button scrollbars cannot obscure them. The containing bounded page scroll keeps
  this form's minimum width from becoming a minimum width for the entire dock.

Signals:
    probeRequested(): "Measure this recording" clicked and nothing blocks it.
    paramsChanged(dict): a value changed; carries :meth:`params`.
    addReferenceSelectionRequested(): deliberately copy the trace's time selection.
    referencePreviewRequested(): preview a valid supplied reference without applying it.
"""

from __future__ import annotations

import html
import math
from typing import Any, Mapping

from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.baseline import (
    BASELINE_PARAM_KEYS,
    BATCH_PROVENANCE_CLAUSE,
    BATCH_SCOPE_KEY,
    BATCH_SOURCE,
    FALLBACK_CLAMPED,
    BaselineConfig,
    BaselineHealth,
    DEFAULT_BASELINE,
    PARAM_SOURCES,
    derive_decim,
)
from ..analysis.baseline_probe import BaselineProposal, proposal_summary
from ..analysis.open_pore_reference import OpenPoreReference

__all__ = ["BaselinePanel"]

#: How each `PARAM_SOURCES` value reads to a user. The badge is the whole staleness
#: story in one word: a row measured with ``'default'`` was never looked at.
_SOURCE_LABELS: dict[str, str] = {
    "default": "app default (this recording was not measured)",
    "proposed": "measured on this recording, and confirmed",
    "user": "set by hand",
    "restored": "restored from the stored analysis",
    # Batch detect copies ONE form onto every open recording. Until sweep 7 the copy
    # arrived stamped 'proposed' and this panel read "measured on this recording,
    # and confirmed" over windows nobody had measured on it (S7-DET1). Built on
    # `baseline.BATCH_PROVENANCE_CLAUSE` -- the ONE clause the result advisory
    # (`baseline.batch_copied_warning`) is built from -- so this badge and the Stats
    # advisory cannot drift apart or claim what the stamp does not know. The first
    # version said "proposed or confirmed for a DIFFERENT recording" here, and said
    # it over windows the user had TYPED into the other recording's form (E4): the
    # resolver stamps every non-default form source 'batch' alike and keeps none.
    "batch": "COPIED from the batch form: " + BATCH_PROVENANCE_CLAUSE,
    # Legacy: the length clamp used to overwrite the source with this (S7-DET3). A
    # project written before sweep 7 can still carry it; a fresh run never does --
    # the clamp is reported on the line under the source, not as the source.
    "fallback": ("restored from an older analysis whose estimator fell back on this "
                 "trace (legacy stamp; the windows were not chosen by anyone)"),
}

#: Health levels, worst first, with the colour each renders in. Matches the level
#: strings `nanoscope.analysis.baseline.BaselineCheck` emits.
_LEVEL_STYLE: dict[str, str] = {
    "refuse": "#ff6b6b",
    "warn": "#ffb454",
    "info": "#8a939c",
    "ok": "#7ec97e",
}
_LEVEL_ORDER = ("refuse", "warn", "info", "ok")

#: The two `PARAM_SOURCES` values that mean A HUMAN DECIDED THESE. Everything else
#: is the app talking to itself. A confirmed value is never seeded over
#: (`BaselinePanel.set_params`) and is dropped only when the recording itself
#: changes (`BaselinePanel.set_recording`), because "the app proposes, the user
#: confirms" is worth nothing if navigating away and back undoes the confirmation.
#: (The drop on a recording change is NOT limited to these two -- see
#: `set_recording` for why every source is per-recording.)
_CONFIRMED_SOURCES = frozenset({"user", "proposed"})

#: The `PARAM_SOURCES` values under which this panel CANNOT say which recording the
#: windows in the form belong to, so `batch_params` names none. ``'default'``:
#: nobody measured anything. `BATCH_SOURCE`: the form was seeded from a result that
#: itself says "copied from the batch form, set for a DIFFERENT recording" -- and
#: the stored result does not record which (the detector drops the scope from what
#: it stores, so a per-load id never reaches a project file).
#: Every other source was set FOR the panel's own target: by a hand edit or a
#: confirmation here, or seeded out of that target's own stored run.
_UNSCOPED_SOURCES = frozenset({"default", BATCH_SOURCE})

_DIM_BUTTON = "color:#6b7178;"
_OK_BUTTON = ""
#: The armed override: an Apply that will accept a CONDEMNED window on the next
#: click. It is neither dim (dim means "blocked") nor plain (plain means "ordinary"),
#: because it is a third thing -- reachable, and a mistake.
_WARN_BUTTON = "color:#ffb454;"

#: The ONE `BaselineCheck` level that certifies. Everything else condemns the windows
#: it was measured on -- which is `BaselineHealth.ok`'s own rule (``all(c.ok)``, and
#: ``BaselineCheck.ok`` is ``level == 'ok'``), spelled from the level so a block
#: restored from provenance -- a dict of rows, with no ``.ok`` anywhere -- is judged
#: identically to a live one.
#:
#: NOT ``{'warn', 'refuse'}``, which is what this was first written as. Those are the
#: two levels the checks emit TODAY; ``'info'`` is in the vocabulary, is rendered by
#: this panel, and already makes ``BaselineHealth.ok`` False. Enumerating the bad
#: levels means the gate and the flag it exists to enforce disagree the day a check
#: starts using one -- and the whole finding here is a panel that disagreed with its
#: own health block. An allow-list of one cannot drift.
_CERTIFYING_LEVEL = "ok"

#: The confirmation on a condemned proposal, once the user has been told. Kept
#: distinct from "Use these values" so the click that accepts a warned window and the
#: click that accepts a clean one are not the same click.
_APPLY_TEXT = "Use these values"
_APPLY_ANYWAY_TEXT = "Use these values anyway"

#: Absolute spin bounds. ``base_s``'s LIVE maximum is tighter -- it is pinned to
#: ``rough_s``, so ``base_s > rough_s`` (which `BaselineConfig` refuses, because a
#: short pass at least as wide as the rough one masks nothing and the two-pass
#: structure is inoperative) can never be typed in. `BaselinePanel._set_windows` opens
#: that cap back to this value while it lands both numbers.
_ROUGH_RANGE = (0.002, 60.0)
_BASE_RANGE = (0.001, 30.0)


class _OpenPoreReferenceEditor(QtWidgets.QWidget):
    """A recording-specific reference form, never a trace reader or estimator.

    Invalid drafts stay visible and serialize as invalid input rather than quietly
    reverting to Automatic. The owner checks ``error()`` before launching work; the
    production resolver independently validates the recipe against the recording.
    """

    changed = QtCore.Signal()
    selectionRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._confirmed = False
        self._duration_s: float | None = None
        self._fs: float | None = None
        self._restore_error = ""
        self._invalid_recipe = None
        self._invalid_reference_flags = {}
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("Automatic (existing estimator)", "automatic")
        self.mode.addItem("Known open-pore current", "constant")
        self.mode.addItem("Selected open-pore intervals", "intervals")
        self.mode.setToolTip(
            "Choose a reference for this recording only. Automatic keeps the existing "
            "estimator. The other modes use information you provide, not independently verified.")
        layout.addWidget(self.mode)

        self.known_box = QtWidgets.QWidget()
        self.known_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        known = QtWidgets.QFormLayout(self.known_box)
        known.setContentsMargins(0, 0, 0, 0)
        known.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.current = QtWidgets.QLineEdit()
        self.current.setPlaceholderText("Signed current, e.g. −120")
        self.current.setToolTip(
            "The signed open-pore current in calibrated pA for THIS recording. "
            "A constant reference is applied to the whole recording; do not enter an absolute magnitude.")
        known.addRow("Open-pore current (pA)", self.current)
        layout.addWidget(self.known_box)

        self.intervals_box = QtWidgets.QWidget()
        self.intervals_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        intervals = QtWidgets.QVBoxLayout(self.intervals_box)
        intervals.setContentsMargins(0, 0, 0, 0)
        intervals.setSpacing(4)
        caption = QtWidgets.QLabel(
            "Mark intervals known to contain open-pore current, not blocked states. "
            "Times are seconds from the recording start, not the displayed clock.")
        caption.setWordWrap(True)
        caption.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        intervals.addWidget(caption)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Start (s)", "End (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setMinimumWidth(0)
        self.table.setFixedHeight(118)
        self.table.setToolTip(
            "Ordered, non-overlapping open-pore intervals relative to this recording's start. "
            "Start is included and end excluded; each interval needs at least five samples. "
            "Double-click a time to edit it.")
        intervals.addWidget(self.table)
        buttons = QtWidgets.QGridLayout()
        self.selection_button = QtWidgets.QPushButton("Add current trace selection")
        self.selection_button.setToolTip(
            "Copy the highlighted time range from the trace. This does not run detection.")
        self.add_button = QtWidgets.QPushButton("Add interval")
        self.add_button.setToolTip("Add a blank interval to enter start and end times by hand.")
        self.remove_button = QtWidgets.QPushButton("Remove selected")
        self.remove_button.setToolTip("Remove only the selected rows from this reference form.")
        buttons.addWidget(self.selection_button, 0, 0, 1, 2)
        buttons.addWidget(self.add_button, 1, 0)
        buttons.addWidget(self.remove_button, 1, 1)
        intervals.addLayout(buttons)
        self.method = QtWidgets.QComboBox()
        self.method.addItem("Constant reference (stable recording)", "constant")
        self.method.addItem("Linear between interval anchors (drift)", "linear")
        self.method.setToolTip(
            "Constant uses the pooled median of selected samples. Linear needs at least two "
            "intervals, interpolates their median levels, and holds the endpoint levels outside "
            "the anchors. Unobserved drift cannot be verified, particularly across long gaps.")
        intervals.addWidget(self.method)
        layout.addWidget(self.intervals_box)

        self.manual_box = QtWidgets.QWidget()
        self.manual_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        manual = QtWidgets.QVBoxLayout(self.manual_box)
        manual.setContentsMargins(0, 0, 0, 0)
        manual.setSpacing(4)
        self.notes = QtWidgets.QLineEdit()
        self.notes.setMaxLength(1000)
        self.notes.setPlaceholderText("Optional source / notes (saved with analysis)")
        self.notes.setToolTip(
            "Where this recording-specific information came from, such as an open-pore measurement "
            "or lab notebook. User-provided information is not independently verified.")
        manual.addWidget(self.notes)
        self.notice = QtWidgets.QLabel(
            "User-provided reference · not independently verified. Preview the baseline, "
            "then run detection deliberately. Existing measurements do not change while you edit.")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color:#8a939c;")
        manual.addWidget(self.notice)
        self.preview_button = QtWidgets.QPushButton("Preview reference and detection…")
        self.preview_button.setToolTip(
            "Preview the reference and candidate events without replacing stored results. "
            "Review the preview before explicitly running detection.")
        manual.addWidget(self.preview_button)
        self.validation_label = QtWidgets.QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextFormat(QtCore.Qt.PlainText)
        self.validation_label.setStyleSheet("color:#ffb454;")
        layout.addWidget(self.manual_box)
        layout.addWidget(self.validation_label)

        self.mode.currentIndexChanged.connect(self._edited)
        self.current.textChanged.connect(self._edited)
        self.method.currentIndexChanged.connect(self._edited)
        self.notes.textChanged.connect(self._edited)
        self.table.cellChanged.connect(self._edited)
        self.add_button.clicked.connect(self._add_blank)
        self.remove_button.clicked.connect(self._remove_selected)
        self.selection_button.clicked.connect(self.selectionRequested)
        self._refresh()

    def reference_mode(self) -> str:
        return str(self.mode.currentData())

    @staticmethod
    def _number(value):
        if "_" in str(value):
            return None
        try:
            number = float(str(value).strip().replace("−", "-"))
        except (ValueError, TypeError):
            return None
        return number if math.isfinite(number) else None

    def _intervals(self):
        return [[self._number(self.table.item(row, col).text())
                 if self.table.item(row, col) else None for col in range(2)]
                for row in range(self.table.rowCount())]

    def params(self) -> dict[str, Any]:
        if self._restore_error:
            return {"baseline_open_pore_reference": self._invalid_recipe,
                    **self._invalid_reference_flags}
        mode = self.reference_mode()
        if mode == "automatic":
            return {}
        recipe = {"version": 1, "mode": mode,
                  "current_pA": self._number(self.current.text()) if mode == "constant" else None,
                  "intervals_s": self._intervals() if mode == "intervals" else [],
                  "interpolation": str(self.method.currentData()) if mode == "intervals" else "constant"}
        if self.notes.text().strip():
            recipe["notes"] = self.notes.text().strip()
        return {"baseline_open_pore_reference": recipe}

    def error(self) -> str:
        if self._restore_error:
            return self._restore_error
        mode = self.reference_mode()
        if mode == "automatic":
            return ""
        if mode == "constant":
            return ("Enter a finite, signed open-pore current in pA."
                    if self._number(self.current.text()) is None else "")
        rows = self._intervals()
        if not rows:
            return "Add at least one trustworthy open-pore interval."
        for row, (start, end) in enumerate(rows, 1):
            if start is None or end is None or not 0 <= start < end:
                return f"Interval {row}: enter finite times with 0 ≤ start < end."
            if self._duration_s is not None and end > self._duration_s:
                return f"Interval {row} ends beyond this recording ({self._duration_s:g} s)."
            if self._fs and (math.ceil(math.nextafter(end * self._fs, -math.inf))
                             - math.ceil(math.nextafter(start * self._fs, -math.inf))) < 5:
                return f"Interval {row} needs at least five samples."
        if any(right[0] < left[1] for left, right in zip(rows, rows[1:])):
            return "Enter open-pore intervals in time order, without overlaps."
        if self.method.currentData() == "linear" and len(rows) < 2:
            return "Linear drift reference needs at least two separate open-pore intervals."
        return ""

    def _refresh(self):
        mode = self.reference_mode()
        self.known_box.setVisible(mode == "constant")
        self.intervals_box.setVisible(mode == "intervals")
        self.manual_box.setVisible(mode != "automatic")
        problem = self.error()
        self.validation_label.setText(problem)
        self.validation_label.setVisible(bool(problem))

    def _edited(self, *_args):
        if self._loading:
            return
        self._confirmed = True
        self._restore_error = ""
        self._invalid_recipe = None
        self._invalid_reference_flags = {}
        self._refresh()
        self.changed.emit()

    def _add_blank(self):
        self._loading = True
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col in range(2):
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(""))
        self._loading = False
        self._edited()
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def _remove_selected(self):
        rows = sorted({item.row() for item in self.table.selectedItems()}, reverse=True)
        if not rows:
            self.validation_label.setText("Select interval rows to remove first.")
            self.validation_label.setVisible(True)
            return
        self._loading = True
        for row in rows:
            self.table.removeRow(row)
        self._loading = False
        self._edited()

    def add_interval(self, start_s, end_s) -> bool:
        start, end = self._number(start_s), self._number(end_s)
        if start is None or end is None or not 0 <= start < end:
            self.validation_label.setText("Select a non-empty time range within this recording.")
            self.validation_label.setVisible(True)
            return False
        if self._duration_s is not None and end > self._duration_s:
            self.validation_label.setText("The selected range extends beyond this recording.")
            self.validation_label.setVisible(True)
            return False
        existing = self._intervals()
        for a, b in existing:
            if a is not None and b is not None and max(a, start) < min(b, end):
                self.validation_label.setText("The selected range overlaps an existing open-pore interval.")
                self.validation_label.setVisible(True)
                return False
        self._loading = True
        self.mode.setCurrentIndex(self.mode.findData("intervals"))
        # Inserting trace selections in order is deterministic and preserves their
        # exact endpoints. Manually entered out-of-order rows remain invalid drafts.
        row = next((index for index, (a, _) in enumerate(existing)
                    if a is not None and start < a), self.table.rowCount())
        self.table.insertRow(row)
        for col, value in enumerate((start, end)):
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(repr(value)))
        self._loading = False
        self._edited()
        return True

    def set_recording(self, fs: float | None, duration_s: float | None):
        self._fs = float(fs) if fs else None
        self._duration_s = (float(duration_s) if duration_s is not None
                            and math.isfinite(float(duration_s)) and duration_s >= 0 else None)
        self._refresh()

    def reset(self):
        self._loading = True
        self.mode.setCurrentIndex(0)
        self.current.clear()
        self.notes.clear()
        self.table.setRowCount(0)
        self.method.setCurrentIndex(0)
        self._confirmed = False
        self._restore_error = ""
        self._invalid_recipe = None
        self._invalid_reference_flags = {}
        self._loading = False
        self._refresh()

    def set_params(self, params: Mapping[str, Any]):
        if self._confirmed:
            return
        active = {k: params[k] for k in ("baseline_reference_applied", "baseline_estimator_bypassed")
                  if params.get(k)}
        if params.get("baseline_reference_restore_error"):
            self._restore_error = str(params["baseline_reference_restore_error"])
            self._invalid_recipe = params.get("baseline_open_pore_reference")
            self._invalid_reference_flags = {
                **active, "baseline_reference_restore_error": self._restore_error}
            self._refresh()
            return
        if active and params.get("baseline_open_pore_reference") is None:
            self._restore_error = ("Stored user reference is missing its recipe. Choose a reference "
                                   "deliberately before detecting; the saved events are unchanged.")
            self._invalid_recipe = None
            self._invalid_reference_flags = active
            self._refresh()
            return
        if "baseline_open_pore_reference" not in params:
            return
        self._invalid_reference_flags = {}
        recipe = params["baseline_open_pore_reference"]
        if recipe is None:
            self.reset()
            return
        try:
            OpenPoreReference.from_dict(recipe)
        except (TypeError, ValueError) as exc:
            self._restore_error = f"Stored reference is invalid or unsupported: {exc}. Review and edit it before detection."
            self._invalid_recipe = recipe
        else:
            self._restore_error = ""
            self._invalid_recipe = None
        if not isinstance(recipe, Mapping) or recipe.get("mode") not in {"constant", "intervals"}:
            self._refresh()
            return
        self._loading = True
        try:
            self.mode.setCurrentIndex(self.mode.findData(recipe["mode"]))
            value = recipe.get("current_pA")
            self.current.setText("" if value is None else str(value))
            self.notes.setText(str(recipe.get("notes", "")))
            method_index = self.method.findData(recipe.get("interpolation", "constant"))
            self.method.setCurrentIndex(max(0, method_index))
            self.table.setRowCount(0)
            rows = recipe.get("intervals_s") or ()
            for values in rows if isinstance(rows, (list, tuple)) else ():
                row = self.table.rowCount()
                self.table.insertRow(row)
                for col in range(2):
                    value = values[col] if isinstance(values, (tuple, list)) and len(values) > col else None
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem(
                        "" if value is None else str(value)))
        finally:
            self._loading = False
        self._refresh()


class BaselinePanel(QtWidgets.QGroupBox):
    """Automatic estimator and opt-in, recording-specific open-pore reference forms.

    Public API:
        params() -> dict                    # the ``baseline_*`` params to run with
        set_params(mapping)                 # seed from a stored detection
        set_recording(fs, target)           # derived decim; ends a confirmation
        set_proposal(proposal)              # a finished probe (or None to clear)
        proposal_text() -> str              # the proposal block AS RENDERED
        set_health(health)                  # the last run's `BaselineHealth`
        set_busy(bool)                      # a probe is in flight
        set_gate(ok, reason)                # can a probe run, and if not why not
        apply_proposal(override=False)      # the confirmation, callable from a test
        condemned_checks() -> tuple[str]    # what condemns the proposal ON OFFER
        accepted_condemned() -> tuple[str]  # what the FORM's values were accepted over
        reference_mode() -> str             # automatic / constant / intervals
        reference_error() -> str            # invalid draft; must block a run
        add_reference_interval(start,end)   # recording-relative seconds; intent only
    """

    probeRequested = QtCore.Signal()
    paramsChanged = QtCore.Signal(dict)
    addReferenceSelectionRequested = QtCore.Signal()
    referencePreviewRequested = QtCore.Signal()
    #: The two VIEW requests, carried separately from `paramsChanged` on purpose:
    #: nothing here is a parameter. Neither changes a measurement, neither belongs in
    #: provenance, and a redraw must never be mistaken for a reason to re-run a
    #: detection -- which is exactly what folding them into the params dict would
    #: have invited.
    showBaselineToggled = QtCore.Signal(bool)
    subtractBaselineToggled = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Event reference · open-pore baseline", parent)
        self.setToolTip(
            "How the recording-specific open-pore current I0(t) is estimated. "
            "Every event's depth is measured against it. Keep the existing Automatic "
            "estimator, supply a known signed current, or select trustworthy open-pore "
            "intervals. User-provided references need review; they are not independently verified.")

        self._fs: float | None = None
        #: Which selection the form currently describes; see `set_recording`.
        self._target: Any = None
        self._proposal: BaselineProposal | None = None
        self._health: Any = None
        self._health_scope: str = "detection"
        self._source: str = "default"
        self._busy = False
        self._gate_ok = False
        self._gate_reason = "Select a current channel to measure its baseline."
        self._loading = False
        #: ``(name, label)`` of every check that CONDEMNS the proposal currently on
        #: offer -- empty unless a proposal and a ``measured_for='proposal'`` health
        #: block are both on screen. See `_recompute_condemnation`.
        self._condemned: tuple[tuple[str, str], ...] = ()
        #: The user has been shown the condemnation and the NEXT click accepts it.
        self._override_armed = False
        #: The labels of the checks that condemned the windows now IN THE FORM,
        #: because the user accepted them anyway. Drives the standing mark in
        #: `source_label`; cleared the moment the values stop being those values.
        self._accepted_condemned: tuple[str, ...] = ()
        #: ``(rough_s, base_s, clamped)`` the detection ON SCREEN actually ran at,
        #: read from its provenance by `set_run_windows`; ``None`` when no detection
        #: is shown. Kept as STATE, not baked into a sentence, because the sentence
        #: contrasts it with the form and the form moves -- see `_refresh_run_windows`.
        self._run_windows: tuple[float, float, bool] | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        self.reference_editor = _OpenPoreReferenceEditor(self)
        root.addWidget(self.reference_editor)
        self.automatic_controls = QtWidgets.QWidget()
        automatic_layout = QtWidgets.QVBoxLayout(self.automatic_controls)
        automatic_layout.setContentsMargins(0, 0, 0, 0)
        automatic_layout.setSpacing(6)
        root.addWidget(self.automatic_controls)

        # -- the two windows ------------------------------------------------
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        form.addWidget(QtWidgets.QLabel("Rough window"), 0, 0)
        self.rough_s = QtWidgets.QDoubleSpinBox()
        self.rough_s.setRange(*_ROUGH_RANGE)
        self.rough_s.setDecimals(3)
        self.rough_s.setSingleStep(0.1)
        self.rough_s.setSuffix(" s")
        self.rough_s.setValue(DEFAULT_BASELINE.rough_s)
        self.rough_s.setToolTip(
            "Pass 1. Decides WHICH excursions count as events. It must be several "
            "times the longest state you want treated as an event -- a rough window "
            "shorter than the state it is sitting in becomes that state, and nothing "
            "gets masked. (Measured on the -600 mV apo file: its 521 ms partial clog "
            "is 15% masked at 0.5 s and 99.96% masked at 2.0 s.)")
        form.addWidget(self.rough_s, 0, 1)

        form.addWidget(QtWidgets.QLabel("Base window"), 1, 0)
        self.base_s = QtWidgets.QDoubleSpinBox()
        self.base_s.setRange(*_BASE_RANGE)
        self.base_s.setDecimals(4)
        self.base_s.setSingleStep(0.05)
        self.base_s.setSuffix(" s")
        self.base_s.setValue(DEFAULT_BASELINE.base_s)
        self.base_s.setToolTip(
            "Pass 2. Decides how fast the baseline may follow the trace. It must be "
            "SHORTER than the shortest genuine conductance level dwell, or a gating "
            "pore's steps get smoothed into the baseline instead of being measured "
            "against it.")
        form.addWidget(self.base_s, 1, 1)

        form.addWidget(QtWidgets.QLabel("Decimation"), 2, 0)
        self.decim_label = QtWidgets.QLabel("auto")
        self.decim_label.setToolTip(
            "Derived from the base window, never set by hand: the estimator's cost "
            "goes as 1/decimation squared, so it is the throttle rather than a "
            "quality knob. Shown here so the cost is visible.")
        form.addWidget(self.decim_label, 2, 1)
        form.setColumnStretch(2, 1)
        automatic_layout.addLayout(form)

        # -- where these numbers came from ----------------------------------
        self.source_label = QtWidgets.QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color:#8a939c;")
        self.source_label.setToolTip(
            "Where the values above came from. This travels with the analysis, so a "
            "stored result says whether anyone ever looked at this recording.")
        automatic_layout.addWidget(self.source_label)

        # -- actions: one primary row and two secondary buttons --------------
        row = QtWidgets.QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.probe_button = QtWidgets.QPushButton("Measure this recording")
        self.probe_button.setToolTip(
            "Look at this recording and propose a rough and base window from its own "
            "timescales, then run the health checks on what it proposes. Usually a "
            "few seconds -- measured 2 s on a 3 M-sample recording and 6 s on 9 M "
            "samples (the button says Measuring... meanwhile; it has not missed the "
            "click). Nothing is applied until you press Use these values.")
        self.apply_button = QtWidgets.QPushButton("Use these values")
        self.apply_button.setToolTip(
            "Accept the proposed windows. This is the confirmation step -- the app "
            "never applies a measured parameter on its own. If the health block "
            "below condemns what was proposed, the first press explains what is "
            "wrong instead of applying it, and the second press accepts it anyway "
            "-- with the panel saying so from then on.")
        self.reset_button = QtWidgets.QPushButton("Back to default")
        self.reset_button.setToolTip(
            f"Return to the app default ({DEFAULT_BASELINE.rough_s:g} s / "
            f"{DEFAULT_BASELINE.base_s:g} s). A defensible floor for an unseen "
            f"recording, and provably wrong for some -- measure instead where you can.")
        row.addWidget(self.probe_button, 0, 0, 1, 2)
        row.addWidget(self.apply_button, 1, 0)
        row.addWidget(self.reset_button, 1, 1)
        row.setColumnStretch(0, 1)
        row.setColumnStretch(1, 1)
        automatic_layout.addLayout(row)
        self.display_group = QtWidgets.QGroupBox('Trace display')
        display_layout = QtWidgets.QVBoxLayout(self.display_group)
        display_layout.setContentsMargins(8, 6, 8, 8)
        display_layout.setSpacing(4)
        root.addWidget(self.display_group)

        # -- SEEING IT (owner: "processing should be visible rather than occurring
        #    entirely in the background") ------------------------------------------
        # Until these two boxes existed, this panel was pure text: a user chose the two
        # windows that decide every depth_pA in the table, pressed Use these values, and
        # never once saw what those windows produced on his own recording. The estimate
        # was a number in a sentence.
        #
        # BOTH BOXES ARE ALWAYS HERE AND ALWAYS ENABLED, and neither ever hides. A
        # control that appears with state shifts its neighbours under the pointer, which
        # is the failure shape three rounds of this project were lost to; and a DISABLED
        # checkbox never receives the click, so it can never answer "why not?" -- the
        # same reasoning that keeps the three buttons above clickable through every gate
        # (`set_gate`). A tick with nothing selected posts the gate's reason and STAYS
        # TICKED: it is a standing preference ("show me the baseline"), so it takes
        # effect on its own as soon as a channel can host it.
        #
        # Stacked, not side by side. Measured in the real window at 1512x900: the
        # Inspector dock's scroll viewport is ~420 px, so a row holding both labels
        # (~172 + ~196 px plus a gap) is already at the edge before a checkbox indicator
        # is counted -- and this panel has watched three named buttons vanish off the
        # right of that same viewport.
        self.show_track = QtWidgets.QCheckBox("Show this baseline on the trace")
        self.show_track.setToolTip(
            "Draw the estimated open-pore level I0(t) over the recording, so you can "
            "see the line every depth is measured down from -- rather than trusting "
            "two numbers in a form. It is the SAME estimate the detection uses (the "
            "windows above, at the detection's polarity), not a redrawing of it. "
            "Measured cost on the 15 s -1000 mV apo recording: 70 ms, which is why "
            "this is a checkbox and not a button.")
        display_layout.addWidget(self.show_track)

        self.subtract_view = QtWidgets.QCheckBox(
            "Subtract baseline (display only)")
        self.subtract_view.setToolTip(
            "Redraw the trace as signal minus baseline, centred on zero, with the "
            "y axis relabelled 'Δ vs baseline' for as long as it is on. Two things "
            "this buys: the drift is gone, so shallow events stop being hidden by it; "
            "and two recordings at different open-pore currents (-566 pA and "
            "-1000 pA, say) can finally be read on ONE y axis. It is a DRAWING, not a "
            "measurement -- no depth, no column and no export changes, because nothing "
            "is measured from a curve.")
        display_layout.addWidget(self.subtract_view)

        # THE TWO THINGS CALLED "BASELINE", ten pixels apart, told apart on screen at
        # last. This group's estimator sets every depth_pA; "Signal filters ▸ Baseline
        # drift" further down the same tab is a DETREND stage that flattens the trace
        # the detector sees and touches no column, and their parameter keys nearly
        # collide (baseline_rough_s/baseline_base_s vs detrend_rough_s/detrend_base_s).
        # A user who ticks a box to check "is my baseline right?" must be told which
        # baseline he is looking at, and the answer belongs beside the boxes rather than
        # in a tooltip nobody hovers. Static: it is a caption, not a state.
        self.which_baseline_label = QtWidgets.QLabel(
            "This open-pore reference is used to measure event depth.")
        self.which_baseline_label.setWordWrap(True)
        self.which_baseline_label.setStyleSheet("color:#8a939c;")
        self.which_baseline_label.setToolTip(
            "Two different measurements share the word 'baseline' on this tab. THIS "
            "one is the open-pore current I0(t) every event's depth_pA is measured "
            "against. The 'Baseline drift' stage in the Signal filters group below is "
            "a detrend: it flattens slow drift in the trace the DETECTOR sees, and "
            "never touches a measured column. They are configured independently and "
            "can disagree.")
        #: The FIRST real blocker, posted on a click that a gate refuses. Never a
        #: downstream one -- see the module docstring.
        #:
        #: IT OWNS ITS ROW WHETHER IT SPEAKS OR NOT. This was a bare word-wrapping
        #: QLabel that appeared on a refusal, and everything below it moved: measured
        #: at 1512x900 on the owner's own recording, posting a refusal here pushed the
        #: Signal-filters stage selector -- the control a user reaches for FIRST on this
        #: tab -- down by 51 px. That is the failure shape this campaign has now lost
        #: four separate rounds to: a disclosure that displaces the thing it is about.
        #: The sibling banner in ``panels.DetectionPanel`` was given a reserved row for
        #: exactly this reason; this one had been missed because it lives in a different
        #: file from the controls it was moving.
        #:
        #: The host carries the reservation so the label keeps its own ``setVisible``:
        #: ``isHidden()`` is how the analysis panel and several tests ask "is the gate
        #: up?", and an always-visible empty label would answer that wrongly.
        self.gate_label = QtWidgets.QLabel()
        self.gate_label.setWordWrap(True)
        self.gate_label.setStyleSheet("color:#ffb454;")
        self.gate_label.setVisible(False)
        # REUSE THE CAPTION'S TWO-LINE ROW. Giving the refusal a second reserved row
        # pushed the Signal-filters stage selector 44 px below the fold at 1280x800,
        # even while the refusal was empty. The static caption and a refusal are never
        # both useful at once, so they share one fixed carrier: the caption is the
        # standing text; a refusal temporarily replaces it. The carrier keeps every
        # neighbour still while either child is shown.
        self._gate_host = QtWidgets.QWidget()
        _gate_box = QtWidgets.QGridLayout(self._gate_host)
        _gate_box.setContentsMargins(0, 0, 0, 0)
        _gate_box.setSpacing(0)
        _gate_box.addWidget(self.which_baseline_label, 0, 0)
        _gate_box.addWidget(self.gate_label, 0, 0)
        # Two lines of this label's own metrics: the refusals this gate posts are the
        # channel-kind sentences, which wrap to two at the widths this panel is used at.
        self._gate_host.setFixedHeight(2 * self.gate_label.fontMetrics().lineSpacing())
        root.addWidget(self._gate_host)

        # -- what the detection on screen ACTUALLY ran at, when that is not the form --
        # The form shows the windows that were ASKED for; on a recording shorter than
        # the tracking window the estimator runs them CLAMPED (0.08 / 0.016 s on a
        # 0.08 s cut of the owner's -600 mV file, asked 0.5 / 0.1 s) and until sweep 7
        # nothing on this panel said so -- it kept showing 0.5 / 0.1 (S7-DET3). Filled
        # from the detection's own provenance by `set_run_windows`, rendered against
        # the live form by `_refresh_run_windows`; hidden when the two agree.
        #
        # UNDER THE BUTTON ROW, LIKE `gate_label` -- NOT under the source line, where
        # it first sat. The line follows the form, and two of the gestures that move
        # the form are clicks on this row: "Use these values" over a detection run at
        # other windows makes it appear, "Back to default" makes it vanish. Above the
        # row it moved the button just pressed. Measured in the real window at the
        # dock's 460 px: Apply moved the row 163 px with the line above it and 142 px
        # without it (the 142 is the staleness banner appearing ABOVE the panel, not
        # this widget's to fix); the 21 px difference is the line. Nothing may move a
        # widget under the pointer, and this panel already has the pattern for text a
        # click makes appear -- the gate's refusal goes under the row, so does this.
        self.run_windows_label = QtWidgets.QLabel()
        self.run_windows_label.setWordWrap(True)
        self.run_windows_label.setStyleSheet("color:#ffb454;")
        self.run_windows_label.setVisible(False)
        root.addWidget(self.run_windows_label)

        # -- the proposal ----------------------------------------------------
        self.proposal_box = QtWidgets.QGroupBox("Proposed for this recording")
        pbl = QtWidgets.QVBoxLayout(self.proposal_box)
        pbl.setContentsMargins(8, 6, 8, 6)
        pbl.setSpacing(4)
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setTextFormat(QtCore.Qt.RichText)
        self.summary_label.setWordWrap(True)
        pbl.addWidget(self.summary_label)
        self.reason_label = QtWidgets.QLabel()
        # PLAIN, DECLARED. A QLabel left on ``AutoText`` guesses, and the reason line
        # is measured prose that now carries comparisons and percentages -- exactly
        # the content a rich-text guess would start editing. Declaring it also means
        # ``reason_label.text()`` IS what the user reads, which is what
        # :meth:`proposal_text` relies on.
        self.reason_label.setTextFormat(QtCore.Qt.PlainText)
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("color:#c3c9d1;")
        pbl.addWidget(self.reason_label)
        self.proposal_box.setVisible(False)
        root.addWidget(self.proposal_box)

        # -- the health block -------------------------------------------------
        self.health_box = QtWidgets.QGroupBox("Baseline health")
        self.health_box.setToolTip(
            "Checks the reference implementation does not perform. They catch the "
            "regimes where this estimator fails SILENTLY -- a mask that found nothing, "
            "a mask that swallowed the recording, drift faster than the rough window, "
            "and a trace too short for the two-pass structure.")
        hbl = QtWidgets.QVBoxLayout(self.health_box)
        hbl.setContentsMargins(8, 6, 8, 6)
        hbl.setSpacing(2)
        # WHAT THESE CHECKS DESCRIBE. The block used to appear only after a detection,
        # populated from that result's provenance, so a block on screen always described
        # the run that had just happened -- and there was nothing to say so. Now it can
        # also describe the PROPOSAL you have not confirmed yet, and those two are not
        # interchangeable: one is the verdict on windows already used to measure a table
        # of depths, the other is the verdict on windows you are about to accept.
        self.health_scope_label = QtWidgets.QLabel()
        self.health_scope_label.setWordWrap(True)
        self.health_scope_label.setStyleSheet("color:#8a939c;")
        hbl.addWidget(self.health_scope_label)
        self.health_label = QtWidgets.QLabel()
        self.health_label.setTextFormat(QtCore.Qt.RichText)
        self.health_label.setWordWrap(True)
        hbl.addWidget(self.health_label)
        self.health_box.setVisible(False)
        root.addWidget(self.health_box)

        # -- wiring (own children only) --------------------------------------
        self.rough_s.valueChanged.connect(self._on_edited)
        self.base_s.valueChanged.connect(self._on_edited)
        self.show_track.toggled.connect(self._on_view_toggled)
        self.subtract_view.toggled.connect(self._on_view_toggled)
        self.probe_button.clicked.connect(self._on_probe_clicked)
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.reference_editor.changed.connect(self._on_reference_edited)
        self.reference_editor.selectionRequested.connect(self.addReferenceSelectionRequested)
        self.reference_editor.preview_button.clicked.connect(self._on_reference_preview_clicked)

        self._recouple()      # the two windows constrain each other from the start
        self._refresh_source()
        self._refresh_decim()
        self._restyle_buttons()
        self._refresh_reference_mode()

    # ==================================================================
    # values
    # ==================================================================
    def config(self) -> BaselineConfig:
        """The estimator the form currently describes.

        Built by ``replace`` off `DEFAULT_BASELINE`, touching only the two fields
        this form owns -- a widget cannot invent a ``mask_sigma``, and
        `DEFAULT_BASELINE` stays the app's single baseline-parameter literal.
        """
        return DEFAULT_BASELINE.replace(
            rough_s=float(self.rough_s.value()),
            base_s=float(self.base_s.value()),
            decim=None,
        )

    def params(self) -> dict[str, Any]:
        """The ``baseline_*`` params to run with, plus ``baseline_param_source``.

        Merged into the detection form's params by the owning AnalysisPanel, read
        back by `nanoscope.analysis.baseline.baseline_config_from_params`, and
        stamped into provenance -- so a stored result records not just WHICH windows
        measured it but whether a human ever chose them.

        Automatic keeps `BASELINE_PARAM_KEYS` plus the source, deliberately NOT the whole
        ``to_provenance`` block: that also carries ``baseline_fingerprint``, which is
        DERIVED from the parameters (and from ``fs``, which this form does not
        resolve). A derived value in a params dict is a value that can disagree with
        the parameters beside it, and the estimator stamps the real one at run time.
        Opt-in modes additionally carry the versioned ``baseline_open_pore_reference``
        recipe. Invalid drafts remain invalid; the owner must check ``reference_error``.
        """
        cfg = self.config()
        prov = cfg.to_provenance()
        out = {k: prov[k] for k in BASELINE_PARAM_KEYS if k in prov}
        out["baseline_param_source"] = self.param_source()
        out.update(self.reference_editor.params())
        return out

    def reference_mode(self) -> str:
        """The per-recording reference mode; Automatic preserves the old contract."""
        return self.reference_editor.reference_mode()

    def reference_error(self) -> str:
        """A visible draft error that must block detection, never trigger fallback."""
        return self.reference_editor.error()

    def add_reference_interval(self, start_s: float, end_s: float) -> bool:
        """Add a recording-relative trace selection deliberately, without analysing."""
        return self.reference_editor.add_interval(start_s, end_s)

    def _on_reference_edited(self) -> None:
        self._refresh_reference_mode()
        self.paramsChanged.emit(self.params())

    def _on_reference_preview_clicked(self) -> None:
        if not self._gate_ok:
            self._post(self._gate_reason)
        elif self._busy:
            self._post("Still measuring this recording — one moment.")
        elif self.reference_mode() == "automatic":
            self._post("Choose a user-provided reference to preview it here.")
        elif self.reference_error():
            self._post(self.reference_error())
        else:
            self._post("")
            self.referencePreviewRequested.emit()

    def _refresh_reference_mode(self) -> None:
        automatic = self.reference_mode() == "automatic"
        self.automatic_controls.setVisible(automatic)
        self.proposal_box.setVisible(automatic and self._proposal is not None)
        if not automatic:
            self.run_windows_label.hide()
        else:
            self._refresh_run_windows()
        self._restyle_buttons()

    def _set_windows(self, rough: float | None, base: float | None) -> None:
        """Set either window without the form ever passing through an invalid state.

        ``base_s > rough_s`` is REFUSED by `BaselineConfig` -- a short pass at least as
        wide as the rough one masks nothing and the two-pass structure is inoperative
        -- so a form that can reach it raises inside a Qt slot, where the exception has
        nowhere to go. ``base_s``'s maximum is therefore pinned to ``rough_s``, exactly
        as this panel's sibling pins ``k_out``'s maximum to just under ``k_in``.

        Ordering is load-bearing for the same reason it is there: applying the values
        under the OLD cap silently truncates one of them, so a proposal of 2.0 / 0.5
        landing on a form holding 0.03 / 0.01 would clamp the new base window down to
        0.03. The cap is opened first, both values land verbatim, and it is re-applied
        around the result.
        """
        self._loading = True
        try:
            self.base_s.setMaximum(_BASE_RANGE[1])
            if rough is not None:
                self.rough_s.setValue(float(rough))
            if base is not None:
                self.base_s.setValue(float(base))
            self._recouple()
        finally:
            self._loading = False
        # The "actually ran at ... not the X above" line quotes the form; the form
        # just moved (a seed, a confirmation, a reset), and `_on_edited` is muted
        # while it does, so the line is re-derived here.
        self._refresh_run_windows()

    def _recouple(self) -> None:
        """Cap ``base_s`` at ``rough_s``. ONE direction, deliberately.

        The constraint is ``base_s <= rough_s``, and it can be enforced from either
        end -- but flooring ``rough_s`` at ``base_s`` as well makes the form
        un-navigable: moving from 0.5 / 0.1 to a gating pore's 0.03 / 0.01 would
        require lowering the base window FIRST, and a user who reaches for the rough
        spin simply finds it will not go down. Capping only the base window means
        lowering the rough one carries the base one with it, which is the resolution
        this form wants anyway.
        """
        self.base_s.setMaximum(min(_BASE_RANGE[1], float(self.rough_s.value())))

    def set_params(self, params: Mapping[str, Any] | None) -> None:
        """SEED the form from a stored detection's params (:meth:`params`' inverse).

        A key that is ABSENT is left alone, as everywhere else in this app's forms:
        a partial dict from an older provenance is a partial update, not a reset to
        defaults over a table that was produced by other numbers.

        A SEED IS NOT A SET, AND IT MAY NOT OVERWRITE A CONFIRMATION
        ------------------------------------------------------------
        Seeding exists so the form shows the settings that produced the table on
        screen; a stored value is the best thing to say when the user has said
        nothing. The moment he has, it is the worst: §0.3 of this change is "the app
        PROPOSES, the user CONFIRMS", and a single global default provably cannot
        exist -- the owner's apo file needs ``rough_s = 2.0`` for its 521 ms partial
        clog while ``ferritin_data3`` needs 0.03.

        It did overwrite it. ``AnalysisPanel._seed_form_from`` runs on every
        ``set_selection``, so ONE click on the channel that was already selected put
        2.0 / 0.5 back to 0.5 / 0.1 with nothing on screen saying so -- and because
        this method does not emit ``paramsChanged``, the staleness check never re-ran
        on the edit, so the ``_show_result`` that followed compared the stored
        fingerprint against the now-reverted form, found them EQUAL, and hid the
        banner too. Detect and Run-ML then both ran the estimator the user had
        replaced.

        So a confirmation wins here, and the reset that ends it is
        :meth:`set_recording` seeing a different recording -- which is where
        "parameters are per-recording" already lives.
        """
        if not params:
            return
        self.reference_editor.set_params(params)
        self._refresh_reference_mode()
        if self._source in _CONFIRMED_SOURCES:
            return
        self._set_windows(params.get("baseline_rough_s"), params.get("baseline_base_s"))
        # Seeded values are not the values the mark was about (a FALLBACK proposal
        # stamps ``'default'``, which is not a confirmed source, so this path can be
        # reached with a mark standing).
        self._accepted_condemned = ()
        src = str(params.get("baseline_param_source", self._source))
        self._source = src if src in PARAM_SOURCES else "user"
        self._refresh_source()
        self._refresh_decim()

    def param_source(self) -> str:
        """How the values in the form were chosen (one of `PARAM_SOURCES`)."""
        if self.reference_mode() != "automatic":
            return "user" if self.reference_editor._confirmed else "restored"
        return self._source

    def batch_params(self) -> dict[str, Any]:
        """:meth:`params` plus the recording the form's source is ABOUT -- for a run
        that will apply this ONE form to MANY recordings.

        Batch detect used to send :meth:`params` as is, so ``'proposed'`` -- "measured
        on this recording, and confirmed" -- was stamped on every recording in the
        batch, and the never-measured advisory (gated on ``'default'``) went silent on
        all of them (S7-DET1). The extra key, `baseline.BATCH_SCOPE_KEY`, names the
        ``(doc_id, channel)`` these windows were proposed / confirmed / restored FOR
        (the panel's own per-recording memory -- the target `set_recording` resets the
        form on), and `baseline.resolve_param_source` stamps every OTHER recording
        ``'batch'``.

        NO SCOPE IS CLAIMED UNDER `_UNSCOPED_SOURCES`. At ``'default'`` nobody
        measured anything, and the honest source for every recording is the one
        already there. Under ``'batch'`` the form was seeded from a result that says
        its windows were set for a DIFFERENT recording -- which one, the stored
        result does not say -- so the panel's own target is precisely the
        recording these windows are NOT about. The first version of this method
        claimed it anyway, and a second Batch detect launched from the never-measured
        recording then stamped the recording the user HAD measured and confirmed
        ``'batch'``: on the owner's -600 mV file (probed, 2.0 s / 0.5 s accepted) the
        provenance, the Stats advisory, this panel's source line and the exported
        ``.meta.json`` all said "this recording was never measured" after a re-batch
        from the -1000 mV form (sweep-7 fix-batch E1). With no scope the resolver
        stamps ``'batch'`` on every run, which is the truth about the form and the
        most this panel can say.

        (What it cannot say -- that a target whose STORED result carries a confirmed
        stamp at these same windows keeps that stamp -- is the batch runner's to say,
        from the stored result; this panel never sees other recordings' results.)
        """
        out = self.params()
        target = self._target
        # A real selection is ``(doc_id, channel >= 0)``; the owning panel's
        # nothing-selected sentinel carries channel -1 and is not a recording.
        real = (isinstance(target, tuple) and len(target) == 2
                and isinstance(target[1], int) and target[1] >= 0)
        if self._source not in _UNSCOPED_SOURCES and real:
            out[BATCH_SCOPE_KEY] = target
        return out

    def set_run_windows(self, prov: Mapping[str, Any] | None) -> None:
        """Record what the detection on screen ACTUALLY ran at, from its provenance.

        Read from the result's own ``baseline_*`` provenance -- the truth about the
        run -- and rendered by `_refresh_run_windows` whenever it disagrees with the
        form: a clamped run (``baseline_fallback == FALLBACK_CLAMPED``) names the
        widths it used and why, beside the widths the form asks for. ``None``, or a
        provenance without both windows, clears the record and hides the line.

        This method stores; it does not compose the sentence. The first version did
        both, reading the form ONCE at push time and baking ``not the 0.5 s / 0.1 s
        above`` into the text -- and the form is a pair of live spins. Measured on a
        4000-sample cut of the owner's -600 mV file: after Detect the line read "ran
        at rough 80 ms / base 16 ms, not the 0.5 s / 0.1 s above"; Rough window
        turned to 1.0 s and the spin read 1.000 s while the line two rows under it
        went on saying 0.5 s (sweep-7 fix-batch E2 / X-SURF-4). A sentence that
        points at the widget above it has to be re-derived when that widget moves.
        """
        self._run_windows = None
        if prov:
            try:
                rough = float(prov.get("baseline_rough_s"))
                base = float(prov.get("baseline_base_s"))
            except (TypeError, ValueError):
                pass
            else:
                clamped = str(prov.get("baseline_fallback") or "") == FALLBACK_CLAMPED
                self._run_windows = (rough, base, clamped)
        self._refresh_run_windows()

    @staticmethod
    def _windows_text(rough: float, base: float) -> str:
        """``"0.08 s / 0.016 s"`` -- ONE unit for both halves of the run-windows line.

        Seconds, because that is the unit the two spins above the line carry (their
        suffix is `` s``) and the line exists to be read against them. The first
        version printed the run in ms and the form in s ("80 ms ... 0.5 s"), which is
        two units for one quantity on one line.
        """
        return f"{rough:g} s / {base:g} s"

    def _refresh_run_windows(self) -> None:
        """Render `_run_windows` against the form AS IT IS NOW. The single writer.

        Called from every path that moves the form -- a hand edit (`_on_edited`), a
        seed / confirmation / reset (`_set_windows`) -- and from `set_run_windows`
        itself, so the "above" half of the sentence can never quote windows the form
        no longer shows.

        Hidden when nothing is shown, or when the run agrees with the form and was
        not clamped. A clamped run stays on screen even when the form matches it
        exactly (the user typed the clamped widths in): the depths in the table are
        still not comparable with ones measured at the configured windows, and the
        contrast clause alone is dropped, because "ran at 0.08 s, not the 0.08 s
        above" is not a sentence.
        """
        run = self._run_windows
        if run is None:
            self.run_windows_label.setVisible(False)
            self.run_windows_label.clear()
            return
        rough, base, clamped = run
        asked_r, asked_b = float(self.rough_s.value()), float(self.base_s.value())
        same = abs(rough - asked_r) < 1e-9 and abs(base - asked_b) < 1e-9
        if same and not clamped:
            self.run_windows_label.setVisible(False)
            self.run_windows_label.clear()
            return
        contrast = "" if same else f", not the {self._windows_text(asked_r, asked_b)} above"
        why = (" -- CLAMPED to this recording's length, which is shorter than the "
               "tracking window; depths are not comparable with ones measured at the "
               "configured windows" if clamped else "")
        self.run_windows_label.setText(
            f"The detection on screen actually ran at rough {rough:g} s / base "
            f"{base:g} s{contrast}{why}.")
        self.run_windows_label.setVisible(True)

    def set_recording(self, fs: float | None, target: Any = None,
                      duration_s: float | None = None) -> None:
        """The selected recording's sample rate, so the derived decimation is real.

        Clears any proposal: a proposal belongs to the recording it was measured on,
        and one carried across a selection change is the same class of defect as a
        surrogate study's answer landing on the wrong file.

        ``target`` is an opaque, hashable identity for the selection -- the owning
        panel passes ``(doc_id, channel)``. It is the OTHER half of the confirmation
        rule in :meth:`set_params`: a value the user confirmed is protected from
        being seeded over, and this is what ends that protection. "Parameters are
        per-recording" is the first thing the spec says about them, so 2.0 / 0.5
        chosen for a 521 ms clog must not silently become the estimator for the next
        file opened -- but re-selecting the SAME channel is navigation, not a new
        recording, and must cost nothing.

        THE RESET IS FOR EVERY SOURCE, NOT ONLY A CONFIRMED ONE. Whatever the form
        holds, it holds FOR the recording just left: typed or confirmed there, or
        seeded out of that recording's own stored run. The first version reset only
        ``'user'`` / ``'proposed'``, reasoning that anything else would be seeded over
        by the next recording's stored run -- which is true only when the next
        recording HAS one. Measured (sweep-7 fix-batch E6): measure + confirm on the
        owner's -600 mV file, Batch detect, click the -1000 mV file (form seeds
        2.0 / 0.5 ``'batch'`` from its result), then open the -700 mV file, which has
        no result and was never batch-run -- the form still read 2.0 / 0.5 "COPIED
        from the batch form", and Detect there measured 1,172 events at those windows
        and stamped them ``'batch'``, though no batch ever ran on it. A
        ``'restored'`` form carried the same way. So a target change puts the form at
        the app default, and the new recording's own stored run seeds it (the owning
        panel calls :meth:`set_params` right after this) -- or nothing does, and the
        default is the honest answer for a recording nobody has looked at.

        ``target=None`` means "do not judge", so every existing caller and every test
        that only sets a rate keeps its behaviour exactly.
        """
        self._fs = float(fs) if fs else None
        if target is not None and target != self._target:
            self._target = target
            self.reference_editor.reset()
            self._set_windows(DEFAULT_BASELINE.rough_s, DEFAULT_BASELINE.base_s)
            self._source = "default"
            # A mark left standing over ANOTHER recording's numbers is the
            # surrogate-answer-on-the-wrong-file defect this method already exists to
            # prevent, one field along.
            self._accepted_condemned = ()
            self._refresh_source()
            # ...and the "actually ran at" line, which described the other recording's
            # detection; the owning panel re-pushes it when this one's result is shown.
            self.set_run_windows(None)
        self.set_proposal(None)
        self.set_health(None)
        self.reference_editor.set_recording(self._fs, duration_s)
        self._refresh_reference_mode()
        self._refresh_decim()

    # ==================================================================
    # the proposal
    # ==================================================================
    def set_proposal(self, proposal: BaselineProposal | None) -> None:
        """Show a finished probe's proposal. Applies NOTHING -- that is the click."""
        self._proposal = proposal
        # A new proposal is a new verdict-to-be. Whatever the last one was condemned
        # for, and whatever the user was told about it, does not carry over -- the
        # health block for THIS one arrives immediately after (`set_health`), and
        # until it does there is nothing certified and nothing condemned.
        self._condemned = ()
        self._override_armed = False
        self.set_busy(False)
        if proposal is None:
            self.proposal_box.setVisible(False)
            self.summary_label.clear()
            self.reason_label.clear()
            self._restyle_buttons()
            return
        rows = proposal_summary(proposal, self._fs)
        # ESCAPED, for the reason :meth:`_check_html` spells out at length: this label
        # is RICH TEXT, and Qt's parser silently swallows a "<" and everything up to
        # the next ">" -- there it deleted a whole criterion on its way to the screen
        # while the HTML source still contained it. These rows now carry measured
        # values (the intrusion ranking), which is precisely the kind of content that
        # grows a comparison operator one day; interpolating it raw would be the same
        # defect one label along.
        self.summary_label.setText(
            "<table cellpadding='1'>"
            + "".join(f"<tr><td style='color:#8a939c;'>{html.escape(str(k))}"
                      f"&nbsp;&nbsp;</td>"
                      f"<td><b>{html.escape(str(v))}</b></td></tr>" for k, v in rows)
            + "</table>")
        # The fallback says so IN THE PROPOSAL TEXT, per the spec: a proposal the
        # probe could not make must not look like one it did.
        prefix = "" if proposal.confident else "Could not measure -- "
        self.reason_label.setText(prefix + proposal.reason)
        self.reason_label.setStyleSheet(
            "color:#c3c9d1;" if proposal.confident else "color:#ffb454;")
        self.proposal_box.setVisible(self.reference_mode() == "automatic")
        self._restyle_buttons()

    def proposal(self) -> BaselineProposal | None:
        """The proposal currently on offer, or None."""
        return self._proposal

    def proposal_text(self) -> str:
        """The proposal block AS RENDERED — the characters that reach the screen.

        The summary table and the reason line, in the order they are drawn, both taken
        through a :class:`QtGui.QTextDocument` for the reason :meth:`health_text` is:
        the summary is rich text, Qt's parser edits it on the way to the pixel, and an
        assertion made on ``summary_label.text()`` is an assertion about the markup
        rather than about what the user read. The intrusion ranking is rendered here,
        so this is the surface any test of it has to make its claim on.

        Empty string when no proposal is on offer.
        """
        if self._proposal is None:
            return ""
        doc = QtGui.QTextDocument()
        doc.setHtml(self.summary_label.text())
        summary = doc.toPlainText()
        return f"{summary}\n{self.reason_label.text()}".strip()

    def apply_proposal(self, *, override: bool = False) -> bool:
        """THE CONFIRMATION. Copy the proposed values into the form; True if applied.

        Public so a test can press it without synthesising a mouse event, and so
        MainWindow could offer the same action from a menu without reaching into the
        buttons.

        **Returns False, and applies nothing, on a proposal the health block
        condemns** unless ``override`` (or a previous refused click has already
        armed one). The gate lives HERE rather than only in the click handler for
        exactly the reason this method is public: a menu action, a driver or a
        headless batch reaching past the button must not be the one path on which a
        condemned window is accepted silently. The refusal posts its reason to
        :attr:`gate_label` and arms the override, so the second call -- click or
        not -- goes through.
        """
        if self.reference_mode() != "automatic":
            self._post("Automatic estimator proposals do not apply to a user-provided reference.")
            return False
        p = self._proposal
        if p is None:
            return False
        if self._condemned and not (override or self._override_armed):
            self._post(self._condemnation_notice())
            self._override_armed = True
            self._restyle_buttons()
            return False
        self._set_windows(p.config.rough_s, p.config.base_s)
        self._source = p.param_source if p.param_source in PARAM_SOURCES else "user"
        # THE STANDING MARK. The refusal above is a moment; this is for as long as the
        # form holds these numbers, because the screen that has to say "these windows
        # were condemned" is the screen the user reads the resulting depth on, not the
        # one they clicked through ten minutes earlier.
        self._accepted_condemned = tuple(
            label or name for name, label in self._condemned)
        if self._condemned:
            self._override_armed = True   # accepted: do not re-refuse the same window
        self._refresh_source()
        self._refresh_decim()
        self._restyle_buttons()
        self.paramsChanged.emit(self.params())
        return True

    # ==================================================================
    # health
    # ==================================================================
    @staticmethod
    def _check_rows(health: Any) -> list[dict[str, Any]]:
        """EVERY field of every check, from EITHER form. The single extraction.

        A live run hands over a `BaselineHealth` whose checks are `BaselineCheck`
        objects; a RESTORED one has been through JSON and comes back as the
        ``baseline_health`` dict in provenance, whose checks are mappings. Two
        readers of that pair -- the block on screen (:meth:`_health_checks`) and the
        Apply gate (:meth:`_recompute_condemnation`) -- must not be able to disagree
        about what this recording's checks say, so they project out of one function
        rather than each doing the ``getattr``-or-``Mapping`` dance themselves.

        ``name`` is the machine token (``'base_vs_state'``); the renderer does not
        want it and the gate cannot work without it, which is why this returns the
        whole row and the two callers take what they need.
        """
        checks = getattr(health, "checks", None)
        if checks is None and isinstance(health, Mapping):
            checks = health.get("checks") or ()
        out: list[dict[str, Any]] = []
        for c in checks or ():
            if isinstance(c, Mapping):
                out.append({"name": str(c.get("name", "")),
                            "level": str(c.get("level", "info")),
                            "label": str(c.get("label", "")),
                            "message": str(c.get("message", "")),
                            "value": c.get("value"),
                            "threshold": str(c.get("threshold", ""))})
            else:
                out.append({"name": str(getattr(c, "name", "")),
                            "level": str(c.level), "label": str(c.label),
                            "message": str(c.message), "value": c.value,
                            "threshold": str(c.threshold)})
        return out

    @classmethod
    def _health_checks(cls, health: Any) -> list[tuple[str, str, str, Any, str]]:
        """``[(level, label, message, value, threshold), ...]`` from EITHER form.

        A live run hands over a `BaselineHealth`; a RESTORED one has been through
        JSON and comes back as the ``baseline_health`` dict in provenance. Both are
        the same information and the panel renders both, because a health block that
        vanished when a project was reopened would quietly turn "this baseline is
        suspect" into silence -- and silence is what this whole change is removing.

        ``value`` and ``threshold`` are carried through because they are what makes a
        green line CHECKABLE -- see :meth:`_check_html`. `BaselineCheck` has always
        held both; this widget used to drop them on the floor.

        EXACTLY THE FIVE FIELDS `_check_html` TAKES, POSITIONALLY, AND THAT IS THE
        WHOLE POINT OF THE SHAPE. The Apply gate needs a sixth (the machine ``name``)
        and it does NOT get it by growing this tuple: the rows go to
        ``_check_html(*row)``, so one field inserted here shifts every criterion on
        screen by one silently, and this tuple is read positionally outside this
        module as well. The extra field lives in :meth:`_check_rows`, which both this
        and the gate project out of.
        """
        return [(r["level"], r["label"], r["message"], r["value"], r["threshold"])
                for r in cls._check_rows(health)]

    #: The verdict word for each level. A GLYPH-BEARING WORD, not colour: a health
    #: block reaches other people as a screenshot, and green is the first thing a
    #: colour-blind reader, a greyscale print or a dim projector loses.
    _VERDICT = {"ok": "PASS", "info": "note", "warn": "WARNING", "refuse": "REFUSED"}

    @staticmethod
    def _fmt_value(value: Any) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "" if value is None else str(value)
        return f"{v:.4g}"

    @classmethod
    def _check_html(cls, level: str, label: str, message: str,
                    value: Any, threshold: str) -> str:
        """One check as a row that STATES ITS OWN CRITERION.

        A CHECK THAT DOES NOT SAY WHAT WOULD FAIL IT CANNOT BE READ. The overshoot
        line rendered as a green bullet saying "overshoot 0.000 (241 samples above
        +4 sigma, 1,061,865 below; healthy corpus values 0.09-0.11)" -- a measured
        value an order of magnitude outside the range its own message calls healthy,
        coloured as a pass. Both halves were true: the check is deliberately
        ONE-SIDED (it flags HIGH overshoot only, ``<= 0.3``, because low overshoot is
        what a good window looks like), and the message never said so, so the reader
        was left to reconcile "0.000" against "0.09-0.11" with nothing to go on.

        `BaselineCheck` already carried the answer in ``threshold`` and the panel
        threw it away. Printing ``measured 0.0002, passes <= 0.3`` beside the verdict
        makes the one-sidedness visible on every check at once, rather than rewording
        one message -- and it is the fix that keeps working when a threshold moves.

        EVERY PIECE OF MEASURED TEXT IS ESCAPED, BECAUSE THIS LABEL IS RICH TEXT.
        The first version of this method interpolated ``threshold`` raw, and the
        thresholds that matter most here begin with ``<``: ``<= 0.3`` for overshoot,
        ``<= 2x`` for the wide-window mask comparison. Qt's rich-text parser reads
        ``passes <= 0.3)</span>`` as a malformed tag and drops it together with the
        closing paren and the span, so the rendered line came out

            PASS  overshoot (measured 0.0002269, passes — overshoot 0.000 (241 …

        -- the criterion this whole method exists to show, deleted on its way to the
        screen, on exactly the check the finding was about. The HTML source contained
        it, which is why a test asserting on the source could not see the defect; see
        :meth:`health_text`, which now returns what is RENDERED.
        """
        colour = _LEVEL_STYLE.get(level, "#c3c9d1")
        verdict = cls._VERDICT.get(level, level.upper())
        shown = html.escape(cls._fmt_value(value))
        thr = html.escape(str(threshold or ""))
        lbl = html.escape(str(label or ""))
        msg = html.escape(str(message or ""))
        crit = ""
        if shown and thr:
            crit = f" <span style='color:#8a939c;'>(measured {shown}, passes {thr})</span>"
        elif thr:
            crit = f" <span style='color:#8a939c;'>(passes {thr})</span>"
        return (f"<div style='color:{colour};'>&bull; <b>{verdict}</b> &nbsp;{lbl}"
                f"{crit} &mdash; {msg}</div>")

    #: What a health block on screen is ABOUT. Keyed by the caller's own word so the
    #: two producers cannot describe themselves differently.
    #:
    #: ``{n}`` is filled from the rows actually rendered, and it is a template rather
    #: than a number because the count is DATA-DEPENDENT: the two state-timescale
    #: checks need a wide pass to read, so a block carries 5 to 8 rows depending on
    #: the recording and the window. These strings said "These six checks" while
    #: eight were drawn beneath them — a caption that miscounts its own evidence, in
    #: the one sentence a reader uses to decide how much the block covers.
    _HEALTH_SCOPE = {
        "proposal": ("These {n} checks were run on the windows PROPOSED above, before "
                     "anything was measured with them. This is the evidence for the "
                     "Apply click."),
        "detection": ("These {n} checks describe the windows the detection on screen "
                      "was measured under — read back from its own provenance, not "
                      "from the form above."),
    }

    def set_health(self, health: BaselineHealth | Mapping[str, Any] | None,
                   *, measured_for: str = "detection") -> None:
        """Render a health block where the user will see it, saying what it is about.

        Accepts the live `BaselineHealth` or the ``baseline_health`` dict a stored
        detection carries in its provenance -- see :meth:`_health_checks`.

        ``measured_for`` is ``'proposal'`` (the probe measured these checks for the
        windows it is offering) or ``'detection'`` (they came out of a stored result's
        provenance). It is a required distinction, not decoration: the same green
        bullets mean "you may trust the windows you are about to accept" in one case
        and "the depths already in the table were measured under these" in the other.
        """
        self._health = health
        self._health_scope = measured_for
        rows = self._health_checks(health)
        # ONLY A PROPOSAL-SCOPE BLOCK MOVES THE GATE, and this is where that matters
        # most. A detection's block arrives here on every result shown -- so a panel
        # that recomputed the gate from whatever was last pushed in would have a
        # user's Detect click SILENTLY LIFT the condemnation on a proposal still
        # sitting on offer above it: probe -> two red warnings -> Detect -> Apply,
        # accepted in one press with nothing said. The verdict on a proposal is ended
        # by a new proposal (`set_proposal`), not by a result about other windows.
        #
        # A new proposal-scope verdict DOES disarm a pending override: arming means
        # "you have read THIS condemnation", and carrying it across a re-measure
        # would let one informed click authorise a warning nobody has seen.
        if measured_for == "proposal":
            self._recompute_condemnation()
        if not rows:
            self.health_box.setVisible(False)
            self.health_label.clear()
            self.health_scope_label.clear()
            self._restyle_buttons()
            return
        rows.sort(key=lambda r: _LEVEL_ORDER.index(r[0]) if r[0] in _LEVEL_ORDER else 9)
        self.health_scope_label.setText(
            self._HEALTH_SCOPE.get(measured_for, "").format(n=len(rows)))
        self.health_label.setText("".join(self._check_html(*row) for row in rows))
        self.health_box.setVisible(True)
        self._restyle_buttons()

    def health_scope(self) -> str:
        """``'proposal'`` / ``'detection'`` — what the block on screen describes."""
        return getattr(self, "_health_scope", "detection")

    def health_text(self) -> str:
        """The health block AS RENDERED — the characters that reach the screen.

        This used to return ``health_label.text()``, the rich-text SOURCE, and the
        difference is not academic: Qt silently ate ``<= 0.3`` out of the overshoot
        line on its way to the pixel while the source still contained it, so a guard
        test written against the source passed over a defect a reader could see. Any
        assertion about what the user reads has to be made on this side of the
        rich-text parser. :meth:`health_html` is the source, for anything that really
        wants the markup.
        """
        doc = QtGui.QTextDocument()
        doc.setHtml(self.health_label.text())
        return doc.toPlainText()

    def health_html(self) -> str:
        """The rich-text SOURCE of the health block (see :meth:`health_text`)."""
        return self.health_label.text()

    def health(self) -> Any:
        """The health block last pushed in (either form), or None."""
        return self._health

    # ==================================================================
    # the condemned proposal
    # ==================================================================
    # WHY APPLY IS NOT SIMPLY DISABLED HERE.
    # `BaselineHealth.ok` False means the estimator's own checks say the windows on
    # offer must not be read a depth off. On the owner's -700 mV apo recording that
    # was not theoretical: the probe proposed 0.1 / 0.03 s, `base_vs_state` read 1.38x
    # against a floor of 3.0 and `baseline_intrusion` 0.0416 against 0.03, and
    # accepting it stamped an open pore of -51.661 pA where three independent estimates
    # of that pore agree on -48.218 (7.14% out) -- 97.00 pA of median depth against
    # 101.01, and a dI/I0 of 1.900 against 2.127.
    #
    # THAT PROPOSAL NO LONGER EXISTS. `baseline_probe.base_floor_s` now floors the
    # proposed `base_s` at `baseline.BASE_STATE_RATIO_MIN` times the same median state
    # dwell this check divides by, so the -700 mV file is proposed at 0.5 / 0.1 s with
    # all eight checks green, and no recording in the owner's corpus reaches this gate
    # any more (measured: -500, -600, -700, ferritin_data1/2/3, health.ok True on all
    # six proposals). **Do not delete the gate on that evidence.** It is now a
    # BACKSTOP rather than a workflow step, and there are three ways to reach a
    # condemned config that the floor cannot touch: a window typed into the spins by
    # hand, one restored from a project file or a seeded session, and a proposal
    # condemned by one of the SEVEN other checks, none of which the proposer bounds.
    #
    # It is still not a `setEnabled(False)`, for the reason this module's docstring
    # gives (a disabled button cannot answer "why not?") and for a second one that is
    # specific to this button: **reproducing an old analysis is a legitimate thing to
    # want**. The -700 mV table already in the owner's report was measured at 0.1 /
    # 0.03; a panel that refuses to set those windows makes the app unable to
    # reproduce its own published numbers, and the user's answer to that is to type
    # 0.03 into the spin by hand -- which reaches the same estimator with no warning
    # attached at all. Refusing the click does not remove the risk, it removes the
    # WARNING from the path the user then takes.
    #
    # So: refuse the FIRST click, spell out what is condemned and what it costs, and
    # accept the SECOND -- and then keep saying so on screen for as long as the form
    # holds those values (`_accepted_condemned` -> `_refresh_source`). What a user
    # must never be able to do is accept a condemned window WITHOUT the screen having
    # said so, and that is the invariant the two-step and the standing mark enforce
    # together.
    def _recompute_condemnation(self) -> None:
        """Which checks condemn the proposal on offer. Disarms any pending override.

        Only a ``measured_for='proposal'`` block can condemn, and only such a block
        calls this: the two scopes are not interchangeable (see :meth:`set_health`).
        A ``'detection'`` block is the verdict on windows that have ALREADY measured
        the table on screen -- it is evidence about that table, and gating a future
        click on it would refuse the Apply that repairs it, while recomputing FROM it
        would quietly clear a condemnation that still stands.
        """
        judging = self._proposal is not None and self._health_scope == "proposal"
        self._condemned = tuple(
            (r["name"], r["label"]) for r in self._check_rows(self._health)
            if r["level"] != _CERTIFYING_LEVEL
        ) if judging else ()
        self._override_armed = False

    def condemned_checks(self) -> tuple[str, ...]:
        """Names of the checks condemning the proposal on offer (``()`` if none).

        Public so a driver or a test can ask the panel the same question the Apply
        gate asks, rather than re-deriving it from a health object and drifting.
        """
        return tuple(name for name, _label in self._condemned)

    def accepted_condemned(self) -> tuple[str, ...]:
        """Labels of the checks the values IN THE FORM were accepted over."""
        return self._accepted_condemned

    def _condemnation_notice(self) -> str:
        """The refusal the first Apply click on a condemned proposal posts.

        It does NOT say "re-run the probe", and that survives
        `baseline_probe.base_floor_s` teaching the proposer the ratio. This notice is
        posted only over a proposal ON OFFER, and the proposer is deterministic: the
        probe that produced the condemned windows will produce them again. What the
        floor changed is which windows can be on offer at all -- ``base_vs_state`` can
        no longer be the check that condemns a proposal, so anything reaching this
        text is condemned by one of the other seven, which the proposer bounds not at
        all. Naming an action that provably cannot work is worse than naming none: it
        spends the one instruction the user will follow. The two actions offered here
        are the two that exist.

        (``base_vs_state``'s own remedy, which is rendered in the health block below
        this notice, DOES now name the probe -- because that message is also shown
        over windows that were typed or restored rather than proposed, and there the
        probe is the action that works. The two are not in conflict: this one is about
        a proposal, that one is about a configuration.)

        THE CONSEQUENCE SENTENCE IS NOT ONE SENTENCE. Every check that can reach here
        but one is about a window too narrow for the recording, and for those the
        consequence is a measured fact: the baseline rides into the events and the
        depths read shallow. ``level_occupancy`` is the exception -- it is not about
        the window at all, it is about WHICH of two levels the estimator called the
        open pore (`baseline.LEVEL_OCCUPANCY_MIN`) -- and telling a user their
        baseline "has ridden into the events" when the block said no such thing spends
        the one instruction they will follow on the wrong repair. So the level row
        gets its own consequence, and a mixed condemnation gets the window one, which
        is then true of at least one of the rows named.
        """
        names = "; ".join(label or name for name, label in self._condemned)
        n = len(self._condemned)
        only_level = bool(self._condemned) and all(
            name == "level_occupancy" for name, _label in self._condemned)
        consequence = (
            "Accepting them measures every depth against a level this block cannot "
            "certify is the open pore -- the recording spends enough of itself at a "
            "second level that dI/I0 has two readings, and the row below names both "
            "of them."
            if only_level else
            "Accepting them measures every depth against a baseline the checks say "
            "has ridden into the events themselves, so depth_pA and dI/I0 read "
            "shallow while the level, drift, noise and overshoot rows all stay green."
        )
        remedy = ("read the row below and do what it names"
                  if only_level else
                  "type the widened windows they name into the two spins above")
        return (
            f"REFUSED — {n} of the checks in Baseline health CONDEMN the windows "
            f"proposed above ({names}). {consequence} Read the reasons below, then "
            f"EITHER {remedy}, OR press "
            f"this button again — it now reads “{_APPLY_ANYWAY_TEXT}” — to accept a "
            f"condemned window deliberately. The panel will keep saying on screen "
            f"that you did."
        )

    # ==================================================================
    # gating -- styling, never setEnabled(False)
    # ==================================================================
    def set_gate(self, ok: bool, reason: str = "") -> None:
        """Whether a probe can run here, and the FIRST REAL reason it cannot.

        The buttons stay Qt-enabled either way. A disabled QPushButton never
        receives the click, so it can never answer the question the user is
        actually asking, which is "why not?".
        """
        self._gate_ok = bool(ok)
        self._gate_reason = "" if ok else (reason or "This recording cannot be measured.")
        if ok:
            self._post("")
        self._restyle_buttons()

    def set_busy(self, busy: bool) -> None:
        """A probe is in flight: the button says so and a second click is refused."""
        self._busy = bool(busy)
        self.probe_button.setText(
            "Measuring…" if self._busy else "Measure this recording")
        self._restyle_buttons()

    def is_busy(self) -> bool:
        return self._busy

    def _blocker_kind(self, what: str) -> tuple[str, str]:
        """``(reason, kind)`` -- the FIRST real blocker, and WHICH one it is.

        Ordered by what the user would have to fix first. A gate that reported "run a
        probe first" while no recording was selected would send them to a button that
        is not reachable yet -- the downstream blocker, which is the failure mode this
        ordering exists to avoid.

        The ``kind`` exists because ONE of these blockers is answered from the button
        itself. :meth:`_on_apply_clicked` arms the condemnation override on the click
        it refuses, and it must arm it only when the refusal it just posted WAS the
        condemnation: a click refused by "select a current channel" while a condemned
        proposal happened to be on screen would otherwise arm it silently, and the
        next press -- after the user fixed the selection -- would accept a condemned
        window they were never shown. Matching on the reason STRING would work today
        and rot the first time a sentence is reworded.
        """
        if not self._gate_ok:
            return self._gate_reason, "gate"
        if self.reference_mode() != "automatic":
            return ("Automatic estimator proposals do not apply to a user-provided "
                    "reference. Choose Automatic to measure its windows."), "reference"
        if self._busy:
            return "Still measuring this recording — one moment.", "busy"
        if what == "apply" and self._proposal is None:
            return ("Nothing has been proposed yet. Press Measure this recording "
                    "first, then this button applies what it found."), "unproposed"
        # LAST, because it is the last thing that can be wrong: there is a recording,
        # a finished probe and a proposal, and the checks measured ON THAT PROPOSAL
        # say it must not be used. Once the user has been shown that, this stops
        # blocking -- the override is armed, not the condemnation lifted.
        if what == "apply" and self._condemned and not self._override_armed:
            return self._condemnation_notice(), "condemned"
        return "", ""

    def _blocker(self, what: str) -> str:
        """The FIRST real blocker for ``what``, or "" when nothing blocks it."""
        return self._blocker_kind(what)[0]

    def _post(self, message: str) -> None:
        self.gate_label.setText(message)
        speaking = bool(message)
        self.which_baseline_label.setVisible(not speaking)
        self.gate_label.setVisible(speaking)

    def _restyle_buttons(self) -> None:
        """Dim a button whose action is blocked. Styling only -- it stays clickable.

        Apply has THREE states here, not two, and they must not be spelled with two
        appearances: blocked (dim), ordinary (plain), and *armed over a condemned
        proposal* -- reachable, and a mistake. The third gets its own colour AND its
        own label, because a button that looks ordinary and does something
        extraordinary is exactly the screen this change exists to remove.
        """
        self.probe_button.setStyleSheet(
            _OK_BUTTON if not self._blocker("probe") else _DIM_BUTTON)
        armed = bool(self._condemned and self._override_armed)
        self.apply_button.setText(_APPLY_ANYWAY_TEXT if armed else _APPLY_TEXT)
        self.apply_button.setStyleSheet(
            _WARN_BUTTON if armed
            else _OK_BUTTON if not self._blocker("apply") else _DIM_BUTTON)
        preview_ready = (self._gate_ok and not self._busy
                         and self.reference_mode() != "automatic" and not self.reference_error())
        self.reference_editor.preview_button.setStyleSheet(
            _OK_BUTTON if preview_ready else _DIM_BUTTON)

    # ==================================================================
    # internals
    # ==================================================================
    def show_baseline(self) -> bool:
        """Is the user asking for the estimated baseline ON the trace?"""
        return bool(self.show_track.isChecked())

    def subtract_baseline(self) -> bool:
        """Is the user asking for the baseline-SUBTRACTED trace?"""
        return bool(self.subtract_view.isChecked())

    def say_view_problem(self, message: str) -> None:
        """Post why the baseline could not be DRAWN, on this panel's own refusal line.

        The boxes stay ticked, so something has to say why nothing appeared -- an
        estimator that refuses on this trace (a non-finite sample, a segment shorter
        than the tracking window) otherwise reads as a checkbox that does nothing.
        Public because the surface that owns the trace view is the only one that can
        try the draw, and it is not this panel; ``""`` clears the line. Lands on the
        SAME label a refused click uses, because to the user it is the same question.
        """
        self._post(message)

    def _on_view_toggled(self, _on: bool) -> None:
        """Announce both view requests on either box, and say why if neither can run.

        BOTH, ALWAYS, from either box: the two compose (a subtracted trace with its
        track on it is a flat rail at zero, which is the picture that proves the
        subtraction did what it says) and the surface that applies them draws one
        state, not two independent ones. Emitting only the box that moved made the
        drawn state depend on which box was ticked first.

        The gate's refusal is POSTED, not enforced. The tick stays where the user put
        it -- see the block where these boxes are built -- so selecting a current
        channel afterwards brings the line up with no second gesture.
        """
        if not self._gate_ok and (self.show_baseline() or self.subtract_baseline()):
            self._post(self._gate_reason)
        elif self._gate_ok:
            self._post("")
        self.showBaselineToggled.emit(self.show_baseline())
        self.subtractBaselineToggled.emit(self.subtract_baseline())

    def _on_probe_clicked(self) -> None:
        why = self._blocker("probe")
        if why:
            self._post(why)
            return
        self._post("")
        self.set_busy(True)
        self.probeRequested.emit()

    def _on_apply_clicked(self) -> None:
        why, kind = self._blocker_kind("apply")
        if why:
            self._post(why)
            # A condemnation is the one blocker the user can answer FROM THIS BUTTON,
            # so this click is what arms the override. Every other blocker is
            # something to go and fix, and a click refused by one of THOSE must not
            # arm anything -- the user has not been shown the condemnation, they have
            # been shown "select a current channel".
            if kind == "condemned":
                self._override_armed = True
                self._restyle_buttons()
            return
        # A condemned window that is being accepted keeps its notice on screen. The
        # health block below and the source line above both say so from here on, but
        # blanking the sentence the user just read at the instant they act on it is
        # the app agreeing with itself that nothing happened.
        if not self._condemned:
            self._post("")
        self.apply_proposal()

    def _on_reset_clicked(self) -> None:
        self._post("")
        self._set_windows(DEFAULT_BASELINE.rough_s, DEFAULT_BASELINE.base_s)
        self._source = "default"
        self._accepted_condemned = ()
        self._refresh_source()
        self._refresh_decim()
        self._restyle_buttons()
        self.paramsChanged.emit(self.params())

    def _on_edited(self, *_a) -> None:
        if self._loading:
            return
        self._recouple()
        # A hand edit is a hand edit, whatever proposed the number a moment ago.
        self._source = "user"
        # ...and the standing condemnation mark named the values that WERE in the
        # form. These are not those values. (The health block above is now stale for
        # the same reason and says which windows it was measured on; leaving a mark
        # that claims to describe numbers nobody has judged would be worse than the
        # silence it replaced.)
        self._accepted_condemned = ()
        self._refresh_source()
        self._refresh_decim()
        # ...and the line that contrasts the shown detection with THESE values.
        self._refresh_run_windows()
        self.paramsChanged.emit(self.params())

    def _refresh_source(self) -> None:
        """Where the values came from -- AND whether they were condemned when taken.

        ``_SOURCE_LABELS['proposed']`` reads "measured on this recording, and
        confirmed", which is true of a window the health block refused and is the
        most reassuring sentence in the panel. It stays, because it is the
        provenance; what is added beside it is the other half of the truth.
        """
        text = "These values: " + _SOURCE_LABELS.get(self._source, self._source)
        if self._accepted_condemned:
            text += (" — ACCEPTED OVER "
                     + ("A FAILED HEALTH CHECK" if len(self._accepted_condemned) == 1
                        else f"{len(self._accepted_condemned)} FAILED HEALTH CHECKS")
                     + " (" + "; ".join(self._accepted_condemned) + "). Depths and "
                     "dI/I0 measured with these windows are not certifiable — see "
                     "Baseline health below.")
        self.source_label.setText(text)
        self.source_label.setStyleSheet(
            "color:#ffb454;" if self._accepted_condemned else "color:#8a939c;")

    def _refresh_decim(self) -> None:
        if not self._fs:
            self.decim_label.setText("auto (sample rate unknown)")
            return
        try:
            d = derive_decim(float(self.base_s.value()), self._fs)
        except Exception:  # noqa: BLE001 -- a mid-edit value must not break the label
            self.decim_label.setText("auto")
            return
        taps = int(float(self.base_s.value()) * self._fs / max(d, 1))
        self.decim_label.setText(f"{d}  ({taps} taps)")
