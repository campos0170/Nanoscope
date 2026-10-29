"""Dialog for combining several detector tables into one derived event table.

The dialog is intentionally a previewer, not a store writer.  It owns the choice of
source tables and the reconciliation parameters, calls the pure analysis function,
and exposes the resulting :class:`~nanoscope.analysis.reconcile.ReconcileOutcome`.
``MainWindow`` remains the sole object allowed to install that result into a Session.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6 import QtCore, QtWidgets

from ..analysis.events import EventSet
from ..analysis.reconcile import (
    CONSENSUS_CURATION_CONFLICT_COL,
    CONSENSUS_LABEL_CONFLICTS_COL,
    CONSENSUS_MODULE_ID,
    ReconcileConfig,
    ReconcileOutcome,
    reconcile_detections,
)
from ..analysis.scoring import added_mask
from ..core.analysis import AnalysisResult
from ..core.sources import detector_name
from .busy_cursor import begin_wait_cursor, end_wait_cursor

__all__ = ["ReconcileDialog", "can_reconcile_sources"]


def can_reconcile_sources(sources: Sequence[AnalysisResult]) -> bool:
    """Whether ``sources`` contain a useful detector-vs-detector or manual-vs-auto job.

    A single stored detector table is sufficient only when it contains both automatic
    candidates and hand-added candidates.  The core deliberately permits that case so a
    curator can collapse a manually drawn duplicate onto the detector candidate without
    first running a second detector.
    """
    compatible = tuple(
        result for result in sources
        if result.module_id not in (CONSENSUS_MODULE_ID, "detect.manual")
        and isinstance(result.payload, EventSet)
    )
    if len(compatible) >= 2:
        return True
    if len(compatible) != 1:
        return False
    manual = added_mask(compatible[0].payload.df)
    return bool(manual.any() and (~manual).any())


class ReconcileDialog(QtWidgets.QDialog):
    """Choose detector tables, preview their convergence, then return a result.

    Source results are never mutated.  ``outcome`` remains ``None`` until a preview
    succeeds, and every control change invalidates that preview.  This makes the
    meaning of the final button exact: the counts on screen describe the result that
    will be installed, not an earlier parameter combination.
    """

    def __init__(self, sources: Sequence[AnalysisResult], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reconcile detector results")
        self.setMinimumWidth(590)
        self._sources = tuple(
            res for res in sources
            if res.module_id not in (CONSENSUS_MODULE_ID, "detect.manual")
            and isinstance(res.payload, EventSet)
        )
        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        self._outcome: ReconcileOutcome | None = None
        self._preview_state: tuple[object, ...] | None = None

        root = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Combine overlapping candidates from two or more detectors, or converge "
            "manual and automatic candidates already stored in one detector table, into "
            "a <b>new reconciled event table</b>. The source tables are never changed. "
            "Use the result selector above the event table to move between them later."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        source_box = QtWidgets.QGroupBox("Source detections")
        source_lay = QtWidgets.QVBoxLayout(source_box)
        if self._sources:
            for res in self._sources:
                n_rows = len(res.payload.df)
                accepted = (int(res.payload.df["accepted"].to_numpy(dtype=bool).sum())
                            if "accepted" in res.payload.df.columns else n_rows)
                cb = QtWidgets.QCheckBox(
                    f"{detector_name(res.module_id)} — {n_rows:,} candidates "
                    f"({accepted:,} accepted)"
                )
                cb.setChecked(True)
                cb.setToolTip(
                    f"{res.module_id}\nCandidates are read from this stored table; "
                    "the table itself is not edited."
                )
                cb.toggled.connect(self._settings_changed)
                source_lay.addWidget(cb)
                self._checks[res.module_id] = cb
        else:
            source_lay.addWidget(QtWidgets.QLabel("No compatible detector results are stored."))
        root.addWidget(source_box)

        settings = QtWidgets.QGroupBox("Matching rule")
        form = QtWidgets.QFormLayout(settings)

        self.overlap_spin = QtWidgets.QDoubleSpinBox()
        self.overlap_spin.setObjectName("reconcileOverlap")
        self.overlap_spin.setRange(1.0, 100.0)
        self.overlap_spin.setDecimals(1)
        self.overlap_spin.setSingleStep(5.0)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(50.0)
        self.overlap_spin.setToolTip(
            "Two candidates can join when their shared time is at least this fraction "
            "of the shorter candidate."
        )
        form.addRow("Minimum temporal overlap", self.overlap_spin)

        support_row = QtWidgets.QWidget()
        support_lay = QtWidgets.QHBoxLayout(support_row)
        support_lay.setContentsMargins(0, 0, 0, 0)
        self.support_spin = QtWidgets.QSpinBox()
        self.support_spin.setObjectName("reconcileSupport")
        self.support_spin.setRange(1, max(1, len(self._sources)))
        self.support_spin.setValue(1)
        self.support_spin.setToolTip(
            "Minimum number of selected detector tables that must support an automatic "
            "candidate. Manual-only candidates can be retained separately below."
        )
        self.support_note = QtWidgets.QLabel()
        self.support_note.setWordWrap(True)
        self.support_note.setStyleSheet("color:#8b9199;")
        support_lay.addWidget(self.support_spin)
        support_lay.addWidget(self.support_note, 1)
        form.addRow("Required detector support", support_row)

        self.keep_manual = QtWidgets.QCheckBox("Keep manual-only events")
        self.keep_manual.setChecked(True)
        self.keep_manual.setToolTip(
            "Keep a candidate that was added by hand even when no automatic detector "
            "supports it. Recommended when the table will become supervised-learning truth."
        )
        form.addRow("", self.keep_manual)
        root.addWidget(settings)

        preview_box = QtWidgets.QGroupBox("Preview")
        preview_lay = QtWidgets.QVBoxLayout(preview_box)
        self.preview_message = QtWidgets.QLabel(
            "Click Preview to calculate the derived table. Nothing is stored yet. "
            "Large event tables may take several seconds."
        )
        self.preview_message.setObjectName("reconcilePreviewMessage")
        self.preview_message.setWordWrap(True)
        self.preview_message.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        preview_lay.addWidget(self.preview_message)

        summary = QtWidgets.QFormLayout()
        self.summary_labels: dict[str, QtWidgets.QLabel] = {}
        for key, caption in (
            ("n_input_rows", "Input candidates"),
            ("n_clusters", "Overlap groups"),
            ("n_output_rows", "Output candidates"),
            ("n_collapsed", "Rows converged"),
            ("n_filtered", "Groups filtered by support"),
            ("n_manual_kept", "Manual-only events kept"),
            ("n_curation_conflicts", "Contradictory curation decisions"),
            ("n_label_conflicts", "Unresolved label conflicts"),
        ):
            value = QtWidgets.QLabel("—")
            value.setObjectName(f"reconcile_{key}")
            summary.addRow(caption, value)
            self.summary_labels[key] = value
        preview_lay.addLayout(summary)
        root.addWidget(preview_box)

        self.buttons = QtWidgets.QDialogButtonBox()
        self.preview_button = self.buttons.addButton(
            "Preview", QtWidgets.QDialogButtonBox.ActionRole)
        self.create_button = self.buttons.addButton(
            "Create reconciled table", QtWidgets.QDialogButtonBox.AcceptRole)
        self.cancel_button = self.buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.create_button.setEnabled(False)
        self.create_button.setToolTip(
            "Preview these exact settings before creating the derived table."
        )
        self.preview_button.setDefault(True)
        root.addWidget(self.buttons)

        self.preview_button.clicked.connect(self.preview)
        self.buttons.accepted.connect(self._accept_current_preview)
        self.buttons.rejected.connect(self.reject)
        self.overlap_spin.valueChanged.connect(self._settings_changed)
        self.support_spin.valueChanged.connect(self._settings_changed)
        self.keep_manual.toggled.connect(self._settings_changed)
        self._settings_changed()

    @property
    def outcome(self) -> ReconcileOutcome | None:
        """The last successful, still-current preview, or ``None``."""
        return self._outcome

    @property
    def source_checks(self) -> dict[str, QtWidgets.QCheckBox]:
        """Module-id keyed checkboxes (exposed read-only for GUI automation)."""
        return dict(self._checks)

    def selected_sources(self) -> tuple[AnalysisResult, ...]:
        return tuple(res for res in self._sources if self._checks[res.module_id].isChecked())

    def config(self) -> ReconcileConfig:
        return ReconcileConfig(
            min_overlap_fraction=float(self.overlap_spin.value()) / 100.0,
            min_detector_support=int(self.support_spin.value()),
            keep_manual=bool(self.keep_manual.isChecked()),
        )

    def _state(self) -> tuple[object, ...]:
        return (
            tuple(res.module_id for res in self.selected_sources()),
            float(self.overlap_spin.value()),
            int(self.support_spin.value()),
            bool(self.keep_manual.isChecked()),
        )

    @QtCore.Slot()
    def _settings_changed(self, *_args) -> None:
        n_selected = len(self.selected_sources())
        old_support = self.support_spin.value()
        self.support_spin.blockSignals(True)
        self.support_spin.setMaximum(max(1, n_selected))
        self.support_spin.setValue(min(old_support, max(1, n_selected)))
        self.support_spin.blockSignals(False)

        support = self.support_spin.value()
        if support == 1:
            note = "Union: keep a candidate found by any selected detector."
        elif support >= n_selected and n_selected:
            note = "Intersection: require every selected detector to support it."
        else:
            note = f"Agreement: require at least {support} selected detectors."
        self.support_note.setText(note)

        self._outcome = None
        self._preview_state = None
        self.create_button.setEnabled(False)
        for label in self.summary_labels.values():
            label.setText("—")
        valid = can_reconcile_sources(self.selected_sources())
        self.preview_button.setEnabled(valid)
        if valid:
            self.preview_message.setStyleSheet("")
            self.preview_message.setText(
                "Settings changed — click Preview. Nothing is stored until you create the table."
            )
            self.preview_button.setToolTip("")
        else:
            self.preview_message.setStyleSheet("color:#ffd479;")
            self.preview_message.setText(
                "Select at least two detector results. One result is enough only when its "
                "table contains both hand-added and automatic candidates."
            )
            self.preview_button.setToolTip(
                "Select two detector results, or one result containing both manual and "
                "automatic candidates."
            )

    @QtCore.Slot()
    def preview(self) -> None:
        selected = self.selected_sources()
        if not can_reconcile_sources(selected):
            self._settings_changed()
            return
        self._outcome = None
        self._preview_state = None
        self.create_button.setEnabled(False)
        self.preview_message.setStyleSheet("")
        self.preview_message.setText(
            "Calculating reconciliation… Large event tables may take several seconds."
        )
        for check in self._checks.values():
            check.setEnabled(False)
        for widget in (self.overlap_spin, self.support_spin, self.keep_manual,
                       self.preview_button, self.cancel_button):
            widget.setEnabled(False)
        cursor_active = begin_wait_cursor()
        # Paint the busy state before the CPU-bound pure analysis call begins. Excluding
        # input avoids a nested click changing parameters behind the preview being built.
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        try:
            outcome = reconcile_detections(selected, config=self.config())
        except (TypeError, ValueError) as exc:
            self._outcome = None
            self._preview_state = None
            self.preview_message.setStyleSheet("color:#ff8a80;")
            self.preview_message.setText(f"Could not reconcile these results: {exc}")
            return
        finally:
            end_wait_cursor(cursor_active)
            for check in self._checks.values():
                check.setEnabled(True)
            for widget in (self.overlap_spin, self.support_spin, self.keep_manual,
                           self.cancel_button):
                widget.setEnabled(True)
            self.preview_button.setEnabled(can_reconcile_sources(self.selected_sources()))

        self._outcome = outcome
        self._preview_state = self._state()
        n_conflicts = int(outcome.n_label_conflicts)
        n_curation_conflicts = int(outcome.n_curation_conflicts)
        self.preview_message.setStyleSheet(
            "color:#ffd479;"
            if n_conflicts or n_curation_conflicts else "color:#9ccc65;")
        label_conflict_note = (
            f" {n_conflicts:,} candidate(s) have unresolved label conflicts; inspect "
            f"the {CONSENSUS_LABEL_CONFLICTS_COL} column before training."
            if n_conflicts else ""
        )
        curation_conflict_note = (
            f" {n_curation_conflicts:,} candidate(s) combine contradictory human "
            f"accept/reject decisions; inspect the {CONSENSUS_CURATION_CONFLICT_COL} "
            "column and explicitly Accept or Reject each one before training."
            if n_curation_conflicts else ""
        )
        self.preview_message.setText(
            "Preview ready. Review status is carried conservatively; inspect, "
            "accept/reject, and label the result like any other event table. Source "
            f"tables remain unchanged.{curation_conflict_note}{label_conflict_note}"
        )
        for key, label in self.summary_labels.items():
            label.setText(f"{int(getattr(outcome, key)):,}")
        self.create_button.setEnabled(True)
        self.create_button.setToolTip(
            "Store this preview as the separate Reconciled result and show it in the event table."
        )
        self.create_button.setDefault(True)
        self.create_button.setFocus()

    @QtCore.Slot()
    def _accept_current_preview(self) -> None:
        if self._outcome is None or self._preview_state != self._state():
            return
        self.accept()
