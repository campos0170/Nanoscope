"""A single, honest workspace for inspecting and applying baseline correction.

``BaselinePanel`` owns the open-pore reference estimator: its rough/base windows
set the reference used to measure event depth.  The signal chain owns a different
operation, *baseline drift correction*, which can change the trace seen by
detection while retaining a representative open-pore offset.  This widget puts
those contracts together without merging their parameters or doing numerical work
on the GUI thread.

The owner supplies arrays to :meth:`set_preview` after computing them in a worker
and connects the intent signals below to the existing analysis controls.  This is
presentation-only: it neither estimates a baseline nor changes the active trace.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6 import QtCore, QtWidgets

from ..plot.style import BASELINE_TRACK_INK, PALETTE
from .inspector_widgets import Disclosure
from .background_controls import BackgroundControls
from .widget_transfer import prepare_widget_move

__all__ = [
    "BaselineCorrectionWorkspace",
    "DISPLAY_ZERO",
    "RUNTIME_RETAIN_OFFSET",
]


# Values describe the numerical contract rather than a human-facing combo label.
DISPLAY_ZERO = "zero_centered_display"
RUNTIME_RETAIN_OFFSET = "retained_offset_runtime"

_MAX_PREVIEW_SAMPLES = 8_000
_CAPTION_CSS = "color:#8b9199;"
_WARN_CSS = f"color:{PALETTE['warn']};"
_BUSY_CSS = "color:#56B4E9;"
_RAW_INK = PALETTE["data"]
_CORRECTED_INK = PALETTE["accent"]


def _caption(text: str = "") -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(_CAPTION_CSS)
    label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class BaselineCorrectionWorkspace(QtWidgets.QWidget):
    """Presentation layer around an existing open-pore ``BaselinePanel``.

    Args:
        baseline_panel: An already-created open-pore estimator panel.  It is
            reparented into this workspace rather than duplicated, preserving its
            proposal/confirmation and provenance behaviour.
        method_choices: ``(label, value)`` options from the signal-chain baseline
            drift control.  This widget requests a choice; the signal chain remains
            the authoritative owner of its parameters.

    Signals:
        refreshRequested(): Compute a fresh raw/baseline/corrected comparison.
        exportRequested(): Export full-resolution raw, fitted-baseline and corrected
            arrays.  The preview itself is intentionally decimated.
        advancedRequested(): Reveal the signal-chain baseline-drift settings.
        modeChanged(str): One of :data:`DISPLAY_ZERO` or
            :data:`RUNTIME_RETAIN_OFFSET`.
        runtimeEnabledRequested(bool): Request a change to ``detrend_enabled``.
        runtimeMethodRequested(str): Request a change to ``detrend_method``.

    Public owner-facing APIs: :meth:`sync_runtime`, :meth:`sync_display_state`,
    :meth:`set_preview`, :meth:`clear_preview`, :meth:`set_busy`,
    :meth:`set_status`, and :meth:`mark_preview_stale`.
    """

    refreshRequested = QtCore.Signal()
    exportRequested = QtCore.Signal()
    advancedRequested = QtCore.Signal()
    modeChanged = QtCore.Signal(str)
    runtimeEnabledRequested = QtCore.Signal(bool)
    runtimeMethodRequested = QtCore.Signal(str)
    recipeChanged = QtCore.Signal()
    referenceEditRequested = QtCore.Signal()

    def __init__(
        self,
        baseline_panel: QtWidgets.QWidget,
        *,
        method_choices: Iterable[tuple[str, str]] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("baselineCorrectionWorkspace")
        self.setToolTip(
            "Compare the raw current, fitted baseline and corrected output in one "
            "place. The zero-centred view is visual only; runtime drift correction "
            "is a separate signal-chain operation used by detection and processed "
            "trace export.")

        # Deliberately keep a reference to the real existing view, not a clone.
        self.baseline_panel = baseline_panel
        self._updating = False
        self._busy = False
        self._stale = False
        self._stale_reason = ""
        self._workspace_hosted = False
        self._comparison_splitter = None
        self._display_show = False
        self._display_subtract = False
        self._runtime_enabled = False
        self._runtime_method = ""
        self._preview_time_s: np.ndarray | None = None
        self._preview_raw: np.ndarray | None = None
        self._preview_baseline: np.ndarray | None = None
        self._preview_corrected: np.ndarray | None = None
        self._expanded_dialog = None
        self._expanded_status = None
        self._expanded_freshness = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.background = BackgroundControls(self)
        root.addWidget(self.background)

        # The existing form stays the owner of the open-pore estimator and its
        # proposal/confirmation workflow.  Keep it first: its two trace-display
        # checkboxes are the direct, familiar controls and must remain reachable
        # without scrolling through the new comparison workspace in the Inspector.
        # QLayout handles reparenting here.
        if hasattr(self.baseline_panel, 'display_group'):
            prepare_widget_move(self.baseline_panel.display_group)
            root.addWidget(self.baseline_panel.display_group)
        prepare_widget_move(self.baseline_panel)
        root.addWidget(self.baseline_panel)
        self.runtime_box = self._build_contract_box(method_choices)
        self.action_widget = self._build_actions()
        root.addWidget(self.action_widget)
        self.preview_box = self._build_preview_box()
        root.addWidget(self.preview_box)
        root.addWidget(self.runtime_box)

        self.output_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.runtime_enabled.toggled.connect(self._on_runtime_enabled_toggled)
        self.runtime_method.currentIndexChanged.connect(self._on_runtime_method_changed)
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        self.export_button.clicked.connect(self._on_export_clicked)
        self.advanced_button.clicked.connect(self.advancedRequested.emit)
        self.background.changed.connect(self._on_background_changed)
        self._refresh_contract_text()
        self._refresh_display_state()
        self._refresh_freshness()

    # -- construction ---------------------------------------------------------------------------
    def _build_contract_box(
        self, method_choices: Iterable[tuple[str, str]]
    ) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Automatic drift for detection")
        box.setToolTip(
            "Open-pore baseline estimates the reference used for event depth; "
            "baseline drift correction is a runtime signal-chain transformation.")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(5)

        # Labels go ABOVE their controls, not in a QFormLayout column.  The latter
        # sums label + combo minima and would floor the Inspector dock at ~400 px.
        # A narrow dock should show an ordinary horizontal scrollbar only where one
        # is genuinely needed, not permanently steal width from the signal viewer.
        self.output_mode_combo = QtWidgets.QComboBox()
        self.output_mode_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.output_mode_combo.setMinimumContentsLength(14)
        self.output_mode_combo.addItem("Event-reference residual (zero)", DISPLAY_ZERO)
        self.output_mode_combo.addItem("Drift-stage correction (offset retained)",
                                      RUNTIME_RETAIN_OFFSET)
        self.output_mode_combo.setToolTip(
            "Choose the comparison and exported output. This does not change the "
            "main-trace display or detector input. Use the separate display checkboxes "
            "or automatic runtime switch deliberately.")
        self.contract_label = _caption()
        self.contract_label.setObjectName("baselineCorrectionContract")

        self.display_state_label = _caption()
        self.display_state_label.setObjectName("baselineDisplayState")
        self.display_state_label.setToolTip(
            "The state of the main trace, separate because trace display does not "
            "change detector input or processed-trace export.")
        if hasattr(self.baseline_panel, 'display_group'):
            self.baseline_panel.display_group.layout().addWidget(self.display_state_label)
        else:
            layout.addWidget(self.display_state_label)

        self.runtime_enabled = QtWidgets.QCheckBox("Enable automatic drift")
        self.runtime_enabled.setToolTip(
            "Apply signal-chain baseline-drift correction before detection and when "
            "exporting the processed trace. It retains a representative open-pore "
            "offset; it does not use the zero-centred display residual.")
        layout.addWidget(self.runtime_enabled)

        self.runtime_method_label = QtWidgets.QLabel("Drift estimator")
        layout.addWidget(self.runtime_method_label)
        self.runtime_method = QtWidgets.QComboBox()
        self.runtime_method.setToolTip(
            "Estimator used by signal-chain drift correction. Advanced settings "
            "contains its windows and method-specific parameters.")
        for label, value in method_choices:
            self.runtime_method.addItem(str(label), str(value))
        if self.runtime_method.count() == 0:
            self.runtime_method.addItem("Choose in advanced settings", "")
        layout.addWidget(self.runtime_method)

        self.runtime_note = _caption(
            "Retains an open-pore offset. The event reference and display settings "
            "remain separate; the ordinary Median filter removes spikes.")
        layout.addWidget(self.runtime_note)
        self.input_state_label = _caption()
        layout.addWidget(self.input_state_label)
        return box

    def _build_preview_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Comparison preview")
        box.setToolTip(
            "Top: raw current with its fitted baseline. Bottom: the selected output. "
            "The preview is decimated for responsiveness; export writes full arrays.")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)
        self._preview_layout = layout
        self.output_mode_label = QtWidgets.QLabel('Comparison calculation')
        layout.addWidget(self.output_mode_label)
        layout.addWidget(self.output_mode_combo)
        layout.addWidget(self.contract_label)
        self.reference_summary_box = QtWidgets.QGroupBox("Shared event reference")
        reference_layout = QtWidgets.QVBoxLayout(self.reference_summary_box)
        reference_layout.setContentsMargins(8, 6, 8, 8)
        self.reference_summary = _caption("Uses the current event-reference settings.")
        reference_layout.addWidget(self.reference_summary)
        self.edit_reference_button = QtWidgets.QPushButton("Edit event reference…")
        self.edit_reference_button.clicked.connect(self.referenceEditRequested.emit)
        reference_layout.addWidget(self.edit_reference_button)
        reference_layout.addWidget(_caption(
            "Editing the shared reference changes future event measurements, including "
            "ML characterization. Calculating this comparison does not change stored results."))
        layout.addWidget(self.reference_summary_box)
        self.reference_summary_box.hide()
        self.freshness_label = _caption()
        self.freshness_label.setObjectName("baselineComparisonFreshness")
        layout.addWidget(self.freshness_label)
        self._expanded_placeholder = _caption('Comparison is open in a separate window.')
        self._expanded_placeholder.hide()
        layout.addWidget(self._expanded_placeholder)

        self.preview = pg.GraphicsLayoutWidget()
        self.preview.setObjectName("baselineComparisonPreview")
        self.preview.setBackground(None)
        self.preview.setMinimumHeight(240)
        self.preview.setMinimumWidth(0)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                   QtWidgets.QSizePolicy.Policy.Expanding)

        self.raw_plot = self.preview.addPlot(row=0, col=0)
        self.corrected_plot = self.preview.addPlot(row=1, col=0)
        self.corrected_plot.setXLink(self.raw_plot)
        for plot in (self.raw_plot, self.corrected_plot):
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            plot.getAxis("bottom").setScale(1000.)
            plot.getAxis("left").enableAutoSIPrefix(False)
        self.raw_plot.setLabel("left", "Current")
        self.corrected_plot.setLabel("left", "Corrected current")
        self.corrected_plot.setLabel("bottom", "Time from recording start (ms)")
        self.raw_plot.getAxis("bottom").setStyle(showValues=False)
        self.raw_plot.setTitle("Raw current + fitted baseline")
        self.corrected_plot.setTitle("Selected corrected output")

        self.raw_curve = self.raw_plot.plot(
            pen=pg.mkPen(_RAW_INK, width=1.25), connect="finite")
        self.baseline_curve = self.raw_plot.plot(
            pen=pg.mkPen(BASELINE_TRACK_INK, width=1.5,
                          style=QtCore.Qt.PenStyle.DashLine),
            connect="finite")
        self.corrected_curve = self.corrected_plot.plot(
            pen=pg.mkPen(_CORRECTED_INK, width=1.25), connect="finite")
        layout.addWidget(self.preview)

        self.preview_key = _caption(
            "Blue = raw current · white dashed = fitted baseline · green = selected output")
        layout.addWidget(self.preview_key)
        self.preview_status = _caption(
            "Choose a current recording, then refresh the comparison.")
        self.preview_status.setObjectName("baselinePreviewStatus")
        layout.addWidget(self.preview_status)
        self._preview_secondary = QtWidgets.QGridLayout()
        self._preview_secondary.addWidget(self.export_button, 0, 0)
        self._preview_secondary.addWidget(self.expand_button, 1, 0)
        layout.addLayout(self._preview_secondary)
        return box

    def _build_actions(self) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        actions = QtWidgets.QVBoxLayout(host)
        self._action_layout = actions
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self.refresh_button = QtWidgets.QPushButton("Update comparison")
        self.refresh_button.setToolTip(
            "Calculate a fresh raw / fitted-baseline / corrected comparison in the "
            "background for the selected recording.")
        self.export_button = QtWidgets.QPushButton("Export current recipe…")
        self.export_button.setToolTip(
            "Calculate the CURRENT recipe and export full-resolution time, raw, "
            "baseline and corrected current plus provenance. An older visible comparison "
            "is not exported. Raw recordings and detection results are unchanged.")
        self.advanced_button = QtWidgets.QPushButton("Edit drift settings…")
        self.advanced_button.setToolTip(
            "Open the baseline-drift page in Signal chain for runtime windows and "
            "method-specific controls.")
        actions.addWidget(self.refresh_button)
        # The existing Advanced action reveals the authoritative signal-chain
        # controls. It never creates a second owner for these parameters.
        advanced_host = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(advanced_host)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        runtime_layout = self.runtime_note.parentWidget().layout()
        prepare_widget_move(self.runtime_note)
        advanced_layout.addWidget(self.runtime_note)
        self.runtime_details = Disclosure('Correction details', advanced_host)
        runtime_layout.addWidget(self.advanced_button)
        runtime_layout.addWidget(self.runtime_details)
        self.expand_button = QtWidgets.QPushButton('Expand comparison…')
        self.expand_button.clicked.connect(self.expand_comparison)
        return host

    def expand_comparison(self) -> None:
        """Enlarge the existing plots, preserving their arrays and live updates."""
        if self._expanded_dialog is not None:
            self._expanded_dialog.show()
            self._expanded_dialog.raise_()
            self._expanded_dialog.activateWindow()
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Baseline comparison')
        dialog.resize(960, 720)
        layout = QtWidgets.QVBoxLayout(dialog)
        prepare_widget_move(self.preview)
        self._expanded_placeholder.show()
        layout.addWidget(self.preview, 1)
        self._expanded_status = _caption(self.preview_status.text())
        self._expanded_status.setStyleSheet(self.preview_status.styleSheet())
        self._expanded_freshness = _caption()
        layout.insertWidget(0, self._expanded_freshness)
        self._refresh_freshness()
        layout.addWidget(self._expanded_status)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(dialog.close)
        layout.addWidget(close)
        self._expanded_dialog = dialog
        dialog.finished.connect(self._restore_compact_comparison)
        dialog.show()

    def _restore_compact_comparison(self, _result=0) -> None:
        dialog = self._expanded_dialog
        if dialog is None:
            return
        prepare_widget_move(self.preview)
        index = self._preview_layout.indexOf(self._expanded_placeholder)
        self._preview_layout.insertWidget(index + 1, self.preview)
        self._expanded_placeholder.hide()
        self._expanded_dialog = None
        self._expanded_status = None
        self._expanded_freshness = None
        dialog.deleteLater()

    def set_workspace_hosted(self, hosted: bool = True) -> None:
        """Arrange comparison controls beside the plot in the shared workspace.

        The owner may reparent ``baseline_panel``, its ``display_group`` and
        ``runtime_box`` to separate reference/global sections. They remain the
        same widgets, and source changes never show or hide those hosted sections.
        Calling this method again is harmless; it never changes scientific state.
        """
        hosted = bool(hosted)
        if self._workspace_hosted == hosted:
            return
        self._workspace_hosted = hosted
        root = self.layout()
        if hosted:
            if self._comparison_splitter is None:
                self._comparison_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
                self._comparison_splitter.setObjectName("backgroundComparisonSplitter")
                self._comparison_splitter.setChildrenCollapsible(False)
                self._recipe_host = QtWidgets.QWidget()
                self._recipe_layout = QtWidgets.QVBoxLayout(self._recipe_host)
                self._recipe_layout.setContentsMargins(0, 0, 4, 0)
                self._recipe_scroll = QtWidgets.QScrollArea()
                self._recipe_scroll.setWidgetResizable(True)
                self._recipe_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
                self._recipe_scroll.setMinimumWidth(240)
                self._recipe_scroll.setWidget(self._recipe_host)
                self._calculation_box = QtWidgets.QGroupBox("Calculation and scope")
                self._calculation_layout = QtWidgets.QVBoxLayout(self._calculation_box)
                self._comparison_splitter.addWidget(self._recipe_scroll)
            # action_widget may currently belong to the Inspector's pinned action
            # holder. Detach from its actual layout before reclaiming the footer.
            for widget in (self.background, self.preview_box, self.action_widget):
                prepare_widget_move(widget)
            self._recipe_layout.addWidget(self.background)
            for widget in (self.output_mode_label, self.output_mode_combo,
                           self.contract_label, self.reference_summary_box):
                prepare_widget_move(widget)
                self._calculation_layout.addWidget(widget)
            prepare_widget_move(self._calculation_box)
            self._recipe_layout.addWidget(self._calculation_box)
            self._recipe_layout.addStretch(1)
            self._comparison_splitter.addWidget(self.preview_box)
            self._comparison_splitter.setStretchFactor(0, 0)
            self._comparison_splitter.setStretchFactor(1, 1)
            self._comparison_splitter.setSizes([320, 620])
            root.addWidget(self._comparison_splitter, 1)
            root.addWidget(self.action_widget)
            self._action_layout.setDirection(QtWidgets.QBoxLayout.LeftToRight)
            prepare_widget_move(self.export_button)
            self._action_layout.addWidget(self.export_button)
            self._action_layout.addStretch(1)
            self.preview_box.setMinimumWidth(0)
            self._comparison_splitter.show()
        else:
            # The shared host normally lives until teardown, but retaining the
            # compact form makes this presentation seam useful in isolation too.
            prepare_widget_move(self._comparison_splitter)
            prepare_widget_move(self.action_widget)
            prepare_widget_move(self.background)
            prepare_widget_move(self._calculation_box)
            while self._recipe_layout.count():
                shiboken6.delete(self._recipe_layout.takeAt(0))
            for index, widget in enumerate((self.output_mode_label,
                                            self.output_mode_combo,
                                            self.contract_label,
                                            self.reference_summary_box)):
                prepare_widget_move(widget)
                self._preview_layout.insertWidget(index, widget)
            root.insertWidget(0, self.background)
            root.addWidget(self.action_widget)
            prepare_widget_move(self.preview_box)
            root.addWidget(self.preview_box)
            prepare_widget_move(self.export_button)
            self._preview_secondary.addWidget(self.export_button, 0, 0)
            self._action_layout.setDirection(QtWidgets.QBoxLayout.TopToBottom)
            for index in reversed(range(self._action_layout.count())):
                if self._action_layout.itemAt(index).spacerItem() is not None:
                    shiboken6.delete(self._action_layout.takeAt(index))
            self._comparison_splitter.hide()
        self._refresh_contract_text()

    def sync_reference_summary(self, text: str) -> None:
        """Describe the owner's shared reference without creating a second form."""
        self.reference_summary.setText(str(text))

    # -- state exposed to the owner -------------------------------------------------------------
    def output_mode(self) -> str:
        """Return the stable selected output token, never the visible label."""
        return str(self.output_mode_combo.currentData() or DISPLAY_ZERO)

    def form_state(self) -> dict[str, Any]:
        return {"output_mode": self.output_mode(), "draft": self.background.state()}

    def has_recipe_draft(self) -> bool:
        state = self.background.state()
        return (self.output_mode() != DISPLAY_ZERO or state['source'] != 'automatic'
                or bool(state['regions'] or state['anchors'] or state['reference']))

    def restore_form_state(self, state: Mapping[str, Any]) -> None:
        self._updating = True
        try:
            mode = state.get("output_mode", DISPLAY_ZERO)
            index = self.output_mode_combo.findData(mode)
            if index < 0:
                raise ValueError(f"Unknown background output mode: {mode}")
            self.output_mode_combo.setCurrentIndex(index)
            self.background.restore(state.get("draft", {}))
        finally:
            self._updating = False
        self._on_background_changed()

    def _on_background_changed(self) -> None:
        automatic = self.background.source_key() == 'automatic'
        reference = self.background.source_key() == 'intervals'
        # Comparison source is independent of both runtime and main-trace display.
        # In particular it must never disable an already-enabled automatic stage.
        self.runtime_box.setEnabled(True)
        if not self._workspace_hosted:
            self.baseline_panel.setVisible(automatic or reference)
            if hasattr(self.baseline_panel, 'display_group'):
                self.baseline_panel.display_group.show()
        self._refresh_contract_text()
        if not self._updating:
            self.mark_preview_stale('Background recipe changed — calculate comparison to inspect it.')
            self.recipeChanged.emit()

    def is_busy(self) -> bool:
        """Whether a refresh/export job is currently owned by the controller."""
        return self._busy

    def is_preview_stale(self) -> bool:
        """Whether controls changed after the most recently supplied preview."""
        return self._stale

    def sync_runtime(self, params: Mapping[str, Any]) -> None:
        """Mirror authoritative signal-chain ``detrend_*`` params without signals.

        The caller owns the actual parameters.  This wrapper does not write them,
        so restore and Advanced edits cannot create a feedback loop or second source
        of truth.
        """
        enabled = bool(params.get("detrend_enabled", False))
        method_value = str(params.get("detrend_method", "") or "")
        self._updating = True
        try:
            self.runtime_enabled.setChecked(enabled)
            if method_value:
                idx = self.runtime_method.findData(method_value)
                if idx < 0:
                    # A stored method remains legible if a plugin/newer version adds it.
                    self.runtime_method.addItem(method_value.replace("_", " "), method_value)
                    idx = self.runtime_method.count() - 1
                self.runtime_method.setCurrentIndex(idx)
        finally:
            self._updating = False
        self._runtime_enabled = enabled
        self._runtime_method = method_value
        self._refresh_contract_text()

    def sync_display_state(self, show: bool, subtract: bool) -> None:
        """Mirror the main trace's open-pore overlay/residual display state.

        This stays separate from :meth:`sync_runtime`: a zero-centred trace is a
        visual aid and must not be mistaken for detector input.
        """
        self._display_show = bool(show)
        self._display_subtract = bool(subtract)
        self._refresh_display_state()

    def set_preview(
        self,
        time_s: Any | Mapping[str, Any],
        raw: Any | None = None,
        baseline: Any | None = None,
        corrected: Any | None = None,
        units: str = "pA",
        mode: str | None = None,
        message: str = "",
    ) -> None:
        """Render a compact comparison supplied by an off-thread calculation.

        The explicit form is ``(time_s, raw, baseline, corrected, units, mode,
        message)``. A worker mapping is accepted too: ``time_s`` (or ``time``),
        ``raw``, ``baseline``, ``corrected``, and optional ``units``, ``mode``, and
        ``message``. ``baseline`` and ``corrected`` may be ``None``; every supplied
        series otherwise aligns sample-for-sample.
        """
        if isinstance(time_s, Mapping):
            payload = time_s
            time_s = payload.get("time_s", payload.get("time"))
            raw = payload.get("raw", payload.get("raw_current", raw))
            baseline = payload.get("baseline", payload.get("fitted_baseline", baseline))
            corrected = payload.get("corrected", payload.get("corrected_current", corrected))
            units = str(payload.get("units", units))
            mode = payload.get("mode", mode)
            message = str(payload.get("message", message))
        if time_s is None or raw is None:
            raise ValueError("set_preview needs time_s and raw arrays")

        x = self._as_vector(time_s, "time_s")
        y_raw = self._as_vector(raw, "raw")
        if x.size != y_raw.size:
            raise ValueError("time_s and raw must have the same length")
        y_baseline = self._optional_vector(baseline, "baseline", x.size)
        y_corrected = self._optional_vector(corrected, "corrected", x.size)

        index = self._preview_index(x.size)
        self._preview_time_s = x[index]
        self._preview_raw = y_raw[index]
        self._preview_baseline = None if y_baseline is None else y_baseline[index]
        self._preview_corrected = None if y_corrected is None else y_corrected[index]

        self.raw_curve.setData(self._preview_time_s, self._preview_raw)
        self.baseline_curve.setData(
            [] if self._preview_baseline is None else self._preview_time_s,
            [] if self._preview_baseline is None else self._preview_baseline)
        self.corrected_curve.setData(
            [] if self._preview_corrected is None else self._preview_time_s,
            [] if self._preview_corrected is None else self._preview_corrected)

        # Treat an omitted/empty token as the current UI choice.  This keeps the
        # explicit positional form convenient for a caller that has no separate
        # mode field, while unknown non-empty values still follow the non-display
        # branch and remain visible in the owner's status/provenance validation.
        selected_mode = self.output_mode() if not mode else str(mode)
        if selected_mode == DISPLAY_ZERO:
            lower_title = ("Zero-centred residual — inspection/export"
                           if self.background.source_key() != 'automatic' else
                           "Event-reference residual — zero")
            lower_label = "Δ vs baseline"
            default_message = (
                "Comparison/export uses raw minus the custom background. Main-trace "
                "display and detection settings are unchanged."
                if self.background.source_key() != 'automatic' else
                "Comparison/export uses raw minus the event reference. Main-trace "
                "display and detection settings are unchanged.")
        else:
            lower_title = ("Corrected signal — retained background offset"
                           if self.background.source_key() != 'automatic' else
                           "Drift-corrected signal — retained open-pore offset")
            lower_label = "Corrected current"
            default_message = (
                "Comparison/export uses raw minus the custom background plus its "
                "median level. Main-trace display and detection settings are unchanged."
                if self.background.source_key() != 'automatic' else
                "Preview uses the runtime drift-correction contract and retains a "
                "representative open-pore offset for detection and processed export."
                if self._runtime_enabled else
                "This is a runtime-style preview only. Enable automatic drift "
                "before detection or processed export will use it.")
        self.corrected_plot.setTitle(lower_title)
        self.corrected_plot.setLabel("left", lower_label, units=str(units))
        self.raw_plot.setLabel("left", "Current", units=str(units))

        self._stale = False
        self._stale_reason = ""
        self._refresh_freshness()
        self.set_busy(False)
        self._set_preview_status(message or default_message, _CAPTION_CSS)
        if self._preview_time_s.size:
            # Preview arrays are capped at 8k samples, so this is cheap and gives a
            # fresh recording an honest full-span framing.
            self.raw_plot.enableAutoRange()
            self.corrected_plot.enableAutoRange(y=True)
            self.raw_plot.autoRange()
            self.corrected_plot.autoRange()

    def clear_preview(
        self,
        message: str = "Choose a current recording, then refresh the comparison.",
    ) -> None:
        """Remove the visible preview while leaving all numerical controls untouched."""
        self._preview_time_s = None
        self._preview_raw = None
        self._preview_baseline = None
        self._preview_corrected = None
        for curve in (self.raw_curve, self.baseline_curve, self.corrected_curve):
            curve.setData([], [])
        self._stale = False
        self._stale_reason = ""
        self._refresh_freshness()
        self.set_busy(False)
        self._set_preview_status(message, _CAPTION_CSS)

    def set_busy(self, busy: bool, message: str | None = None) -> None:
        """Show an in-flight job without disabling controls that should explain it."""
        self._busy = bool(busy)
        self.refresh_button.setText(
            "Calculating comparison…" if self._busy else "Update comparison")
        # Keep buttons clickable: a second click reports the first real state instead
        # of silently swallowing an intent because a disabled Qt button gets no click.
        dim = "color:#6b7178;" if self._busy else ""
        self.refresh_button.setStyleSheet(dim)
        self.export_button.setStyleSheet(dim)
        if self._busy:
            self._set_preview_status(
                message or "Calculating comparison in the background…", _BUSY_CSS)

    def set_status(self, message: str, *, warning: bool = False) -> None:
        """Report an owner-level refusal, export result, or non-fatal warning."""
        self._set_preview_status(str(message), _WARN_CSS if warning else _CAPTION_CSS)

    def mark_preview_stale(self, reason: str | None = None) -> None:
        """Keep the last preview visible, but mark it as out of date."""
        self._stale = True
        self._stale_reason = reason or "Settings changed — update comparison."
        self._refresh_freshness()
        if not self._busy:
            self._set_preview_status(
                reason or "Settings changed — refresh comparison to update it.", _WARN_CSS)

    # -- user intent -----------------------------------------------------------------------------
    def _on_mode_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._refresh_contract_text()
        self.mark_preview_stale("Output choice changed — refresh comparison to update it.")
        self.modeChanged.emit(self.output_mode())

    def _on_runtime_enabled_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self._runtime_enabled = bool(checked)
        self._refresh_contract_text()
        self.mark_preview_stale("Runtime correction changed — refresh comparison to update it.")
        self.runtimeEnabledRequested.emit(self._runtime_enabled)

    def _on_runtime_method_changed(self, _index: int) -> None:
        if self._updating:
            return
        method = str(self.runtime_method.currentData() or "")
        if not method:
            return
        self._runtime_method = method
        self._refresh_contract_text()
        self.mark_preview_stale("Runtime method changed — refresh comparison to update it.")
        self.runtimeMethodRequested.emit(method)

    def _on_refresh_clicked(self) -> None:
        if self._busy:
            self._set_preview_status(
                "A comparison is already running. It will update this preview when it finishes.",
                _BUSY_CSS)
            return
        self.set_busy(True)
        self.refreshRequested.emit()

    def _on_export_clicked(self) -> None:
        if self._busy:
            self._set_preview_status(
                "Wait for the current comparison to finish before exporting it.", _WARN_CSS)
            return
        self.exportRequested.emit()

    # -- rendering helpers ----------------------------------------------------------------------
    def _refresh_contract_text(self) -> None:
        automatic = self.background.source_key() == 'automatic'
        with QtCore.QSignalBlocker(self.output_mode_combo):
            self.output_mode_combo.setItemText(
                0, "Event-reference residual (zero)" if automatic else
                "Raw − background (zero residual)")
            self.output_mode_combo.setItemText(
                1, "Drift-stage correction (offset retained)" if automatic else
                "Raw − background + median(background)")
        self.output_mode_label.setText(
            "Comparison calculation" if automatic else "Output equation")
        uses_reference = (self.background.source_key() == 'intervals' or
                          (automatic and self.output_mode() == DISPLAY_ZERO))
        self.reference_summary_box.setVisible(self._workspace_hosted and uses_reference)
        self.input_state_label.setText(
            'Drift correction is ON for detection and processed export.' if self._runtime_enabled
            else 'Drift correction is OFF. Other enabled signal-chain stages still apply.')
        if not automatic:
            self.contract_label.setText(
                'Custom comparison and export only. Selecting or calculating this recipe '
                'does not change main-trace display, automatic drift, or stored results. '
                'Retained-offset output adds the median background level.')
        elif self.output_mode() == DISPLAY_ZERO:
            self.contract_label.setText(
                "Estimator: automatic event reference, using Event reference Rough/Base "
                "windows. Output: raw − estimated reference. The drift estimator is separate. "
                "Calculating leaves main-trace display, detection and stored results unchanged.")
        else:
            self.contract_label.setText(
                f"Estimator: automatic drift ({self.runtime_method.currentText()}). Uses "
                "Filters & drift parameters, not Event reference windows. Output: raw − "
                "drift baseline + median(baseline). Other signal-chain stages are excluded; "
                "calculation does not enable this stage.")

    def _refresh_freshness(self) -> None:
        """Keep plot validity separate from transient export and progress messages."""
        if self._stale:
            text = "Out of date — update comparison before relying on this plot."
            css = _WARN_CSS + "font-weight:600;"
        elif self._preview_time_s is None:
            text, css = "No comparison calculated for the current settings.", _CAPTION_CSS
        else:
            text, css = "Current comparison · full-recording calculation, reduced preview", _CAPTION_CSS
        self.freshness_label.setText(text)
        self.freshness_label.setStyleSheet(css)
        self.freshness_label.setToolTip(self._stale_reason)
        if self._expanded_freshness is not None:
            self._expanded_freshness.setText(text)
            self._expanded_freshness.setStyleSheet(css)
            self._expanded_freshness.setToolTip(self._stale_reason)

    def _refresh_display_state(self) -> None:
        if self._display_subtract:
            text = "Main trace now: zero-centred residual (visual only)."
        elif self._display_show:
            text = "Main trace now: raw current with its open-pore baseline rail."
        else:
            text = "Main trace now: raw current; no open-pore display overlay is selected."
        self.display_state_label.setText(text)

    def _set_preview_status(self, text: str, css: str) -> None:
        self.preview_status.setText(text)
        self.preview_status.setStyleSheet(css)
        if self._expanded_status is not None:
            self._expanded_status.setText(text)
            self._expanded_status.setStyleSheet(css)

    @staticmethod
    def _as_vector(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return array

    @classmethod
    def _optional_vector(cls, value: Any | None, name: str, n: int) -> np.ndarray | None:
        if value is None:
            return None
        array = cls._as_vector(value, name)
        if array.size != n:
            raise ValueError(f"{name} must have the same length as time_s")
        return array

    @staticmethod
    def _preview_index(n: int) -> slice | np.ndarray:
        if n <= _MAX_PREVIEW_SAMPLES:
            return slice(None)
        # Include both ends so the visible span and baseline alignment stay faithful.
        return np.linspace(0, n - 1, _MAX_PREVIEW_SAMPLES, dtype=np.intp)
