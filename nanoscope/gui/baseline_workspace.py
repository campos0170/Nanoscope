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
from PySide6 import QtCore, QtWidgets

from ..plot.style import BASELINE_TRACK_INK, PALETTE

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


def _scroll_button_row(layout: QtWidgets.QHBoxLayout) -> QtWidgets.QScrollArea:
    """Keep a wide action row from imposing a permanent dock-width floor."""
    host = QtWidgets.QWidget()
    host.setLayout(layout)
    area = QtWidgets.QScrollArea()
    area.setWidget(host)
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                       QtWidgets.QSizePolicy.Policy.Fixed)
    area.setFixedHeight(host.sizeHint().height() + 4)
    return area


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
        self._display_show = False
        self._display_subtract = False
        self._runtime_enabled = False
        self._runtime_method = ""
        self._preview_time_s: np.ndarray | None = None
        self._preview_raw: np.ndarray | None = None
        self._preview_baseline: np.ndarray | None = None
        self._preview_corrected: np.ndarray | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # The existing form stays the owner of the open-pore estimator and its
        # proposal/confirmation workflow.  Keep it first: its two trace-display
        # checkboxes are the direct, familiar controls and must remain reachable
        # without scrolling through the new comparison workspace in the Inspector.
        # QLayout handles reparenting here.
        root.addWidget(self.baseline_panel)
        root.addWidget(self._build_contract_box(method_choices))
        root.addWidget(self._build_actions())
        root.addWidget(self._build_preview_box())

        self.output_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.runtime_enabled.toggled.connect(self._on_runtime_enabled_toggled)
        self.runtime_method.currentIndexChanged.connect(self._on_runtime_method_changed)
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        self.export_button.clicked.connect(self._on_export_clicked)
        self.advanced_button.clicked.connect(self.advancedRequested.emit)
        self._refresh_contract_text()
        self._refresh_display_state()

    # -- construction ---------------------------------------------------------------------------
    def _build_contract_box(
        self, method_choices: Iterable[tuple[str, str]]
    ) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Baseline correction")
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
        layout.addWidget(QtWidgets.QLabel("Comparison output"))
        self.output_mode_combo = QtWidgets.QComboBox()
        self.output_mode_combo.addItem("Inspect at zero — visual only", DISPLAY_ZERO)
        self.output_mode_combo.addItem("Correct drift — retain open-pore offset",
                                      RUNTIME_RETAIN_OFFSET)
        self.output_mode_combo.setToolTip(
            "Choose what the comparison's lower panel means. Inspect at zero is a "
            "display residual only. Correct drift keeps a representative open-pore "
            "offset and is the contract used by the signal-chain operation.")
        layout.addWidget(self.output_mode_combo)

        self.contract_label = _caption()
        self.contract_label.setObjectName("baselineCorrectionContract")
        layout.addWidget(self.contract_label)

        self.display_state_label = _caption()
        self.display_state_label.setObjectName("baselineDisplayState")
        self.display_state_label.setToolTip(
            "The state of the main trace, separate because trace display does not "
            "change detector input or processed-trace export.")
        layout.addWidget(self.display_state_label)

        self.runtime_enabled = QtWidgets.QCheckBox("Enable runtime correction")
        self.runtime_enabled.setToolTip(
            "Apply signal-chain baseline-drift correction before detection and when "
            "exporting the processed trace. It retains a representative open-pore "
            "offset; it does not use the zero-centred display residual.")
        layout.addWidget(self.runtime_enabled)

        layout.addWidget(QtWidgets.QLabel("Runtime method"))
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
            "When enabled, detection and processed export use the retained-offset "
            "drift correction. It is separate from the open-pore reference below; "
            "the ordinary Median filter is a despiker, not a baseline correction.")
        layout.addWidget(self.runtime_note)
        return box

    def _build_preview_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Comparison preview")
        box.setToolTip(
            "Top: raw current with its fitted baseline. Bottom: the selected output. "
            "The preview is decimated for responsiveness; export writes full arrays.")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

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
            plot.getAxis("left").enableAutoSIPrefix(False)
        self.raw_plot.setLabel("left", "Current")
        self.corrected_plot.setLabel("left", "Corrected current")
        self.corrected_plot.setLabel("bottom", "Time", units="s")
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
        return box

    def _build_actions(self) -> QtWidgets.QScrollArea:
        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self.refresh_button = QtWidgets.QPushButton("Refresh comparison")
        self.refresh_button.setToolTip(
            "Calculate a fresh raw / fitted-baseline / corrected comparison in the "
            "background for the selected recording.")
        self.export_button = QtWidgets.QPushButton("Export raw, baseline & corrected…")
        self.export_button.setToolTip(
            "Export full-resolution time, raw current, fitted baseline and corrected "
            "current together. This never exports the decimated preview.")
        self.advanced_button = QtWidgets.QPushButton("Advanced signal-chain settings")
        self.advanced_button.setToolTip(
            "Open the baseline-drift page in Signal chain for runtime windows and "
            "method-specific controls.")
        for button in (self.refresh_button, self.export_button, self.advanced_button):
            actions.addWidget(button)
        actions.addStretch(1)
        return _scroll_button_row(actions)

    # -- state exposed to the owner -------------------------------------------------------------
    def output_mode(self) -> str:
        """Return the stable selected output token, never the visible label."""
        return str(self.output_mode_combo.currentData() or DISPLAY_ZERO)

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
            lower_title = "Zero-centred residual — display only"
            lower_label = "Δ vs baseline"
            default_message = (
                "Preview is a zero-centred residual. It is visual only: detection "
                "and processed-trace export still use the raw/runtime signal.")
        else:
            lower_title = "Drift-corrected signal — retained open-pore offset"
            lower_label = "Corrected current"
            default_message = (
                "Preview uses the runtime drift-correction contract and retains a "
                "representative open-pore offset for detection and processed export."
                if self._runtime_enabled else
                "This is a runtime-style preview only. Enable drift correction above "
                "before detection or processed export will use it.")
        self.corrected_plot.setTitle(lower_title)
        self.corrected_plot.setLabel("left", lower_label, units=str(units))
        self.raw_plot.setLabel("left", "Current", units=str(units))

        self._stale = False
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
        self.set_busy(False)
        self._set_preview_status(message, _CAPTION_CSS)

    def set_busy(self, busy: bool, message: str | None = None) -> None:
        """Show an in-flight job without disabling controls that should explain it."""
        self._busy = bool(busy)
        self.refresh_button.setText(
            "Refreshing comparison…" if self._busy else "Refresh comparison")
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
        if self.output_mode() == DISPLAY_ZERO:
            self.contract_label.setText(
                "Inspect at zero subtracts a fitted open-pore reference for a clearer "
                "view. It is display-only: no detection, event depth, or processed export changes.")
        elif self._runtime_enabled:
            self.contract_label.setText(
                "Runtime drift correction is enabled. Detection and processed export use "
                "the signal-chain output, which retains a representative open-pore offset.")
        else:
            self.contract_label.setText(
                "Runtime-style comparison selected. Turn on drift correction to use this "
                "output for detection and processed export.")

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
