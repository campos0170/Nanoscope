"""Read-only reference/detection preview; it never applies or stores an analysis."""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


def decimate_reference_trace(
    time_s, current_pA, reference_pA, max_points: int = 20_000,
) -> dict[str, np.ndarray]:
    """Bound plot data while retaining both traces' min/max samples in each bin.

    A shared, sorted sample-index set keeps time/current/reference aligned. The
    first and last samples are retained. This is presentation-only decimation;
    full-resolution detection and measurements must happen before this helper.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (time_s, current_pA, reference_pA)]
    if any(a.ndim != 1 for a in arrays) or len({a.size for a in arrays}) != 1:
        raise ValueError("preview time, current and reference must be equal-length one-dimensional arrays")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("preview trace arrays must contain only finite values")
    if arrays[0].size > 1 and np.any(np.diff(arrays[0]) <= 0):
        raise ValueError("preview time must be strictly increasing")
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 6:
        raise ValueError("max_points must be an integer of at least 6")
    n = int(arrays[0].size)
    if n <= max_points:
        return dict(zip(("time_s", "current_pA", "reference_pA"), (a.copy() for a in arrays)))
    bins = (int(max_points) - 2) // 4
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    selected = {0, n - 1}
    for a, b in zip(edges[:-1], edges[1:]):
        for values in arrays[1:]:
            segment = values[a:b]
            selected.add(int(a + np.argmin(segment)))
            selected.add(int(a + np.argmax(segment)))
    index = np.asarray(sorted(selected), dtype=np.int64)
    return dict(zip(("time_s", "current_pA", "reference_pA"), (a[index] for a in arrays)))


class OpenPorePreviewDialog(QtWidgets.QDialog):
    """Modeless frozen preview with a Close button, not an analysis commit API."""

    def __init__(self, payload: Mapping[str, Any], parent=None):
        super().__init__(parent)
        self.setObjectName("openPoreReferencePreview")
        self.setWindowTitle(str(payload.get("title") or "Open-pore reference preview"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.resize(1060, 700)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(9)
        self.notice_label = QtWidgets.QLabel(
            "Preview only — existing analysis unchanged. Close this window and press Detect to apply the selected reference.")
        self.notice_label.setObjectName("referencePreviewNotice")
        self.notice_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.notice_label.setWordWrap(True)
        font = self.notice_label.font()
        font.setBold(True)
        self.notice_label.setFont(font)
        root.addWidget(self.notice_label)

        trace = decimate_reference_trace(payload["time_s"], payload["current_pA"], payload["reference_pA"])
        starts = np.asarray(payload.get("event_start_s", []), dtype=float)
        ends = np.asarray(payload.get("event_end_s", []), dtype=float)
        levels = np.asarray(payload.get("event_level_pA", []), dtype=float)
        if any(a.ndim != 1 for a in (starts, ends, levels)) or len({starts.size, ends.size, levels.size}) != 1:
            raise ValueError("preview candidate start, end and level arrays must have equal lengths")
        valid = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(levels) & (ends >= starts)
        candidate_count = int(payload.get("event_count", starts.size))
        if candidate_count != starts.size or candidate_count < 0:
            raise ValueError("preview event_count must match the full candidate arrays")
        direction = {-1: "downward", 1: "upward"}.get(payload.get("polarity"), "undetermined")
        noise = payload.get("noise_pA")
        noise_text = f"{float(noise):.4g} pA" if noise is not None and np.isfinite(float(noise)) else "unavailable"
        noun = "candidate event" if candidate_count == 1 else "candidate events"
        summary = f"{candidate_count:,} {noun}  ·  Direction: {direction}  ·  Global residual noise: {noise_text}"
        if np.count_nonzero(valid) != candidate_count:
            missing = candidate_count - int(np.count_nonzero(valid))
            verb = "candidate has" if missing == 1 else "candidates have"
            summary += f"  ·  {missing:,} {verb} no finite display marker"
        self.summary_label = QtWidgets.QLabel(summary)
        self.summary_label.setObjectName("referencePreviewSummary")
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        background = self.palette().color(QtGui.QPalette.ColorRole.Window)
        foreground = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
        dark = background.lightness() < 128
        self.plot = pg.PlotWidget(background=background)
        self.plot.setObjectName("referencePreviewPlot")
        self.plot.setMinimumHeight(260)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=True, y=True, alpha=.12)
        self.plot.setLabel("bottom", "Recording-relative time", units="s")
        self.plot.setLabel("left", "Current", units="pA")
        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            axis.setPen(pg.mkPen(foreground))
            axis.setTextPen(pg.mkPen(foreground))
        self.plot.addLegend(labelTextColor=foreground.name(), offset=(12, 12))
        self.current_curve = self.plot.plot(
            trace["time_s"], trace["current_pA"], name="Conditioned current",
            pen=pg.mkPen("#60a5fa" if dark else "#2367a5", width=1),
        )
        self.reference_curve = self.plot.plot(
            trace["time_s"], trace["reference_pA"], name="Selected open-pore reference",
            pen=pg.mkPen("#e8bd55" if dark else "#a86a05", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.candidate_markers = self.plot.plot(
            (starts[valid] + ends[valid]) / 2, levels[valid], name="Candidate event levels",
            pen=None, symbol="o", symbolSize=5, symbolPen=None,
            symbolBrush=pg.mkBrush("#f27970" if dark else "#c44840"),
        )
        root.addWidget(self.plot, 1)
        caption = QtWidgets.QLabel(
            "Trace display is min/max decimated; candidates are computed at full resolution. Each dot marks an event centre and level, not its duration. Drag to pan; scroll to zoom.")
        caption.setWordWrap(True)
        root.addWidget(caption)

        warnings = list(dict.fromkeys(str(w) for w in payload.get("warnings", []) if w))
        self.details_toggle = QtWidgets.QToolButton()
        self.details_toggle.setObjectName("referencePreviewDetailsToggle")
        self.details_toggle.setText(f"Details and warnings ({len(warnings)})")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.details_toggle.setToolTip("Review assumptions, warnings and reference construction details. These are not drawn on the plot.")
        root.addWidget(self.details_toggle)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setObjectName("referencePreviewDetails")
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(130)
        self.details.setMaximumHeight(210)
        self.details.setPlainText(
            "Warnings and assumptions\n" + ("\n\n".join(f"• {w}" for w in warnings) or "No additional warnings supplied.")
            + "\n\nReference construction\n" + json.dumps(payload.get("reference_details", {}), indent=2, ensure_ascii=False, default=str)
        )
        self.details.hide()
        root.addWidget(self.details)
        self.details_toggle.toggled.connect(self._toggle_details)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

    def _toggle_details(self, expanded: bool) -> None:
        self.details.setVisible(expanded)
        self.details_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)
