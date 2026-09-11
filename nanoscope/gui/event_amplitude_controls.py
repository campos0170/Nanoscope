"""Opt-in amplitude curation, separate from the detector's immutable depth definition.

This widget owns no detector and never runs a fit. Its only writes are optional
event-table columns; the Inspector announces those through amplitudeEdited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis.baseline import BaselineConfig, characterisation_baseline
from ..analysis.event_amplitude import (
    amplitude_is_current, clear_event_amplitude, measure_event_amplitude,
    write_event_amplitude,
)
from ..analysis.modules import (
    DETREND_METHOD_PARAMS, DETREND_PARAM_KEYS, FILTER_PARAM_KEYS,
    characterization_params, detection_signal, detection_signal_key,
    preprocess_params_from_provenance,
)
from ..analysis.reconcile import CONSENSUS_MODULE_ID
from ..core.sources import signal_domain_of

if TYPE_CHECKING:
    from .event_inspector import EventInspectorWindow

METHOD_LABELS = {
    "median": "Median depth (whole event)",
    "mean": "Mean depth (whole event)",
    "peak": "Peak excursion (whole event)",
    "mode": "Histogram mode (whole event)",
    "selected_median": "Median of selected interval / state",
    "selected_mean": "Mean of selected interval / state",
    "manual": "Manual level",
}


@dataclass
class AmplitudeContext:
    current: np.ndarray
    baseline: np.ndarray | float
    polarity: int
    units: str
    provenance: dict


class EventAmplitudeControls(QtWidgets.QWidget):
    """Small explicit Apply/Reset surface plus a drag-to-edit equivalent level."""

    def __init__(self, owner: EventInspectorWindow):
        super().__init__(owner)
        self.owner = owner
        self._updating = False
        self._event_key = None
        self._signal_cache = None
        self._baseline_cache = None
        self._error = ""
        self._selected_state = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Amplitude:"))
        self.method = QtWidgets.QComboBox()
        for key, label in METHOD_LABELS.items():
            self.method.addItem(label, key)
        self.method.setToolTip(
            "Explicit optional measurement. Apply never changes median depth_pA, "
            "dI/I0, SNR, event boundaries or fitted states.")
        row.addWidget(self.method, 1)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_all_button = QtWidgets.QPushButton("Apply to all…")
        self.reset_button = QtWidgets.QPushButton("Reset amplitude")
        self.edit_level = QtWidgets.QCheckBox("Edit red level")
        self.edit_level.setToolTip(
            "Unlock the red line. A drag release saves a manual amplitude for this "
            "event only, separately from median depth. Undo with Edit → Undo.")
        for widget in (self.apply_button, self.apply_all_button, self.reset_button, self.edit_level):
            row.addWidget(widget)
        layout.addLayout(row)

        self.interval_row = QtWidgets.QWidget()
        interval = QtWidgets.QHBoxLayout(self.interval_row)
        interval.setContentsMargins(0, 0, 0, 0)
        interval.addWidget(QtWidgets.QLabel("Interval samples:"))
        self.start = QtWidgets.QSpinBox()
        self.end = QtWidgets.QSpinBox()
        for spin in (self.start, self.end):
            spin.setRange(0, 2_147_483_647)
            spin.setKeyboardTracking(False)
        self.start.setToolTip("Absolute first sample, inclusive. Drag the purple interval handles.")
        self.end.setToolTip("Absolute stop sample, exclusive. Does not resize the detected event.")
        interval.addWidget(self.start)
        interval.addWidget(QtWidgets.QLabel("to (exclusive)"))
        interval.addWidget(self.end)
        interval.addSpacing(10)
        interval.addWidget(QtWidgets.QLabel("Use fitted state:"))
        self.state = QtWidgets.QComboBox()
        interval.addWidget(self.state, 1)
        layout.addWidget(self.interval_row)

        self.manual_row = QtWidgets.QWidget()
        manual = QtWidgets.QHBoxLayout(self.manual_row)
        manual.setContentsMargins(0, 0, 0, 0)
        manual.addWidget(QtWidgets.QLabel("Equivalent level:"))
        self.level = QtWidgets.QDoubleSpinBox()
        self.level.setRange(-1e12, 1e12)
        self.level.setDecimals(8)
        self.level.setKeyboardTracking(False)
        manual.addWidget(self.level)
        manual.addWidget(QtWidgets.QLabel("Type a value and Apply, or drag the unlocked red line."), 1)
        layout.addWidget(self.manual_row)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.note)

        self.method.currentIndexChanged.connect(self._method_changed)
        self.apply_button.clicked.connect(lambda: self.apply_current())
        self.apply_all_button.clicked.connect(self.apply_all)
        self.reset_button.clicked.connect(self.reset_current)
        self.edit_level.toggled.connect(self._toggle_edit)
        self.start.valueChanged.connect(self._interval_changed)
        self.end.valueChanged.connect(self._interval_changed)
        self.state.currentIndexChanged.connect(self._state_selected)

    def install_plot_items(self) -> None:
        self.interval = pg.LinearRegionItem(
            brush=pg.mkBrush(174, 125, 232, 30),
            pen=pg.mkPen("#b282e8", width=1, style=QtCore.Qt.DotLine))
        self.interval.setZValue(-5)
        self.owner._plot.addItem(self.interval, ignoreBounds=True)
        self.interval.sigRegionChangeFinished.connect(self._interval_dragged)
        self.interval.hide()
        self.owner._level_line.sigPositionChangeFinished.connect(self._line_edited)
        self.owner._level_line.sigPositionChanged.connect(self._line_moved)

    def _row(self):
        pos = self.owner._row_for_id(self.owner._current_id)
        if pos is None or self.owner._df is None:
            raise ValueError("No event selected.")
        return pos, self.owner._df.iloc[pos]

    def measurement_signal(self) -> np.ndarray:
        """Replay recorded conditioning; never silently fall back to raw on failure."""
        owner = self.owner
        if owner._doc_id is None or owner._channel is None:
            raise ValueError("No recording is open.")
        raw = np.asarray(owner._session.calibrated(owner._doc_id, owner._channel).data)
        fs = owner._sample_rate()
        prov = owner._provenance or {}
        cutoff = prov.get("filter_cutoff_hz")
        if cutoff is not None and (fs is None or not 0 < float(cutoff) < fs / 2):
            raise ValueError("The saved low-pass configuration cannot be reconstructed.")
        key = (owner._doc_id, owner._channel, id(raw), fs, detection_signal_key(prov))
        if self._signal_cache is None or self._signal_cache[0] != key:
            arr = detection_signal(raw, fs, prov)
            self._signal_cache = (key, np.asarray(arr, dtype=np.float64))
            self._baseline_cache = None
        return self._signal_cache[1]

    def context(self, row) -> AmplitudeContext:
        owner = self.owner
        config = self._reference_config()
        current = self.measurement_signal()
        prov = owner._provenance or {}
        fs = owner._sample_rate()
        start, end = int(row["start_idx"]), int(row["end_idx"])
        if not 0 <= start < end <= len(current):
            raise ValueError("Event boundaries are outside the recording.")
        base_reference = float(row["baseline_pA"])
        pol = prov.get("polarity")
        if pol is None:
            pol = int(np.sign(np.median(current[start:end]) - base_reference))
        if pol not in (-1, 1):
            raise ValueError("A valid event polarity is required.")
        baseline_source = "stored_event_reference"
        baseline = base_reference
        if config is not None and fs is not None:
            key = (id(current), fs, int(pol), config.fingerprint())
            if self._baseline_cache is None or self._baseline_cache[0] != key:
                result = characterisation_baseline(
                    current, fs, config=config, polarity=int(pol), health=False)
                self._baseline_cache = (key, np.asarray(result.baseline, dtype=np.float64))
            baseline = self._baseline_cache[1]
            baseline_source = "recording"
        units = owner._units
        translated = preprocess_params_from_provenance(prov)
        processing_keys = set(DETREND_PARAM_KEYS) | set(FILTER_PARAM_KEYS) | {"polarity"}
        for entries in DETREND_METHOD_PARAMS.values():
            processing_keys.update(key for key, _kernel, _default in entries)
        processing = {key: translated[key] for key in processing_keys if key in translated}
        if prov.get("highpass_localization_only") or prov.get("characterization_hp_omitted"):
            processing = characterization_params(processing)
        for key in ("filter_cutoff_hz", "filter_kind", "filter_order"):
            processing[key] = prov.get(key)
        audit = {
            "signal_domain": signal_domain_of(prov),
            "detection_signal_key": list(detection_signal_key(prov)),
            "baseline_source": baseline_source,
            "baseline_fingerprint": str(prov.get("baseline_fingerprint", "unknown")),
            "baseline_config": config.to_provenance(fs) if config is not None else None,
            "processing": processing,
            "sample_rate_hz": fs,
            "calibration_fingerprint": list(owner._session.calibration(
                owner._doc_id, owner._channel).fingerprint()),
        }
        return AmplitudeContext(current, baseline, int(pol), units, audit)

    def _reference_config(self):
        """Resolve saved reference instructions without guessing a replacement.

        Context callers report failures through the Inspector's existing render
        guard. Note-only refreshes must handle the same refusal before enabling
        any manual amplitude write.
        """
        prov = self.owner._provenance or {}
        config = BaselineConfig.reproducible_from(prov)
        if config is not None and config.open_pore_reference is not None:
            from ..analysis.reference_context import validate_open_pore_context
            validate_open_pore_context(
                {**prov, "baseline_config": config},
                units=self.owner._units,
                module_id=self.owner._module_id,
            )
            if str(prov.get("units", self.owner._units)).strip() != "pA":
                raise ValueError(
                    "The saved open-pore reference is not in pA. Verify its "
                    "calibration and units before editing event amplitudes."
                )
        return config

    def _allowed(self) -> bool:
        return (self.owner._df is not None and self.owner._current_id is not None
                and self.owner._module_id != CONSENSUS_MODULE_ID
                and not self.owner._stale_columns and not self._error)

    def sync(self, row=None, *, force=False) -> None:
        self._updating = True
        try:
            if row is None:
                self.hide()
                self.interval.hide()
                self.owner._level_line.setMovable(False)
                self._event_key = None
                return
            self.show()
            key = (self.owner._doc_id, self.owner._channel, int(row["event_id"]),
                   int(row["start_idx"]), int(row["end_idx"]))
            changed_event = key != self._event_key
            if changed_event or force:
                self._event_key = key
                self._selected_state = None
                if changed_event:
                    self.edit_level.setChecked(False)
                current = amplitude_is_current(row)
                method = str(row.get("amplitude_method", "median")) if current else "median"
                self.method.setCurrentIndex(max(0, self.method.findData(method)))
                start, end = int(row["start_idx"]), int(row["end_idx"])
                self.start.setRange(start, end - 1)
                self.end.setRange(start + 1, end)
                self.start.setValue(int(row["amplitude_start_idx"]) if current else start)
                self.end.setValue(int(row["amplitude_end_idx"]) if current else end)
            self.level.setSuffix(" " + self.owner._units)
            self.level.setValue(float(self.owner._level_line.value()))
            self._populate_states(row)
            self._update_enabled()
            self._update_note(row)
        finally:
            self._updating = False
        self._method_changed()

    def _populate_states(self, row):
        old = self.state.currentData()
        self.state.clear()
        self.state.addItem("Choose an existing fitted state…", None)
        levels = self.owner._levels
        if levels is not None:
            for lv in levels[levels["event_id"] == int(row["event_id"])].itertuples():
                self.state.addItem(f"L{int(lv.level_idx)} · samples {int(lv.start_idx)}–{int(lv.end_idx)}",
                                   (int(lv.start_idx), int(lv.end_idx), int(lv.level_idx)))
        if old is not None:
            self.state.setCurrentIndex(max(0, self.state.findData(old)))

    def _update_enabled(self):
        allowed = self._allowed()
        for widget in (self.method, self.apply_button, self.edit_level,
                       self.start, self.end, self.state, self.level):
            widget.setEnabled(allowed)
        self.owner._level_line.setMovable(allowed and self.edit_level.isChecked())
        self.apply_all_button.setEnabled(allowed and self.method.currentData() in
                                        ("median", "mean", "peak", "mode"))
        try:
            _, row = self._row()
            has_saved = pd.notna(row.get("amplitude_method"))
        except ValueError:
            has_saved = False
        # Clearing a stale optional measurement remains possible, and undoable.
        self.reset_button.setEnabled(bool(has_saved) and self.owner._module_id != CONSENSUS_MODULE_ID)

    def _update_note(self, row):
        config = None
        if not self._error and not self.owner._stale_columns:
            try:
                config = self._reference_config()
            except (ValueError, TypeError, OverflowError) as exc:
                self._error = str(exc)
                self._update_enabled()
        if self._error:
            text = "Amplitude unavailable: " + self._error
        elif self.owner._stale_columns:
            text = "Measurements are stale. Re-measure the events before editing their amplitude."
        elif self.owner._module_id == CONSENSUS_MODULE_ID:
            text = "Reconciled results are read-only for amplitude; curate a source detector table."
        elif amplitude_is_current(row):
            text = (f"Selected amplitude: {float(row['amplitude_pA']):.6g} {row['amplitude_units']} "
                    f"({METHOD_LABELS.get(str(row['amplitude_method']), row['amplitude_method'])}). "
                    "Original median depth, dI/I0 and SNR are unchanged.")
        elif pd.notna(row.get("amplitude_method")):
            text = "Saved amplitude is stale; choose a method and Apply again, or Reset amplitude."
        else:
            text = "No override. Red line = stored median-depth equivalent level; Apply saves a separate amplitude."
        if not self._error and not self.owner._stale_columns:
            text += " A horizontal level uses the interval's mean baseline; it is not a fitted plateau."
            if config is None:
                text += " Legacy reference: the saved event baseline is used as a constant."
        self.note.setText(text)

    def _method_changed(self):
        if self._updating:
            return
        method = self.method.currentData()
        if method != "manual" and self.edit_level.isChecked():
            self.edit_level.setChecked(False)
        selected = method in ("selected_median", "selected_mean")
        self.interval_row.setVisible(selected)
        self.manual_row.setVisible(method == "manual" or self.edit_level.isChecked())
        self.interval.setVisible(selected and self.isVisible())
        self._interval_changed()
        self._update_enabled()

    def _toggle_edit(self, checked):
        if self._updating:
            return
        if checked:
            self.method.setCurrentIndex(self.method.findData("manual"))
        self._method_changed()

    def _interval_changed(self):
        if self._updating or self.owner._doc_id is None:
            return
        # Hand-adjusted bounds are no longer exactly the chosen fitted state.
        self._selected_state = None
        xs = self.owner._session.x_axis(self.owner._doc_id).values
        if not len(xs):
            return
        a, b = self.start.value(), self.end.value()
        self._updating = True
        try:
            self.interval.setRegion((float(xs[min(a, len(xs) - 1)]),
                                     self._x_at_stop(b, xs)))
        finally:
            self._updating = False

    def _x_at_stop(self, stop, xs):
        if stop < len(xs):
            return float(xs[stop])
        step = float(xs[-1] - xs[-2]) if len(xs) > 1 else 1.0
        return float(xs[-1]) + step

    def _interval_dragged(self):
        if self._updating or self.owner._doc_id is None:
            return
        xs = self.owner._session.x_axis(self.owner._doc_id).values
        a, b = self.interval.getRegion()
        self.start.setValue(int(np.searchsorted(xs, a)))
        self.end.setValue(int(np.searchsorted(xs, b)))
        self._selected_state = None
        self._interval_changed()

    def _state_selected(self):
        if self._updating or self.state.currentData() is None:
            return
        a, b, state = self.state.currentData()
        self.start.setValue(a)
        self.end.setValue(b)
        self._selected_state = state

    def _line_moved(self):
        if not self._updating and self.edit_level.isChecked():
            self.level.setValue(float(self.owner._level_line.value()))

    def _line_edited(self):
        if not self._updating and self.edit_level.isChecked() and self._allowed():
            self.apply_current(method="manual", manual_level=float(self.owner._level_line.value()))

    def _measurement(self, row, method, manual_level=None):
        context = self.context(row)
        selection = ((self.start.value(), self.end.value())
                     if method in ("selected_median", "selected_mean") else None)
        audit = dict(context.provenance)
        if selection is not None and self._selected_state is not None:
            audit["selected_state"] = self._selected_state
        return measure_event_amplitude(
            context.current, context.baseline, row, polarity=context.polarity,
            method=method, units=context.units, selection=selection,
            manual_level=self.level.value() if method == "manual" and manual_level is None else manual_level,
            provenance=audit)

    def apply_current(self, *, method=None, manual_level=None):
        if not self._allowed():
            return False
        pos, row = self._row()
        try:
            measured = self._measurement(row, method or self.method.currentData(), manual_level)
            changed = write_event_amplitude(self.owner._df, pos, measured)
        except (ValueError, TypeError, OverflowError) as exc:
            self.note.setText(f"Amplitude not changed: {exc}")
            self.owner._render_event(pos, preserve_view=True)
            self.note.setText(f"Amplitude not changed: {exc}")
            return False
        if changed:
            self._announce(pos, int(row["event_id"]))
        return changed

    def _announce(self, pos, event_id):
        self.owner._render_event(pos, preserve_view=True)
        self.owner._refresh_readout(pos)
        self.sync(self.owner._df.iloc[pos], force=True)
        self.owner.amplitudeEdited.emit(event_id)

    def reset_current(self):
        pos, row = self._row()
        if self.owner._module_id == CONSENSUS_MODULE_ID:
            return False
        if clear_event_amplitude(self.owner._df, pos):
            self._announce(pos, int(row["event_id"]))
            return True
        return False

    def apply_all(self):
        method = self.method.currentData()
        if not self._allowed() or method not in ("median", "mean", "peak", "mode"):
            return
        df = self.owner._df
        answer = QtWidgets.QMessageBox.question(
            self, "Apply amplitude method to every event?",
            f"Apply {METHOD_LABELS[method]} to all {len(df)} events, including rejected rows?\n\n"
            "This replaces optional amplitude choices, including manual edits. Original median "
            "depth and all detector results stay unchanged. One Undo restores the prior choices.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        # Compute and validate everything before touching the shared frame.
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            measurements = [self._measurement(row, method) for _, row in df.iterrows()]
            staged = df.copy(deep=True)
            changed = False
            for pos, measured in enumerate(measurements):
                changed = write_event_amplitude(staged, pos, measured) or changed
            if changed:
                from ..analysis.event_amplitude import AMPLITUDE_COLUMNS
                for column in AMPLITUDE_COLUMNS:
                    df[column] = staged[column]
        except (ValueError, TypeError, OverflowError) as exc:
            self.note.setText(f"No amplitudes changed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if changed:
            pos, _ = self._row()
            self._announce(pos, -1)
