"""Opt-in amplitude curation, separate from the detector's immutable depth definition.

This widget owns no detector and never runs a fit. Its only writes are optional
event-table columns; the Inspector announces those through amplitudeEdited.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis.baseline import BaselineConfig, characterisation_baseline, require_finite
from ..analysis.event_amplitude import (
    METHODS, amplitude_is_current, clear_event_amplitude, measure_event_amplitude,
    write_event_amplitude,
)
from ..analysis.modules import (
    DETREND_METHOD_PARAMS, DETREND_PARAM_KEYS, FILTER_PARAM_KEYS,
    characterization_params, detection_signal, detection_signal_key,
    preprocess_params_from_provenance,
)
from ..analysis.reconcile import CONSENSUS_MODULE_ID
from ..core.sources import signal_domain_of
from .busy_cursor import begin_wait_cursor, end_wait_cursor

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
        row = QtWidgets.QVBoxLayout()
        self.method = QtWidgets.QComboBox()
        for key, label in METHOD_LABELS.items():
            self.method.addItem(label, key)
        self.method.setToolTip(
            "Explicit optional measurement. Apply never changes median depth_pA, "
            "dI/I0, SNR, event boundaries or fitted states.")
        self.method.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.method.setMinimumContentsLength(16)
        row.addWidget(self.method)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_all_button = QtWidgets.QPushButton("Apply to all…")
        self.reset_button = QtWidgets.QPushButton("Reset amplitude")
        self.edit_level = QtWidgets.QCheckBox("Edit red level")
        self.edit_level.setToolTip(
            "Unlock the red line. A drag release saves a manual amplitude for this "
            "event only, separately from median depth. Undo with Edit → Undo.")
        actions = QtWidgets.QHBoxLayout()
        for widget in (self.apply_button, self.reset_button):
            actions.addWidget(widget)
        row.addLayout(actions)
        row.addWidget(self.edit_level)
        row.addWidget(self.apply_all_button)
        layout.addLayout(row)

        self.interval_row = QtWidgets.QWidget()
        interval = QtWidgets.QVBoxLayout(self.interval_row)
        interval.setContentsMargins(0, 0, 0, 0)
        interval.addWidget(QtWidgets.QLabel("Interval samples (end exclusive):"))
        self.start = QtWidgets.QSpinBox()
        self.end = QtWidgets.QSpinBox()
        for spin in (self.start, self.end):
            spin.setRange(0, 2_147_483_647)
            spin.setKeyboardTracking(False)
        self.start.setToolTip("Absolute first sample, inclusive. Drag the purple interval handles.")
        self.end.setToolTip("Absolute stop sample, exclusive. Does not resize the detected event.")
        bounds = QtWidgets.QHBoxLayout()
        bounds.addWidget(self.start)
        bounds.addWidget(QtWidgets.QLabel("to"))
        bounds.addWidget(self.end)
        interval.addLayout(bounds)
        interval.addWidget(QtWidgets.QLabel("Use fitted state:"))
        self.state = QtWidgets.QComboBox()
        self.state.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.state.setMinimumContentsLength(12)
        interval.addWidget(self.state)
        layout.addWidget(self.interval_row)

        self.manual_row = QtWidgets.QWidget()
        manual = QtWidgets.QVBoxLayout(self.manual_row)
        manual.setContentsMargins(0, 0, 0, 0)
        manual.addWidget(QtWidgets.QLabel("Equivalent level:"))
        self.level = QtWidgets.QDoubleSpinBox()
        self.level.setRange(-1e12, 1e12)
        self.level.setDecimals(8)
        self.level.setKeyboardTracking(False)
        manual.addWidget(self.level)
        manual_note = QtWidgets.QLabel("Type a value and Apply, or drag the unlocked red line.")
        manual_note.setWordWrap(True)
        manual.addWidget(manual_note)
        layout.addWidget(self.manual_row)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.note.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.note)
        for control in (self.method, self.start, self.end, self.state, self.level,
                        self.apply_button, self.apply_all_button, self.reset_button):
            control.setMinimumHeight(28)

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
            # Mask resizing and amplitude curation measure the same recording.
            # Reuse its conditioned float32 array instead of running the filter
            # again and retaining a second whole-recording float64 copy.
            arr = (owner._conditioned(raw, fs) if fs is not None
                   else detection_signal(raw, fs, prov))
            self._signal_cache = (key, arr)
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
                # Amplitude remains strict: distant NaNs must not become an
                # implicit repaired reference. Once the input is known finite,
                # use the row-rebuild cache policy so a mask drag reuses this
                # identical whole-recording estimate. No sample is interpolated.
                require_finite(current, stage="measure event amplitude")
                result = characterisation_baseline(
                    current, fs, config=config, polarity=int(pol), health=False,
                    on_nonfinite="interpolate")
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
                # A changed mask can make a saved measurement stale without
                # changing the user's estimator choice.
                method = str(row.get("amplitude_method", "median"))
                self.method.setCurrentIndex(max(0, self.method.findData(method)))
                start, end = int(row["start_idx"]), int(row["end_idx"])
                selection = ((int(row["amplitude_start_idx"]), int(row["amplitude_end_idx"]))
                             if current else self._saved_interval(row)
                             if method in ("selected_median", "selected_mean") else None)
                a, b = selection if selection is not None else (start, end)
                # Keep an out-of-mask saved selection visible for correction.
                # Applying it still fails the estimator's containment check.
                self.start.setRange(min(start, a), max(end - 1, a))
                self.end.setRange(min(start + 1, b), max(end, b))
                self.start.setValue(a)
                self.end.setValue(b)
            self.level.setSuffix(" " + self.owner._units)
            self.level.setValue(float(self.owner._level_line.value()))
            self._populate_states(row)
            self._update_enabled()
            self._update_note(row)
        finally:
            self._updating = False
        self._method_changed()

    @staticmethod
    def _saved_interval(row):
        try:
            metadata = json.loads(row.get("amplitude_provenance_json"))
            interval = metadata.get("interval") if isinstance(metadata, dict) else None
            if (isinstance(interval, list) and len(interval) == 2
                    and all(isinstance(value, int) and not isinstance(value, bool)
                            for value in interval)
                    and 0 <= interval[0] < interval[1] <= np.iinfo(np.int32).max):
                return tuple(interval)
        except (TypeError, ValueError):
            pass
        return None

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
            selection = self._saved_interval(row)
            if (str(row.get("amplitude_method")).startswith("selected_")
                    and selection is not None
                    and not int(row["start_idx"]) <= selection[0] < selection[1] <= int(row["end_idx"])):
                text = (f"Saved amplitude is stale: interval [{selection[0]}, {selection[1]}) "
                        "is outside the event mask. Adjust the amplitude interval and Apply "
                        "again, or Reset amplitude.")
            else:
                text = "Saved amplitude is stale; review this choice and Apply again, or Reset amplitude."
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

    def remeasure_after_bounds_edit(self, pos, previous_row) -> bool:
        """Refresh a saved estimator, exact interval or level within the boundary undo step."""
        method = previous_row.get("amplitude_method")
        if (not isinstance(method, str) or method not in METHODS
                or not amplitude_is_current(previous_row)
                or self.owner._stale_columns
                or self.owner._module_id == CONSENSUS_MODULE_ID):
            return False
        try:
            row = self.owner._df.iloc[pos]
            context = self.context(row)
            identity = dict(context.provenance)
            prior_identity = json.loads(previous_row["amplitude_provenance_json"])["signal_identity"]
            # A fit may be retired by a mask edit. Validate the old signal
            # context, but carry only the exact sample interval into the new
            # measurement, never a claim that the fitted state survived.
            if method.startswith("selected_") and "selected_state" in prior_identity:
                identity["selected_state"] = prior_identity["selected_state"]
            if not amplitude_is_current(previous_row, provenance=identity):
                return False
            selection = ((int(previous_row["amplitude_start_idx"]),
                          int(previous_row["amplitude_end_idx"]))
                         if method.startswith("selected_") else None)
            measured = measure_event_amplitude(
                context.current, context.baseline, row, polarity=context.polarity,
                method=method, units=context.units, selection=selection,
                manual_level=float(previous_row["amplitude_level_pA"]) if method == "manual" else None,
                provenance=context.provenance)
            return write_event_amplitude(self.owner._df, pos, measured)
        except (ValueError, TypeError, OverflowError):
            # The caller has already invalidated the old value. Keep it stale
            # when the new measurement cannot be reconstructed safely.
            return False

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
        # A text status also works on macOS, where the native wait cursor is unsafe.
        self.note.setText(f"Calculating {METHOD_LABELS[method]} for {len(df)} events…")
        self.note.repaint()
        cursor_active = begin_wait_cursor()
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
            end_wait_cursor(cursor_active)
        if changed:
            pos, _ = self._row()
            self._announce(pos, -1)
        else:
            self._update_note(self._row()[1])
