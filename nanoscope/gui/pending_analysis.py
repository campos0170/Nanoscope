"""Draft controls beside a committed analytical picture, without copying its data."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from functools import wraps

from PySide6 import QtCore, QtWidgets

from .plot_workflow import PendingState


class PendingAnalysis:
    def __init__(self, page, controls, *, has_result, additive=()):
        self.page, self.controls, self.has_result = page, controls, has_result
        #: Controls added after saved drafts existed. A draft is written with their values
        #: beside ``inputs`` (and only when they differ from the applied ones), so a build
        #: without the control still reads the draft, and a draft saved without them reads
        #: as "unchanged" here.
        self.additive = tuple(additive)
        self.depth = 0
        self.applied = self.read()

    @property
    def managed(self):
        return bool(getattr(self.page, '_nanoscope_figure_managed', False))

    def read(self):
        return {key: deepcopy(read()) for key, (_widget, read, _write) in self.controls.items()}

    def state(self):
        if not self.managed:
            return PendingState()
        changed = tuple(key.replace('_', ' ') for key, value in self.read().items()
                        if value != self.applied[key])
        warnings = getattr(self.page, '_pending_analysis_warnings', lambda: ())()
        changed += tuple(warnings)
        if not changed and not self.has_result():
            changed = ('not plotted yet',)
        return PendingState(len(changed), changed)

    def defer(self):
        if not self.managed or self.depth or self.page._updating:
            return False
        self.page.pendingChanged.emit()
        self.page.workspaceChanged.emit()
        return True

    def commit(self):
        self.applied = self.read()
        self.page.pendingChanged.emit()

    def saved_draft(self):
        draft = self.read()
        if not self.managed or draft == self.applied:
            return None
        if not self.additive:
            return {'version': 1, 'inputs': draft}
        saved = {'version': 1, 'inputs': {key: value for key, value in draft.items()
                                          if key not in self.additive}}
        later = {key: draft[key] for key in self.additive if draft[key] != self.applied[key]}
        if later:
            saved['additive_inputs'] = later
        return saved

    def _saved_inputs(self, raw):
        """A saved draft's inputs with every additive control filled in."""
        inputs = raw.get('inputs')
        if not self.additive or not isinstance(inputs, dict):
            return inputs
        later = raw.get('additive_inputs')
        later = later if isinstance(later, dict) else {}
        return {**inputs, **{key: later.get(key, self.applied[key]) for key in self.additive}}

    @contextmanager
    def applying(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def _write(self, values):
        if not isinstance(values, dict) or set(values) != set(self.controls):
            raise ValueError('The pending analytical controls are incomplete or unsupported.')
        for key, value in values.items():
            before = self.applied[key]
            if type(value) is not type(before) and not (
                    type(before) in (int, float) and type(value) in (int, float)):
                raise ValueError(f'Invalid pending analytical value for {key}.')
        # Source controls establish the ranges and comparison choices used by
        # later controls. Serialized mappings may be alphabetically ordered.
        for key in self.controls:
            self.controls[key][2](values[key])

    @contextmanager
    def _blocked(self):
        previous, self.page._updating = self.page._updating, True
        try:
            with ExitStack() as stack:
                for widget in dict.fromkeys(control[0] for control in self.controls.values()):
                    stack.enter_context(QtCore.QSignalBlocker(widget))
                yield
        finally:
            self.page._updating = previous

    @contextmanager
    def committed(self):
        """Read/format/export the shown result while leaving the user's draft intact."""
        if not self.managed or self.depth:
            yield
            return
        draft = self.read()
        if draft == self.applied:
            yield
            return
        # A draft recording can have a shorter duration. Preserve the original
        # spin bounds while temporarily exposing the committed segment values.
        bounds = []
        with self.applying(), self._blocked():
            for key, (widget, _read, _write) in self.controls.items():
                if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    bounds.append((widget, widget.minimum(), widget.maximum()))
                    widget.setRange(min(widget.minimum(), self.applied[key]),
                                    max(widget.maximum(), self.applied[key]))
            try:
                self._write(self.applied)
                yield
            finally:
                self._write(draft)
                for widget, low, high in bounds:
                    widget.setRange(low, high)

    def restore(self, raw):
        self.commit()
        if raw is not None:
            if not isinstance(raw, dict) or raw.get('version') != 1:
                raise ValueError('Unsupported pending analytical controls.')
            with self._blocked():
                try:
                    self._write(self._saved_inputs(raw))
                except (ValueError, TypeError, KeyError):
                    self._write(self.applied)
                    raise
            sync = getattr(self.page, '_sync_pending_analysis_controls', None)
            if callable(sync):
                sync()
        self.page.pendingChanged.emit()


def committed_analysis(method):
    @wraps(method)
    def call(page, *args, **kwargs):
        pending = getattr(page, '_pending_analysis', None)
        if pending is None:
            return method(page, *args, **kwargs)
        with pending.committed():
            return method(page, *args, **kwargs)
    return call


def combo_value(combo, *, data=False):
    read = combo.currentData if data else combo.currentText
    def write(value):
        index = combo.findData(value) if data else combo.findText(value)
        if index < 0:
            raise ValueError(f'The saved analytical choice {value!r} is unavailable.')
        combo.setCurrentIndex(index)
    return combo, read, write


def spin_value(spin):
    return spin, spin.value, spin.setValue


def check_value(check):
    return check, check.isChecked, check.setChecked
