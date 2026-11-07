"""A record-only controller double for shell routing; no analytical surface is created."""
from copy import deepcopy
from types import SimpleNamespace as NS
from uuid import uuid4

from PySide6 import QtCore

from nanoscope.gui.plot_workflow import HistoryState
from nanoscope.core.figure_style import builtin_styles
from nanoscope.core.figure_style_registry import StyleRegistry, resolve_ref


def record(name="Figure", kind="iv"):
    return NS(spec=NS(figure_id=str(uuid4()), kind=kind, name=name,
                      result=NS(status="current")), resources=NS())


class Controller(QtCore.QObject):
    recordsChanged = QtCore.Signal()
    recordChanged = QtCore.Signal(str)
    currentChanged = QtCore.Signal(str)
    requestFailed = QtCore.Signal(str, str)
    selectionChanged = QtCore.Signal()

    def __init__(self, *records):
        super().__init__()
        self._records = {r.spec.figure_id: r for r in records}
        self._current = next(iter(self._records), None)
        self._selected = (self._current,) if self._current else ()
        self._deleted = {}
        self._generation = {key: 0 for key in self._records}
        self._marks = dict(self._generation)
        styles = tuple(builtin_styles())
        self._styles = StyleRegistry(styles=styles, default_ref=styles[0].ref)
        self._epoch = 0
        self.calls = []

    def records(self): return tuple(self._records.values())
    def get(self, key): return self._records.get(key)
    def current_id(self): return self._current
    def selected_ids(self): return self._selected
    def available_styles(self): return self._styles.available()
    def style_registry(self): return self._styles
    def default_style(self): return resolve_ref(self._styles, self._styles.default_ref)
    def is_read_only(self): return False

    def set_selected(self, ids):
        self._selected = tuple(key for key in ids if key in self._records)
        self.selectionChanged.emit()

    def create(self, kind, initial_recipe=None, binding=None):
        self.calls.append(("create", kind, initial_recipe, binding))
        value = record("New figure", kind)
        key = value.spec.figure_id
        self._records[key] = value
        self._generation[key] = self._marks[key] = 0
        self.recordsChanged.emit()
        self.activate(key)
        return key

    def duplicate(self, key):
        value = deepcopy(self._records[key])
        value.spec.figure_id = str(uuid4())
        value.spec.name += " (copy)"
        new = value.spec.figure_id
        self._records[new] = value
        self._generation[new] = self._marks[new] = 0
        self.calls.append(("duplicate", key))
        self.recordsChanged.emit()
        return new

    def rename(self, key, name):
        self.calls.append(("rename", key, name))
        self._records[key].spec.name = name
        self.recordChanged.emit(key)
        return True

    def delete(self, key):
        self.calls.append(("delete", key))
        self._deleted[key] = self._records.pop(key)
        self._selected = tuple(k for k in self._selected if k != key)
        if self._current == key:
            self._current = next(iter(self._records), None)
        self.recordsChanged.emit()
        return True

    def restore_deleted(self, key=None):
        key = key or next(reversed(self._deleted), None)
        if key is None:
            return None
        self._records[key] = self._deleted.pop(key)
        self.recordsChanged.emit()
        return key

    def activate(self, key):
        self.calls.append(("activate", key))
        if key not in self._records:
            return False
        self._current = key
        self.currentChanged.emit(key)
        return True

    def apply_look_to_figure(self, key, look, keep_hand):
        self.calls.append(("style", key, look, keep_hand))
        self._generation[key] += 1
        self._marks[key] += 1
        self.recordChanged.emit(key)
        return True

    def history(self, key):
        return HistoryState(can_undo=self._marks[key] > 0, mark=self._marks[key],
                            undo_text="Undo: style")

    def revision_token(self, key):
        return id(self._records[key]), self._generation[key]

    def undo(self, key):
        self.calls.append(("undo", key))
        self._marks[key] -= 1
        self._generation[key] += 1
        self.recordChanged.emit(key)
        return True
