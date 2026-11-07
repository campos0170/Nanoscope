"""Frozen figure details for the Explorer's selected Snapshot or Composition.

The project Figures list is the only inventory. Existing snapshot operations are
shared with the standalone library so their frozen-data and modal safety rules
remain the same when invoked from the Explorer.
"""
from pathlib import Path
import re

from PySide6 import QtCore

from .figure_library import FigureLibrary
from .export_paths import ask_export_path
from .plot_workflow import ExportEntry


class SavedFigureView(FigureLibrary):
    entryCreated = QtCore.Signal(str)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, embedded=True, **kwargs)
        self.setObjectName("savedFigureView")

    def current_id(self):
        entry = self.current_entry()
        return entry.artifact_id if entry is not None else None

    def current_status(self):
        entry = self.current_entry()
        return self._status(entry) if entry is not None else ("Unavailable", "This saved figure is no longer in the project.")

    def exports(self):
        available = lambda: self.current_entry() is not None
        return [ExportEntry("Figure…", self.export_current, kind="figure", enabled=available),
                ExportEntry("Data…", self.export_data, kind="data", enabled=available),
                ExportEntry("Bundle…", lambda: self.export_current(bundle=True), kind="bundle", enabled=available)]

    def export_data(self, path: str | Path | None = None):
        entry = self.current_entry()
        if entry is None:
            return None
        if path is None:
            entry_id, project = entry.artifact_id, self._project_token()
            name = re.sub(r"\s*[·/]\s*", " - ", str(entry.payload.get("name", "Figure")))
            stem = re.sub(r"[^\w .-]", "_", name)[:100]
            path = ask_export_path(self, title="Export saved figure data", suggestion=stem + ".csv",
                                   filters="CSV data (*.csv)", outputs=lambda chosen: [chosen.with_suffix(".csv")])
            if path is None:
                return None
            if self._project_token() != project or self.current_id() != entry_id:
                self.statusBar().showMessage("The selected figure changed; nothing was exported.", 10000)
                return None
        return self.export_current(Path(path).with_suffix(".csv"))

    def duplicate_current(self):
        entry = super().duplicate_current()
        if entry is not None:
            self.entryCreated.emit(entry.artifact_id)
        return entry

    def _save_composition(self, dialog, components, *, lineage=None):
        entry = super()._save_composition(dialog, components, lineage=lineage)
        if entry is not None:
            self.entryCreated.emit(entry.artifact_id)
        return entry
