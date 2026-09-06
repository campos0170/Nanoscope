"""The project Figure Library: view/export frozen figures without opening analyses."""
from __future__ import annotations

from copy import deepcopy
import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.figure_library import duplicate_figure, rename_figure
from ..core.project import WorkspaceArtifact
from .export_paths import ask_export_path


def _export_path(entry: WorkspaceArtifact | None) -> Path | None:
    raw = None if entry is None else entry.payload.get("export_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        path = Path(raw)
        return path if path.is_absolute() and path.is_file() else None
    except (OSError, ValueError):
        return None


def _preview_png(entry: WorkspaceArtifact) -> bytes:
    """A corrupt optional preview must not block access to retained numerical data."""
    png = entry.blobs.get("png", b"")
    metadata = entry.payload.get("snapshot")
    stamps = metadata.get("sha256") if isinstance(metadata, dict) else None
    expected = stamps.get("png") if isinstance(stamps, dict) else None
    if not isinstance(png, bytes) or expected != hashlib.sha256(png).hexdigest():
        return b""
    return png


def _scaled_image(png: bytes, size: QtCore.QSize) -> QtGui.QImage:
    """Decode catalogue icons at thumbnail size, not one full canvas per figure."""
    buffer = QtCore.QBuffer()
    buffer.setData(png)
    buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
    reader = QtGui.QImageReader(buffer, b"PNG")
    native = reader.size()
    if (not native.isValid() or native.width() * native.height() > 100_000_000
            or max(native.width(), native.height()) > 32768):
        return QtGui.QImage()
    reader.setScaledSize(native.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
    return reader.read()


class FigureLibrary(QtWidgets.QMainWindow):
    """Presentation-only catalogue; the owner retains all project resources."""

    changed = QtCore.Signal()

    def __init__(self, parent=None, *, entries: Callable,
                 status: Callable, append: Callable, project_token: Callable | None = None):
        super().__init__(parent)
        self._entries = entries
        self._status = status
        self._append = append
        self._project_token = project_token or (lambda: None)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self.refresh)
        self.setWindowTitle("NanoScope — Figure Library")
        self.resize(1120, 740)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        intro = QtWidgets.QLabel(
            "Figure Library · saved with this project, separate from the Event Gallery. "
            "Figures are frozen: opening or exporting one never runs analysis.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        actions = QtWidgets.QHBoxLayout()
        self._rename_btn = QtWidgets.QPushButton("Rename…")
        self._duplicate_btn = QtWidgets.QPushButton("Duplicate")
        self._export_btn = QtWidgets.QPushButton("Export saved figure…")
        self._bundle_btn = QtWidgets.QPushButton("Save figure bundle…")
        self._reveal_btn = QtWidgets.QPushButton("Reveal file")
        self._refresh_btn = QtWidgets.QPushButton("Check status")
        for button in (self._rename_btn, self._duplicate_btn, self._export_btn,
                       self._bundle_btn, self._reveal_btn, self._refresh_btn):
            actions.addWidget(button)
        actions.addStretch(1)
        root.addLayout(actions)
        composition_row = QtWidgets.QHBoxLayout()
        self._compose_btn = QtWidgets.QPushButton("Compose multi-panel figure…")
        self._compose_btn.setToolTip(
            "Arrange saved figures with panel labels and optional compatible shared axes. "
            "Saves a new snapshot without changing the originals.")
        self._compose_btn.clicked.connect(lambda: self.compose_current())
        composition_row.addWidget(self._compose_btn)
        self._revise_btn = QtWidgets.QPushButton("Revise composition…")
        self._revise_btn.setToolTip(
            "Reopen embedded original panels, revise their layout and save a new figure. "
            "The selected composition and live analyses remain unchanged.")
        self._revise_btn.clicked.connect(lambda: self.revise_current())
        composition_row.addWidget(self._revise_btn)
        composition_row.addStretch(1)
        root.addLayout(composition_row)
        self._rename_btn.clicked.connect(self.rename_current)
        self._duplicate_btn.clicked.connect(self.duplicate_current)
        self._export_btn.clicked.connect(lambda: self.export_current())
        self._bundle_btn.clicked.connect(lambda: self.export_current(bundle=True))
        self._reveal_btn.clicked.connect(self.reveal_current)
        self._refresh_btn.clicked.connect(self.refresh)

        split = QtWidgets.QSplitter()
        self._list = QtWidgets.QListWidget()
        self._list.setObjectName("figureLibraryList")
        self._list.setIconSize(QtCore.QSize(170, 110))
        self._list.setMinimumWidth(275)
        self._list.setWordWrap(True)
        self._list.currentItemChanged.connect(self._show_current)
        split.addWidget(self._list)
        right = QtWidgets.QWidget()
        details = QtWidgets.QVBoxLayout(right)
        self._summary = QtWidgets.QLabel()
        self._summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._summary.setWordWrap(True)
        details.addWidget(self._summary)
        self._preview = QtWidgets.QLabel("Save a figure from the Analysis Explorer to begin.")
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(280, 220)
        self._preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                    QtWidgets.QSizePolicy.Policy.Expanding)
        self._preview.setStyleSheet("background:#24272b; border:1px solid #666;")
        details.addWidget(self._preview, 1)
        self._metadata = QtWidgets.QPlainTextEdit()
        self._metadata.setReadOnly(True)
        self._metadata.setMaximumHeight(180)
        self._metadata_toggle = QtWidgets.QToolButton()
        self._metadata_toggle.setText("Analysis settings and provenance")
        self._metadata_toggle.setCheckable(True)
        self._metadata_toggle.toggled.connect(self._metadata.setVisible)
        details.addWidget(self._metadata_toggle)
        details.addWidget(self._metadata)
        self._metadata.hide()
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 760])
        root.addWidget(split, 1)
        self.setCentralWidget(central)
        self._pixmap = QtGui.QPixmap()
        self.refresh()

    def current_entry(self) -> WorkspaceArtifact | None:
        selected = self._list.currentItem()
        key = selected.data(QtCore.Qt.ItemDataRole.UserRole) if selected else None
        return self._entry_by_id(key)

    def _entry_by_id(self, key) -> WorkspaceArtifact | None:
        return next((entry for entry in self._entries() if entry.artifact_id == key), None)

    def refresh(self, select_id: str | None = None) -> None:
        old = self.current_entry()
        selected = select_id or (old.artifact_id if old is not None else None)
        self._list.clear()
        for entry in self._entries():
            status, reason = self._status(entry)
            item = QtWidgets.QListWidgetItem(
                f"{entry.payload.get('name', 'Figure')}\n{status}\n"
                f"{entry.payload.get('source_label', '')}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry.artifact_id)
            item.setToolTip(reason)
            preview = QtGui.QPixmap.fromImage(_scaled_image(_preview_png(entry), QtCore.QSize(340, 220)))
            if not preview.isNull():
                item.setIcon(QtGui.QIcon(preview))
            item.setSizeHint(QtCore.QSize(290, 122))
            self._list.addItem(item)
            if entry.artifact_id == selected:
                self._list.setCurrentItem(item)
        if self._list.currentRow() < 0 and self._list.count():
            self._list.setCurrentRow(0)
        self._show_current()

    def _show_current(self, *_args) -> None:
        entry = self.current_entry()
        self._compose_btn.setEnabled(len(self._entries()) >= 2)
        self._revise_btn.setEnabled(entry is not None and entry.payload.get("page_id") == "composition")
        for button in (self._rename_btn, self._duplicate_btn, self._export_btn,
                       self._bundle_btn):
            button.setEnabled(entry is not None)
        export_path = _export_path(entry)
        self._reveal_btn.setEnabled(export_path is not None)
        self._reveal_btn.setToolTip(str(export_path or "Export this saved figure first."))
        self._pixmap = QtGui.QPixmap()
        if entry is None:
            self._summary.setText("No saved figures in this project.")
            self._metadata.clear()
            self._preview.setText("Save a figure from the Analysis Explorer to begin.")
            return
        status, reason = self._status(entry)
        payload = entry.payload
        self._summary.setText(
            f"{payload.get('name', 'Figure')} · {status}\n"
            f"{payload.get('source_label', '')}\n"
            f"Saved {payload.get('created_at', '')}\n{reason}")
        self._pixmap = QtGui.QPixmap.fromImage(
            _scaled_image(_preview_png(entry), QtCore.QSize(1600, 1200)))
        self._render_preview()
        self._metadata.setPlainText(json.dumps({
            "analysis": payload.get("analysis"),
            "source_recipes": payload.get("source_recipes"),
            "snapshot": payload.get("snapshot"),
        }, indent=2, ensure_ascii=False))

    def schedule_refresh(self) -> None:
        if self.isVisible():
            self._refresh_timer.start()

    def _render_preview(self) -> None:
        if self._pixmap.isNull():
            self._preview.setText("Saved preview is unavailable; stored data are retained.")
        else:
            self._preview.setPixmap(self._pixmap.scaled(
                self._preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_pixmap"):
            self._render_preview()

    def showEvent(self, event):  # noqa: N802
        self.refresh()
        super().showEvent(event)

    def rename_current(self, name: str | None = None) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        entry_id = entry.artifact_id
        if not isinstance(name, str):
            name, accepted = QtWidgets.QInputDialog.getText(
                self, "Rename figure", "Figure name:", text=entry.payload.get("name", ""))
            if not accepted:
                return
        # A worker may checkpoint (cloning all artifacts) inside the modal loop.
        # Mutate the current project's artifact, never the detached pre-dialog copy.
        entry = self._entry_by_id(entry_id)
        if entry is None:
            self.statusBar().showMessage("The figure is no longer in the current project.", 10000)
            return
        try:
            rename_figure(entry, name)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Rename figure", str(exc))
            return
        self.changed.emit()
        self.refresh(entry.artifact_id)

    def duplicate_current(self) -> WorkspaceArtifact | None:
        entry = self.current_entry()
        if entry is None:
            return None
        duplicate = duplicate_figure(entry)
        self._append(duplicate)
        self.changed.emit()
        self.refresh(duplicate.artifact_id)
        return duplicate

    def compose_current(self) -> WorkspaceArtifact | None:
        from .figure_composer import FigureComposerDialog

        entries = list(self._entries())
        if len(entries) < 2:
            return None
        project = self._project_token()
        originals = {entry.artifact_id: (dict(entry.blobs), deepcopy(entry.payload.get("snapshot")))
                     for entry in entries}
        current = self.current_entry()
        dialog = FigureComposerDialog(
            entries, self, current_id=None if current is None else current.artifact_id)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        # A project switch/checkpoint can happen in the modal event loop. The
        # lifecycle token catches even an identical project reopened; frozen stamps
        # allow ordinary checkpoints without trusting mutable artifact references.
        components = [self._entry_by_id(key) for key in dialog.panel_ids]
        if (self._project_token() != project or not components or any(entry is None for entry in components)
                or any(entry.artifact_id not in originals
                       or entry.blobs != originals[entry.artifact_id][0]
                       or entry.payload.get("snapshot") != originals[entry.artifact_id][1]
                       for entry in components if entry is not None)):
            self.statusBar().showMessage(
                "The source figures changed or are no longer in the current project; nothing was saved.", 10000)
            return None
        return self._save_composition(dialog, components)

    def revise_current(self) -> WorkspaceArtifact | None:
        """Revise embedded originals as a new figure, even without separate source entries."""
        from ..core.figure_library import FIGURE_KIND, source_recipes
        from .figure_bundle import FigureSnapshot
        from .figure_composer import FigureComposerDialog
        from .figure_composition_revision import extract_composition

        entry = self.current_entry()
        if entry is None or entry.payload.get("page_id") != "composition":
            return None
        project, entry_id = self._project_token(), entry.artifact_id
        original_blobs = dict(entry.blobs)
        original_metadata = deepcopy(entry.payload.get("snapshot"))
        try:
            snapshot = FigureSnapshot.from_storage(original_metadata, original_blobs)
            panels, layout = extract_composition(snapshot)
            analysis = entry.payload.get("analysis", {}).get("payload", {})
            names = analysis.get("component_names", [])
            origin_ids = analysis.get("original_component_ids", analysis.get("component_ids", []))
            embedded = []
            for index, panel in enumerate(panels):
                metadata, blobs = panel.snapshot.to_storage()
                name = (names[index] if isinstance(names, list) and index < len(names)
                        and isinstance(names[index], str) else f"Embedded panel {index + 1}")
                origin_id = (origin_ids[index] if isinstance(origin_ids, list) and index < len(origin_ids)
                             and isinstance(origin_ids[index], str) else None)
                provenance = panel.snapshot.provenance()
                recipes = source_recipes(WorkspaceArtifact(
                    artifact_id="embedded", kind="frozen", payload={"provenance": provenance}))
                embedded.append(WorkspaceArtifact(
                    artifact_id=panel.panel_id, kind=FIGURE_KIND, schema_version=1,
                    payload={"name": name, "source_label": "Embedded original · " + str(provenance.get("source", "saved panel")),
                             "origin_figure_id": origin_id, "source_recipes": recipes,
                             "snapshot": metadata}, blobs=blobs))
            name = str(entry.payload.get("name", "Composition"))[:189] + " (revision)"
            dialog = FigureComposerDialog(embedded, self, initial_layout=layout, name=name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Revise composition", str(exc))
            return None
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        current = self._entry_by_id(entry_id)
        if (self._project_token() != project or current is None
                or current.blobs != original_blobs or current.payload.get("snapshot") != original_metadata):
            self.statusBar().showMessage(
                "The project or original composition changed while editing; nothing was saved.", 10000)
            return None
        by_id = {panel.artifact_id: panel for panel in embedded}
        if any(key not in by_id for key in dialog.panel_ids):
            return None
        components = [by_id[key] for key in dialog.panel_ids]
        lineage = {"revision_of": entry_id,
                   "revision_source_sha256": deepcopy(original_metadata["sha256"]),
                   "original_component_ids": [panel.payload.get("origin_figure_id") for panel in components],
                   "revision_contract": "Layout revision uses embedded original snapshots; parent and live analyses are unchanged."}
        return self._save_composition(dialog, components, lineage=lineage)

    def _save_composition(self, dialog, components, *, lineage=None) -> WorkspaceArtifact | None:
        from dataclasses import replace
        from ..core.figure_library import make_figure

        snapshot = dialog.snapshot
        if snapshot is None or not components:
            return None
        try:
            if lineage:
                # Keep revision lineage in portable bundles too, not just the
                # Library entry. This changes metadata only, never the approved image.
                snapshot = replace(snapshot,
                    config_json_bytes=json.dumps({**snapshot.config(), "revision": lineage},
                        sort_keys=True, allow_nan=False, ensure_ascii=False).encode("utf-8"),
                    provenance_json_bytes=json.dumps({**snapshot.provenance(), "revision": lineage},
                        sort_keys=True, allow_nan=False, ensure_ascii=False).encode("utf-8"))
            metadata, blobs = snapshot.to_storage()
            analysis = WorkspaceArtifact(
                artifact_id="composition", kind="nanoscope.figure-composition", schema_version=1,
                payload={"layout": snapshot.config(),
                         "component_ids": list(dialog.panel_ids),
                         "component_id_contract": "embedded panel identifiers" if lineage else "project figure identifiers",
                         "component_names": [entry.payload.get("name", "Figure") for entry in components],
                         "figure_sources": [recipe for entry in components
                                            for recipe in entry.payload.get("source_recipes", [])],
                         **(lineage or {})})
            composition = make_figure(
                name=dialog.figure_name(), page_id="composition",
                source_label=f"{len(components)} saved panels · component names and source details in provenance",
                analysis=analysis, snapshot_blobs=blobs, snapshot_metadata=metadata,
                live_state=None, capture_status="snapshot")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Compose publication figure", str(exc))
            return None
        self._append(composition)
        self.changed.emit()
        self.refresh(composition.artifact_id)
        self.statusBar().showMessage(
            "Composition saved as a new snapshot. Save Project to persist it to disk.", 10000)
        return composition

    def export_current(self, path: str | Path | None = None, *, bundle=False,
                       options=None) -> Path | None:
        from .figure_bundle import FigureSnapshot, write_figure_bundle, write_frozen_figure_image
        from .plot_export_options import choose_export_options, resolve_export_options
        from ..core.atomic_write import staged_write, take_identity_notes

        entry = self.current_entry()
        if entry is None:
            return None
        try:
            snapshot = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export saved figure", str(exc))
            return None
        suffix = ".zip" if bundle else ".svg"
        if path is None:
            if not bundle:
                options = choose_export_options(
                    self, (snapshot.width, snapshot.height), initial_format=".svg",
                    preview_image=_scaled_image(snapshot.png_bytes, QtCore.QSize(1200, 900)))
                if options is None:
                    return None
                suffix = options.extension
            stem = re.sub(r"[^\w .-]", "_", str(entry.payload.get("name", "Figure")))[:100]
            target = ask_export_path(
                self, title="Save figure bundle" if bundle else "Export saved figure",
                suggestion=stem + suffix,
                filters=("Figure bundle (*.zip)" if bundle else f"Figure (*{suffix})"),
                outputs=lambda chosen: [chosen.with_suffix(suffix)])
            if target is None:
                return None
            path = target.with_suffix(suffix)
        out = Path(path)
        if not out.suffix:
            out = out.with_suffix(suffix)
        entry = self._entry_by_id(entry.artifact_id)
        if entry is None:
            self.statusBar().showMessage("The figure is no longer in the current project; nothing was exported.", 10000)
            return None
        try:
            if bundle:
                if out.suffix.lower() != ".zip":
                    raise ValueError("Choose a .zip filename for the figure bundle.")
                write_figure_bundle(out, snapshot)
                notes = take_identity_notes(out)
            else:
                if out.suffix.lower() == ".csv":
                    with staged_write(out) as stream:
                        stream.write(snapshot.data_csv_bytes)
                    notes = take_identity_notes(out)
                else:
                    if options is None:
                        options = resolve_export_options(
                            (snapshot.width, snapshot.height), extension=out.suffix, native=True)
                    notes = write_frozen_figure_image(out, snapshot, options)
        except Exception as exc:  # GUI export errors must not lose the previous file
            QtWidgets.QMessageBox.warning(self, "Export saved figure", str(exc))
            return None
        entry.payload["export_path"] = str(out.absolute())
        self.changed.emit()
        self._show_current()
        self.statusBar().showMessage(f"Saved {out}", 10000)
        if notes:
            QtWidgets.QMessageBox.information(self, "Export saved figure", "\n\n".join(notes))
        return out

    def reveal_current(self) -> None:
        entry = self.current_entry()
        path = _export_path(entry)
        if path is None:
            return
        if sys.platform == "darwin":
            # Argument list, never shell interpolation of a user-provided filename.
            subprocess.Popen(["/usr/bin/open", "-R", str(Path(path).absolute())])
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(path).parent)))
