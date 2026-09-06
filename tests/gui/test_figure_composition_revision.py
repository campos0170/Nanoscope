"""Layout revisions use embedded originals, preserve history, and guard project identity."""
from copy import deepcopy
from dataclasses import replace
import json
import zipfile

import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.figure_library import make_figure
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, WorkspaceArtifact, load_project, save_project
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.figure_composition import compose_figures
from nanoscope.gui.figure_composition_revision import extract_composition
from nanoscope.gui.figure_library import FigureLibrary
from .test_figure_composer import entries
from .test_figure_composition import sources


def _snapshot(entry):
    return FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)


def _entry(snapshot, components):
    metadata, blobs = snapshot.to_storage()
    return make_figure(name="Paper figure", page_id="composition", source_label="Saved panels",
        analysis=WorkspaceArtifact("composition", "nanoscope.figure-composition", payload={
            "component_ids": [entry.artifact_id for entry in components],
            "component_names": [entry.payload["name"] for entry in components]}),
        snapshot_metadata=metadata, snapshot_blobs=blobs, live_state=None, capture_status="snapshot")


@pytest.fixture
def composed(entries):
    return _entry(compose_figures([_snapshot(entry) for entry in entries], ["Control", "Treated", "Recovery"], 2), entries)


def test_extraction_uses_exact_embedded_originals_not_duplicate_metadata(composed, entries):
    snapshot = _snapshot(composed)
    config = snapshot.config()
    config["panels"][0]["source_config"] = {"invented": "ignored"}
    changed = replace(snapshot, config_json_bytes=json.dumps(config).encode())
    panels, layout = extract_composition(changed)
    assert layout == {"columns": 2, "share_x": False, "share_y": False,
                      "panel_labels": ["Control", "Treated", "Recovery"]}
    assert len(panels) == 3
    for panel, original in zip(panels, entries):
        assert panel.snapshot.to_storage() == _snapshot(original).to_storage()


@pytest.mark.parametrize("corrupt", ["version", "bool_version", "columns", "shared", "labels", "count",
                                    "duplicate_id", "alias", "missing_blob", "checksum", "child_version"])
def test_invalid_layout_or_resources_refused_without_repair(composed, corrupt):
    snapshot = _snapshot(composed)
    config = snapshot.config()
    resources = dict(snapshot.additional_resources)
    if corrupt == "version":
        config["version"] = 2
    elif corrupt == "bool_version":
        config["version"] = True
    elif corrupt == "columns":
        config["columns"] = 0
    elif corrupt == "shared":
        config["share_x"] = "false"
    elif corrupt == "labels":
        config["panel_labels"][0] = "different"
    elif corrupt == "count":
        config["panel_labels"].pop()
    elif corrupt == "duplicate_id":
        config["panels"][1]["panel_id"] = "panel_001"
    elif corrupt == "alias":
        config["panels"][0]["blob_resources"]["png"] = "panels/panel_002/figure.png"
    elif corrupt == "missing_blob":
        resources.pop("panels/panel_001/figure.png")
    elif corrupt == "checksum":
        config["panels"][0]["snapshot"]["sha256"]["png"] = "0" * 64
    else:
        config["panels"][0]["snapshot"]["version"] = 2
    changed = replace(snapshot, config_json_bytes=json.dumps(config).encode(), additional_resources=tuple(resources.items()))
    with pytest.raises(ValueError):
        extract_composition(changed)


def test_nested_and_single_panel_compositions_remain_opaque(composed, entries):
    inner = _snapshot(composed)
    outer = compose_figures([inner, _snapshot(entries[0])], ["Nested", "Other"], 2)
    panels, _ = extract_composition(outer)
    assert len(panels) == 2 and panels[0].snapshot.to_storage() == inner.to_storage()
    single = compose_figures([inner], ["Only"], 1)
    one, _ = extract_composition(single)
    assert len(one) == 1 and one[0].snapshot.to_storage() == inner.to_storage()


def test_shared_revision_always_starts_from_original_ranges(sources):
    _, snapshots, _ = sources
    shared = compose_figures(snapshots, ["A", "B"], 2, share_x=True, share_y=True)
    panels, _ = extract_composition(shared)
    for panel, original in zip(panels, snapshots):
        assert panel.snapshot.to_storage() == original.to_storage()
    independent = compose_figures([p.snapshot for p in panels], ["B", "A"], 1)
    panels_again, _ = extract_composition(independent)
    shared_again = compose_figures([p.snapshot for p in panels_again], ["A", "B"], 2,
                                  share_x=True, share_y=True)
    assert shared_again.config()["view_ranges"] == shared.config()["view_ranges"]


def test_revision_with_only_composition_loaded_saves_new_entry_and_exact_resources(
        composed, entries, qtbot, monkeypatch, tmp_path):
    original = deepcopy(composed)
    library_entries = [composed]  # Separate source figures/recordings need not exist.
    library = FigureLibrary(entries=lambda: library_entries, status=lambda _: ("Saved snapshot", "Frozen"),
                            append=library_entries.append)
    qtbot.addWidget(library)
    assert library._revise_btn.isEnabled() and not library._compose_btn.isEnabled()
    approved = []
    def revise(dialog):
        assert dialog._columns.value() == 2
        assert dialog._table.rowCount() == 3
        assert dialog._table.item(0, 1).text() == "Control"
        assert dialog._table.item(2, 1).text() == "Recovery"
        dialog._table.selectRow(2)
        dialog._move(-1)
        dialog._move(-1)
        dialog._table.item(0, 1).setText("Reordered")
        dialog._columns.setValue(1)
        dialog.update_preview()
        assert dialog.snapshot is not None
        approved.append(dialog.snapshot)
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(FigureComposerDialog, "exec", revise)
    with qtbot.waitSignal(library.changed):
        result = library.revise_current()
    assert result is library_entries[-1] and len(library_entries) == 2
    assert result.artifact_id != composed.artifact_id
    assert composed.payload == original.payload and composed.blobs == original.blobs
    snapshot = _snapshot(result)
    assert snapshot.png_bytes == approved[0].png_bytes and snapshot.svg_bytes == approved[0].svg_bytes
    panels, layout = extract_composition(snapshot)
    assert layout["columns"] == 1 and layout["panel_labels"] == ["Reordered", "Control", "Treated"]
    for panel, source in zip(panels, [entries[2], entries[0], entries[1]]):
        assert panel.snapshot.to_storage() == _snapshot(source).to_storage()
    assert snapshot.provenance()["revision"]["revision_of"] == composed.artifact_id
    path = tmp_path / "revisions.nanoscope"
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=library_entries)))
    restored = load_project(path).workspace.artifacts
    assert restored[0].blobs == composed.blobs and restored[1].blobs == result.blobs
    bundle = library.export_current(tmp_path / "revision.zip", bundle=True)
    with zipfile.ZipFile(bundle) as archive:
        assert json.loads(archive.read("provenance.json"))["revision"]["revision_of"] == composed.artifact_id


@pytest.mark.parametrize("action", ["cancel", "project", "checkpoint", "mutation"])
def test_revision_modal_boundaries(composed, qtbot, monkeypatch, action):
    stored = [composed]
    epoch = [0]
    library = FigureLibrary(entries=lambda: stored, status=lambda _: ("Saved snapshot", "Frozen"),
                            append=stored.append, project_token=lambda: epoch[0])
    qtbot.addWidget(library)
    def finish(dialog):
        dialog.update_preview()
        if action == "cancel":
            return QtWidgets.QDialog.DialogCode.Rejected
        if action in ("checkpoint", "project"):
            stored[:] = deepcopy(stored)  # Same UUIDs/resources, different workspace object.
        if action == "project":
            epoch[0] += 1
        if action == "mutation":
            stored[0].payload["snapshot"]["width"] += 1  # In-place edit, not a replacement object.
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(FigureComposerDialog, "exec", finish)
    result = library.revise_current()
    assert (result is not None) == (action == "checkpoint")
    assert len(stored) == (2 if action == "checkpoint" else 1)


@pytest.mark.parametrize("action", ["project", "checkpoint", "mutation"])
def test_creation_uses_same_project_guard_and_frozen_stamps(entries, qtbot, monkeypatch, action):
    epoch = [0]
    library = FigureLibrary(entries=lambda: entries, status=lambda _: ("Saved snapshot", "Frozen"),
                            append=entries.append, project_token=lambda: epoch[0])
    qtbot.addWidget(library)
    def finish(dialog):
        dialog.update_preview()
        if action in ("project", "checkpoint"):
            entries[:] = deepcopy(entries)
        if action == "project":
            epoch[0] += 1
        if action == "mutation":
            entries[0].payload["snapshot"]["width"] += 1
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(FigureComposerDialog, "exec", finish)
    result = library.compose_current()
    assert (result is not None) == (action == "checkpoint")
    assert len(entries) == (4 if action == "checkpoint" else 3)


def test_project_epoch_changes_on_reset_not_checkpoint(qtbot):
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    epoch = window._figure_project_epoch
    window._capture_analytical_workspace()
    assert window._figure_project_epoch == epoch
    window._open_figure_library()
    assert window._figure_library._project_token() == epoch
    window._reset_analytical_workspace()
    assert window._figure_project_epoch == epoch + 1
    assert window._figure_library._project_token() == epoch + 1
    window._dirty = False


@pytest.mark.parametrize("alias", ["resource_01", "resource_5"])
def test_noncanonical_child_resource_ordinals_cannot_be_silently_renumbered(composed, alias):
    snapshot = _snapshot(composed)
    config = snapshot.config()
    panel = config["panels"][0]
    metadata = panel["snapshot"]
    filename = next(name for name, key in metadata["resources"].items() if key == "resource_0")
    metadata["resources"][filename] = alias
    metadata["sha256"][alias] = metadata["sha256"].pop("resource_0")
    panel["blob_resources"][alias] = panel["blob_resources"].pop("resource_0")
    changed = replace(snapshot, config_json_bytes=json.dumps(config).encode())
    with pytest.raises(ValueError, match="noncanonical resource"):
        extract_composition(changed)


def test_revision_dialog_restores_layout_and_requires_fresh_preview(composed, qtbot, tmp_path):
    panels, layout = extract_composition(_snapshot(composed))
    embedded = []
    for panel in panels:
        metadata, blobs = panel.snapshot.to_storage()
        embedded.append(WorkspaceArtifact(panel.panel_id, "nanoscope.saved-figure",
            payload={"name": panel.label, "snapshot": metadata}, blobs=blobs))
    dialog = FigureComposerDialog(embedded, initial_layout=layout, name="Paper figure (revision)")
    qtbot.addWidget(dialog)
    dialog.resize(1000, 720)
    dialog.show()
    assert not dialog._save_btn.isEnabled()
    assert all(dialog._table.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked for row in range(3))
    dialog.update_preview()
    assert dialog._save_btn.isEnabled()
    approved = dialog.snapshot
    dialog._name.setText("Revised for journal")
    assert dialog.snapshot is approved
    dialog._columns.setValue(1)
    assert dialog.snapshot is None and not dialog._save_btn.isEnabled()
    dialog.update_preview()
    qtbot.wait(30)
    assert dialog.width() == 1000 and dialog.height() == 720
    assert dialog._preview.height() >= 220
    bottom = dialog._save_btn.mapTo(dialog, dialog._save_btn.rect().bottomRight())
    assert dialog.rect().contains(bottom)
    assert dialog.grab().save(str(tmp_path / "composition-revision.png"))


def test_unknown_composition_version_keeps_export_available(composed, qtbot, monkeypatch, tmp_path):
    snapshot = _snapshot(composed)
    config = snapshot.config()
    config["version"] = 2
    future = replace(snapshot, config_json_bytes=json.dumps(config).encode())
    stored = [_entry(future, [])]
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _p, _t, message: messages.append(message))
    library = FigureLibrary(entries=lambda: stored, status=lambda _: ("Saved snapshot", "Frozen"), append=stored.append)
    qtbot.addWidget(library)
    assert library.revise_current() is None
    assert "supported version" in messages[0] and len(stored) == 1
    target = library.export_current(tmp_path / "future.svg")
    assert target.read_bytes() == future.svg_bytes
