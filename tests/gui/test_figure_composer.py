"""User-facing composition commits only the approved frozen picture."""
from copy import deepcopy

import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.figure_library import make_figure
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, WorkspaceArtifact, load_project, save_project
from nanoscope.gui.figure_bundle import FigureSnapshot, capture_figure_snapshot
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.figure_library import FigureLibrary


@pytest.fixture
def entries(qtbot):
    entries = []
    for i, colour in enumerate(("#ff0000", "#008800", "#0000ff")):
        plot = pg.PlotWidget(background="white")
        qtbot.addWidget(plot)
        plot.resize(430, 300)
        plot.show()
        plot.setLabel("bottom", "Voltage", units="V")
        plot.setLabel("left", "Current", units="A")
        data = pd.DataFrame({"x": [0., 1., 2.], "y": [float(i), 1. + i, 4. + i]})
        plot.plot(data.x.to_numpy(), data.y.to_numpy(), pen=pg.mkPen(colour, width=3))
        qtbot.wait(5)
        snapshot = capture_figure_snapshot(plot.getPlotItem(), data=data,
                                           config={"source": i}, provenance={"source": f"source {i}"})
        metadata, blobs = snapshot.to_storage()
        entries.append(make_figure(
            name=f"Figure {i}", page_id="test", source_label=f"source {i}",
            analysis=WorkspaceArtifact(artifact_id="test", kind="test", payload={}),
            snapshot_metadata=metadata, snapshot_blobs=blobs, live_state=None))
    return entries


def test_preview_invalidation_and_order(entries, qtbot):
    dialog = FigureComposerDialog(entries, current_id=entries[1].artifact_id)
    qtbot.addWidget(dialog)
    dialog.show()
    assert not dialog._save_btn.isEnabled()
    dialog.update_preview()
    assert dialog._save_btn.isEnabled()
    assert dialog.panel_ids == [entries[1].artifact_id, entries[0].artifact_id]
    approved = dialog.snapshot
    dialog._name.setText("Figure for paper")
    assert dialog.snapshot is approved  # Name is metadata, not picture layout.
    dialog._table.selectRow(0)
    dialog._move(1)
    assert dialog.snapshot is None
    assert not dialog._save_btn.isEnabled()
    dialog.update_preview()
    assert dialog.panel_ids == [entries[0].artifact_id, entries[1].artifact_id]
    dialog._columns.setValue(1)
    assert dialog.snapshot is None
    dialog._table.item(0, 1).setText("First condition")
    dialog.update_preview()
    assert dialog.snapshot is not None
    dialog._name.clear()
    assert not dialog._save_btn.isEnabled()
    dialog.accept()
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted


def test_selection_limits_and_shared_axis_failure_are_actionable(entries, qtbot):
    dialog = FigureComposerDialog(entries)
    qtbot.addWidget(dialog)
    dialog._table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    dialog.update_preview()
    assert "2 and 12" in dialog._message.text()
    assert dialog.snapshot is None
    dialog._table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    dialog._share_x.setChecked(True)
    dialog.update_preview()
    assert dialog.snapshot is None  # Legacy snapshots have no replay model.
    assert dialog._message.text() and not dialog._save_btn.isEnabled()
    dialog._share_x.setChecked(False)
    dialog.update_preview()
    assert dialog.snapshot is not None


def _library(entries, qtbot):
    library = FigureLibrary(entries=lambda: entries,
                            status=lambda entry: ("Saved snapshot", "Frozen"), append=entries.append)
    qtbot.addWidget(library)
    return library


def test_save_composition_roundtrip_preserves_originals(entries, qtbot, monkeypatch, tmp_path):
    originals = deepcopy(entries)
    library = _library(entries, qtbot)
    def approve(dialog):
        dialog.update_preview()
        assert dialog.snapshot is not None
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(FigureComposerDialog, "exec", approve)
    with qtbot.waitSignal(library.changed):
        saved = library.compose_current()
    assert saved is entries[-1]
    assert len(entries) == 4
    assert saved.payload["page_id"] == "composition"
    assert saved.payload["capture_status"] == "snapshot"
    for old, original in zip(entries[:3], originals):
        assert old.blobs == original.blobs
        assert old.payload == original.payload
    path = tmp_path / "composition.nanoscope"
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=entries)))
    restored = load_project(path).workspace.artifacts[-1]
    assert restored.blobs == saved.blobs
    snapshot = FigureSnapshot.from_storage(restored.payload["snapshot"], restored.blobs)
    assert not snapshot.dataframe().empty
    out = library.export_current(tmp_path / "composition.svg")
    assert out.read_bytes() == snapshot.svg_bytes
    assert library.export_current(tmp_path / "composition.zip", bundle=True).is_file()


def test_cancel_and_project_switch_do_not_append(entries, qtbot, monkeypatch):
    library = _library(entries, qtbot)
    monkeypatch.setattr(FigureComposerDialog, "exec", lambda _: QtWidgets.QDialog.DialogCode.Rejected)
    assert library.compose_current() is None
    assert len(entries) == 3
    def change_project(dialog):
        dialog.update_preview()
        entries.clear()
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(FigureComposerDialog, "exec", change_project)
    assert library.compose_current() is None
    assert entries == []
    assert "nothing was saved" in library.statusBar().currentMessage()


def test_preview_fits_common_screen(entries, qtbot, tmp_path):
    dialog = FigureComposerDialog(entries)
    qtbot.addWidget(dialog)
    dialog.resize(1000, 700)
    dialog.show()
    dialog.update_preview()
    qtbot.wait(30)
    assert dialog.width() == 1000 and dialog.height() == 700
    assert dialog._preview.height() >= 220
    bottom = dialog._save_btn.mapTo(dialog, dialog._save_btn.rect().bottomRight())
    assert dialog.rect().contains(bottom)
    assert dialog.grab().save(str(tmp_path / "figure-composer.png"))
