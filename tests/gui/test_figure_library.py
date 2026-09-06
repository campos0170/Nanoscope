"""Figure Library integration freezes exact numbers and never recreates analyses."""
from copy import deepcopy
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import iv
from nanoscope.core.figure_library import figures
from nanoscope.core.project import ProjectState, save_project, load_project
from nanoscope.gui.analysis_explorer import PAGE_IV
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_export_options import resolve_export_options


@pytest.fixture
def window(qtbot, monkeypatch):
    def unexpected(_parent, title, message, *args, **kwargs):
        pytest.fail(f"Unexpected dialog: {title}: {message}")
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", unexpected)
    win = MainWindow(recovery_enabled=False)
    qtbot.addWidget(win)
    win._open_iv()
    panel = win._iv_window
    panel._points = [iv.IVPoint(v, 100 * v + 2, "pA", "measured", str(i), f"run {i}")
                     for i, v in enumerate((-.2, -.5, -.8, -1.0))]
    panel._redraw()
    qtbot.wait(20)
    yield win
    # Test teardown is not a user's request to discard an unsaved scientific project.
    win._dirty = False
    if win._figure_library is not None:
        win._figure_library.close()


def test_save_duplicate_rename_and_bundle_do_not_recompute(window, qtbot, monkeypatch, tmp_path):
    panel = window._iv_window
    before_data = panel._plot_data().copy(deep=True)
    before_fit = deepcopy(panel.fit())
    panel._plot.setRange(xRange=(-.85, -.35), yRange=(-70, -5), padding=0)
    qtbot.wait(10)
    before_zoom = deepcopy(panel._plot.viewRange())
    before_items = tuple(panel._plot.items)
    def forbid(*args, **kwargs):
        pytest.fail("Opening/saving/exporting the library reran an analysis")
    monkeypatch.setattr(panel, "refresh", forbid)
    monkeypatch.setattr(panel, "_redraw", forbid)
    entry = window._save_analysis_figure(PAGE_IV, "I–V for paper")
    assert entry is not None
    assert window._has_work_to_lose() and window._dirty
    library = window._figure_library
    assert library._list.count() == 1
    assert not library._pixmap.isNull()
    snapshot = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
    pd.testing.assert_frame_equal(snapshot.dataframe(), before_data, check_exact=True)
    assert snapshot.config()["analysis_kind"] == "nanoscope.iv"
    assert snapshot.provenance()["analysis_recomputed"] is False

    library.rename_current("Final I–V")
    duplicate = library.duplicate_current()
    assert len(figures(window._analytical_workspace)) == 2
    assert duplicate.blobs == entry.blobs
    assert duplicate.artifact_id != entry.artifact_id
    out = library.export_current(tmp_path / "figure.zip", bundle=True)
    with zipfile.ZipFile(out) as archive:
        pd.testing.assert_frame_equal(
            pd.read_parquet(io.BytesIO(archive.read("data.parquet"))), before_data, check_exact=True)
        config = json.loads(archive.read("config.json"))
        assert set(config["tables"].values()) <= set(archive.namelist())
    assert library._reveal_btn.isEnabled()
    assert duplicate.payload["export_path"] == str(out)
    assert panel.fit() == before_fit
    assert tuple(panel._plot.items) == before_items
    np.testing.assert_array_equal(panel._plot.viewRange(), before_zoom)
    pd.testing.assert_frame_equal(panel._plot_data(), before_data, check_exact=True)


def test_project_reopen_keeps_library_without_constructing_analysis(window, qtbot, monkeypatch, tmp_path):
    entry = window._save_analysis_figure(PAGE_IV, "Saved snapshot")
    project = tmp_path / "library.nanoscope"
    save_project(project, ProjectState(workspace=window._capture_analytical_workspace()))
    incoming = load_project(project)
    other = MainWindow(recovery_enabled=False)
    qtbot.addWidget(other)
    other._analytical_workspace = incoming.workspace
    monkeypatch.setattr(other, "_open_iv", lambda: pytest.fail("Library opened an analysis"))
    other._open_figure_library()
    assert other._iv_window is None
    assert other._analysis_explorer is None
    assert other._figure_library._list.count() == 1
    restored = other._figure_library.current_entry()
    assert restored.blobs == entry.blobs
    assert other._figure_status(restored)[0] == "Saved snapshot"
    out = other._figure_library.export_current(tmp_path / "restored.svg")
    assert out.read_bytes() == entry.blobs["svg"]
    other._reset_analytical_workspace()
    assert other._figure_library._list.count() == 0
    other._dirty = False
    other._figure_library.close()


@pytest.mark.parametrize("extension", ["png", "svg", "pdf"])
def test_saved_figure_export_has_publication_sizing(window, tmp_path, extension):
    entry = window._save_analysis_figure(PAGE_IV, "Publication")
    snapshot = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
    options = resolve_export_options(
        (snapshot.width, snapshot.height), extension=extension, native=False,
        width=85, height=60, dpi=300)
    out = window._figure_library.export_current(tmp_path / f"figure.{extension}", options=options)
    assert out is not None and out.stat().st_size > 100


def test_explorer_bundle_action_does_not_create_library_entry(window, tmp_path):
    out = window._export_analysis_bundle(PAGE_IV, tmp_path / "current.zip")
    assert out.is_file()
    assert not window._figure_entries()
    assert window._figure_library is None


def test_library_reveal_uses_exact_exported_filename(window, tmp_path, monkeypatch):
    window._save_analysis_figure(PAGE_IV, "Revealing")
    library = window._figure_library
    out = library.export_current(tmp_path / "special $(text) 'plot'.svg")
    import nanoscope.gui.figure_library as module
    calls = []
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.subprocess, "Popen", lambda args: calls.append(args))
    library.reveal_current()
    assert calls == [["/usr/bin/open", "-R", str(out)]]


def test_library_preview_and_actions_fit_window(window, qtbot, tmp_path):
    window._save_analysis_figure(PAGE_IV, "I–V comparison · publication snapshot")
    library = window._figure_library
    library.resize(1120, 740)
    qtbot.wait(30)
    assert library._preview.width() > 280
    assert library._preview.height() > 220
    assert all(not button.isHidden() for button in (
        library._rename_btn, library._duplicate_btn, library._export_btn,
        library._bundle_btn, library._reveal_btn))
    assert library.grab().save(str(tmp_path / "figure-library.png"))


@pytest.mark.parametrize("bad_path", [123, [], {}, "relative.svg", "\u0000bad"])
def test_invalid_optional_export_path_cannot_block_library(window, bad_path):
    entry = window._save_analysis_figure(PAGE_IV, "Still readable")
    entry.payload["export_path"] = bad_path
    library = window._figure_library
    library.refresh()
    assert library.current_entry().artifact_id == entry.artifact_id
    assert not library._reveal_btn.isEnabled()
    assert not library._pixmap.isNull()
    library.reveal_current()


def test_modal_rename_targets_current_artifact_after_worker_checkpoint(window, monkeypatch):
    from nanoscope.core.analysis_workspace import capture_analysis_workspace
    entry = window._save_analysis_figure(PAGE_IV, "Original")
    def during_dialog(*_args, **_kwargs):
        window._analytical_workspace = capture_analysis_workspace(window._analytical_workspace)
        return "Requested new name", True
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", during_dialog)
    window._figure_library.rename_current()
    assert window._figure_entries()[0] is not entry
    assert window._figure_entries()[0].payload["name"] == "Requested new name"


def test_modal_export_tracks_current_artifact_after_worker_checkpoint(window, monkeypatch, tmp_path):
    import nanoscope.gui.figure_library as module
    from nanoscope.core.analysis_workspace import capture_analysis_workspace
    entry = window._save_analysis_figure(PAGE_IV, "Original")
    out = tmp_path / "snapshot.zip"
    def during_dialog(*_args, **_kwargs):
        window._analytical_workspace = capture_analysis_workspace(window._analytical_workspace)
        return out
    monkeypatch.setattr(module, "ask_export_path", during_dialog)
    assert window._figure_library.export_current(bundle=True) == out
    assert window._figure_entries()[0] is not entry
    assert window._figure_entries()[0].payload["export_path"] == str(out)


def test_project_switch_during_export_dialog_aborts_before_writing(window, monkeypatch, tmp_path):
    import nanoscope.gui.figure_library as module
    window._save_analysis_figure(PAGE_IV, "Previous project")
    out = tmp_path / "should-not-exist.zip"
    def during_dialog(*_args, **_kwargs):
        window._reset_analytical_workspace()
        return out
    monkeypatch.setattr(module, "ask_export_path", during_dialog)
    assert window._figure_library.export_current(bundle=True) is None
    assert not out.exists()
