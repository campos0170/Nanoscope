"""Explicit snapshots hydrate frozen records and cannot cross a modal owner change."""
from copy import deepcopy

import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import iv
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.main_window import MainWindow


def _window_with_frozen_points(qtbot):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    owner.resident_limit = 2
    key = owner.create("iv")
    page = owner.page(key)
    page._points = [iv.IVPoint(.1, 12., "pA", "nominal", "", "Frozen point")]
    page._exclusions = []
    page._fit = None
    page._render_workspace_snapshot()
    assert owner.checkpoint(key)
    owner.create("iv")
    owner.create("iv")
    assert owner.page(key) is None
    return window, owner, key


def test_inactive_snapshot_and_bundle_preserve_frozen_values_and_survive_origin_delete(qapp, qtbot, tmp_path, monkeypatch):
    window, owner, key = _window_with_frozen_points(qtbot)
    def forbidden(*args, **kwargs):
        raise AssertionError("Snapshot hydration must not gather or fit")
    monkeypatch.setattr(IVWindow, "refresh", forbidden)
    monkeypatch.setattr(iv, "iv_fit", forbidden)
    saved = window._save_analysis_figure(key, name="Frozen capture")
    assert saved is not None and owner.page(key) is not None
    assert owner.page(key).points()[0].i_open == 12.
    assert saved.payload["origin_figure_id"] == key
    assert saved.payload["lineage"]["origin_figure_id"] == key
    payload, blobs = deepcopy(saved.payload), dict(saved.blobs)
    assert owner.suspend(key) and owner.page(key) is None
    bundle = tmp_path / "frozen.zip"
    assert window._export_analysis_bundle(key, path=bundle) == bundle and bundle.is_file()
    assert owner.delete(key)
    retained = next(item for item in window._capture_analytical_workspace().artifacts
                    if item.artifact_id == saved.artifact_id)
    assert retained.payload == payload and retained.blobs == blobs


@pytest.mark.parametrize("change", ["project", "same_id_replacement"])
def test_snapshot_name_dialog_cannot_capture_a_replaced_owner(qapp, qtbot, monkeypatch, change):
    window, owner, key = _window_with_frozen_points(qtbot)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    def choose(*args, **kwargs):
        if change == "project":
            window._figure_project_epoch += 1
        else:
            owner.install_workspace(owner.capture_workspace())
        return "Wrong owner", False
    monkeypatch.setattr(window, "_figure_capture_options", choose)
    assert window._save_analysis_figure(key) is None
    assert not window._figure_entries()
    assert warnings and "nothing was saved" in warnings[-1]


def test_bundle_path_dialog_cannot_export_after_project_change(qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.gui import export_paths
    window, owner, key = _window_with_frozen_points(qtbot)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    output = tmp_path / "wrong-owner.zip"
    def choose(*args, **kwargs):
        window._figure_project_epoch += 1
        return output
    monkeypatch.setattr(export_paths, "ask_export_path", choose)
    assert window._export_analysis_bundle(key) is None
    assert not output.exists() and warnings


def test_inactive_capture_reports_busy_residency_refusal(qapp, qtbot, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    owner.resident_limit = 1
    owner.create("iv")
    key = owner.create("iv")
    assert owner.page(key) is None
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    assert window._save_analysis_figure(key, name="Cannot draw yet") is None
    assert warnings and "figure slots are busy" in warnings[-1]
    assert owner.get(key) is not None
