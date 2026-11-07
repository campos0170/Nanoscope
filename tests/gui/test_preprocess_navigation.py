"""Integrated Preprocess routes preserve one form and truthful processing/preview state."""
from __future__ import annotations

from copy import deepcopy
import json
import os

import nanoscope  # noqa: F401 - Qt bootstrap before importing Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow


def _document(tmp_path, name="preprocess"):
    fs = 1000.
    t = np.arange(12000) / fs
    raw = 100. + .3 * np.sin(2 * np.pi * t)
    meta = RecordingMeta(path=tmp_path/f"{name}.dat", format_name="synthetic",
                         n_samples=len(t), sample_rate_hz=fs,
                         rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(index=0, name="current", raw_units="pA",
                                               kind=ChannelKind.CURRENT),))
    return TraceDocument(meta=meta, channels=(raw.astype(np.float32),))


def _install(win, doc):
    did = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(did, 0)
    return did


@pytest.fixture
def bench(qapp, qtbot, tmp_path):
    win = MainWindow(recovery_enabled=False)
    qtbot.addWidget(win)
    did = _install(win, _document(tmp_path))
    win.resize(1440, 1000)
    win.show()
    yield win, win._analysis_panel, did
    win._analysis_panel.abort_background_previews()
    win._analysis_controller.abort()


def _snapshot(panel):
    return deepcopy((panel.detection_panel.params(), panel.detection_panel.ml_params(),
                     panel.baseline_panel.params(), panel.baseline_workspace.form_state()))


def _wait(panel, ws, qtbot):
    qtbot.waitUntil(lambda: panel._pending is None and not ws.is_busy(), timeout=20000)


def _anchors(comparison):
    editor = comparison.background
    editor.source.setCurrentIndex(editor.source.findData("anchors"))
    editor._append(editor.regions, ["A", "0", "12000"])
    editor._append(editor.anchors, ["A", "0", "100"])
    editor._append(editor.anchors, ["A", "12000", "100"])
    return editor


def test_inspector_routes_open_one_workspace_without_editing_shared_settings(bench, qtbot):
    win, panel, _did = bench
    det = panel.detection_panel
    det.detrend_window_s.setValue(.8)
    panel.baseline_panel.rough_s.setValue(1.6)
    panel.baseline_panel.base_s.setValue(.2)
    before = _snapshot(panel)
    win._reveal_detection_panel()
    det.show_preprocess_tab()
    assert set(det.preprocess_route_buttons) == {"filters", "reference", "comparison"}
    owned = None
    for key in ("reference", "comparison", "filters", "reference"):
        det.preprocess_route_buttons[key].click()
        det.preprocess_open_button.click()
        ws = win._signal_conditioning
        assert ws is not None and ws.isVisible()
        if owned is None:
            owned = ws
        assert ws is owned
        assert ws.current_workspace_section() == key
        assert det.current_preprocess_section() == key
        assert _snapshot(panel) == before
        ws.close()
    # Existing controls move once. Every operation still reads the same form.
    assert panel.baseline_panel.window() is owned
    assert panel.baseline_workspace.window() is owned
    assert det.filter_stack.window() is owned
    assert det.filter_preview.window() is owned
    assert panel.baseline_workspace.baseline_panel is panel.baseline_panel
    assert det.params()["detrend_window_s"] == .8
    assert panel.baseline_panel.params()["baseline_rough_s"] == 1.6


@pytest.mark.parametrize("source", ["anchors", "intervals", "recording"])
def test_custom_recipe_keeps_active_runtime_and_display_controls_operable(bench, qtbot, source):
    win, panel, _did = bench
    win._open_preprocess_workspace("comparison")
    ws = win._signal_conditioning
    comp = panel.baseline_workspace
    reference = panel.baseline_panel
    reference.show_track.setChecked(True)
    reference.subtract_view.setChecked(True)
    comp.runtime_enabled.setChecked(True)
    comp.background.source.setCurrentIndex(comp.background.source.findData(source))
    assert panel.detection_panel.params()["detrend_enabled"] is True
    assert comp.runtime_enabled.isChecked() and comp.runtime_enabled.isEnabled()
    assert comp.runtime_enabled.isVisibleTo(ws)
    assert comp.advanced_button.isEnabled() and comp.advanced_button.isVisibleTo(ws)
    for control in (reference.show_track, reference.subtract_view):
        assert control.isChecked() and control.isEnabled()
        assert control.isVisibleTo(ws)
    qtbot.mouseClick(comp.runtime_enabled, QtCore.Qt.LeftButton)
    assert panel.detection_panel.params()["detrend_enabled"] is False
    qtbot.mouseClick(reference.subtract_view, QtCore.Qt.LeftButton)
    assert reference.subtract_baseline() is False
    assert comp.background.source_key() == source


def test_export_new_recipe_preserves_stale_badge_and_exports_new_values(bench, qtbot, tmp_path):
    win, panel, did = bench
    win._open_preprocess_workspace("comparison")
    comp = panel.baseline_workspace
    editor = _anchors(comp)
    original_raw = win._session.calibrated(did, 0).data.copy()
    comp.refresh_button.click()
    _wait(panel, comp, qtbot)
    assert not comp.is_preview_stale()
    old_plot = comp.corrected_curve.getData()[1].copy()
    for row in range(2):
        editor.anchors.item(row, 2).setText("101")
    assert comp.is_preview_stale()
    path = tmp_path / "new-recipe.csv"
    assert panel.export_baseline_correction(path)
    _wait(panel, comp, qtbot)
    assert "Wrote" in comp.preview_status.text()
    assert comp.is_preview_stale()
    assert "Out of date" in comp.freshness_label.text()
    assert comp.freshness_label.isVisibleTo(win._signal_conditioning)
    np.testing.assert_array_equal(comp.corrected_curve.getData()[1], old_plot)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 2], 101.)
    np.testing.assert_allclose(data[:, 3], data[:, 1] - 101., atol=1e-5)
    assert json.loads(path.with_suffix(".meta.json").read_text())["recipe"]["source"] == "anchors"
    np.testing.assert_array_equal(win._session.calibrated(did, 0).data, original_raw)
    comp.refresh_button.click()
    _wait(panel, comp, qtbot)
    assert not comp.is_preview_stale()
    assert "Out of date" not in comp.freshness_label.text()


def test_source_switch_restores_custom_draft_and_runtime_without_replacing_widgets(bench, qtbot, tmp_path):
    win, panel, did = bench
    win._open_preprocess_workspace("comparison")
    ws = win._signal_conditioning
    comp = panel.baseline_workspace
    editor = _anchors(comp)
    editor.anchors.item(0, 2).setText("unfinished")
    comp.runtime_enabled.setChecked(True)
    before = _snapshot(panel)
    other = _install(win, _document(tmp_path, "other"))
    assert other != did
    assert comp.background.source_key() == "automatic"
    win._on_selection(did, 0)
    assert _snapshot(panel) == before
    assert comp.runtime_enabled.isChecked() and comp.runtime_enabled.isEnabled()
    assert editor.anchors.item(0, 2).text() == "unfinished"
    ws.close()
    win._open_preprocess_workspace("reference")
    assert win._signal_conditioning is ws
    assert panel.baseline_panel.window() is ws
    assert comp.window() is ws
    assert _snapshot(panel) == before


@pytest.mark.parametrize("kind", ["reference", "processed"])
def test_real_display_refusals_remain_visible_across_workspace_routes(bench, qtbot, kind):
    win, panel, _did = bench
    win._open_preprocess_workspace("comparison")
    ws = win._signal_conditioning
    ref = panel.baseline_panel
    det = panel.detection_panel
    if kind == "reference":
        ref.reference_editor.mode.setCurrentIndex(ref.reference_editor.mode.findData("constant"))
        ref.reference_editor.current.setText("invalid")
        ref.show_track.setChecked(True)
        qtbot.waitUntil(lambda: bool(ref.view_status()), timeout=20000)
        label = ws.workspace_view_status
        assert ref.reference_error()
    else:
        det.set_params({"detrend_enabled": True, "detrend_method": "block_median",
                        "detrend_window_s": 60., "filter_cutoff_hz": None})
        det.filter_preview.setChecked(True)
        qtbot.waitUntil(lambda: bool(det.preview_error_label.text()), timeout=20000)
        label = det.preview_error_label
        assert "window" in label.text().lower()
        assert det.filter_preview.isChecked()
    message = label.text()
    before = _snapshot(panel)
    for route in ("reference", "filters", "comparison"):
        ws.show_workspace_section(route)
        assert label.isVisibleTo(ws)
        assert label.text() == message
        assert _snapshot(panel) == before
    if kind == "reference":
        ref.reference_editor.current.setText("100")
        qtbot.waitUntil(lambda: not ref.view_status(), timeout=20000)
    else:
        det.detrend_window_s.setValue(.5)
        qtbot.waitUntil(lambda: not det.preview_error_label.text(), timeout=20000)
    assert not label.isVisibleTo(ws)
