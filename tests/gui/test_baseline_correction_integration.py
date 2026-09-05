"""The unified baseline workspace runs the app's real numerical contracts.

The presentation widget has its own focused tests.  These exercise the other half:
the owner-facing wiring through ``AnalysisPanel`` and its shared worker, including the
important promise that a zero-centred inspection residual does not become the runtime
signal by accident.
"""

from __future__ import annotations

# Qt bootstrap must precede any Qt import.
import os

import nanoscope  # noqa: F401,E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nanoscope.analysis.conditioning import (  # noqa: E402
    BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME,
    BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
)
from nanoscope.core.document import TraceDocument  # noqa: E402
from nanoscope.core.types import (  # noqa: E402
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.main_window import MainWindow  # noqa: E402
from nanoscope.gui import analysis_panel as analysis_panel_mod  # noqa: E402


FS = 50_000.0


def _drifting_document(tmp_path) -> TraceDocument:
    n = 120_000
    time_s = np.arange(n, dtype=np.float64) / FS
    rng = np.random.default_rng(8)
    raw = 120.0 + 18.0 * np.sin(2.0 * np.pi * 0.45 * time_s)
    raw += rng.normal(0.0, 0.8, n)
    for start in (18_000, 50_000, 86_000):
        raw[start:start + 450] -= 28.0
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "baseline-workspace.dat", format_name="synthetic",
        n_samples=n, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(np.asarray(raw, dtype=np.float32),))


def _install(window: MainWindow, document: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    window._on_selection(doc_id, 0)
    return doc_id, 0


@pytest.fixture
def window(qapp, qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win._analysis_controller.abort()


def _wait_for_workspace(panel, workspace, qtbot) -> None:
    qtbot.waitUntil(
        lambda: panel._pending is None and not workspace.is_busy(), timeout=20_000)


def test_workspace_runs_visual_and_runtime_contracts_then_exports_runtime_snapshot(
        window, qtbot, tmp_path):
    doc_id, channel = _install(window, _drifting_document(tmp_path))
    panel = window._analysis_panel
    workspace = panel.baseline_workspace
    raw = np.asarray(window._session.calibrated(doc_id, channel).data)

    # The initial zero-centred comparison is an inspection residual.  It is computed
    # in the controller worker and leaves the signal-chain detrend switch untouched.
    assert workspace.output_mode() == BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY
    assert not panel.detection_panel.params()["detrend_enabled"]
    workspace.refresh_button.click()
    _wait_for_workspace(panel, workspace, qtbot)
    _x_zero, y_zero = workspace.corrected_curve.getData()
    assert y_zero is not None and y_zero.size > 100
    assert abs(float(np.median(y_zero))) < 3.0
    assert "Visual residual only" in workspace.preview_status.text()
    assert not panel.detection_panel.params()["detrend_enabled"]

    # Changing the mode explicitly restores the raw TraceView display, then the
    # separate runtime opt-in becomes the exact DetectionPanel signal-chain setting.
    workspace.output_mode_combo.setCurrentIndex(1)
    assert workspace.output_mode() == BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME
    assert not panel.baseline_panel.subtract_baseline()
    workspace.runtime_enabled.setChecked(True)
    assert panel.detection_panel.params()["detrend_enabled"] is True
    workspace.refresh_button.click()
    _wait_for_workspace(panel, workspace, qtbot)
    _x_runtime, y_runtime = workspace.corrected_curve.getData()
    assert y_runtime is not None and y_runtime.size > 100
    assert abs(float(np.median(y_runtime)) - float(np.median(raw))) < 12.0
    assert abs(float(np.median(y_runtime))) > 50.0, "retained-offset output was re-zeroed"
    assert "Runtime Baseline drift is enabled" in workspace.preview_status.text()

    # The compact selector is a proxy, not a second parameter owner: Advanced opens
    # the actual signal-chain page and its baseline-drift stage.
    workspace.advanced_button.click()
    assert panel.detection_panel.preprocess_sections.currentIndex() == 0
    assert "Baseline drift" in panel.detection_panel.filter_selector.currentText()

    # An ordinary Median filter is a different stage.  It may be enabled in the
    # live chain, but the baseline-only comparison must name it as excluded rather
    # than claiming it produced the exported arrays.
    panel.detection_panel.median_enabled.setChecked(True)

    output = tmp_path / "runtime-baseline.csv"
    assert panel.export_baseline_correction(output)
    _wait_for_workspace(panel, workspace, qtbot)
    meta = json.loads(output.with_suffix(".meta.json").read_text())
    assert meta["correction_mode"] == BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME
    assert meta["runtime_correction_active"] is True
    assert meta["baseline_source"] == "baseline_drift_estimate"
    assert meta["runtime_params"]["detrend_enabled"] is True
    assert "median_enabled" not in meta["runtime_params"]
    assert meta["comparison_scope"] == "baseline_drift_stage_only"
    assert "median" in meta["excluded_runtime_stages"]
    assert "Wrote" in workspace.preview_status.text()


def test_late_preview_cannot_overwrite_settings_changed_while_it_ran(
        window, qtbot, tmp_path, monkeypatch):
    """A worker result belongs to the complete launch-time recipe, not just the file."""
    _install(window, _drifting_document(tmp_path))
    panel = window._analysis_panel
    workspace = panel.baseline_workspace
    started = threading.Event()
    release = threading.Event()
    calculate = analysis_panel_mod._baseline_workspace_arrays

    def delayed(*args, **kwargs):
        started.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("test did not release baseline worker")
        return calculate(*args, **kwargs)

    monkeypatch.setattr(analysis_panel_mod, "_baseline_workspace_arrays", delayed)
    workspace.refresh_button.click()
    qtbot.waitUntil(started.is_set, timeout=10_000)
    workspace.output_mode_combo.setCurrentIndex(1)
    release.set()
    _wait_for_workspace(panel, workspace, qtbot)

    assert workspace.is_preview_stale()
    assert "settings changed while this comparison ran" in workspace.preview_status.text()
    assert workspace.corrected_curve.getData() == (None, None)


def test_export_completion_names_its_launch_snapshot_after_recipe_change(
        window, qtbot, tmp_path, monkeypatch):
    """A completed file stays valid, but its late status must not call it current."""
    _install(window, _drifting_document(tmp_path))
    panel = window._analysis_panel
    workspace = panel.baseline_workspace
    workspace.output_mode_combo.setCurrentIndex(1)
    workspace.runtime_enabled.setChecked(True)
    launch_method = str(workspace.runtime_method.currentData())

    started = threading.Event()
    release = threading.Event()
    write_export = analysis_panel_mod.write_baseline_correction

    def delayed_write(*args, **kwargs):
        started.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("test did not release baseline export")
        return write_export(*args, **kwargs)

    monkeypatch.setattr(analysis_panel_mod, "write_baseline_correction", delayed_write)
    output = tmp_path / "launch-snapshot.csv"
    assert panel.export_baseline_correction(output)
    qtbot.waitUntil(started.is_set, timeout=10_000)
    next_index = (workspace.runtime_method.currentIndex() + 1) % workspace.runtime_method.count()
    workspace.runtime_method.setCurrentIndex(next_index)
    assert str(workspace.runtime_method.currentData()) != launch_method
    release.set()
    _wait_for_workspace(panel, workspace, qtbot)

    meta = json.loads(output.with_suffix(".meta.json").read_text())
    assert meta["runtime_params"]["detrend_method"] == launch_method
    assert "settings captured when export started" in workspace.preview_status.text()
    assert "current baseline settings changed" in workspace.preview_status.text()


def test_zero_export_stamps_actual_short_trace_fallback(window, qtbot, tmp_path):
    """The sidecar records the estimator that ran after length adaptation."""
    n = 96
    raw = np.linspace(118.0, 121.0, n, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    document = TraceDocument(
        meta=RecordingMeta(
            path=tmp_path / "short-baseline.dat", format_name="synthetic",
            n_samples=n, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
            channels=(info,),
        ),
        channels=(raw,),
    )
    _install(window, document)
    panel = window._analysis_panel
    output = tmp_path / "short-zero.csv"

    assert panel.baseline_workspace.output_mode() == BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY
    assert panel.export_baseline_correction(output)
    _wait_for_workspace(panel, panel.baseline_workspace, qtbot)
    meta = json.loads(output.with_suffix(".meta.json").read_text())

    assert meta["comparison_scope"] == "open_pore_reference_display_only"
    assert meta["baseline_params"]["baseline_param_source"] == "default"
    assert "baseline_median_pA" in meta["baseline_params"]
    assert "baseline_fallback" in meta["baseline_params"]
    assert any("short" in warning.lower() or "window" in warning.lower()
               for warning in meta["warnings"])
