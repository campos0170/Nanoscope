"""Stage 3's MainWindow boundary: one writer, lazy readers, no widget-owned loss."""

from __future__ import annotations

import os
from pathlib import Path

import nanoscope  # noqa: F401 - Qt bootstrap precedes PySide

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore
import numpy as np
import pandas as pd

from nanoscope.core.analysis_workspace import (
    EVENT_PLOTS_ARTIFACT_ID,
    IV_ARTIFACT_ID,
    PLOTS_ARTIFACT_ID,
    PSD_ARTIFACT_ID,
    SUBPOPULATION_ARTIFACT_ID,
    VOLTAGE_DEPENDENCE_ARTIFACT_ID,
    capture_event_plots,
    capture_iv,
    _decode_like,
)
from nanoscope.analysis.events import EventSet, SCHEMA, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, WorkspaceArtifact
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as main_window_module
from nanoscope.gui.main_window import MainWindow


def _window(qtbot) -> MainWindow:
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    return window


def _add_trace(window: MainWindow, path: Path, value) -> tuple[str, int]:
    samples = (np.full(200, value, dtype=np.float32)
               if np.isscalar(value) else np.ascontiguousarray(value, dtype=np.float32))
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=len(samples),
        sample_rate_hz=10_000.0, rate_provenance=Provenance.FILE_META,
        channels=(info,))
    document = TraceDocument(meta, (samples,))
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    return doc_id, 0


def _install_events(window: MainWindow, target: tuple[str, int]) -> AnalysisResult:
    doc_id, channel = target
    rows = []
    for event_id, (start, end, depth) in enumerate(
            ((300, 340, 20.0), (900, 960, 35.0), (1600, 1680, 50.0))):
        rows.append({
            "event_id": event_id, "recording": "events.dat", "pore_id": "P",
            "start_idx": start, "end_idx": end, "t_start_s": start / 10_000.0,
            "dwell_s": (end - start) / 10_000.0, "baseline_pA": 100.0,
            "depth_pA": depth, "frac_depth": depth / 100.0,
            "ecd_pA_s": depth * (end - start) / 10_000.0,
            "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1e-4, "fall_s": 1e-4, "snr": 10.0,
            "n_levels": 1, "filter_limited": False, "accepted": True, "flag": "",
        })
    frame = pd.concat([empty_df(), pd.DataFrame(rows)], ignore_index=True)
    frame = frame.astype({name: spec[0] for name, spec in SCHEMA.items()})
    result = AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=doc_id,
        channel=channel, params={}, calibration_fingerprint=None,
        payload=EventSet(frame, {}))
    window._session.add_result(result)
    window._on_selection(doc_id, channel)
    return result


def test_analysis_form_edit_creates_workspace_only_unsaved_work(qapp, qtbot):
    window = _window(qtbot)
    assert not window._analytical_workspace.artifacts
    assert not window._has_work_to_lose()

    before = window._session_revision
    spin = window._analysis_panel.merge_controls.window_ms
    spin.setValue(spin.value() + spin.singleStep())

    assert window._session_revision > before
    assert window._dirty
    assert window._workspace_has_artifact("builtin.analysis-panel")
    assert window._has_work_to_lose(), "workspace-only work bypassed the close guard"


def test_recording_navigation_does_not_create_an_analytical_edit(qapp, qtbot, tmp_path):
    window = _window(qtbot)
    first = _add_trace(window, tmp_path / "first.dat", 1.0)
    second = _add_trace(window, tmp_path / "second.dat", 2.0)
    window._on_selection(*first)
    window._project_path = str(tmp_path / "existing.nanoscope")
    window._mark_clean()
    before = window._session_revision

    window._on_selection(*second)

    assert window._session_revision == before
    assert not window._dirty


def test_navigation_does_not_recompute_visible_frozen_snapshots(
        qapp, qtbot, tmp_path, monkeypatch):
    window = _window(qtbot)
    target = _add_trace(window, tmp_path / "target.dat", 1.0)
    window._event_plot_window = object()
    window._psd_window = object()
    window._workspace_frozen.update({EVENT_PLOTS_ARTIFACT_ID, PSD_ARTIFACT_ID})
    monkeypatch.setattr(
        main_window_module, "_on_screen", lambda value: value is not None)
    monkeypatch.setattr(
        window, "_refresh_event_plots",
        lambda: (_ for _ in ()).throw(AssertionError("event analysis reran")))
    monkeypatch.setattr(
        window, "_point_psd_at_selection",
        lambda: (_ for _ in ()).throw(AssertionError("PSD kernel reran")))

    window._on_selection(*target)


def test_first_psd_result_is_immediately_project_work(qapp, qtbot, tmp_path):
    window = _window(qtbot)
    rng = np.random.default_rng(17)
    target = _add_trace(
        window, tmp_path / "noise.dat", rng.normal(0.0, 1.0, 4096))
    window._on_selection(*target)
    window._mark_clean()
    before = window._session_revision

    window._open_psd()

    assert window._psd_window.has_data()
    assert window._workspace_has_artifact(PSD_ARTIFACT_ID)
    assert window._session_revision > before
    assert window._dirty


def test_unknown_artifact_survives_main_window_gather_byte_for_byte(qapp, qtbot):
    window = _window(qtbot)
    opaque = WorkspaceArtifact(
        artifact_id="plugin.example/result",
        kind="example.future-analysis",
        schema_version=7,
        payload={"future": {"answer": 42}},
        blobs={"model.bin": b"\x00opaque\xff"},
        extra={"producer-build": "later"},
    )
    window._analytical_workspace = AnalyticalWorkspace(
        artifacts=[opaque], extra={"plugin-order": ["plugin.example/result"]})

    state = window._gather_project_state()

    carried = state.workspace.artifacts[0]
    assert carried.artifact_id == opaque.artifact_id
    assert carried.kind == opaque.kind
    assert carried.schema_version == 7
    assert carried.payload == opaque.payload
    assert carried.blobs == opaque.blobs
    assert carried.extra == opaque.extra
    from nanoscope.core.acquisition_state import WORKSPACE_KEY
    assert {key: value for key, value in state.workspace.extra.items() if key != WORKSPACE_KEY} == {
        "plugin-order": ["plugin.example/result"]}


def test_loaded_workspace_is_lazy_and_does_not_open_surface_windows(qapp, qtbot):
    window = _window(qtbot)
    artifact = WorkspaceArtifact(
        artifact_id=PLOTS_ARTIFACT_ID,
        kind="nanoscope.plots",
        payload={},
    )

    window._begin_restore(ProjectState(
        workspace=AnalyticalWorkspace(artifacts=[artifact])))

    assert window._plots_window is None
    assert window._psd_window is None
    assert window._iv_window is None
    assert window._vdep_window is None
    assert PLOTS_ARTIFACT_ID in window._workspace_pending
    assert not window._dirty, "adopting a workspace created a phantom edit"


def test_saved_surface_gates_stay_open_when_live_inputs_are_unavailable(qapp, qtbot):
    window = _window(qtbot)
    cases = (
        (PLOTS_ARTIFACT_ID, window._gate_populations),
        (EVENT_PLOTS_ARTIFACT_ID, window._gate_event_plots),
        (PSD_ARTIFACT_ID, window._gate_psd),
        (IV_ARTIFACT_ID, window._gate_iv),
        (VOLTAGE_DEPENDENCE_ARTIFACT_ID, window._gate_vdep),
    )
    for artifact_id, gate in cases:
        window._analytical_workspace = AnalyticalWorkspace(artifacts=[WorkspaceArtifact(
            artifact_id=artifact_id, kind="saved-view", payload={})])
        assert gate()[0], f"{artifact_id} cannot open its self-contained saved view"


def test_plots_close_snapshots_before_its_deliberate_reset_and_reopens(qapp, qtbot):
    window = _window(qtbot)
    window.show()
    window._open_plots()
    plots = window._plots_window
    plots.set_title("Durable publication title")
    qtbot.wait(5)

    plots.close()
    qtbot.waitUntil(lambda: not plots.isVisible())

    assert window._workspace_has_artifact(PLOTS_ARTIFACT_ID)
    assert PLOTS_ARTIFACT_ID in window._workspace_pending
    assert plots.style().title == "", "the widget itself no longer has the saved title"

    window._open_plots()
    qtbot.wait(5)
    assert window._plots_window is plots
    assert plots.style().title == "Durable publication title"
    assert PLOTS_ARTIFACT_ID in window._workspace_frozen


def test_style_edit_does_not_turn_a_restored_snapshot_back_into_live_refresh(
        qapp, qtbot):
    window = _window(qtbot)
    window.show()
    window._open_plots()
    plots = window._plots_window
    plots.set_title("saved snapshot")
    plots.close()
    qtbot.waitUntil(lambda: not plots.isVisible())
    window._open_plots()
    assert PLOTS_ARTIFACT_ID in window._workspace_frozen

    # A cosmetic edit is durable but is not the explicit Refresh/Run boundary that
    # permits future session navigation to replace the snapshot's numerical result.
    plots.set_title("restyled snapshot")

    assert PLOTS_ARTIFACT_ID in window._workspace_frozen
    artifact = window._workspace_artifact(PLOTS_ARTIFACT_ID)
    restored_config = _decode_like(plots.config(), artifact.payload["config"], artifact)
    assert restored_config.style.title == "restyled snapshot"


def test_closing_an_already_checkpointed_plot_is_not_a_project_edit(qapp, qtbot):
    window = _window(qtbot)
    window.show()
    window._open_plots()
    window._plots_window.set_title("Already saved")
    window._mark_clean()
    before = window._session_revision

    window._plots_window.close()
    qtbot.waitUntil(lambda: not window._plots_window.isVisible())

    assert window._session_revision == before
    assert not window._dirty
    assert window._workspace_artifact(PLOTS_ARTIFACT_ID) is not None


def test_real_save_open_keeps_closed_plot_and_panel_draft_lazy_and_clean(
        qapp, qtbot, tmp_path):
    source = _window(qtbot)
    source.show()
    source._open_plots()
    source._plots_window.set_title("Project-owned title")
    source._plots_window.close()
    draft = source._analysis_panel.merge_controls.window_ms
    draft.setValue(draft.value() + 2.5)
    expected_ms = draft.value()
    project = tmp_path / "analytical.nanoscope"
    assert source._write_project(str(project))

    restored = _window(qtbot)
    with qtbot.waitSignal(restored.projectRestored, timeout=10_000):
        restored._open_project(str(project))

    assert restored._plots_window is None, "project open eagerly created a plot window"
    assert restored._analysis_panel.merge_controls.window_ms.value() == expected_ms
    assert not restored._dirty, "adapter replay marked a clean project dirty"
    restored._open_plots()
    assert restored._plots_window.style().title == "Project-owned title"


def test_saved_subpopulation_entry_point_never_dispatches_analysis(
        qapp, qtbot, monkeypatch):
    window = _window(qtbot)
    artifact = WorkspaceArtifact(
        artifact_id=SUBPOPULATION_ARTIFACT_ID,
        kind="nanoscope.subpopulation-report",
        payload={},
    )
    window._analytical_workspace = AnalyticalWorkspace(artifacts=[artifact])
    window._workspace_pending.add(SUBPOPULATION_ARTIFACT_ID)
    window._sync_saved_subpop_action()
    launched = []
    monkeypatch.setattr(
        window._analysis_controller, "run_callable",
        lambda *args, **kwargs: launched.append((args, kwargs)))
    monkeypatch.setattr(window, "_restore_workspace_surface", lambda *_args: True)

    window._act_show_saved_subpop.trigger()

    assert window._subpop_window is not None
    assert window._subpop_window.isVisible()
    assert launched == []


def test_lazy_iv_restore_does_not_gather_or_refit(qapp, qtbot, monkeypatch):
    # Capture an empty but valid IV snapshot through the real adapter first.
    source = _window(qtbot)
    source_iv = __import__("nanoscope.gui.iv_panel", fromlist=["IVWindow"])
    live = source_iv.IVWindow(source._session, source)
    artifact = capture_iv(live)
    live.close()

    window = _window(qtbot)
    window._analytical_workspace = AnalyticalWorkspace(artifacts=[artifact])
    window._workspace_pending.add(IV_ARTIFACT_ID)
    monkeypatch.setattr(source_iv.iv, "iv_points",
                        lambda *_: (_ for _ in ()).throw(AssertionError("gather ran")))
    monkeypatch.setattr(source_iv.iv, "iv_fit",
                        lambda *_: (_ for _ in ()).throw(AssertionError("fit ran")))

    window._open_iv()

    assert window._iv_window is not None
    assert IV_ARTIFACT_ID in window._workspace_frozen


def test_lazy_event_plot_restore_binds_source_without_analysis_kernels(
        qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import event_shape
    from nanoscope.gui.event_plot_panel import EventPlotWindow

    window = _window(qtbot)
    rng = np.random.default_rng(23)
    signal = 100.0 + rng.normal(0.0, 0.5, 4096)
    for start, end, depth in ((300, 340, 20.0), (900, 960, 35.0),
                              (1600, 1680, 50.0)):
        signal[start:end] -= depth
    target = _add_trace(window, tmp_path / "events.dat", signal)
    result = _install_events(window, target)
    source = EventPlotWindow(window._session)
    qtbot.addWidget(source)
    source.set_source(result.doc_id, result.channel, result.payload)
    artifact = capture_event_plots(source)
    expected = source.density_counts()

    window._analytical_workspace = AnalyticalWorkspace(artifacts=[artifact])
    window._workspace_pending.add(EVENT_PLOTS_ARTIFACT_ID)
    for name in ("density_2d", "event_ensemble", "charge_deficit"):
        monkeypatch.setattr(
            event_shape, name,
            lambda *_a, _name=name, **_k: (_ for _ in ()).throw(
                AssertionError(f"{_name} reran")))

    window._open_event_plots()

    assert EVENT_PLOTS_ARTIFACT_ID in window._workspace_frozen
    np.testing.assert_array_equal(window._event_plot_window.density_counts(), expected)


def test_hiding_cached_surface_makes_its_project_copy_pending(qapp, qtbot):
    window = _window(qtbot)
    window.show()
    window._open_plots()
    plots = window._plots_window
    plots.set_title("cached")
    plots.close()
    qtbot.waitUntil(lambda: not plots.isVisible())

    # Exercise the event-loop edge explicitly: Hide is the point after Plots reset.
    QtCore.QCoreApplication.processEvents()
    assert PLOTS_ARTIFACT_ID in window._workspace_pending
