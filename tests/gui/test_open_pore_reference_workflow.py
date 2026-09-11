"""Reference workflow integration: explicit run, safe preview, recording scope."""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace

import nanoscope  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 10_000.0
N = 20_000


def document(tmp_path, name="reference.dat"):
    data = np.random.default_rng(45).normal(-120, 1, N).astype(np.float32)
    data[5_000:7_000] -= 30
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(index=0, name="current", raw_units="pA",
                                               kind=ChannelKind.CURRENT),), t0_s=61.3)
    return TraceDocument(meta=meta, channels=(data,))


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    doc = document(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    yield window, window._analysis_panel, (doc_id, 0)
    window._analysis_controller.abort()


def choose_current(panel, value=-120):
    editor = panel.baseline_panel.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData("constant"))
    editor.current.setText(str(value))
    return editor


def install_current_result(window, panel, target):
    session = window._session
    result = DetectSchmitt().run(SimpleAnalysisContext(
        session.calibrated(*target), session.document(target[0]).meta, panel._form_params()))
    session.add_result(result)
    return result


def capture_preview(monkeypatch, window, panel):
    jobs = []
    monkeypatch.setattr(window._analysis_controller, "run_callable",
                        lambda job, **kwargs: jobs.append((job, kwargs)) or True)
    panel._on_reference_preview_requested()
    assert len(jobs) == 1
    assert jobs[0][1]["label"] == "open_pore_reference_preview"
    return jobs[0][0]


@pytest.mark.parametrize("engine", ["detect.schmitt", "detect.cusum"])
def test_preview_reuses_actual_detector_arrays(rig, monkeypatch, engine):
    import nanoscope.gui.analysis_panel as ap
    window, panel, _target = rig
    choose_current(panel)
    panel._set_active_detection(engine)
    job = capture_preview(monkeypatch, window, panel)
    def duplicate(*args, **kwargs):
        raise AssertionError("Preview repeated conditioning/baseline after detection")
    monkeypatch.setattr(ap, "conditioned_trace", duplicate)
    monkeypatch.setattr(ap, "characterisation_baseline", duplicate)
    payload = job()
    assert payload["event_count"] > 0


def test_reference_view_conditioning_cache_and_debounce(rig, monkeypatch, qtbot):
    import nanoscope.gui.analysis_panel as ap
    window, panel, target = rig
    raw = window._session.calibrated(*target).data
    calls = []
    original = ap.conditioned_trace
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(ap, "conditioned_trace", observed)
    first = panel._conditioned_for_baseline_view(raw, FS, {})
    assert panel._conditioned_for_baseline_view(raw, FS, {"k_in": 8}) is first
    assert len(calls) == 1  # thresholds don't change a conditioned display trace
    panel._conditioned_for_baseline_view(raw, FS, {"filter_cutoff_hz": 1000})
    assert len(calls) == 2
    draws = []
    panel._baseline_view_timer.timeout.disconnect()
    panel._baseline_view_timer.timeout.connect(lambda: draws.append(True))
    monkeypatch.setattr(panel.baseline_panel, "show_baseline", lambda: True)
    for _ in range(10):
        panel._schedule_baseline_view_refresh()
    assert not draws
    qtbot.waitUntil(lambda: len(draws) == 1, timeout=2000)
    panel._schedule_baseline_view_refresh()
    panel._show_empty()
    assert not panel._baseline_view_timer.isActive()
    assert panel._baseline_conditioned_cache is None


def test_signed_reference_resolves_without_installing_or_dispatching(rig, monkeypatch):
    window, panel, target = rig
    result = install_current_result(window, panel, target)
    snapshot = result.payload.df.copy(deep=True)
    starts = []
    monkeypatch.setattr(window._analysis_controller, "start", lambda *a, **k: starts.append((a, k)))
    choose_current(panel, -119.25)
    assert panel._live_baseline_config().open_pore_reference.current_pA == -119.25
    assert panel.current_detection() is result
    assert result.payload.df.equals(snapshot)
    assert starts == []
    assert panel.can_run()


def test_missing_reference_input_gates_detect_and_preview(rig, monkeypatch):
    window, panel, _target = rig
    choose_current(panel, "")
    calls = []
    monkeypatch.setattr(window._analysis_controller, "start", lambda *a, **k: calls.append("detect"))
    monkeypatch.setattr(window._analysis_controller, "run_callable", lambda *a, **k: calls.append("preview"))
    refusals = []
    panel.runRefused.connect(refusals.append)
    assert not panel.can_run()
    panel.trigger_detection()
    panel._on_reference_preview_requested()
    assert calls == []
    assert len(refusals) == 2


def test_current_trace_selection_uses_sample_coordinates_not_displayed_clock(rig, monkeypatch):
    _window, panel, target = rig
    monkeypatch.setattr(panel._trace_view, "measure_target", lambda: target)
    monkeypatch.setattr(panel._trace_view, "measure_bounds", lambda: (100, 300))
    panel._add_open_pore_selection()
    assert panel.baseline_panel.params()["baseline_open_pore_reference"]["intervals_s"] == [[0.01, 0.03]]
    assert panel._live_baseline_config().open_pore_reference.sample_intervals(N, FS) == ((100, 300),)


@pytest.mark.parametrize("bounds", [(-1, 100), (10, N + 1), (200, 100)])
def test_outside_selection_is_refused_not_clipped(rig, monkeypatch, bounds):
    _window, panel, target = rig
    monkeypatch.setattr(panel._trace_view, "measure_target", lambda: target)
    monkeypatch.setattr(panel._trace_view, "measure_bounds", lambda: bounds)
    messages = []
    panel.runRefused.connect(messages.append)
    panel._add_open_pore_selection()
    assert messages
    assert panel.baseline_panel.reference_mode() == "automatic"


def test_reference_change_requires_redetection_not_amplitude_remeasurement(rig, monkeypatch):
    window, panel, target = rig
    result = install_current_result(window, panel, target)
    choose_current(panel)
    assert panel._reference_redetection_needed(result)
    assert panel._stale._btn.text() == "Run detection"
    calls = []
    monkeypatch.setattr(panel, "_on_detect_requested", lambda params: calls.append("detect"))
    monkeypatch.setattr(panel, "remeasure_active_detection", lambda: calls.append("remeasure"))
    panel._on_stale_action("baseline")
    assert calls == ["detect"]


def test_stale_reference_action_preserves_curation_when_replacement_is_cancelled(rig, monkeypatch):
    window, panel, target = rig
    result = install_current_result(window, panel, target)
    result.payload.df.loc[0, "accepted"] = False
    before = result.payload.df.copy(deep=True)
    choose_current(panel)
    calls = []
    monkeypatch.setattr(panel, "confirm_replacing_curation", lambda module: False)
    monkeypatch.setattr(panel, "_run_detection", lambda params: calls.append(params))
    panel._on_stale_action("baseline")
    assert calls == []
    assert panel.current_detection() is result
    assert result.payload.df.equals(before)
    assert panel._reference_redetection_needed(result)


def test_simultaneous_calibration_and_reference_change_detects_live_recipe(rig, monkeypatch):
    window, panel, target = rig
    result = install_current_result(window, panel, target)
    choose_current(panel, -119)
    old = window._session.calibration(*target)
    window._session.set_calibration(*target, replace(old, gain=1.1), write_sidecar=False)
    assert panel._reference_redetection_needed(result)
    assert panel._stale._btn.text() == "Run detection"
    calls = []
    monkeypatch.setattr(panel, "confirm_replacing_curation", lambda module: True)
    monkeypatch.setattr(panel, "_run_detection", lambda params: calls.append(params) or True)
    panel._on_stale_action("calibration")
    assert len(calls) == 1
    assert calls[0]["baseline_open_pore_reference"]["current_pA"] == -119
    assert calls[0]["baseline_param_source"] == "user"


def test_preview_worker_runs_production_detection_without_replacing_results(rig, monkeypatch):
    window, panel, target = rig
    result = install_current_result(window, panel, target)
    snapshot = result.payload.df.copy(deep=True)
    choose_current(panel)
    job = capture_preview(monkeypatch, window, panel)
    assert panel._pending == "open_pore_reference_preview"
    assert panel.baseline_panel.is_busy()
    payload = job()
    assert isinstance(payload, dict)
    assert payload["event_count"] > 0
    assert payload["polarity"] == -1
    assert payload["noise_pA"] > 0
    assert payload["settings"] == panel._form_params()
    assert panel.current_detection() is result
    assert result.payload.df.equals(snapshot)
    # Exercise the common completion sink; its preview route must never install.
    finished = []
    monkeypatch.setattr(panel, "_finish_reference_preview", lambda obj, selection: finished.append((obj, selection)))
    panel._on_controller_finished(payload)
    assert finished == [(payload, target)]
    assert panel.current_detection() is result
    assert result.payload.df.equals(snapshot)


@pytest.mark.parametrize("change", ["settings", "target", "gain", "units"])
def test_preview_is_discarded_if_its_inputs_changed(rig, monkeypatch, tmp_path, change):
    window, panel, target = rig
    choose_current(panel)
    payload = {"settings": deepcopy(panel._form_params()),
               "calibration": tuple(window._session.calibration(*target).fingerprint())}
    messages = []
    panel.runRefused.connect(messages.append)
    if change == "settings":
        panel.baseline_panel.reference_editor.current.setText("-125")
    elif change == "target":
        doc = document(tmp_path, "other.dat")
        other = window._session.add_document(doc)
        window._create_default_layers(doc)
        window._on_selection(other, 0)
    else:
        old = window._session.calibration(*target)
        updated = replace(old, gain=1.1) if change == "gain" else replace(old, target_units="mV")
        window._session.set_calibration(*target, updated, write_sidecar=False)
    panel._finish_reference_preview(payload, target)
    assert messages and "discarded" in messages[-1]
    assert not panel.baseline_panel.is_busy()


def test_recording_selection_resets_and_restores_only_its_own_recipe(rig, tmp_path):
    window, panel, target = rig
    choose_current(panel, -118)
    install_current_result(window, panel, target)
    doc = document(tmp_path, "other.dat")
    other = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(other, 0)
    assert panel.baseline_panel.reference_mode() == "automatic"
    assert "baseline_open_pore_reference" not in panel._form_params()
    window._on_selection(*target)
    assert panel._live_baseline_config().open_pore_reference.current_pA == -118
    assert panel.baseline_panel.param_source() == "restored"


@pytest.mark.parametrize("slot", ["_on_sweep_requested", "_on_null_requested", "_on_recovery_requested"])
def test_surrogate_studies_refuse_recording_specific_references(rig, monkeypatch, slot):
    window, panel, _target = rig
    choose_current(panel)
    calls, messages = [], []
    panel.runRefused.connect(messages.append)
    monkeypatch.setattr(window._analysis_controller, "run_callable", lambda *a, **k: calls.append(True))
    getattr(panel, slot)(panel._detection.params())
    assert calls == []
    assert messages and "surrogates" in messages[-1]


@pytest.mark.parametrize("module", ["detect.ml", "detect.domain"])
def test_unsupported_detector_refuses_without_dispatch(rig, monkeypatch, module):
    window, panel, _target = rig
    choose_current(panel)
    calls = []
    monkeypatch.setattr(window._analysis_controller, "start", lambda *a, **k: calls.append(True))
    params = {**panel._form_params(), "detector": module}
    if module == "detect.ml":
        assert panel._launch_ml(params) is False
    else:
        assert panel._run_detection(params) is False
    assert calls == []


def test_legacy_baseline_workspace_refuses_preview_and_export(rig, monkeypatch):
    window, panel, _target = rig
    choose_current(panel)
    calls = []
    monkeypatch.setattr(window._analysis_controller, "run_callable", lambda *a, **k: calls.append(True))
    panel._on_baseline_workspace_refresh()
    panel._on_baseline_workspace_export()
    assert calls == []
    assert "Preview reference and detection" in panel.baseline_workspace.preview_status.text()


def test_baseline_overlay_uses_supplied_signed_current_without_results(rig, monkeypatch):
    _window, panel, target = rig
    captured = []
    monkeypatch.setattr(panel._trace_view, "set_baseline_track", lambda *args: captured.append(args))
    choose_current(panel, -118.5)
    panel.baseline_panel.show_track.setChecked(True)
    assert captured
    assert captured[-1][:2] == target
    np.testing.assert_array_equal(captured[-1][-1], -118.5)
    assert panel.current_detection() is None
