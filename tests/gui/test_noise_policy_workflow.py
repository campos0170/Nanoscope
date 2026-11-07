"""Noise controls through a real worker, trace rails, projects and exported evidence."""
import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detection_thresholds import RAIL_ENTER, RAIL_EXIT
from nanoscope.analysis.modules import depth_floor_from_provenance
from nanoscope.analysis.scoring import write_training_export
from nanoscope.core.project import load_project, save_project
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.types import Provenance
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel
from nanoscope.io.base import open_document

FS = 10_000.0
N = 100_000
NOISE_PREFIXES = ("noise_policy", "noise_window", "noise_local", "noise_profile", "noise_algorithm")


def _raw():
    x = np.random.default_rng(75661).normal(size=N)
    x[40_000:60_000] *= 8
    x += 100
    for start in (10_000, 80_000):
        x[start:start + 200] -= 30
    return x


def _window(qtbot):
    w = MainWindow(recovery_enabled=False)
    qtbot.addWidget(w)
    w.resize(1450, 950)
    w.show()
    return w


@pytest.fixture
def noise_window(qapp, qtbot, tmp_path):
    path = tmp_path / "variance-burst.csv"
    pd.DataFrame({"time": np.arange(N) / FS, "current": _raw()}).to_csv(path, index=False)
    w = _window(qtbot)
    doc = open_document(path)
    did = w._session.add_document(doc)
    w._session.set_calibration(did, 0, ChannelCalibration(
        gain=1, offset=0, target_units="pA", provenance=Provenance.USER), write_sidecar=False)
    w._create_default_layers(doc)
    w._on_selection(did, 0)
    p = w._analysis_panel
    p.detection_panel.set_params({"polarity": -1, "noise_policy": "auto", "noise_window_s": .25,
                                  "filter_cutoff_hz": None, "min_dwell_s": .0005})
    p.baseline_panel.set_params({"baseline_open_pore_reference": {
        "mode": "constant", "current_pA": 100}})
    qtbot.wait(20)
    yield w, did, path
    w._analysis_controller.abort()
    w._controller.abort()


def _run(w, qtbot, *, gesture=False, **settings):
    p = w._analysis_panel
    p.detection_panel.set_params(settings)
    assert p._open_pore_reference_problem(p.detection_params()) is None
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        if gesture:
            assert p.detection_panel.detect_button.isEnabled()
            p.detection_panel.detect_button.click()
        else:
            assert p._run_detection(p.detection_params())
    qtbot.wait(25)
    result = p._stored_detection()
    assert result is not None
    return result


def _tracks(w):
    cache = w._analysis_panel._threshold_cache
    assert cache is not None
    outcome = cache[1]
    assert not isinstance(outcome, str), outcome
    return {r.label: r for r in outcome.rails}


@pytest.mark.parametrize("policy", ["auto", "local", "global"])
def test_controls_roundtrip_emit_and_use_seconds_while_displaying_milliseconds(qapp, qtbot, policy):
    form = DetectionPanel()
    qtbot.addWidget(form)
    with qtbot.waitSignal(form.paramsChanged):
        form.set_params({"noise_policy": policy, "noise_window_s": .125})
    assert form.params()["noise_policy"] == policy
    assert form.params()["noise_window_s"] == .125
    assert "125" in form.noise_window_s.text()
    assert form.noise_window_s.isEnabled() is (policy != "global")
    assert form.surrogate_params()["noise_policy"] == policy
    form.set_detector("detect.cusum")
    form.set_detector("detect.schmitt")
    assert form.params()["noise_policy"] == policy


def test_real_detect_button_uses_local_profile_and_rails_follow_it(noise_window, qtbot):
    w, did, _path = noise_window
    before = w._session.document(did).raw(0).copy()
    result = _run(w, qtbot, gesture=True)
    prov = result.payload.provenance
    assert result.params["noise_policy"] == "auto"
    assert prov["noise_policy_effective"] == "local"
    assert prov["noise_local_scale_max_pA"] > 5 * prov["noise_local_scale_min_pA"]
    assert prov["threshold_out_pA"] is None
    assert depth_floor_from_provenance(prov, column="depth_pA") is None
    assert any("SNR" in warning for warning in result.warnings)
    assert len(result.payload.df) == 2
    np.testing.assert_allclose(result.payload.df.snr, result.payload.df.depth_pA / prov["noise_rms_pA"])
    w._act_thresholds.setChecked(True)
    rails = _tracks(w)
    assert set(rails) == {RAIL_ENTER, RAIL_EXIT}
    assert w._trace_view.has_threshold_tracks(did, 0)
    assert "local sigma" in w._analysis_panel.detection_thresholds_note()
    for rail, multiplier in ((rails[RAIL_ENTER], prov["k_in"]), (rails[RAIL_EXIT], prov["k_out"])):
        ends = np.asarray(prov["noise_profile_tile_ends"])
        scales = np.asarray(prov["noise_profile_scales_pA"])
        expected = 100 - multiplier * scales[np.searchsorted(ends, rail.idx, side="right")]
        np.testing.assert_allclose(rail.values, expected, rtol=1e-3, atol=.001)
    np.testing.assert_array_equal(w._session.document(did).raw(0), before)


def test_policy_change_recomputes_detection_and_visible_rails(noise_window, qtbot):
    w, _did, _path = noise_window
    local = _run(w, qtbot)
    w._act_thresholds.setChecked(True)
    local_key = w._analysis_panel._threshold_cache[0]
    local_rail = _tracks(w)[RAIL_ENTER].values.copy()
    global_result = _run(w, qtbot, noise_policy="global")
    global_prov = global_result.payload.provenance
    assert global_prov["noise_policy_effective"] == "global"
    assert len(global_result.payload.df) > len(local.payload.df)
    assert w._analysis_panel._threshold_cache[0] != local_key
    rail = _tracks(w)[RAIL_ENTER].values
    np.testing.assert_allclose(rail, 100 - global_prov["threshold_in_pA"], atol=.001)
    assert np.ptp(local_rail) > 20 and np.ptp(rail) < .001


@pytest.mark.parametrize("legacy", [False, True, "baseline_only", "no_params"])
def test_project_restores_own_policy_and_legacy_uses_global(noise_window, qtbot, tmp_path, legacy):
    w, _did, path = noise_window
    result = _run(w, qtbot, noise_policy="global" if legacy else "local", noise_window_s=.125)
    project = tmp_path / ("legacy.nanoscope" if legacy else "local.nanoscope")
    assert w._write_project(str(project))
    if legacy:
        state = load_project(project)
        for record in state.results:
            record.params = {k: v for k, v in record.params.items() if not k.startswith(NOISE_PREFIXES)}
            record.provenance = {k: v for k, v in record.provenance.items() if not k.startswith(NOISE_PREFIXES)}
            if legacy == "baseline_only":
                record.params = {k: v for k, v in record.params.items() if k.startswith("baseline_")}
            elif legacy == "no_params":
                record.params = {}
        save_project(project, state)
    reopened = _window(qtbot)
    try:
        with qtbot.waitSignal(reopened.projectRestored, timeout=30_000):
            reopened._open_project(str(project))
        qtbot.wait(30)
        did = reopened._session.doc_id_for_source(str(path.resolve()))
        reopened._on_selection(did, 0)
        qtbot.wait(20)
        p = reopened._analysis_panel
        restored = p._stored_detection()
        assert restored is not None
        expected = "global" if legacy else "local"
        assert p.detection_panel.params()["noise_policy"] == expected
        assert p._rerun_params(restored)["noise_policy"] == expected
        pd.testing.assert_frame_equal(restored.payload.df, result.payload.df)
        reopened._act_thresholds.setChecked(True)
        assert reopened._trace_view.has_threshold_tracks(did, 0), p.detection_thresholds_note()
        if not legacy:
            assert p.detection_panel.params()["noise_window_s"] == .125
            assert restored.payload.provenance["noise_profile_sha256"] == result.payload.provenance["noise_profile_sha256"]
    finally:
        reopened._analysis_controller.abort()
        reopened._controller.abort()


def test_highpass_localization_exports_profile_and_keeps_measurement_snr(noise_window, qtbot, tmp_path):
    w, did, path = noise_window
    # A supplied physical I0 cannot be applied to the high-pass candidate trace.
    w._analysis_panel.baseline_panel.set_params({"baseline_open_pore_reference": None})
    result = _run(w, qtbot, noise_policy="local", hp_enabled=True, hp_cutoff_hz=5.)
    prov = result.payload.provenance
    assert prov["noise_policy_effective"] == "local"
    assert prov["highpass_localization_only"] is True
    assert prov["threshold_in_pA"] is None and prov["threshold_out_pA"] is None
    assert "profile" in prov["threshold_noise_source"]
    assert depth_floor_from_provenance(prov, column="depth_pA") is None
    assert len(result.payload.df) >= 2
    np.testing.assert_allclose(result.payload.df.snr, result.payload.df.depth_pA / prov["measurement_noise_rms_pA"])
    w._act_thresholds.setChecked(True)
    assert not w._trace_view.has_threshold_tracks(did, 0)
    assert "high-passed" in w._analysis_panel.detection_thresholds_note()
    out = write_training_export(result.payload.df, tmp_path / "local-highpass.csv", provenance=prov,
                                source_file=str(path), detector=result.module_id, include_all=True)
    sidecar = json.loads(out["meta"].read_text())
    contract = sidecar["detrend_contract"]
    for key in ("noise_policy", "noise_policy_effective", "noise_window_s", "noise_window_max_samples", "noise_algorithm_version",
                "noise_profile_tile_ends", "noise_profile_scales_pA", "noise_profile_sha256",
                "threshold_in_pA", "threshold_out_pA", "threshold_noise_source"):
        assert contract[key] == prov[key]
    assert len(pd.read_csv(out["data"])) == len(result.payload.df)
