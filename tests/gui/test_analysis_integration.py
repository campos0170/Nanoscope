"""End-to-end GUI integration (PORT_PLAN §6/§8, offscreen).

Detection is driven THROUGH THE APP PATH — a real :class:`MainWindow`, its
:class:`AnalysisController`, its :class:`AnalysisPanel` and the shared
:class:`TraceView` — exactly as a click would, and the results are asserted on
the live widgets:

* a synthetic five-event document: run detection, assert the five events land in
  the Session's ResultStore, the EventTable fills, and ONE batched
  :class:`SpanOverlayItem` (not five region items) is painted on the current
  channel's pane;
* clicking an event row zooms the trace and drops the single MARKER overlay;
* revising the calibration marks the result stale (the user's deferred
  current-gain workflow);
* ``requires_time_base`` disables detection honestly on a rate-unknown document;
* a REAL Axon ABF opened offscreen detects events through the same path.

``app.main()`` registers the modules + renderers before building the window; the
``qapp`` conftest fixture does the same for tests, so the seam is live here.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import time

import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.analysis.events import EventSet
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import SpanOverlayItem

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]
_ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0004_"
            "process _ -1000 mV_ 385-400 s time trace.abf")


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path, *, fs: float | None = FS, n: int = 100_000,
               name: str = "pore_p1.dat") -> TraceDocument:
    arr = _event_trace(n)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=arr.size,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(arr,))


# A drifting variant for the baseline-wander-removal (detrend) path: the five events
# ride on a slow open-pore wander. n=300_000 -> 3 s at FS, so a 0.3 s detrend window
# forms ~10 blocks.
_DRIFT_EVENTS = [(30_000, 30_300), (90_000, 90_250), (150_000, 150_400),
                 (210_000, 210_200), (270_000, 270_350)]


def _drift_trace(n: int = 300_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    drift = 45.0 * np.sin(2 * np.pi * 0.5 * t) + 30.0 * (t / t[-1])
    x = 100.0 + drift + rng.normal(0.0, 1.5, n)
    for s, e in _DRIFT_EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _drift_doc(tmp_path, name: str = "drift.dat") -> TraceDocument:
    arr = _drift_trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Add a document to the window the way the load pipeline would, then select
    its current channel — the real selection path (calibration/file-info/analysis
    panels all repoint, the Run action re-gates)."""
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _run_detection(window: MainWindow, qtbot, *, min_dwell_s: float = 5e-4) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)   # let the queued add_result -> resultsChanged -> overlay draw settle


def _span_items(pane) -> list[SpanOverlayItem]:
    return [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# the core end-to-end path
# ================================================================================================

def test_detection_through_app_path_stores_result_and_paints_spans(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))

    # Run action is enabled for a CURRENT channel on a rate-known doc.
    assert window._act_run_detection.isEnabled()

    _run_detection(window, qtbot)

    # 1) the result is in the Session's ResultStore (installed on the main thread).
    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert res is not None
    assert isinstance(res.payload, EventSet)
    assert len(res.payload) == len(_EVENTS)

    # 2) the EventTable filled.
    assert window._analysis_panel._table._proxy.rowCount() == len(_EVENTS)

    # 3) exactly ONE batched span item on the current channel's pane (never N region items).
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert pane is not None
    spans = _span_items(pane)
    assert len(spans) == 1
    assert spans[0].n_spans == len(_EVENTS)
    assert not [it for it in pane.plot.items if isinstance(it, pg.LinearRegionItem)]

    # 4) the overlay id is tracked so a re-run/close can remove it.
    assert window._analysis_panel._overlay_ids


def _two_channel_doc(tmp_path) -> TraceDocument:
    """A voltage channel (index 0) AND a current channel (index 1) — the real ABF shape."""
    cur = _event_trace()
    volt = np.full(cur.size, -600.0, dtype=np.float32)
    v = ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    c = ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "two_ch.dat", format_name="synthetic", n_samples=cur.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(v, c))
    return TraceDocument(meta=meta, channels=(volt, cur))


def test_analysis_sticks_to_current_channel_when_voltage_selected(window, qtbot, tmp_path):
    """Regression: clicking the VOLTAGE channel (e.g. to calibrate it) must NOT move detection off
    the current channel and grey out Detect. Analysis is a current-channel operation and stays put."""
    doc = _two_channel_doc(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)

    # Select the VOLTAGE channel (index 0), as clicking that row would.
    window._on_selection(doc_id, 0)
    assert window._analysis_panel._sel == (doc_id, 1), "analysis abandoned the current channel"
    assert window._act_run_detection.isEnabled(), "Detect greyed out after selecting voltage"

    # Selecting the dead/other channels likewise stays on current; detection still runs on it.
    window._on_selection(doc_id, 1)
    assert window._analysis_panel._sel == (doc_id, 1)
    _run_detection(window, qtbot)
    assert window._session.results.get(doc_id, 1, "detect.schmitt") is not None


def test_rerun_replaces_overlays_not_accumulates(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    _run_detection(window, qtbot)   # second run

    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    # A re-run must remove the previous spans, not stack a second batched item.
    assert len(_span_items(pane)) == 1
    assert len(window._analysis_panel._overlay_ids) == 1


def test_event_row_click_zooms_and_marks(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    panel = window._analysis_panel
    anchor = window._trace_view.anchor_plot()
    before = tuple(anchor.vb.viewRange()[0])

    panel._table.view.selectRow(0)
    qtbot.wait(10)

    # a MARKER overlay was dropped on the dived-into event...
    assert panel._marker_id is not None
    assert panel._marker_id in window._trace_view._overlays
    # ...and the trace zoomed in (a narrower x-window than the full trace).
    after = tuple(anchor.vb.viewRange()[0])
    assert (after[1] - after[0]) < (before[1] - before[0])


def test_calibration_change_marks_result_stale(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    panel = window._analysis_panel
    assert panel._stale.isHidden()   # fresh result: not stale

    # The user revises the current gain (the explicitly deferred workflow).
    window._session.set_calibration(
        doc_id, channel,
        ChannelCalibration(gain=2.0, offset=0.0, target_units="pA",
                           provenance=Provenance.UNKNOWN))
    qtbot.wait(10)
    assert not panel._stale.isHidden()   # banner now offers a re-run


def test_requires_time_base_disables_detection(window, qtbot, tmp_path):
    """A rate-unknown document (legacy TDMS shape) must disable detection with an
    honest reason, never fabricate a dwell time."""
    doc_id, channel = _install(
        window, _synth_doc(tmp_path, fs=None, n=8_000, name="no_rate.dat"))

    # Menu action AND the panel's Detect button are both gated off.
    assert not window._act_run_detection.isEnabled()
    assert not window._analysis_panel._detection.detect_button.isEnabled()
    assert not window._analysis_panel._detection._gate_label.isHidden()

    # Even if forced, the run is a no-op (no result appears).
    window._analysis_panel.trigger_detection()
    qtbot.wait(30)
    assert window._session.results.get(doc_id, channel, "detect.schmitt") is None


def test_selection_change_moves_overlays(window, qtbot, tmp_path):
    """Switching away removes the current selection's overlays; switching back
    re-draws them from the stored result (overlays follow the selection)."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert len(_span_items(pane)) == 1

    # Switch to "no channel" (a closed/absent doc id) then back.
    window._analysis_panel.set_selection("does-not-exist", 0)
    assert len(_span_items(pane)) == 0

    window._on_selection(doc_id, channel)
    qtbot.wait(10)
    assert len(_span_items(pane)) == 1


# ================================================================================================
# baseline-drift removal through the app path
# ================================================================================================

def test_detrend_through_app_records_provenance_and_finds_events(window, qtbot, tmp_path):
    """Enabling baseline-drift removal in the panel feeds the flattened signal to
    detection: the result's provenance records the transform and every event is still
    found on the drifting trace."""
    doc_id, channel = _install(window, _drift_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.detrend_enabled.setChecked(True)
    panel._detection.set_detrend_method("block_median")
    panel._detection.detrend_window_s.setValue(0.3)

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert res is not None
    prov = res.payload.provenance
    assert prov["detrend_enabled"] is True
    assert prov["detrend_method"] == "block_median"
    assert prov["detrend_window_s"] == pytest.approx(0.3)
    assert len(res.payload) == len(_DRIFT_EVENTS)


def test_truth_kwargs_strip_detrend_keys():
    """The surrogate studies (sweep/null/recovery) bake the detrend into the wrapped
    signal, then re-detect. The detrend keys must be stripped from the detect kwargs —
    detect_events has no such parameters and would raise a TypeError."""
    from nanoscope.gui.analysis_panel import AnalysisPanel

    params = {"polarity": None, "k_in": 5.0, "min_dwell_s": 5e-4,
              "detrend_enabled": True, "detrend_method": "asls", "detrend_window_s": 0.4,
              "filter_cutoff_hz": None, "merge_gap_s": None, "max_dwell_s": None}
    kw = AnalysisPanel._detect_kwargs_for_truth(params)
    for k in ("detrend_enabled", "detrend_method", "detrend_window_s", "polarity", "k_in"):
        assert k not in kw
    assert kw["min_dwell_s"] == 5e-4


def test_detrend_preview_curve_reflects_flattened_signal(window, qtbot, tmp_path):
    """The 'show on trace' preview draws the ACTUAL preprocessed signal detection sees.
    With baseline removal on and the low-pass off, that is exactly
    remove_baseline_wander of the raw channel."""
    from nanoscope.analysis.baseline import remove_baseline_wander

    doc_id, channel = _install(window, _drift_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.detrend_enabled.setChecked(True)   # detrend only; low-pass stays off
    panel._detection.detrend_window_s.setValue(0.3)
    panel._detection.filter_preview.setChecked(True)    # show it on the trace
    qtbot.wait(20)

    tv = window._trace_view
    assert tv.has_filter_preview()
    curve = tv._filter_preview
    assert curve is not None

    ch = window._session.calibrated(doc_id, channel)
    expected = remove_baseline_wander(np.asarray(ch.data), FS, window_s=0.3)
    got = np.asarray(curve.yData)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-3)


# ================================================================================================
# the rest of the filter suite through the app path
# ================================================================================================

def test_notch_through_app_records_provenance_and_finds_events(window, qtbot, tmp_path):
    """Enabling the notch in the panel feeds the notched signal to detection: the result's
    provenance records the transform and every (fast) event still survives it."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.set_notch_enabled(True)
    panel._detection.set_notch_harmonics(2)
    panel._detection.set_notch_q(25.0)

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert res is not None
    prov = res.payload.provenance
    assert prov["notch_enabled"] is True
    assert prov["notch_harmonics"] == 2
    assert prov["notch_q"] == pytest.approx(25.0)
    assert len(res.payload) == len(_EVENTS)


def test_lowpass_through_app_path_runs_iir_filter_on_worker(window, qtbot, tmp_path):
    """Regression for the worker stack size. The low-pass runs zero-phase IIR filtering
    (scipy sosfiltfilt) INSIDE detect_events on the analysis worker thread — the same
    Accelerate/LAPACK path notch and high-pass take. On a default-stack QThread that
    SIGBUSed the whole process; the worker now sizes its stack for it. Enabling the
    low-pass and detecting must simply work and record the measured rise time."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.filter_enabled.setChecked(True)
    panel._detection.filter_cutoff_hz.setValue(20_000.0)

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert res is not None
    assert len(res.payload) == len(_EVENTS)
    # The detector stamps the rise time it measured through the exact filter it applied.
    assert res.payload.provenance.get("filter_rise_s")


def test_truth_kwargs_strip_filter_keys():
    """The surrogate studies (sweep/null/recovery) bake notch/high-pass/savgol into the
    wrapped signal, then re-detect. Those keys must be stripped from the detect kwargs —
    detect_events has no such parameters and would raise a TypeError."""
    from nanoscope.gui.analysis_panel import AnalysisPanel

    params = {"polarity": None, "k_in": 5.0, "min_dwell_s": 5e-4,
              "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 3, "notch_q": 30.0,
              "hp_enabled": True, "hp_cutoff_hz": 100.0,
              "savgol_enabled": True, "savgol_window_s": 1e-3, "savgol_polyorder": 3,
              "filter_cutoff_hz": None, "merge_gap_s": None, "max_dwell_s": None}
    kw = AnalysisPanel._detect_kwargs_for_truth(params)
    for k in ("notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q",
              "hp_enabled", "hp_cutoff_hz",
              "savgol_enabled", "savgol_window_s", "savgol_polyorder", "polarity", "k_in"):
        assert k not in kw
    assert kw["min_dwell_s"] == 5e-4


def test_filter_chain_preview_curve_reflects_preprocessed_signal(window, qtbot, tmp_path):
    """The 'show on trace' preview draws the ACTUAL preprocessed signal detection sees.
    With notch + Savitzky-Golay on and the low-pass off, that is exactly
    preprocess_for_detection of the raw channel — the whole chain, not just one stage."""
    from nanoscope.analysis.modules import preprocess_for_detection

    doc_id, channel = _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.set_notch_enabled(True)         # notch + savgol; low-pass stays off
    panel._detection.set_notch_harmonics(1)
    panel._detection.set_savgol_enabled(True)
    panel._detection.set_savgol_window_s(2e-4)
    panel._detection.filter_preview.setChecked(True)  # show it on the trace
    qtbot.wait(20)

    tv = window._trace_view
    assert tv.has_filter_preview()
    curve = tv._filter_preview
    assert curve is not None

    ch = window._session.calibrated(doc_id, channel)
    expected = preprocess_for_detection(np.asarray(ch.data), FS, panel._detection.params())
    got = np.asarray(curve.yData)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-3)


# ================================================================================================
# a real ABF through the same path
# ================================================================================================

@pytest.mark.skipif(not _ABF.exists(), reason="corpus ABF not present")
def test_real_abf_detection_through_app_path(window, qtbot):
    from nanoscope.io.base import open_document

    doc = open_document(_ABF, None)      # the same reader the LoadWorker calls
    doc_id, channel = _install(window, doc)
    assert doc.meta.channels[channel].kind is ChannelKind.CURRENT

    _run_detection(window, qtbot, min_dwell_s=5e-5)

    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert res is not None
    events = res.payload
    assert isinstance(events, EventSet)
    assert len(events) > 0, "detection found no events in the real recording"

    # spans painted on the real current pane, one batched item covering every event.
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert pane is not None
    spans = _span_items(pane)
    assert len(spans) == 1
    assert spans[0].n_spans == len(events)


# ================================================================================================
# The user's own filter parameters, end to end through the app path
# ================================================================================================

def test_selected_lowpass_design_reaches_the_detector_and_lands_in_provenance(
        window, qtbot, tmp_path):
    """The panel's type/order controls are not decorative: they steer the filter the
    detector applies AND the rise time it measures through it, and both are recorded."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.filter_enabled.setChecked(True)
    panel._detection.filter_cutoff_hz.setValue(5_000.0)
    panel._detection.set_filter_kind("butter")
    panel._detection.set_filter_order(2)

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    prov = window._session.results.get(doc_id, channel, "detect.schmitt").payload.provenance
    assert prov["filter_kind"] == "butter"
    assert prov["filter_order"] == 2
    assert prov["filter_zero_phase"] is True
    # Measured through THAT design, not the 8-pole Bessel default.
    from nanoscope.analysis.filters import measure_rise_time
    assert prov["filter_rise_s"] == pytest.approx(
        measure_rise_time(FS, 5_000.0, order=2, kind="butter"), rel=1e-9)


def test_gaussian_lowpass_runs_end_to_end(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.filter_enabled.setChecked(True)
    panel._detection.filter_cutoff_hz.setValue(5_000.0)
    panel._detection.set_filter_kind("gaussian")

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert len(res.payload) == len(_EVENTS)
    prov = res.payload.provenance
    assert prov["filter_kind"] == "gaussian"
    assert prov["filter_order"] is None      # a symmetric kernel has no order


def test_high_pass_design_reaches_preprocessing_and_lands_in_provenance(
        window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.set_hp_enabled(True)
    panel._detection.set_hp_cutoff_hz(50.0)
    panel._detection.set_hp_kind("bessel")
    panel._detection.set_hp_order(8)

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    prov = window._session.results.get(doc_id, channel, "detect.schmitt").payload.provenance
    assert prov["hp_enabled"] is True
    assert prov["hp_kind"] == "bessel"
    assert prov["hp_order"] == 8


def test_detrend_shape_parameters_reach_the_estimator_and_land_in_provenance(
        window, qtbot, tmp_path):
    """ASLS's smoothness used to be frozen at 6e4 with no way to retune it."""
    doc_id, channel = _install(window, _drift_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.detrend_enabled.setChecked(True)
    panel._detection.set_detrend_method("asls")
    panel._detection.detrend_window_s.setValue(0.3)
    panel._detection.detrend_asls_lam.setValue(1e6)
    panel._detection.detrend_asls_p.setValue(0.01)

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    prov = window._session.results.get(doc_id, channel, "detect.schmitt").payload.provenance
    assert prov["detrend_method"] == "asls"
    assert prov["asls_lam"] == pytest.approx(1e6)
    assert prov["asls_p"] == pytest.approx(0.01)


def test_block_median_detrend_records_no_shape_parameters(window, qtbot, tmp_path):
    """It genuinely has none — provenance must not invent them."""
    doc_id, channel = _install(window, _drift_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.detrend_enabled.setChecked(True)
    panel._detection.set_detrend_method("block_median")
    panel._detection.detrend_window_s.setValue(0.3)

    _run_detection(window, qtbot, min_dwell_s=5e-4)

    prov = window._session.results.get(doc_id, channel, "detect.schmitt").payload.provenance
    assert prov["detrend_method"] == "block_median"
    for key in ("percentile", "asls_lam", "asls_p", "asls_niter", "asls_decim"):
        assert key not in prov


def test_panel_ranges_follow_the_selected_recordings_nyquist(window, tmp_path):
    """Selecting a channel re-ranges the frequency spins; the panel is told fs by the
    same set_recording path that drives the rise-time readout."""
    _install(window, _synth_doc(tmp_path))       # FS = 100 kHz -> Nyquist 50 kHz
    panel = window._analysis_panel._detection
    assert 49_000.0 < panel.filter_cutoff_hz.maximum() < 50_000.0
    assert 49_000.0 < panel.hp_cutoff_hz.maximum() < 50_000.0
    assert 49_000.0 < panel.notch_freq_hz.maximum() < 50_000.0


def test_filter_preview_uses_the_users_design_not_the_kernel_default(
        window, qtbot, tmp_path):
    """The 'show processed trace' curve must be the filter detection will apply — a
    preview of a different filter is worse than no preview."""
    from nanoscope.analysis.filters import lowpass

    doc_id, channel = _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.filter_enabled.setChecked(True)
    panel._detection.filter_cutoff_hz.setValue(5_000.0)
    panel._detection.set_filter_kind("butter")
    panel._detection.set_filter_order(2)
    panel._detection.filter_preview.setChecked(True)
    qtbot.wait(20)

    assert window._trace_view.has_filter_preview()
    drawn = np.asarray(window._trace_view._filter_preview.yData, dtype=np.float64)

    raw = np.asarray(window._session.calibrated(doc_id, channel).data)
    expected = np.asarray(lowpass(raw, FS, 5_000.0, order=2, kind="butter"), dtype=np.float64)
    default = np.asarray(lowpass(raw, FS, 5_000.0), dtype=np.float64)
    # The two designs really are different, so the assertion below has teeth.
    assert not np.allclose(expected, default)
    assert drawn.size == expected.size, "the preview is decimated; compare like for like"
    assert np.allclose(drawn, expected, atol=1e-3)
    assert not np.allclose(drawn, default, atol=1e-3)


# ================================================================================================
# Sweep / Null control / Recovery map no longer die silently on a bad preprocessing value
# ================================================================================================

@pytest.mark.parametrize("handler,label", [
    ("_on_sweep_requested", "Sweep threshold"),
    ("_on_null_requested", "Null control"),
    ("_on_recovery_requested", "Recovery map"),
])
def test_surrogate_studies_report_a_bad_preprocessing_value(
        window, tmp_path, monkeypatch, qapp, handler, label):
    """A kernel ValueError in the conditioning chain reaches the user, and does so through
    the SAME door Detect uses.

    History, because the route changed and the guarantee did not. The chain used to run
    inline on the GUI THREAD, so a ValueError escaped into the Qt slot: PySide6 printed a
    traceback and the button did NOTHING, forever, with no dialog — while the identical
    value produced a proper 'Analysis failed' box on Detect. That was first fixed with a
    try/except and a panel-owned QMessageBox, which cured the silence but kept two
    different experiences for one error AND kept the freeze (measured 81-361 ms on the
    owner's smallest ABF, linear in n). The chain now runs INSIDE the job, so the worker
    reports it exactly as it reports a bad Detect parameter: one ``failed`` signal, one
    message naming the real problem, ``_pending`` cleared, no phantom run left in flight.
    """
    from PySide6 import QtWidgets

    _install(window, _synth_doc(tmp_path))
    panel = window._analysis_panel
    # MainWindow answers `failed` with a modal; capture it rather than face it.
    shown: list[str] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self, *a, **k: shown.append(self.text()))

    params = dict(panel._detection.params())
    params["detrend_enabled"] = True
    params["detrend_window_s"] = 60.0        # far longer than the 1 s recording

    getattr(panel, handler)(params)
    deadline = time.time() + 10.0
    while panel._pending is not None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert len(shown) == 1, f"{label} produced no message"
    assert "longer than the recording" in shown[0]
    assert panel._pending is None            # and no phantom run was left in flight


# ================================================================================================
# CUSUM sub-state level fitting: an ADDED capability, and only ever user-initiated.
# ================================================================================================

def test_level_fitting_never_runs_by_itself(window, qtbot, tmp_path):
    """THE contract for this feature. Detection must not produce a level fit.

    CUSUM is a second opinion on n_levels with its own parameters. One that arrives
    without being asked for is indistinguishable, to the user, from the app changing its
    mind about their data — so it is not an AnalysisModule, not in the fit dispatch list,
    and not wired to any signal. Only the menu action runs it.
    """
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    assert window._level_fit is None, "detection must not fit levels"
    assert not window._act_export_levels.isEnabled()
    assert window._session.results.get(
        window._analysis_panel.current_detection().doc_id, 0, "fit.levels") is None
    # ...and the registry still holds only the detectors.
    #
    # THE NAMING TRAP, and why this assertion is spelled the way it is. The claim being
    # pinned is "the LEVEL FITTER is not a registered analysis module" -- level fitting is
    # a deliberate, explicit action, never something that runs by itself off the back of a
    # detection. It used to be spelled `not any("cusum" in m)`, which was a fine proxy
    # while `analysis/cusum.py` (the sub-state LEVEL fitter) was the only thing in the tree
    # with that word in its name. It stopped being one when a CUSUM-based EVENT DETECTOR
    # shipped under "detect.cusum": that IS a detector and DOES belong in the registry, so
    # the old spelling failed a test about level fitting because of an unrelated module
    # that merely shares an algorithm name.
    from nanoscope.core.analysis import ANALYSIS_REGISTRY
    assert "fit.levels" not in ANALYSIS_REGISTRY
    assert not any(m.startswith("fit.") for m in ANALYSIS_REGISTRY)
    assert "detect.cusum" in ANALYSIS_REGISTRY, "the CUSUM DETECTOR is a detector"


def test_the_fit_action_is_gated_on_having_a_detection(window, qtbot, tmp_path):
    assert not window._act_fit_levels.isEnabled(), "nothing to fit before a detection"
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    assert window._act_fit_levels.isEnabled()


def test_running_the_action_fits_levels_and_enables_the_export(window, qtbot, tmp_path):
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)

    res = window._level_fit
    assert res is not None and res.module_id == "fit.levels"
    assert not res.payload.empty
    assert window._act_export_levels.isEnabled()
    # It is stored like any other derived fit, and carries BOTH staleness stamps.
    det = window._analysis_panel.current_detection()
    assert window._session.results.get(det.doc_id, det.channel, "fit.levels") is not None
    assert res.calibration_fingerprint is not None
    assert res.baseline_fingerprint, "a level depth is measured against a baseline"
    assert res.params["cusum_fingerprint"]


def test_the_fit_reaches_an_open_event_inspector(window, qtbot, tmp_path):
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._open_event_inspector()
    insp = window._event_inspector
    assert "CUSUM levels" not in insp._readout.text()

    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)
    assert "CUSUM levels" in insp._readout.text()


def test_a_new_detection_drops_the_previous_level_fit(window, qtbot, tmp_path):
    """A level table belongs to ONE event table. Carried across a re-detection it would
    export under the wrong provenance and match another run's event_ids."""
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)
    assert window._level_fit is not None

    _run_detection(window, qtbot)          # a second, different detection
    assert window._level_fit is None, "a stale level fit must not survive a re-detection"
    assert not window._act_export_levels.isEnabled()


def test_exported_levels_carry_the_cusum_parameters(window, qtbot, tmp_path, monkeypatch):
    import json

    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)

    target = tmp_path / "levels.csv"
    monkeypatch.setattr("nanoscope.gui.main_window.ask_export_path",
                        lambda *a, **k: target)
    window._export_levels()

    assert target.exists()
    meta = json.loads((tmp_path / "levels.meta.json").read_text())
    assert meta["format"] == "nanoscope-levels"
    assert meta["layout"] == "long"
    # The numbers that decide how many levels an event HAS must travel with the table.
    for key in ("step_sigma", "threshold_sigma", "min_level_samples", "cusum_fingerprint"):
        assert key in meta["cusum_params"], key


def test_exporting_without_a_fit_explains_rather_than_writing(window, qtbot, tmp_path, monkeypatch):
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    called = []
    monkeypatch.setattr("nanoscope.gui.main_window.ask_export_path",
                        lambda *a, **k: called.append(1))
    shown = []
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a)))
    window._export_levels()
    assert not called, "must not open a save dialog with nothing to write"
    assert shown, "must say why"

# ------------------------------------------------------------------------------------------------
# The fitted levels reach the MAIN TRACE (owner note #18): OverlayKind.STEP finally has a
# producer (AnalysisPanel._level_step_hint) and a renderer (overlays.render_step).
# ------------------------------------------------------------------------------------------------

def _step_overlay_items(window) -> list:
    """The STEP overlay's PlotDataItems among the trace view's tracked overlays.

    Identified by zValue 6 (render_step's, above spans / below the marker) — the only
    other PlotDataItem an overlay can be is the HLINE baseline track at zValue -4."""
    out = []
    for items in window._trace_view._overlays.values():
        for it in items:
            if isinstance(it, pg.PlotDataItem) and it.zValue() == 6:
                out.append(it)
    return out


def test_the_fit_draws_step_segments_on_the_main_trace(window, qtbot, tmp_path):
    """Sub-state levels used to show ONLY in the Event Inspector; the main panel showed
    the same events with no levels at all. The fit must land on the trace the moment it
    completes — one batched item, y values equal to the stored table's level_pA."""
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    assert _step_overlay_items(window) == [], "no fit yet, so no step overlay"

    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)
    items = _step_overlay_items(window)
    assert len(items) == 1, "one batched STEP item, not one per level"
    xs, ys = items[0].getData()
    levels = window._level_fit.payload
    # NaN-separated triplets: 3 points per level, drawn at that level's own current.
    assert xs.size == 3 * len(levels)
    drawn = ys[0::3]
    assert np.allclose(np.sort(drawn), np.sort(levels["level_pA"].to_numpy()))


def test_the_step_overlay_retires_with_the_fit(window, qtbot, tmp_path):
    """The store drop() emits no signal — the explicit redraw_overlays() call in
    _invalidate_level_fit is what takes the segments off the trace. A re-detection
    invalidates the fit, so the overlay must be gone right then, not on the next edit."""
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)
    assert len(_step_overlay_items(window)) == 1

    _run_detection(window, qtbot)          # drops the fit via _invalidate_level_fit
    assert _step_overlay_items(window) == [], "a retired fit must leave the trace"


def test_the_step_overlay_follows_curation(window, qtbot, tmp_path):
    """Levels are drawn for ACCEPTED events only — the same set whose spans are shaded.
    Reject everything and both spans and levels must clear."""
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)
    assert len(_step_overlay_items(window)) == 1

    panel = window._analysis_panel
    df = panel.current_detection().payload.df
    df["accepted"] = False
    panel.resync_curation()
    assert _step_overlay_items(window) == []
