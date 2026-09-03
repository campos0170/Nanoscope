"""GUI + module wiring for the runtime-loaded Keras ML detector (Phase 2, offscreen).

The ML detector produces the SAME :class:`~nanoscope.analysis.events.EventSet` the
Schmitt detector does and must flow into the identical pipeline (event table, span
overlays, event inspector, plots, exports). These tests prove that end to end.

Strategy (mirrors the ml_detect Phase 1 split so the pytest-qt suite never imports
TensorFlow -- which aborts the shiboken/pytest-qt harness):

* **Panel plumbing** -- the DetectionPanel's ML controls (Load-model label, run
  gating, param dict, exit<=enter clamp, the TF-unavailable message). Pure view,
  no TF, always runs.
* **Through the controller with a FAKE model** -- the whole app path (panel ->
  AnalysisController off-thread -> MLDetect -> characterize -> AnalysisResult ->
  ResultStore -> EventTable + batched SpanOverlayItem), driven by a fake Keras
  model injected by monkeypatching ``load_keras_model`` and forcing
  ``TENSORFLOW_AVAILABLE`` True. Imports ZERO TensorFlow, so it runs on CI.
* **The REAL model, in a SUBPROCESS** -- guarded on TF + the model file. A
  standalone (non-pytest-qt) Qt process loads the real ``.keras`` and runs ML
  detection through the real controller + AnalysisPanel, asserting an EventSet
  lands in the ResultStore and the spans paint. Isolated in a child process
  exactly as the Phase 1 ml_detect corpus test is, and for the same reason.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis import ml_detect
from nanoscope.analysis.events import EventSet
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import SpanOverlayItem
from nanoscope.gui.panels import DetectionPanel

FS = 100_000.0
FAKE_W = 3072                                   # match the real model's window size
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]

MODEL_PATH = Path("/Users/abrahamcampos/Downloads/student_raw_global_raw_best.keras")
ABF_PATH = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
_HAVE_REAL = ml_detect.TENSORFLOW_AVAILABLE and MODEL_PATH.exists() and ABF_PATH.exists()


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
               name: str = "pore_ml.dat") -> TraceDocument:
    arr = _event_trace(n)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=fs, rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


class _FakeEventModel:
    """A stand-in Keras model that fires (prob ~1) on the deep, normalised event
    samples and stays ~0 on the open pore.

    ``ml_detect_regions`` feeds it globally-normalised windows: after detrending, the
    ~-30 pA blockades dominate the max-abs, so they normalise to ~-1 while the noise
    sits near 0. Thresholding ``|x| > 0.5`` therefore recovers exactly the injected
    events -- a deterministic detector with no TensorFlow involved."""

    def __init__(self, window: int) -> None:
        self.window = window

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=np.float32)           # (B, W, 1)
        return (np.abs(x) > 0.5).astype(np.float16)


def _fake_wrapper(window: int = FAKE_W) -> ml_detect.KerasEventModel:
    return ml_detect.KerasEventModel(
        model=_FakeEventModel(window), window=window, path="<fake>", mtime=0.0,
        input_shape=(None, window, 1), output_shape=(None, window, 1))


def _use_fake_model(monkeypatch, wrapper=None) -> ml_detect.KerasEventModel:
    """Make the whole ML path run with a fake model and NO TensorFlow: pretend TF is
    importable (so the gate opens) and swap the loader for one returning the fake
    wrapper (so nothing is ever loaded from disk)."""
    wrapper = wrapper or _fake_wrapper()
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _path: wrapper)
    return wrapper


def _span_items(pane) -> list[SpanOverlayItem]:
    return [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]


def _run_ml(window: MainWindow, qtbot, *, range_source: str = "whole") -> None:
    panel = window._analysis_panel
    panel._detection.ml_range.setCurrentIndex(0 if range_source == "visible" else 1)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._run_ml_detection()
    qtbot.wait(20)   # let the queued add_result -> resultsChanged -> overlay draw settle


def _run_schmitt(window: MainWindow, qtbot, *, min_dwell_s: float = 5e-4) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# A. DetectionPanel ML controls — pure view, no TensorFlow.
# ================================================================================================

def test_load_model_updates_label_and_path(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)

    assert panel.model_path() is None
    assert panel.ml_model_label.text() == "none"

    panel.set_model_path("/some/dir/student_raw_global_raw_best.keras")
    assert panel.model_path() == "/some/dir/student_raw_global_raw_best.keras"
    assert panel.ml_model_label.text() == "student_raw_global_raw_best.keras"

    panel.set_model_path(None)   # clearing returns to "none"
    assert panel.model_path() is None
    assert panel.ml_model_label.text() == "none"


def test_ml_model_changed_signal_carries_path(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.mlModelChanged, timeout=1000) as blocker:
        panel.set_model_path("/x/y/m.keras")
    assert blocker.args == ["/x/y/m.keras"]


def test_run_ml_gated_on_model_and_channel(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)

    # No channel, no model -> disabled.
    assert not panel.ml_detect_button.isEnabled()

    # Channel OK but still no model -> disabled, and the gate says so.
    panel.set_ml_runnable(True, "")
    assert not panel.ml_detect_button.isEnabled()
    assert "model" in panel.ml_gate_label.text().lower()
    assert not panel.ml_gate_label.isHidden()

    # Model loaded AND channel OK -> enabled, gate hidden.
    panel.set_model_path("/x/model.keras")
    assert panel.ml_detect_button.isEnabled()
    assert panel.ml_gate_label.isHidden()

    # Channel goes not-runnable (e.g. voltage / no rate) -> disabled with the reason.
    panel.set_ml_runnable(False, "Detection runs on CURRENT channels; this is a voltage channel.")
    assert not panel.ml_detect_button.isEnabled()
    assert "voltage channel" in panel.ml_gate_label.text()


def test_run_ml_disabled_when_tensorflow_absent(qtbot, monkeypatch):
    """TensorFlow unavailable -> a clear disabled message, and neither loading nor
    running is offered (there is nothing a model could run on)."""
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", False)
    panel = DetectionPanel()
    qtbot.addWidget(panel)

    panel.set_ml_runnable(True, "")
    panel.set_model_path("/x/model.keras")   # even with a model + channel...

    assert not panel.ml_detect_button.isEnabled()    # ...run stays disabled
    assert not panel.ml_load_button.isEnabled()      # ...and loading is disabled
    assert "TensorFlow" in panel.ml_gate_label.text()
    assert not panel.ml_gate_label.isHidden()


def test_ml_params_carry_model_thresholds_and_range(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_model_path("/x/model.keras")
    panel.ml_enter.setValue(0.6)
    panel.ml_exit.setValue(0.25)
    panel.ml_range.setCurrentIndex(1)   # Whole recording

    params = panel.ml_params()
    assert params["model_path"] == "/x/model.keras"
    assert params["enter"] == pytest.approx(0.6)
    assert params["exit"] == pytest.approx(0.25)
    assert params["range_source"] == "whole"

    panel.ml_range.setCurrentIndex(0)   # Visible view (the default)
    assert panel.ml_params()["range_source"] == "visible"


def test_ml_exit_clamped_at_or_below_enter(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.ml_enter.setValue(0.4)
    panel.ml_exit.setValue(0.9)          # adversarial: exit above enter
    assert panel.ml_exit.value() <= panel.ml_enter.value()


def test_run_ml_button_emits_ml_detect_requested(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_ml_runnable(True, "")
    panel.set_model_path("/x/model.keras")

    with qtbot.waitSignal(panel.mlDetectRequested, timeout=1000) as blocker:
        panel.ml_detect_button.click()
    payload = blocker.args[0]
    assert payload["model_path"] == "/x/model.keras"
    assert "enter" in payload and "exit" in payload and "range_source" in payload


# ================================================================================================
# B. Through the app path with a FAKE model — no TensorFlow, runs everywhere.
# ================================================================================================

def test_ml_detection_through_app_stores_result_and_paints_spans(window, qtbot, tmp_path, monkeypatch):
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))

    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    assert window._analysis_panel.can_run_ml()
    window._sync_ml_detection_action()
    assert window._act_run_ml_detection.isEnabled()

    _run_ml(window, qtbot, range_source="whole")

    # 1) an ML result is in the ResultStore under its own module id.
    res = window._session.results.get(doc_id, channel, "detect.keras")
    assert res is not None
    assert isinstance(res.payload, EventSet)
    assert len(res.payload) == len(_EVENTS)
    assert res.payload.provenance["detector"] == "ml_keras"

    # 2) the ML result is the ACTIVE detection and the EventTable filled from it.
    assert window._analysis_panel._active_detection == "detect.keras"
    assert window._analysis_panel._table._proxy.rowCount() == len(_EVENTS)

    # 3) exactly ONE batched span item covering every event (the SAME overlay path
    #    Schmitt uses — never N region items).
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert pane is not None
    spans = _span_items(pane)
    assert len(spans) == 1
    assert spans[0].n_spans == len(_EVENTS)
    assert window._analysis_panel._overlay_ids

    # 4) the events landed at their true FULL-FILE sample coordinates.
    starts = res.payload.df["start_idx"].to_numpy()
    for (s, _e), got in zip(_EVENTS, sorted(starts)):
        assert abs(int(got) - s) <= 5


def test_ml_result_feeds_export_and_inspector_actions(window, qtbot, tmp_path, monkeypatch):
    """The ML result lights up the same MainWindow actions a Schmitt result does."""
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    _run_ml(window, qtbot)

    assert window._analysis_panel.current_detection().module_id == "detect.keras"
    assert window._act_export_events.isEnabled()
    assert window._act_export_figure.isEnabled()
    assert window._act_event_inspector.isEnabled()
    assert window._act_plots.isEnabled()


def test_ml_params_carry_model_and_resolved_range(window, qtbot, tmp_path, monkeypatch):
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    window._analysis_panel._detection.ml_enter.setValue(0.55)
    window._analysis_panel._detection.ml_exit.setValue(0.2)
    _run_ml(window, qtbot, range_source="whole")

    res = window._session.results.get(doc_id, channel, "detect.keras")
    assert res.params["model_path"] == "/fake/model.keras"
    assert res.params["enter"] == pytest.approx(0.55)
    assert res.params["exit"] == pytest.approx(0.2)
    # The panel resolved "whole" into concrete sample indices spanning the recording.
    assert res.params["range_start"] == 0
    assert res.params["range_stop"] == window._session.document(doc_id).meta.n_samples
    prov = res.payload.provenance
    assert prov["analyzed_duration_s"] == pytest.approx(window._session.document(doc_id).meta.n_samples / FS)
    assert prov["enter"] == pytest.approx(0.55) and prov["exit"] == pytest.approx(0.2)


def test_ml_visible_range_restricts_and_offsets_to_full_file(window, qtbot, tmp_path, monkeypatch):
    """'Visible view' runs the model only over the on-screen window, and the events
    come back in FULL-FILE coordinates (offset by the range start)."""
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    window._analysis_panel._detection.set_model_path("/fake/model.keras")

    # Zoom to a window around the middle event (50_000..50_400) only.
    window._trace_view.zoom_x_to(0.48, 0.52)
    qtbot.wait(10)

    # The indices the panel will resolve from the CURRENT view (whatever padding Qt used).
    n = window._session.document(doc_id).meta.n_samples
    i0, i1 = window._analysis_panel._visible_sample_range(doc_id, channel, n)
    n_expected = sum(1 for s, e in _EVENTS if s >= i0 and e <= i1)
    assert n_expected >= 1        # the window frames at least the middle event
    assert n_expected < len(_EVENTS)  # ...but not all of them (it is a real sub-range)

    _run_ml(window, qtbot, range_source="visible")

    res = window._session.results.get(doc_id, channel, "detect.keras")
    assert res is not None
    assert res.params["range_source"] == "visible"
    assert res.params["range_start"] == i0 and res.params["range_stop"] == i1
    # Only the framed events were detected...
    assert len(res.payload) == n_expected
    # ...and every one is reported inside the analysed window (full-file coords).
    for s in res.payload.df["start_idx"].to_numpy():
        assert i0 <= int(s) <= i1


def test_ml_and_schmitt_coexist_active_switches_on_rerun(window, qtbot, tmp_path, monkeypatch):
    """Running Schmitt then ML (and back) keeps both results stored and flips which one
    the panel shows; each drives the shared table + a single batched span overlay."""
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")

    # Schmitt first.
    _run_schmitt(window, qtbot)
    assert window._analysis_panel._active_detection == "detect.schmitt"
    assert window._session.results.get(doc_id, channel, "detect.schmitt") is not None
    assert len(_span_items(pane)) == 1

    # Then ML: it becomes active; both results now coexist in the store.
    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    _run_ml(window, qtbot)
    assert window._analysis_panel._active_detection == "detect.keras"
    assert window._session.results.get(doc_id, channel, "detect.keras") is not None
    assert window._session.results.get(doc_id, channel, "detect.schmitt") is not None
    # Still exactly one batched span item on the pane (ML replaced Schmitt's spans).
    assert len(_span_items(pane)) == 1
    assert window._analysis_panel.current_detection().module_id == "detect.keras"

    # Re-running Schmitt flips the active detection back.
    _run_schmitt(window, qtbot)
    assert window._analysis_panel._active_detection == "detect.schmitt"
    assert len(_span_items(pane)) == 1


def test_ml_event_row_click_zooms_and_marks(window, qtbot, tmp_path, monkeypatch):
    """Navigation works on an ML result exactly as on a Schmitt one (shared table)."""
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    _run_ml(window, qtbot)

    panel = window._analysis_panel
    anchor = window._trace_view.anchor_plot()
    before = tuple(anchor.vb.viewRange()[0])
    panel._table.view.selectRow(0)
    qtbot.wait(10)

    assert panel._marker_id is not None
    assert panel._marker_id in window._trace_view._overlays
    after = tuple(anchor.vb.viewRange()[0])
    assert (after[1] - after[0]) < (before[1] - before[0])


def test_ml_curation_redraws_accepted_only_spans(window, qtbot, tmp_path, monkeypatch):
    """Rejecting an ML event refreshes the batched span overlay to the accepted set —
    curation flows to the trace, exactly as for Schmitt events."""
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    _run_ml(window, qtbot)

    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert _span_items(pane)[0].n_spans == len(_EVENTS)

    # Reject the first event through the shared table model.
    window._analysis_panel._table._model.set_accepted(0, False)
    qtbot.wait(10)
    assert _span_items(pane)[0].n_spans == len(_EVENTS) - 1


def test_ml_calibration_change_marks_result_stale(window, qtbot, tmp_path, monkeypatch):
    """An ML result carries the calibration fingerprint and goes stale when the gain
    is revised — the same deferred-current-gain workflow the Schmitt result honours."""
    from nanoscope.core.calibration import ChannelCalibration

    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    _run_ml(window, qtbot)

    panel = window._analysis_panel
    assert panel._stale.isHidden()

    window._session.set_calibration(
        doc_id, channel,
        ChannelCalibration(gain=2.0, offset=0.0, target_units="pA",
                           provenance=Provenance.UNKNOWN))
    qtbot.wait(10)
    assert not panel._stale.isHidden()


def test_ml_run_is_a_noop_without_a_model(window, qtbot, tmp_path, monkeypatch):
    """Triggering ML with no model loaded does nothing (the gate holds even if forced)."""
    _use_fake_model(monkeypatch)   # TF 'available', but no model set
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    assert not window._analysis_panel.can_run_ml()

    window._analysis_panel.trigger_ml_detection()
    qtbot.wait(30)
    assert window._session.results.get(doc_id, channel, "detect.keras") is None


# ================================================================================================
# C. The REAL model, through the controller, in a subprocess (guarded on TF + files).
#
#    Why a subprocess: importing tensorflow into the pytest-qt process aborts the
#    shiboken/pytest-qt harness (see tests/analysis/test_ml_detect.py). A standalone
#    Qt process has no such conflict, so the real model runs through the real
#    controller + AnalysisPanel there and reports back a small JSON result.
# ================================================================================================
_REPO_ROOT = Path(__file__).resolve().parents[2]

_SUBPROCESS_PROGRAM = r"""
import os, sys, json
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
REPO_ROOT, MODEL, ABF = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO_ROOT)

import nanoscope  # noqa: F401  bootstrap
import numpy as np
from pathlib import Path
from nanoscope.gui.app import create_app
app = create_app([])
from nanoscope.analysis import register_all
from nanoscope.gui.overlays import register_all_renderers, SpanOverlayItem
register_all(); register_all_renderers()
from nanoscope.core.session import Session
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.trace_view import TraceView
from nanoscope.gui.analysis_worker import AnalysisController
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.analysis.events import EventSet
from PySide6 import QtCore

FS = 50000.0
from neo.rawio import AxonRawIO
r = AxonRawIO(filename=ABF); r.parse_header()
total = r.get_signal_size(block_index=0, seg_index=0)
n = min(150000, total)                       # ~3 s @ 50 kHz — small = fast
a = max(0, (total - n) // 2)                  # middle: where the translocations are
raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=a, i_stop=a + n)
cur = np.ascontiguousarray(r.rescale_signal_raw_to_float(raw, dtype="float32")[:, 2])

info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
meta = RecordingMeta(path=Path(ABF), format_name="abf-slice", n_samples=cur.size,
                     sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
doc = TraceDocument(meta=meta, channels=(cur,))

session = Session()
tv = TraceView(session)
controller = AnalysisController(session)
panel = AnalysisPanel(session, controller, tv)
doc_id = session.add_document(doc)
session.add_layer(doc_id, 0)
panel.set_selection(doc_id, 0)
panel._detection.set_model_path(MODEL)
panel._detection.ml_range.setCurrentIndex(1)   # whole (the doc is already the slice)

out = {}
def _fin(_obj):
    out["finished"] = True
    QtCore.QTimer.singleShot(50, app.quit)      # let the panel's redraw settle first
def _fail(msg):
    out["error"] = msg
    app.quit()
controller.finished.connect(_fin)
controller.failed.connect(_fail)
QtCore.QTimer.singleShot(280000, app.quit)      # hard safety timeout
panel.trigger_ml_detection()
app.exec()

res = session.results.get(doc_id, 0, "detect.keras")
out["has_result"] = res is not None
if res is not None:
    out["is_eventset"] = isinstance(res.payload, EventSet)
    out["n_events"] = int(len(res.payload))
    out["module_id"] = res.module_id
    out["prov_detector"] = res.payload.provenance.get("detector")
    out["prov_window"] = int(res.payload.provenance.get("model_window", -1))
    out["in_bounds"] = bool(
        (res.payload.df["end_idx"].to_numpy() <= n).all()
        and (res.payload.df["start_idx"].to_numpy() >= 0).all()
    ) if len(res.payload) else True
    pane = tv.pane_for_layer(f"{doc_id}#0")
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)] if pane else []
    out["n_span_items"] = len(spans)
    out["n_spans"] = int(spans[0].n_spans) if spans else 0
print("__RESULT__" + json.dumps(out))
"""


def _run_subprocess() -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROGRAM,
         str(_REPO_ROOT), str(MODEL_PATH), str(ABF_PATH)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"ML subprocess failed rc={proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-3000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise AssertionError(
        f"no __RESULT__ line from ML subprocess\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr[-2000:]}")


@pytest.mark.corpus
@pytest.mark.skipif(not _HAVE_REAL, reason="TensorFlow and/or the real .keras model / ABF absent")
def test_real_keras_ml_detection_through_controller_subprocess():
    r = _run_subprocess()
    assert r.get("error") is None, f"run failed: {r.get('error')}"
    # The real model, loaded at runtime and run through the real controller + panel,
    # produced an EventSet that landed in the ResultStore...
    assert r["has_result"] is True
    assert r["is_eventset"] is True
    assert r["module_id"] == "detect.keras"
    assert r["prov_detector"] == "ml_keras"
    assert r["prov_window"] == 3072
    # ...with real events in-bounds, and the spans painted as ONE batched item.
    assert r["n_events"] >= 1
    assert r["in_bounds"] is True
    assert r["n_span_items"] == 1
    assert r["n_spans"] == r["n_events"]
