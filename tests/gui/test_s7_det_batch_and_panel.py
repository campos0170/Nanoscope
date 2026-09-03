"""S7-DET1 / S7-DET3 / S7-DET4 / S7-JRN1 through the REAL widgets.

* Batch detect (S7-DET1): the recording the form was confirmed for keeps
  ``'proposed'``; every other recording is stamped ``'batch'``, carries the
  copied-windows advisory, records the resolved source in its params, and the
  BaselinePanel names it honestly when that recording is selected.
* The clamp line (S7-DET3): the panel states the windows a clamped detection
  actually ran at, beside the form's.
* Merge all (S7-DET4): the installed result's ``n_events`` and composed warnings
  describe the merged table; the subpopulation resolver reads a truthful count.
* The probe tooltip (S7-JRN1) states the measured seconds, not a fifth of one.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis import baseline as B
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import register_all
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _signal(spans, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._batch_controller.abort()


# ===========================================================================
# BaselinePanel on its own
# ===========================================================================
def test_the_panel_names_a_batch_stamp_and_scopes_its_batch_params(qapp, qtbot):
    panel = BaselinePanel()
    qtbot.addWidget(panel)
    panel.set_recording(FS, ("doc-A", 0))
    # At the default nothing was measured: no scope, the honest source for everyone.
    assert B.BATCH_SCOPE_KEY not in panel.batch_params()
    panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": "proposed"})
    bp = panel.batch_params()
    assert bp["baseline_param_source"] == "proposed"
    assert bp[B.BATCH_SCOPE_KEY] == ("doc-A", 0), "the recording these windows are ABOUT"
    assert B.BATCH_SCOPE_KEY not in panel.params(), "the Detect button's params are untouched"
    # ...and a stored 'batch' result seeds an honest label, never "measured on this
    # recording, and confirmed".
    panel.set_recording(FS, ("doc-B", 0))
    panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": "batch"})
    assert panel.param_source() == "batch"
    assert "COPIED from the batch form" in panel.source_label.text()
    assert "confirmed" not in panel.source_label.text().split("COPIED")[0]


def test_the_probe_tooltip_states_the_measured_seconds(qapp, qtbot):
    panel = BaselinePanel()
    qtbot.addWidget(panel)
    tip = panel.probe_button.toolTip()
    assert "fifth of a second" not in tip
    assert "2 s on a 3 M-sample" in tip and "6 s on 9 M" in tip


def test_the_panel_states_the_windows_a_clamped_run_actually_used(qapp, qtbot):
    panel = BaselinePanel()
    qtbot.addWidget(panel)
    panel.set_recording(FS, ("doc-A", 0))
    assert not panel.run_windows_label.isVisibleTo(panel)
    panel.set_run_windows({"baseline_rough_s": 0.08, "baseline_base_s": 0.016,
                           "baseline_fallback": B.FALLBACK_CLAMPED})
    assert panel.run_windows_label.isVisibleTo(panel)
    text = panel.run_windows_label.text()
    # One unit for both halves (sweep-7 fix-batch X-SURF-4): the run in seconds,
    # like the spins it is read against -- not "80 ms ... 0.5 s".
    assert "0.08 s / base 0.016 s" in text and "CLAMPED" in text and "0.5 s / 0.1 s" in text
    # A run at the form's own windows has nothing to add.
    panel.set_run_windows({"baseline_rough_s": 0.5, "baseline_base_s": 0.1})
    assert not panel.run_windows_label.isVisibleTo(panel)
    panel.set_run_windows({"baseline_rough_s": 0.08, "baseline_base_s": 0.016,
                           "baseline_fallback": B.FALLBACK_CLAMPED})
    panel.set_run_windows(None)
    assert not panel.run_windows_label.isVisibleTo(panel)


# ===========================================================================
# Batch detect through the real window (S7-DET1)
# ===========================================================================
def test_batch_detect_keeps_the_confirmed_source_at_home_and_stamps_batch_elsewhere(
        win, qtbot, tmp_path):
    a = win._session.add_document(_doc(tmp_path, _signal(_FRAGMENTS), name="a.dat"))
    b = win._session.add_document(_doc(tmp_path, _signal(_FRAGMENTS, seed=1), name="b.dat"))
    for d in (a, b):
        win._create_default_layers(win._session.document(d))
    win._on_selection(a, 0)
    qtbot.wait(20)
    bl = win._analysis_panel.baseline_panel
    # The form was filled in (as a confirmed proposal would) FOR recording a.
    bl.set_params({"baseline_rough_s": 0.2, "baseline_base_s": 0.05,
                   "baseline_param_source": "proposed"})
    assert bl.param_source() == "proposed"
    params = win._analysis_panel.batch_detection_params()
    assert params[B.BATCH_SCOPE_KEY] == (a, 0)

    with qtbot.waitSignal(win._batch_controller.finished, timeout=60_000):
        win._batch_detect()
    qtbot.wait(50)

    ra = win._session.results.get(a, 0, "detect.schmitt")
    rb = win._session.results.get(b, 0, "detect.schmitt")
    assert ra is not None and rb is not None
    assert ra.payload.provenance["baseline_param_source"] == "proposed"
    assert not any("COPIED FROM THE BATCH FORM" in w for w in ra.warnings)
    assert rb.payload.provenance["baseline_param_source"] == "batch"
    assert rb.payload.provenance["baseline_rough_s"] == pytest.approx(0.2)
    hit = [w for w in rb.warnings if "COPIED FROM THE BATCH FORM" in w]
    assert hit and "0.2 s / 0.05 s" in hit[0] and "never measured" in hit[0]
    # The stored params say what the run resolved, and never carry a per-load id.
    assert rb.params["baseline_param_source"] == "batch"
    assert ra.params["baseline_param_source"] == "proposed"
    assert B.BATCH_SCOPE_KEY not in rb.params and B.BATCH_SCOPE_KEY not in ra.params

    # Selecting the never-measured recording tells the truth on every surface.
    win._on_selection(b, 0)
    qtbot.wait(50)
    assert bl.param_source() == "batch"
    assert "COPIED from the batch form" in bl.source_label.text()
    assert "COPIED FROM THE BATCH FORM" in win._analysis_panel._stats.warnings_label.text()


# ===========================================================================
# Merge all through the real button (S7-DET4)
# ===========================================================================
def _install_detection(win, doc_id, signal) -> EventSet:
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    es = EventSet(df, prov)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=es, scalars={"n_events": float(len(df))}))
    return es


def test_merge_all_restamps_n_events_and_the_report_stops_counting_merges_as_curation(
        win, qtbot, tmp_path):
    signal = _signal(_FRAGMENTS)
    doc_id = win._session.add_document(_doc(tmp_path, signal, name="frag.dat"))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    es = _install_detection(win, doc_id, signal)
    assert len(es.df) == 4 and es.provenance["n_events"] == 4

    panel = win._analysis_panel
    panel.merge_controls.set_window_s(0.010)
    with qtbot.waitSignal(panel.mergeFinished, timeout=30_000):
        panel.merge_controls.merge_all_btn.click()
    qtbot.wait(20)

    res = panel.current_detection()
    assert len(res.payload.df) == 2
    assert res.payload.provenance["n_events"] == 2
    assert res.scalars["n_events"] == 2.0
    # The report's curation line: nothing curated out, so no CURATION line at all --
    # before the fix it read "2 of the detector's 4 region(s) were curated OUT".
    meta, note = SR.rows_in_hand(res.payload.accepted, int(res.payload.df["accepted"].sum()))
    assert meta["n_detected"] == 2 and note is None


# ===========================================================================
# Persistence: a 'batch' stamp survives a project round-trip (S7-DET1)
# ===========================================================================
def _csv_doc(tmp_path, signal, *, name) -> tuple[TraceDocument, "os.PathLike"]:
    """A time,current CSV the TextTraceReader reopens as one CURRENT channel, and an
    in-memory doc whose meta.path IS that CSV -- the test_project_io pattern."""
    import pandas as pd
    path = tmp_path / name
    t = np.arange(signal.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(signal, dtype=np.float64)}).to_csv(
        path, index=False)
    return _doc(tmp_path, signal, name=name), path


def test_a_batch_stamp_survives_save_and_reopen_and_the_panel_still_tells_the_truth(
        qapp, qtbot, tmp_path):
    from pathlib import Path

    w1 = MainWindow()
    qtbot.addWidget(w1)
    doc_a, path_a = _csv_doc(tmp_path, _signal(_FRAGMENTS), name="a.csv")
    doc_b, path_b = _csv_doc(tmp_path, _signal(_FRAGMENTS, seed=1), name="b.csv")
    a = w1._session.add_document(doc_a)
    b = w1._session.add_document(doc_b)
    for d in (a, b):
        w1._create_default_layers(w1._session.document(d))
    w1._on_selection(a, 0)
    qtbot.wait(20)
    w1._analysis_panel.baseline_panel.set_params(
        {"baseline_rough_s": 0.2, "baseline_base_s": 0.05, "baseline_param_source": "proposed"})
    with qtbot.waitSignal(w1._batch_controller.finished, timeout=60_000):
        w1._batch_detect()
    qtbot.wait(50)
    assert w1._session.results.get(b, 0, "detect.schmitt").payload.provenance[
        "baseline_param_source"] == "batch"
    proj = tmp_path / "batch.nanoscope"
    w1._write_project(str(proj))
    w1._analysis_controller.abort()
    w1._batch_controller.abort()

    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    s2 = w2._session
    live_b = s2.doc_id_for_source(str(Path(path_b).resolve()))
    live_a = s2.doc_id_for_source(str(Path(path_a).resolve()))
    assert live_a is not None and live_b is not None
    rb = s2.results.get(live_b, 0, "detect.schmitt")
    ra = s2.results.get(live_a, 0, "detect.schmitt")
    assert rb is not None and ra is not None
    assert rb.payload.provenance["baseline_param_source"] == "batch"
    assert ra.payload.provenance["baseline_param_source"] == "proposed"
    assert any("COPIED FROM THE BATCH FORM" in w for w in rb.warnings)
    w2._on_selection(live_b, 0)
    qtbot.wait(50)
    assert "COPIED from the batch form" in w2._analysis_panel.baseline_panel.source_label.text()
    w2._analysis_controller.abort()
    w2._batch_controller.abort()
