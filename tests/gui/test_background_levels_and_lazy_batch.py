"""Responsive level fitting and lazy immutable batch inputs preserve the science."""
from __future__ import annotations

import gc
import os
import threading
import weakref
from dataclasses import replace

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.analysis.baseline import baseline_config_from_params, characterisation_baseline
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import detection_signal, fit_levels
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_worker import BatchDetectController, FrozenChannelSource
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui import level_fit_job

FS = 10000.


def document(path, n=20000):
    x = (100 + np.random.default_rng(903).normal(0, .4, n)).astype(np.float32)
    for a in (3000, 8000, 15000):
        if a + 400 < n:
            x[a:a+200] -= 20
            x[a+200:a+400] -= 30
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=n,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta, (x,))


@pytest.fixture
def detection_window(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    doc = document(tmp_path / "levels.dat")
    doc_id = w._session.add_document(doc)
    doc = w._session.document(doc_id)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(doc.raw(0), FS, baseline=100., min_dwell_s=.001)
    res = AnalysisResult("detect.schmitt", "1", doc_id, 0,
                         {"min_dwell_s": .001}, (1., 0.), EventSet(df, prov))
    w._session.add_result(res)
    yield w, res, doc
    w._analysis_controller.abort()


def pause_job(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    threads = []
    original = level_fit_job.detection_signal
    def paused(*args, **kwargs):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(10), "test did not release the worker"
        return original(*args, **kwargs)
    monkeypatch.setattr(level_fit_job, "detection_signal", paused)
    return entered, release, threads


def test_levels_leave_qt_responsive_and_match_previous_synchronous_math(
        detection_window, qtbot, monkeypatch):
    w, source, doc = detection_window
    prov = dict(source.payload.provenance)
    signal = detection_signal(doc.raw(0), FS, prov)
    cfg, origin = baseline_config_from_params(prov)
    base = characterisation_baseline(np.asarray(signal, dtype=np.float64), FS,
                                    config=cfg, polarity=prov.get("polarity"), param_source=origin)
    expected = fit_levels(source.payload.accepted.df, np.asarray(signal, dtype=np.float64), FS,
                          doc_id=source.doc_id, channel=0, baseline=base.baseline,
                          polarity=int(prov.get("polarity", 1)), provenance=prov,
                          calibration_fingerprint=(1., 0.), baseline_fingerprint=cfg.fingerprint(FS))
    entered, release, threads = pause_job(monkeypatch)
    stored = []
    w._session.resultsChanged.connect(lambda _, module: stored.append(module))
    w._fit_levels()
    qtbot.waitUntil(entered.is_set)
    try:
        pulse = []
        QtCore.QTimer.singleShot(0, lambda: pulse.append(True))
        qtbot.waitUntil(lambda: bool(pulse))
        assert threads[0] != threading.get_ident()
        assert w._level_fit is None
        assert w._analysis_controller.is_busy()
        worker = w._analysis_controller._worker
        w._fit_levels()
        assert w._analysis_controller._worker is worker
    finally:
        release.set()
    qtbot.waitUntil(lambda: w._level_fit is not None, timeout=10000)
    pd.testing.assert_frame_equal(w._level_fit.payload, expected.payload, check_exact=True)
    assert w._level_fit.params == expected.params
    assert w._level_fit.baseline_fingerprint == expected.baseline_fingerprint
    assert stored.count("fit.levels") == 1
    assert w._level_fit_pending is None
    assert w._level_fit_inputs is not None


@pytest.mark.parametrize("change", ["acceptance", "bounds", "provenance", "params", "calibration", "closed", "replaced", "rate"])
def test_stale_background_fit_never_installs(detection_window, qtbot, monkeypatch, change):
    w, res, doc = detection_window
    entered, release, _ = pause_job(monkeypatch)
    w._fit_levels()
    qtbot.waitUntil(entered.is_set)
    try:
        if change == "acceptance":
            res.payload.df.loc[res.payload.df.index[0], "accepted"] = False
        elif change == "bounds":
            res.payload.df.loc[res.payload.df.index[0], "end_idx"] += 1
        elif change == "provenance":
            res.payload.provenance["polarity"] = 1
        elif change == "params":
            res.params["min_dwell_s"] = .002
        elif change == "calibration":
            w._session.set_calibration(res.doc_id, 0, ChannelCalibration(gain=2), write_sidecar=False)
        elif change == "closed":
            w._session.close_document(res.doc_id)
        elif change == "replaced":
            w._session.add_result(replace(res, payload=EventSet(res.payload.df.copy(), dict(res.payload.provenance))))
        else:
            w._session._documents[res.doc_id] = replace(doc, meta=replace(doc.meta, sample_rate_hz=FS * 2))
    finally:
        release.set()
    qtbot.waitUntil(lambda: not w._analysis_controller.is_busy(), timeout=10000)
    qtbot.waitUntil(lambda: w._level_fit_pending is None)
    assert w._session.results.get(res.doc_id, 0, "fit.levels") is None
    assert w._level_fit is None
    assert "discarded" in w.statusBar().currentMessage()


def test_cancel_discards_level_fit_and_clears_pending(detection_window, qtbot, monkeypatch):
    w, res, _ = detection_window
    entered, release, _ = pause_job(monkeypatch)
    w._fit_levels()
    qtbot.waitUntil(entered.is_set)
    try:
        w._cancel_load()
        assert w._level_fit_pending is None
        assert not w._analysis_controller.is_busy()
    finally:
        release.set()
    qtbot.wait(100)
    assert w._session.results.get(res.doc_id, 0, "fit.levels") is None


def test_batch_does_not_eagerly_calibrate_and_freezes_nested_settings(qapp, tmp_path, monkeypatch):
    session = Session()
    ids = [session.add_document(document(tmp_path / f"batch{i}.dat", n=1000)) for i in range(3)]
    for doc_id in ids:
        session.set_calibration(doc_id, 0, ChannelCalibration(gain=2., offset=3.), write_sidecar=False)
    params = {"nested": {"value": 7}}
    controller = BatchDetectController(session)
    captured = []
    monkeypatch.setattr(controller, "_spin_up", captured.append)
    monkeypatch.setattr(session, "calibrated", lambda *_: pytest.fail("eager calibrated() on GUI thread"))
    weak_arrays, seen = [], []
    class Module:
        id = "test.lazy"
        def run(self, ctx):
            gc.collect()
            assert all(ref() is None for ref in weak_arrays)
            weak_arrays.append(weakref.ref(ctx.channel.data))
            seen.append((ctx.params["nested"]["value"], ctx.params["_calibration_fingerprint"], ctx.channel.data.copy()))
            return None
    assert controller.start([(doc_id, 0) for doc_id in ids], params, module=Module())
    assert len(captured) == 1
    worker = captured[0]
    for job in worker._jobs:
        assert job.channel_data is None
        assert job.source is not None
        assert not job.source.raw.flags.writeable
        assert np.shares_memory(job.source.raw, session.document(job.doc_id).raw(0))
    params["nested"]["value"] = 999
    session.set_calibration(ids[1], 0, ChannelCalibration(gain=10), write_sidecar=False)
    outcomes = []
    worker.finished.connect(outcomes.append)
    worker.run()
    assert outcomes and not outcomes[0].errors
    assert len(seen) == 3
    for doc_id, (value, fingerprint, actual) in zip(ids, seen):
        assert value == 7 and fingerprint == (2., 3.)
        np.testing.assert_array_equal(actual, session.document(doc_id).raw(0) * np.float32(2) + np.float32(3))
    gc.collect()
    assert all(ref() is None for ref in weak_arrays)


@pytest.mark.parametrize("calibration", [ChannelCalibration(), ChannelCalibration(gain=-2, offset=3), ChannelCalibration(gain=1e37)])
def test_frozen_channel_matches_session_numerics_units_and_damage(qapp, tmp_path, calibration):
    session = Session()
    doc_id = session.add_document(document(tmp_path / "materialize.dat", n=100))
    session.set_calibration(doc_id, 0, calibration, write_sidecar=False)
    expected = session.calibrated(doc_id, 0)
    actual = FrozenChannelSource.capture(session.document(doc_id), 0, calibration).materialize()
    np.testing.assert_array_equal(actual.data, expected.data)
    assert actual.units == expected.units
    assert actual.issues == expected.issues
    assert actual.sample_rate_hz == expected.sample_rate_hz
