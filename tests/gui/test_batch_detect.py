"""GUI tests for batch detection (pytest-qt, offscreen).

Batch detection runs the CURRENT Schmitt parameters over EVERY open recording that
can be detected -- a live current channel + a known sample rate -- on ONE background
pass, storing an ``EventSet`` per document in the ResultStore. It is the step the
user runs before plotting the voltage dependence of their -1000..-200 mV series.

Two layers are exercised:

* the pure machinery -- :class:`~nanoscope.gui.analysis_worker.BatchDetectController`
  driven to a :class:`~nanoscope.gui.analysis_worker.BatchOutcome` on a real
  ``QThread``: every queued recording detects and lands one ``detect.schmitt``
  result; one bad recording is reported in ``errors`` without sinking the batch;
  ``abort`` detaches the worker so a late signal is dropped;

* the wiring through the real :class:`MainWindow` -- ``_detectable_targets`` skips a
  recording with no current channel / no rate, the ``Analysis > Batch detect...``
  action gates on >= 1 detectable recording, and ``_batch_detect`` installs one
  result per detectable document on the main thread while the non-detectable ones
  are skipped.

Detection is the REAL detector on synthetic square-event traces (mirroring
``test_analysis_integration``), so the batch genuinely detects rather than being
handed pre-made events.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from PySide6 import QtCore

from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import register_all
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.analysis_worker import BatchDetectController, BatchOutcome
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()   # idempotent; makes detect.schmitt resolvable
    yield


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _current_doc(tmp_path, *, nominal_mV: float, name: str, fs: float | None = FS,
                 n: int = 100_000) -> TraceDocument:
    """A single-current-channel recording with five square events and a nominal voltage."""
    arr = _event_trace(n)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=fs, rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,), nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _voltage_only_doc(tmp_path, *, name: str, n: int = 4096) -> TraceDocument:
    """A recording with a VOLTAGE channel but NO current channel -> never a detect target."""
    arr = np.ascontiguousarray(np.full(n, -500.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> str:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    return doc_id


# ================================================================================================
# BatchDetectController -- the core machinery
# ================================================================================================

def test_batch_controller_detects_every_recording(qapp, qtbot, tmp_path):
    session = Session()
    ids = [session.add_document(_current_doc(tmp_path, nominal_mV=mV, name=f"p_{int(-mV)}.dat"))
           for mV in (-200.0, -500.0, -1000.0)]
    controller = BatchDetectController(session)

    with qtbot.waitSignal(controller.finished, timeout=60_000) as blocker:
        controller.start([(d, 0) for d in ids], {"min_dwell_s": 5e-4})
    outcome = blocker.args[0]

    assert isinstance(outcome, BatchOutcome)
    assert outcome.errors == ()
    assert len(outcome.results) == len(ids)
    assert {r.doc_id for r in outcome.results} == set(ids)
    for res in outcome.results:
        assert res.module_id == "detect.schmitt"
        assert isinstance(res.payload, EventSet)
        assert len(res.payload) == len(_EVENTS)                  # the five placed events
        assert res.payload.provenance.get("analyzed_duration_s")  # so rate_hz() works
    assert not controller.is_busy()


def test_batch_controller_reports_one_bad_recording_without_sinking_the_batch(
        qapp, qtbot, tmp_path):
    """A rate-unknown recording raises inside the detector; it is caught into errors
    while the good recording still detects."""
    session = Session()
    good = session.add_document(_current_doc(tmp_path, nominal_mV=-500.0, name="good.dat"))
    bad = session.add_document(
        _current_doc(tmp_path, nominal_mV=-700.0, name="norate.dat", fs=None, n=8_000))
    controller = BatchDetectController(session)

    with qtbot.waitSignal(controller.finished, timeout=60_000) as blocker:
        controller.start([(good, 0), (bad, 0)], {"min_dwell_s": 5e-4})
    outcome = blocker.args[0]

    assert len(outcome.results) == 1
    assert outcome.results[0].doc_id == good
    assert len(outcome.errors) == 1
    label, msg = outcome.errors[0]
    assert label == "norate.dat"
    assert "sample rate" in msg


def test_batch_controller_abort_drops_late_finished(qapp, tmp_path):
    """abort() detaches the batch worker and emits NOTHING.

    ``seen`` used to be connected to ``finished`` only, so the one signal abort()
    actually did emit -- ``cancelled``, because the detach never worked and the sender
    guard passed None against None (C40) -- was the one signal nothing watched. Listen
    to every terminal signal."""
    session = Session()
    ids = [session.add_document(_current_doc(tmp_path, nominal_mV=mV, name=f"a_{int(-mV)}.dat"))
           for mV in (-200.0, -500.0)]
    controller = BatchDetectController(session)
    seen: list[str] = []
    controller.finished.connect(lambda _o: seen.append("finished"))
    controller.failed.connect(lambda _m: seen.append("failed"))
    controller.cancelled.connect(lambda: seen.append("cancelled"))

    controller.start([(d, 0) for d in ids], {"min_dwell_s": 5e-4})
    controller.abort()

    for _ in range(20):
        qapp.processEvents()
        QtCore.QThread.msleep(20)
    qapp.processEvents()

    assert not controller.is_busy()
    assert seen == [], f"a detached worker's terminal signal reached the controller: {seen}"


# ================================================================================================
# MainWindow wiring -- targets, gating, end-to-end install
# ================================================================================================

def test_detectable_targets_skips_no_current_and_no_rate(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    a = _install(win, _current_doc(tmp_path, nominal_mV=-200.0, name="a.dat", n=4_000))
    b = _install(win, _current_doc(tmp_path, nominal_mV=-500.0, name="b.dat", n=4_000))
    _install(win, _voltage_only_doc(tmp_path, name="volt_only.dat"))          # no current
    _install(win, _current_doc(tmp_path, nominal_mV=-700.0, name="norate.dat",
                               fs=None, n=4_000))                              # no rate

    targets = win._detectable_targets()
    assert {doc_id for doc_id, _ch, _name in targets} == {a, b}               # only the two
    assert all(ch == 0 for _doc, ch, _name in targets)


def test_batch_action_gated_on_at_least_one_detectable(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win._sync_batch_detect_action()
    assert not win._act_batch_detect.isEnabled()                              # nothing open

    _install(win, _voltage_only_doc(tmp_path, name="volt_only.dat"))
    win._sync_batch_detect_action()
    assert not win._act_batch_detect.isEnabled()                              # no current channel

    _install(win, _current_doc(tmp_path, nominal_mV=-500.0, name="c.dat", n=4_000))
    win._sync_batch_detect_action()
    assert win._act_batch_detect.isEnabled()                                  # one detectable -> live


def test_batch_detect_installs_one_result_per_recording(qapp, qtbot, tmp_path):
    """Fresh docs (none already detected) -> no prompt; the batch installs a result each."""
    win = MainWindow()
    qtbot.addWidget(win)
    ids = [_install(win, _current_doc(tmp_path, nominal_mV=mV, name=f"pore_{int(-mV)}.dat"))
           for mV in (-200.0, -1000.0)]

    win._batch_detect()   # runs directly: no existing detections means no dialog
    qtbot.waitUntil(
        lambda: all(win._session.results.get(d, 0, "detect.schmitt") is not None for d in ids),
        timeout=60_000)

    for d in ids:
        res = win._session.results.get(d, 0, "detect.schmitt")
        assert isinstance(res.payload, EventSet)
        assert len(res.payload) == len(_EVENTS)
    assert not win._batch_controller.is_busy()


def test_batch_detect_skips_nondetectable_recording(qapp, qtbot, tmp_path):
    """A voltage-only recording is not a target: after the batch it carries no detection."""
    win = MainWindow()
    qtbot.addWidget(win)
    cur = _install(win, _current_doc(tmp_path, nominal_mV=-500.0, name="cur.dat"))
    volt = _install(win, _voltage_only_doc(tmp_path, name="volt_only.dat"))

    assert len(win._detectable_targets()) == 1   # the voltage-only doc is skipped

    win._batch_detect()
    qtbot.waitUntil(
        lambda: win._session.results.get(cur, 0, "detect.schmitt") is not None, timeout=60_000)

    assert win._session.results.for_doc(volt) == []   # skipped -> no result stored


def test_vdep_lights_up_after_batch_detect(qapp, qtbot, tmp_path):
    """The end-to-end payoff: batch-detecting a two-voltage series enables Voltage dependence."""
    win = MainWindow()
    qtbot.addWidget(win)
    ids = [_install(win, _current_doc(tmp_path, nominal_mV=mV, name=f"v_{int(-mV)}.dat"))
           for mV in (-300.0, -900.0)]

    win._sync_vdep_action()
    assert not win._act_vdep.isEnabled()          # detected nothing yet

    win._batch_detect()
    qtbot.waitUntil(
        lambda: all(win._session.results.get(d, 0, "detect.schmitt") is not None for d in ids),
        timeout=60_000)

    win._sync_vdep_action()
    assert win._act_vdep.isEnabled()              # two recordings with voltage + detection
