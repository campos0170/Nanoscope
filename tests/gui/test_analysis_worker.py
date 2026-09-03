"""Phase 6 worker tests (PORT_PLAN §6, gap #2).

Drives a real detection run through ``AnalysisController`` -> ``AnalysisWorker`` on
a background ``QThread`` to completion, offscreen, and asserts:

  * ``started`` fires with the module id; ``progressed`` is relayed; ``finished``
    carries an ``AnalysisResult`` with the five placed events.
  * The controller injects ``_calibration_fingerprint`` (from Session) and
    ``pore_id`` into the module's params (the module never touches the Session).
  * ``finished(AnalysisResult)`` installed on the MAIN THREAD via
    ``Session.add_result`` stores + emits ``resultsChanged``.
  * A module that raises (rate-unknown channel -> ``requires_time_base``) surfaces
    as ``failed(str)``, never an escaped traceback.
  * ``run_callable`` gives the non-module jobs a generic ``finished(object)`` path.
  * ``abort()`` detaches the in-flight worker so a late ``finished`` is dropped.
"""

from __future__ import annotations

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
from nanoscope.gui.analysis_worker import AnalysisCancelled, AnalysisController

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


def _make_event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    """Five square blockades on a flat, noisy baseline — contiguous float32."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _make_doc(tmp_path, *, fs: float | None = FS, n: int = 100_000,
              name: str = "pore_p1.dat") -> TraceDocument:
    arr = _make_event_trace(n)
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


def _open(tmp_path, **kw) -> tuple[Session, str]:
    sess = Session()
    doc = _make_doc(tmp_path, **kw)
    doc_id = sess.add_document(doc)
    return sess, doc_id


class _Capture:
    """Collect a controller's terminal signals and quit an event loop on the first."""

    def __init__(self, controller: AnalysisController) -> None:
        self.started: list[str] = []
        self.progress: list[tuple[float, str]] = []
        self.finished: list[object] = []
        self.failed: list[str] = []
        self.cancelled = 0
        self._loop = QtCore.QEventLoop()
        controller.started.connect(self.started.append)
        controller.progressed.connect(lambda f, m: self.progress.append((f, m)))
        controller.finished.connect(self._on_finished)
        controller.failed.connect(self._on_failed)
        controller.cancelled.connect(self._on_cancelled)

    def _on_finished(self, r):
        self.finished.append(r)
        self._loop.quit()

    def _on_failed(self, m):
        self.failed.append(m)
        self._loop.quit()

    def _on_cancelled(self):
        self.cancelled += 1
        self._loop.quit()

    def wait(self, timeout_ms: int = 20_000) -> None:
        """Block until the first terminal signal (finished/failed/cancelled) or timeout."""
        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(self._loop.quit)
        timer.start(timeout_ms)
        self._loop.exec()
        timer.stop()

    def drain_until(self, predicate, app, timeout_ms: int = 20_000) -> None:
        """Pump the event loop until ``predicate()`` holds or the timeout elapses.

        Poll-based rather than a nested exec so it is insensitive to the exact
        order in which a detached worker's late signal and the live worker's
        terminal signal arrive under load."""
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while not predicate() and deadline.elapsed() < timeout_ms:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
            QtCore.QThread.msleep(2)


@pytest.fixture(autouse=True)
def _registered():
    register_all()  # idempotent
    yield


# ---------------------------------------------------------------------------
# happy path: a real detection run to completion on a QThread
# ---------------------------------------------------------------------------
def test_detection_run_to_completion(qapp, tmp_path):
    sess, doc_id = _open(tmp_path)
    controller = AnalysisController(sess)
    cap = _Capture(controller)

    controller.start("detect.schmitt", doc_id, 0,
                     params={"min_dwell_s": 5e-4}, pore_id="P1")
    cap.wait()

    assert not cap.failed, cap.failed
    assert cap.cancelled == 0
    assert cap.started == ["detect.schmitt"]
    assert len(cap.finished) == 1

    res = cap.finished[0]
    assert isinstance(res, AnalysisResult)
    assert res.module_id == "detect.schmitt"
    assert res.doc_id == doc_id
    assert res.channel == 0

    assert isinstance(res.payload, EventSet)
    assert len(res.payload) == len(_EVENTS)
    assert (res.payload.df["pore_id"] == "P1").all()

    # controller injected the Session's fingerprint (identity for a fresh CURRENT channel)
    assert res.calibration_fingerprint == sess.calibration(doc_id, 0).fingerprint()
    assert res.calibration_fingerprint == (1.0, 0.0)

    # progress was relayed off-thread: first 0.0, last 1.0
    assert cap.progress, "no progress relayed"
    assert cap.progress[0][0] == 0.0
    assert cap.progress[-1][0] == 1.0

    # the run really was off the main thread and has been torn down
    assert not controller.is_busy()


def test_finished_installed_via_add_result(qapp, tmp_path):
    """The main-thread consumer stores the result and Session announces it."""
    sess, doc_id = _open(tmp_path)
    controller = AnalysisController(sess)

    seen: list[tuple[str, str]] = []
    sess.resultsChanged.connect(lambda d, m: seen.append((d, m)))
    controller.finished.connect(lambda r: sess.add_result(r))

    cap = _Capture(controller)
    controller.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-4})
    cap.wait()

    assert len(cap.finished) == 1
    res = cap.finished[0]
    assert seen == [(doc_id, "detect.schmitt")]
    stored = sess.results.get(doc_id, 0, "detect.schmitt")
    assert stored is res


def test_pore_id_from_params_takes_precedence(qapp, tmp_path):
    sess, doc_id = _open(tmp_path)
    controller = AnalysisController(sess)
    cap = _Capture(controller)

    # pore_id supplied inside params must win over the keyword default
    controller.start("detect.schmitt", doc_id, 0,
                     params={"min_dwell_s": 5e-4, "pore_id": "FROM_PARAMS"},
                     pore_id="FROM_KW")
    cap.wait()

    assert not cap.failed, cap.failed
    res = cap.finished[0]
    assert (res.payload.df["pore_id"] == "FROM_PARAMS").all()


def test_module_instance_accepted(qapp, tmp_path):
    """start() also accepts an AnalysisModule instance, not only an id string."""
    from nanoscope.analysis.modules import DetectSchmitt

    sess, doc_id = _open(tmp_path)
    controller = AnalysisController(sess)
    cap = _Capture(controller)

    controller.start(DetectSchmitt(), doc_id, 0, params={"min_dwell_s": 5e-4})
    cap.wait()

    assert not cap.failed, cap.failed
    assert isinstance(cap.finished[0], AnalysisResult)


# ---------------------------------------------------------------------------
# failure path: module raises -> failed(str), no escaped traceback
# ---------------------------------------------------------------------------
def test_rate_unknown_channel_fails_gracefully(qapp, tmp_path):
    sess, doc_id = _open(tmp_path, fs=None, n=4_000, name="no_rate.dat")
    controller = AnalysisController(sess)
    cap = _Capture(controller)

    controller.start("detect.schmitt", doc_id, 0, params={})
    cap.wait()

    assert not cap.finished
    assert len(cap.failed) == 1
    assert "sample rate" in cap.failed[0]
    assert "RuntimeError" in cap.failed[0]
    assert not controller.is_busy()


# ---------------------------------------------------------------------------
# generic callable path (null/sweep/recovery)
# ---------------------------------------------------------------------------
def test_run_callable_emits_finished(qapp):
    controller = AnalysisController(session=None)
    cap = _Capture(controller)

    payload = {"tau_null": 1.23, "n": 7}
    controller.run_callable(lambda: payload, label="null.model")
    cap.wait()

    assert cap.started == ["null.model"]
    assert cap.finished == [payload]
    assert not cap.failed


def test_run_callable_reports_failure(qapp):
    controller = AnalysisController(session=None)
    cap = _Capture(controller)

    def boom():
        raise ValueError("surrogate exploded")

    controller.run_callable(boom, label="sweep")
    cap.wait()

    assert not cap.finished
    assert len(cap.failed) == 1
    assert "surrogate exploded" in cap.failed[0]
    assert "ValueError" in cap.failed[0]


def test_run_callable_takes_no_argument_by_default(qapp):
    """The contract every existing caller relies on: ``fn()``, zero arguments.

    Pinned because the progress channel below made the arity a CHOICE. Deciding
    it by inspecting the callable would make "does this job report progress?" a
    property of how ``fn`` happens to be spelled -- a partial, a bound method and
    a lambda with a default argument each answer differently.
    """
    controller = AnalysisController(session=None)
    cap = _Capture(controller)
    seen: list[tuple] = []

    def zero_arg(*args):
        seen.append(args)
        return "ok"

    controller.run_callable(zero_arg, label="null.model")
    cap.wait()

    assert seen == [()]
    assert cap.finished == ["ok"]
    assert cap.progress == []


def test_run_callable_with_progress_reaches_the_controllers_progressed_signal(qapp):
    """``wants_progress=True`` -> ``fn(report)``, and ``report`` is the SAME channel a
    module's :class:`AnalysisWorker` uses, so the status bar cannot tell them apart.

    Without this a minutes-long callable could only show a bar that never moves,
    which is indistinguishable from a hang. The surrogate studies lived with that
    because they take seconds; the subpopulation pipeline (480 cross-validated
    mixture fits plus a 200-resample bootstrap) does not.
    """
    controller = AnalysisController(session=None)
    cap = _Capture(controller)

    def reporting(report):
        report(0.25, "quarter")
        report(1.0, "done")
        return "payload"

    controller.run_callable(reporting, label="subpopulations", wants_progress=True)
    cap.wait()

    assert cap.finished == ["payload"]
    assert cap.progress == [(0.25, "quarter"), (1.0, "done")]


def test_run_callable_cancel_flag_maps_to_cancelled(qapp):
    controller = AnalysisController(session=None)
    cap = _Capture(controller)

    def cooperative():
        raise AnalysisCancelled()

    controller.run_callable(cooperative, label="recovery")
    cap.wait()

    assert cap.cancelled == 1
    assert not cap.finished
    assert not cap.failed


# ---------------------------------------------------------------------------
# teardown: abort detaches the worker so a late terminal signal is dropped
# ---------------------------------------------------------------------------
def test_abort_drops_late_finished(qapp, tmp_path):
    """abort() detaches the worker and emits NOTHING -- not finished, not failed, and
    not cancelled either.

    This test used to check only ``finished``/``failed``, which is why it sat green
    through C40: the detach silently did nothing (``worker.disconnect()`` raises
    ``TypeError`` under PySide6) and ``_is_active()`` waved the orphan through whenever
    ``sender()`` and ``_worker`` were both None, so abort() DID emit ``cancelled`` --
    hiding the progress bar in the middle of a project restore, since ``_clear_session``
    aborts. Assert the whole contract, or the next hole hides in the part not asserted."""
    sess, doc_id = _open(tmp_path)
    controller = AnalysisController(sess)
    cap = _Capture(controller)

    controller.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-4})
    controller.abort()

    # let any queued (now-orphaned) signal deliver
    for _ in range(20):
        qapp.processEvents()
        QtCore.QThread.msleep(20)
    qapp.processEvents()

    assert not controller.is_busy()
    assert not cap.finished, "a detached worker's finished must not reach the controller"
    assert not cap.failed
    assert cap.cancelled == 0, "abort() emits nothing -- including cancelled"


def test_start_aborts_previous_run(qapp, tmp_path):
    """A second start() aborts the first; only the second result is delivered.

    The first run is detached by the second start()'s abort(), so its terminal
    signal (if any) is dropped by the sender guard and never reaches the
    controller. We drain until the controller is idle to be insensitive to the
    order in which the detached and live signals arrive under load."""
    sess, doc_id = _open(tmp_path)
    controller = AnalysisController(sess)
    cap = _Capture(controller)

    controller.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-4})
    controller.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-4})

    cap.drain_until(lambda: not controller.is_busy() and cap.finished, qapp)

    # exactly one finished (the second run); the first was aborted before delivery
    assert len(cap.finished) == 1
    assert not cap.failed
    assert not controller.is_busy()
