"""C19 + C40 regressions: Cancel must stop the work, not the application.

C19 -- the original failure, reproduced four times out of four on the owner's own
data before the fix. ``AnalysisController.abort()`` answered a worker that would
not stop with::

    thread.quit()
    if not thread.wait(5000):
        thread.terminate()
        thread.wait()          # <- UNBOUNDED

``terminate()`` kills a thread wherever it happens to be, which for a numpy-heavy
analysis means while it holds the CPython GIL; the GIL is then never released and
the unbounded ``wait()`` can never be satisfied. Cancelling ``sweep_threshold``
0.5 s into an 18.9 s run on the 2.99 M-sample -600 mV apo recording hung the
process every time -- 0% CPU, ``take_gil``/``__psynch_mutexwait``, Force Quit the
only exit, every unsaved detection/measurement/calibration lost. The same three
lines were copied into the batch controller and the load controller, and the same
``abort()`` is reached from ``closeEvent``, from File > Open Project, and from
``AnalysisController.start`` -- so clicking Detect during a sweep took this path.

C40 -- the "detached worker" protection had NEVER executed: ``worker.disconnect()``
raises ``TypeError: not enough arguments`` under PySide6, the ``except`` swallowed
it, the slots stayed connected; and ``_is_active()`` returned True whenever BOTH
``_worker`` and ``sender()`` were None, so ``abort()`` -- documented to emit
nothing -- emitted ``cancelled``.

Timings, same machine, ``sweep_threshold`` on the -600 mV apo ABF, Cancel at 0.5 s:

===================================  =====================  ====================
                                     before                 after
===================================  =====================  ====================
abort() returns after                18.40 s / 19.09 s /    0.05-0.08 s
                                     never (2 of 4 runs)
worst GUI freeze                     >= 17.8 s              0.08 s
outcome                              4 of 4 runs HUNG       4 of 4 clean
===================================  =====================  ====================
"""

from __future__ import annotations

import ast
import pathlib
import threading
import time

import numpy as np
import pytest
from PySide6 import QtCore

import nanoscope.gui.analysis_worker as aw_mod
import nanoscope.gui.workers as workers_mod
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.truth import SynthRecording, sweep_threshold
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_worker import AnalysisController, BatchDetectController
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.worker_thread import (
    JOB_NONE,
    detach_worker,
    detached_count,
    drain_detached,
    retire_worker_thread,
    start_worker_thread,
)

FS = 100_000.0

#: The owner's smallest apo recording (2.99 M samples, -600 mV) -- the one the C19
#: measurements were taken on. Optional: the deterministic synthetic test below covers
#: the same failure without it.
_REAL_ABF = pathlib.Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf"
)


# ================================================================================================
# helpers
# ================================================================================================

def _event_trace(n: int = 600_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, -100.0) + rng.normal(0.0, 2.0, n)
    for s in range(5_000, n - 5_000, 4_000):
        x[s:s + 250] += 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _sweep_job(n: int = 600_000):
    """A REAL, slow analysis job: ~2 s, the shape of what the user actually cancels."""
    rec = SynthRecording(data=_event_trace(n), sample_rate_hz=FS, polarity=1, name="synth")
    return lambda: sweep_threshold(rec)


class _Heartbeat:
    """A main-thread QTimer. The largest gap between ticks IS the GUI freeze."""

    def __init__(self, interval_ms: int = 20) -> None:
        self._ticks: list[float] = []
        self._timer = QtCore.QTimer()
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(lambda: self._ticks.append(time.perf_counter()))

    def __enter__(self) -> "_Heartbeat":
        self._timer.start()
        return self

    def __exit__(self, *exc) -> None:
        self._timer.stop()

    def worst_gap_s(self) -> float:
        if len(self._ticks) < 2:
            return float("inf")   # never ticked: the GUI thread never ran
        return float(np.diff(np.asarray(self._ticks)).max())


def _pump(qapp, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 5)


def _doc(tmp_path, n: int = 200_000, name: str = "cancel.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    arr = _event_trace(n)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


class _StuckWorker(QtCore.QObject):
    """A worker that ignores cancel entirely -- the case that used to be terminated."""

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Event()

    def cancel(self) -> None:
        pass   # deliberately deaf

    @QtCore.Slot()
    def run(self) -> None:
        self.entered.set()
        self.release.wait(30.0)
        self.finished.emit("done")


# ================================================================================================
# C19: the deadlock itself
# ================================================================================================

def test_no_gui_module_terminates_a_thread():
    """``QThread.terminate()`` must not appear anywhere. It kills a thread at an
    arbitrary instruction -- for this app, almost always one holding the GIL -- and no
    amount of care around the call makes that safe. Three copies of it existed."""
    offenders = []
    for path in sorted(pathlib.Path(aw_mod.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "terminate"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"QThread.terminate() called at {offenders}"


def test_no_controller_waits_on_a_thread_without_a_bound():
    """``thread.wait()`` with no timeout is the second half of the deadlock: it needs the
    GIL the terminated worker never released."""
    offenders = []
    for path in sorted(pathlib.Path(aw_mod.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "wait" and not node.args and not node.keywords):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"unbounded QThread.wait() at {offenders}"


def test_cancelling_a_long_analysis_returns_at_once_and_the_gui_stays_alive(qapp):
    """THE regression. A real ~2 s ``sweep_threshold`` on a background thread, cancelled
    0.2 s in.

    Before the fix ``abort()`` blocked the GUI thread for the whole remaining run (and,
    on the owner's real data where the run is 18.9 s, then terminated the worker and
    hung the process). After it, abort returns in well under a tenth of a second because
    the work genuinely stops."""
    controller = AnalysisController(session=None)
    finished: list[object] = []
    cancelled: list[int] = []
    controller.finished.connect(finished.append)
    controller.cancelled.connect(lambda: cancelled.append(1))

    with _Heartbeat() as beat:
        controller.run_callable(_sweep_job(), label="sweep_threshold")
        _pump(qapp, 0.2)

        t0 = time.perf_counter()
        stopped = controller.abort_stopped()
        abort_s = time.perf_counter() - t0

        _pump(qapp, 0.4)

    assert abort_s < 0.5, f"Cancel blocked the GUI thread for {abort_s:.2f}s"
    assert stopped, "the worker did not actually stop; it should have, the job polls"
    assert beat.worst_gap_s() < 0.5, f"GUI froze for {beat.worst_gap_s():.2f}s"
    assert not controller.is_busy()
    assert finished == [], "a cancelled run installed its result"
    assert cancelled == [], "abort() emits nothing (C40)"


def test_a_cancelled_detection_installs_no_result(qapp, tmp_path):
    """"Partial state must be discarded, not installed." The whole point of Cancel is
    that the session is left exactly as it was.

    A contract test rather than a reproduction: the pre-fix worker did drop the result
    it had already computed. What it could NOT do was stop computing it, and what the
    controller could not do was keep a DETACHED worker's result out -- see
    ``test_cancelling_a_long_analysis_...`` and the C40 group below for those."""
    sess = Session()
    doc_id = sess.add_document(_doc(tmp_path))
    controller = AnalysisController(sess)
    controller.finished.connect(sess.add_result)

    controller.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-5})
    controller.abort()
    _pump(qapp, 0.5)

    assert sess.results.get(doc_id, 0, "detect.schmitt") is None
    assert not controller.is_busy()


def test_clicking_detect_during_a_sweep_does_not_block_the_gui(qapp, tmp_path):
    """``AnalysisController.start`` aborts the in-flight run first, so Detect-during-a-
    sweep went through the SAME deadlock path as the Cancel button -- with the added
    insult that the user had not asked to stop anything, they had asked to start
    something. The new run must begin promptly and be the only one delivered."""
    sess = Session()
    doc_id = sess.add_document(_doc(tmp_path))
    controller = AnalysisController(sess)
    finished: list[object] = []
    controller.finished.connect(finished.append)

    controller.run_callable(_sweep_job(), label="sweep_threshold")
    _pump(qapp, 0.2)

    t0 = time.perf_counter()
    controller.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-5})
    start_s = time.perf_counter() - t0

    deadline = time.perf_counter() + 10.0
    while not finished and time.perf_counter() < deadline:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 5)
    _pump(qapp, 0.4)

    assert start_s < 0.5, f"launching Detect blocked the GUI for {start_s:.2f}s"
    assert len(finished) == 1, "the pre-empted sweep also delivered a result"
    from nanoscope.core.analysis import AnalysisResult
    assert isinstance(finished[0], AnalysisResult)
    drain_detached(5000)


def test_a_worker_that_will_not_stop_is_detached_not_killed(qapp):
    """The honest answer for a job that ignores cancel: sever it, park it, tell the
    truth. Never terminate it -- that is what deadlocked the app."""
    worker = _StuckWorker()
    # The funnel takes a REQUIRED domain + label so no controller can fall outside
    # the run gate by omission; this test is about thread lifetime only, so it opts
    # out on purpose.
    thread = start_worker_thread(worker, domain=JOB_NONE, label="stuck")
    assert worker.entered.wait(5.0)

    t0 = time.perf_counter()
    stopped = retire_worker_thread(worker, thread, wait_ms=50)
    elapsed = time.perf_counter() - t0

    assert stopped is False, "a stuck worker must be reported as NOT stopped"
    assert elapsed < 1.0, f"retire blocked for {elapsed:.2f}s"
    assert thread.isRunning(), "the thread must be left to finish, not killed"
    assert detached_count() == 1

    worker.release.set()
    assert drain_detached(5000) == 0
    assert detached_count() == 0


def test_a_detached_worker_cannot_reach_its_controller(qapp):
    """Its result goes nowhere: detaching is what makes leaving it running harmless."""
    controller = AnalysisController(session=None)
    seen: list[str] = []
    controller.finished.connect(lambda _r: seen.append("finished"))
    controller.failed.connect(lambda _m: seen.append("failed"))
    controller.cancelled.connect(lambda: seen.append("cancelled"))

    worker = _StuckWorker()
    controller._spin_up(worker)           # same path start()/run_callable() use
    worker.finished.connect(controller._on_finished)
    assert worker.entered.wait(5.0)

    assert controller.abort_stopped(wait_ms=50) is False
    worker.release.set()
    _pump(qapp, 0.5)

    assert seen == [], f"a detached worker reached the controller: {seen}"
    drain_detached(5000)


# ================================================================================================
# C40: the detach that never detached, and the guard that let it through
# ================================================================================================

def test_detach_worker_actually_severs_the_signal():
    """``worker.disconnect()`` -- what every controller called -- raises
    ``TypeError: not enough arguments`` under PySide6 and disconnects NOTHING."""
    worker = _StuckWorker()
    hits: list[int] = []
    worker.cancelled.connect(lambda: hits.append(1))

    with pytest.raises(TypeError):
        worker.disconnect()               # the old spelling: still raises
    worker.cancelled.emit()
    assert hits == [1], "sanity: the slot is connected"

    detach_worker(worker)                 # the spelling PySide6 actually binds
    worker.cancelled.emit()
    assert hits == [1], "detach_worker did not sever the connection"


@pytest.mark.parametrize("controller_cls", [AnalysisController, BatchDetectController])
def test_is_active_rejects_a_signal_when_no_worker_is_current(controller_cls):
    """``sender()`` is None on a direct call AND on a queued signal whose sender has been
    collected. Against ``_worker is None`` that compared EQUAL, so the guard passed and a
    detached worker's terminal signal was honoured -- which is how ``abort()``, whose
    docstring says it emits nothing, emitted ``cancelled``."""
    controller = controller_cls(session=None)
    seen: list[str] = []
    controller.cancelled.connect(lambda: seen.append("cancelled"))
    controller.finished.connect(lambda _r: seen.append("finished"))

    assert controller._worker is None
    controller._on_cancelled()            # sender() is None here
    controller._on_finished(object())

    assert seen == [], "a signal from no-one was treated as the current worker's"


def test_abort_emits_nothing_even_after_the_worker_stops(qapp, tmp_path):
    """``abort()``'s contract, now true: MainWindow resets its own chrome, and a stale
    ``cancelled`` arriving later must not hide the progress bar of whatever came next."""
    sess = Session()
    doc_id = sess.add_document(_doc(tmp_path))
    controller = AnalysisController(sess)
    seen: list[str] = []
    for name in ("finished", "failed", "cancelled"):
        getattr(controller, name).connect(lambda *_a, n=name: seen.append(n))

    controller.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-5})
    controller.abort()
    _pump(qapp, 0.6)

    assert seen == [], f"abort() emitted {seen}"


def test_batch_abort_emits_nothing(qapp, tmp_path):
    sess = Session()
    ids = [sess.add_document(_doc(tmp_path, name=f"b{i}.dat")) for i in range(2)]
    controller = BatchDetectController(sess)
    seen: list[str] = []
    for name in ("finished", "failed", "cancelled"):
        getattr(controller, name).connect(lambda *_a, n=name: seen.append(n))

    controller.start([(d, 0) for d in ids], {"min_dwell_s": 5e-5})
    controller.abort()
    _pump(qapp, 0.6)

    assert seen == [], f"batch abort() emitted {seen}"


def test_clear_session_does_not_hide_the_restores_progress_bar(qtbot, qapp):
    """The user-visible face of C40, end to end. File > Open Project calls
    ``_clear_session``, which aborts every controller; abort() emitting a stale
    ``cancelled`` ran ``_on_analysis_cancelled`` -- ``_set_progress_visible(False)`` --
    in the MIDDLE of the restore that had already started, so the restore ran blind.

    Whether the stale signal survives to delivery depends on when the orphaned worker is
    collected, so this is the symptom test; the mechanism is pinned deterministically by
    ``test_is_active_rejects_a_signal_when_no_worker_is_current``."""
    win = MainWindow()
    qtbot.addWidget(win)

    win._analysis_controller.run_callable(_sweep_job(300_000), label="sweep_threshold")
    _pump(qapp, 0.1)
    win._clear_session()

    # the restore that follows shows its own progress
    win._set_progress_visible(True)
    _pump(qapp, 0.8)     # a stale cancelled() would be delivered in here

    assert win._progress.isVisibleTo(win.statusBar()), (
        "abort() emitted a terminal signal and it hid the restore's progress bar")
    drain_detached(5000)


def test_load_controller_shares_the_same_retirement(qapp):
    """The third copy of the deadlock lived in ``LoadController.abort``."""
    assert hasattr(workers_mod.LoadController, "abort_stopped")
    controller = workers_mod.LoadController()
    assert controller.abort_stopped() is True     # nothing running: trivially stopped


# ================================================================================================
# the UI must not lie
# ================================================================================================

def test_cancel_button_is_only_offered_while_something_is_running(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)

    win._set_progress_visible(True)
    assert win._cancel_btn.isVisibleTo(win.statusBar())
    assert not win._cancel_btn.isEnabled(), "Cancel offered with nothing to cancel"

    win._set_progress_visible(False)
    assert not win._cancel_btn.isEnabled()


def test_cancel_reports_what_actually_happened(qtbot, tmp_path, qapp):
    """"Cancelled." is a claim about the world. It is only printed when it is true."""
    win = MainWindow()
    qtbot.addWidget(win)

    win._analysis_controller.run_callable(_sweep_job(300_000), label="sweep_threshold")
    win._set_progress_visible(True)
    assert win._cancel_btn.isEnabled()
    _pump(qapp, 0.15)

    win._cancel_load()
    assert "Cancelled." in win.statusBar().currentMessage()
    assert not win._progress.isVisibleTo(win.statusBar())
    assert not win._analysis_controller.is_busy()

    # ...and a job that will not stop is described as still running, not as cancelled.
    worker = _StuckWorker()
    win._analysis_controller._spin_up(worker)
    assert worker.entered.wait(5.0)
    win._cancel_load()
    assert "background" in win.statusBar().currentMessage()

    worker.release.set()
    drain_detached(5000)


def test_cancel_clears_the_panels_pending_job(qtbot, qapp):
    """``abort()`` emits nothing, so nothing reaches ``_on_controller_cancelled`` -- a
    ``_pending`` left pointing at an abandoned sweep would route the next answer wrong."""
    win = MainWindow()
    qtbot.addWidget(win)
    panel = win._analysis_panel
    panel._pending, panel._pending_sel = "sweep", ("doc", 0)

    win._cancel_load()

    assert panel._pending is None
    assert panel._pending_sel is None


# ================================================================================================
# the real thing, on the owner's own recording
# ================================================================================================

@pytest.mark.skipif(not _REAL_ABF.exists(), reason="real ABF not present")
def test_cancel_a_real_sweep_on_the_owners_apo_recording(qapp, capsys):
    """The measured case. 2.99 M samples, ``sweep_threshold`` = 18.9 s uncancelled;
    Cancel at 0.5 s used to hang the process 4 times out of 4."""
    from nanoscope.analysis.baseline import derive_polarity
    from nanoscope.io.base import open_document

    doc = open_document(_REAL_ABF, None)
    ch = next(c.index for c in doc.meta.channels if c.kind is ChannelKind.CURRENT)
    data = np.ascontiguousarray(doc.channels[ch], dtype=np.float32)
    rec = SynthRecording(data=data, sample_rate_hz=float(doc.meta.sample_rate_hz),
                         polarity=int(derive_polarity(data)), name=_REAL_ABF.name)

    controller = AnalysisController(session=None)
    installed: list[object] = []
    controller.finished.connect(installed.append)

    with _Heartbeat() as beat:
        controller.run_callable(lambda: sweep_threshold(rec), label="sweep_threshold")
        _pump(qapp, 0.5)
        t0 = time.perf_counter()
        stopped = controller.abort_stopped()
        abort_s = time.perf_counter() - t0
        _pump(qapp, 0.5)

    with capsys.disabled():
        print(f"\n[C19 real-abf] samples={data.size:,} abort={abort_s:.3f}s "
              f"stopped={stopped} worst_gui_gap={beat.worst_gap_s():.3f}s "
              f"installed={len(installed)}")

    assert abort_s < 1.0, f"Cancel blocked the GUI for {abort_s:.2f}s (was 18.4-19.1s)"
    assert beat.worst_gap_s() < 1.0
    assert installed == []
    assert not controller.is_busy()
    drain_detached(5000)


@pytest.mark.skipif(not _REAL_ABF.exists(), reason="real ABF not present")
def test_a_cancelled_real_sweep_raises_rather_than_returning_a_partial_frame(qapp):
    from nanoscope.analysis.baseline import derive_polarity
    from nanoscope.io.base import open_document

    doc = open_document(_REAL_ABF, None)
    ch = next(c.index for c in doc.meta.channels if c.kind is ChannelKind.CURRENT)
    data = np.ascontiguousarray(doc.channels[ch], dtype=np.float32)
    rec = SynthRecording(data=data, sample_rate_hz=float(doc.meta.sample_rate_hz),
                         polarity=int(derive_polarity(data)), name=_REAL_ABF.name)

    flag = threading.Event()
    timer = threading.Timer(0.4, flag.set)
    timer.start()
    try:
        with pytest.raises(OperationCancelled):
            sweep_threshold(rec, cancel=flag.is_set)
    finally:
        timer.cancel()
