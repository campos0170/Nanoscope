"""C23 + C20 + §2.6 — one background job at a time, no empty recordings, a clean quit.

Three defects, one theme: **the app must never end up in a state it cannot describe.**

* **C23** — two detections could write the same ``ResultStore`` key. ``_batch_detect``
  guarded ``batch_controller.is_busy()``; ``AnalysisPanel._run_detection`` had no busy
  check at all, and the Detect button stayed live during a batch. Measured through a real
  ``MainWindow`` on four open recordings: two writes to one key, ``(20 events, k_in=12.0)``
  from the user's click and then ``(20 events, k_in=5.0)`` from the batch, and the batch's
  answer is the one that survived. The fix is a run gate in
  :mod:`nanoscope.gui.worker_thread` -- the seam every background job already passes
  through -- not a second ``is_busy`` check at the second call site.

* **C20** — a zero-sample recording. HDF5 and TDMS both loaded one with ``issues=()``;
  selecting it raised out of the calibration panel HALFWAY THROUGH ``_on_selection``, so
  the window was left pointing at two different recordings at once and Detect ran on the
  one the user was no longer looking at.

* **§2.6** — quitting during a job that will not stop aborted the process (SIGABRT, exit
  134, 4/4 runs) because ``drain_detached``'s return value was discarded.

Every test here drives the real widgets / the real readers.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from PySide6 import QtCore

from nanoscope.analysis.cancel import OperationCancelled, cancel_scope
from nanoscope.analysis.modules import preprocess_for_detection, register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import worker_thread as wt
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
#: The package under test — taken from the imported module, never from this file's
#: location, so the AST guard below scans whichever copy of nanoscope is actually
#: loaded (a reverted out-of-tree oracle included).
_PKG = Path(nanoscope.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s in range(10_000, n - 1_000, 20_000):
        x[s:s + 300] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str, *, n: int = 200_000, seed: int = 0,
         fs: float | None = FS) -> TraceDocument:
    arr = _event_trace(n, seed)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _window(qtbot, tmp_path, n_docs: int = 4, n: int = 200_000):
    win = MainWindow()
    qtbot.addWidget(win)
    docs = [_doc(tmp_path, f"r{i}.dat", n=n, seed=i) for i in range(n_docs)]
    for d in docs:
        win._session.add_document(d)
    win._on_selection(docs[0].doc_id, 0)
    return win, docs


def _settle(qapp, predicate, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    for _ in range(20):
        qapp.processEvents()


def _idle(win) -> bool:
    return not (win._analysis_controller.is_busy() or win._batch_controller.is_busy())


class _StuckWorker(QtCore.QObject):
    """Ignores cancel entirely — the ``_import_tf`` shape, in miniature."""

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, hold_s: float = 30.0) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._hold = hold_s

    def cancel(self) -> None:
        pass

    @QtCore.Slot()
    def run(self) -> None:
        self.entered.set()
        self.release.wait(self._hold)
        self.finished.emit(None)


# ================================================================================================
# C23 — the run gate
# ================================================================================================

def test_a_detect_during_a_batch_writes_a_second_answer_to_the_same_key(qtbot, tmp_path, qapp):
    """THE defect, measured the way it was measured: writes to one ResultStore key.

    Before: ``[(20, 12.0), (20, 5.0)]`` — the user's own k_in=12 detection landed, and the
    batch then overwrote it with k_in=5. Two answers, one key, last writer wins, nothing
    said. After: exactly one write, and it is not the one that was refused."""
    win, docs = _window(qtbot, tmp_path)
    key = (docs[0].doc_id, 0, "detect.schmitt")

    writes: list[tuple[int, float]] = []
    original = win._session.add_result

    def spy(res):
        if (res.doc_id, res.channel, res.module_id) == key:
            writes.append((len(res.payload.df), res.params.get("k_in")))
        return original(res)

    win._session.add_result = spy

    win._batch_detect()
    assert win._batch_controller.is_busy()

    strict = dict(win._analysis_panel._form_params())
    strict["k_in"] = 12.0
    started = win._analysis_panel._run_detection(strict)

    assert started is False, "the single run must be refused while a batch holds the domain"
    assert not win._analysis_controller.is_busy(), "and it must not have spun a thread"

    _settle(qapp, lambda: _idle(win))
    assert len(writes) == 1, f"one key, one answer; got {writes}"
    assert writes[0][1] == 5.0, "the batch's answer — the refused click wrote nothing"


def test_the_refusal_names_the_run_that_is_in_the_way(qtbot, tmp_path, qapp):
    """A gate that cannot explain itself is the disabled-button failure in another form."""
    win, _docs = _window(qtbot, tmp_path)
    win._batch_detect()

    win._analysis_panel._run_detection(win._analysis_panel._form_params())
    msg = win.statusBar().currentMessage()

    assert "batch detection" in msg.lower(), msg
    assert "nothing was started" in msg.lower(), msg
    assert "cancel" in msg.lower(), msg          # ...and how to get out of it
    _settle(qapp, lambda: _idle(win))


def test_the_detect_button_stays_enabled_during_a_batch(qtbot, tmp_path, qapp):
    """Qt never delivers a click to a disabled widget, so a Detect greyed out BY the batch
    could not name the batch. The click is allowed to land and the gate answers it."""
    win, _docs = _window(qtbot, tmp_path)
    win._batch_detect()

    assert win._analysis_panel.can_run() is True
    assert win._act_run_detection.isEnabled() is True

    win._run_detection()                       # the menu action, not the panel method
    assert "already running" in win.statusBar().currentMessage().lower()
    _settle(qapp, lambda: _idle(win))


def test_a_batch_during_a_single_run_is_refused_by_the_same_gate(qtbot, tmp_path, qapp):
    """The mirror direction. ``_batch_detect`` used to answer this with a bare ``return``:
    a menu item that did nothing and said nothing."""
    win, _docs = _window(qtbot, tmp_path)
    assert win._analysis_panel._run_detection(win._analysis_panel._form_params())

    win._batch_detect()

    assert not win._batch_controller.is_busy()
    assert "already running" in win.statusBar().currentMessage().lower()
    _settle(qapp, lambda: _idle(win))


def test_refusing_is_not_cancelling(qtbot, tmp_path, qapp):
    """A start that is not going to happen must not stop the run that is already happening.

    The tempting alternative — "you cannot start, so I will stop the other one and start
    yours" — turns a stray click on Detect into a silent cancel of a batch that may be
    twenty recordings in. Nothing about the in-flight run may move: not its thread, not
    its progress chrome, not one of its answers."""
    win, docs = _window(qtbot, tmp_path)
    win._batch_detect()
    assert win._progress.isVisible() or win._progress.isVisibleTo(win.statusBar())

    win._analysis_panel._run_detection(win._analysis_panel._form_params())

    assert win._batch_controller.is_busy(), "the batch must survive the refused click"
    assert win._progress.isVisibleTo(win.statusBar()), "...and so must its progress bar"

    _settle(qapp, lambda: _idle(win))
    installed = [d for d in docs
                 if win._session.results.get(d.doc_id, 0, "detect.schmitt") is not None]
    assert len(installed) == len(docs), "the batch finished all of them"


def test_a_refused_run_leaves_no_pending_job_on_the_panel(qtbot, tmp_path, qapp):
    """``_pending`` names the in-flight job so the next controller answer can be routed.
    Set for a run that never started, it makes the NEXT job's answer look like this one's."""
    win, _docs = _window(qtbot, tmp_path)
    win._batch_detect()

    win._analysis_panel._run_detection(win._analysis_panel._form_params())
    assert win._analysis_panel._pending is None

    win._analysis_panel._on_sweep_requested(win._analysis_panel._form_params())
    assert win._analysis_panel._pending is None
    _settle(qapp, lambda: _idle(win))


def test_a_refused_remeasure_does_not_claim_to_have_started(qtbot, tmp_path, qapp):
    """This surface guarantees exactly one terminal message per STARTED job. A
    "Re-measuring 1641 events…" that no job will ever finish is the stuck status line that
    guarantee exists to prevent."""
    win, docs = _window(qtbot, tmp_path)
    assert win._analysis_panel._run_detection(win._analysis_panel._form_params())
    _settle(qapp, lambda: _idle(win))
    assert win._analysis_panel.current_detection() is not None

    # Straight to the controller: ``_batch_detect`` would first ask "re-run all, or only
    # the missing ones?" about the detection this test just made, and this test is about
    # the gate rather than about that dialog.
    win._batch_controller.start([(d.doc_id, 0) for d in docs],
                                win._analysis_panel.detection_params())
    token = win._analysis_panel.remeasure_active_detection()

    assert token == "busy"
    assert win._analysis_panel._remeasure_target is None
    _settle(qapp, lambda: _idle(win))


def test_a_controller_may_still_replace_its_own_run(qtbot, tmp_path, qapp):
    """Same domain, SAME owner: allowed, and deliberately. ``start()`` aborts its own
    predecessor first, which detaches it and drops its answer, so there is no race to lose
    — and refusing here would break every re-run button in the app."""
    win, _docs = _window(qtbot, tmp_path)
    assert win._analysis_panel._run_detection(win._analysis_panel._form_params())
    assert win._analysis_panel._run_detection(win._analysis_panel._form_params())
    _settle(qapp, lambda: _idle(win))


def test_a_cancelled_batch_stops_blocking_the_next_run(qtbot, tmp_path, qapp):
    """A DETACHED job is de-registered from the gate: its answer is already discarded by
    the sender guard, so it must not go on refusing the user's next click."""
    win, _docs = _window(qtbot, tmp_path)
    win._batch_detect()
    win._cancel_load()

    assert wt.blocking_job(wt.JOB_ANALYSIS, win._analysis_controller) is None
    assert win._analysis_panel._run_detection(win._analysis_panel._form_params())
    _settle(qapp, lambda: _idle(win))


def test_loading_a_file_is_not_blocked_by_an_analysis(qtbot, tmp_path, qapp):
    """Two domains, not one. Opening a second recording while a detection runs is a normal
    thing to do and the two touch different data; a gate that stopped it would be a gate
    that had over-reached."""
    win, _docs = _window(qtbot, tmp_path)
    assert win._analysis_panel._run_detection(win._analysis_panel._form_params())

    assert wt.blocking_job(wt.JOB_LOAD, win._controller) is None
    _settle(qapp, lambda: _idle(win))


def test_the_load_controller_retires_its_thread_when_a_load_succeeds(qtbot, tmp_path, qapp):
    """``LoadController.is_busy()`` used to answer True forever after the first successful
    load: only ``abort()`` cleared ``_thread``, and the terminal slots did not. It is what
    the Cancel button consults, and now what the run gate consults."""
    from nanoscope.gui.workers import LoadController

    csv = tmp_path / "tiny.csv"
    csv.write_text("time,current\n"
                   + "".join(f"{i * 2e-5:.6f},{100.0 + 0.1 * i:.3f}\n" for i in range(64)))

    ctrl = LoadController()
    seen: list[object] = []
    ctrl.loaded.connect(seen.append)
    ctrl.failed.connect(lambda m: seen.append(RuntimeError(m)))
    ctrl.start(csv)
    _settle(qapp, lambda: bool(seen), timeout_s=30.0)

    assert seen and not isinstance(seen[0], Exception), seen
    assert ctrl.is_busy() is False
    assert wt.blocking_job(wt.JOB_LOAD, None) is None


# ================================================================================================
# C23 — structural: nothing can fall outside the gate by omission
# ================================================================================================

def test_every_worker_thread_declares_a_domain():
    """``start_worker_thread`` takes ``domain`` and ``label`` as REQUIRED keywords, so a
    controller written later cannot fall outside the run gate by forgetting a check. This
    asserts the shape at every call site, in case a default is ever added back."""
    calls = 0
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name != "start_worker_thread":
                continue
            calls += 1
            kw = {k.arg for k in node.keywords}
            assert "domain" in kw and "label" in kw, (
                f"{path.name}:{node.lineno} starts a worker thread without declaring its "
                f"run-gate domain")
    assert calls >= 3, f"expected the three controllers to spin threads, saw {calls}"


def test_the_gate_ignores_a_job_its_own_owner_holds():
    """The unit rule underneath the GUI tests above."""
    import weakref

    class _Owner:
        pass

    a, b = _Owner(), _Owner()
    thread = QtCore.QThread()
    wt._LIVE.append(wt.LiveJob(domain=wt.JOB_ANALYSIS, label="detect.schmitt",
                               thread=thread, _owner=weakref.ref(a)))
    try:
        assert wt.blocking_job(wt.JOB_ANALYSIS, a) is None
        assert wt.blocking_job(wt.JOB_ANALYSIS, b) is not None
        assert wt.blocking_job(wt.JOB_LOAD, b) is None
        assert wt.blocking_job(wt.JOB_NONE, b) is None
    finally:
        wt._LIVE.clear()


def test_an_unknown_job_token_reads_awkwardly_rather_than_anonymously():
    """A job added later with no prose entry must still name itself in the refusal."""
    assert wt.describe_job("detect.schmitt") == "A detection"
    assert wt.describe_job("some.new.study") == "some.new.study"
    assert wt.describe_job("") == "A background analysis"


def test_refusal_names_the_window_with_cancel_without_claiming_shared_recordings():
    job = wt.LiveJob(domain=wt.JOB_ANALYSIS, label="detect.schmitt", thread=QtCore.QThread())
    message = wt.refusal_text(job, "batch.detect")
    assert "one analysis job at a time" in message
    assert "nothing was started" in message
    assert "Cancel in the window running that job" in message
    assert "same recording" not in message
    assert "batch detection" in message


# ================================================================================================
# C20 — a recording with no samples
# ================================================================================================

def _empty_hdf5(tmp_path) -> Path:
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "armed_and_stopped.h5"
    with h5py.File(p, "w") as h:
        h.create_dataset("current", data=np.zeros(0, dtype=np.float64))
        h.create_dataset("voltage", data=np.zeros(0, dtype=np.float64))
    return p


def _empty_tdms(tmp_path) -> Path:
    nptdms = pytest.importorskip("nptdms")
    p = tmp_path / "armed_and_stopped.tdms"
    with nptdms.TdmsWriter(p) as w:
        w.write_segment([
            nptdms.RootObject(),
            nptdms.ChannelObject("grp", "current", np.zeros(0, dtype=np.float64),
                                 properties={"wf_increment": 1.0 / FS,
                                             "wf_start_time": 0.0}),
        ])
    return p


@pytest.mark.parametrize("make", [_empty_hdf5, _empty_tdms], ids=["hdf5", "tdms"])
def test_a_zero_sample_recording_is_refused_by_probe_and_by_load(tmp_path, make):
    """Both readers accepted it, both reported ``n_samples=0`` with ``issues=()``.

    Both doors are tested because they are separately reachable: the probe is what the
    file dialog and the >15 M-sample range prompt read, so without it the app offers to
    load, and puts a progress bar in front of, a file it is about to refuse."""
    from nanoscope.io.base import open_document, probe_path
    from nanoscope.io.errors import EmptyRecordingError

    path = make(tmp_path)
    with pytest.raises(EmptyRecordingError):
        probe_path(path)
    with pytest.raises(EmptyRecordingError):
        open_document(path)


def test_the_empty_recording_message_is_the_same_from_both_doors(tmp_path):
    """The user cannot tell which door refused, so they must not be able to tell from the
    wording either — and it must not call an intact file corrupt."""
    from nanoscope.io.base import probe_path
    from nanoscope.io.errors import EmptyRecordingError, empty_recording_message

    path = _empty_hdf5(tmp_path)
    with pytest.raises(EmptyRecordingError) as exc:
        probe_path(path)

    assert str(exc.value) == empty_recording_message(path.name)
    assert path.name in str(exc.value)
    assert "corrupt" not in str(exc.value).lower()
    assert "no samples" in str(exc.value)


def test_no_producer_can_build_a_zero_sample_recording(tmp_path):
    """The rule lives at the ONE door every recording passes through — the four readers
    and every segment cut from them — rather than four times over in the readers."""
    from nanoscope.io.errors import EmptyRecordingError

    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "z.dat", format_name="synthetic", n_samples=0,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    with pytest.raises(EmptyRecordingError):
        TraceDocument(meta=meta, channels=(np.zeros(0, dtype=np.float32),))


def test_a_two_sample_recording_still_opens(tmp_path):
    """The rule is ``== 0``, not "suspiciously short". ``core.segment`` already allows a
    two-sample hand-cut segment and this must not quietly raise that floor."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "tiny.dat", format_name="synthetic", n_samples=2,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(np.zeros(2, dtype=np.float32),))
    assert doc.meta.n_samples == 2


def test_a_raise_in_one_consumer_cannot_leave_the_window_split(qtbot, tmp_path, capsys):
    """C20's real damage. A zero-sample recording made the calibration panel raise at step
    four of ``_on_selection``, and the remaining twelve consumers never ran: the title bar,
    the calibration panel and ``_current_selection`` were on the new recording while the
    analysis panel, the Inspector and every gate were still on the old one — so Detect ran
    on the recording the user was no longer looking at.

    The empty recording is refused at the reader now, but the split was never about empty
    recordings: it is what ANY exception in ANY consumer does. Injected here directly."""
    win = MainWindow()
    qtbot.addWidget(win)
    a = _doc(tmp_path, "a.dat", n=50_000, seed=1)
    b = _doc(tmp_path, "b.dat", n=50_000, seed=2)
    win._session.add_document(a)
    win._session.add_document(b)
    win._on_selection(a.doc_id, 0)

    def boom(*_a, **_k):
        raise ValueError("zero-size array to reduction operation minimum")

    win._calibration_panel.set_selection = boom
    win._on_selection(b.doc_id, 0)

    assert win._current_selection == (b.doc_id, 0)
    assert win._analysis_panel._sel == (b.doc_id, 0), "the analysis target followed"
    assert win._act_run_detection.isEnabled() is True, "...and the gates were re-derived"
    msg = win.statusBar().currentMessage()
    assert "Calibration" in msg and "could not follow" in msg, msg
    # ...and ONLY the consumer that raised is named: the other fifteen all ran.
    assert "Analysis" not in msg and "Undo" not in msg, msg
    # ...and whoever is debugging still gets the traceback PySide6 would have printed.
    assert "ValueError" in capsys.readouterr().err


# ================================================================================================
# §2.6 — quitting while a worker will not stop
# ================================================================================================

def test_teardown_reports_how_many_threads_would_not_stop(qtbot, tmp_path, qapp):
    """The return value used to be discarded, and that discard IS the SIGABRT."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._teardown() == 0, "an idle window leaves nothing running"

    worker = _StuckWorker()
    win._analysis_controller._spin_up(worker, "sweep_threshold")
    worker.finished.connect(win._analysis_controller._on_finished)
    assert worker.entered.wait(5.0)

    try:
        assert win._teardown() == 1
    finally:
        worker.release.set()
        wt.drain_detached(5000)


def test_settings_are_synced_not_merely_set(qtbot, monkeypatch):
    """The quit may end in ``exit_without_destroying_live_threads``, which runs no
    destructors — so an unsynced QSettings would be lost on exactly the quits that are
    already going least well."""
    win = MainWindow()
    qtbot.addWidget(win)
    synced: list[int] = []
    monkeypatch.setattr(QtCore.QSettings, "sync",
                        lambda self: synced.append(1))
    win._save_settings()
    assert synced, "_save_settings must flush"


def test_clear_session_says_when_a_run_would_not_stop(qtbot, tmp_path, qapp):
    """At quit there is nobody left to tell. On File ▸ Open Project there is."""
    win = MainWindow()
    qtbot.addWidget(win)
    worker = _StuckWorker()
    win._analysis_controller._spin_up(worker, "sweep_threshold")
    worker.finished.connect(win._analysis_controller._on_finished)
    assert worker.entered.wait(5.0)

    try:
        win._clear_session()
        assert "would not stop" in win.statusBar().currentMessage()
        assert "discarded" in win.statusBar().currentMessage()
    finally:
        worker.release.set()
        wt.drain_detached(5000)


def test_quitting_with_an_unstoppable_worker_does_not_abort_the_process(tmp_path):
    """The measurement that made this a defect, end to end through the real ``gui.app.main``.

    A worker that ignores cancel, a quit while it runs. The **control** arm skips only the
    last-resort exit: the process finalises, ``worker_thread._DETACHED`` is cleared, shiboken
    deletes the C++ QThread under a live OS thread and Qt calls ``qFatal`` — SIGABRT, plus a
    macOS crash report, on every quit of this shape. The **real** arm is ``main()`` exactly
    as shipped."""
    script = textwrap.dedent(f'''
        import os, sys, time
        sys.path.insert(0, {str(_PKG.parent)!r})
        import nanoscope
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtCore, QtWidgets
        from nanoscope.gui import app as appmod
        from nanoscope.gui import main_window as mw
        from nanoscope.gui.analysis_worker import AnalysisController

        CONTROL = sys.argv[1] == "control"

        def burn():
            t0 = time.monotonic()
            while time.monotonic() - t0 < 8.0:
                pass

        class _Window:
            """Stands in for MainWindow: one uncancellable job, then quit."""

            def __init__(self):
                self.ctrl = AnalysisController(session=None)
                self.ctrl.run_callable(burn, label="sweep_threshold")

            def show(self):
                QtCore.QTimer.singleShot(700, self._quit)

            def _quit(self):
                self.ctrl.abort()                       # what _teardown does
                QtWidgets.QApplication.instance().quit()

        mw.MainWindow = _Window
        if CONTROL:
            # Exactly main(), minus the last-resort exit — i.e. the tree as it was.
            # Patched on worker_thread, not on app: main() imports the name at call time.
            from nanoscope.gui import worker_thread as wt
            wt.exit_without_destroying_live_threads = lambda code: None
        sys.exit(appmod.main([]))
    ''')
    path = tmp_path / "quit_probe.py"
    path.write_text(script)

    control = subprocess.run([sys.executable, str(path), "control"],
                             capture_output=True, timeout=180)
    real = subprocess.run([sys.executable, str(path), "real"],
                          capture_output=True, timeout=180)

    # -6 from subprocess (SIGABRT), which a shell reports as 134. Both spellings, one signal.
    assert control.returncode in (-6, 134), (
        f"the control arm should abort; got {control.returncode}\n{control.stderr[-2000:]}")
    assert b"Destroyed while thread" in control.stderr
    assert real.returncode == 0, (real.returncode, real.stderr[-2000:])
    assert b"Destroyed while thread" not in real.stderr


def test_import_tf_polls_the_cancel_token():
    """4.10 s, polling nothing. The poll on the way IN is the one that earns its place:
    ``MLDetect.run`` conditions the whole recording BEFORE it reaches a model, so a user
    who cancelled during preprocessing used to go on to pay the full import of a library
    the run had already been told to abandon."""
    from nanoscope.analysis import ml_detect

    saved = ml_detect._TF
    ml_detect._TF = None
    try:
        with cancel_scope(lambda: True):
            with pytest.raises(OperationCancelled):
                ml_detect._import_tf()
        assert ml_detect._TF is None, "and it did not import TensorFlow on the way out"
    finally:
        ml_detect._TF = saved


def test_preprocess_for_detection_stops_between_stages():
    """Each stage is one monolithic scipy call nothing can interrupt, so a poll BETWEEN
    them is the finest grain available. Worst-case latency: one stage, not the chain."""
    x = np.ascontiguousarray(np.random.default_rng(0).normal(0, 1, 200_000), np.float32)
    params = {"notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0,
              "notch_harmonics": 3, "savgol_enabled": True, "savgol_window_s": 0.0005,
              "savgol_polyorder": 3}

    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            preprocess_for_detection(x, FS, params)

    # ...and with no token installed it is exactly the function it always was.
    out = preprocess_for_detection(x, FS, params)
    assert out.shape == x.shape and out.dtype == np.float32


def test_a_cancel_arriving_mid_chain_is_seen_at_the_next_stage():
    """A token that flips after the first poll: the chain stops, it does not run to the
    end and check afterwards (which is what made Cancel a lie for these jobs)."""
    x = np.ascontiguousarray(np.random.default_rng(1).normal(0, 1, 200_000), np.float32)
    ticks = {"n": 0}

    def flag() -> bool:
        ticks["n"] += 1
        return ticks["n"] > 2

    params = {"notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0,
              "notch_harmonics": 3,
              "median_enabled": True, "median_window_s": 0.0002,
              "rmean_enabled": True, "rmean_window_s": 0.0002,
              "savgol_enabled": True, "savgol_window_s": 0.0005, "savgol_polyorder": 3}
    with cancel_scope(flag):
        with pytest.raises(OperationCancelled):
            preprocess_for_detection(x, FS, params)
    assert ticks["n"] <= 4, "it stopped at a stage boundary, not at the end"


def test_the_surrogate_chain_runs_off_the_gui_thread(qtbot, tmp_path, qapp):
    """Measured on the owner's 2,993,798-sample ABF: 81 ms with no stage enabled, 130 ms
    with the notch on, 361 ms with all six — linear in n, so his 50 M-sample ATP file pays
    ~17x that. It used to run inline, in front of the click."""
    win, _docs = _window(qtbot, tmp_path, n_docs=1, n=100_000)
    seen: list[str] = []
    import nanoscope.gui.analysis_panel as ap_mod

    real = ap_mod.preprocess_for_detection

    def spy(sig, fs, params):
        seen.append(threading.current_thread().name)
        return real(sig, fs, params)

    ap_mod.preprocess_for_detection = spy
    try:
        params = dict(win._analysis_panel._form_params())
        params["notch_enabled"] = True
        win._analysis_panel._on_sweep_requested(params)
        _settle(qapp, lambda: bool(seen), timeout_s=30.0)
    finally:
        ap_mod.preprocess_for_detection = real
        win._cancel_load()
        wt.drain_detached(5000)

    assert seen, "the chain never ran"
    assert seen[0] != threading.main_thread().name, (
        f"conditioning ran on {seen[0]} — that is the GUI thread")


def test_a_controller_that_vanished_mid_run_does_not_block_the_app_forever(qapp):
    """The one path that never reaches ``retire_worker_thread``: a controller collected
    while its worker was still running (a window dropped mid-run). Its registry entry can
    never receive an answer, so it must not go on refusing everybody else's — and holding
    it would also pin the QThread it names."""
    import gc
    import weakref

    from nanoscope.gui.analysis_worker import AnalysisController

    doomed = AnalysisController(session=None)
    worker = _StuckWorker()
    doomed._spin_up(worker, "sweep_threshold")
    assert worker.entered.wait(5.0)

    survivor = AnalysisController(session=None)
    assert wt.blocking_job(wt.JOB_ANALYSIS, survivor) is not None, "a LIVE job blocks"

    ref = weakref.ref(doomed)
    del doomed
    gc.collect()
    assert ref() is None, "the controller really was collected"

    try:
        assert wt.blocking_job(wt.JOB_ANALYSIS, survivor) is None
    finally:
        worker.release.set()
        wt.drain_detached(5000)


def test_the_gate_stays_shut_until_the_controller_retires_the_job(qtbot, tmp_path, qapp):
    """A worker's ``finished`` ends the thread's event loop BEFORE its controller's slot
    installs the result. A gate that opened on ``thread.isFinished()`` would let a second
    owner start while the first answer was still queued — the same two-answers-for-one-key
    defect, in a smaller window."""
    win, _docs = _window(qtbot, tmp_path, n_docs=1, n=100_000)
    assert win._analysis_panel._run_detection(win._analysis_panel._form_params())

    saw_open_gate_while_busy = False
    deadline = time.time() + 30.0
    while win._analysis_controller.is_busy() and time.time() < deadline:
        if wt.blocking_job(wt.JOB_ANALYSIS, win._batch_controller) is None:
            saw_open_gate_while_busy = True
            break
        qapp.processEvents()

    assert not saw_open_gate_while_busy, (
        "the gate opened while the controller still reported itself busy")
    _settle(qapp, lambda: _idle(win))
    assert wt.blocking_job(wt.JOB_ANALYSIS, win._batch_controller) is None
