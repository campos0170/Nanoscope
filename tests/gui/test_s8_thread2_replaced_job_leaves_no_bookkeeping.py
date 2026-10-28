"""S8-THREAD2 (and S8-THREAD1, same cause) -- a REPLACED job leaves its bookkeeping behind.

MEASURED DEFECT (sweep-8 concurrency lens, owner's -500 mV recording). Start a Merge-all or a
Re-measure, press Detect while it runs: the status bar says "Replaced merge_all with
detect.schmitt. The previous run's result will not be applied.", the detection lands, the
controller is idle -- and from then on the Manual study's and the Acid study's "prepare review"
actions, and the main window's study-import path, all answer "Wait for main-workspace analysis
and editing operations to finish first." with nothing running. Nothing clears it: not selecting
another recording, not a second Detect, not closing every recording. The panel records what the
job needs in ``AnalysisPanel._merge_job`` / ``_remeasure_target`` and clears them only when the job
finishes, fails or is cancelled -- and ``AnalysisController.start()`` / ``run_callable()`` replace
a running job with ``abort()``, which emits NOTHING. ``study_event_bridge._assert_idle`` reads the
leftovers as "an operation is running". S8-THREAD1 is the same cause one widget over: Detect during
"Measure this recording" leaves the probe button reading "Measuring..." (and refusing a second
click) for as long as the recording stays selected, and the Baseline-correction workspace's
"Refreshing comparison..." is stuck the same way.

THE FIX is at the one place a job is replaced. ``AnalysisController`` now announces it
(``replaced(label)``, from the shared ``_supersede`` that both ``start()`` and ``run_callable()``
go through; nothing is emitted when nothing was running, when the run gate refused the start, or
on an explicit ``abort()``), and ``AnalysisPanel`` listens (``_on_controller_replaced``): it forgets
every record of the dropped job and releases the busy button that job owned -- unless the job
taking over owns the same button (an export replacing a comparison keeps it lit). The hazard the
finding names -- the dropped worker's late answer -- turned out to have a hole of its own: the
controller's sender guard compares objects, the retired worker was freed at once, and a stale
answer naming a recycled address passed for the NEW job's, ended it and detached it (about a third
of 300 replacements in a tight loop). ``AnalysisController._stop`` now keeps a retired worker alive
until the signals it had already queued have been delivered.

These tests drive the real MainWindow: the Merge all button, the re-measure and compare entry
points, the Detect button, the probe and Refresh buttons, the status-bar Cancel, and the study
bridge's own guard. Jobs are held in flight by wrapping the function the job calls, so "something
really runs" is a fact and not a timing hope. The tests marked "passes before and after" pin
what the fix must not break: a late answer from the replaced worker must not clear or end the
NEW job's records, a replacement by a job that owns the same busy button must keep it lit, and
the abandonment routes that already worked (Cancel, project reset, closing the recording) still
do. Numbers are untouched: nothing here reaches detection maths or export.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import types

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.analysis import baseline as baseline_mod
from nanoscope.analysis import baseline_probe as probe_mod
from nanoscope.analysis import remeasure as remeasure_mod
from nanoscope.analysis.baseline import BASELINE_PARAM_KEYS, BaselineConfig
from nanoscope.analysis.cancel import check_cancelled
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import DetectSchmitt, register_all
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import analysis_panel as ap
from nanoscope.gui import worker_thread as wt
from nanoscope.gui.analysis_worker import AnalysisController
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 50_000
OLD = BaselineConfig(method="block_median", window_s=0.01)
#: What the study bridge says while the main workspace is busy -- the sentence in the finding.
WAIT = "Wait for main-workspace analysis and editing operations to finish first."
#: A cluster of fragments one translocation was split into (Merge all has something to do),
#: then evenly spaced single events.
_CLUSTER = [(20_000, 20_120), (20_140, 20_260), (20_280, 20_400)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


# ================================================================================================
# builders
# ================================================================================================
def _signal(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, -100.0) + rng.normal(0.0, 2.0, N)
    for s, e in _CLUSTER:
        x[s:e] -= 60.0
    t = 500
    while t + 250 < N:
        if not 19_000 < t < 21_500:
            x[t:t + 250] -= 60.0
        t += 2000
    return x.astype(np.float32)


def _doc(tmp_path, signal, name) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _open(win, tmp_path, name="a.dat", seed=3, *, detected=True) -> str:
    """A recording on the window, selected, with a stored Schmitt detection (block-median
    baseline, so the re-measure entry points have a configuration to measure against)."""
    signal = _signal(seed)
    doc_id = win._session.add_document(_doc(tmp_path, signal, name))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    if detected:
        df, prov = detect_events(signal, FS, baseline_config=OLD, polarity=-1,
                                 min_dwell_s=1e-4, recording=name, pore_id="pore")
        block = OLD.to_provenance()
        params = {"min_dwell_s": 1e-4, **{k: block[k] for k in BASELINE_PARAM_KEYS if k in block}}
        win._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params=params,
            calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
            payload=EventSet(df, prov), baseline_fingerprint=prov.get("baseline_fingerprint")))
    return doc_id


class _Hold:
    """Keeps a job in flight until the test lets it go, so "something really runs" is a fact.

    ``fn`` is a plain function (so it also binds as a method) to patch in over the one the job
    calls. Call *n* parks on its own slot, which lets a test hold two jobs of one kind apart.
    A cooperative slot polls the cancel token the way every real kernel does and leaves the
    moment the controller aborts it; an UNCOOPERATIVE slot ignores it -- the worker that will
    not stop, whose thread is detached and whose answer, if any, arrives late."""

    def __init__(self, real, *, uncooperative=(), free=()):
        self._real = real
        self._uncooperative = set(uncooperative)
        self._free = set(free)
        self._lock = threading.Lock()
        self.calls = 0
        self.slots = [types.SimpleNamespace(entered=threading.Event(), release=threading.Event(),
                                            returned=threading.Event()) for _ in range(4)]
        for i in self._free:
            self.slots[i].release.set()
        self.fn = lambda *a, **k: self._body(*a, **k)

    def _body(self, *args, **kwargs):
        with self._lock:
            i, self.calls = self.calls, self.calls + 1
        slot = self.slots[i]
        slot.entered.set()
        while not slot.release.wait(0.005):
            if i not in self._uncooperative:
                check_cancelled()
        try:
            return self._real(*args, **kwargs)
        finally:
            slot.returned.set()

    def release_all(self) -> None:
        for slot in self.slots:
            slot.release.set()


@pytest.fixture
def win(qapp, qtbot, monkeypatch):
    w = MainWindow(recovery_enabled=False)
    qtbot.addWidget(w)
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: "discard")
    yield w
    w._dirty = False
    w._teardown()
    wt.drain_detached(5000)


@pytest.fixture
def holds(win):
    """Built on ``win`` so that it is torn down FIRST: every parked job is let go before the
    window quits, and the quit never waits on a worker the test itself is holding."""
    made: list[_Hold] = []

    def make(real, **kw) -> _Hold:
        made.append(_Hold(real, **kw))
        return made[-1]

    yield make
    for h in made:
        h.release_all()


class _Terminals:
    """The controller's terminal signals. A replaced job must produce none of them."""

    def __init__(self, ctl: AnalysisController) -> None:
        self.finished: list[object] = []
        self.failed: list[str] = []
        self.cancelled = 0
        ctl.finished.connect(self.finished.append)
        ctl.failed.connect(self.failed.append)
        ctl.cancelled.connect(self._cancelled)

    def _cancelled(self) -> None:
        self.cancelled += 1


# ================================================================================================
# what the user does, and what the study bridge asks
# ================================================================================================
def _guard(win) -> str | None:
    """The guard in front of the Manual / Acid study's "prepare review" actions and the main
    window's study-import path. ``None`` = it lets the user through; else its refusal."""
    from nanoscope.gui.study_event_bridge import _assert_idle
    try:
        _assert_idle(win)
    except ValueError as exc:
        return str(exc)
    return None


def _import_refusal(win) -> str:
    """What the main window's study-import door says. ``snapshot_study_events`` asks the same
    guard first; with an empty record it stops at "Prepare this study recording..." once past it."""
    with pytest.raises(ValueError) as caught:
        win.snapshot_study_events({})
    return str(caught.value)


def _start_job(win, holds, monkeypatch, kind: str, **hold_kw) -> _Hold:
    """Start a Merge-all / Re-measure / Compare through the panel and hold it in flight."""
    panel = win._analysis_panel
    if kind == "merge":
        hold = holds(ap.merge_by_gap, **hold_kw)
        monkeypatch.setattr(ap, "merge_by_gap", hold.fn)
        panel.merge_controls.set_window_s(0.01)
        panel.merge_controls.merge_all_btn.click()
    else:
        hold = holds(remeasure_mod.remeasure_events, **hold_kw)
        monkeypatch.setattr(remeasure_mod, "remeasure_events", hold.fn)
        entry = (panel.remeasure_active_detection if kind == "remeasure"
                 else panel.compare_baseline_estimators)
        assert entry() == "started"
    assert hold.slots[0].entered.wait(10), f"{kind}: the job never reached its worker"
    assert win._analysis_controller.is_busy()
    return hold


def _hold_probe(holds, monkeypatch) -> _Hold:
    """Hold "Measure this recording" in flight. The probe answers with a stand-in that is not a
    ``BaselineProposal`` (so the panel ignores it on arrival) and the health pass is refused, which
    the job already tolerates: these tests are about the button, not about the proposal."""
    proposal = types.SimpleNamespace(config=None, probe=None)
    probe = holds(lambda data, fs, *, polarity: proposal)

    def no_health(*_a, **_k):
        raise RuntimeError("no health pass in this test")

    monkeypatch.setattr(probe_mod, "probe_and_propose", probe.fn)
    monkeypatch.setattr(baseline_mod, "characterisation_baseline", no_health)
    return probe


def _press_detect(win) -> None:
    button = win._analysis_panel._detection.detect_button
    assert button.isEnabled(), "the Detect button is gated off on this recording"
    button.click()


def _settle(win, qtbot) -> None:
    qtbot.waitUntil(lambda: not win._analysis_controller.is_busy(), timeout=60_000)
    qtbot.wait(20)


def _replace_with_detect(win, holds, monkeypatch, qtbot, kind: str) -> _Terminals:
    """Job A of ``kind`` in flight, Detect pressed over it, everything settled."""
    terminals = _Terminals(win._analysis_controller)
    started: list[str] = []
    win._analysis_controller.started.connect(started.append)
    _start_job(win, holds, monkeypatch, kind)
    assert _guard(win) == WAIT, "control: the guard must refuse while the job really runs"
    _press_detect(win)
    _settle(win, qtbot)
    assert started[-1] == "detect.schmitt", started
    assert terminals.cancelled == 0 and not terminals.failed, (
        "the replaced job must not report a terminal signal; that is what makes it a replacement")
    assert len(terminals.finished) == 1 and isinstance(terminals.finished[0], AnalysisResult)
    return terminals


KINDS = ["merge", "remeasure", "compare"]


# ================================================================================================
# 1. THE DEFECT: the study guard after Detect replaced a Merge-all / Re-measure / Compare
# ================================================================================================
@pytest.mark.parametrize("kind", KINDS)
def test_the_study_guard_passes_once_detect_has_replaced_the_job(
        win, holds, monkeypatch, qtbot, tmp_path, kind):
    """FAILING BEFORE -- the finding's own sequence. The guard refused for the rest of the
    session with the controller idle; both doors the finding names must now let the user in."""
    _open(win, tmp_path)
    _replace_with_detect(win, holds, monkeypatch, qtbot, kind)

    assert not win._analysis_controller.is_busy()
    assert _guard(win) is None, "the study guard still sees the replaced job as running"
    assert WAIT not in _import_refusal(win), "the study-import path still says to wait"


@pytest.mark.parametrize("kind", ["merge", "remeasure"])
@pytest.mark.parametrize("step", ["another recording selected", "a second Detect",
                                  "every recording closed"])
def test_nothing_brings_the_leftover_back_afterwards(
        win, holds, monkeypatch, qtbot, tmp_path, kind, step):
    """FAILING BEFORE -- the finding measured the guard refusing after each of these as well; the
    fix must leave it open through every one of them, not merely right after the replacement."""
    first = _open(win, tmp_path)
    second = _open(win, tmp_path, "b.dat", seed=4, detected=False)
    win._on_selection(first, 0)
    _replace_with_detect(win, holds, monkeypatch, qtbot, kind)

    if step == "another recording selected":
        win._on_selection(second, 0)
        qtbot.wait(20)
    elif step == "a second Detect":
        _press_detect(win)
        _settle(win, qtbot)
    else:
        for doc in list(win._session.documents()):
            win._session.close_document(doc.doc_id)
        qtbot.wait(20)
    assert _guard(win) is None, f"the guard refuses again after: {step}"


def test_a_job_that_replaces_it_need_not_be_a_detection(win, holds, monkeypatch, qtbot, tmp_path):
    """FAILING BEFORE -- the fix is at the controller, not at Detect. "Measure this recording"
    also replaces a running Merge-all; while the probe runs its button reads "Measuring..."
    (the replacement must not take the NEW job's busy state away), and when it ends the guard
    is open and the button is back."""
    _open(win, tmp_path)
    probe = _hold_probe(holds, monkeypatch)
    _start_job(win, holds, monkeypatch, "merge")
    bp = win._analysis_panel.baseline_panel

    bp.probe_button.click()
    assert probe.slots[0].entered.wait(10)
    assert bp.probe_button.text() == "Measuring…" and bp.is_busy(), (
        "the replacement took the probe's own busy state away")
    assert _guard(win) == WAIT                      # the probe really runs
    probe.slots[0].release.set()
    _settle(win, qtbot)

    assert _guard(win) is None
    assert bp.probe_button.text() == "Measure this recording" and not bp.is_busy()


# ================================================================================================
# 2. S8-THREAD1: the busy buttons of the job that was replaced
# ================================================================================================
def test_detect_during_measure_this_recording_releases_the_button(
        win, holds, monkeypatch, qtbot, tmp_path):
    """FAILING BEFORE -- S8-THREAD1: the button read "Measuring..." for ever, refused a second
    click, and only a recording change cleared it."""
    _open(win, tmp_path)
    probe = _hold_probe(holds, monkeypatch)
    bp = win._analysis_panel.baseline_panel

    bp.probe_button.click()
    assert probe.slots[0].entered.wait(10)
    assert bp.probe_button.text() == "Measuring…"
    _press_detect(win)
    _settle(win, qtbot)

    assert bp.probe_button.text() == "Measure this recording" and not bp.is_busy()
    bp.probe_button.click()                          # a working button starts a probe
    assert probe.slots[1].entered.wait(10), "the second click started nothing"
    assert win._analysis_controller.is_busy()


def test_detect_during_a_reference_preview_releases_the_button(
        win, holds, monkeypatch, qtbot, tmp_path):
    """FAILING BEFORE -- "Preview reference and detection" lights the same button as the probe,
    and Detect replaced it with the same result: "Measuring..." for as long as the recording
    stayed selected."""
    _open(win, tmp_path, detected=False)
    panel = win._analysis_panel
    bp = panel.baseline_panel
    editor = bp.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData("constant"))
    editor.current.setText("-100")
    preview = holds(DetectSchmitt.run, free=(1,))           # call 0 is the preview, 1 the Detect
    monkeypatch.setattr(DetectSchmitt, "run", preview.fn)

    panel._on_reference_preview_requested()
    assert preview.slots[0].entered.wait(10)
    assert bp.is_busy() and bp.probe_button.text() == "Measuring…"
    _press_detect(win)
    _settle(win, qtbot)

    assert preview.calls == 2
    assert not bp.is_busy() and bp.probe_button.text() == "Measure this recording"
    assert _guard(win) is None


def test_detect_during_a_comparison_refresh_releases_the_refresh_button(
        win, holds, monkeypatch, qtbot, tmp_path):
    """FAILING BEFORE -- the Baseline-correction workspace is stuck the same way: its Refresh
    button read "Refreshing comparison..." after Detect had replaced the job it stood for, and
    its status line went on saying the comparison was being calculated."""
    _open(win, tmp_path)
    compare = holds(ap._baseline_workspace_arrays)
    monkeypatch.setattr(ap, "_baseline_workspace_arrays", compare.fn)
    workspace = win._analysis_panel._baseline_workspace

    workspace.refresh_button.click()
    assert compare.slots[0].entered.wait(10)
    assert workspace.is_busy() and workspace.refresh_button.text() == "Refreshing comparison…"
    _press_detect(win)
    _settle(win, qtbot)

    assert not workspace.is_busy()
    assert workspace.refresh_button.text() == "Refresh comparison"
    assert "Calculating" not in workspace.preview_status.text(), (
        "nothing is calculating any more, and the status line must not say so")
    assert _guard(win) is None


def test_an_export_replacing_a_running_comparison_keeps_the_button_busy(
        win, holds, monkeypatch, qtbot, tmp_path):
    """Passes before and after -- it pins the hazard of the fix. The workspace's own buttons
    refuse a second press while busy, but the export entry the menu and scripts use is not
    behind them: it REPLACES the running comparison. Releasing the busy state for the dropped
    comparison must not release the one the export needs."""
    _open(win, tmp_path)
    compare = holds(ap._baseline_workspace_arrays)
    monkeypatch.setattr(ap, "_baseline_workspace_arrays", compare.fn)
    panel = win._analysis_panel
    workspace = panel._baseline_workspace

    workspace.refresh_button.click()
    assert compare.slots[0].entered.wait(10)
    assert panel.export_baseline_correction(tmp_path / "baseline.csv")
    assert compare.slots[1].entered.wait(10)
    assert workspace.is_busy() and workspace.refresh_button.text() == "Refreshing comparison…", (
        "the replaced comparison released the busy state the export is using")

    compare.release_all()
    _settle(win, qtbot)
    assert not workspace.is_busy() and workspace.refresh_button.text() == "Refresh comparison"
    assert (tmp_path / "baseline.csv").exists()


def test_the_busy_buttons_follow_the_job_that_took_over(win):
    """Each busy button is released when a job that does not own it takes over, and only then.
    The three jobs that own the Baseline panel's button and the two that own the workspace's
    are named here so a label that drifts from the call sites fails loudly."""
    panel, ctl = win._analysis_panel, win._analysis_controller
    bp, ws = panel.baseline_panel, panel._baseline_workspace
    cases = [
        # label of the job taking over,       Baseline panel stays busy, workspace stays busy
        ("baseline_probe",                    True,  False),
        ("open_pore_reference_preview",       True,  False),
        ("baseline_correction_preview",       False, True),
        ("baseline_correction_export",        False, True),
        ("detect.schmitt",                    False, False),
        ("merge_all",                         False, False),
        ("subpopulations",                    False, False),
    ]
    for label, probe_stays, workspace_stays in cases:
        bp.set_busy(True)
        ws.set_busy(True)
        ctl.replaced.emit(label)
        assert bp.is_busy() is probe_stays, label
        assert ws.is_busy() is workspace_stays, label


# ================================================================================================
# 3. THE HAZARD: a late signal from the replaced worker must not reach the NEW job's records
# ================================================================================================
def test_a_finished_signal_already_queued_when_the_job_is_replaced_is_ignored(
        win, holds, monkeypatch, qtbot, tmp_path):
    """FAILING BEFORE -- at the end, where the replaced Merge-all's check-list still blocks the
    study guard, and now and then in the middle, where the stale answer takes the Detect for its
    sender (the hole ``AnalysisController._stop`` closes; the controller-level test below pins
    it deterministically). The Merge-all's worker has FINISHED and its answer is queued for the
    main thread, which has not run its event loop yet, when Detect is pressed. When the loop
    turns, that answer is delivered against a controller that now owns the Detect: it must be
    dropped -- not applied, not announced, and not mistaken for the Detect's own end (which would
    end the Detect, detach it and lose its result while the progress bar says it is running)."""
    _open(win, tmp_path)
    merge = holds(ap.merge_by_gap, free=(0,))
    monkeypatch.setattr(ap, "merge_by_gap", merge.fn)
    detect = holds(DetectSchmitt.run)
    monkeypatch.setattr(DetectSchmitt, "run", detect.fn)
    reports: list[object] = []
    win._analysis_panel.mergeFinished.connect(reports.append)
    ctl = win._analysis_controller
    terminals = _Terminals(ctl)
    before = win._analysis_panel.current_detection()

    win._analysis_panel.merge_controls.set_window_s(0.01)
    win._analysis_panel.merge_controls.merge_all_btn.click()
    assert merge.slots[0].returned.wait(10)
    QtCore.QThread.msleep(150)                       # no event loop: the answer stays queued
    assert ctl.is_busy(), "control: the finished answer must still be waiting in the queue"
    _press_detect(win)
    assert detect.slots[0].entered.wait(10)
    qtbot.wait(100)                                  # the queued answer is delivered now

    assert not reports, f"the replaced merge's answer was announced: {reports}"
    assert ctl.is_busy(), (
        "the stale answer ended the Detect that replaced its job "
        f"(finished={terminals.finished!r} failed={terminals.failed!r} "
        f"cancelled={terminals.cancelled})")
    assert _guard(win) == WAIT                       # the Detect really runs
    assert win._analysis_panel.current_detection() is before, "the stale answer was applied"

    detect.release_all()
    _settle(win, qtbot)
    assert win._analysis_panel.current_detection() is not before
    assert not reports
    assert _guard(win) is None


def test_a_late_answer_from_the_replaced_job_does_not_touch_the_new_jobs_bookkeeping(
        win, holds, monkeypatch, qtbot, tmp_path):
    """Passes before and after -- job identity, not job kind. Merge-all A will not stop; Detect
    replaces it; a SECOND Merge-all C then runs. A's worker finally returns while C is in
    flight. C is the same kind as A, so a guard that compared kinds would take A's answer for
    C's: C's records must survive it and C must still deliver its own answer, exactly once."""
    _open(win, tmp_path)
    merge = _start_job(win, holds, monkeypatch, "merge", uncooperative=(0,))
    reports: list[object] = []
    win._analysis_panel.mergeFinished.connect(reports.append)
    ctl = win._analysis_controller

    _press_detect(win)
    _settle(win, qtbot)
    win._analysis_panel.merge_controls.merge_all_btn.click()      # C
    assert merge.slots[1].entered.wait(10)
    assert not reports

    merge.slots[0].release.set()                     # A's late answer
    assert merge.slots[0].returned.wait(10)
    qtbot.wait(100)

    assert ctl.is_busy() and _guard(win) == WAIT, "A's late answer ended C"
    assert not reports, f"A's late answer was announced: {reports}"
    merge.slots[1].release.set()
    _settle(win, qtbot)
    assert len(reports) == 1, "C must deliver exactly one answer of its own"
    assert reports[0].status in {"ok", "noop", "low_occupancy"}
    assert _guard(win) is None


# ================================================================================================
# 4. The ways of abandoning a job that already worked, and the ordinary end of one
# ================================================================================================
@pytest.mark.parametrize("kind", ["merge", "remeasure"])
def test_a_job_that_runs_to_its_end_clears_its_own_records(
        win, holds, monkeypatch, qtbot, tmp_path, kind):
    """Passes before and after: the guard refuses while the job really runs and opens when it
    finishes. Pins that the fix did not trade one half for the other."""
    _open(win, tmp_path)
    hold = _start_job(win, holds, monkeypatch, kind)
    assert _guard(win) == WAIT
    hold.release_all()
    _settle(win, qtbot)
    assert _guard(win) is None


@pytest.mark.parametrize("route", ["status-bar Cancel", "project reset", "recording closed"])
def test_cancel_reset_and_close_leave_the_guard_open(
        win, holds, monkeypatch, qtbot, tmp_path, route):
    """Passes before and after -- the routes that already cleared their records. Closing the
    recording does not stop the job (it runs to its end and drops its answer), so the guard
    rightly refuses until then."""
    doc_id = _open(win, tmp_path)
    hold = _start_job(win, holds, monkeypatch, "merge")
    assert _guard(win) == WAIT

    if route == "status-bar Cancel":
        win._cancel_btn.click()
    elif route == "project reset":
        win._clear_session()
    else:
        win._session.close_document(doc_id)
        assert _guard(win) == WAIT, "a job is still running"
        hold.release_all()
        _settle(win, qtbot)
    assert _guard(win) is None


# ================================================================================================
# 5. The controller's side of the contract
# ================================================================================================
def _plain_doc(tmp_path):
    return _doc(tmp_path, _signal(), "c.dat")


@pytest.mark.parametrize("outcome", ["answer", "failure"])
def test_a_queued_answer_of_a_replaced_job_never_reaches_the_job_that_replaced_it(qapp, outcome):
    """FAILING BEFORE -- the hole in the sender guard. Job A ends just as job B replaces it, so
    A's answer (or its failure) is already in the main thread's queue when B starts. It must be
    dropped. The controller tells A's signal from B's by ``sender()``, but A's worker was freed
    the moment it was retired and B's was built at the address it left; the stale signal then
    named B as its sender, passed the guard, ended B and detached it -- B's own answer was never
    delivered, and a failure would have opened "Analysis failed" over a job that was fine.
    Measured in this very loop: about a third of 300 replacements lost the new job."""
    ctl = AnalysisController(Session())
    delivered: list[object] = []
    ctl.finished.connect(lambda r: delivered.append(("finished", r)))
    ctl.failed.connect(lambda m: delivered.append(("failed", m)))
    lost = 0

    def boom():
        raise RuntimeError("A failed")

    for i in range(40):
        delivered.clear()
        job_a = boom if outcome == "failure" else (lambda i=i: ("A", i))
        assert ctl.run_callable(job_a, label="a")
        QtCore.QThread.msleep(15)            # A has ended; its signal is queued, undelivered
        release = threading.Event()

        def parked():
            while not release.wait(0.002):
                check_cancelled()
            return "B"

        assert ctl.run_callable(parked, label="b")
        for _ in range(4):
            qapp.processEvents(QtCore.QEventLoop.AllEvents, 5)
        if not ctl.is_busy() or delivered:
            lost += 1                        # A's signal was taken for B's
        release.set()
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while ctl.is_busy() and deadline.elapsed() < 5000:
            qapp.processEvents(QtCore.QEventLoop.AllEvents, 5)
            QtCore.QThread.msleep(2)
        assert delivered in ([], [("finished", "B")]), f"round {i}: {delivered}"
        ctl.abort()
    wt.drain_detached(3000)
    assert lost == 0, f"A's stale signal took B for its sender in {lost} of 40 replacements"


def test_the_controller_announces_a_replacement_once_and_only_then(qapp, tmp_path):
    """``replaced(label)`` fires from ``start()`` and ``run_callable()`` alike, after the old
    job is gone and before the new one has started -- the moment no caller has recorded its new
    job yet. Not for a first job, not for a start the run gate refused, not for ``abort()``."""
    session = Session()
    doc_id = session.add_document(_plain_doc(tmp_path))
    ctl = AnalysisController(session)
    seen: list[tuple[str, str, bool]] = []
    ctl.started.connect(lambda label: seen.append(("started", label, ctl.is_busy())))
    ctl.replaced.connect(lambda label: seen.append(("replaced", label, ctl.is_busy())))
    release = threading.Event()

    def parked():
        while not release.wait(0.005):
            check_cancelled()
        return None

    try:
        assert ctl.run_callable(parked, label="first")
        assert seen == [("started", "first", True)], "nothing was running: nothing to announce"

        assert ctl.run_callable(parked, label="second")
        assert seen[1:] == [("replaced", "second", False), ("started", "second", True)]

        assert ctl.start("detect.schmitt", doc_id, 0, params={"min_dwell_s": 5e-4})
        assert seen[3:] == [("replaced", "detect.schmitt", False),
                            ("started", "detect.schmitt", True)]

        ctl.abort()
        assert len(seen) == 5, "abort() announces nothing; only a replacement does"

        other = AnalysisController(session)             # a second owner holds the analysis domain
        assert other.run_callable(parked, label="theirs")
        refused: list[str] = []
        ctl.refused.connect(refused.append)
        assert not ctl.run_callable(parked, label="mine")
        assert refused and len(seen) == 5, "a refused start replaces nothing and announces nothing"
        assert other.is_busy(), "the refused start must not have stopped the other owner's job"
        other.abort()
    finally:
        release.set()
        wt.drain_detached(5000)
