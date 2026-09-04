"""GREEN — teardown stops what it OWNS, and the status bar is not a second author.

Two fixes, both of them about the gap this batch exists to close: *the mechanism is
good and the thing it says to the user is not tested against the wired application*.

**(1) ``MainWindow._teardown`` stopped workers BY NAME.** It aborted ``_controller``,
``_analysis_controller`` and ``_batch_controller`` — a list somebody has to remember to
extend — and the Plots window's fit-note / bootstrap-CI worker was added without being
added to it. ``PlotsWindow.closeEvent`` retires that worker, but Qt does not deliver a
close event to a child window when its parent closes. Measured through a real
``MainWindow`` before the fix::

    plots window parent is the MainWindow: True
    note thread started: True   running: True
    _teardown() returned                : 0        <- "nothing was still running"
    detached_count() after teardown     : 0        <- "nothing is parked"
    note thread STILL RUNNING           : True     <- ...and yet
    plots window still holds its worker : True

Both numbers are the app telling itself the quit was clean while a QThread it owns runs
on. ``gui.app.main`` reads exactly those numbers back to decide whether it must leave
without running destructors, so the one live thread went into interpreter shutdown —
which is the ``qFatal`` / exit-134 the drain-return-value fix was written to prevent,
arriving from a direction that fix could not see.

The fix is at the invariant: teardown no longer enumerates. It DISCOVERS what this
window owns — its own attributes and the Qt object tree beneath it — and asks each
holder to stop through every protocol it implements (``abort()`` for a controller,
``close()`` for a window, whose ``closeEvent`` is where a window retires its own
workers). The tests below therefore do not check that the Plots window is stopped;
they check that a holder **nothing in this tree has ever heard of** is stopped.

**(2) The status bar stated a bare class count.** H4 established that
``SubpopReport.headline()`` is the single author of the claim — the count welded to the
separation statistic and the continuum verdict — and fixed the report and the report
window. The app's status bar was left out of that pass (declared as a fence violation,
``test_fix_subpop_honesty_header.py`` says so in its own module docstring), so on the
owner's -700 mV apo run the line a reader carried away was still::

    Subpopulations: 985 of 986 region(s) passed the gate; 7 subpopulation(s)
    at n_PCs=11; 10 caveat(s).

over a partition the report itself calls NO SUBSTANTIAL STRUCTURE. The same was true of
the second status-bar sentence, *"Trace spans coloured by 7 subpopulation(s)…"*, which
counted the classes for itself out of the assignment dict.

EVERY TEST HERE DRIVES A WIRED ``MainWindow`` AND ASSERTS THE STRING THE USER SEES.
None of them asserts ``headline()``'s wording: the H4b pass owns that method and may be
rewording it right now. What is pinned is that the surface repeats it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import pathlib
import threading
import time

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import worker_thread as WT
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N_SAMPLES = 420_000
N_EVENTS = 100
_SPACING = 4_000


# ================================================================================================
# 0. where the code under test came from
# ================================================================================================

def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """Passes in BOTH trees, so a failure below cannot be an import that never happened.

    A revert run started from inside the repo silently tests the LIVE tree —
    ``PYTHONPATH`` does not beat ``cwd`` when ``cwd`` contains the package — so every
    run of this file prints where it actually imported from.
    """
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    from nanoscope.gui import main_window as MW

    for mod in (MW, SR, WT):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} came from outside {root}"
        assert p.stat().st_size > 5_000, f"{p} is a stub, not the real module"
    print(f"\n[GREEN] nanoscope resolved from {root}")


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """This module's own QSettings namespace — nothing here reads or leaves app state.

    Module-scoped, for the reason ``test_subpop_galleries`` measured: clearing the store
    between windows makes every ``MainWindow`` still alive in the process re-read its
    persisted state, and that cost grows with the number of windows.
    """
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("FupWiringSuiteGreenPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


# ================================================================================================
# 1. teardown stops what this window OWNS — including a holder nobody listed
# ================================================================================================

class _CooperativeWorker(QtCore.QObject):
    """A worker with the shape every real one in this tree has: a terminal signal, a
    plain (non-slot) ``cancel`` setting a ``threading.Event``, and a loop that polls it.

    ``cancel`` is not a ``@Slot`` for the reason ``_NoteWorker`` and ``AnalysisWorker``
    both record: a slot on an object living in the worker's thread queues BEHIND
    ``run()`` and arrives after the work it was meant to stop.
    """

    finished = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancel = threading.Event()
        self.entered = threading.Event()
        self.saw_cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        self.entered.set()
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            if self._cancel.is_set():
                self.saw_cancel.set()
                break
            time.sleep(0.005)
        self.finished.emit()


class _UnlistedWindow(QtWidgets.QWidget):
    """A standalone window this window owns, which NOTHING in ``main_window.py`` names.

    This is the whole test. The defect was not "the Plots window was forgotten" — it was
    that teardown could only stop holders somebody had remembered to write down. So the
    holder used here is one no list could contain: it is created by the test, parented to
    the MainWindow exactly as ``PlotsWindow(self._session, self)`` is, and it retires its
    worker in its own ``closeEvent``, which is the seam every standalone window in this
    tree uses.
    """

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.worker = _CooperativeWorker()
        self.thread = WT.start_worker_thread(
            self.worker, domain=WT.JOB_NONE, label="unlisted-ci", owner=self)
        self.retired = False

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.retired = True
        worker, thread = self.worker, self.thread
        self.worker, self.thread = None, None
        if worker is not None:
            worker.cancel()
            WT.retire_worker_thread(worker, thread, wait_ms=2000)
        super().closeEvent(event)


class _UnlistedController(QtCore.QObject):
    """A controller this window owns, which nothing names either. Stops via ``abort()``."""

    def __init__(self) -> None:
        super().__init__()
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


@pytest.fixture
def bare_win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    WT.drain_detached(5000)


def test_teardown_stops_a_worker_owning_window_no_list_has_ever_named(bare_win, qapp):
    """THE INVARIANT. Not "the Plots window is stopped" — *an unlisted holder* is.

    Fails against the enumerating ``_teardown``: that one aborts three controllers by
    attribute name and this window is none of them, so the thread runs on and
    ``detached_count()`` reports the quit as clean.
    """
    win = bare_win
    held = _UnlistedWindow(win)
    held.show()
    assert held.worker.entered.wait(5.0), "the worker never started; nothing to stop"
    assert held.thread.isRunning()

    assert WT.detached_count() == 0
    still_running = win._teardown()

    assert held.retired, (
        "teardown never asked the window to stop — it is still enumerating holders")
    assert held.worker is None
    assert still_running == 0, (
        "teardown reported a worker it could not stop; a cooperative one stops in ms")
    assert WT.detached_count() == 0, (
        "a live QThread was left parked at quit — gui.app.main reads exactly this number")


def test_teardown_stops_an_unlisted_controller_through_abort(bare_win):
    """The other protocol, discovered the same way: an owned object with ``abort()``."""
    win = bare_win
    win._some_controller_added_next_year = _UnlistedController()

    win._teardown()

    assert win._some_controller_added_next_year.aborted, (
        "an owned controller was not aborted — teardown is still using a name list")


def test_teardown_still_stops_the_three_controllers_it_used_to_name(bare_win, monkeypatch):
    """Widening the net must not have dropped anything that was in the old list."""
    win = bare_win
    stopped: list[str] = []
    for name in ("_controller", "_analysis_controller", "_batch_controller"):
        controller = getattr(win, name)
        monkeypatch.setattr(controller, "abort",
                            lambda *_a, _n=name, **_k: stopped.append(_n))

    win._teardown()

    assert sorted(stopped) == ["_analysis_controller", "_batch_controller", "_controller"]


def test_the_real_plots_window_worker_is_stopped_by_a_quit(bare_win, qapp):
    """The measured defect itself, on the real ``PlotsWindow`` and its real note worker.

    The job polls the ambient cancel token, which is what every note this panel defers
    actually does (``fit_dwell_peaked``'s bootstrap loop polls it inside its replicates).
    """
    from nanoscope.analysis.cancel import check_cancelled

    win = bare_win
    win._open_plots()
    plots = win._plots_window
    assert plots is not None
    assert win._analysis_explorer.isAncestorOf(plots)
    assert plots.property("nanoscopeAnalysisExplorerPage") is True

    started = threading.Event()

    def slow_note():
        started.set()
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            check_cancelled()          # the ambient token _NoteWorker installs
            time.sleep(0.005)
        return "never reached"

    plots._pending_notes = {("ci", "slow"): slow_note}
    plots._settle_notes()
    thread = plots._note_thread
    assert started.wait(5.0), "the note worker never started"
    assert thread.isRunning()

    still_running = win._teardown()

    assert plots._note_worker is None, "the plots window still holds its worker"
    assert still_running == 0
    assert WT.detached_count() == 0
    assert not thread.isRunning()


def test_teardown_never_terminates_and_never_waits_unbounded(bare_win):
    """The standing rule, at the seam this fix added.

    Read off the SOURCE of the two methods the fix introduced, because a runtime check
    cannot prove the absence of a call that this run happened not to take. Neither may
    name ``terminate``, and neither may call ``wait`` at all — stopping a thread is
    ``worker_thread``'s job, and that is the one place the bounded-join rule lives.
    """
    from nanoscope.gui import main_window as MW

    src = pathlib.Path(MW.__file__).read_text()
    tree = ast.parse(src)
    bodies = {n.name: n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name in ("_teardown", "_stop_owned_background_work",
                             "_owned_background_holders")}
    assert set(bodies) == {"_teardown", "_stop_owned_background_work",
                           "_owned_background_holders"}, "the fix's methods are not all present"
    called = {n.func.attr for b in bodies.values() for n in ast.walk(b)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "terminate" not in called
    assert "wait" not in called
    # ...and the discovery is actually REACHED. Defining the two methods and leaving
    # ``_teardown`` on its old three-name list would satisfy everything above.
    reached = {n.func.attr for n in ast.walk(bodies["_teardown"])
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "_stop_owned_background_work" in reached, (
        "_teardown does not go through the discovery — it is enumerating again")
    assert "abort" not in reached, (
        "_teardown names a controller directly again; that is the defect, not the fix")


def test_no_gui_widget_turns_close_into_a_delete(bare_win):
    """THE PRECONDITION of stopping windows by ``close()``.

    Teardown asks every owned top-level widget to close, which for most of them (Qt's own
    hidden menu popups) is a no-op. It stops being a no-op the moment a widget carries
    ``Qt::WA_DeleteOnClose``: ``close()`` then DELETES it, and teardown would be
    destroying part of the window it is tearing down rather than stopping work.

    Nothing in ``nanoscope/gui`` sets that attribute today. This is that fact being
    checked rather than assumed — it is the assumption the broad discovery rests on.

    Read from the AST, not with ``in src``: the comment recording this precondition
    lives in ``main_window.py`` beside the discovery, and a substring search matches its
    own documentation.
    """
    from PySide6 import QtCore as _QtCore

    root = pathlib.Path(nanoscope.__file__).resolve().parent / "gui"
    offenders = []
    for p in sorted(root.glob("*.py")):
        tree = ast.parse(p.read_text())
        if any(isinstance(n, ast.Attribute) and n.attr == "WA_DeleteOnClose"
               for n in ast.walk(tree)):
            offenders.append(p.name)
    assert offenders == [], (
        f"{offenders} set WA_DeleteOnClose — teardown's close() would delete a window")

    # ...and the live tree agrees: no holder teardown would touch carries it.
    win = bare_win
    win._open_plots()
    flagged = [type(h).__name__ for h in win._owned_background_holders()
               if isinstance(h, QtWidgets.QWidget)
               and h.testAttribute(_QtCore.Qt.WA_DeleteOnClose)]
    assert flagged == [], flagged


def test_discovery_finds_the_live_holders_without_walking_into_anything_costly(bare_win):
    """The discovery is over the whole object tree; that must stay cheap and correct.

    Pins the two things a reviewer would want measured rather than asserted in prose:
    the live holders are all there, and the walk is not something a quit would feel.
    """
    win = bare_win
    win._open_plots()

    t0 = time.perf_counter()
    holders = win._owned_background_holders()
    cost_ms = (time.perf_counter() - t0) * 1000.0

    names = {type(h).__name__ for h in holders}
    assert {"LoadController", "AnalysisController", "BatchDetectController",
            "PlotsWindow"} <= names, sorted(names)
    assert win not in holders, "the window would ask itself to close"
    assert len(holders) == len(set(map(id, holders))), "a holder is listed twice"
    assert cost_ms < 250.0, f"discovery took {cost_ms:.1f} ms"
    print(f"\n[GREEN] {len(holders)} holders discovered in {cost_ms:.1f} ms "
          f"({len(win.findChildren(QtCore.QObject))} QObjects in the tree)")


def test_one_refusing_holder_cannot_block_the_quit(bare_win):
    """A holder that raises must not keep the others from being asked, or the settings
    from being saved. The drain — not any holder — decides how the quit ends."""
    win = bare_win

    class _Angry(QtCore.QObject):
        def abort(self):
            raise ValueError("no")

    win._angry = _Angry()
    win._polite = _UnlistedController()

    assert win._teardown() == 0
    assert win._polite.aborted, "a raising holder swallowed the ones after it"


# ================================================================================================
# 2. the status bar repeats the report's headline; it does not write one
# ================================================================================================

def _trace(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 100.0) + rng.normal(0.0, 1.5, N_SAMPLES)
    for i in range(N_EVENTS):
        s = 2_000 + i * _SPACING
        length = int(rng.integers(60, 600))
        x[s:s + length] -= float(rng.uniform(15.0, 45.0))
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: pathlib.Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


class _Cheap(SR.SubpopConfig):
    """The same pipeline over a small grid — as in ``test_subpop_gui._Cheap``."""

    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2, 3))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        super().__init__(**kw)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    arr = _trace()
    csv = tmp_path_factory.mktemp("green_wiring") / "many.csv"
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        csv, index=False)
    return csv, arr


@pytest.fixture(scope="module")
def reported(qapp, recording, _private_settings):
    """A WIRED MainWindow that has adopted a REAL report over its own REAL detection.

    Never a hand-written stub: the report is built by the same function the worker calls
    and adopted through the same slot the worker's ``finished`` signal reaches, so the
    string asserted below is the one the app puts on screen. Module-scoped because the
    detection plus the mixture is the expensive part and nothing here mutates it.
    """
    from nanoscope.analysis.modules import detection_signal

    csv, arr = recording
    w = MainWindow()
    doc_id = w._session.add_document(_doc(csv, arr))
    w._create_default_layers(w._session.document(doc_id))
    ch = next(c.index for c in w._session.document(doc_id).meta.channels
              if c.kind is ChannelKind.CURRENT and not c.is_dead)
    w._on_selection(doc_id, ch)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)

    done = threading.Event()
    w._analysis_controller.finished.connect(lambda *_: done.set())
    w._analysis_panel.trigger_detection()
    deadline = time.perf_counter() + 60.0
    while not done.is_set() and time.perf_counter() < deadline:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 20)
    assert done.is_set(), "the detection never finished; nothing below is about the wiring"
    qapp.processEvents()

    events = w._analysis_panel.current_detection().payload
    raw = np.asarray(w._session.calibrated(doc_id, ch).data)
    prov = dict(events.provenance or {})
    signal = detection_signal(raw, FS, prov)
    report = SR.run_subpopulations(
        signal, raw, FS, events.accepted, recording="synthetic",
        duration_s=N_SAMPLES / FS,
        config=_Cheap(polarity=SR.polarity_for_features(prov.get("polarity"))))
    assert report.quantification is not None, (
        "the fixture produced no mixture, so nothing below is about the wiring")

    w._subpop_signal_pending = signal
    w._subpop_pending = True
    w._on_subpop_finished(report)          # the real slot, the real fan-out
    qapp.processEvents()
    assert w._subpop_window is not None
    yield w, report
    w._teardown()


def _bar_after_the_run(reported, qapp) -> str:
    """Re-issue the run's own status message and return what the user then reads.

    THE FIXTURE IS MODULE-SCOPED AND ITS STATUS BAR IS A SHARED, MUTABLE SURFACE. Found
    in this fix's own verify pass: the colouring test below overwrites the run's message,
    so a test that merely read ``currentMessage()`` was asserting about whichever message
    happened to be up — it passed only because pytest ran the file top to bottom, and it
    would have gone green over a broken bar in any other order. And ``showMessage`` here
    carries a 12 s timeout, which a slow module can outlive. So every test about a
    message re-issues that message through the real slot first.
    """
    win, report = reported
    win._subpop_pending = True
    win._on_subpop_finished(report)
    qapp.processEvents()
    return win.statusBar().currentMessage()


def test_the_status_bar_after_a_run_is_the_reports_headline_verbatim(reported, qapp):
    """THE H4 SURFACE THIS PASS OWNS. The status bar is not a second author.

    ``headline()``'s WORDING is deliberately not asserted — the H4b pass owns that
    method and may be changing it. What is pinned is that this surface repeats whatever
    it says, so a future change to how the claim is qualified reaches the status bar
    without anyone remembering to come here.
    """
    win, report = reported
    shown = _bar_after_the_run(reported, qapp)
    print(f"\n[GREEN] status bar: {shown!r}")

    assert report.headline() in shown, (
        f"the status bar does not repeat the report's headline\n  bar: {shown!r}\n"
        f"  headline: {report.headline()!r}")
    assert shown.startswith("Subpopulations: "), shown
    # ...and NOTHING beyond the label. The old string's own words are gone.
    assert shown == f"Subpopulations: {report.headline()}"


def test_the_status_bar_no_longer_states_a_count_of_its_own(reported, qapp):
    """The exact sentence the -700 mV run printed, in its general form.

    ``"N subpopulation(s) at n_PCs=P"`` may appear in the bar ONLY as part of the
    headline. Asserting "the phrase is absent" would be wrong (the headline contains it,
    with the silhouette welded on); what must be absent is the surface's own sentence
    around it — the region/gate/caveat clauses main_window.py used to compose.
    """
    win, report = reported
    shown = _bar_after_the_run(reported, qapp)

    # THE assertion, and it is wording-independent: everything except the label came
    # from the report. Nothing here may depend on how ``headline()`` phrases itself —
    # H4b owns that method and is changing it in this same batch.
    assert shown.replace("Subpopulations: ", "", 1) == report.headline()

    # ...and the old sentence's own markers, stated explicitly so the failure names the
    # defect. Each is only held against the bar when the headline does not itself
    # produce it, so a future rewording of ``headline()`` cannot false-fail this.
    head = report.headline()
    for marker in ("region(s) passed the gate;", "caveat(s)."):
        if marker not in head:
            assert marker not in shown, (
                f"the hand-built sentence is still there: {marker!r}")


def test_colouring_the_trace_says_what_the_classes_are_worth(reported, qapp):
    """The SECOND status-bar sentence — Report window ▸ "Colour the trace".

    It used to count the classes itself out of the assignment dict
    (``len({v for v in ... if v >= 0})``) and print "coloured by N subpopulation(s)"
    with nothing beside it. Now it says what was coloured and repeats the report for
    what it is worth.
    """
    win, report = reported
    assert win._subpop_assignment, "the fixture produced no assignment to colour with"

    win._colour_trace_by_subpops()
    qapp.processEvents()
    shown = win.statusBar().currentMessage()
    print(f"\n[GREEN] colour-the-trace status bar: {shown!r}")

    assert shown.startswith("Trace spans coloured by subpopulation for "), shown
    assert report.headline() in shown, (
        f"the colouring message does not repeat the report's headline: {shown!r}")
    for k in range(1, 9):
        assert f"coloured by {k} subpopulation" not in shown, (
            "the message counts the classes for itself again")
    # The headline is quoted LAST and nothing is appended after it, so this surface
    # never has to assume how another module's sentence ends (a "." after it would
    # read as ".." the day headline() acquires its own).
    assert shown.endswith(report.headline()), shown


def test_the_headline_the_bar_repeats_is_the_one_the_report_window_shows(reported, qapp):
    """One claim, two surfaces, and they cannot drift: both repeat the same accessor.

    The report window is H4b's; this asserts only that the two agree, never what either
    of them says.
    """
    win, report = reported
    shown = _bar_after_the_run(reported, qapp)
    assert report.headline() in win._subpop_window.header()
    assert report.headline() in shown


def test_main_window_never_formats_a_class_count_of_its_own():
    """``headline()`` is documented as the ONLY place the count becomes a sentence.

    A source guard, because the two call sites above are not the only places somebody
    could write ``f"{q.fit.k} subpopulation(s)"`` tomorrow — and the run that produced
    the H4 finding had three surfaces doing it independently.

    READ FROM THE AST, NOT WITH ``in src``. The prose above and the comments at both
    call sites quote the sentence they replaced, deliberately, so a substring search over
    the file matches its own history and fails forever. Only executable nodes count: an
    ``x.fit.k`` attribute chain, and an f-string that would render a class count.
    """
    from nanoscope.gui import main_window as MW

    tree = ast.parse(pathlib.Path(MW.__file__).read_text())

    reads_k = [n for n in ast.walk(tree)
               if isinstance(n, ast.Attribute) and n.attr == "k"
               and isinstance(n.value, ast.Attribute) and n.value.attr == "fit"]
    assert reads_k == [], (
        f"main_window.py reads the raw class count again at line(s) "
        f"{[n.lineno for n in reads_k]}")

    # An f-string is a JoinedStr; a docstring or comment is not, so this sees only code.
    guilty = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)
              for part in n.values
              if isinstance(part, ast.Constant) and isinstance(part.value, str)
              and ("subpopulation(s)" in part.value or "n_PCs=" in part.value)]
    assert guilty == [], (
        f"main_window.py composes a sentence about the mixture again at line(s) {guilty}")


# ================================================================================================
# 3. the widened cross-panel guard still catches a real violation
# ================================================================================================

def _the_guard_itself():
    """``(_NON_PANEL_GUI_MODULES, _canonical_gui_imports)`` FROM THE GUARD'S OWN FILE.

    Loaded from ``test_measure_navigation.py`` by path rather than re-implemented here.
    A hand-copied canonicaliser is exactly the failure mode that file's own docstring
    warns about — a second implementation drifts from the first and then passes while the
    guard it stands in for has a hole (``from . import x`` was such a hole). Loading by
    path also makes this independent of how the runner happens to name the test package.
    """
    import importlib.util

    path = pathlib.Path(__file__).with_name("test_measure_navigation.py")
    spec = importlib.util.spec_from_file_location("green_probe_measure_navigation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._NON_PANEL_GUI_MODULES, mod._canonical_gui_imports

def test_widening_the_non_panel_allowlist_did_not_disarm_the_cross_panel_guard():
    """``worker_thread`` joined ``_NON_PANEL_GUI_MODULES``. Prove what still fails.

    A guard nobody has seen fail is a guard nobody has tested, and a guard that was just
    WIDENED is the one most worth re-proving. This drives the guard's own canonicaliser —
    the same function the real test calls, never a hand-copied second implementation —
    over injected source, so it cannot pass because the copy drifted.

    Still caught after the widening:
      * a panel imported by any of the five spellings, including ``from . import x``;
      * a NON-panel module that is not on the allowlist;
      * a module on the allowlist that starts reaching a panel itself (the two-hop
        route), which is now checked by ``test_the_non_panel_allowlist_holds_no_panel``.
    """
    _NON_PANEL_GUI_MODULES, _canonical_gui_imports = _the_guard_itself()

    assert "worker_thread" in _NON_PANEL_GUI_MODULES

    # (a) every spelling of a panel-to-panel import is still seen...
    for src in ("from .file_dock import FileDock",
                "from ..gui.file_dock import FileDock",
                "from nanoscope.gui.file_dock import FileDock",
                "import nanoscope.gui.file_dock",
                "from . import file_dock",
                "from .psd_panel import PSDWindow",
                "from . import measure_panel"):
        reached = _canonical_gui_imports(src)
        assert reached - _NON_PANEL_GUI_MODULES, f"{src!r} slipped past the widened set"
        assert [m for m in reached if m.endswith("_panel") or m == "file_dock"], src

    # (b) ...and the widening admitted exactly ONE new name, not a pattern.
    assert _canonical_gui_imports("from .worker_thread import start_worker_thread") == {
        "worker_thread"}
    assert not (_canonical_gui_imports("from .worker_thread import start_worker_thread")
                - _NON_PANEL_GUI_MODULES)
    for lookalike in ("from .workers import LoadController",
                      "from .analysis_worker import AnalysisController",
                      "from .worker_thread_helpers import x"):
        assert _canonical_gui_imports(lookalike) - _NON_PANEL_GUI_MODULES, lookalike


def test_the_allowlist_guard_catches_a_helper_that_reaches_a_panel():
    """The two-hop route, injected. ``worker_thread`` was admitted on the claim that it
    reaches nothing in ``gui``; this is that claim being checked rather than trusted."""
    _, _canonical_gui_imports = _the_guard_itself()

    innocent = pathlib.Path(WT.__file__).read_text()
    assert _canonical_gui_imports(innocent) == set(), (
        "worker_thread reaches a gui sibling — it does not belong on the allowlist")
    guilty = innocent + "\nfrom .file_dock import FileDock\n"
    assert _canonical_gui_imports(guilty) == {"file_dock"}, (
        "the allowlist guard would not have seen a helper that reaches a panel")
