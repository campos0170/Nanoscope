"""GREEN2 — a clean quit reported over a live thread, and a suite that wrote the owner's
real preferences.

Two defects, four hunks, and every test below is written to FAIL against an out-of-tree
revert of them:

  A. ``MainWindow._owned_background_holders`` admitted a widget only if ``isWindow()``,
     and applied that filter to the ``abort()`` protocol as well as to ``close()``. Every
     panel in this tree is a non-window child widget, so a panel holding a worker was not
     a holder at all. Measured in the wired app before the fix::

         isWindow(): False   has abort(): True
         discovered as a holder: False    (21 holders found)
         teardown() returned: 0    detached_count(): 0
         panel.abort() was called: False
         its QThread STILL RUNNING: True
         QThread: Destroyed while thread '' is still running

  B. ``MainWindow._teardown`` returned ``drain_detached()``, which can only see threads
     that reached ``retire_worker_thread``. A thread nobody ever asked to stop is absent
     from ``_DETACHED`` — and from the run gate's ``_LIVE`` too, when the job was started
     with ``JOB_NONE`` — so the quit reported itself clean over a running OS thread.

  C. ``worker_thread`` now registers EVERY started thread, which is the only source of
     truth that is a property of the thread rather than of somebody's intent.

  D. ``tests/conftest.py`` redirects the QSettings STORE (format + path), not the app's
     NAME, and ``tests/test_integration.py``'s shadowing session ``qapp`` fixture is gone.
     Measured before: a suite run changed
     ``~/Library/Preferences/com.nanoscope-local.NanoScope.plist``
     (md5 f58fe4d1… -> faf8fd28…) and left eleven pytest temporary-directory paths in the
     shipped app's Recent Projects.

Run:  QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -m pytest \
          tests/test_rk_teardown_and_prefs_green2.py -v
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import pathlib
import sys
import threading
import time

import pytest
from PySide6 import QtCore, QtWidgets

import conftest as CT
from nanoscope.gui import worker_thread as WT
from nanoscope.gui.main_window import MainWindow

_TREE = pathlib.Path(nanoscope.__file__).resolve().parent.parent


def _drain(ms: int) -> int:
    """Cleanup that works in BOTH trees.

    ``drain_worker_threads`` is one of the hunks under test, so calling it from a fixture
    would make an out-of-tree revert run fail in TEARDOWN — an ``AttributeError`` that
    tells you nothing about whether the test's own assertion held. The tests that are
    ABOUT the new function name it directly; everything else cleans up through this."""
    return getattr(WT, "drain_worker_threads", WT.drain_detached)(ms)


def _which_tree() -> str:
    return (f"nanoscope <- {nanoscope.__file__}\n"
            f"main_window <- {MainWindow.__module__} "
            f"{pathlib.Path(__import__('nanoscope.gui.main_window', fromlist=['x']).__file__)}")


def test_which_tree_this_ran_against(capsys):
    """Passes in BOTH trees. A revert run from inside the repo silently tests the LIVE
    tree — ``PYTHONPATH`` does not beat cwd when cwd contains the package — so every run
    of this file states where it imported from."""
    from nanoscope.gui import main_window as MW

    print("\n[GREEN2] " + _which_tree())
    assert pathlib.Path(MW.__file__).resolve().parent.parent.parent == _TREE
    assert pathlib.Path(CT.__file__).resolve().parent.parent == _TREE, (
        f"conftest.py came from a different tree than nanoscope: {CT.__file__}")


# ================================================================================================
# Workers and holders. Real QThreads, started through the real seam.
# ================================================================================================

class _SpinWorker(QtCore.QObject):
    """Cooperative: stops when told, like every job in this tree."""

    finished = QtCore.Signal()

    def __init__(self, hold_s: float = 30.0) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._hold = hold_s

    @QtCore.Slot()
    def run(self) -> None:
        self.entered.set()
        self.release.wait(self._hold)
        self.finished.emit()


class _PanelWithAbort(QtWidgets.QWidget):
    """EXACTLY what the three original controllers are — except it is an inner widget,
    which is what every panel in this tree is."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.aborted = False
        self.worker: _SpinWorker | None = None
        self.thread: QtCore.QThread | None = None

    def start(self) -> None:
        self.worker = _SpinWorker()
        self.thread = WT.start_worker_thread(
            self.worker, domain=WT.JOB_NONE, label="rk-panel", owner=self)

    def abort(self, wait_ms: int = WT.DEFAULT_JOIN_MS) -> None:
        self.aborted = True
        worker, thread = self.worker, self.thread
        self.worker = None
        if worker is not None:
            worker.release.set()          # the cooperative stop every real job has
        WT.retire_worker_thread(worker, thread, wait_ms=2000)


class _MutePanel(QtWidgets.QWidget):
    """Owns a worker and speaks NEITHER stop protocol — the case the return value is for."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.worker: _SpinWorker | None = None
        self.thread: QtCore.QThread | None = None

    def start(self) -> None:
        self.worker = _SpinWorker()
        self.thread = WT.start_worker_thread(
            self.worker, domain=WT.JOB_NONE, label="rk-mute", owner=self)


@pytest.fixture
def win(qapp, qtbot):
    """A REAL, fully wired MainWindow — not a widget built in isolation."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qapp.processEvents()
    yield w
    _drain(5000)


# ================================================================================================
# A — a panel that owns a worker is stopped BECAUSE IT OWNS ONE
# ================================================================================================

def test_a_panel_that_owns_a_worker_is_stopped_although_it_is_not_a_window(win, qapp, capsys):
    """THE INVARIANT of hunk A, in the wired app, with the exact facts the defect printed.

    Fails against the revert: ``discovered as a holder: False``, ``abort() was called:
    False``, and the QThread is still running after ``_teardown()`` returned 0.
    """
    panel = _PanelWithAbort(win)
    win._rk_panel = panel                      # an attribute AND in the Qt tree
    panel.start()
    qapp.processEvents()
    thread = panel.thread
    assert panel.worker is not None and panel.worker.entered.wait(5.0), "worker never started"
    assert thread.isRunning(), "nothing to stop — the harness is not exercising the defect"

    assert panel.isWindow() is False, "the harness stopped being about an INNER widget"
    holders = win._owned_background_holders()
    discovered = any(h is panel for h in holders)

    rc = win._teardown()

    print(f"\n[GREEN2-A] isWindow(): {panel.isWindow()}   has abort(): True\n"
          f"           discovered as a holder: {discovered}    ({len(holders)} holders)\n"
          f"           teardown() returned: {rc}    detached_count(): {WT.detached_count()}\n"
          f"           panel.abort() was called: {panel.aborted}\n"
          f"           its QThread STILL RUNNING: {thread.isRunning()}")

    assert discovered, (
        "a non-window widget that owns a worker was not discovered as a holder — the "
        "isWindow() filter is back on admission instead of on close()")
    assert panel.aborted, "teardown never asked the panel to stop"
    assert not thread.isRunning(), "the panel's QThread survived the quit"
    assert rc == 0, f"a cooperative worker stops in ms; teardown reported {rc}"
    assert WT.detached_count() == 0, "a live QThread was left parked at quit"


def test_the_window_filter_now_guards_close_and_nothing_else(win):
    """The other half of hunk A: the fix must not have over-corrected into calling
    ``close()`` on inner widgets — that would be a state change, not a stop.

    Fails against the revert with ``AttributeError: _stop_protocols``.
    """
    inner_with_abort = _PanelWithAbort(win)
    inner_plain = QtWidgets.QWidget(win)
    a_window = QtWidgets.QWidget(win)
    a_window.setWindowFlag(QtCore.Qt.Window, True)

    assert win._stop_protocols(inner_with_abort) == ["abort"], (
        "an inner widget must be aborted and must NOT be closed")
    assert win._stop_protocols(inner_plain) == [], (
        "a plain inner widget speaks no stop protocol and is not a holder")
    assert a_window.isWindow()
    assert win._stop_protocols(a_window) == ["close"], (
        "a window that owns no abort() is still stopped through close()")

    holders = win._owned_background_holders()
    assert any(h is inner_with_abort for h in holders)
    assert all(h is not inner_plain for h in holders), (
        "discovery widened to every child widget; that is 1600 close() calls, not a fix")


def test_the_holders_the_old_list_named_are_all_still_stopped(win, monkeypatch):
    """Nothing chartered moved: the three controllers and the Plots window still stop.

    (Passes in both trees — it is the regression half, not the revert half.)
    """
    stopped: list[str] = []
    for name in ("_controller", "_analysis_controller", "_batch_controller"):
        monkeypatch.setattr(getattr(win, name), "abort",
                            lambda *_a, _n=name, **_k: stopped.append(_n))
    win._open_plots()
    assert win._plots_window is not None

    win._teardown()

    assert sorted(stopped) == ["_analysis_controller", "_batch_controller", "_controller"]
    assert not win._plots_window.isVisible(), "the Plots window was not closed on quit"


# ================================================================================================
# B — the return value is measured from the threads
# ================================================================================================

def test_teardown_will_not_report_a_clean_quit_over_a_thread_nobody_stopped(win, qapp, capsys):
    """Hunk B. A holder that speaks no stop protocol, with a ``JOB_NONE`` job: absent from
    ``_DETACHED`` (nothing retired it) and absent from ``_LIVE`` (JOB_NONE is ungated), so
    both of the old candidate counts say zero while an OS thread runs.

    Fails against the revert: ``_teardown()`` returns 0.
    """
    panel = _MutePanel(win)
    win._rk_mute = panel
    panel.start()
    assert panel.worker.entered.wait(5.0), "worker never started"
    try:
        # The two candidate counts the old return value could have been read off, BEFORE
        # anything new is named — so an out-of-tree revert fails on ``rc``, the fact this
        # test is about, and not on an AttributeError in its own preamble.
        assert WT.drain_detached(0) == 0, "the harness detached it; that is not this case"
        assert len(WT.live_jobs()) == 0, "JOB_NONE is registered at the gate; harness is wrong"

        t0 = time.perf_counter()
        rc = win._teardown()
        cost_s = time.perf_counter() - t0

        print(f"\n[GREEN2-B] teardown() returned: {rc}   detached_count(): "
              f"{WT.detached_count()}   ({cost_s * 1000:.0f} ms)\n"
              f"           thread still running: {panel.thread.isRunning()}")
        assert rc == 1, (
            f"teardown reported {rc} with a live worker thread — the count is still being "
            f"read off _DETACHED, which nothing ever put this thread into")
        assert WT.detached_count() == 1, (
            "the survivor was not parked; gui.app.main's final drain_detached() would see "
            "nothing and interpreter shutdown would delete a running QThread")
    finally:
        panel.worker.release.set()
        assert _drain(5000) == 0
    assert WT.detached_count() == 0, "the parked thread was not released once it finished"


def test_an_idle_wired_window_still_reports_a_clean_quit_and_costs_nothing(win, capsys):
    """The other side of the same value, and the interactive cost of hunk B."""
    assert WT.detached_count() == 0
    t0 = time.perf_counter()
    rc = win._teardown()
    cost_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n[GREEN2-B] idle _teardown(): {rc} in {cost_ms:.1f} ms")
    assert rc == 0
    assert cost_ms < 250.0, f"teardown of an idle window took {cost_ms:.1f} ms"


# ================================================================================================
# C — the thread registry, at the seam
# ================================================================================================

def test_a_JOB_NONE_worker_is_invisible_to_both_old_registries_and_visible_to_the_new_one(qapp):
    """Hunk C, at ``worker_thread``'s own level.

    Fails against the revert with ``AttributeError: unstopped_worker_threads``.
    """
    worker = _SpinWorker()
    thread = WT.start_worker_thread(worker, domain=WT.JOB_NONE, label="rk-registry",
                                    owner=None)
    try:
        assert worker.entered.wait(5.0)
        assert WT.blocking_job(WT.JOB_ANALYSIS) is None, "JOB_NONE must not gate anything"
        assert all(j.thread is not thread for j in WT.live_jobs())
        assert WT.detached_count() == 0
        assert any(t is thread for t in WT.unstopped_worker_threads()), (
            "a running thread is invisible to every registry — that is the lie")
    finally:
        worker.release.set()
        assert _drain(5000) == 0
    assert all(t is not thread for t in WT.unstopped_worker_threads()), (
        "a finished thread is still registered; the registry leaks")


def test_the_registry_holds_threads_weakly(qapp):
    """The registry must not be the reason a worker (and through it a controller) lives
    forever — the leak ``_park_orphan`` documents from the other direction."""
    import gc
    import weakref

    worker = _SpinWorker(hold_s=0.05)
    thread = WT.start_worker_thread(worker, domain=WT.JOB_NONE, label="rk-weak", owner=None)
    ref = weakref.ref(thread)
    assert WT.drain_worker_threads(5000) == 0   # named directly: this test is about it
    del thread, worker
    gc.collect()
    assert ref() is None, "worker_thread._STARTED is holding a strong reference"


def test_no_terminate_and_no_unbounded_wait_survived_the_change():
    """The standing rule, read off the SOURCE of both files I touched."""
    from nanoscope.gui import main_window as MW

    for mod in (WT, MW):
        src = pathlib.Path(mod.__file__).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "terminate", f"{mod.__name__} terminates a thread"
                if node.func.attr == "wait":
                    assert mod is WT, f"{mod.__name__} joins a thread; only worker_thread may"
                    assert node.args, "an UNBOUNDED wait() — every join must be bounded"


# ================================================================================================
# D — the suite may not write the owner's real preferences
# ================================================================================================

def test_the_suite_keys_its_settings_somewhere_private(qapp, win):
    """The redirect, asserted on the store a REAL MainWindow actually writes.

    The location is checked BEFORE anything is written, so that against the revert this
    test fails without having written the owner's file to prove it.
    """
    where = pathlib.Path(QtCore.QSettings().fileName())
    real = CT.real_settings_store()
    print(f"\n[GREEN2-D] QSettings() -> {where}\n           real store -> {real}")

    assert CT._SETTINGS_ROOT in where.parents, (
        f"the suite's QSettings is not inside the private root {CT._SETTINGS_ROOT}")
    assert where != real
    assert real.parent not in where.parents, (
        f"the suite writes into the real preferences directory {real.parent}")

    before = CT.real_store_signature()
    win._save_settings()                       # the real quit path, on a real window
    assert where.exists(), "the private store was not actually written"
    assert CT.real_store_signature() == before, "a MainWindow save reached the real store"


def test_renaming_the_application_cannot_escape_the_private_store(qapp):
    """THE INVARIANT of hunk D. ``create_app`` sets the shipped names — that is its job,
    and it is exactly what the shadowing fixture in ``test_integration.py`` did. Isolation
    that a rename can step around is not isolation.

    Nothing is WRITTEN here: the assertion is on ``fileName()``, so against the revert
    this fails instead of rewriting the owner's plist to make its point.
    """
    from nanoscope.gui.app import create_app

    org, name, domain = (qapp.organizationName(), qapp.applicationName(),
                         qapp.organizationDomain())
    try:
        create_app([])
        assert qapp.applicationName() == "NanoScope", "create_app stopped naming the app"
        where = pathlib.Path(QtCore.QSettings().fileName())
        print(f"\n[GREEN2-D] after create_app([]) -> {where}")
        assert CT._SETTINGS_ROOT in where.parents, (
            f"renaming the app escaped the private store: {where}")
        assert where != CT.real_settings_store()
    finally:
        qapp.setOrganizationName(org)
        qapp.setApplicationName(name)
        qapp.setOrganizationDomain(domain)


#: The calls that RENAME the application. Every one of them re-points a name-based
#: QSettings namespace, which is what made the shadowing fixture dangerous.
_RENAMERS = frozenset({"create_app", "setApplicationName", "setOrganizationName",
                       "setOrganizationDomain"})


def test_no_test_module_shadows_the_shared_qapp_fixture_with_a_renaming_one(capsys):
    """Hunk D's other half. A fixture named ``qapp`` in a test MODULE shadows conftest's
    for that module; the one in ``test_integration.py`` was ``return create_app([])``,
    which is precisely a rename back to the shipped app's own name. Fails against the
    revert, naming that file and line.

    Shadows that do NOT rename are reported, not failed: ``test_zoom_navigation.py`` has
    a function-scoped ``qapp`` that only returns the existing instance, and with the store
    redirected by path and format rather than by name it cannot reach the owner's file.
    """
    renaming: list[str] = []
    benign: list[str] = []
    for path in sorted((_TREE / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "qapp":
                continue
            if not any("fixture" in ast.dump(d) for d in node.decorator_list):
                continue
            where = f"{path.relative_to(_TREE)}:{node.lineno}"
            called = {n.func.attr if isinstance(n.func, ast.Attribute) else
                      getattr(n.func, "id", "")
                      for n in ast.walk(node) if isinstance(n, ast.Call)}
            (renaming if called & _RENAMERS else benign).append(where)

    print(f"\n[GREEN2-D] shadowing qapp fixtures that rename the app: {renaming}\n"
          f"           shadowing qapp fixtures that do not: {benign}")
    assert renaming == [], (
        f"these modules define their own ``qapp`` and rename the application inside it, "
        f"which is how the owner's real preferences came to be written: {renaming}")


def test_the_real_store_guard_actually_fails_when_the_store_moves(tmp_path):
    """The guard is not ``None == None``: hand it a stand-in file and watch it fire.

    Uses a temporary file, never the owner's — a guard proved by damaging the thing it
    guards is not a proof anybody wants.
    """
    stand_in = tmp_path / "com.example.StandIn.plist"
    stand_in.write_bytes(b"a")
    sig = CT.real_store_signature(stand_in)

    assert CT.check_real_store_untouched(sig, stand_in) == sig, "the guard cried wolf"

    time.sleep(0.01)
    stand_in.write_bytes(b"ab")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        CT.check_real_store_untouched(sig, stand_in)
    assert "wrote the owner's real settings store" in str(excinfo.value)
    assert stand_in.name in str(excinfo.value)


def test_the_guard_is_autouse_so_every_test_carries_it(request):
    """It has to reach tests nobody remembered to decorate — that is the whole point."""
    assert "_real_settings_store_is_never_written" in request.fixturenames, (
        "the real-store guard is no longer autouse")


# ================================================================================================
# A + B together, in a WHOLE PROCESS: what the user's machine actually does on ⌘Q
# ================================================================================================

_EXIT_PROBE = '''
import os, sys, threading
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import nanoscope
print("nanoscope <-", nanoscope.__file__)
from PySide6 import QtCore, QtWidgets
QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                         sys.argv[1])
from nanoscope.gui import app as APP, worker_thread as wt

class _Stuck(QtCore.QObject):
    finished = QtCore.Signal()
    def __init__(self):
        super().__init__(); self.entered = threading.Event()
    @QtCore.Slot()
    def run(self):
        self.entered.set()
        threading.Event().wait(20.0)          # ignores cancel entirely
        self.finished.emit()

class _Mute(QtWidgets.QWidget):
    pass

def fake_exec(self):
    w = [x for x in QtWidgets.QApplication.topLevelWidgets()
         if type(x).__name__ == "MainWindow"][0]
    p = _Mute(w); w._probe_mute = p
    p.worker = _Stuck()
    p.thread = wt.start_worker_thread(p.worker, domain=wt.JOB_NONE, label="probe", owner=p)
    assert p.worker.entered.wait(5.0)
    print("worker running:", p.thread.isRunning())
    w.close()                                  # the real quit path: closeEvent -> _teardown
    print("after close: detached_count() =", wt.detached_count())
    return 7                                   # a distinctive rc

QtWidgets.QApplication.exec = fake_exec
rc = APP.main([])
sys.stdout.flush()
sys.exit(rc)
'''


def test_a_quit_over_an_unstoppable_worker_leaves_the_process_at_its_own_exit_code(tmp_path):
    """THE THING THE USER'S MACHINE DOES. A whole process, through ``gui.app.main``'s real
    quit path, with a worker that ignores cancel and a holder that speaks no stop protocol.

    ``app.main`` decides between an orderly return and
    ``exit_without_destroying_live_threads`` by reading ``drain_detached()`` — so a
    survivor that ``_teardown`` never parked leaves that guard with nothing to see.
    Measured, same script, both trees::

        live tree  : exit 7  (the app's own rc), detached_count() after close = 1
        reverted   : exit 134 SIGABRT, detached_count() = 0,
                     "QThread: Destroyed while thread '' is still running"

    ``gui/app.py`` is not mine to edit and did not need to be: it was already right, and
    was being handed a false zero.
    """
    import subprocess
    script = tmp_path / "exit_probe.py"
    script.write_text(_EXIT_PROBE)
    # sys.path[0] for ``python /tmp/.../exit_probe.py`` is the SCRIPT's directory, not the
    # cwd — so without this the subprocess imports whichever ``nanoscope`` is installed in
    # site-packages, which on this machine is the live tree. That is the trap this project
    # keeps naming ("PYTHONPATH does not beat cwd"), arriving from its other side; the
    # ``nanoscope <-`` line asserted below is what caught it.
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(_TREE))
    proc = subprocess.run([sys.executable, str(script), str(tmp_path / "qs")],
                          cwd=str(_TREE), capture_output=True, text=True, timeout=180,
                          env=env)
    print(f"\n[GREEN2-AB] exit={proc.returncode}\n{proc.stdout.strip()}")

    assert str(_TREE) in proc.stdout, f"the subprocess imported another tree:\n{proc.stdout}"
    assert "worker running: True" in proc.stdout, "the probe never got a live worker"
    assert "detached_count() = 1" in proc.stdout, (
        "_teardown did not park the thread it could not stop; app.main's final "
        "drain_detached() has nothing to see")
    assert proc.returncode == 7, (
        f"the process left with {proc.returncode} (134 = SIGABRT / qFatal), not the app's "
        f"own exit code:\n{proc.stderr[-2000:]}")
    assert "Destroyed while thread" not in proc.stderr, proc.stderr[-2000:]
