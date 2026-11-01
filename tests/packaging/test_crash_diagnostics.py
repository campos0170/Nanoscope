"""Q11 — the frozen app's log must tell a crash from a quit, and say which build crashed where.

The recoverability investigation could not reconstruct a reported crash from
``~/Library/Logs/NanoScope/nanoscope.log``: its start banner named no build, nothing was written
at exit, faulthandler was never enabled (a segfault in Qt left the log simply stopping), and no
hook dated an exception that escaped a Qt slot. ``packaging/nanoscope_app.py`` now writes the
build ID and pid into the banner, a "clean exit rc=N" line on every controlled way out (so a
banner with no exit line after it marks a crash), enables faulthandler on the log, and installs
sys.excepthook / threading.excepthook.

Every test drives the REAL launcher in a subprocess with HOME pointed at a scratch folder, like
test_s4_bundle_log_rotation.py. Only ``nanoscope.gui.app`` is replaced, by a stand-in whose
``main`` plays one way of ending; the real one would build a MainWindow, and macOS QSettings
ignore HOME. The segfault test leaves one python3.x crash report in DiagnosticReports, as any
crashing subprocess does: faulthandler re-raises the signal so the OS still records the crash.
"""
from __future__ import annotations

import os
import plistlib
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
PACKAGING = TREE / "packaging"

_CHILD = r'''
import os, sys, types
sys.path.insert(0, {pkg!r})
import nanoscope_app

SCENARIO = os.environ["NS_SCENARIO"]
RC = int(os.environ.get("NS_RC", "0"))


def _create_app(argv):
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv[:1])


def _crash_inside_the_app():
    import faulthandler
    faulthandler._sigsegv()


def _slot_that_raises():
    raise ValueError("BOOM-IN-A-SLOT")


def _thread_that_raises():
    raise RuntimeError("BOOM-IN-A-THREAD")


def _app_main(argv):
    import nanoscope
    from PySide6 import QtCore, QtWidgets
    print("nanoscope <- " + os.path.dirname(nanoscope.__file__))
    print("CHILD-PID=%d" % os.getpid())
    app = QtWidgets.QApplication.instance()
    if SCENARIO == "return":
        return RC
    if SCENARIO == "raise-systemexit":
        raise SystemExit(RC)
    if SCENARIO == "raise-systemexit-none":
        raise SystemExit(None)
    if SCENARIO == "raise-systemexit-text":  # Python prints the text and exits 1
        raise SystemExit("leaving with a message")
    if SCENARIO == "hard-exit":  # the quit over a worker that will not stop
        from nanoscope.gui.worker_thread import exit_without_destroying_live_threads
        exit_without_destroying_live_threads(RC)
    if SCENARIO == "sys-exit-in-slot":  # PySide6 exits from inside exec; main never returns
        QtCore.QTimer.singleShot(0, lambda: sys.exit(RC))
        return app.exec()
    if SCENARIO == "fatal":
        raise RuntimeError("BOOM-AT-THE-TOP")
    if SCENARIO == "segfault":
        _crash_inside_the_app()
    if SCENARIO == "slot-raises":
        QtCore.QTimer.singleShot(0, _slot_that_raises)
        QtCore.QTimer.singleShot(100, app.quit)
        return app.exec()
    if SCENARIO in ("thread-raises", "thread-exits"):
        import threading
        target = _thread_that_raises if SCENARIO == "thread-raises" else (lambda: sys.exit(9))
        worker = threading.Thread(target=target, name="probe-worker")
        worker.start()
        worker.join()
        return RC
    raise AssertionError(SCENARIO)


fake = types.ModuleType("nanoscope.gui.app")
fake.create_app = _create_app
fake.main = _app_main
sys.modules["nanoscope.gui.app"] = fake

# _report_fatal would block on a modal QMessageBox offscreen; keep its log half only.
nanoscope_app._report_fatal = lambda exc: sys.stderr.write("REPORTED %r\n" % (exc,))

if os.environ.get("NS_NO_EXIT_HOOK"):  # a bundle that lacks nanoscope/clean_exit.py
    sys.modules["nanoscope.clean_exit"] = None

if os.environ.get("NS_PREVIOUS_HOOK"):
    def _previous(exc_type, exc, tb):
        sys.stderr.write("PREVIOUS-HOOK-SAW-%s\n" % exc_type.__name__)
    sys.excepthook = _previous

sys.exit(nanoscope_app.main())
'''

_BANNER = re.compile(r"^===== NanoScope start \S+ pid=(\d+) build=(\S+) =====$", re.M)
_EXIT = re.compile(r"^===== NanoScope clean exit rc=(\S+) \S+ pid=(\d+) =====$", re.M)


def _log_path(home: Path) -> Path:
    """Where the launcher puts the log on this platform (mirrors nanoscope_app._log_dir)."""
    if sys.platform == "darwin":
        return home / "Library" / "Logs" / "NanoScope" / "nanoscope.log"
    if sys.platform == "win32":
        return home / "AppData" / "Local" / "NanoScope" / "Logs" / "nanoscope.log"
    return home / ".local" / "state" / "NanoScope" / "nanoscope.log"


def _env(home: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    env["QT_QPA_PLATFORM"] = "offscreen"
    # sys.path[0] of a script is ITS directory, so without this the child would import whichever
    # nanoscope site-packages points at rather than this tree's (asserted below).
    env["PYTHONPATH"] = str(TREE)
    env["NANOSCOPE_RECOVERY_ROOT"] = str(home / "Recovery")  # no window is built; belt and braces
    for stale in ("NANOSCOPE_MPLCONFIGDIR", "PYTHONFAULTHANDLER"):
        env.pop(stale, None)
    env.update(extra)
    return env


def _run(tmp_path: Path, scenario: str, rc: int = 0, *, no_log: bool = False, **extra: str):
    """Start the launcher once; return (process, log text).

    `no_log` puts a FILE where the log directory belongs, so the stdio redirect fails the way an
    unwritable ~/Library/Logs would and the child keeps the stdout/stderr pipes it was given.
    """
    script = tmp_path / "start_nanoscope.py"
    script.write_text(_CHILD.format(pkg=str(PACKAGING)))
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    log = _log_path(home)
    if no_log:
        log.parent.parent.mkdir(parents=True, exist_ok=True)
        log.parent.write_text("not a directory")
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), timeout=180,
                          env=_env(home, NS_SCENARIO=scenario, NS_RC=str(rc), **extra),
                          capture_output=True, text=True)
    text = proc.stdout + proc.stderr if no_log else log.read_text(errors="replace")
    assert f"nanoscope <- {TREE / 'nanoscope'}" in text, (
        f"the child did not import this tree's nanoscope:\n{text}\n{proc.stderr[-2000:]}")
    return proc, text


def test_the_start_banner_names_the_build_and_the_pid(tmp_path):
    """An unfrozen run says so; the pid is the launcher's own, so its exit line can be paired."""
    proc, text = _run(tmp_path, "return")

    assert proc.returncode == 0, proc.stderr[-2000:]
    banners = _BANNER.findall(text)
    assert len(banners) == 1, text
    pid, build = banners[0]
    assert build == "source", text
    assert f"CHILD-PID={pid}" in text, text


@pytest.mark.parametrize(("scenario", "rc", "logged"), [
    ("return", 0, "0"),
    ("return", 3, "3"),
    ("raise-systemexit", 6, "6"),
    ("raise-systemexit-none", 0, "0"),
    ("raise-systemexit-text", 1, "1"),
    ("hard-exit", 5, "5"),               # os._exit: no atexit, so worker_thread writes it
    ("sys-exit-in-slot", 4, "unknown"),  # only atexit sees this one, and it cannot see the code
    ("fatal", 1, "1"),                   # reported, then ended on purpose: not a crash
])
def test_every_controlled_way_out_ends_with_one_clean_exit_line(tmp_path, scenario, rc, logged):
    proc, text = _run(tmp_path, scenario, rc)

    assert proc.returncode == rc, f"{scenario}: rc {proc.returncode}\n{text}\n{proc.stderr[-2000:]}"
    exits = _EXIT.findall(text)
    assert len(exits) == 1, f"{scenario}: {exits}\n{text}"
    assert exits[0] == (logged, _BANNER.findall(text)[0][0]), f"{scenario}:\n{text}"
    assert text.index("clean exit") > text.index("CHILD-PID="), f"{scenario}:\n{text}"


def test_a_missing_exit_hook_never_stops_the_app_from_starting(tmp_path):
    """Diagnostics must never be what stops the app: without the hook module only the os._exit
    quit would lose its line."""
    proc, text = _run(tmp_path, "return", 3, NS_NO_EXIT_HOOK="1")

    assert proc.returncode == 3, f"rc {proc.returncode}\n{text}\n{proc.stderr[-2000:]}"
    assert "REPORTED" not in text, text
    assert [rc for rc, _pid in _EXIT.findall(text)] == ["3"], text


def test_a_segfault_leaves_a_python_traceback_in_the_log_and_no_exit_line(tmp_path):
    """The point of the whole item: a native crash used to leave the log simply stopping."""
    proc, text = _run(tmp_path, "segfault")

    assert proc.returncode == -signal.SIGSEGV, f"rc {proc.returncode}\n{text}"
    assert "Fatal Python error: Segmentation fault" in text, text
    # faulthandler names the Python frames that were running, innermost first.
    assert re.search(r'File ".*start_nanoscope\.py", line \d+ in _crash_inside_the_app', text), text
    assert re.search(r'File ".*start_nanoscope\.py", line \d+ in _app_main', text), text
    assert not _EXIT.findall(text), f"a crash must not look like a quit:\n{text}"


def test_an_exception_in_a_qt_slot_is_dated_and_named_and_logged_once(tmp_path):
    """PySide6 hands it to sys.excepthook and carries on, so the app still quits cleanly."""
    proc, text = _run(tmp_path, "slot-raises")

    assert proc.returncode == 0, proc.stderr[-2000:]
    pid = _BANNER.findall(text)[0][0]
    assert re.search(rf"^----- unhandled ValueError in thread 'MainThread' \S+ pid={pid} -----$",
                     text, re.M), text
    assert text.count("ValueError: BOOM-IN-A-SLOT") == 1, f"logged twice or not at all:\n{text}"
    assert _EXIT.findall(text) == [("0", pid)], text


def test_a_hook_installed_before_the_launchers_still_runs_after_it(tmp_path):
    proc, text = _run(tmp_path, "slot-raises", NS_PREVIOUS_HOOK="1")

    assert proc.returncode == 0, proc.stderr[-2000:]
    logged = text.find("----- unhandled ValueError")
    assert logged >= 0, text
    assert text.find("PREVIOUS-HOOK-SAW-ValueError") > logged, text


def test_an_exception_in_a_python_thread_is_dated_and_named_and_logged_once(tmp_path):
    proc, text = _run(tmp_path, "thread-raises")

    assert proc.returncode == 0, proc.stderr[-2000:]
    pid = _BANNER.findall(text)[0][0]
    assert re.search(rf"^----- unhandled RuntimeError in thread 'probe-worker' \S+ pid={pid} -----$",
                     text, re.M), text
    assert text.count("RuntimeError: BOOM-IN-A-THREAD") == 1, text
    assert "Exception in thread" not in text, f"the default hook printed it again:\n{text}"


def test_a_thread_that_calls_sys_exit_is_not_an_error(tmp_path):
    """The default threading hook is silent about SystemExit: a thread may end that way."""
    proc, text = _run(tmp_path, "thread-exits")

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "unhandled" not in text and "SystemExit" not in text, text


def test_without_a_log_an_unhandled_exception_still_reaches_stderr(tmp_path):
    """The redirect can fail (an unwritable log folder). The hook replaces the default printer,
    so it must then print to stderr itself rather than drop the traceback."""
    proc, text = _run(tmp_path, "slot-raises", no_log=True)

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "----- unhandled ValueError in thread 'MainThread'" in proc.stderr, text
    assert proc.stderr.count("ValueError: BOOM-IN-A-SLOT") == 1, text


# --- the build ID a frozen app puts in its banner --------------------------------------------

_FROZEN_CHILD = r'''
import os, sys
sys.path.insert(0, {pkg!r})
import nanoscope_app
sys.frozen = True
sys.executable = {exe!r}
sys._MEIPASS = {meipass!r}
nanoscope_app._redirect_stdio_to_log()
'''


def _fake_bundle(root: Path, plist_id: str | None, record_id: str | None) -> tuple[Path, Path]:
    """A NanoScope.app's shape: Contents/MacOS/<exe>, Contents/Info.plist, Frameworks/nanoscope."""
    contents = root / "NanoScope.app" / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    meipass = contents / "Frameworks"
    (meipass / "nanoscope").mkdir(parents=True)
    if plist_id is not None:
        with (contents / "Info.plist").open("wb") as fh:
            plistlib.dump({"CFBundleName": "NanoScope", "NanoScopeBuildID": plist_id}, fh)
    if record_id is not None:
        (meipass / "nanoscope" / "_build_info.json").write_text(
            '{"schema_version": 1, "build_id": "%s"}' % record_id, encoding="utf-8")
    return contents / "MacOS" / "NanoScope", meipass


def test_a_frozen_banner_carries_the_bundles_build_id(tmp_path):
    exe, meipass = _fake_bundle(tmp_path, "20261101T013909067040Z-d09f1d2e438b-PLIST", None)
    script = tmp_path / "start_frozen.py"
    script.write_text(_FROZEN_CHILD.format(pkg=str(PACKAGING), exe=str(exe), meipass=str(meipass)))
    home = tmp_path / "home"
    home.mkdir()

    subprocess.run([sys.executable, str(script)], env=_env(home), timeout=120, check=True,
                   capture_output=True)

    banners = _BANNER.findall(_log_path(home).read_text())
    assert [b[1] for b in banners] == ["20261101T013909067040Z-d09f1d2e438b-PLIST"], banners


def _import_launcher():
    sys.path.insert(0, str(PACKAGING))
    try:
        import nanoscope_app
    finally:
        sys.path.pop(0)
    return nanoscope_app


@pytest.mark.parametrize(("plist_id", "record_id", "expected"), [
    ("ID-FROM-PLIST", "ID-FROM-RECORD", "ID-FROM-PLIST"),  # the one `defaults read` shows
    (None, "ID-FROM-RECORD", "ID-FROM-RECORD"),            # Windows: no Info.plist
    ("", "ID-FROM-RECORD", "ID-FROM-RECORD"),              # a blank key is no ID
    (None, None, "unknown"),                               # a damaged bundle says so
])
def test_a_frozen_app_reads_its_build_id_from_its_own_bundle(tmp_path, monkeypatch,
                                                              plist_id, record_id, expected):
    launcher = _import_launcher()
    exe, meipass = _fake_bundle(tmp_path, plist_id, record_id)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)

    assert launcher._build_id() == expected
