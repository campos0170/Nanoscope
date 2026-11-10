"""Frozen-app entry script. PyInstaller builds THIS; it is not used by the source checkout.

Four jobs beyond calling `nanoscope.gui.app.main`:

1. Send stdout/stderr to a log file at the *file-descriptor* level, so C/C++ chatter (TensorFlow's
   core, Qt's plugin loader, HDF5) lands there too — not just Python `print`. A double-clicked
   .app has no terminal; without this the owner has no way to see why anything failed.
2. Turn an unhandled exception into a Qt dialog with the log path, instead of an icon that bounces
   once and vanishes.
3. Open the documents macOS hands us. `Info.plist` advertises .abf/.h5/.tdms/.csv/.txt, so the app
   shows up under "Open With" and can be a double-click target — but macOS delivers those as
   Cocoa open-document AppleEvents, which Qt turns into QFileOpenEvent, NOT as argv. Without the
   relay below the app would claim to handle those files and then open empty.
4. Leave a trail in that log that tells a crash from a quit and says which build crashed where:
   the build ID and pid in the start banner, a "clean exit rc=N" line on every controlled way
   out, faulthandler on the log, and hooks that date and name the thread of an exception that
   escaped a Qt slot or a Python thread. See _install_crash_diagnostics.
"""
from __future__ import annotations

import atexit
import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime

APP_NAME = "NanoScope"


def _log_dir() -> str:
    """Where this platform expects an application's log to live.

    Windows has no ~/Library: writing there would put a literal "Library\\Logs" folder in the
    user's profile, and %LOCALAPPDATA% is the location a locked-down corporate profile
    reliably allows writes to. This function is the reason the frozen app can be built for
    either OS from the same launcher.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, APP_NAME, "Logs")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Logs", APP_NAME)
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, APP_NAME)


_LOG_DIR = _log_dir()
_LOG_PATH = os.path.join(_LOG_DIR, "nanoscope.log")
_LOCK_PATH = _LOG_PATH + ".lock"
_MAX_LOG_BYTES = 2 * 1024 * 1024

# Deliberately kept alive for the whole process: the shared lock taken in _redirect_stdio_to_log
# is what tells a LATER instance "someone is writing to nanoscope.log right now, do not rename it
# out from under them". Closing this fd (or letting it be garbage-collected) would drop the lock.
_LOG_LOCK_FD: int | None = None

# The log as _redirect_stdio_to_log opened it (append mode), also kept for the whole process:
# faulthandler writes a crash's traceback through this descriptor from its signal handler, when
# no Python code can run any more, and the exit line and the exception hooks write here even if
# something rebinds sys.stderr.
_LOG_FILE = None
_EXIT_LOGGED = False


def _build_id() -> str:
    """Which build is writing the log, for the start banner: "source" for an unfrozen run.

    A frozen app reads the NanoScopeBuildID its own bundle's Info.plist carries (the value an
    owner can read back with `defaults read`), else the _build_info.json PyInstaller placed under
    nanoscope/ (Windows has no Info.plist; About reads the same record). Both are written from the
    one build record by NanoScope.spec. An ID, never a path.
    """
    if not getattr(sys, "frozen", False):
        return "source"
    try:
        import plistlib

        contents = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
        with open(os.path.join(contents, "Info.plist"), "rb") as fh:
            found = plistlib.load(fh).get("NanoScopeBuildID")
        if isinstance(found, str) and found:
            return found
    except Exception:
        pass
    try:
        import json

        record = os.path.join(sys._MEIPASS, "nanoscope", "_build_info.json")
        with open(record, encoding="utf-8") as fh:
            found = json.load(fh).get("build_id")
        if isinstance(found, str) and found:
            return found
    except Exception:
        pass
    return "unknown"


def _lock(fd: int, exclusive: bool, blocking: bool) -> bool:
    """Take an advisory lock on `fd`. Returns True if it was granted.

    Every OS this ships to spells this differently and one of them cannot do it at all, so the
    caller must be able to carry on without a lock. Returning False (never raising) is what lets
    the rotation below choose the safe branch — skip rotating — rather than race.
    """
    try:
        if sys.platform == "win32":
            # msvcrt has no shared mode; an exclusive byte-range lock is all Windows offers here.
            # LK_LOCK retries for ~10 s then raises; LK_NBLCK fails immediately.
            import msvcrt

            if not exclusive:
                return False
            msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            return True
        import fcntl

        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, mode if blocking else mode | fcntl.LOCK_NB)
        return True
    except (OSError, ImportError, ValueError):
        return False


def _redirect_stdio_to_log() -> str:
    """Point fd 1 and fd 2 at this platform's log file (see _log_dir). Returns the path.

    os.dup2 on the real descriptors (not just rebinding sys.stdout) is the point: TensorFlow and
    Qt write from C++ straight to fd 2 and would otherwise be lost.

    ROTATION IS LOCKED. It used to be a bare check-then-act pair, and two instances starting at
    the same moment (a shell/script double start, Login Items racing a startup script, a relaunch
    overlapping a still-shutting-down copy) both saw the same oversized log: the first renamed the
    old 2.5 MB log to nanoscope.log.1 and opened a fresh one, then the second renamed THAT fresh
    log on top of nanoscope.log.1 — destroying the previous session's diagnostics, the one thing
    "keep one previous run around" exists to promise. It also left the first instance writing into
    a file no longer reachable as nanoscope.log, so half a live session went where nobody looks.

    The lock file next to the log carries two claims:
      * EXCLUSIVE, taken non-blocking, means "no other instance is live" — only then may this
        process rename anything. It is held across the whole check → rename → open sequence, so
        the losing instance cannot see a half-rotated directory.
      * SHARED, held for the process lifetime, is that "I am live" claim. A later instance that
        cannot get the exclusive lock does not rotate at all; it appends to the same file the
        running instance is writing to, which is exactly where the reader will look.
    Rotation is therefore skipped, never guessed at, whenever it could not be done safely — and
    the skip is written into the log rather than being silent.
    """
    global _LOG_LOCK_FD, _LOG_FILE
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)

        if _LOG_LOCK_FD is not None:  # a second call in one process: drop the previous claim
            try:
                os.close(_LOG_LOCK_FD)
            except OSError:
                pass
            _LOG_LOCK_FD = None

        lock_fd, exclusive = None, False
        try:
            lock_fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError:
            lock_fd = None  # unwritable log dir; the open() below will report the real problem
        if lock_fd is not None:
            exclusive = _lock(lock_fd, exclusive=True, blocking=False)
            if not exclusive:
                # Someone else is starting or already running. Wait for them to finish rotating
                # (they hold the exclusive lock for microseconds) so we open the file they chose.
                _lock(lock_fd, exclusive=False, blocking=True)

        # Single-file rotation: keep one previous run around, don't grow without bound.
        note = ""
        oversized = os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > _MAX_LOG_BYTES
        if oversized and exclusive:
            try:
                os.replace(_LOG_PATH, _LOG_PATH + ".1")
            except OSError as exc:
                # Windows refuses to rename a file another process still has open. Say so and
                # keep appending — a log that grows past its cap beats no log at all.
                note = f"[log rotation failed, appending instead: {type(exc).__name__}: {exc}]\n"
        elif oversized:
            note = (f"[log is over {_MAX_LOG_BYTES} bytes but was NOT rotated: another {APP_NAME}"
                    " instance is using it, and renaming it would destroy the kept previous run"
                    " and split that instance's log]\n")

        log = open(_LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
        _LOG_FILE = log
        # The pid pairs this banner with its own exit line when two instances share the log.
        log.write(f"\n===== {APP_NAME} start {datetime.now().isoformat(timespec='seconds')}"
                  f" pid={os.getpid()} build={_build_id()} =====\n")
        if note:
            log.write(note)
        log.flush()

        if lock_fd is not None:
            # Downgrade to the shared "I am live" claim and hold it until the process exits. The
            # log file is already open above, so an instance that takes the exclusive lock during
            # this downgrade finds a freshly rotated (small) log and correctly does not rotate.
            if exclusive and not _lock(lock_fd, exclusive=False, blocking=False):
                os.close(lock_fd)  # no shared mode here (Windows): release rather than hold EX
                lock_fd = None
            _LOG_LOCK_FD = lock_fd
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace")
        sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace")
    except Exception:  # logging must never be the thing that kills the app
        pass
    return _LOG_PATH


def _restore_mpl_cache_dir() -> str:
    """Give the runtime hook's persistent MPLCONFIGDIR the last word. Returns a line for the log.

    packaging/rthooks/rthook_nanoscope_env.py points MPLCONFIGDIR at a per-user cache directory so
    matplotlib's ~10 s font-cache build happens once, ever. That hook cannot win on its own:
    PyInstaller runs custom runtime hooks FIRST and its own afterwards, and pyi_rth_mplconfig
    (always collected alongside matplotlib) then does an unconditional

        os.environ['MPLCONFIGDIR'] = secure_mkdtemp()   +   atexit rmtree of that directory

    so every launch got an empty cache dir, spent ~10 s rebuilding fontlist-v*.json into it, and
    deleted it on exit. Nothing was ever cached and nothing could be read back — which is why
    seeding the hook's directory by hand changed nothing either.

    THIS script runs after every runtime hook, and nothing above imports matplotlib, so putting
    the variable back here is equivalent to having set it before the hooks ran. The note is
    returned rather than printed because in the GUI path stdout is not a log file yet.
    """
    wanted = os.environ.get("NANOSCOPE_MPLCONFIGDIR")
    if not wanted:
        # Not frozen (the hook only runs inside the bundle), or the hook could not choose a
        # directory. Leave whatever matplotlib would do alone rather than invent a location, and
        # say nothing: there is no NanoScope decision here to report.
        return ""
    current = os.environ.get("MPLCONFIGDIR")
    if current == wanted:
        return f"MPLCONFIGDIR={wanted} (font cache persists here)"
    os.environ["MPLCONFIGDIR"] = wanted
    late = " — but matplotlib was ALREADY imported, so this launch still rebuilt the font cache" \
        if "matplotlib" in sys.modules else ""
    return (f"MPLCONFIGDIR was {current!r} (a throwaway directory from PyInstaller's own "
            f"matplotlib runtime hook); re-pointed at {wanted} so the font cache survives{late}")


def _install_file_open_relay(argv: list[str]):
    """Route macOS open-document requests into MainWindow.open_paths(). Returns the relay.

    Two sources, one sink:
      * argv — `open -a NanoScope.app --args file.abf`, or launching from a terminal;
      * QFileOpenEvent — double-clicking a document in the Finder, dropping files on the Dock
        icon, or "Open With > NanoScope". This is how macOS actually does it, and it can happen
        both before the window exists and at any time afterwards.

    The window is found by searching top-level widgets rather than being passed in, because
    `nanoscope.gui.app.main` owns MainWindow's construction and we deliberately do not
    reimplement it here. A zero-delay singleShot fires at the first turn of the event loop, by
    which time main() has built and shown the window.
    """
    from PySide6 import QtCore, QtWidgets

    class _FileOpenRelay(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__()
            self._pending: list[str] = []

        def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt's name
            if event.type() == QtCore.QEvent.Type.FileOpen:
                name = event.file()
                if name:
                    self.queue([name])
                return True
            return False

        def queue(self, paths) -> None:
            self._pending.extend(str(p) for p in paths if os.path.exists(p))
            if self._pending:
                QtCore.QTimer.singleShot(0, self.flush)

        def flush(self) -> None:
            if not self._pending:
                return
            window = next(
                (w for w in QtWidgets.QApplication.topLevelWidgets()
                 if hasattr(w, "open_paths")), None)
            if window is None:
                QtCore.QTimer.singleShot(200, self.flush)  # window not built yet; try again
                return
            paths, self._pending = self._pending, []
            print(f"opening {len(paths)} document(s) from macOS: {paths}")
            window.open_paths(paths)

    relay = _FileOpenRelay()
    app = QtWidgets.QApplication.instance()
    app.installEventFilter(relay)
    # `-psn_0_12345` is the process-serial-number argument LaunchServices appends; never a path.
    relay.queue([a for a in argv[1:] if not a.startswith("-")])
    return relay


def _report_fatal(exc: BaseException) -> None:
    """Last-ditch: write the traceback to the log and show it in a dialog if Qt still works."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle(f"{APP_NAME} could not start")
        box.setText(f"{APP_NAME} hit an unrecoverable error and has to close.")
        box.setInformativeText(f"The full details were written to:\n{_LOG_PATH}")
        box.setDetailedText(text)
        box.exec()
    except Exception:
        pass


def _exit_code(code) -> int:
    """The exit status `raise SystemExit(code)` gives the process (None is 0, a message is 1)."""
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def _log_clean_exit(rc: int | None) -> None:
    """Write the "clean exit" line, once: this process is ending under its own control.

    A start banner whose pid has no such line after it marks a crash, a force quit or a power
    loss. `rc` is the exit code; None means the caller cannot know it (atexit, reached by a
    sys.exit inside the event loop, which PySide6 turns into an exit that never returns from
    exec). Called from main() and atexit here, and through nanoscope.clean_exit by the os._exit
    in gui.worker_thread, which skips atexit.
    """
    global _EXIT_LOGGED
    if _EXIT_LOGGED or _LOG_FILE is None:
        return
    _EXIT_LOGGED = True
    try:
        _LOG_FILE.write(f"===== {APP_NAME} clean exit rc={'unknown' if rc is None else rc}"
                        f" {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} =====\n")
        _LOG_FILE.flush()
    except Exception:
        pass


def _log_unhandled(exc_type, exc, tb, thread_name: str) -> None:
    """Write a dated traceback that names its thread and pid to the log. Never raises."""
    try:
        out = _LOG_FILE or sys.stderr
        out.write(f"----- unhandled {exc_type.__name__} in thread {thread_name!r}"
                  f" {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} -----\n")
        out.write("".join(traceback.format_exception(exc_type, exc, tb)))
        out.flush()
    except Exception:
        pass


def _install_crash_diagnostics() -> None:
    """After the stdio redirect: make the log say how this process ended, and where it crashed.

    * faulthandler on the log. A segfault or a qFatal abort in Qt or another C/C++ library kills
      the process before Python can report anything, so the log used to just stop. Now it ends
      in "Fatal Python error: Segmentation fault" (or "Aborted") and every thread's Python stack.
      faulthandler re-raises the signal afterwards, so macOS still writes its own crash report.
    * sys.excepthook and threading.excepthook. An exception that escapes a Qt slot does not end
      the app: PySide6 hands it to sys.excepthook and carries on. The default hook prints a bare
      traceback among Qt's chatter; these date it and name the thread and pid first. A hook
      installed before these still runs after them. The interpreter's own default does not: it
      would print the same traceback into the same log a second time.
    * the clean-exit line on a way out that never returns to main(): atexit, for a sys.exit inside
      the event loop. (main() arms nanoscope.clean_exit for gui.worker_thread's os._exit.)
    Each step stands alone and none may raise: diagnostics must never be what stops the app.
    """
    try:
        faulthandler.enable(file=_LOG_FILE or sys.stderr, all_threads=True)
    except Exception:
        pass  # no log and no usable stderr (a windowed launch whose redirect failed)

    previous = sys.excepthook

    def _excepthook(exc_type, exc, tb) -> None:
        _log_unhandled(exc_type, exc, tb, threading.current_thread().name)
        if previous is not sys.__excepthook__:
            previous(exc_type, exc, tb)

    sys.excepthook = _excepthook

    previous_thread_hook = threading.excepthook

    def _thread_excepthook(args) -> None:
        if not issubclass(args.exc_type, SystemExit):  # the default is silent about it too
            name = args.thread.name if args.thread is not None else "<unknown>"
            _log_unhandled(args.exc_type, args.exc_value, args.exc_traceback, name)
        if previous_thread_hook is not threading.__excepthook__:
            previous_thread_hook(args)

    threading.excepthook = _thread_excepthook
    atexit.register(_log_clean_exit, None)


def main() -> int:
    # Before ANY import that could pull in matplotlib, and before either branch below: the last
    # runtime hook to touch MPLCONFIGDIR is PyInstaller's, not ours. See _restore_mpl_cache_dir.
    mpl_note = _restore_mpl_cache_dir()

    # `--self-test` runs the in-bundle smoke test on the terminal instead of starting the GUI:
    # readers, worker thread, TensorFlow, figure export. Keep this BEFORE the stdio redirect so
    # its output goes to the terminal the owner ran it from. See nanoscope_selftest.py.
    if "--self-test" in sys.argv:
        # Redirected output on Windows is encoded with the ANSI code page (cp1252), which has no
        # character for the check marks the acid analysis prints, so the print raised and failed
        # that check. The GUI branch already writes its log as UTF-8; do the same here.
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (OSError, ValueError):
                    pass
        import nanoscope_selftest

        if mpl_note:
            print(mpl_note)
        args = [a for a in sys.argv[1:] if a != "--self-test" and not a.startswith("-psn_")]
        return nanoscope_selftest.run(args)

    _redirect_stdio_to_log()
    _install_crash_diagnostics()
    if mpl_note:
        print(mpl_note)
    print(f"{APP_NAME} frozen={getattr(sys, 'frozen', False)} python={sys.version.split()[0]}")
    print(f"executable={sys.executable}")
    print(f"QT_PLUGIN_PATH={os.environ.get('QT_PLUGIN_PATH', '<unset>')}")
    try:
        from nanoscope import clean_exit

        # A quit over a worker that will not stop leaves through os._exit, which skips atexit.
        clean_exit.arm(_log_clean_exit)
    except Exception:  # noqa: BLE001 - a diagnostic never stops the app; that quit loses its line
        pass
    try:
        from nanoscope.gui.app import create_app, main as app_main

        # create_app is idempotent and returns the singleton, so calling it here just brings the
        # QApplication into existence early enough to attach the relay; main() then reuses it.
        create_app(sys.argv)
        relay = _install_file_open_relay(sys.argv)  # noqa: F841 - must outlive main()

        rc = int(app_main(sys.argv))
        _log_clean_exit(rc)
        return rc
    except BaseException as exc:  # noqa: BLE001 - this is the top of the stack
        if isinstance(exc, SystemExit):
            _log_clean_exit(_exit_code(exc.code))
            raise
        _report_fatal(exc)
        _log_clean_exit(1)
        return 1


if __name__ == "__main__":
    # macOS multiprocessing/forkserver safety: any library that re-executes the frozen binary as a
    # worker (joblib via scikit-learn, TF's data pipeline) must not re-enter the GUI here.
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
