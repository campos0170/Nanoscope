"""Where the frozen app's "clean exit" log line is reached from the ways out that skip atexit.

packaging/nanoscope_app.py (the frozen entry point) writes a start banner into the app log and a
"clean exit rc=N" line when the process ends under its own control, so a banner with no exit line
after it marks a crash: a segfault, a qFatal abort, a force quit. The launcher writes that line
itself when the app returns to it, and atexit covers a sys.exit inside the event loop. One way out
reaches neither: ``gui.worker_thread.exit_without_destroying_live_threads`` ends the process with
``os._exit``, which runs no atexit handler. It calls :func:`note_clean_exit` first, and the
launcher passes its writer to :func:`arm`, so the two meet here without the package importing the
launcher.

Unarmed (a source run, a test, a script) :func:`note_clean_exit` does nothing.
"""
from __future__ import annotations

from collections.abc import Callable

_writer: Callable[[int | None], None] | None = None


def arm(writer: Callable[[int | None], None] | None) -> None:
    """Make ``writer(rc)`` what :func:`note_clean_exit` calls. None disarms."""
    global _writer
    _writer = writer


def note_clean_exit(rc: int | None) -> None:
    """Hand ``rc`` to the armed writer. Never raises: it runs on the way out of the process."""
    writer = _writer
    if writer is None:
        return
    try:
        writer(rc)
    except Exception:
        pass
