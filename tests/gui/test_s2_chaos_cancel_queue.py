"""chaos2-001 — CANCEL DURING A MULTI-FILE OPEN ABANDONS THE WHOLE OPEN, QUEUE AND ALL.

``_cancel_load`` aborted the active load and then called ``_finish_active()`` — which
pops the open queue and STARTS THE NEXT LOAD inside the same slot. The status bar read
"Cancelled." (its docstring: "It must not claim the work stopped when it did not") at
the exact moment the next queued file began loading; that file then opened as if
Cancel had never been pressed, and the only way to stop the rest of a multi-select
File ▸ Open (or multi-file drop) was one Cancel per file, racing each load start. The
"result is discarded" variant was outright false for the queued load, whose result was
kept and shown. Nothing but ``_clear_session`` ever cleared the queue.

THE FIX THESE TESTS PIN. Cancel empties the pending open queue BEFORE
``_finish_active`` runs, and the message says how many queued files were dropped —
they did not merely stop, they never started. The beta chrome contract
(tests/gui/test_beta_chaos_load_chrome_midsweep.py) and the "Cancelled." honesty tests
(tests/gui/test_cancel_deadlock.py) are unchanged and stay green beside this file.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.gui.main_window import MainWindow

FS = 100_000.0


def _write_csv(path: Path, n: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": arr}).to_csv(path, index=False)
    return path


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._controller.abort()


def _settle(window, qtbot) -> None:
    """Let every in-flight or (wrongly) queued load run to its end, so the assertions
    see the world the user would."""
    qtbot.waitUntil(lambda: window._active_open is None and not window._open_queue,
                    timeout=30_000)
    qtbot.wait(150)


def test_cancel_mid_multi_open_drops_the_queue_and_says_so(window, qtbot, tmp_path):
    # A biggish first file so its load is honestly in flight, and a second one queued
    # behind it — the user's multi-select File ▸ Open.
    first = _write_csv(tmp_path / "first.csv", 400_000, seed=0)
    second = _write_csv(tmp_path / "second.csv", 20_000, seed=1)
    window.open_paths([str(first), str(second)])
    assert window._active_open is not None, "the first load must be in flight"
    assert [Path(p) for p in window._open_queue] == [second], \
        "the second file must be QUEUED when Cancel is pressed"

    window._cancel_load()

    # The queue is emptied INSIDE the cancel — before _finish_active can pop it.
    assert window._open_queue == [], \
        "Cancel left the open queue standing; the next file will load as if Cancel " \
        "had never been pressed"
    msg = window.statusBar().currentMessage()
    assert "Cancel" in msg, msg
    assert "1 more queued file(s) were not opened" in msg, (
        "the dropped queue is not disclosed — the user cannot know the second file "
        "will never open:\n" + repr(msg))

    _settle(window, qtbot)
    names = [d.meta.path.name for d in window._session.documents()]
    assert "second.csv" not in names, (
        "the queued file opened anyway — Cancel cancelled only the active load:\n"
        f"{names}")


def test_cancel_of_a_single_load_keeps_the_plain_message(window, qtbot, tmp_path):
    """No queue, no queue clause: the beta 'Cancelled.' honesty wording must come
    through untouched when there is nothing queued to drop."""
    only = _write_csv(tmp_path / "only.csv", 400_000, seed=2)
    window.open_paths([str(only)])
    assert window._active_open is not None

    window._cancel_load()

    msg = window.statusBar().currentMessage()
    assert "Cancel" in msg, msg
    assert "queued file" not in msg, (
        "a queue clause with no queue behind it is noise that will be read as a "
        "second failure:\n" + repr(msg))
    _settle(window, qtbot)


def test_a_multi_open_left_alone_still_opens_everything(window, qtbot, tmp_path):
    """The healthy path the fix must not touch: without a Cancel, the queue drains and
    both files open."""
    a = _write_csv(tmp_path / "a.csv", 20_000, seed=3)
    b = _write_csv(tmp_path / "b.csv", 20_000, seed=4)
    window.open_paths([str(a), str(b)])
    _settle(window, qtbot)

    names = sorted(d.meta.path.name for d in window._session.documents())
    assert names == ["a.csv", "b.csv"], names
