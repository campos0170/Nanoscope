"""Wait for the real asynchronous Inspector fit workflow, including terminal UI delivery."""
import time
from PySide6 import QtWidgets


def wait_for_fit(stats, timeout=60):
    deadline = time.monotonic() + timeout
    while getattr(stats, '_fit_busy', False) and time.monotonic() < deadline:
        # 2026-11-11: this used to wait with QtTest.QTest.qWait(10). On PySide6 6.11 qWait keeps
        # the GIL while it waits, so a Python worker thread is starved: a thread of 1500 small
        # numpy sorts took 0.01 s with the two lines below and 10 s under qWait(10). The fit is
        # such a thread (the dwell fit with a measured eta(t) is a bootstrap of many small numpy
        # calls). The deadline and the final assert are unchanged.
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    assert not getattr(stats, '_fit_busy', False), 'Inspector fit worker did not finish'
