"""Wait for the real asynchronous Inspector fit workflow, including terminal UI delivery."""
import time
from PySide6 import QtTest


def wait_for_fit(stats, timeout=60):
    deadline = time.monotonic() + timeout
    while getattr(stats, '_fit_busy', False) and time.monotonic() < deadline:
        QtTest.QTest.qWait(10)
    assert not getattr(stats, '_fit_busy', False), 'Inspector fit worker did not finish'
