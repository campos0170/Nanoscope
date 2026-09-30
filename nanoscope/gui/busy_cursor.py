"""Paired wait cursors without the affected macOS image-conversion path."""
from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets


def begin_wait_cursor() -> bool:
    """Push a wait cursor where safe; return whether this call pushed one.

    Qt 6.11.1's Cocoa wait cursor converts a PNG through QImage::toCGImage,
    where a released CGColorSpaceRef can crash the process. Python cannot
    catch that native fault. On macOS the caller's status text supplies busy
    feedback instead; never pop an unrelated override when no cursor was pushed.
    """
    if sys.platform == "darwin":
        return False
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    return True


def end_wait_cursor(active: bool) -> None:
    """Restore only an override pushed by the matching begin call."""
    if active:
        QtWidgets.QApplication.restoreOverrideCursor()
