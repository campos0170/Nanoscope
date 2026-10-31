"""One rule for a list of shortcut keys: register each RESOLVED sequence once.

What ``QtGui.QKeySequence.StandardKey.Redo`` resolves to is a platform-theme decision, and
the themes disagree: Ctrl+Y under the generic theme the offscreen test platform falls back
to, Ctrl+Shift+Z (⇧⌘Z) under the Mac one (measured on the cocoa platform, 2026-10-30). A
key list that names the standard key AND ``"Ctrl+Shift+Z"`` therefore registers ⇧⌘Z twice
on a Mac, and Qt answers a doubled registration with "Ambiguous shortcut overload" and
triggers NOTHING (measured: 0 of 1 presses with the pair, 1 of 1 with one registration).
That took the main window's Redo key away on exactly the platform the app is used on, while
every test, run offscreen where the two differ, stayed green.

So every list that pairs a standard key with a literal goes through
:func:`unique_sequences`. Sequences resolve against the running application's theme, so
call it after the ``QGuiApplication`` exists, never at import time.
"""

from __future__ import annotations

from PySide6 import QtGui

__all__ = ["REDO_KEYS", "UNDO_KEYS", "redo_sequences", "undo_sequences", "unique_sequences"]

#: Undo: the platform's own key (⌘Z on a Mac, Ctrl+Z elsewhere).
UNDO_KEYS: tuple = (QtGui.QKeySequence.StandardKey.Undo,)
#: Redo: the platform's own key plus Ctrl/⌘+Shift+Z by name, which every platform's users
#: also reach for. On a Mac the two are the same sequence, which is why this goes through
#: :func:`unique_sequences`.
REDO_KEYS: tuple = (QtGui.QKeySequence.StandardKey.Redo, "Ctrl+Shift+Z")


def unique_sequences(*keys) -> list[QtGui.QKeySequence]:
    """``keys`` (standard keys, strings or sequences) resolved, in order, each once.

    An empty resolution (a standard key the theme does not bind) is dropped.
    """
    out: list[QtGui.QKeySequence] = []
    for key in keys:
        seq = QtGui.QKeySequence(key)
        if seq.isEmpty() or any(seq == seen for seen in out):
            continue
        out.append(seq)
    return out


def undo_sequences() -> list[QtGui.QKeySequence]:
    """The Undo keys for this platform, each once."""
    return unique_sequences(*UNDO_KEYS)


def redo_sequences() -> list[QtGui.QKeySequence]:
    """The Redo keys for this platform, each once."""
    return unique_sequences(*REDO_KEYS)
