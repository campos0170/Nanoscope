"""Qt plugin-path fix. MUST be imported before any Qt import, in the app AND in every test/script.

WHY (source checkout / conda): /opt/anaconda3/bin/qt.conf sits next to the python binary and sets
Prefix=/opt/anaconda3. Qt reads qt.conf from the executable's directory, so PySide6's Qt6 hunts for
plugins in /opt/anaconda3/plugins, which contains conda's Qt5 plugins (PyQt5 5.15.10 is installed).
Qt6 cannot load Qt5 plugins => 'Could not find the Qt platform plugin "cocoa"'.
Do NOT rely on shell env vars — that breaks for other users. This is diagnosed and solved.

WHY THE FROZEN GUARD: inside a PyInstaller bundle there is no conda qt.conf to defend against —
the interpreter is the app's own binary in Contents/MacOS — and PyInstaller's PySide6 runtime hook
has ALREADY pointed QT_PLUGIN_PATH at the bundled Contents/Frameworks/PySide6/Qt/plugins and
written a matching qt.conf. Recomputing the path from ``PySide6.__file__`` there would aim Qt at
Contents/Resources/PySide6/Qt/plugins (the *data* half of the split-layout bundle) and fight the
hook. So: apply the fix only when NOT frozen, and never clobber a plugin path someone else set.
"""
import os
import sys
from pathlib import Path as _Path


def _is_frozen() -> bool:
    """True inside a PyInstaller (or similar) one-dir/.app bundle."""
    return bool(getattr(sys, "frozen", False)) or hasattr(sys, "_MEIPASS")


if not _is_frozen():
    import PySide6 as _PySide6

    _plugins = _Path(_PySide6.__file__).parent / "Qt" / "plugins"
    os.environ["QT_PLUGIN_PATH"] = str(_plugins)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_plugins / "platforms")
