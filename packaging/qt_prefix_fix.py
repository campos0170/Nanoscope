"""Point PyInstaller's Qt introspection at PySide6's OWN Qt tree instead of Anaconda's.

THIS IS LANDMINE #1, MOVED TO BUILD TIME. `nanoscope/bootstrap.py` exists because
/opt/anaconda3/bin/qt.conf declares `Prefix = /opt/anaconda3`, and Qt reads qt.conf from the
directory of the running executable. PyInstaller determines where Qt's plugins and libraries live
by importing PySide6 in an isolated subprocess — a subprocess whose executable is
/opt/anaconda3/bin/python3 — and asking QLibraryInfo. It therefore gets told:

    version      = 6.11.1          <- PySide6's Qt6, correct
    PrefixPath   = /opt/anaconda3  <- conda's tree, WRONG
    PluginsPath  = /opt/anaconda3/plugins
    LibrariesPath= /opt/anaconda3/lib

/opt/anaconda3/plugins holds conda's **Qt 5.15** plugins (PyQt5 is installed in this env). So a
build without this fix silently ships Qt5's libqcocoa.dylib inside a Qt6 application, plus
libQt5Core/libQt5Gui pulled in as its dependencies, and the frozen app dies at launch with

    Plugin uses incompatible Qt library (5.15.0)
    qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""

Setting QT_PLUGIN_PATH in the build environment does NOT help: QLibraryInfo reports the compiled-
in/qt.conf prefix and ignores that variable. The fix is to rewrite the paths PyInstaller derived,
re-rooting every one of them from the conda prefix onto <site-packages>/PySide6/Qt, which is where
the Qt that PySide6 actually loads lives.
"""
from __future__ import annotations

import pathlib


def force_pyside6_qt_prefix(verbose: bool = True) -> dict[str, str]:
    """Re-root PyInstaller's PySide6 location table onto the PySide6 package. Idempotent.

    Returns the corrected location mapping (also useful as a build-log receipt).
    """
    import PySide6
    from PyInstaller.utils.hooks import qt as qt_hooks

    info = qt_hooks.pyside6_library_info
    location = dict(info.location)  # first access triggers the isolated QLibraryInfo query

    package_qt = (pathlib.Path(PySide6.__file__).resolve().parent / "Qt").resolve()
    old_prefix = pathlib.Path(location["PrefixPath"]).resolve()

    if old_prefix == package_qt or package_qt in old_prefix.parents:
        if verbose:
            print(f"[qt_prefix_fix] PrefixPath already inside PySide6 ({old_prefix}); nothing to do")
        return location

    if not (package_qt / "plugins" / "platforms").is_dir():
        raise SystemExit(
            f"REFUSING TO BUILD: expected PySide6's Qt plugins at {package_qt / 'plugins'}, "
            f"but that directory does not exist. PySide6 install looks wrong."
        )

    remapped: dict[str, str] = {}
    for key, value in location.items():
        try:
            rel = pathlib.Path(value).resolve().relative_to(old_prefix)
        except ValueError:
            remapped[key] = value  # already outside the conda prefix; leave it alone
            continue
        remapped[key] = str(package_qt / rel)

    info.location = remapped
    # Recompute the two values _load_qt_info derived from the (wrong) table.
    info.qt_inside_package = True
    info.qt_lib_dir = pathlib.Path(remapped["LibrariesPath"]).resolve()

    if verbose:
        print(f"[qt_prefix_fix] re-rooted Qt paths {old_prefix} -> {package_qt}")
        for key in ("PrefixPath", "PluginsPath", "LibrariesPath"):
            print(f"[qt_prefix_fix]   {key} = {remapped[key]}")
    return remapped
