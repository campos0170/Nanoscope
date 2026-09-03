"""Post-build integrity checks on NanoScope.app. Run by build_app.sh before signing.

Five jobs, all of which exist because the failure they catch is invisible until launch or leaks
details of the build machine into the release artifact:

1. QT MAJOR VERSION. This build runs inside an Anaconda env whose qt.conf misdirects Qt's own
   path introspection at /opt/anaconda3/plugins — Qt **5.15** plugins (see qt_prefix_fix.py).
   A bundle that picked those up builds, signs and verifies perfectly, then dies at launch with
   'Could not find the Qt platform plugin "cocoa"'. Assert the platform plugin links Qt6.

2. DANGLING SYMLINKS. PyInstaller mirrors each collected Qt framework as a symlink in both
   Contents/Frameworks and Contents/Resources. When the spec's _prune_toc drops a framework's
   payload (QtQml, QtQuick, QtPdf — none reachable from this app), the two symlinks survive
   pointing at nothing, and `codesign --deep` fails the entire bundle with a bare
   "No such file or directory". Remove them — but only after proving no Mach-O in the bundle
   actually references the name, by scanning for the exact NUL-terminated load-command string
   rather than a substring (plain grep for "@rpath/QtQml" also matches "@rpath/QtQmlModels").

3. DEPLOYMENT TARGET. ``LSMinimumSystemVersion`` is a hand-written constant in the spec while
   the real floor is a property of whatever the build env's wheels were compiled against, so
   the two drift silently. They had: the plist said 11.0 while eleven PySide6/shiboken6
   binaries were built for 15.0. Below that floor macOS PERMITS the launch and then dyld
   refuses PySide6 — and since the app's own error dialog is PySide6, it cannot report it: the
   icon bounces and vanishes. The bundle builds, signs, verifies and passes its self-test on
   the developer's machine, which is by definition above the floor. So derive the truth from
   the Mach-O load commands and refuse to ship a manifest that promises more than the binaries
   can deliver.

4. NON-HERMETIC RPATHS. Third-party wheels can retain absolute ``LC_RPATH`` entries from their
   build host (for example ``/opt/gfortran-darwin-arm64-cross/...``). Even when a later
   ``@loader_path`` entry currently finds the bundled library, leaving the absolute entry first
   makes lookup depend on the tester's machine and exposes a path from the build environment.
   Delete every absolute rpath except Apple's two system roots before the final signature.

5. INSTALLER PROVENANCE. ``copy_metadata()`` preserves PEP 610 ``direct_url.json`` files. An
   editable install records the developer's checkout as ``file:///Users/...``; conda packages
   record private temporary build paths. None is needed by ``importlib.metadata``, so remove only
   those files inside ``*.dist-info`` directories while retaining METADATA and entry points.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MACHO_MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}
_SYSTEM_RPATH_ROOTS = ("/System/Library", "/usr/lib")


def _is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) in MACHO_MAGIC
    except OSError:
        return False


def _macho_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        if _is_macho(p):
            out.append(p)
    return out


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def check_deployment_target(app: Path) -> None:
    """Assert Info.plist's LSMinimumSystemVersion is not below what the binaries require.

    ``otool -l`` reports ``minos`` under LC_BUILD_VERSION and ``version`` under the older
    LC_VERSION_MIN_MACOSX; both are the deployment target. The bundle's floor is the MAXIMUM
    over every Mach-O it ships, because one unloadable dylib is enough — and the one that bit
    us was Qt, which the app cannot run without and cannot report the loss of.

    The key must be read WITHIN its load command, not matched by name across the whole dump:
    LC_SOURCE_VERSION also prints a bare ``version`` line, and reading that gave "macOS 1230.1"
    from the main executable's source version on the first attempt at this check.
    """
    import plistlib

    plist_path = app / "Contents" / "Info.plist"
    with plist_path.open("rb") as fh:
        declared = str(plistlib.load(fh).get("LSMinimumSystemVersion", "0"))

    worst, culprits = (0,), []
    for binary in _macho_files(app):
        try:
            out = subprocess.run(["otool", "-l", str(binary)], capture_output=True,
                                 text=True, check=False).stdout
        except OSError:
            continue
        cmd = ""
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "cmd":
                cmd = parts[1]
                continue
            wanted = ("minos" if cmd == "LC_BUILD_VERSION"
                      else "version" if cmd == "LC_VERSION_MIN_MACOSX" else None)
            if wanted is None or len(parts) != 2 or parts[0] != wanted:
                continue
            key = _version_key(parts[1])
            if not key:
                continue
            if key > worst:
                worst, culprits = key, [binary]
            elif key == worst and len(culprits) < 12:
                culprits.append(binary)

    required = ".".join(str(n) for n in worst)
    if worst > _version_key(declared):
        names = "\n      ".join(str(p.relative_to(app)) for p in culprits)
        raise SystemExit(
            f"FATAL: Info.plist declares LSMinimumSystemVersion {declared}, but {len(culprits)} "
            f"bundled binaries are built for macOS {required} and cannot load below it:\n"
            f"      {names}\n"
            f"    On an older Mac the app launches and then dyld refuses these — and if Qt is "
            f"among them the app cannot even show an error. Set LSMinimumSystemVersion to "
            f"{required} in NanoScope.spec and say the same in INSTALL.txt.")
    print(f"    deployment target: binaries need macOS {required}, plist declares {declared}")


def check_qt_major(app: Path) -> None:
    cocoa = app / "Contents/Frameworks/PySide6/Qt/plugins/platforms/libqcocoa.dylib"
    if not cocoa.is_file():
        raise SystemExit("FATAL: no cocoa platform plugin in the bundle")

    links = subprocess.run(["otool", "-L", str(cocoa)], capture_output=True, text=True).stdout
    if "libQt5" in links:
        raise SystemExit(f"FATAL: libqcocoa.dylib links Qt5 — qt_prefix_fix did not take.\n{links}")

    qtgui = [ln.strip() for ln in links.splitlines() if "@rpath/QtGui" in ln]
    if not qtgui:
        raise SystemExit(f"FATAL: libqcocoa.dylib links no QtGui at all.\n{links}")
    if "current version 6." not in qtgui[0]:
        raise SystemExit(f"FATAL: libqcocoa.dylib is not built against Qt6: {qtgui[0]}")

    stray = list((app / "Contents/Frameworks").glob("libQt5*"))
    if stray:
        raise SystemExit(f"FATAL: conda Qt5 libraries leaked into the bundle: {stray}")
    print(f"    cocoa plugin OK -> {qtgui[0]}")


def _rpaths_from_otool(output: str) -> list[str]:
    """Return LC_RPATH values from ``otool -l`` output, including paths with spaces."""
    command = ""
    rpaths: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) == 2 and parts[0] == "cmd":
            command = parts[1]
            continue
        if command != "LC_RPATH" or not stripped.startswith("path "):
            continue
        value = stripped.removeprefix("path ").rsplit(" (offset ", 1)[0]
        if value:
            rpaths.append(value)
        command = ""
    return rpaths


def _is_non_system_absolute_rpath(path: str) -> bool:
    if not path.startswith("/"):
        return False
    return not any(path == root or path.startswith(f"{root}/")
                   for root in _SYSTEM_RPATH_ROOTS)


def clean_non_system_absolute_rpaths(app: Path) -> None:
    """Remove build-host LC_RPATH entries before the bundle receives its final signature."""
    removed: list[tuple[Path, str]] = []
    for binary in _macho_files(app):
        inspected = subprocess.run(
            ["otool", "-l", str(binary)], capture_output=True, text=True, check=False)
        if inspected.returncode != 0:
            raise SystemExit(
                f"FATAL: otool could not inspect {binary}: {inspected.stderr.strip()}")

        leaked = list(dict.fromkeys(
            path for path in _rpaths_from_otool(inspected.stdout)
            if _is_non_system_absolute_rpath(path)
        ))
        for rpath in leaked:
            changed = subprocess.run(
                ["install_name_tool", "-delete_rpath", rpath, str(binary)],
                capture_output=True, text=True, check=False)
            if changed.returncode != 0:
                raise SystemExit(
                    f"FATAL: could not remove LC_RPATH {rpath!r} from {binary}: "
                    f"{changed.stderr.strip()}")
            removed.append((binary, rpath))

        if leaked:
            verified = subprocess.run(
                ["otool", "-l", str(binary)], capture_output=True, text=True, check=False)
            remaining = [
                path for path in _rpaths_from_otool(verified.stdout)
                if _is_non_system_absolute_rpath(path)
            ]
            if verified.returncode != 0 or remaining:
                raise SystemExit(
                    f"FATAL: non-system absolute LC_RPATH remains in {binary}: {remaining}")

    if not removed:
        print("    no non-system absolute LC_RPATH entries")
        return
    print(f"    removed {len(removed)} non-system absolute LC_RPATH entr"
          f"{'y' if len(removed) == 1 else 'ies'}:")
    for binary, rpath in removed:
        print(f"      {binary.relative_to(app)} -> {rpath}")


def clean_direct_url_metadata(app: Path) -> None:
    """Remove PEP 610 build/install paths while keeping useful distribution metadata."""
    metadata = [
        path for path in (app / "Contents").rglob("direct_url.json")
        if path.parent.name.endswith(".dist-info")
    ]
    for path in metadata:
        path.unlink()
    print(f"    removed {len(metadata)} direct_url.json metadata file(s)")


def clean_dangling_symlinks(app: Path) -> None:
    dangling = [p for p in app.rglob("*") if p.is_symlink() and not p.exists()]
    if not dangling:
        print("    no dangling symlinks")
        return

    machos = _macho_files(app)
    names = {p.name for p in dangling}
    # One pass over every Mach-O, collecting which of the candidate names are genuinely referenced.
    referenced: set[str] = set()
    needles = {n: f"@rpath/{n}\0".encode() for n in names}
    for path in machos:
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        for name, needle in needles.items():
            if name not in referenced and needle in blob:
                referenced.add(name)

    if referenced:
        raise SystemExit(
            f"FATAL: {sorted(referenced)} are still linked by bundled binaries but their payload "
            f"was pruned. Remove the matching entries from _PRUNE_FRAGMENTS in NanoScope.spec."
        )

    for link in dangling:
        link.unlink()
    print(f"    removed {len(dangling)} unreferenced symlink(s): {sorted(names)}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: verify_bundle.py /path/to/NanoScope.app", file=sys.stderr)
        return 2
    app = Path(argv[0]).resolve()
    if not app.is_dir():
        raise SystemExit(f"FATAL: {app} is not a directory")
    check_qt_major(app)
    check_deployment_target(app)
    clean_dangling_symlinks(app)
    clean_non_system_absolute_rpaths(app)
    clean_direct_url_metadata(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
