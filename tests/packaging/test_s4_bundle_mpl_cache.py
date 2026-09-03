"""BD4-01 — matplotlib's font cache must survive a launch, and the self-test must check that.

The packaging promises the ~10 s font-cache build is a one-time cost: the runtime hook points
MPLCONFIGDIR at ``~/Library/Caches/NanoScope/matplotlib`` ("First run builds the font cache"),
and INSTALL.txt tells the labmate "Later launches are much faster." Neither was true. PyInstaller
runs CUSTOM runtime hooks first and ITS OWN afterwards, and because matplotlib is collected it
always includes ``pyi_rth_mplconfig``, whose entire body is

    configdir = secure_mkdtemp(); os.environ['MPLCONFIGDIR'] = configdir
    atexit.register(shutil.rmtree, configdir, ignore_errors=True)

so every launch got an empty directory, rebuilt fontlist-v390.json into it (~10 s, measured 9.8 s
here) and deleted it on exit. Nothing was ever written to the persistent directory and nothing
was ever read back — seeding it by hand changed nothing either.

These tests reproduce that launch ORDER (custom hook → PyInstaller's hook → entry script) in a
subprocess and then let the real matplotlib decide where its cache goes. The .app cannot be
rebuilt from this test, so the source modules are exercised directly; a bundle rebuild is what
carries the fix into dist/NanoScope.app.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[2] / "packaging"
RTHOOK = PACKAGING / "rthooks" / "rthook_nanoscope_env.py"

# One simulated frozen launch. Only PyInstaller's own hook is transcribed rather than imported
# (its secure_mkdtemp helper exists only inside a bundle); everything else is the real code.
_LAUNCH = """
import atexit, importlib.util, json, os, shutil, sys, tempfile, time
from pathlib import Path

exec(compile(open({rthook!r}).read(), {rthook!r}, "exec"), {{"__name__": "rthook_nanoscope_env"}})

configdir = tempfile.mkdtemp()                     # pyi_rth_mplconfig, verbatim in effect
os.environ["MPLCONFIGDIR"] = configdir
atexit.register(shutil.rmtree, configdir, ignore_errors=True)

spec = importlib.util.spec_from_file_location("nanoscope_app", {app!r})
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
note = app._restore_mpl_cache_dir()

t0 = time.time()
import matplotlib
import matplotlib.font_manager
elapsed = time.time() - t0

persistent = os.environ.get("NANOSCOPE_MPLCONFIGDIR", "")
cache = Path(matplotlib.get_cachedir())
print(json.dumps({{
    "note": note,
    "throwaway": configdir,
    "persistent": persistent,
    "cachedir": str(cache),
    "fontlists": [p.name for p in sorted(Path(persistent).glob("fontlist-v*.json"))] if persistent else [],
    "seconds": elapsed,
}}))
"""


def _launch(tmp_path: Path, home: Path) -> dict:
    script = tmp_path / "frozen_launch.py"
    script.write_text(_LAUNCH.format(rthook=str(RTHOOK), app=str(PACKAGING / "nanoscope_app.py")))
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    env["TMPDIR"] = str(tmp_path / "tmp")
    (tmp_path / "tmp").mkdir(exist_ok=True)
    for stale in ("MPLCONFIGDIR", "NANOSCOPE_MPLCONFIGDIR"):
        env.pop(stale, None)
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def launches(tmp_path_factory):
    """Two simulated frozen launches sharing one HOME. Module-scoped because the first one really
    does spend ~10 s building the font list — the very cost this defect is about."""
    work = tmp_path_factory.mktemp("frozen")
    home = work / "home"
    home.mkdir()
    return home, _launch(work, home), _launch(work, home)


def test_the_font_cache_is_built_once_and_reused_on_the_next_launch(launches):
    home, first, second = launches

    persistent = Path(first["persistent"])
    # The hook's own per-user directory, not PyInstaller's throwaway one, is where the cache goes.
    expected = {"darwin": home / "Library" / "Caches" / "NanoScope" / "matplotlib",
                "win32": home / "AppData" / "Local" / "NanoScope" / "matplotlib"}.get(
                    sys.platform, home / ".cache" / "NanoScope" / "matplotlib")
    assert persistent == expected, first
    assert Path(first["cachedir"]).resolve() == persistent.resolve(), first
    assert first["cachedir"] != first["throwaway"], first
    assert first["fontlists"] and first["fontlists"] == second["fontlists"], (first, second)
    # And the cost the packaging promised to pay once is in fact paid once. The build is ~10 s and
    # a load is ~0.2 s; a factor of three is far outside the noise and far inside that gap.
    assert second["seconds"] * 3 < first["seconds"], (first["seconds"], second["seconds"])


def test_the_launcher_says_which_directory_it_had_to_repair(launches):
    """No silent fix: the log line names the throwaway directory it overrode and the one it chose."""
    info = launches[1]

    assert info["throwaway"] in info["note"], info["note"]
    assert info["persistent"] in info["note"], info["note"]
    assert "PyInstaller's own matplotlib runtime hook" in info["note"]


def test_outside_the_bundle_the_launcher_says_nothing_and_changes_nothing(tmp_path, monkeypatch):
    """Unfrozen, no runtime hook has run, so there is no NanoScope decision to report or enforce."""
    sys.path.insert(0, str(PACKAGING))
    try:
        import nanoscope_app
    finally:
        sys.path.pop(0)

    monkeypatch.delenv("NANOSCOPE_MPLCONFIGDIR", raising=False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "somewhere"))

    assert nanoscope_app._restore_mpl_cache_dir() == ""
    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "somewhere")


def test_a_deliberately_exported_mplconfigdir_is_still_honoured(tmp_path, monkeypatch):
    """The repair restores the hook's EFFECTIVE choice, which respects an operator's own export."""
    sys.path.insert(0, str(PACKAGING))
    try:
        import nanoscope_app
    finally:
        sys.path.pop(0)

    mine = tmp_path / "my_own_cache"
    mine.mkdir()
    monkeypatch.setenv("NANOSCOPE_MPLCONFIGDIR", str(mine))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "pyinstaller_throwaway"))

    note = nanoscope_app._restore_mpl_cache_dir()

    assert os.environ["MPLCONFIGDIR"] == str(mine)
    assert str(mine) in note
