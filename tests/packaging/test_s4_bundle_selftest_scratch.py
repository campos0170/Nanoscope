"""BD4-03 — `--self-test` must not abandon its scratch directory in the user's temp dir.

``run()`` opened ``tempfile.mkdtemp(prefix="nanoscope_selftest_")`` and nothing ever removed it:
populations.h5 (14.4 MB), synthetic.h5 (4.8 MB), five PNGs, a PDF, a parquet and two CSVs —
19.5 MB per invocation, growing linearly forever. This machine's TMPDIR had accumulated 92 such
directories (1.66 GB), the oldest six days old, and on Windows (build_app.ps1 runs `--self-test`
on every build, %TEMP% has no reaper) the growth is unbounded. Every other scratch user in the
codebase already cleans up after itself — ml_detect.py's ``finally: shutil.rmtree``,
make_icon.py, build_app.sh — so this module was the outlier.

The first test drives the whole shipped command (`NanoScope --self-test`, via the real
``nanoscope_app.main``) with TMPDIR redirected into the test's own directory. It costs ~15 s
because it really runs the checks; nothing cheaper can prove the temp dir is empty afterwards.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[2] / "packaging"

_LAUNCH = """
import sys, tempfile
sys.path.insert(0, {pkg!r})
sys.argv = ["NanoScope", "--self-test"]

# THE OWNER'S REAL PREFERENCES ARE NOT THIS TEST'S TO WRITE. The self-test builds a real
# MainWindow under the shipped application name, and MainWindow stores recent files / window
# state through QSettings; without this the run appends test paths to
# ~/Library/Preferences/com.nanoscope-local.NanoScope.plist. Same redirect tests/conftest.py
# installs for the in-process suite: move the STORE, before any QApplication exists.
import nanoscope  # noqa: F401 - bootstrap must precede any Qt import
from PySide6 import QtCore
QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                         tempfile.mkdtemp(prefix="s4-qsettings-"))

import nanoscope_app
raise SystemExit(nanoscope_app.main())
"""


@pytest.fixture()
def selftest():
    """The real packaging/nanoscope_selftest.py, imported by path (it is not on sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "nanoscope_selftest_scratch", PACKAGING / "nanoscope_selftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_real_self_test_run_leaves_nothing_behind_and_says_what_it_wrote(tmp_path):
    script = tmp_path / "launch.py"
    script.write_text(_LAUNCH.format(pkg=str(PACKAGING)))
    scratch = tmp_path / "tmpdir"
    scratch.mkdir()
    env = dict(os.environ)
    env["TMPDIR"] = str(scratch)

    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                         text=True, timeout=900)

    # This is a dev tree, not a bundle, so some checks legitimately fail (frozen bundle,
    # TensorFlow); what matters here is the housekeeping around them.
    assert "checks passed" in out.stdout, out.stdout[-3000:]
    assert "SKIPPED (no file passed)" in out.stdout  # a bare run still skips honestly
    leftovers = sorted(p.name for p in scratch.iterdir()
                       if p.name.startswith("nanoscope_selftest_"))
    assert leftovers == [], f"the self-test abandoned {leftovers} in TMPDIR"
    # Removed, and said so: the size is the receipt for what was thrown away.
    scratch_line = [ln for ln in out.stdout.splitlines() if ln.startswith("scratch:")]
    assert len(scratch_line) == 1, out.stdout[-3000:]
    assert "nanoscope_selftest_" in scratch_line[0] and "— removed" in scratch_line[0]
    assert "MB in" in scratch_line[0]


def test_the_scratch_directory_is_removed_even_when_the_run_blows_up(tmp_path, selftest,
                                                                    monkeypatch, capsys):
    """A crash inside the checks must not turn into 19.5 MB of litter.

    ``run()`` owns the directory and hands it to ``_run_checks``; the cleanup is in a ``finally``,
    so it survives the one path @check cannot catch (a BaseException) as well as a plain return.
    """
    seen: dict[str, Path] = {}

    def _explode(argv, tmp):
        seen["tmp"] = tmp
        (tmp / "populations.h5").write_bytes(b"x" * 4096)
        raise KeyboardInterrupt("operator hit ctrl-C mid-run")

    monkeypatch.setattr(selftest, "_run_checks", _explode)
    # tempfile caches its directory on first use, so setting TMPDIR here would be ignored and the
    # test would litter the owner's real temp dir with what it is asserting about.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(KeyboardInterrupt):
        selftest.run([])

    assert seen["tmp"].parent == tmp_path
    assert not seen["tmp"].exists(), f"{seen['tmp']} survived the crash"
    assert "1 file(s), 0.0 MB" in capsys.readouterr().out


def test_a_refused_command_line_creates_no_scratch_directory_at_all(tmp_path, selftest,
                                                                    monkeypatch):
    """The argument refusal (BD4-02) happens before anything is created."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    assert selftest.run([str(tmp_path / "nope.abf")]) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == []
