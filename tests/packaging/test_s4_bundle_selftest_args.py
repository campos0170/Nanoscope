"""BD4-02 — `--self-test` must not report a green run for work it never did.

Three checks used to fold "no file was given" into "the file you gave me is not there":

    if data_file is None or not data_file.exists(): return "SKIPPED (no file passed)"

``@check`` records a RETURN as a PASS, so one mistyped character, a stray flag, or a path pasted
without quotes (this owner's recordings are named "500 pM Apo ferritin_1 x PBS_22 nm_0004_process
_ -1000 mV_ 385-400 s time trace.abf") produced "[PASS] real recording end to end  SKIPPED (no
file passed)", "18/18 checks passed" and exit 0 — byte-identical to a run that loaded a
749,651-sample ABF and ran TensorFlow end to end. This is the gate a build is signed off with.

Every test here goes through the shipped entry point, `nanoscope_app.main()` with `--self-test`
on argv, which is what the frozen binary runs.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[2] / "packaging"

_LAUNCH = """
import sys, tempfile
sys.path.insert(0, {pkg!r})
sys.argv = ["NanoScope", "--self-test"] + sys.argv[1:]

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


@pytest.fixture(scope="module")
def selftest():
    """The real packaging/nanoscope_selftest.py, imported by path (it is not on sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "nanoscope_selftest_under_test", PACKAGING / "nanoscope_selftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    script = tmp_path / "launch.py"
    script.write_text(_LAUNCH.format(pkg=str(PACKAGING)))
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)  # nothing of this test may land in the owner's temp dir
    return subprocess.run([sys.executable, str(script), *args], env=env,
                          capture_output=True, text=True, timeout=300)


def test_a_mistyped_recording_path_is_named_and_stops_the_run(tmp_path):
    missing = tmp_path / "500 pM Apo ferritin_1 x PBS_22 nm_0004_proces _ -1000 mV.abf"

    out = _run(tmp_path, str(missing))

    assert out.returncode == 2, out.stdout
    assert "BAD ARGUMENTS — nothing was checked" in out.stdout
    assert f"the data file you passed does not exist: {missing}" in out.stdout
    # The lie this replaces, in both of its halves:
    assert "SKIPPED (no file passed)" not in out.stdout
    assert "checks passed" not in out.stdout


def test_a_mistyped_model_path_is_named_even_with_a_good_recording(tmp_path):
    recording = tmp_path / "recording.abf"
    recording.write_bytes(b"not read: the run stops before any check")
    missing_model = tmp_path / "student_v2.keras"

    out = _run(tmp_path, str(recording), str(missing_model))

    assert out.returncode == 2, out.stdout
    assert f"the model file you passed does not exist: {missing_model}" in out.stdout
    assert "SKIPPED (no model passed)" not in out.stdout


def test_an_unquoted_path_with_spaces_is_refused_rather_than_silently_truncated(tmp_path):
    """The shell splits it; argv[2:] used to be discarded without a word."""
    out = _run(tmp_path, "/Users/abrahamcampos/500", "pM", "Apo")

    assert out.returncode == 2, out.stdout
    assert "the data file you passed does not exist: /Users/abrahamcampos/500" in out.stdout
    assert "argument(s) after FILE.abf and MODEL.keras were ignored: ['Apo']" in out.stdout
    assert "has to be quoted" in out.stdout


def test_a_stray_flag_is_not_treated_as_a_data_file(tmp_path):
    out = _run(tmp_path, "--help")

    assert out.returncode == 2, out.stdout
    assert "unknown option '--help'" in out.stdout
    assert "checks passed" not in out.stdout


def test_paths_that_exist_and_an_empty_command_line_are_not_argument_errors(tmp_path, selftest):
    """The positive control: the refusal must be about missing paths, not about having any."""
    recording = tmp_path / "recording.abf"
    recording.write_bytes(b"")
    model = tmp_path / "student.keras"
    model.write_bytes(b"")

    assert selftest._argument_problems([]) == []
    assert selftest._argument_problems([str(recording)]) == []
    assert selftest._argument_problems([str(recording), str(model)]) == []
    # ~ is the operator's own spelling of their home directory, and validation must expand it
    # the same way _run_checks does — otherwise a perfectly good "~/trace.abf" would be refused.
    tilde = "~/.nanoscope_s4_probe_that_does_not_exist"
    assert selftest._argument_problems([tilde]) == [
        f"the data file you passed does not exist: {Path(tilde).expanduser()}"]
