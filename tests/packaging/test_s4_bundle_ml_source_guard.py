"""BD4-06 — the self-test's ML guard may only accuse real signal when real signal was used.

The check prefers a slice of the owner's own recording and falls back to the bundle's synthetic
trace when the reader refuses the file ("fall back rather than fail the ML check"). Its zero-event
guard was armed by ``source != "synthetic"`` — but the fallback sets

    source = f"synthetic (real slice unavailable: {type(exc).__name__}: {exc})"

which is not equal to ``"synthetic"``. So a refused recording (the owner's CPYA-variant ABF is the
real-world instance) plus a model that finds nothing ended the run with

    the ML detector found NOTHING in synthetic (real slice unavailable: UnsupportedFormatError:
    ...) — TensorFlow ran but the model produced no events on real signal

which is false about the run and points the operator at their recording instead of at the reader
refusal that actually happened.

Both tests run the REAL check — real AnalysisController, real worker thread, real HDF5 reader,
real MLDetect module and decode pipeline — in a subprocess. The one thing standing in is the
Keras model's weights: `load_keras_model` returns a model whose probability track is all zeros,
i.e. a detector that finds nothing. That is the condition being tested and cannot be shipped as
a file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PACKAGING = Path(__file__).resolve().parents[2] / "packaging"

_DRIVER = """
import importlib.util, json, os, sys, tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLBACKEND", "Agg")

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

import numpy as np
from nanoscope.analysis import ml_detect


class _FindsNothing:
    def predict(self, x, *a, **kw):
        x = np.asarray(x)
        return np.zeros((x.shape[0], x.shape[1], 1), dtype=np.float32)

    def __call__(self, x, *a, **kw):
        return self.predict(x)


def _load(path):
    p = os.path.abspath(os.fspath(path))
    return ml_detect.KerasEventModel(model=_FindsNothing(), window=2048, path=p,
                                     mtime=os.path.getmtime(p), channels=1)


ml_detect.load_keras_model = _load
ml_detect.TENSORFLOW_AVAILABLE = True

spec = importlib.util.spec_from_file_location("nanoscope_selftest", {selftest!r})
selftest = importlib.util.module_from_spec(spec)
sys.modules["nanoscope_selftest"] = selftest
spec.loader.exec_module(selftest)

tmp = Path(tempfile.mkdtemp(prefix="nanoscope_selftest_"))
try:
    selftest._run_checks([sys.argv[1], sys.argv[2]], tmp)
finally:
    selftest._remove_scratch(tmp)

hit = [r for r in selftest._results if r[0] == "ML detection through the GUI controller"]
name, ok, detail = hit[0]
print("RESULT " + json.dumps({{"ok": ok, "detail": detail}}))
"""


def _drive(tmp_path: Path, data_file: Path) -> dict:
    driver = tmp_path / "drive_ml_check.py"
    driver.write_text(_DRIVER.format(selftest=str(PACKAGING / "nanoscope_selftest.py")))
    model = tmp_path / "student.keras"
    model.write_bytes(b"weights are stubbed by the driver; only the path has to exist")
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)
    out = subprocess.run([sys.executable, str(driver), str(data_file), str(model)], env=env,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, out.stdout[-3000:]
    return json.loads(line[0][len("RESULT "):])


def _unreadable_recording(tmp_path: Path) -> Path:
    """A file the readers refuse — the shape of the owner's CPYA-variant ABF, without needing it."""
    path = tmp_path / "refused_by_the_reader.abf"
    path.write_bytes(b"NOPE" + os.urandom(4096))
    return path


def _real_recording(tmp_path: Path) -> Path:
    """A recording the app really reads, long enough for the check to slice it (>= 20 000)."""
    import h5py
    import numpy as np

    path = tmp_path / "recording.h5"
    n, fs = 200_000, 50_000.0
    rng = np.random.default_rng(11)
    sig = rng.normal(1000.0, 8.0, n)
    sig[50_000:50_400] -= 220.0
    with h5py.File(path, "w") as fh:
        fh.create_dataset("time", data=np.arange(n, dtype=np.float64) / fs)
        fh.create_dataset("current", data=sig)
        fh.create_dataset("voltage", data=np.full(n, -200.0))
    return path


def test_a_refused_recording_is_not_reported_as_a_failure_of_real_signal(tmp_path):
    result = _drive(tmp_path, _unreadable_recording(tmp_path))

    assert result["ok"] is True, result["detail"]
    assert "no events on real signal" not in result["detail"]
    # And the reason the recording was not used is still on the report line, not swallowed.
    assert "synthetic (real slice unavailable:" in result["detail"]
    assert "0 event(s)" in result["detail"]


def test_the_guard_still_fires_when_the_model_finds_nothing_in_real_signal(tmp_path):
    """The positive control: standing the guard down on the fallback must not disarm it."""
    result = _drive(tmp_path, _real_recording(tmp_path))

    assert result["ok"] is False, result["detail"]
    assert "produced no events on real signal" in result["detail"]
    assert "real samples from recording.h5" in result["detail"]
