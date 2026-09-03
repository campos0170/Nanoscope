"""H-4 — a failing analysis must go RED, never hang the run.

THE DEFECT. ``MainWindow._on_analysis_failed`` answers the analysis controller's ``failed``
signal with a real ``QMessageBox.exec()``. That is right in the app and fatal in the suite: the
signal is delivered inside whatever ``processEvents()`` pump the test is running, so the modal
opens with nobody to dismiss it and the nested event loop never returns. Measured (three-arm
control, ``test_project_io.py::test_a_dwell_and_depth_fit_survive_save_and_reopen``):

    unpatched                            1 passed, 2.9 s
    detection raises, modal neutralised  1 FAILED, 31.5 s, exact traceback
    detection raises, modal live         still running at 75 s, 0 % CPU, no output, SIGKILLed

and it is not curable by waiting harder — the timeout timer *does* fire under the modal and *does*
call ``quit()``; ``exec()`` cannot return while a nested loop sits above it. The cost is that the
modal EATS THE REGRESSION THAT OPENED IT: a numerical defect in detection presents as "the suite
hangs", not as "test X failed". Two prior rounds were burned exactly that way.

Reproduced here on the same shape at ``main_window.py:1328``: a detrend window longer than the
recording, driven through a real ``MainWindow`` + ``AnalysisController``. Guarded, the subprocess
finishes in ~3 s carrying the ValueError; with the guard removed it was killed at 45 s having
printed nothing past ``PROBE_START``.

THE TEST MUST NOT BE ABLE TO HANG THE SUITE ITSELF. That is why the failing analysis runs in a
SUBPROCESS under a hard timeout: if the guard ever regresses, this file goes red in
``_HANG_BUDGET_S`` seconds instead of taking the run down with it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PySide6 import QtGui, QtWidgets

_REPO_ROOT = Path(nanoscope.__file__).resolve().parent.parent
_GUI_DIR = _REPO_ROOT / "nanoscope" / "gui"

# Generous: the guarded run measured 2.9 s wall including interpreter start and MainWindow
# construction. Paid in full ONLY when the guard has regressed, and the point of the number is
# to bound how long a regression can pretend to be a slow test.
_HANG_BUDGET_S = 45.0


# ================================================================================================
# 1. The failure itself: a raising analysis reaches the user and the process stays alive.
# ================================================================================================

_PROBE = r'''
"""Drive a RAISING analysis through a real MainWindow. argv: <repo-root>"""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
root = sys.argv[1]

import nanoscope
print("NANOSCOPE_FROM", os.path.dirname(os.path.abspath(nanoscope.__file__)), flush=True)

# Import the suite's conftest BY PATH and install its guard. If the file has no guard (the
# reverted "before" arm) say so and carry on -- the run must then hang, which is the defect.
import importlib.util
spec = importlib.util.spec_from_file_location("_ns_conftest", os.path.join(root, "tests", "conftest.py"))
cft = importlib.util.module_from_spec(spec)
sys.modules["_ns_conftest"] = cft
spec.loader.exec_module(cft)
install = getattr(cft, "_install_modal_guard", None)
if install is None:
    print("GUARD_ABSENT", flush=True)
else:
    install()
    print("GUARD_INSTALLED", flush=True)

import numpy as np
from pathlib import Path
from PySide6 import QtCore
from nanoscope.gui.app import create_app

app = create_app([])
# A PRIVATE QSettings namespace: this process must not read or write either the developer's
# real layout or the suite's shared one (conftest's qapp fixture owns that).
app.setOrganizationName("NanoScopeTest")
app.setApplicationName("NanoScopeModalGuardProbe")
QtCore.QSettings().clear()

from nanoscope.analysis import register_all
from nanoscope.gui.overlays import register_all_renderers
register_all()
register_all_renderers()

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
rng = np.random.default_rng(0)
arr = np.full(100_000, 100.0) + rng.normal(0.0, 1.5, 100_000)
arr[10_000:10_300] -= 30.0
arr = np.ascontiguousarray(arr, dtype=np.float32)
info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
meta = RecordingMeta(path=Path("probe.dat"), format_name="synthetic", n_samples=arr.size,
                     sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))

w = MainWindow()
doc = TraceDocument(meta=meta, channels=(arr,))
doc_id = w._session.add_document(doc)
w._create_default_layers(doc)
w._on_selection(doc_id, 0)

seen = []
w._analysis_controller.failed.connect(seen.append)

panel = w._analysis_panel
params = dict(panel._detection.params())
params["detrend_enabled"] = True
params["detrend_window_s"] = 60.0          # far longer than the 1 s recording

print("PROBE_START", flush=True)
panel._on_detect_requested(params)

deadline = time.time() + 30.0
while not seen and time.time() < deadline:
    app.processEvents()
    time.sleep(0.005)
for _ in range(100):                       # let the queued slot -- and its modal -- run
    app.processEvents()
    time.sleep(0.005)

print("FAILED_SIGNAL", bool(seen), flush=True)
print("MESSAGE", (seen[0] if seen else "")[:200], flush=True)
for bm in getattr(cft, "_BLOCKED_MODALS", ()):
    print("MODAL", bm.kind, "|", bm.where, "|", bm.text[:200], flush=True)
print("SURVIVED", flush=True)
os._exit(0)                                # no closeEvent, so no QSettings write
'''


def test_a_raising_analysis_surfaces_as_a_failure_instead_of_hanging(tmp_path):
    script = tmp_path / "raising_analysis_probe.py"
    script.write_text(_PROBE)

    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, str(script), str(_REPO_ROOT)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(tmp_path))
    try:
        out = proc.communicate(timeout=_HANG_BUDGET_S)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        out = proc.communicate()[0]
        pytest.fail(
            f"THE HANG IS BACK: a failing analysis blocked in a modal for "
            f"{_HANG_BUDGET_S:.0f} s. Everything after PROBE_START is missing:\n{out}")
    elapsed = time.monotonic() - t0

    # The run under test must be the run we think it is (pytest's rootdir insertion has
    # silently turned a 'before' arm into an 'after' arm on this project twice).
    assert f"NANOSCOPE_FROM {_REPO_ROOT / 'nanoscope'}" in out, out
    assert "GUARD_INSTALLED" in out, f"tests/conftest.py grew no modal guard:\n{out}"

    assert "PROBE_START" in out, out
    assert "SURVIVED" in out, f"the probe died before reporting:\n{out}"
    assert "FAILED_SIGNAL True" in out, f"the analysis did not fail at all:\n{out}"

    # The failure REACHED THE USER: the box MainWindow._on_analysis_failed opens was built,
    # and it carries the real reason rather than a generic apology.
    modals = [ln for ln in out.splitlines() if ln.startswith("MODAL ")]
    assert modals, f"the failure opened no message box at all:\n{out}"
    assert any("main_window.py" in ln and "longer than the recording" in ln for ln in modals), \
        f"the failure box did not carry the reason:\n{modals}"

    # And it did so promptly. 3x headroom over the measured 2.9 s; the point is that this is
    # nowhere near the hang budget, so a future regression is unambiguous.
    assert elapsed < _HANG_BUDGET_S / 3, f"the guarded run took {elapsed:.1f} s"


# ================================================================================================
# 2. The guard is at the SEAM, not on a list of method names.
# ================================================================================================

# Qt's blocking modal entry points that do NOT route through ``QDialog.exec``: each builds and
# runs its own modal in C++. There is no seam under them, so they are patched by name -- and this
# is the list that must stay complete.
_GUARDED_STATICS = {
    "QMessageBox": {"information", "warning", "critical", "question", "about", "aboutQt"},
    "QFileDialog": {"getOpenFileName", "getOpenFileNames", "getSaveFileName",
                    "getExistingDirectory", "getOpenFileUrl", "getOpenFileUrls",
                    "getSaveFileUrl", "getExistingDirectoryUrl"},
    "QInputDialog": {"getText", "getMultiLineText", "getInt", "getDouble", "getItem"},
    "QColorDialog": {"getColor"},
    "QFontDialog": {"getFont"},
}


def _guarded(fn) -> bool:
    """True when ``fn`` is one of conftest's stubs rather than Qt's own blocking call."""
    from conftest import is_modal_guard

    return is_modal_guard(fn)


def test_one_patch_of_QDialog_exec_covers_every_dialog_class_qt_gives_us(qapp):
    """The reason the guard is not a list of names: ``exec`` lives in exactly one ``__dict__``.

    If a future Qt moves ``exec`` down into the concrete classes, this fails and the guard has
    to grow -- which is precisely the day you want to be told.
    """
    assert "exec" in QtWidgets.QDialog.__dict__
    for name in ("QMessageBox", "QFileDialog", "QInputDialog", "QColorDialog",
                 "QFontDialog", "QProgressDialog", "QWizard"):
        cls = getattr(QtWidgets, name)
        assert cls.exec is QtWidgets.QDialog.exec, (
            f"{name} carries its OWN exec; one patch of QDialog no longer covers it")
    assert _guarded(QtWidgets.QDialog.exec), "QDialog.exec is not guarded"
    assert _guarded(QtWidgets.QMenu.exec), "QMenu.exec is not guarded"
    # PySide6 keeps the PyQt-era `exec_` alias as its own C++ entry; `exec` does not cover it.
    assert _guarded(QtWidgets.QDialog.exec_), "QDialog.exec_ is not guarded"
    assert _guarded(QtWidgets.QMenu.exec_), "QMenu.exec_ is not guarded"


def test_every_static_modal_the_gui_actually_calls_is_guarded(qapp):
    """AST-enumerate ``nanoscope/gui`` and demand the guard covers what it finds.

    Adding ``QMessageBox.question(...)`` to a panel is a one-line change that reintroduces the
    hang; this test is what notices.
    """
    called: set[tuple[str, str]] = set()
    for path in sorted(_GUI_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            recv = ast.unparse(node.func.value).split(".")[-1]
            if recv in _GUARDED_STATICS:
                called.add((recv, node.func.attr))

    assert called, "found no static modal calls at all — the scan is broken, not the app"
    for cls_name, meth in sorted(called):
        assert meth in _GUARDED_STATICS[cls_name], (
            f"nanoscope/gui calls {cls_name}.{meth}, which no guard covers: it will HANG the "
            f"suite the first time a test reaches it. Add it to _install_modal_guard().")
        assert _guarded(getattr(getattr(QtWidgets, cls_name), meth)), \
            f"{cls_name}.{meth} is listed as guarded but is not patched"


def test_no_gui_class_overrides_exec_and_slips_under_the_seam(qapp):
    """A Python-level ``def exec`` in a dialog subclass would bypass ``QDialog.exec`` entirely."""
    offenders = []
    for path in sorted(_GUI_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name in ("exec", "exec_"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"these override exec and escape the guard: {offenders}"


# ================================================================================================
# 3. The guard answers like a user who cancelled — and never overrules a test that wants an answer.
# ================================================================================================

def test_a_blocked_message_box_declines_and_is_recorded(qapp, blocked_modals):
    box = QtWidgets.QMessageBox()
    box.setText("Analysis failed")
    box.setInformativeText("because of a reason")
    destroy = box.addButton("Delete everything", QtWidgets.QMessageBox.DestructiveRole)
    box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)

    assert box.exec() != QtWidgets.QDialog.Accepted
    # Every `box.exec(); box.clickedButton()` site in nanoscope/gui reads None as "cancelled",
    # which is the branch that keeps the user's data.
    assert box.clickedButton() is None
    assert box.clickedButton() is not destroy

    assert len(blocked_modals) == 1, blocked_modals
    rec = blocked_modals[0]
    assert rec.kind == "QMessageBox"
    assert "Analysis failed" in rec.text and "because of a reason" in rec.text
    assert rec.where.startswith("test_modal_guard.py:")


def test_the_blocking_statics_answer_as_a_cancel(qapp, blocked_modals):
    assert QtWidgets.QFileDialog.getSaveFileName(None, "t", "x.csv", "*")[0] == ""
    assert QtWidgets.QFileDialog.getOpenFileNames(None, "t")[0] == []
    assert QtWidgets.QFileDialog.getExistingDirectory(None, "t") == ""
    assert QtWidgets.QInputDialog.getText(None, "t", "label") == ("", False)
    assert QtWidgets.QInputDialog.getInt(None, "t", "label")[1] is False
    assert not QtWidgets.QColorDialog.getColor().isValid()
    assert QtWidgets.QMessageBox.question(None, "t", "really?") != QtWidgets.QMessageBox.Yes
    QtWidgets.QMessageBox.warning(None, "Could not write", "the disk is full")

    assert len(blocked_modals) == 8
    warn = blocked_modals[-1]
    assert warn.kind == "QMessageBox.warning"
    assert warn.title == "Could not write" and "disk is full" in warn.text


def test_a_test_that_owns_a_dialogs_answer_still_overrules_the_guard(qapp, monkeypatch):
    """The guard supplies a DEFAULT, never a verdict.

    A previous attempt at H-4 neutralised ``AnalysisPanel._show_comparison`` by name and
    silently deleted the Accepted branch of the estimator-comparison report — with the test
    that covered it. Patching the seam instead of the method keeps that branch reachable: a
    per-test patch of the concrete dialog class shadows ``QDialog.exec``.
    """
    class _Report(QtWidgets.QDialog):
        pass

    dlg = _Report()
    assert dlg.exec() == QtWidgets.QDialog.Rejected          # the guard's default

    monkeypatch.setattr(_Report, "exec", lambda self: QtWidgets.QDialog.Accepted)
    assert dlg.exec() == QtWidgets.QDialog.Accepted          # the test's answer wins
    monkeypatch.undo()
    assert dlg.exec() == QtWidgets.QDialog.Rejected          # ...and the guard is back


def test_the_estimator_comparison_method_is_left_alone(qapp):
    """Named explicitly because deleting it is the known way to get this wrong."""
    from nanoscope.gui.analysis_panel import AnalysisPanel

    fn = AnalysisPanel._show_comparison
    assert getattr(fn, "__module__", "").startswith("nanoscope"), (
        "_show_comparison has been neutralised by name; the Accepted 'Re-measure now' branch "
        "and test_baseline_migration's coverage of it go with it")


def test_the_guard_does_not_touch_the_loops_the_suite_runs_on_purpose(qapp):
    """``QApplication.exec`` is the app's own loop and ``QEventLoop.exec`` is nested on purpose
    by ``tests/gui/test_analysis_worker.py``. Neither is a modal; neither may be stubbed."""
    from PySide6 import QtCore

    assert not _guarded(QtCore.QCoreApplication.exec)
    assert not _guarded(QtCore.QEventLoop.exec)


def test_an_invalid_colour_is_what_a_cancelled_picker_returns(qapp):
    """Pins the return SHAPE of the picker stubs — a wrong shape is a TypeError deep in a
    panel, which is a worse failure than the hang it replaced."""
    assert isinstance(QtWidgets.QColorDialog.getColor(), QtGui.QColor)
    ok, font = QtWidgets.QFontDialog.getFont()
    assert ok is False and isinstance(font, QtGui.QFont)
