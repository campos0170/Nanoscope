"""Regression tests for DET-01: Analysis ▸ Load ML model… rides the TensorFlow gate.

With TensorFlow absent the DetectionPanel disables its Load-model button with an
actionable reason (test_ml_detection.py::test_run_ml_disabled_when_tensorflow_absent:
"neither loading nor running is offered") — but the menu action for the SAME operation
stayed enabled, opened the file dialog and accepted a model that can never run, with
no TensorFlow message anywhere in that flow. Two surfaces for one operation
disagreed, and the menu path left the user believing the model was usable.

The fix gates the action in ``_sync_ml_detection_action`` — re-derived on the
Analysis menu's own ``aboutToShow``, exactly where Run-ML and batch-detect already
re-gate — with the panel gate label's sentence as the tooltip, and makes
``_load_ml_model`` refuse with that same reason if it ever fires while stale.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import ml_detect
from nanoscope.gui.main_window import MainWindow


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _resync_via_menu(window) -> None:
    """Drive the re-gate the way opening the menu does: its own aboutToShow."""
    window._analysis_menu.aboutToShow.emit()


# ================================================================================================
# TensorFlow absent — the menu and the panel agree, down to the reason
# ================================================================================================

def test_load_ml_model_action_disabled_without_tensorflow(qapp, qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", False)
    w = MainWindow()
    qtbot.addWidget(w)

    _resync_via_menu(w)

    # The two surfaces for the same operation may never disagree about reality.
    assert not w._analysis_panel._detection.ml_load_button.isEnabled()
    assert not w._act_load_ml_model.isEnabled()
    # Disabled WITH its reason — the panel gate label's own sentence.
    assert "TensorFlow" in w._act_load_ml_model.toolTip()
    w._analysis_controller.abort()


def test_disabled_action_never_opens_the_dialog_or_accepts_a_model(
        window, monkeypatch, tmp_path):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", False)
    _resync_via_menu(window)

    opened: list[int] = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (opened.append(1), (str(tmp_path / "m.keras"), ""))[1]))

    window._act_load_ml_model.trigger()   # a disabled QAction must not activate

    assert opened == []
    assert window._analysis_panel._detection.model_path() is None


def test_stale_trigger_refuses_with_the_reason_not_silently(
        window, monkeypatch, tmp_path):
    """A trigger that beats the first aboutToShow resync (the action starts life
    enabled) must refuse with the gate's sentence — never accept a model that can
    never run. This is the slot itself, bypassing the disabled-action guard."""
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", False)
    opened: list[int] = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (opened.append(1), (str(tmp_path / "m.keras"), ""))[1]))

    window._load_ml_model()

    assert opened == [], "no file dialog while TensorFlow is absent"
    assert window._analysis_panel._detection.model_path() is None
    assert "TensorFlow" in window.statusBar().currentMessage()
    # ...and the refusal re-derived the action, so the stale enablement is gone too.
    assert not window._act_load_ml_model.isEnabled()


# ================================================================================================
# TensorFlow present — the working flow is untouched
# ================================================================================================

def test_load_ml_model_action_enabled_and_delegates_with_tensorflow(
        window, monkeypatch, tmp_path):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    _resync_via_menu(window)

    assert window._act_load_ml_model.isEnabled()
    assert "TensorFlow" not in window._act_load_ml_model.toolTip()

    model = tmp_path / "student.keras"
    model.write_bytes(b"\x00not-a-real-model")
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(model), "Keras model (*.keras)")))

    window._act_load_ml_model.trigger()

    det = window._analysis_panel._detection
    assert det.model_path() == str(model)
    assert det.ml_model_label.text() == "student.keras"
