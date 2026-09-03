"""LAY-W3b — the SECOND gate banner in the Detection panel owns its row too.

The panel-wide Schmitt gate (``_gate_host``) was given a reserved row last round. Its
sibling one page along — ``ml_gate_label``, same panel, same amber, same
show-and-hide-with-state — was not, so the anti-pattern survived one widget along, and the
reachability round escalated it by putting live controls directly underneath it.

Measured on a real ``MainWindow`` at 1512x900 on a fresh profile, ML detector page,
BEFORE this fix (content y of ``ml_preset_save_btn``, one of the preset verbs below the
banner; ``ml_preset_delete_btn`` moves identically):

    gate up, one line   ("Load a .keras model to run ML detection.")        y=475..500
    gate up, two lines  ("Detection runs on CURRENT channels; this is a …") y=490..515
    gate hidden         (TensorFlow present + a model loaded)               y=454..479

— i.e. the two refusals sit 15 px apart and hiding the banner moves the row 21 px the
other way, a 36 px range on a 25 px button, under whatever pointer was aiming at it.

WHAT WALKS THE BANNER BETWEEN THOSE THREE, measured rather than assumed: loading or
clearing a model, TensorFlow's presence, a recording with no live current channel or no
known sample rate, and a Calibration-panel "treat this channel as" override on the target
(driven on the apo ABF, marking IN15 voltage produces exactly the long refusal below, and
clearing it returns the one-line gate). NOT selecting a different channel: ``set_selection``
sticks to the recording's current channel through ``Session.resolve_current_channel``, so
clicking IN13 or IN14 on that recording leaves the target on IN15 and the gate unchanged.
Whichever door is used, the user is then reaching for the preset row underneath it.

The remedy is the one the sibling banner already uses, and nothing else changes: a
``_FixedLinesNote(2)`` inside a fixed-height host. The note keeps its own show/hide, so
``ml_gate_label.isHidden()`` still answers "is the gate up?" (five test modules ask), and
``text()`` still returns the FULL reason even when the paint is elided.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import ml_detect
from nanoscope.gui.main_window import _INSPECTOR_DEFAULT_WIDTH, MainWindow
from nanoscope.gui.panels import DetectionPanel

#: The real wrong-channel refusal ``AnalysisPanel`` pushes when the TARGET channel is not
#: a current channel — in practice a Calibration-panel override, since a plain selection
#: resolves back to the current channel — two lines at the shipped width, against one for
#: the "load a model" gate.
_LONG_REASON = ("Detection runs on CURRENT channels; this is a voltage channel "
                "(your Calibration-panel override; the file says current).")

#: Controls that live BELOW the banner and only became reachable last round.
_BELOW = ("ml_preset", "ml_preset_save_btn", "ml_preset_delete_btn")


def _settle(root: QtWidgets.QWidget, rounds: int = 3) -> None:
    for _ in range(rounds):
        QtWidgets.QApplication.processEvents()
        for w in root.findChildren(QtWidgets.QWidget):
            lay = w.layout()
            if lay is not None:
                lay.activate()
        if root.layout() is not None:
            root.layout().activate()
        QtWidgets.QApplication.processEvents()


def _ml_page(panel: DetectionPanel) -> None:
    tabs = panel.findChild(QtWidgets.QTabWidget)
    for i in range(tabs.count()):
        if tabs.tabText(i) == "ML detector":
            tabs.setCurrentIndex(i)
            return
    raise AssertionError("no ML detector page")


def _geometry(root: QtWidgets.QWidget, panel: DetectionPanel) -> dict[str, tuple[int, int]]:
    _settle(root)
    out = {}
    for name in _BELOW:
        w = getattr(panel, name)
        y = w.mapTo(root, QtCore.QPoint(0, 0)).y()
        out[name] = (y, y + w.height())
    return out


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    p = DetectionPanel()
    qtbot.addWidget(p)
    p.resize(_INSPECTOR_DEFAULT_WIDTH - 40, 900)
    p.show()
    _settle(p)
    _ml_page(p)
    _settle(p)
    return p


def test_the_ml_gate_moves_nothing_when_it_speaks_differently(panel):
    """A one-line refusal and a two-line refusal reserve the SAME row.

    The two states the gate speaks in: runnable target with no model -> "Load a .keras
    model to run ML detection."; a target that is not a current channel -> the
    122-character wrong-channel refusal. ``set_ml_runnable`` is the door
    ``AnalysisPanel._regate`` pushes them through, whatever moved the gate upstream.
    """
    panel.set_ml_runnable(True, "")
    short = _geometry(panel, panel)
    panel.set_ml_runnable(False, _LONG_REASON)
    long_ = _geometry(panel, panel)
    moved = {k: (short[k], long_[k]) for k in short if short[k] != long_[k]}
    assert moved == {}, f"a longer refusal moved the controls below it: {moved}"


def test_the_ml_gate_moves_nothing_when_it_appears_or_goes(panel):
    """And the same for up vs down — the 36 px jump loading a model used to make."""
    panel.set_ml_runnable(True, "")
    gated = _geometry(panel, panel)
    assert not panel.ml_gate_label.isHidden(), "precondition: the gate is up"
    panel.set_model_path("/x/model.keras")
    _settle(panel)
    assert panel.ml_gate_label.isHidden(), "precondition: the gate went down"
    ungated = _geometry(panel, panel)
    moved = {k: (gated[k], ungated[k]) for k in gated if gated[k] != ungated[k]}
    assert moved == {}, f"hiding the gate banner moved: {moved}"


def test_the_reserved_row_keeps_the_gate_readable(panel):
    """Reserving the row may not cost the reason.

    ``text()`` is the FULL refusal whatever fitted on screen (the ML tests read it), the
    whole of it is in the tooltip, and an empty gate is still HIDDEN — an always-visible
    blank banner would answer "is the gate up?" wrongly.
    """
    panel.set_ml_runnable(False, _LONG_REASON)
    _settle(panel)
    assert panel.ml_gate_label.text() == _LONG_REASON
    assert not panel.ml_gate_label.isHidden()
    panel.set_ml_runnable(True, "")
    panel.set_model_path("/x/model.keras")
    _settle(panel)
    assert panel.ml_gate_label.isHidden() and panel.ml_gate_label.text() == ""


def test_in_the_real_window_the_ml_gate_moves_nothing(qtbot, monkeypatch):
    """The same diff of rectangles on a real ``MainWindow`` at the owner's 1512x900."""
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    org, app_name = QtWidgets.QApplication.organizationName(), QtWidgets.QApplication.applicationName()
    QtWidgets.QApplication.setOrganizationName("NanoScopeTest")
    QtWidgets.QApplication.setApplicationName("WireLayW3bMlGate")
    QtCore.QSettings().clear()
    try:
        win = MainWindow()
        qtbot.addWidget(win)
        win.resize(1512, 900)
        win.show()
        _settle(win)
        tabs = win._inspector.widget()
        for i in range(tabs.count()):
            if tabs.tabText(i).startswith("Analysis"):
                tabs.setCurrentIndex(i)
        _settle(win)
        from nanoscope.gui.analysis_panel import AnalysisPanel
        det = win.findChild(AnalysisPanel)._detection
        _ml_page(det)
        _settle(win)

        det.set_ml_runnable(True, "")            # current channel, no model
        before = _geometry(win, det)
        det.set_ml_runnable(False, _LONG_REASON)  # the target stops being a current channel
        after = _geometry(win, det)
        det.set_ml_runnable(True, "")
        det.set_model_path("/x/model.keras")      # ...and then loads a model
        gone = _geometry(win, det)
        moved = {k: (before[k], after[k], gone[k]) for k in before
                 if not (before[k] == after[k] == gone[k])}
        assert moved == {}, f"the ML gate banner moved the controls below it: {moved}"
    finally:
        QtCore.QSettings().clear()
        QtWidgets.QApplication.setOrganizationName(org)
        QtWidgets.QApplication.setApplicationName(app_name)
