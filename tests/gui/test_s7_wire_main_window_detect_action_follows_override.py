"""Sweep 7, S7-CAL1 / S7-CAL2, the MainWindow half: ``Analysis ▸ Run detection`` follows a
Calibration-panel override the moment it is applied, and re-derives itself whenever the
menu opens.

Agent C2 made the AnalysisPanel's gate follow the session's EFFECTIVE channel kind (a
"treat this channel as current" override), but MainWindow synced ``_act_run_detection``
only in ``_on_selection`` / ``_on_document_closed`` — so after Apply the panel's Detect
button was live while the menu action stayed grey (repro_S7-CAL1 "Run detection False"),
and after the reverse override the button was dead while the action stayed live
(repro_S7-CAL2 §B "Run detection True"), until the next click on a file row. Two surfaces
for one gate, disagreeing. The action is now synced on ``Session.calibrationChanged`` and
on the Analysis menu's ``aboutToShow``, the way the ML action already was.

Driven through the real Calibration panel (combo, units, gain, Apply) on tmp_path-backed
synthetic recordings — the sidecar Apply writes lands beside the tmp file, never beside a
real recording.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 20_000


def _doc(path, channels) -> TraceDocument:
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    arrays = tuple(np.zeros(N, dtype=np.float32) for _ in channels)
    return TraceDocument(meta=meta, channels=arrays)


def _undeclared(tmp_path) -> TraceDocument:
    """The shape of the owner's nisin TDMS / a neutral-header CSV: nothing declared."""
    return _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)))


def _declared(tmp_path) -> TraceDocument:
    """The owner's ABF shape: IN14 voltage, IN15 current, declared by the header."""
    return _doc(tmp_path / "declared.abf", (
        ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
        ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _open(window, doc, channel):
    did = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(did, channel)
    window._calibration_panel.set_selection(did, channel)
    return did


def _apply_kind(window, qtbot, kind: ChannelKind, units: str, gain: str) -> None:
    cp = window._calibration_panel
    cp._treat_as.setCurrentIndex(cp._treat_as.findData(kind))
    cp._units.clear()
    qtbot.keyClicks(cp._units, units)
    cp._gain.clear()
    qtbot.keyClicks(cp._gain, gain)
    cp._btn_apply.click()
    qtbot.wait(50)
    assert cp.status_text().startswith("Applied"), cp.status_text()


def _analysis_menu(window) -> QtWidgets.QMenu:
    menus = [m for m in window.menuBar().findChildren(QtWidgets.QMenu)
             if m.title().replace("&", "") == "Analysis"]
    assert menus, [m.title() for m in window.menuBar().findChildren(QtWidgets.QMenu)]
    return menus[0]


def test_correcting_an_undeclared_channel_to_current_enables_the_menu_action_at_once(
        window, qtbot, tmp_path):
    did = _open(window, _undeclared(tmp_path), 0)
    qtbot.wait(20)
    panel = window._analysis_panel
    assert not panel.can_run()
    assert not window._act_run_detection.isEnabled()

    _apply_kind(window, qtbot, ChannelKind.CURRENT, "pA", "2.5")
    assert window._session.effective_kind(did, 0) is ChannelKind.CURRENT
    assert panel.can_run(), "the panel's gate still reads the header (C2's half regressed)"
    # NO _sync_detection_action() here: the wiring itself has to have done it.
    assert window._act_run_detection.isEnabled(), (
        "Analysis ▸ Run detection kept its pre-override state; the panel's Detect button "
        "and the menu action disagree about one recording")


def test_the_reverse_override_disables_the_menu_action_at_once(window, qtbot, tmp_path):
    did = _open(window, _declared(tmp_path), 1)
    qtbot.wait(20)
    panel = window._analysis_panel
    assert panel.can_run()
    assert window._act_run_detection.isEnabled()

    _apply_kind(window, qtbot, ChannelKind.VOLTAGE, "mV", "1")
    assert window._session.effective_kind(did, 1) is ChannelKind.VOLTAGE
    assert not panel.can_run()
    assert not window._act_run_detection.isEnabled(), (
        "the action stayed live after the override made the only current channel a "
        "voltage channel — Run detection offered on a recording with nothing to run on")


def test_opening_the_analysis_menu_re_derives_the_action_from_the_panel_gate(
        window, qtbot, tmp_path):
    """The belt to the calibration braces: whatever moved the gate, the menu opens true."""
    _open(window, _undeclared(tmp_path), 0)
    qtbot.wait(20)
    assert not window._analysis_panel.can_run()
    window._act_run_detection.setEnabled(True)          # a stale state, planted
    _analysis_menu(window).aboutToShow.emit()
    qtbot.wait(5)
    assert not window._act_run_detection.isEnabled()

    # ...and the other way round, so this is a re-derivation and not a one-way clamp.
    _apply_kind(window, qtbot, ChannelKind.CURRENT, "pA", "1")
    window._act_run_detection.setEnabled(False)
    _analysis_menu(window).aboutToShow.emit()
    qtbot.wait(5)
    assert window._act_run_detection.isEnabled()


def test_the_ml_action_is_synced_on_the_same_calibration_edit(window, qtbot, tmp_path):
    """Both detector actions ride the one calibration hop; the ML one must at least be
    re-derived (it stays off here — no model is loaded — but from the gate, not from
    its planted state)."""
    _open(window, _declared(tmp_path), 1)
    qtbot.wait(20)
    window._act_run_ml_detection.setEnabled(True)       # planted; no model is loaded
    _apply_kind(window, qtbot, ChannelKind.CURRENT, "pA", "1")
    assert window._act_run_ml_detection.isEnabled() == window._analysis_panel.can_run_ml()
