"""Sweep 7, S7-CAL1 (major), the AnalysisPanel half: the Detect gate resolves the channel's
kind through the SESSION -- the user's Calibration-panel override, else the header -- so the
Detect button, the Analysis menu and Batch detect give ONE answer about one channel.

Measured on the pre-fix tree (docs/sweep7/repros/repro_S7-CAL1.py, a 2 s CSV of the owner's
own -700 mV recording with neutral column headers): after "Treat this channel as: current"
+ pA, Run detection stayed disabled with "Detection runs on CURRENT channels; this is a
unknown channel." -- quoting the header the user had just corrected -- while Batch detect
one menu item over ran on that exact channel and stored 15 events. The reverse override
(repro_S7-CAL2.py section B: IN15, header current, marked VOLTAGE on the -1000 mV ABF) left
Run detection ENABLED on a channel the user had just called a command voltage.

The fix: ``AnalysisPanel._resolve_analysis_channel(doc_id, channel)`` delegates to
``Session.resolve_current_channel`` and every gate goes through ``_gate_for`` (the session's
``effective_kind``); ``_gate`` names the override when it is the reason; a calibration edit
re-gates the target it lands on; the title says what the channel is TREATED AS. Every test
drives a real MainWindow -- the override is committed through the Calibration panel's own
combo + Apply where the shipped gesture is what is under test.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import AnalysisPanel
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


# ================================================================================================
# the finding: an undeclared channel corrected to current
# ================================================================================================

def test_the_detect_gate_honours_a_channel_corrected_to_current_in_the_calibration_panel(
        window, qtbot, tmp_path):
    did = _open(window, _undeclared(tmp_path), 0)
    panel: AnalysisPanel = window._analysis_panel
    det = panel._detection
    qtbot.wait(20)
    # Before the correction the refusal is true and reads the header.
    assert not panel.can_run()
    assert not det.detect_button.isEnabled()
    assert "unknown channel" in det._gate_label.text()
    assert not window._act_run_detection.isEnabled()

    # THE USER'S GESTURE, through the real panel: Treat this channel as current, pA, ×2.5.
    cp = window._calibration_panel
    cp._treat_as.setCurrentIndex(cp._treat_as.findData(ChannelKind.CURRENT))
    cp._units.clear()
    qtbot.keyClicks(cp._units, "pA")
    cp._gain.clear()
    qtbot.keyClicks(cp._gain, "2.5")
    cp._btn_apply.click()
    qtbot.wait(50)
    assert cp.status_text().startswith("Applied")
    assert window._session.effective_kind(did, 0) is ChannelKind.CURRENT

    # The panel's gate, its Detect button and the menu action all follow the session.
    assert panel.can_run(), "the gate still reads the header the user just corrected"
    assert det.detect_button.isEnabled()
    assert det._gate_label.text() == ""
    window._sync_detection_action()
    assert window._act_run_detection.isEnabled()
    # ...and Batch detect, which honoured the override all along, still agrees.
    assert [t[1] for t in window._detectable_targets()] == [0]


def test_the_title_says_what_the_channel_is_treated_as_and_what_the_file_says(
        window, qtbot, tmp_path):
    did = _open(window, _undeclared(tmp_path), 0)
    panel: AnalysisPanel = window._analysis_panel
    assert "unknown" in panel._title.text()
    window._session.set_calibration(
        did, 0, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.CURRENT),
        write_sidecar=False)
    qtbot.wait(20)
    assert "current (file says unknown)" in panel._title.text(), panel._title.text()
    # Removing the override puts the header's word back, unqualified.
    window._session.set_calibration(did, 0, ChannelCalibration(1.0, 0.0, "pA"),
                                    write_sidecar=False)
    qtbot.wait(20)
    assert "(file says" not in panel._title.text()
    assert "unknown" in panel._title.text()


def test_the_panel_follows_the_resolver_when_an_override_makes_another_channel_current(
        window, qtbot, tmp_path):
    """The user clicked ch1 of an undeclared recording (nothing to resolve to, so the panel
    kept ch1) and then corrected ch0 to current. The pipeline's channel is now ch0 --
    the workflow strip and Batch detect say so -- and the panel must be pointed there
    too, exactly as MainWindow's own push would point it."""
    did = _open(window, _undeclared(tmp_path), 1)
    panel: AnalysisPanel = window._analysis_panel
    assert panel._sel == (did, 1)
    window._session.set_calibration(
        did, 0, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.CURRENT),
        write_sidecar=False)
    qtbot.wait(20)
    assert panel._sel == (did, 0), panel._sel
    assert panel.can_run()
    assert [t[1] for t in window._detectable_targets()] == [0]


# ================================================================================================
# the reverse direction: a declared current channel the user calls a voltage
# ================================================================================================

def test_the_reverse_override_disables_detection_and_the_refusal_names_the_override(
        window, qtbot, tmp_path):
    did = _open(window, _declared(tmp_path), 1)
    panel: AnalysisPanel = window._analysis_panel
    det = panel._detection
    qtbot.wait(20)
    assert panel.can_run() and det.detect_button.isEnabled()

    window._session.set_calibration(
        did, 1, ChannelCalibration(1.0, 0.0, "pA", Provenance.MEASURED,
                                   "this is really the command channel",
                                   kind=ChannelKind.VOLTAGE),
        write_sidecar=False)
    qtbot.wait(20)
    assert window._session.effective_kind(did, 1) is ChannelKind.VOLTAGE
    assert not panel.can_run(), "Run detection would go ahead on a declared command voltage"
    assert not det.detect_button.isEnabled()
    reason = det._gate_label.text()
    assert "voltage channel" in reason
    assert "your Calibration-panel override" in reason, reason
    assert "the file says current" in reason, reason
    # The baseline probe rides the same gate and says the same thing.
    assert panel.baseline_panel._blocker("probe")
    window._sync_detection_action()
    assert not window._act_run_detection.isEnabled()
    # ...and nothing starts if a caller bypasses the buttons.
    assert panel._run_detection(panel.detection_params()) is False
    assert panel._pending is None
    assert not list(window._session.results.for_doc(did))


# ================================================================================================
# the wording, pinned
# ================================================================================================

def test_a_refusal_with_no_override_reads_as_it_always_did():
    """CONTROL (passes before and after): the header-only sentence is unchanged, so no
    existing surface that quotes it moved."""
    ok, why = AnalysisPanel._gate(ChannelKind.VOLTAGE, FS)
    assert not ok
    assert why == "Detection runs on CURRENT channels; this is a voltage channel."


def test_a_refusal_caused_by_the_override_says_so_and_a_matching_header_does_not():
    ok, why = AnalysisPanel._gate(ChannelKind.VOLTAGE, FS, header_kind=ChannelKind.CURRENT)
    assert not ok
    assert why == ("Detection runs on CURRENT channels; this is a voltage channel "
                   "(your Calibration-panel override; the file says current).")
    # An override that AGREES with the header is not worth a parenthesis.
    assert AnalysisPanel._gate(ChannelKind.VOLTAGE, FS, header_kind=ChannelKind.VOLTAGE) == (
        False, "Detection runs on CURRENT channels; this is a voltage channel.")
    # A current channel passes whatever the header said (the CAL1 direction).
    assert AnalysisPanel._gate(ChannelKind.CURRENT, FS, header_kind=ChannelKind.UNKNOWN) == (
        True, "")
