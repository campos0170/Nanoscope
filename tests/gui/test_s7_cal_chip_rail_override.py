"""Sweep 7: the status-bar calibration chip and the workflow strip agree with the rest of
the app about WHICH channel is the current channel, and never certify a ruined one.

The skeptics' MISSED (major) item: on a 2 s CSV of the owner's own -700 mV recording with
neutral column headers, corrected in the Calibration panel to "current" + pA + gain 2.5 and
then batch-detected (15 events stored on ch0), the chip read "No current channel" and the
strip locked CALIBRATE / FILTER / DETECT with "No current channel on this recording" — a
flatly false statement about a channel the app had just calibrated, drawn with a pA axis
and detected on. Root cause shared with S7-CAL1: ``workflow_status`` read the header
(``ChannelInfo.kind``) where the panes and Batch detect read the override.

S7-CAL4 (minor): with gain 1e38 on IN15 of the -1000 mV ABF the chip read "✓ Gain set:
×1e+38 → pA" / "reported magnitudes are physical" over a channel whose every sample was
-inf, while the File info panel one dock over said "every one of 749,651 samples is
non-finite". The chip and the strip now render that as DAMAGED (amber, the issue as the
tooltip) and CALIBRATE is not done.

The pure model is asserted through ``compute_status`` and the two real widgets are driven
with it; the last test drives the whole thing through a real MainWindow and the Calibration
panel's own combo + Apply button.
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
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.status_chip import AMBER, GREEN, NEUTRAL, CalibrationChip
from nanoscope.gui.workflow_rail import GLYPH_LOCKED, WorkflowStrip
from nanoscope.gui.workflow_status import CalibrationState, Stage, compute_status

FS = 50_000.0
N = 20_000


def _doc(path, channels, arrays=None) -> TraceDocument:
    n = N if arrays is None else int(arrays[0].size)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    if arrays is None:
        arrays = tuple(np.zeros(n, dtype=np.float32) for _ in channels)
    return TraceDocument(meta=meta, channels=tuple(arrays))


def _undeclared(tmp_path) -> TraceDocument:
    return _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)))


def _declared(tmp_path, arr=None) -> TraceDocument:
    chans = (ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
             ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT))
    if arr is None:
        return _doc(tmp_path / "declared.abf", chans)
    return _doc(tmp_path / "declared.abf", chans, (np.zeros(arr.size, np.float32), arr))


# ================================================================================================
# the pure model + the two real widgets
# ================================================================================================

def test_the_chip_and_strip_count_a_channel_corrected_to_current_as_the_current_channel(
        qapp, qtbot, tmp_path):
    s = Session()
    did = s.add_document(_undeclared(tmp_path))
    before = compute_status(s, did, 0)
    assert before.calibration.state is CalibrationState.NO_CHANNEL
    assert not before.stage(Stage.DETECT).available

    # The user's gesture, as the panel commits it: "Treat this channel as: current", pA, ×2.5.
    s.set_calibration(did, 0, ChannelCalibration(2.5, 0.0, "pA", kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    after = compute_status(s, did, 0)
    assert after.calibration.state is CalibrationState.CALIBRATED
    assert after.calibration.channel == 0
    assert after.calibration.detail == "×2.5 → pA"
    for stage in (Stage.CALIBRATE, Stage.FILTER, Stage.DETECT):
        assert after.stage(stage).available, stage
        assert "No current channel" not in after.stage(stage).reason
    assert after.stage(Stage.CALIBRATE).done
    assert after.current_stage is Stage.DETECT

    chip = CalibrationChip()
    qtbot.addWidget(chip)
    chip.set_status(after.calibration)
    assert chip.text() == "✓ Gain set: ×2.5 → pA"
    assert chip.accent() == GREEN

    strip = WorkflowStrip()
    qtbot.addWidget(strip)
    strip.resize(2400, 30)
    strip.set_status(after)
    assert strip.glyph(Stage.DETECT) != GLYPH_LOCKED
    assert "No current channel" not in strip.tooltip(Stage.DETECT)


def test_a_header_current_channel_re_declared_as_voltage_locks_the_pipeline_and_names_the_override(
        qapp, qtbot, tmp_path):
    """The reverse direction, and the locked reason must say WHY: "No current channel" on
    a recording whose header plainly declares IN15 reads as "this recording cannot be
    analysed" unless the override is named as the cause."""
    s = Session()
    did = s.add_document(_declared(tmp_path))
    assert compute_status(s, did, 1).calibration.state is CalibrationState.UNCALIBRATED

    s.set_calibration(did, 1, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.VOLTAGE),
                      write_sidecar=False)
    st = compute_status(s, did, 1)
    assert st.calibration.state is CalibrationState.NO_CHANNEL
    for stage in (Stage.CALIBRATE, Stage.FILTER, Stage.DETECT):
        assert not st.stage(stage).available, stage
        reason = st.stage(stage).reason
        assert "No current channel" in reason
        assert "IN15" in reason and "override treats it as voltage" in reason, reason

    chip = CalibrationChip()
    qtbot.addWidget(chip)
    chip.set_status(st.calibration)
    assert chip.text() == "No current channel" and chip.accent() == NEUTRAL


def test_a_gain_that_overflows_float32_renders_damaged_not_calibrated(qapp, qtbot, tmp_path):
    """Measured shape: raw reaching ±250 pA, gain 1e38 -> every calibrated sample ±inf in
    float32. ``Session.calibration_issues`` is the authority; the chip and strip follow it."""
    arr = np.linspace(-250.0, 40.0, N, dtype=np.float32)
    s = Session()
    did = s.add_document(_declared(tmp_path, arr))

    s.set_calibration(did, 1, ChannelCalibration(2.5, 0.0, "pA"), write_sidecar=False)
    fine = compute_status(s, did, 1)
    assert fine.calibration.state is CalibrationState.CALIBRATED, "a sane gain is still green"
    assert fine.stage(Stage.CALIBRATE).done

    s.set_calibration(did, 1, ChannelCalibration(1e38, 0.0, "pA"), write_sidecar=False)
    assert s.calibration_issues(did, 1), "precondition: the session reports the damage"
    st = compute_status(s, did, 1)
    assert st.calibration.state is CalibrationState.DAMAGED
    assert st.calibration.is_damaged and not st.calibration.is_calibrated
    assert st.calibration.detail == "×1e+38 → pA"
    assert "non-finite" in st.calibration.issue
    cal_stage = st.stage(Stage.CALIBRATE)
    assert cal_stage.available and not cal_stage.done
    assert "ruined the channel" in cal_stage.reason and "non-finite" in cal_stage.reason
    assert st.current_stage is Stage.CALIBRATE, "the remedy is the current step"

    chip = CalibrationChip()
    qtbot.addWidget(chip)
    chip.set_status(st.calibration)
    assert chip.accent() == AMBER
    assert chip.text().startswith("⚠ Gain set: ×1e+38 → pA")
    assert "not physical" in chip.text()
    assert "non-finite" in chip.toolTip() and "Click to open the Calibration panel" in chip.toolTip()
    assert "magnitudes are physical" not in chip.toolTip()


# ================================================================================================
# through the real window: the panel's own combo + Apply, then the chip and the strip
# ================================================================================================

@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def test_the_real_chip_and_strip_follow_an_override_committed_through_the_calibration_panel(
        window, qtbot, tmp_path):
    doc = _undeclared(tmp_path)
    did = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(did, 0)
    window._calibration_panel.set_selection(did, 0)
    qtbot.wait(20)
    assert window._calibration_chip.text() == "No current channel"
    assert window._workflow_status.stage(Stage.DETECT).available is False

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

    assert window._calibration_chip.text() == "✓ Gain set: ×2.5 → pA"
    assert window._calibration_chip.accent() == GREEN
    status = window._workflow_status
    assert status.calibration.channel == 0
    for stage in (Stage.CALIBRATE, Stage.FILTER, Stage.DETECT):
        assert status.stage(stage).available, stage
    assert window._workflow_strip.glyph(Stage.DETECT) != GLYPH_LOCKED
    assert "No current channel" not in window._workflow_strip.tooltip(Stage.DETECT)
