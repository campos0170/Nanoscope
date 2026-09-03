"""Sweep-7 fix-up: when the user's OWN "Treat this channel as" override is why a recording
has no current channel, every guided surface says so -- the rail's three locked reasons
AND the sentence the calibration chip carries -- on undeclared recordings as well as on
ones whose header declared the channel.

Two defects the skeptics found in the wave-1 fix (``_overridden_away_note``):

* X8 -- the note only listed channels the FILE declared current. On the undeclared CSV
  shape the override exists for (2 s of the owner's -700 mV recording with neutral column
  headers), corrected to current, detected on (15 events) and then set back to "Treat this
  channel as: voltage", the Detect gate read "this is a voltage channel (your
  Calibration-panel override; the file says unknown)" while the rail read "No current
  channel on this recording to detect on." and the chip "This recording has no live
  current channel to calibrate." -- the user's own override, the only reason the channel
  the app detected on a minute ago is gone, named nowhere on the two guided surfaces.
* D3 -- after the reverse override on IN15 of the -1000 mV ABF the strip's CALIBRATE
  tooltip named the override but the chip one glance away still read the fixed "This
  recording has no live current channel to calibrate.": the reason was derived once and
  threaded to three of its four consumers. ``CalibrationStatus.issue`` now carries the
  same sentence as the CALIBRATE stage, so the chip can show it verbatim (the chip's own
  consumer change lives in ``status_chip.py``).

Everything here is the pure model plus the two real widgets; no MainWindow, no sidecar.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.status_chip import NEUTRAL, CalibrationChip
from nanoscope.gui.workflow_rail import WorkflowStrip
from nanoscope.gui.workflow_status import CalibrationState, Stage, compute_status

FS = 50_000.0
N = 20_000
LOCKED = (Stage.CALIBRATE, Stage.FILTER, Stage.DETECT)


def _doc(path, channels) -> TraceDocument:
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    return TraceDocument(meta=meta,
                         channels=tuple(np.zeros(N, dtype=np.float32) for _ in channels))


def _undeclared(tmp_path) -> TraceDocument:
    """The CAL fixture's shape: ``time,Untitled,Untitled 1`` -- both channels ``unknown``."""
    return _doc(tmp_path / "undeclared_700mV.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)))


def _declared(tmp_path) -> TraceDocument:
    """The owner's ABF shape: IN14 voltage, IN15 current, both declared by the header."""
    return _doc(tmp_path / "declared.abf", (
        ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
        ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)))


def _treat_as(session: Session, did: str, ch: int, kind: ChannelKind, units: str) -> None:
    """What the Calibration panel's combo + Apply commits (never a sidecar here)."""
    session.set_calibration(did, ch, ChannelCalibration(2.5, 0.0, units, kind=kind),
                            write_sidecar=False)


# ================================================================================================
# X8 -- the undeclared recording set back to voltage: the rail names the override
# ================================================================================================

def test_an_undeclared_channel_set_back_to_voltage_locks_the_rail_with_the_override_named(
        qapp, qtbot, tmp_path):
    s = Session()
    did = s.add_document(_undeclared(tmp_path))
    _treat_as(s, did, 0, ChannelKind.CURRENT, "pA")
    assert compute_status(s, did, 0).calibration.state is CalibrationState.CALIBRATED

    _treat_as(s, did, 0, ChannelKind.VOLTAGE, "mV")           # the user changes their mind
    st = compute_status(s, did, 0)
    assert st.calibration.state is CalibrationState.NO_CHANNEL
    for stage in LOCKED:
        reason = st.stage(stage).reason
        assert not st.stage(stage).available, stage
        assert "No current channel on this recording" in reason, reason
        assert "Calibration-panel override" in reason, reason
        assert "Untitled" in reason and "voltage" in reason, reason
        assert "the file says Untitled is unknown" in reason, reason
    assert st.stage(Stage.DETECT).reason == (
        "No current channel on this recording to detect on (the file says Untitled is "
        "unknown and your Calibration-panel override treats it as voltage).")

    strip = WorkflowStrip()
    qtbot.addWidget(strip)
    strip.resize(2400, 30)
    strip.set_status(st)
    for stage in LOCKED:
        assert "Calibration-panel override" in strip.tooltip(stage), strip.tooltip(stage)


# ================================================================================================
# D3 -- ONE sentence for the strip and the chip
# ================================================================================================

def test_the_no_channel_status_carries_the_same_override_sentence_as_the_calibrate_stage(
        qapp, tmp_path):
    """The chip's tooltip and the strip's CALIBRATE tooltip must read alike: the reason is
    derived once and the status carries it to BOTH consumers. The header-current wording
    is pinned byte-for-byte because the wave-1 test and the strip already show it."""
    s = Session()
    did = s.add_document(_declared(tmp_path))
    _treat_as(s, did, 1, ChannelKind.VOLTAGE, "mV")
    st = compute_status(s, did, 1)
    assert st.calibration.state is CalibrationState.NO_CHANNEL
    expected = ("No current channel on this recording to calibrate (IN15 is declared current "
                "by the file but your Calibration-panel override treats it as voltage).")
    assert st.stage(Stage.CALIBRATE).reason == expected
    assert st.calibration.issue == expected

    s2 = Session()
    did2 = s2.add_document(_undeclared(tmp_path))
    _treat_as(s2, did2, 0, ChannelKind.VOLTAGE, "mV")
    st2 = compute_status(s2, did2, 0)
    assert st2.calibration.state is CalibrationState.NO_CHANNEL
    assert st2.calibration.issue == st2.stage(Stage.CALIBRATE).reason
    assert "Untitled" in st2.calibration.issue and "override" in st2.calibration.issue


def test_a_recording_that_simply_has_no_current_channel_carries_no_override_sentence(
        qapp, tmp_path):
    """When no override is involved the chip's fixed tooltip is exact, so ``issue`` stays
    empty and the locked reasons carry no parenthetical -- a voltage-only recording, a
    dead current channel, and a re-declaration between two non-current kinds (which is
    not why the recording has no current channel, so it is not named as one)."""
    s = Session()
    only_voltage = s.add_document(_doc(tmp_path / "v.abf", (
        ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),)))
    dead = s.add_document(_doc(tmp_path / "dead.abf", (
        ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT,
                    is_dead=True),)))
    _treat_as(s, only_voltage, 0, ChannelKind.CONDUCTANCE, "nS")
    for did in (only_voltage, dead):
        st = compute_status(s, did, 0)
        assert st.calibration.state is CalibrationState.NO_CHANNEL
        assert st.calibration.issue == ""
        for stage in LOCKED:
            assert "override" not in st.stage(stage).reason, st.stage(stage).reason
            assert st.stage(stage).reason.endswith("."), st.stage(stage).reason


def test_the_chip_still_renders_a_no_channel_status_that_carries_an_issue(qapp, qtbot, tmp_path):
    """Carrying the override sentence on a NO_CHANNEL status must not trip the chip into
    its DAMAGED rendering: text, accent and state are the neutral ones."""
    s = Session()
    did = s.add_document(_declared(tmp_path))
    _treat_as(s, did, 1, ChannelKind.VOLTAGE, "mV")
    st = compute_status(s, did, 1)
    assert st.calibration.issue, "precondition: the status carries the override sentence"

    chip = CalibrationChip()
    qtbot.addWidget(chip)
    chip.set_status(st.calibration)
    assert chip.text() == "No current channel"
    assert chip.accent() == NEUTRAL
    assert chip.status().state is CalibrationState.NO_CHANNEL
    assert not chip.status().is_damaged
