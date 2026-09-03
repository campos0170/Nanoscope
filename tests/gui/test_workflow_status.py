"""Pure unit tests for :mod:`nanoscope.gui.workflow_status` — no Qt, no QApplication.

These pin the two ease-of-use affordances' whole state machine against a REAL
:class:`~nanoscope.core.session.Session` driven exactly as the app drives it (open a
recording, set a gain, store a detection, accept an event), so the chip and rail can
be trusted without ever opening a window:

* the calibration chip's NO_CHANNEL / UNCALIBRATED / CALIBRATED state, and that
  "calibrated" means the same thing the sidecar means (differs from default_for);
* each stage's done / available / reason, the optional non-blocking FILTER, and how
  ``current_stage`` walks Open → Calibrate → Detect → Curate → Analyze.

Nothing here needs a QApplication: ``workflow_status`` imports no Qt, and a Session is
a plain QObject that constructs and emits without an event loop. The frozen 20-column
EventSet schema is used verbatim (``empty_df``); nothing here mutates it.
"""

from __future__ import annotations

import nanoscope  # noqa: F401  -> runs bootstrap.py (import-order discipline; still no Qt widgets)

import pandas as pd

from nanoscope.analysis.events import EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.workflow_status import (
    CalibrationState,
    PipelineStatus,
    Stage,
    compute_status,
)

import numpy as np

FS = 50_000.0
N = 20_000


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path, *, name="rec.dat", fs=FS,
         channels: tuple[ChannelInfo, ...] | None = None) -> TraceDocument:
    if channels is None:
        channels = (ChannelInfo(index=0, name="current", raw_units="pA",
                                kind=ChannelKind.CURRENT),)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=channels)
    sig = np.zeros(N, dtype=np.float32)
    return TraceDocument(meta=meta, channels=tuple(sig for _ in channels))


def _eventset(accepted: list[bool]) -> EventSet:
    """A minimal valid EventSet with one row per entry in ``accepted``."""
    n = len(accepted)
    df = empty_df()
    if n:
        rows = pd.DataFrame({
            "event_id": np.arange(n, dtype=np.int64),
            "recording": pd.array(["rec"] * n, dtype="string"),
            "pore_id": pd.array(["p0"] * n, dtype="string"),
            "start_idx": np.arange(n, dtype=np.int64) * 100,
            "end_idx": np.arange(n, dtype=np.int64) * 100 + 50,
            "t_start_s": np.zeros(n),
            "dwell_s": np.full(n, 1e-3),
            "baseline_pA": np.full(n, -50.0),
            "depth_pA": np.full(n, 20.0),
            "frac_depth": np.full(n, 0.4),
            "ecd_pA_s": np.full(n, 0.02),
            "max_dev_pA": np.full(n, 25.0),
            "mean_dev_pA": np.full(n, 20.0),
            "rise_s": np.full(n, 1e-4),
            "fall_s": np.full(n, 1e-4),
            "snr": np.full(n, 8.0),
            "n_levels": np.ones(n, dtype=np.int64),
            "filter_limited": np.zeros(n, dtype=bool),
            "accepted": np.asarray(accepted, dtype=bool),
            "flag": pd.array([""] * n, dtype="string"),
        })
        df = pd.concat([df, rows], ignore_index=True)
    return EventSet(df, {"analyzed_duration_s": 1.0, "recording": "rec"})


def _install_detection(session: Session, doc_id: str, ev: EventSet, channel: int = 0) -> None:
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=ev))


def _st(status: PipelineStatus, stage: Stage):
    return status.stage(stage)


# ================================================================================================
# empty session
# ================================================================================================

def test_empty_session_only_open_is_actionable():
    session = Session()
    status = compute_status(session, None, None)

    assert status.calibration.state is CalibrationState.NO_CHANNEL
    assert status.current_stage is Stage.OPEN
    assert _st(status, Stage.OPEN).done is False
    assert _st(status, Stage.OPEN).available is True
    # nothing downstream is reachable yet
    for stage in (Stage.CALIBRATE, Stage.FILTER, Stage.DETECT, Stage.CURATE, Stage.ANALYZE):
        assert _st(status, stage).available is False
        assert _st(status, stage).done is False


def test_stage_set_is_exactly_the_pipeline_in_order():
    session = Session()
    status = compute_status(session, None, None)
    assert tuple(s.stage for s in status.stages) == (
        Stage.OPEN, Stage.CALIBRATE, Stage.FILTER, Stage.DETECT, Stage.CURATE, Stage.ANALYZE)


# ================================================================================================
# a recording is open (uncalibrated current channel)
# ================================================================================================

def test_open_current_doc_uncalibrated_advances_to_calibrate(tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    status = compute_status(session, doc_id, 0)

    assert status.calibration.state is CalibrationState.UNCALIBRATED
    assert status.calibration.channel == 0
    assert _st(status, Stage.OPEN).done is True
    assert _st(status, Stage.CALIBRATE).available is True
    assert _st(status, Stage.CALIBRATE).done is False
    assert _st(status, Stage.DETECT).available is True
    assert _st(status, Stage.DETECT).done is False
    assert _st(status, Stage.CURATE).available is False
    assert _st(status, Stage.ANALYZE).available is False
    assert status.current_stage is Stage.CALIBRATE
    assert _st(status, Stage.CALIBRATE).current is True


def test_filter_is_optional_available_and_never_current(tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    status = compute_status(session, doc_id, 0)
    flt = _st(status, Stage.FILTER)
    assert flt.optional is True
    assert flt.available is True         # a current channel exists
    assert flt.done is False             # never required-done
    assert flt.current is False          # excluded from the current-stage walk
    assert status.current_stage is not Stage.FILTER


# ================================================================================================
# calibrated -> current advances to Detect (reuses the sidecar's user-vs-default notion)
# ================================================================================================

def test_setting_a_nondefault_gain_marks_calibrated_and_advances(tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    session.set_calibration(doc_id, 0, ChannelCalibration(gain=2500.0, offset=0.0,
                                                          target_units="pA"))
    status = compute_status(session, doc_id, 0)

    assert status.calibration.state is CalibrationState.CALIBRATED
    assert "2500" in status.calibration.detail
    assert _st(status, Stage.CALIBRATE).done is True
    assert status.current_stage is Stage.DETECT
    assert _st(status, Stage.DETECT).current is True


def test_calibrated_matches_the_default_for_notion_exactly(tmp_path):
    """A VOLTAGE-mV channel's DEFAULT is gain 0.1 (see CalibrationSet.default_for). Re-applying
    that exact default must NOT read as user-calibrated — the chip and the sidecar share one
    notion. A voltage channel resolves to the recording's current channel for calibration, so
    exercise the identity here on the current channel instead: re-applying its default (identity)
    leaves it UNCALIBRATED."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    # Re-applying the current channel's own default (identity) is NOT a user calibration.
    session.set_calibration(doc_id, 0, ChannelCalibration())
    status = compute_status(session, doc_id, 0)
    assert status.calibration.state is CalibrationState.UNCALIBRATED


# ================================================================================================
# a detection is stored -> Detect done, Curate/Analyze unlock, current = Curate
# ================================================================================================

def _calibrate(session: Session, doc_id: str, ch: int = 0) -> None:
    session.set_calibration(doc_id, ch, ChannelCalibration(gain=2500.0, target_units="pA"))


def test_storing_a_detection_unlocks_curate_and_analyze(tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    _calibrate(session, doc_id)                    # realistic flow: gain set before detecting
    _install_detection(session, doc_id, _eventset([False, False]))
    status = compute_status(session, doc_id, 0)

    assert _st(status, Stage.DETECT).done is True
    assert _st(status, Stage.CURATE).available is True
    assert _st(status, Stage.CURATE).done is False        # nothing accepted yet
    assert _st(status, Stage.ANALYZE).available is True
    assert _st(status, Stage.ANALYZE).done is False        # terminal, never "done"
    assert status.current_stage is Stage.CURATE


def test_accepting_an_event_completes_curate_and_lands_on_analyze(tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    _calibrate(session, doc_id)
    _install_detection(session, doc_id, _eventset([True, False]))
    status = compute_status(session, doc_id, 0)

    assert _st(status, Stage.CURATE).done is True
    assert status.current_stage is Stage.ANALYZE
    assert _st(status, Stage.ANALYZE).current is True


def test_detection_before_calibration_points_back_to_calibrate(tmp_path):
    """Detection does not require calibration, but if the user skipped it the current step is
    still CALIBRATE — the whole point of the chip is to flag the uncalibrated gain, so the rail
    must not let a premature detection hide it."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    _install_detection(session, doc_id, _eventset([True]))   # detected, but never calibrated
    status = compute_status(session, doc_id, 0)
    assert status.calibration.state is CalibrationState.UNCALIBRATED
    assert _st(status, Stage.DETECT).done is True
    assert status.current_stage is Stage.CALIBRATE


# ================================================================================================
# selection points at the voltage channel -> still reflects the current channel
# ================================================================================================

def test_selecting_voltage_channel_resolves_to_current_channel(tmp_path):
    session = Session()
    chans = (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="voltage", raw_units="mV", kind=ChannelKind.VOLTAGE),
    )
    doc_id = session.add_document(_doc(tmp_path, channels=chans))
    # Select the voltage channel (index 1) — calibration still targets the current channel.
    status = compute_status(session, doc_id, 1)
    assert status.calibration.channel == 0
    assert status.calibration.state is CalibrationState.UNCALIBRATED
    assert _st(status, Stage.DETECT).available is True


# ================================================================================================
# a recording with no current channel -> NO_CHANNEL, calibrate/detect locked with a reason
# ================================================================================================

def test_recording_without_a_current_channel_is_no_channel(tmp_path):
    session = Session()
    chans = (ChannelInfo(index=0, name="voltage", raw_units="mV", kind=ChannelKind.VOLTAGE),)
    doc_id = session.add_document(_doc(tmp_path, channels=chans))
    status = compute_status(session, doc_id, 0)

    assert status.calibration.state is CalibrationState.NO_CHANNEL
    assert status.calibration.channel is None
    assert _st(status, Stage.OPEN).done is True            # a recording IS open
    assert _st(status, Stage.CALIBRATE).available is False
    assert _st(status, Stage.DETECT).available is False
    assert "current channel" in _st(status, Stage.CALIBRATE).reason
    # nothing in the pipeline is actionable on this recording
    assert status.current_stage is None


def test_dead_current_channel_is_no_channel(tmp_path):
    session = Session()
    chans = (ChannelInfo(index=0, name="current", raw_units="pA",
                         kind=ChannelKind.CURRENT, is_dead=True),)
    doc_id = session.add_document(_doc(tmp_path, channels=chans))
    status = compute_status(session, doc_id, 0)
    assert status.calibration.state is CalibrationState.NO_CHANNEL
    assert _st(status, Stage.DETECT).available is False


def test_unknown_doc_id_is_robust(tmp_path):
    session = Session()
    session.add_document(_doc(tmp_path))
    # A stale/closed doc_id: OPEN is still done (a recording is open), nothing else actionable.
    status = compute_status(session, "no-such-doc", 0)
    assert status.calibration.state is CalibrationState.NO_CHANNEL
    assert _st(status, Stage.OPEN).done is True
    assert status.current_stage is None


def test_detection_on_a_different_channel_does_not_count(tmp_path):
    """DETECT.done is about the RESOLVED current channel; a result filed under another channel
    must not masquerade as this channel's detection."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    _install_detection(session, doc_id, _eventset([True]), channel=7)  # wrong channel
    status = compute_status(session, doc_id, 0)
    assert _st(status, Stage.DETECT).done is False
    assert _st(status, Stage.CURATE).available is False
