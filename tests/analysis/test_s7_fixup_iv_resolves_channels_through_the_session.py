"""Sweep-7 X2: :mod:`nanoscope.analysis.iv` must resolve BOTH channels through the
Session, not through the file header.

The defect the X-combos skeptic measured: on a 2 s CSV of the owner's -700 mV
recording with neutral column headers (both channels ``unknown``), ch0 corrected to
current / pA / x2.5 and ch1 to voltage / mV through the real Calibration panel and
Detect storing 15 events on ch0, ``Analysis > I-V / Conductance...`` was ENABLED --
``MainWindow._has_iv_data`` asks ``Session.effective_kind`` for both of its tests --
while the window it opened read "No usable recordings. Open a recording with a
current channel and a known voltage, then Refresh." ``iv.iv_point`` answered ``None``
with the refusal "no live current channel", ``iv.applied_voltage_readout`` answered
the FILENAME nominal instead of the corrected voltage channel, and the
voltage-dependence gate counted the recording as not ready.

``iv_point`` is the ONE resolver the I-V window, the exclusion list and the
voltage-dependence series share, so all three denied the recording together. These
tests pin the resolver itself (no Qt): the corrected channels are found, the reverse
override is refused, and -- the number-safety half -- a recording that declares its
channels in the header lands on exactly the same point it always did.

Every one of the override tests fails against the pre-fix ``analysis/iv.py`` (its
``_current_channel(meta)`` walked ``c.kind is ChannelKind.CURRENT`` and
``applied_voltage_readout`` walked ``c.kind is ChannelKind.VOLTAGE``); the
header-declared test passes there too, which is the point of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis import iv
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)

I_OPEN_RAW = -49.0      # raw counts; x2.5 gain -> -122.5 pA, the shape of the CAL fixture
V_RAW = -700.0          # raw counts on the command channel; gain 1 -> -700 mV
NOMINAL_mV = -700.0     # what the FILENAME says -- the value the header walk fell back to


def _two_channel_doc(tmp_path: Path, *, kinds: tuple[ChannelKind, ChannelKind],
                     name: str, n: int = 4096) -> TraceDocument:
    """A two-channel recording whose header kinds are whatever the caller says.

    ``kinds=(UNKNOWN, UNKNOWN)`` is the undeclared CSV / nisin-TDMS shape the
    Calibration-panel override exists for; ``(CURRENT, VOLTAGE)`` is an honest ABF.
    Both carry a filename ``nominal_voltage_mV`` so the *wrong* answer (the nominal
    fallback) is available and a test can tell the two apart.
    """
    cur = np.ascontiguousarray(np.full(n, I_OPEN_RAW, dtype=np.float32))
    volt = np.ascontiguousarray(np.full(n, V_RAW, dtype=np.float32))
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=n,
        sample_rate_hz=50_000.0,
        rate_provenance=Provenance.FILE_META,
        channels=(
            ChannelInfo(index=0, name="col0", raw_units="", kind=kinds[0]),
            ChannelInfo(index=1, name="col1", raw_units="", kind=kinds[1]),
        ),
        nominal_voltage_mV=NOMINAL_mV,
        voltage_provenance=Provenance.FILENAME,
    )
    return TraceDocument(meta=meta, channels=(cur, volt))


def _corrected_session(tmp_path: Path, *, name: str,
                       voltage_units: str = "mV") -> tuple[Session, str]:
    """The undeclared recording with the two Calibration-panel overrides applied.

    ``write_sidecar=False`` on both: nothing may be written beside a recording from a
    test (the panel is the only door allowed to write a user-owned .npcal.json).
    """
    session = Session()
    doc_id = session.add_document(
        _two_channel_doc(tmp_path, kinds=(ChannelKind.UNKNOWN, ChannelKind.UNKNOWN), name=name))
    session.set_calibration(doc_id, 0, ChannelCalibration(
        gain=2.5, target_units="pA", kind=ChannelKind.CURRENT), write_sidecar=False)
    session.set_calibration(doc_id, 1, ChannelCalibration(
        gain=1.0, target_units=voltage_units, kind=ChannelKind.VOLTAGE), write_sidecar=False)
    return session, doc_id


def test_iv_point_reads_the_channel_the_user_corrected_to_current(tmp_path):
    session, doc_id = _corrected_session(tmp_path, name="undeclared_700mV.csv")

    point = iv.iv_point(session, doc_id)

    assert point is not None, "the corrected current channel is the one the app detected on"
    assert iv.iv_point_refusal(session, doc_id) == ""
    assert point.i_open == pytest.approx(I_OPEN_RAW * 2.5)
    assert point.current_units == "pA"


def test_iv_point_takes_its_voltage_from_the_corrected_channel_not_the_filename(tmp_path):
    session, doc_id = _corrected_session(tmp_path, name="undeclared_700mV.csv")

    got = iv.applied_voltage_readout(session, doc_id)

    assert got == iv.AppliedVoltage(value=V_RAW, units="mV", source="measured")
    # The pre-fix walk answered the FILENAME nominal here -- same value shape, but
    # "nominal", and it would have kept answering it however the user calibrated ch1.
    assert iv.iv_point(session, doc_id).voltage_source == "measured"
    assert iv.applied_voltage_mV(session, doc_id) == (pytest.approx(V_RAW), "measured")


def test_the_i_v_window_gathers_the_corrected_recording(tmp_path):
    """``iv_points`` is what the window's "No usable recordings." message counts."""
    session, doc_id = _corrected_session(tmp_path, name="undeclared_700mV.csv")

    points = iv.iv_points(session)

    assert [p.doc_id for p in points] == [doc_id]
    assert points[0].v_volts == pytest.approx(V_RAW / 1000.0)


def test_the_exclusion_list_names_the_corrected_recording_when_its_voltage_units_are_blank(
        tmp_path):
    """The refusal the window LISTS has to see the same channels the plot does.

    With the voltage corrected but its units left undeclared, the recording must
    appear as an exclusion ("declare the units in Calibration"), not vanish: the
    pre-fix walk could not find a current channel at all, so it fell out of both
    lists and the window said nothing about it.
    """
    session, doc_id = _corrected_session(tmp_path, name="blank_units.csv", voltage_units="")

    excluded = iv.iv_exclusions(session)

    assert [x.doc_id for x in excluded] == [doc_id]
    assert excluded[0].i_open == pytest.approx(I_OPEN_RAW * 2.5)
    assert excluded[0].v_units == ""
    assert iv.iv_points(session) == []


def test_a_channel_re_declared_as_voltage_stops_being_the_current_channel(tmp_path):
    """The reverse override, which is what makes this one resolver and not two.

    A header-CURRENT channel the user re-declares as a command voltage greys the
    menu item out (``_has_iv_data`` reads the effective kind); the window behind it
    must refuse for the same reason rather than plot a point off it.
    """
    session = Session()
    doc_id = session.add_document(_two_channel_doc(
        tmp_path, kinds=(ChannelKind.CURRENT, ChannelKind.VOLTAGE), name="reversed.abf"))
    session.set_calibration(doc_id, 0, ChannelCalibration(
        gain=1.0, target_units="mV", kind=ChannelKind.VOLTAGE), write_sidecar=False)

    assert iv.iv_point(session, doc_id) is None
    assert iv.iv_point_refusal(session, doc_id) == iv.IV_NO_CURRENT_CHANNEL
    assert iv.iv_points(session) == []


def test_a_header_declared_recording_is_untouched_by_the_session_resolver(tmp_path):
    """The numbers-do-not-move half: with no override the session's answer IS the
    header's, so an honest ABF-shaped recording gives the identical point."""
    session = Session()
    doc_id = session.add_document(_two_channel_doc(
        tmp_path, kinds=(ChannelKind.CURRENT, ChannelKind.VOLTAGE), name="honest.abf"))
    session.set_calibration(doc_id, 0, ChannelCalibration(
        gain=2.5, target_units="pA"), write_sidecar=False)
    session.set_calibration(doc_id, 1, ChannelCalibration(
        gain=1.0, target_units="mV"), write_sidecar=False)

    point = iv.iv_point(session, doc_id)

    assert point is not None
    assert point.i_open == pytest.approx(I_OPEN_RAW * 2.5)
    assert point.current_units == "pA"
    assert point.v_volts == pytest.approx(V_RAW / 1000.0)
    assert point.voltage_source == "measured"
