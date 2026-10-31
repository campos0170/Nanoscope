"""Reset to default restores what the channel OPENED with, and resets nothing else.

Design §1.2 defects 5 and 6, §6 (the calibration and sidecar rows), decision 5.

**Defect 5.** Opening a recording seeds each channel with ``default_for(info, median)``, so a
host-guest voltage channel that reads its applied bias (75 mV) opens at gain 1.0. Reset to
default installed ``default_for(info)`` WITHOUT the median, i.e. gain 0.1 -- 10x on every
voltage, conductance and V-dependence figure. The sidecar payload and the workflow chip
compared with the same median-less default, so calibrating only the current channel wrote
the untouched voltage channel into the sidecar, and the chip's notion of "the user set this"
said yes for a channel nobody had touched.

**Defect 6.** Reset also cleared the channel's note and its "Treat this channel as" override:
three things reset by a button that names one.

Synthetic two-channel recordings only -- a current (pA) and a voltage (mV) channel -- each
with ``meta.path`` under ``tmp_path``, so every sidecar lands in the test's own folder.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from dataclasses import replace

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration, sidecar_path
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.workflow_status import CalibrationState, _is_user_calibrated, compute_status

FS = 50_000.0
N = 4_000
HOST_GUEST_MV = 75.07        # reads the applied bias itself -> opens at gain 1.0
FERRITIN_MV = -6998.9        # the 10x-inflated encoding -> opens at gain 0.1
NOTE = "bias checked against the protocol sheet, 2026-10-30"

CURRENT = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
VOLTAGE = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
#: The owner's text/TDMS shape: a signal column the file never declared, beside the voltage.
UNDECLARED = ChannelInfo(index=0, name="signal", raw_units="", kind=ChannelKind.UNKNOWN)


# ================================================================================================
# builders
# ================================================================================================
def _doc(tmp_path, volt_mV: float, *, first: ChannelInfo = CURRENT,
         name: str = "rec.abf") -> TraceDocument:
    rng = np.random.default_rng(0)
    signal = rng.normal(-100.0, 5.0, N).astype(np.float32)
    voltage = (volt_mV + rng.normal(0.0, 0.05, N)).astype(np.float32)
    path = tmp_path / name
    path.write_bytes(b"ABF ")            # the recording is where it was opened from
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(first, VOLTAGE))
    return TraceDocument(meta=meta, channels=(signal, voltage))


def _open(tmp_path, volt_mV: float, **kw) -> tuple[Session, str]:
    session = Session()
    return session, session.add_document(_doc(tmp_path, volt_mV, **kw))


def _panel(qtbot, session: Session, doc_id: str, channel: int, *,
           remove: bool = True, lose_note: bool = True) -> tuple[CalibrationPanel, list]:
    """The panel with both of its confirmations answered without a live modal (the
    injectable seams ``test_sidecar_safety`` uses), and every question it asked recorded."""
    panel = CalibrationPanel(session)
    qtbot.addWidget(panel)
    asked: list = []
    panel._confirm_sidecar_removal = lambda path, **kw: (
        asked.append(("remove", kw.get("action"))), remove)[1]
    panel._confirm_note_loss = lambda path, lost, **kw: (
        asked.append(("note", kw.get("action"), lost)), lose_note)[1]
    panel.set_selection(doc_id, channel)
    return panel, asked


def _apply(panel: CalibrationPanel, *, gain: str | None = None, units: str | None = None,
           treat_as: ChannelKind | None | str = "keep") -> None:
    """Type into the form the way a user does, then press Apply."""
    if gain is not None:
        panel._gain.setText(gain)
    if units is not None:
        panel._units.setText(units)
    if treat_as != "keep":
        panel._treat_as.setCurrentIndex(panel._treat_as.findData(treat_as))
    panel._apply()


def _sidecar(tmp_path, name: str = "rec.abf") -> dict:
    return json.loads(sidecar_path(tmp_path / name).read_text())["channels"]


# ================================================================================================
# defect 5 -- the gain Reset installs
# ================================================================================================
def test_reset_returns_a_host_guest_voltage_channel_to_the_gain_it_opened_with(qtbot, tmp_path):
    """THE DEFECT. 75 mV is a plausible bias, so the channel opened at 1.0; Reset gave 0.1.

    The channel also carries a note (as a sidecar or a project hands one back) and the
    user's explicit "treat as voltage", and Reset resets neither."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    assert s.calibration(doc_id, 1).gain == 1.0, "setup: a plausible bias opens at 1.0"
    s.set_calibration(doc_id, 1, replace(s.calibration(doc_id, 1), note=NOTE),
                      write_sidecar=False)
    panel, _asked = _panel(qtbot, s, doc_id, 1)
    _apply(panel, gain="2", treat_as=ChannelKind.VOLTAGE)
    assert s.calibration(doc_id, 1).gain == 2.0

    panel._reset()

    cal = s.calibration(doc_id, 1)
    assert cal.gain == 1.0, f"Reset installed gain {cal.gain:g}; the channel opened at 1.0"
    assert (cal.offset, cal.target_units) == (0.0, "mV")
    assert cal.note == NOTE, "Reset cleared the channel's note"
    assert cal.kind is ChannelKind.VOLTAGE, "Reset cleared the 'Treat this channel as' choice"
    assert panel._gain.text() == "1", "the form reads back what Reset committed"
    # The kept override still differs from the opened default, so the entry -- and the
    # note in it -- stays in the user's file.
    entry = _sidecar(tmp_path)["1"]
    assert (entry["gain"], entry["note"], entry["treat_as_kind"]) == (1.0, NOTE, "voltage")


def test_reset_returns_a_ferritin_voltage_channel_to_its_10x_correction(qtbot, tmp_path):
    """CONTROL / PASSES BOTH. -6999 mV is no bias anyone applied, so the 10x-inflated
    encoding is detected and the channel opens at 0.1 -- and that is what Reset restores."""
    s, doc_id = _open(tmp_path, FERRITIN_MV)
    assert s.calibration(doc_id, 1).gain == pytest.approx(0.1)
    panel, _asked = _panel(qtbot, s, doc_id, 1)
    _apply(panel, gain="0.25")
    assert s.calibration(doc_id, 1).gain == 0.25

    panel._reset()

    cal = s.calibration(doc_id, 1)
    assert cal.gain == pytest.approx(0.1)
    assert (cal.offset, cal.target_units) == (0.0, "mV")


def test_reset_restores_the_provenance_the_channel_opened_with(qtbot, tmp_path):
    """CONTROL / PASSES BOTH. The badge says where the numbers came from: after Reset they
    came from the file again (FILE_META), not from a user edit (UNKNOWN)."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    panel, _asked = _panel(qtbot, s, doc_id, 1)
    _apply(panel, gain="2")
    assert s.calibration(doc_id, 1).provenance is Provenance.UNKNOWN

    panel._reset()

    assert s.calibration(doc_id, 1).provenance is Provenance.FILE_META


# ================================================================================================
# defect 5 -- the sidecar and the chip compare with the same opened default
# ================================================================================================
def test_calibrating_the_current_channel_leaves_the_untouched_voltage_channel_out(qtbot,
                                                                                  tmp_path):
    """Before: the voltage channel (opened at 1.0, never touched) read as "differs from 0.1"
    and was written beside the current channel's gain."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    panel, asked = _panel(qtbot, s, doc_id, 0)
    _apply(panel, gain="2.5", units="pA")

    channels = _sidecar(tmp_path)
    assert set(channels) == {"0"}, f"the sidecar stored channels {sorted(channels)}"
    assert channels["0"]["gain"] == 2.5
    assert asked == []


def test_a_typed_point_one_on_a_host_guest_voltage_channel_is_stored(qtbot, tmp_path):
    """The same comparison the other way round. A user who decides this file IS the 10x
    encoding types 0.1 -- which equalled the median-less default, so nothing was written and
    the next open put the channel back at 1.0 without a word."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    panel, _asked = _panel(qtbot, s, doc_id, 1)
    _apply(panel, gain="0.1")

    assert _sidecar(tmp_path)["1"]["gain"] == pytest.approx(0.1)
    again, again_id = _open(tmp_path, HOST_GUEST_MV)
    assert again.calibration(again_id, 1).gain == pytest.approx(0.1)


def test_an_old_sidecar_that_stored_the_untouched_channel_opens_the_same(qtbot, tmp_path):
    """A sidecar an older build wrote that way carries the voltage channel exactly as the
    open seeds it, note included: it reads back with the same numbers, the next write leaves
    the entry out, and dropping that note is not a loss to ask about -- the next open writes
    it back word for word."""
    first = Session()
    seeded = first.calibration(first.add_document(_doc(tmp_path, HOST_GUEST_MV)), 1)
    assert seeded.gain == 1.0 and seeded.note, "setup: what an older build stored for ch1"
    sidecar_path(tmp_path / "rec.abf").write_text(json.dumps({"version": 2, "channels": {
        "0": {"gain": 2.5, "offset": 0.0, "target_units": "pA", "note": "",
              "name": "IN15", "kind": "current"},
        "1": {"gain": seeded.gain, "offset": seeded.offset, "target_units": seeded.target_units,
              "note": seeded.note, "name": "IN14", "kind": "voltage"}}}))
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    volt = s.calibration(doc_id, 1)
    assert (volt.gain, volt.offset, volt.target_units, volt.note) == (
        seeded.gain, seeded.offset, seeded.target_units, seeded.note)
    assert s.calibration(doc_id, 0).gain == 2.5

    panel, asked = _panel(qtbot, s, doc_id, 0)
    _apply(panel, gain="3")

    assert set(_sidecar(tmp_path)) == {"0"}
    assert asked == [], f"an ordinary Apply raised {asked}"


def test_the_chip_does_not_call_the_untouched_voltage_channel_calibrated(tmp_path):
    """The chip's "gain settings entered" is the sidecar's own comparison, and it said yes
    for a voltage channel that opened at 1.0 and was never touched."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    assert not _is_user_calibrated(s, doc_id, VOLTAGE)

    # Where the chip itself reads that channel: the user says it is the current. Nothing
    # about its gain, offset or units was entered, and the chip must not claim otherwise.
    s.set_calibration(doc_id, 1, replace(s.calibration(doc_id, 1), kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    status = compute_status(s, doc_id, 1)
    assert status.calibration.channel == 1
    assert status.calibration.state is CalibrationState.UNCALIBRATED, status.calibration.label


# ================================================================================================
# defect 6 -- "Treat this channel as" keeps its own reset
# ================================================================================================
def test_reset_keeps_the_treat_as_override_and_the_sidecar_stays(qtbot, tmp_path):
    """An undeclared column corrected to a current, in pA, at gain 2.5. Reset puts the numbers
    back; the channel is still the current -- its pane, its detection, its sidecar entry --
    and nothing was asked, because nothing in the file is lost."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV, first=UNDECLARED)
    panel, asked = _panel(qtbot, s, doc_id, 0)
    _apply(panel, gain="2.5", units="pA", treat_as=ChannelKind.CURRENT)
    assert s.resolve_current_channel(doc_id, None) == 0

    panel._reset()

    cal = s.calibration(doc_id, 0)
    assert (cal.gain, cal.offset, cal.target_units) == (1.0, 0.0, "")
    assert cal.kind is ChannelKind.CURRENT
    assert panel._treat_as.currentData() is ChannelKind.CURRENT
    assert s.resolve_current_channel(doc_id, None) == 0, "detection still acts on it"
    assert asked == []
    channels = _sidecar(tmp_path)
    assert set(channels) == {"0"}
    assert (channels["0"]["gain"], channels["0"]["treat_as_kind"]) == (1.0, "current")

    # The override's own reset: "as the file declares", then Apply. Everything is then at
    # the default it opened with, so the file goes -- after asking.
    _apply(panel, treat_as=None)
    assert s.calibration(doc_id, 0).kind is None
    assert asked == [("remove", "Apply")]
    assert not sidecar_path(tmp_path / "rec.abf").exists()


# ================================================================================================
# the confirmations still guard Reset
# ================================================================================================
def test_reset_still_asks_before_it_deletes_the_sidecar(qtbot, tmp_path):
    """Calibrate the current channel only; Reset it. Every channel is then at its opened
    default, so the commit deletes the user's file -- which it may only do when asked."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    panel, asked = _panel(qtbot, s, doc_id, 0, remove=False)
    _apply(panel, gain="2", units="pA")
    sc = sidecar_path(tmp_path / "rec.abf")
    before = sc.read_text()

    panel._reset()

    assert asked == [("remove", "Reset")]
    assert sc.read_text() == before, "declined, yet the file changed"
    assert s.calibration(doc_id, 0).gain == 2.0, "declined, yet the calibration was reset"
    assert "Nothing changed" in panel.status_text()

    panel._confirm_sidecar_removal = lambda path, **kw: (
        asked.append(("remove", kw.get("action"))), True)[1]
    panel._reset()

    assert not sc.exists()
    assert s.calibration(doc_id, 0).gain == 1.0
    assert "deleted" in panel.status_text()


def test_reset_still_asks_before_it_drops_a_note_out_of_a_surviving_file(qtbot, tmp_path):
    """The note stays on the channel in this session, but its entry leaves the file (the
    channel is back at its opened default), so on the next open it would be gone. That is
    what the second confirmation is for, and declining it changes nothing."""
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    s.set_calibration(doc_id, 0, ChannelCalibration(gain=2.0, target_units="pA", note=NOTE))
    s.set_calibration(doc_id, 1, ChannelCalibration(gain=0.5, target_units="mV"))
    panel, asked = _panel(qtbot, s, doc_id, 0, lose_note=False)

    panel._reset()

    assert asked == [("note", "Reset", ((0, NOTE),))]
    assert _sidecar(tmp_path)["0"]["note"] == NOTE
    assert s.calibration(doc_id, 0).gain == 2.0
    assert "Nothing changed" in panel.status_text()


# ================================================================================================
# the record itself
# ================================================================================================
def test_the_opened_default_is_recorded_before_the_sidecar_lands(qtbot, tmp_path):
    """The default a channel "opened with" is the app's, not what a saved calibration installed
    on open: the record is taken before the sidecar is applied, so Reset on a channel the
    sidecar set to 2.5 goes back to 1.0 (keeping the sidecar's note), and asks before the file
    goes."""
    sidecar_path(tmp_path / "rec.abf").write_text(json.dumps({"version": 2, "channels": {
        "0": {"gain": 2.5, "offset": 0.0, "target_units": "pA", "note": NOTE,
              "name": "IN15", "kind": "current"}}}))
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    assert s.calibration(doc_id, 0).gain == 2.5
    assert s.opened_default(doc_id, 0) == ChannelCalibration()
    panel, asked = _panel(qtbot, s, doc_id, 0)

    panel._reset()

    cal = s.calibration(doc_id, 0)
    assert (cal.gain, cal.offset, cal.target_units, cal.note) == (1.0, 0.0, "", NOTE)
    assert asked == [("remove", "Reset")]
    assert not sidecar_path(tmp_path / "rec.abf").exists()


def test_a_segment_resets_to_its_parents_opened_default(qtbot, tmp_path):
    """A segment is a window of its parent, and the voltage encoding belongs to the file.
    A window cut where the bias was off reads 0 mV -- a "plausible bias" on its own -- in a
    recording whose channel is the 10x encoding throughout."""
    rng = np.random.default_rng(1)
    voltage = (FERRITIN_MV + rng.normal(0.0, 0.05, N)).astype(np.float32)
    voltage[1000:2000] = 0.0                       # the bias switched off for a while
    path = tmp_path / "rec.abf"
    path.write_bytes(b"ABF ")
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(CURRENT, VOLTAGE))
    s = Session()
    pid = s.add_document(TraceDocument(
        meta=meta, channels=(rng.normal(-100.0, 5.0, N).astype(np.float32), voltage)))
    assert s.opened_default(pid, 1).gain == pytest.approx(0.1)

    seg_id = s.create_segment(pid, 1000, 2000, name="bias off")
    assert s.opened_default(seg_id, 1).gain == pytest.approx(0.1)
    panel, _asked = _panel(qtbot, s, seg_id, 1)
    _apply(panel, gain="0.3")
    panel._reset()

    assert s.calibration(seg_id, 1).gain == pytest.approx(0.1)
    assert not sidecar_path(path).exists(), "a segment never writes beside its parent"


def test_closing_a_recording_drops_its_opened_defaults(tmp_path):
    s, doc_id = _open(tmp_path, HOST_GUEST_MV)
    assert {k[0] for k in s._calibration.opened_defaults} == {doc_id}
    s.close_document(doc_id)
    assert s._calibration.opened_defaults == {}
    assert s._calibration.entries == {}
