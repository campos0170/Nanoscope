"""EXPL2-5 regression: the I-V window must not fabricate mV / nS from a voltage
channel whose units are UNDECLARED.

CF-5 fixed the Calibration panel: a measured command voltage with units ``""``
(the HDF5/CSV loaders leave them blank on purpose; docs/USER_GUIDE.md says the
user sets them in Calibration) is shown as "<value> (units unknown)", the nS
conductance is refused, and the note says why. The I-V window read the SAME
document through ``iv._applied_voltage_mV`` and asserted the raw number to be
millivolts ("V = 0.52 mV [measured]") and headlined a nanosiemens conductance —
1000x wrong wherever the channel is really volts, and in direct contradiction
with the Calibration panel in the same session.

These tests pin the CF-5 contract onto this surface, through the REAL widgets:

* an undeclared-units voltage never appears with an mV suffix anywhere in the
  window; the recording is listed with "(units unknown)" and the exact
  ``iv.conductance_note`` reason the Calibration panel shows (one voice);
* no nS headline/fit is fabricated from such recordings;
* declared mV and declared V (the ABF 10x path) keep working; and
* declaring the units in Calibration — the documented fix — brings the
  recording back into the fit.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.iv_panel import IVWindow


# ================================================================================================
# builder — a voltage + current recording whose voltage units we control
# ================================================================================================

def _doc(tmp_path, *, v_raw: float, i_pA: float, name: str,
         v_units: str = "", n: int = 4096) -> TraceDocument:
    """A two-channel recording: VOLTAGE (units ``v_units``; ``""`` == undeclared,
    exactly what io/hdf5.py produces) + CURRENT in pA."""
    volt = np.ascontiguousarray(np.full(n, v_raw, dtype=np.float32))
    cur = np.ascontiguousarray(np.full(n, i_pA, dtype=np.float32))
    v_info = ChannelInfo(index=0, name="IN14", raw_units=v_units, kind=ChannelKind.VOLTAGE)
    c_info = ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
        channels=(v_info, c_info))
    return TraceDocument(meta=meta, channels=(volt, cur))


# ================================================================================================
# the defect scenario — every recording units-undeclared
# ================================================================================================

def test_undeclared_voltage_units_never_asserted_mV(qapp, qtbot, tmp_path):
    """The EXPL2-5 repro through the real window: no mV label, no nS headline,
    the values still shown — with "(units unknown)" — and the reason named."""
    session = Session()
    session.add_document(_doc(tmp_path, v_raw=0.30, i_pA=-60.0, name="b1.h5"))
    session.add_document(_doc(tmp_path, v_raw=0.52, i_pA=-104.0, name="b2.h5"))

    w = IVWindow(session)
    qtbot.addWidget(w)
    text = w.details_text()

    # The fabrication is gone: no undeclared value is asserted to be millivolts,
    # and no conductance in nS is fitted from unit-less voltages.
    assert "V = 0.3 mV" not in text
    assert "V = 0.52 mV" not in text
    assert w.fit() is None
    assert "nS" not in w._headline.text()

    # The refusal is not silent: each recording is listed with its value in the
    # honest form and the reason, and the user is pointed at the fix.
    assert "V = 0.3 (units unknown)" in text
    assert "V = 0.52 (units unknown)" in text
    assert "voltage units (unknown) -> conductance not in nS" in text
    assert "Calibration" in text

    exl = w.exclusions()
    assert [e.v_units for e in exl] == ["", ""]
    assert [e.v_value for e in exl] == pytest.approx([0.30, 0.52], rel=1e-5)


def test_iv_window_and_calibration_panel_speak_with_one_voice(qapp, qtbot, tmp_path):
    """Two surfaces, one session, ONE claim: the reason string the I-V window
    prints for an undeclared-units voltage is literally the note the Calibration
    panel (CF-5) shows for the same document."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, v_raw=0.30, i_pA=-60.0, name="b1.h5"))
    session.add_document(_doc(tmp_path, v_raw=0.52, i_pA=-104.0, name="b2.h5"))

    cal = CalibrationPanel(session)
    qtbot.addWidget(cal)
    cal.set_selection(doc_id, 1)
    assert "units unknown" in cal._cond_voltage.text()
    assert cal._cond_g.text() == "—"
    cal_reason = cal._cond_note.text()

    w = IVWindow(session)
    qtbot.addWidget(w)
    (e, _) = w.exclusions()
    # The wording is shared, not merely similar: both come from iv.conductance_note.
    assert e.reason in cal_reason
    assert e.reason == iv.conductance_note("pA", "")
    assert e.reason in w.details_text()


# ================================================================================================
# declared units keep working — mV directly, and V scaled by 1000 (the ABF path)
# ================================================================================================

def test_declared_mV_recordings_still_fit_and_undeclared_one_is_excluded(qapp, qtbot, tmp_path):
    """A mixed session: the declared-mV pair fits the known 100 nS line; the
    undeclared recording is excluded WITH its value, never silently dropped."""
    session = Session()
    session.add_document(_doc(tmp_path, v_raw=-500.0, i_pA=-50_000.0, name="m1.abf",
                              v_units="mV"))
    session.add_document(_doc(tmp_path, v_raw=-1000.0, i_pA=-100_000.0, name="m2.abf",
                              v_units="mV"))
    session.add_document(_doc(tmp_path, v_raw=0.52, i_pA=-104.0, name="u.h5"))

    w = IVWindow(session)
    qtbot.addWidget(w)

    assert len(w.points()) == 2
    assert w.fit() is not None
    assert w.fit().conductance_nS == pytest.approx(100.0, rel=1e-3)

    text = w.details_text()
    assert "V = -500 mV" in text and "V = -1000 mV" in text     # declared: still mV
    assert "V = 0.52 mV" not in text                            # undeclared: never mV
    assert "u.h5" in text and "V = 0.52 (units unknown)" in text


def test_declared_V_units_scale_to_mV_as_before(qapp, qtbot, tmp_path):
    """Units "V" (the calibrated ABF voltage path) still convert x1000 to mV."""
    session = Session()
    session.add_document(_doc(tmp_path, v_raw=-0.5, i_pA=-50_000.0, name="v1.abf",
                              v_units="V"))
    session.add_document(_doc(tmp_path, v_raw=-1.0, i_pA=-100_000.0, name="v2.abf",
                              v_units="V"))

    w = IVWindow(session)
    qtbot.addWidget(w)
    assert [p.v_volts for p in w.points()] == pytest.approx([-0.5, -1.0])
    assert all(p.voltage_source == "measured" for p in w.points())
    assert w.fit().conductance_nS == pytest.approx(100.0, rel=1e-3)
    assert w.exclusions() == []


# ================================================================================================
# the pure resolver — the invariant itself
# ================================================================================================

def test_resolver_refuses_mV_for_undeclared_units(qapp, tmp_path):
    """``applied_voltage_mV`` answers ONLY in mV, so undeclared units get None;
    ``applied_voltage_readout`` still hands out the value with its honest units;
    ``iv_point`` never fabricates a point (its documented contract)."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, v_raw=0.52, i_pA=-104.0, name="u.h5"))

    assert iv.applied_voltage_mV(session, doc_id) is None
    got = iv.applied_voltage_readout(session, doc_id)
    assert got == iv.AppliedVoltage(value=pytest.approx(0.52), units="", source="measured")
    assert iv.iv_point(session, doc_id) is None

    (e,) = iv.iv_exclusions(session)
    assert (e.v_value, e.v_units) == (pytest.approx(0.52), "")
    assert e.reason == iv.conductance_note("pA", "")


def test_declaring_units_in_calibration_restores_the_point(qapp, qtbot, tmp_path):
    """The documented fix (USER_GUIDE: set the units in Calibration) works: once
    the voltage channel is calibrated to "V", the recording joins the fit at the
    now-meaningful voltage and leaves the excluded list."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, v_raw=0.52, i_pA=-104.0, name="u.h5"))

    w = IVWindow(session)
    qtbot.addWidget(w)
    assert w.points() == [] and len(w.exclusions()) == 1

    session.set_calibration(
        doc_id, 0,
        ChannelCalibration(gain=1.0, offset=0.0, target_units="V",
                           provenance=Provenance.MEASURED),
        write_sidecar=False)
    w.refresh()

    assert w.exclusions() == []
    (p,) = w.points()
    assert p.v_volts == pytest.approx(0.52)
    assert "V = 520 mV" in w.details_text()
