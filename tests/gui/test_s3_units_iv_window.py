"""S3 units-honesty on the I-V / Conductance window: the refusal names the box the
user has to change, and one unit declaration gets one verdict.

Through the REAL :class:`~nanoscope.gui.iv_panel.IVWindow` (and the real
:class:`~nanoscope.gui.calibration_panel.CalibrationPanel` where two surfaces are
compared), on the recording shape every HDF5/CSV arrives in: no units declared on
any channel.

JHT-02 — a freshly opened HDF5 was refused with a headline naming the VOLTAGE
units and a per-recording row naming the CURRENT units. The gate is the voltage
alone, so declaring the current units the row asked for left the recording
excluded and merely flipped the reason. One message must not give two
instructions, and the one it gives must be the one that works.

JHT-03 — a current channel declared in amperes ("A" is in ``iv._I_TO_A``, and the
Calibration panel prints "A / mV -> nS" and a conductance for it) was thrown out
of the mixed-unit fit as "current units “A” unknown", because the window kept a
private conversion table that omitted "A". With two recordings open the fit
vanished and the window advised "open another at a different voltage" above a
Points list naming two recordings at two different voltages.

The general rule these pin, at the seam every surface goes through:
:func:`nanoscope.analysis.iv.current_scale` is the ONLY current-unit conversion
(no window keeps a copy), and a dropped point is named as dropped everywhere the
window lists or counts its points.
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
# builder — the HDF5 shape: VOLTAGE + CURRENT, units undeclared unless we say otherwise
# ================================================================================================

def _doc(tmp_path, *, name: str, v_raw: float, i_raw: float,
         v_units: str = "", i_units: str = "", n: int = 4096) -> TraceDocument:
    """io/hdf5.py leaves ``raw_units=""`` on EVERY channel; that is the default here."""
    volt = np.ascontiguousarray(np.full(n, v_raw, dtype=np.float32))
    cur = np.ascontiguousarray(np.full(n, i_raw, dtype=np.float32))
    v_info = ChannelInfo(index=0, name="voltage", raw_units=v_units, kind=ChannelKind.VOLTAGE)
    c_info = ChannelInfo(index=1, name="current", raw_units=i_units, kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
        channels=(v_info, c_info))
    return TraceDocument(meta=meta, channels=(volt, cur))


def _set_units(session: Session, doc_id: str, channel: int, units: str,
               gain: float = 1.0) -> None:
    session.set_calibration(
        doc_id, channel, ChannelCalibration(gain=gain, offset=0.0, target_units=units))


# ================================================================================================
# JHT-02 — the whole message points at ONE box, and it is the box that works
# ================================================================================================

def test_fresh_hdf5_exclusion_row_blames_the_voltage_not_the_current(qapp, qtbot, tmp_path):
    """The repro's core: headline, block header and per-recording row must agree."""
    session = Session()
    session.add_document(_doc(tmp_path, name="pore.h5", v_raw=-600.0, i_raw=-60000.0))

    w = IVWindow(session)
    qtbot.addWidget(w)
    text = w.details_text()

    assert "No recording has a voltage in declared units" in text     # the headline
    (e,) = w.exclusions()
    assert e.reason == "voltage units (unknown) -> conductance not in nS"
    assert "current units" not in text            # nothing in the window blames the current
    # The values are still shown, both in their honest form — neither is given a unit.
    assert "V = -600 (units unknown)" in text
    assert "I₀ = -6e+04 (units unknown)" in text


def test_following_the_row_does_nothing_and_the_row_never_asks_for_it(qapp, qtbot, tmp_path):
    """Declaring the CURRENT units (what the shipped row asked for) leaves the
    recording excluded with an unchanged reason — so the reason must not name it."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, name="pore.h5", v_raw=-600.0, i_raw=-60000.0))

    w = IVWindow(session)
    qtbot.addWidget(w)
    before = w.exclusions()[0].reason
    assert "current" not in before

    _set_units(session, doc_id, 1, "pA")          # follow the old advice, change nothing else
    w.refresh()
    assert w.points() == []
    assert w.exclusions()[0].reason == before     # nothing moved; the advice was not this


def test_following_the_named_box_clears_the_exclusion(qapp, qtbot, tmp_path):
    """...and the box the window DOES name is the one that fixes it, with the current
    units still undeclared."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, name="pore.h5", v_raw=-600.0, i_raw=-60000.0))

    w = IVWindow(session)
    qtbot.addWidget(w)
    _set_units(session, doc_id, 0, "mV")
    w.refresh()

    assert w.exclusions() == []
    assert len(w.points()) == 1
    # No unit is invented for the still-undeclared current channel, on the axis either.
    assert "units unknown" in w._plot.getAxis("left").labelText


def test_iv_row_and_calibration_panel_agree_on_a_fresh_hdf5(qapp, qtbot, tmp_path):
    """Two surfaces, one session, one document, no units anywhere: the I-V row's
    reason is a claim the Calibration panel also makes (it adds the current units,
    which cost the nS conductance there but exclude nothing here)."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, name="pore.h5", v_raw=-600.0, i_raw=-60000.0))

    cal = CalibrationPanel(session)
    qtbot.addWidget(cal)
    cal.set_selection(doc_id, 1)
    cal_note = cal._cond_note.text()

    w = IVWindow(session)
    qtbot.addWidget(w)
    (e,) = w.exclusions()

    # The Calibration panel names BOTH undeclared units (both block a nS number)...
    assert "current units (unknown)" in cal_note and "voltage units (unknown)" in cal_note
    # ...and the I-V refusal names the one that is its gate. No contradiction: the
    # voltage clause is word-for-word the same on both surfaces.
    assert iv.voltage_units_note("") in cal_note
    assert e.reason == iv.voltage_units_note("")


# ================================================================================================
# JHT-03 — amperes are amperes on every surface, in every mix
# ================================================================================================

def _two_pore_session(tmp_path) -> tuple[Session, list[str]]:
    """A 100 nS pore measured at -600 and -300 mV, voltages declared, currents open."""
    session = Session()
    ids = []
    for mV, i_pA in ((-600.0, -60000.0), (-300.0, -30000.0)):
        doc_id = session.add_document(
            _doc(tmp_path, name=f"pore_{int(-mV)}.h5", v_raw=mV, i_raw=i_pA))
        _set_units(session, doc_id, 0, "mV")
        ids.append(doc_id)
    return session, ids


def test_amperes_survive_a_mixed_unit_fit(qapp, qtbot, tmp_path):
    """One recording in A, one in pA: both belong on the nA axis and the fit is the
    same 100 nS the all-amperes control reports."""
    session, ids = _two_pore_session(tmp_path)
    _set_units(session, ids[0], 1, "A", gain=1e-12)     # -6e-8 A
    _set_units(session, ids[1], 1, "A", gain=1e-12)     # -3e-8 A

    w = IVWindow(session)
    qtbot.addWidget(w)
    control = w.fit()
    assert control is not None
    assert control.conductance_nS == pytest.approx(100.0, rel=1e-3)

    _set_units(session, ids[1], 1, "pA", gain=1.0)      # ...now mix the units
    w.refresh()
    mixed = w.fit()
    assert mixed is not None, "the amperes recording was dropped from the mixed fit"
    assert mixed.n == 2
    assert mixed.conductance_nS == pytest.approx(100.0, rel=1e-3)

    text = w.details_text()
    assert "unknown — dropped" not in text
    assert "Only one recording" not in text
    assert "I-V over 2 recording(s), current in nA" in text


@pytest.mark.parametrize("other", ["pA", "nA", "uA"])
def test_amperes_pair_with_every_other_declared_current_unit(qapp, qtbot, tmp_path, other):
    """Not a pA special case: "A" converts against every unit ``iv._I_TO_A`` holds."""
    session, ids = _two_pore_session(tmp_path)
    _set_units(session, ids[0], 1, "A", gain=1e-12)
    _set_units(session, ids[1], 1, other, gain=iv.current_scale("pA", other))

    w = IVWindow(session)
    qtbot.addWidget(w)
    fit = w.fit()
    assert fit is not None and fit.n == 2
    assert fit.conductance_nS == pytest.approx(100.0, rel=1e-3)


def test_the_window_keeps_no_unit_table_of_its_own(qapp, qtbot, tmp_path):
    """The rule, not the instance: every unit ``iv`` can scale, the window converts —
    a second table in the GUI is what let "A" mean two things at once."""
    for unit in iv._I_TO_A:
        partner = "nA" if unit == "pA" else "pA"    # force the MIXED-unit branch
        pt = iv.IVPoint(v_volts=-0.6, i_open=-1.0, current_units=unit,
                        voltage_source="measured", doc_id="d", label="x")
        other = iv.IVPoint(v_volts=-0.3, i_open=-1.0, current_units=partner,
                           voltage_source="measured", doc_id="e", label="y")
        _V, I, units, _notes, dropped = IVWindow._resolve_units([pt, other])
        assert units == "nA"
        assert dropped == [], f"{unit} was dropped by the window but iv can convert it"
        assert I[0] == pytest.approx(-1.0 * iv.current_scale(unit, "nA"))


# ================================================================================================
# a genuinely unconvertible point is still dropped — and SAID to be, everywhere
# ================================================================================================

def test_a_dropped_point_is_named_dropped_in_the_points_list(qapp, qtbot, tmp_path):
    """Undeclared current units on one of two recordings: it cannot go on a shared
    nA axis, so it is dropped — but the Points list and the one-recording advice
    must not read as though it were fitted or as though it were not open."""
    session, ids = _two_pore_session(tmp_path)
    _set_units(session, ids[0], 1, "", gain=1.0)        # undeclared: unconvertible
    _set_units(session, ids[1], 1, "pA", gain=1.0)

    w = IVWindow(session)
    qtbot.addWidget(w)
    text = w.details_text()

    assert "I-V over 1 recording(s)" in text
    assert "pore_600.h5" in text and "DROPPED, not in the fit" in text
    # The advice names the real cause and the real fix instead of "open another".
    assert "Only one recording reached the fit" in text
    assert "1 of the 2 recordings" in text
    assert "Calibration" in text


def test_a_fit_in_undeclared_units_labels_the_slope_and_the_axis(qapp, qtbot, tmp_path):
    """Undeclared current units cost the nS conductance and nothing else — the fit
    still runs, so its slope, leak and axis must name the unit state rather than
    print a bare "/V" the reader will fill in with pA."""
    session, ids = _two_pore_session(tmp_path)      # currents left undeclared

    w = IVWindow(session)
    qtbot.addWidget(w)
    fit = w.fit()
    assert fit is not None and fit.conductance_nS is None

    text = w.details_text()
    assert "(units unknown)/V" in text              # the slope
    assert "(units unknown) at 0 V" in text         # the leak
    assert "conductance not in nS" in text
    assert "(units unknown)/V" in w._headline.text()
    assert "units unknown" in w._plot.getAxis("left").labelText


def test_the_plain_one_recording_advice_survives_when_nothing_was_dropped(qapp, qtbot, tmp_path):
    """A single open recording is still simply a single open recording."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, name="solo.h5", v_raw=-600.0, i_raw=-60000.0))
    _set_units(session, doc_id, 0, "mV")
    _set_units(session, doc_id, 1, "pA")

    w = IVWindow(session)
    qtbot.addWidget(w)
    text = w.details_text()
    assert "Only one recording — open another at a different voltage" in text
    assert "DROPPED" not in text
