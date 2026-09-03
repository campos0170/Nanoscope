"""S3 units-honesty, the pure core: a unit note names what is actually blocking,
and there is exactly ONE table of current-unit conversions in the app.

Two invariants, both of them second instances of rules earlier sweeps fixed
elsewhere (CF-5 / EXPL2-5 pinned "never assert a unit the file did not declare";
these pin "never blame the wrong unit" and "never keep a private copy of the unit
table"):

1. **A refusal names the thing that is actually blocking it.**
   :func:`iv.conductance_note` answers "why is there no nS number", which BOTH
   units can block -- so when both are undeclared (the state every fresh HDF5/CSV
   channel is in) it must name both, not stop at the first. And the I-V
   exclusion, whose gate is the voltage units ALONE
   (:func:`iv._applied_voltage_mV`), must use a reason that names the voltage --
   :func:`iv.voltage_units_note` -- because the current units never exclude
   anything: a point with undeclared current units plots and fits, it merely has
   no conductance in nS.

   The shipped code did the opposite: ``iv_exclusions`` borrowed
   ``conductance_note``, which tested the CURRENT units first and returned on the
   first failure, so a freshly opened HDF5 was refused with "current units
   (unknown)" while the headline above it said the voltage units were missing.
   Declaring the current units as the row asked left the recording excluded
   (JHT-02).

2. **One unit table.** ``iv._I_TO_A`` is what the module docstring calls "the
   single place that scale lives". :func:`iv.current_scale` is the seam that lets
   a GUI convert currents without copying it; the I-V window's copy omitted "A"
   and therefore dropped a channel declared in amperes as "unknown" while
   ``to_nS`` was converting that same declaration on the Calibration panel
   (JHT-03). The parity test below fails the moment a table drifts again.
"""

from __future__ import annotations

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


# ================================================================================================
# builder — the HDF5 shape: a VOLTAGE + CURRENT recording with units we control
# ================================================================================================

def _doc(tmp_path, *, name: str, v_raw: float, i_raw: float,
         v_units: str = "", i_units: str = "", n: int = 2048) -> TraceDocument:
    """Two channels whose raw units default to ``""`` — exactly what io/hdf5.py and
    io/text.py produce (docs/USER_GUIDE.md line 88: units are not declared in the
    file, the user sets them in Calibration)."""
    volt = np.ascontiguousarray(np.full(n, v_raw, dtype=np.float32))
    cur = np.ascontiguousarray(np.full(n, i_raw, dtype=np.float32))
    v_info = ChannelInfo(index=0, name="voltage", raw_units=v_units, kind=ChannelKind.VOLTAGE)
    c_info = ChannelInfo(index=1, name="current", raw_units=i_units, kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
        channels=(v_info, c_info))
    return TraceDocument(meta=meta, channels=(volt, cur))


# ================================================================================================
# 1a. conductance_note — the whole bill, not the first line of it
# ================================================================================================

def test_conductance_note_names_both_units_when_both_are_undeclared():
    """The default state of every .h5/.csv. Naming only the current units sent the
    user to fill in one box, after which the note flipped to the other one."""
    note = iv.conductance_note("", "")
    assert "current units (unknown)" in note
    assert "voltage units (unknown)" in note
    assert note.endswith("-> conductance not in nS")


def test_conductance_note_single_failures_are_unchanged():
    """The one-missing-unit wording is a contract with the Calibration panel and the
    I-V exclusion list; it must not drift."""
    assert iv.conductance_note("", "mV") == "current units (unknown) -> conductance not in nS"
    assert iv.conductance_note("pA", "") == "voltage units (unknown) -> conductance not in nS"
    assert iv.conductance_note("pA", "mV") == "pA / mV -> nS"
    # An unrecognised spelling is quoted back rather than called "(unknown)".
    assert "counts" in iv.conductance_note("pA", "counts")


# ================================================================================================
# 1b. voltage_units_note — the I-V gate's own voice
# ================================================================================================

def test_voltage_units_note_names_the_voltage_only():
    assert iv.voltage_units_note("") == "voltage units (unknown) -> conductance not in nS"
    assert "current" not in iv.voltage_units_note("")
    assert "counts" in iv.voltage_units_note("counts")


def test_voltage_units_note_is_empty_when_the_voltage_units_are_usable():
    """Nothing to refuse: mV and V are exactly what `_applied_voltage_mV` accepts."""
    assert iv.voltage_units_note("mV") == ""
    assert iv.voltage_units_note("V") == ""


def test_voltage_units_note_matches_conductance_notes_voltage_clause():
    """One voice: the I-V refusal and the Calibration panel's note are the same words
    for the same fact, so the two surfaces cannot drift apart."""
    assert iv.voltage_units_note("") == iv.conductance_note("pA", "")
    assert iv.voltage_units_note("counts") == iv.conductance_note("nA", "counts")


# ================================================================================================
# 1c. iv_exclusions — the reason is the GATE's reason, whatever the current units are
# ================================================================================================

def test_exclusion_reason_names_the_voltage_on_a_fresh_hdf5(tmp_path):
    """JHT-02: no units declared anywhere — the reason must still name the voltage,
    because the voltage units are the only thing that excludes the recording."""
    session = Session()
    session.add_document(_doc(tmp_path, name="pore.h5", v_raw=-600.0, i_raw=-60000.0))

    (e,) = iv.iv_exclusions(session)
    assert e.reason == "voltage units (unknown) -> conductance not in nS"
    assert "current units" not in e.reason


def test_exclusion_reason_is_the_same_whatever_the_current_units_say(tmp_path):
    """The row's advice has to be actionable: declaring the current units (what the
    old reason asked for) changes nothing, so it must never have been asked for."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, name="pore.h5", v_raw=-600.0, i_raw=-60000.0))
    before = iv.iv_exclusions(session)[0].reason

    session.set_calibration(doc_id, 1,
                            ChannelCalibration(gain=1.0, offset=0.0, target_units="pA"))
    after = iv.iv_exclusions(session)
    assert len(after) == 1                    # still excluded — nothing was fixed
    assert after[0].reason == before          # ...and the reason never moved


def test_declaring_the_voltage_units_is_what_clears_the_exclusion(tmp_path):
    """The falsifiable half: follow the reason and the recording joins the fit,
    with the current units still undeclared."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, name="pore.h5", v_raw=-600.0, i_raw=-60000.0))
    session.set_calibration(doc_id, 0,
                            ChannelCalibration(gain=1.0, offset=0.0, target_units="mV"))

    assert iv.iv_exclusions(session) == []
    (pt,) = iv.iv_points(session)
    assert pt.v_volts == pytest.approx(-0.6, rel=1e-3)
    assert pt.current_units == ""             # undeclared current units cost only the nS


# ================================================================================================
# 2. one table: current_scale IS _I_TO_A
# ================================================================================================

def test_current_scale_covers_exactly_the_modules_own_unit_table():
    """The parity assertion the I-V window's private ``to_nA`` failed: every unit the
    module can turn into nS can also be converted between units, and no other."""
    for unit in iv._I_TO_A:
        assert iv.current_scale(unit, "nA") is not None, unit
    for unknown in ("", "?", "counts", "V", "nS"):
        assert iv.current_scale(unknown, "nA") is None, unknown
        assert iv.current_scale("pA", unknown) is None, unknown


def test_current_scale_factors_are_the_physics():
    assert iv.current_scale("pA", "nA") == pytest.approx(1e-3)
    assert iv.current_scale("nA", "nA") == pytest.approx(1.0)
    assert iv.current_scale("uA", "nA") == pytest.approx(1e3)
    assert iv.current_scale("A", "nA") == pytest.approx(1e9)
    assert iv.current_scale("nA", "pA") == pytest.approx(1e3)


def test_current_scale_agrees_with_to_nS_on_amperes():
    """The two consumers of ``_I_TO_A`` must not disagree about one declaration: a
    100 nS pore at -0.6 V reads 100 nS whether its current is called A or pA."""
    i_A, v_V = -6e-8, -0.6
    assert iv.to_nS(i_A / v_V, "A", "V") == pytest.approx(100.0)
    i_pA = i_A * iv.current_scale("A", "pA")
    assert iv.to_nS(i_pA / v_V, "pA", "V") == pytest.approx(100.0)
