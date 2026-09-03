"""Tests for :mod:`nanoscope.analysis.iv` -- the pure I-V / conductance core.

The load-bearing claims:
  1. an ohmic pore ``I = G*V + leak`` is recovered by ``iv_fit`` -- slope is the
     conductance, intercept the leak, both to numerical precision;
  2. the units are right: the SAME physical conductance comes out in nS whether
     the current is pA over mV or nA over volts;
  3. ``V == 0`` is guarded (no infinity), and a single-point / single-voltage I-V
     raises rather than inventing a slope;
  4. ``open_pore_current`` is a robust median -- events do not move it;
  5. ``iv_point`` reads voltage from a VOLTAGE channel when present and from
     ``nominal_voltage_mV`` otherwise, and the calibrated current gain flows
     straight through to the reported open-pore current.
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


# ================================================================================================
# builders
# ================================================================================================

def _single_current_doc(
    tmp_path: Path,
    *,
    i_open_pA: float,
    nominal_mV: float,
    name: str,
    n: int = 4096,
    seed: int = 0,
) -> TraceDocument:
    """A current-only recording whose open-pore level is ``i_open_pA`` (+ noise).

    Voltage is carried only as ``nominal_voltage_mV`` (the HDF5/CSV case), so
    ``iv_point`` must fall back to it.
    """
    rng = np.random.default_rng(seed)
    arr = np.ascontiguousarray(i_open_pA + rng.normal(0.0, 1.0, n), dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=arr.size,
        sample_rate_hz=50_000.0,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
        nominal_voltage_mV=nominal_mV,
        voltage_provenance=Provenance.FILENAME,
    )
    return TraceDocument(meta=meta, channels=(arr,))


def _current_plus_voltage_doc(
    tmp_path: Path,
    *,
    i_open_pA: float,
    v_raw_mV: float,
    name: str,
    n: int = 4096,
) -> TraceDocument:
    """Two-channel ABF-shaped doc: a CURRENT (pA) and a VOLTAGE (mV) channel.

    The voltage channel is stored ``v_raw_mV`` so a downstream 0.1 gain lands it
    on the true command voltage (mirrors the ABF IN14 10x-too-large convention).
    """
    cur = np.ascontiguousarray(np.full(n, i_open_pA, dtype=np.float32))
    volt = np.ascontiguousarray(np.full(n, v_raw_mV, dtype=np.float32))
    cur_info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    volt_info = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=n,
        sample_rate_hz=50_000.0,
        rate_provenance=Provenance.FILE_META,
        channels=(cur_info, volt_info),
        nominal_voltage_mV=round(v_raw_mV * 0.1 / 10.0) * 10.0,
        voltage_provenance=Provenance.FILENAME,
    )
    return TraceDocument(meta=meta, channels=(cur, volt))


# ================================================================================================
# open_pore_current
# ================================================================================================

def test_open_pore_current_is_event_robust_median():
    rng = np.random.default_rng(1)
    x = -1000.0 + rng.normal(0.0, 5.0, 20_000)
    # Punch deep blockade events into ~20% of the trace; the median must ignore them.
    x[:4000] += 400.0
    got = iv.open_pore_current(x)
    assert got == pytest.approx(-1000.0, abs=2.0)


def test_open_pore_current_signed_and_units_agnostic():
    # Signed: negative bias -> negative open pore, returned verbatim (no abs()).
    assert iv.open_pore_current(np.full(100, -812.0)) == pytest.approx(-812.0)


def test_open_pore_current_rejects_empty_and_nonfinite_only():
    with pytest.raises(ValueError):
        iv.open_pore_current(np.empty(0))
    with pytest.raises(ValueError):
        iv.open_pore_current(np.array([np.nan, np.inf]))


def test_open_pore_current_unknown_method_raises():
    with pytest.raises(ValueError):
        iv.open_pore_current(np.zeros(10), method="p50")


# ================================================================================================
# conductance + unit scaling
# ================================================================================================

def test_conductance_is_raw_ratio_and_guards_zero_voltage():
    assert iv.conductance(-100.0, -500.0) == pytest.approx(0.2)
    assert iv.conductance(50.0, 0.0) is None
    assert iv.conductance(50.0, float("nan")) is None


def test_unit_scaling_pA_per_mV_and_nA_per_V_agree_in_nS():
    # 100 nS expressed two ways: 100 pA at 1 mV, and 100 nA at 1 V (both = 1e-7 S).
    g_pA_mV = iv.conductance_in_nS(100.0, "pA", 1.0)
    g_nA_V = iv.to_nS(iv.conductance(100.0, 1.0), "nA", "V")
    assert g_pA_mV == pytest.approx(100.0)
    assert g_nA_V == pytest.approx(100.0)


def test_to_nS_scale_factors():
    # pA/mV x1, nA/mV x1000, pA/V x1e-3, nA/V x1.
    assert iv.to_nS(1.0, "pA", "mV") == pytest.approx(1.0)
    assert iv.to_nS(1.0, "nA", "mV") == pytest.approx(1000.0)
    assert iv.to_nS(1.0, "pA", "V") == pytest.approx(1e-3)
    assert iv.to_nS(1.0, "nA", "V") == pytest.approx(1.0)


def test_to_nS_unknown_units_is_none():
    assert iv.to_nS(1.0, "", "mV") is None
    assert iv.to_nS(1.0, "pA", "counts") is None
    assert iv.conductance_in_nS(100.0, "widgets", 100.0) is None


def test_conductance_in_nS_guards_zero_voltage():
    assert iv.conductance_in_nS(100.0, "pA", 0.0) is None


def test_conductance_note_explains_units():
    assert iv.conductance_note("pA", "mV") == "pA / mV -> nS"
    assert "not in nS" in iv.conductance_note("", "mV")
    assert "not in nS" in iv.conductance_note("pA", "counts")


# ================================================================================================
# iv_fit -- slope is the conductance
# ================================================================================================

def test_iv_fit_recovers_conductance_and_leak_pA():
    # Ohmic pore: I[pA] = G[nS]*V[mV] + leak.  In volts the slope is G*1000 pA/V.
    G_nS, leak_pA = 100.0, -15.0
    v_mV = np.array([-1000.0, -700.0, -600.0, -500.0, -200.0])
    v_volts = v_mV / 1000.0
    i_pA = G_nS * v_mV + leak_pA

    fit = iv.iv_fit(v_volts, i_pA, current_units="pA")
    assert fit.n == 5
    assert fit.r2 == pytest.approx(1.0, abs=1e-9)
    # slope is pA/V; conductance in nS is the physical answer.
    assert fit.slope == pytest.approx(G_nS * 1000.0)
    assert fit.conductance_nS == pytest.approx(G_nS)
    assert fit.intercept == pytest.approx(leak_pA)


def test_iv_fit_recovers_conductance_nA():
    # Same 100 nS pore in nA/V: slope in nA/V IS the nS value.
    G_nS = 100.0
    v_volts = np.array([-1.0, -0.7, -0.5, -0.2])
    i_nA = (G_nS * (v_volts * 1000.0)) / 1000.0  # nA = nS*mV/1000
    fit = iv.iv_fit(v_volts, i_nA, current_units="nA")
    assert fit.slope == pytest.approx(G_nS)
    assert fit.conductance_nS == pytest.approx(G_nS)


def test_iv_fit_unknown_units_leaves_nS_none():
    fit = iv.iv_fit([-1.0, -0.5], [-100.0, -50.0], current_units="")
    assert fit.conductance_nS is None
    assert fit.slope == pytest.approx(100.0)  # still a valid slope in raw/V


def test_iv_fit_single_point_raises():
    with pytest.raises(ValueError):
        iv.iv_fit([-0.5], [-50.0], current_units="pA")


def test_iv_fit_degenerate_voltage_raises():
    # Every recording at the same voltage: the slope is undefined.
    with pytest.raises(ValueError):
        iv.iv_fit([-0.5, -0.5, -0.5], [-50.0, -51.0, -49.0], current_units="pA")


def test_iv_fit_predict_is_the_fitted_line():
    fit = iv.iv_fit([-1.0, 0.0, 1.0], [-100.0, 0.0, 100.0], current_units="pA")
    assert np.allclose(fit.predict(np.array([-1.0, 1.0])), [-100.0, 100.0])


# ================================================================================================
# iv_point / iv_points -- reading a Session
# ================================================================================================

def test_iv_point_uses_nominal_voltage_when_current_only(tmp_path):
    session = Session()
    doc = _single_current_doc(tmp_path, i_open_pA=-50.0, nominal_mV=-500.0, name="a.dat")
    doc_id = session.add_document(doc)
    pt = iv.iv_point(session, doc_id)
    assert pt is not None
    assert pt.v_volts == pytest.approx(-0.5)
    assert pt.i_open == pytest.approx(-50.0, abs=0.5)
    assert pt.current_units == "pA"
    assert pt.voltage_source == "nominal"


def test_iv_point_prefers_measured_voltage_channel(tmp_path):
    session = Session()
    # Raw voltage 5000 mV -> default 0.1 gain -> 500 mV measured command voltage.
    doc = _current_plus_voltage_doc(tmp_path, i_open_pA=-250.0, v_raw_mV=-5000.0, name="b.abf")
    doc_id = session.add_document(doc)
    pt = iv.iv_point(session, doc_id)
    assert pt is not None
    assert pt.voltage_source == "measured"
    assert pt.v_volts == pytest.approx(-0.5, abs=1e-3)  # -5000 mV * 0.1 / 1000
    assert pt.i_open == pytest.approx(-250.0, abs=0.5)


def test_iv_point_reflects_calibrated_current_gain(tmp_path):
    session = Session()
    doc = _single_current_doc(tmp_path, i_open_pA=-4.0, nominal_mV=-1000.0, name="c.dat")
    doc_id = session.add_document(doc)
    # Dial a 25x current gain: the open-pore current the point reports scales with it.
    session.set_calibration(doc_id, 0, ChannelCalibration(gain=25.0, offset=0.0, target_units="pA"))
    pt = iv.iv_point(session, doc_id)
    assert pt.i_open == pytest.approx(-100.0, abs=1.0)


def test_iv_point_none_without_current_channel(tmp_path):
    session = Session()
    volt = np.ascontiguousarray(np.full(1024, -700.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=tmp_path / "volt_only.dat", format_name="synthetic", n_samples=1024,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(volt,))
    doc_id = session.add_document(doc)
    assert iv.iv_point(session, doc_id) is None


def test_iv_point_none_without_known_voltage(tmp_path):
    session = Session()
    arr = np.ascontiguousarray(np.full(1024, -30.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "no_v.dat", format_name="synthetic", n_samples=1024,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=None)
    doc = TraceDocument(meta=meta, channels=(arr,))
    doc_id = session.add_document(doc)
    assert iv.iv_point(session, doc_id) is None


def test_iv_points_gathers_a_line_that_fits_to_known_conductance(tmp_path):
    """3 current-only docs on I[pA] = 100*V[mV]; the gathered fit reads 100 nS."""
    session = Session()
    G_nS = 100.0
    for mV in (-200.0, -500.0, -1000.0):
        doc = _single_current_doc(
            tmp_path, i_open_pA=G_nS * mV, nominal_mV=mV, name=f"pore_{int(-mV)}.dat")
        session.add_document(doc)

    pts = iv.iv_points(session)
    assert len(pts) == 3
    fit = iv.iv_fit([p.v_volts for p in pts], [p.i_open for p in pts], current_units="pA")
    assert fit.conductance_nS == pytest.approx(G_nS, rel=1e-3)
    assert abs(fit.intercept) < 5.0  # leak ~ 0


# ================================================================================================
# The ABF voltage encoding is DETECTED, not assumed.
# ================================================================================================
def test_voltage_gain_is_detected_from_the_channels_own_magnitude():
    """A blanket 0.1 was right for one instrument configuration and 10x wrong for the other.

    The owner's FERRITIN ABFs store the command channel 10x too large (-5999 mV on a
    -600 mV run), which is where the 0.1 default came from. His HOST-GUEST ABFs store it
    directly (75.07 mV on a 75 mV run) -- and 0.1 turned those into 7.507 mV, which put a
    0-100 mV V-dependence series on a 0-10 mV axis and made every conductance 10x too
    large (an apo file read G = 2.724 nS where the honest number is 0.273 nS).

    Detection is on magnitude: a bias past _MAX_PLAUSIBLE_BIAS_MV would destroy the
    membrane, so such a reading must be the inflated encoding.
    """
    from nanoscope.core.calibration import _MAX_PLAUSIBLE_BIAS_MV, CalibrationSet
    from nanoscope.core.types import ChannelInfo, ChannelKind

    v = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)

    # Inflated (ferritin): 10x correction, and it lands on the real bias.
    assert CalibrationSet.default_for(v, -5999.15).gain == pytest.approx(0.1)
    assert -5999.15 * CalibrationSet.default_for(v, -5999.15).gain == pytest.approx(-600, abs=1)

    # Direct (host-guest): taken as-is.
    for raw in (75.07, 100.1, 50.11, 25.09, 0.009):
        assert CalibrationSet.default_for(v, raw).gain == pytest.approx(1.0), raw

    # No measurement available -> the historical default stands, so a caller that cannot
    # reach the samples behaves exactly as before.
    assert CalibrationSet.default_for(v).gain == pytest.approx(0.1)

    # The boundary is the stated one, and both branches stay overridable/provenanced.
    assert CalibrationSet.default_for(v, _MAX_PLAUSIBLE_BIAS_MV - 1).gain == pytest.approx(1.0)
    assert CalibrationSet.default_for(v, _MAX_PLAUSIBLE_BIAS_MV + 1).gain == pytest.approx(0.1)
    assert CalibrationSet.default_for(v, 75.07).note
    assert CalibrationSet.default_for(v, -5999.15).note


def test_a_current_channel_is_unaffected_by_the_voltage_detection():
    from nanoscope.core.calibration import CalibrationSet
    from nanoscope.core.types import ChannelInfo, ChannelKind
    i = ChannelInfo(index=0, name="IN11", raw_units="pA", kind=ChannelKind.CURRENT)
    assert CalibrationSet.default_for(i, 9999.0).gain == pytest.approx(1.0)
    assert CalibrationSet.default_for(i, 12.0).gain == pytest.approx(1.0)
