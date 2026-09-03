"""Tests for :mod:`nanoscope.analysis.voltage_dep` -- the pure voltage-dependence core.

The classic nanopore voltage series: the same pore recorded at a ladder of applied
voltages, detected in each, then a per-recording event summary plotted against the
command voltage. The load-bearing claims pinned here:

  1. :func:`vdep_points` emits exactly one ``(V, value)`` per open document that has
     BOTH a known applied voltage (via :func:`nanoscope.analysis.iv.iv_point`) AND a
     stored ``detect.schmitt`` result -- and the ``metric`` switch reads the right
     column/reduction (capture rate / dwell / depth / dI-I0 / count);
  2. ``accepted_only`` is honoured -- the curated-in events alone drive the metric;
  3. :func:`vdep_fit` recovers an INJECTED trend: a straight line for a metric linear
     in V, and the ``A*exp(V/V0)`` capture-rate model for a rate we placed on that
     curve, both to numerical precision (R² -> 1);
  4. robustness is the contract -- a document missing either half of the pair (no
     voltage, or no detection) simply does not contribute a point, and an empty
     detection contributes to a count/rate (0) but not to a median;
  5. the error paths (unknown metric, unknown model, too few / non-positive points)
     raise rather than fabricate.

Metrics are injected DIRECTLY as hand-built :class:`~nanoscope.analysis.events.EventSet`
frames with exactly known medians/counts, so the recovered trend can be checked
against the numbers that went in (not against a detector's output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import voltage_dep
from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)

# The exponential capture-rate model we inject and expect vdep_fit to recover.
_EXP_A = 10.0
_EXP_V0 = -0.5   # negative: the rate rises as V grows more negative (|V| up)


# ================================================================================================
# builders
# ================================================================================================

def _current_doc(
    tmp_path: Path, *, nominal_mV: float, name: str, i_open_pA: float = -200.0, n: int = 2048
) -> TraceDocument:
    """A current-only recording carrying its applied voltage as ``nominal_voltage_mV``.

    That is the HDF5/CSV case where ``iv_point`` falls back to the filename/header
    voltage -- exactly the seam voltage-dependence reuses for the per-recording V.
    """
    arr = np.ascontiguousarray(np.full(n, i_open_pA, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _no_voltage_doc(tmp_path: Path, *, name: str, n: int = 1024) -> TraceDocument:
    """A current-only recording with NO known voltage (no nominal, no voltage channel)."""
    arr = np.ascontiguousarray(np.full(n, -30.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=None)
    return TraceDocument(meta=meta, channels=(arr,))


def _events_df(
    dwell_s, depth_pA, frac_depth, accepted, *, recording: str = "rec", pore_id: str = "P"
) -> pd.DataFrame:
    """A valid EventSet frame from equal-length per-row columns (SCHEMA dtypes).

    Only the fields the voltage-dependence metrics read (``dwell_s`` / ``depth_pA`` /
    ``frac_depth`` / ``accepted``, plus the count) vary per row; the rest carry
    harmless constants so the frame satisfies the 20-column contract.
    """
    n = len(dwell_s)
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": recording, "pore_id": pore_id,
            "start_idx": 100 * i, "end_idx": 100 * i + 50,
            "t_start_s": 1e-3 * i, "dwell_s": float(dwell_s[i]),
            "baseline_pA": -1000.0, "depth_pA": float(depth_pA[i]),
            "frac_depth": float(frac_depth[i]), "ecd_pA_s": 0.05,
            "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False,
            "accepted": bool(accepted[i]), "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _constant_df(
    *, n: int, dwell_s: float, depth_pA: float, frac_depth: float, accepted: bool = True
) -> pd.DataFrame:
    """``n`` identical rows -> every median is exactly the injected value."""
    return _events_df([dwell_s] * n, [depth_pA] * n, [frac_depth] * n, [accepted] * n)


def _install_detection(
    session: Session, doc_id: str, df: pd.DataFrame, *, duration_s: float, channel: int = 0
) -> EventSet:
    """Store a ``detect.schmitt`` result carrying ``df`` (rate needs analyzed_duration_s)."""
    ev = EventSet(df, {"analyzed_duration_s": float(duration_s), "recording": "rec"})
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=ev))
    return ev


# One recording per applied voltage; medians linear in V, capture rate on A*exp(V/V0).
# (V in volts, median dwell_s, median depth_pA, median frac_depth)
#   depth_pA  = -100 * V     (0 leak)  -> slope -100,  intercept 0
#   frac      = -0.1 * V     (0 leak)  -> slope -0.1,  intercept 0
#   dwell_s   = (V + 1.2)e-3           -> slope 1e-3,  intercept 1.2e-3  (falls with |V|)
_LADDER_MV = (-200.0, -500.0, -700.0, -1000.0)
_N_PER_DOC = 50


def _ladder_session(tmp_path: Path) -> Session:
    """4 current-only docs at -200/-500/-700/-1000 mV with injected, known metrics."""
    session = Session()
    for mV in _LADDER_MV:
        v = mV / 1000.0
        doc = _current_doc(tmp_path, nominal_mV=mV, name=f"pore_{int(-mV)}.dat")
        doc_id = session.add_document(doc)
        rate = _EXP_A * np.exp(v / _EXP_V0)                 # capture rate on the exp curve
        df = _constant_df(
            n=_N_PER_DOC, dwell_s=(v + 1.2) * 1e-3, depth_pA=-100.0 * v, frac_depth=-0.1 * v)
        _install_detection(session, doc_id, df, duration_s=_N_PER_DOC / rate)
    return session


def _expected(metric: str) -> dict[float, float]:
    """The injected metric value keyed by applied voltage (volts), for assertions."""
    out: dict[float, float] = {}
    for mV in _LADDER_MV:
        v = mV / 1000.0
        out[v] = {
            "median_dwell_s": (v + 1.2) * 1e-3,
            "median_depth": -100.0 * v,
            "median_frac_depth": -0.1 * v,
            "n_events": float(_N_PER_DOC),
            "capture_rate_hz": _EXP_A * np.exp(v / _EXP_V0),
        }[metric]
    return out


# ================================================================================================
# vdep_points -- one (V, value) per usable recording, per metric
# ================================================================================================

def test_vdep_points_one_per_recording_with_voltage_and_detection(tmp_path):
    session = _ladder_session(tmp_path)
    pts = voltage_dep.vdep_points(session, metric="median_depth")
    assert len(pts) == len(_LADDER_MV)
    # exactly the four applied voltages, in documents() order
    assert [round(p.v_volts, 6) for p in pts] == [mV / 1000.0 for mV in _LADDER_MV]
    assert all(isinstance(p.recording, str) and p.recording for p in pts)


@pytest.mark.parametrize(
    "metric", ["capture_rate_hz", "median_dwell_s", "median_depth", "median_frac_depth", "n_events"])
def test_metric_switch_selects_the_right_value(tmp_path, metric):
    session = _ladder_session(tmp_path)
    pts = voltage_dep.vdep_points(session, metric=metric)
    got = {round(p.v_volts, 6): p.value for p in pts}
    want = {round(v, 6): val for v, val in _expected(metric).items()}
    assert set(got) == set(want)
    for v in want:
        assert got[v] == pytest.approx(want[v], rel=1e-9, abs=1e-12)


def test_recording_name_is_display_name(tmp_path):
    session = _ladder_session(tmp_path)
    pts = voltage_dep.vdep_points(session, metric="n_events")
    names = {p.recording for p in pts}
    assert names == {f"pore_{int(-mV)}.dat" for mV in _LADDER_MV}


def test_unknown_metric_raises(tmp_path):
    session = _ladder_session(tmp_path)
    with pytest.raises(ValueError):
        voltage_dep.vdep_points(session, metric="median_snr")


# ================================================================================================
# accepted_only respected
# ================================================================================================

def test_accepted_only_changes_median_and_rate(tmp_path):
    """3 accepted (dwell 1e-3) + 3 rejected (dwell 9e-3) -> the curation flag moves both."""
    session = Session()
    doc_id = session.add_document(
        _current_doc(tmp_path, nominal_mV=-500.0, name="mixed.dat"))
    df = _events_df(
        dwell_s=[1e-3, 1e-3, 1e-3, 9e-3, 9e-3, 9e-3],
        depth_pA=[20.0] * 6, frac_depth=[0.02] * 6,
        accepted=[True, True, True, False, False, False])
    _install_detection(session, doc_id, df, duration_s=1.0)

    # median dwell: accepted-only sees 1e-3; all events see median([1,1,1,9,9,9]e-3)=5e-3
    acc = voltage_dep.vdep_points(session, metric="median_dwell_s", accepted_only=True)
    allv = voltage_dep.vdep_points(session, metric="median_dwell_s", accepted_only=False)
    assert acc[0].value == pytest.approx(1e-3)
    assert allv[0].value == pytest.approx(5e-3)

    # capture rate over a 1 s analysed window: 3 accepted vs 6 total
    racc = voltage_dep.vdep_points(session, metric="capture_rate_hz", accepted_only=True)
    rall = voltage_dep.vdep_points(session, metric="capture_rate_hz", accepted_only=False)
    assert racc[0].value == pytest.approx(3.0)
    assert rall[0].value == pytest.approx(6.0)


# ================================================================================================
# robustness -- a document missing either half contributes no point
# ================================================================================================

def test_doc_without_detection_is_skipped(tmp_path):
    session = Session()
    session.add_document(_current_doc(tmp_path, nominal_mV=-500.0, name="undetected.dat"))
    assert voltage_dep.vdep_points(session, metric="n_events") == []


def test_doc_without_voltage_is_skipped(tmp_path):
    session = Session()
    doc_id = session.add_document(_no_voltage_doc(tmp_path, name="no_v.dat"))
    _install_detection(session, doc_id, _constant_df(
        n=5, dwell_s=1e-3, depth_pA=20.0, frac_depth=0.02), duration_s=1.0)
    assert voltage_dep.vdep_points(session, metric="n_events") == []


def test_empty_detection_counts_but_has_no_median(tmp_path):
    """An empty event set contributes to a count/rate (0) but not to a median metric."""
    session = Session()
    doc_id = session.add_document(_current_doc(tmp_path, nominal_mV=-500.0, name="empty.dat"))
    _install_detection(session, doc_id, empty_df(), duration_s=2.0)

    assert voltage_dep.vdep_points(session, metric="median_dwell_s") == []   # no median
    n_pts = voltage_dep.vdep_points(session, metric="n_events")
    r_pts = voltage_dep.vdep_points(session, metric="capture_rate_hz")
    assert len(n_pts) == 1 and n_pts[0].value == 0.0
    assert len(r_pts) == 1 and r_pts[0].value == 0.0


def test_only_docs_with_both_halves_place(tmp_path):
    """A mixed session: only the recording with a voltage AND a detection places."""
    session = Session()
    good = session.add_document(_current_doc(tmp_path, nominal_mV=-700.0, name="good.dat"))
    _install_detection(session, good, _constant_df(
        n=7, dwell_s=4e-4, depth_pA=70.0, frac_depth=0.07), duration_s=1.0)
    session.add_document(_current_doc(tmp_path, nominal_mV=-300.0, name="novote.dat"))  # no detection
    novolt = session.add_document(_no_voltage_doc(tmp_path, name="nov.dat"))
    _install_detection(session, novolt, _constant_df(
        n=3, dwell_s=1e-3, depth_pA=10.0, frac_depth=0.01), duration_s=1.0)

    pts = voltage_dep.vdep_points(session, metric="n_events")
    assert len(pts) == 1
    assert pts[0].v_volts == pytest.approx(-0.7)
    assert pts[0].value == 7.0


# ================================================================================================
# vdep_fit -- recover the injected trend
# ================================================================================================

def test_linear_fit_recovers_injected_depth_slope(tmp_path):
    """median_depth = -100*V (0 leak): the linear fit reads slope -100, intercept 0."""
    session = _ladder_session(tmp_path)
    pts = voltage_dep.vdep_points(session, metric="median_depth")
    fit = voltage_dep.vdep_fit([p.v_volts for p in pts], [p.value for p in pts], model="linear")
    assert fit["model"] == "linear"
    assert fit["n"] == 4
    assert fit["params"]["slope"] == pytest.approx(-100.0, rel=1e-9)
    assert fit["params"]["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_linear_fit_recovers_injected_dwell_line(tmp_path):
    """median_dwell = (V + 1.2)e-3: slope 1e-3, intercept 1.2e-3, dwell falling with |V|."""
    session = _ladder_session(tmp_path)
    pts = voltage_dep.vdep_points(session, metric="median_dwell_s")
    fit = voltage_dep.vdep_fit([p.v_volts for p in pts], [p.value for p in pts], model="linear")
    assert fit["params"]["slope"] == pytest.approx(1e-3, rel=1e-9)
    assert fit["params"]["intercept"] == pytest.approx(1.2e-3, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_exponential_fit_recovers_capture_rate_model(tmp_path):
    """capture_rate = A*exp(V/V0): the exponential fit reads back A and V0."""
    session = _ladder_session(tmp_path)
    pts = voltage_dep.vdep_points(session, metric="capture_rate_hz")
    fit = voltage_dep.vdep_fit(
        [p.v_volts for p in pts], [p.value for p in pts], model="exponential")
    assert fit["model"] == "exponential"
    assert fit["n"] == 4
    assert fit["params"]["A"] == pytest.approx(_EXP_A, rel=1e-6)
    assert fit["params"]["V0"] == pytest.approx(_EXP_V0, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_exponential_fit_over_abs_voltage_axis(tmp_path):
    """The fit follows the plotted axis: on |V| the recovered V0 flips sign (+0.5)."""
    session = _ladder_session(tmp_path)
    pts = voltage_dep.vdep_points(session, metric="capture_rate_hz")
    fit = voltage_dep.vdep_fit(
        [abs(p.v_volts) for p in pts], [p.value for p in pts], model="exponential")
    assert fit["params"]["A"] == pytest.approx(_EXP_A, rel=1e-6)
    assert fit["params"]["V0"] == pytest.approx(-_EXP_V0, rel=1e-6)   # +0.5 on |V|


# ================================================================================================
# vdep_fit -- error paths
# ================================================================================================

def test_linear_fit_needs_two_points():
    with pytest.raises(ValueError):
        voltage_dep.vdep_fit([-0.5], [50.0], model="linear")


def test_linear_fit_degenerate_voltage_raises():
    with pytest.raises(ValueError):
        voltage_dep.vdep_fit([-0.5, -0.5, -0.5], [10.0, 11.0, 9.0], model="linear")


def test_exponential_fit_drops_non_positive_points():
    """A non-positive value has no logarithm; it is dropped, the positives still fit."""
    v = [-0.2, -0.5, -0.7, -1.0]
    y = [_EXP_A * np.exp(x / _EXP_V0) for x in v]
    y[0] = 0.0                              # a zero-rate recording
    fit = voltage_dep.vdep_fit(v, y, model="exponential")
    assert fit["n"] == 3                    # the zero was dropped
    assert fit["params"]["V0"] == pytest.approx(_EXP_V0, rel=1e-6)


def test_exponential_fit_all_non_positive_raises():
    with pytest.raises(ValueError):
        voltage_dep.vdep_fit([-0.5, -1.0], [0.0, -5.0], model="exponential")


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        voltage_dep.vdep_fit([-0.5, -1.0], [1.0, 2.0], model="quadratic")


# ================================================================================================
# vdep_predict -- evaluate a fit on the plotted axis
# ================================================================================================

def test_predict_linear_is_the_fitted_line():
    fit = voltage_dep.vdep_fit([-1.0, 0.0, 1.0], [-100.0, 0.0, 100.0], model="linear")
    got = voltage_dep.vdep_predict(fit, np.array([-1.0, 1.0]))
    assert np.allclose(got, [-100.0, 100.0])


def test_predict_exponential_is_A_exp():
    fit = {"model": "exponential", "params": {"A": _EXP_A, "V0": _EXP_V0}, "r2": 1.0, "n": 4}
    got = voltage_dep.vdep_predict(fit, np.array([-0.5, -1.0]))
    assert got == pytest.approx([_EXP_A * np.exp(-0.5 / _EXP_V0),
                                 _EXP_A * np.exp(-1.0 / _EXP_V0)])


def test_predict_unknown_model_raises():
    with pytest.raises(ValueError):
        voltage_dep.vdep_predict({"model": "spline", "params": {}}, [0.0, 1.0])


def test_the_signal_domain_half_reaches_the_voltage_series():
    """The Plots window learned that a Normalize-stage detection cannot pool with a
    picoamp one; the voltage series pooled the same recordings and stayed quiet,
    because ``VDepPoint`` carried no domain and every item defaulted to "pA".

    This is the same class of gap the module docstring already records for the GAIN
    half -- "this half was here in NAME only" -- so it is pinned the same way.
    """
    from nanoscope.analysis.voltage_dep import (
        UNMERGED,
        VDepPoint,
        depends_on_gain,
        depends_on_signal_domain,
        vdep_comparability,
    )

    def _pt(v, val, name, domain):
        return VDepPoint(v_volts=v, value=val, recording=name,
                         baseline_fingerprint="sha256:same", merge_state=UNMERGED,
                         calibration_fingerprint=(0.5, 0.0), signal_domain=domain,
                         analyzed_duration_s=100.0)

    mixed = [_pt(-0.6, 100.0, "pA run", "pA"),
             _pt(-0.7, 87.3, "normalized run", "normalized: zscore — noise sigmas")]

    # An ABSOLUTE amplitude must be warned about...
    got = vdep_comparability(mixed, metric="median_depth")
    assert got.warnings, "a mixed-domain pool must not be silent on median_depth"
    assert any("normal" in w.lower() for w in got.warnings), got.warnings

    # ...and so must dI/I0, which is gain-FREE but NOT domain-free: Normalize is
    # affine and moves the denominator. Measured 872x on a real chain.
    frac = vdep_comparability(mixed, metric="median_frac_depth")
    assert not depends_on_gain("median_frac_depth")
    assert depends_on_signal_domain("median_frac_depth")
    assert frac.warnings, "frac_depth is gain-free but NOT domain-free"

    # A time and a rate never touched an amplitude: still quiet.
    for metric in ("median_dwell_s", "capture_rate_hz", "n_events"):
        assert not depends_on_signal_domain(metric), metric
        assert not vdep_comparability(mixed, metric=metric).warnings, metric

    # A uniform pool says nothing about the DOMAIN. (It may still speak up about the
    # other three halves -- an identity gain reads as uncalibrated, for instance --
    # so this asserts the domain half specifically rather than global silence.)
    uniform = [_pt(-0.6, 100.0, "a", "pA"), _pt(-0.7, 120.0, "b", "pA")]
    assert not [w for w in vdep_comparability(uniform, metric="median_depth").warnings
                if "normal" in w.lower()]
