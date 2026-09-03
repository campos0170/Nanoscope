"""S4 voltage-series — nothing leaves a pooled series unnamed, at the pure layer.

Four defects of the voltage-series group share ONE rule: a pooled comparison may not
lose, drop or silently qualify a recording without saying so, in the same words the
sibling I-V window uses for the same fact.

* VS-03 — a detected recording whose voltage units the file never declared was
  dropped from :func:`vdep_points` by a bare ``continue``, while
  :func:`nanoscope.analysis.iv.iv_exclusions` listed the identical refusal, from the
  identical resolver, with its value and its remedy. The gather now returns the
  points AND :class:`~nanoscope.analysis.voltage_dep.VDepExclusion` for everything
  that could not become one; the reason for the voltage half comes from
  :func:`nanoscope.analysis.iv.iv_point_refusal`, so the two windows cannot disagree.
* VS-08 — :func:`vdep_fit`'s exponential model drops non-positive points (they have
  no logarithm) and returned only the reduced ``n``. It now reports ``n_input``,
  ``dropped`` and ``drop_reason``, so no caller can pass the drop on in silence.
* VS-02 — ``n_events`` is a capture rate TIMES the analysed duration, and was fitted
  across recordings of unequal duration with the word "duration" nowhere. Every point
  now carries ``analyzed_duration_s`` and :func:`duration_comparability` is the rule.
* VS-05 — the x-coordinates of the fitted law can be measured command voltages or
  numbers parsed off a filename, and nothing said which. Every point now carries
  ``voltage_source``.

Metrics are injected as hand-built EventSets so the numbers are exactly what went in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import iv, voltage_dep
from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

WHERE = "(S4 voltage-series: nothing leaves a pooled series unnamed)"


# ================================================================================================
# builders
# ================================================================================================

def _nominal_doc(tmp_path, *, mV: float, name: str, n: int = 2048) -> TraceDocument:
    """A current-only recording whose bias is known only from its FILENAME."""
    arr = np.ascontiguousarray(np.full(n, -200.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _measured_doc(tmp_path, *, raw_mV: float, name: str, n: int = 2048,
                  units: str = "") -> TraceDocument:
    """A recording with a real VOLTAGE channel, whose units the file may not declare."""
    cur = np.ascontiguousarray(np.full(n, -60.0, dtype=np.float32))
    volt = np.ascontiguousarray(np.full(n, raw_mV, dtype=np.float32))
    infos = (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="command", raw_units=units, kind=ChannelKind.VOLTAGE),
    )
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=infos)
    return TraceDocument(meta=meta, channels=(cur, volt))


def _df(n: int, *, dwell_s: float = 1e-3, depth_pA: float = 50.0,
        frac_depth: float = 0.05) -> pd.DataFrame:
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 10 * i, "end_idx": 10 * i + 5, "t_start_s": 1e-3 * i,
            "dwell_s": dwell_s, "baseline_pA": -1000.0, "depth_pA": depth_pA,
            "frac_depth": frac_depth, "ecd_pA_s": 0.05, "max_dev_pA": 60.0,
            "mean_dev_pA": 50.0, "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False, "accepted": True, "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _detect(session: Session, doc_id: str, df: pd.DataFrame, *,
            duration_s: float | None) -> None:
    prov: dict = {"recording": "rec"}
    if duration_s is not None:
        prov["analyzed_duration_s"] = float(duration_s)
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=EventSet(df, prov)))


# ================================================================================================
# 1. VS-03 -- the recording that is NOT in the series is named, with the I-V window's reason
# ================================================================================================

def test_a_recording_with_undeclared_voltage_units_is_named_not_dropped(tmp_path):
    """THE VS-03 REGRESSION. The I-V window listed it; the pooled series said nothing."""
    session = Session()
    good = session.add_document(_nominal_doc(tmp_path, mV=-500.0, name="good.dat"))
    _detect(session, good, _df(10), duration_s=2.0)
    bad = session.add_document(_measured_doc(tmp_path, raw_mV=519.7, name="mystery.h5"))
    _detect(session, bad, _df(7), duration_s=2.0)

    series = voltage_dep.vdep_series(session, metric="capture_rate_hz")
    assert len(series.points) == 1, WHERE
    assert [e.recording for e in series.excluded] == ["mystery.h5"], (
        f"the series lost a detected recording in silence {WHERE}")
    lost = series.excluded[0]
    # The refusal names the state the way every other surface names it, and the fix.
    assert "voltage units (unknown)" in lost.reason, f"{lost.reason} {WHERE}"
    assert "Calibration" in lost.reason, f"{lost.reason} {WHERE}"
    # ...and shows the value the user is about to give units to, honestly.
    assert "519.7" in lost.detail and "units unknown" in lost.detail, f"{lost.detail} {WHERE}"

    # ONE VOICE: the I-V window refuses the same recording for the same reason, and both
    # reasons are built from iv.voltage_units_clause, so the state has one spelling.
    excl = iv.iv_exclusions(session)
    assert [e.label for e in excl] == ["mystery.h5"], WHERE
    assert iv.voltage_units_clause("") in excl[0].reason, WHERE
    assert iv.voltage_units_clause("") in lost.reason, WHERE


def test_declaring_the_units_restores_the_point_and_empties_the_exclusion(tmp_path):
    """The remedy the refusal names actually works, and the disclosure clears with it."""
    session = Session()
    a = session.add_document(_nominal_doc(tmp_path, mV=-500.0, name="a.dat"))
    _detect(session, a, _df(10), duration_s=2.0)
    b = session.add_document(_measured_doc(tmp_path, raw_mV=-600.0, name="b.h5"))
    _detect(session, b, _df(7), duration_s=2.0)

    assert len(voltage_dep.vdep_series(session, metric="capture_rate_hz").excluded) == 1
    session.set_calibration(b, 1, ChannelCalibration(target_units="mV"),
                            write_sidecar=False)
    after = voltage_dep.vdep_series(session, metric="capture_rate_hz")
    assert after.excluded == (), f"{after.excluded} {WHERE}"
    assert len(after.points) == 2, WHERE
    assert sorted(round(p.v_volts, 4) for p in after.points) == [-0.6, -0.5], WHERE


def test_a_recording_with_no_detection_is_named_too(tmp_path):
    """Every silent ``continue`` in the gather is now a named exclusion, not just one."""
    session = Session()
    a = session.add_document(_nominal_doc(tmp_path, mV=-500.0, name="a.dat"))
    _detect(session, a, _df(10), duration_s=2.0)
    session.add_document(_nominal_doc(tmp_path, mV=-600.0, name="undetected.dat"))
    b = session.add_document(_nominal_doc(tmp_path, mV=-700.0, name="empty.dat"))
    _detect(session, b, _df(0), duration_s=2.0)     # a detection with no events

    series = voltage_dep.vdep_series(session, metric="median_dwell_s")
    named = {e.recording: e.reason for e in series.excluded}
    assert set(named) == {"undetected.dat", "empty.dat"}, f"{named} {WHERE}"
    assert "no stored detect.schmitt detection" in named["undetected.dat"], WHERE
    assert "no accepted events" in named["empty.dat"], f"{named['empty.dat']} {WHERE}"
    # The points and the exclusions add up to the open recordings -- the whole point of
    # gathering them in one walk.
    assert len(series.points) + len(series.excluded) == len(session.documents()), WHERE


def test_points_and_exclusions_come_from_one_walk(tmp_path):
    """`vdep_points` and `vdep_exclusions` are two halves of the same gather."""
    session = Session()
    a = session.add_document(_nominal_doc(tmp_path, mV=-500.0, name="a.dat"))
    _detect(session, a, _df(4), duration_s=1.0)
    session.add_document(_measured_doc(tmp_path, raw_mV=0.4, name="nounits.h5"))
    pts = voltage_dep.vdep_points(session, metric="n_events")
    excl = voltage_dep.vdep_exclusions(session, metric="n_events")
    series = voltage_dep.vdep_series(session, metric="n_events")
    assert [p.recording for p in pts] == [p.recording for p in series.points], WHERE
    assert [e.recording for e in excl] == [e.recording for e in series.excluded], WHERE


@pytest.mark.parametrize("units,expect_point", [("mV", True), ("V", True), ("", False)])
def test_iv_point_and_its_refusal_are_exactly_complementary(tmp_path, units, expect_point):
    """`iv_point() is None` iff `iv_point_refusal()` says why -- one gate, two readings."""
    session = Session()
    doc_id = session.add_document(
        _measured_doc(tmp_path, raw_mV=-0.6 if units == "V" else -600.0,
                      name="x.h5", units=units))
    point = iv.iv_point(session, doc_id)
    reason = iv.iv_point_refusal(session, doc_id)
    assert (point is not None) is expect_point, WHERE
    assert (reason == "") is expect_point, f"{reason!r} {WHERE}"


# ================================================================================================
# 2. VS-08 -- the fit says what it left out
# ================================================================================================

def test_the_exponential_fit_reports_the_points_it_could_not_take():
    """THE VS-08 ROOT. A 0 Hz recording is a real low-|V| point, not an error state."""
    fit = voltage_dep.vdep_fit([-0.5, -0.6, -0.7], [4.782993, 0.0, 6.496554],
                               model="exponential")
    assert fit["n"] == 2 and fit["n_input"] == 3, f"{fit} {WHERE}"
    assert fit["dropped"] == (1,), f"{fit['dropped']} {WHERE}"
    assert "no logarithm" in fit["drop_reason"], f"{fit['drop_reason']} {WHERE}"


def test_a_clean_fit_reports_no_drop_at_all():
    """The disclosure is a signal, so it must be silent when there is nothing to say."""
    for model in ("linear", "exponential"):
        fit = voltage_dep.vdep_fit([-0.5, -0.6, -0.7], [1.0, 2.0, 4.0], model=model)
        assert fit["dropped"] == (), f"{model} {fit} {WHERE}"
        assert fit["drop_reason"] == "", f"{model} {WHERE}"
        assert fit["n"] == fit["n_input"] == 3, f"{model} {WHERE}"


def test_the_linear_fit_reports_a_non_finite_pair_it_dropped():
    """The same disclosure for the other model's own cause."""
    fit = voltage_dep.vdep_fit([-0.5, -0.6, -0.7], [1.0, float("nan"), 3.0],
                               model="linear")
    assert fit["dropped"] == (1,) and fit["n"] == 2 and fit["n_input"] == 3, f"{fit} {WHERE}"
    assert "finite" in fit["drop_reason"], f"{fit['drop_reason']} {WHERE}"


def test_dropped_indices_address_the_input_arrays():
    """The indices are the caller's own, or a window cannot name the recording."""
    v = [-0.4, -0.5, -0.6, -0.7]
    y = [0.0, 5.0, 0.0, 7.0]
    fit = voltage_dep.vdep_fit(v, y, model="exponential")
    assert fit["dropped"] == (0, 2), f"{fit['dropped']} {WHERE}"
    assert [y[i] for i in fit["dropped"]] == [0.0, 0.0], WHERE


# ================================================================================================
# 3. VS-02 -- a count across unequal analysed durations
# ================================================================================================

def _duration_ladder(tmp_path, durations) -> Session:
    """One recording per (mV, duration); every EventSet holds the same 40 events."""
    session = Session()
    for mV, dur in durations.items():
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.dat"))
        _detect(session, doc_id, _df(40), duration_s=dur)
    return session


def test_a_count_pooled_across_unequal_durations_is_warned_about(tmp_path):
    """THE VS-02 REGRESSION. The owner's own spans: 250.053 / 59.876 / 181.481 s."""
    session = _duration_ladder(tmp_path, {-500.0: 250.053, -600.0: 59.876,
                                          -700.0: 181.481})
    series = voltage_dep.vdep_series(session, metric="n_events")
    assert series.duration_warnings, f"a count over a 4.18x span said nothing {WHERE}"
    text = series.duration_warnings[0]
    assert "capture rate TIMES the analysed duration" in text, f"{text} {WHERE}"
    for span in ("250.053", "59.876", "181.481"):
        assert span in text, f"{span} missing from {text} {WHERE}"
    assert "4.18x" in text, f"{text} {WHERE}"
    # ...and it names the honest sibling, the way the gain warning names its alternatives.
    assert "capture rate" in text, WHERE


def test_the_duration_travels_on_every_point(tmp_path):
    """The window cannot show a duration the point does not carry."""
    session = _duration_ladder(tmp_path, {-500.0: 250.053, -600.0: 59.876})
    pts = voltage_dep.vdep_points(session, metric="n_events")
    assert sorted(p.analyzed_duration_s for p in pts) == [59.876, 250.053], WHERE
    # ...and the groups the provenance block renders are the same numbers.
    groups = dict(voltage_dep.vdep_series(session, metric="n_events").duration_groups)
    assert set(groups) == {"250.053 s", "59.876 s"}, f"{groups} {WHERE}"


def test_equal_durations_say_nothing(tmp_path):
    """A check that always fires is not a check: equal spans are the quiet case."""
    session = _duration_ladder(tmp_path, {-500.0: 40.0, -600.0: 40.0, -700.0: 40.0})
    assert voltage_dep.vdep_series(session, metric="n_events").duration_warnings == (), WHERE


def test_an_unknown_duration_fails_closed(tmp_path):
    """An ABSENT duration is not evidence of agreement -- the baseline rule's shape."""
    session = Session()
    for mV, dur in ((-500.0, 40.0), (-600.0, None)):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.dat"))
        _detect(session, doc_id, _df(40), duration_s=dur)
    series = voltage_dep.vdep_series(session, metric="n_events")
    assert len(series.points) == 2, WHERE           # a count still needs no duration
    assert series.duration_warnings, WHERE
    assert "recorded no analysed duration" in series.duration_warnings[0], WHERE
    assert "unknown" in dict(series.duration_groups), WHERE


@pytest.mark.parametrize("metric", ["capture_rate_hz", "median_dwell_s", "median_depth",
                                    "median_frac_depth"])
def test_only_the_count_reads_the_duration(tmp_path, metric):
    """PER METRIC, like every other half of the rule: a rate already divided it out."""
    session = _duration_ladder(tmp_path, {-500.0: 250.053, -600.0: 59.876,
                                          -700.0: 181.481})
    assert voltage_dep.depends_on_duration(metric) is False, WHERE
    assert voltage_dep.vdep_series(session, metric=metric).duration_warnings == (), WHERE
    # ...but the durations are still SHOWN for it; only the verdict is per-metric.
    assert len(voltage_dep.vdep_series(session, metric=metric).duration_groups) == 3, WHERE


def test_depends_on_duration_rejects_an_unknown_metric():
    assert voltage_dep.depends_on_duration("n_events") is True, WHERE
    with pytest.raises(ValueError, match="unknown metric"):
        voltage_dep.depends_on_duration("median_snr")


def test_a_pool_with_no_voltage_dependence_at_all_still_trends_as_a_count(tmp_path):
    """WHY THE WARNING MATTERS. Three recordings at the SAME true capture rate of
    4.0 Hz, differing only in how long they ran, are drawn as a steep, clean count
    trend -- the voltage dependence is manufactured entirely by file length."""
    session = Session()
    for mV, dur in ((-500.0, 250.0), (-600.0, 60.0), (-700.0, 180.0)):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.dat"))
        _detect(session, doc_id, _df(int(4.0 * dur)), duration_s=dur)

    rates = voltage_dep.vdep_points(session, metric="capture_rate_hz")
    assert {round(p.value, 9) for p in rates} == {4.0}, WHERE   # no dependence at all
    counts = voltage_dep.vdep_points(session, metric="n_events")
    fit = voltage_dep.vdep_fit([p.v_volts for p in counts], [p.value for p in counts])
    assert abs(fit["params"]["slope"]) > 100.0, f"{fit} {WHERE}"
    # ...and that is exactly the pool the rule now speaks about.
    assert voltage_dep.vdep_series(session, metric="n_events").duration_warnings, WHERE
    assert voltage_dep.vdep_series(session,
                                   metric="capture_rate_hz").duration_warnings == (), WHERE


# ================================================================================================
# 4. VS-05 -- measured x-coordinates against ones read off a filename
# ================================================================================================

def test_every_point_says_where_its_voltage_came_from(tmp_path):
    """THE VS-05 REGRESSION. VDepPoint had no voltage_source field at all."""
    session = Session()
    a = session.add_document(_measured_doc(tmp_path, raw_mV=-700.0, name="meas.abf",
                                           units="mV"))
    _detect(session, a, _df(9), duration_s=2.0)
    b = session.add_document(_nominal_doc(tmp_path, mV=-600.0, name="from_name.csv"))
    _detect(session, b, _df(5), duration_s=2.0)

    series = voltage_dep.vdep_series(session, metric="capture_rate_hz")
    got = {p.recording: p.voltage_source for p in series.points}
    assert got == {"meas.abf": "measured", "from_name.csv": "nominal"}, f"{got} {WHERE}"
    assert dict(series.voltage_source_groups) == {
        "measured": ("meas.abf",), "nominal": ("from_name.csv",)}, WHERE

    # ONE RESOLVER: the I-V window tags the same two recordings the same two ways.
    iv_src = {p.label: p.voltage_source for p in iv.iv_points(session)}
    assert iv_src == got, f"{iv_src} vs {got} {WHERE}"


def test_one_source_is_still_stated(tmp_path):
    """Not only the mixed case: a series entirely off filenames must say so too."""
    session = Session()
    for mV in (-500.0, -700.0):
        doc_id = session.add_document(
            _nominal_doc(tmp_path, mV=mV, name=f"apo_{int(-mV)}.csv"))
        _detect(session, doc_id, _df(6), duration_s=2.0)
    series = voltage_dep.vdep_series(session, metric="capture_rate_hz")
    assert [k for k, _ in series.voltage_source_groups] == ["nominal"], WHERE
