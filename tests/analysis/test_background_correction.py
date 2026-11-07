"""Scientific contracts for explicit, independent background comparisons."""
from dataclasses import replace
import json

import numpy as np
import pytest

from nanoscope.analysis.background_correction import (
    AnchorRegion, BackgroundRecipe, correct_background, export_background,
    background_export_paths,
)


def anchors(points=((0, 10), (1, 12)), *, end=1, **kwargs):
    return BackgroundRecipe(source="anchors", regions=(AnchorRegion(0, end, points),), **kwargs)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("interpolation", ["linear", "pchip"])
def test_signed_drift_and_event_amplitude_are_preserved(sign, interpolation):
    fs = 1000
    t = np.arange(1000) / fs
    background = sign * (100 + 10*t)
    event = np.zeros_like(t)
    event[200:280] = -sign * 25
    raw = background + event
    original = raw.copy()
    recipe = anchors(((0, sign*100), (1, sign*110)), interpolation=interpolation)
    result = correct_background(raw, fs, recipe, units="pA")
    np.testing.assert_allclose(result.baseline, background, atol=1e-12)
    np.testing.assert_allclose(result.corrected, event, atol=1e-12)
    np.testing.assert_array_equal(raw, original)
    # A later caller edit must not change the completed comparison.
    raw[:] = 0
    np.testing.assert_array_equal(result.raw, original)
    for array in (result.raw, result.baseline, result.corrected, result.time_s):
        assert not array.flags.writeable


def test_retained_offset_is_declared_median_not_zero_or_raw_mean():
    raw = np.array([10., 8., 10., 14.])
    recipe = anchors(((0, 10), (1, 14)), mode="retained_offset")
    result = correct_background(raw, 4, recipe, units="pA")
    np.testing.assert_allclose(result.baseline, [10, 11, 12, 13])
    np.testing.assert_allclose(result.corrected, [11.5, 8.5, 9.5, 12.5])
    assert result.provenance["retained_offset"] == 11.5
    assert result.provenance["retained_offset_policy"] == "median background"


@pytest.mark.parametrize("interpolation", ["linear", "pchip"])
def test_explicit_jump_is_not_bridged(interpolation):
    recipe = BackgroundRecipe(source="anchors", interpolation=interpolation, regions=(
        AnchorRegion(0, .5, ((0, 10), (.5, 10))),
        AnchorRegion(.5, 1, ((.5, 30), (1, 30))),
    ))
    raw = np.r_[np.full(50, 10.), np.full(50, 30.)]
    result = correct_background(raw, 100, recipe, units="pA")
    np.testing.assert_array_equal(result.baseline, raw)
    np.testing.assert_array_equal(result.corrected, np.zeros(100))
    assert result.baseline[49] == 10 and result.baseline[50] == 30
    assert "no interpolation" in result.provenance["details"]["boundary_policy"]


def test_constant_regions_preserve_distinct_asserted_levels():
    recipe = BackgroundRecipe(source="anchors", interpolation="constant", regions=(
        AnchorRegion(0, .3, ((.1, -50),)), AnchorRegion(.3, 1, ((.5, -80),)),
    ))
    r = correct_background(np.zeros(10), 10, recipe, units="pA")
    np.testing.assert_array_equal(r.baseline, [-50]*3+[-80]*7)


def test_pchip_is_shape_preserving_without_cubic_overshoot():
    r = correct_background(np.zeros(101), 100, anchors(
        ((0, 0), (.2, 2), (.4, 2), (1, 3)), end=1.01, interpolation="pchip"), units="pA")
    assert r.baseline.min() >= 0 and r.baseline.max() <= 3
    assert np.all(np.diff(r.baseline) >= -1e-14)
    np.testing.assert_array_equal(r.baseline[20:41], np.full(21, 2.))


def test_unsupported_edges_are_held_and_counted_or_refused():
    recipe = anchors(((.2, 10), (.7, 20)))
    result = correct_background(np.zeros(10), 10, recipe, units="pA")
    np.testing.assert_array_equal(result.baseline[:3], [10, 10, 10])
    np.testing.assert_array_equal(result.baseline[7:], [20, 20, 20])
    assert result.provenance["details"]["held_endpoint_samples"] == 4
    assert any("assumed, not measured" in w for w in result.warnings)
    with pytest.raises(ValueError, match="outside anchor support"):
        correct_background(np.zeros(10), 10, replace(recipe, edge_policy="refuse"), units="pA")


@pytest.mark.parametrize("regions", [
    (AnchorRegion(0, .4, ((.2, 1),)), AnchorRegion(.5, 1, ((.7, 2),))),
    (AnchorRegion(0, .6, ((.2, 1),)), AnchorRegion(.5, 1, ((.7, 2),))),
])
def test_gap_or_overlap_between_regions_is_refused(regions):
    with pytest.raises(ValueError, match="adjacent"):
        BackgroundRecipe(source="anchors", interpolation="constant", regions=regions)


@pytest.mark.parametrize("region", [AnchorRegion(.1, 1, ((.2, 10),)), AnchorRegion(0, .5, ((.2, 10),))])
def test_uncovered_target_domain_is_not_silently_filled(region):
    recipe = BackgroundRecipe(source="anchors", interpolation="constant", regions=(region,))
    with pytest.raises(ValueError, match="cover every target sample"):
        correct_background(np.zeros(10), 10, recipe, units="pA")


@pytest.mark.parametrize("bad", [
    {"start_s":0,"end_s":1,"anchors":[[.2,1],[.2,2]]},
    {"start_s":0,"end_s":1,"anchors":[[.8,1],[.2,2]]},
    {"start_s":0,"end_s":1,"anchors":[[-.1,1]]},
    {"start_s":0,"end_s":1,"anchors":[[.1,np.nan]]},
    {"start_s":0,"end_s":1,"anchors":[[np.inf,1]]},
    {"start_s":1,"end_s":0,"anchors":[[.5,1]]},
])
def test_invalid_anchor_recipe_refused(bad):
    with pytest.raises(ValueError):
        AnchorRegion.from_dict(bad)


def test_interval_reference_uses_selected_samples_and_existing_endpoint_policy():
    raw = np.r_[np.full(10, 100.), np.full(10, 80.), np.full(10, 120.)]
    reference = {"mode":"intervals", "intervals_s":[[0,.1],[.2,.3]], "interpolation":"linear"}
    result = correct_background(raw, 100, BackgroundRecipe(source="intervals", interval_reference=reference), units="pA")
    assert result.baseline[0] == 100 and result.baseline[-1] == 120
    assert result.baseline[14] == pytest.approx(109.5)
    assert result.corrected[14] == pytest.approx(-29.5)
    assert result.provenance["details"]["endpoint_policy"].startswith("hold")
    assert result.provenance["recipe"]["interval_reference"]["intervals_s"] == [[0,.1],[.2,.3]]


def test_known_open_pore_reference_is_a_nonmutating_comparison():
    recipe = BackgroundRecipe(source="intervals", interval_reference={"mode":"constant", "current_pA":-100})
    raw = np.array([-100., -70., -80., -100.])
    result = correct_background(raw, 4, recipe, units="pA")
    np.testing.assert_array_equal(result.corrected, [0,30,20,0])
    np.testing.assert_array_equal(raw, [-100,-70,-80,-100])


@pytest.mark.parametrize("reference", [
    {"mode":"intervals","intervals_s":[[0,.02]],"interpolation":"constant"},
    {"mode":"intervals","intervals_s":[[0,2]],"interpolation":"constant"},
])
def test_invalid_selected_intervals_refused(reference):
    recipe = BackgroundRecipe(source="intervals", interval_reference=reference)
    with pytest.raises(ValueError):
        correct_background(np.zeros(100), 100, recipe, units="pA")


def test_recording_alignment_scale_offset_and_origins_are_explicit():
    ref = np.arange(20, dtype=float)
    raw = 2*ref[4:8] + 7 + np.array([0, -5, -5, 0])
    recipe = BackgroundRecipe(source="recording", time_shift_s=.1, scale=2, offset=7)
    result = correct_background(raw, 10, recipe, units="pA", time_start_s=10.5,
        reference=ref, reference_fs=10, reference_units="pA", reference_time_start_s=10)
    np.testing.assert_allclose(result.corrected, [0,-5,-5,0], atol=1e-12)
    np.testing.assert_allclose(result.time_s, [10.5,10.6,10.7,10.8])
    assert result.provenance["details"]["reference_index_start"] == pytest.approx(4)


@pytest.mark.parametrize("shift", [.1,-.1])
def test_partial_overlap_refused_instead_of_trimming_or_extrapolating(shift):
    with pytest.raises(ValueError, match="entire shifted target span"):
        correct_background(np.arange(10.), 10, BackgroundRecipe(source="recording", time_shift_s=shift),
            units="pA", reference=np.arange(10.), reference_fs=10, reference_units="pA")


def test_different_rates_and_fractional_shift_require_explicit_resampling():
    recipe = BackgroundRecipe(source="recording")
    args = dict(units="pA", reference=np.arange(20.)*.5, reference_fs=20, reference_units="pA")
    with pytest.raises(ValueError, match="Sample rates differ"):
        correct_background(np.arange(10.), 10, recipe, **args)
    r = correct_background(np.arange(10.), 10, replace(recipe, reference_resampling="linear"), **args)
    np.testing.assert_array_equal(r.corrected, np.zeros(10))
    shifted = replace(recipe, time_shift_s=-.05)
    args.update(reference=np.arange(20.), reference_fs=10)
    with pytest.raises(ValueError, match="not sample-aligned"):
        correct_background(np.arange(10.), 10, shifted, **args)
    r = correct_background(np.arange(10.)+.5, 10, replace(shifted, reference_resampling="linear"), **args)
    np.testing.assert_allclose(r.corrected, 0, atol=1e-14)
    assert any("not anti-alias filtering" in w for w in r.warnings)


def test_explicit_linear_resampling_still_refuses_partial_overlap():
    with pytest.raises(ValueError, match="partial overlap"):
        correct_background(np.zeros(10),10,BackgroundRecipe(source="recording",reference_resampling="linear"),
            units="pA",reference=np.zeros(5),reference_fs=10,reference_units="pA")


@pytest.mark.parametrize("which", ["target", "reference"])
def test_time_gaps_are_never_interpolated_across(which):
    times = np.arange(10.)/10
    times[5:] += .1
    kwargs = {"sample_times_s":times} if which == "target" else {"reference_sample_times_s":times}
    with pytest.raises(ValueError, match="gap"):
        correct_background(np.zeros(10),10,BackgroundRecipe(source="recording",reference_resampling="linear"),
            units="pA",reference=np.zeros(10),reference_fs=10,reference_units="pA", **kwargs)


@pytest.mark.parametrize("bad", [np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("which", ["target", "reference"])
def test_nonfinite_samples_are_refused(bad, which):
    raw = np.ones(10); ref = np.ones(10)
    (raw if which == "target" else ref)[3] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        correct_background(raw,10,BackgroundRecipe(source="recording"),units="pA",reference=ref,reference_fs=10,reference_units="pA")


@pytest.mark.parametrize("bad", [0,-1,np.nan,np.inf,True,None])
def test_invalid_sample_rate_refused(bad):
    with pytest.raises(ValueError):
        correct_background(np.zeros(10),bad,anchors(),units="pA")


@pytest.mark.parametrize("units", ["", "unknown", "V", "counts", "mV", None])
def test_noncurrent_units_refused(units):
    with pytest.raises(ValueError, match="current units"):
        correct_background(np.zeros(10),10,anchors(),units=units)


def test_scaled_units_cannot_be_silently_reinterpreted():
    with pytest.raises(ValueError, match="recipe units"):
        correct_background(np.zeros(10),10,anchors(),units="nA")
    with pytest.raises(ValueError, match="including scale"):
        correct_background(np.zeros(10),10,BackgroundRecipe(source="recording"),units="pA",
            reference=np.zeros(10),reference_fs=10,reference_units="nA")
    with pytest.raises(ValueError, match="uses pA"):
        BackgroundRecipe(source="intervals",units="nA",interval_reference={"mode":"constant","current_pA":1})


def test_microamp_spelling_alias_does_not_change_scale():
    recipe = anchors(units="µA")
    r=correct_background(np.ones(10),10,recipe,units="uA")
    assert r.recipe.units == "uA"


@pytest.mark.parametrize("field,value", [("version",2),("version",True),("version",1.0),("source",[]),("mode","runtime"),("scale",np.inf),("offset",True),("time_shift_s",np.nan)])
def test_bad_recipe_fields_refused(field,value):
    d=BackgroundRecipe(source="recording").to_dict();d[field]=value
    with pytest.raises(ValueError):BackgroundRecipe.from_dict(d)


def test_recipe_json_roundtrip_and_unknown_fields():
    recipe=BackgroundRecipe(source="anchors",regions=(AnchorRegion(0,1,((0,10),(1,12))),))
    assert BackgroundRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))) == recipe
    with pytest.raises(ValueError,match="Unknown"):
        BackgroundRecipe.from_dict({**recipe.to_dict(),"silent_extrapolation":True})


def test_overflow_refused_before_export():
    with pytest.raises(ValueError,match="nonfinite"):
        correct_background(np.ones(2),1,BackgroundRecipe(source="recording",scale=1e308),units="pA",
            reference=np.full(2,10.),reference_fs=1,reference_units="pA")
    with pytest.raises(ValueError,match="overflowed"):
        correct_background(np.full(2,1e308),1,BackgroundRecipe(source="recording"),units="pA",
            reference=np.full(2,-1e308),reference_fs=1,reference_units="pA")


def test_export_full_precision_data_recipe_and_source_provenance(tmp_path):
    raw=np.linspace(99,110,1001)
    source={"doc_key":"recording-A","channel":2,"cal_gain":2.5,"cal_offset":-1}
    result=correct_background(raw,1000,anchors(end=1.001),units="pA",time_start_s=12.5,source_metadata=source)
    source["doc_key"]="changed-after-run"
    paths=export_background(tmp_path/"comparison",result)
    data=np.loadtxt(paths["data"],delimiter=",",skiprows=1)
    assert data.shape == (1001,4)
    for index, array in enumerate((result.time_s,result.raw,result.baseline,result.corrected)):
        np.testing.assert_array_equal(data[:,index],array)
    meta=json.loads(paths["meta"].read_text())
    assert meta["source"]["metadata"]["doc_key"]=="recording-A"
    assert meta["recipe"]==result.recipe.to_dict()
    assert meta["raw_input_modified"] is False
    assert meta["current_units"]=="pA"
    assert background_export_paths(tmp_path/"comparison")==tuple(paths.values())


def test_reference_provenance_is_a_snapshot(tmp_path):
    meta={"doc_key":"blank-B","cal_gain":1,"channel":0}
    result=correct_background(np.ones(10),10,BackgroundRecipe(source="recording"),units="pA",
        reference=np.zeros(10),reference_fs=10,reference_units="pA",reference_metadata=meta)
    meta["channel"]=4
    assert result.provenance["reference"]["metadata"]["channel"]==0
    assert result.provenance["reference"]["sha256_float64_le"]


@pytest.mark.parametrize("array_name", ["raw","baseline","corrected","time_s"])
def test_export_refuses_modified_completed_arrays(tmp_path,array_name):
    r=correct_background(np.ones(10),10,anchors(),units="pA")
    a=getattr(r,array_name);a.setflags(write=True);a[0]+=1
    with pytest.raises(ValueError,match="changed"):
        export_background(tmp_path/"tampered.csv",r)
    assert not (tmp_path/"tampered.csv").exists()


def test_atomic_export_failure_preserves_previous_pair(tmp_path,monkeypatch):
    result=correct_background(np.ones(10),10,anchors(),units="pA")
    paths=export_background(tmp_path/"existing.csv",result)
    before={k:p.read_bytes() for k,p in paths.items()}
    def fail(*args,**kwargs):raise OSError("simulated write failure")
    monkeypatch.setattr(np,"savetxt",fail)
    with pytest.raises(OSError):export_background(paths["data"],result)
    assert {k:p.read_bytes() for k,p in paths.items()}==before


def test_metadata_nonfinite_refused():
    with pytest.raises(ValueError,match="JSON"):
        correct_background(np.ones(10),10,anchors(),units="pA",source_metadata={"cal_gain":np.nan})
