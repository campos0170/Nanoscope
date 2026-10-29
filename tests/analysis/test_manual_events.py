"""The pure half of the Manual events design (docs/MANUAL_EVENT_TABLE_DESIGN.md): the contract,
one measured hand row, dtype-safe insertion, delete, rebuild, and the re-run carry-over."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import manual_events as me
from nanoscope.analysis.baseline import DEFAULT_BASELINE
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    MANUAL_MODULE_ID, SCHEMA, EventSet, mark_reviewed)
from nanoscope.analysis.truth import synth_trace

FS = 50_000.0
I0 = -1000.0


def _trace(spans, *, n=80_000, depth=300.0, seed=0, noise=8.0):
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, noise, n)
    for s, e in spans:
        x[s:e] += depth
    return np.ascontiguousarray(x, dtype=np.float32)


def _contract(sig):
    return me.build_contract(sig, FS, "pA", config=DEFAULT_BASELINE, recording="r.dat",
                             doc_id="d", channel=0)


def _row(sig, contract, span, event_id, df_columns=None, **kw):
    return me.measure_hand_row(sig, FS, span[0], span[1], provenance=contract.provenance,
                               units="pA", recording="r.dat", pore_id="p", event_id=event_id, **kw)


# ------------------------------------------------------------------------------------------------
# the contract
# ------------------------------------------------------------------------------------------------
def test_the_contract_is_the_detectors_own_basis():
    spans = [(10_000, 10_300), (30_000, 30_300), (50_000, 50_400)]
    sig = _trace(spans)
    c = _contract(sig)
    assert c.polarity == 1                          # I0 < 0 and events run toward zero
    assert c.noise_rms_pA == pytest.approx(8.0, rel=0.1)
    prov = c.provenance
    assert prov["signal_basis"] == "raw" and prov["detector"] == "manual"
    assert prov["fs"] == FS and prov["baseline_fingerprint"] == c.baseline_fingerprint
    assert prov["baseline_method"] and "detected_at" in prov
    # nobody knows how much a hand pass searched, so the table has no span and no rate
    for absent in ("analyzed_duration_s", "analysed_start_idx", "min_dwell_s", "max_dwell_s"):
        assert absent not in prov
    # the same estimator the detector stamped on the same trace
    _df, dprov = detect_events(sig, FS, min_dwell_s=1e-4)
    assert dprov["baseline_fingerprint"] == c.baseline_fingerprint
    assert dprov["noise_rms_pA"] == pytest.approx(c.noise_rms_pA, rel=0.02)


def test_a_flat_trace_has_no_measurable_noise():
    with pytest.raises(ValueError, match="no measurable noise"):
        _contract(np.full(50_000, -50.0, dtype=np.float32))


def test_noise_ignores_non_finite_samples_instead_of_refusing_the_table():
    sig = _trace([(10_000, 10_300)])
    sig[500] = np.nan
    assert _contract(sig).noise_rms_pA == pytest.approx(8.0, rel=0.1)


# ------------------------------------------------------------------------------------------------
# one row
# ------------------------------------------------------------------------------------------------
def test_a_hand_row_is_measured_like_an_event_and_marked_as_drawn():
    sig = _trace([(20_000, 20_400)])
    c = _contract(sig)
    row = _row(sig, c, (20_000, 20_400), 7)
    r = row.iloc[0]
    assert (r["event_id"], r["start_idx"], r["end_idx"]) == (7, 20_000, 20_400)
    assert r["depth_pA"] == pytest.approx(300.0, rel=0.05)
    assert r["dwell_s"] == pytest.approx(400 / FS)
    assert bool(r["accepted"]) and bool(r["added"])
    assert (r["source"], r["review"]) == ("manual", "reviewed")
    assert pd.isna(r["orig_start_idx"]) and pd.isna(r["orig_end_idx"])     # Revert stays off
    assert r["snr"] == pytest.approx(300.0 / c.noise_rms_pA, rel=0.1)


def test_a_span_at_the_recording_edge_is_rejected_by_the_automatic_rule():
    sig = _trace([(0, 400)])
    c = _contract(sig)
    r = _row(sig, c, (0, 400), 0).iloc[0]
    assert not bool(r["accepted"]) and "clipped_by_boundary" in r["flag"]


def test_the_edge_record_travels_with_the_row():
    sig = _trace([(20_000, 20_400)])
    c = _contract(sig)
    edge = me.EdgeRecord(bounds_source="start=fit:step;end=placed", drawn_start=19_990,
                         drawn_end=20_410, fit_json=me.fit_record_json({"method": "step", "v": 1}))
    r = _row(sig, c, (20_000, 20_400), 0, edge=edge, note="a long one").iloc[0]
    assert r["bounds_source"] == "start=fit:step;end=placed"
    assert (int(r["drawn_start_idx"]), int(r["drawn_end_idx"])) == (19_990, 20_410)
    assert r["note"] == "a long one" and '"method":"step"' in r["fit_json"]


# ------------------------------------------------------------------------------------------------
# insertion keeps dtypes, time order, and native study tables
# ------------------------------------------------------------------------------------------------
def test_inserting_rows_keeps_time_order_and_dtypes():
    sig = _trace([(10_000, 10_300), (30_000, 30_300)])
    c = _contract(sig)
    ev = me.empty_manual_events(c)
    df = me.insert_hand_row(ev.df, _row(sig, c, (30_000, 30_300), 0))
    df = me.insert_hand_row(df, _row(sig, c, (10_000, 10_300), 1))
    assert list(df["start_idx"]) == [10_000, 30_000]
    for col, (dtype, _u, _d) in SCHEMA.items():
        if dtype != "string":
            assert df[col].dtype == np.dtype(dtype), col
    assert str(df["added"].dtype) == "bool" and str(df["orig_start_idx"].dtype) == "Int64"
    assert str(df["note"].dtype) == "string"


def test_a_native_study_table_takes_a_hand_row_without_crashing_or_inventing_values():
    """int64 and bool `study_*` columns: the old hand-add cast every column back to its own
    dtype, which raised on the int64 ones and turned NaN into True in the bool ones."""
    sig = _trace([(10_000, 10_300), (30_000, 30_300)])
    c = _contract(sig)
    ev = me.empty_manual_events(c)
    df = me.insert_hand_row(ev.df, _row(sig, c, (10_000, 10_300), 0))
    df["study_rank"] = np.array([3], dtype=np.int64)
    df["study_pass"] = np.array([True])
    df = me.insert_hand_row(df, _row(sig, c, (30_000, 30_300), 1))
    assert df["study_rank"].iloc[0] == 3 and pd.isna(df["study_rank"].iloc[1])
    assert bool(df["study_pass"].iloc[0]) is True and pd.isna(df["study_pass"].iloc[1])


def test_a_detector_table_grows_only_the_columns_the_row_needs():
    sig = _trace([(10_000, 10_300), (30_000, 30_300)])
    c = _contract(sig)
    df, _prov = detect_events(sig, FS, min_dwell_s=1e-4)
    assert "bounds_source" not in df.columns and "note" not in df.columns
    row = _row(sig, c, (50_000, 50_300), 99, edge=me.EdgeRecord(bounds_source="start=drawn;end=drawn"))
    grown = me.insert_hand_row(df, row)
    assert "bounds_source" in grown.columns and "added" in grown.columns
    assert "note" not in grown.columns and "origin" not in grown.columns      # nothing to say
    assert grown.loc[grown["event_id"] == 99, "source"].iloc[0] == "manual"
    assert grown["source"].isna().sum() == len(df)       # the detector rows keep "unknown"


# ------------------------------------------------------------------------------------------------
# who counts as hand-made
# ------------------------------------------------------------------------------------------------
def test_hand_made_means_source_manual_with_added_as_the_fallback():
    df = pd.DataFrame({
        "source": pd.array(["manual", "mixed", "detect.schmitt", pd.NA, pd.NA], dtype="string"),
        "added": [True, True, False, True, False],
    })
    assert list(me.hand_made_mask(df)) == [True, False, False, True, False]
    assert list(me.hand_made_mask(df.drop(columns=["source"]))) == [True, True, False, True, False]
    assert me.hand_made_mask(df.iloc[:0]).size == 0


def test_only_hand_made_rows_can_be_deleted():
    sig = _trace([(10_000, 10_300), (30_000, 30_300)])
    c = _contract(sig)
    det, _ = detect_events(sig, FS, min_dwell_s=1e-4)
    det = me.insert_hand_row(det, _row(sig, c, (50_000, 50_300), 99))
    out, removed = me.delete_hand_rows(det, [0, 99])
    assert removed == [99] and 0 in set(out["event_id"]) and 99 not in set(out["event_id"])
    same, none = me.delete_hand_rows(det, [0])
    assert none == [] and len(same) == len(det)


# ------------------------------------------------------------------------------------------------
# rebuild
# ------------------------------------------------------------------------------------------------
def test_rebuild_moves_times_with_the_rate_and_keeps_ids_indices_and_rulings():
    sig = _trace([(10_000, 10_300), (30_000, 30_300)])
    c = _contract(sig)
    ev = me.empty_manual_events(c)
    df = me.insert_hand_row(ev.df, _row(sig, c, (10_000, 10_300), 0))
    df = me.insert_hand_row(df, _row(sig, c, (30_000, 30_300), 1))
    df.loc[df["event_id"] == 1, "accepted"] = False        # a hand rejection
    df.loc[df["event_id"] == 1, "flag"] = "molecule=apo"   # and a label
    halved = me.build_contract(sig, 2 * FS, "pA", config=DEFAULT_BASELINE, recording="r.dat",
                               doc_id="d", channel=0)
    rebuilt = me.rebuild_manual_events(EventSet(df, c.events_provenance()), sig, halved, units="pA")
    out = rebuilt.df
    assert list(out["event_id"]) == [0, 1] and list(out["start_idx"]) == [10_000, 30_000]
    assert out["dwell_s"].iloc[0] == pytest.approx(300 / (2 * FS))
    assert rebuilt.provenance["fs"] == 2 * FS
    assert not bool(out["accepted"].iloc[1]) and "molecule=apo" in out["flag"].iloc[1]
    assert out["added"].all() and (out["source"] == "manual").all()


# ------------------------------------------------------------------------------------------------
# the re-run carry-over
# ------------------------------------------------------------------------------------------------
def _old_table(sig, spans, *, note=None):
    c = _contract(sig)
    df = me.empty_manual_events(c).df
    for i, span in enumerate(spans):
        df = me.insert_hand_row(df, _row(sig, c, span, i, note=note))
    return df


def _carry(sig, detected_spans, old, **kw):
    """A re-run that finds ``detected_spans`` (with strong events) over the same trace."""
    det_sig = _trace(detected_spans)
    df, prov = detect_events(det_sig, FS, min_dwell_s=1e-4)
    new = EventSet(df, prov)
    return me.carry_hand_rows(new, old, sig, fs=FS, units="pA", old_module_id="detect.schmitt", **kw)


def test_a_hand_row_the_new_run_does_not_find_is_carried_with_its_origin():
    sig = _trace([(10_000, 10_300), (30_000, 30_300), (50_000, 50_300)])
    old = _old_table(sig, [(50_000, 50_300)], note="found by eye")
    out, rep = _carry(sig, [(10_000, 10_300), (30_000, 30_300)], old)
    assert rep.kept == ((0, 2),) and not rep.moved and not rep.left_out
    carried = out.df.loc[out.df["event_id"] == 2].iloc[0]
    assert carried["source"] == "manual" and bool(carried["added"])
    assert carried["origin"] == "detect.schmitt:0" and carried["note"] == "found by eye"
    assert (int(carried["start_idx"]), int(carried["end_idx"])) == (50_000, 50_300)
    assert list(out.df["start_idx"]) == sorted(out.df["start_idx"])
    assert out.df["event_id"].is_unique
    assert out.provenance["carried_hand_rows"]["carried_ids"] == [2]


def test_a_hand_row_the_new_run_also_found_moves_its_labels_onto_the_detection():
    sig = _trace([(10_000, 10_300), (30_000, 30_300)])
    old = _old_table(sig, [(30_005, 30_295)], note="odd one")
    old.loc[:, "flag"] = "molecule=apo"
    out, rep = _carry(sig, [(10_000, 10_300), (30_000, 30_300)], old)
    assert rep.moved == ((0, 1),) and not rep.kept
    assert len(out.df) == 2                                  # no second row for the same event
    det = out.df.loc[out.df["event_id"] == 1].iloc[0]
    assert "molecule=apo" in det["flag"] and det["note"] == "odd one"
    assert det["review"] == "reviewed"
    assert not bool(det.get("added", False))                  # still the detector's own row
    assert "now detected (1)" in rep.sentence("Schmitt", len(out.df))


def test_a_hand_rejection_moves_onto_the_detected_row():
    sig = _trace([(30_000, 30_300)])
    old = _old_table(sig, [(30_000, 30_300)])
    old.loc[:, "accepted"] = False
    out, rep = _carry(sig, [(30_000, 30_300)], old)
    assert rep.moved and not bool(out.df["accepted"].iloc[0])


def test_the_hand_row_outvotes_an_automatically_rejected_detection():
    sig = _trace([(30_000, 30_300)])
    old = _old_table(sig, [(30_000, 30_300)])
    det_sig = _trace([(30_000, 30_300)])
    df, prov = detect_events(det_sig, FS, min_dwell_s=1e-4)
    df.loc[:, "accepted"] = False
    df.loc[:, "flag"] = "clipped_by_boundary"
    out, rep = me.carry_hand_rows(EventSet(df, prov), old, sig, fs=FS, units="pA",
                                  old_module_id="detect.schmitt")
    assert rep.replaced_rejected == ((0, 0),) and rep.kept
    assert len(out.df) == 1 and out.df["source"].iloc[0] == "manual"
    assert out.provenance["carried_hand_rows"]["dropped_ids"] == [0]


def test_a_partial_overlap_keeps_both_and_lists_it():
    sig = _trace([(30_000, 30_300)])
    old = _old_table(sig, [(30_200, 30_900)])                # 100 of its 700 samples overlap
    out, rep = _carry(sig, [(30_000, 30_300)], old)
    assert rep.unresolved and len(out.df) == 2


def test_a_hand_row_outside_what_the_new_run_searched_is_left_out_and_said():
    sig = _trace([(10_000, 10_300), (60_000, 60_300)])
    old = _old_table(sig, [(60_000, 60_300)])
    det_sig = _trace([(10_000, 10_300)])
    df, prov = detect_events(det_sig[:40_000], FS, min_dwell_s=1e-4)
    prov = dict(prov, analysed_start_idx=0, analysed_stop_idx=40_000)
    out, rep = me.carry_hand_rows(EventSet(df, prov), old, sig, fs=FS, units="pA",
                                  old_module_id="detect.schmitt")
    assert rep.left_out and "did not search" in rep.sentence("Schmitt", len(out.df))
    assert len(out.df) == len(df)


def test_a_table_with_no_hand_rows_comes_back_untouched():
    sig = _trace([(10_000, 10_300)])
    det, prov = detect_events(sig, FS, min_dwell_s=1e-4)
    new = EventSet(det, prov)
    out, rep = me.carry_hand_rows(new, det.copy(), sig, fs=FS, units="pA",
                                  old_module_id="detect.schmitt")
    assert out is new and rep == me.CarryReport() and rep.sentence("Schmitt", 1) == \
        "Re-ran Schmitt: 1 events."


# ------------------------------------------------------------------------------------------------
# "Include hand-made events" on the exports
# ------------------------------------------------------------------------------------------------
def _mixed_table():
    sig = _trace([(10_000, 10_300), (30_000, 30_300)])
    c = _contract(sig)
    det, _ = detect_events(sig, FS, min_dwell_s=1e-4)
    mixed = me.insert_hand_row(det, _row(sig, c, (50_000, 50_300), 99))
    return mixed


def test_the_training_export_leaves_hand_made_rows_out_and_says_how_many(tmp_path):
    import json
    from nanoscope.analysis.scoring import build_training_frame, write_training_export

    table = _mixed_table()
    assert len(build_training_frame(table)) == 3
    assert len(build_training_frame(table, include_added=False)) == 2
    out = write_training_export(table, tmp_path / "t.csv", provenance={"fs": FS},
                                include_added=False)
    meta = json.loads(out["meta"].read_text())
    block = meta["curation"]
    assert block["include_added"] is False and block["n_added_excluded"] == 1
    assert pd.read_csv(out["data"])["added"].sum() == 0
    on = write_training_export(table, tmp_path / "on.csv", provenance={"fs": FS})
    assert json.loads(on["meta"].read_text())["curation"]["include_added"] is True
    assert int(pd.read_csv(on["data"])["added"].sum()) == 1


def test_a_training_export_of_the_manual_table_says_added_means_drawn():
    import json
    from nanoscope.analysis.scoring import training_provenance

    table = _mixed_table()
    meta = training_provenance({"fs": FS}, detector="detect.manual", df=table, source_df=table)
    assert "drawn by hand" in meta["added_means"]


def test_the_events_sidecar_names_the_switch(tmp_path):
    import json
    from nanoscope.analysis.scoring import write_events_export

    table = _mixed_table()
    out = write_events_export(table.iloc[:2], tmp_path / "e.csv", provenance={"fs": FS},
                              n_total=2, include_added=False, n_added_excluded=1)
    block = json.loads(out["meta"].read_text())["curation"]
    assert block["include_added"] is False and block["n_added_excluded"] == 1
    assert block["n_withheld"] == 0           # rejected rows only: a choice about scope is not one


# ------------------------------------------------------------------------------------------------
# converting the old measurement list
# ------------------------------------------------------------------------------------------------
def _measurements(rows):
    """A measurement frame from ``(id, doc_key, channel_index, channel, start, end, label)``."""
    from nanoscope.analysis.measure import coerce_measurements

    recs = []
    for mid, key, ci, ch, s, e, label in rows:
        recs.append({"measurement_id": mid, "recording": "r.dat", "doc_key": key,
                     "channel": ch, "channel_index": ci, "start_idx": s, "end_idx": e,
                     "label": label, "units": "pA", "cal_gain": 1.0, "cal_offset": 0.0})
    return coerce_measurements(pd.DataFrame(recs))


def _target(**kw):
    from nanoscope.analysis.measurement_conversion import TargetChannel

    base = dict(doc_key="/data/r.dat", channel_index=0, channel_name="current",
                n_samples=100_000, fs=FS, units="pA", cal_gain=1.0, cal_offset=0.0)
    base.update(kw)
    return TargetChannel(**base)


def test_the_plan_places_a_row_only_on_its_own_recording_and_channel():
    from nanoscope.analysis.measurement_conversion import plan_conversion

    frame = _measurements([
        (0, "/data/r.dat", 0, "current", 1_000, 1_400, "ok"),
        (1, "/data/other.dat", 0, "current", 1_000, 1_400, "elsewhere"),
        (2, "/data/gone.dat", 0, "current", 1_000, 1_400, "closed"),
        (3, "/data/r.dat", 1, "voltage", 1_000, 1_400, "other channel"),
        (4, "", 0, "current", 1_000, 1_400, "blank key"),
        (5, "/data/r.dat", -1, "current", 1_000, 1_400, "no index"),
        (6, "/data/r.dat", 0, "current", 99_900, 100_900, "off the end"),
        (7, "/data/r.dat", 0, "current", 5_000, 5_400, "done already"),
    ])
    plan = plan_conversion(frame, _target(), converted_ids={7},
                           open_doc_keys=["/data/r.dat", "/data/other.dat"])
    by_id = {r.measurement_id: r for r in plan}
    assert by_id[0].placeable and by_id[0].reason == ""
    assert "select that recording" in by_id[1].reason
    assert "not open" in by_id[2].reason
    assert "another channel" in by_id[3].reason
    assert "did not record which recording" in by_id[4].reason
    assert "did not record which channel" in by_id[5].reason
    assert "outside the loaded data" in by_id[6].reason
    assert by_id[7].already_converted and not by_id[7].placeable
    assert [r.measurement_id for r in plan] == list(range(8))        # the list's own order


def test_the_plan_flags_rows_whose_numbers_will_change_and_refuses_without_a_rate():
    from nanoscope.analysis.measurement_conversion import plan_conversion

    frame = _measurements([(0, "/data/r.dat", 0, "current", 1_000, 1_400, "")])
    frame.loc[0, "cal_gain"] = 2.0                  # measured under another gain
    plan = plan_conversion(frame, _target())
    assert plan[0].placeable and plan[0].numbers_change
    none = plan_conversion(frame, _target(fs=None))
    assert not none[0].placeable and "no sample rate" in none[0].reason
    nothing = plan_conversion(frame, None)
    assert not nothing[0].placeable and "select a recording" in nothing[0].reason
    assert plan_conversion(None, _target()) == [] and plan_conversion(frame.iloc[:0], _target()) == []
