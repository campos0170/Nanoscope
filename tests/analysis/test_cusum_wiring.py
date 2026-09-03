"""CUSUM's two integration seams: the derived AnalysisResult, and the export pair.

The algorithm itself is covered in test_cusum.py. What is asserted here is that a level
fit behaves like the other derived fits this app already has -- same result shape, same
staleness stamps, same two-files-or-neither export contract -- because that is what
makes it storable, comparable and re-runnable rather than a number in a panel.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.cusum import LEVEL_SCHEMA, CusumConfig, empty_levels
from nanoscope.analysis.events import SCHEMA as EVENT_SCHEMA
from nanoscope.analysis.modules import fit_levels
from nanoscope.analysis.scoring import (
    levels_export_paths,
    pivot_levels_wide,
    write_levels_export,
)

FS = 50_000.0
SIGMA = 1.0


def _event_frame(spans, baseline_pA=100.0):
    rows = []
    for i, (a, b) in enumerate(spans):
        row = {c: (0 if d == "int64" else np.nan if d == "float64"
                   else False if d == "bool" else "")
               for c, (d, _, _) in EVENT_SCHEMA.items()}
        row.update(event_id=i, start_idx=a, end_idx=b, baseline_pA=baseline_pA,
                   accepted=True, recording="rec", pore_id="pore")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.astype({c: d for c, (d, _, _) in EVENT_SCHEMA.items()})[list(EVENT_SCHEMA)]


@pytest.fixture
def two_level_case():
    """One 2-level event and one flat event, in absolute-current space."""
    rng = np.random.default_rng(11)
    sig = np.full(2_000, 100.0) + rng.normal(0.0, SIGMA, 2_000)
    sig[500:700] += 20.0
    sig[700:900] += 35.0
    sig[1_200:1_400] += 25.0
    return sig, _event_frame([(500, 900), (1_200, 1_400)])


# ================================================================================================
# The derived AnalysisResult.
# ================================================================================================
def test_fit_levels_returns_a_derived_result_shaped_like_the_other_fits(two_level_case):
    sig, events = two_level_case
    res = fit_levels(events, sig, FS, doc_id="d", channel=0, baseline=100.0,
                     polarity=1, provenance={"noise_rms_pA": SIGMA})
    assert res.module_id == "fit.levels"
    assert res.doc_id == "d" and res.channel == 0
    assert list(res.payload.columns) == list(LEVEL_SCHEMA)
    assert len(res.payload) == 3           # 2 levels + 1 level
    assert res.scalars["n_events_fitted"] == 2
    assert res.scalars["n_levels_total"] == 3
    assert res.scalars["frac_multilevel"] == pytest.approx(0.5)


def test_the_params_name_every_number_that_moved_a_level(two_level_case):
    """A stored level table has to be able to say what produced it -- same reason
    write_events_export carries the baseline estimator."""
    sig, events = two_level_case
    cfg = CusumConfig(step_sigma=2.5, threshold_sigma=7.0, min_level_samples=4,
                      warmup_samples=10, max_levels=6)
    res = fit_levels(events, sig, FS, doc_id="d", channel=0, baseline=100.0,
                     polarity=1, provenance={"noise_rms_pA": SIGMA}, config=cfg)
    for key, want in (("step_sigma", 2.5), ("threshold_sigma", 7.0),
                      ("min_level_samples", 4), ("warmup_samples", 10), ("max_levels", 6)):
        assert res.params[key] == want
    assert res.params["cusum_fingerprint"] == cfg.fingerprint()
    assert res.module_version


def test_both_staleness_stamps_are_carried(two_level_case):
    """A level depth is measured against a baseline and a calibration exactly as an
    event depth is, so it goes stale for exactly the same two reasons."""
    sig, events = two_level_case
    res = fit_levels(events, sig, FS, doc_id="d", channel=0, baseline=100.0,
                     polarity=1, provenance={"noise_rms_pA": SIGMA},
                     calibration_fingerprint=(2.0, 0.5), baseline_fingerprint="abc123")
    assert res.calibration_fingerprint == (2.0, 0.5)
    assert res.baseline_fingerprint == "abc123"


def test_an_empty_fit_warns_rather_than_returning_silently(two_level_case):
    sig, events = two_level_case
    events = events.copy()
    events["accepted"] = False
    res = fit_levels(events, sig, FS, doc_id="d", channel=0, baseline=100.0,
                     polarity=1, provenance={"noise_rms_pA": SIGMA})
    assert res.payload.empty
    assert res.warnings, "an empty result must say why"


# ================================================================================================
# Export: the two-files-or-neither contract, and the wide pivot.
# ================================================================================================
def test_levels_export_writes_the_table_and_its_sidecar(tmp_path, two_level_case):
    sig, events = two_level_case
    levels = fit_levels(events, sig, FS, doc_id="d", channel=0, baseline=100.0,
                        polarity=1, provenance={"noise_rms_pA": SIGMA}).payload
    cfg = CusumConfig()
    out = write_levels_export(levels, tmp_path / "lv.csv",
                              provenance={"fs": FS, "polarity": 1, "units": "pA"},
                              cusum_params={"step_sigma": cfg.step_sigma,
                                            "cusum_fingerprint": cfg.fingerprint()},
                              source_file="a.abf", detector="detect.schmitt")
    assert set(out) == {"data", "meta"}
    assert tuple(out.values()) == levels_export_paths(tmp_path / "lv.csv")
    assert out["data"].exists() and out["meta"].exists()

    back = pd.read_csv(out["data"])
    assert list(back.columns) == list(LEVEL_SCHEMA)
    assert len(back) == len(levels)

    meta = json.loads(out["meta"].read_text())
    assert meta["format"] == "nanoscope-levels"
    assert meta["layout"] == "long"
    assert meta["n_rows"] == len(levels)
    assert meta["n_events"] == 2
    # The sidecar must be able to name the parameters that decided the level COUNT.
    assert meta["cusum_params"]["cusum_fingerprint"] == cfg.fingerprint()


def test_wide_export_is_one_row_per_event_with_a_pair_per_substate(tmp_path, two_level_case):
    """The shape the user's own '*Saved Data.txt' files already have."""
    sig, events = two_level_case
    levels = fit_levels(events, sig, FS, doc_id="d", channel=0, baseline=100.0,
                        polarity=1, provenance={"noise_rms_pA": SIGMA}).payload
    out = write_levels_export(levels, tmp_path / "wide.csv", wide=True)
    back = pd.read_csv(out["data"])
    assert len(back) == 2, "one row per EVENT in wide layout"
    assert list(back.columns)[:2] == ["event_id", "n_levels"]
    for col in ("L0_depth_pA", "L0_dwell_s", "L1_depth_pA", "L1_dwell_s"):
        assert col in back.columns
    # The flat event has no second level; that cell is empty, not zero.
    flat = back[back["event_id"] == 1].iloc[0]
    assert pd.isna(flat["L1_depth_pA"])
    assert json.loads(out["meta"].read_text())["layout"] == "wide"


def test_pivot_of_an_empty_table_does_not_explode():
    wide = pivot_levels_wide(empty_levels())
    assert wide.empty and list(wide.columns) == ["event_id", "n_levels"]


def test_export_leaves_nothing_behind_when_the_sidecar_cannot_be_written(tmp_path, two_level_case):
    """Two files or neither -- a table beside a stale sidecar is worse than no export."""
    sig, events = two_level_case
    levels = fit_levels(events, sig, FS, doc_id="d", channel=0, baseline=100.0,
                        polarity=1, provenance={"noise_rms_pA": SIGMA}).payload
    target = tmp_path / "blocked.csv"
    (tmp_path / "blocked.meta.json").mkdir()      # a directory where the sidecar must go
    with pytest.raises(OSError):
        write_levels_export(levels, target)
    assert not target.exists(), "the data file must not survive a failed sidecar"
