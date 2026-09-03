"""Round-trip tests for the pure ``.nanoscope`` (de)serializer (``core/project.py``).

No Qt: this exercises :func:`save_project` / :func:`load_project` on a plain
:class:`ProjectState`, asserting that

* the event DataFrames come back byte-for-byte — including the curation ``accepted``
  column and hand-edited ``start_idx``/``end_idx`` — and the manual measurements too;
* provenance dicts (with their numpy scalars + a nested polarity-evidence dict)
  survive the JSON round-trip intact;
* documents-as-references, calibrations, layers and view mode round-trip;
* a project whose version is missing / older / newer than this build raises a clear
  :class:`ProjectVersionError`, and a non-project file raises :class:`ProjectError`.

The event/provenance fixtures come from the REAL Schmitt detector so the provenance
is the genuine article (numpy dtypes, nested evidence), not a hand-rolled stand-in.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.measure import (
    append_measurement,
    empty_measurements_df,
    measure_selection,
)
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.core.document import CalibratedChannel
from nanoscope.core.project import (
    CalibrationRec,
    DocumentRef,
    FitRec,
    LayerRec,
    ProjectError,
    ProjectState,
    ProjectVersionError,
    ResultRec,
    load_project,
    save_project,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _detection(path: str = "pore_a.csv") -> tuple[pd.DataFrame, dict]:
    """Run the real Schmitt detector and hand back (df, provenance)."""
    arr = _event_trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=arr, units="pA", sample_rate_hz=FS, t0_s=0.0)
    meta = RecordingMeta(
        path=Path(path), format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    ctx = SimpleAnalysisContext(channel=ch, meta=meta, params={"min_dwell_s": 5e-4})
    res = DetectSchmitt().run(ctx)
    return res.payload.df.copy(), dict(res.payload.provenance)


def _measurements() -> pd.DataFrame:
    arr = _event_trace()
    df = empty_measurements_df()
    for i, (s, e) in enumerate(_EVENTS[:2]):
        row = measure_selection(arr, FS, s, e, units="pA")
        row["recording"] = "pore_a.csv"
        row["channel"] = "current"
        row["measurement_id"] = i
        df = append_measurement(df, row)
    return df


def _rewrite_version(src: Path, dst: Path, version) -> Path:
    """Copy a saved project, replacing (or dropping) the manifest version field."""
    with zipfile.ZipFile(src, "r") as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members["manifest.json"])
    if version is None:
        manifest.pop("version", None)
    else:
        manifest["version"] = version
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(dst, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return dst


# ================================================================================================
# the full round-trip
# ================================================================================================

def test_round_trip_preserves_events_curation_provenance_and_tables(tmp_path):
    df_a, prov_a = _detection("pore_a.csv")
    df_b, prov_b = _detection("pore_b.csv")

    # Curate result A: reject event 1, and hand-edit event 0's bounds (the Event Inspector path).
    df_a.loc[1, "accepted"] = False
    df_a.loc[0, "start_idx"] = int(df_a.loc[0, "start_idx"]) + 17
    df_a.loc[0, "end_idx"] = int(df_a.loc[0, "end_idx"]) - 9

    meas = _measurements()
    state = ProjectState(
        documents=[
            DocumentRef(path="/data/pore_a.csv", loaded_range=None),
            DocumentRef(path="/data/pore_b.csv", loaded_range=(0, 50_000)),
        ],
        calibrations=[
            CalibrationRec("/data/pore_a.csv", 0, 0.1, -2.0, "pA", "file_meta"),
            CalibrationRec("/data/pore_b.csv", 0, 1.0, 0.0, "", "unknown"),
        ],
        layers=[
            LayerRec("/data/pore_a.csv", 0, "#4FC3F7", True, 0.0),
            LayerRec("/data/pore_b.csv", 0, "#E57373", False, 0.25),
        ],
        view_mode="stacked",
        results=[
            ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df_a, prov_a),
            ResultRec("/data/pore_b.csv", 0, "detect.keras", df_b, prov_b),
        ],
        measurements=meas,
    )

    out = save_project(tmp_path / "proj.nanoscope", state)
    assert out.exists()
    back = load_project(out)

    # -- documents (references) ---------------------------------------------------------------
    assert [(d.path, d.loaded_range) for d in back.documents] == [
        ("/data/pore_a.csv", None), ("/data/pore_b.csv", (0, 50_000))]

    # -- calibrations -------------------------------------------------------------------------
    ca = next(c for c in back.calibrations if c.doc_id == "/data/pore_a.csv")
    assert (ca.gain, ca.offset, ca.target_units, ca.provenance) == (0.1, -2.0, "pA", "file_meta")

    # -- layers + view mode -------------------------------------------------------------------
    lb = next(lay for lay in back.layers if lay.doc_id == "/data/pore_b.csv")
    assert (lb.color, lb.visible, lb.time_offset_s) == ("#E57373", False, 0.25)
    assert back.view_mode == "stacked"

    # -- results: BOTH module ids, dfs byte-identical incl curation + edited bounds -----------
    assert {r.module_id for r in back.results} == {"detect.schmitt", "detect.keras"}
    ra = next(r for r in back.results if r.module_id == "detect.schmitt")
    assert_frame_equal(ra.events, df_a)
    assert list(ra.events["accepted"]) == list(df_a["accepted"])       # reject survived
    assert int(ra.events.loc[0, "start_idx"]) == int(df_a.loc[0, "start_idx"])   # edit survived
    assert int(ra.events.loc[0, "end_idx"]) == int(df_a.loc[0, "end_idx"])

    rb = next(r for r in back.results if r.module_id == "detect.keras")
    assert_frame_equal(rb.events, df_b)

    # -- provenance: deep-equal through JSON (numpy coerced, nested evidence intact) ----------
    assert isinstance(prov_a.get("polarity_evidence"), dict)
    expected = json.loads(json.dumps(_json_safe(prov_a)))
    assert ra.provenance == expected

    # -- measurements -------------------------------------------------------------------------
    assert_frame_equal(back.measurements, meas)


# ================================================================================================
# v3: a result's PARAMS and its calibration fingerprint (C5 / C6)
# ================================================================================================

def test_result_params_and_calibration_fingerprint_round_trip(tmp_path):
    """REGRESSION: neither field existed in the format, so a reopened detection could not say
    what settings produced it (C5) nor what gain it was computed under (C6)."""
    df, prov = _detection()
    params = {
        "polarity": None, "k_in": 5.0, "k_out": 2.0,
        "min_dwell_s": 0.0022, "max_dwell_s": None, "merge_gap_s": None,
        "filter_cutoff_hz": 10_000.0, "filter_order": 8, "filter_kind": "bessel",
        "detrend_enabled": True, "detrend_method": "percentile", "detrend_percentile": 70.0,
        "_calibration_fingerprint": (0.5, -1.0), "pore_id": "pore_a",
    }
    state = ProjectState(results=[
        ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df, prov,
                  params=params, calibration_fingerprint=(0.5, -1.0)),
    ])
    back = load_project(save_project(tmp_path / "p.nanoscope", state))
    got = back.results[0]

    assert got.params["min_dwell_s"] == 0.0022
    assert got.params["max_dwell_s"] is None          # the detector's off-sentinel survives
    assert got.params["polarity"] is None
    assert got.params["filter_kind"] == "bessel"
    assert got.params["detrend_percentile"] == 70.0
    assert got.params["_calibration_fingerprint"] == [0.5, -1.0]   # tuples land as JSON lists
    assert got.calibration_fingerprint == (0.5, -1.0)


def test_result_module_version_and_baseline_fingerprint_round_trip(tmp_path):
    """v10 keeps producer/staleness stamps even when provenance omits them."""
    df, prov = _detection()
    prov = {key: value for key, value in prov.items()
            if key != "baseline_fingerprint"}
    state = ProjectState(results=[
        ResultRec(
            "/data/pore_a.csv", 0, "detect.reconciled", df, prov,
            module_version="1.0", baseline_fingerprint="baseline-a",
        ),
    ])

    got = load_project(save_project(tmp_path / "v10.nanoscope", state)).results[0]

    assert got.module_version == "1.0"
    assert got.baseline_fingerprint == "baseline-a"


def test_a_result_with_no_fingerprint_reports_none_rather_than_guessing(tmp_path):
    """An unknown stamp must read as unknown. Anything else — a default, a zero pair — would
    be a fingerprint that compares EQUAL to some live calibration and silently declares a
    result fresh."""
    df, prov = _detection()
    state = ProjectState(results=[ResultRec("d", 0, "detect.schmitt", df, prov)])
    back = load_project(save_project(tmp_path / "p.nanoscope", state))
    assert back.results[0].calibration_fingerprint is None
    assert back.results[0].params == {}


def test_a_pre_v3_project_still_loads_without_params_or_fingerprint(tmp_path):
    """Back-compat: v1/v2 manifests have neither key. They must load, with the same empty
    params and absent stamp a project written before the fields existed implied."""
    df, prov = _detection()
    out = save_project(tmp_path / "p.nanoscope", ProjectState(
        results=[ResultRec("d", 0, "detect.schmitt", df, prov, params={"k_in": 5.0},
                           calibration_fingerprint=(2.0, 0.0))]))

    # Strip the v3 keys and re-stamp the manifest as version 2, i.e. an old file.
    members = {}
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            members[name] = zf.read(name)
    manifest = json.loads(members["manifest.json"])
    manifest["version"] = 2
    for r in manifest["results"]:
        r.pop("params", None)
        r.pop("calibration_fingerprint", None)
    members["manifest.json"] = json.dumps(manifest).encode()
    old = tmp_path / "v2.nanoscope"
    with zipfile.ZipFile(old, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    back = load_project(old)
    assert back.results[0].params == {}
    assert back.results[0].calibration_fingerprint is None
    assert_frame_equal(back.results[0].events, df)


def test_a_malformed_fingerprint_reads_as_unknown(tmp_path):
    """A stamp that cannot be compared must never masquerade as one that matches."""
    from nanoscope.core.project import _fingerprint_from_json, _fingerprint_to_json

    for bad in (None, [], [1.0], [1.0, 2.0, 3.0], "1,2", "12", b"12", {"gain": 1.0}):
        assert _fingerprint_from_json(bad) is None
        assert _fingerprint_to_json(bad) is None
    assert _fingerprint_to_json((1.5, -2.0)) == [1.5, -2.0]
    assert _fingerprint_from_json([1.5, -2.0]) == (1.5, -2.0)


def test_empty_project_round_trips(tmp_path):
    out = save_project(tmp_path / "empty.nanoscope", ProjectState())
    back = load_project(out)
    assert back.documents == []
    assert back.results == []
    assert back.measurements is None
    assert back.view_mode == "overlay"


def test_provenance_coerces_numpy_scalars_and_arrays(tmp_path):
    prov = {
        "fs": np.float64(100_000.0),
        "n_events": np.int64(5),
        "flag": np.bool_(True),
        "thresholds": np.array([1.5, 2.5, 3.5]),
        "evidence": {"verdict": "blockade", "score": np.float32(0.87)},
        "rise_s": float("nan"),
    }
    df, _ = _detection()
    state = ProjectState(results=[ResultRec("d", 0, "detect.schmitt", df, prov)])
    back = load_project(save_project(tmp_path / "p.nanoscope", state))
    got = back.results[0].provenance
    assert got["fs"] == 100_000.0 and isinstance(got["fs"], float)
    assert got["n_events"] == 5 and isinstance(got["n_events"], int)
    assert got["flag"] is True
    assert got["thresholds"] == [1.5, 2.5, 3.5]
    assert got["evidence"]["score"] == pytest.approx(0.87, abs=1e-6)
    assert np.isnan(got["rise_s"])


# ================================================================================================
# version + format guards
# ================================================================================================

def test_missing_version_raises_clear_error(tmp_path):
    out = save_project(tmp_path / "p.nanoscope", ProjectState())
    bad = _rewrite_version(out, tmp_path / "noversion.nanoscope", None)
    with pytest.raises(ProjectVersionError, match="no format version"):
        load_project(bad)


def test_older_version_raises_clear_error(tmp_path):
    out = save_project(tmp_path / "p.nanoscope", ProjectState())
    bad = _rewrite_version(out, tmp_path / "old.nanoscope", 0)
    with pytest.raises(ProjectVersionError, match="older project"):
        load_project(bad)


def test_newer_version_raises_clear_error(tmp_path):
    out = save_project(tmp_path / "p.nanoscope", ProjectState())
    bad = _rewrite_version(out, tmp_path / "new.nanoscope", 999)
    with pytest.raises(ProjectVersionError, match="newer NanoScope"):
        load_project(bad)


def test_not_a_project_raises_clear_error(tmp_path):
    junk = tmp_path / "not_a_zip.nanoscope"
    junk.write_text("this is not a zip archive")
    with pytest.raises(ProjectError, match="not a NanoScope project"):
        load_project(junk)


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(ProjectError, match="no such file"):
        load_project(tmp_path / "does_not_exist.nanoscope")


# core.project keeps _json_safe private; import lazily so the public API test above does not
# depend on it existing under that name.
from nanoscope.core.project import _json_safe  # noqa: E402


def test_a_results_caveats_survive_the_round_trip_and_an_old_project_still_reads(tmp_path):
    """MEASURED BEFORE: ``warnings`` was never persisted, so every reopened project
    restored ``warnings=()``. Two consequences, one of them silent:

    * the caveat those numbers were produced under — above all "this recording's
      open-pore level was never probed" — no longer travelled with them;
    * pressing Re-measure on a reopened project then MOVED ``warnings`` from 0 to 1 with
      no number changing at all, which ``core.history`` reads as a state change: it pushed
      an undo step labelled "Re-run detection" (the most destructive click in the app,
      named for an operation the user did not perform) and silently cleared the redo stack.

    The key is additive and optional, so a project written before it still loads and still
    restores ``()`` — asserted here, because a reader that required the key would refuse
    every project the owner already has.
    """
    df, prov = _detection("pore_a.csv")
    caveat = ("This recording's open-pore level was never probed, so these depths came "
              "from an unconfirmed default.")
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        results=[ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df, prov,
                           warnings=(caveat,))],
    )
    out = save_project(tmp_path / "warn.nanoscope", state)
    back = load_project(out)
    assert back.results[0].warnings == (caveat,)

    # A project whose manifest predates the key restores exactly as it always did.
    import json
    import zipfile

    older = tmp_path / "older.nanoscope"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(older, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                for r in manifest["results"]:
                    r.pop("warnings", None)
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)
    assert load_project(older).results[0].warnings == ()


# ================================================================================================
# v6 — DERIVED fits (fit.dwell / fit.depth / fit.levels)
# ================================================================================================
# MEASURED BEFORE: the save path collected only `isinstance(res.payload, EventSet)`, so every
# derived fit was dropped on save with no message. A user could fit a dwell distribution,
# bootstrap its CIs, save, reopen — and find the fit simply gone.
#
# The interesting half is not the numbers (they are scalars) but the CURVE. A FitResult's
# density is a closure and cannot be pickled, so it is rebuilt on load by stats.density_from()
# — the SAME function the fitters call. These tests assert the reconstruction is an identity,
# not an approximation, because a curve that drifts from the tau/AIC printed beside it would
# be the worst kind of failure here: silent, plausible, and wrong.


def _dwell_fit(*, efficiency=None, t_max=None):
    """A real fit_dwell_exponential result — closure, eta and all."""
    from nanoscope.analysis.stats import fit_dwell_exponential

    rng = np.random.default_rng(0)
    t = rng.exponential(2e-3, 900)
    t = t[t > 1e-4]
    if t_max is not None:
        t = t[t <= t_max]
    return fit_dwell_exponential(t, t_min=1e-4, t_max=t_max, n_components=1,
                                 n_boot=20, seed=1, efficiency=efficiency)


@pytest.mark.parametrize("efficiency, t_max, label", [
    (None, None, "plain left-truncated"),
    (None, 1e-2, "doubly truncated"),
    ((np.geomspace(1e-4, 1e-2, 12),
      np.clip(1 - np.exp(-np.geomspace(1e-4, 1e-2, 12) / 5e-4), 0, 1)), None, "eta-weighted"),
])
def test_a_dwell_fits_curve_is_identical_after_a_round_trip(tmp_path, efficiency, t_max, label):
    """The restored density must equal the original POINTWISE — `array_equal`, not
    `allclose`. It can, because both come from stats.density_from().

    The eta-weighted case is the one that made this feature look impossible: the measured
    efficiency curve was neither a parameter nor a scalar, it lived only inside the
    fitter's closure, so a fit could be stored as numbers but never redrawn. It is now
    written into ``meta['eta_grid']``/``meta['eta_vals']`` and comes back exact.
    """
    from nanoscope.analysis.stats import fit_from_json, fit_to_json

    fit = _dwell_fit(efficiency=efficiency, t_max=t_max)
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        fits=[FitRec("/data/pore_a.csv", 0, "fit.dwell", curve=fit_to_json(fit))],
    )
    back = load_project(save_project(tmp_path / "f.nanoscope", state))
    restored = fit_from_json(back.fits[0].curve)

    x = np.geomspace(1e-4, 2e-2, 200)
    assert np.array_equal(fit.predict(x), restored.predict(x)), f"{label}: curve drifted"
    assert restored.params == fit.params
    assert restored.aic == fit.aic and restored.bic == fit.bic
    assert restored.n == fit.n and restored.model == fit.model


def test_a_depth_fits_curve_is_identical_after_a_round_trip(tmp_path):
    """Same guarantee for the Gaussian depth mixture, whose density is untruncated and
    reconstructs from ``params`` alone."""
    from nanoscope.analysis.stats import fit_depth_gaussians, fit_from_json, fit_to_json

    rng = np.random.default_rng(0)
    d = np.r_[rng.normal(0.2, 0.03, 400), rng.normal(0.5, 0.05, 300)]
    fit = fit_depth_gaussians(d, n_components=2, n_boot=20, seed=1)
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        fits=[FitRec("/data/pore_a.csv", 0, "fit.depth", curve=fit_to_json(fit))],
    )
    back = load_project(save_project(tmp_path / "d.nanoscope", state))
    x = np.linspace(0.05, 0.8, 200)
    assert np.array_equal(fit.predict(x), fit_from_json(back.fits[0].curve).predict(x))


@pytest.mark.parametrize("model", ["gamma", "inverse-gaussian"])
def test_a_peaked_dwell_fits_curve_is_identical_after_a_round_trip(tmp_path, model):
    """The peaked families carry their truncation mass in ``meta['t_min'/'t_max']``, so
    they reconstruct exactly too."""
    from nanoscope.analysis.stats import fit_dwell_peaked, fit_from_json, fit_to_json

    rng = np.random.default_rng(0)
    fit = fit_dwell_peaked(rng.gamma(3, 4e-4, 600), model=model, t_min=1e-5)
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        fits=[FitRec("/data/pore_a.csv", 0, "fit.dwell", curve=fit_to_json(fit))],
    )
    back = load_project(save_project(tmp_path / f"{model}.nanoscope", state))
    x = np.geomspace(1e-5, 2e-2, 200)
    assert np.array_equal(fit.predict(x), fit_from_json(back.fits[0].curve).predict(x))


def test_a_level_table_survives_the_round_trip(tmp_path):
    """``fit.levels`` is a DataFrame payload and rides as parquet, exactly like an
    EventSet — dtypes included."""
    levels = pd.DataFrame({
        "event_id": np.array([0, 0, 1], dtype=np.int64),
        "level_idx": np.array([0, 1, 0], dtype=np.int64),
        "n_levels": np.array([2, 2, 1], dtype=np.int64),
        "level_pA": np.array([101.5, 88.25, 97.0], dtype=np.float64),
        "frac_depth": np.array([0.05, 0.19, 0.09], dtype=np.float64),
    })
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        fits=[FitRec("/data/pore_a.csv", 0, "fit.levels", table=levels)],
    )
    back = load_project(save_project(tmp_path / "lv.nanoscope", state))
    assert_frame_equal(back.fits[0].table, levels)
    assert back.fits[0].curve is None


def test_a_fits_staleness_stamps_come_back_from_the_file(tmp_path):
    """BOTH stamps must survive, and must NOT be re-derived on load.

    This is the defect ResultRec's own docstring describes, one level down: a fit measured
    under gain 1 and reopened after a re-gain must still read as stale. Re-deriving either
    stamp from the live session clears a banner that is telling the truth — and for the
    baseline stamp the owner's own -600 mV apo file moves frac_depth by 16x.
    """
    from nanoscope.analysis.stats import fit_to_json

    caveat = "open-pore level was never probed"
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        fits=[FitRec("/data/pore_a.csv", 0, "fit.dwell", curve=fit_to_json(_dwell_fit()),
                     params={"t_min": 1e-4, "n_components": 1},
                     calibration_fingerprint=(2.0, 0.5),
                     baseline_fingerprint="bfp-deadbeef",
                     warnings=(caveat,))],
    )
    back = load_project(save_project(tmp_path / "s.nanoscope", state))
    rec = back.fits[0]
    assert rec.calibration_fingerprint == (2.0, 0.5)
    assert rec.baseline_fingerprint == "bfp-deadbeef"
    assert rec.warnings == (caveat,)
    assert rec.params["t_min"] == 1e-4


def test_a_project_without_the_fits_key_still_loads(tmp_path):
    """`fits` is additive and optional. Every project the owner already has predates it,
    and a reader that required the key would refuse all of them."""
    df, prov = _detection("pore_a.csv")
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        results=[ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df, prov)],
    )
    out = save_project(tmp_path / "old.nanoscope", state)
    older = tmp_path / "older.nanoscope"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(older, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                manifest.pop("fits", None)
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)
    back = load_project(older)
    assert back.fits == []
    assert len(back.results) == 1      # and the rest of the project is untouched


def test_a_fit_from_an_unknown_family_keeps_its_numbers_and_drops_its_curve(tmp_path):
    """Forward compatibility. A fit written by a build that knows a family this one does
    not must restore its params/AIC — and must REFUSE to draw a curve, rather than
    silently guessing one from a family it cannot identify.
    """
    from nanoscope.analysis.stats import fit_from_json, fit_to_json

    blob = fit_to_json(_dwell_fit())
    blob["meta"]["family"] = "weibull-from-the-future"
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        fits=[FitRec("/data/pore_a.csv", 0, "fit.dwell", curve=blob)],
    )
    back = load_project(save_project(tmp_path / "fut.nanoscope", state))
    restored = fit_from_json(back.fits[0].curve)
    assert restored.params and np.isfinite(restored.aic)
    with pytest.raises(RuntimeError, match="carries no density"):
        restored.predict(np.array([1e-3]))


def test_a_missing_level_table_member_is_named_in_the_error(tmp_path):
    """A truncated archive must say WHICH member is gone, matching how a missing
    ``results/<i>.parquet`` already reports."""
    levels = pd.DataFrame({"event_id": [0], "level_idx": [0], "level_pA": [1.0]})
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        fits=[FitRec("/data/pore_a.csv", 0, "fit.levels", table=levels)],
    )
    out = save_project(tmp_path / "trunc.nanoscope", state)
    broken = tmp_path / "broken.nanoscope"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(broken, "w") as dst:
        for item in src.infolist():
            if item.filename.startswith("fits/"):
                continue
            dst.writestr(item, src.read(item.filename))
    with pytest.raises(ProjectError, match=r"fits/0\.parquet"):
        load_project(broken)


def test_the_format_version_bumped_so_an_older_build_refuses_instead_of_stripping(tmp_path):
    """v6 bumps VERSION even though ``fits`` is additive on READ.

    Additive-ness decides whether THIS build can read OLDER files. It says nothing about
    whether older builds can safely read this one — and they cannot: a build that predates
    ``fits`` accepts a version-5 file, ignores the key and the ``fits/<i>.parquet`` members,
    and drops them on its next save, silently, because it never knew they were there. That
    is the loss FitRec exists to end, reintroduced one level up and made worse by living in
    a file that gives no sign of it. (Verified against the build installed at
    /Applications/NanoScope.app, which contains neither ``fits/`` nor ``eta_grid``.)

    The bump costs nothing in the other direction, which is the second half of this test:
    MIN_READ_VERSION stays 1, so every project the owner already has still opens.
    """
    from nanoscope.core.project import MIN_READ_VERSION, VERSION

    assert VERSION >= 6, "a project carrying fits must not be labelled with a version that predates them"
    assert MIN_READ_VERSION == 1, "bumping must not cost backward read compatibility"

    # A v5 project — what every project the owner already has looks like — still loads.
    df, prov = _detection("pore_a.csv")
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        results=[ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df, prov)],
    )
    out = save_project(tmp_path / "now.nanoscope", state)
    older = tmp_path / "v5.nanoscope"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(older, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["version"] = 5
                manifest.pop("fits", None)
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)
    assert len(load_project(older).results) == 1


def test_the_label_palette_survives_the_round_trip(tmp_path):
    """The vocabulary is session-wide, so it belongs to the project rather than to a result.
    A palette that reset between sessions would retrain the user's fingers every time."""
    from nanoscope.analysis.labels import LabelVocabulary

    v = LabelVocabulary()
    v.add("molecule", "apo-ferritin", hotkey="a", note="500 pM")
    v.add("type", "collision")
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        label_vocabulary=v.to_json(),
    )
    back = load_project(save_project(tmp_path / "lab.nanoscope", state))
    got = LabelVocabulary.from_json(back.label_vocabulary)
    assert [e.token for e in got.entries] == ["molecule=apo-ferritin", "type=collision"]
    assert got.entries[0].hotkey == "a" and got.entries[0].note == "500 pM"


def test_a_project_without_a_palette_loads_with_an_empty_one(tmp_path):
    state = ProjectState(documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)])
    out = save_project(tmp_path / "nolab.nanoscope", state)
    older = tmp_path / "older.nanoscope"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(older, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "manifest.json":
                m = json.loads(data)
                m.pop("label_vocabulary", None)
                data = json.dumps(m).encode()
            dst.writestr(item, data)
    assert load_project(older).label_vocabulary == []


# ================================================================================================
# v9 — the colour identity map
# ================================================================================================

def test_the_colour_identity_map_survives_the_round_trip(tmp_path):
    """D1's persistence half. Until v9 a recording's colour was stored for exactly ONE
    of the four surfaces that draw it (``layers[].color``); the Plots panel recoloured
    from position on every refresh, so a hand-picked colour was gone the moment the
    window was reopened, and gone for good on the next save.

    Keyed by the PERSISTENT key like `calibrations` / `layers` / `results`, so it
    re-attaches to a segment re-derived on load rather than to an in-memory id that no
    longer exists.
    """
    from nanoscope.core.layer import ColourRegistry

    reg = ColourRegistry()
    reg.slot("/data/pore_a.csv#0")
    reg.slot("/data/pore_b.csv#0")
    reg.set_ink("/data/pore_b.csv#0", "#FF00FF")

    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        colours=reg.to_json(),
    )
    back = load_project(save_project(tmp_path / "col.nanoscope", state))
    got = ColourRegistry.from_json(back.colours)

    assert got.slots == reg.slots
    assert got.ink("/data/pore_b.csv#0") == "#FF00FF"
    # ...and pore_b is not even an open document here. A colour for a recording this
    # session has not loaded is KEPT, because the recording comes back and the colour
    # must come back with it.
    assert got.is_user_set("/data/pore_b.csv#0")


def test_a_project_without_a_colour_map_loads_with_an_empty_one(tmp_path):
    """v1-v8 have no such key. An empty registry means "nobody has assigned anything",
    and the first discovery then hands out slots in the order it always did — which is
    visually what that project last showed, because a project with no registry is one
    that was drawing positional colours anyway."""
    from nanoscope.core.layer import ColourRegistry

    state = ProjectState(documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)])
    out = save_project(tmp_path / "nocol.nanoscope", state)
    older = tmp_path / "older.nanoscope"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(older, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "manifest.json":
                m = json.loads(data)
                m.pop("colours", None)
                m["version"] = 8
                data = json.dumps(m).encode()
            dst.writestr(item, data)
    back = load_project(older)
    assert back.colours == []
    assert len(ColourRegistry.from_json(back.colours)) == 0


def test_the_layer_ink_is_stored_beside_the_registry_and_not_derived_from_it(tmp_path):
    """``layers[].color`` is what is ON SCREEN; the registry is the RULE that chose it.

    Deriving one from the other on load would silently repaint a project saved by a
    build whose palette differed — which is exactly what this pass just did to
    LAYER_PALETTE, so it is not a hypothetical.
    """
    from nanoscope.core.layer import ColourRegistry

    reg = ColourRegistry()
    reg.slot("/data/pore_a.csv#0")
    state = ProjectState(
        documents=[DocumentRef(path="/data/pore_a.csv", loaded_range=None)],
        layers=[LayerRec("/data/pore_a.csv", 0, "#4FC3F7", True, 0.0)],   # an OLD palette ink
        colours=reg.to_json(),
    )
    back = load_project(save_project(tmp_path / "both.nanoscope", state))
    assert back.layers[0].color == "#4FC3F7", "the restored layer was repainted by the registry"
