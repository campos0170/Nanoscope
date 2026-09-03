"""The four call sites, unified on one `BaselineConfig` -- and the guard that keeps
a fifth from picking its own default.

THE DEFECT
----------
Four places estimated a characterisation baseline, at four settings:

  * ``analysis/detect.py``      -- ``estimate_baseline(cur, fs, polarity=...)``, i.e.
    the silent 10 ms ``block_median`` default, entirely unexposed.
  * ``analysis/ml_modules.py``  -- ``_CHAR_BASELINE_WINDOW_S = 0.20``, widened to fix
    a real zoom-dependence defect, but only here.
  * ``analysis/characterize.py::recharacterize_span`` -- a +-3 ms flank median.
  * ``analysis/truth.py``       -- the 10 ms default again; the site earlier accounts
    of this defect missed entirely.

So ONE event on the owner's -600 mV apo recording read **32.04 pA** (Schmitt),
**37.54 pA** (rebuild) and **67.69 pA** (ML), and the population median depth was
23.23 pA or 111.31 pA depending on which detector found it.

WHY THE GUARDS AND NOT JUST THE PARAMETERS
------------------------------------------
Giving each site a ``baseline_config=None`` parameter fixes those four sites. It
does not fix the CLASS: ``estimate_baseline`` is public, takes ``window_s=``, and
has a silent default, so the fifth site that hits a baseline problem fixes it
locally exactly as ``ml_modules`` did -- for a good reason, with a measurement to
back it up, and the app reports two depths again. The guarantee therefore lives
where every site must pass through:

  * `baseline.characterisation_baseline` is the only door,
  * `baseline.DEFAULT_BASELINE` is the only baseline-parameter literal,
  * and the AST guards at the bottom of this file fail the build if another module
    calls `estimate_baseline`, defines a baseline window constant, or hands the
    door a literal window.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import truth as T
from nanoscope.analysis.characterize import (
    REBUILD_BASELINE_SOURCES,
    characterize,
    recharacterize_span,
)
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import SCHEMA
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.analysis.modules import _CONSUMED_PARAM_KEYS, preprocess_params_from_provenance
from nanoscope.analysis.scoring import (
    BASELINE_CONTRACT_KEYS,
    events_export_paths,
    write_events_export,
)

FS = 50_000.0
I0 = -100.0
DEPTH = 60.0


@pytest.fixture(autouse=True)
def _clear_cache():
    """The whole-recording memo is keyed on the trace OBJECT, so it cannot leak
    between tests that build their own arrays -- but clearing it keeps a failure
    from being explained by a hit."""
    B.clear_baseline_cache()
    yield
    B.clear_baseline_cache()


def trace(n=400_000, *, seed=7, depth=DEPTH, dwell=300, gap=4000, drift=0.0, f=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = I0 + drift * np.sin(2 * np.pi * f * t) + rng.normal(0, 5.0, n)
    for s in range(2000, n - 2000, gap):
        x[s:s + dwell] -= depth
    return x.astype(np.float32)


# ===========================================================================
# 1. ONE CONFIG REACHES ALL FOUR SITES
# ===========================================================================


def test_all_four_call_sites_stamp_the_same_fingerprint():
    """I2. The four sites resolve to the SAME `BaselineConfig` when none is passed,
    asserted the only way that cannot be faked -- by comparing the digest each one
    stamps, not by reading their source.

    This is the whole fix in one assertion. Before it, the four fingerprints would
    have been ``block_median@0.01`` / ``block_median@0.20`` / a flank median with no
    fingerprint at all / ``block_median@0.01``.
    """
    x = trace()
    fp = B.DEFAULT_BASELINE.fingerprint(FS)

    # site 1 -- the Schmitt detector
    df, prov = detect_events(x, FS, polarity=-1)
    assert prov["baseline_fingerprint"] == fp

    # site 2 -- the ML characteriser measures the whole channel through the door
    ml = B.characterisation_baseline(
        np.asarray(x, dtype=np.float64), FS, polarity=-1)
    assert ml.fingerprint == fp

    # site 3 -- a row rebuild reads the config out of the rows' own provenance
    assert B.BaselineConfig.reproducible_from(prov).fingerprint(FS) == fp

    # site 4 -- truth
    rec = T.SynthRecording(data=np.asarray(x), sample_rate_hz=FS, polarity=-1,
                           name="t", meta={})
    _, method = T._baseline_of(rec)
    assert method == B.DEFAULT_BASELINE.method


def test_one_event_reads_the_same_depth_through_every_path():
    """The headline defect: one event, three answers. All three paths must now
    produce the same ``baseline_pA`` and ``depth_pA`` to the last bit."""
    x = trace()
    df, prov = detect_events(x, FS, polarity=-1)
    row = df.iloc[5]
    s, e = int(row.start_idx), int(row.end_idx)

    rebuilt = recharacterize_span(x, FS, s, e, provenance=prov, prior=row)
    assert rebuilt.attrs["baseline_source"] == "recording"

    # The ML characteriser's own arithmetic, verbatim: whole-channel baseline
    # through the door, sliced, then the shared `characterize`.
    full = np.asarray(x, dtype=np.float64)
    base_full = B.characterisation_baseline(full, FS, polarity=-1).baseline
    ml_row = characterize(
        full, base_full, np.array([s]), np.array([e]), FS, -1,
        noise_rms=float(prov["noise_rms_pA"]), clip_bounds=(0, full.size),
    )

    assert float(rebuilt["baseline_pA"].iloc[0]) == float(row.baseline_pA)
    assert float(ml_row["baseline_pA"].iloc[0]) == float(row.baseline_pA)
    assert float(rebuilt["depth_pA"].iloc[0]) == float(row.depth_pA)
    assert float(ml_row["depth_pA"].iloc[0]) == float(row.depth_pA)


def test_a_non_default_config_flows_all_the_way_to_the_rebuilt_row():
    """Not just the default: whatever the user confirms must reach every path.
    A config that only reached the detector would put a rebuilt row back on the
    default, which is the defect wearing a different hat."""
    cfg = B.BaselineConfig(rough_s=1.0, base_s=0.05, decim=40, mask_sigma=4.0)
    x = trace()
    df, prov = detect_events(x, FS, polarity=-1, baseline_config=cfg,
                             baseline_param_source="user")
    assert prov["baseline_fingerprint"] == cfg.fingerprint(FS)
    assert prov["baseline_param_source"] == "user"
    assert prov["baseline_rough_s"] == 1.0 and prov["baseline_decim"] == 40

    row = df.iloc[5]
    rebuilt = recharacterize_span(x, FS, int(row.start_idx), int(row.end_idx),
                                  provenance=prov, prior=row)
    assert float(rebuilt["baseline_pA"].iloc[0]) == float(row.baseline_pA)
    # ...and it is genuinely a DIFFERENT number from the default's answer
    d_df, _ = detect_events(x, FS, polarity=-1)
    assert float(d_df.iloc[5].baseline_pA) != float(row.baseline_pA)


def test_merge_lands_every_rebuilt_row_on_the_detectors_baseline():
    """R8. The merge was the easiest site to miss because it never called
    `estimate_baseline` at all. A merged row must sit on the same open-pore
    reference as the rows it did not merge."""
    from nanoscope.analysis.events import EventSet

    n_ev = 400_000 // 4000
    x = trace(gap=4000)
    # two sub-events 1 ms apart, so a 2 ms merge collapses each pair
    for s in range(2000 + 350, 400_000 - 2000, 4000):
        x[s:s + 200] -= DEPTH
    df, prov = detect_events(x, FS, polarity=-1)
    merged, n_merged = merge_by_gap(EventSet(df, prov), x, FS, max_gap_s=2e-3)
    assert n_merged > 0

    base_full = B.characterisation_baseline(x, FS, polarity=-1).baseline
    for _, r in merged.df.iterrows():
        span = base_full[int(r.start_idx):int(r.end_idx)]
        # `characterize` stamps the MEAN of the baseline over the event's own
        # samples; a rebuilt row must be reading THAT array, not a local flank median.
        assert float(r.baseline_pA) == pytest.approx(float(np.mean(span)), abs=1e-9)


def test_the_estimator_is_reached_once_per_trace_not_once_per_event():
    """The memo is what makes (2) affordable. Without it a merge of N runs pays N
    whole-recording estimates; with it, one."""
    x = trace()
    df, prov = detect_events(x, FS, polarity=-1)
    calls = {"n": 0}
    real = B.estimate_baseline_result

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    B.estimate_baseline_result = counting
    try:
        for _, row in df.head(20).iterrows():
            recharacterize_span(x, FS, int(row.start_idx), int(row.end_idx),
                                provenance=prov, prior=row)
    finally:
        B.estimate_baseline_result = real
    assert calls["n"] == 1, f"{calls['n']} estimates for 20 rebuilds"


def test_the_memo_cannot_serve_a_different_trace_under_a_recycled_id():
    """The cache key is object identity validated through a weakref. If the entry's
    referent is gone the key is dropped, so a recycled ``id()`` can never hit."""
    cfg = B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50)
    a = trace(n=200_000, seed=1)
    B.characterisation_baseline(a, FS, config=cfg, polarity=-1, health=False)
    key = B._BaselineCache._key(a, FS, cfg, -1, False)
    assert B._CACHE.get(key, a) is not None
    b = trace(n=200_000, seed=2)
    assert B._CACHE.get(key, b) is None      # same key, different object -> miss
    assert B._CACHE.get(key, a) is None      # ...and the stale entry was dropped


# ===========================================================================
# 2. recharacterize_span -- the site that needs different treatment
# ===========================================================================


def test_rebuild_falls_back_to_flanks_when_no_estimator_can_be_named():
    """A row detected before this change carries no ``baseline_fingerprint``, and one
    whose detector was HANDED a baseline array carries ``baseline_method='provided'``.
    Neither names a reproducible estimator. Guessing one would be worse than the
    local median the function has always used, so it declines and says so."""
    x = trace()
    df, prov = detect_events(x, FS, polarity=-1)
    row = df.iloc[5]

    legacy = {k: v for k, v in prov.items() if k != "baseline_fingerprint"}
    out = recharacterize_span(x, FS, int(row.start_idx), int(row.end_idx),
                              provenance=legacy, prior=row)
    assert out.attrs["baseline_source"] == "flanks"

    _, prov_provided = detect_events(x, FS, polarity=-1, baseline=np.full(x.size, I0))
    assert prov_provided["baseline_method"] == "provided"
    assert "baseline_fingerprint" not in prov_provided
    assert B.BaselineConfig.reproducible_from(prov_provided) is None


def test_rebuild_prefers_an_explicitly_passed_baseline_array():
    x = trace()
    df, prov = detect_events(x, FS, polarity=-1)
    row = df.iloc[5]
    flat = np.full(x.size, -12.5)
    out = recharacterize_span(x, FS, int(row.start_idx), int(row.end_idx),
                              provenance=prov, prior=row, baseline=flat)
    assert out.attrs["baseline_source"] == "provided"
    assert float(out["baseline_pA"].iloc[0]) == pytest.approx(-12.5)


def test_a_non_finite_pad_still_poisons_the_row_and_outranks_the_estimator():
    """The poisoning rule is what routes a corrupt measurement window into
    `characterize`'s ``non_finite`` token. A clean whole-recording baseline read
    around it would launder the row into a finite, accepted, UNFLAGGED one -- the
    precise failure the rule was written to stop."""
    x = trace().astype(np.float64)
    df, prov = detect_events(x.astype(np.float32), FS, polarity=-1)
    row = df.iloc[5]
    s, e = int(row.start_idx), int(row.end_idx)
    x[s - 50] = np.nan                       # inside the pad, outside the event
    out = recharacterize_span(x, FS, s, e, provenance=prov, prior=row)
    assert out.attrs["baseline_source"] == "non_finite"
    assert not bool(out["accepted"].iloc[0])
    assert "non_finite" in str(out["flag"].iloc[0])


def test_one_bad_sample_far_away_leaves_every_distant_rebuilt_row_bit_identical():
    """The regression the whole-recording rebuild introduced, and the reason
    `repair_nonfinite` exists.

    Re-deriving the detector's baseline is what makes a rebuilt row land on its
    neighbours' numbers -- but the estimator REFUSES a non-finite trace, by design
    (a NaN silently disables its event mask), so one bad sample anywhere downgraded
    EVERY rebuild in the table to the local flank median. Measured on the owner's
    -600 mV recording: one ``inf`` at sample 1.9 M moved row 0's ``baseline_pA``
    from -27.16 to -31.13 pA. That is the blast radius going from "the pad" to "the
    table", which is precisely what this project's non-finite rule forbids.

    The estimate is taken over the repaired trace; the ROW-level poisoning is
    untouched, so the rows that actually read the bad sample still lose their
    numbers and still carry the token.
    """
    x = trace().astype(np.float64)
    df, prov = detect_events(x.astype(np.float32), FS, polarity=-1)
    near, far = df.iloc[40], df.iloc[5]
    victim = int(near.start_idx) - 50
    poisoned = x.copy()
    poisoned[victim] = np.inf

    # The mechanism: the door still refuses this trace outright.
    with pytest.raises(ValueError, match="non-finite|NaN|Inf"):
        B.characterisation_baseline(poisoned, FS, polarity=-1, health=False)

    def rebuild(sig, row):
        return recharacterize_span(sig, FS, int(row.start_idx), int(row.end_idx),
                                   provenance=prov, prior=row)

    clean_far, dirty_far = rebuild(x, far), rebuild(poisoned, far)
    assert dirty_far.attrs["baseline_source"] == "recording"
    for col in SCHEMA:
        a, b = clean_far[col].to_numpy(), dirty_far[col].to_numpy()
        assert a.tobytes() == b.tobytes() if a.dtype.kind not in "OU" \
            else list(a) == list(b), col

    dirty_near = rebuild(poisoned, near)
    assert dirty_near.attrs["baseline_source"] == "non_finite"
    assert not bool(dirty_near["accepted"].iloc[0])


def test_repair_nonfinite_only_moves_samples_near_the_defect():
    x = trace().astype(np.float64)
    y = x.copy()
    y[123_456] = np.nan
    fixed, n_bad = B.repair_nonfinite(y)
    assert n_bad == 1
    assert np.isfinite(fixed).all()
    moved = np.flatnonzero(fixed != x)
    assert moved.tolist() == [123_456]
    # an all-non-finite trace is handed back untouched so the caller still refuses
    allbad = np.full(100, np.nan)
    back, n = B.repair_nonfinite(allbad)
    assert n == 100 and back is allbad


def test_a_rebuild_never_raises_because_the_estimator_could_not_run():
    """A rebuild is a UI action on ONE event. A trace too short for the configured
    estimator must cost that row its exactness, never the edit."""
    x = trace(n=3000, gap=900, dwell=100)
    prov = B.DEFAULT_BASELINE.to_provenance(FS) | {"polarity": -1, "fs": FS}
    out = recharacterize_span(x, FS, 1200, 1300, provenance=prov, prior=None)
    assert out.attrs["baseline_source"] in REBUILD_BASELINE_SOURCES
    assert len(out) == 1


def test_rebuild_reproduces_the_stored_estimator_not_the_live_default():
    """A merge must not silently apply an estimator change the user has not
    confirmed. The config comes from the ROW's provenance; staleness is the
    fingerprint's job, not the merge's."""
    stored = B.BaselineConfig(rough_s=1.5, base_s=0.2, decim=80)
    x = trace()
    df, prov = detect_events(x, FS, polarity=-1, baseline_config=stored)
    row = df.iloc[5]
    out = recharacterize_span(x, FS, int(row.start_idx), int(row.end_idx),
                              provenance=prov, prior=row)
    assert float(out["baseline_pA"].iloc[0]) == float(row.baseline_pA)
    # the live default would have produced something else
    live = B.characterisation_baseline(np.asarray(x, np.float64), FS, polarity=-1)
    assert float(np.median(live.baseline[int(row.start_idx):int(row.end_idx)])) != \
        float(row.baseline_pA)


# ===========================================================================
# 3. PROVENANCE
# ===========================================================================


def test_provenance_carries_every_parameter_and_a_fingerprint():
    x = trace()
    _, prov = detect_events(x, FS, polarity=-1)
    for key in B.BASELINE_PARAM_KEYS:
        assert key in prov, key
    for key in ("baseline_param_source", "baseline_median_pA", "baseline_polarity",
                "baseline_mask_fraction", "baseline_health"):
        assert key in prov, key
    assert prov["baseline_fingerprint"].startswith("sha256:")
    json.dumps(prov, default=str)            # the whole block round-trips as JSON


def test_the_fingerprint_moves_with_every_hashed_parameter():
    """A digest that ignores a parameter is a staleness stamp that lies."""
    base = B.BaselineConfig()
    seen = {base.fingerprint(FS)}
    for field, value in (("method", "robust_median"), ("rough_s", 0.7),
                         ("base_s", 0.2), ("decim", 33), ("mask_sigma", 4.0),
                         ("dilate_ms", 2.0), ("window_s", 0.25), ("despike_size", 3)):
        fp = base.replace(**{field: value}).fingerprint(FS)
        assert fp not in seen, f"{field} does not move the fingerprint"
        seen.add(fp)


def test_preprocess_params_round_trips_every_baseline_key_unchanged():
    """I4. Param name IS provenance name, so `preprocess_params_from_provenance`
    needs no translation entry -- assert that rather than trusting it."""
    x = trace()
    _, prov = detect_events(x, FS, polarity=-1)
    params = preprocess_params_from_provenance(prov)
    for key in B.BASELINE_PARAM_KEYS:
        assert params[key] == prov[key], key
    cfg = B.BaselineConfig.from_provenance(prov)
    assert cfg.fingerprint(FS) == prov["baseline_fingerprint"]


def test_re_running_a_stored_detection_reproduces_its_own_estimator():
    """"Same detection, new calibration" must re-measure with the estimator that
    produced the rows. Reconstructing the settings and dropping the baseline block
    would silently move depth_pA by up to 15x -- the exact change the fingerprint
    exists to make visible, applied invisibly."""
    from nanoscope.analysis.modules import detection_params_from_provenance

    stored = B.BaselineConfig(rough_s=2.0, base_s=0.5, decim=64, mask_sigma=4.0)
    x = trace()
    _, prov = detect_events(x, FS, polarity=-1, baseline_config=stored,
                            baseline_param_source="user")
    params = detection_params_from_provenance(prov)
    cfg, source = B.baseline_config_from_params(params)
    assert cfg == stored.resolve(FS)
    assert cfg.fingerprint(FS) == prov["baseline_fingerprint"]
    assert source == "restored"


def test_analysis_result_carries_the_fingerprint_beside_the_calibration_one():
    from nanoscope.core.analysis import AnalysisResult

    fields = AnalysisResult.__dataclass_fields__
    assert "baseline_fingerprint" in fields
    assert fields["baseline_fingerprint"].default is None
    assert AnalysisResult(
        module_id="m", module_version="1", doc_id="d", channel=0, params={},
        calibration_fingerprint=(1.0, 0.0),
    ).baseline_fingerprint is None


def test_the_baseline_params_are_consumed_not_forwarded_as_kwargs():
    """`detect_events` has no ``baseline_rough_s`` parameter and never will: the
    config OBJECT is the point. A loose per-key kwarg is how the four sites drifted
    apart in the first place."""
    for key in B.BASELINE_PARAM_KEYS:
        assert key in _CONSUMED_PARAM_KEYS, key


# ===========================================================================
# 4. THE EXPORT -- without touching the frozen schema
# ===========================================================================


def test_export_writes_the_sidecar_and_adds_no_column_of_its_own(tmp_path):
    """I5. The estimator's identity reaches a CSV consumer through a sibling file,
    never through a column of this function's invention.

    Stated as "adds none", not "writes exactly the 20", because the second is not
    true of the app and this test used to claim it was: it built its frame with a
    bare ``detect_events``, which never passes through the scoring/curation path, so
    it never saw the ``added`` marker column that ``AnalysisPanel._show_result``
    attaches the moment a detection is displayed. The real ``Analysis > Export
    events...`` therefore writes 21 columns while this passed. That extra column is
    PRE-EXISTING and deliberate (``scoring.ADDED_COLUMN``; the pre-change tree's bare
    ``to_csv`` wrote it too) and the sidecar's ``columns`` field lists it honestly --
    but a guarantee that only holds for frames the app does not produce is worse than
    none, so the assertion is now made on a frame that HAS the extra column.
    """
    from nanoscope.analysis.scoring import ADDED_COLUMN, ensure_added_column

    x = trace()
    df, prov = detect_events(x, FS, polarity=-1)
    out = write_events_export(df, tmp_path / "ev.csv", provenance=prov,
                              source_file="rec.abf", detector="detect.schmitt")
    assert out["data"].exists() and out["meta"].exists()
    assert events_export_paths(tmp_path / "ev.csv") == (out["data"], out["meta"])

    written = pd.read_csv(out["data"])
    assert list(written.columns) == list(SCHEMA)

    # ...and on the frame the APP actually holds: the schema, in order, plus the
    # curation marker it arrived with, and nothing this function invented.
    shown = ensure_added_column(df.copy())
    assert ADDED_COLUMN in shown.columns
    out2 = write_events_export(shown, tmp_path / "ev2.csv", provenance=prov,
                               source_file="rec.abf", detector="detect.schmitt")
    written2 = pd.read_csv(out2["data"])
    assert list(written2.columns) == list(SCHEMA) + [ADDED_COLUMN]
    meta2 = json.loads(out2["meta"].read_text())
    assert meta2["columns"] == list(written2.columns), (
        "the sidecar must list the columns actually written, or a consumer reading "
        "positionally is misled by the very file that exists to describe the table"
    )

    meta = json.loads(out["meta"].read_text())
    assert meta["provenance"]["baseline_fingerprint"] == prov["baseline_fingerprint"]
    for key in B.BASELINE_PARAM_KEYS:
        assert key in meta["provenance"]


def test_the_training_sidecar_also_names_the_estimator():
    from nanoscope.analysis.scoring import training_provenance

    x = trace()
    _, prov = detect_events(x, FS, polarity=-1)
    meta = training_provenance(prov, source_file="r.abf", detector="detect.schmitt")
    contract = meta["detrend_contract"]
    for key in BASELINE_CONTRACT_KEYS:
        if key in prov:
            assert key in contract, key
    assert contract["baseline_fingerprint"] == prov["baseline_fingerprint"]


# ===========================================================================
# 5. THE LENGTH ADAPTATION -- fit the windows to the trace, name it, never search
# ===========================================================================
# This section used to assert the opposite: that a short trace fell back to the
# pre-change 10 ms block median, on the reasoning that shrinking a window to fit is
# the degenerate auto-tune (R5) the design refuses. The reasoning conflated two
# different things and the measurement settled it. Over 27 real events on
# ``..._0006_Process_-600 mV.abf``, cut into segments centred on them, depth as a
# fraction of the depth the whole-recording masked median gives the SAME event:
#
#     segment length      1 000    1 536    3 072    5 000   10 000   50 000
#     block median 10 ms  0.020    0.187    0.011    0.014    0.014    0.014
#     clamped masked      0.013    0.014    0.958    0.976    0.990    0.998
#
# The fallback that existed to avoid a degenerate answer WAS the degenerate answer:
# 1.4 % of the depth, the baseline sitting inside the event, at every length. And
# clamping is not a search -- there is one width the data can host, no candidate set,
# no metric consulted, and the ratio between the two passes (which is what decides
# the two-pass structure) is the user's own, preserved exactly.


def test_a_short_trace_runs_the_configured_estimator_at_clamped_windows():
    """The configured estimator RUNS, at the only widths this many samples can host,
    with its two-pass ratio intact -- and says so out loud under its own token."""
    x = trace(n=3000, gap=900, dwell=100)
    res = B.characterisation_baseline(x, FS, polarity=-1)
    assert res.config.method == "masked_median", (
        "the pre-change block median must not stand in: on the owner's own recording "
        "it reports 1.4 % of the depth the whole-recording estimator reports"
    )
    assert res.config.rough_s == pytest.approx(3000 / FS)
    assert res.config.base_s / res.config.rough_s == pytest.approx(
        B.DEFAULT_BASELINE.base_s / B.DEFAULT_BASELINE.rough_s
    ), "the ratio between the passes is the user's, not the clamp's"
    # The source stays the CALLER's (nobody chose these windows: 'default'), and the
    # clamp is recorded beside it. It used to become 'fallback' here, which turned
    # the never-measured advisory off on every short segment (S7-DET3).
    assert res.param_source == "default"
    assert res.fallback_token == B.FALLBACK_CLAMPED
    assert res.fingerprint != B.DEFAULT_BASELINE.fingerprint(FS)
    assert res.warnings and "CLAMPED" in res.warnings[0]
    prov = res.provenance()
    assert prov["baseline_fallback"] == B.FALLBACK_CLAMPED
    # and the estimator's own strict contract is untouched -- it still REFUSES the
    # config it was handed rather than re-tuning behind its caller
    with pytest.raises(ValueError, match="tracking"):
        B.estimate_baseline_result(x, FS, polarity=-1)


def test_the_clamp_cannot_reach_a_recording_long_enough_for_its_own_windows():
    """The invariant that keeps every shipped number where it is: one sample either
    side of the boundary, and the config is either untouched or clamped -- never
    partly re-tuned."""
    need = int(np.ceil(B.DEFAULT_BASELINE.tracking_window_s * FS))
    for n in (need, need + 1, 4 * need):
        cfg, token = B.adapt_to_length(B.DEFAULT_BASELINE, n, FS)
        assert token is None and cfg is B.DEFAULT_BASELINE, n
    cfg, token = B.adapt_to_length(B.DEFAULT_BASELINE, need - 1, FS)
    assert token == B.FALLBACK_CLAMPED
    assert cfg.rough_s == pytest.approx((need - 1) / FS)


def test_a_trace_too_short_for_any_tracking_window_is_a_named_global_median():
    """Below the point where even the clamped short pass has `MIN_TAPS` taps there is
    nothing to track, and the honest answer is the global median under its own name --
    not a two-pass estimator's name over a one-sample baseline."""
    x = trace(n=10, gap=900, dwell=2)
    res = B.characterisation_baseline(x, FS, polarity=-1)
    assert res.config.method == "constant"
    assert res.provenance()["baseline_fallback"] == B.FALLBACK_TOO_SHORT
    assert res.warnings and "GLOBAL MEDIAN" in res.warnings[0]
    assert np.allclose(res.baseline, np.median(np.asarray(x, np.float64)))


def test_the_two_rungs_are_told_apart_in_provenance():
    """``baseline_fallback`` used to be stamped as the same constant whatever
    happened, so no consumer could distinguish a clamped MEASUREMENT from an
    untracked global median."""
    short = B.characterisation_baseline(trace(n=3000, gap=900, dwell=100), FS,
                                        polarity=-1)
    tiny = B.characterisation_baseline(trace(n=10, gap=900, dwell=2), FS, polarity=-1)
    assert short.provenance()["baseline_fallback"] != \
        tiny.provenance()["baseline_fallback"]


def test_the_app_never_refuses_a_legal_segment_for_its_length():
    """``core.segment.MIN_SEGMENT_SAMPLES`` is 2, so a two-sample cut is a document
    the app lets the user create, select and name. Every one of them used to raise
    below 500 samples -- with a message naming a 10 ms ``window_s`` the user had
    never set, because it belonged to the internal fallback."""
    from nanoscope.core.segment import MIN_SEGMENT_SAMPLES

    for n in (MIN_SEGMENT_SAMPLES, 3, 10, 16, 50, 200, 499, 500, 3072):
        x = trace(n=n, gap=max(2, n // 3), dwell=max(1, n // 10))
        res = B.characterisation_baseline(x, FS, polarity=-1)
        assert res.baseline.shape == (n,), n
        assert np.isfinite(res.baseline).all(), n


def test_constant_has_no_window_to_be_too_short_for():
    """``'constant'`` is a global median. It reads no window, and ``window_s`` rides on
    the config only so five methods share one object -- so refusing a 200-sample trace
    because a window it never touches does not fit was a plain defect, and it made the
    two entry points disagree about the same method."""
    x = trace(n=200, gap=60, dwell=10)
    cfg = B.BaselineConfig(method="constant")
    res = B.estimate_baseline_result(x, FS, config=cfg, polarity=-1, health=False)
    assert np.allclose(res.baseline, np.median(np.asarray(x, np.float64)))
    assert np.allclose(res.baseline,
                       B.estimate_baseline(x, FS, method="constant", polarity=-1))


def test_the_clamped_run_reaches_the_detection_as_a_warning():
    from nanoscope.analysis.modules import _build_warnings

    x = trace(n=3000, gap=900, dwell=100)
    df, prov = detect_events(x, FS, polarity=-1)
    assert prov["baseline_method"] == "masked_median"
    # 'default', not 'fallback': the clamp is a fact about the run, recorded in
    # ``baseline_fallback``; the source is a fact about who chose the windows, and
    # overwriting it silenced the never-measured advisory here (S7-DET3).
    assert prov["baseline_param_source"] == "default"
    assert prov["baseline_fallback"] == B.FALLBACK_CLAMPED
    warns = _build_warnings(prov, df)
    assert any("CLAMPED" in w for w in warns)
    assert B.UNPROBED_DEFAULT_WARNING in warns


# ===========================================================================
# 6. POLARITY -- the door measures it in the frame the algorithm assumes
# ===========================================================================


def test_the_door_derives_polarity_on_a_detrended_trace():
    """`derive_polarity` measures from a GLOBAL-MEDIAN centre, so on a drifting raw
    trace it counts the DRIFT, not the events -- +1 on the owner's own ferritin_data2
    and data3. The one-sided mask pointed the wrong way moves the baseline by up to
    312 pA. `derive_polarity` is frozen; the door hands it a rough-detrended trace,
    which is the frame the reference algorithm assumes."""
    x = trace(drift=400.0, depth=120.0)
    assert B.derive_polarity(np.asarray(x, np.float64)) == +1      # the drift
    assert B._polarity_for(x, FS, B.DEFAULT_BASELINE) == -1        # the events
    assert B.characterisation_baseline(x, FS).polarity == -1


def test_an_explicit_polarity_always_wins():
    x = trace(drift=400.0, depth=120.0)
    assert B.characterisation_baseline(x, FS, polarity=+1).polarity == +1


# ---------------------------------------------------------------------------
# 6b. ...and the frame is a property of the RECORDING, not of the parameters
# ---------------------------------------------------------------------------
# THE DEFECT. The door used to detrend at ONE window -- the configured estimator's
# own ``rough_s``, at a stride derived from its ``base_s`` -- so the mask direction
# was decided by parameters nobody had confirmed. Measured on ``ferritin_data3``:
# at a 0.5 s rough window, stride 264 gives -1 and stride 333 gives +1, and the
# stride is derived from a PASS-2 parameter times a cost constant. And at the
# baseline probe's own frame on that recording (5 s) NOTHING crosses 5 sigma, so
# the +1 handed back is `derive_polarity`'s documented flat-trace default, reported
# as a measurement. The cost of getting it wrong is not subtle: the one-sided mask
# pointed backwards moves the baseline by 312 pA on data2 and 282 pA on data3, and
# a probe handed +1 reads the apo file's decisive 496 ms clog as 27.8 ms and
# proposes ``rough_s = 0.1`` against a known-good 2.0 -- with ``confident=True``.
#
# `derive_polarity` is FROZEN and is not touched by any of this.

FAST_GATE_STEP_PA = 290.0
FAST_GATE_LEVEL_S = 0.02


def gating_trace(*, n=1_200_000, level_s=FAST_GATE_LEVEL_S, step=FAST_GATE_STEP_PA,
                 depth=120.0, dwell=100, gap=3000, seed=5):
    """A pore whose OPEN level steps between two conductances, with real blockades.

    The blockades run DOWN (polarity -1) and they are 120 pA against a 290 pA step,
    so anything that counts the STEPS instead of the events answers +1. This is the
    shape ``ferritin_data3`` has and the shape the single-window derivation got wrong.
    """
    rng = np.random.default_rng(seed)
    x = np.full(n, -400.0) + rng.normal(0.0, 14.0, n)
    k = int(level_s * FS)
    hi = False
    for s in range(0, n, k):
        if hi:
            x[s:s + k] += step
        hi = not hi
    s = 2000
    while s < n - dwell - 2000:
        x[s:s + dwell] -= depth
        s += int(rng.exponential(gap)) + dwell
    return x.astype(np.float64)


#: Every baseline configuration this app can be in: the shipped default, each of the
#: four known-good corpus sets, the probe's own wide frame, and the legacy fallback.
_EVERY_SHIPPED_CONFIG = (
    B.DEFAULT_BASELINE,
    B.BaselineConfig(method="robust_median", window_s=0.5, decim=50),
    B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50),
    B.BaselineConfig(rough_s=0.03, base_s=0.01, decim=10),
    B.BaselineConfig(rough_s=2.0, base_s=0.5),
    B.BaselineConfig(rough_s=5.0, base_s=1.25),
    B.BaselineConfig(method="block_median", window_s=0.01),
)


def test_the_mask_direction_does_not_depend_on_the_configuration():
    """THE invariant. Same recording, every configuration the app can be in, one answer.

    Before the ladder this fixture answered ``+1`` at the shipped default and at the
    probe's own 5 s frame, and ``-1`` at the 0.03 s and legacy configurations -- the
    direction of the one-sided mask decided by a window nobody had confirmed.
    """
    x = gating_trace()
    # A FRESH COPY PER CONFIGURATION, or this test would be asserting the memo rather
    # than the ladder: `_PolarityMemo` is keyed on object identity, so the second and
    # later questions about one array are answered from the first. A copy holds the
    # same samples and misses, so each configuration really re-measures.
    answers = {B._polarity_for(x.copy(), FS, cfg) for cfg in _EVERY_SHIPPED_CONFIG}
    assert answers == {-1}, f"the direction moved with the configuration: {answers}"
    assert {B.measured_polarity(x.copy(), FS, config=c)
            for c in _EVERY_SHIPPED_CONFIG} == {-1}
    assert B.measured_polarity(x.copy(), FS) == -1

    # ...and the memo really is transparent: same array, every configuration, and the
    # answer the ladder gives on a copy that has never been asked about.
    assert {B._polarity_for(x, FS, cfg) for cfg in _EVERY_SHIPPED_CONFIG} == {-1}


def test_a_rung_that_measured_nothing_does_not_vote_a_default():
    """`derive_polarity` documents a ``+1`` fallback for a trace with nothing to
    measure. At a wide detrend window on a gating pore that is exactly what happens --
    the residual keeps the whole conductance swing, its MAD is enormous, and no run
    reaches 5 sigma -- and the single-window derivation reported that ``+1`` as though
    it had been measured. The ladder asks the frozen function itself, by negating the
    residual: a measured answer must flip, both no-evidence exits return ``+1`` either
    way."""
    flat = np.zeros(100_000)
    assert B.derive_polarity(flat) == +1                 # the documented default
    assert B._polarity_measured(flat) == (+1, False)     # ...and it is named as one

    rng = np.random.default_rng(3)
    quiet = rng.normal(0.0, 1.0, 200_000)                # noise, no coherent excursion
    assert B._polarity_measured(quiet)[1] is False

    real = quiet.copy()
    for s in range(1000, 199_000, 5000):
        real[s:s + 60] -= 30.0
    vote, measured = B._polarity_measured(real)
    assert (vote, measured) == (-1, True)

    x = gating_trace()
    rungs = B._polarity_ladder(x, FS)
    assert [w for w, _, _ in rungs] == list(B._POLARITY_LADDER_S)
    assert any(not m for _, _, m in rungs), "the wide rungs must report no evidence"
    assert not all(v == -1 for _, v, m in rungs if m), (
        "the fixture must reproduce the trap: some measured rung has to disagree"
    )


def test_the_narrowest_rung_is_the_strongest_voice_but_cannot_dictate():
    """Harmonic weights, narrowest first. The narrowest rung is the frame in which the
    least of the recording's own wander can have survived, so it carries the most
    weight -- but 1 < 1/2 + 1/3 + 1/4 + 1/5, so the rest of the ladder can still
    overturn it and no single window decides."""
    w = [1.0 / (i + 1) for i in range(len(B._POLARITY_LADDER_S))]
    assert w[0] < sum(w[1:]), "the narrowest rung must be overrulable"
    assert w[0] > w[1], "and it must be the strongest single voice"
    assert B._POLARITY_LADDER_S == tuple(sorted(B._POLARITY_LADDER_S))


def test_a_recording_too_short_for_any_rung_still_answers():
    """The ladder needs `_POLARITY_LADDER_WINDOWS` windows of its narrowest rung. Below
    that there is no ladder, and the door falls back to the frozen function on the
    trace as given rather than raising -- which is what it returned before there was
    one."""
    x = trace(n=400)                       # 8 ms at 50 kHz
    assert B._polarity_rung_windows(x.size, FS) == ()
    assert B._polarity_for(x, FS) in (-1, +1)


def test_the_ladder_costs_what_it_says_it_costs():
    """The vote stops as soon as the rungs still to come cannot overturn it, so an
    agreeing recording pays two rungs, not five."""
    x = trace(drift=400.0, depth=120.0, dwell=100)
    assert [v for _, v, _ in B._polarity_ladder(x, FS)] == [-1] * 5, "fixture drifted"
    seen: list[float] = []
    real = B._polarity_rungs

    def spy(sig, fs):
        for rung in real(sig, fs):
            seen.append(rung[0])
            yield rung

    B._polarity_rungs = spy          # type: ignore[assignment]
    try:
        assert B._polarity_for(x, FS) == -1
    finally:
        B._polarity_rungs = real     # type: ignore[assignment]
    assert len(seen) == 2, f"a unanimous ladder must stop at rung 2, ran {seen}"


def _rung_spy(monkeypatch) -> list[float]:
    """Record every ladder rung actually computed, without changing any answer."""
    seen: list[float] = []
    real = B._polarity_rungs

    def spy(sig, fs):
        for rung in real(sig, fs):
            seen.append(rung[0])
            yield rung

    monkeypatch.setattr(B, "_polarity_rungs", spy)
    return seen


def test_the_direction_is_measured_once_per_trace_not_once_per_call(monkeypatch):
    """The ladder is 2 to 5 detrends of the WHOLE recording, and
    `characterisation_baseline` needs the direction before it can build its cache key
    -- so without a memo the ladder re-ran on every call, cache hit or miss. Measured
    on the owner's -600 mV apo recording, five detections over one array with the
    baseline already memoised: 1.508 s before the ladder, 2.283 s with it and no memo
    (+155 ms each), 0.929 s with it. `sweep_threshold` is 32 such detections.

    THE TRAP THIS PINS is the key. ``characterisation_baseline`` converts to float64
    on every call and the app's conditioned trace is float32, so a memo keyed on the
    array the ladder actually runs on would miss every single time and save nothing.
    It is keyed on the caller's object instead -- the same object `_BaselineCache` is
    keyed on -- which is why ``trace()`` here returns float32, as the reader does.
    """
    x = trace(drift=400.0, depth=120.0, dwell=100)
    assert x.dtype == np.float32, "the fixture must reproduce the reader's dtype"

    seen = _rung_spy(monkeypatch)
    first = B.characterisation_baseline(x, FS, health=False)
    ran_once = len(seen)
    assert ran_once >= 2, "the first call has to actually measure something"

    for _ in range(4):
        again = B.characterisation_baseline(x, FS, health=False)
        assert np.array_equal(again.baseline, first.baseline)
    assert len(seen) == ran_once, f"the ladder re-ran on a repeat call: {seen}"

    # A DIFFERENT array with identical contents misses and re-measures -- correct,
    # just slower, exactly as `_BaselineCache` documents.
    B.characterisation_baseline(x.copy(), FS, health=False)
    assert len(seen) == 2 * ran_once, "a distinct trace must be measured on its own"


def test_the_memo_changes_no_answer_and_the_one_reset_drops_both(monkeypatch):
    """It is a memo, not a decision: same direction as the unmemoised vote, on the
    ladder's own hard case, and `clear_baseline_cache` -- the one reset the door
    offers -- has to drop it or a test that clears the cache is still answered from it.
    """
    x = gating_trace()
    direct = B._polarity_vote(np.asarray(x, dtype=np.float64), FS)
    assert B._polarity_for(x, FS) == direct
    assert B.measured_polarity(x, FS) == direct

    seen = _rung_spy(monkeypatch)
    B.measured_polarity(x, FS)
    assert seen == [], "already measured; this must not re-run the ladder"
    B.clear_baseline_cache()
    assert B.measured_polarity(x, FS) == direct
    assert seen, "clear_baseline_cache must drop the direction too"


def test_a_memoised_direction_never_answers_a_trace_that_should_be_refused():
    """The memo sits AFTER ``require_finite``. An ``interpolate`` call leaves an entry
    keyed on an array that still holds the NaN; the ``refuse`` path must still refuse
    rather than be answered out of it."""
    x = trace().astype(np.float64)
    x[12_345] = np.nan
    ok = B.characterisation_baseline(x, FS, health=False, on_nonfinite="interpolate")
    assert np.all(np.isfinite(ok.baseline))

    # The hazard is real, not hypothetical: that call really did leave an entry keyed
    # on THIS array -- the one still holding the NaN. Move the lookup above
    # ``require_finite`` and the refusal below turns into an answer.
    key = B._PolarityMemo._key(x, FS)
    assert B._POLARITY_MEMO.get(key, x) in (-1, +1), "the interpolate call must memoise"

    with pytest.raises(ValueError):
        B.characterisation_baseline(x, FS, health=False)      # on_nonfinite='refuse'


# ===========================================================================
# 7. BIT-IDENTITY -- a stored result re-measured under its OWN fingerprint
# ===========================================================================
# Every digest below was PROVED equal to the pre-batch PyInstaller bundle
# (`dist/NanoScope/NanoScope`, Jul 28 21:13) before it was written down. That
# binary embeds this tree's 71 modules as marshalled bytecode and is the only
# witness to what the code used to do -- `git ls-files` returns nothing here. It
# is provably pre-change: its `METHODS` has four entries, its `estimate_baseline`
# defaults to `window_s=0.01`, and its `ml_modules` has no
# `_CHAR_BASELINE_WINDOW_S` at all.
#
# Three claims were checked against it, all PASS:
#   A  `detect_events` with an EXPLICIT baseline array -- all 20 columns
#      byte-identical, 240 rows. Isolates the PIPELINE from the estimator.
#   B  `detect_events` under `BaselineConfig("block_median", 0.01)` against the
#      oracle's own default -- all 20 columns byte-identical, 197 rows. THIS is
#      the migration claim: a stored result re-measured under the fingerprint it
#      was stored with does not move.
#   C  the old ML characterisation baseline (`block_median` @ 0.20 s) --
#      byte-identical over 400 000 samples.
# Harness: scratchpad/BUILD/seam/{extract_oracle,verify_oracle_detect}.py
#
# NEVER RE-BASELINE THESE. A failure here means a stored project's numbers moved.

_ORACLE_TRACE_SHA = "29027606946beba95a7f02c48f7b0baeff37c4283046b224a89492c9507093f4"
_ORACLE_COLUMNS = {
    "event_id": "c2840e3020bfdac5", "recording": "f6fed448e4ed5eeb",
    "pore_id": "f6fed448e4ed5eeb", "start_idx": "a57c71ee2065dcce",
    "end_idx": "464b693e93d2f138", "t_start_s": "5c10a029538f97c7",
    "dwell_s": "f494a9ab8819d870", "baseline_pA": "fc9e67a57a7778e0",
    "depth_pA": "fc7891641a69a77c", "frac_depth": "e5d577550e9f9f1d",
    "ecd_pA_s": "536672c577f61b81", "max_dev_pA": "060a2b08e254890b",
    "mean_dev_pA": "2a4cea34428c82c7", "rise_s": "a2543e790132a457",
    "fall_s": "be333d937bee7f9d", "snr": "cc0fbae0c93163c3",
    "n_levels": "cd92abb8440dc9e5", "filter_limited": "8ce53fa730feb908",
    "accepted": "5784ac90b7462326", "flag": "f6fed448e4ed5eeb",
}


def _oracle_trace():
    import hashlib

    rng = np.random.default_rng(25)
    t = np.arange(400_000) / FS
    x = -1000.0 + 180.0 * np.sin(2 * np.pi * 0.4 * t) + rng.normal(0, 12.0, 400_000)
    for k in range(240):
        s = 900 + k * 1600
        if s + 320 < 400_000:
            x[s:s + 320] -= 300.0
    x = x.astype(np.float32)
    assert hashlib.sha256(x.tobytes()).hexdigest() == _ORACLE_TRACE_SHA
    return x


def test_a_stored_result_re_measured_under_its_own_fingerprint_is_unchanged():
    """All 20 columns, against the pre-batch bundle. The migration promise in one
    test: opening an old project and re-measuring it under the estimator it RECORDS
    reproduces the numbers it already has, to the byte."""
    import hashlib

    x = _oracle_trace()
    cfg = B.BaselineConfig.from_provenance({"baseline_method": "block_median",
                                            "baseline_window_s": 0.01})
    df, prov = detect_events(x, FS, polarity=-1, min_dwell_s=2e-5,
                             baseline_config=cfg)
    assert len(df) == 197
    assert prov["baseline_fingerprint"] == "sha256:236cdc0579e4"
    for col, want in _ORACLE_COLUMNS.items():
        v = df[col].to_numpy()
        raw = (v.tobytes() if v.dtype.kind not in "OU"
               else "\x1f".join(map(str, v)).encode())
        assert hashlib.sha256(raw).hexdigest()[:16] == want, col


def test_a_provided_baseline_array_still_bypasses_the_estimator_entirely():
    """Claim A: with the baseline handed in, nothing this change touched runs."""
    x = _oracle_trace()
    flat = np.full(x.size, -1000.0)
    df, prov = detect_events(x, FS, polarity=-1, baseline=flat, min_dwell_s=2e-5)
    assert prov["baseline_method"] == "provided"
    assert prov["baseline_provided"] is True
    assert "baseline_fingerprint" not in prov
    assert float(prov["baseline_median_pA"]) == -1000.0


def test_the_old_ml_characterisation_window_still_reproduces_its_own_numbers():
    """Claim C: `block_median` at 0.20 s -- what `_CHAR_BASELINE_WINDOW_S` selected --
    is byte-identical through the new door."""
    import hashlib

    x = _oracle_trace()
    cfg = B.BaselineConfig(method="block_median", window_s=0.20)
    got = B.characterisation_baseline(
        np.asarray(x, np.float64), FS, config=cfg, polarity=None, health=False)
    assert hashlib.sha256(got.baseline.tobytes()).hexdigest()[:16] == \
        "a05df4fa8c2bf355"


# ===========================================================================
# THE STRUCTURAL GUARDS -- a fifth call site cannot express its own default
# ===========================================================================
_SRC = Path(__file__).resolve().parents[2] / "nanoscope"

#: The one module allowed to reach the raw estimator, own a default, or spell a
#: baseline window. Everything else goes through `characterisation_baseline`.
_BASELINE_OWNER = "nanoscope/analysis/baseline.py"


def _called_name(node: ast.Call) -> str | None:
    """Both spellings: a bare name and a module attribute. A guard on a call SHAPE
    that only sees ``ast.Name`` is a guard an ``import ... as`` walks straight
    past -- ``tests/analysis/test_flag_invariant.py`` learned that the hard way."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _modules():
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC.parent))
        if rel == _BASELINE_OWNER:
            continue
        yield rel, ast.parse(path.read_text())


def test_no_module_outside_baseline_calls_the_raw_estimator():
    """`estimate_baseline` returns a bare array with a silent ``window_s`` default and
    no fingerprint. Every one of the four historical settings was expressed through
    it. The door -- `characterisation_baseline` -- returns a `BaselineResult` that
    knows its own parameters and stamps them, so a call site cannot measure without
    saying what it measured with.

    `modules.py`'s DETRENDING stage is a different measurement (wander removal, which
    keeps events) and goes through `remove_baseline_wander`; it does not appear here.
    """
    offenders = []
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) == "estimate_baseline":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"estimate_baseline() is called from {offenders}; call "
        "baseline.characterisation_baseline so the parameters are one object and "
        "reach provenance"
    )


def test_no_module_outside_baseline_defines_a_baseline_window_constant():
    """I3. ``_CHAR_BASELINE_WINDOW_S = 0.20`` is the shape of this defect: a module
    fixing its own baseline locally, correctly, for its own recording. The next one
    will look just as reasonable."""
    offenders = []
    for rel, tree in _modules():
        # MODULE LEVEL ONLY, deliberately. ``_CHAR_BASELINE_WINDOW_S`` was a module
        # constant, and that is the shape being banned. `MLPreprocessConfig`'s
        # ``baseline_window_s`` / ``baseline_decim`` FIELDS in `ml_preprocess` are a
        # different measurement entirely -- the ML model's own training-time
        # preprocessing chain, bit-exact against the owner's notebooks and frozen --
        # and a guard that walked class bodies would ban them for their spelling.
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                name = t.id.lower()
                if "baseline" in name and ("window" in name or "rough" in name
                                           or "base_s" in name or "decim" in name):
                    offenders.append(f"{rel}:{node.lineno}:{t.id}")
    assert not offenders, (
        f"a baseline window constant is defined in {offenders}; the only owner of a "
        "baseline default is baseline.DEFAULT_BASELINE"
    )


def test_no_call_site_hands_the_door_a_literal_window():
    """``characterisation_baseline(x, fs, config=BaselineConfig(base_s=0.2))`` at a
    call site is the same defect with an extra keyword. Configs are built by the two
    resolvers (`baseline_config_from_params` forward,
    `BaselineConfig.reproducible_from` backward), never inline."""
    offenders = []
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) == "BaselineConfig":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"a BaselineConfig is constructed at {offenders}; resolve it with "
        "baseline_config_from_params / BaselineConfig.reproducible_from"
    )


def test_the_default_is_written_exactly_once():
    """`DEFAULT_BASELINE = BaselineConfig()` -- one literal, in one module. Grepping
    for a second one is the cheapest possible defence against R4."""
    src = (_SRC / "analysis" / "baseline.py").read_text()
    assert src.count("DEFAULT_BASELINE = ") == 1
    others = [rel for rel, tree in _modules()
              for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "DEFAULT_BASELINE"
                      for t in node.targets)]
    assert not others, others
