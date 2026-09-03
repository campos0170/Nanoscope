"""S7-DET1 / S7-DET3 -- ``baseline_param_source`` says who chose the windows, for THIS
recording, and nothing else overwrites it.

* S7-DET1: Batch detect sent the selected recording's ``'proposed'`` ("measured on
  this recording, and confirmed") to every recording in the batch, which also
  silenced the never-measured advisory on all of them. Now a params map may carry
  `BATCH_SCOPE_KEY` -- the recording the form was filled in for -- and
  `resolve_param_source` stamps every OTHER recording ``'batch'``, whose advisory
  names the copied windows.
* S7-DET3: the length clamp overwrote the caller's source with ``'fallback'``, which
  silenced the same advisory on every short segment. The clamp now stays in
  ``baseline_fallback`` and the source is left alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis import modules as M

FS = 50_000.0


def _trace(n=3000, gap=900, dwell=100, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, -100.0) + rng.normal(0.0, 2.0, n)
    t = 200
    while t + dwell < n:
        x[t:t + dwell] -= 60.0
        t += gap
    return x


# ===========================================================================
# the resolver
# ===========================================================================
@pytest.mark.parametrize("source", ["proposed", "user", "restored"])
def test_a_confirmed_source_is_kept_for_its_own_recording_and_becomes_batch_elsewhere(source):
    params = {"baseline_param_source": source, B.BATCH_SCOPE_KEY: ("doc-A", 0)}
    assert B.resolve_param_source(params, ("doc-A", 0), source) == source
    assert B.resolve_param_source(params, ("doc-B", 0), source) == B.BATCH_SOURCE
    assert B.resolve_param_source(params, ("doc-A", 1), source) == B.BATCH_SOURCE


def test_the_default_is_the_honest_source_for_every_recording_and_needs_no_scope():
    params = {"baseline_param_source": "default", B.BATCH_SCOPE_KEY: ("doc-A", 0)}
    assert B.resolve_param_source(params, ("doc-B", 0), "default") == "default"
    assert B.resolve_param_source({"baseline_param_source": "proposed"}, ("doc-B", 0),
                                  "proposed") == "proposed", "no scope: the Detect button"
    assert B.resolve_param_source(None, ("doc-B", 0), "proposed") == "proposed"


def test_batch_is_a_legal_source_the_params_resolver_accepts_on_a_re_run():
    assert B.BATCH_SOURCE in B.PARAM_SOURCES
    cfg, src = B.baseline_config_from_params(
        {"baseline_rough_s": 2.0, "baseline_base_s": 0.5, "baseline_param_source": "batch"})
    assert src == "batch" and cfg.rough_s == 2.0


def test_the_run_resolver_records_the_resolved_source_and_drops_the_scope_key():
    params = {"k_in": 4.0, "baseline_rough_s": 2.0, "baseline_base_s": 0.5,
              "baseline_param_source": "proposed", B.BATCH_SCOPE_KEY: ("doc-A", 0)}
    cfg, src, rec = M._resolve_run_baseline(params, ("doc-B", 0))
    assert src == "batch"
    assert rec["baseline_param_source"] == "batch"
    assert B.BATCH_SCOPE_KEY not in rec, "a per-load doc_id must not reach a project file"
    assert rec["k_in"] == 4.0 and cfg.rough_s == 2.0
    _, src_a, rec_a = M._resolve_run_baseline(params, ("doc-A", 0))
    assert src_a == "proposed" and rec_a["baseline_param_source"] == "proposed"


# ===========================================================================
# the advisory
# ===========================================================================
def test_a_batch_run_carries_the_never_measured_advisory_naming_the_copied_windows():
    x = _trace(n=60_000, gap=2000, dwell=100)
    cfg = B.DEFAULT_BASELINE.replace(rough_s=0.2, base_s=0.05)
    res = B.characterisation_baseline(x, FS, config=cfg, polarity=-1,
                                      param_source=B.BATCH_SOURCE)
    assert res.param_source == "batch"
    warns = res.warnings
    hit = [w for w in warns if "COPIED FROM THE BATCH FORM" in w]
    assert hit, warns
    assert "0.2 s / 0.05 s" in hit[0]
    assert "never measured" in hit[0]
    assert B.UNPROBED_DEFAULT_WARNING not in warns
    assert res.provenance()["baseline_param_source"] == "batch"


def test_a_detection_stamped_batch_reaches_the_result_warnings():
    x = _trace(n=60_000, gap=2000, dwell=100)
    df, prov = detect_events(x, FS, polarity=-1, baseline_param_source="batch")
    assert prov["baseline_param_source"] == "batch"
    assert any("COPIED FROM THE BATCH FORM" in w for w in M._build_warnings(prov, df))


# ===========================================================================
# the clamp (S7-DET3)
# ===========================================================================
def test_the_length_clamp_records_itself_beside_the_source_instead_of_over_it():
    x = _trace(n=3000, gap=900, dwell=100)
    res = B.characterisation_baseline(x, FS, polarity=-1)          # 'default' asked
    assert res.fallback_token == B.FALLBACK_CLAMPED and res.fallback
    assert res.param_source == "default", (
        "the clamp chose nothing; overwriting the source silenced the never-measured "
        "advisory on the weakest estimate of all")
    assert B.UNPROBED_DEFAULT_WARNING in res.warnings
    assert any("CLAMPED" in w for w in res.warnings)
    prov = res.provenance()
    assert prov["baseline_param_source"] == "default"
    assert prov["baseline_fallback"] == B.FALLBACK_CLAMPED


@pytest.mark.parametrize("source", ["proposed", "user", "batch"])
def test_the_clamp_leaves_every_other_source_alone_too(source):
    x = _trace(n=3000, gap=900, dwell=100)
    res = B.characterisation_baseline(x, FS, polarity=-1, param_source=source)
    assert res.param_source == source
    assert res.fallback_token == B.FALLBACK_CLAMPED


def test_a_clamped_un_probed_segment_says_both_things_on_the_detection():
    x = _trace(n=3000, gap=900, dwell=100)
    df, prov = detect_events(x, FS, polarity=-1)
    warns = M._build_warnings(prov, df)
    assert B.UNPROBED_DEFAULT_WARNING in warns
    assert any("CLAMPED" in w for w in warns)
