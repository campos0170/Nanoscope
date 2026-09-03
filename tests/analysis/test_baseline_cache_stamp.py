"""M-2 — the baseline memo must hold the MEASUREMENT, never the stamp on it.

``_BaselineCache``'s own docstring says it memoises "the MEASUREMENT ... how the parameters
were chosen is not part of it, and must not be inherited from whoever computed it first".
That was enforced for ``param_source`` and for nothing else: ``characterisation_baseline``
applied ``_with_fallback`` BEFORE ``_CACHE.put``, so the stored object carried a previous
call's ``fallback`` / ``fallback_token`` / ``tracking_window_fits`` check, and the key is
built from the FITTED config so an honest later call collides with it by construction.

Two consequences, and the second is the one that reaches a file:

* a repeat fallback call comes back with ``tracking_window_fits`` stamped TWICE — 6 checks
  to 7, 2 warnings to 3, stable thereafter;
* a config that FITS the trace exactly comes back asserting
  ``baseline_fallback='windows_clamped_to_recording'`` and *"Depths are NOT comparable …
  this estimate is inoperative"*, with ``health.ok`` flipping True→False on a bit-identical
  measurement — purely because a default-config call ran on the same array first.

``baseline_fallback`` is in ``scoring.BASELINE_CONTRACT_KEYS``, so it rides into every
export sidecar and every project file. No number moves; the provenance lies.

The fix is at the ``put`` (cache the undecorated result, decorate on the way out through
one function), NOT at the cache-hit branch — patching the hit leaves the false-stamp
direction wide open, which is the call-site-instead-of-invariant shape this project keeps
repeating.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import baseline as B

FS = 50000.001192092925


def _api(name: str):
    """Look the new symbol up LATE and say so when it is absent, so a run against a
    pre-fix package fails on this test's own assertion rather than on an ImportError."""
    obj = getattr(B, name, None)
    assert obj is not None, f"nanoscope.analysis.baseline has no {name!r}"
    return obj


def _trace(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(np.full(n, -34.8) + rng.normal(0, 4.84, n), dtype=np.float32)


def _checks(res) -> list[str]:
    return [c.name for c in res.health.checks]


# ================================================================================================
# (a) the reported double stamp
# ================================================================================================

@pytest.mark.parametrize("n", [2000, 4000, 4999])
def test_a_repeat_call_does_not_stamp_the_clamp_twice(n):
    """Cold and warm must be the same answer. Before: 6 checks/2 warnings cold, 7/3 warm,
    with ``tracking_window_fits`` appearing twice on every subsequent call."""
    x = _trace(n)
    B.clear_baseline_cache()
    cold = B.characterisation_baseline(x, FS, polarity=-1)
    warm = B.characterisation_baseline(x, FS, polarity=-1)
    third = B.characterisation_baseline(x, FS, polarity=-1)

    assert _checks(cold).count("tracking_window_fits") == 1, "the premise: this config clamps"
    assert _checks(warm) == _checks(cold), (
        f"a cache hit added a check: {_checks(warm)} vs {_checks(cold)}")
    assert _checks(third) == _checks(cold)
    assert list(warm.health.warnings) == list(cold.health.warnings)
    assert warm.health.ok is cold.health.ok
    assert warm.fallback == cold.fallback
    assert float(warm.median_pA) == float(cold.median_pA)


def test_the_double_stamp_reached_the_persisted_provenance():
    """The pair of keys the hunter measured differing run-to-run through ``detect_events``
    were exactly ``baseline_health`` / ``baseline_warnings``. Asserted on the provenance
    dict itself, which is what a project file and an export sidecar carry."""
    x = _trace(4000)
    B.clear_baseline_cache()
    cold = B.characterisation_baseline(x, FS, polarity=-1).provenance()
    warm = B.characterisation_baseline(x, FS, polarity=-1).provenance()
    differing = sorted(k for k in set(cold) | set(warm)
                       if repr(cold.get(k)) != repr(warm.get(k)))
    assert differing == [], f"provenance differs by cache history alone: {differing}"


# ================================================================================================
# (b) the false stamp — worse, and unreported before the verify pass
# ================================================================================================

def test_a_config_that_fits_never_inherits_a_fallback_it_did_not_have():
    """THE LOAD-BEARING TEST. The key is built from the FITTED config, so the config a
    previous call was clamped DOWN to is exactly the config a later honest call passes in —
    which is what ``modules.baseline_config_from_params`` hands back from provenance. That
    later call ran at its configured windows and nothing was clamped, and it used to come
    back asserting otherwise, with ``health.ok`` False on a bit-identical measurement."""
    n = 4000
    x = _trace(n)
    fitted, token = B.adapt_to_length(B.DEFAULT_BASELINE, n, FS)
    assert token is not None, "premise: the default config clamps at this length"
    assert B.adapt_to_length(fitted, n, FS)[1] is None, "premise: the fitted config fits"

    B.clear_baseline_cache()
    cold = B.characterisation_baseline(x, FS, config=fitted, polarity=-1, param_source="user")
    assert cold.fallback is None
    assert "tracking_window_fits" not in _checks(cold)
    assert "baseline_fallback" not in cold.provenance()

    B.clear_baseline_cache()
    B.characterisation_baseline(x, FS, polarity=-1)          # the poisoning default call
    warm = B.characterisation_baseline(x, FS, config=fitted, polarity=-1, param_source="user")

    assert warm.fallback is None, (
        f"a call that clamped nothing inherited a fallback: {warm.fallback!r}")
    assert warm.fallback_token == cold.fallback_token
    assert _checks(warm) == _checks(cold)
    assert warm.health.ok is cold.health.ok
    assert "baseline_fallback" not in warm.provenance()
    assert warm.param_source == "user" == cold.param_source
    assert float(warm.median_pA) == float(cold.median_pA)


def test_the_trigger_is_the_tracking_window_not_five_thousand_samples():
    """Scope, corrected by the verifier: ``n < 5000`` is a property of DEFAULT_BASELINE.
    The real condition is ``n < tracking_window_s * fs``, so at a probed ``rough_s = 5 s``
    an ordinary 0.8 s hand-cut segment reproduces it."""
    cfg = B.BaselineConfig(rough_s=5.0, base_s=1.0)
    for n in (40_000, 45_000):
        x = _trace(n, seed=n)
        B.clear_baseline_cache()
        cold = B.characterisation_baseline(x, FS, config=cfg, polarity=-1)
        warm = B.characterisation_baseline(x, FS, config=cfg, polarity=-1)
        assert _checks(cold).count("tracking_window_fits") == 1, n
        assert _checks(warm) == _checks(cold), n
        assert list(warm.health.warnings) == list(cold.health.warnings), n


# ================================================================================================
# the seam itself
# ================================================================================================

def test_the_decoration_is_one_function_both_paths_go_through():
    """The invariant, not the symptom. A hit and a miss must be stamped by the SAME code —
    two call sites deciding it separately is what produced the leak, and a future third
    path (another estimator, another fallback rung) has to be unable to reintroduce it."""
    decorated = _api("_decorated")
    import inspect

    src = inspect.getsource(B.characterisation_baseline)
    assert src.count("_decorated(") == 2, (
        "characterisation_baseline must reach the decoration through exactly two calls "
        "(the cache hit and the fresh measurement), both of _decorated")
    assert "_CACHE.put" in src
    put_at = src.index("_CACHE.put")
    # The put must come BEFORE any decoration of the object it stores.
    assert src.index("return _decorated(res", put_at) > put_at, (
        "the UNDECORATED measurement must be what is cached")
    assert decorated is not None


def test_a_non_clamping_length_is_unaffected_in_both_directions():
    """The control. Nothing about a recording long enough for its tracking window changes,
    warm or cold — this is the owner's own 2,993,798-sample case in miniature."""
    x = _trace(200_000)
    B.clear_baseline_cache()
    cold = B.characterisation_baseline(x, FS, polarity=-1)
    warm = B.characterisation_baseline(x, FS, polarity=-1)
    assert "tracking_window_fits" not in _checks(cold)
    assert _checks(warm) == _checks(cold)
    assert cold.fallback is None and warm.fallback is None
    assert float(warm.median_pA) == float(cold.median_pA)
