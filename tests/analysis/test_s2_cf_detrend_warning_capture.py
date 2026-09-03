"""S2 regression — CF2-3, kernel level: the baseline module's warnings are collectable.

The 1-block degeneracy warning ("the baseline is effectively constant and cannot
track drift") used to exist only as a ``warnings.warn`` — invisible to every user
surface, while the detrend stage silently returned its input unchanged. The fix
gives the module ONE warn door (``_warn_baseline``) that also feeds a THREAD-LOCAL
sink (``collect_baseline_warnings``) — thread-local rather than
``warnings.catch_warnings`` because the stdlib filter list is process-global and the
GUI computes previews on its thread while a detection worker runs this kernel on
another (see ``stats.quiet_runtime_warnings``).

Pinned here:

  * the degenerate case emits ``ConstantBaselineWarning`` — a SUBCLASS of
    ``BaselineWarning``, so every pre-existing ``pytest.warns(BaselineWarning)``
    and filter rule keeps matching;
  * ``collect_baseline_warnings`` captures the instance (typed, so callers key
    "the stage did nothing" on isinstance, not on prose) while ``warnings.warn``
    still fires;
  * the degenerate detrend really is a no-op (output == input) — the fact the
    warning must disclose;
  * nesting: an inner collect does not swallow warnings from the outer;
  * ``modules._build_warnings`` carries ``prov["detrend_warnings"]`` into a
    result's warning tuple (the Stats panel's surface).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import (
    BaselineWarning,
    ConstantBaselineWarning,
    collect_baseline_warnings,
    remove_baseline_wander,
)
from nanoscope.analysis.events import empty_df
from nanoscope.analysis.modules import _build_warnings, preprocess_for_detection

FS = 50_000.0
N = 50_000            # 1 s


def _drifting(n: int = N) -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(n, 100.0) + np.linspace(0.0, 25.0, n) + rng.normal(0.0, 1.5, n)
    return np.ascontiguousarray(x, dtype=np.float32)


# ================================================================================================
# the sink captures the kernel's own warning, typed
# ================================================================================================

def test_collect_captures_the_constant_baseline_warning():
    x = _drifting()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # capture must not depend on the filter state
        with collect_baseline_warnings() as sink:
            out = remove_baseline_wander(x, FS, window_s=1.0, polarity=-1)
    assert len(sink) == 1
    assert isinstance(sink[0], ConstantBaselineWarning)
    assert "cannot track drift" in str(sink[0])
    # ...and the warning is true: one block == a constant baseline == a no-op detrend.
    assert np.allclose(out, x, atol=1e-4), "the degenerate detrend should change nothing"


def test_warnings_warn_still_fires_and_the_subclass_matches_the_base():
    """Compatibility: every existing ``pytest.warns(BaselineWarning)`` keeps passing —
    the degeneracy is a subclass, and the stdlib emission was not replaced by the sink."""
    with pytest.warns(BaselineWarning, match="cannot track drift"):
        remove_baseline_wander(_drifting(), FS, window_s=1.0, polarity=-1)


def test_no_sink_installed_is_the_old_behaviour():
    """Outside any collect block the kernel just warns — no state leaks between calls."""
    with pytest.warns(ConstantBaselineWarning):
        remove_baseline_wander(_drifting(), FS, window_s=1.0, polarity=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BaselineWarning)
        # A healthy window raises nothing and appends nowhere.
        remove_baseline_wander(_drifting(), FS, window_s=0.1, polarity=-1)


def test_nested_collect_blocks_both_see_the_warning():
    x = _drifting()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with collect_baseline_warnings() as outer:
            with collect_baseline_warnings() as inner:
                remove_baseline_wander(x, FS, window_s=1.0, polarity=-1)
            assert len(inner) == 1
        assert len(outer) == 1, "an inner collect must not swallow the outer's view"


def test_preprocess_for_detection_degenerate_detrend_is_captured():
    """The sink works through the whole preprocess chain — the exact call
    DetectSchmitt.run wraps."""
    x = _drifting()
    params = {"detrend_enabled": True, "detrend_window_s": 1.0, "polarity": -1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with collect_baseline_warnings() as sink:
            out = preprocess_for_detection(x, FS, params)
    assert any(isinstance(w, ConstantBaselineWarning) for w in sink)
    assert np.allclose(out, x, atol=1e-4)


# ================================================================================================
# the result surface — _build_warnings reads the stamped provenance
# ================================================================================================

def test_build_warnings_carries_detrend_warnings():
    msg = ("window_s=1s yields 1 block(s) over 1s: the baseline is effectively "
           "constant and cannot track drift.")
    warns = _build_warnings({"detrend_warnings": (msg,)}, pd.DataFrame(empty_df()))
    assert any("cannot track drift" in w for w in warns)


def test_build_warnings_without_detrend_warnings_is_unchanged():
    assert _build_warnings({}, pd.DataFrame(empty_df())) == ()
