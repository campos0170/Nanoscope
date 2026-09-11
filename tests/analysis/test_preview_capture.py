from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.modules import DetectCusum, DetectSchmitt
from nanoscope.analysis.preview_capture import capture_detection_trace, publish_detection_trace


@pytest.mark.parametrize("module", [DetectSchmitt, DetectCusum])
@pytest.mark.parametrize("level", [-120., 120.])
@pytest.mark.parametrize("cutoff", [None, 1000.])
@pytest.mark.parametrize("reference", [False, True])
def test_capture_is_output_only(context_factory, module, level, cutoff, reference):
    x = np.random.default_rng(4).normal(level, .5, 20000).astype(np.float32)
    x[5000:6000] -= np.sign(level) * 30
    original = x.copy()
    settings = {"filter_cutoff_hz": cutoff, "min_dwell_s": .001}
    if reference:
        settings["baseline_open_pore_reference"] = {
            "version": 1, "mode": "constant", "current_pA": level,
            "intervals_s": [], "interpolation": "constant", "notes": "test"}
    ctx = context_factory(x, fs=10000., units="pA", params=settings)
    ordinary = module().run(ctx)
    with capture_detection_trace() as capture:
        observed = module().run(ctx)
    current, baseline = capture.require_single()
    assert current.shape == baseline.shape == x.shape
    assert not current.flags.writeable and not baseline.flags.writeable
    assert x.flags.writeable
    np.testing.assert_array_equal(x, original)
    pd.testing.assert_frame_equal(ordinary.payload.df, observed.payload.df, check_exact=True)
    # Wall-clock run stamps necessarily differ; every scientific provenance field is exact.
    stable = lambda p: {k: v for k, v in p.items() if k != "detected_at"}
    assert stable(ordinary.payload.provenance) == stable(observed.payload.provenance)
    assert ordinary.warnings == observed.warnings
    if reference:
        np.testing.assert_array_equal(baseline, np.full(x.size, level))


def test_capture_context_isolated_nested_and_after_failure():
    x = np.arange(10, dtype=np.float32)
    with capture_detection_trace() as outer:
        publish_detection_trace(x, x)
        with pytest.raises(RuntimeError):
            with capture_detection_trace() as inner:
                publish_detection_trace(x, x)
                raise RuntimeError("job failed")
        assert inner.count == outer.count == 1
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(publish_detection_trace, x, x).result()
        assert outer.count == 1
    publish_detection_trace(x, x)
    assert outer.count == 1
    assert np.shares_memory(outer.current, x)
    with pytest.raises(ValueError):
        outer.current[0] = 9
    assert x.flags.writeable


def test_missing_or_multiple_traces_are_not_substituted():
    with capture_detection_trace() as capture:
        with pytest.raises(ValueError, match="exactly one"):
            capture.require_single()
        publish_detection_trace(np.zeros(2), np.zeros(2))
        publish_detection_trace(np.ones(2), np.ones(2))
        with pytest.raises(ValueError, match="exactly one"):
            capture.require_single()
