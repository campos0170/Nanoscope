import gc
import tracemalloc
import weakref

import numpy as np
import pytest

from nanoscope.analysis.baseline import baseline_subtracted
from nanoscope.gui.display_residual import DisplayResidualCache


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stride", [1, 2])
def test_bounded_subtraction_is_exact_and_input_preserving(dtype, stride):
    rng = np.random.default_rng(3)
    x = rng.normal(-123, 19, 1031).astype(dtype)[::stride]
    b = rng.normal(-121, 4, 1031).astype(dtype)[::stride]
    before = x.copy(), b.copy()
    cache = DisplayResidualCache(chunk_samples=100)
    actual = cache.residual("x", x, b)
    np.testing.assert_array_equal(actual, baseline_subtracted(x, b))
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(b, before[1])
    assert actual.dtype == np.float32 and not actual.flags.writeable
    assert cache.residual("x", x, b) is actual
    cache.invalidate("x")
    assert cache.residual("x", x, b) is not actual


def test_lru_budget_replacement_and_input_lifetimes():
    cache = DisplayResidualCache(max_bytes=80)
    x, b = np.ones(10, np.float32), np.zeros(10, np.float32)
    a = cache.residual("a", x, b)
    cache.residual("b", x, b)
    assert cache.residual("a", x, b) is a
    cache.residual("c", x, b)
    assert list(cache._entries) == ["a", "c"]
    assert cache.nbytes == 80
    replacement = x.copy()
    assert cache.residual("a", replacement, b) is not a
    ref = weakref.ref(replacement)
    del replacement
    gc.collect()
    assert ref() is None
    cache.retain(set())
    assert cache.nbytes == 0
    large = np.ones(21, np.float32)
    cache.residual("large", large, large)
    assert cache.nbytes == 0


def test_temporary_allocation_is_bounded_not_full_float64_copies():
    x = np.full(1_000_000, -123.456, dtype=np.float32)
    b = np.full(x.size, -120.123, dtype=np.float32)
    cache = DisplayResidualCache(chunk_samples=32768)
    tracemalloc.start()
    out = cache.residual("x", x, b)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < out.nbytes + 2 * 1024 * 1024
    np.testing.assert_array_equal(out, baseline_subtracted(x, b))


def test_real_view_rebuild_reuses_and_explicit_changes_invalidate(qapp, qtbot, tmp_path):
    from nanoscope.core.calibration import ChannelCalibration
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.session import Session
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    from nanoscope.gui.trace_view import TraceView
    x = np.linspace(-125, -115, 1000, dtype=np.float32)
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "cache.dat", format_name="synthetic", n_samples=x.size,
                         sample_rate_hz=10000., rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta, (x,)))
    session.add_layer(doc_id, 0)
    view = TraceView(session)
    qtbot.addWidget(view)
    key = f"{doc_id}#0"
    b = np.full(x.size, -120., np.float32)
    view.set_baseline_subtracted(doc_id, 0, b)
    first = view._residual_cache._entries[key][2]
    view.rebuild()
    assert view._residual_cache._entries[key][2] is first
    b[:] = -119.
    view.set_baseline_subtracted(doc_id, 0, b)  # explicit same-object replacement
    second = view._residual_cache._entries[key][2]
    assert second is not first
    np.testing.assert_array_equal(second, baseline_subtracted(x, b))
    session.set_calibration(doc_id, 0, ChannelCalibration(gain=2.), write_sidecar=False)
    third = view._residual_cache._entries[key][2]
    assert third is not second
    np.testing.assert_array_equal(third, baseline_subtracted(session.calibrated(doc_id, 0).data, b))
    session.close_document(doc_id)
    assert view._residual_cache.nbytes == 0
    assert key not in view._subtracted
