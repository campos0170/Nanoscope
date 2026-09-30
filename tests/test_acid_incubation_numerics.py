"""The quantized-current baseline must equal the stated moving median."""
import numpy as np
import pytest

from nanoscope.analysis.acid_incubation_numerics import median_filter_1d


@pytest.mark.parametrize('length', [1, 2, 5, 5001])
@pytest.mark.parametrize('window', [1, 3, 1001])
@pytest.mark.parametrize('mode,padding', [('reflect', 'symmetric'), ('nearest', 'edge')])
def test_quantized_current_matches_explicit_order_statistic(length, window, mode, padding):
    rng = np.random.default_rng(68)
    values = (rng.integers(-1000, -930, length) * .30517578125).astype(np.float32)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode=padding)
    result = median_filter_1d(values, window, mode)
    indices = np.unique(np.linspace(0, length - 1, min(length, 51), dtype=int))
    expected = np.asarray([np.median(padded[i:i + window]) for i in indices], dtype=np.float32)
    np.testing.assert_array_equal(result[indices], expected)
    assert result.dtype == values.dtype


def test_recording_length_cannot_change_an_unaffected_window():
    values = (np.random.default_rng(7).integers(-980, -950, 6001) * .30517578125).astype(np.float32)
    full = median_filter_1d(values, 1001)
    prefix = median_filter_1d(values[:3500], 1001)
    np.testing.assert_array_equal(full[:3000], prefix[:3000])
