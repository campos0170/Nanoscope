"""Regression tests for :func:`nanoscope.analysis.filters.normalize`.

The kernel behind the owner-requested "Normalize" preprocessing stage. Its whole
reason for existing is to divide the picoamps out of a trace, so the tests here pin
the two things that would be silently wrong otherwise: that each mode really is the
rescale its name claims (a wrong divisor produces plausible-looking numbers that are
simply the wrong size), and that the DEGENERATE inputs raise instead of returning an
array of NaN/inf that looks like a filter result.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.filters import NORMALIZE_MODES, noise_rms, normalize

FS = 50_000.0


def _trace(n: int = 20_000, *, seed: int = 7) -> np.ndarray:
    """Open pore at 500 pA with 10 pA noise -- the app's usual shape, in pA."""
    rng = np.random.default_rng(seed)
    return (500.0 + rng.normal(0.0, 10.0, n)).astype(np.float32)


@pytest.mark.parametrize("mode", NORMALIZE_MODES)
def test_returns_contiguous_float32_of_the_same_length(mode):
    """Same output contract as every other kernel in the module."""
    x = _trace()
    y = normalize(x, mode=mode)
    assert y.dtype == np.float32
    assert y.shape == x.shape
    assert y.flags["C_CONTIGUOUS"]


def test_zscore_standardises_on_mean_and_std():
    x = _trace()
    y = np.asarray(normalize(x, mode="zscore"), dtype=np.float64)
    assert float(np.mean(y)) == pytest.approx(0.0, abs=1e-5)
    assert float(np.std(y)) == pytest.approx(1.0, rel=1e-4)


def test_robust_divides_by_the_noise_rms_the_detector_thresholds_on():
    """'robust' is not merely "a z-score with a median": its divisor is the SAME
    1.4826*MAD sigma `detect_events` sets its threshold from, which is what makes a
    value of 1.0 mean "one baseline noise sigma" rather than "one sigma of
    noise-plus-events"."""
    x = _trace()
    y = np.asarray(normalize(x, mode="robust"), dtype=np.float64)
    expected = (np.asarray(x, dtype=np.float64) - np.median(x)) / noise_rms(x)
    assert np.allclose(y, expected, atol=1e-5)


def test_robust_is_not_dragged_by_events_the_way_zscore_is():
    """The reason 'robust' is offered at all. Contaminate a clean trace with deep
    blockades: the z-score divisor inflates (every depth in sigma shrinks), the robust
    one barely moves."""
    x = _trace().astype(np.float64)
    contaminated = x.copy()
    contaminated[::50] -= 200.0                     # 2% of samples, 20-sigma deep
    z_clean = np.std(x)
    z_dirty = np.std(contaminated)
    r_clean = noise_rms(x)
    r_dirty = noise_rms(contaminated)
    assert z_dirty > 1.5 * z_clean                  # std is wrecked...
    assert r_dirty == pytest.approx(r_clean, rel=0.1)   # ...MAD is not
    # And that shows up in the normalised depth: the same 200 pA blockade reads far
    # shallower in z-score units than in robust ones.
    zs = np.asarray(normalize(contaminated, mode="zscore"), dtype=np.float64)
    rb = np.asarray(normalize(contaminated, mode="robust"), dtype=np.float64)
    assert abs(zs.min()) < abs(rb.min())


def test_minmax_maps_onto_the_documented_interval():
    x = _trace()
    centred = np.asarray(normalize(x, mode="minmax", center=True), dtype=np.float64)
    assert float(centred.min()) == pytest.approx(-0.5, abs=1e-6)
    assert float(centred.max()) == pytest.approx(+0.5, abs=1e-6)
    plain = np.asarray(normalize(x, mode="minmax", center=False), dtype=np.float64)
    assert float(plain.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(plain.max()) == pytest.approx(1.0, abs=1e-6)


def test_center_false_keeps_the_offset_for_the_dividing_modes():
    """For z-score/robust, ``center=False`` divides WITHOUT subtracting, so a 500 pA
    open pore stays firmly positive instead of straddling zero."""
    x = _trace()
    y = np.asarray(normalize(x, mode="zscore", center=False), dtype=np.float64)
    assert float(np.mean(y)) == pytest.approx(float(np.mean(x)) / float(np.std(x)),
                                              rel=1e-4)
    assert float(y.min()) > 0.0


def test_the_unit_really_is_destroyed():
    """The property the GUI's warning banner exists for: scaling the INPUT by an
    arbitrary gain leaves the output unchanged, i.e. nothing downstream can recover
    picoamps from a normalised trace."""
    x = _trace().astype(np.float64)
    a = np.asarray(normalize(x, mode="zscore"), dtype=np.float64)
    b = np.asarray(normalize(x * 37.0, mode="zscore"), dtype=np.float64)
    assert np.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("mode", NORMALIZE_MODES)
def test_constant_trace_raises_rather_than_returning_nan(mode):
    """A constant trace has no scale to divide by. Returning inf/NaN here would look
    like a filter result; the GUI surfaces this message in its preview-error label."""
    with pytest.raises(ValueError, match="constant"):
        normalize(np.full(1000, 42.0, dtype=np.float32), mode=mode)


def test_empty_trace_raises():
    with pytest.raises(ValueError, match="empty"):
        normalize(np.array([], dtype=np.float32), mode="zscore")


def test_unknown_mode_raises_and_names_the_accepted_set():
    with pytest.raises(ValueError, match="zscore"):
        normalize(_trace(), mode="whitening")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_samples_raise_instead_of_poisoning_every_sample(bad):
    x = _trace().astype(np.float64)
    x[123] = bad
    with pytest.raises(ValueError, match="NaN/inf"):
        normalize(x, mode="zscore")
