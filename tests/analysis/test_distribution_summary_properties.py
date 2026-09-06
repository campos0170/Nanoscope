"""Independent numerical-reference and unit-change invariants for new summaries."""
import numpy as np
import pytest
from scipy.stats import ecdf

from nanoscope.analysis.distribution_summary import summarize_distribution


@pytest.mark.parametrize("size", [1, 2, 3, 4, 10, 100, 1000])
def test_tied_samples_match_reference_without_mutating_input(size):
    rng = np.random.default_rng(906 + size)
    for _ in range(10):
        values = rng.integers(-20, 20, size=size).astype(float)
        before = values.copy()
        reference = ecdf(values)
        for kind, expected in (("ecdf", reference.cdf), ("survival", reference.sf)):
            result = summarize_distribution(values, kind=kind)
            np.testing.assert_array_equal(result.x, expected.quantiles)
            np.testing.assert_allclose(result.y, expected.probabilities, rtol=0, atol=2e-16)
            assert not result.x.flags.writeable and not result.y.flags.writeable
        box = summarize_distribution(values, kind="box")
        q1, median, q3 = np.quantile(values, [.25, .5, .75], method="linear")
        assert (box.stats["q1"], box.stats["median"], box.stats["q3"]) == (q1, median, q3)
        inside = values[(values >= q1 - 1.5 * (q3 - q1)) & (values <= q3 + 1.5 * (q3 - q1))]
        assert box.stats["lower_whisker"] == inside.min()
        assert box.stats["upper_whisker"] == inside.max()
        np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("kind", ["ecdf", "survival", "box", "violin"])
@pytest.mark.parametrize("scale", [.001, 1., 1000.])
def test_positive_unit_change_and_offset_have_expected_effect(kind, scale):
    values = np.array([1., 1., 2., 3., 4., 5., 8., 13., 21.])
    original = summarize_distribution(values, kind=kind)
    changed = summarize_distribution(values * scale + 2.5, kind=kind)
    np.testing.assert_allclose(changed.x, original.x * scale + 2.5, rtol=2e-12, atol=2e-12)
    if kind in ("ecdf", "survival"):
        np.testing.assert_array_equal(changed.y, original.y)
    elif kind == "box":
        for key, value in original.stats.items():
            expected = value * scale + (0 if key == "iqr" else 2.5)
            assert changed.stats[key] == pytest.approx(expected, rel=2e-12, abs=2e-12)
    else:
        np.testing.assert_allclose(changed.y, original.y / scale, rtol=2e-11, atol=2e-11)
        assert changed.stats["bandwidth"] == pytest.approx(original.stats["bandwidth"] * scale)


def test_scott_bandwidth_matches_explicit_unweighted_formula():
    values = np.array([1., 1., 2., 4.])
    result = summarize_distribution(values, kind="violin")
    assert result.stats["scott_factor"] == pytest.approx(len(values) ** (-.2))
    assert result.stats["bandwidth"] == pytest.approx(len(values) ** (-.2) * np.std(values, ddof=1))


@pytest.mark.parametrize("kind", ["ecdf", "survival", "box", "violin"])
def test_empty_and_nonfinite_observations_are_disclosed(kind):
    for values in ([], [np.nan, np.inf, -np.inf]):
        result = summarize_distribution(np.asarray(values, dtype=float), kind=kind)
        assert result.n == 0 and result.n_excluded == len(values)
        assert len(result.x) == len(result.y) == 0 and result.warnings


@pytest.mark.parametrize("values", [[1.], [1., 2.], [4., 4., 4., 4.]])
def test_sparse_and_constant_violin_uses_only_observed_points(values):
    result = summarize_distribution(values, kind="violin")
    np.testing.assert_array_equal(result.x, np.unique(values))
    assert not result.y.any() and not result.stats
    assert any("No violin density estimated" in note for note in result.warnings)


@pytest.mark.parametrize("values", [[True, False], [1+1j], [[1., 2.]], [2**53+1],
                                    [-(2**53+1)], np.array([2**64-1], dtype=np.uint64),
                                    np.array([-(2**63)], dtype=np.int64)])
def test_nonreal_nonvector_or_inexact_integer_inputs_refused(values):
    with pytest.raises(ValueError):
        summarize_distribution(values)
