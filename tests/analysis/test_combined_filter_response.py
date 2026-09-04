"""The full linear conditioning chain has one honest frequency response.

These tests pin the scientific seam rather than any plot: every stage is evaluated
on one shared grid, zero-phase IIR stages retain their two-pass semantics, and a
narrow notch cannot disappear between sparse log-grid samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import filters


FS = 250_000.0


def test_combined_response_is_the_pointwise_product_on_one_shared_grid():
    stages = [
        {"stage": "highpass", "cutoff_hz": 20.0, "order": 2, "kind": "butter"},
        {"stage": "running_mean", "window_s": 80e-6},
        {"stage": "savgol", "window_s": 120e-6, "polyorder": 3},
        {"stage": "lowpass", "cutoff_hz": 12_000.0, "order": 4, "kind": "bessel"},
    ]
    f, combined = filters.combined_response(stages, FS, n=2048, f_min_hz=1.0)

    expected = np.ones_like(combined)
    for spec in stages:
        kwargs = dict(spec)
        stage = kwargs.pop("stage")
        stage_f, magnitude = filters.response(
            stage, FS, n=f.size, f_min_hz=f[0], **kwargs
        )
        # This is the contract: no interpolation between independently chosen grids.
        assert np.array_equal(stage_f, f)
        expected *= magnitude

    # scipy's repeated SOS accumulation can differ by a few hundred ULPs between
    # otherwise identical calls; this remains far below any physical precision.
    np.testing.assert_allclose(combined, expected, rtol=1e-11, atol=0.0)


def test_zero_phase_default_squares_each_iir_design_exactly_once():
    stages = [
        {"stage": "highpass", "cutoff_hz": 10.0, "order": 2, "kind": "butter"},
        {"stage": "notch", "freq_hz": 60.0, "q": 20.0, "harmonics": 2},
        {"stage": "lowpass", "cutoff_hz": 10_000.0, "order": 3, "kind": "bessel"},
    ]
    f_single, single = filters.combined_response(
        stages, FS, zero_phase=False, n=512, f_min_hz=1.0
    )
    f_applied, applied = filters.combined_response(
        stages, FS, zero_phase=True, n=512, f_min_hz=1.0
    )

    assert np.array_equal(f_single, f_applied)
    np.testing.assert_allclose(applied, single**2, rtol=2e-14, atol=1e-300)


def test_a_stage_can_explicitly_override_the_chain_zero_phase_default():
    stages = [
        {
            "stage": "highpass",
            "cutoff_hz": 20.0,
            "order": 2,
            "kind": "butter",
            "zero_phase": False,
        },
        {"stage": "lowpass", "cutoff_hz": 8_000.0, "order": 2, "kind": "butter"},
    ]
    f, combined = filters.combined_response(stages, FS, zero_phase=True, f_min_hz=1.0)
    _, high = filters.response(
        "highpass", FS, cutoff_hz=20.0, order=2, kind="butter",
        zero_phase=False, n=f.size, f_min_hz=f[0],
    )
    _, low = filters.response(
        "lowpass", FS, cutoff_hz=8_000.0, order=2, kind="butter",
        zero_phase=True, n=f.size, f_min_hz=f[0],
    )
    # At Nyquist SciPy's independently repeated SOS accumulation may differ by one
    # machine epsilon around a mathematical zero; that is numerical dust, not a
    # response disagreement.
    np.testing.assert_allclose(combined, high * low, rtol=2e-14, atol=1e-15)


def test_notch_automatically_refines_but_keeps_a_geometric_shared_grid():
    # Thirty-two points across four decades would miss this 2 Hz-wide trough almost
    # completely.  The API raises the count enough that even the nearest (not exactly
    # centred) sample truthfully shows the applied two-pass notch below -40 dB.
    f, magnitude = filters.combined_response(
        [{"stage": "notch", "freq_hz": 60.0, "q": 30.0, "harmonics": 1}],
        10_000.0,
        n=32,
        f_min_hz=1.0,
    )

    assert f.size > 32
    assert f[0] == pytest.approx(1.0)
    assert f[-1] == pytest.approx(5_000.0)
    assert np.allclose(f[1:] / f[:-1], f[1] / f[0], rtol=2e-12)
    nearest = int(np.argmin(np.abs(f - 60.0)))
    assert magnitude[nearest] < 0.01


def test_notch_refinement_still_uses_the_same_grid_for_every_stage():
    stages = [
        {"stage": "notch", "freqs": (x for x in (60.0, 120.0, 180.0)), "q": 30.0},
        {"stage": "lowpass", "cutoff_hz": 1_000.0, "order": 2, "kind": "butter"},
    ]
    f, combined = filters.combined_response(stages, 10_000.0, n=64, f_min_hz=1.0)
    _, notch = filters.response(
        "notch", 10_000.0, freqs=(60.0, 120.0, 180.0), q=30.0,
        n=f.size, f_min_hz=f[0],
    )
    _, low = filters.response(
        "lowpass", 10_000.0, cutoff_hz=1_000.0, order=2, kind="butter",
        n=f.size, f_min_hz=f[0],
    )
    np.testing.assert_allclose(combined, notch * low, rtol=2e-14, atol=0.0)


def test_highpass_defaults_describe_the_filter_that_highpass_actually_applies():
    f, combined = filters.combined_response(
        [{"stage": "highpass", "cutoff_hz": 100.0}], FS, n=1024, f_min_hz=1.0
    )
    _, actual_default = filters.response(
        "highpass", FS, cutoff_hz=100.0, order=4, kind="butter",
        n=f.size, f_min_hz=f[0],
    )
    np.testing.assert_allclose(combined, actual_default, rtol=0.0, atol=0.0)


def test_empty_chain_is_the_flat_identity():
    f, magnitude = filters.combined_response([], FS, n=37)
    assert f.size == 37
    assert np.all(magnitude == 1.0)


@pytest.mark.parametrize(
    "stages,match",
    [
        ("lowpass", "sequence of stage mappings"),
        (["lowpass"], r"stages\[0\] must be a mapping"),
        ([{}], "missing required key"),
        ([{"stage": "median", "window_s": 1e-3}], "non-linear rank filter"),
        ([{"stage": "wavelet"}], "stage must be one of"),
        ([{"stage": "lowpass", "cutoff_hz": 1_000.0, "q": 30.0}], "unexpected"),
        ([{"stage": "lowpass"}], "needs cutoff_hz"),
        ([{"stage": "notch", "freq_hz": 60.0, "q": np.nan}], "positive and finite"),
        ([{"stage": "notch", "freq_hz": 60.0, "harmonics": 1.5}], "harmonics"),
        ([{"stage": "savgol", "window_s": 1e-3, "polyorder": 2.5}], "polyorder"),
        ([{"stage": "lowpass", "cutoff_hz": 1_000.0, "zero_phase": "yes"}], "boolean"),
    ],
)
def test_invalid_stage_specifications_are_refused_with_stage_context(stages, match):
    with pytest.raises(ValueError, match=match):
        filters.combined_response(stages, FS)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n": 1}, "integer >= 2"),
        ({"n": True}, "integer >= 2"),
        ({"f_min_hz": 0.0}, "positive, finite"),
        ({"f_min_hz": FS / 2}, "below Nyquist"),
        ({"zero_phase": "yes"}, "boolean"),
    ],
)
def test_invalid_chain_grid_controls_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        filters.combined_response([], FS, **kwargs)


def test_extreme_notch_resolution_is_refused_instead_of_allocating_without_bound():
    with pytest.raises(ValueError, match=r"needs .* points.*raise f_min_hz"):
        filters.combined_response(
            [{"stage": "notch", "freq_hz": 60.0, "q": 3_000.0}],
            FS,
            n=32,
            f_min_hz=1.0,
        )
