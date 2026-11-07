"""The full linear conditioning chain has one honest frequency response.

These tests pin the scientific seam rather than any plot: every stage is evaluated
on one shared grid, zero-phase IIR stages retain their two-pass semantics, and a
narrow notch cannot disappear between sparse log-grid samples.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import signal

from nanoscope.analysis import filters


FS = 250_000.0


def test_combined_response_is_the_pointwise_product_on_one_shared_grid(monkeypatch):
    stages = [
        {"stage": "highpass", "cutoff_hz": 20.0, "order": 2, "kind": "butter"},
        {"stage": "running_mean", "window_s": 80e-6},
        {"stage": "savgol", "window_s": 120e-6, "polyorder": 3},
        {"stage": "lowpass", "cutoff_hz": 12_000.0, "order": 4, "kind": "bessel"},
    ]
    evaluated = []
    real_response = filters.response

    def record_response(stage, fs, **kwargs):
        result = real_response(stage, fs, **kwargs)
        evaluated.append((stage, kwargs, result))
        return result

    monkeypatch.setattr(filters, 'response', record_response)
    f, combined = filters.combined_response(stages, FS, n=2048, f_min_hz=1.0)

    expected = np.ones_like(combined)
    assert len(evaluated) == len(stages)
    for spec, (stage, kwargs, (stage_f, magnitude)) in zip(stages, evaluated):
        assert stage == spec['stage']
        assert all(kwargs[key] == value for key, value in spec.items() if key != 'stage')
        # This is the contract: no interpolation between independently chosen grids.
        assert np.array_equal(stage_f, f)
        expected *= magnitude

    # Check the responses actually consumed by the chain, exactly. Independently
    # repeating sosfreqz is not an exact oracle: at 1 Hz its high-pass numerator
    # condition exceeds 6e9, amplifying sub-ULP complex-evaluation differences.
    # The independent 80-digit check below separately bounds that numerical error.
    np.testing.assert_array_equal(combined, expected)


def _high_precision_sos_power(sos, frequencies, fs):
    """Exact stored-coefficient response and polynomial condition, at 80 digits.

    |a+b*z+c*z^2|^2 = a²+b²+c²+2*b*(a+c)*cos(w)+2*a*c*cos(2*w).
    Evaluating this identity in Decimal avoids both SciPy's complex Horner path
    and its double-precision cancellation; the input coefficients remain the
    actual binary64 filter, rather than an idealized unquantized design.
    """
    values, conditions = [], []
    pi = Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628')
    with localcontext() as context:
        context.prec = 80
        sections = [[Decimal.from_float(float(v)) for v in row] for row in sos]
        for frequency in frequencies:
            angle = 2*pi*Decimal.from_float(float(frequency))/Decimal.from_float(float(fs))
            cosine = term = Decimal(1)
            for k in range(1, 100):
                term *= -angle*angle/Decimal((2*k-1)*(2*k))
                cosine += term
                if abs(term) < Decimal('1e-78'):
                    break
            else:
                raise AssertionError('High-precision cosine did not converge')
            power, condition = Decimal(1), Decimal(0)
            for row in sections:
                squared = []
                for coefficients in (row[:3], row[3:]):
                    a, b, c = coefficients
                    magnitude_squared = (a*a+b*b+c*c+2*b*(a+c)*cosine
                                         +2*a*c*(2*cosine*cosine-1))
                    squared.append(magnitude_squared)
                    condition += sum(abs(v) for v in coefficients)/magnitude_squared.sqrt()
                power *= squared[0]/squared[1]
            values.append(float(power))
            conditions.append(float(condition))
    return np.asarray(values), np.asarray(conditions)


def test_low_cutoff_highpass_response_matches_high_precision_quantized_sos():
    sos = filters._design_sos(FS, 20., 2, 'butter', btype='high')
    frequency, magnitude = filters.response('highpass', FS, cutoff_hz=20., order=2,
                                             kind='butter', n=2048, f_min_hz=1.)
    reference, condition = _high_precision_sos_power(sos, frequency, FS)
    # Standard polynomial evaluation sensitivity is sum|coeff| / |P(z)|.
    # Propagate numerator + denominator conditioning through the squared magnitude.
    # gamma_64 is a conservative first-order envelope for the complex arithmetic,
    # phase evaluation, division and squaring; it is NOT a fixed relaxed tolerance.
    epsilon = np.finfo(float).eps
    gamma = 64*epsilon/(1-64*epsilon)
    relative_error = np.abs(magnitude/reference-1)
    assert condition[0] > 1e9, 'the fixture must exercise severe cancellation'
    assert np.all(relative_error <= gamma*condition), (
        frequency[np.argmax(relative_error/(gamma*condition))],
        float(np.max(relative_error/(gamma*condition))))


def _capture_stage_responses(monkeypatch):
    """Observe the stage values the chain actually consumed, without recomputing them."""
    captured = []
    real_response = filters.response

    def record(stage, fs, **kwargs):
        frequency, magnitude = real_response(stage, fs, **kwargs)
        captured.append((stage, dict(kwargs), frequency, magnitude))
        return frequency, magnitude

    monkeypatch.setattr(filters, "response", record)
    return captured


def _assert_requested_stage_parameters(expected, captured):
    """The numerical oracle must describe the requested design, not altered inputs."""
    assert [item[0] for item in captured] == [spec["stage"] for spec in expected]
    for spec, (_stage, kwargs, _frequency, _magnitude) in zip(expected, captured, strict=True):
        for name, value in spec.items():
            if name != "stage":
                assert kwargs[name] == value, (name, kwargs[name], value)


def _assert_iir_stage_matches_high_precision(stage, kwargs, frequency, magnitude, fs):
    """Independently check single/two-pass semantics of the quantized SOS design.

    A repeated SciPy SOS evaluation is not an exact numerical oracle near a
    cancelling numerator. The independent Decimal response and its polynomial
    condition bound both phase modes without an arbitrary fixed tolerance.
    """
    if stage in ("highpass", "lowpass"):
        sos = filters._design_sos(
            fs, kwargs["cutoff_hz"], kwargs["order"], kwargs["kind"],
            btype="high" if stage == "highpass" else "low")
    else:
        assert stage == "notch"
        lines = filters.notch_frequencies(
            fs, kwargs.get("freq_hz", 60.0),
            harmonics=kwargs.get("harmonics", 3), freqs=kwargs.get("freqs"))
        sos = np.vstack([
            np.hstack(signal.iirnotch(line, kwargs.get("q", 30.0), fs=fs))[None, :]
            for line in lines])
    power, condition = _high_precision_sos_power(sos, frequency, fs)
    reference = power if kwargs["zero_phase"] else np.sqrt(power)
    assert np.all(np.isfinite(reference) & (reference > 0)), "fixture needs a finite positive reference"
    assert np.all(np.isfinite(condition) & (condition > 0)), "fixture needs finite polynomial conditioning"
    assert np.all(np.isfinite(magnitude))
    epsilon = np.finfo(float).eps
    gamma = 64*epsilon/(1-64*epsilon)
    relative_error = np.abs(magnitude/reference-1)
    assert np.all(relative_error <= gamma*condition), (
        stage, kwargs["zero_phase"],
        frequency[np.argmax(relative_error/(gamma*condition))],
        float(np.max(relative_error/(gamma*condition))))


def _assert_captured_iir_product(frequency, combined, captured, fs):
    expected = np.ones_like(combined)
    for stage, kwargs, stage_frequency, magnitude in captured:
        np.testing.assert_array_equal(stage_frequency, frequency)
        expected *= magnitude
        _assert_iir_stage_matches_high_precision(
            stage, kwargs, stage_frequency, magnitude, fs)
    np.testing.assert_array_equal(combined, expected)


def test_zero_phase_default_squares_each_iir_design_exactly_once(monkeypatch):
    stages = [
        {"stage": "highpass", "cutoff_hz": 10.0, "order": 2, "kind": "butter"},
        {"stage": "notch", "freq_hz": 60.0, "q": 20.0, "harmonics": 2},
        {"stage": "lowpass", "cutoff_hz": 10_000.0, "order": 3, "kind": "bessel"},
    ]
    captured = _capture_stage_responses(monkeypatch)
    f_single, single = filters.combined_response(
        stages, FS, zero_phase=False, n=512, f_min_hz=1.0
    )
    single_stages = list(captured)
    captured.clear()
    f_applied, applied = filters.combined_response(
        stages, FS, zero_phase=True, n=512, f_min_hz=1.0
    )

    np.testing.assert_array_equal(f_single, f_applied)
    _assert_requested_stage_parameters(stages, single_stages)
    _assert_requested_stage_parameters(stages, captured)
    assert all(item[1]["zero_phase"] is False for item in single_stages)
    assert all(item[1]["zero_phase"] is True for item in captured)
    # Omitted stage flags inherit the chain default. Verify each consumed stage
    # independently as |H| or |H|², then its exact contribution to the product.
    _assert_captured_iir_product(f_single, single, single_stages, FS)
    _assert_captured_iir_product(f_applied, applied, captured, FS)


def test_a_stage_can_explicitly_override_the_chain_zero_phase_default(monkeypatch):
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
    captured = _capture_stage_responses(monkeypatch)
    f, combined = filters.combined_response(stages, FS, zero_phase=True, f_min_hz=1.0)
    _assert_requested_stage_parameters(stages, captured)
    assert captured[0][1]["zero_phase"] is False  # explicit stage override
    assert captured[1][1]["zero_phase"] is True   # omitted flag inherits default
    _assert_captured_iir_product(f, combined, captured, FS)


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


def test_notch_refinement_still_uses_the_same_grid_for_every_stage(monkeypatch):
    stages = [
        {"stage": "notch", "freqs": (x for x in (60.0, 120.0, 180.0)), "q": 30.0},
        {"stage": "lowpass", "cutoff_hz": 1_000.0, "order": 2, "kind": "butter"},
    ]
    captured = _capture_stage_responses(monkeypatch)
    f, combined = filters.combined_response(stages, 10_000.0, n=64, f_min_hz=1.0)
    _assert_requested_stage_parameters(
        [{**stages[0], "freqs": (60.0, 120.0, 180.0)}, stages[1]], captured)
    assert all(item[1]["zero_phase"] is True for item in captured)
    _assert_captured_iir_product(f, combined, captured, 10_000.0)


def test_highpass_defaults_describe_the_filter_that_highpass_actually_applies():
    f, combined = filters.combined_response(
        [{"stage": "highpass", "cutoff_hz": 100.0}], FS, n=1024, f_min_hz=1.0
    )
    # The kernel's default is order-4 Butterworth. Compare to its independent
    # coefficient oracle, not a second cancellation-sensitive SOS accumulation.
    sos = filters._design_sos(FS, 100., 4, 'butter', btype='high')
    reference, condition = _high_precision_sos_power(sos, f, FS)
    epsilon = np.finfo(float).eps
    gamma = 64*epsilon/(1-64*epsilon)
    assert np.all(np.abs(combined/reference-1) <= gamma*condition)


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
