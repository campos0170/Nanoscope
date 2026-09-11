"""Stored filter requests cannot silently become an unfiltered/default-order trace."""
import numpy as np
import pytest

from nanoscope.analysis.filters import lowpass
from nanoscope.analysis.modules import detection_signal, preprocess_for_detection


FS = 50_000.


@pytest.fixture
def trace():
    x = np.random.default_rng(410).normal(100., 2., 20_000).astype(np.float32)
    x[2000:2500] -= 30.
    return x


@pytest.mark.parametrize("cutoff", ["invalid", [], {}, 0., -1., np.nan, np.inf, -np.inf, FS / 2, FS])
def test_explicit_invalid_cutoff_refuses_without_any_conditioning(trace, cutoff, monkeypatch):
    import nanoscope.analysis.modules as modules

    calls = []
    monkeypatch.setattr(modules, "preprocess_for_detection", lambda *args: calls.append(args))
    before = trace.copy()
    with pytest.raises(ValueError, match="saved low-pass cutoff"):
        detection_signal(trace, FS, {"filter_cutoff_hz": cutoff})
    assert calls == []
    np.testing.assert_array_equal(trace, before)


@pytest.mark.parametrize("fs", [None, 0., -1., np.nan, np.inf, -np.inf])
def test_requested_lowpass_requires_valid_timebase(trace, fs):
    with pytest.raises(ValueError, match="saved low-pass filter requires"):
        detection_signal(trace, fs, {"filter_cutoff_hz": 2000.})


@pytest.mark.parametrize("kind", ["bessel", "butter"])
@pytest.mark.parametrize("order", ["invalid", "4.0", 0, -1, 3.5, np.nan, np.inf, -np.inf, True, False, [], {}])
def test_explicit_invalid_iir_order_never_defaults_or_truncates(trace, kind, order):
    with pytest.raises(ValueError, match="saved low-pass IIR order"):
        detection_signal(trace, FS, {"filter_cutoff_hz": 2000., "filter_kind": kind,
                                     "filter_order": order})


@pytest.mark.parametrize("kind", ["bessel", "butter"])
@pytest.mark.parametrize("order,expected", [(None, 8), (2, 2), (4., 4), ("4", 4), (np.int64(6), 6)])
def test_valid_iir_orders_and_null_default_replay_bit_identically(trace, kind, order, expected):
    prov = {"filter_cutoff_hz": 2000., "filter_kind": kind, "filter_order": order}
    np.testing.assert_array_equal(detection_signal(trace, FS, prov),
                                  lowpass(trace, FS, 2000., order=expected, kind=kind))


@pytest.mark.parametrize("prov", [
    {"filter_cutoff_hz": 2000.},
    {"filter_cutoff_hz": "2000", "filter_kind": None, "filter_order": None},
    {"filter_cutoff_hz": 2000., "filter_kind": "", "filter_order": 8},
])
def test_legacy_omitted_filter_kind_and_order_keep_original_defaults(trace, prov):
    np.testing.assert_array_equal(detection_signal(trace, FS, prov),
                                  lowpass(trace, FS, 2000., kind="bessel", order=8))


@pytest.mark.parametrize("order", [None, 0, -1, 1.25, "unused", np.nan, np.inf, {}, []])
def test_gaussian_unused_order_remains_irrelevant(trace, order):
    np.testing.assert_array_equal(
        detection_signal(trace, FS, {"filter_cutoff_hz": 2000., "filter_kind": "gaussian",
                                    "filter_order": order}),
        lowpass(trace, FS, 2000., kind="gaussian"))


@pytest.mark.parametrize("fs", [None, FS])
@pytest.mark.parametrize("prov", [{}, {"filter_cutoff_hz": None},
                                   {"filter_cutoff_hz": None, "filter_kind": "not-used",
                                    "filter_order": "not-used"}])
def test_unrequested_filter_keeps_legacy_unconditioned_identity(trace, fs, prov):
    np.testing.assert_array_equal(detection_signal(trace, fs, prov), trace)


def test_valid_preprocessing_plus_lowpass_replays_in_unchanged_order(trace):
    params = {"detrend_enabled": True, "detrend_method": "percentile", "polarity": -1,
              "detrend_window_s": .05, "detrend_percentile": 90.}
    provenance = dict(params, percentile=params["detrend_percentile"], filter_cutoff_hz=2000.,
                      filter_kind="butter", filter_order=4)
    expected = lowpass(preprocess_for_detection(trace, FS, params), FS, 2000., kind="butter", order=4)
    np.testing.assert_array_equal(detection_signal(trace, FS, provenance), expected)


def test_unknown_kind_does_not_fall_back_to_bessel(trace):
    with pytest.raises(ValueError, match="kind"):
        detection_signal(trace, FS, {"filter_cutoff_hz": 2000., "filter_kind": "missing-plugin"})
