"""Validation boundary for provenance-only acquisition-bandwidth declarations."""
import json
from dataclasses import replace

import numpy as np
import pytest

from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.types import ChannelInfo, ChannelKind


@pytest.mark.parametrize("value", [
    AcquisitionBandwidth(),
    AcquisitionBandwidth(source="file_header", note="ABF1 cutoff unavailable",
                         header_fields=(("fTelegraphFilter", "nan"),)),
    AcquisitionBandwidth(status="disabled", source="file_header", note="Reported stage only"),
    AcquisitionBandwidth(2000, "reported", "user", "Bessel", 8, "Verified amplifier settings"),
    AcquisitionBandwidth(5000, "reported", "file_header", header_fields=(
        ("fTelegraphFilter", "5000.0"), ("fPostProcessLowpassFilter", "1000.0"))),
])
def test_bandwidth_strict_json_roundtrip(value):
    payload = json.loads(json.dumps(value.to_dict(), allow_nan=False))
    restored = AcquisitionBandwidth.from_dict(payload)
    assert restored == value
    assert restored.fingerprint() == value.fingerprint()
    assert hash(restored.fingerprint()) == hash(value.fingerprint())


@pytest.mark.parametrize("cutoff", [0, -1, float("nan"), float("inf"),
                                   float("-inf"), True, "2000", [], {}])
def test_invalid_cutoff_cannot_enter_via_construction_or_restore(cutoff):
    payload = {"cutoff_hz": cutoff, "status": "reported", "source": "user"}
    with pytest.raises(ValueError):
        AcquisitionBandwidth(**payload)
    with pytest.raises(ValueError):
        AcquisitionBandwidth.from_dict(payload)


@pytest.mark.parametrize("payload", [
    {"status": "guessed"},
    {"source": "measured"},
    {"cutoff_hz": 2000},
    {"cutoff_hz": 2000, "status": "disabled", "source": "user"},
    {"status": "reported", "source": "user"},
    {"status": "reported", "cutoff_hz": 2000},
    {"status": "disabled"},
    {"note": None},
    {"filter_type": 3},
    {"filter_order": 0},
    {"filter_order": 2.5},
    {"filter_order": True},
    {"filter_order": 2},
    {"header_fields": {"fSignalLowpassFilter": "2000"}},
    {"header_fields": [["x", 2000]]},
    {"header_fields": [["", "2000"]]},
    {"header_fields": [["x", "2"], ["x", "3"]]},
    {"header_fields": ["x"]},
    {"unexpected": 1},
])
def test_malformed_provenance_is_refused_not_downgraded_silently(payload):
    with pytest.raises(ValueError):
        AcquisitionBandwidth.from_dict(payload)


@pytest.mark.parametrize("payload", [None, [], "unknown", 5])
def test_non_object_restore_is_refused(payload):
    with pytest.raises(ValueError):
        AcquisitionBandwidth.from_dict(payload)


def test_numpy_scalars_normalize_and_header_lists_become_immutable():
    value = AcquisitionBandwidth(np.float64(1000), "reported", "user", "Bessel",
                                 np.int64(8), header_fields=[["stage", "verified"]])
    assert type(value.cutoff_hz) is float
    assert type(value.filter_order) is int
    assert value.header_fields == (("stage", "verified"),)
    assert isinstance(hash(value), int)


def test_user_confirmation_cannot_overwrite_original_channel_header():
    original = AcquisitionBandwidth(5000, "reported", "file_header",
                                    header_fields=(("fSignalLowpassFilter", "5000.0"),))
    channel = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT,
                          acquisition_bandwidth=original)
    override = AcquisitionBandwidth(2000, "reported", "user", "Bessel", 8,
                                    "Amplifier front-panel setting")
    assert original != override
    assert channel.acquisition_bandwidth is original
    # Dead-channel annotation at load must retain the unrelated acquisition declaration.
    assert replace(channel, is_dead=True).acquisition_bandwidth is original


def test_old_channel_constructors_still_mean_unknown_not_unfiltered():
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT, False, None)
    assert info.acquisition_bandwidth.status == "unknown"
    assert info.acquisition_bandwidth.cutoff_hz is None
