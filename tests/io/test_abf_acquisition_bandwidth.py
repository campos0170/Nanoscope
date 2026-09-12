"""ABF header provenance must follow Neo's channel mapping and leave samples alone."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.types import SampleRange
from nanoscope.io.abf import AbfReader


def _reader(header, ids=("0",), names=None):
    if names is None:
        names = tuple(f"IN{i}" for i in ids)
    signal_channels = np.array(list(zip(names, ids, ["pA"] * len(ids))),
                               dtype=[("name", "U32"), ("id", "U16"), ("units", "U8")])
    return SimpleNamespace(_axon_info=header, header={"signal_channels": signal_channels})


def _abf2(adc):
    return _reader({"fFileVersionNumber": 2.0, "listADCInfo": [adc]})


def _bandwidth(adc):
    return AbfReader._channel_infos(_abf2(adc))[0].acquisition_bandwidth


def test_abf1_uses_physical_adc_id_not_display_position():
    lowpass = np.full(16, 999., dtype=float)
    lowpass[15], lowpass[2] = 5000., 2000.
    reader = _reader({
        "fFileVersionNumber": 1.83, "nADCSamplingSeq": [15, 2] + [-1] * 14,
        "fSignalLowpassFilter": lowpass, "nLowpassFilterType": [3] * 16,
    }, ids=("15", "2"))
    infos = AbfReader._channel_infos(reader)
    assert [info.acquisition_bandwidth.cutoff_hz for info in infos] == [5000, 2000]
    assert dict(infos[0].acquisition_bandwidth.header_fields)["physical_ADC_index"] == "15"
    assert infos[0].acquisition_bandwidth.filter_type == "Bessel"


def test_abf2_uses_adc_entry_index_not_physical_adc_number_or_sequence():
    reader = _reader({
        "fFileVersionNumber": 2.1,
        "listADCInfo": [
            {"nADCNum": 15, "nADCSamplingSeq": 1, "fSignalLowpassFilter": 5000},
            {"nADCNum": 2, "nADCSamplingSeq": 0, "fSignalLowpassFilter": 2000},
        ],
    }, ids=("0", "1"), names=("IN15", "IN2"))
    infos = AbfReader._channel_infos(reader)
    assert [info.acquisition_bandwidth.cutoff_hz for info in infos] == [5000, 2000]
    assert dict(infos[0].acquisition_bandwidth.header_fields)["nADCNum"] == "15"


def test_current_neo_abf1_absent_fields_remain_explicitly_unknown():
    reader = _reader({"fFileVersionNumber": 1.83, "nTelegraphEnable": [0] * 16},
                     ids=("15",))
    result = AbfReader._channel_infos(reader)[0].acquisition_bandwidth
    assert result.status == "unknown"
    assert result.cutoff_hz is None
    assert "Neo ABF1 parser does not expose" in result.note


@pytest.mark.parametrize("header", [None, {}, {"fFileVersionNumber": float("nan")},
                                    {"fFileVersionNumber": 2, "listADCInfo": []}])
def test_absent_or_unmappable_header_does_not_prevent_channel_loading(header):
    result = AbfReader._channel_infos(_reader(header))[0].acquisition_bandwidth
    assert result.status == "unknown"
    assert result.cutoff_hz is None


@pytest.mark.parametrize("cutoff", [float("nan"), float("inf"), float("-inf"),
                                   0, -1, "bad", None, True])
def test_invalid_header_cutoff_is_preserved_as_text_but_never_used(cutoff):
    result = _bandwidth({"fSignalLowpassFilter": cutoff, "nLowpassFilterType": 3})
    assert result.status == "unknown"
    assert result.cutoff_hz is None
    assert dict(result.header_fields)["fSignalLowpassFilter"] == str(cutoff)
    assert AcquisitionBandwidth.from_dict(json.loads(
        json.dumps(result.to_dict(), allow_nan=False))) == result


def test_signal_conditioner_bypass_is_not_a_100khz_corner():
    result = _bandwidth({"fSignalLowpassFilter": 100000, "nLowpassFilterType": 0})
    assert result.status == "disabled"
    assert result.cutoff_hz is None
    assert "does not establish an unfiltered acquisition chain" in result.note


def test_telegraph_default_is_not_claimed_to_be_bandwidth_or_filter_disabled():
    result = _bandwidth({"fTelegraphFilter": 100000, "nTelegraphEnable": 1})
    assert result.status == "unknown"
    assert result.cutoff_hz is None


def test_disabling_telegraph_readback_does_not_mean_hardware_filter_disabled():
    result = _bandwidth({"fTelegraphFilter": 1000, "nTelegraphEnable": 0})
    assert result.status == "unknown"
    assert result.cutoff_hz is None


def test_multiple_stages_preserve_declared_values_without_equivalent_response():
    result = _bandwidth({"fTelegraphFilter": 3000, "nTelegraphEnable": 1,
                         "fSignalLowpassFilter": 5000, "nLowpassFilterType": 3,
                         "fPostProcessLowpassFilter": 1000,
                         "nPostProcessLowpassFilterType": b"B"})
    assert result.status == "reported"
    assert result.cutoff_hz == 3000
    assert result.filter_type == ""  # Bessel describes signal conditioner, NOT amplifier.
    assert result.filter_order is None
    assert dict(result.header_fields)["fSignalLowpassFilter"] == "5000"
    assert dict(result.header_fields)["fPostProcessLowpassFilter"] == "1000"
    assert "not the combined effective response" in result.note
    assert "no rise time is inferred" in result.note


def test_bypassed_signal_stage_does_not_hide_a_reported_amplifier_cutoff():
    result = _bandwidth({"fTelegraphFilter": 3000, "nTelegraphEnable": 1,
                         "fSignalLowpassFilter": 100000})
    assert result.status == "reported"
    assert result.cutoff_hz == 3000


def test_postprocessing_cutoff_is_not_misclassified_as_hardware():
    result = _bandwidth({"fPostProcessLowpassFilter": 1000})
    assert result.cutoff_hz is None
    assert result.status == "unknown"
    assert "not treated as hardware bandwidth" in result.note


def test_no_filter_type_conflicting_with_positive_signal_cutoff_is_unknown():
    result = _bandwidth({"fSignalLowpassFilter": 5000, "nLowpassFilterType": 0})
    assert result.cutoff_hz is None
    assert result.status == "unknown"
    assert "conflicts" in result.note


def test_missing_telegraph_enable_does_not_activate_stale_telegraph_value():
    assert _bandwidth({"fTelegraphFilter": 2000}).cutoff_hz is None


@pytest.mark.parametrize("sample_range", [None, SampleRange(10, 190)])
def test_probe_load_and_partial_load_keep_metadata_and_exact_samples(tmp_path, monkeypatch,
                                                                    sample_range):
    path = tmp_path / "acquisition.abf"
    path.write_bytes(b"ABF2")
    raw = np.column_stack([np.linspace(0, 10, 256), np.linspace(10, 30, 256)]).astype(np.float32)
    reader = _reader({
        "fFileVersionNumber": 2,
        "listADCInfo": [{"nADCNum": 15, "fSignalLowpassFilter": 5000},
                        {"nADCNum": 2, "fSignalLowpassFilter": 2000}],
    }, ids=("0", "1"))
    reader.get_signal_size = lambda **kwargs: len(raw)
    reader.get_signal_sampling_rate = lambda: 50000
    reader.get_signal_t_start = lambda *args: 0.
    reader.block_count = lambda: 1
    reader.segment_count = lambda *args: 1
    reader.get_analogsignal_chunk = lambda **kwargs: raw[kwargs["i_start"]:kwargs["i_stop"]]
    reader.rescale_signal_raw_to_float = lambda data, **kwargs: data
    monkeypatch.setattr(AbfReader, "_open_reader", lambda self: reader)
    abf = AbfReader(path)
    probe = abf.probe()
    doc = abf.load(sample_range)
    interval = slice(None) if sample_range is None else slice(sample_range.start, sample_range.stop)
    for index in range(2):
        assert doc.meta.channels[index].acquisition_bandwidth == probe.meta.channels[index].acquisition_bandwidth
        np.testing.assert_array_equal(doc.channels[index], raw[interval, index])
