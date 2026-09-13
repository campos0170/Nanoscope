"""Adversarial clocks and channel layouts from the beta sweep, using synthetic files."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from nptdms import ChannelObject, TdmsWriter

from nanoscope.core.types import Provenance, SampleRange
from nanoscope.io.base import open_document, probe_path
from nanoscope.io.errors import LoadCancelled, ReaderError


def _recording(path, times, values=None):
    values = np.arange(len(times), dtype=np.float32) if values is None else values
    if path.suffix == ".h5":
        with h5py.File(path, "w") as h:
            h["time"] = times
            h["current"] = values
    else:
        with path.open("w") as f:
            f.write("time,current\n")
            for t, v in zip(times, values):
                f.write(f"{t:.17g},{v:.9g}\n")
    return path


@pytest.mark.parametrize("suffix", [".csv", ".h5"])
@pytest.mark.parametrize("damage", ["nan", "infinite", "backwards", "compensated"])
def test_middle_clock_damage_cannot_drive_seconds_based_analysis(tmp_path, suffix, damage):
    n, fs = 10_000, 50_000.0
    times = np.arange(n) / fs
    if damage == "compensated":
        steps = np.full(n - 1, 1 / fs)
        steps[2000:2200] *= 2
        steps[2200:2600] *= 0.5
        times = np.r_[0, np.cumsum(steps)]
        # Endpoints, head, and tail agree; the source's local duration is twice
        # the duration the previous loader would have assigned to these samples.
        assert times[-1] == pytest.approx((n - 1) / fs)
        assert times[2200] - times[2000] == pytest.approx(0.008)
        assert 200 / fs == 0.004
    else:
        times[3000] = {"nan": np.nan, "infinite": np.inf,
                       "backwards": times[2999] - 1}[damage]
    doc = open_document(_recording(tmp_path / (damage + suffix), times))
    np.testing.assert_array_equal(doc.channels[0], np.arange(n, dtype=np.float32))
    assert doc.meta.sample_rate_hz is None
    assert doc.meta.rate_provenance is Provenance.UNKNOWN
    assert np.isfinite(doc.meta.t0_s)
    assert any("loaded samples" in issue and "Sample rate is unknown" in issue
               for issue in doc.meta.issues)


@pytest.mark.parametrize("suffix", [".csv", ".h5"])
@pytest.mark.parametrize("damage", ["nan", "backwards", "gap"])
def test_clock_check_spans_chunk_boundaries(tmp_path, monkeypatch, suffix, damage):
    from nanoscope.io import hdf5, text
    monkeypatch.setattr(hdf5 if suffix == ".h5" else text, "LOAD_CHUNK_SAMPLES", 2048)
    times = np.arange(6000) / 50_000
    if damage == "nan":
        times[4096] = np.nan
    elif damage == "backwards":
        times[4096] = times[4095] - 1
    else:
        times[4096:] += 0.001
    doc = open_document(_recording(tmp_path / (damage + suffix), times))
    assert doc.meta.sample_rate_hz is None
    assert any("4,096" in issue for issue in doc.meta.issues)


@pytest.mark.parametrize("suffix", [".csv", ".h5"])
def test_nan_at_window_start_has_finite_origin_and_unknown_rate(tmp_path, suffix):
    times = np.arange(10_000) / 50_000
    times[5000] = np.nan
    doc = open_document(_recording(tmp_path / ("window" + suffix), times),
                        SampleRange(5000, 6000))
    assert doc.meta.loaded_range == SampleRange(5000, 6000)
    assert doc.meta.sample_rate_hz is None
    assert np.isfinite(doc.meta.t0_s)
    assert doc.meta.issues
    np.testing.assert_array_equal(doc.channels[0], np.arange(5000, 6000, dtype=np.float32))


@pytest.mark.parametrize("suffix", [".csv", ".h5"])
@pytest.mark.parametrize("window", [None, SampleRange(3000, 5000)])
def test_uniform_loaded_clock_and_values_are_unchanged(tmp_path, suffix, window):
    times = 60.00002 + np.arange(10_000) / 50_000
    doc = open_document(_recording(tmp_path / ("uniform" + suffix), times), window)
    start, stop = (0, len(times)) if window is None else (window.start, window.stop)
    assert doc.meta.sample_rate_hz == pytest.approx(50_000)
    assert doc.meta.rate_provenance is Provenance.MEASURED
    assert doc.meta.t0_s == pytest.approx(times[start])
    assert doc.meta.issues == ()
    np.testing.assert_array_equal(doc.channels[0], np.arange(start, stop, dtype=np.float32))


@pytest.mark.parametrize("suffix", [".csv", ".h5"])
def test_clean_window_after_acquisition_pause_keeps_its_true_clock(tmp_path, suffix):
    times = np.arange(10_000) / 50_000
    times[2500:] += 5
    doc = open_document(_recording(tmp_path / ("paused" + suffix), times),
                        SampleRange(3000, 5000))
    assert doc.meta.sample_rate_hz == pytest.approx(50_000)
    assert doc.meta.t0_s == pytest.approx(times[3000])
    np.testing.assert_array_equal(doc.channels[0], np.arange(3000, 5000, dtype=np.float32))


@pytest.mark.parametrize("index", [500, 3000, 4096])
@pytest.mark.parametrize("window", [False, True])
def test_extra_csv_fields_are_refused_in_preview_late_or_first_window_row(
        tmp_path, monkeypatch, index, window):
    from nanoscope.io import text
    monkeypatch.setattr(text, "LOAD_CHUNK_SAMPLES", 2048)
    path = _recording(tmp_path / "ragged.csv", np.arange(6000) / 50_000)
    rows = path.read_text().splitlines()
    rows[index + 1] += ",999999"
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ReaderError, match="ragged|fields"):
        open_document(path, SampleRange(index, index + 1000) if window else None)


def test_missing_late_csv_signal_is_refused(tmp_path):
    path = _recording(tmp_path / "short_row.csv", np.arange(6000) / 50_000)
    rows = path.read_text().splitlines()
    rows[3001] = rows[3001].split(",")[0]
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ReaderError, match="ragged table"):
        open_document(path)


@pytest.mark.parametrize("values", [np.full(5000, np.nan), -np.arange(5000) / 50_000])
@pytest.mark.parametrize("call", [probe_path, open_document])
def test_unusable_hdf5_clock_never_exposes_nonfinite_origin(tmp_path, values, call):
    result = call(_recording(tmp_path / "bad_time.h5", values))
    assert result.meta.sample_rate_hz is None
    assert np.isfinite(result.meta.t0_s)
    assert result.meta.issues


@pytest.mark.parametrize("link_kind", ["soft", "external"])
@pytest.mark.parametrize("member", ["current", "time"])
@pytest.mark.parametrize("call", [probe_path, open_document])
def test_dangling_hdf5_member_is_named_in_issues(tmp_path, link_kind, member, call):
    path = _recording(tmp_path / "dangling.h5", np.arange(2000) / 50_000)
    with h5py.File(path, "a") as h:
        h["voltage"] = np.ones(2000)
        del h[member]
        h[member] = (h5py.SoftLink("/missing") if link_kind == "soft"
                     else h5py.ExternalLink(str(tmp_path / "missing.h5"), "/signal"))
    meta = call(path).meta
    assert meta.n_samples == 2000
    assert any(f"/{member}" in issue and "target" in issue for issue in meta.issues)
    if member == "current":
        assert [c.name for c in meta.channels] == ["voltage"]
    else:
        assert meta.sample_rate_hz is None


def test_resolving_hdf5_link_stays_silent(tmp_path):
    path = _recording(tmp_path / "linked.h5", np.arange(2000) / 50_000)
    with h5py.File(path, "a") as h:
        h.move("current", "raw")
        h["current"] = h5py.SoftLink("/raw")
    doc = open_document(path)
    assert doc.meta.issues == ()
    np.testing.assert_array_equal(doc.channels[0], np.arange(2000, dtype=np.float32))


def test_lone_signal_with_dangling_clock_is_not_called_a_clockless_export(tmp_path):
    path = _recording(tmp_path / "dangling_time.h5", np.arange(2000) / 50_000)
    with h5py.File(path, "a") as h:
        del h["time"]
        h["time"] = h5py.SoftLink("/missing")
    with pytest.raises(ReaderError, match="/time present but unusable"):
        open_document(path)


@pytest.mark.parametrize("extra", [np.array(["start", "note", "stop"]),
                                   np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[us]")])
@pytest.mark.parametrize("call", [probe_path, open_document])
def test_tdms_nonnumeric_channel_cannot_truncate_or_poison_waveform(tmp_path, extra, call):
    path = tmp_path / "mixed.tdms"
    values = np.arange(20_000, dtype=np.float32)
    with TdmsWriter(path) as writer:
        writer.write_segment([
            ChannelObject("Data", "current", values, properties={"wf_increment": 2e-5}),
            ChannelObject("Data", "log", extra, properties={"wf_increment": -1}),
        ])
    result = call(path)
    assert result.meta.n_samples == len(values)
    assert result.meta.sample_rate_hz == pytest.approx(50_000)
    assert [c.name for c in result.meta.channels] == ["Data/current"]
    assert any("Data/log" in issue and "dtype" in issue for issue in result.meta.issues)
    if call is open_document:
        np.testing.assert_array_equal(result.channels[0], values)


@pytest.mark.parametrize("extra", [np.array(["1", "2", "3"]),
                                   np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[us]")])
@pytest.mark.parametrize("call", [probe_path, open_document])
def test_tdms_without_real_numeric_signals_is_refused(tmp_path, extra, call):
    path = tmp_path / "not_signals.tdms"
    with TdmsWriter(path) as writer:
        writer.write_segment([ChannelObject("Data", "log", extra)])
    with pytest.raises(ReaderError, match="no readable numeric signal channels"):
        call(path)


def test_hdf5_clock_validation_can_be_cancelled(tmp_path, monkeypatch):
    from nanoscope.io import hdf5
    monkeypatch.setattr(hdf5, "LOAD_CHUNK_SAMPLES", 2048)
    path = _recording(tmp_path / "cancel.h5", np.arange(10_000) / 50_000)
    progress = []
    with pytest.raises(LoadCancelled):
        open_document(path, progress=lambda p, message: progress.append((p, message)),
                      cancel=lambda: len(progress) >= 2)
    assert len(progress) == 2
    assert all("/time" in message for _, message in progress)


@pytest.mark.parametrize("index", [10, 3000])
def test_quoted_thousands_value_is_refused_with_numeric_field_guidance(tmp_path, index):
    path = _recording(tmp_path / "grouped.csv", np.arange(5000) / 50_000)
    rows = path.read_text().splitlines()
    rows[index + 1] = f'{index / 50_000},"-500,000.00"'
    path.write_text("\n".join(rows) + "\n")
    probe_path(path)  # Structural probing does not claim to parse every signal value.
    with pytest.raises(ReaderError) as exc:
        open_document(path)
    said = str(exc.value)
    assert "time or signal column" in said and "without thousands separators" in said
    assert "could not convert string to float" not in said


def test_quoted_ungrouped_numeric_values_and_terminal_progress_are_preserved(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text("time,current\n" + "".join(
        f'"{i / 50_000}","{-500000 + i}.00"\n' for i in range(3000)))
    progress = []
    doc = open_document(path, progress=lambda value, _message: progress.append(value))
    np.testing.assert_array_equal(doc.channels[0], np.arange(3000, dtype=np.float32) - 500000)
    assert doc.meta.issues == ()
    assert progress[-1] == 1.0
    assert np.all(np.diff(progress) >= 0)


@pytest.mark.parametrize("call", [probe_path, open_document])
def test_zero_byte_recordings_have_one_honest_refusal_for_every_format(tmp_path, call):
    from nanoscope.io.errors import CorruptFileError
    messages = []
    for suffix in (".abf", ".csv", ".h5", ".tdms"):
        path = tmp_path / ("empty" + suffix)
        path.write_bytes(b"")
        with pytest.raises(CorruptFileError) as exc:
            call(path)
        said = str(exc.value)
        assert "0 bytes" in said and "recover the original" in said
        assert "can be deleted" not in said
        assert "file signature" not in said and "no channels" not in said
        messages.append(said.replace(path.name, "FILE"))
    assert len(set(messages)) == 1


@pytest.mark.parametrize("suffix", [".h5", ".csv"])
@pytest.mark.parametrize("damaged", [False, True])
def test_epoch_clock_rounding_does_not_hide_ten_percent_interval_changes(tmp_path, suffix, damaged):
    n, fs = 10_000, 50_000.0
    relative = np.arange(n) / fs
    if damaged:
        steps = np.full(n - 1, 1 / fs)
        steps[2000:2200] *= 1.1
        steps[2200:2400] *= 0.9
        relative = np.r_[0, np.cumsum(steps)]
    times = 1_700_000_000.0 + relative
    doc = open_document(_recording(tmp_path / ("epoch" + suffix), times))
    if damaged:
        assert times[2200] - times[2000] == pytest.approx(0.0044, abs=5e-7)
        assert doc.meta.sample_rate_hz is None
        assert any("non-uniform interval" in issue for issue in doc.meta.issues)
    else:
        # The existing CSV median estimator is quantized at this large origin;
        # the audit must tolerate real representational rounding, not a 10% change.
        assert doc.meta.sample_rate_hz == pytest.approx(fs, rel=.002)
        assert not any("in the loaded samples" in issue for issue in doc.meta.issues)


@pytest.mark.parametrize("damaged", [False, True])
def test_uncovered_hdf5_window_requires_valid_extrapolation_clock(tmp_path, damaged):
    steps = np.full(500, 2e-5)
    if damaged:
        steps[1::2] *= 2
    times = np.r_[0, np.cumsum(steps)]
    path = _recording(tmp_path / "partial.h5", times, np.arange(3000, dtype=np.float32))
    doc = open_document(path, SampleRange(1000, 2000))
    np.testing.assert_array_equal(doc.channels[0], np.arange(1000, 2000, dtype=np.float32))
    if damaged:
        assert doc.meta.sample_rate_hz is None
        assert doc.meta.t0_s == 0
        assert any("extrapolate" in issue for issue in doc.meta.issues)
    else:
        assert doc.meta.sample_rate_hz == pytest.approx(50_000)
        assert doc.meta.t0_s == pytest.approx(.02)


@pytest.mark.parametrize("damaged", [False, True])
def test_float32_hdf5_clock_keeps_source_rounding_precision_across_chunks(
        tmp_path, monkeypatch, damaged):
    from nanoscope.io import hdf5
    monkeypatch.setattr(hdf5, "LOAD_CHUNK_SAMPLES", 12506)
    n, fs = 100_000, 50_000
    times = np.arange(n, dtype=np.float32) / np.float32(fs)
    if damaged:
        steps = np.full(n - 1, 1 / fs)
        steps[20000:20200] *= 1.1
        steps[20200:20400] *= .9
        times = np.r_[0, np.cumsum(steps)].astype(np.float32)
    doc = open_document(_recording(tmp_path / "float32.h5", times))
    if damaged:
        assert doc.meta.sample_rate_hz is None
        assert any("non-uniform interval" in issue for issue in doc.meta.issues)
    else:
        assert doc.meta.sample_rate_hz == pytest.approx(fs)
        assert not any("in the loaded samples" in issue for issue in doc.meta.issues)


def test_quoted_empty_late_csv_field_is_a_record_not_a_blank_line(tmp_path):
    path = _recording(tmp_path / "empty_field.csv", np.arange(5000) / 50_000)
    rows = path.read_text().splitlines()
    rows[3001] = '""'
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ReaderError, match="ragged table"):
        open_document(path)


def test_actual_blank_csv_lines_do_not_change_loaded_row_indices(tmp_path):
    path = _recording(tmp_path / "blank_lines.csv", np.arange(5000) / 50_000)
    rows = path.read_text().splitlines()
    rows.insert(3001, "   ")
    rows.insert(3501, "")
    path.write_text("\n".join(rows) + "\n")
    doc = open_document(path, SampleRange(2500, 4000))
    assert doc.meta.issues == ()
    np.testing.assert_array_equal(doc.channels[0], np.arange(2500, 4000, dtype=np.float32))
