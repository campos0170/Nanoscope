"""Third-sweep loader defects, part 1: ONE CLOCK, and it is the file's own.

Every test here is an instance of a single rule the previous two sweeps fixed one reader at a
time, so the sweep after each fix found the next reader with the old answer:

    **The experimental time a document carries is read from the file wherever the file
    states it, computed only where it does not, and never reported as measured when the
    measurement cannot be believed.**

  IO3-01  TDMS ``wf_start_offset`` was still read from the first timing channel only, after
          sweep 2 had made ``wf_increment`` per-channel. An offset declared on any later
          channel was dropped (t0 = 0.0 for a file that says 12.5 s) and two channels
          declaring different offsets collapsed to whichever was written first — both in
          complete silence, while the identical disagreement about the INCREMENT is refused
          loudly with every channel's rate named.

  IO3-02  A mid-file CSV window reported ``file_t0 + start/fs``, extrapolated from a rate
          measured over the first 1,000 rows, while the true value sat in the file's own time
          column at that row. On a 10 kHz CSV with a 5 s acquisition pause the window's clock
          was 5 s early, ``issues=()``; the byte-identical time base written as HDF5 reported
          the right value beside it, because HDF5 reads ``/time`` at the window's own index.

  IO3-05  An HDF5 ``/time`` stored as int64 MICROSECONDS was read as seconds: 0.01 Hz stamped
          ``Provenance.MEASURED`` for a 10 kHz recording, duration 500,000 s, Issues box empty
          — against a guide that promises "a file that shows nothing there was read clean".
          Every uniformity and span check passes trivially on an integer clock, so the file
          that is most wrong is the one that looks cleanest.

The "no false positive" tests matter as much: a well-formed recording of any of the three
formats must gain no issue and no note it did not have before.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.core.types import Provenance, SampleRange
from nanoscope.io.base import open_document, probe_path
from nanoscope.io.hdf5 import Hdf5Reader
from nanoscope.io.tdms import TdmsReader
from nanoscope.io.text import TextTraceReader

pytest.importorskip("nptdms")


def _text(strings) -> str:
    return " || ".join(strings)


def _write_tdms(path: Path, per_channel_props: list[dict], n: int = 2_000) -> Path:
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    data = np.linspace(0.0, 1.0, n)
    objs = [RootObject(properties={}), GroupObject("Untitled", properties={})]
    for i, props in enumerate(per_channel_props):
        objs.append(ChannelObject("Untitled", f"ai{i}", data, properties=dict(props)))
    with TdmsWriter(str(path)) as w:
        w.write_segment(objs)
    return path


def _write_h5(path: Path, **datasets) -> Path:
    with h5py.File(path, "w") as h:
        for name, value in datasets.items():
            h[name] = value
    return path


def _write_csv(path: Path, times, values) -> Path:
    rows = ["time,current"]
    rows += [f"{t!r},{v!r}" for t, v in zip(times, values)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _paused_times(n: int = 6_000, dt: float = 1e-4, *, pause_at: int = 3_000,
                  pause_s: float = 5.0) -> list[float]:
    """A 10 kHz time base whose acquisition stopped for 5 s in the middle and resumed.

    The whole point of the fixture: it is uniform on both sides of the pause, so nothing about
    it can be recovered from a rate measured at the top of the file. Only the column itself
    knows what time row 3,500 carries.
    """
    out = []
    for i in range(n):
        out.append(i * dt + (pause_s if i >= pause_at else 0.0))
    return out


# =================================================================================================
# IO3-01: a TDMS start offset is every channel's testimony, exactly as the increment is
# =================================================================================================

def test_a_start_offset_declared_on_a_later_channel_is_honoured(tmp_path):
    """Was: t0_s = 0.0, issues=(), notes=() — the whole recording placed 12.5 s early because
    the offset sat on the second channel and only ``timing[0]`` was consulted."""
    p = _write_tdms(tmp_path / "later_offset.tdms", [
        {"wf_increment": 2e-5, "unit_string": "Volts"},
        {"wf_increment": 2e-5, "wf_start_offset": 12.5, "unit_string": "Volts"},
    ])
    meta = probe_path(p).meta
    assert meta.t0_s == pytest.approx(12.5), (
        f"an offset declared anywhere in the file must be honoured, got {meta.t0_s}")
    assert meta.issues == (), "one channel declaring the offset is not a disagreement"
    assert open_document(p).meta.t0_s == pytest.approx(12.5), "the load door must agree"


def test_channels_declaring_different_start_offsets_are_named(tmp_path):
    """The mirror of test_a_mixed_rate_tdms_refuses_a_single_clock_and_names_both_rates: the
    same disagreement about the other half of the same clock was silent."""
    p = _write_tdms(tmp_path / "mixed_offset.tdms", [
        {"wf_increment": 2e-5, "wf_start_offset": 0.0, "unit_string": "Volts"},
        {"wf_increment": 2e-5, "wf_start_offset": 30.0, "unit_string": "Volts"},
    ])
    meta = probe_path(p).meta
    said = _text(meta.issues)
    assert "different start offsets" in said, f"the conflict was not named: {meta.issues!r}"
    assert "ai0" in said and "ai1" in said, "the issue must name the disagreeing channels"
    assert "30" in said, "the issue must name what the other channel declared"
    assert str(meta.t0_s) in said or f"{meta.t0_s:g}" in said, (
        "the issue must say which declaration the recording was stamped with")


def test_the_disagreement_is_reported_whichever_channel_is_written_first(tmp_path):
    """The defect was also order-dependent: the same two channels gave t0 = 0.0 or 30.0
    depending purely on which nptdms wrote first, and said nothing either way."""
    swapped = _write_tdms(tmp_path / "swapped.tdms", [
        {"wf_increment": 2e-5, "wf_start_offset": 30.0, "unit_string": "Volts"},
        {"wf_increment": 2e-5, "wf_start_offset": 0.0, "unit_string": "Volts"},
    ])
    said = _text(probe_path(swapped).meta.issues)
    assert "different start offsets" in said, (
        "file order must not decide whether the user is told")


def test_agreeing_start_offsets_gain_no_issue(tmp_path):
    """No false positive: the real corpus file declares wf_start_offset = 0.0 identically on
    Dev2/ai0 and Dev2/ai1, which is what a stock NI-DAQmx acquisition writes."""
    p = _write_tdms(tmp_path / "agree.tdms",
                    [{"wf_increment": 2e-5, "wf_start_offset": 0.0,
                      "unit_string": "Volts"}] * 2)
    meta = probe_path(p).meta
    assert meta.t0_s == 0.0
    assert meta.issues == ()


def test_a_legacy_tdms_that_declares_only_a_start_offset_still_carries_it(tmp_path):
    """The generalisation, not the repro: the offset is now read by ONE rule for all three
    TDMS layouts, so a file whose rate is unknown still gets placed where it says it is
    instead of being silently stamped 0.0."""
    p = _write_tdms(tmp_path / "legacy_offset.tdms", [
        {"wf_start_offset": 7.5, "unit_string": "Volts"},
        {"wf_start_offset": 7.5, "unit_string": "Volts"},
    ])
    meta = probe_path(p).meta
    assert meta.sample_rate_hz is None, "no increment is declared: the rate stays unknown"
    assert meta.t0_s == pytest.approx(7.5), (
        "an unknown rate is no reason to discard a declared experimental time")


def test_a_windowed_tdms_still_gets_the_windows_own_clock(tmp_path):
    """The boundary the offset fix must not move: window_t0 still adds start/fs on top."""
    p = _write_tdms(tmp_path / "win.tdms", [
        {"wf_increment": 2e-5, "unit_string": "Volts"},
        {"wf_increment": 2e-5, "wf_start_offset": 12.5, "unit_string": "Volts"},
    ])
    doc = TdmsReader(p).load(SampleRange(1_000, 1_500))
    assert doc.meta.t0_s == pytest.approx(12.5 + 1_000 * 2e-5)


# =================================================================================================
# IO3-02: a CSV window's clock is READ from the time column, like HDF5's is
# =================================================================================================

def test_a_csv_window_reports_the_time_its_own_column_gives_that_row(tmp_path):
    """Was: t0_s = 0.350000 s where the file's column 0 says 5.350000 s, issues=()."""
    times = _paused_times()
    values = list(np.arange(len(times), dtype=float))
    p = _write_csv(tmp_path / "paused.csv", times, values)

    start, stop = 3_500, 4_500
    doc = TextTraceReader(p).load(SampleRange(start, stop))
    assert doc.meta.t0_s == pytest.approx(times[start], abs=1e-9), (
        f"the window's clock must be the file's own value at row {start:,}")
    assert doc.channels[0][0] == pytest.approx(values[start]), (
        "and it must be the clock of the row that was actually loaded")


def test_the_csv_and_hdf5_windows_of_one_time_base_carry_one_clock(tmp_path):
    """The defect was the DISAGREEMENT: the same experiment saved in two formats put two
    clocks seconds apart in front of the user, and the CSV leg was the silent one."""
    times = _paused_times()
    values = list(np.arange(len(times), dtype=float))
    csv = _write_csv(tmp_path / "twin.csv", times, values)
    h5 = _write_h5(tmp_path / "twin.h5", current=np.asarray(values),
                   voltage=np.full(len(times), -0.6), time=np.asarray(times))

    window = SampleRange(3_500, 4_500)
    csv_t0 = TextTraceReader(csv).load(window).meta.t0_s
    h5_t0 = Hdf5Reader(h5).load(window).meta.t0_s
    assert csv_t0 == pytest.approx(h5_t0, abs=1e-9), (
        f"two formats of one recording disagree about the window's clock: {csv_t0} / {h5_t0}")


def test_a_uniform_csv_window_is_exactly_what_it_always_was(tmp_path):
    """No false positive, and the boundary of the fix: on a file whose interval never changes
    the read value and the computed one are the same number, which is why the defect could
    hide behind every uniform fixture in the suite."""
    n, dt, t0 = 20_000, 2e-5, 5.0
    times = [t0 + i * dt for i in range(n)]
    p = _write_csv(tmp_path / "uniform.csv", times, np.zeros(n))
    doc = TextTraceReader(p).load(SampleRange(10_000, 15_000))
    assert doc.meta.t0_s == pytest.approx(t0 + 10_000 * dt, abs=1e-9)
    assert doc.meta.issues == ()
    assert TextTraceReader(p).load().meta.t0_s == pytest.approx(t0, abs=1e-12), (
        "a whole-file load must be bit-identical to what it always was")


# =================================================================================================
# IO3-05: a rate MEASURED from a time column whose units are undeclared must be believable
# =================================================================================================

def test_an_integer_microsecond_hdf5_time_base_refuses_a_rate_instead_of_reporting_0_01_hz(
        tmp_path):
    """Was: fs = 0.01 Hz, Provenance.MEASURED, duration 500,000 s for 0.5 s of samples, with
    issues=() — 10^6x wrong and certified clean by the one box the user is told to read."""
    n = 5_000
    p = _write_h5(tmp_path / "int_us.h5",
                  current=np.zeros(n), voltage=np.full(n, -0.6),
                  time=(np.arange(n) * 100).astype(np.int64))     # microseconds
    meta = probe_path(p).meta
    assert meta.sample_rate_hz is None, (
        f"0.01 Hz is not a rate this app can report as measured: {meta.sample_rate_hz}")
    assert meta.rate_provenance is Provenance.UNKNOWN
    assert meta.duration_s is None
    said = _text(meta.issues)
    assert "less than one sample per second" in said, f"nothing was said: {meta.issues!r}"
    assert "int64" in said, "the integer dtype is part of the evidence and must be named"
    assert "will not guess" in said, "the reader must not be read as having converted anything"


def test_the_same_rule_covers_a_csv_time_column_in_microseconds(tmp_path):
    """The generalisation the repro did not ask for: the text reader measures its rate from a
    time column whose units are declared exactly as nowhere as HDF5's, and reported 0.01 Hz
    for the same recording with the same silence."""
    n = 5_000
    p = _write_csv(tmp_path / "int_us.csv", [float(i * 100) for i in range(n)], np.zeros(n))
    meta = probe_path(p).meta
    assert meta.sample_rate_hz is None
    assert meta.rate_provenance is Provenance.UNKNOWN
    assert "less than one sample per second" in _text(meta.issues), meta.issues


def test_an_ordinary_recording_still_reports_its_measured_rate(tmp_path):
    """The no-false-positive half, in both formats: 50 kHz is nowhere near the floor and must
    gain nothing at all."""
    n, dt = 4_000, 2e-5
    h5 = _write_h5(tmp_path / "ok.h5", current=np.zeros(n), voltage=np.full(n, -0.6),
                   time=np.arange(n) * dt)
    csv = _write_csv(tmp_path / "ok.csv", [i * dt for i in range(n)], np.zeros(n))
    for label, path in (("hdf5", h5), ("csv", csv)):
        meta = probe_path(path).meta
        assert meta.sample_rate_hz == pytest.approx(50_000.0), label
        assert meta.rate_provenance is Provenance.MEASURED, label
        assert meta.issues == (), f"{label} gained an issue it should not have: {meta.issues!r}"


def test_an_integer_time_base_that_could_be_seconds_is_read_and_says_it_is_integer(tmp_path):
    """The boundary between the two halves of the fix. An integer clock at 1 Hz is not
    IMPLAUSIBLE the way a microsecond one is, so it is read — but the dtype is still a fact
    about the file, and this reader already says it for an integer SIGNAL dataset
    ("/current is stored as int32 — raw ADC counts"). The mirror sentence was missing."""
    n = 2_000
    p = _write_h5(tmp_path / "int_s.h5", current=np.zeros(n), voltage=np.full(n, -0.6),
                  time=np.arange(n).astype(np.int64))             # 1 s per sample
    meta = probe_path(p).meta
    assert meta.sample_rate_hz == pytest.approx(1.0)
    said = _text(meta.notes)
    assert "integer time base" in said, f"the integer clock was not mentioned: {meta.notes!r}"
    assert "microseconds" in said, "the note must name the units that would make it wrong"
