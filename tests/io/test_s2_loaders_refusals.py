"""Second-sweep loader defects: every refusal names its real cause, no format disagrees.

Six measured defects, all of the same family the first sweep worked in — a reader answering
honestly for one format and wrongly (or silently) for another:

  LDR2-01  an ABF whose header lies "zero samples" over a 9 MB data section was refused with
           the empty-recording sentence, whose tail is deletion advice — for a recording the
           verifier proved recoverable by restoring 8 header bytes.
  LDR2-02  a vanished / permission-locked / directory .h5 or .tdms leaked raw h5py/nptdms
           errno strings into the "Could not open file" dialog wrapped in corruption wording,
           while the identical inputs as .abf/.csv got classified sentences.
  LDR2-03  a /time that EXISTS but is structurally unusable (string dtype, 2-D, one sample)
           was dropped silently: rate "unknown" with issues=(), indistinguishable from a file
           that never carried a time base.
  LDR2-04  a windowed HDF5 load starting past the end of a partial /time carried the FILE's
           t0, putting every experimental time start/fs seconds early.
  LDR2-05  TDMS timing was read from the first channel only: mixed-rate files got one silent
           clock, and a bare first channel produced the false note "declares no sample rate".
  LDR2-06  a zero-overlap saved window on an intact-but-shorter CSV was refused as "no data
           rows were read" — corruption wording — where the other three readers say the
           request "selects nothing in an N-sample recording".
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.core.types import Provenance, SampleRange
from nanoscope.io.base import open_document, probe_path
from nanoscope.io.errors import (
    CorruptFileError,
    EmptyRecordingError,
    ReaderError,
    UnsupportedFormatError,
)

pytest.importorskip("nptdms")

# The real quick ABF: the only way to build an ABF that DECODES with a lying header is to
# start from one that decodes. Tests that need it skip honestly when it is not on this machine.
_QUICK_ABF = Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0004_process _ "
    "-1000 mV_ 385-400 s time trace.abf")
_ABF_DATA_START = 12 * 512      # lDataSectionPtr = 12 blocks of 512 bytes in this file

needs_quick_abf = pytest.mark.skipif(
    not _QUICK_ABF.exists(), reason=f"real corpus file not present: {_QUICK_ABF.name}")


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


# ================================================================================================
# LDR2-01: a zero-sample header over a data section is damage, not an empty recording
# ================================================================================================

def _crashed_abf(tmp_path: Path) -> Path:
    """The state an acquisition crash leaves: header counts initialised to zero, data section
    flushed. Built from the real quick ABF by zeroing the two header ints the crash never got
    to write (lActualAcqLength at byte 10, lSynchArraySize at byte 96)."""
    liar = tmp_path / "crashed_acquisition.abf"
    shutil.copyfile(_QUICK_ABF, liar)
    with liar.open("r+b") as fh:
        fh.seek(10)
        fh.write(struct.pack("<i", 0))
        fh.seek(96)
        fh.write(struct.pack("<i", 0))
    return liar


@needs_quick_abf
@pytest.mark.parametrize("door", [probe_path, open_document], ids=["probe", "load"])
def test_a_zero_sample_header_over_megabytes_of_data_is_never_called_deletable(tmp_path, door):
    """Was: "…this file is the expected result and can be deleted" for 9 MB of recoverable
    samples. The TDMS reader already reroutes its analogous case ("a cut recording is neither
    expected nor safe to delete"); this pins the same reroute for ABF, at both doors."""
    liar = _crashed_abf(tmp_path)
    with pytest.raises(CorruptFileError) as exc:
        door(liar)
    message = str(exc.value)
    assert "can be deleted" not in message
    assert "do not delete" in message
    assert "holds no samples" in message, "the refusal must state what the header claims"
    expected_bytes = liar.stat().st_size - _ABF_DATA_START
    assert f"{expected_bytes:,}" in message, "the refusal must quote the data the file holds"
    assert "recover" in message


@needs_quick_abf
def test_a_genuinely_header_only_abf_keeps_the_honest_empty_sentence(tmp_path):
    """The other side of the reroute: an acquisition armed and stopped without recording
    writes a bare header and nothing after it. THAT file really is the expected result, and
    the empty-recording sentence — deletion advice included — stays its answer."""
    tiny = tmp_path / "armed_and_stopped.abf"
    blob = bytearray(_QUICK_ABF.read_bytes()[:_ABF_DATA_START])
    struct.pack_into("<i", blob, 10, 0)
    struct.pack_into("<i", blob, 96, 0)
    tiny.write_bytes(bytes(blob))
    with pytest.raises(EmptyRecordingError) as exc:
        probe_path(tiny)
    assert "can be deleted" in str(exc.value)


# ================================================================================================
# LDR2-02: gone / locked / a-folder .h5 and .tdms are classified, never leaked
# ================================================================================================

def _valid_recording(tmp_path: Path, suffix: str) -> Path:
    n = 2_000
    if suffix == ".h5":
        return _write_h5(tmp_path / f"v{suffix}", current=np.zeros(n),
                         voltage=np.full(n, -0.6), time=np.arange(n) * 2e-5)
    return _write_tdms(tmp_path / f"v{suffix}",
                       [{"wf_increment": 2e-5, "unit_string": "Volts"}] * 2, n)


@pytest.mark.parametrize("suffix", [".h5", ".tdms"])
@pytest.mark.parametrize("door", [probe_path, open_document], ids=["probe", "load"])
def test_a_vanished_recording_is_called_gone_not_unreadable(tmp_path, suffix, door):
    """The race the GUI actually runs: ProbeWorker finishes, the file vanishes (unmounted
    share, synced-folder deletion), LoadWorker starts. Was: raw h5py/nptdms "[Errno 2] …"
    internals wrapped in "not a readable HDF5/TDMS file" — corruption wording for a file
    that is merely gone."""
    path = _valid_recording(tmp_path, suffix)
    path.unlink()
    with pytest.raises(CorruptFileError) as exc:
        door(path)
    message = str(exc.value)
    assert "moved, renamed or deleted" in message, message
    assert "[Errno" not in message
    assert str(tmp_path) not in message, "no absolute paths in the dialog"
    assert "not a readable" not in message, "a gone file is not an unreadable-format file"


@pytest.mark.parametrize("suffix", [".h5", ".tdms"])
def test_a_directory_named_like_a_recording_is_a_folder(tmp_path, suffix):
    folder = tmp_path / f"dir{suffix}"
    folder.mkdir()
    with pytest.raises(CorruptFileError) as exc:
        probe_path(folder)
    message = str(exc.value)
    assert "is a folder, not a file" in message, message
    assert "[Errno" not in message
    assert "bytes actually read" not in message, "h5py's internal failure log leaked"


@pytest.mark.parametrize("suffix", [".h5", ".tdms"])
def test_a_permission_locked_recording_says_permission_denied(tmp_path, suffix):
    if os.geteuid() == 0:  # pragma: no cover - root reads through chmod 000
        pytest.skip("running as root: chmod 000 does not deny anything")
    path = _valid_recording(tmp_path, suffix)
    path.chmod(0o000)
    try:
        with pytest.raises(CorruptFileError) as exc:
            probe_path(path)
    finally:
        path.chmod(0o644)   # let tmp_path clean itself up
    message = str(exc.value)
    assert "cannot be read: permission denied" in message, message
    assert "[Errno" not in message


def test_the_four_readers_agree_about_a_gone_file(tmp_path):
    """abf/csv already answered this honestly; the fix is that h5/tdms now give the SAME
    sentence — two dialogs for one condition must not disagree."""
    messages = set()
    for name in ("v.abf", "v.csv", "v.h5", "v.tdms"):
        gone = tmp_path / name      # never created
        with pytest.raises(CorruptFileError) as exc:
            probe_path(gone)
        messages.add(str(exc.value).removeprefix(name))
    assert len(messages) == 1, f"the four readers disagree about a gone file: {messages!r}"


# ================================================================================================
# LDR2-03: a /time that exists but cannot be used SAYS SO
# ================================================================================================

def _h5_with_time(tmp_path: Path, label: str, time_value) -> Path:
    n = 2_000
    return _write_h5(tmp_path / f"{label}.h5", current=np.zeros(n),
                     voltage=np.full(n, -0.6), time=time_value)


@pytest.mark.parametrize("label,time_value", [
    ("string_dtype", np.array([b"2026-01-01"] * 2_000)),
    ("two_d_block", np.zeros((2, 2_000))),
    ("single_sample", np.array([60.0])),
])
def test_an_unusable_time_dataset_is_reported_not_silently_absent(tmp_path, label, time_value):
    """Was: rate=None, issues=() for all three shapes — indistinguishable from a file that
    never had /time, while the value-level defects of the same dataset (non-monotonic,
    partial, oversized) were all loud. io/contract.py question 3: "A refusal the user cannot
    see is indistinguishable from a file that never had the channel"."""
    meta = probe_path(_h5_with_time(tmp_path, label, time_value)).meta
    assert meta.sample_rate_hz is None
    assert meta.rate_provenance is Provenance.UNKNOWN
    said = _text(meta.issues)
    assert "/time is present" in said, f"unusable /time reported nothing: {meta.issues!r}"
    assert "not usable as a time base" in said
    assert "sample rate unknown" in said


def test_a_file_that_never_had_time_still_gains_no_issue(tmp_path):
    """The other half of the fix's contract: absence stays silent — only a REFUSAL speaks."""
    n = 2_000
    p = _write_h5(tmp_path / "no_time.h5", current=np.zeros(n), voltage=np.full(n, -0.6))
    meta = probe_path(p).meta
    assert meta.sample_rate_hz is None
    assert not any("/time" in issue for issue in meta.issues), meta.issues


def test_the_unusable_time_issue_survives_the_load(tmp_path):
    doc = open_document(_h5_with_time(tmp_path, "loaded", np.zeros((2, 2_000))))
    assert "not usable as a time base" in _text(doc.meta.issues)


# ================================================================================================
# LDR2-04: a window past a partial /time carries the WINDOW's clock
# ================================================================================================

def _partial_time_h5(tmp_path: Path, *, n: int, n_time: int, t0: float, dt: float) -> Path:
    return _write_h5(tmp_path / "partial_time.h5",
                     current=np.zeros(n), voltage=np.full(n, -0.6),
                     time=t0 + np.arange(n_time) * dt)


def test_a_window_past_a_partial_time_base_gets_its_own_t0(tmp_path):
    """Was: meta.t0_s = the file's origin whenever start >= len(/time), so the x-axis, cursor
    readouts and every segment rebased from the document were start/fs seconds early — while
    the same load's partial-timebase issue promised "the rest is assumed to continue at the
    same rate". Measured error grew linearly with the window start; the fix extrapolates
    exactly as promised."""
    t0, dt, n = 60.00002, 2e-5, 20_000
    p = _partial_time_h5(tmp_path, n=n, n_time=100, t0=t0, dt=dt)
    for start in (5_000, 10_000):
        doc = open_document(p, SampleRange(start, start + 1_000))
        expected = t0 + start * dt
        assert doc.meta.t0_s == pytest.approx(expected, abs=1e-6), (
            f"start={start}: t0_s={doc.meta.t0_s} is not the window's own {expected}")
        assert "covers only part" in _text(doc.meta.issues), "the partial issue must still fire"


def test_a_window_still_covered_by_the_partial_time_base_reads_it_directly(tmp_path):
    """The in-range branch is untouched: /time itself is the better answer where it exists."""
    t0, dt, n = 60.00002, 2e-5, 20_000
    p = _partial_time_h5(tmp_path, n=n, n_time=100, t0=t0, dt=dt)
    doc = open_document(p, SampleRange(50, 1_050))
    assert doc.meta.t0_s == pytest.approx(t0 + 50 * dt, abs=1e-9)


# ================================================================================================
# LDR2-05: TDMS timing is every channel's testimony, not the first channel's
# ================================================================================================

def test_a_mixed_rate_tdms_refuses_a_single_clock_and_names_both_rates(tmp_path):
    """Was: ai0's 50 kHz applied to both channels, issues=() — every dwell time and duration
    on ai1 5x wrong by the file's own metadata. One recording carries one time axis, so the
    rate is left unknown WITH the conflict named (contract question 2: the rate is never
    silently corrected — saying that the file disagrees is the fix)."""
    p = _write_tdms(tmp_path / "mixed.tdms", [
        {"wf_increment": 2e-5, "wf_start_offset": 0.0, "unit_string": "Volts"},
        {"wf_increment": 1e-4, "wf_start_offset": 0.0, "unit_string": "Volts"},
    ])
    meta = probe_path(p).meta
    assert meta.sample_rate_hz is None, "no single clock may be adopted for a mixed-rate file"
    assert meta.rate_provenance is Provenance.UNKNOWN
    said = _text(meta.issues)
    assert "different sample rates" in said, f"the conflict was not named: {meta.issues!r}"
    assert "ai0" in said and "ai1" in said, "the issue must name the disagreeing channels"
    assert "50,000" in said and "10,000" in said, "the issue must name both declared rates"
    assert not any("declares no sample rate" in note for note in meta.notes), (
        "a file that declares two rates must not be described as declaring none")


def test_a_bare_first_channel_does_not_hide_the_second_channels_declared_rate(tmp_path):
    """Was: classified "TDMS (legacy)" with the note "This file declares no sample rate" —
    false text, the file declares 50 kHz on ai1."""
    p = _write_tdms(tmp_path / "bare_first.tdms", [
        {"unit_string": "Volts"},
        {"wf_increment": 2e-5, "wf_start_offset": 0.25, "unit_string": "Volts"},
    ])
    meta = probe_path(p).meta
    assert meta.sample_rate_hz == pytest.approx(50_000.0)
    assert meta.rate_provenance is Provenance.FILE_META
    assert meta.t0_s == pytest.approx(0.25), "t0 comes from the channel that declared timing"
    assert not any("declares no sample rate" in note for note in meta.notes), meta.notes


def test_agreeing_channels_still_load_with_one_clock_and_no_issue(tmp_path):
    """No false positive: the corpus layout — every channel declaring the same increment —
    gains nothing from the per-channel read."""
    p = _write_tdms(tmp_path / "agree.tdms",
                    [{"wf_increment": 2e-5, "unit_string": "Volts"}] * 2)
    meta = probe_path(p).meta
    assert meta.sample_rate_hz == pytest.approx(50_000.0)
    assert meta.issues == ()


def test_an_invalid_increment_on_a_later_channel_is_still_refused_by_name(tmp_path):
    """The declared-but-impossible-interval refusal now reads every channel, so a poisoned
    SECOND channel is caught too — and the sentence names which channel declared it."""
    p = _write_tdms(tmp_path / "poisoned.tdms", [
        {"wf_increment": 2e-5, "unit_string": "Volts"},
        {"wf_increment": 0.0, "unit_string": "Volts"},
    ])
    with pytest.raises(CorruptFileError) as exc:
        probe_path(p)
    message = str(exc.value)
    assert "not a valid sample interval" in message
    assert "ai1" in message


# ================================================================================================
# LDR2-06: a zero-overlap window on a shorter CSV is a request problem, not corruption
# ================================================================================================

def _short_csv(tmp_path: Path, rows: int = 500) -> Path:
    p = tmp_path / "short.csv"
    with p.open("w") as f:
        f.write("time,current\n")
        for i in range(rows):
            f.write(f"{i / 1000.0:.6f},{0.1 * i:.3f}\n")
    return p


def test_a_zero_overlap_window_on_a_csv_says_selects_nothing_not_corruption(tmp_path):
    """Was: CorruptFileError "no data rows were read." — names neither the request nor the
    file's real length, and its category says the intact file failed to read. The request
    outran the file (a project restored against a re-exported shorter CSV, or a Load-a-range
    start past the real last row, offered because the probe's n_samples is a byte estimate)."""
    with pytest.raises(UnsupportedFormatError) as exc:
        open_document(_short_csv(tmp_path), SampleRange(10_000, 11_000))
    message = str(exc.value)
    assert "selects nothing" in message, message
    assert "[10,000, 11,000)" in message, "the refusal must name the request"
    assert "500-sample recording" in message, "the refusal must name the file's real length"
    assert "no data rows were read" not in message


def test_the_csv_and_hdf5_zero_overlap_sentences_are_identical(tmp_path):
    """The defect was the DISAGREEMENT: the same request to .h5 already got the honest
    sentence from clamp_request. Pin that the two can never drift apart again."""
    csv = _short_csv(tmp_path)
    h5 = _write_h5(tmp_path / "short.h5", current=np.zeros(500),
                   time=np.arange(500) / 1000.0)

    def refusal(path: Path) -> str:
        with pytest.raises(ReaderError) as exc:
            open_document(path, SampleRange(10_000, 11_000))
        return str(exc.value).removeprefix(path.name)

    assert refusal(csv) == refusal(h5)


def test_a_partial_overlap_window_on_a_shorter_csv_still_loads_with_the_honest_issue(tmp_path):
    """The boundary the fix must not move: PARTIAL overlap keeps loading what exists and
    keeps short_window_issue's project-restore sentence."""
    doc = open_document(_short_csv(tmp_path), SampleRange(400, 1_000))
    assert doc.meta.n_samples == 100
    said = _text(doc.meta.issues)
    assert "asked for 600 samples" in said
    assert "Nothing is wrong with the file" in said
