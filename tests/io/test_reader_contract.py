"""A file must never load quietly wrong.

Every test here reproduces a measured defect in which a reader answered one of the four
questions of :mod:`nanoscope.io.contract` — how many samples are really here, what is the
sample rate and how sure are we, which channels exist and why was one rejected, what encoding —
in a way that produced a plausible number and said nothing. All of them fail against a pre-P2
copy of the tree.

The "no false positive" tests at the end are the other half of the contract and matter just as
much: a well-formed file must gain no issue and no note it did not have before, or the box the
Files dock renders becomes noise nobody reads.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.core.types import Provenance, RecordingMeta, SampleRange
from nanoscope.io.abf import AbfReader
from nanoscope.io.base import Claim, open_document, probe_path, select_reader
from nanoscope.io.contract import ReaderReport, clamp_request, window_loaded
from nanoscope.io.errors import CorruptFileError, ReaderError, UnsupportedFormatError
from nanoscope.io.hdf5 import Hdf5Reader
from nanoscope.io.tdms import TdmsReader
from nanoscope.io.text import TextTraceReader

nptdms_writer = pytest.importorskip("nptdms")


# ================================================================================================
# Builders
# ================================================================================================

def _write_csv(path: Path, times, values, *, header="time,current", sep=",",
               encoding="utf-8") -> Path:
    rows = [header] if header else []
    rows += [f"{t!r}{sep}{v!r}" for t, v in zip(times, values)]
    path.write_bytes(("\n".join(rows) + "\n").encode(encoding))
    return path


def _uniform_times(n: int, dt: float) -> list[float]:
    return [i * dt for i in range(n)]


def _noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(-50.0, 1.5, n)


def _write_tdms(path: Path, data: np.ndarray, *, properties: dict, channels: int = 2) -> Path:
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    objs = [RootObject(properties={}), GroupObject("Untitled", properties={})]
    for i in range(channels):
        objs.append(ChannelObject("Untitled", f"ai{i}", data, properties=dict(properties)))
    with TdmsWriter(str(path)) as w:
        w.write_segment(objs)
    return path


def _write_h5(path: Path, **datasets) -> Path:
    with h5py.File(path, "w") as h:
        for name, value in datasets.items():
            h[name] = value
    return path


def _text(strings) -> str:
    return " || ".join(strings)


# ================================================================================================
# 1. How many samples are really here
# ================================================================================================

def test_a_truncated_tdms_says_the_recording_is_not_the_one_you_started(tmp_path):
    """MEASURED on the tree as found: 20,000 samples became 5,983, issues=(), notes=().

    That number is the denominator of every event rate in the app. nptdms recovers what it can
    and says so to a logger, which in this app means to nobody.
    """
    full = _write_tdms(tmp_path / "full.tdms", np.linspace(0.0, 1.0, 20_000),
                       properties={"wf_increment": 2e-5, "unit_string": "Volts"})
    raw = full.read_bytes()
    cut = tmp_path / "cut.tdms"
    cut.write_bytes(raw[: int(len(raw) * 0.8)])

    complete = probe_path(full)
    assert complete.meta.issues == (), "a complete TDMS must gain no issue"

    partial = probe_path(cut)
    assert partial.meta.n_samples < complete.meta.n_samples, "expected a short read to reproduce"
    said = _text(partial.meta.issues)
    assert "truncated" in said, f"a truncated TDMS reported nothing: {partial.meta.issues!r}"
    assert f"{partial.meta.n_samples:,}" in said, "the issue must quote the count you actually got"
    assert "event rate" in said, "the issue must name what the count is the denominator of"

    # and the same sentence survives the load, which is where the number is used
    assert "truncated" in _text(open_document(cut).meta.issues)


def test_the_tdms_decoders_own_complaint_reaches_the_user(tmp_path):
    """nptdms is the only component that knows a segment was short. It logs, and logging is not
    a place a user looks."""
    full = _write_tdms(tmp_path / "full.tdms", np.linspace(0.0, 1.0, 20_000),
                       properties={"wf_increment": 2e-5, "unit_string": "Volts"})
    cut = tmp_path / "cut.tdms"
    cut.write_bytes(full.read_bytes()[: int(full.stat().st_size * 0.8)])

    said = _text(probe_path(cut).meta.issues)
    assert "TDMS decoder reported" in said, "the decoder's own words never reached meta.issues"
    assert "less data than expected" in said


def test_a_tdms_cut_before_any_sample_is_not_called_deletable(tmp_path):
    """The empty-recording door tells the truth about the count and the WRONG thing about the
    cause: its sentence ends "this file is the expected result and can be deleted"."""
    full = _write_tdms(tmp_path / "full.tdms", np.linspace(0.0, 1.0, 20_000),
                       properties={"wf_increment": 2e-5, "unit_string": "Volts"})
    cut = tmp_path / "cut.tdms"
    cut.write_bytes(full.read_bytes()[: int(full.stat().st_size * 0.3)])

    with pytest.raises(ReaderError) as exc:
        probe_path(cut)
    message = str(exc.value)
    assert "truncated" in message
    assert "can be deleted" not in message
    assert "recover the original" in message


def test_a_tdms_whose_final_segment_declares_no_end_says_the_acquisition_was_interrupted(tmp_path):
    """LabVIEW writes 0xFFFFFFFFFFFFFFFF as the next-segment offset while a segment is still
    being written, and leaves it there if the program dies. The file opens; it is not complete."""
    import struct

    p = _write_tdms(tmp_path / "unclosed.tdms", np.linspace(0.0, 1.0, 20_000),
                    properties={"wf_increment": 2e-5, "unit_string": "Volts"})
    blob = bytearray(p.read_bytes())
    struct.pack_into("<Q", blob, 12, 0xFFFFFFFFFFFFFFFF)   # lead-in: next-segment offset
    p.write_bytes(bytes(blob))

    said = _text(probe_path(p).meta.issues)
    assert "interrupted rather than stopped" in said, said
    assert "truncated" in said


def test_a_complete_tdms_gains_nothing_from_the_chain_scan(tmp_path):
    full = _write_tdms(tmp_path / "full.tdms", np.linspace(0.0, 1.0, 5_000),
                       properties={"wf_increment": 2e-5, "unit_string": "Volts"})
    meta = open_document(full).meta
    assert meta.issues == ()
    assert meta.n_samples == 5_000


# ================================================================================================
# 2. What is the sample rate, and how sure are we
# ================================================================================================

def _changing_interval_csv(path: Path) -> tuple[Path, float]:
    """1,500 rows at 1e-4 s then 1,800 rows at 1e-3 s. The head scan sees only the first kind."""
    times: list[float] = []
    t = 0.0
    for _ in range(1500):
        times.append(t)
        t += 1e-4
    for _ in range(1800):
        times.append(t)
        t += 1e-3
    _write_csv(path, times, _noise(len(times)))
    return path, times[-1] - times[0]


def test_a_csv_whose_interval_changes_says_the_rate_does_not_describe_the_file(tmp_path):
    """MEASURED on the tree as found: 0.0600 s reported for a 0.3300 s recording, issues=().

    The cheap probe describes the disagreement; loading checks every timestamp and
    withholds the invalid uniform-rate interpretation. Clean rates are not re-estimated.
    """
    path, true_span = _changing_interval_csv(tmp_path / "changing.csv")

    meta = probe_path(path).meta
    said = _text(meta.issues)
    assert said, "a file whose sample interval changes probed with issues=()"
    assert "interval near the END" in said
    assert "dwell time" in said, "the issue must name what is wrong, not just what was seen"

    doc_meta = open_document(path).meta
    exact = _text(doc_meta.issues)
    assert "the rest of the file disagrees" in exact, f"load reported nothing exact: {exact!r}"
    # the sentence must quote the real span, not the one the rate implies
    assert f"{true_span:,.6g}" in exact
    assert doc_meta.sample_rate_hz is None
    assert doc_meta.rate_provenance is Provenance.UNKNOWN


def test_an_invalid_loaded_clock_does_not_get_a_guessed_replacement_rate(tmp_path):
    """Refuse the invalid clock instead of guessing a median or average replacement."""
    path, _ = _changing_interval_csv(tmp_path / "changing.csv")
    assert open_document(path).meta.sample_rate_hz is None
    assert open_document(path).meta.rate_provenance is Provenance.UNKNOWN


def test_an_hdf5_whose_time_base_does_not_cover_the_signal_says_so(tmp_path):
    n = 5_000
    _write_h5(tmp_path / "short_time.h5",
              current=_noise(n), voltage=np.full(n, -0.6),
              time=np.arange(1_000) * 2e-5)
    said = _text(probe_path(tmp_path / "short_time.h5").meta.issues)
    assert "/time holds" in said and "signal samples" in said


def test_an_hdf5_whose_time_base_jumps_says_the_rate_does_not_describe_the_file(tmp_path):
    n = 5_000
    t = np.arange(n) * 2e-5
    t[2_500:] += 3.0                       # a 3 s gap: dt at the head still says 20 us
    _write_h5(tmp_path / "gap.h5", current=_noise(n), voltage=np.full(n, -0.6), time=t)
    said = _text(probe_path(tmp_path / "gap.h5").meta.issues)
    assert "the rest of the file disagrees" in said


# ================================================================================================
# 3. Which channels exist, and why was one rejected
# ================================================================================================

def test_an_hdf5_current_stored_as_raw_counts_is_a_channel(tmp_path):
    """MEASURED on the tree as found: channels ['voltage'], issues (). The only channel event
    detection can run on was absent, and nothing said why."""
    n = 5_000
    p = _write_h5(tmp_path / "int.h5",
                  current=(_noise(n) * 100).astype(np.int32),
                  voltage=np.full(n, -0.6), time=np.arange(n) * 2e-5)
    meta = probe_path(p).meta
    assert [c.name for c in meta.channels] == ["current", "voltage"]
    assert "raw ADC counts" in _text(meta.notes), "an integer channel must say what it holds"

    doc = open_document(p)
    assert doc.channels[0].dtype == np.float32
    with h5py.File(p, "r") as h:
        assert np.array_equal(doc.channels[0], np.asarray(h["current"], dtype=np.float32))


def test_raw_counts_too_large_for_float32_say_the_samples_are_rounded(tmp_path):
    """float32 stops representing every integer at 2**24. A raw count above it is rounded, and a
    rounded count is a current step the user will never see."""
    n = 4_000
    counts = np.full(n, 1 << 25, dtype=np.int32)
    counts[0] = -(1 << 25)
    p = _write_h5(tmp_path / "big.h5", current=counts, voltage=np.full(n, -0.6),
                  time=np.arange(n) * 2e-5)
    said = _text(open_document(p).meta.issues)
    assert "cannot represent every integer" in said, said
    assert "rounded" in said


def test_ordinary_raw_counts_say_nothing_about_rounding(tmp_path):
    n = 4_000
    p = _write_h5(tmp_path / "small.h5", current=(_noise(n) * 100).astype(np.int16),
                  voltage=np.full(n, -0.6), time=np.arange(n) * 2e-5)
    assert open_document(p).meta.issues == ()


def test_an_hdf5_current_stored_as_one_row_of_a_block_is_a_channel(tmp_path):
    n = 5_000
    values = _noise(n)
    p = _write_h5(tmp_path / "twod.h5",
                  current=values.reshape(1, n),
                  voltage=np.full((n, 1), -0.6), time=np.arange(n) * 2e-5)
    doc = open_document(p)
    assert [c.name for c in doc.meta.channels] == ["current", "voltage"]
    assert doc.channels[0] == pytest.approx(values.astype(np.float32))
    assert doc.meta.n_samples == n


def test_an_hdf5_channel_that_still_cannot_be_read_says_why(tmp_path):
    n = 5_000
    p = _write_h5(tmp_path / "block.h5",
                  current=np.stack([_noise(n), _noise(n, 1)]),   # (2, N): genuinely ambiguous
                  voltage=np.full(n, -0.6), time=np.arange(n) * 2e-5)
    meta = probe_path(p).meta
    assert [c.name for c in meta.channels] == ["voltage"]
    said = _text(meta.issues)
    assert "/current is present in this file but is not loaded" in said, said
    assert "(2, 5000)" in said, "the reason must name the shape it actually found"


def test_a_lone_unreadable_channel_is_refused_with_the_reason_attached(tmp_path):
    n = 5_000
    p = _write_h5(tmp_path / "lone.h5", current=np.stack([_noise(n), _noise(n, 1)]))
    with pytest.raises(ReaderError) as exc:
        probe_path(p)
    assert "not loaded as a channel" in str(exc.value)


def test_a_zero_sample_channel_is_still_the_empty_recording_not_a_rejection(tmp_path):
    """P1's rule, which widening the channel gate is one line away from breaking: an acquisition
    armed and stopped writes a structurally perfect file with nothing in it, and it must get the
    ONE sentence written for that — not a channel-rejection notice per format."""
    p = _write_h5(tmp_path / "armed.h5",
                  current=np.zeros(0), voltage=np.zeros(0), time=np.arange(10) * 2e-5)
    for call in (lambda: probe_path(p), lambda: open_document(p)):
        with pytest.raises(ReaderError) as exc:
            call()
        assert "holds no samples" in str(exc.value)
        assert "not loaded as a channel" not in str(exc.value)


def test_a_real_recording_gains_no_rejection_noise(tmp_path):
    n = 5_000
    p = _write_h5(tmp_path / "clean.h5",
                  current=_noise(n), voltage=np.full(n, -0.6), time=np.arange(n) * 2e-5)
    meta = probe_path(p).meta
    assert meta.issues == ()
    assert "raw ADC" not in _text(meta.notes)


# ================================================================================================
# 4. What encoding is this text in
# ================================================================================================

def test_a_latin1_csv_loads_with_its_column_name_intact(tmp_path):
    """MEASURED on the tree as found: probed clean with a channel called 'curr?nt' and a
    plausible rate, then failed at load with a raw codec error. Routine off Windows instrument
    software."""
    p = tmp_path / "latin1.csv"
    times = _uniform_times(3_000, 1e-5)
    _write_csv(p, times, _noise(3_000), header="time;currènt", sep=";", encoding="latin-1")

    meta = probe_path(p).meta
    assert [c.name for c in meta.channels] == ["currènt"]
    assert "not valid UTF-8" in _text(meta.notes)

    doc = open_document(p)                     # this used to raise CorruptFileError
    assert doc.meta.n_samples == 3_000
    assert [c.name for c in doc.meta.channels] == ["currènt"]


def test_a_csv_with_a_bad_byte_deep_inside_is_refused_by_the_ROW_not_by_a_codec_offset(tmp_path):
    """The head decodes as UTF-8 and a byte past the encoding probe does not. Re-reading as
    Latin-1 gets past the codec and reaches the thing that is actually wrong — a data cell that
    is not a number — which is what the user can act on. Before, the message was
    "'utf-8' codec can't decode byte 0xe9 in position 233412", which points at nothing."""
    p = tmp_path / "mixed.csv"
    n = 40_000
    rows = ["time,current"] + [f"{i * 1e-5!r},{-50.0 + (i % 7) * 0.01!r}" for i in range(n)]
    blob = bytearray(("\n".join(rows) + "\n").encode("utf-8"))
    blob[len(blob) // 2] = 0xE9                # a Latin-1 'é' deep inside the file
    p.write_bytes(bytes(blob))

    with pytest.raises(CorruptFileError) as exc:
        open_document(p)
    message = str(exc.value)
    assert "not valid UTF-8" in message, message
    assert "read as Latin-1 instead" in message
    assert "That row is the one to look at" in message


def test_a_utf8_csv_gains_no_encoding_note(tmp_path):
    p = _write_csv(tmp_path / "plain.csv", _uniform_times(3_000, 2e-5), _noise(3_000))
    assert "UTF-8" not in _text(probe_path(p).meta.notes)


# ================================================================================================
# 5. The load window records what was READ, never what was asked for
# ================================================================================================

def test_a_csv_window_past_the_end_records_what_was_read(tmp_path):
    """MEASURED on the tree as found: n_samples=1500 beside loaded_range=(500, 8000) — a value
    that then went into the project manifest and was replayed on the next restore."""
    p = _write_csv(tmp_path / "short.csv", _uniform_times(2_000, 2e-5), _noise(2_000))
    doc = open_document(p, SampleRange(500, 8_000))
    assert doc.meta.n_samples == 1_500
    assert doc.meta.loaded_range == SampleRange(500, 2_000)


def test_every_reader_records_a_window_whose_length_is_its_sample_count(tmp_path):
    """The inconsistency WAS the defect: three readers clamped, one did not."""
    n = 2_000
    made: dict[str, Path] = {
        "csv": _write_csv(tmp_path / "a.csv", _uniform_times(n, 2e-5), _noise(n)),
        "h5": _write_h5(tmp_path / "a.h5", current=_noise(n), voltage=np.full(n, -0.6),
                        time=np.arange(n) * 2e-5),
        "tdms": _write_tdms(tmp_path / "a.tdms", _noise(n),
                            properties={"wf_increment": 2e-5, "unit_string": "Volts"}),
    }
    for label, path in made.items():
        doc = open_document(path, SampleRange(500, 8_000))
        lr = doc.meta.loaded_range
        assert lr is not None, label
        assert lr.stop - lr.start == doc.meta.n_samples, f"{label}: {lr} vs {doc.meta.n_samples}"
        assert lr.start == 500, label


def test_a_window_that_selects_nothing_is_refused_by_every_reader(tmp_path):
    n = 2_000
    for path in (_write_h5(tmp_path / "b.h5", current=_noise(n), voltage=np.full(n, -0.6),
                           time=np.arange(n) * 2e-5),
                 _write_tdms(tmp_path / "b.tdms", _noise(n),
                             properties={"wf_increment": 2e-5, "unit_string": "Volts"})):
        with pytest.raises(ReaderError) as exc:
            open_document(path, SampleRange(9_000, 9_500))
        assert "selects nothing" in str(exc.value), path.suffix


def test_a_meta_whose_window_disagrees_with_its_length_is_refused():
    """The door a fifth reader has to pass through."""
    kwargs = dict(path=Path("/tmp/x.abf"), format_name="Axon ABF", sample_rate_hz=50_000.0,
                  rate_provenance=Provenance.FILE_META, channels=())
    RecordingMeta(n_samples=1_500, loaded_range=SampleRange(500, 2_000), **kwargs)  # consistent
    with pytest.raises(ValueError) as exc:
        RecordingMeta(n_samples=1_500, loaded_range=SampleRange(500, 8_000), **kwargs)
    assert "records what was READ" in str(exc.value)


def test_window_loaded_is_computed_from_the_read_not_the_request():
    assert window_loaded(None, 0, 100) is None
    assert window_loaded(SampleRange(500, 8_000), 500, 1_500) == SampleRange(500, 2_000)


def test_clamp_request_refuses_an_empty_selection():
    assert clamp_request(SampleRange(500, 8_000), 2_000, name="x") == (500, 2_000)
    with pytest.raises(UnsupportedFormatError):
        clamp_request(SampleRange(9_000, 9_500), 2_000, name="x")


# ================================================================================================
# 6. A declared-but-impossible sample interval
# ================================================================================================

def test_a_tdms_declaring_a_zero_sample_interval_is_refused_not_called_rateless(tmp_path):
    """`if wf_increment:` is FALSE at 0.0, so the guard written for exactly that value was
    unreachable and the file loaded carrying the note "This file declares no sample rate" — which
    is not true. It declares one, and the one it declares is impossible."""
    p = _write_tdms(tmp_path / "zero.tdms", _noise(500),
                    properties={"wf_increment": 0.0, "unit_string": "Volts"})
    with pytest.raises(CorruptFileError) as exc:
        probe_path(p)
    assert "not a valid sample interval" in str(exc.value)


def test_a_tdms_declaring_a_negative_interval_is_refused_the_same_way(tmp_path):
    p = _write_tdms(tmp_path / "neg.tdms", _noise(500),
                    properties={"wf_increment": -2e-5, "unit_string": "Volts"})
    with pytest.raises(CorruptFileError):
        probe_path(p)


def test_a_legacy_tdms_with_no_increment_still_says_the_rate_is_unknown(tmp_path):
    p = _write_tdms(tmp_path / "legacy.tdms", _noise(500), properties={})
    meta = probe_path(p).meta
    assert meta.sample_rate_hz is None
    assert meta.rate_provenance is Provenance.UNKNOWN
    assert "declares no sample rate" in _text(meta.notes)


# ================================================================================================
# 7. A .h5 that is not HDF5 is still ours
# ================================================================================================

def test_a_non_hdf5_h5_is_told_it_is_not_hdf5(tmp_path):
    """Claim's own docstring warns about exactly this: NO sent the user hunting for a plugin
    that was never missing, from an app whose file dialog lists .h5."""
    p = tmp_path / "notes.h5"
    p.write_bytes(b"a note to self, saved with the wrong extension\n" * 20)

    assert Hdf5Reader.sniff(p) is Claim.WEAK
    assert select_reader(p) is Hdf5Reader
    with pytest.raises(ReaderError) as exc:
        probe_path(p)
    message = str(exc.value)
    assert "not a readable HDF5 file" in message
    assert "matches no reader" not in message


def test_a_real_hdf5_is_still_claimed_strong(tmp_path):
    n = 2_000
    p = _write_h5(tmp_path / "real.h5", current=_noise(n), voltage=np.full(n, -0.6),
                  time=np.arange(n) * 2e-5)
    assert Hdf5Reader.sniff(p) is Claim.STRONG


# ================================================================================================
# 8. An episodic ABF is a fraction of the file
# ================================================================================================

class _FakeAbf:
    def __init__(self, blocks: int, sweeps: int) -> None:
        self._b, self._s = blocks, sweeps

    def block_count(self) -> int:
        return self._b

    def segment_count(self, _block: int) -> int:
        return self._s


def test_a_gapfree_abf_says_nothing_about_sweeps():
    assert AbfReader._sweep_report(_FakeAbf(1, 1), 2_993_798).issues == ()


def test_an_episodic_abf_says_only_the_first_sweep_is_loaded():
    """get_signal_size(0, 0) is the FIRST SWEEP's length, and it is reported as the recording."""
    said = _text(AbfReader._sweep_report(_FakeAbf(1, 40), 12_000).issues)
    assert "40 sweeps" in said
    assert "Only the first sweep (12,000 samples) is loaded" in said
    assert "every rate derived from them" in said


class _RatedAbf:
    def __init__(self, rate) -> None:
        self._rate = rate

    def get_signal_sampling_rate(self):
        return self._rate


@pytest.mark.parametrize("rate", [0.0, -50_000.0, float("nan"), float("inf")])
def test_an_abf_declaring_an_impossible_sample_rate_is_refused(rate):
    """Nothing downstream errors on a zero rate — it produces inf seconds and a plot with no time
    axis, which is the shape of every defect in this tier."""
    reader = AbfReader(Path("/tmp/x.abf"))
    with pytest.raises(CorruptFileError) as exc:
        reader._rate(_RatedAbf(rate))
    assert "divides by it" in str(exc.value)


def test_an_abf_declaring_a_real_rate_is_untouched():
    assert AbfReader(Path("/tmp/x.abf"))._rate(_RatedAbf(50_000.0)) == 50_000.0


@pytest.mark.parametrize("bad", ["not a number", None, float("nan"), float("inf")])
def test_a_tdms_start_offset_that_is_not_a_number_is_no_start_offset(bad):
    """float() on whatever the file stored used to escape the scan as a raw ValueError, where the
    reader had promised a sentence."""
    from nanoscope.io.tdms import _finite_or_zero

    assert _finite_or_zero(bad) == 0.0
    assert _finite_or_zero(1.25) == 1.25


def test_a_neo_without_sweep_counts_says_nothing_rather_than_crashing():
    class _Old:
        def block_count(self):
            raise AttributeError("this neo does not expose it")

    assert AbfReader._sweep_report(_Old(), 10).issues == ()


# ================================================================================================
# 9. The accumulator
# ================================================================================================

def test_a_report_never_repeats_itself():
    r = ReaderReport(issues=("a", "a"), notes=("n",))
    r.issue("a")
    r.note("n")
    r.issue("")
    assert r.issues == ("a",)
    assert r.notes == ("n",)


# ================================================================================================
# 10. The load window's OTHER half: what time does sample 0 carry
# ================================================================================================

def test_every_reader_gives_a_window_the_windows_own_clock(tmp_path):
    """``meta.t0_s`` is the time of the FIRST LOADED SAMPLE, because that is what
    ``session.x_axis`` (``arange(n)/fs + t0_s``) and ``segment.make_segment``
    (``parent.t0_s + a/fs``) both read it as.

    HDF5 read it at the window's own index; the other three left it at the file's origin, so
    the same window of the same experiment carried two clocks depending on the format. On the
    owner's 606 MB ATP recording — the one file over SOFT_SAMPLE_CEILING, so a window IS its
    normal load path — a window at sample 30,000,000 put every experimental time 600 s early.
    """
    n, dt, t0 = 20_000, 2e-5, 5.0
    start, stop = 10_000, 15_000
    want = t0 + start * dt
    sig, volt = _noise(n), np.full(n, -0.6)

    h5 = _write_h5(tmp_path / "w.h5", current=sig, voltage=volt,
                   time=np.arange(n) * dt + t0)
    csv = _write_csv(tmp_path / "w.csv", [t0 + i * dt for i in range(n)], sig)
    tdms = _write_tdms(tmp_path / "w.tdms", sig,
                       properties={"wf_increment": dt, "wf_start_offset": t0})

    for label, doc in (("hdf5", Hdf5Reader(h5).load(SampleRange(start, stop))),
                       ("text", TextTraceReader(csv).load(SampleRange(start, stop))),
                       ("tdms", TdmsReader(tdms).load(SampleRange(start, stop)))):
        assert doc.meta.t0_s == pytest.approx(want, abs=1e-6), label
        assert doc.meta.loaded_range == SampleRange(start, stop), label

    # A whole-file load is bit-identical to what it always was.
    for label, doc in (("hdf5", Hdf5Reader(h5).load()),
                       ("text", TextTraceReader(csv).load()),
                       ("tdms", TdmsReader(tdms).load())):
        assert doc.meta.t0_s == pytest.approx(t0, abs=1e-9), label


def test_window_t0_leaves_a_whole_file_load_alone():
    from nanoscope.io.contract import window_t0

    assert window_t0(61.3, 0, 50_000.0) == 61.3
    assert window_t0(61.3, 1_000_000, None) == 61.3        # rate unknown -> never fabricate
    assert window_t0(61.3, 1_000_000, 0.0) == 61.3
    assert window_t0(0.0, 500, 1_000.0) == pytest.approx(0.5)


# ================================================================================================
# 11. A UTF-16 CSV can be rate-checked (guard #4 used to be skipped outright)
# ================================================================================================

def _hiccup_rows(n_head: int = 1500, n_tail: int = 500,
                 dt0: float = 1e-4, dt1: float = 1e-3):
    times = [i * dt0 for i in range(n_head)]
    times += [times[-1] + (i + 1) * dt1 for i in range(n_tail)]
    return times, list(_noise(n_head + n_tail))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_a_rate_that_stops_describing_the_file_is_reported_in_every_encoding(tmp_path, encoding):
    """_BYTE_SEEKABLE excluded UTF-16, so ``_tail_times`` returned [] and BOTH halves of
    guard #4 were gated off — for the one encoding Windows instrument software ("Unicode
    Text") writes by default. Measured before: the UTF-8 file raised two issues and its
    byte-identical UTF-16 twin loaded with ``issues=()``, reporting 0.2000 s for a 0.6490 s
    recording."""
    times, values = _hiccup_rows()
    p = _write_csv(tmp_path / f"h_{encoding}.csv", times, values, encoding=encoding)

    probe = TextTraceReader(p).probe().meta
    assert any("interval near the END" in s for s in probe.issues), probe.issues

    meta = TextTraceReader(p).load().meta
    assert any("the rest of the file disagrees" in s for s in meta.issues), meta.issues
    assert meta.n_samples == len(times)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_a_uniform_csv_gains_nothing_in_any_encoding(tmp_path, encoding):
    n = 4000
    p = _write_csv(tmp_path / f"u_{encoding}.csv", _uniform_times(n, 1e-4), _noise(n),
                   encoding=encoding)
    assert TextTraceReader(p).probe().meta.issues == ()
    meta = TextTraceReader(p).load().meta
    assert meta.issues == ()
    assert meta.n_samples == n


# ================================================================================================
# 12. A TDMS cut inside a segment HEADER
# ================================================================================================

def _streamed_tdms(path: Path, *, segments: int = 8, per_segment: int = 2000) -> Path:
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    props = {"wf_increment": 1e-5, "wf_start_offset": 0.0}
    with TdmsWriter(str(path)) as w:
        for s in range(segments):
            objs = [RootObject(properties={}), GroupObject("Untitled", properties={})]
            for c in range(2):
                objs.append(ChannelObject("Untitled", f"ai{c}",
                                          _noise(per_segment, seed=s * 10 + c),
                                          properties=dict(props)))
            w.write_segment(objs)
    return path


def _segment_offsets(raw: bytes) -> list[int]:
    import struct

    out, pos = [], 0
    while pos + 28 <= len(raw) and raw[pos:pos + 4] == b"TDSm":
        out.append(pos)
        pos += 28 + struct.unpack("<q", raw[pos + 12:pos + 20])[0]
    return out


def test_a_tdms_cut_inside_a_segment_header_is_reported(tmp_path):
    """``_segment_chain_defect``'s loop is ``while pos + 28 <= size``, so a file ending 1–27
    bytes into the next lead-in fell out of it and returned None — the trailing fragment,
    which is the proof of truncation, was discarded. Measured before: 14,000 of 20,000
    samples per channel, ``issues=()``, ``notes=()``, and nptdms silent as well."""
    src = _streamed_tdms(tmp_path / "stream.tdms")
    raw = src.read_bytes()
    offs = _segment_offsets(raw)
    assert len(offs) >= 6

    cut = tmp_path / "cut_header.tdms"
    cut.write_bytes(raw[:offs[5] + 12])          # 12 bytes into a 28-byte lead-in
    meta = TdmsReader(cut).probe().meta
    assert any("into the header of segment" in s for s in meta.issues), meta.issues
    assert any("Do not pool them" in s for s in meta.issues), meta.issues


def test_a_complete_tdms_stream_gains_nothing(tmp_path):
    """The control for the check above: a file that ends exactly where its last segment does
    is complete, and the trailing-bytes rule must never fire on it."""
    src = _streamed_tdms(tmp_path / "whole.tdms")
    assert TdmsReader(src).probe().meta.issues == ()


# ================================================================================================
# 13. An HDF5 /time LONGER than the signal
# ================================================================================================

def test_a_timebase_longer_than_the_signal_is_reported(tmp_path):
    """The mirror of the ``n_time < n_signal`` branch, and it was the silent one:
    ``_common_length`` only cross-checks signal channels against each other."""
    n = 4000
    p = _write_h5(tmp_path / "long_time.h5", current=_noise(n), voltage=np.full(n, -0.6),
                  time=np.arange(2 * n) * 2e-5)
    meta = Hdf5Reader(p).probe().meta
    assert any("runs" in s and "past the last sample" in s for s in meta.issues), meta.issues


def test_a_timebase_that_matches_the_signal_is_silent(tmp_path):
    n = 4000
    p = _write_h5(tmp_path / "ok_time.h5", current=_noise(n), voltage=np.full(n, -0.6),
                  time=np.arange(n) * 2e-5)
    assert Hdf5Reader(p).probe().meta.issues == ()


# ================================================================================================
# 14. The text reader converts its OSErrors like the other three
# ================================================================================================

def test_a_csv_that_cannot_be_opened_is_a_sentence_not_a_traceback(tmp_path):
    """The only reader that converted no OSError: a moved file, an unmounted share, a
    permission and a DIRECTORY named ``*.csv`` all escaped as raw FileNotFoundError /
    PermissionError / IsADirectoryError with an absolute path."""
    missing = tmp_path / "gone.csv"
    with pytest.raises(ReaderError) as exc:
        TextTraceReader(missing).probe()
    assert "no longer there" in str(exc.value)

    folder = tmp_path / "adir.csv"
    folder.mkdir()
    with pytest.raises(ReaderError) as exc:
        TextTraceReader(folder).probe()
    assert "is a folder" in str(exc.value)


# ================================================================================================
# 15. A window that outruns the file
# ================================================================================================

def test_a_window_the_file_cannot_fill_says_so(tmp_path):
    """``loaded_range`` is clamped to the truth (C28) and used to say nothing about the rows
    that were asked for and are not there — which is what a project restore looks like when
    the CSV has been re-exported shorter since the project was saved."""
    n = 2000
    p = _write_csv(tmp_path / "two_k.csv", _uniform_times(n, 1e-4), _noise(n))
    meta = open_document(p, SampleRange(500, 8000)).meta
    assert meta.n_samples == 1500
    assert meta.loaded_range == SampleRange(500, 2000)
    assert any("asked for 7,500 samples" in s for s in meta.issues), meta.issues

    inside = open_document(p, SampleRange(500, 1500)).meta
    assert inside.issues == ()


def test_a_window_that_hits_the_end_still_gets_the_exact_rate_check(tmp_path):
    """The exact half of guard #4 was gated on ``sample_range is None``. A window that ran off
    the end of the file settles the file's exact row count just as a whole-file load does
    (``start + n`` IS the length), so it gets the same exact answer."""
    times, values = _hiccup_rows()
    p = _write_csv(tmp_path / "hiccup_w.csv", times, values)
    meta = TextTraceReader(p).load(SampleRange(0, 99_999)).meta
    assert any("the rest of the file disagrees" in s for s in meta.issues), meta.issues


# ================================================================================================
# The clamp reports, in every reader that clamps  (found by the bundle verification)
# ================================================================================================
#
# `short_window_issue` existed and read well, and exactly ONE reader emitted it. ABF, HDF5 and
# TDMS all clamped correctly -- `loaded_range` never disagreed with `n_samples`, the arithmetic
# stayed self-consistent, t0 rebased properly -- and then said nothing at all. The cost is not a
# wrong number; it is silence in the one situation that needs a sentence: reopening a project
# against a recording that has been re-exported SHORTER since the project was saved. The stored
# results were measured on a span this document does not hold, and nothing said so.
#
# Fixed at the invariant rather than at three call sites: `clamp_request` is the only place that
# can see the discrepancy (it alone knows both the request and the true length), it is the one
# function every clamping reader already has to call, and it now takes the report. A fifth reader
# gets the sentence by calling the function it cannot avoid.

def _asked_and_got(issues) -> str:
    return _text(issues)


def test_every_clamping_reader_says_the_window_outran_the_file(tmp_path):
    """The regression. All three silent readers, one property: clamped => reported."""
    n, dt = 2_000, 2e-5
    sig, volt = _noise(n), np.full(n, -0.6)

    h5 = _write_h5(tmp_path / "s.h5", current=sig, voltage=volt,
                   time=np.arange(n) * dt)
    tdms = _write_tdms(tmp_path / "s.tdms", sig,
                       properties={"wf_increment": dt, "wf_start_offset": 0.0})

    for label, doc in (("hdf5", Hdf5Reader(h5).load(SampleRange(500, 8_000))),
                       ("tdms", TdmsReader(tdms).load(SampleRange(500, 8_000)))):
        blob = _asked_and_got(doc.meta.issues)
        assert "7,500" in blob and "500" in blob, f"{label}: {doc.meta.issues}"
        assert "1,500" in blob, f"{label} never said what it actually holds: {doc.meta.issues}"
        # the clamp itself is unchanged -- this fix adds a sentence, not an arithmetic change
        assert doc.meta.loaded_range == SampleRange(500, 2_000), label
        assert doc.meta.n_samples == 1_500, label


def test_a_window_inside_the_file_stays_silent(tmp_path):
    """The control that makes the test above mean something: an ordinary windowed load of a
    file long enough to satisfy it must gain nothing. This is the normal path for the owner's
    606 MB recording, so a false alarm here would fire on every routine load of it."""
    n, dt = 2_000, 2e-5
    h5 = _write_h5(tmp_path / "i.h5", current=_noise(n), voltage=np.full(n, -0.6),
                   time=np.arange(n) * dt)
    tdms = _write_tdms(tmp_path / "i.tdms", _noise(n),
                       properties={"wf_increment": dt, "wf_start_offset": 0.0})

    for label, doc in (("hdf5", Hdf5Reader(h5).load(SampleRange(500, 1_500))),
                       ("tdms", TdmsReader(tdms).load(SampleRange(500, 1_500)))):
        assert "asked for" not in _asked_and_got(doc.meta.issues), label
        assert doc.meta.loaded_range == SampleRange(500, 1_500), label


def test_a_whole_file_load_stays_silent(tmp_path):
    """The second control: no window, no sentence, on every reader."""
    n, dt = 2_000, 2e-5
    h5 = _write_h5(tmp_path / "w.h5", current=_noise(n), voltage=np.full(n, -0.6),
                   time=np.arange(n) * dt)
    tdms = _write_tdms(tmp_path / "w.tdms", _noise(n),
                       properties={"wf_increment": dt, "wf_start_offset": 0.0})

    for label, doc in (("hdf5", Hdf5Reader(h5).load()), ("tdms", TdmsReader(tdms).load())):
        assert "asked for" not in _asked_and_got(doc.meta.issues), label
        assert doc.meta.loaded_range is None, label


def test_clamp_request_reports_only_when_it_actually_clamps():
    """The invariant, unit-level: the report is a function of whether the span shrank."""
    short = ReaderReport()
    assert clamp_request(SampleRange(500, 8_000), 2_000, name="x", report=short) == (500, 2_000)
    assert len(short.issues) == 1 and "asked for 7,500 samples" in short.issues[0]

    inside = ReaderReport()
    assert clamp_request(SampleRange(500, 1_500), 2_000, name="x", report=inside) == (500, 1_500)
    assert inside.issues == ()

    # and the argument is optional, so the pre-existing callers keep working untouched
    assert clamp_request(SampleRange(500, 8_000), 2_000, name="x") == (500, 2_000)
