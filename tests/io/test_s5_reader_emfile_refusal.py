"""EX5-02 — descriptor exhaustion is a system-resource refusal, not a damaged file.

Opening a healthy recording after the process ran out of file descriptors (EMFILE; macOS
default soft limit is 256) used to fall through every reader's content classification and
come out as corruption wording — for ABF, "the file is shorter than its header promises …
Recover the original file." — sending a researcher to re-copy, or discard, an intact
recording whose bytes were never even read.

The invariant restored, at the seam all four readers share: ``io.base.filesystem_refusal``
now classifies EMFILE/ENFILE from the exception chain (the one place the errno exists —
stat costs no descriptor, so no disk check can see this condition) and answers with the
limit's own name: the app has too many files open, the recording was never read, close
documents or restart. The durable disk facts still win when both are true.
"""

from __future__ import annotations

import errno
import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.io.base import filesystem_refusal, probe_path, select_reader
from nanoscope.io.errors import ReaderError

# What the shared sentence must and must not say. "shorter than its header" /
# "Recover the original" / "not a readable" are the corruption spellings this defect wore.
MUST_SAY = ("open-file limit", "never read")
MUST_NOT_SAY = ("shorter than its header", "Recover the original", "not a readable",
                "corrupt", "damaged", "signature")


@contextmanager
def _fd_exhausted():
    """Genuinely exhaust this process's descriptor table for the duration of the block.

    Held-open real descriptors, not an rlimit set below the live count — the same ground the
    sweep's verifier stood on. Everything the block needs that costs a descriptor must have
    happened before entry (reader modules are imported below, before the squeeze).
    """
    holds: list[int] = []
    try:
        try:
            while True:
                holds.append(os.open(os.devnull, os.O_RDONLY))
        except OSError as exc:
            assert exc.errno == errno.EMFILE, f"expected EMFILE while exhausting, got {exc!r}"
        yield
    finally:
        for fd in holds:
            os.close(fd)


@pytest.fixture()
def recordings(tmp_path):
    """One openable path per reader, plus reader modules pre-imported (a long session — the
    realistic host of this condition — has imported them long before the fd table fills)."""
    h5 = tmp_path / "rec.h5"
    with h5py.File(h5, "w") as h:
        h["current"] = np.arange(1_000, dtype=np.float64)
        h["time"] = np.arange(1_000, dtype=np.float64) / 50_000.0
    abf = tmp_path / "rec.abf"
    abf.write_bytes(b"ABF " + bytes(2_048))          # real v1 magic; neo never gets to parse it
    tdms = tmp_path / "rec.tdms"
    tdms.write_bytes(b"TDSm" + bytes(2_048))
    csv = tmp_path / "rec.csv"
    csv.write_text("time,current\n" + "".join(f"{i/50_000.0},{100+i%5}\n" for i in range(200)))
    for p in (h5, abf, tdms, csv):
        select_reader(p)                             # imports every reader module once
    return {"h5": h5, "abf": abf, "tdms": tdms, "csv": csv}


@pytest.mark.parametrize("fmt", ["h5", "abf", "tdms", "csv"])
def test_probe_at_the_descriptor_ceiling_names_the_limit_not_the_file(recordings, fmt):
    path = recordings[fmt]
    with _fd_exhausted():
        with pytest.raises(ReaderError) as excinfo:
            probe_path(path)
    message = str(excinfo.value)
    for fragment in MUST_SAY:
        assert fragment in message, f"[{fmt}] refusal must name '{fragment}': {message!r}"
    for fragment in MUST_NOT_SAY:
        assert fragment not in message, f"[{fmt}] refusal still says '{fragment}': {message!r}"
    assert path.name in message


def test_after_the_squeeze_the_same_file_opens_clean(recordings):
    # The refusal was about the PROCESS: once descriptors are back, the identical path
    # probes without complaint — proof the file was never the problem.
    with _fd_exhausted():
        with pytest.raises(ReaderError):
            probe_path(recordings["h5"])
    probe = probe_path(recordings["h5"])
    assert probe.meta.n_samples == 1_000


# ================================================================================================
# the classifier itself: chain-walking, ENFILE spelling, and durable facts still first
# ================================================================================================

def test_classifier_reads_the_errno_off_a_wrapped_cause(tmp_path):
    p = tmp_path / "x.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")
    inner = OSError(errno.EMFILE, "Too many open files")
    wrapper = RuntimeError("library wording")
    wrapper.__cause__ = inner
    message = filesystem_refusal(p, wrapper)
    assert message is not None and "open-file limit" in message


def test_classifier_spells_enfile_as_the_whole_computer(tmp_path):
    p = tmp_path / "x.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")
    message = filesystem_refusal(p, OSError(errno.ENFILE, "Too many open files in system"))
    assert message is not None
    assert "computer" in message and "across all programs" in message


def test_durable_disk_facts_still_beat_the_errno(tmp_path):
    # An EMFILE on a since-deleted file is still, most usefully, a gone file: the durable
    # fact wins, exactly as the classifier's order rule says.
    gone = tmp_path / "gone.h5"
    message = filesystem_refusal(gone, OSError(errno.EMFILE, "Too many open files"))
    assert message is not None and "no longer there" in message
    folder = tmp_path / "folder.h5"
    folder.mkdir()
    message = filesystem_refusal(folder, OSError(errno.EMFILE, "Too many open files"))
    assert message is not None and "is a folder" in message


def test_unclaimed_extension_at_the_ceiling_names_the_limit_not_empty_bytes(tmp_path):
    # A recording renamed .dat lands in select_reader's fallthrough, where read_magic's
    # swallowed OSError used to become "its first bytes (b'') match no known format" — a
    # quote of bytes nobody read. The fallthrough now opens the file itself and classifies
    # the failure.
    p = tmp_path / "renamed.dat"
    p.write_bytes(b"\x00" * 64)
    from nanoscope.io.base import _load_readers
    _load_readers()                                  # a long session imported them long ago
    with _fd_exhausted():
        with pytest.raises(ReaderError) as excinfo:
            select_reader(p)
    message = str(excinfo.value)
    assert "open-file limit" in message
    assert "b''" not in message and "match no known format" not in message


def test_cold_start_lazy_import_at_zero_descriptors_is_classified_too(tmp_path):
    # At exactly zero spare descriptors on a FRESH process, the first open to fail is
    # select_reader's own lazy import of the reader modules, which used to escape as a raw
    # "[Errno 24] Too many open files: …/nanoscope/io/hdf5.py". A subprocess is the only
    # honest way to a cold import cache. (macOS/Linux: fd exhaustion is per-process.)
    import subprocess
    import sys
    h5 = tmp_path / "rec.h5"
    with h5py.File(h5, "w") as h:
        h["current"] = np.arange(100, dtype=np.float64)
        h["time"] = np.arange(100, dtype=np.float64) / 50_000.0
    child = (
        "import os, sys\n"
        "import nanoscope\n"
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "from nanoscope.io.base import probe_path\n"
        "from nanoscope.io.errors import ReaderError\n"
        "import nanoscope.io.base as B\n"
        "assert not B._readers_loaded, 'import cache not cold'\n"
        "holds = []\n"
        "try:\n"
        "    while True:\n"
        "        holds.append(os.open(os.devnull, os.O_RDONLY))\n"
        "except OSError:\n"
        "    pass\n"
        "try:\n"
        "    try:\n"
        f"        probe_path({str(h5)!r})\n"
        "    except ReaderError as exc:\n"
        "        print('READER_ERROR:', str(exc))\n"
        "    else:\n"
        "        print('NO_ERROR')\n"
        "finally:\n"
        "    for fd in holds: os.close(fd)\n"
    )
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, f"child crashed: {proc.stderr[-800:]}"
    assert "READER_ERROR:" in proc.stdout, f"no classified refusal: {proc.stdout!r}"
    assert "open-file limit" in proc.stdout
    assert "Too many open files:" not in proc.stdout   # the raw import traceback wording


def test_ordinary_readable_file_with_no_errno_is_still_none(tmp_path):
    # The content door is untouched: a decode failure on a readable file stays the
    # readers' business.
    p = tmp_path / "fine.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")
    assert filesystem_refusal(p, ValueError("some decode failure")) is None
    assert filesystem_refusal(p) is None
