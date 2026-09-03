"""Third-sweep loader defects, part 3: one condition, one sentence, whatever the extension.

  IO3-03  An unreadable FOLDER was classified two ways depending on the name it wore: ``.abf``
          and ``.csv`` said "cannot be read: permission denied" (they tested ``PermissionError``
          first, and opening a mode-000 directory raises exactly that), ``.h5`` and ``.tdms``
          said "is a folder, not a file". The ``.abf`` dialog then FLIPPED once the user acted
          on its advice — it had sent them to fix a permission that was never the reason.

  IO3-07  The text scan split on the delimiter without RFC 4180 quoting while its own
          ``pd.read_csv`` honours quotes, so two ordinary CSVs that pandas parses into 3,000
          clean rows were refused — one as a "ragged table … event-table export … land in v2"
          because a quoted header field contains a comma, one as "column 0 is not numeric" of
          a column that is numeric.

  IO3-08  Exactly one of the twelve wrong-extension combinations got the "these bytes are
          actually X" redirect the LDR-01/LDR2-02 fixes established. The rest got the decoder's
          own words — h5py's "file signature not found" for an intact ABF, nptdms's "Segment
          does not start with b'TDSm'" — and ``.csv`` got a confidently FALSE description: a
          binary ABF called a ragged event-table export.

  IO3-10  A TDMS with one full channel and one never-filled channel was refused as an empty
          recording, whose sentence ends "this file is the expected result and can be deleted"
          — over 20,000 samples that are in the file, while the differing-lengths issue the
          reader had already built for it was never shown.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from nanoscope.io.base import open_document, probe_path
from nanoscope.io.errors import (
    CorruptFileError,
    EmptyRecordingError,
    ReaderError,
    UnsupportedFormatError,
)

pytest.importorskip("nptdms")

_EXTENSIONS = (".abf", ".h5", ".tdms", ".csv")

as_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="running as root: chmod 000 does not deny anything")


def _text(strings) -> str:
    return " || ".join(strings)


def _refusal(path: Path) -> str:
    with pytest.raises(ReaderError) as exc:
        probe_path(path)
    return str(exc.value)


# =================================================================================================
# IO3-03: being a folder is the durable fact, and it wins
# =================================================================================================

@as_root
def test_an_unreadable_folder_is_a_folder_under_every_extension(tmp_path):
    """Was: .abf/.csv "cannot be read: permission denied", .h5/.tdms "is a folder, not a
    file" — for one directory, from one app, in the same dialog."""
    said = {}
    for ext in _EXTENSIONS:
        folder = tmp_path / f"session{ext}"
        folder.mkdir()
        folder.chmod(0o000)
        try:
            said[ext] = _refusal(folder)
        finally:
            folder.chmod(0o755)
    for ext, message in said.items():
        assert "is a folder, not a file" in message, f"{ext}: {message}"
        assert "permission denied" not in message, (
            f"{ext}: chmod cannot make a folder into a file, so it must not be the advice")


@as_root
def test_the_folder_answer_does_not_change_when_the_mode_is_relaxed(tmp_path):
    """The proof that the old answer was wrong rather than merely different: the user follows
    the advice, and the dialog tells them something else."""
    for ext in _EXTENSIONS:
        folder = tmp_path / f"relax{ext}"
        folder.mkdir()
        folder.chmod(0o000)
        try:
            locked = _refusal(folder)
        finally:
            folder.chmod(0o755)
        assert locked == _refusal(folder), f"{ext}: the refusal changed with the mode alone"


@as_root
def test_a_permission_locked_FILE_still_says_permission_denied(tmp_path):
    """No false positive: for a real file, permission IS the durable fact and the advice is
    actionable."""
    p = tmp_path / "locked.csv"
    p.write_text("time,current\n0.0,-50.0\n0.0001,-50.1\n0.0002,-50.2\n")
    p.chmod(0o000)
    try:
        message = _refusal(p)
    finally:
        p.chmod(0o644)
    assert "cannot be read: permission denied" in message, message


def test_the_four_readers_still_agree_about_a_gone_file(tmp_path):
    """The sweep-2 invariant this fix shares a classifier with: it must not have moved."""
    messages = {str(_refusal(tmp_path / f"v{ext}")).removeprefix(f"v{ext}")
                for ext in _EXTENSIONS}
    assert len(messages) == 1, f"the four readers disagree about a gone file: {messages!r}"


# =================================================================================================
# IO3-07: the scan and the loader must agree about what the file says
# =================================================================================================

def _quoted_header_csv(tmp_path: Path, n: int = 3_000) -> Path:
    p = tmp_path / "quoted_header.csv"
    lines = ['time,"current, pA","voltage, mV"']
    lines += [f"{i / 10000.0:.6f},{-119.8 - 0.001 * i:.4f},-600.0" for i in range(n)]
    p.write_text("\n".join(lines) + "\n")
    return p


def _quote_all_csv(tmp_path: Path, n: int = 3_000) -> Path:
    p = tmp_path / "quote_all.csv"
    lines = ['"time","current","voltage"']
    lines += [f'"{i / 10000.0:.6f}","{-119.8 - 0.001 * i:.4f}","-600.0"' for i in range(n)]
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.mark.parametrize("builder", [_quoted_header_csv, _quote_all_csv],
                         ids=["quoted header field", "quoted numeric fields"])
def test_an_ordinary_quoted_csv_is_a_trace_not_damage(tmp_path, builder):
    """Was: "ragged table (3 fields x999, 5 fields x1). This looks like an event-table export,
    not a trace. Event tables land in v2." and "column 0 is not numeric, so it is not a time
    column." — both false, about files pandas parses perfectly."""
    p = builder(tmp_path)
    doc = open_document(p)
    truth = pd.read_csv(p)
    assert doc.meta.n_samples == len(truth), "the loader and the scan must count one table"
    assert len(doc.channels) == 2, "two signal columns beside the time column"
    assert doc.channels[0][0] == pytest.approx(float(truth.iloc[0, 1]), rel=1e-5)
    assert doc.meta.sample_rate_hz == pytest.approx(10_000.0)


def test_a_quoted_header_names_the_channels_the_way_it_is_written(tmp_path):
    """The quoted field is one name, not two: the comma inside it belongs to the name."""
    meta = probe_path(_quoted_header_csv(tmp_path)).meta
    names = [c.name for c in meta.channels]
    assert names == ["current, pA", "voltage, mV"], names


def test_a_genuinely_ragged_table_is_still_refused(tmp_path):
    """No false positive: quoting is not an excuse to accept the user's event-table exports,
    which are what the raggedness guard exists for."""
    p = tmp_path / "events.csv"
    p.write_text("id,t_start,dwell\n1,0.1,0.002\n2,0.2,0.003,EXTRA\n3,0.3,0.004\n" * 10)
    with pytest.raises(UnsupportedFormatError) as exc:
        probe_path(p)
    assert "ragged table" in str(exc.value)


# =================================================================================================
# IO3-08: a renamed file is told what it is, in every direction
# =================================================================================================

def _hdf5_bytes(tmp_path: Path) -> Path:
    p = tmp_path / "src.h5"
    with h5py.File(p, "w") as h:
        h["current"] = np.zeros(2_000)
        h["voltage"] = np.full(2_000, -0.6)
        h["time"] = np.arange(2_000) * 2e-5
    return p


def _tdms_bytes(tmp_path: Path) -> Path:
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    p = tmp_path / "src.tdms"
    objs = [RootObject(properties={}), GroupObject("Untitled", properties={})]
    for i in range(2):
        objs.append(ChannelObject("Untitled", f"ai{i}", np.zeros(2_000),
                                  properties={"wf_increment": 2e-5, "unit_string": "Volts"}))
    with TdmsWriter(str(p)) as w:
        w.write_segment(objs)
    return p


def _abf_bytes(tmp_path: Path) -> Path:
    """An ABF signature over a header-shaped block. Only the MAGIC matters here: every reader
    under test is refusing before it decodes anything, and the question is what it says."""
    p = tmp_path / "src.abf"
    p.write_bytes(b"ABF " + b"\x00" * 4_096)
    return p


_FORMATS = {
    "hdf5": (_hdf5_bytes, ".h5", "an HDF5 file"),
    "tdms": (_tdms_bytes, ".tdms", "TDMS file"),
    "abf": (_abf_bytes, ".abf", "an Axon ABF recording"),
}


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
@pytest.mark.parametrize("worn", _EXTENSIONS)
def test_every_wrong_extension_names_the_format_the_bytes_really_are(tmp_path, fmt, worn):
    """Was: one of twelve combinations answered this way (HDF5 named .abf). The owner's own
    ABF named .h5 got h5py's "file signature not found", named .tdms got nptdms's "Segment
    does not start with b'TDSm'", and named .csv was described as a ragged event-table export
    — a binary recording called a text table."""
    builder, own, description = _FORMATS[fmt]
    src = builder(tmp_path)
    if worn == own:
        pytest.skip("the name matches the bytes: nothing to redirect")
    renamed = tmp_path / f"recording{worn}"
    shutil.copy(src, renamed)

    message = _refusal(renamed)
    assert "its bytes are" in message, f"{fmt} as {worn}: {message}"
    assert description in message, f"{fmt} as {worn} was not identified: {message}"
    assert own in message, f"{fmt} as {worn} does not say where it is read: {message}"
    # And none of the three decoders' internals, which io/errors.py forbids reaching the UI.
    for leak in ("file signature not found", "TDSm", "ragged table", "Errno"):
        assert leak not in message, f"{fmt} as {worn} leaked '{leak}': {message}"


def test_a_recording_under_its_own_extension_gains_no_redirect(tmp_path):
    """No false positive: a real HDF5 named .h5 opens, and a DAMAGED one still gets the
    reader's own diagnosis rather than being told to rename itself."""
    good = _hdf5_bytes(tmp_path)
    assert probe_path(good).meta.n_samples == 2_000

    truncated = tmp_path / "cut.h5"
    truncated.write_bytes(good.read_bytes()[:64])
    message = _refusal(truncated)
    assert "its bytes are" not in message, message
    assert "not a readable HDF5 file" in message, message


def test_a_cpya_abf_under_another_extension_is_named_without_a_false_promise(tmp_path):
    """CPYA is an ABF variant this build does not decode, so naming it must not come with
    "rename it and open it again" — the rename would not work."""
    p = tmp_path / "cpya.h5"
    p.write_bytes(b"CPYA" + b"\x00" * 4_096)
    message = _refusal(p)
    assert "CPYA" in message, message
    assert "does not read" in message, message
    assert "Re-export it from Clampex" in message, message


def test_plain_text_under_a_binary_extension_keeps_the_readers_own_words(tmp_path):
    """The stated BOUNDARY of the redirect, pinned so it is a decision and not a drift: text
    has no magic bytes, "these bytes are text" is not "these bytes are a trace table", and
    this app must not tell somebody to rename their notes .csv."""
    p = tmp_path / "notes.h5"
    p.write_bytes(b"a note to self, saved with the wrong extension\n" * 20)
    message = _refusal(p)
    assert "not a readable HDF5 file" in message, message
    assert "Rename it" not in message, message


# =================================================================================================
# IO3-10: an empty channel beside a full one is not an empty recording
# =================================================================================================

def _one_empty_channel_tdms(tmp_path: Path, n: int = 20_000) -> Path:
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    p = tmp_path / "one_never_filled.tdms"
    props = {"wf_increment": 2e-5, "unit_string": "Volts"}
    with TdmsWriter(str(p)) as w:
        w.write_segment([
            RootObject(properties={}),
            GroupObject("Untitled", properties={}),
            ChannelObject("Untitled", "ai0", np.linspace(-120.0, -90.0, n),
                          properties=dict(props)),
            ChannelObject("Untitled", "ai1", np.zeros(0), properties=dict(props)),
        ])
    return p


@pytest.mark.parametrize("door", [probe_path, open_document], ids=["probe", "load"])
def test_a_tdms_with_one_full_and_one_empty_channel_is_never_called_deletable(tmp_path, door):
    """Was: EmptyRecordingError — "the file decoded but holds no samples … this file is the
    expected result and can be deleted" — over 20,000 samples nptdms reads out of the same
    file. The HDF5 reader answers the identical input "signal channels have mismatched
    lengths (/current=20,000, /voltage=0)"."""
    p = _one_empty_channel_tdms(tmp_path)
    with pytest.raises(CorruptFileError) as exc:
        door(p)
    message = str(exc.value)
    assert "can be deleted" not in message, message
    assert "mismatched lengths" in message, message
    assert "ai0" in message and "20,000" in message, "the surviving channel must be named"
    assert "ai1" in message, "so must the empty one"


def test_a_genuinely_empty_tdms_keeps_the_honest_empty_sentence(tmp_path):
    """The boundary: a file where NOTHING was recorded is what the empty-recording door is
    for, deletion advice included — an acquisition armed and stopped writes exactly this."""
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    p = tmp_path / "armed_and_stopped.tdms"
    props = {"wf_increment": 2e-5, "unit_string": "Volts"}
    with TdmsWriter(str(p)) as w:
        w.write_segment([
            RootObject(properties={}),
            GroupObject("Untitled", properties={}),
            ChannelObject("Untitled", "ai0", np.zeros(0), properties=dict(props)),
            ChannelObject("Untitled", "ai1", np.zeros(0), properties=dict(props)),
        ])
    with pytest.raises(EmptyRecordingError) as exc:
        probe_path(p)
    assert "can be deleted" in str(exc.value)


def test_a_ragged_but_non_empty_tdms_still_loads_with_its_issue(tmp_path):
    """The other boundary, and a locked behaviour: 'example.tdms' in the real corpus has
    channels of 20,400 / 20,200 samples and MUST load, truncated to the common minimum with
    the differing-lengths issue shown."""
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    p = tmp_path / "ragged.tdms"
    props = {"wf_increment": 2e-5, "unit_string": "Volts"}
    with TdmsWriter(str(p)) as w:
        w.write_segment([
            RootObject(properties={}),
            GroupObject("Untitled", properties={}),
            ChannelObject("Untitled", "ai0", np.zeros(2_000), properties=dict(props)),
            ChannelObject("Untitled", "ai1", np.zeros(1_800), properties=dict(props)),
        ])
    doc = open_document(p)
    assert doc.meta.n_samples == 1_800
    assert "differing lengths" in _text(doc.meta.issues)
