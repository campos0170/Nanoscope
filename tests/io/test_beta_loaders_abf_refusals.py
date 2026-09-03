"""LDR-01: an ABF that cannot be opened must refuse with a sentence, never a neo internal.

MEASURED on the tree as found, via File > Open: a 0-byte .abf and an HDF5 file misnamed
.abf both put "'NoneType' object is not subscriptable" in the "Could not open file"
dialog, a deleted .abf put "[Errno 2] No such file or directory: '/abs/path'" there, and
a cut one "mmap length is greater than file size". io/errors.py promises "Carries a
USER-FACING message. Never leak a neo/h5py/nptdms traceback to the UI", and the identical
inputs with a .h5 or .csv extension already came back as sentences. The leak was
AbfReader._open_reader (and the mid-read wrap in load()) stuffing str(exc) of ANY failure
into CorruptFileError; the fix classifies the refusal from the bytes on disk instead.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.io.base import open_document, probe_path
from nanoscope.io.errors import CorruptFileError, ReaderError, UnsupportedFormatError

# The verbatim fragments the defect leaked. None of them may ever reach a refusal again.
_INTERNALS = ("NoneType", "not subscriptable", "[Errno", "Traceback", "mmap", "unpack")


def _refusal_text(path: Path, expected=ReaderError, via=probe_path) -> str:
    with pytest.raises(expected) as exc:
        via(path)
    message = str(exc.value)
    for fragment in _INTERNALS:
        assert fragment not in message, f"internal leaked to the UI: {message!r}"
    assert path.name in message, f"the refusal must name the file: {message!r}"
    return message


def test_a_missing_abf_names_the_moved_renamed_deleted_cause(tmp_path):
    """Was: "gone.abf: [Errno 2] No such file or directory: '/abs/path'"."""
    gone = tmp_path / "gone.abf"               # never created
    message = _refusal_text(gone, CorruptFileError)
    assert "moved, renamed or deleted" in message
    assert str(tmp_path) not in message, "no absolute paths in the dialog"


def test_a_zero_byte_abf_is_called_empty_not_a_typeerror(tmp_path):
    """Was: "empty.abf: 'NoneType' object is not subscriptable"."""
    empty = tmp_path / "empty.abf"
    empty.write_bytes(b"")
    message = _refusal_text(empty, CorruptFileError)
    assert "0 bytes" in message
    assert "recover the original" in message


def test_an_hdf5_file_misnamed_abf_is_identified_and_redirected(tmp_path):
    """The bytes say \\x89HDF — a format this app reads natively. Say so and point at the
    rename, instead of calling a perfectly good file corrupt (or 'NoneType')."""
    liar = tmp_path / "liar.abf"
    with h5py.File(liar, "w") as h:
        h["current"] = np.zeros(100)
    message = _refusal_text(liar, UnsupportedFormatError)
    assert "HDF5" in message
    assert ".h5" in message, "the refusal must point at the rename that fixes it"


def test_a_cut_abf_with_a_real_signature_is_called_damaged(tmp_path):
    """Was: "mmap length is greater than file size" / "unpack requires a buffer of 32
    bytes" — a truncated ABF must be blamed on the copy, not on struct internals."""
    cut = tmp_path / "cut.abf"
    cut.write_bytes(b"ABF " + b"\x00" * 200)
    message = _refusal_text(cut, CorruptFileError)
    assert "interrupted copy" in message


def test_random_bytes_named_abf_are_refused_as_not_an_abf(tmp_path):
    noise = tmp_path / "noise.abf"
    noise.write_bytes(b"time,current\n0.0,-50.0\n" * 20)
    message = _refusal_text(noise, CorruptFileError)
    assert "not an ABF file" in message


def test_a_folder_named_abf_is_a_folder(tmp_path):
    folder = tmp_path / "folder.abf"
    folder.mkdir()
    assert "is a folder, not a file" in _refusal_text(folder, CorruptFileError)


def test_the_load_door_refuses_with_the_same_sentence_as_the_probe(tmp_path):
    """probe() and load() share _open_reader; the user must not be able to tell which one
    refused from the wording (the same rule empty_recording_message states)."""
    empty = tmp_path / "empty.abf"
    empty.write_bytes(b"")
    via_probe = _refusal_text(empty, CorruptFileError, via=probe_path)
    via_load = _refusal_text(empty, CorruptFileError, via=open_document)
    assert via_probe == via_load


def test_the_cpya_refusal_is_not_regressed(tmp_path):
    """The CPYA message predates this fix and is good; the classifier must not eat it."""
    cpya = tmp_path / "variant.abf"
    cpya.write_bytes(b"CPYA" + b"\x00" * 200)
    message = _refusal_text(cpya, UnsupportedFormatError)
    assert "CPYA" in message
    assert "Re-export from Clampex" in message
