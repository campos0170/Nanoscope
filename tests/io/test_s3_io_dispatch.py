"""Third-sweep loader defects, part 4: the door NO reader claims.

This file exists because of IO3-09. That defect itself lives in ``gui/main_window.py`` (the
restore banner throws the reader's sentence away and calls every failure "could not be
found"), which is not this group's to edit — but re-verifying it forced the question one layer
down: *does the io layer itself ever describe a present file as missing, or an unread file by
its format?* It did, at one door, and that door is in ``io/base.py``.

``select_reader`` dispatches by extension. Every refusal that reaches a reader therefore goes
through that reader's ``_refusal``, which calls the two shared rules — ``filesystem_refusal``
(IO3-03) and ``foreign_format_redirect`` (IO3-08). The FALLTHROUGH, taken whenever no reader
claims the extension, called neither. So the two rules held for ``.abf``/``.h5``/``.hdf5``/
``.tdms``/``.csv``/``.txt`` and stopped dead at ``.dat``, ``.bin``, ``.rec`` and no extension
at all — which is not an exotic corner: it is a folder the user picked by mistake, a file whose
permissions deny reading, a recording renamed by an acquisition script, and every stale project
or recent-files entry pointing at any of them.

Measured on the tree as it stood before this file was written:

    a folder named `recordings`     "not a recognised recording. Its extension ('none')
                                     matches no reader and its first bytes (b'') match no
                                     known format."
    the same folder as `.h5`        "recordings.h5 is a folder, not a file."
    a mode-000 `locked.dat`         the format sentence again — for bytes never read
    a missing `gone.dat`            "gone.dat: no such file."
    the same file as `gone.h5`      "gone.h5 is no longer there — it was moved, renamed or
                                     deleted, or it lives on a drive that is not mounted."
    an HDF5 recording as `.dat`     "its first bytes (b'\\x89HDF') match no known format"

The last line is the whole batch theme in one sentence: ``_FOREIGN_FORMATS``, thirty lines up
in the same module, lists ``b"\\x89HDF\\r\\n\\x1a\\n"`` as a format this app reads natively. One
file disagreeing with itself about one fact.

The tests below pin the rule, not the repro: every disk condition is asserted across a reader
extension AND a non-reader extension AND no extension, through BOTH doors (probe and load),
so a future reader added without wiring is caught by the parametrisation rather than by the
next sweep.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.io.base import open_document, probe_path
from nanoscope.io.errors import CorruptFileError, ReaderError, UnsupportedFormatError

# Names no reader claims, beside one that every reader-claimed path already covered. The point
# of running the SAME assertion over both groups is that the answer must not depend on which
# group the name falls into — that dependence WAS the defect.
_CLAIMED = (".abf", ".h5", ".tdms", ".csv")
_UNCLAIMED = (".dat", ".bin", "")

as_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="running as root: chmod 000 does not deny anything")


def _both_doors(path: Path) -> list[tuple[type, str]]:
    """(exception type, message) from probe AND load. A user meets the probe in the file
    dialog and the load on open/restore; the sweep-2 work established they must not be able
    to tell which one refused from the wording, so both are measured everywhere here."""
    out: list[tuple[type, str]] = []
    for call in (probe_path, open_document):
        try:
            call(path)
        except ReaderError as exc:
            out.append((type(exc), str(exc)))
        else:                                    # pragma: no cover - a failing assert reports it
            out.append((type(None), "OPENED"))
    return out


# =================================================================================================
# The durable disk facts: gone, folder, unreadable
# =================================================================================================

@pytest.mark.parametrize("suffix", _CLAIMED + _UNCLAIMED)
def test_a_folder_is_a_folder_whatever_extension_its_name_carries(tmp_path, suffix):
    """A folder is not a format problem, and it never becomes one by being called `.dat`.

    Before: `recordings.h5` was "a folder, not a file" while `recordings.dat` and plain
    `recordings` were "not a recognised recording … its first bytes (b'') match no known
    format" — a sentence about bytes a folder does not have (the b'' is ``read_magic``
    swallowing the OSError), sending the user to convert or re-export a directory."""
    folder = tmp_path / f"recordings{suffix}"
    folder.mkdir()
    for exc_type, message in _both_doors(folder):
        assert exc_type is CorruptFileError, message
        assert message == f"recordings{suffix} is a folder, not a file."


@pytest.mark.parametrize("suffix", _CLAIMED + _UNCLAIMED)
def test_a_missing_file_gets_one_sentence_whatever_extension_it_wore(tmp_path, suffix):
    """One fact, one sentence. ``io/base`` used to hold a SECOND spelling of "it is not there"
    — a terse "gone.dat: no such file." reachable only through the unclaimed-extension door —
    beside the one ``filesystem_refusal`` gives every reader. The user cannot tell which door
    they came through, so they must not be able to tell from the words."""
    gone = tmp_path / f"gone{suffix}"
    for exc_type, message in _both_doors(gone):
        assert exc_type is CorruptFileError, message
        assert message == (f"gone{suffix} is no longer there — it was moved, renamed or "
                           f"deleted, or it lives on a drive that is not mounted.")
        assert "no such file" not in message.lower()


@as_root
@pytest.mark.parametrize("suffix", _CLAIMED + _UNCLAIMED)
def test_a_file_the_app_may_not_read_says_so_instead_of_describing_its_format(tmp_path, suffix):
    """Describing the FORMAT of a file whose bytes were never readable is a claim the app
    cannot support: ``read_magic`` returns b"" on any OSError, so `locked.dat` was refused as
    "its first bytes (b'') match no known format" — of a perfectly ordinary recording."""
    locked = tmp_path / f"locked{suffix}"
    locked.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    os.chmod(locked, 0o000)
    try:
        for exc_type, message in _both_doors(locked):
            assert exc_type is CorruptFileError, message
            assert message == f"locked{suffix} cannot be read: permission denied."
            assert "match no known format" not in message
    finally:
        os.chmod(locked, 0o644)


@as_root
def test_the_folder_answer_still_wins_over_permission_at_the_unclaimed_door(tmp_path):
    """The ORDER is the rule, and it has to be the same order here as in every reader: opening
    a mode-000 DIRECTORY raises ``PermissionError``, so a classifier that asks about
    permissions first sends the user to ``chmod`` — after which the folder is still a folder
    and the message changes its story. Being a folder is not fixable by ``chmod``."""
    folder = tmp_path / "locked_folder"           # no extension: the fallthrough door
    folder.mkdir()
    os.chmod(folder, 0o000)
    try:
        first = [m for _, m in _both_doors(folder)]
    finally:
        os.chmod(folder, 0o755)
    relaxed = [m for _, m in _both_doors(folder)]
    assert set(first) == {"locked_folder is a folder, not a file."}
    assert relaxed == first, "the answer changed once the user acted on it"


# =================================================================================================
# What the bytes really are
# =================================================================================================

def _write_hdf5(path: Path, n: int = 500) -> None:
    with h5py.File(path, "w") as fh:
        fh["current"] = np.linspace(-120.0, -118.0, n).astype(np.float32)
        fh["time"] = (np.arange(n) / 10_000.0).astype(np.float64)


@pytest.mark.parametrize("suffix", _UNCLAIMED)
def test_a_recording_renamed_beyond_every_reader_extension_is_still_named(tmp_path, suffix):
    """IO3-08 generalised to the door it did not reach. The redirect was wired into the four
    readers' refusal paths, so it answered for the twelve reader-extension combinations and
    for none of the names outside that set — even though those are the ones an acquisition
    script or an archive tool actually produces."""
    src = tmp_path / "real.h5"
    _write_hdf5(src)
    renamed = tmp_path / f"renamed{suffix}"
    shutil.copy2(src, renamed)

    for exc_type, message in _both_doors(renamed):
        assert exc_type is UnsupportedFormatError, message
        assert message == (
            f"renamed{suffix} is named {suffix or 'without an extension'} but its bytes are "
            f"an HDF5 file, which this app reads natively. Rename it with a .h5 extension "
            f"and open it again.")
        # The sentence it replaced asserted the opposite of what _FOREIGN_FORMATS says.
        assert "match no known format" not in message


def test_a_cpya_abf_outside_every_reader_extension_is_named_without_a_false_promise(tmp_path):
    """CPYA is an ABF variant this build does not decode, so naming it must NOT come with
    "rename it and open it again" — the boundary the IO3-08 fix drew, held at the new door."""
    cpya = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0011_Process_-200 mV_30-35.abf")
    if not cpya.exists():                          # pragma: no cover - corpus-dependent
        pytest.skip("the CPYA corpus file is not on this machine")
    renamed = tmp_path / "renamed.dat"
    shutil.copy2(cpya, renamed)
    for exc_type, message in _both_doors(renamed):
        assert exc_type is UnsupportedFormatError, message
        assert "'CPYA' variant, which this build does not read" in message
        assert "Re-export it from Clampex" in message
        assert "Rename it with a .abf extension" not in message


def test_plain_text_outside_every_reader_extension_keeps_the_generic_sentence(tmp_path):
    """THE BOUNDARY, pinned. ``foreign_format_redirect`` answers only for formats that carry
    magic bytes; text carries none, and "these bytes are text" is not "these bytes are a trace
    table". A shopping list named `.dat` must not be sent to be renamed `.csv`, so it keeps the
    honest generic sentence — which is the same one it had before this change."""
    notes = tmp_path / "notes.dat"
    notes.write_text("milk\neggs\nbread\n", encoding="utf-8")
    for exc_type, message in _both_doors(notes):
        assert exc_type is UnsupportedFormatError, message
        assert "not a recognised recording" in message
        assert "matches no reader" in message
        assert "This build reads: " in message
        assert "Rename it" not in message


def test_an_unrecognised_binary_still_gets_the_generic_sentence_with_its_real_bytes(tmp_path):
    """The fallthrough is still reachable and still true. Its sentence is only wrong when it
    speaks for a file that was never opened or whose bytes ARE known; for a readable file of
    genuinely unknown format it is the right answer, and it must keep quoting the bytes it
    actually read rather than the b'' of a failed open."""
    junk = tmp_path / "scope_capture.bin"
    junk.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 128)
    for exc_type, message in _both_doors(junk):
        assert exc_type is UnsupportedFormatError, message
        assert "scope_capture.bin: not a recognised recording." in message
        assert r"first bytes (b'\x7fELF')" in message


# =================================================================================================
# Nothing that already worked moved
# =================================================================================================

def test_a_recording_under_its_own_extension_still_just_opens(tmp_path):
    """The rules run in the fallthrough only — after STRONG/WEAK dispatch — so a file a reader
    claims never reaches them and nothing about the ordinary path changed."""
    good = tmp_path / "fine.h5"
    _write_hdf5(good, 800)
    probe = probe_path(good)
    assert probe.meta.n_samples == 800
    doc = open_document(good)
    assert doc.meta.n_samples == 800


def test_the_two_doors_never_disagree_across_the_whole_matrix(tmp_path):
    """The cross-cutting assertion, stated once over every condition x every naming: the file
    dialog's probe and the open/restore load give the SAME type and the SAME sentence. A
    surface that refuses one way when browsing and another way when opening is the same defect
    as two extensions disagreeing, one axis over."""
    src = tmp_path / "src.h5"
    _write_hdf5(src)

    subjects: list[Path] = []
    for suffix in _CLAIMED + _UNCLAIMED:
        folder = tmp_path / f"dir{suffix}"
        folder.mkdir()
        subjects.append(folder)
        subjects.append(tmp_path / f"absent{suffix}")
        renamed = tmp_path / f"copy{suffix}"
        shutil.copy2(src, renamed)
        subjects.append(renamed)

    for path in subjects:
        (probe_answer, load_answer) = _both_doors(path)
        assert probe_answer == load_answer, f"{path.name}: probe {probe_answer} vs load {load_answer}"


def test_no_refusal_at_this_door_calls_a_present_file_missing(tmp_path):
    """The io half of IO3-09's invariant, pinned here so it cannot rot while the GUI half is
    fixed. Whatever is wrong with a file that IS on disk, the reader layer never says it is
    not — the restore banner's "could not be found" has no counterpart down here."""
    src = tmp_path / "src.h5"
    _write_hdf5(src)

    present: list[Path] = []
    for suffix in _CLAIMED + _UNCLAIMED:
        folder = tmp_path / f"d{suffix}"
        folder.mkdir()
        present.append(folder)
        renamed = tmp_path / f"c{suffix}"
        shutil.copy2(src, renamed)
        present.append(renamed)
    notes = tmp_path / "notes.dat"
    notes.write_text("not a recording\n", encoding="utf-8")
    present.append(notes)

    for path in present:
        for _, message in _both_doors(path):
            lowered = message.lower()
            assert "no longer there" not in lowered, f"{path.name}: {message}"
            assert "no such file" not in lowered, f"{path.name}: {message}"
            assert "could not be found" not in lowered, f"{path.name}: {message}"


# =================================================================================================
# The guide describes the thing actually used
# =================================================================================================

_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "USER_GUIDE.md"


def test_the_guide_promises_the_redirect_the_app_actually_gives(tmp_path):
    """§2 said the redirect fires for "an HDF5 recording named `.abf`, or an ABF named `.csv`"
    — the reader-extension combinations, which is all it used to cover. It now covers every
    name, so the guide names the unclaimed ones too. Measured here before it is read, because
    a guide that is checked only against itself is a document, not a promise."""
    src = tmp_path / "real.h5"
    _write_hdf5(src)
    renamed = tmp_path / "acquired.dat"
    shutil.copy2(src, renamed)
    with pytest.raises(UnsupportedFormatError) as excinfo:
        open_document(renamed)
    assert "its bytes are an HDF5 file" in str(excinfo.value)

    guide = _GUIDE.read_text(encoding="utf-8")
    assert "`.dat`, `.bin`, or no" in guide
    # The boundary is still stated, and still true (pinned above by the notes.dat test).
    assert "these bytes are text" in guide


def test_the_guide_quotes_the_three_disk_sentences_the_app_really_uses(tmp_path):
    """Each fragment below is measured from a real refusal first, then required in the guide,
    so the two cannot drift: if a sentence is reworded in ``io/base.filesystem_refusal`` this
    test fails on the guide rather than the guide quietly becoming wrong."""
    gone = tmp_path / "absent.dat"
    folder = tmp_path / "adir.dat"
    folder.mkdir()

    measured = []
    for path in (gone, folder):
        with pytest.raises(CorruptFileError) as excinfo:
            open_document(path)
        measured.append(str(excinfo.value))

    # Markdown hard-wraps at 100 columns, so the guide is compared as flowed text: the
    # sentence the user reads is the same one whether or not a newline lands inside it.
    guide = " ".join(_GUIDE.read_text(encoding="utf-8").split())
    assert "is no longer there — it was moved, renamed or deleted" in measured[0]
    assert "is no longer there — it was moved, renamed or deleted, or it" in guide
    assert "is a folder, not a file" in measured[1]
    assert "is a folder, not a file" in guide
    assert "cannot be read: permission denied" in guide
