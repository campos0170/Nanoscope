from __future__ import annotations

__all__ = [
    "ReaderError",
    "UnsupportedFormatError",
    "CorruptFileError",
    "EmptyRecordingError",
    "LoadCancelled",
    "empty_recording_message",
]

# This module imports NOTHING. That is deliberate and load-bearing: it lets
# ``core.document`` -- which every reader's output passes through, and which is where
# EmptyRecordingError is actually raised -- import from here without creating the
# ``io -> core.document -> io`` cycle that ``io/base.py`` already has to work around.


class ReaderError(Exception):
    """Carries a USER-FACING message. Never leak a neo/h5py/nptdms traceback to the UI."""


class UnsupportedFormatError(ReaderError):
    ...


class CorruptFileError(ReaderError):
    ...


class EmptyRecordingError(ReaderError):
    """The file decoded, and there is nothing in it: ``n_samples == 0``.

    A ``ReaderError`` (not a ValueError) so the load worker reports it as the readable
    sentence it carries rather than as "ValueError: ...", and so every existing
    ``except ReaderError`` in the open pipeline already handles it.

    Its own category rather than :class:`CorruptFileError` because the file is not
    necessarily damaged -- an acquisition that was armed and stopped writes a
    structurally perfect HDF5 or TDMS with zero samples in it -- and telling a user
    their file is corrupt when it is merely empty sends them looking for the wrong
    problem."""


class LoadCancelled(ReaderError):
    ...


def empty_recording_message(name: str) -> str:
    """The ONE sentence for "this file holds no samples".

    Two places raise :class:`EmptyRecordingError` — the probe (``io.base.probe_path``)
    and the load (``core.document.TraceDocument``) — and the user cannot tell which of
    them refused, so they must not be able to tell from the wording either."""
    return (f"{name or 'this recording'}: the file decoded but holds no samples, so there "
            f"is nothing to plot, measure or detect. If the acquisition was armed and "
            f"stopped without recording, this file is the expected result and can be "
            f"deleted.")
