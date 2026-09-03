from __future__ import annotations

import errno
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np

from ..core.document import Document
from ..core.types import SOFT_SAMPLE_CEILING, RecordingMeta, SampleRange
from .errors import (
    CorruptFileError,
    EmptyRecordingError,
    UnsupportedFormatError,
    empty_recording_message,
)

__all__ = [
    "Claim",
    "RecordingProbe",
    "ProgressFn",
    "CancelFn",
    "BaseReader",
    "READERS",
    "register",
    "read_magic",
    "select_reader",
    "probe_path",
    "open_document",
    "MIN_PLAUSIBLE_RATE_HZ",
    "filesystem_refusal",
    "foreign_format_redirect",
    "implausible_measured_rate_issue",
    "DowncastAudit",
    "downcast_needs_audit",
]


class Claim(Enum):
    """Separates "this file is MINE" from "I can DECODE it". Conflating them makes the CPYA file
    report the lie "no reader for .abf", sending the user hunting for a missing plugin."""

    NO = "no"
    WEAK = "weak"       # extension matches
    STRONG = "strong"   # magic bytes verified


@dataclass(frozen=True)
class RecordingProbe:
    """Cheap. Loads NO bulk data. MANDATORY: without it the BINDING ">15M samples: warn and offer
    a range" rule is unimplementable, because you cannot warn about a size you can only learn by
    loading — and by then you have already spent the memory and the wall clock you were avoiding.
    It also gives the progress bar a denominator.

    [VERIFIED] probing is essentially free in every format:
        606 MB ABF (50,495,999 samples): 0.001 s via parse_header() + get_signal_size()
        169 MB HDF5:                     0.010 s via .shape + /time[:1000]
    """

    meta: RecordingMeta

    @property
    def exceeds_ceiling(self) -> bool:
        """[VERIFIED] only 'March_25_ATP...0005_Process.abf' trips this: 50,495,999 samples =
        3.37x the ceiling. Every normal file is under it and must NEVER be nagged."""
        return self.meta.n_samples > SOFT_SAMPLE_CEILING


ProgressFn = Callable[[float, str], None]   # frac 0..1, message
CancelFn = Callable[[], bool]               # True => abort by raising LoadCancelled


class BaseReader(ABC):
    format_name: ClassVar[str]
    extensions: ClassVar[tuple[str, ...]]

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    @abstractmethod
    def sniff(cls, path: Path) -> Claim:
        """ONLY "is this file mine". NOT "can I decode it". Decoding failures are raised from
        probe()/load() as typed errors carrying a user-facing message."""

    @abstractmethod
    def probe(self) -> RecordingProbe:
        """Cheap; no bulk reads. Does NOT set ChannelInfo.is_dead (that needs sampled data;
        load() sets it). Raises UnsupportedFormatError for decodable-shaped-but-unsupported files."""

    @abstractmethod
    def load(self, sample_range: SampleRange | None = None,
             progress: ProgressFn | None = None,
             cancel: CancelFn | None = None) -> Document:
        """Eager, full-signal (or the chosen range). Runs OFF the Qt main thread.
        MUST: read in LOAD_CHUNK_SAMPLES chunks, call progress() and check cancel() between
        chunks (this is what makes Cancel real on the 606 MB file), return contiguous 1-D float32
        per channel, run detect_dead() per channel, and raise LoadCancelled if cancelled."""


READERS: list[type[BaseReader]] = []


def register(r: type[BaseReader]) -> type[BaseReader]:
    READERS.append(r)
    return r


def read_magic(path: Path, n: int = 8) -> bytes:
    """First n bytes, short-read tolerant. Every reader's sniff() dispatches on these, so the
    open-and-truncate is written once here rather than four times with four error behaviours.
    An unreadable file yields b"" -> every sniff falls back to its extension claim, and the real
    OSError surfaces from probe()/load() where it can carry a user-facing message."""
    try:
        with path.open("rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


# =================================================================================================
# The three rules every reader's refusal goes through
#
# Four readers refuse four formats, and a user who renames a file, chmods a directory or points
# the app at the wrong thing gets ONE condition described four ways unless the classification
# lives in one place. Each rule below was fixed at a single reader at least once before it was
# moved here; the sweep that followed found the next reader with the old answer. So they are
# stated once, above the readers, and every reader calls them.
# =================================================================================================

def _fd_exhaustion_errno(exc: BaseException | None) -> int | None:
    """``errno.EMFILE``/``ENFILE`` if ``exc`` — or anything in its cause/context chain —
    carries one, else None.

    The chain is walked rather than only the top exception read, because the decoding
    libraries hand the condition over in different wrappers: neo and nptdms raise the
    ``OSError(24)`` bare, h5py re-raises it as its own ``OSError`` subclass (errno kept),
    and a mid-parse failure can arrive wrapped in whatever the library tripped over next
    with the original error as ``__cause__``. The errno is the only durable fact about
    this condition, so it is looked for wherever it survived."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        eno = getattr(exc, "errno", None)
        if eno in (errno.EMFILE, errno.ENFILE):
            return eno
        exc = exc.__cause__ or exc.__context__
    return None


def filesystem_refusal(path: Path, exc: BaseException | None = None) -> str | None:
    """The sentence for "this cannot be opened at all", classified from the DISK — or None
    when the path is an ordinary readable file and the failure is about its CONTENT.

    ``exc`` is only a hint. The decoding libraries' own words ("[Errno 21] Is a directory",
    "'NoneType' object is not subscriptable", "Unable to open file …errno = 2") identify
    nothing a user can act on and ``io/errors.py`` promises they never reach the UI, so the
    cause is re-derived from the filesystem and the exception is used only to break ties.

    THE ORDER IS THE RULE, and it is why this is one function rather than four: the DURABLE
    fact wins. A mode-000 FOLDER raises ``PermissionError`` on open, so a classifier that asks
    "was it a permission error?" first calls it a permission problem — advice the user can act
    on, after which the folder is still a folder and the dialog changes its story. Being a
    folder is not fixable by ``chmod``, so it is reported first. Measured before this was
    shared: one folder-symlink got "cannot be read: permission denied" as ``.abf``/``.csv``
    and "is a folder, not a file" as ``.h5``/``.tdms``.

    One deliberate exception to "the disk decides": descriptor exhaustion (EMFILE/ENFILE),
    which no stat can see and only the errno names — see the last branch below.
    """
    if isinstance(exc, FileNotFoundError) or not path.exists():
        return (f"{path.name} is no longer there — it was moved, renamed or deleted, or it "
                f"lives on a drive that is not mounted.")
    if isinstance(exc, IsADirectoryError) or path.is_dir():
        return f"{path.name} is a folder, not a file."
    if isinstance(exc, PermissionError) or not os.access(path, os.R_OK):
        return f"{path.name} cannot be read: permission denied."
    # THE ONE ERRNO FAMILY THE DISK CANNOT ARBITRATE. Every branch above re-derives the cause
    # by statting the path, because the bytes on disk outlast whatever the library tripped
    # over. Out-of-descriptors is the exception that proves that rule: the failure lives in
    # the PROCESS (EMFILE) or the whole machine (ENFILE) — the file was never opened and its
    # bytes never read — and no stat of the path can reveal it, because stat costs no
    # descriptor and so passes every check above. Before this branch, the condition fell
    # through to the readers' CONTENT sentences: probing a healthy 9 MB ABF at the macOS
    # default 256-descriptor ceiling told its owner the file "is shorter than its header
    # promises … Recover the original file." — corruption wording, for a failure that never
    # touched the file. Classified from the exception chain, the one place the errno exists,
    # and placed LAST so the durable disk facts still win when both are true (an EMFILE on a
    # since-deleted file is still, most usefully, a gone file).
    eno = _fd_exhaustion_errno(exc)
    if eno is not None:
        scope = ("this app has hit its open-file limit" if eno == errno.EMFILE
                 else "this computer has hit its open-file limit, across all programs")
        return (f"{path.name} could not be opened because {scope} — too many files are open "
                f"at once (errno {eno}). The recording itself was never read and there is no "
                f"reason to think anything is wrong with it. Close some open documents or "
                f"other applications, or restart this app, then open the file again.")
    return None


# What the first bytes of a file say it REALLY is, and where that format is read. The magics are
# the same constants the readers' own sniff() methods dispatch on; they are gathered here because
# the redirect is about the file the user has, not about the reader that happened to be asked.
_FOREIGN_FORMATS: tuple[tuple[bytes, tuple[str, ...], str, str], ...] = (
    (b"\x89HDF\r\n\x1a\n", (".h5", ".hdf5"),
     "an HDF5 file, which this app reads natively",
     "Rename it with a .h5 extension and open it again."),
    (b"TDSm", (".tdms",),
     "a National Instruments TDMS file, which this app reads natively",
     "Rename it with a .tdms extension and open it again."),
    (b"ABF ", (".abf",),
     "an Axon ABF recording, which this app reads natively",
     "Rename it with a .abf extension and open it again."),
    (b"ABF2", (".abf",),
     "an Axon ABF recording, which this app reads natively",
     "Rename it with a .abf extension and open it again."),
    # CPYA is an ABF variant this build deliberately does not decode. Naming it is still the
    # honest answer — "these bytes are a CPYA ABF" is actionable where "not a readable HDF5
    # file (file signature not found)" is not — but it must not promise a rename will work.
    (b"CPYA", (".abf",),
     "an Axon ABF recording of the 'CPYA' variant, which this build does not read",
     "Re-export it from Clampex as the standard b'ABF ' float format to open it here."),
)



def foreign_format_redirect(path: Path) -> str | None:
    """"These bytes are not the format this name claims, and here is what they are" — or None
    when the bytes match the extension (so the real problem is the file, not its name).

    A renamed file is one condition with twelve spellings (four formats x three wrong
    extensions), and every reader used to answer it in its decoder's own words: the owner's
    ABF as ``.h5`` got h5py's "file signature not found" (corruption wording for an intact
    recording), as ``.tdms`` nptdms's "Segment does not start with b'TDSm'", and as ``.csv``
    "ragged table … This looks like an event-table export" — a binary ABF described as a text
    table, with the user sent to wait for a v2 feature. Exactly one combination, HDF5-as-.abf,
    had the sentence that says what the file IS. It is now the answer for all of them.

    THE BOUNDARY, stated because it is a choice and not an oversight: this answers only for
    the formats that carry MAGIC BYTES. A file of plain text under a binary extension gets no
    redirect, because "these bytes are text" is not "these bytes are a trace table" — the same
    sniff passes for a shopping list, and telling somebody to rename their notes ``.csv`` is a
    promise this app cannot keep. Those three combinations keep their readers' own wording,
    which is unhelpful but not false ("not a readable HDF5 file", "does not start with an ABF
    signature"). What is fixed here is every case where a decodable RECORDING was called
    damaged, or — the .csv case — confidently described as something it is not.

    Cheap and read-only: one open and 8 bytes. The three binary readers call it only on the
    refusal path; the text reader calls it before its own guards run, because those guards are
    the ones that produced the false description and they cannot be allowed to speak first.
    """
    head = read_magic(path, 8)
    if not head:
        return None
    suffix = path.suffix.lower()
    for magic, extensions, description, advice in _FOREIGN_FORMATS:
        if not head.startswith(magic):
            continue
        if suffix in extensions:
            return None            # the name is right; whatever is wrong is inside the file
        return (f"{path.name} is named {suffix or 'without an extension'} but its bytes are "
                f"{description}. {advice}")
    return None


# A sample rate MEASURED from a file's own time column is only as good as that column's UNITS,
# and no format this app reads declares them: an integer /time in microseconds and a float /time
# in seconds are the same numbers with the same dtype rules and a 10^6 difference in meaning.
# Below this rate the reading cannot be right — no recording this app opens takes more than a
# second per sample — so it is refused out loud instead of being reported as a measurement.
# Deliberately NOT applied to a DECLARED rate (the ABF header's Hz, TDMS's wf_increment): those
# carry their unit with them, so a strange value there is the file's statement, not our guess.
MIN_PLAUSIBLE_RATE_HZ = 1.0


def implausible_measured_rate_issue(*, source: str, dt: float, hz: float,
                                    stored_as: str | None = None) -> str:
    """The one sentence for "this time column cannot be in seconds".

    Says the measurement, says why it cannot be believed, says what was done instead, and does
    NOT guess the unit: microseconds and nanoseconds are both common in hand-written exporters
    and picking one would turn a visible refusal into an invisible 1,000x error.
    """
    dtype_clause = (f" {source} is stored as {stored_as}, and an integer clock is never "
                    f"seconds at a recording rate." if stored_as else "")
    return (f"{source} steps by {dt:,.6g} per sample. Read as seconds that is {hz:,.6g} Hz — "
            f"less than one sample per second, which no recording this app opens is."
            f"{dtype_clause} The unit of this time base is not declared in the file "
            f"(microseconds and nanoseconds are what LabVIEW and hand-written exporters "
            f"write), and this reader will not guess it: the sample rate is left unknown and "
            f"the x-axis is in samples, rather than every duration, dwell time and event rate "
            f"here being wrong by a factor of a million in silence. Re-export the time base in "
            f"seconds to analyse this recording here.")


# =================================================================================================
# The float64 -> float32 downcast, audited
#
# Every reader's contract is "contiguous 1-D float32 per channel" (BaseReader.load), and three
# of the four formats can carry values on disk that float32 cannot: HDF5 and TDMS store float64
# natively, and a text file can spell any decimal at all. A float64 sample whose magnitude is
# below float32's smallest subnormal (~1.4e-45) becomes EXACTLY 0.0 in that downcast — a number
# changing without anyone being told, the precise thing the HDF5 reader's integer-counts
# advisory (peak >= 2^24) already exists to say. The overflow direction of the same cast is loud
# twice over (numpy warns, and core.document reports every-sample-non-finite); underflow was the
# one representability failure of this cast that said nothing. [MEASURED] a valid HDF5 whose
# /current was all-nonzero float64 at 1e-310 loaded as identically zero with issues=() and a
# dead-hint of "constant 0.0" — a property no sample in the file has.
#
# The audit is one rule stated once, here, because the hole was one rule broken at three sites:
# each reader passes every source block and its float32 destination through observe(), and any
# destruction is reported through the same sentence. The ABF reader deliberately does NOT wire
# it: its disk dtypes are int16 and float32 (neo rescales with gains of order 0.1-1), so no
# value narrower than float32 can exist in the file to be destroyed — see abf.load.
# =================================================================================================

# float32's smallest positive subnormal: below half of this, a float64 rounds to exactly 0.0.
_FLOAT32_TINY: float = 1.401298464324817e-45   # == np.finfo(np.float32).smallest_subnormal


def downcast_needs_audit(dtype) -> bool:
    """True for source dtypes that can hold a nonzero magnitude float32 cannot: floats wider
    than 32 bits. Integers are >= 1 in magnitude, float16 bottoms out at ~6e-8, and a float32
    source casts to itself — none of them can be flushed to zero, so their reads skip the
    audit's per-chunk pass entirely."""
    dtype = np.dtype(dtype)
    return bool(np.issubdtype(dtype, np.floating)) and dtype.itemsize > 4


class DowncastAudit:
    """Watches one channel's chunked downcast to float32 and says so when it destroys samples.

    ``observe(src, dst)`` is called with each source block as read from the file and the
    float32 slice the app keeps; it counts the finite nonzero source samples that came out
    exactly 0.0. Cheap: two vectorised comparisons per chunk, the same order of work as the
    integer-peak scan the HDF5 reader already runs. ``issue()`` is the one sentence for the
    result, and ``dead_reason()`` is the honest replacement for detect_dead's
    "constant 0.0 (sampled N)" — which, on a flushed channel, asserts a property of the FILE
    that no sample in the file has.
    """

    __slots__ = ("channel", "source", "flushed", "nonzero")

    def __init__(self, channel: str, source: str) -> None:
        self.channel = channel      # how this reader names the channel to the user ("/current")
        self.source = source        # what the file stores ("float64")
        self.flushed = 0            # nonzero source samples that became exactly 0.0
        self.nonzero = 0            # nonzero source samples seen at all

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        src_nonzero = src != 0                       # NaN counts as nonzero; casts to NaN, not 0
        self.nonzero += int(np.count_nonzero(src_nonzero))
        self.flushed += int(np.count_nonzero(src_nonzero & (dst == np.float32(0.0))))

    def issue(self) -> str | None:
        """The advisory, or None when the cast was faithful (the overwhelmingly normal case)."""
        if self.flushed == 0:
            return None
        every = self.flushed == self.nonzero
        scope = (f"every one of its {self.nonzero:,} nonzero samples lies"
                 if every else f"{self.flushed:,} of its {self.nonzero:,} nonzero samples lie")
        flat = (" The loaded trace is identically zero — a fact about this conversion, not "
                "about the recording." if every else "")
        return (f"channel {self.channel} is stored as {self.source} and {scope} below "
                f"~{_FLOAT32_TINY:.2g}, the smallest magnitude float32 — the precision this "
                f"app stores traces in — can hold: those samples became exactly 0.0 when the "
                f"file was read.{flat} No physical signal in a standard unit is this small, "
                f"so the export that wrote this file is the place to look — re-export the "
                f"channel scaled into a larger unit to analyse the real values here.")

    def dead_reason(self) -> str | None:
        """What to put in ChannelInfo.dead_reason when the flushed channel loaded flat.

        Returned only when samples were flushed; the caller substitutes it for detect_dead's
        "constant 0.0 (sampled N)" so the File Info hint and the Issues box tell one story
        instead of the hint blaming the file for a flat line the cast produced."""
        if self.flushed == 0:
            return None
        return (f"constant 0.0 after conversion to float32 — the file's {self.source} samples "
                f"are nonzero but below float32 range (see Issues)")


# Reader modules are imported LAZILY inside the functions below, never at module scope: each
# reader does `from .base import BaseReader, register, ...`, so importing them here would be a
# circular import. Registration is a one-time side effect of that import.
_READER_MODULES: tuple[str, ...] = ("abf", "hdf5", "tdms", "text")
_readers_loaded = False


def _load_readers() -> None:
    """Import every reader module once, populating READERS via @register."""
    global _readers_loaded
    if _readers_loaded:
        return
    for name in _READER_MODULES:
        import_module(f"{__package__}.{name}")
    _readers_loaded = True


def select_reader(path: str | Path) -> type[BaseReader]:
    """STRONG beats WEAK. Raises a ReaderError carrying an honest message, never a bare
    "no reader" — ``CorruptFileError`` when the path cannot be opened at all (gone, a folder,
    unreadable) and ``UnsupportedFormatError`` when it opens and is not a format this build
    reads. Those are the same two types, for the same two conditions, that the four readers
    raise when one of them does claim the extension.

    Dispatch is by extension AND magic bytes, because neither alone is sufficient here:
      - magic alone cannot claim .csv/.txt (text has none -> WEAK on extension only);
      - extension alone cannot tell b'ABF ' from b'CPYA'. Both are .abf and both are OURS; the
        CPYA file is claimed STRONG precisely so probe() can tell the truth about it instead of
        the app reporting "no reader for .abf".
    """
    path = Path(path)
    try:
        _load_readers()
    except OSError as exc:
        # At exactly zero spare descriptors the lazy import itself is the first open that
        # fails, and it used to escape as a raw "[Errno 24] Too many open files:
        # …/nanoscope/io/hdf5.py" — a traceback about OUR module, for a condition of the
        # user's session, from a function whose own docstring promises "an honest message,
        # never a bare" one. Same classification as every reader's refusal; anything the
        # classifier does not recognise (a genuinely broken install) is not masked.
        message = filesystem_refusal(path, exc)
        if message is not None:
            raise CorruptFileError(message) from exc
        raise

    strong: list[type[BaseReader]] = []
    weak: list[type[BaseReader]] = []
    for reader in READERS:
        claim = reader.sniff(path)
        if claim is Claim.STRONG:
            strong.append(reader)
        elif claim is Claim.WEAK:
            weak.append(reader)

    if strong:
        return strong[0]
    if weak:
        return weak[0]

    # NOTHING CLAIMED THIS PATH — and that is exactly where the two rules above used to stop.
    #
    # A reader claims by EXTENSION, so every refusal that goes through a reader (.abf, .h5,
    # .hdf5, .tdms, .csv, .txt) already gets `filesystem_refusal` and `foreign_format_redirect`
    # from that reader's own `_refusal`. This fallthrough is the one door the rules did not
    # reach, and it is not a rare one: it is where a folder lands, where a file whose
    # permissions deny reading lands, where a recording that was renamed to `.dat`/`.bin`/no
    # extension at all lands — anything the user points the app at whose name no reader knows.
    # Measured before this block existed, all four answers were wrong in the same way — they
    # described the FORMAT of something the app had never managed to look at:
    #
    #   a folder named `recordings`   -> "not a recognised recording. Its extension ('none')
    #                                     matches no reader and its first bytes (b'') match no
    #                                     known format."   (a folder has no first bytes; the
    #                                     b'' is read_magic swallowing the OSError)
    #   the same folder as `.h5`      -> "recordings.h5 is a folder, not a file."
    #   a mode-000 file `locked.dat`  -> the format sentence again, for bytes never read
    #   a missing `gone.dat`          -> "gone.dat: no such file."
    #   the same file as `gone.h5`    -> "gone.h5 is no longer there — it was moved, renamed
    #                                     or deleted, or it lives on a drive that is not
    #                                     mounted."
    #   an HDF5 recording as `.dat`   -> "its first bytes (b'\\x89HDF') match no known format"
    #
    # The last one is the sharpest: `_FOREIGN_FORMATS`, thirty lines up in this same module,
    # knows perfectly well that b'\x89HDF' IS a known format. One sentence contradicting one
    # table in one file is the "two surfaces never disagree about one fact" rule broken at the
    # shortest possible range.
    #
    # So the rules are applied HERE too, in the same order they carry everywhere else: the
    # DURABLE disk fact first (a folder is still a folder after any chmod, and no rename makes
    # a missing file present), then what the bytes really are, and only then — for a file that
    # exists, opens, and genuinely is not a format this build reads — the format sentence.
    # `CorruptFileError` rather than `UnsupportedFormatError` for the first, because that is
    # the type the four readers already raise for these same three conditions, and a condition
    # that changes its exception class depending on the extension is the same defect one layer
    # down. This also retires the module's second spelling of "the file is not there": one
    # fact, one sentence, whatever the name on it.
    gone_locked_or_folder = filesystem_refusal(path)
    if gone_locked_or_folder is not None:
        raise CorruptFileError(gone_locked_or_folder)

    # The format sentence below quotes the file's first bytes, so it must not speak unless
    # those bytes were actually READ: `read_magic` swallows every OSError to b"", which is
    # honest input to a sniff (fall back to the extension claim) and a lie in a message
    # ("its first bytes (b'') match no known format" for bytes nobody saw). One direct open
    # classifies the failure — at the descriptor ceiling this is where an unclaimed
    # extension lands, since EMFILE is invisible to the stat-based rule above.
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        unopenable = filesystem_refusal(path, exc)
        if unopenable is not None:
            raise CorruptFileError(unopenable) from exc
        raise CorruptFileError(
            f"{path.name} cannot be opened: {exc.strerror or exc}.") from exc

    renamed = foreign_format_redirect(path)
    if renamed is not None:
        raise UnsupportedFormatError(renamed)

    known = sorted({ext for r in READERS for ext in r.extensions})
    magic = read_magic(path, 4)
    raise UnsupportedFormatError(
        f"{path.name}: not a recognised recording. Its extension "
        f"('{path.suffix or 'none'}') matches no reader and its first bytes ({magic!r}) match no "
        f"known format. This build reads: {', '.join(known)}.")


def probe_path(path: str | Path) -> RecordingProbe:
    """Probe ``path``, refusing a recording with no samples in it.

    The load half of this rule lives in ``TraceDocument.__post_init__`` (the one door
    every producer uses, segments included). A probe builds no document, so the same rule
    has to be stated once more here — and it is worth stating: the probe is what the File
    dialog and the >15 M-sample range prompt read, so without it the app would offer to
    load, and put a progress bar in front of, a file it is about to refuse. Both HDF5 and
    TDMS were measured probing at ``n_samples=0`` with ``issues=()``.
    """
    path = Path(path)
    probe = select_reader(path)(path).probe()
    if probe.meta.n_samples <= 0:
        raise EmptyRecordingError(empty_recording_message(path.name))
    return probe


def open_document(path: str | Path, sample_range: SampleRange | None = None,
                  progress: ProgressFn | None = None,
                  cancel: CancelFn | None = None) -> Document:
    """-> Document, NOT -> Recording. Not every openable file is a trace."""
    path = Path(path)
    return select_reader(path)(path).load(sample_range, progress, cancel)
