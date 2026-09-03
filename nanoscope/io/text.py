from __future__ import annotations

import codecs
import csv
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.document import Document, TraceDocument
from ..core.types import (
    LOAD_CHUNK_SAMPLES,
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
    SampleRange,
    detect_dead,
)
from .base import (
    MIN_PLAUSIBLE_RATE_HZ,
    BaseReader,
    Claim,
    DowncastAudit,
    ProgressFn,
    CancelFn,
    RecordingProbe,
    filesystem_refusal,
    foreign_format_redirect,
    implausible_measured_rate_issue,
    register,
)
from .contract import (
    ReaderReport,
    fallback_encoding_note,
    lossy_encoding_issue,
    rate_disagrees_issue,
    rate_interval_changes_issue,
    short_window_issue,
    window_loaded,
    window_t0,
)
from .errors import CorruptFileError, LoadCancelled, UnsupportedFormatError

FORMAT_NAME = "Text trace"

# UNITS are never inferred from a column name (see §12.4 / [VERIFIED]: the same `time,current`
# header carries -48.83 pA (negative) in one file and +0.295 nA (positive) in another) — raw_units
# stays "" and the user sets units in the Calibration panel. KIND, however, IS taken from an explicit
# column name, exactly as the HDF5 reader maps a `/current` dataset to CURRENT: a column literally
# named "current" is the ionic-current signal regardless of its (unknown) units, and labelling it so
# is what lets event detection run on a CSV. Names that match nothing stay UNKNOWN.
_UNITS_NOTE = "Units are not declared in this file. Set them in the Calibration panel."

_NAME_KIND_TOKENS: tuple[tuple[tuple[str, ...], ChannelKind], ...] = (
    (("conductance", "cond"), ChannelKind.CONDUCTANCE),   # before 'current' (neither substrings the other, but explicit)
    (("current", "curr", "ionic"), ChannelKind.CURRENT),
    (("voltage", "bias", "command"), ChannelKind.VOLTAGE),
)


def _kind_from_name(name: str) -> ChannelKind:
    """Kind from an explicit signal column name (units are left unknown). Conservative: only clear
    names map; anything else is UNKNOWN so we never mislabel an arbitrary column."""
    n = (name or "").strip().lower()
    for tokens, kind in _NAME_KIND_TOKENS:
        if any(t in n for t in tokens):
            return kind
    return {"i": ChannelKind.CURRENT, "v": ChannelKind.VOLTAGE,
            "g": ChannelKind.CONDUCTANCE}.get(n, ChannelKind.UNKNOWN)

# The ONE place a filename is consulted, and only when no VOLTAGE channel exists (which — since
# every text signal column is kind UNKNOWN — is always). Single match only; 0 or >1 => None.
_MV_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*mV", re.IGNORECASE)

# How many lines the cheap probe scan reads. Raggedness in the user's event-table exports first
# appears at row 2 (-1000 mV) and row 17 (-500 mV) — both well inside this window.
_SCAN_LINES = 1000

# The head block decoded strictly to settle the file's encoding, and the tail block read back to
# ask whether the interval measured at the START of the file still holds at its END. Both are one
# seek and one read: the tail check is what makes the sample rate an answer about the whole file
# instead of about its first 1000 rows. [MEASURED] on the owner's 332 MB / 9,800,724-row
# 'nisin_fibrin_trace.csv': probe 0.9 ms -> 1.5 ms, full load unchanged at 0.83 s.
_ENCODING_PROBE_BYTES = 1 << 16
_TAIL_PROBE_BYTES = 1 << 16
_TAIL_PROBE_ROWS = 200

# Bytes per code unit, for the tail seek. A UTF-16 file has no arbitrary seek offset — landing
# between the two bytes of a code unit decodes the rest of the block as garbage — but it does have
# an ALIGNED one, and the alignment is known: the BOM is two bytes, so every code unit begins at an
# even offset. Skipping the tail for those files (which is what an encoding whitelist did) left the
# whole of guard #4 dead on the one encoding Windows instrument software writes by default —
# [MEASURED]: the C6 fixture saved as UTF-16 reported a 0.2000 s recording for a 0.6490 s one,
# 3.25x wrong, issues=(), while its byte-identical UTF-8 twin raised two loud issues.
_TAIL_UNIT_BYTES: dict[str, int] = {"utf-8": 1, "utf-8-sig": 1, "latin-1": 1, "utf-16": 2}

# What one character of the detected encoding costs on disk, for the row-count estimate. The
# str.encode() spelling matters: "utf-16" prepends a BOM to EVERY call, which would add two
# phantom bytes to every row.
_LENGTH_CODEC = {"utf-8-sig": "utf-8", "utf-16": "utf-16-le"}

# A sample interval measured at the end of the file may differ from the one at the start by this
# much before it is a different rate rather than round-off in the time column.
_INTERVAL_TOL = 1e-2
# The whole-file duration may differ from the one the measured rate implies by this much before
# the rate is wrong for this file. Round-off over a 9.8 M-row file accumulates to ~1e-16 relative.
_SPAN_TOL = 1e-3


@dataclass(frozen=True)
class _Scan:
    """Result of the cheap header scan, shared by probe() and load() so the two never disagree
    about delimiter, header presence, column layout, rate, encoding or t0."""
    delimiter: str
    has_header: bool
    ncols: int
    names: tuple[str, ...]          # per-column display names (header text, or col0/col1/...)
    t0_s: float
    sample_rate_hz: float | None       # None: the column's own units make it unreadable as one
    rate_provenance: Provenance
    est_n_samples: int
    header_bytes: int               # bytes consumed by a header line (0 if none)
    notes: tuple[str, ...]
    issues: tuple[str, ...]
    nominal_voltage_mV: float | None
    voltage_provenance: Provenance
    encoding: str                   # what pandas must be told, NOT a guess made twice
    last_t_s: float | None          # time of the file's final row, read by one tail seek
    estimate_note: str              # probe-only: n_samples is a guess until the file is read


@contextmanager
def _readable(path: Path):
    """Turn "this file cannot be opened" into a sentence, like the other three readers do.

    The text reader was the only one that converted no ``OSError``: a ``.csv`` that had been
    moved, one on an unmounted share, one whose permissions were wrong and a DIRECTORY named
    ``*.csv`` all escaped as ``FileNotFoundError`` / ``PermissionError`` / ``IsADirectoryError``
    with an absolute path, straight out to the load worker's ``except Exception``. The same three
    inputs with a ``.h5`` extension already came back as a ``CorruptFileError`` with a sentence.

    The classification itself is ``io.base.filesystem_refusal``, shared with the other three
    readers rather than written a fourth time here — which is what fixes the ORDER. Catching
    ``PermissionError`` before ``IsADirectoryError`` looks equivalent and is not: opening a
    mode-000 DIRECTORY raises ``PermissionError``, so a folder was reported as a permission
    problem by this reader and as a folder by the ``.h5``/``.tdms`` ones, and the user who
    followed this dialog's advice found the same file refused with a different sentence.
    """
    try:
        yield
    except OSError as exc:
        message = filesystem_refusal(path, exc)
        if message is None:
            raise CorruptFileError(
                f"{path.name} cannot be read: {exc.strerror or exc}.") from exc
        raise CorruptFileError(message) from None


def _detect_encoding(path: Path) -> tuple[str, str | None]:
    """``(encoding, why_it_is_not_utf8)`` — settled ONCE, and used by every later read.

    The probe used to open with ``errors="replace"`` while ``pd.read_csv`` used neither an
    encoding nor a replacement policy, so the two disagreed about what the file said: a Latin-1
    CSV off Windows instrument software probed clean, with a channel named ``curr?nt`` and a
    plausible sample rate, and then failed at load with a raw codec error. Deciding here means
    the name in the Files list is the name pandas will read.

    Latin-1 as the fallback because it is total — every byte sequence decodes — so the failure
    mode becomes "a column name may have the wrong accent", never "the file will not open".
    """
    with _readable(path):
        with path.open("rb") as fh:
            head = fh.read(_ENCODING_PROBE_BYTES)
    for bom, enc in ((codecs.BOM_UTF8, "utf-8-sig"),
                     (codecs.BOM_UTF16_LE, "utf-16"),
                     (codecs.BOM_UTF16_BE, "utf-16")):
        if head.startswith(bom):
            return enc, None
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut in half by the end of the probe block is not a property of
        # the file, only of where we stopped reading. A bad byte anywhere else is.
        if exc.start < max(0, len(head) - 4):
            bad = head[exc.start]
            return "latin-1", f"byte 0x{bad:02x} at offset {exc.start:,} is not valid UTF-8"
    return "utf-8", None


def _read_sample_lines(path: Path, n: int, encoding: str) -> list[tuple[str, int]]:
    """Return up to n non-blank lines as (stripped_text, raw_byte_length). Byte length includes
    the newline so it can drive a file-size row-count estimate."""
    codec = _LENGTH_CODEC.get(encoding, encoding)
    out: list[tuple[str, int]] = []
    with path.open("r", encoding=encoding, errors="replace", newline="") as fh:
        for _ in range(n):
            raw = fh.readline()
            if not raw:
                break
            stripped = raw.rstrip("\r\n")
            if stripped == "":
                continue
            out.append((stripped, len(raw.encode(codec, errors="replace"))))
    return out


def _tail_times(path: Path, encoding: str, delim: str, ncols: int) -> list[float]:
    """Column 0 of the last few complete rows, from ONE seek to the end of the file.

    This is what turns the sample rate from a statement about the first 1000 rows into a
    statement about the file. Measured on the tree as found: a CSV whose interval changes at row
    1500 reported 0.0600 s for a 0.3300 s recording — 5.5x wrong, ``issues=()`` — because the
    only rows anyone looked at were the ones before the change.

    Returns ``[]`` whenever the tail cannot be trusted (an encoding whose unit size is unknown,
    a short file already covered by the head scan, rows of the wrong width). An empty answer is
    silence, never a finding: this check may add an issue, never remove one.

    For a multi-byte encoding the seek is ALIGNED to a code-unit boundary and decoded with the
    endian-specific codec (there is no BOM in the middle of a file), which is what lets a UTF-16
    CSV be rate-checked at all.
    """
    unit = _TAIL_UNIT_BYTES.get(encoding)
    if unit is None:
        return []
    size = path.stat().st_size
    offset = max(0, size - _TAIL_PROBE_BYTES)
    codec, data_start = encoding, 0
    if unit > 1:
        with path.open("rb") as fh:
            bom = fh.read(2)
        codec = "utf-16-be" if bom == codecs.BOM_UTF16_BE else "utf-16-le"
        data_start = 2 if bom in (codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE) else 0
        offset -= (offset - data_start) % unit          # land on a code-unit boundary
        offset = max(data_start, offset)
    with path.open("rb") as fh:
        fh.seek(offset)
        blob = fh.read()
    lines = blob.decode(codec, errors="replace").splitlines()
    if offset > data_start:
        lines = lines[1:]              # line 0 is a fragment of whatever row we landed inside
    values: list[float] = []
    # Same splitter as the head scan (quotes and all): a quoted CSV whose tail rows were split
    # the naive way came out the wrong width, every row was discarded, and guard #4 fell silent
    # on the exact files the quoting fix exists for.
    for fields in _split_rows([t.strip() for t in lines], delim):
        if len(fields) != ncols:
            continue
        try:
            values.append(float(fields[0]))
        except ValueError:
            continue
    return values[-_TAIL_PROBE_ROWS:]


def _split_rows(texts: list[str], delim: str) -> list[list[str]]:
    """Lines split into fields THE WAY ``pd.read_csv`` will split them — quoting included.

    The scan used bare ``str.split(delim)`` while ``load()`` hands the file to pandas, so the
    two disagreed about what the file says for any ordinary RFC 4180 CSV, and the scan is the
    half that refuses. A header ``time,"current, pA","voltage, mV"`` — what an instrument or
    Excel writes for a column named ``current, pA`` — counted 5 fields against the data rows'
    3 and was refused as a "ragged table … This looks like an event-table export, not a trace.
    Event tables land in v2", both halves false and the user sent to wait for a feature.
    Excel's QUOTE_ALL output (``"0.000000","-119.8109","-600.0"``) made ``float()`` raise and
    was refused as "column 0 is not numeric, so it is not a time column" — of a column that is
    numeric. In the same script ``pd.read_csv`` parsed both files into 3,000 clean rows.

    ``csv.reader`` is this module's own tool (already imported for ``Sniffer``) and implements
    what pandas' C engine implements, so the guards now judge the table the loader will build.

    The ``csv.Error`` fallback is for BINARY content wearing a ``.csv`` name: ``csv.reader``
    refuses a line containing NUL, and a refusal path must not raise an exception of its own on
    the way to explaining itself. Splitting such a line naively is exactly as meaningless as it
    always was, and the raggedness guard below still refuses the file.
    """
    try:
        return list(csv.reader(texts, delimiter=delim))
    except csv.Error:
        return [t.split(delim) for t in texts]


def _detect_delimiter(path: Path, sample_text: str) -> str:
    """Sniff among comma/tab/semicolon; fall back to ',' for .csv and '\\t' for .txt."""
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=",\t;").delimiter
    except csv.Error:
        return "," if path.suffix.lower() == ".csv" else "\t"


def _looks_numeric(fields: list[str]) -> bool:
    for f in fields:
        try:
            float(f)
        except ValueError:
            return False
    return bool(fields)


def _voltage_from_filename(stem: str) -> tuple[float | None, Provenance]:
    matches = _MV_RE.findall(stem)
    if len(matches) == 1:
        try:
            return float(matches[0]), Provenance.FILENAME
        except ValueError:
            return None, Provenance.UNKNOWN
    return None, Provenance.UNKNOWN


def _scan(path: Path, encoding: str | None = None) -> _Scan:
    """`_scan_impl` with every filesystem failure converted to a sentence (see `_readable`).

    One wrapper rather than a guard at each of the four places this reader touches the disk
    (the encoding probe, the line read, the tail seek, two ``stat()`` calls), so a fifth cannot
    be added without it.
    """
    with _readable(path):
        # Before any text guard runs: are these bytes text at all? A binary recording renamed
        # .csv used to be described by the raggedness guard as "ragged table (1 fields x696,
        # 2 fields x153, …). This looks like an event-table export, not a trace" — a confident,
        # false description of an intact ABF, and the only one of the twelve wrong-extension
        # combinations that was not merely unhelpful but wrong. The shared redirect names what
        # the bytes really are; see io.base.foreign_format_redirect.
        renamed = foreign_format_redirect(path)
        if renamed is not None:
            raise UnsupportedFormatError(renamed)
        return _scan_impl(path, encoding)


def _scan_impl(path: Path, encoding: str | None = None) -> _Scan:
    """Cheap: reads at most _SCAN_LINES lines, one 64 KB tail block, and the file size. Enforces
    the ragged / monotonic / >=2-column guards, then derives encoding, rate and t0.

    ``encoding`` is normally settled here; ``load()`` passes one explicitly when a mid-file byte
    forced it to fall back, so the second scan reads the file the way pandas will.
    """
    why_not_utf8: str | None = None
    if encoding is None:
        encoding, why_not_utf8 = _detect_encoding(path)
    lines = _read_sample_lines(path, _SCAN_LINES, encoding)
    if not lines:
        raise UnsupportedFormatError(f"{path.name}: file is empty.")

    sample_text = "\n".join(t for t, _ in lines[:50])
    delim = _detect_delimiter(path, sample_text)

    # ONE split of the sampled lines, honouring quotes, shared by every guard below — so the
    # width count, the header test and the time column all describe the same table, and it is
    # the table pandas will parse. See `_split_rows`.
    rows = _split_rows([t for t, _ in lines], delim)

    # Header detection: a first row whose fields are all numeric is data, not a header.
    first_fields = rows[0]
    has_header = not _looks_numeric(first_fields)

    # --- RAGGEDNESS GUARD (guard #1) --------------------------------------------------------
    widths = [len(r) for r in rows]
    width_counts = Counter(widths)
    if len(width_counts) > 1:
        summary = ", ".join(f"{w} fields x{c}" for w, c in sorted(width_counts.items()))
        raise UnsupportedFormatError(
            f"{path.name}: ragged table ({summary}). This looks like an event-table export, "
            f"not a trace. Event tables land in v2.")

    ncols = widths[0]

    # Data rows only (drop a header line) for the numeric guards below.
    data_lines = lines[1:] if has_header else lines
    data_rows = rows[1:] if has_header else rows
    if len(data_lines) < 2:
        raise UnsupportedFormatError(
            f"{path.name}: too few data rows to establish a time base ({len(data_lines)}).")

    # Column 0 as float over the sampled data rows.
    try:
        col0 = np.array([float(r[0]) for r in data_rows], dtype=np.float64)
    except (ValueError, IndexError):
        raise UnsupportedFormatError(
            f"{path.name}: column 0 is not numeric, so it is not a time column.")

    # --- MONOTONICITY GUARD (guard #2) ------------------------------------------------------
    if not np.all(np.diff(col0) > 0):
        raise UnsupportedFormatError(
            f"{path.name}: column 0 is not monotonically increasing, so it is not a time column.")

    # --- >= 2 columns (guard #3) ------------------------------------------------------------
    if ncols < 2:
        raise UnsupportedFormatError(
            f"{path.name}: only {ncols} column(s); a trace needs a time column plus at least one "
            f"signal column.")

    # --- time base --------------------------------------------------------------------------
    t0_s = float(col0[0])
    diffs = np.diff(col0)
    dt = float(np.median(diffs))
    if not (dt > 0) or not np.isfinite(dt):
        raise UnsupportedFormatError(
            f"{path.name}: could not derive a sample interval from column 0.")
    sample_rate_hz: float | None = 1.0 / dt
    rate_provenance = Provenance.MEASURED

    report = ReaderReport(notes=(_UNITS_NOTE,))
    if why_not_utf8:
        report.note(fallback_encoding_note(encoding, why=why_not_utf8))

    # --- IS THIS COLUMN IN SECONDS AT ALL? --------------------------------------------------
    # The same question the HDF5 reader asks of /time, for the same reason and with the same
    # answer: a time column logged in microseconds is perfectly uniform, spans exactly what its
    # own step implies, and passes every check below — so the file that is most wrong is the
    # one that looks cleanest, and the rate comes out 10^6x low with issues=(). The unit is
    # declared nowhere in a CSV, so it is refused rather than guessed. See
    # io.base.MIN_PLAUSIBLE_RATE_HZ.
    if sample_rate_hz < MIN_PLAUSIBLE_RATE_HZ:
        report.issue(implausible_measured_rate_issue(
            source="this file's time column", dt=dt, hz=sample_rate_hz))
        sample_rate_hz = None
        rate_provenance = Provenance.UNKNOWN

    rel_jitter = float(np.std(diffs) / dt) if dt else 0.0
    if rel_jitter > 1e-3 and sample_rate_hz is not None:
        report.issue(f"time column is non-uniform (std/dt = {rel_jitter:.3g}); "
                     f"sample rate taken as the median interval")

    # --- DOES THAT RATE DESCRIBE THE WHOLE FILE? (guard #4) ----------------------------------
    # dt above is measured from the first _SCAN_LINES rows and then used as the sample rate of
    # the entire recording. One seek to the end of the file says whether that is true. Comparing
    # two directly measured intervals needs no row count, so this is exact even though the
    # probe's n_samples is still an estimate.
    # Skipped when the rate above was refused as implausible: both halves of guard #4 are
    # sentences ABOUT that rate ("the sample rate X Hz was measured from the first N rows, and
    # the rest of the file disagrees"), and a rate this reader has already refused to report is
    # not a thing to then argue with. The refusal itself is the louder statement.
    tail = _tail_times(path, encoding, delim, ncols)
    last_t: float | None = None
    if len(tail) >= 3:
        last_t = float(tail[-1])
        tail_diffs = np.diff(np.asarray(tail, dtype=np.float64))
        tail_diffs = tail_diffs[np.isfinite(tail_diffs) & (tail_diffs > 0)]
        if tail_diffs.size >= 2 and sample_rate_hz is not None:
            dt_tail = float(np.median(tail_diffs))
            if abs(dt_tail - dt) > _INTERVAL_TOL * dt:
                report.issue(rate_interval_changes_issue(
                    hz=sample_rate_hz, measured_over=len(data_lines), source="rows",
                    dt_head=dt, dt_tail=dt_tail))

    # Column names: header text if present, else positional.
    if has_header:
        names = tuple(f.strip() for f in first_fields)
    else:
        names = tuple(f"col{i}" for i in range(ncols))

    # n_samples estimate for the probe: file_size / mean_bytes_per_data_row. load() sets exact.
    data_bytes = [b for _, b in (lines[1:] if has_header else lines)]
    mean_row_bytes = max(1.0, float(np.mean(data_bytes)))
    header_bytes = lines[0][1] if has_header else 0
    file_size = path.stat().st_size
    est_n = max(len(data_lines), int((file_size - header_bytes) / mean_row_bytes))
    # NOT in `notes`: load() knows the exact count and must not repeat a caveat that has stopped
    # being true. Kept as its own field so the two callers differ by which one adds it, not by a
    # string match on the note text.
    estimate_note = f"n_samples is an estimate ({est_n:,}) until the file is fully loaded."

    nominal_v, v_prov = _voltage_from_filename(path.stem)

    return _Scan(
        delimiter=delim,
        has_header=has_header,
        ncols=ncols,
        names=names,
        t0_s=t0_s,
        sample_rate_hz=sample_rate_hz,
        rate_provenance=rate_provenance,
        est_n_samples=est_n,
        header_bytes=header_bytes,
        notes=report.notes,
        issues=report.issues,
        nominal_voltage_mV=nominal_v,
        voltage_provenance=v_prov,
        encoding=encoding,
        last_t_s=last_t,
        estimate_note=estimate_note,
    )


@register
class TextTraceReader(BaseReader):
    format_name = FORMAT_NAME
    extensions = (".csv", ".txt")

    @classmethod
    def sniff(cls, path: Path) -> Claim:
        # Text has no magic bytes; claim WEAK on extension only (never STRONG).
        return Claim.WEAK if path.suffix.lower() in cls.extensions else Claim.NO

    def _channel_infos(self, scan: _Scan, dead: dict[int, tuple[bool, str | None]] | None
                       ) -> tuple[ChannelInfo, ...]:
        infos: list[ChannelInfo] = []
        for out_idx, file_col in enumerate(range(1, scan.ncols)):
            is_dead, reason = dead.get(out_idx, (False, None)) if dead else (False, None)
            name = scan.names[file_col] if file_col < len(scan.names) else f"col{file_col}"
            infos.append(ChannelInfo(
                index=out_idx,
                name=name,
                raw_units="",                 # units are NEVER inferred from a column name
                kind=_kind_from_name(name),   # kind IS, from an explicit name (like the HDF5 reader)
                is_dead=is_dead,
                dead_reason=reason,
            ))
        return tuple(infos)

    def probe(self) -> RecordingProbe:
        scan = _scan(self.path)
        meta = RecordingMeta(
            path=self.path,
            format_name=FORMAT_NAME,
            n_samples=scan.est_n_samples,
            sample_rate_hz=scan.sample_rate_hz,
            rate_provenance=scan.rate_provenance,
            channels=self._channel_infos(scan, dead=None),
            t0_s=scan.t0_s,
            loaded_range=None,
            nominal_voltage_mV=scan.nominal_voltage_mV,
            voltage_provenance=scan.voltage_provenance,
            notes=scan.notes + (scan.estimate_note,),
            issues=scan.issues,
        )
        return RecordingProbe(meta=meta)

    def _read_columns(self, scan: _Scan, signal_cols: list[int],
                      skiprows, nrows: int | None, est_total: int,
                      progress: ProgressFn | None, cancel: CancelFn | None,
                      header: int | None = None
                      ) -> tuple[list[list[np.ndarray]], list[DowncastAudit]]:
        """One streamed pass over the file, in the encoding ``scan`` settled on.

        ``header`` is ``0`` for a read that starts at the top of the file and ``None`` for a
        windowed one, where ``skiprows`` has already consumed the header line along with
        everything before the window (see ``_window_start``). Columns are selected positionally
        either way, so dropping the header row changes nothing about which column is which.

        Parsed at float64 and downcast to float32 HERE, not by handing pandas
        ``dtype=np.float32`` — which is what this method used to do, and which performed the
        same two steps invisibly (the C tokenizer parses doubles and casts at the end). A text
        file can spell any decimal at all, including magnitudes below float32's smallest
        subnormal that the cast flushes to exactly 0.0, so the cast is taken back from pandas
        and put through the shared audit (``io.base.DowncastAudit`` — one rule with the HDF5
        and TDMS readers, whose float64 datasets have the same hole). The returned audits are
        aligned with ``signal_cols``; the float64 exists one chunk at a time, never whole.
        """
        buffers: list[list[np.ndarray]] = [[] for _ in signal_cols]
        audits: list[DowncastAudit] = [
            DowncastAudit(
                f"'{scan.names[c]}'" if c < len(scan.names) else f"col{c}",
                "decimal text")
            for c in signal_cols
        ]
        rows_read = 0
        reader = pd.read_csv(
            self.path,
            sep=scan.delimiter,
            header=header,
            usecols=signal_cols,
            dtype=np.float64,
            chunksize=LOAD_CHUNK_SAMPLES,
            skiprows=skiprows,
            nrows=nrows,
            encoding=scan.encoding,
            engine="c",
        )
        for chunk in reader:
            if cancel is not None and cancel():
                raise LoadCancelled(f"{self.path.name}: load cancelled.")
            for j in range(len(signal_cols)):
                src = chunk.iloc[:, j].to_numpy(dtype=np.float64, copy=False)
                dst = src.astype(np.float32)
                audits[j].observe(src, dst)
                buffers[j].append(dst)
            rows_read += len(chunk)
            if progress is not None:
                progress(min(1.0, rows_read / est_total),
                         f"Reading {self.path.name}: {rows_read:,} rows")
        return buffers, audits

    def _count_data_rows(self, scan: _Scan) -> int:
        """The file's exact data-row count, by one streamed counting pass.

        Exists for the zero-overlap refusal in ``load()``, which must name the recording's
        real length the way ``clamp_request`` does for the readers that know theirs up front.
        Counts non-blank lines (pandas' default ``skip_blank_lines`` does the same) minus a
        header line, in the encoding the scan settled on.
        """
        with _readable(self.path):
            with self.path.open("r", encoding=scan.encoding, errors="replace",
                                newline="") as fh:
                total = sum(1 for line in fh if line.strip())
        return max(0, total - (1 if scan.has_header else 0))

    def _window_start(self, scan: _Scan, start: int) -> tuple[int, float | None]:
        """Where data row ``start`` really begins, and what the file's own clock says there.

        Returns ``(physical lines to skip, the time column's value at that row or None)``.

        TWO defects, one seam, because they are one mistake made twice: a window was described
        in a unit the file is not indexed in, and its clock was computed where it could be read.

        THE INDEX. ``skiprows=range(1, 1 + start)`` is applied by pandas to PHYSICAL LINES,
        before its own ``skip_blank_lines=True`` drops blanks — while the whole-file load, and
        ``_read_sample_lines``, and ``_count_data_rows``, all index the file by DATA ROW. One
        blank line after the header (what Excel and several instrument exporters write) is
        therefore enough to slide a window one row early, with ``meta.issues`` empty and
        ``meta.loaded_range`` still naming the range that was asked for. [MEASURED] on a
        5,000-row fixture whose values equal their row numbers: ``whole[2000] = 2000.0`` beside
        ``window[0] = 1999.0``, and 1997.0 with a blank every 500 rows. Adjacent windows of one
        file then no longer tile: a row is duplicated at every seam and the tail rows are lost.

        THE CLOCK. A CSV is the one format GUARANTEED to hold a real time column — guard #2
        refuses any file whose column 0 is not monotonically increasing — so the experimental
        time of the first loaded sample is a value this reader can READ, which is what the HDF5
        reader does with ``/time`` at the window's own index. It was instead computed as
        ``file_t0 + start/fs`` from a rate measured over the first 1,000 rows. On a 10 kHz CSV
        whose acquisition paused 5 s mid-file, a window at row 3,500 reported t0 = 0.350000 s
        where the file's own column 0 says 5.350000 s — 5 s of every cursor readout, event
        time, hand measurement and segment rebase, silently, while the byte-identical time base
        written as HDF5 reported the right value beside it.

        Both are answered by the same single streamed pass, which walks the file counting DATA
        rows and reads column 0 off the row it stops on. It costs one text scan of the skipped
        prefix — the same prefix ``pd.read_csv`` is about to walk anyway — and it runs only for
        ``start > 0``, so a whole-file load and a window at the top of the file are untouched.
        [MEASURED] on the owner's 332 MB / 9,554,560-row 'nisin_fibrin_trace.csv', a window at
        sample 5,000,000: the pass is 1.36 s of a 4.14 s load, and it returns the file's own
        99.99999999999999 s — byte-identical to row 5,000,000 of the file — where the old
        computed answer was right only by the accident of that file being uniform.
        """
        skip_lines = 0
        data_seen = 0
        header_pending = scan.has_header
        t_at_start: float | None = None
        with _readable(self.path):
            with self.path.open("r", encoding=scan.encoding, errors="replace",
                                newline="") as fh:
                for raw in fh:
                    if raw.strip() == "":
                        skip_lines += 1          # a blank line is not a data row, for either of us
                        continue
                    if header_pending:
                        header_pending = False
                        skip_lines += 1
                        continue
                    if data_seen == start:
                        fields = _split_rows([raw.rstrip("\r\n")], scan.delimiter)[0]
                        try:
                            t_at_start = float(fields[0])
                        except (ValueError, IndexError):
                            t_at_start = None    # caller falls back to the rate, and says so
                        break
                    data_seen += 1
                    skip_lines += 1
        return skip_lines, t_at_start

    def load(self, sample_range: SampleRange | None = None,
             progress: ProgressFn | None = None,
             cancel: CancelFn | None = None) -> Document:
        scan = _scan(self.path)
        report = ReaderReport(notes=scan.notes, issues=scan.issues)

        signal_cols = list(range(1, scan.ncols))  # positional; column 0 is the time axis, dropped

        # `start` is clamped at 0 only. There is no honest upper clamp here: a CSV's length is an
        # ESTIMATE until its last row is parsed, so clamping against the estimate would be
        # guessing. What the document actually holds is settled below, from what was read.
        skiprows = None
        nrows = None
        start = 0
        header = 0 if scan.has_header else None
        window_t0_read: float | None = None
        if sample_range is not None:
            start = max(0, int(sample_range.start))
            stop = max(start, int(sample_range.stop))
            nrows = stop - start
            if start > 0:
                # DATA rows, not physical lines, and the window's clock read out of the file
                # rather than computed — see `_window_start`, which answers both in one pass.
                # The header line is skipped along with everything before the window and
                # `header=None` is passed instead, because the columns are selected
                # positionally: there is nothing left for a header row to name.
                skip_lines, window_t0_read = self._window_start(scan, start)
                skiprows = range(0, skip_lines)
                header = None

        est_total = max(1, len(sample_range) if sample_range is not None else scan.est_n_samples)

        # Aligned with signal_cols; empty means nothing was read, so nothing was downcast.
        audits: list[DowncastAudit] = []
        try:
            buffers, audits = self._read_columns(scan, signal_cols, skiprows, nrows, est_total,
                                                 progress, cancel, header=header)
        except LoadCancelled:
            raise
        except pd.errors.EmptyDataError:
            # `skiprows` consumed every line in the file: the window starts at or past the last
            # data row. Nothing is wrong with the FILE, so this must not become the corruption
            # sentence — it is the zero-overlap case, and the refusal below names the request
            # and the recording's real length the way the other three readers do. (Reachable
            # only for a window, because the scan has already proved the file has data rows.)
            buffers = [[] for _ in signal_cols]
        except UnicodeDecodeError as exc:
            # The head decoded as UTF-8 and a byte further in did not. Re-read as Latin-1 rather
            # than refusing on the codec error alone: a codec offset is not something a user can
            # act on, and Latin-1 either recovers the file or reaches the row that is actually
            # wrong and can name it.
            where = f"byte 0x{exc.object[exc.start]:02x} at offset {exc.start:,}"
            scan = _scan(self.path, encoding="latin-1")
            report = ReaderReport(notes=scan.notes, issues=scan.issues)
            report.issue(lossy_encoding_issue(self.path.name, "UTF-8", why=where))
            report.note(fallback_encoding_note("latin-1", why="a byte past the file header"))
            if start > 0:
                # The window is re-located in the encoding pandas will now read: blank lines
                # and the value of column 0 are facts about the DECODED file, and this branch
                # has just changed which decoding that is.
                skip_lines, window_t0_read = self._window_start(scan, start)
                skiprows = range(0, skip_lines)
            try:
                buffers, audits = self._read_columns(scan, signal_cols, skiprows, nrows,
                                                     est_total, progress, cancel, header=header)
            except LoadCancelled:
                raise
            except pd.errors.EmptyDataError:
                buffers = [[] for _ in signal_cols]
            except (ValueError, OSError, UnicodeDecodeError, pd.errors.ParserError) as exc2:
                raise CorruptFileError(
                    f"{self.path.name} is not valid UTF-8 ({where}); read as Latin-1 instead, it "
                    f"then failed on the data itself: {exc2}. That row is the one to look at, "
                    f"not the encoding.") from exc2
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            raise CorruptFileError(f"{self.path.name}: {exc}") from exc

        channels: list[np.ndarray] = []
        for buf in buffers:
            if buf:
                arr = np.concatenate(buf)
            else:
                arr = np.empty(0, dtype=np.float32)
            channels.append(np.ascontiguousarray(arr, dtype=np.float32))

        n = len(channels[0]) if channels else 0
        if n == 0:
            if sample_range is not None:
                # Zero rows read past `start` from a file whose scan already saw data rows:
                # nothing is wrong with the FILE — the REQUEST outran it. This is the
                # zero-overlap boundary of the short-window case (partial overlap gets
                # `short_window_issue` below), and it happens on a project restore whose CSV
                # was re-exported shorter, or live, because the probe's n_samples is a byte
                # estimate that can overshoot the real row count. It used to fall through to
                # the corruption sentence below, which told the user their intact file failed
                # to read; the other three readers refuse the identical request through
                # `clamp_request` with the selects-nothing sentence, and two dialogs must not
                # disagree about the same condition. The exact row count is settled by one
                # cheap counting pass — bounded by `start` itself, since the read just proved
                # the file holds no rows past it.
                #
                # The gate used to be `start > 0`, which left the start == 0 boundary of the
                # same condition outside it: SampleRange(0, 0) — an empty request, which the
                # other three readers answer through `clamp_request` with exactly the sentence
                # below — fell through to "no data rows were read", corruption wording naming
                # neither the request nor the file's real length, for an intact recording. A
                # window is empty because of where it starts AND where it stops; only the
                # request can be at fault here, because the scan above already proved this file
                # has data rows.
                n_rows = self._count_data_rows(scan)
                raise UnsupportedFormatError(
                    f"{self.path.name}: the requested sample range "
                    f"[{sample_range.start:,}, {sample_range.stop:,}) selects nothing in a "
                    f"{n_rows:,}-sample recording.")
            raise CorruptFileError(f"{self.path.name}: no data rows were read.")

        # --- the exact half of guard #4 ---------------------------------------------------------
        # The row count is no longer an estimate, and the file's final timestamp came free with
        # the scan's tail seek, so "does the measured rate describe this file?" now has an exact
        # answer instead of the two-interval comparison the probe had to settle for.
        # A WINDOW that ran off the end of the file also settles the file's exact row count —
        # start + n IS the file length — so the same exact check runs there. A window that
        # stopped short of the end does not, and gets the probe's two-interval check only. That
        # is a KNOWN GAP, left open deliberately: the alternative is to test the file's span
        # against the row-count ESTIMATE, which on a file with variable-width rows would raise
        # false alarms on well-formed recordings.
        exact_rows: int | None = None
        if sample_range is None:
            exact_rows = n
        elif n < (int(sample_range.stop) - start):        # the read hit EOF
            exact_rows = start + n
        if (exact_rows is not None and scan.last_t_s is not None
                and scan.sample_rate_hz is not None and scan.sample_rate_hz > 0):
            dt = 1.0 / scan.sample_rate_hz
            implied = (exact_rows - 1) * dt
            actual = float(scan.last_t_s) - scan.t0_s
            if abs(actual - implied) > max(_SPAN_TOL * abs(implied), 2.0 * dt):
                report.issue(rate_disagrees_issue(
                    hz=scan.sample_rate_hz, measured_over=min(_SCAN_LINES, exact_rows),
                    source="rows", implied_s=implied, actual_s=actual))

        # The window asked for rows this file does not have. `loaded_range` is clamped to the
        # truth (that is C28's fix) and used to say nothing about it — the one place
        # `short_read_issue` belonged and was not wired. It is how a project restore behaves when
        # the CSV has been re-exported shorter since the project was saved.
        if sample_range is not None:
            asked = int(sample_range.stop) - start
            if n < asked:
                report.issue(short_window_issue(
                    self.path.name, start=start, asked=asked, got=n))

        # A computed clock is allowed, but never in silence: the contract for t0 is "the
        # experimental time of the first loaded sample", and a value derived from the rate is
        # that time only if the interval never changes before it. This fires when column 0 of
        # the window's first row could not be read as a number — the one case the pass above
        # cannot answer from the file itself.
        if sample_range is not None and start > 0 and window_t0_read is None:
            report.note(
                f"the time column could not be read at row {start:,}, so this window's start "
                f"time was computed from the sample rate instead of read from the file. If the "
                f"interval changes anywhere before that row, this recording's absolute times "
                f"are off by the difference; the samples themselves are unaffected.")

        # What the float64->float32 downcast destroyed, if anything — the shared rule's
        # sentence (see _read_columns). issue() is None for a faithful cast, which
        # ReaderReport.issue drops, so a normal file adds nothing here.
        for audit in audits:
            report.issue(audit.issue())

        dead = {}
        for i, a in enumerate(channels):
            is_dead, reason = detect_dead(a)
            if (is_dead and reason is not None and reason.startswith("constant 0.0")
                    and i < len(audits) and audits[i].dead_reason() is not None):
                # The flat line is the downcast's doing, not the file's: detect_dead reads the
                # loaded float32 and its "constant 0.0" would assert a property no value in
                # the file has — same substitution the HDF5 and TDMS readers make.
                reason = audits[i].dead_reason()
            dead[i] = (is_dead, reason)

        meta = RecordingMeta(
            path=self.path,
            format_name=FORMAT_NAME,
            n_samples=n,
            sample_rate_hz=scan.sample_rate_hz,
            rate_provenance=scan.rate_provenance,
            channels=self._channel_infos(scan, dead=dead),
            # The window's own clock — READ from the file's time column at the window's own
            # row where that value exists, exactly as the HDF5 reader reads /time at the
            # window's own index, and only computed from the rate (contract.window_t0) when it
            # does not: a whole-file load, a window at row 0, or a row whose column 0 could not
            # be parsed. A computed t0 is right only while the interval is constant through the
            # whole file, and a CSV whose acquisition paused, was concatenated, or changed
            # logging rate is exactly the file where it is not — see `_window_start`.
            t0_s=(window_t0(scan.t0_s, start, scan.sample_rate_hz)
                  if window_t0_read is None else window_t0_read),
            loaded_range=window_loaded(sample_range, start, n),
            nominal_voltage_mV=scan.nominal_voltage_mV,
            voltage_provenance=scan.voltage_provenance,
            notes=report.notes,
            issues=report.issues,
        )
        return TraceDocument(meta=meta, channels=tuple(channels))
