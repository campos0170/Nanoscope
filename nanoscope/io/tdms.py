"""TDMS reader (National Instruments) via nptdms.

Two layouts, distinguished by the presence of ``wf_increment`` on the waveform channel:

  MODERN  [VERIFIED] '4_detection 100 pM nisin with fibrin -008.tdms':
      one group 'Untitled', channels 'Dev2/ai0'/'Dev2/ai1', len 13,428,995,
      wf_increment = 1.9999999999999998e-05 -> fs = 50 kHz, unit_string 'Volts'.
      Volts are RAW off the ADC -> ChannelKind.RAW_VOLTAGE. NEVER presented as pA.

  LEGACY  [VERIFIED] 'example.tdms':
      groups named 'Current2.925730E-1' / 'Current4.576672E-1' / 'Current7.750115E-1',
      each with channels 'Untitled' / 'Untitled 1', NO wf_increment.
      Sample rate is GENUINELY UNKNOWN -> sample_rate_hz None, Provenance.UNKNOWN.
      DO NOT fabricate 50 kHz. Channel names carry no unit -> raw_units "", kind UNKNOWN.

DEVIATION FROM THE LITERAL CONTRACT, forced by the real corpus (reported honestly):
  §12.3 says "read all channels of all groups; if lengths differ across channels raise" and
  claims "no evidence of this in the corpus". [VERIFIED] that premise is FALSE: 'example.tdms'
  has THREE groups whose channels are 20400 / 20400 / 20200 / 20200 / 20400 / 20400 samples —
  lengths DO differ. Test 21.13 (`test_legacy_tdms_rate_is_none`) is a LOCKED test that requires
  this exact file to LOAD (not raise). Raising would fail that test and would refuse real user
  data. A TraceDocument is rectangular (single n_samples), so the ragged groups are reconciled by
  truncating every channel to the common MINIMUM length and recording a loud ``meta.issues`` entry
  naming the offending channels and their original lengths. Nothing is hidden; all channels remain
  visible; the file loads. This preserves the contract's intent ("do not build a heterogeneous
  model", "name the offending channels") while honouring the locked test.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import ClassVar

import numpy as np
from nptdms import TdmsFile

from ..core.document import Document, TraceDocument
from ..core.types import (
    LOAD_CHUNK_SAMPLES,
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
    SampleRange,
    detect_dead,
    kind_for_units,
    normalise_units,
)
from .base import (
    BaseReader,
    Claim,
    DowncastAudit,
    ProgressFn,
    CancelFn,
    RecordingProbe,
    downcast_needs_audit,
    filesystem_refusal,
    foreign_format_redirect,
    read_magic,
    register,
)
from .contract import (
    ReaderReport,
    clamp_request,
    decoder_message_issue,
    decoder_messages,
    short_read_issue,
    window_loaded,
    window_t0,
)
from .errors import (
    CorruptFileError,
    LoadCancelled,
    ReaderError,
    UnsupportedFormatError,
)

_LEGACY_NOTE = "This file declares no sample rate. The x-axis is in samples."

# TDMS lead-in: b"TDSm", ToC mask (u32), version (u32), next-segment offset (u64), raw-data
# offset (u64). The offsets are counted from the END of the 28-byte lead-in.
_LEAD_IN_BYTES = 28
_TOC_BIG_ENDIAN = 1 << 6
_NO_NEXT_SEGMENT = 0xFFFFFFFFFFFFFFFF   # written when the acquisition never closed the file


def _finite_or_zero(value) -> float:
    """A declared start offset that is not a finite number is no start offset.

    Same class as the ``wf_increment`` guard directly below it: ``float()`` on whatever the file
    happened to store used to escape ``_scan`` as a raw ``ValueError`` — a traceback where the
    reader had promised a sentence. t0 has no defensible fallback other than 0.0, which is what
    a file that declares nothing gets.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def _declared_t0(offsets: list[tuple[str, float]], issues: list[str]) -> float:
    """The file's declared start offset, read from EVERY channel that declares one.

    ``wf_start_offset`` is the other half of the clock ``wf_increment`` describes, and it was
    the half still read from one channel after the increment had been made per-channel. Two
    consequences, both silent: an offset declared on any channel but the first timing one was
    dropped (t0 = 0.0 for a file that says 12.5 s), and two channels declaring different
    offsets collapsed to whichever was written first — so the SAME two channels gave t0 = 0.0
    or t0 = 30.0 depending on file order. Meanwhile the identical disagreement about
    ``wf_increment`` is refused loudly with every channel's declared rate named.

    So: an offset declared anywhere is honoured, and a disagreement is NAMED. It is not
    refused the way a mixed rate is, because the two are not the same kind of damage — a wrong
    RATE makes every dwell time and duration wrong (there is an honest "unknown" to fall back
    to, and ``sample_rate_hz = None`` is it), while t0 only places the recording on the
    experimental clock and has no representable "unknown": 0.0 is itself a claim, and a false
    one for every channel here. The first declaration is adopted, and the issue says which one
    was adopted and what the others declared, so the number on the x-axis is never a number
    nobody was told about.
    """
    if not offsets:
        return 0.0
    chosen_name, chosen = offsets[0]
    if any(not math.isclose(value, chosen, rel_tol=1e-9, abs_tol=1e-12)
           for _n, value in offsets):
        detail = ", ".join(f"'{n}' declares wf_start_offset = {v:g} s" for n, v in offsets)
        issues.append(
            f"this file's channels declare different start offsets ({detail}). One "
            f"recording carries one time axis, so the experimental time of sample 0 was taken "
            f"from '{chosen_name}' ({chosen:g} s); every other channel here is placed on that "
            f"clock. Dwell times and durations are unaffected — only the absolute time this "
            f"recording is stamped with. Export each channel to its own file if their start "
            f"offsets are meant to differ.")
    return chosen


def _segment_chain_defect(path: Path) -> str | None:
    """Walk the segment lead-in chain and return what is WRONG with this file, or None.

    This is the answer to "how many samples are really here", asked of the file itself rather
    than of the decoder's best effort. nptdms recovers what it can from a short file and reports
    the recovered length as though it were the recording — measured on a file cut to 30 %:
    20,000 samples became 5,983, ``issues=()``, ``notes=()``. That number is the denominator of
    every event rate in the app, so recovering quietly is the whole defect. The equivalent
    truncated ABF is refused outright; TDMS was the only reader that said nothing.

    Costs one seek and 28 bytes per segment: [MEASURED] 7 ms warm over the 1,227 segments of the
    215 MB corpus file, against 0.21 s cold — which the metadata pass nptdms runs immediately
    afterwards was going to pay anyway.
    """
    size = path.stat().st_size
    pos = 0
    segments = 0
    try:
        with path.open("rb") as fh:
            while pos + _LEAD_IN_BYTES <= size:
                fh.seek(pos)
                lead = fh.read(_LEAD_IN_BYTES)
                if len(lead) < _LEAD_IN_BYTES or lead[:4] != b"TDSm":
                    if segments == 0:
                        return None          # not a segmented TDMS at all; nptdms will say so
                    return (f"its segment chain breaks {pos:,} bytes in, after {segments:,} "
                            f"complete segments")
                toc = struct.unpack_from("<I", lead, 4)[0]
                order = ">" if toc & _TOC_BIG_ENDIAN else "<"
                _version, next_offset, _raw_offset = struct.unpack_from(order + "IQQ", lead, 8)
                segments += 1
                if next_offset == _NO_NEXT_SEGMENT:
                    return (f"its final segment declares no end, which is what an acquisition "
                            f"that was interrupted rather than stopped leaves behind "
                            f"({segments:,} segments present)")
                if next_offset == 0:
                    return None              # a legitimately empty trailing segment
                nxt = pos + _LEAD_IN_BYTES + int(next_offset)
                if nxt > size:
                    return (f"its final segment declares {nxt - size:,} more bytes than the file "
                            f"contains ({size:,} bytes present, {segments:,} segments)")
                pos = nxt
    except OSError:
        return None                          # the real error surfaces from nptdms, with a message

    # The loop condition is `pos + _LEAD_IN_BYTES <= size`, so a file that ends PART-WAY THROUGH
    # the next segment's 28-byte lead-in falls out of it with `pos < size` — and the trailing
    # fragment, which is proof of truncation, used to be discarded. [MEASURED] on a 10-segment
    # interleaved stream (20,000 samples/channel) cut 12 bytes into segment #7's lead-in: 14,000
    # samples, `issues=()`, `notes=()`, and nptdms silent as well (0 log records). A cut EXACTLY
    # at a boundary is a different matter and is deliberately not reported: such a file is
    # byte-for-byte a complete shorter recording, so there is nothing to detect and a guess would
    # be a false alarm on every legitimately short acquisition.
    trailing = size - pos
    if segments and 0 < trailing < _LEAD_IN_BYTES:
        return (f"it stops {trailing:,} bytes into the header of segment {segments + 1:,}, which "
                f"needs {_LEAD_IN_BYTES} ({segments:,} complete segments present) — the "
                f"acquisition was cut while writing that segment")
    return None


class _ChannelPlan:
    """Everything probe() and load() need about one channel, minus the bulk array."""

    __slots__ = ("name", "raw_units", "kind", "tdms_group", "tdms_channel", "length")

    def __init__(self, name: str, raw_units: str, kind: ChannelKind,
                 tdms_group: str, tdms_channel: str, length: int) -> None:
        self.name = name
        self.raw_units = raw_units
        self.kind = kind
        self.tdms_group = tdms_group
        self.tdms_channel = tdms_channel
        self.length = length


class _Scan:
    """Cheap structural read shared by probe() and load(). Reads only metadata + channel lengths
    (nptdms exposes len(channel) from the file index without loading the samples)."""

    def __init__(self, format_name: str, sample_rate_hz: float | None,
                 rate_provenance: Provenance, t0_s: float,
                 plans: list[_ChannelPlan], base_len: int,
                 notes: tuple[str, ...], issues: tuple[str, ...]) -> None:
        self.format_name = format_name
        self.sample_rate_hz = sample_rate_hz
        self.rate_provenance = rate_provenance
        self.t0_s = t0_s
        self.plans = plans
        self.base_len = base_len
        self.notes = notes
        self.issues = issues


@register
class TdmsReader(BaseReader):
    format_name: ClassVar[str] = "TDMS"          # refined per layout in meta.format_name
    extensions: ClassVar[tuple[str, ...]] = (".tdms",)

    # -- sniff -----------------------------------------------------------------------------------
    @classmethod
    def sniff(cls, path: Path) -> Claim:
        if path.suffix.lower() != ".tdms":
            return Claim.NO
        if read_magic(path, 4) == b"TDSm":
            return Claim.STRONG
        return Claim.WEAK

    # -- structural scan -------------------------------------------------------------------------
    def _scan(self) -> _Scan:
        # Everything nptdms says about this file while we read it, captured here rather than left
        # on stderr. It is the only component in the app that knows a segment was short, and it
        # says so to a logger the user has never seen.
        with decoder_messages("nptdms") as decoder_said:
            scan = self._scan_open()
        report = ReaderReport(notes=scan.notes, issues=scan.issues)
        defect = _segment_chain_defect(self.path)
        if defect is not None:
            if scan.base_len <= 0:
                # Nothing survived. Refused HERE rather than by the empty-recording door, which
                # would tell the truth about the sample count and the wrong thing about the
                # cause: its sentence ends "this file is the expected result and can be
                # deleted", and a cut recording is neither expected nor safe to delete.
                raise CorruptFileError(
                    f"{self.path.name} is truncated — {defect} — and at least one channel "
                    f"decoded to no samples at all, so there is no window every channel shares. "
                    f"The recording this belongs to is longer than this file: recover the "
                    f"original rather than deleting it.")
            report.issue(short_read_issue(self.path.name, recovered=scan.base_len, detail=defect))
        elif scan.base_len <= 0 and any(p.length > 0 for p in scan.plans):
            # The SECOND way this reader can hand out a zero-sample document, and it reached
            # the empty-recording door in the same silence the truncation case above was
            # rerouted away from. A file whose 'Untitled/ai0' holds all 20,000 samples of the
            # acquisition and whose 'Untitled/ai1' was never filled is not an empty recording:
            # `base_len = min(lengths)` is 0, so the whole recording disappears behind a
            # sentence that ends "this file is the expected result and can be deleted" —
            # advice to delete a file that holds the data. The differing-lengths issue the
            # scan already built for exactly this file was constructed and then never shown,
            # because nothing survived to carry it. Refused here, naming every channel's
            # length, in the words the HDF5 reader gives the identical input
            # ("signal channels have mismatched lengths (/current=20,000, /voltage=0)").
            detail = ", ".join(f"'{p.name}'={p.length:,}" for p in scan.plans)
            longest = max(p.length for p in scan.plans)
            raise CorruptFileError(
                f"{self.path.name}: channels have mismatched lengths ({detail}), and one of "
                f"them holds no samples at all, so there is no window every channel shares. "
                f"This is not an empty recording — {longest:,} samples are in this file and "
                f"nothing here is safe to delete. Re-export the channels that hold data, or "
                f"the run that filled only one of them, to open the recording here.")
        for message in decoder_said:
            report.issue(decoder_message_issue("TDMS", message))
        scan.notes = report.notes
        scan.issues = report.issues
        return scan

    def _open_refusal(self, exc: Exception) -> ReaderError:
        """The user-facing refusal for an nptdms open failure, classified from the FILESYSTEM,
        not from ``exc``.

        nptdms's own words — "[Errno 2] No such file or directory: '/abs/path'" for a gone
        file, "[Errno 13] Permission denied" for a locked one, "[Errno 21]" for a directory —
        are library internals io/errors.py promises never reach the UI, and wrapping them in
        "not a readable TDMS file" calls a merely-gone file unreadable-format. The real cause
        is re-derived from the disk by ``io.base.filesystem_refusal`` — the same call
        ``abf._refusal``, ``hdf5._open_refusal`` and ``text._readable`` make, so the dialogs for
        one condition cannot disagree by format as the four readers are edited apart.
        """
        gone_locked_or_folder = filesystem_refusal(self.path, exc)
        if gone_locked_or_folder is not None:
            return CorruptFileError(gone_locked_or_folder)
        renamed = foreign_format_redirect(self.path)
        if renamed is not None:
            # An ABF, an HDF5 or a text export wearing a .tdms name. Was: nptdms's "Segment
            # does not start with b'TDSm', but with b'ABF '" — the decoder's words for a file
            # that is not damaged at all, only misnamed, while the mirror case (HDF5 bytes
            # named .abf) was told exactly what it was and where to rename it.
            return UnsupportedFormatError(renamed)
        return CorruptFileError(f"{self.path.name}: not a readable TDMS file ({exc}).")

    def _scan_open(self) -> _Scan:
        try:
            tf = TdmsFile.open(str(self.path))
        except UnsupportedFormatError:
            raise
        except Exception as exc:  # nptdms parse failure -> honest user-facing message
            raise self._open_refusal(exc) from exc

        try:
            plans: list[_ChannelPlan] = []
            # EVERY channel that declares wf_increment, as (channel name, raw value, its own
            # props) in file order — not just the first channel encountered. Timing used to be
            # read from `first_props` alone, which made two false statements possible: a
            # mixed-rate file (a legal NI multi-rate acquisition) got the first channel's
            # clock for every channel with issues=(), and a file whose FIRST channel happened
            # to carry no wf_increment was labelled "declares no sample rate" while its second
            # channel declared one. What each channel says about its own clock is read from
            # that channel.
            timing: list[tuple[str, object, dict]] = []
            # And EVERY channel that declares a start offset, on the same terms and for the
            # same reason — see `_declared_t0`. Collected here, in the one pass that already
            # visits every channel, so the offset cannot go back to being the first channel's
            # opinion the next time this loop is edited.
            offsets: list[tuple[str, float]] = []
            for group in tf.groups():
                for ch in group.channels():
                    props = dict(ch.properties)
                    name = f"{group.name}/{ch.name}"
                    if "wf_increment" in props:
                        timing.append((name, props["wf_increment"], props))
                    if "wf_start_offset" in props:
                        offsets.append((name, _finite_or_zero(props["wf_start_offset"])))
                    unit = props.get("unit_string") or props.get("NI_UnitDescription") or ""
                    raw_units = normalise_units(str(unit)) if unit else ""
                    kind = kind_for_units(raw_units) if raw_units else ChannelKind.UNKNOWN
                    plans.append(_ChannelPlan(
                        name=name,
                        raw_units=raw_units,
                        kind=kind,
                        tdms_group=group.name,
                        tdms_channel=ch.name,
                        length=len(ch),
                    ))

            if not plans:
                raise UnsupportedFormatError(
                    f"{self.path.name}: no channels found in any TDMS group.")

            # Layout: presence of wf_increment on ANY waveform channel.
            notes: list[str] = []
            issues: list[str] = []

            # `"wf_increment" in props`, NOT truthiness of the value. `if wf_increment:` is
            # FALSE at 0.0, so the `inc <= 0.0` guard inside this branch was unreachable for
            # exactly the value it was written for: a file declaring wf_increment = 0.0 fell
            # through to the legacy branch and loaded carrying the note "This file declares no
            # sample rate" — which is not true. It declares one, and the one it declares is
            # impossible. A negative increment was correctly refused all along; zero now joins
            # it rather than being described wrongly, and a non-numeric or non-finite one is
            # refused by the same sentence instead of escaping as a raw ValueError from float().
            if timing:
                declared: list[tuple[str, float]] = []
                for ch_name, raw, _props in timing:
                    try:
                        inc = float(raw)
                    except (TypeError, ValueError):
                        raise CorruptFileError(
                            f"{self.path.name}: wf_increment on '{ch_name}' is {raw!r}, which "
                            f"is not a number, so this file's sample interval cannot be "
                            f"read.") from None
                    if not np.isfinite(inc) or inc <= 0.0:
                        raise CorruptFileError(
                            f"{self.path.name}: wf_increment on '{ch_name}' is {inc}, which is "
                            f"not a valid sample interval.")
                    declared.append((ch_name, inc))

                base_inc = declared[0][1]
                agree = all(math.isclose(inc, base_inc, rel_tol=1e-9)
                            for _n, inc in declared)
                if agree:
                    format_name = "TDMS (modern)"
                    sample_rate_hz: float | None = 1.0 / base_inc
                    rate_provenance = Provenance.FILE_META
                else:
                    # The file disagrees with ITSELF about its clock. A TraceDocument is
                    # rectangular — one time axis for every channel — so picking any one
                    # channel's rate would make every dwell time and duration on the others
                    # wrong by the ratio, silently. Refuse the single clock and say exactly
                    # which channels declare what (io/contract.py question 2: the rate is not
                    # silently corrected — "saying that the rest of the file disagrees is the
                    # fix"). The file still loads; detection is disabled honestly, on a named
                    # cause, rather than enabled on a clock that is false for some channels.
                    format_name = "TDMS (modern)"
                    sample_rate_hz = None
                    rate_provenance = Provenance.UNKNOWN
                    detail = ", ".join(
                        f"'{n}' declares wf_increment = {inc:g} ({1.0 / inc:,.6g} Hz)"
                        for n, inc in declared)
                    issues.append(
                        f"this file's channels declare different sample rates ({detail}). One "
                        f"recording here carries one time axis, and adopting any one channel's "
                        f"clock would silently make every dwell time and duration on the "
                        f"others wrong by the ratio, so the sample rate is left unknown and "
                        f"the x-axis is in samples. Export each rate's channels to their own "
                        f"file to analyse them here.")
            else:
                format_name = "TDMS (legacy)"
                sample_rate_hz = None
                rate_provenance = Provenance.UNKNOWN
                notes.append(_LEGACY_NOTE)

            # ONE t0 rule, applied to all three layouts above rather than inside the one that
            # happened to have the bug. A legacy file that declares a start offset without an
            # increment, and a mixed-rate file whose clock this reader refuses, both still
            # carry an experimental time the acquisition wrote down; there is no reason for
            # either to be silently stamped 0.0 while the third branch reads the property.
            t0_s = _declared_t0(offsets, issues)

            lengths = [p.length for p in plans]
            base_len = min(lengths)
            if len(set(lengths)) > 1:
                detail = ", ".join(f"'{p.name}'={p.length}" for p in plans)
                issues.append(
                    f"Channels have differing lengths ({detail}); truncated all to the common "
                    f"minimum of {base_len} samples so they share one time axis.")

            return _Scan(format_name, sample_rate_hz, rate_provenance, t0_s,
                         plans, base_len, tuple(notes), tuple(issues))
        finally:
            tf.close()

    def _meta(self, scan: _Scan, channels: tuple[ChannelInfo, ...],
              n_samples: int, loaded_range: SampleRange | None, start: int = 0) -> RecordingMeta:
        return RecordingMeta(
            path=self.path,
            format_name=scan.format_name,
            n_samples=n_samples,
            sample_rate_hz=scan.sample_rate_hz,
            rate_provenance=scan.rate_provenance,
            channels=channels,
            # The window's own clock, not the file's — see contract.window_t0.
            t0_s=window_t0(scan.t0_s, start, scan.sample_rate_hz),
            loaded_range=loaded_range,
            nominal_voltage_mV=None,          # no VOLTAGE-kind channel in TDMS; never fabricate
            voltage_provenance=Provenance.UNKNOWN,
            notes=scan.notes,
            issues=scan.issues,
        )

    # -- probe -----------------------------------------------------------------------------------
    def probe(self) -> RecordingProbe:
        scan = self._scan()
        channels = tuple(
            ChannelInfo(index=i, name=p.name, raw_units=p.raw_units, kind=p.kind)
            for i, p in enumerate(scan.plans)
        )
        return RecordingProbe(self._meta(scan, channels, scan.base_len, loaded_range=None))

    # -- load ------------------------------------------------------------------------------------
    def load(self, sample_range: SampleRange | None = None,
             progress: ProgressFn | None = None,
             cancel: CancelFn | None = None) -> Document:
        scan = self._scan()

        # Resolve the load window against the (possibly truncated) common length — through the
        # shared clamp, so a window that selects nothing is refused with the same sentence the
        # other three readers use rather than producing a zero-sample document.
        # Carried on the scan (which is what `_meta` reads) so a clamp reported here survives
        # the bulk read below, which rebuilds the report only when the decoder says something.
        report = ReaderReport(notes=scan.notes, issues=scan.issues)
        if sample_range is None:
            start, stop = 0, scan.base_len
        else:
            start, stop = clamp_request(
                sample_range, scan.base_len, name=self.path.name, report=report)
            scan.issues = report.issues
        n = stop - start

        n_ch = len(scan.plans)
        arrays: list[np.ndarray] = [np.empty(n, dtype=np.float32) for _ in range(n_ch)]

        total = max(1, n_ch * n)
        done = 0
        if progress is not None:
            progress(0.0, f"Loading {self.path.name}")

        # The bulk read is a second chance for nptdms to notice something the metadata pass did
        # not, so it is captured on the same terms: nothing the decoder says about the user's
        # file may end its life in a log handler.
        audits: list[DowncastAudit | None] = [None] * n_ch
        with decoder_messages("nptdms") as decoder_said:
            self._read_into(arrays, scan, start, stop, n, total, done, progress, cancel,
                            audits=audits)
        audit_issues = [msg for a in audits if a is not None for msg in (a.issue(),) if msg]
        if decoder_said or audit_issues:
            report = ReaderReport(notes=scan.notes, issues=scan.issues)
            for message in decoder_said:
                report.issue(decoder_message_issue("TDMS", message))
            for message in audit_issues:
                report.issue(message)
            scan.issues = report.issues

        channels: list[ChannelInfo] = []
        for i, plan in enumerate(scan.plans):
            is_dead, reason = detect_dead(arrays[i])
            audit = audits[i]
            if (is_dead and reason is not None and reason.startswith("constant 0.0")
                    and audit is not None and audit.dead_reason() is not None):
                # The flat line is the float64->float32 downcast's doing, not the file's:
                # detect_dead reads the loaded float32 and cannot know the file's samples were
                # nonzero, so its "constant 0.0" would assert a property no sample in the file
                # has — same substitution the HDF5 reader makes, for the same reason.
                reason = audit.dead_reason()
            channels.append(ChannelInfo(
                index=i, name=plan.name, raw_units=plan.raw_units, kind=plan.kind,
                is_dead=is_dead, dead_reason=reason,
            ))

        if progress is not None:
            progress(1.0, f"Loaded {self.path.name}")

        meta = self._meta(scan, tuple(channels), n, window_loaded(sample_range, start, n),
                          start=start)
        return TraceDocument(meta=meta, channels=tuple(arrays))

    def _read_into(self, arrays: list[np.ndarray], scan: _Scan, start: int, stop: int, n: int,
                   total: int, done: int, progress: ProgressFn | None,
                   cancel: CancelFn | None,
                   audits: list[DowncastAudit | None] | None = None) -> None:
        try:
            tf = TdmsFile.open(str(self.path))
        except Exception as exc:
            raise self._open_refusal(exc) from exc

        try:
            for ci, plan in enumerate(scan.plans):
                channel = tf[plan.tdms_group][plan.tdms_channel]
                dst = arrays[ci]
                # A TDMS channel can be float64 on disk, and the in-place downcast below
                # flushes any nonzero |x| < ~1.4e-45 to exactly 0.0 in silence. The shared
                # audit (io.base.DowncastAudit — same rule and sentence as the HDF5 and text
                # readers) counts what the cast destroys. The disk dtype is only knowable from
                # the first block nptdms hands over, so the audit is created there; a float32
                # or integer channel skips the per-chunk pass entirely.
                audit: DowncastAudit | None = None
                first_block = True
                pos = start
                while pos < stop:
                    if cancel is not None and cancel():
                        raise LoadCancelled(f"{self.path.name}: load cancelled.")
                    end = min(pos + LOAD_CHUNK_SAMPLES, stop)
                    # float64 on disk -> in-place downcast to the float32 destination slice
                    block = np.asarray(channel[pos:end])
                    if first_block:
                        first_block = False
                        if downcast_needs_audit(block.dtype):
                            audit = DowncastAudit(plan.name, str(block.dtype))
                    seg = dst[pos - start:end - start]
                    seg[:] = block
                    if audit is not None:
                        audit.observe(block, seg)
                    done += end - pos
                    pos = end
                    if progress is not None:
                        progress(done / total, f"Loading {plan.name}")
                if audits is not None:
                    audits[ci] = audit
        finally:
            tf.close()
