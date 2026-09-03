from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np

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
    downcast_needs_audit,
    filesystem_refusal,
    foreign_format_redirect,
    implausible_measured_rate_issue,
    read_magic,
    register,
)
from .contract import (
    ReaderReport,
    clamp_request,
    oversized_timebase_issue,
    partial_timebase_issue,
    rate_disagrees_issue,
    rejected_channel_issue,
    window_loaded,
)
from .errors import CorruptFileError, LoadCancelled, ReaderError, UnsupportedFormatError

__all__ = ["Hdf5Reader"]

# HDF5 superblock signature (first 8 bytes of every HDF5 file).
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# The only dataset names this flat format recognises as SIGNAL channels. Kind comes from the NAME
# (unambiguous, unlike a CSV header); UNITS do NOT — they are not declared in these files.
# [VERIFIED] a genuine recording pairs current/voltage/(conductance) with a /time base. A lone
# /current dataset with no time and no companion channel is a data export (e.g. an ML training
# split, 'training_split.h5'), NOT a recording — see probe()'s acceptance gate below.
_NAME_KIND: dict[str, ChannelKind] = {
    "current": ChannelKind.CURRENT,
    "voltage": ChannelKind.VOLTAGE,
    "conductance": ChannelKind.CONDUCTANCE,
}
# Canonical channel order: stable ChannelInfo.index regardless of h5py's alphabetical key order.
_SIGNAL_ORDER: tuple[str, ...] = ("current", "voltage", "conductance")

_UNITS_NOTE = ("Units are not declared in this file. Magnitudes suggest nA / V / nS — verify and "
               "set them in the Calibration panel.")

# float32 represents every integer up to 2**24 exactly and thereafter starts skipping them. A
# 16- or 24-bit ADC count is always below this; a raw count above it would be silently rounded,
# which is a number changing without anyone being told.
_FLOAT32_EXACT_INT = 1 << 24

# The whole-file duration may differ from the one the measured rate implies by this much before
# the rate is wrong for this file.
_SPAN_TOL = 1e-3


class _Signal:
    """One accepted signal dataset, and how to read a window of it as 1-D samples.

    A recording's channel is a single run of samples. On disk that is USUALLY a 1-D float
    dataset, and this reader used to accept nothing else — so a ``/current`` written as int32
    (raw ADC counts, which is what several acquisition programs write) or as ``(1, N)`` (which
    is what h5py gives you for a row of a 2-D block) was dropped from the recording without a
    word: measured on the tree as found, ``channels: ['voltage'], issues: ()``. The current
    channel — the only one event detection can run on — was simply not there.

    Accepting a shape is not the whole fix; the other half is :meth:`Hdf5Reader._reject_reason`,
    which makes anything still refused SAY SO.
    """

    __slots__ = ("name", "dataset", "length", "_axis", "is_integer")

    def __init__(self, name: str, dataset: h5py.Dataset, axis: int | None) -> None:
        self.name = name
        self.dataset = dataset
        self._axis = axis                 # None: 1-D. 0: shape (1, N). 1: shape (N, 1).
        self.length = int(dataset.shape[0] if axis is None or axis == 1 else dataset.shape[1])
        self.is_integer = bool(np.issubdtype(dataset.dtype, np.integer))

    def read(self, a: int, b: int) -> np.ndarray:
        if self._axis is None:
            return self.dataset[a:b]
        if self._axis == 0:
            return self.dataset[0, a:b]
        return self.dataset[a:b, 0]


@register
class Hdf5Reader(BaseReader):
    """Flat HDF5 recordings: top-level 1-D float datasets /current, /voltage, /conductance plus a
    /time base. Tolerates non-signal GROUPS (e.g. /display_settings). Units are NOT declared on
    disk, so raw_units is "" for every channel; kind is derived from the dataset name only.

    [VERIFIED] on this machine, both corpus recordings:
        '2_3 electroporation-005_Cut_1.h5'                  N=2,000,000  fs=50000  t0=60.0000200
            channels current, voltage; /display_settings empty group; gzip float64 on disk
        '4_detection 100 pM nisin with fibrin -014_..._Copy1.h5'  N=9,800,724  fs=50000  t0=0.0
            channels current, voltage, conductance; conductance == current/voltage
    and rejects the non-recording .h5 files in the same directory (Keras models with
    /model_weights+/optimizer_weights groups; 'training_split.h5' with a bare /current and no /time;
    MEArec files with no signal dataset).
    """

    format_name: ClassVar[str] = "HDF5 flat"
    extensions: ClassVar[tuple[str, ...]] = (".h5", ".hdf5")

    @classmethod
    def sniff(cls, path: Path) -> Claim:
        """Format ownership only: an HDF5 file with a recognised extension is claimed STRONG. This
        is NOT "can I decode it" — probe() raises UnsupportedFormatError for HDF5 files that are
        Keras models or bare data exports rather than recordings."""
        if path.suffix.lower() not in cls.extensions:
            return Claim.NO
        if read_magic(path, 8) == _HDF5_MAGIC:
            return Claim.STRONG
        # WEAK, not NO. A .h5 that is not HDF5 is still OURS — see Claim's own docstring. With
        # NO, select_reader fell through to "its extension ('.h5') matches no reader", from an
        # app whose file dialog lists .h5, which sends the user looking for a plugin that was
        # never missing. Claimed here, probe() names the real problem: it is not an HDF5 file.
        return Claim.WEAK

    # -- helpers -------------------------------------------------------------------------------
    def _open_refusal(self, exc: OSError) -> ReaderError:
        """The user-facing refusal for an h5py open/read failure, classified from the
        FILESYSTEM, not from ``exc``.

        h5py's own words for these failures — "[Errno 2] Unable to open file (unable to open
        file: name = '/abs/path', errno = 2, …)" for a gone one, the same shape for a locked
        one or a directory — are library internals that io/errors.py promises never reach the
        UI, wrapped in "not a readable HDF5 file", which is corruption wording for a file
        that is merely gone, locked, or a folder. The real cause is re-derived from the disk by
        ``io.base.filesystem_refusal`` — the same call ``abf._refusal``, ``tdms._open_refusal``
        and ``text._readable`` make, so the dialogs for one condition cannot disagree by format
        as the four readers are edited apart.
        """
        gone_locked_or_folder = filesystem_refusal(self.path, exc)
        if gone_locked_or_folder is not None:
            return CorruptFileError(gone_locked_or_folder)
        renamed = foreign_format_redirect(self.path)
        if renamed is not None:
            # An ABF, a TDMS or a text export wearing an .h5 name. Was: h5py's "file signature
            # not found" — corruption wording for an intact recording of another format, while
            # the mirror case (HDF5 bytes named .abf) was told exactly what it was and where to
            # rename it. One condition, one sentence, in every direction.
            return UnsupportedFormatError(renamed)
        return CorruptFileError(f"{self.path.name}: not a readable HDF5 file ({exc}).")

    @staticmethod
    def _as_signal(name: str, obj) -> _Signal | None:
        """``obj`` as a run of samples, or None if it is not one shape/dtype we can read."""
        if not isinstance(obj, h5py.Dataset):
            return None
        dtype_ok = (np.issubdtype(obj.dtype, np.floating)
                    or np.issubdtype(obj.dtype, np.integer))
        if not dtype_ok:
            return None
        if obj.ndim == 1:
            return _Signal(name, obj, None)
        if obj.ndim == 2 and obj.shape[0] == 1:
            return _Signal(name, obj, 0)
        if obj.ndim == 2 and obj.shape[1] == 1:
            return _Signal(name, obj, 1)
        return None

    @staticmethod
    def _reject_reason(obj) -> str:
        """Why a dataset this reader recognises by NAME still cannot be a channel.

        Exists because the silence was the defect: a rejected channel that says nothing is
        indistinguishable, in the Files list, from a file that never had one.
        """
        if obj is None:
            return "it is not present"
        if not isinstance(obj, h5py.Dataset):
            return f"it is a {type(obj).__name__.lower()}, not a dataset of samples"
        if not (np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.integer)):
            return (f"its dtype is {obj.dtype}, which is not a number this reader can turn into "
                    f"a signal")
        return (f"its shape is {tuple(int(v) for v in obj.shape)}; a channel must be a single "
                f"run of samples (1-D, or 2-D with one row or one column)")

    def _signals(self, h: h5py.File) -> tuple[list[_Signal], ReaderReport]:
        """Accepted signal channels in canonical order, plus what was said about the rest."""
        found: list[_Signal] = []
        report = ReaderReport()
        for name in _SIGNAL_ORDER:
            obj = h.get(name)
            if obj is None:
                continue
            sig = self._as_signal(name, obj)
            if sig is None:
                report.issue(rejected_channel_issue(f"/{name}", self._reject_reason(obj)))
                continue
            # A ZERO-LENGTH dataset is deliberately NOT a rejection. It is a structurally valid
            # channel that holds nothing, which is the empty-recording condition and belongs to
            # the door that owns it — ``probe_path`` and ``TraceDocument`` — so that an
            # acquisition armed and stopped without recording gets the one sentence written for
            # it ("the file decoded but holds no samples…") rather than a channel-rejection
            # notice that describes the wrong problem in different words for each format.
            found.append(sig)
            if sig.is_integer:
                report.note(
                    f"/{name} is stored as {sig.dataset.dtype} — raw ADC counts, not a physical "
                    f"unit. It is read as-is; set a gain and units in the Calibration panel.")
        return found, report

    def _time_base(self, h: h5py.File) -> _Signal | None:
        """/time as a structurally valid channel, WHATEVER it holds — or None if it is not one.

        Two different questions are asked about /time and they had one answer between them:

        * "does this file carry a time base at all?" — which is what the acceptance gate in
          :meth:`probe` / :meth:`load` asks, to tell a RECORDING from a bare data export; and
        * "can a sample interval be measured from it?" — which is :meth:`_time_signal`, and
          needs at least two samples to difference.

        Answering the first with the second denied a /time that is plainly there. An
        acquisition armed and stopped writes ``/time`` and ``/current`` both zero-length; the
        gate saw "no time base", fell into the lone-signal branch and refused the file with
        "/current present but no /time base … this looks like a data export, not a recording
        (top-level members: current, time)" — a sentence that denies /time in the same breath
        as listing it, and sends the user hunting for an export tool that never made the file.
        Adding one more EMPTY dataset (/voltage) flipped the same file to the right sentence,
        so one condition had two stories depending on channel count.

        A zero-length /time is exactly what :meth:`_signals` already says a zero-length CHANNEL
        is: structurally valid, holding nothing, and the business of the empty-recording door
        (``probe_path``/``TraceDocument``), which has one sentence written for it. This method
        is that rule applied to the clock. Measurability is unchanged and still reported: a
        /time too short to difference gets its issue from :meth:`_timebase` and the rate stays
        UNKNOWN, never fabricated.
        """
        return self._as_signal("time", h.get("time"))

    def _time_signal(self, h: h5py.File) -> _Signal | None:
        """/time usable as a RATE SOURCE: a structural time base with at least 2 samples."""
        sig = self._time_base(h)
        return sig if sig is not None and sig.length >= 2 else None

    def _common_length(self, signals: list[_Signal]) -> int:
        """The one length every channel of a rectangular recording shares, or a refusal."""
        n = signals[0].length
        if any(s.length != n for s in signals):
            detail = ", ".join(f"/{s.name}={s.length:,}" for s in signals)
            raise CorruptFileError(
                f"{self.path.name}: signal channels have mismatched lengths ({detail}).")
        return n

    def _reject_message(self, h: h5py.File, signals: list[_Signal],
                        rejected: tuple[str, ...] = ()) -> str:
        members = ", ".join(sorted(h.keys())) or "(none)"
        tail = (" " + " ".join(rejected)) if rejected else ""
        if signals:
            only = ", ".join(f"/{s.name}" for s in signals)
            obj = h.get("time")
            if obj is not None:
                # /time IS in the file, it just is not a run of samples this reader can read
                # (a group, a string dtype, a 2-D block). Reaching for "no /time base" here
                # would deny a member the same sentence goes on to list. Name what is actually
                # wrong with it, and do not call the file an export: something wrote a clock.
                return (f"{self.path.name}: {only} present and /time present but unusable as a "
                        f"time base ({self._reject_reason(obj)}), and there is no second signal "
                        f"channel to read this file without one "
                        f"(top-level members: {members}).{tail}")
            # A lone signal dataset with no /time and no companion channel: a data export
            # (e.g. an ML training split), not a recording. Say so honestly.
            return (f"{self.path.name}: {only} present but no /time base and no second signal "
                    f"channel — this looks like a data export, not a recording "
                    f"(top-level members: {members}).{tail}")
        return (f"{self.path.name}: no readable /current, /voltage or /conductance dataset found "
                f"(top-level members: {members}).{tail}")

    def _build_meta(self, *, n_samples: int, sample_rate_hz: float | None,
                    rate_provenance: Provenance, t0_s: float,
                    loaded_range: SampleRange | None,
                    channels: tuple[ChannelInfo, ...],
                    report: ReaderReport) -> RecordingMeta:
        return RecordingMeta(
            path=self.path,
            format_name=self.format_name,
            n_samples=n_samples,
            sample_rate_hz=sample_rate_hz,
            rate_provenance=rate_provenance,
            channels=channels,
            t0_s=t0_s,
            loaded_range=loaded_range,
            nominal_voltage_mV=None,             # units unknown => no honest mV
            voltage_provenance=Provenance.UNKNOWN,
            notes=(_UNITS_NOTE,) + report.notes,
            issues=report.issues,
        )

    def _timebase(self, h: h5py.File, start: int, n_signal: int,
                  report: ReaderReport) -> tuple[float | None, Provenance, float]:
        """Derive (sample_rate_hz, provenance, t0_s) from /time WITHOUT loading it whole.

        Reads only the first ~1000 samples for dt, plus one sample at `start` for t0, plus — and
        this is the part that was missing — the LAST sample, which HDF5 hands over in O(1).

        dt itself is measured exactly as it always was, from the head, deliberately: deriving it
        differently would move the sample rate, and with it every dwell time, on every
        well-formed file in the corpus. What the last sample buys is the ANSWER TO WHETHER THAT
        RATE DESCRIBES THE FILE — the same blind spot that let a CSV report 0.06 s for a 0.33 s
        recording. Here it is free.

        If /time is absent the rate is genuinely unknown (never fabricated): (None, UNKNOWN, 0.0).

        A /time that EXISTS but cannot be used — string dtype, a 2-D block, a single sample —
        gets the same (None, UNKNOWN, 0.0) answer WITH an issue saying so. It used to get the
        answer silently, which made the file indistinguishable from one that never carried a
        time base (io/contract.py question 3: "A refusal the user cannot see is
        indistinguishable from a file that never had the channel") while the value-level
        defects of the very same dataset — non-monotonic, non-uniform, partial — were all
        reported loudly. The user whose acquisition software wrote /time in an unreadable
        shape deserves to know the file carries a clock this reader refused, and why.
        """
        obj = h.get("time")
        if obj is None:
            return None, Provenance.UNKNOWN, 0.0
        tsig = self._time_signal(h)
        if tsig is None:
            partial = self._as_signal("time", obj)
            why = (f"it holds only {partial.length:,} sample(s), and at least 2 are needed "
                   f"to measure a sample interval"
                   if partial is not None else self._reject_reason(obj))
            report.issue(f"/time is present in this file but is not usable as a time base "
                         f"({why}); sample rate unknown")
            return None, Provenance.UNKNOWN, 0.0
        n_time = tsig.length
        head = np.asarray(tsig.read(0, 1000), dtype=np.float64)
        d = np.diff(head)
        if d.size == 0 or not np.all(d > 0):
            # Degenerate/non-increasing time head: cannot trust it as a rate source.
            report.issue("/time is not usable as a monotonic time base; sample rate unknown")
            t0 = float(tsig.read(start, start + 1)[0]) if start < n_time else 0.0
            return None, Provenance.UNKNOWN, t0
        dt = float(d.mean())
        sample_rate_hz = 1.0 / dt

        # --- IS THIS CLOCK IN SECONDS AT ALL? ---------------------------------------------------
        # Every check below this line — uniformity, span, partial/oversized — passes TRIVIALLY on
        # a /time written in microseconds or nanoseconds: integer counts are perfectly uniform and
        # their span agrees with their own step exactly. So the file that is most wrong is the one
        # that looks cleanest. [MEASURED] an int64 /time in µs (dt = 100) reported 0.01 Hz,
        # Provenance.MEASURED, duration 500,000 s for 5,000 samples of a 10 kHz recording — 10^6x
        # wrong, issues=(), against a guide that promises "a file that shows nothing there was
        # read clean". The unit is not declared anywhere in this format, so the reading is refused
        # rather than converted by a guess: see io.base.MIN_PLAUSIBLE_RATE_HZ, which the text
        # reader applies to a CSV's time COLUMN for the same reason.
        if sample_rate_hz < MIN_PLAUSIBLE_RATE_HZ:
            report.issue(implausible_measured_rate_issue(
                source="/time", dt=dt, hz=sample_rate_hz,
                stored_as=str(tsig.dataset.dtype) if tsig.is_integer else None))
            return None, Provenance.UNKNOWN, 0.0
        if tsig.is_integer:
            # The mirror of the sentence _signals() already gives an integer SIGNAL dataset
            # ("/current is stored as int32 — raw ADC counts…"). Same fact, same file, same
            # obligation: an integer clock that survived the plausibility check above is being
            # read as seconds, and the user is the only one who can confirm that is what it is.
            report.note(
                f"/time is stored as {tsig.dataset.dtype} — an integer time base, read as "
                f"whole seconds ({sample_rate_hz:,.6g} Hz). If this file's clock is in "
                f"milliseconds, microseconds or nanoseconds, every duration and rate here is "
                f"wrong by that factor: re-export /time as float seconds.")
        std = float(d.std())
        if std > 1e-6 * dt:
            report.issue(f"/time is non-uniform (std/dt = {std / dt:.3g})")

        if n_time < n_signal:
            report.issue(partial_timebase_issue(n_time=n_time, n_signal=n_signal))
        else:
            # The MIRROR of the branch above, and it was the silent one. A /time longer than the
            # signal means the signal datasets stopped early or the time base is not these
            # samples'. [MEASURED] before this: /current and /voltage of 4,000 beside a /time of
            # 8,000 probed with issues=() — a recording described as 0.08 s long whose own clock
            # says 0.16 s. `_common_length` only cross-checks the signal channels against each
            # other, so nothing else in this reader was ever going to notice.
            if n_time > n_signal:
                report.issue(oversized_timebase_issue(
                    n_time=n_time, n_signal=n_signal, dt=dt))
            last = float(tsig.read(n_time - 1, n_time)[0])
            implied = (n_time - 1) * dt
            actual = last - float(head[0])
            if abs(actual - implied) > max(_SPAN_TOL * abs(implied), 2.0 * dt):
                report.issue(rate_disagrees_issue(
                    hz=sample_rate_hz, measured_over=int(head.size), source="samples of /time",
                    implied_s=implied, actual_s=actual))

        if start < n_time:
            t0_s = float(tsig.read(start, start + 1)[0])
        else:
            # A window starting PAST the end of a partial /time. Extrapolate the window's own
            # t0 from the measured rate — head[0] + start·dt — exactly as the in-range branch
            # would have read it had /time reached this far, and exactly what the partial-
            # timebase issue above already promised ("the rest is assumed to continue at the
            # same rate"). Falling back to head[0] — the FILE's origin — put every cursor
            # readout, x-axis value and segment rebase of such a window start/fs seconds
            # early, the precise defect class contract.window_t0 was built to end.
            t0_s = float(head[0]) + start * dt
        return sample_rate_hz, Provenance.MEASURED, t0_s

    # -- probe ---------------------------------------------------------------------------------
    def probe(self) -> RecordingProbe:
        try:
            with h5py.File(self.path, "r") as h:
                signals, report = self._signals(h)
                # STRUCTURAL, not measurable: a /time this reader recognises as a channel means
                # the file carries a clock, even when it is too short to measure a rate from —
                # see _time_base. Whether a rate can be derived is _timebase's separate answer.
                has_time = self._time_base(h) is not None
                # Acceptance gate: at least one signal dataset AND either a time base or >=2 signal
                # channels. This rejects the bare-/current export ('training_split.h5') and every
                # Keras/MEArec .h5 (no signal dataset at all) while still accepting a multi-channel
                # recording that happens to lack /time (rate then reported UNKNOWN, never faked).
                if not signals or (not has_time and len(signals) < 2):
                    raise UnsupportedFormatError(
                        self._reject_message(h, signals, report.issues))

                n = self._common_length(signals)
                sample_rate_hz, rate_prov, t0_s = self._timebase(h, 0, n, report)

                channels = tuple(
                    ChannelInfo(index=i, name=s.name, raw_units="", kind=_NAME_KIND[s.name])
                    for i, s in enumerate(signals)
                )
                meta = self._build_meta(
                    n_samples=n, sample_rate_hz=sample_rate_hz,
                    rate_provenance=rate_prov, t0_s=t0_s, loaded_range=None,
                    channels=channels, report=report,
                )
                return RecordingProbe(meta=meta)
        except ReaderError:
            raise
        except OSError as e:
            raise self._open_refusal(e) from e
        except Exception as e:  # h5py internals must never leak to the UI
            raise CorruptFileError(f"{self.path.name}: {e}") from e

    # -- load ----------------------------------------------------------------------------------
    def load(self, sample_range: SampleRange | None = None,
             progress: ProgressFn | None = None,
             cancel: CancelFn | None = None) -> Document:
        try:
            with h5py.File(self.path, "r") as h:
                signals, report = self._signals(h)
                has_time = self._time_base(h) is not None      # structural; see probe()
                if not signals or (not has_time and len(signals) < 2):
                    raise UnsupportedFormatError(
                        self._reject_message(h, signals, report.issues))

                full_n = self._common_length(signals)
                if sample_range is None:
                    start, stop = 0, full_n
                else:
                    start, stop = clamp_request(
                        sample_range, full_n, name=self.path.name, report=report)
                n = stop - start

                # full_n, not n: whether /time covers the RECORDING is a fact about the file,
                # and must not change depending on how much of it the user chose to load.
                sample_rate_hz, rate_prov, t0_s = self._timebase(h, start, full_n, report)

                # --- read each channel: gzip float64 on disk -> preallocated contiguous float32.
                # `dst[a:b] = sig.read(a, b)` with a float32 dst does the downcast in place, no
                # float64 temporary of the whole channel. Progress/cancel poll per chunk (gzip
                # inflate is the slow path this exists for).
                total_work = n * len(signals)
                done = 0
                arrays: list[np.ndarray] = []
                # One audit per wide-float channel, kept aligned with `signals` so the
                # dead-hint substitution below can find its channel's audit again. The mirror
                # of the integer-peak scan two branches down: overflow of this same downcast is
                # reported by core.document (every sample non-finite) and integer rounding by
                # the peak check, so underflow — nonzero float64 below float32's smallest
                # subnormal flushing to exactly 0.0 — was the one destruction that said
                # nothing. The rule and its sentence are io.base's, shared with the TDMS and
                # text readers, which read the same wide sources.
                audits: list[DowncastAudit | None] = []
                for sig in signals:
                    dst = np.empty(n, dtype=np.float32)
                    audit = (DowncastAudit(f"/{sig.name}", str(sig.dataset.dtype))
                             if downcast_needs_audit(sig.dataset.dtype) else None)
                    peak = 0.0
                    a = 0
                    while a < n:
                        if cancel is not None and cancel():
                            raise LoadCancelled(f"{self.path.name}: load cancelled.")
                        b = min(a + LOAD_CHUNK_SAMPLES, n)
                        block = sig.read(start + a, start + b)
                        if sig.is_integer:
                            # min/max separately, NOT max(abs(block)): abs() of the most negative
                            # value of a signed integer dtype overflows back to itself, so the
                            # exactly-representable check would pass on the one value it exists
                            # to catch.
                            peak = max(peak, abs(float(block.min())), abs(float(block.max())))
                        dst[a:b] = block                       # -> float32 in place
                        if audit is not None:
                            audit.observe(np.asarray(block).reshape(-1), dst[a:b])
                        done += (b - a)
                        if progress is not None:
                            progress(done / total_work, f"Reading {sig.name} ({b}/{n})")
                        a = b
                    if sig.is_integer and peak >= _FLOAT32_EXACT_INT:
                        report.issue(
                            f"/{sig.name} holds raw {sig.dataset.dtype} counts up to "
                            f"{peak:,.0f}, above the {_FLOAT32_EXACT_INT:,} beyond which float32 "
                            f"cannot represent every integer. Individual sample values in this "
                            f"channel are rounded, so the smallest current steps in it are lost.")
                    if audit is not None:
                        report.issue(audit.issue())
                    arrays.append(dst)
                    audits.append(audit)

                # detect_dead per channel (needs sampled data; probe never sets it)
                channels: list[ChannelInfo] = []
                for i, sig in enumerate(signals):
                    is_dead, reason = detect_dead(arrays[i])
                    audit = audits[i]
                    if (is_dead and reason is not None and reason.startswith("constant 0.0")
                            and audit is not None and audit.dead_reason() is not None):
                        # The flat line is the downcast's doing, not the file's: detect_dead
                        # reads the loaded float32 and cannot know the file's float64 samples
                        # were all nonzero, so its "constant 0.0" would assert a property no
                        # sample in the file has — the File Info hint and the Issues box must
                        # not tell two stories about one channel.
                        reason = audit.dead_reason()
                    channels.append(ChannelInfo(index=i, name=sig.name, raw_units="",
                                                kind=_NAME_KIND[sig.name],
                                                is_dead=is_dead, dead_reason=reason))

                meta = self._build_meta(
                    n_samples=n, sample_rate_hz=sample_rate_hz,
                    rate_provenance=rate_prov, t0_s=t0_s,
                    loaded_range=window_loaded(sample_range, start, n),
                    channels=tuple(channels), report=report,
                )
                return TraceDocument(meta=meta, channels=tuple(arrays))
        except ReaderError:
            raise
        except OSError as e:
            raise self._open_refusal(e) from e
        except Exception as e:
            raise CorruptFileError(f"{self.path.name}: {e}") from e
