from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .types import ChannelInfo, RecordingMeta

__all__ = ["Document", "TraceDocument", "CalibratedChannel", "XAxis"]


@runtime_checkable
class Document(Protocol):
    """Anything the app can hold open. NOT everything openable is a trace: the user's
    '*Saved Data.txt' files are ragged event tables. [VERIFIED] 375 rows (367x7 + 6x9 + 2x11)
    and 788 rows (461x7 + 95x9 + 169x11 + 52x13 + 4x15 + 4x17 + 3x23) — all odd, +2 increments.
    This union is a v1 FACT, not a v2 speculation. Dispatch returns Document, never Recording.
    """

    @property
    def doc_id(self) -> str: ...

    @property
    def source_key(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def meta(self) -> RecordingMeta: ...


@dataclass(frozen=True)
class TraceDocument:
    """Immutable loaded signal. A value object. NEVER serialised. Holds NO calibration —
    calibration is Session state, so the same 180 MB array is never trapped with user edits.

    channels: one CONTIGUOUS 1-D float32 array per channel, NOT an (n, n_ch) block.
    [VERIFIED] at N=15M: block `data[:,1]*g+o` = 12.14 ms (C_CONTIGUOUS False);
    per-channel contiguous = 8.50 ms; forced ascontiguousarray = 9.47 ms + 60 MB.
    torch.from_numpy() on a strided slice yields a strided tensor that copies downstream.
    Readers pay the contiguity cost ONCE, off the main thread, where it is invisible.
    """

    meta: RecordingMeta
    channels: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        n = self.meta.n_samples
        assert len(self.channels) == len(self.meta.channels), "channels/meta.channels length mismatch"
        for i, a in enumerate(self.channels):
            assert a.dtype == np.float32, f"ch{i}: {a.dtype} — float32 is BINDING"
            assert a.ndim == 1, f"ch{i}: expected 1-D, got {a.ndim}-D"
            assert a.flags["C_CONTIGUOUS"], f"ch{i}: non-contiguous"
            assert len(a) == n, f"ch{i}: len {len(a)} != meta.n_samples {n}"
        self._reject_empty()
        self._stamp_load_uid()
        self._report_nonfinite()

    def _reject_empty(self) -> None:
        """A recording with NO SAMPLES is refused here, at the one door all of them use.

        HERE, and not in the four readers, for exactly the reason
        :meth:`_report_nonfinite` gives directly below: a rule written in a reader has to
        be written four times and still misses the fifth producer. Measured on the tree as
        found, an HDF5 with ``/current`` and ``/voltage`` both length 0, and a TDMS with a
        zero-sample channel, BOTH loaded happily with ``n_samples=0`` and ``issues=()``
        (the text reader was the only one that already refused, with "no data rows were
        read"). ``core.segment`` refuses below ``MIN_SEGMENT_SAMPLES``, so this closes the
        remaining producers in one line.

        WHY IT IS FATAL rather than an ``issues`` note, which is how non-finite samples are
        handled. A recording with NaN in it is still a recording: it draws, it has a
        duration, every panel can describe it. A recording with no samples is not one —
        it has no time base to plot against, no duration to divide by, and no value for
        any statistic. What actually happened on the tree as found: selecting it raised
        ``ValueError: zero-size array to reduction operation minimum which has no
        identity`` out of the calibration panel, HALFWAY THROUGH ``_on_selection``, so the
        window was left split — the title bar, the calibration panel and
        ``_current_selection`` on the empty recording, the analysis panel still on the
        previous one. Pressing Detect then ran on the recording the user was no longer
        looking at, and reported its event count as if it were this one.
        """
        if self.meta.n_samples > 0:
            return
        from ..io.errors import EmptyRecordingError, empty_recording_message

        raise EmptyRecordingError(empty_recording_message(self.meta.path.name))

    def _stamp_load_uid(self) -> None:
        """Give this load its own identity, HERE and nowhere else.

        Same argument as :meth:`_report_nonfinite` directly below, applied to identity
        instead of to NaNs: every path that produces a recording — the four readers, and a
        segment cut from any of them — builds this object, so stamping here is the one place
        a load can be identified without the fix having to be written four times and still
        missing the fifth.

        WHAT IT BUYS. ``doc_id`` used to be the resolved path, so a *different load of the
        same file* was indistinguishable from the load that is open: close a recording while
        a detection runs, reopen it (at a different range, or after the file was re-exported
        over itself) and the in-flight result passed every
        ``session.has_document(res.doc_id)`` check and landed on the new document. Measured
        on a re-opened 4,000-sample load of a 100,000-sample recording: all five events past
        the end of the file, ``analyzed_duration_s`` 1.0 s against a true 0.04 s — the
        capture rate wrong by 25x — and no staleness banner, because the calibration had not
        changed. See :mod:`nanoscope.core.identity`.

        IDEMPOTENT, and load-bearing rather than tidy. This is a frozen VALUE object that the
        app rebuilds with ``dataclasses.replace`` (``Session.add_document`` folds the
        sidecar's issues into ``meta`` that way), which re-enters ``__post_init__``. Minting a
        second token there would change ``doc_id`` after the session had already keyed the
        document under the first one — every layer, calibration, result and undo stack of that
        recording orphaned in one line. An existing token is therefore always kept.
        """
        if not self.meta.load_uid:
            from dataclasses import replace

            from .identity import new_load_uid
            object.__setattr__(self, "meta", replace(self.meta, load_uid=new_load_uid()))

    def _report_nonfinite(self) -> None:
        """Name every channel carrying NaN/inf samples in ``meta.issues``.

        HERE, at construction, and not in the four readers — because the reader is the one
        place a fix would have to be written four times and would still miss the fifth.
        Every path that produces a recording (ABF, HDF5, TDMS, text, and a segment cut from
        any of them) builds this object, so every path reports.

        Why it matters at all: a single blank cell in a CSV becomes one NaN, and Welch
        averages segment periodograms, so that ONE sample poisons every bin of the spectrum
        and ``band_rms`` returned ``0.0``. The panel then printed "Total RMS noise: 0 pA"
        for a trace whose true σ was 1.49 pA — the most convincing-looking QC result a
        noise panel can produce. The load reported ``issues: ()``.

        Non-FATAL by design: the recording still opens and is still viewable. The
        guarantee is only that the app never pretends it did not notice. (The other half
        of the guarantee is :func:`nanoscope.analysis.psd.band_rms`, which now returns NaN
        rather than 0.0 for a poisoned band.)

        IDEMPOTENT, and that is not decoration. This is a frozen VALUE object, so the app
        rebuilds it with ``dataclasses.replace`` to carry extra ``meta.issues`` (see
        ``Session.add_document`` folding in the sidecar's) — which re-enters ``__post_init__``
        over a meta that already holds this very sentence. Appending unconditionally reported
        the same NaN twice, and once more per rebuild after that.
        """
        from dataclasses import replace

        from .types import nonfinite_issue, scan_nonfinite

        already = set(self.meta.issues)
        found: list[str] = []
        for i, a in enumerate(self.channels):
            count, first = scan_nonfinite(a)
            if count:
                name = self.meta.channels[i].name if i < len(self.meta.channels) else ""
                issue = nonfinite_issue(i, name, count, first, len(a))
                if issue not in already:
                    found.append(issue)
        if found:
            # Frozen value objects: rebuild meta rather than mutate it. dataclasses.replace
            # reuses everything by reference, so this costs nothing beyond the tuple.
            object.__setattr__(
                self, "meta", replace(self.meta, issues=tuple(self.meta.issues) + tuple(found)))

    @property
    def source_key(self) -> str:
        """WHICH FILE — the resolved source path, stable across loads and sessions.

        This is the answer to "is this recording already open?" and to "which file did this
        result come from?"; it is what a file-backed document persists under (see
        :func:`nanoscope.core.segment.persistent_key`, which returns this for a file and the
        uid for a segment). It is NOT the session identity — see :attr:`doc_id`.
        """
        return str(self.meta.path.resolve())

    @property
    def doc_id(self) -> str:
        """WHICH LOAD — the source key plus this load's own token.

        Unique to this load: reopening the same file yields a different ``doc_id``, which is
        what makes ``session.has_document(res.doc_id)`` mean "the document this was actually
        computed from" rather than merely "some document with the same name". Opaque — only
        :mod:`nanoscope.core.identity` may build or read its shape.
        """
        from .identity import make_doc_id
        return make_doc_id(self.source_key, self.meta.load_uid)

    @property
    def display_name(self) -> str:
        return self.meta.path.name

    def raw(self, channel: int) -> np.ndarray:
        """Uncalibrated, signed, float32. Calibrated data comes from Session.calibrated()."""
        return self.channels[channel]


# --- v2 SEAM. Declared, deliberately NOT implemented. -------------------------------------------
# @dataclass(frozen=True)
# class EventTableDocument:
#     """v2. The user's '*Saved Data.txt' exports are already this shape on disk: one row per
#     event; 7 base columns plus one (amplitude, dwell) pair per additional substate (v4).
#     Column semantics are INFERRED and UNCONFIRMED — confirm with the user before trusting any
#     mapping. Exact duplicate rows exist, so the importer needs a dedup flag.
#     v1 REFUSES these files with a specific message (see io/text.py)."""
#     meta: RecordingMeta
#     events: "pandas.DataFrame"
#     source_doc_id: str | None


@dataclass(frozen=True)
class CalibratedChannel:
    """What Session.calibrated() returns and what an AnalysisContext hands a v2 module.
    Carries the metadata needed to INTERPRET the data. Does NOT carry the time array —
    that is large, shared, and layer-offset-dependent; get it from Session.x_axis()."""

    info: ChannelInfo
    data: np.ndarray          # float32, contiguous, len == meta.n_samples
    units: str                # calibration.target_units if set, else info.raw_units; "" == unknown
    sample_rate_hz: float | None
    t0_s: float
    issues: tuple[str, ...] = ()
    """What is wrong with THIS ARRAY — the calibrated one, which is what every analysis reads.

    ``RecordingMeta.issues`` describes the recording AS LOADED, and the non-finite gate on
    :meth:`TraceDocument._report_nonfinite` fills it from the raw channels. Nothing re-checked
    the array after ``ChannelCalibration.apply``, so a calibration that RUINED the channel
    reported nothing at all: measured on the owner's 2,993,798-sample ABF, a gain of 1.5e36
    overflows float32 and gives 2,420 non-finite calibrated samples with ``meta.issues=()``,
    while the PSD panel's own text sends the user to the File info panel's Issues box — which
    says "(none)". A box that is empty because nobody looked is worse than no box.

    Empty means the calibration added nothing to what the load already reported; it never
    means "not checked". Populated by :meth:`nanoscope.core.session.Session.calibrated`,
    which is the one door every consumer of a calibrated array passes through.
    """


@dataclass(frozen=True)
class XAxis:
    """The x values for one layer. `is_time` False means the sample rate is genuinely unknown
    and the axis is SAMPLES — the app says so rather than fabricating seconds."""

    values: np.ndarray        # float64 ALWAYS. See Rule 2.
    unit: str                 # "s" or "sample"
    is_time: bool
