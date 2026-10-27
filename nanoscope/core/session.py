from __future__ import annotations

import warnings
from dataclasses import replace
from numbers import Integral

import numpy as np
from PySide6.QtCore import QObject, Signal

from .analysis import AnalysisResult, ResultStore
from .acquisition import AcquisitionBandwidth
from .cache import LruArrayCache
from .sample_rate import SampleRateDeclaration, TimeBaseAsRead
from .calibration import (
    SIDECAR_NONE,
    SIDECAR_RECORDING_MOVED,
    CalibrationSet,
    ChannelCalibration,
    load_sidecar,
    notes_lost_by,
    save_sidecar,
    sidecar_effect,
)
from .document import CalibratedChannel, Document, TraceDocument, XAxis
from .identity import same_source
from .layer import LAYER_INKS, ColourRegistry, Layer, ViewMode, colour_key
from .source_style import SourceLabelRegistry, validate_source_appearance
from .segment import (
    follow_parent_clock,
    SegmentDocument,
    build_segment_document,
    derive_segment_uid,
    persistent_key,
)
from .types import CACHE_BUDGET_MB, ChannelKind, Provenance, nonfinite_reason, scan_nonfinite

__all__ = ["UnknownSampleRateError", "Session"]


def _same_clock(a: Document, b: Document) -> bool:
    """Two documents carry one clock: rate, provenance, unit scale, refused rate and — for
    segments — where the cut sat in the experiment."""
    ma, mb = a.meta, b.meta
    if (ma.sample_rate_hz, ma.rate_provenance, ma.time_scale_s, ma.refused_rate_hz) != (
            mb.sample_rate_hz, mb.rate_provenance, mb.time_scale_s, mb.refused_rate_hz):
        return False
    oa, ob = getattr(a, "origin", None), getattr(b, "origin", None)
    return (oa is None and ob is None) or (oa is not None and ob is not None
                                           and oa.t0_abs_s == ob.t0_abs_s)


class UnknownSampleRateError(RuntimeError):
    """Raised instead of fabricating a sample rate for files with no timing metadata."""


class Session(QObject):
    """The ONLY mutable store. Panels hold exactly one reference — this — and talk to each other
    ONLY by calling mutators and subscribing to signals. Idiomatic Qt: no event bus, no observer
    framework. This is the entire answer to "how do panels observe state without a tangle".

    THREADING: mutated on the Qt main thread ONLY. Workers return objects; main thread installs
    them. No locks anywhere.
    """

    documentOpened = Signal(str)              # doc_id
    documentClosed = Signal(str)              # doc_id
    layersChanged = Signal()                  # add/remove/visible/color/offset/reorder
    calibrationChanged = Signal(str, int)     # doc_id, channel_index
    viewModeChanged = Signal()
    cursorMoved = Signal(float)               # x in the current axis unit
    resultsChanged = Signal(str, str)         # doc_id, module_id  — v2; never emitted in v1
    sourceAppearanceChanged = Signal(str)     # persistent PlotSource key; presentation only
    acquisitionBandwidthChanged = Signal(str, int)  # metadata only, never recalibration
    sampleRateChanged = Signal(str)           # doc_id — its time base was re-declared; every
                                              # seconds-based number (axis, dwell, PSD) moves

    def __init__(self, cache_budget_mb: int = CACHE_BUDGET_MB) -> None:
        super().__init__()
        self._documents: dict[str, Document] = {}
        self._layers: list[Layer] = []
        self._calibration = CalibrationSet()
        self._acquisition_overrides: dict[tuple[str, int], AcquisitionBandwidth] = {}
        # The user's sample-rate / time-unit declarations (core/sample_rate.py), keyed by LIVE
        # doc_id, beside the time base each document was ADDED with so "use the file's own
        # time base" can always be restored exactly. Unlike the acquisition override above,
        # a declaration is applied by REBUILDING the document's meta (see
        # set_sample_rate_declaration): ~80 consumers read meta.sample_rate_hz directly.
        self._rate_declarations: dict[str, SampleRateDeclaration] = {}
        self._time_base_as_read: dict[str, TimeBaseAsRead] = {}
        self._cache = LruArrayCache(cache_budget_mb)
        self._results = ResultStore()
        # What the CALIBRATION did to a channel, keyed exactly like the cached calibrated
        # array — (doc_id, ch, gain, offset) — so the scan runs once per distinct calibration
        # and an A->B->A edit reuses the answer instead of re-scanning 3 M samples.
        self._calibration_issues: dict[tuple, tuple[str, ...]] = {}
        # Was the recording's file AT ``meta.path`` when it was opened? Read once, at
        # add_document, so :meth:`set_calibration` can tell "the user moved the file behind
        # the app's back" (it was there, now it is not) from "this document never had a
        # file" (a synthetic/in-memory document, whose sidecar path is a perfectly good place
        # to write). Only the first is a reason not to write beside it.
        self._path_seen_on_open: dict[str, bool] = {}
        self._view_mode = ViewMode.OVERLAY
        self._cursor_x: float = 0.0
        # The project-lifetime colour identity map shared by the trace view and every
        # source-comparison surface.  It is public for the same reason ``results`` is:
        # project restore replaces the whole value before it re-creates layers, while
        # ordinary callers ask it for stable inks throughout the session.
        #
        # The keys are PERSISTENT recording keys, never live ``doc_id`` values; see
        # :func:`colour_key` and :meth:`add_layer`.  Initialising it here makes the
        # identity rule the normal Session behaviour rather than an optional GUI seam.
        self.colours: ColourRegistry = ColourRegistry()
        self.source_labels = SourceLabelRegistry()

    def set_source_appearance(self, key: str, *, label: str = "", color: str | None = None,
                              reset_color: bool = False) -> None:
        """Set project source defaults without touching analysis inputs/results.

        A blank label clears the alias. ``color=None`` preserves an existing user
        ink unless ``reset_color=True`` explicitly requests the palette. Detector-specific
        keys remain detector-specific; only an
        unsuffixed recording/channel key shares its explicit ink with the trace.
        """
        entry = {"label": label, **({"color": color} if color is not None else {})}
        clean = validate_source_appearance({key: entry}).get(key, {})
        previous = (self.source_labels.label(key), self.colours.inks.get(key))
        self.source_labels.set_label(key, clean.get("label", ""))
        if "color" in clean:
            self.colours.set_ink(key, clean["color"])
        elif reset_color:
            self.colours.clear_ink(key)
        current = (self.source_labels.label(key), self.colours.inks.get(key))
        if current == previous:
            return
        changed_layer = False
        for layer in self._layers:
            doc = self._documents.get(layer.doc_id)
            if doc is not None and colour_key(persistent_key(doc), layer.channel) == key:
                ink = self.colours.ink(key, LAYER_INKS)
                if layer.color != ink:
                    layer.color = ink
                    changed_layer = True
        if changed_layer:
            self.layersChanged.emit()
        self.sourceAppearanceChanged.emit(key)

    # -- documents -------------------------------------------------------------------------------
    def add_document(self, doc: Document) -> str:
        """Seeds CalibrationSet.default_for() for every channel, then applies any .npcal.json
        sidecar over the top. Emits documentOpened. Does NOT create layers — main_window does,
        so the default-visible policy lives in one place."""
        doc_id = doc.doc_id
        if doc_id in self._documents:
            raise ValueError(f"document already open: {doc_id}")

        for info in doc.meta.channels:
            # The voltage default is DETECTED from the channel's own magnitude, not assumed
            # (see CalibrationSet.default_for): blanket 0.1 was right for the ferritin ABFs
            # and 10x wrong for the host-guest ones. A strided median is enough and costs
            # nothing next to the read that just happened.
            med = None
            if info.kind is ChannelKind.VOLTAGE:
                try:
                    arr = np.asarray(doc.raw(info.index))
                    if arr.size:
                        step = max(1, arr.size // 20_000)
                        med = float(np.median(np.asarray(arr[::step], dtype=np.float64)))
                except Exception:      # noqa: BLE001 - a default must never break a load
                    med = None
            self._calibration.set(doc_id, info.index,
                                  CalibrationSet.default_for(info, med))

        # A segment is an in-memory derived recording, not a file: it has no sidecar of its own,
        # and its meta.path points at the PARENT file (whose sidecar belongs to the parent, not
        # the segment). Skip sidecar loading for segments; Session.create_segment instead inherits
        # the parent's LIVE calibration directly.
        if not isinstance(doc, SegmentDocument):
            # The channel list is passed so the sidecar loader can reject an entry for a
            # channel this recording does not have (C8) instead of installing it invisibly.
            issues = load_sidecar(doc.meta.path, doc_id, self._calibration, doc.meta.channels)
            if issues:
                # RecordingMeta and TraceDocument are frozen VALUE objects: rebuild rather than
                # object.__setattr__ past the freeze. dataclasses.replace reuses the signal arrays
                # by reference, so this is free — no 180 MB copy.
                doc = replace(doc, meta=replace(doc.meta, issues=doc.meta.issues + tuple(issues)))

        self._documents[doc_id] = doc
        self._time_base_as_read[doc_id] = TimeBaseAsRead(
            doc.meta.sample_rate_hz, doc.meta.rate_provenance, float(doc.meta.t0_s),
            float(doc.meta.time_scale_s), doc.meta.refused_rate_hz)
        try:
            self._path_seen_on_open[doc_id] = bool(doc.meta.path.exists())
        except (OSError, AttributeError, TypeError):   # noqa: BLE001 - a stat must never break a load
            self._path_seen_on_open[doc_id] = False
        self.documentOpened.emit(doc_id)
        # A segment whose parent was closed kept the clock it was cut with; the parent
        # landing again (a file with its own clock, or a segment re-cut at the same window
        # and so the same uid) is what it follows now — as it would after a project reopen.
        # Free for a document with no orphans: nothing is found, nothing fires.
        self._follow_parent_clock(doc_id)
        return doc_id

    def close_document(self, doc_id: str) -> None:
        """Drops layers for the doc, invalidates its cache entries and calibration.
        Emits documentClosed, then layersChanged if any layer went away."""
        if doc_id not in self._documents:
            return
        del self._documents[doc_id]

        before = len(self._layers)
        self._layers = [lay for lay in self._layers if lay.doc_id != doc_id]
        dropped = len(self._layers) != before

        self._cache.invalidate_doc(doc_id)
        self._calibration.drop_doc(doc_id)
        for key in [k for k in self._acquisition_overrides if k[0] == doc_id]:
            del self._acquisition_overrides[key]
        self._rate_declarations.pop(doc_id, None)
        self._time_base_as_read.pop(doc_id, None)
        for key in [k for k in self._calibration_issues if k[0] == doc_id]:
            del self._calibration_issues[key]
        self._path_seen_on_open.pop(doc_id, None)
        self._results.drop_doc(doc_id)

        self.documentClosed.emit(doc_id)
        if dropped:
            self.layersChanged.emit()

    def document(self, doc_id: str) -> Document:
        return self._documents[doc_id]

    def documents(self) -> tuple[Document, ...]:
        return tuple(self._documents.values())

    def has_document(self, doc_id: str) -> bool:
        """Is THIS LOAD open?

        Since a ``doc_id`` carries a per-load token (see :mod:`nanoscope.core.identity`)
        this now means what its ~50 call sites always assumed it meant: a result, a
        selection or an overlay that names a document names the load it was computed from,
        and a reopened file answers False here for the previous load's id. Ask
        :meth:`doc_id_for_source` instead when the question is "is this FILE open?".
        """
        return doc_id in self._documents

    def doc_id_for_source(self, source_key: str) -> str | None:
        """The live ``doc_id`` of the open document loaded from ``source_key``, or None.

        THE "is this file already open?" question, which is a different one from
        :meth:`has_document` and has to be asked differently now that a path is not an
        identity. The open pipeline uses it to keep its existing policy — a second File ▸
        Open of a recording that is already open focuses it rather than spending another
        180 MB — a policy that would otherwise silently become "load it twice".

        The comparison is :func:`~nanoscope.core.identity.same_source`, not ``==``. A source
        key is a resolved path, and ``Path.resolve()`` canonicalises ``sub/../`` and
        symlinks but NOT case — so on the owner's case-insensitive APFS volume two
        spellings of one path used to pass this guard and load the recording twice, leaving
        two documents with two independent calibrations behind one sidecar. Identity is a
        question for the filesystem; see that function for why folding case by hand would
        be the worse answer.
        """
        for doc_id, doc in self._documents.items():
            if same_source(doc.source_key, source_key):
                return doc_id
        return None

    def duplicates_of(self, doc_id: str) -> tuple[str, ...]:
        """Open docs sharing meta.stem_key. [VERIFIED] '..._-600 mV' exists as .abf (3 channels)
        and .csv (time,current only — the voltage channel is gone, so it can never have MEASURED
        voltage provenance). Inform via a non-blocking info bar. Never block, never auto-dedup."""
        key = self._documents[doc_id].meta.stem_key
        return tuple(other for other, doc in self._documents.items()
                     if other != doc_id and doc.meta.stem_key == key)

    # -- segments (derived recordings) -----------------------------------------------------------
    def create_segment(self, parent_doc_id: str, i0: int, i1: int, *, name: str,
                       segment_uid: str | None = None) -> str:
        """Derive a new in-memory SEGMENT document from an open parent recording and register it
        as a full document — its own ResultStore entries, its own selection, analysed with the
        exact same per-recording machinery as any file. Returns the new document's ``doc_id``.

        Emits ``documentOpened`` (so the workflow strip / dirty guard react for free, exactly as
        for a file open). The segment INHERITS the parent's per-channel calibration: calibration
        is Session state (never on the document), so it is copied here — straight onto the
        CalibrationSet, so no sidecar is written next to the parent file and no calibrationChanged
        storm fires for a brand-new document.

        ``segment_uid`` is normally derived deterministically from the parent + window; project
        RESTORE passes the saved uid so the segment re-derives onto the same identity and its
        persisted results re-attach.
        """
        parent = self._documents[parent_doc_id]
        if not isinstance(parent, TraceDocument):
            raise TypeError(f"{parent.display_name} carries no signal and cannot be segmented.")

        parent_key = persistent_key(parent)
        if segment_uid is None:
            existing = {d.origin.segment_uid for d in self._documents.values()
                        if isinstance(d, SegmentDocument)}
            segment_uid = derive_segment_uid(parent_key, i0, i1, existing)

        seg = build_segment_document(parent, i0, i1, name=name, segment_uid=segment_uid,
                                     parent_key=parent_key)
        seg_id = self.add_document(seg)   # seeds default calibration + emits documentOpened

        # Inherit the parent's LIVE per-channel calibration over the seeded defaults. Set straight
        # on the CalibrationSet (not via set_calibration) so no sidecar is written and no signal
        # fires: a freshly derived segment is not a user calibration edit.
        for info in seg.meta.channels:
            self._calibration.set(seg_id, info.index,
                                  self._calibration.get(parent_doc_id, info.index))
            override = self.acquisition_bandwidth_override(parent_doc_id, info.index)
            if override is not None:
                self._acquisition_overrides[(seg_id, info.index)] = override
        return seg_id

    def segment_documents(self) -> tuple[SegmentDocument, ...]:
        """Every open segment (derived) document, in insertion order. Project SAVE uses this to
        persist each segment as a spec plus its results, keyed by the persistent segment uid."""
        return tuple(d for d in self._documents.values() if isinstance(d, SegmentDocument))

    # -- calibration -----------------------------------------------------------------------------
    def calibration(self, doc_id: str, ch: int) -> ChannelCalibration:
        return self._calibration.get(doc_id, ch)

    def acquisition_bandwidth_override(self, doc_id: str, ch: int) -> AcquisitionBandwidth | None:
        self._acquisition_channel(doc_id, ch)
        return self._acquisition_overrides.get((doc_id, ch))

    def _acquisition_channel(self, doc_id: str, ch: int):
        channels = self._documents[doc_id].meta.channels
        if isinstance(ch, bool) or not isinstance(ch, Integral) or not 0 <= ch < len(channels):
            raise ValueError("acquisition metadata requires a valid channel index")
        return channels[ch]

    def acquisition_bandwidth(self, doc_id: str, ch: int) -> AcquisitionBandwidth:
        header = self._acquisition_channel(doc_id, ch).acquisition_bandwidth
        return self._acquisition_overrides.get((doc_id, ch), header)

    def set_acquisition_bandwidth(self, doc_id: str, ch: int,
                                  value: AcquisitionBandwidth | None) -> None:
        """Set acquisition provenance only; None restores the immutable file declaration.

        No samples, cache entries, detections, fits or calibration are changed. Existing
        results retain the metadata snapshot they were created with. Saved in projects,
        never written beside raw recordings.
        """
        self._acquisition_channel(doc_id, ch)
        if value is not None and (not isinstance(value, AcquisitionBandwidth) or value.source != "user"):
            raise ValueError("an acquisition override must be user-provided metadata")
        key = (doc_id, ch)
        if self._acquisition_overrides.get(key) == value:
            return
        if value is None:
            self._acquisition_overrides.pop(key, None)
        else:
            self._acquisition_overrides[key] = value
        self.acquisitionBandwidthChanged.emit(doc_id, ch)

    # -- the time base: what the file said, and what the user declared --------------------------
    def time_base_as_read(self, doc_id: str) -> TimeBaseAsRead:
        """The document's clock as the reader reported it (as the cut made it, for a segment):
        what a declaration is resolved against and what ``None`` restores."""
        if doc_id not in self._documents:
            raise KeyError(doc_id)
        return self._time_base_as_read[doc_id]

    def sample_rate_declaration(self, doc_id: str) -> SampleRateDeclaration | None:
        if doc_id not in self._documents:
            raise KeyError(doc_id)
        return self._rate_declarations.get(doc_id)

    def set_sample_rate_declaration(self, doc_id: str,
                                    value: SampleRateDeclaration | None) -> None:
        """Declare this recording's time base — its column's unit, or its rate — or, with
        ``None``, return to the file's own. Emits ``sampleRateChanged`` once per change.

        THE DOCUMENT IS REBUILT. A rate is not a side-table value like an acquisition
        override: ``Session.calibrated`` snapshots ``meta.sample_rate_hz`` onto every
        channel a worker is handed, ``x_axis`` builds the axis from it, segments inherit it,
        and some eighty callers read it off the document directly. So the frozen meta is
        replaced (``dataclasses.replace`` — the one sanctioned way, and the same move
        ``add_document`` makes to fold sidecar issues in), which keeps ``load_uid`` and
        therefore the ``doc_id``, and the cached x-axis and calibrated arrays for the
        document are dropped so nothing serves the old clock. Every seconds-based number on
        screen moves; stored results do NOT — they keep the rate they were measured at and
        are judged stale against the live one (``analysis_panel._judge_staleness``), never
        silently re-scaled. Saved in projects, never written beside raw recordings.

        SEGMENTS FOLLOW. A segment is a window of its parent, so a declaration is made on
        the parent recording (a segment refuses one) and every segment cut from it — live
        ones here, re-derived ones on a project restore — is re-made on the parent's new
        clock (:func:`~nanoscope.core.segment.follow_parent_clock`), each announced by
        its own ``sampleRateChanged``.
        """
        doc = self._documents[doc_id]
        if isinstance(doc, SegmentDocument):
            raise ValueError("a segment follows the time base of the recording it was cut "
                             "from; declare it on that recording")
        if value is not None and (not isinstance(value, SampleRateDeclaration)
                                  or value.source != "user"):
            raise ValueError("a sample-rate declaration must be user-provided metadata")
        if self._rate_declarations.get(doc_id) == value:
            return
        read = self._time_base_as_read[doc_id]
        if value is None:
            rate, prov, t0, scale = (read.sample_rate_hz, read.rate_provenance, read.t0_s,
                                     read.time_scale_s)
        else:
            rate, t0, scale = value.resolve(read)   # ValueError if it cannot be
            prov = Provenance.USER
        self._documents[doc_id] = replace(doc, meta=replace(
            doc.meta, sample_rate_hz=rate, rate_provenance=prov, t0_s=t0, time_scale_s=scale))
        if value is None:
            self._rate_declarations.pop(doc_id, None)
        else:
            self._rate_declarations[doc_id] = value
        self._cache.invalidate_doc(doc_id)
        self.sampleRateChanged.emit(doc_id)
        self._follow_parent_clock(doc_id)

    def _follow_parent_clock(self, parent_id: str) -> None:
        """Re-make every segment cut from ``parent_id`` on the parent's current clock, and
        theirs in turn (a segment of a segment follows the same rule)."""
        parent = self._documents[parent_id]
        parent_key = persistent_key(parent)
        for seg_id, seg in list(self._documents.items()):
            # same_source, not ==: a reopened parent may spell its path differently
            # (/Case.csv vs /case.csv on the owner's case-insensitive disk); for a segment
            # uid it is plain string equality.
            if (not isinstance(seg, SegmentDocument)
                    or not same_source(seg.origin.parent_key, parent_key)):
                continue
            followed = follow_parent_clock(seg, parent)
            if _same_clock(followed, seg):
                continue                    # already on this clock: nothing moves, nothing fires
            self._documents[seg_id] = followed
            self._time_base_as_read[seg_id] = TimeBaseAsRead(
                followed.meta.sample_rate_hz, followed.meta.rate_provenance,
                float(followed.meta.t0_s), float(followed.meta.time_scale_s),
                followed.meta.refused_rate_hz)
            self._cache.invalidate_doc(seg_id)
            self.sampleRateChanged.emit(seg_id)
            self._follow_parent_clock(seg_id)

    # -- what a channel IS: the one resolver ----------------------------------------------------
    def effective_kind(self, doc_id: str, ch: int) -> ChannelKind:
        """*** WHAT THIS CHANNEL IS TREATED AS, everywhere. *** The user's Calibration-panel
        override when they set one (``ChannelCalibration.kind``, persisted as the sidecar's
        ``treat_as_kind``), else the header's own declaration.

        THIS LIVES ON THE SESSION, NOT IN THE TRACE VIEW, because the trace view was where it
        used to live and that is exactly how the app came to contradict itself. Measured on a
        2 s CSV of the owner's own -700 mV recording with neutral column headers (both
        channels ``unknown``), after "Treat this channel as: current" + pA + gain 2.5 was
        applied through the real panel: the panes drew ch0 on top with a pA axis, Batch
        detect ran on it and stored 15 events -- both read ``trace_view.effective_kind`` --
        while the status-bar chip read "No current channel", the workflow strip locked
        CALIBRATE / FILTER / DETECT with "No current channel on this recording", the
        Calibration panel's own title still said ``unknown`` two rows above its combo
        reading ``current`` and hid its conductance box, and the Detect button refused with
        "this is a unknown channel" -- every one of those read ``meta.channels[ch].kind``,
        the header. The reverse override was worse: IN15 (header current) marked VOLTAGE
        greyed out Batch detect and the PSD and left Run detection enabled on a channel the
        user had just called a command voltage.

        A resolver that any surface can reach without importing a GUI module is what stops
        the next surface reading the header again. ``gui.trace_view.effective_kind`` is now
        a thin delegate to this.
        """
        cal = self._calibration.get(doc_id, ch)
        if cal.kind is not None:
            return cal.kind
        return self._documents[doc_id].meta.channels[ch].kind

    def is_live_current(self, doc_id: str, ch: int) -> bool:
        """Effective kind CURRENT and not flagged dead at load -- the channel the pipeline
        (calibrate / filter / detect) acts on."""
        info = self._documents[doc_id].meta.channels[ch]
        return self.effective_kind(doc_id, ch) is ChannelKind.CURRENT and not info.is_dead

    def resolve_current_channel(self, doc_id: str, channel: int | None) -> int | None:
        """Map a selection to the current channel the pipeline acts on -- THE policy the
        analysis panel, the workflow strip and the calibration chip share: keep the selection
        if it is already a live current channel, else the recording's first live current
        channel (so clicking the voltage channel to calibrate it does not lose the analyte
        channel), else ``None`` when the recording has none.

        "Current" here is :meth:`effective_kind`, so a channel the user corrected to
        current in the Calibration panel resolves like one the file declared, and a
        channel the user re-declared as a voltage stops resolving -- on every surface that
        asks here, at once.
        """
        channels = self._documents[doc_id].meta.channels
        if channel is not None and 0 <= channel < len(channels) \
                and self.is_live_current(doc_id, channel):
            return channel
        for info in channels:
            if self.is_live_current(doc_id, info.index):
                return info.index
        return None

    def recording_moved(self, doc_id: str) -> bool:
        """Was the recording's file at ``meta.path`` when it was opened, and is it gone now?

        True means the user moved (or renamed, or deleted) the file behind the app's back
        while the samples stayed in memory. A document that never had a file -- a
        synthetic or in-memory one -- answers False: nothing moved. A segment answers
        False too: its ``meta.path`` is its PARENT's, and it never writes beside it.
        """
        if not self._path_seen_on_open.get(doc_id, False):
            return False
        doc = self._documents.get(doc_id)
        if doc is None or isinstance(doc, SegmentDocument):
            return False
        try:
            return not doc.meta.path.exists()
        except OSError:
            return True

    def sidecar_effect_of(self, doc_id: str, ch: int, cal: ChannelCalibration) -> str:
        """What committing ``cal`` WOULD do to ``<datafile>.npcal.json``, without committing it.

        One of ``SIDECAR_WRITTEN`` / ``SIDECAR_REMOVED`` / ``SIDECAR_NONE``. This is how a
        caller asks BEFORE it destroys something: a commit that leaves every channel at its
        default deletes the user's sidecar, note and all, and a confirmation has to happen
        before the unlink, not after it.

        Evaluated against a COPY of the calibration set, so asking costs nothing and changes
        nothing. A segment never has a sidecar of its own, so it always answers
        ``SIDECAR_NONE`` — exactly matching what :meth:`set_calibration` will skip.
        """
        doc = self._documents[doc_id]
        if isinstance(doc, SegmentDocument):
            return SIDECAR_NONE
        if self.recording_moved(doc_id):
            return SIDECAR_RECORDING_MOVED      # nothing beside the old path is touched
        probe = CalibrationSet(entries=dict(self._calibration.entries))
        probe.set(doc_id, ch, cal)
        return sidecar_effect(doc.meta.path, doc_id, probe, doc.meta.channels)

    def sidecar_notes_lost_by(self, doc_id: str, ch: int,
                              cal: ChannelCalibration) -> tuple[tuple[int, str], ...]:
        """``((channel, note), …)`` this commit would erase from the user's sidecar.

        The companion to :meth:`sidecar_effect_of`, and the reason there are two: the effect
        is about the FILE and fires only when the whole file goes, while the thing that
        cannot be recreated is one channel's ``note``. Resetting one channel of a recording
        whose OTHER channel is still calibrated leaves the file in place and drops that
        channel's note out of it — ``SIDECAR_WRITTEN``, no confirmation, and a status line
        that reads like an ordinary save.

        Same probe-a-copy discipline: asking changes nothing. A segment has no sidecar, so
        it can lose nothing.
        """
        doc = self._documents[doc_id]
        if isinstance(doc, SegmentDocument):
            return ()
        if self.recording_moved(doc_id):
            return ()                            # the orphan sidecar is not rewritten
        probe = CalibrationSet(entries=dict(self._calibration.entries))
        probe.set(doc_id, ch, cal)
        return notes_lost_by(doc.meta.path, doc_id, probe, doc.meta.channels)

    def set_calibration(self, doc_id: str, ch: int, cal: ChannelCalibration, *,
                        write_sidecar: bool = True,
                        reason: list[str] | None = None) -> str:
        """Invalidates the affected cache entry, writes the sidecar if non-default, emits
        calibrationChanged(doc_id, ch). Consumers RE-SET DATA on affected curves — they must NOT
        rebuild the view, which would throw away the user's zoom.

        ``write_sidecar=False`` installs the calibration WITHOUT touching
        ``<datafile>.npcal.json`` next to the user's raw data. That is not an optimisation, it
        is a data-safety contract: the sidecar is a USER-OWNED file outside the app, holding a
        ``note`` field this app cannot round-trip through a project (ProjectState carries no
        note). Writing it from a non-user-edit path — project RESTORE being the one that
        mattered — rewrote it with a blank note, or ``unlink``ed it outright when the restored
        values happened to equal the defaults. Only a genuine calibration EDIT (the Calibration
        panel) may write a sidecar; everything reconstructive passes False.

        Returns what happened to the sidecar (``SIDECAR_*``), so no caller can delete the
        user's file without being told. ``SIDECAR_NONE`` when ``write_sidecar`` is False.
        ``reason`` is passed straight through to :func:`save_sidecar`: it collects the OS's
        own words for a ``SIDECAR_FAILED``, so the panel can say WHY instead of guessing.

        ``SIDECAR_RECORDING_MOVED`` when the recording's file was at ``meta.path`` on open
        and is not there now (:meth:`recording_moved`): the calibration is installed for
        this session exactly as before, but NOTHING is written beside the old path. Measured
        on a copy of the owner's 15 s -1000 mV ABF: Apply gain 0.75 wrote the sidecar, the
        .abf was then moved to another folder, Apply gain 0.9 reported "Applied · saved to
        pore.abf.npcal.json" and rewrote the file in the folder the recording had LEFT --
        beside nothing -- and reopening the recording where it now lived gave gain 1.0 with
        nothing said. A save that lands where the data is not is not a save; saying so is
        the honest outcome, and it leaves the user free to move the sidecar after the file.
        """
        old = self._calibration.get(doc_id, ch)
        self._cache.invalidate(("cal", doc_id, ch) + old.fingerprint())
        self._calibration.set(doc_id, ch, cal)

        doc = self._documents[doc_id]
        # Never write a sidecar for a segment: its meta.path is the PARENT file, and a segment's
        # calibration lives only in the project, not next to the parent on disk.
        outcome = SIDECAR_NONE
        if write_sidecar and not isinstance(doc, SegmentDocument):
            meta = doc.meta
            if self.recording_moved(doc_id):
                outcome = SIDECAR_RECORDING_MOVED
            else:
                outcome = save_sidecar(meta.path, doc_id, self._calibration, meta.channels,
                                       reason)
        self.calibrationChanged.emit(doc_id, ch)
        return outcome

    def calibrated(self, doc_id: str, ch: int) -> CalibratedChannel:
        """On demand + bounded cache; never eager for all channels (60 MB each).
        IMPORTANT: when the calibration is identity, return doc.raw(ch) DIRECTLY and do NOT cache
        it — caching a reference to the raw array double-counts it against the budget and
        evicting it frees nothing.

        THE RETURNED CHANNEL CARRIES ITS OWN ``issues``. ``meta.issues`` describes the
        recording as loaded, and nothing re-checked the array after ``apply()`` — so a
        calibration that ruined the channel reported nothing anywhere. This is the one door
        every consumer of a calibrated array passes through, which is why the re-check lives
        here rather than in the panel that happened to notice first."""
        doc = self._documents[doc_id]
        if not isinstance(doc, TraceDocument):
            raise TypeError(f"{doc.display_name} is not a trace document; it carries no signal.")

        info = doc.meta.channels[ch]
        cal = self._calibration.get(doc_id, ch)
        units = cal.target_units or info.raw_units

        # Snapshot effective metadata for workers; meta.channels still holds the header.
        info = replace(info, acquisition_bandwidth=self.acquisition_bandwidth(doc_id, ch))
        return CalibratedChannel(info=info, data=self.calibrated_array(doc_id, ch), units=units,
                                 sample_rate_hz=doc.meta.sample_rate_hz, t0_s=doc.meta.t0_s,
                                 issues=self.calibration_issues(doc_id, ch))

    def calibration_issues(self, doc_id: str, ch: int) -> tuple[str, ...]:
        """What the CALIBRATION did to this channel that the load did not already report.

        Only the damage ``apply()`` itself caused: a channel that arrived with NaN in it is
        already named in ``meta.issues`` and is not repeated here, so the two lists add up
        instead of double-counting. Empty at an identity calibration without touching the
        array at all.

        MEMOISED per ``(doc_id, channel, gain, offset)`` — the same key and the same lifetime
        as the cached calibrated array itself, so nothing here runs more than once per
        distinct calibration. ``calibrated()`` is called once per repaint by the trace view;
        a per-call scan of 3 M samples would be a per-frame cost.

        IT DOES NOT SCAN THE CALIBRATED ARRAY IN THE NORMAL CASE, and that is deliberate
        rather than an optimisation. ``apply()`` is affine and monotone, so the largest
        output magnitude is produced by the smallest or largest RAW sample; the only way a
        finite sample becomes non-finite is float32 overflow, which those two numbers decide.
        Scanning the calibrated array unconditionally would force it to be MATERIALISED —
        200 MB for one channel of the owner's 50 M-sample recording — merely to fill an
        information box. The predicate is deliberately CONSERVATIVE: when it says overflow is
        possible we do the exact scan, so a false "possible" costs time and never accuracy.
        """
        doc = self._documents[doc_id]
        if not isinstance(doc, TraceDocument):
            return ()
        cal = self._calibration.get(doc_id, ch)
        if cal.fingerprint() == (1.0, 0.0):
            return ()
        key = (doc_id, int(ch)) + cal.fingerprint()
        cached = self._calibration_issues.get(key)
        if cached is not None:
            return cached

        issues = self._scan_calibration_damage(doc, ch, cal)
        self._calibration_issues[key] = issues
        return issues

    def raw_extremes(self, doc_id: str, ch: int) -> tuple[float, float]:
        """``(min, max)`` of a raw channel, EXACT over every sample, ignoring NaN; cached for
        the document's lifetime.

        Public because two surfaces need the same two numbers: :meth:`calibration_issues`
        decides float32 overflow from them, and the Calibration panel prints raw/calibrated
        min, max and span from them. The panel used to take its extremes from a <=20,000
        sample stride instead -- and on the owner's -500 mV recording (12,502,631 samples,
        every 625th read) that printed a maximum of -7.63 pA for a channel whose true
        maximum is +5.49 pA, i.e. it said the current never went positive when it does.
        A tail order statistic is biased inward by construction; the exact scan of that
        12.5 M-sample float32 channel costs a few milliseconds once, and the calibrated
        extremes follow from these two by monotone affine arithmetic.

        ``(nan, nan)`` for a channel with no finite sample; non-finite when the channel
        holds an infinity, which the caller must read as "I cannot decide from these two
        numbers" rather than as a bound. ``(0.0, 0.0)`` for an empty channel.
        """
        return self._raw_extremes(doc_id, ch)

    def _raw_extremes(self, doc_id: str, ch: int) -> tuple[float, float]:
        key = ("rawext", doc_id, int(ch))
        cached = self._calibration_issues.get(key)
        if cached is not None:
            return cached                                  # type: ignore[return-value]
        raw = self._documents[doc_id].raw(ch)
        if raw.size == 0:
            extremes = (0.0, 0.0)
        else:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                # An all-NaN channel makes nanmin warn and return NaN; NaN is the honest
                # answer and the warning is noise on a path the trace view repaints from.
                warnings.simplefilter("ignore", RuntimeWarning)
                extremes = (float(np.nanmin(raw)), float(np.nanmax(raw)))
        self._calibration_issues[key] = extremes           # type: ignore[assignment]
        return extremes

    def _scan_calibration_damage(self, doc: TraceDocument, ch: int,
                                 cal: ChannelCalibration) -> tuple[str, ...]:
        lo, hi = self._raw_extremes(doc.doc_id, ch)
        f32_max = float(np.finfo(np.float32).max)
        reach = max(abs(lo), abs(hi)) * abs(cal.gain) + abs(cal.offset)
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(reach) and reach <= f32_max:
            return ()                       # no finite sample can leave float32; nothing to say

        raw_bad, _first_raw = scan_nonfinite(doc.raw(ch))
        data = self.calibrated_array(doc.doc_id, ch)
        count, first = scan_nonfinite(data)
        if count <= raw_bad:
            return ()                       # the load already reported everything there is
        info = doc.meta.channels[ch]
        return (
            f"channel {ch} ({info.name or 'unnamed'}) AFTER calibration "
            f"(gain {cal.gain:g}, offset {cal.offset:g}): "
            f"{nonfinite_reason(count, int(data.size), first)} The recording as loaded "
            f"carried {raw_bad:,} of them; this calibration made the rest, by overflowing "
            f"float32. Any statistic computed over them — RMS noise, baseline, event depth "
            f"— is undefined. Lower the gain, or work in the raw units.",
        )

    def calibrated_array(self, doc_id: str, ch: int) -> np.ndarray:
        """The calibrated samples alone, without building a :class:`CalibratedChannel`.

        Exists so :meth:`calibration_issues` can read the array without recursing through
        the constructor that asks it for the issues.
        """
        doc = self._documents[doc_id]
        cal = self._calibration.get(doc_id, ch)
        if cal.fingerprint() == (1.0, 0.0):
            return doc.raw(ch)
        return self._cache.get_or_compute(
            ("cal", doc_id, ch) + cal.fingerprint(),
            lambda: cal.apply(doc.raw(ch)),
        )

    # -- x axis ----------------------------------------------------------------------------------
    def x_axis(self, doc_id: str, time_offset_s: float = 0.0) -> XAxis:
        """float64 ALWAYS (Rule 2). Built once per (doc, shift) and shared BY REFERENCE across
        every curve of that document. 120 MB at the ceiling; goes through the SAME bounded cache
        as calibrated arrays, so it is evictable.

        values = np.arange(n, dtype=np.float64) / fs + (meta.t0_s + time_offset_s), unit "s".
        If meta.sample_rate_hz is None: values = np.arange(n, dtype=np.float64), unit "sample",
        is_time False, time_offset_s IGNORED. Never fabricate seconds.
        """
        meta = self._documents[doc_id].meta
        n = meta.n_samples
        fs = meta.sample_rate_hz

        if fs is None:
            values = self._cache.get_or_compute(
                ("time", doc_id, 0.0),
                lambda: np.arange(n, dtype=np.float64),
            )
            return XAxis(values=values, unit="sample", is_time=False)

        t_shift = meta.t0_s + time_offset_s
        values = self._cache.get_or_compute(
            ("time", doc_id, t_shift),
            lambda: np.arange(n, dtype=np.float64) / fs + t_shift,
        )
        return XAxis(values=values, unit="s", is_time=True)

    def time_to_index(self, doc_id: str, ch: int, x: float, time_offset_s: float = 0.0) -> int | None:
        """O(1) inverse map for the crosshair — arithmetic, not a search over the array.
        Returns None if out of range. Raises UnknownSampleRateError only if callers ask for
        seconds on a rate-unknown document.

        On a rate-unknown document the axis IS samples (see x_axis), so x is a sample index and a
        non-zero time_offset_s is by definition a seconds-flavoured request — that is the one
        thing this refuses rather than silently answering wrong.
        """
        meta = self._documents[doc_id].meta
        fs = meta.sample_rate_hz

        if fs is None:
            if time_offset_s != 0.0:
                raise UnknownSampleRateError(
                    f"{meta.path.name} declares no sample rate; its x-axis is in samples, so a "
                    f"time offset of {time_offset_s} s has no meaning.")
            idx = int(round(x))
        else:
            idx = int(round((x - (meta.t0_s + time_offset_s)) * fs))

        return idx if 0 <= idx < meta.n_samples else None

    # -- layers ----------------------------------------------------------------------------------
    def layers(self) -> tuple[Layer, ...]:
        return tuple(self._layers)

    def layer(self, layer_id: str) -> Layer | None:
        return next((lay for lay in self._layers if lay.layer_id == layer_id), None)

    def has_layer(self, doc_id: str, ch: int) -> bool:
        return self.layer(f"{doc_id}#{ch}") is not None

    def add_layer(self, doc_id: str, ch: int) -> Layer:
        """No-op returning the existing Layer if ``(doc_id, ch)`` is already present.

        The colour comes from the session's identity-keyed :class:`ColourRegistry`, using
        the recording's PERSISTENT key.  Asking the registry assigns a stable slot the first
        time this source is seen; removing/re-adding a layer or closing another recording
        therefore cannot repaint it.  The layer still stores the concrete dark-pane ink so
        old projects and registry-free consumers retain their established fallback.

        Labels the layer ``"<file> — <channel>"`` and emits :attr:`layersChanged`.
        """
        existing = self.layer(f"{doc_id}#{ch}")
        if existing is not None:
            return existing

        doc = self._documents[doc_id]
        info = doc.meta.channels[ch]
        key = colour_key(persistent_key(doc), ch)
        color = self.colours.ink(key, LAYER_INKS)

        layer = Layer(doc_id=doc_id, channel=ch, color=color,
                      label=f"{doc.display_name} — {info.name}")
        self._layers.append(layer)
        self.layersChanged.emit()
        return layer

    def remove_layer(self, layer_id: str) -> None:
        before = len(self._layers)
        self._layers = [lay for lay in self._layers if lay.layer_id != layer_id]
        if len(self._layers) != before:
            self.layersChanged.emit()

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        layer = self.layer(layer_id)
        if layer is None or layer.visible == visible:
            return
        layer.visible = visible
        self.layersChanged.emit()

    def set_layer_excluded(self, layer_id: str, excluded: bool) -> None:
        """Record the user's own include/exclude decision for this channel.

        THE NON-DESTRUCTIVE UNTICK. Excluding takes the channel off the trace and keeps
        everything about it — its colour, its ``time_offset_s`` alignment, its place in the
        layer list — so re-ticking restores what was there rather than a fresh layer with
        the next palette colour and dt=0. Including it puts it back on screen.

        The distinction from :meth:`set_layer_visible` is the whole point and is stated on
        :class:`~nanoscope.core.layer.Layer`: ``visible`` is "is this the recording on
        screen", ``excluded`` is "the user said no". :meth:`show_only` honours the second,
        which is what lets the untick stop being a deletion.
        """
        layer = self.layer(layer_id)
        if layer is None:
            return
        excluded = bool(excluded)
        want_visible = not excluded
        if layer.excluded == excluded and layer.visible == want_visible:
            return
        layer.excluded = excluded
        layer.visible = want_visible
        self.layersChanged.emit()

    def show_only(self, doc_id: str) -> bool:
        """Make ``doc_id`` THE SHOWN recording: every layer of it visible, every layer of every
        other document hidden. Returns True if anything actually changed.

        This is the mutator behind "opening (or selecting) a recording shows THAT recording".
        It moves VISIBILITY and never touches layer membership, and that distinction is the
        whole point: a recording you switch away from keeps its layer set, its colours and its
        time-offset alignment, so switching back restores exactly what you were looking at
        rather than a fresh layer with the next palette colour and dt=0. Layer membership stays
        what ``Layer`` says it is — the authoritative include/exclude decision for a channel —
        which merely looking at something else has no business rewriting.

        Emits ``layersChanged`` ONCE for the whole switch, not once per layer: every emission
        rebuilds the entire trace view, so a per-layer loop over ``set_layer_visible`` would
        rebuild it N times to reach one state.

        A layer the user EXCLUDED (unticked) stays off. Switching to a recording is not a
        statement about which of its channels you wanted, and re-showing an unticked one was
        the reason the untick had to delete the layer to stick — see
        :meth:`set_layer_excluded` and :class:`~nanoscope.core.layer.Layer`.
        """
        changed = False
        for lay in self._layers:
            want = lay.doc_id == doc_id and not lay.excluded
            if lay.visible != want:
                lay.visible = want
                changed = True
        if changed:
            self.layersChanged.emit()
        return changed

    def shown_doc_ids(self) -> frozenset[str]:
        """The documents currently DRAWN — i.e. carrying at least one visible layer. Distinct
        from :meth:`documents` (everything open) and from the set of documents with layers
        (everything ever displayed). Analyses read ``documents()``; the viewer reads this."""
        return frozenset(lay.doc_id for lay in self._layers if lay.visible)

    def set_layer_time_offset(self, layer_id: str, dt_s: float) -> None:
        layer = self.layer(layer_id)
        if layer is None or layer.time_offset_s == dt_s:
            return
        layer.time_offset_s = dt_s
        self.layersChanged.emit()

    def set_layer_color(
            self, layer_id: str, color: str, *, update_registry: bool = True) -> None:
        """Set a layer's concrete ink and, normally, its shared colour rule.

        ``Layer.color`` is the concrete ink saved for backwards compatibility and to
        preserve the exact on-screen appearance across palette changes.
        :attr:`colours` is the identity-keyed rule shared with plot sources.  A user-facing
        edit updates both so the trace swatch and Plots window keep the same chosen colour.

        ``update_registry=False`` is the project-restore seam.  ``layers[].color`` and the
        top-level colour registry are independently persisted facts: the former is the exact
        dark-pane ink that was saved, while a slot without a user ink is rendered through
        each surface's current palette.  A restore must install the concrete ink without
        turning it into a new user override in the registry.
        """
        layer = self.layer(layer_id)
        if layer is None:
            return
        color = str(color)
        if not update_registry:
            if layer.color == color:
                return
            layer.color = color
            self.layersChanged.emit()
            return
        doc = self._documents.get(layer.doc_id)
        if doc is None:  # defensive: close_document normally removes the layer first
            if layer.color == color:
                return
            layer.color = color
            self.layersChanged.emit()
            return

        key = colour_key(persistent_key(doc), layer.channel)
        if layer.color == color and self.colours.ink(key, LAYER_INKS) == color:
            return
        self.colours.set_ink(key, color)
        layer.color = color
        self.layersChanged.emit()

    # -- view ------------------------------------------------------------------------------------
    @property
    def view_mode(self) -> ViewMode:
        return self._view_mode

    def set_view_mode(self, mode: ViewMode) -> None:
        if mode is self._view_mode:
            return
        self._view_mode = mode
        self.viewModeChanged.emit()

    def set_cursor(self, x: float) -> None:
        self._cursor_x = x
        self.cursorMoved.emit(x)

    @property
    def cursor_x(self) -> float:
        return self._cursor_x

    # -- results (v2 sink) -----------------------------------------------------------------------
    @property
    def results(self) -> ResultStore:
        return self._results

    def add_result(self, res: AnalysisResult) -> None:
        """Store an AnalysisResult and announce it. Called on the MAIN THREAD only
        (the worker returns the object; the controller installs it here), so no
        lock — same discipline as every other mutator. Emits
        resultsChanged(doc_id, module_id) so panels refresh and redraw overlays."""
        self._results.put(res)
        self.resultsChanged.emit(res.doc_id, res.module_id)
