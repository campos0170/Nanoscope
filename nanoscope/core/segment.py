"""Signal segmentation: a *segment* is a DERIVED RECORDING.

Cropping a time-region of an open recording produces a brand-new in-memory
:class:`Document` that lands in the Files list and is analysed with the *exact same*
per-recording machinery (detection / merge / plots / PSD / measure) as any file — its
own results, its own selection, independent of the parent and of sibling segments.

This module is PURE (no Qt, no :class:`~nanoscope.core.session.Session`): it is the
value-object layer.

* :class:`SegmentOrigin` — the provenance stamped on a derived document (what it was
  cut from and where), so a segment is identifiable AND traceable to its parent.
* :class:`RebasedMeta` — the same origin carried on the ``meta`` itself, because
  ``core.clock`` (the one seam every clock label in the app reads) is handed a meta and
  would otherwise see a manufactured ``t0_s = 0`` and call it a file's declared zero.
* :class:`SegmentDocument` — a :class:`~nanoscope.core.document.TraceDocument`
  subclass, so ``isinstance(seg, TraceDocument)`` holds and every existing code path
  (``Session.calibrated``, detection, IV, calibration panel, …) runs on it unchanged.
  Its ``doc_id`` is a stable id derived from the segment uid — NOT the parent's path —
  so re-deriving the same segment on project load lands it on the same identity.
* :class:`SegmentSpec` — the persistable description (uid, parent key, window, name)
  that lets the project re-derive the segment from its parent on load.
* :func:`build_segment_document` — the pure cropper.
* :func:`derive_segment_uid` — a deterministic (never random) id, so tests reproduce.
* :func:`persistent_key` — the identity that survives a project round-trip (source path
  for a file recording, segment uid for a segment); results/calibrations/layers persist
  under THIS key, never the volatile in-memory ``doc_id``.
* :func:`spec_window_refusal` — the one rule that says when a SAVED window may no longer
  be re-derived. The builder clamps (a live cut is a request); a persisted spec is an
  assertion and must refuse rather than quietly become a different recording.

CALIBRATION is Session state, never document state (a :class:`TraceDocument` holds no
calibration by design). So a segment inherits its parent's per-channel calibration in
:meth:`~nanoscope.core.session.Session.create_segment`, not in the pure builder here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields as dc_fields, replace
from pathlib import Path

import numpy as np

from . import clock
from .document import Document, TraceDocument
from .types import RecordingMeta

__all__ = [
    "MIN_SEGMENT_SAMPLES",
    "RebasedMeta",
    "SegmentOrigin",
    "SegmentSpec",
    "SegmentDocument",
    "derive_segment_uid",
    "build_segment_document",
    "persistent_key",
    "spec_window_refusal",
]

# A 0- or 1-sample crop carries no analysable signal (no dwell, no baseline flank); reject it.
MIN_SEGMENT_SAMPLES: int = 2


def spec_window_refusal(parent_n_samples: int, i0: int, i1: int) -> str | None:
    """Why a SAVED window ``[i0, i1)`` can no longer be re-derived from a parent holding
    ``parent_n_samples`` samples — or ``None`` when it still fits exactly.

    THE DISTINCTION THIS FUNCTION EXISTS FOR. :func:`build_segment_document` CLAMPS its
    window into ``[0, n]``, deliberately and permanently: a live rubber-band cut is a
    REQUEST ("crop what I dragged"), the drag can legitimately run past the end of the
    trace, and ``tests/core/test_segment.py`` pins that. A :class:`SegmentSpec` restored
    from a project is not a request — it is an ASSERTION that a particular 20,000-sample
    stretch of a particular recording IS this segment, under whose identity a whole
    curated event table is filed. Re-deriving that assertion against a shorter parent
    substitutes a DIFFERENT recording under the same identity and says nothing:
    measured, a ``[40000, 60000)`` segment came back as ``[40000, 45000)`` — a quarter of
    the analysed window — with ``meta.issues == ()``, all 4 saved events reattached, and
    3 of them lying entirely past the end of the segment they were attached to. Their
    dwell/depth/ECD had been measured on data no longer on disk, and the next save
    rewrote ``SegmentRec.i1`` to 45,000, so restoring the original recording from backup
    no longer recovered the window that was actually analysed.

    So the rule lives HERE, next to the clamp it contradicts, rather than in the restore
    loop: a reader who finds the clamp finds the exception to it in the same file, and a
    second restore path cannot re-derive an assertion without passing through it.

    Note the app already got the ADJACENT case right — a window landing entirely past the
    new end leaves <2 samples, the builder raises, and the restore reports it. Partial
    overlap (``i0 <= n-2 < i1``) was the sibling that lost it, and that band is exactly
    what this closes.
    """
    n = int(parent_n_samples)
    a, b = int(i0), int(i1)
    if a < 0 or b < a:
        return (f"its saved window [{a:,}, {b:,}) is not a window at all")
    if b > n:
        return (f"its saved window [{a:,}, {b:,}) needs {b:,} samples but its parent "
                f"recording now holds {n:,}")
    if b - a < MIN_SEGMENT_SAMPLES:
        return (f"its saved window [{a:,}, {b:,}) is {b - a} sample(s); a segment needs "
                f"at least {MIN_SEGMENT_SAMPLES}")
    return None


@dataclass(frozen=True)
class RebasedMeta(RecordingMeta):
    """A :class:`~nanoscope.core.types.RecordingMeta` whose ``t0_s`` this app SET, and the
    instrument-clock instant it set it away from.

    WHY THE REBASE NEEDS A WITNESS. A segment's own axis starts at 0 — that is deliberate,
    it is what lets every per-recording panel run on a segment unchanged, and
    ``tests/core/test_segment.py`` pins it. But ``meta.t0_s == 0`` then means two
    completely different things in the app: "the file declared t₀ = 0" (the -700 mV apo
    recording) and "this app manufactured a zero for a cut of a recording that started at
    1154.9909 s". Read through ``t0_s`` alone they are indistinguishable, so every clock
    label in the app told the user its two clocks agreed and there was nothing to convert,
    on a document the SAME window names ``[1159-1165 s]`` and whose Notes say
    ``t₀=1158.99s``.

    ``rebased_from_t0_s`` is that instant, under the name :data:`nanoscope.core.clock.REBASE_ATTR`
    — the attribute ``core.clock`` looks for. Putting it on the META rather than only on
    :class:`SegmentOrigin` is the point: ``clock.clock_offset_s(document.meta, …)`` is the
    ONE seam the trace axis label, the event-table header, the PSD panel note and the
    manual-measurement table all read their clock story through, and it is handed a meta,
    not a document. A witness anywhere else has to be plumbed to each surface separately,
    which is precisely how the four surfaces came to disagree.

    Additive and default-valued, so every ``dataclasses.replace`` on a segment's meta
    (``TraceDocument.__post_init__`` stamps ``load_uid`` that way, ``Session.add_document``
    folds in sidecar issues) preserves the witness for free, and no reader, project field
    or schema changes.
    """

    rebased_from_t0_s: float = 0.0


def _rebased_meta(base: RecordingMeta, rebased_from_t0_s: float) -> RebasedMeta:
    """``base`` re-made as a :class:`RebasedMeta`, field for field.

    Enumerated from ``dataclasses.fields`` rather than by hand so an additive change to
    :class:`~nanoscope.core.types.RecordingMeta` cannot silently drop a field here — the
    same reasoning ``core.history`` gives for comparing results field-wise.
    """
    values = {f.name: getattr(base, f.name) for f in dc_fields(base)
              if f.init and f.name != "rebased_from_t0_s"}   # nesting: base may already be one
    return RebasedMeta(**values, rebased_from_t0_s=float(rebased_from_t0_s))


@dataclass(frozen=True)
class SegmentOrigin:
    """Provenance stamped on a derived :class:`SegmentDocument`.

    ``parent_key`` is the parent's PERSISTENT key — the resolved source path for a
    file-backed recording, or the parent segment's own ``segment_uid`` when a segment is
    cut from another segment (nesting). ``i0``/``i1`` are the half-open ``[i0, i1)`` sample
    window IN THE PARENT; ``t0_abs_s`` is the absolute experimental time of sample ``i0``
    in the parent (the segment's own axis is rebased to 0, so this is the only trace back
    to where it sat). ``name`` is the user-facing display name.
    """

    parent_key: str
    i0: int
    i1: int
    t0_abs_s: float
    segment_uid: str
    name: str


@dataclass(frozen=True)
class SegmentSpec:
    """The persistable description of a segment — everything the project needs to
    RE-DERIVE it from its parent on load. Goes in the manifest's ``segments`` list."""

    segment_uid: str
    parent_key: str
    i0: int
    i1: int
    name: str
    t0_abs_s: float = 0.0


@dataclass(frozen=True)
class SegmentDocument(TraceDocument):
    """A derived recording: one channel-crop of a parent trace, added to the Session as a
    full in-memory document.

    It IS a :class:`TraceDocument` (same frozen value-object contract: float32,
    contiguous, one array per channel), so every per-recording path runs on it unchanged.
    The extra ``origin`` stamp makes it identifiable as a segment and traceable to its
    parent, and drives project persistence.

    IDENTITY: a segment deliberately does NOT override ``doc_id``. Its ``meta.path`` is a
    UNIQUE synthetic path (the parent's path with the segment uid folded into the stem; see
    :func:`build_segment_document`), so its ``source_key`` is unique exactly like every file
    recording's, and its ``doc_id`` is that plus its own per-load token — which is exactly
    the id every analysis module stamps onto its result
    (``AnalysisResult.doc_id = identity.doc_id_for_meta(ctx.meta)``). That single fact is
    what makes a detection run on a segment land under the SEGMENT, not the parent.
    ``display_name`` still shows the user's name; the PROJECT identity is the uid (see
    :func:`persistent_key`), so persisted results re-attach after re-derivation regardless
    of the exact synthetic path or of which load re-derived it.
    """

    origin: SegmentOrigin = None  # type: ignore[assignment]  # always set by build_segment_document

    @property
    def display_name(self) -> str:
        return self.origin.name

    @property
    def spec(self) -> SegmentSpec:
        """The persistable :class:`SegmentSpec` for this segment."""
        o = self.origin
        return SegmentSpec(segment_uid=o.segment_uid, parent_key=o.parent_key,
                           i0=o.i0, i1=o.i1, name=o.name, t0_abs_s=o.t0_abs_s)


def persistent_key(doc: Document) -> str:
    """The identity that survives a project round-trip.

    A segment's stable ``segment_uid``; otherwise the document's ``source_key`` (the
    resolved source path for a file-backed recording). Results, calibrations and layers are
    persisted under THIS key so they re-attach after re-derivation, never under the
    volatile in-memory ``doc_id``.

    NOTE the file case is ``source_key``, NOT ``doc_id``: a ``doc_id`` now carries a
    per-load token (see :mod:`nanoscope.core.identity`) and would make every project
    round-trip fail to re-attach, because the reopened load's token is by construction a
    different one. The VALUES this function returns are exactly what it returned before that
    token existed, so projects written by either build read identically.
    """
    if isinstance(doc, SegmentDocument):
        return doc.origin.segment_uid
    return doc.source_key


def _short_key(s: str) -> str:
    """A short, stable digest of an arbitrary key (keeps a uid readable even when the
    parent key is a long absolute path)."""
    return hashlib.blake2s(s.encode("utf-8"), digest_size=4).hexdigest()


def derive_segment_uid(parent_key: str, i0: int, i1: int,
                       existing=()) -> str:
    """A DETERMINISTIC segment id — never ``uuid4``/random, so tests reproduce exactly.

    Derived from ``(parent_key, i0, i1)``; disambiguated with a trailing counter only when
    that base id is already taken among ``existing`` (two identical crops of one parent).
    """
    base = f"seg-{_short_key(str(parent_key))}-{int(i0)}-{int(i1)}"
    existing = set(existing)
    if base not in existing:
        return base
    k = 2
    while f"{base}-{k}" in existing:
        k += 1
    return f"{base}-{k}"


def build_segment_document(parent_doc: TraceDocument, i0: int, i1: int, *,
                           name: str, segment_uid: str,
                           parent_key: str | None = None) -> SegmentDocument:
    """Crop ``parent_doc`` to the half-open sample window ``[i0, i1)`` and return a
    derived :class:`SegmentDocument`.

    PURE — no Session, no Qt. Everything that lives ON the document is inherited: the
    sample rate, every channel's :class:`~nanoscope.core.types.ChannelInfo` (name, kind,
    ``is_dead``, ``raw_units``) and the nominal voltage (so the segment groups with its
    parent in the Files list). Every channel's signal is a float32 slice **copy** (owned +
    contiguous), so the segment is independent of the parent's array — closing the parent
    never strands the segment, and vice-versa. The time axis is rebased so ``meta.t0_s``
    is 0 (``meta.duration_s`` is then ``(i1 - i0) / fs``), and an :class:`SegmentOrigin` is
    stamped for persistence + traceability.

    THE REBASE IS WITNESSED. The returned ``meta`` is a :class:`RebasedMeta` carrying
    ``rebased_from_t0_s`` — the instrument-clock instant of the cut's first sample, the
    same number that goes on :class:`SegmentOrigin`. ``core.clock`` reads it, so a
    segment's axis label, event-table tooltip and saved measurement stamps name the
    experiment time this data was recorded at instead of asserting that its t₀ is 0.

    Per-channel CALIBRATION is Session state, not document state, so it is inherited by
    :meth:`~nanoscope.core.session.Session.create_segment`, not here.

    ``i0``/``i1`` are clamped into ``[0, n]`` and ordered; a crop shorter than
    :data:`MIN_SEGMENT_SAMPLES` (empty or a single sample) is rejected with a clear
    :class:`ValueError`.
    """
    if not isinstance(parent_doc, TraceDocument):
        raise TypeError(
            f"{parent_doc!r} is not a trace document; only trace recordings can be segmented.")

    n = parent_doc.meta.n_samples
    a = max(0, min(int(min(i0, i1)), n))
    b = max(0, min(int(max(i0, i1)), n))
    if b - a < MIN_SEGMENT_SAMPLES:
        raise ValueError(
            f"segment window [{i0}, {i1}) clamps to [{a}, {b}) = {b - a} sample(s); a segment "
            f"needs at least {MIN_SEGMENT_SAMPLES} samples.")

    fs = parent_doc.meta.sample_rate_hz
    # float32 slice COPY per channel: owned + C-contiguous, so the segment never keeps the
    # parent's whole array alive and survives the parent being closed.
    channels = tuple(
        np.array(parent_doc.channels[c][a:b], dtype=np.float32, order="C")
        for c in range(len(parent_doc.channels))
    )
    # WHERE THE CUT SAT IN THE EXPERIMENT. Through `clock.instrument_origin_s`, not
    # `parent.meta.t0_s`, because a parent can itself be a rebased segment: cutting a
    # segment OF a segment read the inner cut's manufactured 0 and quietly threw the
    # grandparent's origin away, so a nested cut of the 1154.9909 s recording came back
    # claiming its data was recorded at 2 s. One rule, every generation.
    parent_origin_s = clock.instrument_origin_s(parent_doc.meta)
    if parent_origin_s is None:                     # no sample rate -> no clock at all
        parent_origin_s = float(parent_doc.meta.t0_s)
    t0_abs_s = parent_origin_s + (a / fs if fs else 0.0)

    if parent_key is None:
        parent_key = persistent_key(parent_doc)

    origin = SegmentOrigin(parent_key=str(parent_key), i0=a, i1=b, t0_abs_s=t0_abs_s,
                           segment_uid=str(segment_uid), name=str(name))

    where = f"samples {a:,}–{b:,}" + (f"  ·  t₀={t0_abs_s:.6g}s" if fs else "")
    meta = replace(
        parent_doc.meta,
        path=_segment_path(parent_doc.meta.path, str(segment_uid)),
        n_samples=b - a,
        t0_s=0.0,               # rebased: the segment's own clock starts at 0
        loaded_range=None,      # a derived crop is not a partial FILE load
        notes=(f"Derived segment of {parent_doc.display_name} ({where}).",),
        issues=(),              # the parent's load issues are not the segment's
        load_uid="",            # a crop is its OWN load: cleared so __post_init__ mints a
                                # fresh token rather than inheriting the parent's identity
    )
    # ...and the rebase does not get to be invisible. `t0_s = 0` above is a zero this app
    # made; `RebasedMeta` carries the instant it was made away from, so every surface that
    # asks core/clock.py "which clock is this and where is it" gets 1158.9909 s instead of
    # the reassurance that there is nothing to convert. See RebasedMeta's docstring.
    meta = _rebased_meta(meta, t0_abs_s)
    return SegmentDocument(meta=meta, channels=channels, origin=origin)


def follow_parent_clock(seg: "SegmentDocument", parent_doc: TraceDocument) -> "SegmentDocument":
    """``seg`` re-made on its parent's CURRENT clock — what a fresh cut of the same window
    would carry (rate, provenance, unit scale, ``t0_abs_s``, the note that names it).

    A segment is a window of its parent, so when the parent's time base is re-declared
    (:meth:`~nanoscope.core.session.Session.set_sample_rate_declaration`) the segment's
    clock follows — live, exactly as it does on a project restore, where segments are
    re-derived from the restored parent. Without this the two disagreed: a segment cut
    before the declaration kept the old clock until the project was reopened, and then
    silently came back on the new one. Samples are untouched; only the meta and origin
    move, by the same arithmetic as :func:`build_segment_document`.
    """
    a, b = int(seg.origin.i0), int(seg.origin.i1)
    fs = parent_doc.meta.sample_rate_hz
    parent_origin_s = clock.instrument_origin_s(parent_doc.meta)
    if parent_origin_s is None:
        parent_origin_s = float(parent_doc.meta.t0_s)
    t0_abs_s = parent_origin_s + (a / fs if fs else 0.0)
    where = f"samples {a:,}–{b:,}" + (f"  ·  t₀={t0_abs_s:.6g}s" if fs else "")
    meta = replace(
        seg.meta,
        sample_rate_hz=fs,
        rate_provenance=parent_doc.meta.rate_provenance,
        time_scale_s=parent_doc.meta.time_scale_s,
        refused_rate_hz=parent_doc.meta.refused_rate_hz,
        notes=(f"Derived segment of {parent_doc.display_name} ({where}).",),
    )
    meta = _rebased_meta(meta, t0_abs_s)
    return replace(seg, meta=meta, origin=replace(seg.origin, t0_abs_s=t0_abs_s))


def _segment_path(parent_path: Path, segment_uid: str) -> Path:
    """A UNIQUE synthetic path for a segment — the parent path with the uid folded into the
    STEM (before the extension), so ``str(path.resolve())`` is a stable, unique ``doc_id``
    that no other recording collides with, and the parent's ``stem_key`` (which strips the
    extension) never accidentally matches (no spurious duplicate-representation banner)."""
    marker = f"·segment·{segment_uid}"
    try:
        return parent_path.with_stem(f"{parent_path.stem} {marker}")
    except ValueError:               # a name with no stem (e.g. a dotfile) — append instead
        return parent_path.with_name(f"{parent_path.name} {marker}")
