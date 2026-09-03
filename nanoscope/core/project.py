"""Persist a whole analysis session to a reopenable ``.nanoscope`` project file.

This module is the PURE (de)serializer — no Qt, no :class:`~nanoscope.core.session.Session`.
It moves a plain :class:`ProjectState` (documents-as-references, calibrations,
layers, view mode, per-result event tables + provenance, manual measurements) to
and from a single self-contained file, so the GUI restore code (main_window) can
stay a thin translator between ``Session`` and this dataclass.

FILE FORMAT
-----------
A ``.nanoscope`` file is a ZIP archive containing:

* ``manifest.json`` — the whole project *except* the bulk event/measurement tables::

      {
        "format": "nanoscope-project",
        "version": 1,
        "app_version": "0.1.0",
        "documents":    [{"path": "<abspath>", "loaded_range": [start, stop] | null,
                          "source": {"n_samples", "size", "digest", "mtime_ns"} | null}],
        "segments":     [{"segment_uid", "parent_key", "i0", "i1", "name", "t0_abs_s"}],
        "calibrations": [{"doc_id", "channel", "channel_name", "channel_kind", "gain",
                          "offset", "target_units", "provenance", "note"}],
        "layers":       [{"doc_id", "channel", "channel_name", "channel_kind", "color",
                          "visible", "time_offset_s", "excluded"}],
        "view_mode":    "overlay" | "stacked",
        "results":      [{"doc_id", "channel", "module_id", "module_version",
                          "events": "results/<i>.parquet", "provenance": {...},
                          "params": {...}, "calibration_fingerprint": [gain, offset]|null,
                          "baseline_fingerprint": "..."|null}],
        "label_vocabulary": [{"key", "value", "hotkey", "note"}],
        "colours":      [{"key", "slot", "ink"?}],   # v9 — the colour identity map
        "fits":         [{"doc_id", "channel", "module_id",
                          "table": "fits/<i>.parquet" | absent,
                          "curve": {"params", "ci", "loglik", "aic", "bic", "n",
                                    "model", "meta"} | absent,
                          "params": {...}, "calibration_fingerprint": [gain, offset]|null,
                          "baseline_fingerprint": "..."|null, "warnings": [...]}],
        "measurements": "measurements.parquet" | null
      }

  ``documents`` holds only FILE-backed recordings (persisted as path references).
  A ``segments`` entry is a DERIVED recording: a crop ``[i0, i1)`` of its parent (a
  file's path, or another segment's uid when nested), re-derived from the parent on load
  rather than stored as signal. The ``doc_id`` in ``calibrations`` / ``layers`` /
  ``results`` is the PERSISTENT key — a file's path, or a segment's ``segment_uid`` — never
  the volatile in-memory document id, so those records re-attach after re-derivation.
  ``segments`` is additive: a version-1 project has no such key and loads unchanged.

* ``results/<i>.parquet`` — one per :class:`AnalysisResult` EventSet ``.df`` (the
  FULL curated frame: ``accepted`` + edited ``start_idx``/``end_idx`` + recomputed
  columns), stored losslessly with pandas dtypes via pyarrow. WHOLE-FRAME, and that
  is load-bearing: the frame carries columns that are deliberately OUTSIDE
  ``events.SCHEMA`` (``added``, ``orig_start_idx``/``orig_end_idx``, ``source``/
  ``review``), and they are how the app knows who found a row and whether anybody has
  ruled on it. Never narrow this member to the frozen 20 to "save space" — measured on
  the owner's 15 s ‑1000 mV file (115 events), the member is 26,964 bytes for the frozen
  20 and 31,475 for all 25 columns, i.e. the five description columns cost 4.5 kB and
  are not re-derivable from anything else in the file. See :func:`_restore_event_extras`
  for what load does about their dtypes.
* ``fits/<i>.parquet`` — one per DERIVED table result (today: the ``fit.levels`` CUSUM
  level frame). Distribution fits have no table and ride in the manifest as ``curve``;
  their density is rebuilt on load rather than stored. See :class:`FitRec`.
* ``measurements.parquet`` — the manual :class:`MeasurePanel` table, when non-empty.

WHY THIS SHAPE
--------------
The raw 180 MB signal is NEVER stored: documents are persisted as REFERENCES
(absolute path + the loaded :class:`SampleRange`) and reopened from disk on load.
What a user cannot cheaply recreate — calibration, layer view state, curated
detections, manual measurements — is what we keep.

Provenance dicts are coerced JSON-safe (numpy scalars/arrays down to Python
builtins) so they survive the round-trip. The per-result overlay hints are NOT
stored: their sample-index arrays are rebuilt from the event ``df`` on restore, so
there is never a numpy array in the JSON.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
import pandas as pd

from . import atomic_write
from .identity import SourceFingerprint

__all__ = [
    "FORMAT",
    "VERSION",
    "MIN_READ_VERSION",
    "MANIFEST_NAME",
    "ProjectError",
    "ProjectVersionError",
    "DocumentRef",
    "SegmentRec",
    "CalibrationRec",
    "LayerRec",
    "ResultRec",
    "FitRec",
    "ProjectState",
    "save_project",
    "load_project",
]

FORMAT: str = "nanoscope-project"
# v2 ADDS the `segments` list (derived recordings). v3 ADDS each result's `params` (the
# settings it ran under) and `calibration_fingerprint` (the gain/offset it was computed
# under). v4 ADDS each document's `source` fingerprint — what the file on disk CONTAINED
# when the project was saved, so a restore can tell "the same file" from "a different file
# with the same name". v5 ADDS each calibration's `note` — the one field a project could not
# carry, which is why committing a calibration after a restore blanked the user's sidecar (see
# CalibrationRec). v6 ADDS the `fits` list — the DERIVED results (fit.dwell / fit.depth /
# fit.levels), which until then were dropped on save without a word (see FitRec). All five
# are purely ADDITIVE extensions: an older project simply
# lacks those keys, so this build still reads it (see MIN_READ_VERSION) and it restores
# exactly as before — a missing `params` is {}, a missing calibration fingerprint falls back
# to the restored calibration, a missing `source` reads as UNKNOWN, which raises no
# warning rather than a false one, a missing `note` reads as "this file has no opinion",
# which leaves the note the sidecar already installed alone, and a missing `fits` is [].
#
# WHY v6 BUMPS THE VERSION EVEN THOUGH IT IS ADDITIVE. Additive-ness governs whether THIS
# build can read OLDER files; it says nothing about whether older builds can safely read
# THIS one. They cannot. A build that predates `fits` accepts a version-5 file, ignores the
# key and the `fits/<i>.parquet` members, and then DROPS THEM ON ITS NEXT SAVE — silently,
# because it never knew they were there. That is exactly the loss FitRec exists to end,
# reintroduced one level up and made worse by living in a file that gives no sign of it.
# The bump converts that silent strip into ProjectVersionError's clean refusal, and costs
# nothing in the other direction: MIN_READ_VERSION is still 1, so every project the owner
# already has still opens. (Every previous additive key here bumped for the same reason;
# only per-result `warnings` did not, and no released build was ever on the losing side of
# that one.)
#
# v7 ADDS `channel_name` / `channel_kind` to CalibrationRec, LayerRec, ResultRec and FitRec.
# All four filed their record under a bare channel INDEX, and an index is a position, not an
# identity: re-export a recording with its signal columns in the other order — same sample
# count, same channel count, byte-identical size — and a `gain=0.5, offset=-1.0, 'pA'`
# calibration lands on the VOLTAGE channel (the -600 mV trace then reads -301.0 pA), the real
# current channel falls back to gain 1.0, an 11-event detection is reattached to the voltage
# trace, and the result's stored fingerprint MATCHES the misplaced calibration so the
# staleness system positively certifies those events as current-valid. The identity is
# optional on read for the usual reason: a v1-v6 project simply carries None, which
# `calibration.resolve_channel` treats as "resolve by index, exactly as before, silently".
# It bumps VERSION on the same argument v6 did — an older build reads a v7 file, ignores the
# keys, and strips them on its next save.
#
# v8 ADDS `kind` to CalibrationRec: the user's "treat this channel as current/voltage/
# conductance" override. It is NOT `channel_kind`, which v7 added and which records what the
# FILE HEADER declared so `resolve_channel` can find the channel again. This one is the
# correction the user makes when the header is wrong or silent, and it is load-bearing in the
# viewer: `trace_view._kind` feeds BOTH the pane ORDER (current on top) and the OVERLAY
# GROUPING key, so without it a channel the user relabelled to pA still sorts as UNKNOWN and
# sits below the voltage pane — the reversal Abraham asked about. Optional on read for the
# usual reason: a v1-v7 project carries None, which means "no override, the header stands",
# and `to_calibration` then leaves whatever the sidecar already installed alone rather than
# blanking it (see CalibrationRec). Bumps VERSION on v6's argument — an older build reads a
# v8 file, ignores the key, and strips the user's override on its next save.
#
# v9 ADDS `colours` — the project's `core.layer.ColourRegistry`: which SLOT each
# recording+channel holds in the categorical cycle, and the ink the user assigned it by
# hand. Until now a recording's colour was persisted for exactly one of the four surfaces
# that draw it: `layers[].color` carried the trace view's ink, and the Plots panel
# recoloured everything from POSITION on every refresh, so "the Zn run is the green one"
# was true in one window, false in another, and false again after closing a neighbour. The
# registry is keyed by the persistent key (a path, or a segment uid) exactly as
# `calibrations` / `layers` / `results` are, so it re-attaches after re-derivation and
# survives a reopen. Optional on read for the usual reason: a v1-v8 project has no such key
# and loads with an EMPTY registry, meaning "nobody has assigned anything yet"; the first
# discovery then re-assigns slots in the order it always did, which is visually identical
# to what that project last showed, since a project with no registry is one that was
# drawing positional colours anyway. `layers[].color` is NOT removed and NOT derived from
# the registry on load: it is the ink actually on screen for that layer, the registry is
# the rule that chose it, and a restore that recomputed one from the other would silently
# repaint a project saved by a build whose palette differed. Bumps VERSION on v6's argument
# — an older build reads a v9 file, ignores the key, and strips every hand-picked colour on
# its next save.
#
# v10 ADDS the event result's own ``module_version`` and top-level
# ``baseline_fingerprint``. Detector provenance normally duplicates the baseline stamp,
# but plugins and derived results need not; restore also used the currently installed
# module version (or ``""`` for the unregistered reconciler), which rewrote history.
# Both fields are optional on read so v1-v9 remain compatible. The version bumps because
# an older NanoScope would accept and then strip these audit/staleness stamps on re-save.
VERSION: int = 10
MIN_READ_VERSION: int = 1   # oldest project this build can still read (v1 predates segments)
MANIFEST_NAME: str = "manifest.json"


class ProjectError(ValueError):
    """A ``.nanoscope`` file could not be read (not a project, corrupt, or missing parts)."""


class ProjectVersionError(ProjectError):
    """A ``.nanoscope`` file declares a version this build cannot read."""


@contextmanager
def _manifest_section(name: str, section: str) -> Iterator[None]:
    """Turn ANY decoding failure in one manifest section into a :class:`ProjectError`.

    :func:`load_project` promises exactly one exception type — "not a project, corrupt, or
    an unsupported version" is one answer, and the caller catches one class. It kept that
    promise for the shapes somebody had thought of (missing manifest, bad JSON, absent
    parquet member) and broke it for every OTHER malformation of the same file: a
    ``layers`` entry without ``color`` raised a bare ``KeyError('color')``; a
    ``loaded_range`` written as a string raised ``TypeError``; a ``calibrations`` entry
    with a non-numeric ``channel`` raised ``ValueError`` — six shapes in all, each escaping
    as whatever the decoding expression happened to throw.

    One guard around each section rather than a check per field, because the failure is not
    a field: it is "this manifest is not shaped like a manifest", and the useful thing to
    say is WHICH PART of it and what Python found there. The GUI's blanket
    ``except Exception`` meant no user ever saw a traceback for this, which is exactly why
    it stayed wrong — the contract is for the API's other callers and for the next reader.
    """
    try:
        yield
    except ProjectError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        detail = f"missing key {exc}" if isinstance(exc, KeyError) else str(exc)
        raise ProjectError(
            f"{name}: the {section!r} section of {MANIFEST_NAME} is malformed ({detail})."
        ) from exc


# ================================================================================================
# ProjectState — the plain, Qt-free description of a saved session.
# ================================================================================================

@dataclass
class DocumentRef:
    """An open document persisted as a REFERENCE, never its signal.

    ``path`` is the resolved absolute path (== the document's ``source_key``, which is also
    its persistent key); ``loaded_range`` is the ``[start, stop)`` sample window that was
    loaded, or None for a whole-file load. Reopened from disk on restore.

    ``source`` is v4: WHAT THAT PATH CONTAINED. A path is a name, not a recording, and the
    restore used to check nothing but ``path.exists()``. Re-exporting from Clampex over the
    same filename therefore reattached 5 saved events to a 5,000-sample flat trace — 5 of
    them entirely past the end of the file, event 0 reporting a 30.2 pA blockade in a signal
    whose whole standard deviation is 1.1 pA — with no warning of any kind. ``None`` means
    the stamp is simply unknown (a v1–v3 project, or a source we could not read), which the
    restore must treat as "no evidence", never as "verified identical". See
    :class:`nanoscope.core.identity.SourceFingerprint`.
    """

    path: str
    loaded_range: tuple[int, int] | None = None
    source: SourceFingerprint | None = None


@dataclass
class SegmentRec:
    """A derived segment persisted as a SPEC — enough to RE-DERIVE it from its parent on
    load (the signal is never stored).

    ``parent_key`` is the parent's persistent key: the parent file's path, or the parent
    segment's ``segment_uid`` when this segment is nested. ``i0``/``i1`` are the half-open
    ``[i0, i1)`` sample window in the parent; ``segment_uid`` is this segment's own
    persistent key, under which its calibrations / layers / results are stored.
    """

    segment_uid: str
    parent_key: str
    i0: int
    i1: int
    name: str
    t0_abs_s: float = 0.0


@dataclass
class CalibrationRec:
    """One channel's calibration, keyed by ``(doc_id, channel)``.

    THE FIELD LIST HERE IS CHECKED AGAINST
    :class:`~nanoscope.core.calibration.ChannelCalibration` by a test, and that check is the
    point of this class rather than a tidiness rule. ``note`` used to be missing, and the
    consequence was not that a project "lost a note": the restore built a
    ``ChannelCalibration`` with the default empty note, the panel's *Apply* then faithfully
    carried that empty note forward with ``note=existing.note``, and ``save_sidecar``
    overwrote ``<datafile>.npcal.json`` — the user's own file, next to their raw data — with
    ``"note": ""``. Measured end to end: open a project, change a gain 0.5 → 0.75, press
    Apply, and *"electrode replaced 2026-07-01 — gain measured against the 1 MΩ standard"*
    became ``""`` with **zero prompts** and a status line reading "Applied · saved to …".
    The confirmation that guards the sidecar never fires, because this is a WRITE, not the
    removal it watches for.

    So the record has to carry every field the value object has, and
    :meth:`to_calibration` / :meth:`from_calibration` are the only translation, so a field
    added to one and forgotten in the other is a failing test rather than a silent blanking.

    ``note = None`` means **this project file did not record one** (any project written
    before v5). That is not the same as ``""``, and conflating them is exactly how the note
    was lost: a v4 project has nothing to say about the note, so :meth:`to_calibration`
    keeps whatever the channel already has — normally the sidecar's, loaded moments earlier
    by ``Session.add_document``. A v5 project always records a string, so an empty one there
    really does mean "no note".
    """

    doc_id: str
    channel: int
    gain: float
    offset: float
    target_units: str
    provenance: str
    note: str | None = None
    #: v7 — WHICH channel, not merely which position. See the note above :data:`VERSION`
    #: and :func:`nanoscope.core.calibration.resolve_channel`. ``None`` means a pre-v7
    #: project that did not record it, which resolves by index exactly as before.
    channel_name: str | None = None
    channel_kind: str | None = None
    #: v8 — the USER'S "treat this channel as" override
    #: (:attr:`~nanoscope.core.calibration.ChannelCalibration.kind`), which is NOT
    #: ``channel_kind`` above: that one records what the FILE HEADER declared, and exists
    #: so `resolve_channel` can find the same channel again after a re-export. This one is
    #: the correction the user made when the header was wrong or silent, and it decides
    #: pane order and overlay grouping in the trace view.
    #:
    #: ``None`` == no override (a pre-v8 project, or a channel the user never corrected),
    #: which is the same three-state rule as ``note``: writing a confident "no override"
    #: for a project that simply predates the field is how the restore would carry a
    #: default forward and let the panel's *Apply* overwrite the user's own sidecar.
    kind: str | None = None

    @classmethod
    def from_calibration(cls, doc_id: str, channel: int, cal, info=None) -> "CalibrationRec":
        """Persist a :class:`~nanoscope.core.calibration.ChannelCalibration` in full.

        ``info`` is the channel's :class:`~nanoscope.core.types.ChannelInfo`, whose name +
        kind become the record's identity. Optional only so the pure serializer's own
        tests can build a record without a document; every save passes it.
        """
        from .calibration import channel_identity

        name, kind = (None, None) if info is None else channel_identity(info)
        return cls(doc_id=doc_id, channel=int(channel), gain=float(cal.gain),
                   offset=float(cal.offset), target_units=str(cal.target_units),
                   provenance=cal.provenance.value, note=str(cal.note),
                   channel_name=name, channel_kind=kind,
                   kind=(None if cal.kind is None else cal.kind.value))

    def to_calibration(self, existing=None):
        """Rebuild the value object. ``existing`` is the channel's CURRENT calibration, whose
        only job is to supply a ``note`` this record does not carry (a pre-v5 project).

        Raises ``ValueError`` for a gain/offset a :class:`ChannelCalibration` refuses — a
        project is a file on disk and can be edited or damaged, and the caller decides what
        to do about one bad channel.
        """
        from .calibration import ChannelCalibration
        from .types import Provenance

        try:
            prov = Provenance(self.provenance)
        except ValueError:
            prov = Provenance.UNKNOWN
        note = self.note
        if note is None:
            note = "" if existing is None else str(existing.note)
        # The "treat this channel as" override gets the note's treatment for the note's
        # reason: a pre-v8 project has no opinion about it, so it must not overwrite the
        # one the sidecar supplied moments ago in `Session.add_document`. Restoring None
        # here and letting the panel's Apply carry that default forward is precisely how
        # the note was blanked out of the user's own `.npcal.json`.
        kind = self.kind
        if kind is None and existing is not None:
            kind = existing.kind
        return ChannelCalibration(gain=self.gain, offset=self.offset,
                                  target_units=self.target_units, provenance=prov,
                                  note=str(note), kind=kind)


@dataclass
class LayerRec:
    """One overlay layer's view state, keyed by (doc_id, channel)."""

    doc_id: str
    channel: int
    color: str
    visible: bool
    time_offset_s: float
    #: The user's own untick, distinct from "not the recording on screen" (see
    #: :class:`nanoscope.core.layer.Layer`). Additive and optional on read: a project
    #: written before it existed has no such key and restores as False, which is
    #: exactly what it meant — the untick used to DELETE the layer, so a layer present
    #: in an older project was never an excluded one.
    excluded: bool = False
    #: v7 — the identity of the channel this record belongs to (name + kind), so a
    #: re-export that reorders the signal columns cannot move it onto another trace.
    #: ``None`` == a pre-v7 project, which resolves by index exactly as before. See
    #: the note above :data:`VERSION` and
    #: :func:`nanoscope.core.calibration.resolve_channel`.
    channel_name: str | None = None
    channel_kind: str | None = None


@dataclass
class ResultRec:
    """One detection/ML :class:`AnalysisResult`, reduced to what re-open needs: the
    EventSet ``df`` (curated, in full), its provenance, the PARAMS it ran under, the
    calibration fingerprint it was computed under, and where it belongs.

    ``params`` and ``calibration_fingerprint`` are v3 additions, and both exist because a
    result is only reproducible and only auditable if it carries them:

    * ``params`` is the exact settings dict the module was handed (``min_dwell_s``,
      ``k_in``/``k_out``, the whole preprocessing chain; for an ML run the model path,
      thresholds, range and pipeline configs). Without it a reopened project could offer a
      "Re-run" that silently substituted the PANEL'S DEFAULTS — re-detecting with different
      settings and destroying every hand curation in the process.
    * ``calibration_fingerprint`` is the ``(gain, offset)`` in force WHEN THE RESULT WAS
      COMPUTED — not the live one. It is the staleness stamp: the app compares it against
      the channel's current calibration to decide whether the depths on screen are still
      valid. Re-deriving it from the live calibration on load made every reopened project
      look freshly-computed, clearing a "calibration changed" banner that was telling the
      truth. ``None`` means a pre-v3 project that never stored it.

    ``warnings`` is the caveat the module attached to these very numbers — above all
    the "this recording's open-pore level was never probed, so the depths came from an
    unconfirmed default" advisory. Dropping it on reopen left the table on screen
    looking unqualified while the numbers still carried the qualification, which is the
    same defect ``calibration_fingerprint`` exists to prevent one level down. Optional
    on read: a project written before it simply has no key and restores as ``()``,
    exactly as before.
    """

    doc_id: str
    channel: int
    module_id: str
    events: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    calibration_fingerprint: tuple[float, float] | None = None
    warnings: tuple[str, ...] = ()
    #: v7 — the identity of the channel this record belongs to (name + kind), so a
    #: re-export that reorders the signal columns cannot move it onto another trace.
    #: ``None`` == a pre-v7 project, which resolves by index exactly as before. See
    #: the note above :data:`VERSION` and
    #: :func:`nanoscope.core.calibration.resolve_channel`.
    channel_name: str | None = None
    channel_kind: str | None = None
    #: v10 — the producer version and the baseline stamp belong to the stored result,
    #: not to whichever module/config happens to be installed when a project reopens.
    module_version: str = ""
    baseline_fingerprint: str | None = None


@dataclass
class FitRec:
    """One DERIVED result — a distribution fit (``fit.dwell`` / ``fit.depth``) or a CUSUM
    level table (``fit.levels``) — reduced to what re-open needs.

    These are separate from :class:`ResultRec` because their payload is not an EventSet.
    Until v6 the save path collected only ``isinstance(res.payload, EventSet)``, so every
    derived fit was silently dropped: a user could fit a dwell distribution, bootstrap its
    CIs over a minute of compute, save, reopen — and find the fit gone with no message. The
    numbers were never at risk (they are re-derivable from the events, which DID persist),
    but "re-derivable" is not "there", and silence is the part that made it a defect.

    Exactly one of ``table`` / ``curve`` is set:

    * ``table`` — the ``fit.levels`` frame (long-form, one row per level), stored as parquet
      exactly like an EventSet.
    * ``curve`` — a :class:`~nanoscope.analysis.stats.FitResult` flattened to JSON:
      ``params``/``ci``/``loglik``/``aic``/``bic``/``n``/``model``/``meta``. The density
      itself is NOT stored — it is a closure. It is rebuilt on load by
      :func:`~nanoscope.analysis.stats.density_from`, the same function the fitters
      themselves call, from ``params`` + ``meta``. Storing a second copy of those formulae
      here would let the restored curve drift away from the tau/AIC printed beside it.

    Both staleness stamps ride along for the reason spelled out in :class:`ResultRec`:
    a fit re-derived on load looks fresh, and a fit that looks fresh stops warning that the
    calibration or the open-pore baseline moved under it.
    """

    doc_id: str
    channel: int
    module_id: str
    table: pd.DataFrame | None = None
    curve: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    calibration_fingerprint: tuple[float, float] | None = None
    baseline_fingerprint: str | None = None
    warnings: tuple[str, ...] = ()
    #: v7 — the identity of the channel this record belongs to (name + kind), so a
    #: re-export that reorders the signal columns cannot move it onto another trace.
    #: ``None`` == a pre-v7 project, which resolves by index exactly as before. See
    #: the note above :data:`VERSION` and
    #: :func:`nanoscope.core.calibration.resolve_channel`.
    channel_name: str | None = None
    channel_kind: str | None = None


@dataclass
class ProjectState:
    """The whole persisted session, as plain data (DataFrames + dicts + scalars)."""

    documents: list[DocumentRef] = field(default_factory=list)
    segments: list[SegmentRec] = field(default_factory=list)
    calibrations: list[CalibrationRec] = field(default_factory=list)
    layers: list[LayerRec] = field(default_factory=list)
    view_mode: str = "overlay"
    results: list[ResultRec] = field(default_factory=list)
    measurements: pd.DataFrame | None = None
    app_version: str = ""
    # v6: a project written before it has no `fits` key and restores as [], exactly as
    # every project did before. See FitRec, and the note above VERSION for why a key that
    # is additive on read still bumps the format version.
    fits: list[FitRec] = field(default_factory=list)
    #: v6. The user's label palette (analysis.labels.LabelVocabulary.to_json()). Session-wide
    #: rather than per-result, because it is a vocabulary: the same keys apply to every
    #: recording in the project, and a palette that reset between sessions would retrain the
    #: user's fingers each time. Folded into v6 rather than bumping again because v6 has not
    #: shipped — extending an unreleased format is not a compatibility event.
    label_vocabulary: list[dict[str, str]] = field(default_factory=list)
    #: v9. The project's colour identity map — :class:`nanoscope.core.layer.ColourRegistry`
    #: as ``to_json()`` rows. Session-wide and keyed by the PERSISTENT key, because a
    #: recording's colour is a property of the recording and not of the window that happens
    #: to be drawing it; that is the whole of D1 and the reason this is here rather than in
    #: the Plots panel's ``StyleSettings`` (which is per-window, unpersisted, and discarded
    #: on close — see the note in ``core.layer``). Stored as plain rows rather than as the
    #: registry object so this module keeps its one job, moving PLAIN DATA in and out of a
    #: file; ``ColourRegistry.from_json`` is the one decoder.
    colours: list[dict[str, Any]] = field(default_factory=list)


# ================================================================================================
# JSON-safety — provenance dicts carry numpy scalars/arrays; JSON does not.
# ================================================================================================

def _json_safe(obj: Any) -> Any:
    """Recursively coerce ``obj`` into JSON-encodable builtins.

    numpy scalars become Python scalars, numpy arrays become nested lists, tuples
    become lists, dict keys become strings. Anything already JSON-safe passes
    through; an unrecognised leaf falls back to ``str`` so a stray object can never
    abort a save. NaN/inf floats are preserved (the reader is Python's json too).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):          # np.float32/np.int64/np.bool_/... -> Python scalar
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ================================================================================================
# parquet helpers — lossless DataFrame <-> bytes via pyarrow (preserves pandas dtypes).
# ================================================================================================

def _fingerprint_to_json(fp: Any) -> list[float] | None:
    """A ``(gain, offset)`` staleness stamp as a JSON pair, or None when there is none.

    Anything that is not a 2-element numeric pair becomes None rather than a half-written
    stamp: a fingerprint that cannot be compared must read as "unknown", never as "matches".
    """
    if fp is None or isinstance(fp, (str, bytes)):
        return None   # a string is iterable; "12" must not read as the pair (1.0, 2.0)
    try:
        vals = [float(v) for v in fp]
    except (TypeError, ValueError):
        return None
    return vals if len(vals) == 2 else None


def _identity_to_json(rec: Any) -> dict[str, str]:
    """The v7 ``channel_name`` / ``channel_kind`` keys, or nothing.

    ABSENT, never null: a record loaded from a pre-v7 project genuinely does not know which
    channel it was measured on, and writing a name for it would turn "I don't know" into a
    confident claim — the same three-state spelling ``note`` needed, for the same reason.
    ``resolve_channel`` reads absence as "resolve by index, exactly as before".
    """
    out: dict[str, str] = {}
    if getattr(rec, "channel_name", None) is not None:
        out["channel_name"] = str(rec.channel_name)
    if getattr(rec, "channel_kind", None) is not None:
        out["channel_kind"] = str(rec.channel_kind)
    return out


def _identity_from_json(raw: dict[str, Any]) -> dict[str, str | None]:
    """Inverse of :func:`_identity_to_json`. Absent stays None — see there."""
    return {
        "channel_name": (None if raw.get("channel_name") is None
                         else str(raw["channel_name"])),
        "channel_kind": (None if raw.get("channel_kind") is None
                         else str(raw["channel_kind"])),
    }


def _fingerprint_from_json(raw: Any) -> tuple[float, float] | None:
    """Inverse of :func:`_fingerprint_to_json`; None for absent or malformed stamps."""
    if raw is None or isinstance(raw, (str, bytes)):
        return None
    try:
        vals = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    return vals if len(vals) == 2 else None  # type: ignore[return-value]


def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=True)
    return buf.getvalue()


def _parquet_bytes_to_df(blob: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(blob), engine="pyarrow")


def _restore_event_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Put ``source``/``review`` back in ``string`` dtype after a parquet round-trip.

    NOT a migration and NOT a repair of values. Nothing is added, nothing is dropped,
    no cell changes: a project WITHOUT these columns comes back without them and reads
    "unknown" for every row, which is the entire point of declaring them outside
    ``events.SCHEMA`` (the owner was promised he would not have to migrate his saved
    projects, and an absent column is how that promise is kept). This function only
    settles the DTYPE, and only for these two columns.

    WHY THE DTYPE IS WORTH A FUNCTION — RE-MEASURED, AND IT IS NOT A CRASH ANY MORE.
    A project file is written by whichever build the user had, so it is exactly the door
    a ``boolean``-dtyped ``review`` arrives through, and it survives parquet as
    ``boolean``: measured with this function neutralised, pandas 2.2.3 / pyarrow, a
    ``boolean`` column holding ``pd.NA`` reloads as ``boolean`` with ``is_bool_dtype``
    True. THE FAILURE THIS PARAGRAPH USED TO CITE IS NOW UNREACHABLE, and the same change
    that added this guard is what retired it. ``merge._merged_row`` does carry a
    non-schema column with ``if pd.api.types.is_bool_dtype(series):
    series.to_numpy(dtype=bool).any()``, which does raise ``ValueError: cannot convert to
    'bool'-dtype NumPy array with missing values`` on a missing value — but ``source`` and
    ``review`` are routed to ``merge._combined_provenance_cell`` on the line ABOVE that
    branch and never reach it. Measured on the frame the old WHY said would fail: the
    unguarded ``boolean`` frame reloads, "Merge all" SUCCEEDS, and the merged rows read
    unknown out of ``combine_reviews``. Do not paste the crash back into this docstring;
    if it ever becomes true again it will be because merge's routing line was deleted,
    which is a defect in ``merge``, not evidence for this function.

    WHAT IT PROTECTS TODAY, measured the same way (guard off, one saved project): the
    wrong dtype is PERMANENT, because nothing downstream repairs it and load is the only
    place it passes through. It reloads ``boolean``; re-saved it goes back into the
    project file as ``boolean``; it comes out of a re-measure install
    (`remeasure.merge_late_curation`) still ``boolean``; and concatenated with a frame a
    detector stamped it becomes ``object`` — neither of the two dtypes ``events.py``
    declares. That is the whole benefit and it is worth stating plainly: this is a
    NORMALISER on the one door legacy dtypes come through, not a guard against a crash.
    It is also the last thing standing between a ``boolean`` ``review`` and merge's
    generic bool carry, which is one routing line away — defence in depth, named as such.

    A value this build does not recognise ("True" from that boolean column) is KEPT
    VERBATIM and simply reads as unknown through `events.review_of`, which already
    refuses to treat an unrecognised spelling as evidence that a person ruled. Rewriting
    it into one of the two tokens would be inventing a curator's decision.

    Never raises: a column that will not convert is left exactly as it was found. A
    dtype nicety is not worth failing a project load over, and every reader of these two
    columns (`events.event_status`, `events.source_of`) goes through ``astype('string')``
    of its own anyway — the value of doing it HERE is that the LIVE frame the rest of
    the session shares is normal, not that any one reader depends on it.

    The column NAMES are imported, never re-typed: a second spelling is a second place to
    forget one. The import is function-local for the reason every other ``analysis``
    import in ``core`` is (see :mod:`nanoscope.core.history`) — persistence must not
    drag the analysis stack into its own import graph.
    """
    from ..analysis.events import REVIEW_COL, SOURCE_COL

    for col in (SOURCE_COL, REVIEW_COL):
        if col not in df.columns or df[col].dtype == "string":
            continue
        try:
            df[col] = df[col].astype("string")
        except (TypeError, ValueError):          # keep the values; a dtype is not worth a load
            pass
    return df


# ================================================================================================
# save
# ================================================================================================

def _default_app_version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return ""


@contextmanager
def _staged_write(target: Path, notes: list[str] | None = None) -> Iterator[BinaryIO]:
    """Yield a writable handle whose bytes land at ``target`` ONLY if the block succeeds.

    THIS IS THE INVARIANT, not a helper for one caller: no code in this module may
    open ``target`` for writing directly. Everything is staged in a temporary sibling
    (same directory, therefore the same filesystem, therefore :func:`os.replace` is a
    genuine atomic rename rather than a copy that can itself fail half-way) and moved
    into place as the last act. A caller cannot half-write the user's file because it
    never holds a handle to the user's file.

    THE RULE ITSELF NOW LIVES IN :mod:`nanoscope.core.atomic_write`, and this is a
    one-line pass-through kept for the callers in this module. It moved because the
    invariant was stated for "every file the app writes over user data" and implemented
    for exactly one of them: ``calibration.save_sidecar`` was a truncate-in-place
    ``write_text`` on the very file the sidecar confirmation exists to protect, and the
    training export, the measurement export and the figure writer were the same shape.
    Along the way it gained the three things this local copy got wrong — it turned a
    SYMLINK into a regular file (leaving the canonical project holding the previous
    save), destroyed extended attributes on every write, and could not stage a name of
    239 characters at all.
    """
    with atomic_write.staged_write(Path(target), notes=notes) as fh:
        yield fh


def save_project(path: str | Path, state: ProjectState,
                 notes: list[str] | None = None) -> Path:
    """Write ``state`` to a ``.nanoscope`` ZIP at ``path``; return the path written.

    Event tables and the measurements table go to parquet members; everything else
    (including JSON-coerced provenance) goes to ``manifest.json``.

    ATOMIC. The file at ``path`` is REPLACED, never edited in place: everything is
    written to a temporary sibling first and moved over the target with
    :func:`os.replace` only once the whole archive is on disk. If ANYTHING fails —
    a full disk on the second parquet member, an unserialisable frame, the process
    dying — the existing project at ``path`` is byte-for-byte untouched and the
    temporary is removed.

    This is not defensive decoration. ``zipfile.ZipFile(path, "w")`` truncates the
    target the instant it opens, so the pre-atomic version of this function destroyed
    a good project *before* it knew whether it could write a new one: an injected
    ENOSPC on the second member turned a 6949-byte project into a 3251-byte corpse
    that then failed to load with "missing its manifest.json". A save that fails must
    cost the user nothing.

    ``notes``, when a list is passed, collects sentences about what this write changed
    about the FILE — that a symlink was followed to its destination, or that the project
    had other hard-linked names which now hold the previous contents. An ordinary save
    adds nothing; see :func:`nanoscope.core.atomic_write.identity_notes`.

    PASSING IT IS NO LONGER WHAT MAKES THOSE SENTENCES EXIST. The seam computes them for
    every write, and a writer that has nowhere to put them leaves them in the module ledger
    (:func:`nanoscope.core.atomic_write.take_identity_notes`) instead of dropping them —
    this function was the only caller in the app that ever opted in, which is precisely how
    the promise ``docs/USER_GUIDE.md`` §11 makes for the project, the calibration sidecar,
    the exported events and the exported figures came to hold for the project alone. Passing
    ``notes`` here now means "this caller takes delivery": the sentences go to the list and
    are NOT also queued in the ledger, so they cannot be shown twice. They arrive only if the
    save LANDED — a save that raised replaced nothing, so no other name for the project went
    stale and the list stays empty, exactly as it does for an ordinary save.
    """
    path = Path(path)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "app_version": state.app_version or _default_app_version(),
        "documents": [
            {
                "path": d.path,
                "loaded_range": [int(d.loaded_range[0]), int(d.loaded_range[1])]
                if d.loaded_range is not None else None,
                # v4. Absent (never null-vs-missing ambiguous: both read as unknown).
                "source": d.source.to_json() if d.source is not None else None,
            }
            for d in state.documents
        ],
        "segments": [
            {
                "segment_uid": str(s.segment_uid),
                "parent_key": str(s.parent_key),
                "i0": int(s.i0),
                "i1": int(s.i1),
                "name": str(s.name),
                "t0_abs_s": float(s.t0_abs_s),
            }
            for s in state.segments
        ],
        "calibrations": [
            {
                "doc_id": c.doc_id,
                "channel": int(c.channel),
                "gain": float(c.gain),
                "offset": float(c.offset),
                "target_units": str(c.target_units),
                "provenance": str(c.provenance),
                # v5, and the three-state spelling is deliberate. An empty string means
                # "there is no note"; the key being ABSENT means "this file has no opinion",
                # which is what a v1–v4 project says and what a record that came from one
                # must keep saying. Writing "" for a None would turn "I don't know" into a
                # confident "there is none" the moment an old project was re-saved — and a
                # confident "there is none" is exactly what blanks the user's sidecar.
                **({} if c.note is None else {"note": str(c.note)}),
                # v8 — the user's "treat this channel as" override, three-state for the
                # same reason as `note` above: an ABSENT key is a project that predates
                # the field, and it must keep saying nothing rather than assert "no
                # override" over a correction the sidecar is still carrying.
                **({} if c.kind is None else {"kind": str(c.kind)}),
                # v7 — WHICH channel, not merely which position. Three-state like
                # `note`: an ABSENT key means the record came from a project that
                # did not know, and must keep saying so rather than asserting an
                # identity nobody recorded. See `calibration.resolve_channel`.
                **_identity_to_json(c),
            }
            for c in state.calibrations
        ],
        "layers": [
            {
                "doc_id": lay.doc_id,
                "channel": int(lay.channel),
                "color": str(lay.color),
                "visible": bool(lay.visible),
                "time_offset_s": float(lay.time_offset_s),
                "excluded": bool(lay.excluded),
                # v7 — WHICH channel, not merely which position. Three-state like
                # `note`: an ABSENT key means the record came from a project that
                # did not know, and must keep saying so rather than asserting an
                # identity nobody recorded. See `calibration.resolve_channel`.
                **_identity_to_json(lay),
            }
            for lay in state.layers
        ],
        "view_mode": str(state.view_mode),
        "results": [],
        "fits": [],
        "label_vocabulary": _json_safe(list(state.label_vocabulary or [])),
        # v9 — the colour identity map. `_json_safe` because a caller may have built the
        # rows from numpy-flavoured ints (a slot read back off a frame), and a numpy int64
        # is not JSON.
        "colours": _json_safe(list(state.colours or [])),
        "measurements": None,
    }

    with _staged_write(path, notes) as fh:
        with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, res in enumerate(state.results):
                member = f"results/{i}.parquet"
                zf.writestr(member, _df_to_parquet_bytes(res.events))
                manifest["results"].append(
                    {
                        "doc_id": res.doc_id,
                        "channel": int(res.channel),
                        "module_id": res.module_id,
                        # v10. Historical producer identity; never substitute the version
                        # installed at reopen time for the version that made these rows.
                        "module_version": str(res.module_version),
                        "events": member,
                        "provenance": _json_safe(res.provenance),
                        # v3. `params` is what makes a stored detection RE-RUNNABLE;
                        # `calibration_fingerprint` is what makes it AUDITABLE (see ResultRec).
                        "params": _json_safe(res.params or {}),
                        "calibration_fingerprint": _fingerprint_to_json(
                            res.calibration_fingerprint),
                        # v10. Kept top-level like AnalysisResult so staleness survives
                        # even for plugins/derived tables that did not duplicate it into
                        # EventSet provenance.
                        "baseline_fingerprint": res.baseline_fingerprint,
                        # Additive and optional, so no VERSION bump: a reader that does
                        # not know the key ignores it, and a project that lacks it
                        # restores as () exactly as every project did before.
                        "warnings": [str(w) for w in (res.warnings or ())],
                        # v7 -- see `_identity_to_json` and the note above VERSION.
                        **_identity_to_json(res),
                    }
                )

            # v6 -- derived fits. Additive on READ (a project without the key loads as []),
            # but the format VERSION still bumps, because an older build would silently
            # strip these on its next save. See the note above VERSION.
            for i, fit in enumerate(state.fits):
                entry: dict[str, Any] = {
                    "doc_id": fit.doc_id,
                    "channel": int(fit.channel),
                    "module_id": fit.module_id,
                    "params": _json_safe(fit.params or {}),
                    # BOTH staleness stamps -- a fit is measured against a calibration AND
                    # an open-pore baseline, and re-deriving either on load would clear a
                    # banner that is telling the truth. See FitRec / ResultRec.
                    "calibration_fingerprint": _fingerprint_to_json(
                        fit.calibration_fingerprint),
                    "baseline_fingerprint": fit.baseline_fingerprint,
                    "warnings": [str(w) for w in (fit.warnings or ())],
                    # v7 -- see `_identity_to_json` and the note above VERSION.
                    **_identity_to_json(fit),
                }
                if fit.table is not None:
                    member = f"fits/{i}.parquet"
                    zf.writestr(member, _df_to_parquet_bytes(fit.table))
                    entry["table"] = member
                if fit.curve is not None:
                    entry["curve"] = _json_safe(fit.curve)
                manifest["fits"].append(entry)

            if state.measurements is not None and len(state.measurements) > 0:
                member = "measurements.parquet"
                zf.writestr(member, _df_to_parquet_bytes(state.measurements))
                manifest["measurements"] = member

            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))

    return path


# ================================================================================================
# load
# ================================================================================================

def _check_version(manifest: dict[str, Any], name: str) -> None:
    """Reject anything that is not exactly this build's format+version, with a message
    that names the problem (not a project / missing / older / newer)."""
    fmt = manifest.get("format")
    if fmt != FORMAT:
        raise ProjectError(
            f"{name} is not a NanoScope project (format={fmt!r}, expected {FORMAT!r})."
        )
    version = manifest.get("version")
    if version is None:
        raise ProjectVersionError(
            f"{name} declares no format version and cannot be read."
        )
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProjectVersionError(
            f"{name} has a non-integer format version {version!r} and cannot be read."
        )
    # Read a compatible RANGE, not one exact version: this build writes VERSION but still reads
    # every project back to MIN_READ_VERSION (older formats are strict subsets — e.g. v1 simply
    # lacks the additive `segments` key).
    if version < MIN_READ_VERSION:
        raise ProjectVersionError(
            f"{name} is an older project (version {version}); this build reads versions "
            f"{MIN_READ_VERSION}–{VERSION}."
        )
    if version > VERSION:
        raise ProjectVersionError(
            f"{name} needs a newer NanoScope (project version {version} > {VERSION})."
        )


def _looks_like_zip(path: Path) -> bool:
    """Do the file's LEADING bytes carry the ZIP magic?

    ``zipfile.is_zipfile`` answers a different question — it looks for the end-of-central-
    directory record at the TAIL, which is precisely what an interrupted copy (USB pull,
    network drop, crash mid-transfer) destroys. Telling the two apart is the difference
    between "you opened the wrong file" and "your project is damaged — restore the backup":
    a file that BEGINS with ``PK\\x03\\x04`` (a local-file header, the first thing every
    save here writes) or ``PK\\x05\\x06`` (an empty archive) was an archive once, whatever
    happened to its tail.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return head in (b"PK\x03\x04", b"PK\x05\x06")


def load_project(path: str | Path) -> ProjectState:
    """Read a ``.nanoscope`` file back into a :class:`ProjectState`.

    Raises :class:`ProjectError` (or its :class:`ProjectVersionError` subclass, both
    ``ValueError``) if the file is not a project, is corrupt, or declares an
    unsupported version. EXACTLY that family, whatever the malformation: the refusal
    taxonomy is *not a project / damaged / missing parts / older / newer*, and every
    shape of damage — a truncated archive, a member whose CRC fails, a manifest section
    that is not shaped like one — must fold into it (see :func:`_manifest_section`, and
    the ``BadZipFile`` boundary below).
    """
    path = Path(path)
    if not path.exists():
        raise ProjectError(f"{path.name}: no such file.")
    if not zipfile.is_zipfile(path):
        # Two refusals, not one. `is_zipfile` failing only proves the archive's TAIL is
        # gone; a file that still LEADS with the ZIP magic is a project that got damaged
        # (interrupted copy, truncation), and telling its owner it "is not a NanoScope
        # project" steered them toward a wrong-file mixup when the honest remedies are
        # re-copying or restoring a backup. Measured: losing the last 200 bytes of a
        # 912 KB project — 99.98% intact — used to get the wrong-file sentence.
        if _looks_like_zip(path):
            raise ProjectError(
                f"{path.name} is damaged or truncated (it begins like a NanoScope project "
                f"but its archive is incomplete — a copy or save may have been "
                f"interrupted). Restore it from a backup, or copy the original again.")
        raise ProjectError(f"{path.name} is not a NanoScope project (not a ZIP archive).")

    try:
        state = _read_project(path)
    except (zipfile.BadZipFile, zlib.error) as exc:
        # The one-exception contract, kept at the boundary. `is_zipfile` passes any file
        # whose end-of-central-directory survives, so IN-PLACE corruption (a CRC-damaged
        # member, a broken deflate stream) surfaces later, from `zf.read` — as
        # `zipfile.BadZipFile` ("Bad CRC-32 for file 'results/0.parquet'") or
        # `zlib.error`, neither of which is the documented ProjectError. An API caller
        # catching ProjectError per the docs crashed, and the GUI's blanket handler
        # printed raw zip internals instead of the corrupt-project sentence.
        raise ProjectError(
            f"{path.name} is damaged: its contents failed the archive's integrity check "
            f"({exc}). Restore the project from a backup.") from exc
    return state


def _read_project(path: Path) -> ProjectState:
    """The body of :func:`load_project` — every zip read lives inside the caller's
    ``BadZipFile``/``zlib.error`` boundary."""
    with zipfile.ZipFile(path, "r") as zf:
        try:
            raw = zf.read(MANIFEST_NAME)
        except KeyError:
            raise ProjectError(
                f"{path.name} is missing its {MANIFEST_NAME}; it is not a NanoScope project."
            ) from None
        try:
            manifest = json.loads(raw)
        except ValueError as exc:
            raise ProjectError(f"{path.name}: {MANIFEST_NAME} is not valid JSON ({exc}).") from exc
        if not isinstance(manifest, dict):
            raise ProjectError(f"{path.name}: {MANIFEST_NAME} is not a JSON object.")

        _check_version(manifest, path.name)

        # Every section decodes inside `_manifest_section`, which is the one place a
        # malformed manifest becomes a ProjectError. Without it each expression below
        # escaped as whatever it happened to throw — KeyError('color'), TypeError on a
        # `loaded_range` written as a string, ValueError on a non-numeric channel — through
        # an API that documents exactly one exception type.
        with _manifest_section(path.name, "documents"):
            documents = [
                DocumentRef(
                    path=str(d["path"]),
                    loaded_range=(
                        (int(d["loaded_range"][0]), int(d["loaded_range"][1]))
                        if d.get("loaded_range") else None
                    ),
                    # v4, optional on read: a v1-v3 project has no stamp and gets None.
                    source=SourceFingerprint.from_json(d.get("source")),
                )
                for d in manifest.get("documents", [])
            ]
        # Additive + back-compatible: a version-1 project has no `segments` key, so this is [].
        with _manifest_section(path.name, "segments"):
            segments = [
                SegmentRec(
                    segment_uid=str(s["segment_uid"]),
                    parent_key=str(s["parent_key"]),
                    i0=int(s["i0"]),
                    i1=int(s["i1"]),
                    name=str(s.get("name", "")),
                    t0_abs_s=float(s.get("t0_abs_s", 0.0)),
                )
                for s in manifest.get("segments", [])
            ]
        with _manifest_section(path.name, "calibrations"):
            calibrations = [
                CalibrationRec(
                    doc_id=str(c["doc_id"]),
                    channel=int(c["channel"]),
                    gain=float(c["gain"]),
                    offset=float(c["offset"]),
                    target_units=str(c.get("target_units", "")),
                    provenance=str(c.get("provenance", "unknown")),
                    # None, NOT "" — a pre-v5 project genuinely does not know, and telling
                    # the two apart is what keeps the restore from blanking the sidecar's
                    # note.
                    note=None if c.get("note") is None else str(c["note"]),
                    kind=None if c.get("kind") is None else str(c["kind"]),
                    **_identity_from_json(c),
                )
                for c in manifest.get("calibrations", [])
            ]
        with _manifest_section(path.name, "layers"):
            layers = [
                LayerRec(
                    doc_id=str(lay["doc_id"]),
                    channel=int(lay["channel"]),
                    color=str(lay["color"]),
                    visible=bool(lay["visible"]),
                    time_offset_s=float(lay.get("time_offset_s", 0.0)),
                    excluded=bool(lay.get("excluded", False)),
                    **_identity_from_json(lay),
                )
                for lay in manifest.get("layers", [])
            ]

        results: list[ResultRec] = []
        with _manifest_section(path.name, "results"):
            for r in manifest.get("results", []):
                member = r.get("events")
                try:
                    # `_restore_event_extras` settles the dtype of the description
                    # columns and NOTHING else — a project written before they existed
                    # comes back without them, and reads "unknown". See that function.
                    df = (_restore_event_extras(_parquet_bytes_to_df(zf.read(member)))
                          if member else pd.DataFrame())
                except KeyError as exc:
                    raise ProjectError(
                        f"{path.name}: result table {member!r} is missing from the archive."
                    ) from exc
                results.append(
                    ResultRec(
                        doc_id=str(r["doc_id"]),
                        channel=int(r["channel"]),
                        module_id=str(r["module_id"]),
                        module_version=str(r.get("module_version", "")),
                        events=df,
                        provenance=dict(r.get("provenance") or {}),
                        # Both v3 keys are optional on read: a v1/v2 project has neither,
                        # and gets the same {} / None the pre-v3 code produced.
                        params=dict(r.get("params") or {}),
                        calibration_fingerprint=_fingerprint_from_json(
                            r.get("calibration_fingerprint")),
                        baseline_fingerprint=(
                            None if r.get("baseline_fingerprint") is None
                            else str(r["baseline_fingerprint"])),
                        warnings=tuple(str(w) for w in (r.get("warnings") or ())),
                        **_identity_from_json(r),
                    )
                )

        fits: list[FitRec] = []
        with _manifest_section(path.name, "fits"):
            for r in manifest.get("fits", []):
                member = r.get("table")
                try:
                    tbl = _parquet_bytes_to_df(zf.read(member)) if member else None
                except KeyError as exc:
                    raise ProjectError(
                        f"{path.name}: level table {member!r} is missing from the archive."
                    ) from exc
                curve = r.get("curve")
                fits.append(
                    FitRec(
                        doc_id=str(r["doc_id"]),
                        channel=int(r["channel"]),
                        module_id=str(r["module_id"]),
                        table=tbl,
                        curve=dict(curve) if curve else None,
                        params=dict(r.get("params") or {}),
                        calibration_fingerprint=_fingerprint_from_json(
                            r.get("calibration_fingerprint")),
                        baseline_fingerprint=(
                            None if r.get("baseline_fingerprint") is None
                            else str(r["baseline_fingerprint"])),
                        warnings=tuple(str(w) for w in (r.get("warnings") or ())),
                        **_identity_from_json(r),
                    )
                )

        measurements: pd.DataFrame | None = None
        member = manifest.get("measurements")
        if member:
            try:
                measurements = _parquet_bytes_to_df(zf.read(member))
            except KeyError as exc:
                raise ProjectError(
                    f"{path.name}: measurements table {member!r} is missing from the archive."
                ) from exc

    with _manifest_section(path.name, "view_mode / app_version"):
        return ProjectState(
            documents=documents,
            segments=segments,
            calibrations=calibrations,
            layers=layers,
            view_mode=str(manifest.get("view_mode", "overlay")),
            results=results,
            measurements=measurements,
            app_version=str(manifest.get("app_version", "")),
            fits=fits,
            label_vocabulary=[dict(r) for r in (manifest.get("label_vocabulary") or [])],
            # v9. Absent in v1-v8 → []. Rows are handed on verbatim; deciding which of them
            # is well-formed is `ColourRegistry.from_json`'s job, and it drops a bad row
            # rather than costing the project (this section is inside the same
            # `_manifest_section` guard as the rest, so a `colours` that is not even a list
            # still surfaces as ProjectError and not as a bare TypeError).
            colours=[dict(r) for r in (manifest.get("colours") or [])],
        )
