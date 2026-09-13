from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import numpy as np

from . import atomic_write
from .types import ChannelInfo, ChannelKind, Provenance

__all__ = [
    "ChannelCalibration",
    "CalibrationSet",
    "KIND_OVERRIDES",
    "channel_identity",
    "resolve_channel",
    "SIDECAR_VERSION",
    "SIDECAR_MIN_READ_VERSION",
    "SIDECAR_WRITTEN",
    "SIDECAR_REMOVED",
    "SIDECAR_NONE",
    "SIDECAR_FAILED",
    "SIDECAR_DELETE_FAILED",
    "SIDECAR_RECORDING_MOVED",
    "sidecar_path",
    "sidecar_effect",
    "notes_lost_by",
    "save_sidecar",
    "load_sidecar",
]


#: What a :attr:`ChannelCalibration.kind` override is allowed to say. Deliberately NOT every
#: :class:`ChannelKind`: ``RAW_VOLTAGE`` and ``UNKNOWN`` are what an undeclared channel already
#: IS, so overriding TO one of them says nothing the file did not already say, and an override
#: that can restate the default is an override a user can leave set without meaning anything.
KIND_OVERRIDES: tuple[ChannelKind, ...] = (
    ChannelKind.CURRENT, ChannelKind.VOLTAGE, ChannelKind.CONDUCTANCE)


@dataclass(frozen=True)
class ChannelCalibration:
    gain: float = 1.0
    offset: float = 0.0
    target_units: str = ""          # "" == display raw_units unchanged
    provenance: Provenance = Provenance.UNKNOWN
    note: str = ""
    kind: ChannelKind | None = None
    # ^^ *** WHAT THIS CHANNEL IS, when the file did not say (or said it wrong). *** ``None`` ==
    # no override: ``ChannelInfo.kind`` off the header stands, which is every recording that
    # declares its channels honestly.
    #
    # It lives HERE, beside target_units, because those two answer the same question about the
    # same channel and were answered from different places. ``target_units`` already lets a user
    # correct an undeclared channel to 'pA'; ``kind`` did not move with it, and the view reads
    # BOTH — the y-axis label from the calibration, the pane's order and its overlay grouping
    # from the header. The owner's opening question ("why does the app place the pA on the top
    # panel for some traces, and the voltage on the bottom yet reverses in others") is that
    # split: a channel relabelled to pA still ranked UNKNOWN and stayed BELOW the voltage pane,
    # so two recordings of the same experiment came up in opposite orders. And on his nisin TDMS
    # both channels are (RAW_VOLTAGE, 'V'), so in OVERLAY they grouped into ONE pane sharing one
    # y-axis with no way to separate a current from its command voltage. One override fixes both,
    # because the view derives the rank AND the grouping key from the same one answer.
    #
    # NOT part of :meth:`fingerprint`, and for the same reason ``target_units`` is not: this
    # changes no sample. It IS part of what a sidecar stores (see ``_sidecar_payload``) — a
    # correction the user had to make once must not have to be made again on every open.

    def __post_init__(self) -> None:
        """A gain/offset that would DESTROY the channel is refused HERE, at construction —
        not at each reader.

        ``apply()`` multiplies the whole channel by ``gain``, so a bad value turns every
        sample of a 180 MB recording into garbage, silently, forever. There are at least four
        ways in: a hand-edited ``.npcal.json`` (``json.loads`` happily accepts bare ``NaN`` and
        ``Infinity``), a project file, ``1e999`` typed into the Gain box — ``QDoubleValidator``
        accepts it and ``float()`` returns ``inf`` — and any future loader. Validating in the
        sidecar reader would have covered exactly one of them, which is why the check lives on
        the value object every one of them must build.

        TWO rules, and the second is not the first one with a wider net:

        * **non-finite** gain/offset — every calibrated sample becomes NaN/inf. Loud
          downstream: the PSD read-out says "unavailable", the non-finite gate rejects
          every event row.
        * **zero gain** — every calibrated sample becomes exactly ``offset``, a perfectly
          finite constant, and NOTHING downstream can tell that from a recording. Measured on
          the owner's 2,993,798-sample ABF through ``Session.calibrated``: ``gain=0`` gave
          ``non-finite 0/2993798``, ``meta.issues=()`` and **"Total RMS noise: 0 pA"** — the
          exact symptom C4 exists to prevent, and the most convincing-looking QC result a
          noise panel can produce. It is also irreversible: multiplying by zero does not
          scale a recording, it deletes it.

        The rule is ``gain == 0`` after the float32 cast ``apply()`` will make, NOT "gain is
        suspiciously small". A gain of 1e-30 is a strange unit choice and is honoured; a gain
        that UNDERFLOWS float32 to zero destroys the channel exactly as completely as a typed
        zero and is refused for the same reason, not for being small. A NEGATIVE gain is
        legitimate and stays legitimate — headstages invert, and a sign flip loses nothing.

        Raises ValueError; callers reading user files (the sidecar loader, project restore)
        already treat that as "malformed, ignore it and say so".
        """
        for name in ("gain", "offset"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"calibration {name} must be a number, got "
                                 f"{getattr(self, name)!r}") from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"calibration {name} must be finite, got {value!r} — a non-finite "
                    f"{name} makes every calibrated sample non-finite")
            object.__setattr__(self, name, value)   # normalise np.float64/str-free to float
        if float(np.float32(self.gain)) == 0.0:
            raise ValueError(
                f"calibration gain must not be zero (got {self.gain!r}) — every calibrated "
                f"sample would become exactly the offset, so the recording is deleted rather "
                f"than scaled, and no read-out can tell that flat line apart from a "
                f"perfectly quiet pore")
        # The kind override is NORMALISED here rather than trusted, on the same principle as
        # the two above: it arrives from a hand-editable `.npcal.json` and from a project file,
        # and every reader would otherwise have to know that it is sometimes a string. "" and
        # None both mean "no override" — a sidecar written by a build that has the field but
        # whose user never set it stores "", and that must not become an override to nothing.
        kind = self.kind
        if kind is None or kind == "":
            object.__setattr__(self, "kind", None)
        elif not isinstance(kind, ChannelKind):
            try:
                object.__setattr__(self, "kind", ChannelKind(str(kind)))
            except ValueError as exc:
                raise ValueError(
                    f"calibration kind must be one of "
                    f"{[k.value for k in ChannelKind]} or None, got {kind!r}") from exc

    @property
    def is_identity(self) -> bool:
        """Nothing about this channel has been overridden — display AND meaning.

        ``kind`` belongs in it for the same reason ``target_units`` does: this predicate is
        "would a user notice anything if this entry vanished", and a channel the user told the
        app to treat as current is one whose panes move when it does."""
        return (self.gain == 1.0 and self.offset == 0.0
                and not self.target_units and self.kind is None)

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """float32 in -> float32 out. np.float32 scalars are MANDATORY: a bare Python float
        silently upcasts the result to float64, doubling memory and violating Rule 5."""
        if self.gain == 1.0 and self.offset == 0.0:
            return raw
        out = raw * np.float32(self.gain) + np.float32(self.offset)
        assert out.dtype == np.float32
        return out

    def fingerprint(self) -> tuple[float, float]:
        """Cache-key component. Only gain/offset alter the ARRAY; target_units is a label."""
        return (self.gain, self.offset)


#: A raw mV voltage channel past this cannot be a bias someone applied to a nanopore -- the
#: field would break the membrane long before it. The owner's corpus separates cleanly either
#: side: inflated files read 4999-9999, direct files read 0.09-100.1, and nothing lies between.
#: Above it the channel is the 10x-inflated encoding; below it the number IS the bias.
_MAX_PLAUSIBLE_BIAS_MV: float = 1200.0


@dataclass
class CalibrationSet:
    """Keyed by (doc_id, channel_index). Owned by Session, NEVER by the document — that is what
    makes it JSON-persistable and what lets the 180 MB TraceDocument stay an immutable value.

    NOTE: no revision counter. The gain/offset fingerprint is a CONTENT key and strictly better:
    it is immune to A->B->A cycles that a counter would needlessly invalidate.
    """

    entries: dict[tuple[str, int], ChannelCalibration] = field(default_factory=dict)

    def get(self, doc_id: str, ch: int) -> ChannelCalibration:
        """Identity for an unseeded channel; Session seeds default_for() at add_document."""
        return self.entries.get((doc_id, ch), ChannelCalibration())

    def set(self, doc_id: str, ch: int, cal: ChannelCalibration) -> None:
        self.entries[(doc_id, ch)] = cal

    def fingerprint(self, doc_id: str, ch: int) -> tuple[float, float]:
        return self.get(doc_id, ch).fingerprint()

    def drop_doc(self, doc_id: str) -> None:
        for key in [k for k in self.entries if k[0] == doc_id]:
            del self.entries[key]

    @staticmethod
    def default_for(info: ChannelInfo, raw_median: float | None = None) -> ChannelCalibration:
        """Keyed off units+kind, NEVER off the literal name 'IN14'.

        [VERIFIED] on the owner's FERRITIN recordings ABF IN14 reads 10x too large: median
        -6998.90 mV on a -700 mV run, -9999 on -1000, -4999.1 on -500. Hence a 0.1 gain.

        BUT NOT EVERY ABF DOES THIS, and applying 0.1 blindly is wrong the other way. On the
        owner's HOST-GUEST files the same channel already reads the applied bias directly:
        75.07 mV on a 75 mV run, 100.1 on 100, 50.11 on 50. Scaling those by 0.1 reported
        7.507 mV for 75 mV -- which put the V-dependence x axis at 0-10 mV for a 0-100 mV
        series and made every conductance 10x too large (an apo file read G = 2.724 nS where
        the honest number is 0.273 nS).

        So the encoding is DETECTED, not assumed, from the channel's own magnitude: a value
        past :data:`_MAX_PLAUSIBLE_BIAS_MV` cannot be a bias anyone applied to a nanopore, so
        it must be the 10x-inflated form. Both branches stay FILE_META, both carry a note
        saying which was taken, and both remain user-overridable in the Calibration panel --
        the point is to stop guessing silently, not to start guessing better.

        ``raw_median`` is optional: with nothing to measure the old 0.1 default stands, so a
        caller that cannot reach the samples behaves exactly as before.
        """
        if info.kind is ChannelKind.VOLTAGE and info.raw_units == "mV":
            if raw_median is not None and abs(float(raw_median)) <= _MAX_PLAUSIBLE_BIAS_MV:
                return ChannelCalibration(
                    1.0, 0.0, "mV", Provenance.FILE_META,
                    f"voltage reads {raw_median:.3g} mV, a plausible applied bias, so it is "
                    f"taken as-is (no 10x correction); verify against your protocol")
            return ChannelCalibration(0.1, 0.0, "mV", Provenance.FILE_META,
                                      "ABF voltage is stored 10x too large; verify against your protocol")
        if info.kind is ChannelKind.RAW_VOLTAGE:
            return ChannelCalibration(1.0, 0.0, "", Provenance.UNKNOWN,
                                      "raw Volts — set a gain to convert to current")
        return ChannelCalibration()


# --- channel identity ----------------------------------------------------------------------------
# A CHANNEL INDEX IS A POSITION, NOT AN IDENTITY, and everything this app persists about a
# channel used to be filed under the position alone. Measured: re-export a recording with its
# signal columns in the other order (byte-identical file size, same sample count, same channel
# count — only the column order moved) and a `gain=0.5, offset=-1.0, units='pA'` calibration
# lands on the VOLTAGE channel, so the -600 mV trace reads -301.0 pA; the real current channel
# falls back to gain 1.0 / units ''; an 11-event detection is reattached to the voltage trace;
# and because the result's stored calibration fingerprint MATCHES the misplaced live
# calibration, the staleness system positively CERTIFIES those events as current-valid. On the
# `.npcal.json` path there is no project, no fingerprint and no banner: `meta.issues == ()`,
# zero warnings, wrong number on screen.
#
# So the identity travels with every persisted record, and this is the one function that reads
# it. It lives here rather than in `core/project.py` because the SIDECAR is the silent path and
# the sidecar is this module's: a resolver in the project layer would have left the worse half
# unguarded, which is the call-site-instead-of-invariant shape this project keeps repeating.

def channel_identity(info: ChannelInfo) -> tuple[str, str]:
    """``(name, kind)`` — what a stored record records about the channel it belongs to.

    Name AND kind, not either alone. The name is what tells two current channels apart
    ('IN15' vs 'IN0'); the kind is what tells a current channel from the voltage channel
    that took its index, which is the misattribution that actually costs a number.
    """
    return (str(info.name), info.kind.value)


def resolve_channel(
    channels: tuple[ChannelInfo, ...], index: int,
    name: str | None = None, kind: str | None = None,
) -> tuple[int | None, str | None]:
    """Which LIVE channel a record stored under ``index`` belongs to.

    Returns ``(resolved_index, what_changed)``. ``resolved_index`` is None when the record
    must NOT be applied; ``what_changed`` is a phrase for the caller's sentence, and is
    None on the ordinary case so an unchanged recording says nothing.

    Four answers, and the first two are the ones that must not move:

    * **No identity recorded** (``name is None``) — a project written before v7, or a
      version-1 sidecar. Behave EXACTLY as this app always has: the index if it is in
      range, nothing if it is not, and no note either way. ``MIN_READ_VERSION = 1`` is
      frozen and a warning on every old project is how a real one gets ignored.
    * **The channel at that index still has that identity** — the index, no note. This is
      every ordinary open, and it is bit-identical to the behaviour before this existed.
    * **Exactly one other channel carries the identity** — RELOCATE to it and say so. The
      record describes a channel that is still here; refusing would throw away a
      calibration whose subject is present, and guessing by position is what produced the
      defect.
    * **No channel carries it, or more than one does** — refuse, and say what was expected
      and what is there instead. The record is then kept in the project rather than
      applied (see ``MainWindow._capture_unrestorable``), so a refusal costs nothing
      permanent and a re-export that restores the channel restores the record with it.
    """
    n = len(channels)
    idx = int(index)
    if name is None and kind is None:
        return (idx if 0 <= idx < n else None), None
    want = (str(name or ""), str(kind or ""))
    here = [channel_identity(c) for c in channels]
    if 0 <= idx < n and here[idx] == want:
        return idx, None
    matches = [i for i, ident in enumerate(here) if ident == want]
    if len(matches) == 1:
        return matches[0], (
            f"{want[0]!r} ({want[1]}) is now channel {matches[0]}, not channel {idx}")
    now = (f"channel {idx} now holds {here[idx][0]!r} ({here[idx][1]})" if 0 <= idx < n
           else f"this recording has no channel {idx}")
    if len(matches) > 1:
        return None, (f"{len(matches)} channels are named {want[0]!r} ({want[1]}) and "
                      f"{now}, so which one this belongs to is not decidable")
    return None, f"no channel is named {want[0]!r} ({want[1]}) any more — {now}"


# --- sidecar persistence -------------------------------------------------------------------------
# <datafile>.npcal.json next to the data file. This is the payoff for splitting calibration out of
# the document. WRITE ONLY when at least one channel differs from CalibrationSet.default_for() —
# do not litter a sidecar next to every file the user merely opened.
# Format: {"version": 2, "channels": {"<index>": {"gain":…, "offset":…, "target_units":…,
#                                                 "note":…, "name":…, "kind":…,
#                                                 "treat_as_kind":…}}}
# A malformed/unreadable sidecar is IGNORED with a meta.issues entry. It never blocks a load.
#
# v2 ADDS "name"/"kind" — the identity of the channel the entry was measured on. An index is a
# POSITION: re-export the same recording with its signal columns swapped and a v1 sidecar
# silently installs a pA calibration on the voltage channel, with no project, no fingerprint
# and no banner anywhere to say so (see `resolve_channel`). Reading stays compatible back to
# SIDECAR_MIN_READ_VERSION: a v1 entry carries no identity, so it resolves by index exactly as
# it always did and raises no warning — the owner's own `.npcal.json` files are v1.
#
# "treat_as_kind" is THE USER'S OVERRIDE of what the channel is (ChannelCalibration.kind), and
# is NOT the same key as "kind": "kind" is the header's own declaration, recorded so the entry
# can be re-attached to the right channel after a re-export, and reusing it for the override
# would make `resolve_channel` hunt for a channel that never existed. Written only when set,
# absent everywhere else, and read as "no override" when absent — so every sidecar already on
# disk keeps its exact meaning.
#
# IT DOES NOT BUMP THE VERSION, and that is a deliberate difference from the v1->v2 bump. That
# bump existed to make an OLDER build REFUSE the file, because applying a v2 entry by index is
# how a pA calibration lands on a voltage trace. Ignoring "treat_as_kind" is not dangerous — an
# older build shows the panes in the order it always did — so raising the version would make it
# throw away a perfectly good gain and offset to avoid a pane ordering, which is a worse trade
# in exactly the direction this file cares about.

SIDECAR_VERSION: int = 2
#: Oldest sidecar this build still applies. v1 predates channel identity.
SIDECAR_MIN_READ_VERSION: int = 1


def sidecar_path(data_path: Path) -> Path:
    """'<datafile>.npcal.json' — the FULL name plus a suffix, not with_suffix().
    [VERIFIED] the corpus has '..._-600 mV.abf' and '..._-600 mV.csv' side by side; keying the
    sidecar on the stem alone would make them share (and corrupt) one calibration file."""
    return data_path.with_name(data_path.name + ".npcal.json")


def _sidecar_payload(doc_id: str, calib: CalibrationSet,
                     channels: tuple[ChannelInfo, ...]) -> dict[str, dict[str, object]]:
    """The channels that differ from their default — i.e. what a sidecar would contain.

    THE one place that determination lives. :func:`save_sidecar` acts on it and
    :func:`sidecar_effect` predicts it, so "what will this write do" and "what this write
    does" cannot drift apart.
    """
    payload: dict[str, dict[str, object]] = {}
    for info in channels:
        cal = calib.get(doc_id, info.index)
        default = CalibrationSet.default_for(info)
        # THE KIND OVERRIDE IS PART OF "DIFFERS FROM DEFAULT". Without it a user who only
        # tells the app that an undeclared channel is a current — no gain, no units — has
        # made a change that writes nothing, so the panes come back in the old order on the
        # next open and the correction has to be made again every time the file is opened.
        if cal.fingerprint() == default.fingerprint() \
                and cal.target_units == default.target_units \
                and cal.kind == default.kind:
            continue
        name, kind = channel_identity(info)
        payload[str(info.index)] = {
            "gain": cal.gain,
            "offset": cal.offset,
            "target_units": cal.target_units,
            "note": cal.note,
            # v2. WHICH channel this was measured on, so a re-export that reorders the
            # signal columns cannot silently move a pA calibration onto the voltage trace.
            "name": name,
            "kind": kind,
        }
        if cal.kind is not None:
            # WRITTEN ONLY WHEN SET, and under its own key — see the format note above. An
            # entry with no override is byte-identical to what this always wrote, so no
            # existing sidecar changes shape merely because this build can read one.
            payload[str(info.index)]["treat_as_kind"] = cal.kind.value
    return payload


# What a save_sidecar call did (or, from sidecar_effect, would do) to <datafile>.npcal.json.
SIDECAR_WRITTEN = "written"     # created or overwritten with the current calibration
SIDECAR_REMOVED = "removed"     # an EXISTING sidecar was deleted (everything is default again)
SIDECAR_NONE = "none"           # nothing on disk to change, and nothing to write
SIDECAR_FAILED = "failed"       # the WRITE was refused by the filesystem (nothing saved)
# The failed WRITE and the failed DELETE are opposite outcomes and MUST stay distinct: after a
# failed write there is no calibration next to the data and the session's values are lost on
# reopen; after a failed delete the OLD sidecar survives and its values RE-APPLY on reopen.
# One shared "failed" made the panel tell a user who had just confirmed "Reset and delete the
# file" that their calibration "will be gone when you reopen" — exactly backwards.
SIDECAR_DELETE_FAILED = "delete_failed"  # the removal was refused; the OLD sidecar survives
# The recording's file was beside the sidecar path when it was opened and is not there now:
# the user moved it behind the app's back. NOTHING is written or deleted at the old path --
# a sidecar beside nothing is not a save, and an orphan there would re-apply to whatever is
# later dropped into that folder under that name. The calibration stays live in the session
# (and in a project); the caller must SAY it was not stored beside the data. Decided by
# ``Session.recording_moved``, which is the only place that knows the file was ever there.
SIDECAR_RECORDING_MOVED = "recording_moved"


def sidecar_effect(data_path: Path, doc_id: str, calib: CalibrationSet,
                   channels: tuple[ChannelInfo, ...]) -> str:
    """What :func:`save_sidecar` WOULD do right now, without doing it.

    Exists so a caller can ask before it commits — specifically so the app can confirm
    before ``SIDECAR_REMOVED``. That branch deletes a file the app does not own, sitting
    next to the user's raw data, carrying a ``note`` field a project file cannot round-trip;
    it is reached by any commit that leaves every channel at its default, which the
    Calibration panel's *Reset to default* button does by definition and its *Apply* button
    does whenever the typed values happen to be the defaults. Silent deletion from either
    was C12.
    """
    if _sidecar_payload(doc_id, calib, channels):
        return SIDECAR_WRITTEN
    return SIDECAR_REMOVED if sidecar_path(data_path).exists() else SIDECAR_NONE


def notes_lost_by(data_path: Path, doc_id: str, calib: CalibrationSet,
                  channels: tuple[ChannelInfo, ...]) -> tuple[tuple[int, str], ...]:
    """``((channel, note), …)`` that are on disk now and would NOT survive this write.

    :func:`sidecar_effect` answers a question about the FILE, and the confirmation built on
    it therefore only fires when the whole file goes. The thing that cannot be recreated is
    finer than the file: it is one channel's ``note``. Measured on the tree as found — two
    calibrated channels, a note on the current one, *Reset to default* on that channel —
    the sidecar SURVIVED (the other channel still differs from its default), the entry and
    its note were dropped from it, ``sidecar_effect`` said ``SIDECAR_WRITTEN``, no
    confirmation fired, and the status line read *"Applied · saved to pore.csv.npcal.json"*,
    which is what a successful save looks like.

    So the question a caller has to be able to ask is "what does this write DESTROY", and
    this is that question. Note that it reads the FILE, not the in-memory set: the note
    matters because it is in the user's own file, and a note this session never loaded is
    exactly the one nobody can retype. It is derived from :func:`_sidecar_payload`, the same
    one place the write is derived from, so the prediction cannot drift from the act.

    A note that is being REPLACED by another note is not a loss and is not reported; an
    empty note is not a loss either, and neither is a channel whose entry survives.
    """
    path = sidecar_path(data_path)
    try:
        existing = json.loads(path.read_text(), parse_constant=_reject_json_constant)
        on_disk = existing["channels"]
    except (OSError, ValueError, KeyError, TypeError):
        return ()                      # unreadable or absent: there is nothing to lose here
    if not isinstance(on_disk, dict):
        return ()
    after = _sidecar_payload(doc_id, calib, channels)
    lost: list[tuple[int, str]] = []
    for key, entry in on_disk.items():
        note = str(entry.get("note", "")) if isinstance(entry, dict) else ""
        if not note.strip():
            continue
        kept = after.get(str(key))
        if kept is not None and str(kept.get("note", "")).strip():
            continue                   # the entry survives and still carries a note
        try:
            lost.append((int(key), note))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(lost))


def save_sidecar(data_path: Path, doc_id: str, calib: CalibrationSet,
                 channels: tuple[ChannelInfo, ...], reason: list[str] | None = None) -> str:
    """Write the sidecar iff at least one channel differs from its default; otherwise remove any
    stale sidecar so that resetting every channel truly reverts to the defaults on next open.
    Write failures are swallowed: a read-only data directory must never break calibration.

    Returns what it did (:data:`SIDECAR_WRITTEN` / :data:`SIDECAR_REMOVED` /
    :data:`SIDECAR_NONE` / :data:`SIDECAR_FAILED` / :data:`SIDECAR_DELETE_FAILED`). The
    return value is not decoration: a removal destroys a user-owned file, and a caller that
    cannot see that happened cannot tell the user about it. Callers who need to ask FIRST
    use :func:`sidecar_effect`.

    A refused WRITE and a refused DELETE are reported as different outcomes because their
    consequences are opposite. :data:`SIDECAR_FAILED` means nothing was saved — the session's
    calibration is gone on reopen. :data:`SIDECAR_DELETE_FAILED` means the OLD sidecar could
    not be removed — it survives next to the data and its values re-apply on reopen, which is
    the reverse of what the user who confirmed the deletion was trying to achieve. Collapsing
    both into one code forced the caller's message to be wrong for one of them.

    ``reason``, when a list is passed, collects the OS's own words for either failure. The
    panel used to report every failure as "(the folder is not
    writable)", which is one of at least five things an ``OSError`` here means — a full
    disk, a read-only file, a disconnected network volume, a name too long, a permission
    denied — and telling a user to check a permission that is already correct sends them
    looking in the wrong place.
    """
    payload = _sidecar_payload(doc_id, calib, channels)
    path = sidecar_path(data_path)
    if not payload:
        # DELETE branch: every channel is back at its default, so any stale sidecar must go.
        # Its failure is its own outcome (SIDECAR_DELETE_FAILED, never SIDECAR_FAILED): the
        # file that could not be unlinked still holds the OLD calibration and will re-apply
        # it on next open — the opposite of a failed write, where nothing is saved at all.
        try:
            existed = path.exists()
            path.unlink(missing_ok=True)
            return SIDECAR_REMOVED if existed else SIDECAR_NONE
        except OSError as exc:
            if reason is not None:
                reason.append(str(exc) if isinstance(exc, PermissionError)
                              else f"{exc.strerror or exc}".strip())
            return SIDECAR_DELETE_FAILED
    try:
        # STAGED, not written in place. PORT_PLAN rule 21 claimed every write over user data
        # was staged and this one — on the exact file the removal confirmation exists to
        # protect — was a truncate-in-place `write_text`: a full disk or a crash between the
        # truncate and the last byte left a user-owned calibration file that no longer parsed,
        # and `load_sidecar` would then discard the whole thing with an issues line. See
        # nanoscope.core.atomic_write.
        blob = json.dumps({"version": SIDECAR_VERSION, "channels": payload}, indent=2)
        with atomic_write.staged_write(path) as fh:
            fh.write(blob.encode("utf-8"))
        return SIDECAR_WRITTEN
    except OSError as exc:
        if reason is not None:
            reason.append(str(exc) if isinstance(exc, PermissionError)
                          else f"{exc.strerror or exc}".strip())
        return SIDECAR_FAILED


def _reject_json_constant(token: str) -> NoReturn:
    """``json.loads`` hook that refuses the non-standard bare ``NaN``/``Infinity`` literals.

    Python's json accepts all three by default and hands back a float, so a sidecar reading
    ``{"gain": NaN}`` parsed cleanly and installed a NaN gain. They are not JSON (RFC 8259 has
    no such literals), so refusing them is a correctness fix, not a policy choice — and it
    catches the value before it can reach anything, rather than after.
    """
    raise ValueError(f"{token} is not valid JSON and is not a usable calibration value")


def load_sidecar(data_path: Path, doc_id: str, calib: CalibrationSet,
                 channels: tuple[ChannelInfo, ...] = ()) -> list[str]:
    """Apply '<datafile>.npcal.json' over the already-seeded defaults. Returns meta.issues
    entries; a malformed sidecar is skipped, never raised — it must not block a load.

    ``channels`` is the recording's real channel list. When given, a sidecar entry whose key
    does not name one of them is REJECTED with an issue instead of being installed: the file
    is hand-editable, and ``"-1"`` or ``"99"`` used to load silently into a CalibrationSet
    keyed by an index no channel has — invisible, unreachable from the Calibration panel, and
    (for a negative key) capable of colliding with a real channel through Python's negative
    indexing elsewhere. Passing nothing keeps the old, unvalidated behaviour for callers that
    genuinely do not know the channel list; :meth:`Session.add_document` always knows it.

    Non-finite gain/offset is refused by :class:`ChannelCalibration` itself — see its
    ``__post_init__`` — and surfaces here as a per-channel issue.

    v2 entries also name the channel they were measured on, and :func:`resolve_channel`
    decides where they land: unchanged recordings are unaffected, a reordered one has its
    entries re-attached by identity (said out loud), and an entry whose channel is gone is
    NOT applied — because installing it by position is how a pA calibration ended up on the
    voltage trace reading -301 pA with ``meta.issues == ()``. A v1 sidecar carries no
    identity and resolves by index exactly as before, silently.

    ``treat_as_kind`` — the user's "this channel is really a current" override — is OPTIONAL
    and absent from every sidecar written before it existed, which reads back as no override.
    It is deliberately a different key from the v2 ``kind``; see the format note above.
    """
    path = sidecar_path(data_path)
    if not path.exists():
        return []
    try:
        blob = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    except (OSError, ValueError) as exc:
        return [f"{path.name} could not be read and was ignored ({exc})."]

    if not isinstance(blob, dict):
        return [f"{path.name} must contain a JSON object and was ignored."]
    version = blob.get("version")
    if (not isinstance(version, int)
            or isinstance(version, bool)
            or not (SIDECAR_MIN_READ_VERSION <= version <= SIDECAR_VERSION)):
        return [f"{path.name} has unsupported version {version!r} and was ignored."]
    raw_channels = blob.get("channels")
    if not isinstance(raw_channels, dict):
        return [f"{path.name} has no 'channels' object and was ignored."]

    known = {int(info.index) for info in channels}
    issues: list[str] = []
    for key, entry in raw_channels.items():
        moved = None
        try:
            ch = int(key)
        except (TypeError, ValueError) as exc:
            issues.append(f"{path.name}: channel {key!r} entry is malformed and was "
                          f"ignored ({exc}).")
            continue
        # WHERE the entry lands. `resolve_channel` is the sole authority when the caller
        # knows the channel list (Session.add_document always does): a v2 entry names the
        # channel it was measured ON and is re-attached by that or refused, and a v1 entry
        # carries no identity, so it resolves by index and is refused when out of range —
        # exactly the C8 guard this replaces, with the same reach. An empty `channels`
        # means the caller genuinely does not know, and the old unvalidated behaviour stands.
        if channels:
            target, moved = resolve_channel(
                channels, ch,
                entry.get("name") if isinstance(entry, dict) else None,
                entry.get("kind") if isinstance(entry, dict) else None)
            if target is None:
                why = moved or f"this recording has no channel {ch} (it has {sorted(known)})"
                issues.append(
                    f"{path.name}: the calibration saved for channel {ch} was NOT applied "
                    f"— {why}. Nothing on this recording is calibrated by it; re-apply it "
                    f"to the right channel if it still belongs here.")
                continue
            ch = target
        try:
            cal = ChannelCalibration(
                gain=float(entry["gain"]),
                offset=float(entry["offset"]),
                target_units=str(entry.get("target_units", "")),
                # The sidecar records the user's own edits; it declares no measurement or header
                # evidence, so its provenance is UNKNOWN rather than a flattering MEASURED.
                provenance=Provenance.UNKNOWN,
                note=str(entry.get("note", "")),
                # "treat_as_kind", NOT "kind" — the latter is the header's own declaration,
                # recorded so `resolve_channel` can re-attach the entry, and it is present in
                # every v2 sidecar this build has ever written. Absent (every sidecar written
                # before this field existed) reads back as None: no override, header stands.
                # ChannelCalibration normalises the string and refuses one that names no kind.
                kind=entry.get("treat_as_kind"),
            )
        except (TypeError, ValueError, KeyError) as exc:
            issues.append(f"{path.name}: channel {key!r} entry is malformed and was ignored ({exc}).")
            continue
        if moved is not None:
            issues.append(
                f"{path.name}: the calibration saved for channel {key} was applied to "
                f"channel {ch} — {moved}. The signal columns of this recording have "
                f"moved since it was written.")
        calib.set(doc_id, ch, cal)
    return issues
