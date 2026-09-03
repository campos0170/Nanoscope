"""WHICH LOAD OF WHICH FILE — the one module that answers both halves of that question.

Three different identities were, until this module existed, one string (the resolved
source path). Conflating them is what produced three separate data-safety defects, and
they are here together because they are one question asked at three depths:

``doc_id`` — **this load, in this session.**
    A per-load token. Two *different* loads of the same file are two different
    documents, even though they share a path, and a result computed on one may never
    land on the other. Built by :func:`make_doc_id`, stamped once at
    :class:`~nanoscope.core.document.TraceDocument` construction, opaque everywhere
    else. NOT persistable: a fresh session re-loads and gets a fresh token.

``persistent_key`` — **the same thing, across sessions.**
    A file's resolved path, a segment's ``segment_uid`` — see
    :func:`nanoscope.core.segment.persistent_key`, which is where that lives because
    only it knows what a segment is. Results, calibrations, layers and manual
    measurements persist under THIS, never under ``doc_id``, so a reopened project
    re-attaches to the same recording. :func:`source_of` recovers it from a ``doc_id``.

:class:`SourceFingerprint` — **the bytes on disk.**
    A path is a *name*, not a recording. Re-exporting from Clampex over the same
    filename leaves the name identical and the contents completely different, and a
    project restore that checks only ``path.exists()`` will reattach five saved events to
    a 5,000-sample flat trace — all five past the end of it, event 0 reporting a 30.2 pA
    blockade in a signal whose whole standard deviation is 1.08 pA — and say nothing.
    This is what a project stamps so that it can notice.

WHY A PER-LOAD ``doc_id`` IS THE FIX AND NOT ONE MORE GUARD
-----------------------------------------------------------
Sixty call sites ask ``session.has_document(...)``; eight of them are a result sink
guarding on ``res.doc_id``, each with a comment explaining that a file's ``doc_id`` is its
path and a late result must therefore be dropped. Every one of them was RIGHT about a
closed recording and WRONG about a re-opened one: close a recording while a detection is
in flight, reopen it (at a different range, or after re-exporting the file), and the
check passes against a document the result was never computed from. Measured on a
re-opened 4,000-sample load of a 100,000-sample recording: all five events past the end
of the file, ``analyzed_duration_s`` 1.0 s against a true 0.04 s — the capture rate wrong
by 25x — and no staleness banner, because nothing about the calibration had changed.

``SourceFingerprint`` is the same lesson one level down. It is not a fourth guard bolted
onto the restore: it is the missing half of the identity a project was already storing.

Adding a ninth guard would have fixed one call site. Making the IDENTITY carry the load
fixes all eight at once and makes the ninth right before it is written: a result stamped
with a load token can only match the document that produced it, so every existing
``has_document`` check becomes exactly true without being edited.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "LOAD_SEP",
    "new_load_uid",
    "make_doc_id",
    "source_of",
    "same_source",
    "doc_id_for_meta",
    "SourceFingerprint",
    "SAMPLE_BLOCK_BYTES",
    "fingerprint_source",
    "describe_source_change",
    "describe_restored_source",
]

#: Separator between a document's source key and its per-load token. Appended LAST, so
#: :func:`source_of` splits on the last occurrence — and the tail must LOOK like a token
#: (:data:`LOAD_UID_HEX` hex digits) before it is treated as one, so a source path that
#: happens to contain this literal is returned whole rather than truncated.
LOAD_SEP: str = "#load-"

#: Hex digits in a load token. 12 (48 bits of ``os.urandom``) — a session opens tens of
#: recordings, not billions, so this is about being unmistakable, not about birthday bounds.
LOAD_UID_HEX: int = 12

#: Bytes read from each of the head / middle / tail of a source file for its content
#: digest. 256 KiB x 3 = 768 KiB regardless of file size. Measured on the owner's 606 MB
#: ``March_25_ATP...0005_Process.abf``, interleaved, min of 5: **0.49 ms** against
#: **0.298 s** for a whole-file SHA-256 — 611x — and that is a cost every project save AND
#: every project open would otherwise pay, per document.
SAMPLE_BLOCK_BYTES: int = 256 * 1024


def new_load_uid() -> str:
    """A fresh per-load token. Short, hex, and collision-free for the purpose.

    ``os.urandom`` rather than a counter: a counter would restart at 0 in the next
    process, and the whole point of this token is that it is never confused with the one
    a previous load carried. It deliberately does NOT encode the time or the path —
    nothing may ever be inferred from it, because the moment something is, the token has
    become a second (and wrong) persistent key.
    """
    return os.urandom(LOAD_UID_HEX // 2).hex()


def _is_load_uid(tail: str) -> bool:
    """Does ``tail`` look like a token this module minted? See :data:`LOAD_SEP`."""
    return len(tail) == LOAD_UID_HEX and all(c in "0123456789abcdef" for c in tail)


def make_doc_id(source_key: str, load_uid: str) -> str:
    """THE ONLY writer of a ``doc_id``. Opaque to every caller.

    ``load_uid`` empty means "not stamped", and the id degrades to the bare source key —
    the pre-load-identity behaviour. That case is reachable only from a
    :class:`~nanoscope.core.types.RecordingMeta` that never became a document (a
    ``probe()`` result, say), and never from anything the session holds:
    :class:`~nanoscope.core.document.TraceDocument` stamps at construction, which is the
    one funnel every recording — ABF, HDF5, TDMS, text, and a segment cut from any of
    them — passes through.
    """
    return f"{source_key}{LOAD_SEP}{load_uid}" if load_uid else str(source_key)


def source_of(doc_id: str) -> str:
    """THE ONLY reader of a ``doc_id``'s shape: the source key it was built from.

    For a file recording that is the resolved path; for a segment, its synthetic path
    (whose PERSISTENT key is the uid — see
    :func:`nanoscope.core.segment.persistent_key`, the one function that knows the
    difference). Returns ``doc_id`` unchanged when it carries no load token.

    Used where a *human* has to read the identity of a document that is no longer open
    (an export's ``source_file`` provenance, a restore warning), which is precisely
    where a raw per-load token would be noise.
    """
    head, sep, tail = str(doc_id).rpartition(LOAD_SEP)
    return head if sep and _is_load_uid(tail) else str(doc_id)


def same_source(a: Any, b: Any) -> bool:
    """Do these two source keys name the SAME recording? THE answer to that question.

    ``str(path.resolve())`` is what a source key is, and comparing two of them as strings
    is right for everything ``resolve()`` canonicalises — ``sub/../`` spellings and
    symlinks, both verified. It is WRONG for case, because ``Path.resolve()`` does not fold
    it, and the owner's disk (macOS's default APFS) is case-INSENSITIVE: measured, opening
    ``…/relocated_copy.abf`` and then ``…/Relocated_Copy.abf`` — one file, ``os.path.samefile``
    True — passed the "is this file already open?" guard twice. The recording was decoded and
    held in RAM a second time, two File Manager rows appeared with names differing only in
    case (identical when the case variant is in a parent DIRECTORY), and the two loads then
    carried two independent calibrations behind ONE ``.npcal.json``: one Apply wrote gain 9.0
    to the sidecar and left the other load at 3.0, so every ``depth_pA``, ECD and conductance
    read off it was 3x wrong with nothing on screen saying the two were one file.

    The robust test is the FILESYSTEM's own, not a string transformation: two names are the
    same file when they resolve to the same ``(st_dev, st_ino)`` — which is what
    ``os.path.samefile`` asks. Case folding by hand (``casefold``/``normcase``) would be a
    guess about the volume: a case-SENSITIVE volume mounted on the same machine genuinely
    has two files there, and folding would then merge two different recordings, which is the
    worse error. Asking the filesystem is right on both.

    Never raises and never says True on a guess: a key that names nothing on disk (a
    segment's synthetic path, a recording whose volume has gone away) falls back to the
    string compare, which is exactly the previous behaviour for those.
    """
    ka, kb = str(a), str(b)
    if ka == kb:
        return True
    try:
        return os.path.samefile(ka, kb)
    except (OSError, ValueError):
        # Not both readable — a synthetic segment path, a detached volume, a file that has
        # been deleted. "Cannot tell" is answered as "not the same", which is what the
        # string compare above already said.
        return False


def doc_id_for_meta(meta: Any) -> str:
    """The ``doc_id`` of the document ``meta`` belongs to.

    THE call every analysis module makes to stamp its result, and the reason it is a
    function rather than two lines in each module: ``str(ctx.meta.path.resolve())``
    appeared in :mod:`nanoscope.analysis.modules` and
    :mod:`nanoscope.analysis.ml_modules`, identically, and a third detector would have
    copied it and been wrong in the same way. An AST guard in the suite fails the build
    if that expression comes back.
    """
    return make_doc_id(str(meta.path.resolve()), str(getattr(meta, "load_uid", "") or ""))


# ================================================================================================
# The bytes on disk — what a project stamps so a re-export cannot pass for the original.
# ================================================================================================

@dataclass(frozen=True)
class SourceFingerprint:
    """What a source file WAS when a project was saved from it.

    ``n_samples`` is the count the app actually loaded and analysed (from the document,
    not re-derived from the file) — the strongest and cheapest check there is, and the
    one that speaks the user's language: "8 events against a 5,000-sample recording".
    ``size`` and ``digest`` describe the bytes. ``mtime_ns`` is carried for the
    MESSAGE only and is never a mismatch on its own: copying a file preserves it and
    re-saving one in a sync folder changes it, so a modification time that moved says
    nothing about whether the contents did.
    """

    n_samples: int = 0
    size: int = 0
    digest: str = ""
    mtime_ns: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"n_samples": int(self.n_samples), "size": int(self.size),
                "digest": str(self.digest), "mtime_ns": int(self.mtime_ns)}

    @classmethod
    def from_json(cls, raw: Any) -> "SourceFingerprint | None":
        """Parse a manifest entry; ``None`` for absent or unusable ones.

        A project written before this stamp existed simply has no entry, and a
        fingerprint that cannot be parsed must read as UNKNOWN rather than as a
        mismatch — an unreadable stamp is not evidence that the file changed, and
        raising a false alarm on every old project would teach the user to ignore the
        real one.
        """
        if not isinstance(raw, dict):
            return None
        try:
            return cls(n_samples=int(raw.get("n_samples", 0) or 0),
                       size=int(raw.get("size", 0) or 0),
                       digest=str(raw.get("digest", "") or ""),
                       mtime_ns=int(raw.get("mtime_ns", 0) or 0))
        except (TypeError, ValueError):
            return None


def fingerprint_source(path: str | Path, n_samples: int = 0) -> SourceFingerprint | None:
    """Fingerprint the file at ``path``; ``None`` when it cannot be read.

    The digest covers the file SIZE plus up to three :data:`SAMPLE_BLOCK_BYTES` blocks —
    head, middle and tail. Sampled rather than whole-file because this runs on every
    project save and every project open, and the owner's largest recording is 606 MB:
    whole-file SHA-256 is 0.298 s there against 0.49 ms for this, and no acquisition
    program rewrites a recording's head, middle AND tail to byte-identical content while
    changing anything that matters. A file smaller than one block is hashed entire.

    Note the strongest check is not this one: it is ``n_samples``, which costs nothing
    because the app already knows it, and which :func:`describe_source_change` reports
    first because it is the one a user can act on.

    Never raises: a source on a disconnected volume, or one the user cannot read, must
    leave the project openable. ``None`` means "unknown", which
    :func:`describe_source_change` treats as no evidence either way.
    """
    p = Path(path)
    try:
        st = p.stat()
        size = int(st.st_size)
        h = hashlib.sha256()
        h.update(str(size).encode("ascii"))
        with open(p, "rb") as fh:
            if size <= 3 * SAMPLE_BLOCK_BYTES:
                h.update(fh.read())
            else:
                for off in (0, (size - SAMPLE_BLOCK_BYTES) // 2, size - SAMPLE_BLOCK_BYTES):
                    fh.seek(off)
                    h.update(fh.read(SAMPLE_BLOCK_BYTES))
        return SourceFingerprint(n_samples=int(n_samples), size=size,
                                 digest=f"sha256:{h.hexdigest()[:32]}",
                                 mtime_ns=int(st.st_mtime_ns))
    except OSError:
        return None


def describe_source_change(saved: SourceFingerprint | None,
                           now: SourceFingerprint | None,
                           *, loaded_n_samples: int | None = None) -> str:
    """THE ONE writer of "this is not the file the project was saved from".

    Returns an empty string when there is nothing to say — which includes every case
    where we do not actually KNOW: a project written before this stamp existed, a source
    we could not read, a stamp we could not parse. Silence here means "no evidence",
    never "verified identical", and that distinction is the difference between a warning
    the user believes and one they learn to click past.

    ``loaded_n_samples`` is what the reopened document actually holds, compared against
    what the project recorded. It is checked separately from the digest because it
    catches the case the digest cannot: a file whose bytes we could not re-read, or one
    the reader now interprets differently.

    A changed modification time is NEVER a mismatch on its own — see
    :class:`SourceFingerprint` — but it is reported alongside a real one, because "and
    it was last written on Saturday" is often the sentence that makes the user remember
    what they did.
    """
    if saved is None:
        return ""
    reasons: list[str] = []

    if saved.n_samples and loaded_n_samples is not None \
            and int(loaded_n_samples) != int(saved.n_samples):
        reasons.append(f"it now holds {int(loaded_n_samples):,} samples where the project "
                       f"recorded {int(saved.n_samples):,}")

    if now is not None:
        if saved.size and now.size and now.size != saved.size:
            reasons.append(f"it is {now.size:,} bytes where the project recorded "
                           f"{saved.size:,}")
        elif saved.digest and now.digest and now.digest != saved.digest:
            reasons.append("its contents differ")

    if not reasons:
        return ""

    when = ""
    if now is not None and now.mtime_ns and now.mtime_ns != saved.mtime_ns:
        import datetime as _dt
        stamp = _dt.datetime.fromtimestamp(now.mtime_ns / 1e9)
        when = f" (last modified {stamp:%d %b %Y %H:%M})"
    return " and ".join(reasons) + when


def describe_restored_source(display_name: str,
                             saved: SourceFingerprint | None,
                             now: SourceFingerprint | None,
                             *,
                             loaded_n_samples: int | None = None,
                             restored_events: int = 0,
                             events_past_end: int = 0) -> str:
    """THE ONE writer of what a restore says about the recording it reattached to.

    :func:`describe_source_change` answers "did this file change?", and answers it only
    from EVIDENCE: no saved fingerprint means no evidence, and silence is the right
    reply. This function answers the question the user actually has after a restore --
    "do the events on my screen still fit the file they are drawn on?" -- and that one
    has an answer whether or not a fingerprint was ever written, because it is a
    comparison between the project's own saved event indices and the sample count of the
    recording now in hand.

    Splitting the two is the fix for a real silence. The consequence sentence used to be
    composed inside the fingerprint branch, so a project carrying no stamp -- every
    project written before version 4 (``project.MIN_READ_VERSION`` is frozen at 1 on
    purpose, "so every project the owner already has still opens"), and any project whose
    source could not be read at save time (:func:`fingerprint_source` returns ``None``
    for a disconnected volume by design) -- restored its whole saved event table onto a
    re-exported, shorter recording and said nothing at all. Measured: 11 events
    reattached, 9 of them lying entirely past the end of a 20,000-sample file, with the
    same project saved WITH a stamp naming those 9 in one sentence. Two surfaces, one
    fact, two different stories, decided by which version of the app wrote the file.

    So: the fingerprint half stays exactly as evidence-gated as it was (an unknown stamp
    still raises no "the file changed" claim -- a false alarm on every old project is how
    a real one gets ignored), and the events-past-the-end half is reported whenever it is
    TRUE, with the sentence saying plainly which of the two things is known.

    Args:
        display_name: The document's name, as the user sees it.
        saved: The fingerprint the project stored, or ``None`` when it stored none.
        now: The source's fingerprint re-read at restore time, or ``None`` when it could
            not be read. Callers may skip re-reading it when ``saved`` is ``None``:
            with nothing to compare against there is nothing it could establish.
        loaded_n_samples: What the reopened document actually holds.
        restored_events: How many events the restore reattached to this document.
        events_past_end: How many of those end past ``loaded_n_samples``. The caller
            counts them, because only the session knows which results belong to the
            document.

    Returns:
        The sentence for the restore banner, or ``""`` when there is nothing to say.
    """
    what = describe_source_change(saved, now, loaded_n_samples=loaded_n_samples)
    rows = int(restored_events)
    past = int(events_past_end)
    advice = ("Re-detect on it, or point the project at the original recording — the "
              "events on screen were measured on different data.")

    if what:
        # UNCHANGED, to the character: this is the sentence a stamped project already
        # produced, and a fix that reworded it would make two builds disagree about one
        # file for no reason.
        harm = ""
        if rows:
            harm = (f" Its {rows} restored event(s) were reattached anyway"
                    + (f", and {past} of them now lie past the end of this recording"
                       if past else "")
                    + ".")
        return (f"“{display_name}” is not the file this project was saved from: "
                f"{what}.{harm} {advice}")

    if not past:
        # Either nothing changed, or nothing is known and nothing landed wrong. Both are
        # silence, and silence still means "no evidence", never "verified identical".
        return ""

    where = ("" if loaded_n_samples is None
             else f" as it stands now ({int(loaded_n_samples):,} samples)")
    if saved is None:
        # No stamp: we cannot say WHAT changed about the file, and we do not pretend to.
        # What we can say is measured from live data, and it is the part that matters.
        why = ("This project carries no fingerprint of the recording it was saved from "
               "— it was written by an older version of NanoScope, or the file could "
               "not be read when it was saved — so the app cannot say what changed "
               "about the file, only that the events no longer fit it.")
    else:
        # A stamp that matched, and events that still do not fit: the file is not the
        # thing that moved, so do not let the sentence imply that it is.
        why = ("The recording still matches the fingerprint this project saved, so what "
               "does not fit is the project's own event table.")
    return (f"“{display_name}”: {past} of {rows} restored event(s) lie past the "
            f"end of this recording{where}. {why} {advice}")
