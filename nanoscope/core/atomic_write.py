"""THE door every write over user data goes through.

PORT_PLAN rule 21 states the guarantee this module owns:

    *No write may destroy what it is replacing until the replacement exists.*

That rule was implemented once, inside ``core/project._staged_write``, and the plan's own
sentence — "every file the app writes over user data" — was false the moment it was written:
``calibration.save_sidecar`` was a ``path.write_text`` (truncate in place) **on the exact file
the sidecar-deletion confirmation exists to protect**, and ``scoring.write_training_export``,
``analysis.measure.export_measurements`` and ``plot.figures.save_figure`` were the same shape.
Fixing the project alone is the shape this project has watched come back three times, so the
guarantee moved here, where every writer has to pass, and a test (``test_persist_invariant``)
fails the build if any module under ``nanoscope/`` writes a path it did not stage.

THREE THINGS ARE GUARANTEED, and they are three different failures.

1. **A write never half-lands.** Bytes are staged in a temporary sibling and moved into place
   with a single :func:`os.replace`. This is the C13 rule and it is unchanged.

2. **A write never changes WHICH FILE it wrote** (§2.7). ``os.replace`` installs a new inode,
   so a naive stage-and-replace silently:

   * turned a **symlink** into a regular file — measured: ``link.is_symlink()`` False, the real
     file's inode unchanged, and the canonical project still holding the OLD content, so the
     user's next open read the save they thought they had just made;
   * destroyed **extended attributes** (Finder tags, ``user.note``) on *every* save —
     measured ``['com.apple.provenance', 'user.note']`` → ``['com.apple.provenance']``;
   * raised ``OSError[63] File name too long`` for a name ≥ 239 characters, naming a hidden
     ``.tmp`` path the user has never seen and cannot act on.

   :func:`resolve_write_target` and :func:`dress_like` close those. What ``os.replace``
   *cannot* preserve is a **hard link**: an atomic replace is a new inode by definition, so a
   second name for the same file necessarily goes stale. Writing in place is the only
   alternative and it is precisely the destructive path guarantee 1 exists to forbid, so the
   trade is deliberate — and instead of being silent it is REPORTED, through ``notes``.

   That report is now produced BY DEFAULT, at every seam here, not only for a caller that
   remembered to ask: it was an opt-in, exactly one writer in the app
   (``project.save_project``) opted in, and so the guarantee held for the project and
   silently failed for every export. See :func:`take_identity_notes`.

   Three rules keep that channel honest, and they are the rules a disclosure channel needs
   to *be* one rather than a bucket of sentences:

   * **A note is only produced for a write that happened.** The sentences are computed
     before the replace (``st_nlink`` describes the file being replaced) and published
     after it. A save that fails, a staged set that is abandoned, a member withdrawn by
     :meth:`StagedSet.unstage` — none of them changed any file's identity, and none of them
     leaves a sentence behind saying they did.
   * **A note knows which file it is about.** Ledger entries are ``(target, sentence)`` and
     :func:`take_identity_notes` takes a ``for_paths`` scope, so an export prints only the
     sentences for the files it actually wrote and never one left over from another
     document's save.
   * **Nothing is dropped in silence.** The ledger is capped; when the cap throws pending
     sentences away, the next drain says how many and that they cannot be recovered.

3. **A SET of files never half-lands** (§2.10). One figure export writes twelve files under one
   ``figures_*`` stem. Nine failing left three new files beside nine old ones with no marker,
   disclosed only by a 12-second status line — a folder of mixed vintages that looks exactly
   like a folder of one export. :class:`StagedSet` stages every member first and commits only
   when all of them staged, and it restores what it replaced if a commit fails part way.

4. **A failure names the file the USER chose.** Because staging happens in a hidden sibling,
   a folder the process cannot write into fails *at the temporary*, and the raw ``OSError``
   the GUI prints verbatim read ``Permission denied:
   '/vol/share/.study.nanoscope.saving-83391-1f1bd2c4.tmp'`` — a path that does not exist,
   that the user never chose, and that gives them nothing to act on. That is the harm
   :func:`temp_sibling` already names; closing the one errno it was measured on
   (ENAMETOOLONG) left every other one leaking. :func:`restate_staging_failure` re-addresses
   the failure to the real file, with the real reason and an action, and keeps the original
   as ``__cause__`` for whoever is debugging the app rather than using it.

WHAT THIS MODULE DOES NOT DO. It does not decide *whether* to write, does not ask the user
anything, and does not know what a project or a figure is. It is a filesystem primitive, and
being Qt-free and app-free is what lets every writer in the app use it.
"""

from __future__ import annotations

import errno as _errno
import os
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

__all__ = [
    "NAME_MAX_FALLBACK",
    "TEMP_SUFFIX",
    "resolve_write_target",
    "temp_sibling",
    "dress_like",
    "refuse_if_write_protected",
    "identity_notes",
    "take_identity_notes",
    "peek_identity_notes",
    "staged_write",
    "staged_path",
    "StagedSet",
    "staged_write_set",
]

# Every filesystem this app is likely to meet allows at least 255 bytes per component; used
# only when pathconf cannot answer (a path whose directory does not exist yet, a mount that
# refuses the query).
NAME_MAX_FALLBACK: int = 255

TEMP_SUFFIX: str = ".tmp"


# ================================================================================================
# which file are we actually replacing
# ================================================================================================

def resolve_write_target(target: str | Path) -> tuple[Path, Path | None]:
    """``(the file to replace, the symlink the caller named)``.

    A user who keeps ``~/analysis/current.nanoscope`` as a symlink into a dated folder means
    "write the file this points at". A stage-and-replace against the *link* path instead
    unlinks the link and drops a regular file in its place: the link is gone, the real file
    still holds the previous save, and nothing says so. Resolving first makes the symlink
    behave the way every editor on the machine already behaves.

    ``os.path.realpath`` and not ``Path.resolve``: both resolve, but realpath is defined on a
    path whose final component does not exist yet (the first save of a new project), which
    ``Path.resolve(strict=True)`` is not.

    A BROKEN symlink resolves to its (missing) destination and is written there, which is
    also what an editor does — the link starts working again.

    Returns the resolved path plus the link the caller named, or ``None`` when the caller
    named the file itself (the overwhelmingly common case, and the one that costs nothing).
    The second value reports only a link at the FINAL component — a symlinked *directory*
    somewhere above (``/tmp`` → ``/private/tmp`` on every Mac) is resolved silently, because
    the file the user named and the file that gets written are the same file.
    """
    named = Path(target)
    real = Path(os.path.realpath(named))
    followed = named if named.is_symlink() else None
    return real, followed


def _name_max(directory: Path) -> int:
    try:
        value = int(os.pathconf(str(directory), "PC_NAME_MAX"))
    except (OSError, ValueError, AttributeError, KeyError):
        return NAME_MAX_FALLBACK
    return value if value > 0 else NAME_MAX_FALLBACK


def temp_sibling(target: Path, *, tag: str = "saving", suffix: str | None = None) -> Path:
    """A staging path in ``target``'s own directory — therefore the same filesystem, therefore
    :func:`os.replace` is a genuine atomic rename and not a copy that can itself fail half way.

    The name is per-process and per-call (``pid`` + 8 random hex) so two NanoScopes saving the
    same project cannot stage into one temp and interleave their bytes.

    THE STEM IS TRUNCATED to fit ``PC_NAME_MAX``. Decorating a 239-character filename produced
    a 267-character temp and ``OSError[63] File name too long``, and the message named the
    hidden ``.tmp`` path — a file the user has never seen, cannot find and did not create, for
    a save they had every reason to expect to work. The random suffix carries the uniqueness,
    so shortening the stem costs nothing; the leading characters are kept because that is what
    identifies the temp if one is ever left behind.
    """
    directory = target.parent
    decoration = f".{tag}-{os.getpid()}-{uuid.uuid4().hex[:8]}{suffix or TEMP_SUFFIX}"
    budget = _name_max(directory) - len(decoration.encode()) - 1   # -1 for the leading dot
    stem = target.name
    if budget < 1:
        stem = ""
    elif len(stem.encode()) > budget:
        # Truncate on a BYTE budget without splitting a character.
        stem = stem.encode()[:budget].decode("utf-8", "ignore")
    return directory / f".{stem}{decoration}"


# ================================================================================================
# crash-orphaned temps: the residue of the atomicity guarantee, swept and DISCLOSED
# ================================================================================================

#: What a :func:`temp_sibling` name looks like from the outside: ``.<stem>.<tag>-<pid>-
#: <8 hex><suffix>``. The pid is captured because it is the liveness test: a temp whose
#: writing process is gone can never be finished, renamed into place, or cleaned by the
#: exception handlers that normally remove it — it is debris by construction.
_ORPHAN_RE = re.compile(
    r"\A\..*\.(?:saving|replacing)-(\d+)-[0-9a-f]{8}\.[^.]+\Z")


def _pid_alive(pid: int) -> bool:
    """Is a process with this pid still running? Errs on the side of ALIVE: a temp we are
    not sure about is left alone, because deleting an in-flight staging file would break
    the very write it belongs to."""
    if pid <= 0:                      # 0 would signal our whole process group; never probe it
        return True
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:                   # EPERM: exists, owned by someone else — alive
        return True
    except OverflowError:             # a pid this OS cannot represent — unknowable, keep it
        return True
    return True


def _sweep_stale_temps(directory: Path) -> list[str]:
    """Remove crash-orphaned staging temps in ``directory`` and RETURN the sentences
    saying so — one disclosure channel entry per sweep, through the same ledger the
    identity notes use.

    THE HARM THIS CLOSES: a SIGKILL / power loss / force-quit mid-save leaves the hidden
    staging temp behind (measured: one partial ``.events.csv.saving-*`` of 35 MB; 464 MB
    total after ten killed exports), and nothing in the app ever removed or mentioned
    them — the next write in the same folder succeeded with an empty disclosure ledger,
    while the temps sat invisibly (dotfiles) next to the user's raw data, growing on
    every crash forever. ``temp_sibling``'s own docstring concedes the identifiability
    ("that is what identifies the temp if one is ever left behind"); this is the code
    that finally acts on it, at the seam every writer already passes through.

    SAFETY RULE: only a temp whose embedded pid is DEAD is touched. A live pid — this
    process included — may still be mid-write; its temp is that write's own business.
    A dead pid's temp is unfinishable: no exception handler will ever reach it, and its
    contents are an incomplete copy of a write that never landed (the real file was
    never touched — that is the atomicity guarantee working).

    A removal that itself fails (the folder went read-only since) is NOT reported as a
    removal: the sentence then says the file is still there and where, so the user can
    act on it — a claim of success is made only on success.
    """
    removed: list[str] = []
    stuck: list[str] = []
    freed = 0
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return []
    for entry in entries:
        match = _ORPHAN_RE.match(entry.name)
        if match is None:
            continue
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            if _pid_alive(int(match.group(1))):
                continue
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        try:
            os.unlink(entry.path)
        except OSError:
            stuck.append(entry.name)
            continue
        removed.append(entry.name)
        freed += size
    sentences: list[str] = []
    if removed:
        shown = ", ".join(sorted(removed)[:3])
        more = f" and {len(removed) - 3} more" if len(removed) > 3 else ""
        sentences.append(
            f"{len(removed)} leftover staging file(s) from an interrupted NanoScope save "
            f"or export were found in {directory} and removed ({shown}{more}; "
            f"{freed:,} bytes). They were incomplete hidden working copies from a NanoScope "
            f"that did not shut down cleanly — the files they were staging were never "
            f"touched, so nothing of yours was lost.")
    if stuck:
        sentences.append(
            f"{len(stuck)} leftover staging file(s) from an interrupted NanoScope save or "
            f"export are still in {directory} and could not be removed "
            f"({', '.join(sorted(stuck)[:3])}). They are incomplete hidden working copies "
            f"and safe to delete by hand.")
    return sentences


# ================================================================================================
# a failure names the file the USER chose, never the staging file
# ================================================================================================

#: The errno values that mean "this folder will not take a new file from you". Kept small and
#: explicit: everything else (ENOSPC, EDQUOT, EIO, ENAMETOOLONG…) gets the plain restatement,
#: because "remove the write protection" would be the wrong advice for a full disk.
_DENIED = frozenset(
    e for e in (getattr(_errno, name, None) for name in ("EACCES", "EPERM", "EROFS"))
    if e is not None)


def _names(exc: OSError, path: Path) -> bool:
    """Does ``exc`` point the user at ``path``? Both the structured filename slots and the
    rendered text, because an exception raised by somebody else's library (``savefig`` on the
    staged path) may carry the name only in its message."""
    text = str(path)
    for candidate in (getattr(exc, "filename", None), getattr(exc, "filename2", None)):
        if candidate is not None and str(candidate) == text:
            return True
    return text in str(exc)


def restate_staging_failure(exc: BaseException, staged: Path,
                            target: Path) -> BaseException | None:
    """The same failure, addressed to the file the user actually named — or ``None`` when
    the exception never mentioned the staging file and is already about something the user
    can find.

    THE HARM THIS CLOSES is the one :func:`temp_sibling` already names as the thing the
    staging design must not inflict: "the message named the hidden ``.tmp`` path — a file
    the user has never seen, cannot find and did not create, for a save they had every
    reason to expect to work." Shortening the temp NAME fixed the one errno that was
    measured (ENAMETOOLONG) and left every other one leaking: a save or an export into a
    folder the process cannot write fails at the temporary, so the raw ``OSError`` the GUI
    prints verbatim reads ``Permission denied: '/vol/share/.study.nanoscope.saving-83391-
    1f1bd2c4.tmp'`` — a path that does not exist, for a file the user never chose.

    The replacement follows the sentence this module already owns for the neighbouring case
    (:func:`refuse_if_write_protected`): the real file, the real reason, and an action. The
    ORIGINAL is kept as ``__cause__`` by every raise site, so a traceback or a log still
    holds the errno and the staging path for whoever is debugging the app rather than using
    it.

    The exception TYPE is preserved (``PermissionError`` stays a ``PermissionError``) because
    callers catch by type; only the sentence changes.
    """
    if not isinstance(exc, OSError) or not _names(exc, staged):
        return None
    reason = exc.strerror or exc.__class__.__name__
    if exc.errno in _DENIED:
        message = (
            f"{target.name} was not written. Its folder {target.parent} cannot be written "
            f"to ({reason}), so nothing could be staged there for {target}. Nothing was "
            f"changed — save under a different folder, or remove the write protection on "
            f"{target.parent}.")
    else:
        # No advice here, deliberately: ENOSPC, EIO and a name too long each need a
        # different one, and a guess would send the user looking in the wrong place — the
        # mistake `calibration.save_sidecar` already documents. It says WHAT and WHERE, and
        # it does not claim the target "is as it was" (a first save has no target yet).
        message = f"{target} was not written: {reason}. Nothing was changed."
    try:
        return type(exc)(message)
    except Exception:       # pragma: no cover - an OSError subclass with an odd __init__
        return OSError(message)


# ================================================================================================
# carrying the replaced file's identity onto its replacement
# ================================================================================================

def _darwin_xattr():
    """``(list, get, set)`` bound to libc's xattr calls, or ``None``.

    macOS has extended attributes but CPython exposes ``os.*xattr`` only where the Linux
    signatures apply, and ``shutil.copystat`` therefore copies nothing here — which is why
    Finder tags and ``user.note`` were being destroyed on every save with no code in sight
    doing it. Bound lazily and cached on the function so a platform without libc, or a build
    without these symbols, degrades to "attributes not copied" rather than to an import error.
    """
    cached = getattr(_darwin_xattr, "_cached", "unset")
    if cached != "unset":
        return cached
    result = None
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            libc.listxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                       ctypes.c_size_t, ctypes.c_int]
            libc.listxattr.restype = ctypes.c_ssize_t
            libc.getxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                                      ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
            libc.getxattr.restype = ctypes.c_ssize_t
            libc.setxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                                      ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
            libc.setxattr.restype = ctypes.c_int

            def _list(path: Path) -> list[bytes]:
                raw = str(path).encode()
                size = libc.listxattr(raw, None, 0, 0)
                if size <= 0:
                    return []
                buf = ctypes.create_string_buffer(size)
                size = libc.listxattr(raw, buf, size, 0)
                if size <= 0:
                    return []
                return [n for n in buf.raw[:size].split(b"\x00") if n]

            def _get(path: Path, name: bytes) -> bytes | None:
                raw = str(path).encode()
                size = libc.getxattr(raw, name, None, 0, 0, 0)
                if size < 0:
                    return None
                buf = ctypes.create_string_buffer(size) if size else ctypes.create_string_buffer(1)
                got = libc.getxattr(raw, name, buf, size, 0, 0)
                return b"" if got <= 0 else buf.raw[:got]

            def _set(path: Path, name: bytes, value: bytes) -> bool:
                return libc.setxattr(str(path).encode(), name, value, len(value), 0, 0) == 0

            result = (_list, _get, _set)
        except Exception:       # pragma: no cover - no libc / missing symbols
            result = None
    _darwin_xattr._cached = result           # type: ignore[attr-defined]
    return result


def _copy_xattrs(src: Path, dst: Path) -> int:
    """Copy every extended attribute of ``src`` onto ``dst``; return how many landed.

    Per-attribute failures are swallowed on purpose: some names are owned by the system and
    refuse to be set by a normal process (``com.apple.provenance`` on a sandboxed volume),
    and a Finder tag that could not be carried must not turn a working save into a failed one.
    """
    landed = 0
    if hasattr(os, "listxattr"):                     # Linux
        for name in os.listxattr(str(src), follow_symlinks=True):
            try:
                os.setxattr(str(dst), name, os.getxattr(str(src), name), follow_symlinks=True)
                landed += 1
            except OSError:
                pass
        return landed
    fns = _darwin_xattr()
    if fns is None:
        return 0
    lst, get, setx = fns
    for name in lst(src):
        try:
            value = get(src, name)
            if value is not None and setx(dst, name, value):
                landed += 1
        except Exception:
            pass
    return landed


def dress_like(staged: Path, target: Path) -> None:
    """Give ``staged`` the identity ``target`` already has, BEFORE the replace.

    Permissions, ownership and extended attributes are applied to the temporary while it is
    still a temporary, so the file that lands is fully dressed the instant it becomes visible
    under the user's name — there is no window in which the project exists with the process
    umask's permissions or with a Finder tag missing.

    Every step is individually best-effort. ``chown`` needs privileges the app does not have
    outside a shared-volume setup; an xattr can be refused by name. None of that may turn a
    save that would otherwise work into one that fails, which is the same reasoning
    ``save_sidecar`` already applies to a read-only data directory.

    Does nothing when ``target`` does not exist — a first save has no identity to inherit,
    and the process umask is then the right answer.
    """
    try:
        st = target.stat()
    except OSError:
        return
    try:
        os.chmod(staged, st.st_mode & 0o7777)
    except OSError:
        pass
    try:
        if (st.st_uid, st.st_gid) != (os.getuid(), os.getgid()):
            os.chown(staged, st.st_uid, st.st_gid)
    except (OSError, AttributeError):
        pass
    _copy_xattrs(target, staged)


def refuse_if_write_protected(target: Path) -> None:
    """Honour a read-only file. Raises :class:`PermissionError`; silent otherwise.

    ``os.replace`` needs write permission on the DIRECTORY and pays no attention at all to
    the file's own mode, so a project the user deliberately ``chmod 400``-ed was measured
    being overwritten without a word — the read-only bit was decorative. A rule that only
    the truncating write it replaced happened to enforce.

    Checked at the target, not at the temporary, and only when the target already exists:
    a first save has nothing to protect.
    """
    if target.exists() and not os.access(target, os.W_OK):
        raise PermissionError(
            f"{target.name} is marked read-only and was not changed. Remove the write "
            f"protection on {target}, or save under a different name.")


def identity_notes(target: Path, followed: Path | None) -> list[str]:
    """What this write is about to change about the FILE, in the user's words.

    Only the things an atomic replace cannot preserve, and only when they are actually
    present, so an ordinary save says nothing at all. Silence here means "nothing unusual",
    never "not checked" — the same rule the project's source-fingerprint warning follows.
    """
    notes: list[str] = []
    if followed is not None:
        notes.append(
            f"“{followed.name}” is a link to {target}; that file was written, and the link "
            f"still points at it.")
    try:
        links = target.stat().st_nlink
    except OSError:
        links = 1
    if links > 1:
        notes.append(
            f"{target.name} had {links - 1} other name(s) on this disk (hard links). Saving "
            f"replaces the file rather than editing it in place, so those other names still "
            f"hold the PREVIOUS contents. Delete and re-create them if you rely on them.")
    return notes


#: Every identity note this process has produced and nobody has taken yet, newest last.
#:
#: THE DISCLOSURE IS NOW A DEFAULT, NOT AN OPT-IN. :func:`identity_notes` used to be computed
#: only for a caller that passed a ``notes`` list, and ``core.project.save_project`` was the
#: only one in the app that did — so the promise this module's docstring makes ("instead of
#: being silent it is REPORTED, through ``notes``") and the one ``docs/USER_GUIDE.md`` §11
#: makes for "the project, the calibration sidecar, exported events, exported figures" held
#: for exactly one of that list. A hard-linked ``events.csv`` re-exported over went stale in
#: the manuscript folder with no dialog, no status clause and nothing anywhere that could say
#: so afterwards, while the identical situation on a ``.nanoscope`` produced a full dialog in
#: the same session.
#:
#: Every seam in this module now records unconditionally, so a writer inherits the disclosure
#: by passing through the door rather than by remembering to ask for it. ``notes=`` still
#: works and is still the direct channel for a caller that has a place to put the sentences
#: (the project save shows them in a dialog); the ledger is for the writers that do not.
#:
#: EACH ENTRY CARRIES THE FILE IT IS ABOUT, ``(target, sentence)``. A ledger of bare sentences
#: is a shared bucket: the app writes a calibration sidecar, a project and four figure files
#: in one session, so the next surface to drain it would show the user a sentence about
#: ``rec.csv.npcal.json`` underneath "Wrote 4 figure(s)" — a note that does not describe the
#: thing it is printed beside. :func:`take_identity_notes` therefore takes a ``for_paths``
#: scope, and a surface that knows which files it just wrote (all of them do — the export
#: commands hold the paths they passed to the seam) drains exactly those.
_identity_ledger: list[tuple[Path, str]] = []

#: The ledger is a disclosure channel, not a log: it holds what has not been shown yet, and a
#: process that never drains it must not grow without bound. Oldest sentences fall off first.
LEDGER_MAX: int = 64

#: How many pending sentences the cap has thrown away before anything showed them. Dropping
#: them SILENTLY would be the defect this whole channel exists to close, one level up: the
#: fact that a hard link was broken is not recoverable after the fact (an atomic replace
#: leaves no trace of the link), so a discarded disclosure is a disclosure lost for good.
#: Counted here and said out loud by the next drain.
_identity_discarded: int = 0


def _discard_note(count: int) -> str:
    """The sentence for ``count`` disclosures the cap threw away before anyone read them."""
    return (f"{count} earlier file-identity note(s) about other writes in this session were "
            f"discarded before anything showed them — more than {LEDGER_MAX} went unread. "
            f"What they said cannot be recovered: replacing a file leaves no trace of a hard "
            f"link it broke or a symlink it followed.")


def _queue_identity(entries: list[tuple[Path, str]]) -> None:
    """Put entries nobody has taken in the ledger, capped, and COUNT what the cap drops."""
    global _identity_discarded
    _identity_ledger.extend(entries)
    overflow = len(_identity_ledger) - LEDGER_MAX
    if overflow > 0:
        del _identity_ledger[:overflow]
        _identity_discarded += overflow


def _deliver_identity(target: Path, sentences: list[str],
                      notes: list[str] | None) -> list[str]:
    """Put the sentences for a write that HAS NOW HAPPENED where they can be found.

    Called after the ``os.replace``, never before it. The sentences have to be COMPUTED
    first — ``st_nlink`` describes the file being replaced, and after the replace the target
    is a brand-new inode with one name — but a write that then fails changed nothing about
    any file's identity, and "those other names still hold the PREVIOUS contents" printed for
    a save that did not happen is a note about a write the app never made.

    ONE PLACE, NEVER TWO. A caller that passed a ``notes`` list has taken delivery — it has
    somewhere to put the sentences and it is going to — so those notes do not also go in the
    ledger, or the next surface to drain it would repeat a disclosure the user has already
    been shown, attached to a different file. The ledger holds exactly what nobody has taken.

    A note is never dropped for being a duplicate of one already waiting: two re-exports over
    the same hard-linked file are two facts about two writes. The ledger is capped instead —
    and the cap is not silent (:data:`_identity_discarded`)."""
    if not sentences:
        return sentences
    if notes is not None:
        notes.extend(sentences)
        return sentences
    _queue_identity([(target, note) for note in sentences])
    return sentences


def _scope(for_paths) -> set[Path] | None:
    """The resolved paths a drain is scoped to, or ``None`` for "everything pending".

    Resolved through :func:`resolve_write_target` because that is the identity the ledger
    recorded: a caller that wrote through ``current.nanoscope`` (a symlink) has an entry
    filed under the destination, and asking for the name it used must still find it."""
    if for_paths is None:
        return None
    # One path, passed bare, is the obvious spelling for a surface that wrote one file --
    # and iterating a ``str`` would quietly scope the drain to its CHARACTERS, matching
    # nothing and returning silence that looks exactly like "nothing to disclose".
    if isinstance(for_paths, (str, Path)):
        for_paths = [for_paths]
    return {resolve_write_target(p)[0] for p in for_paths}


def take_identity_notes(for_paths=None) -> list[str]:
    """Remove and return the identity notes for the writes named by ``for_paths``.

    For a surface that has just finished a write and is about to tell the user what
    happened: whatever this returns is what that write changed about the FILE (a symlink
    followed to its destination, hard-linked names left holding the previous contents) and
    is not obtainable anywhere else afterwards — an atomic replace leaves no trace of the
    link it broke.

    ``for_paths`` is the files this surface is about to talk about; anything pending for a
    different file is LEFT pending, so an export never prints a sentence about a file it did
    not write. Omitting it drains everything, which is right only for a surface that speaks
    for the whole session.

    Draining is the point: a note that has been shown must not be shown again beside the
    next export. If the cap discarded pending notes since the last drain, the count of them
    comes back FIRST — a lost disclosure is disclosed, not swallowed."""
    global _identity_discarded
    wanted = _scope(for_paths)
    taken: list[str] = []
    kept: list[tuple[Path, str]] = []
    for target, note in _identity_ledger:
        if wanted is None or target in wanted:
            taken.append(note)
        else:
            kept.append((target, note))
    _identity_ledger[:] = kept
    if _identity_discarded:
        taken.insert(0, _discard_note(_identity_discarded))
        _identity_discarded = 0
    return taken


def peek_identity_notes(for_paths=None) -> tuple[str, ...]:
    """What :func:`take_identity_notes` would return, without draining it (tests, asserts)."""
    wanted = _scope(for_paths)
    pending = [note for target, note in _identity_ledger
               if wanted is None or target in wanted]
    if _identity_discarded:
        pending.insert(0, _discard_note(_identity_discarded))
    return tuple(pending)


# ================================================================================================
# one file
# ================================================================================================

@contextmanager
def staged_write(target: str | Path, *, notes: list[str] | None = None) -> Iterator[BinaryIO]:
    """Yield a writable BINARY handle whose bytes land at ``target`` only if the block succeeds.

    On ANY exception — including :class:`KeyboardInterrupt` — the temporary is removed and
    ``target`` is left exactly as it was. A caller cannot half-write the user's file because
    it never holds a handle to the user's file.

    Durability is best-effort and deliberately separate from atomicity: the temp is fsynced
    before the rename so a crash cannot expose a rename of unflushed bytes, but a filesystem
    that refuses fsync (some network mounts) must not turn a working save into a failed one.

    ``notes``, when given, collects the sentences from :func:`identity_notes` — the things
    this write changes about the file that the caller may want to tell the user. It is no
    longer what DECIDES whether they are produced: they are always produced, and a caller
    that passes nothing can still find them through :func:`take_identity_notes`.

    One deliberate behaviour, inherited from the project save this generalises: staging needs
    write permission on the DIRECTORY, not just on the file, so a writable file inside a
    read-only folder now fails instead of being rewritten in place. That failure costs the
    user nothing — the existing file is untouched — whereas the in-place write it replaces is
    the destructive path, and least recoverable exactly there. It is reported by the name the
    user chose, never by the staging file's (:func:`restate_staging_failure`).
    """
    real, followed = resolve_write_target(target)
    refuse_if_write_protected(real)
    # Debris from an earlier CRASHED write in this folder is removed now and said out
    # loud with this write's own disclosures. Computed before staging so the sweep can
    # never mistake this write's brand-new temp for an orphan (the pid guard already
    # prevents that; the ordering makes it obvious).
    swept = _sweep_stale_temps(real.parent)
    # Read the identity of the file we are about to replace WHILE it is still that file; say
    # it only once the replace has happened (:func:`_deliver_identity`).
    pending = identity_notes(real, followed)
    tmp = temp_sibling(real)
    try:
        with open(tmp, "wb") as fh:
            yield fh
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        dress_like(tmp, real)
        os.replace(tmp, real)
    except BaseException as exc:
        _discard(tmp)
        # The sweep HAPPENED even though this write did not; its sentences go to the
        # ledger rather than dying with the failed write — a removal is never silent.
        if swept:
            _queue_identity([(real, s) for s in swept])
        restated = restate_staging_failure(exc, tmp, real)
        if restated is None:
            raise
        raise restated from exc
    _deliver_identity(real, swept + pending, notes)


@contextmanager
def staged_path(target: str | Path, *, notes: list[str] | None = None) -> Iterator[Path]:
    """Like :func:`staged_write` but yields the temporary PATH instead of a handle.

    For writers that insist on opening the file themselves — ``DataFrame.to_csv``,
    ``Figure.savefig``, ``DataFrame.to_parquet``. They get a path they may do anything to;
    what lands at ``target`` is whatever is at that path when the block ends.

    The temporary keeps ``target``'s SUFFIX. That is not cosmetic: matplotlib picks its
    backend from the extension, and ``to_parquet``/``to_csv`` are chosen by the caller but
    read back by extension elsewhere. A ``.tmp`` handed to ``savefig`` raises rather than
    writing a PDF.
    """
    real, followed = resolve_write_target(target)
    refuse_if_write_protected(real)
    swept = _sweep_stale_temps(real.parent)       # crash debris out, and said — see there
    pending = identity_notes(real, followed)      # before; delivered after the replace
    tmp = temp_sibling(real, suffix=real.suffix or TEMP_SUFFIX)
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(
                f"nothing was written to the staged path for {real.name}")
        dress_like(tmp, real)
        os.replace(tmp, real)
    except BaseException as exc:
        _discard(tmp)
        # The sweep HAPPENED even though this write did not — its sentences must not
        # die with the failed write.
        if swept:
            _queue_identity([(real, s) for s in swept])
        # The block that failed is the CALLER's -- ``savefig``/``to_csv`` opened the staged
        # path itself, so its OSError carries the staging name and nothing else does.
        restated = restate_staging_failure(exc, tmp, real)
        if restated is None:
            raise
        raise restated from exc
    _deliver_identity(real, swept + pending, notes)


def _discard(tmp: Path) -> None:
    try:
        os.unlink(tmp)
    except OSError:
        pass


# ================================================================================================
# a set of files
# ================================================================================================

class StagedSet:
    """A group of files that must land together or not at all (§2.10).

    The failure this exists for: one *Export figures* writes twelve files under one
    ``figures_*`` stem. When nine of them failed, the folder held three files from today and
    nine from a previous export, under one name, with no marker — a mixed-vintage folder that
    is indistinguishable from a consistent one, disclosed only by a status line that clears
    after twelve seconds. A figure and the ``.csv`` of numbers behind it disagreeing is worse
    than either being missing.

    Usage::

        with staged_write_set() as ss:
            for path, payload in items:
                write(payload, ss.stage(path))       # a temp path, per member
        # everything landed here, or nothing did

    Stage everything, then commit. A member that raises while being written aborts the whole
    set before a single file has been replaced. If the COMMIT itself fails part way — a
    permission change between staging and replacing — every member already replaced is
    restored from a copy taken before its replacement, so the set still does not split.

    NOT a general transaction: it does not lock, and another process writing the same files
    concurrently will still interleave. It closes the failure that was measured, which is one
    process's own partial write.
    """

    def __init__(self, notes: list[str] | None = None) -> None:
        self._staged: list[tuple[Path, Path]] = []      # (temp, real target)
        self._notes = notes
        #: Identity sentences for members that are STAGED but not yet placed, held here
        #: until :meth:`commit` actually replaces them. A set exists precisely because it
        #: may end in nothing being written, and "events.csv had 1 other name(s) on this
        #: disk... those other names still hold the PREVIOUS contents" describes a write
        #: that HAPPENED — published from `stage` it would be a note about a file the
        #: export never touched, printed beside a message saying nothing was written.
        #: :meth:`unstage` withdraws a member's sentences with the member; :meth:`abandon`
        #: drops all of them; :meth:`commit` publishes exactly the ones it placed.
        self._pending_identity: list[tuple[Path, str]] = []
        #: Directories THIS set created, innermost last. `stage` has to make the parent
        #: before it can write a temp file into it, and that happens before anything can
        #: fail — so an abandoned set used to leave a brand-new empty folder behind, named
        #: after an export that never happened. `abandon` removes them, innermost first,
        #: and only while they are still empty (another process, or a later export to the
        #: same folder, always wins).
        self._made_dirs: list[Path] = []
        #: Sentences from :func:`_sweep_stale_temps` — crash debris this set found and
        #: removed on its way in, filed under the member that led it there. Kept APART
        #: from ``_pending_identity`` because their lifetime differs: an identity note
        #: describes a write and dies with an abandoned set, while a sweep describes
        #: removals that HAPPENED regardless of how the set ends — `commit` delivers
        #: them with its own disclosures and `abandon` queues them in the ledger.
        self._swept_notes: list[tuple[Path, str]] = []
        #: Directories already swept by THIS set, so a twelve-member figure export scans
        #: its one folder once, not twelve times.
        self._swept_dirs: set[Path] = set()

    @property
    def targets(self) -> tuple[Path, ...]:
        return tuple(target for _tmp, target in self._staged)

    def stage(self, target: str | Path) -> Path:
        """Reserve ``target`` and return the temporary path to write in its place.

        THE STAGING PATH IS PROVEN WRITABLE BEFORE IT IS HANDED OUT. A set member is written
        by the CALLER (``savefig`` opens the path itself), outside any context manager this
        class can wrap — so a folder the process cannot write into surfaced as the caller's
        own ``OSError`` naming the hidden staging file, the exact message
        :func:`restate_staging_failure` exists to prevent, from the one seam that could not
        catch it afterwards. Creating and immediately removing the temp turns that into a
        refusal from ``stage`` itself, addressed to the file the user chose, before a single
        member of the set has been written.

        The probe is removed again: :meth:`commit` reads "the temp does not exist" as "this
        member was never written", and a leftover probe would make a forgotten member look
        finished.
        """
        real, followed = resolve_write_target(target)
        refuse_if_write_protected(real)
        # Remember what we create, outermost first, so `abandon` can undo exactly it.
        missing = []
        probe = real.parent
        while not probe.exists() and probe != probe.parent:
            missing.append(probe)
            probe = probe.parent
        try:
            real.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Only a DENIAL is re-addressed here. The path a failing mkdir names is a real
            # directory the user can go and look at, so for anything else (a file sitting
            # where the folder should be, a name too long) its own message is the more
            # useful one and restating it would throw away the detail.
            restated = (restate_staging_failure(exc, real.parent, real)
                        if exc.errno in _DENIED else None)
            if restated is None:
                raise
            raise restated from exc
        self._made_dirs.extend(reversed(missing))
        # Crash debris from an EARLIER NanoScope in this folder is removed now and the
        # sentence held for whichever ending this set gets (see ``_swept_notes``). A
        # directory this set just created cannot hold any, and each folder is swept once.
        if real.parent not in self._swept_dirs:
            self._swept_dirs.add(real.parent)
            self._swept_notes.extend(
                (real, note) for note in _sweep_stale_temps(real.parent))
        # Computed NOW -- `st_nlink` has to be read while the file being replaced is still
        # the file being replaced -- but published only by `commit`. See
        # ``_pending_identity``.
        self._pending_identity.extend(
            (real, note) for note in identity_notes(real, followed))
        tmp = temp_sibling(real, suffix=real.suffix or TEMP_SUFFIX)
        try:
            with open(tmp, "xb"):
                pass
            os.unlink(tmp)
        except OSError as exc:
            _discard(tmp)
            restated = restate_staging_failure(exc, tmp, real)
            if restated is None:
                raise
            raise restated from exc
        self._staged.append((tmp, real))
        return tmp

    def unstage(self, targets) -> None:
        """Withdraw the named ``targets`` from the set: their temps are deleted and they
        are neither committed nor counted. The rest of the set is untouched.

        The failure this exists for: a caller staging SEVERAL independent families into one set
        (Export figures) hits an error while WRITING one family — the temp was reserved
        by :meth:`stage` but never (fully) written. Leaving those members in place
        poisons the whole set: :meth:`commit` refuses over the never-written temp
        ("staged file(s) were never written") and the healthy families are thrown away
        with it, misreported as a placement failure. The failing family withdraws
        exactly its own members — never a partial subset of them, or the family splits
        into mixed vintages, the very thing the set exists to prevent.

        Directories created for a withdrawn member are left to :meth:`abandon` /
        :meth:`commit` exactly as before: a later commit means the folder now holds the
        surviving export, and a full abandon still removes what stayed empty.

        Targets are matched by the same identity :meth:`stage` recorded — the RESOLVED
        path from :func:`resolve_write_target` — not by the spelling the caller used.
        ``stage`` stores ``os.path.realpath(target)``, so a member staged through a
        symlinked directory (``/tmp`` → ``/private/tmp``, ``/var`` → ``/private/var`` on
        every Mac) lives in the set under its real path; comparing the caller's bare
        spelling against it matched nothing, the never-written temp stayed in the set,
        and :meth:`commit` then refused the WHOLE set with "staged file(s) were never
        written" — the exact healthy-figures-sunk, environment-blamed failure the method
        exists to prevent, for any export directory reached through a symlink.

        A named target that resolves to nothing in the set is genuinely nothing to
        withdraw — the one in-app caller (``save_figure``) records a member's name just
        before staging it, so a ``stage`` that itself refuses (write-protected target)
        leaves a recorded name with no staged entry behind it. That is not a silent
        no-op: there is no temp to delete and no membership to retract.
        """
        wanted = {resolve_write_target(t)[0] for t in targets}
        kept: list[tuple[Path, Path]] = []
        for tmp, real in self._staged:
            if real in wanted:
                _discard(tmp)
            else:
                kept.append((tmp, real))
        self._staged = kept
        # A withdrawn member is not going to be written, so its identity sentences describe
        # nothing that happened. They leave with it.
        self._pending_identity = [(real, note) for real, note in self._pending_identity
                                  if real not in wanted]

    # -- the two endings -------------------------------------------------------------------
    def abandon(self) -> None:
        """Delete every staged temporary, and any directory this set had to create.

        "Nothing on disk changes" is the promise; an empty folder named after an export
        that never happened is a change, and a confusing one — the user goes looking in it
        for the twelve files the status line said were not written.

        Innermost first, and each removal is conditional on the directory still being
        empty, so a concurrent write or a later successful export into the same folder is
        never disturbed.
        """
        for tmp, _target in self._staged:
            _discard(tmp)
        self._staged.clear()
        # Nothing was placed, so nothing about any target's identity changed. Publishing the
        # sentences anyway would tell the user a link was broken by a write that never ran.
        self._pending_identity.clear()
        # The SWEEP is different: those removals happened whether or not this set landed,
        # and a removal is never silent. Nobody is taking delivery here, so they go to
        # the ledger for the next surface that speaks about these files.
        if self._swept_notes:
            _queue_identity(self._swept_notes)
            self._swept_notes = []
        self._swept_dirs.clear()
        for made in reversed(self._made_dirs):
            try:
                made.rmdir()        # refuses on a non-empty directory; that is the guard
            except OSError:
                break               # not empty (or gone) -- and neither is anything above
        self._made_dirs.clear()

    def commit(self) -> list[Path]:
        """Replace every target. All of them, or none of them."""
        missing = [target.name for tmp, target in self._staged if not tmp.exists()]
        if missing:
            self.abandon()
            raise FileNotFoundError(
                f"{len(missing)} staged file(s) were never written ({', '.join(missing)}); "
                f"nothing was replaced.")

        backups: list[tuple[Path, Path]] = []      # (backup, target) for targets that existed
        replaced: list[Path] = []
        # Which member the commit is on, so a failure can be re-addressed to the file the
        # user named rather than to the staging or backup sibling it actually landed on.
        here: tuple[Path, Path] | None = None
        # The backup being COPIED right now, before it has earned its place in `backups`.
        # ``_copy_file`` creates the file and can raise half-way through it (the measured
        # case: ENOSPC on a nearly-full disk, exactly when copying an 8 MB target), and the
        # except path below restores/discards only members of ``backups`` while ``abandon``
        # removes only staged temps — so a partial backup leaked as a hidden
        # ``.replacing-*`` file on the very disk that had just run out of space, and every
        # retry (the natural response to a full-disk error) leaked another. Tracked
        # separately so the failure path can free it FIRST — before anything else needs
        # that space back.
        backup_in_progress: Path | None = None
        try:
            for tmp, target in self._staged:
                here = (tmp, target)
                if target.exists():
                    backup = temp_sibling(target, tag="replacing")
                    here = (backup, target)
                    backup_in_progress = backup
                    _copy_file(target, backup)
                    dress_like(backup, target)   # so a rollback restores the file, not just bytes
                    backups.append((backup, target))
                    backup_in_progress = None
                    here = (tmp, target)
                dress_like(tmp, target)
                os.replace(tmp, target)
                replaced.append(target)
        except BaseException as exc:
            # A half-copied backup is the one file the rollback below does not know about
            # (it never reached `backups`). Delete it before anything else: it is garbage
            # by construction, and on the ENOSPC path it is squatting on the space the
            # user needs back.
            if backup_in_progress is not None:
                _discard(backup_in_progress)
            # The rollback restores CONTENTS, and `abandon` below drops the pending identity
            # sentences with them: after a failed commit the user is told nothing was placed,
            # and "those other names still hold the PREVIOUS contents" would be false --
            # after a rollback the previous contents are also the current ones.
            for backup, target in backups:
                try:
                    os.replace(backup, target)
                except OSError:
                    pass
            for target in replaced:
                if not any(target == t for _b, t in backups):
                    _discard(target)
            self.abandon()
            for backup, _target in backups:
                _discard(backup)
            restated = (restate_staging_failure(exc, here[0], here[1])
                        if here is not None else None)
            if restated is None:
                raise
            raise restated from exc
        for backup, _target in backups:
            _discard(backup)
        self._staged.clear()
        self._made_dirs.clear()   # they hold the export now; they are not ours to remove
        # The writes happened -- NOW the identity sentences describe something real. Only
        # for the members actually placed: a set that reached here placed all of them, and
        # anything withdrawn on the way lost its sentences with `unstage`.
        placed = set(replaced)
        landed = [(target, note) for target, note in self._pending_identity
                  if target in placed]
        self._pending_identity.clear()
        # The sweep's sentences ride along with the commit's own disclosures — the sweep
        # happened on this set's way in, and this is the ending that takes delivery.
        landed = self._swept_notes + landed
        self._swept_notes = []
        self._swept_dirs.clear()
        if landed:
            if self._notes is not None:
                self._notes.extend(note for _target, note in landed)
            else:
                _queue_identity(landed)
        return replaced


def _copy_file(src: Path, dst: Path) -> None:
    with open(src, "rb") as rfh, open(dst, "wb") as wfh:
        while True:
            block = rfh.read(1 << 20)
            if not block:
                break
            wfh.write(block)


@contextmanager
def staged_write_set(*, notes: list[str] | None = None) -> Iterator[StagedSet]:
    """A :class:`StagedSet` that commits on a clean exit and abandons on any exception."""
    group = StagedSet(notes)
    try:
        yield group
    except BaseException:
        group.abandon()
        raise
    group.commit()
