"""User labels on events: a general-purpose ``key=value`` vocabulary carried in ``flag``.

WHAT THIS IS FOR. Curation answers "is this event real?" — one bit, and the app has had it
from the start. A label answers "what IS it?", and that question has no fixed answer: on one
bench a label is a molecule (``molecule=apo-ferritin``), on another it is an event class
(``type=collision``), and for settling how many conductance states an event really has it is
a count the person read off the screen (``levels=3``). So the vocabulary is the USER'S, not
an enum in this file.

WHY IT LIVES IN ``flag`` AND NOT IN A NEW COLUMN. Three reasons, in order of weight:

1. ``events.SCHEMA`` is frozen at 20 columns. That freeze is load-bearing — every export,
   every reader, every stored project agrees on it — and "add a column for the new feature"
   is exactly the pressure it exists to resist.
2. ``flag`` is already a ``;``-separated token set with a curator's own mark in it
   (:data:`~nanoscope.analysis.events.FLAG_CURATED`), so a person writing into this column
   is the established design, not a new one.
3. Most importantly, it makes labels behave correctly for free. Tokens outside
   :data:`~nanoscope.analysis.events.DERIVED_FLAGS` are INHERITED VERBATIM by
   :func:`~nanoscope.analysis.events.inherited_curation` when a row is rebuilt — so a label
   survives a merge, a boundary drag and a hand-add, and a re-detection that would erase it
   is already counted as destroying hand curation, which already raises the warning and the
   undo step. A separate column would have had to re-implement all of that, and would have
   got the merge case wrong first.

THE TOKEN FORM. A label is ``key=value``; a flag is a bare token. Nothing in the derived
vocabulary contains ``=`` (``dwell>max_dwell_s`` is the closest and uses ``>``), so the two
never collide and a pre-label project reads back unchanged. One key holds one value per
event — setting it again replaces it — but an event may carry as many keys as you like,
which is what lets ``molecule=apo-ferritin;type=translocation;levels=3`` be one row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .events import FLAG_SEP, LABEL_SEP, flag_tokens

__all__ = [
    "LABEL_SEP",
    "CONFLICT_SEP",
    "label_conflicts",
    "parse_labels",
    "set_label",
    "clear_label",
    "label_column",
    "label_keys",
    "format_labels",
    "validate_label",
    "validate_label_hotkey",
    "RESERVED_LABEL_HOTKEYS",
    "LabelEntry",
    "LabelVocabulary",
]

#: Joins the values of a key that ended up with more than one — only a merge can do that.
#: Chosen because it is not legal inside a value (validate_label refuses it), so a joined
#: value can never be mistaken for one somebody typed.
CONFLICT_SEP = "|"

#: What separates a label's key from its value inside one token. Defined in
#: :mod:`nanoscope.analysis.events` and re-exported here: ``inherited_curation`` needs it to
#: tell a label from a rejection reason, and it cannot import this module.

#: Characters a key or value may not contain: the two separators, plus whitespace at the
#: ends. A label with a ``;`` in it would silently split into two tokens; one with an ``=``
#: would parse with the wrong value.
_BANNED = (FLAG_SEP, LABEL_SEP, CONFLICT_SEP)

# Bare keys already owned by curation/navigation on at least one label-writing window.
# Keep one conservative set for both windows: a palette that changes meaning when the
# curator moves from Gallery to Inspector is more dangerous than one with three fewer
# letter choices. Modifiers are not accepted as label hotkeys, so Ctrl/Cmd+A and Ctrl/Cmd+Z
# do not need entries here; their unmodified letters remain available unless listed.
RESERVED_LABEL_HOTKEYS = frozenset({"a", "r", "t", " "})


def validate_label_hotkey(hotkey: Any) -> str:
    """Return the canonical safe one-character shortcut, or ``""`` for automatic.

    Letter shortcuts are case-insensitive because Qt maps ``c`` and ``C`` to the same
    unmodified key. Multi-key sequences and keys already used for curation are refused so
    the GUI never installs an ambiguous shortcut. This validator is deliberately separate
    from :class:`LabelEntry` construction: an older project containing a now-reserved or
    malformed hotkey must still load its label definitions; :meth:`LabelVocabulary.hotkeys`
    simply leaves that legacy shortcut unarmed until the user fixes it in the editor.
    """
    raw = "" if hotkey is None else str(hotkey)
    if raw == "":
        return ""
    # A literal space is a meaningful attempted shortcut, not the same thing as the blank
    # cell that asks for an automatic digit.
    if raw == " ":
        raise ValueError("hotkey Space is reserved for acceptance toggling")
    text = raw.strip()
    if len(text) != 1 or not text.isprintable() or text.isspace():
        raise ValueError("hotkey must be one printable character (or blank for 1–9)")
    canonical = text.lower()
    # Some Unicode lowercase mappings expand to multiple code points. Qt can still bind the
    # original glyph; retaining it is safer than pretending it is a different single key.
    if len(canonical) != 1:
        canonical = text
    if canonical in RESERVED_LABEL_HOTKEYS:
        names = {"a": "Accept", "r": "Reject", "t": "Show in trace", " ": "Toggle"}
        raise ValueError(
            f"hotkey {text!r} is reserved for {names[canonical]}"
        )
    return canonical


def validate_label(key: str, value: str) -> tuple[str, str]:
    """Return the cleaned ``(key, value)``, or raise ValueError saying which is wrong.

    Raises rather than sanitising: a label quietly rewritten is a label that will not match
    the one the user typed yesterday, and matching is the entire point of a vocabulary.
    """
    k, v = str(key).strip(), str(value).strip()
    for name, s in (("key", k), ("value", v)):
        if not s:
            raise ValueError(f"label {name} is empty")
        for bad in _BANNED:
            if bad in s:
                raise ValueError(
                    f"label {name} {s!r} contains {bad!r}, which separates tokens "
                    f"({FLAG_SEP!r}) or keys from values ({LABEL_SEP!r})")
    return k, v


def parse_labels(flag: Any) -> dict[str, str]:
    """The ``key=value`` labels in one flag cell, in first-appearance order.

    Bare tokens (the derived flags and ``flagged``) are ignored — this returns labels only.
    A malformed token (``a=b=c``) splits on the FIRST separator, so the value keeps the rest.
    """
    out: dict[str, str] = {}
    for token in flag_tokens(flag):
        if LABEL_SEP in token:
            k, _, v = token.partition(LABEL_SEP)
            k, v = k.strip(), v.strip()
            if not k or not v:
                continue
            if k not in out:
                out[k] = v
            elif v not in out[k].split(CONFLICT_SEP):
                # A CONFLICT, and it must be visible. Merging two fragments labelled
                # `molecule=apo-ferritin` and `molecule=holo-ferritin` unions both tokens
                # (inherited_curation is generic and cannot know one key holds one value),
                # and returning just the first made the merged row claim one molecule while
                # carrying evidence of two, with nobody told. Joining them means the tile,
                # the readout and the export column all show `apo-ferritin|holo-ferritin`
                # and the curator fixes it — which is the only correct outcome, because the
                # app does not know which fragment was right.
                out[k] = out[k] + CONFLICT_SEP + v
    return out


def set_label(flag: Any, key: str, value: str) -> str:
    """``flag`` with ``key`` set to ``value``, every other token preserved in order.

    Replaces an existing value for the same key rather than appending a second one, so a
    row can never end up claiming two molecules at once.
    """
    k, v = validate_label(key, value)
    out: list[str] = []
    replaced = False
    for token in flag_tokens(flag):
        if LABEL_SEP in token and token.partition(LABEL_SEP)[0].strip() == k:
            if not replaced:
                out.append(f"{k}{LABEL_SEP}{v}")
                replaced = True
            continue
        out.append(token)
    if not replaced:
        out.append(f"{k}{LABEL_SEP}{v}")
    return FLAG_SEP.join(out)


def clear_label(flag: Any, key: str) -> str:
    """``flag`` with ``key`` removed. Unknown key is a no-op, not an error."""
    k = str(key).strip()
    return FLAG_SEP.join(
        t for t in flag_tokens(flag)
        if not (LABEL_SEP in t and t.partition(LABEL_SEP)[0].strip() == k))


def label_column(flags: Iterable[Any], key: str) -> np.ndarray:
    """Vectorised read: the value of ``key`` for every cell, ``''`` where unset.

    Returns an object array so a missing label is an empty string rather than NaN — the
    table, the exports and ``groupby`` all treat "unlabelled" as a category, not as a hole.
    """
    k = str(key).strip()
    return np.array([parse_labels(f).get(k, "") for f in np.asarray(list(flags), dtype=object)],
                    dtype=object)


def label_keys(flags: Iterable[Any]) -> list[str]:
    """Every label key present across these cells, in first-appearance order."""
    out: list[str] = []
    for f in flags:
        for k in parse_labels(f):
            if k not in out:
                out.append(k)
    return out


def label_conflicts(flag: Any) -> dict[str, list[str]]:
    """``{key: [values]}`` for every key a merge left with more than one value."""
    return {k: v.split(CONFLICT_SEP) for k, v in parse_labels(flag).items()
            if CONFLICT_SEP in v}


def format_labels(flag: Any) -> str:
    """``"molecule=apo-ferritin, type=collision"`` — for a readout, not for storage."""
    return ", ".join(f"{k}{LABEL_SEP}{v}" for k, v in parse_labels(flag).items())


@dataclass(frozen=True)
class LabelEntry:
    """One entry in the palette: what a key press writes.

    ``hotkey`` is retained verbatim enough for old projects to load. The safe, canonical
    assignment is decided by :meth:`LabelVocabulary.hotkeys`; the editor reports malformed,
    reserved and duplicate assignments where a person can fix them.
    """

    key: str
    value: str
    hotkey: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        # Normalise, do not merely check: validate_label already returns the cleaned pair,
        # and storing the raw one let " molecule " sit beside "molecule" as a second entry
        # that remove("molecule", ...) could never reach.
        key, value = validate_label(self.key, self.value)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", value)
        raw_hotkey = str(self.hotkey)
        # Preserve a legacy literal-Space assignment so the safety gate can recognise and
        # leave it unarmed. Collapsing it to the automatic blank would silently remap that
        # label onto a digit when the project opens.
        object.__setattr__(self, "hotkey", " " if raw_hotkey == " " else raw_hotkey.strip())

    @property
    def token(self) -> str:
        return f"{self.key}{LABEL_SEP}{self.value}"


@dataclass
class LabelVocabulary:
    """The label palette for a session: an ordered list of entries the user defined.

    Ordered because the order IS the hotkey layout when entries carry no explicit one, and
    a palette that reshuffled itself between sessions would retrain the user's fingers every
    time they opened the app.
    """

    entries: list[LabelEntry] = field(default_factory=list)

    def add(self, key: str, value: str, hotkey: str = "", note: str = "") -> LabelEntry:
        entry = LabelEntry(key=key, value=value, hotkey=hotkey, note=note)
        for i, existing in enumerate(self.entries):
            if (existing.key, existing.value) == (entry.key, entry.value):
                self.entries[i] = entry
                return entry
        self.entries.append(entry)
        return entry

    def remove(self, key: str, value: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries
                        if (e.key, e.value) != (str(key).strip(), str(value).strip())]
        return len(self.entries) < before

    def hotkeys(self) -> dict[str, LabelEntry]:
        """``{character: entry}`` — THE assignment, and the only one.

        Both the palette display and the key handler must read this, or they disagree.
        Measured before it existed: with one explicit hotkey in the list, the combo printed
        "molecule=apo-ferritin  [2]" while pressing 2 wrote *holo*-ferritin, because the
        display numbered ALL entries by position and the handler numbered only the ones
        WITHOUT an explicit key. Wrong labels, written in bulk, silently.

        Safe explicit hotkeys win; the rest take the digits 1-9 in order, skipping any
        digit an explicit hotkey has already claimed. Invalid/reserved legacy assignments
        and case-insensitive duplicates stay unarmed (and do not silently fall back to a
        different digit). An entry that gets no key gets none — the palette says so rather
        than pretending.
        """
        out: dict[str, LabelEntry] = {}
        for e in self.entries:
            if not e.hotkey:
                continue
            try:
                hotkey = validate_label_hotkey(e.hotkey)
            except ValueError:
                continue
            if hotkey and hotkey not in out:
                out[hotkey] = e
        digits = [d for d in "123456789" if d not in out]
        for e in self.entries:
            if e.hotkey or not digits:
                continue
            out[digits.pop(0)] = e
        return out

    def hotkey_for(self, entry: LabelEntry) -> str:
        """The character that writes ``entry``, or "" when it has none to spare."""
        for ch, e in self.hotkeys().items():
            if (e.key, e.value) == (entry.key, entry.value):
                return ch
        return ""

    def for_hotkey(self, ch: str) -> LabelEntry | None:
        """The entry a key press means. Reads :meth:`hotkeys`, so it can never disagree
        with what the palette printed."""
        try:
            canonical = validate_label_hotkey(ch)
        except ValueError:
            return None
        return self.hotkeys().get(canonical)

    def keys(self) -> list[str]:
        out: list[str] = []
        for e in self.entries:
            if e.key not in out:
                out.append(e.key)
        return out

    # -- persistence (rides a project's JSON-safe params) ---------------------------------
    def to_json(self) -> list[dict[str, str]]:
        return [{"key": e.key, "value": e.value, "hotkey": e.hotkey, "note": e.note}
                for e in self.entries]

    @classmethod
    def from_json(cls, blob: Any) -> "LabelVocabulary":
        """Rebuild from :meth:`to_json`. A malformed entry is skipped, not fatal — a broken
        palette must not stop a project from opening."""
        out = cls()
        for row in (blob or []):
            try:
                out.add(str(row["key"]), str(row["value"]),
                        str(row.get("hotkey", "")), str(row.get("note", "")))
            except (KeyError, TypeError, ValueError):
                continue
        return out


def apply_label(df: pd.DataFrame, rows: Iterable[int], key: str, value: str) -> int:
    """Write ``key=value`` onto ``rows`` of a shared events frame IN PLACE.

    Returns how many rows actually changed. Writes through ``set_label`` rather than over
    the cell, so a row's derived flags and its ``flagged`` mark survive — overwriting the
    cell is the exact defect FLAG_CURATED's docstring records.
    """
    if "flag" not in df.columns:
        raise KeyError("frame has no 'flag' column; it is not an events table")
    col = df.columns.get_loc("flag")
    changed = 0
    for r in rows:
        before = df.iat[r, col]
        after = set_label(before, key, value)
        if after != (before if isinstance(before, str) else ""):
            df.iat[r, col] = after
            changed += 1
    return changed


def clear_labels(df: pd.DataFrame, rows: Iterable[int], key: str) -> int:
    """Remove ``key`` from ``rows`` in place; returns how many changed."""
    col = df.columns.get_loc("flag")
    changed = 0
    for r in rows:
        before = df.iat[r, col]
        after = clear_label(before, key)
        if after != (before if isinstance(before, str) else ""):
            df.iat[r, col] = after
            changed += 1
    return changed
