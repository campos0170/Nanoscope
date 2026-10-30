"""What the curation windows browse and count: filters, navigation masks, progress, notes.

The Event inspector and the Event gallery may not import each other (the sibling-import
guards), yet they must agree on what "Unreviewed" means, which events a label filter
holds and how far a curation pass has got. Every such rule lives here, on plain numpy /
pandas, so both windows ask one function the same question.

UNREVIEWED INCLUDES "NEVER RECORDED". A row is reviewed only when its ``review`` cell
normalises to ``reviewed`` (:func:`~nanoscope.analysis.events.reviewed_mask`). A row with
no review record -- an old detection, or one whose detector never stamped the column --
counts as unreviewed, and the progress tooltip says how many of them there are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .events import (
    KIND_MANUAL,
    KIND_MIXED,
    REVIEW_COL,
    SOURCE_COL,
    _normalise_review,
    mark_reviewed,
    reviewed_mask,
    source_kind,
)
from .labels import CONFLICT_SEP, LabelVocabulary, apply_label, clear_labels, parse_labels

__all__ = [
    "FILTER_ALL",
    "FILTER_ACCEPTED",
    "FILTER_REJECTED",
    "FILTER_UNREVIEWED",
    "FILTER_UNLABELLED",
    "FILTER_HAND_MADE",
    "FILTER_DETECTED",
    "FILTER_MIXED",
    "SOURCE_FILTERS",
    "source_masks",
    "LABEL_FILTER_PREFIX",
    "FIXED_FILTERS",
    "label_filter",
    "parse_filter",
    "is_filter",
    "filter_depends_on_labels_or_review",
    "filter_mask",
    "labelled_mask",
    "token_mask",
    "next_member_after",
    "prev_member_before",
    "first_member",
    "count_before",
    "landing_after_removal",
    "CurationProgress",
    "curation_progress",
    "format_progress",
    "describe_progress",
    "filter_options",
    "end_of_filter_note",
    "no_unreviewed_note",
    "label_rows",
    "unlabel_rows",
    "same_events",
    "row_unchanged",
    "filter_display_candidates",
    "label_display_candidates",
]

FILTER_ALL = "All"
FILTER_ACCEPTED = "Accepted only"
FILTER_REJECTED = "Rejected only"
FILTER_UNREVIEWED = "Unreviewed"
FILTER_UNLABELLED = "Unlabelled"
#: The source filters: rows a person drew, rows a detector found, rows merged from both. Offered
#: only once the table holds a hand-made (or merged) row, so a plain detection's combo is the
#: four it always was.
FILTER_HAND_MADE = "Hand-made only"
FILTER_DETECTED = "Detected only"
FILTER_MIXED = "Merged (hand + detected)"
SOURCE_FILTERS = (FILTER_HAND_MADE, FILTER_DETECTED, FILTER_MIXED)
#: A label filter's display text is this prefix plus the ``key=value`` token. The text IS
#: the filter's identity: both windows select filters by their combo text.
LABEL_FILTER_PREFIX = "Labelled: "
#: Always offered, in this order. Unlabelled and the label filters follow when a palette
#: or a label exists.
FIXED_FILTERS = (FILTER_ALL, FILTER_ACCEPTED, FILTER_REJECTED, FILTER_UNREVIEWED)

_KIND_BY_NAME = {
    FILTER_ALL: "all",
    FILTER_ACCEPTED: "accepted",
    FILTER_REJECTED: "rejected",
    FILTER_UNREVIEWED: "unreviewed",
    FILTER_UNLABELLED: "unlabelled",
    FILTER_HAND_MADE: "hand_made",
    FILTER_DETECTED: "detected",
    FILTER_MIXED: "mixed",
}


# -- filter names -----------------------------------------------------------------------------
def label_filter(token: str) -> str:
    """``"type=collision"`` -> ``"Labelled: type=collision"``."""
    return f"{LABEL_FILTER_PREFIX}{str(token).strip()}"


def parse_filter(mode: Any) -> tuple[str, str, str] | None:
    """``(kind, key, value)`` for a filter's display text, or ``None`` if it is not one.

    ``kind`` is one of all / accepted / rejected / unreviewed / unlabelled / label; the key
    and value are ``""`` except for a label filter, which splits on the FIRST ``=`` (the
    value keeps the rest, as :func:`~nanoscope.analysis.labels.parse_labels` does).
    """
    text = str(mode) if mode is not None else ""
    kind = _KIND_BY_NAME.get(text)
    if kind is not None:
        return kind, "", ""
    if not text.startswith(LABEL_FILTER_PREFIX):
        return None
    key, sep, value = text[len(LABEL_FILTER_PREFIX):].partition("=")
    key, value = key.strip(), value.strip()
    if not sep or not key or not value:
        return None
    return "label", key, value


def is_filter(mode: Any) -> bool:
    return parse_filter(mode) is not None


def filter_depends_on_labels_or_review(mode: Any) -> bool:
    """True when a label write or a review stamp can move an event in or out of ``mode``
    -- the filters whose membership is not decided by ``accepted`` alone."""
    parsed = parse_filter(mode)
    return parsed is not None and parsed[0] in ("unreviewed", "unlabelled", "label")


# -- masks ------------------------------------------------------------------------------------
def _per_distinct(values: Any, fn: Callable[[Any], bool]) -> np.ndarray:
    """``fn`` over every cell, evaluated once per distinct value (and once for missing).

    A flag or review column holds a handful of distinct strings whatever its length, so
    this keeps a per-cell rule out of a per-row Python loop.
    """
    codes, uniques = pd.factorize(pd.Series(np.asarray(values, dtype=object)),
                                  use_na_sentinel=True)
    hit = np.fromiter((bool(fn(u)) for u in uniques), dtype=bool, count=len(uniques))
    out = np.empty(len(codes), dtype=bool)
    known = codes >= 0
    if len(uniques):
        out[known] = hit[codes[known]]
    if not known.all():
        out[~known] = bool(fn(pd.NA))
    return out


def _flags(df: pd.DataFrame) -> np.ndarray:
    if "flag" in df.columns:
        return df["flag"].to_numpy(dtype=object)
    return np.full(len(df), "", dtype=object)


def labelled_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows carrying at least one ``key=value`` label. Bare flag tokens are not labels."""
    if not len(df):
        return np.zeros(0, dtype=bool)
    return _per_distinct(_flags(df), lambda f: bool(parse_labels(f)))


def token_mask(df: pd.DataFrame, key: str, value: str) -> np.ndarray:
    """Rows labelled ``key=value``. A merge conflict (``apo|holo``) matches each value."""
    if not len(df):
        return np.zeros(0, dtype=bool)
    k, v = str(key).strip(), str(value).strip()
    return _per_distinct(
        _flags(df), lambda f: v in parse_labels(f).get(k, "").split(CONFLICT_SEP))


def source_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(hand_made, detected, mixed)`` boolean masks over ``df``'s rows.

    Hand-made is ``source`` reading manual, falling back to ``added`` only where ``source`` was
    never recorded; mixed is a merge across sources (``source`` reads mixed); detected is
    everything else."""
    n = len(df)
    if n == 0:
        z = np.zeros(0, dtype=bool)
        return z, z.copy(), z.copy()
    if SOURCE_COL in df.columns:
        src = df[SOURCE_COL].to_numpy(dtype=object)
        kind = np.array(_per_distinct_value(src, source_kind), dtype=object)
        known = np.array(_per_distinct_value(src, lambda v: v is not None and not pd.isna(v)),
                         dtype=bool)
    else:
        kind = np.full(n, "unknown", dtype=object)
        known = np.zeros(n, dtype=bool)
    added = (df["added"].to_numpy(dtype=bool) if "added" in df.columns
             else np.zeros(n, dtype=bool))
    hand = np.where(known, kind == KIND_MANUAL, added)
    mixed = known & (kind == KIND_MIXED)
    return hand, ~(hand | mixed), mixed


def _per_distinct_value(values: Any, fn: Callable[[Any], Any]) -> list[Any]:
    """``fn`` over every cell, evaluated once per distinct value."""
    codes, uniques = pd.factorize(pd.Series(np.asarray(values, dtype=object)),
                                  use_na_sentinel=True)
    results = [fn(u) for u in uniques]
    na_result = fn(pd.NA)
    return [results[c] if c >= 0 else na_result for c in codes]


def filter_mask(df: pd.DataFrame, mode: str) -> np.ndarray:
    """The rows ``mode`` keeps, in row order. Raises ValueError for an unknown filter."""
    parsed = parse_filter(mode)
    if parsed is None:
        raise ValueError(f"unknown curation filter {mode!r}")
    kind, key, value = parsed
    n = len(df)
    if kind == "all":
        return np.ones(n, dtype=bool)
    if kind in ("accepted", "rejected"):
        accepted = (df["accepted"].to_numpy(dtype=bool) if "accepted" in df.columns
                    else np.ones(n, dtype=bool))
        return accepted if kind == "accepted" else ~accepted
    if kind == "unreviewed":
        return ~reviewed_mask(df)
    if kind == "unlabelled":
        return ~labelled_mask(df)
    if kind in ("hand_made", "detected", "mixed"):
        hand, detected, mixed = source_masks(df)
        return {"hand_made": hand, "detected": detected, "mixed": mixed}[kind]
    return token_mask(df, key, value)


# -- navigation over a mask -------------------------------------------------------------------
def next_member_after(mask: np.ndarray, row: int) -> int | None:
    hits = np.flatnonzero(np.asarray(mask, dtype=bool)[int(row) + 1:])
    return int(hits[0]) + int(row) + 1 if hits.size else None


def prev_member_before(mask: np.ndarray, row: int) -> int | None:
    hits = np.flatnonzero(np.asarray(mask, dtype=bool)[:max(int(row), 0)])
    return int(hits[-1]) if hits.size else None


def first_member(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(hits[0]) if hits.size else None


def count_before(mask: np.ndarray, row: int) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)[:max(int(row), 0)]))


def same_events(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    """Do two frames hold the same events in the same rows -- identical ``event_id`` and
    ``start_idx`` arrays? The test a re-feed must pass before a window keeps its place:
    a merge renumbers ``event_id``, so a kept id could otherwise name another event."""
    if a is None or b is None or len(a) != len(b):
        return False
    for col in ("event_id", "start_idx"):
        if (col in a.columns) != (col in b.columns):
            return False
        if col in a.columns and not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
            return False
    return "event_id" in a.columns


def superset_by_id(old: pd.DataFrame | None, new: pd.DataFrame | None) -> bool:
    """Does ``new`` hold every event ``old`` held, each still at its ``start_idx``?

    The test a re-feed passes after a hand-made event joins a table: ids are never reused
    by an add, so a kept id still names the same event, and a window may keep its filter,
    its current event and its selection (``same_events`` demands an identical frame, which
    an add never is). Event ids must be unique in ``new``."""
    if old is None or new is None or "event_id" not in old.columns or "event_id" not in new.columns:
        return False
    if "start_idx" not in old.columns or "start_idx" not in new.columns:
        return False
    if len(new) < len(old):
        return False
    new_ids = pd.Index(new["event_id"].to_numpy())
    if not new_ids.is_unique:
        return False
    pos = new_ids.get_indexer(old["event_id"].to_numpy())
    if (pos < 0).any():
        return False
    return bool(np.array_equal(new["start_idx"].to_numpy()[pos], old["start_idx"].to_numpy()))


def can_keep_place(old: pd.DataFrame | None, new: pd.DataFrame | None) -> bool:
    """May a window that was showing ``old`` keep its place when ``new`` arrives?

    Yes for the very same events (an edit), for ``new`` holding all of ``old`` (a hand-made
    event joined), and for ``new`` being ``old`` less some events (a Delete, the undo of an
    add): the same table less some rows, not a new detection. The window keeps its filter;
    when the event it stood on is one of those gone, it lands beside it
    (:func:`landing_after_removal`). This used to also ask that the event on screen survived,
    which the inspector's own Delete never satisfies -- the event on screen is the one
    deleted -- so a curator in Hand-made only was put back in All, on a detected event."""
    return same_events(old, new) or superset_by_id(old, new) or superset_by_id(new, old)


def landing_after_removal(order: Any, gone_id: int, members: Any) -> int | None:
    """Where a window goes when the event it stood on, ``gone_id``, left the table: the first
    event after it in ``order`` (the window's list as it was, by ``event_id``) that
    ``members`` (the ids its list holds now) still holds, else the last one before it.
    ``None`` when there is neither: the list is empty now.

    One rule, each window asking it of its own list: the inspector's filter runs in table
    order, the gallery's tiles in the order they are drawn (a sort by depth is not time)."""
    order = np.asarray(order, dtype=np.int64)
    at = np.flatnonzero(order == int(gone_id))
    if not at.size:
        return None
    still = np.isin(order, np.asarray(members, dtype=np.int64))
    row = next_member_after(still, int(at[0]))
    if row is None:
        row = prev_member_before(still, int(at[0]))
    return None if row is None else int(order[row])


def row_unchanged(a: pd.DataFrame, b: pd.DataFrame, row: int) -> bool:
    """Does positional ``row`` hold the same values in ``a`` and ``b`` (missing equals
    missing)? Asked after an undo: did it touch the event a window would return to?"""
    if set(a.columns) != set(b.columns) or not (0 <= int(row) < min(len(a), len(b))):
        return False
    for col in a.columns:
        x, y = a[col].iloc[int(row)], b[col].iloc[int(row)]
        x_na, y_na = _is_missing(x), _is_missing(y)
        if x_na or y_na:
            if x_na != y_na:
                return False
            continue
        try:
            if bool(np.all(np.asarray(x == y))):
                continue
        except (TypeError, ValueError):
            pass
        return False
    return True


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False          # an array-like cell is a value, not a missing one


def filter_display_candidates(mode: str) -> tuple[str, ...]:
    """What a narrow closed filter combo may show for ``mode``, longest first. A label
    filter drops its prefix, then its key, so the VALUE is what survives the width."""
    parsed = parse_filter(mode)
    if parsed is None or parsed[0] != "label":
        return (str(mode),)
    _kind, key, value = parsed
    return (str(mode), f"{key}={value}", value)


def label_display_candidates(token: str, hotkey: str | None = None) -> tuple[str, ...]:
    """The same for a palette entry in a Label combo. The hotkey is in every form: the
    last one is what a face too narrow for any of them cuts in the middle."""
    value = str(token).partition("=")[2] or str(token)
    if not hotkey:
        return tuple(dict.fromkeys((str(token), value)))
    return tuple(dict.fromkeys((f"{token}   [{hotkey}]", f"{value}   [{hotkey}]",
                                f"{value} [{hotkey}]")))


# -- progress ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class CurationProgress:
    """How far curation of one detection has got. Counts are for the whole frame."""

    n_total: int = 0
    n_reviewed: int = 0
    #: Rows with no review record at all (counted as unreviewed).
    n_unknown_review: int = 0
    n_accepted: int = 0
    #: ``(token, n)``: the palette's tokens in palette order (zero counts included), then
    #: labels found in the frame but not in the palette, in first-appearance order.
    label_counts: tuple[tuple[str, int], ...] = ()
    orphan_tokens: tuple[str, ...] = ()
    n_labelled: int = 0
    n_conflicted: int = 0
    has_palette: bool = False
    #: Rows a person drew / rows merged across sources (0 for a plain detection).
    n_hand_made: int = 0
    n_mixed: int = 0

    @property
    def n_unreviewed(self) -> int:
        return self.n_total - self.n_reviewed


def curation_progress(df: pd.DataFrame | None,
                      vocab: LabelVocabulary | None = None) -> CurationProgress:
    palette = [e.token for e in (vocab.entries if vocab is not None else [])]
    if df is None or not len(df):
        return CurationProgress(label_counts=tuple((t, 0) for t in palette),
                                has_palette=bool(palette))
    n = len(df)
    reviewed = int(np.count_nonzero(reviewed_mask(df)))
    if REVIEW_COL in df.columns:
        unknown = int(np.count_nonzero(_per_distinct(
            df[REVIEW_COL].to_numpy(dtype=object), lambda r: _normalise_review(r) is None)))
    else:
        unknown = n
    accepted = (int(np.count_nonzero(df["accepted"].to_numpy(dtype=bool)))
                if "accepted" in df.columns else n)
    counts: dict[str, int] = {}
    n_labelled = n_conflicted = 0
    for flag, k in pd.Series(_flags(df)).value_counts(dropna=False, sort=False).items():
        labels = parse_labels(flag)
        if not labels:
            continue
        n_labelled += int(k)
        if any(CONFLICT_SEP in v for v in labels.values()):
            n_conflicted += int(k)
        for key, joined in labels.items():
            for value in joined.split(CONFLICT_SEP):
                token = f"{key}={value}"
                counts[token] = counts.get(token, 0) + int(k)
    # First-appearance order for the labels the palette does not name: value_counts is
    # unsorted above, but a stable order needs the row order, read once over distinct cells.
    seen: list[str] = []
    if counts:
        for flag in pd.unique(pd.Series(_flags(df))):
            for key, joined in parse_labels(flag).items():
                for value in joined.split(CONFLICT_SEP):
                    token = f"{key}={value}"
                    if token not in palette and token not in seen:
                        seen.append(token)
    label_counts = tuple((t, counts.get(t, 0)) for t in palette) + tuple(
        (t, counts[t]) for t in seen)
    hand, _detected, mixed = source_masks(df)
    return CurationProgress(
        n_total=n, n_reviewed=reviewed, n_unknown_review=unknown, n_accepted=accepted,
        label_counts=label_counts, orphan_tokens=tuple(seen), n_labelled=n_labelled,
        n_conflicted=n_conflicted, has_palette=bool(palette),
        n_hand_made=int(np.count_nonzero(hand)), n_mixed=int(np.count_nonzero(mixed)))


def format_progress(p: CurationProgress) -> str:
    """``"Reviewed 312 / 870 · accepted 801 · labels: type=collision 40, type=junk 2"``."""
    parts = [f"Reviewed {p.n_reviewed} / {p.n_total}", f"accepted {p.n_accepted}"]
    used = [(t, k) for t, k in p.label_counts if k]
    if used:
        parts.append("labels: " + ", ".join(f"{t} {k}" for t, k in used))
    elif p.has_palette:
        parts.append("labels: none yet")
    return " · ".join(parts)


def describe_progress(p: CurationProgress) -> str:
    """The progress line's tooltip: what each count means, and what it is counted over."""
    lines = [f"Reviewed: {p.n_reviewed} of {p.n_total} events — a person has ruled on them "
             "(accept, reject or a label)."]
    unrev = f"Not yet reviewed: {p.n_unreviewed}"
    if p.n_unknown_review:
        unrev += (f" — {p.n_unknown_review} of them have no review record and are counted "
                  "as unreviewed here")
    lines.append(unrev + ".")
    lines.append(f"Accepted: {p.n_accepted} of {p.n_total}.")
    if p.n_hand_made or p.n_mixed:
        lines.append(f"Hand-made: {p.n_hand_made} · detected: "
                     f"{p.n_total - p.n_hand_made - p.n_mixed}"
                     + (f" · merged from both: {p.n_mixed}" if p.n_mixed else "") + ".")
    palette = [(t, k) for t, k in p.label_counts if t not in p.orphan_tokens]
    orphans = [(t, k) for t, k in p.label_counts if t in p.orphan_tokens]
    if palette:
        lines.append("Labels (palette): " + " · ".join(f"{t} {k}" for t, k in palette))
    if orphans:
        lines.append("Labels not in the palette: " + " · ".join(f"{t} {k}" for t, k in orphans))
    unl = f"Unlabelled: {p.n_total - p.n_labelled}"
    if p.n_conflicted:
        unl += f" · with a merge conflict: {p.n_conflicted}"
    lines.append(unl + ".")
    lines.append("Unreviewed means no review stamp, so a rejection made without one counts "
                 "here even where the events table's status calls it a person's rejection.")
    lines.append("Counts are for the whole detection, not the current filter.")
    return "\n".join(lines)


def filter_options(progress: CurationProgress | None, vocab: LabelVocabulary | None,
                   current: str | None = None) -> tuple[str, ...]:
    """The filters a window offers: the fixed four, then (with a palette or any label)
    Unlabelled and one ``Labelled:`` entry per palette token and per label found only in
    the frame. ``current`` is appended when it is a filter and not already offered -- a
    window never drops the filter the user is browsing."""
    out = list(FIXED_FILTERS)
    if progress is not None and (progress.n_hand_made or progress.n_mixed):
        out.extend((FILTER_HAND_MADE, FILTER_DETECTED))
        if progress.n_mixed:
            out.append(FILTER_MIXED)
    tokens = [e.token for e in (vocab.entries if vocab is not None else [])]
    if progress is not None:
        tokens += [t for t in progress.orphan_tokens if t not in tokens]
    if tokens or (vocab is not None and vocab.entries):
        out.append(FILTER_UNLABELLED)
        out.extend(label_filter(t) for t in tokens)
    if current and is_filter(current) and current not in out:
        out.append(current)
    return tuple(out)


# -- notes ------------------------------------------------------------------------------------
def end_of_filter_note(mode: str, *, noun: str, n_earlier: int = 0, n_unreviewed: int = 0,
                       first_key: str = "") -> tuple[str, str]:
    """``(short, full)`` for an advance that found nothing after the decided event.

    The short form fits a narrow status slot (~40 characters); the full sentence goes in
    the tooltip. ``first_key`` names the First-unreviewed shortcut as the window prints it.
    """
    key = f" ({first_key})" if first_key else ""
    if mode == FILTER_UNREVIEWED:
        if n_unreviewed == 0:
            return ("Every event reviewed",
                    "Every event in this detection has been reviewed.")
        if n_earlier > 0:
            return (f"End of Unreviewed · {n_earlier} earlier{key}",
                    f"End of Unreviewed — no unreviewed {noun} after this one; {n_earlier} "
                    f"earlier remain. First unreviewed{key} jumps to the first.")
        return ("End of Unreviewed",
                f"End of Unreviewed — no unreviewed {noun} after this one.")
    return (f"End of {mode}",
            f"End of {mode} — this was the last {noun}; the decision was written and the "
            "window stayed on it.")


def no_unreviewed_note(mode: str, *, noun: str, n_unreviewed: int,
                       chooser: str) -> tuple[str, str]:
    """``(short, full)`` for First unreviewed finding nothing in ``mode``."""
    if n_unreviewed == 0:
        return ("Every event reviewed", "Every event in this detection has been reviewed.")
    return (f"None unreviewed here · {n_unreviewed} elsewhere",
            f"No unreviewed {noun} in {mode} — {n_unreviewed} remain in this detection "
            f"({chooser} ▸ Unreviewed shows them).")


# -- label writes shared by both windows ------------------------------------------------------
def label_rows(df: pd.DataFrame, rows: Iterable[int], key: str, value: str) -> tuple[int, int]:
    """Write ``key=value`` onto ``rows`` and stamp them reviewed: ``(labels, reviews)``
    changed. Applying a label is a human inspection even when the value was already there.
    A frame with no ``flag`` column is not an events table: nothing is written."""
    rows = [int(r) for r in rows]
    if not rows or "flag" not in df.columns:
        return 0, 0
    changed = apply_label(df, rows, key, value)
    reviewed = mark_reviewed(df, rows=np.asarray(rows, dtype=np.int64))
    return changed, reviewed


def unlabel_rows(df: pd.DataFrame, rows: Iterable[int], key: str) -> tuple[int, int]:
    """Remove label ``key`` from ``rows`` and stamp them reviewed, as the gallery's Clear
    always has: ``(labels, reviews)`` changed."""
    rows = [int(r) for r in rows]
    if not rows or "flag" not in df.columns:
        return 0, 0
    changed = clear_labels(df, rows, key)
    reviewed = mark_reviewed(df, rows=np.asarray(rows, dtype=np.int64))
    return changed, reviewed
