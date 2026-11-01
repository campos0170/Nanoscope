"""Which event edges a PERSON put where they are -- the question the trace marks answer.

A detected span, a span whose edge was dragged in the Event Inspector, one a merge stretched
over its neighbour, and one drawn by hand (with the trace keys' edge fits or without) all used
to shade the trace identically, so the picture could not say which boundaries are the
detector's and which are somebody's judgement. Both facts are already recorded; this reads
them:

* HAND-MADE ROWS are `scoring.hand_made_mask` -- the one definition the source filter,
  Delete and Copy ask. Every edge of such a row is a person's. The trace keys only ever FILE
  such a row (their fits place a new event's edges; they never move a detected one), so
  their edges are marked as hand-made, not as corrected.
* A CORRECTED EDGE is a detected row's edge that is no longer where the detector put it:
  ``start_idx != orig_start_idx`` (or the end likewise). `events.stamp_original_bounds`
  writes that pair once, at detection, and no edit rewrites it, so the comparison covers
  every way an edge moves -- a mask drag, a merge (the merged row carries its first
  fragment's original pair, so the far edge differs) -- and Revert, which puts the edge back
  and so takes its mark away with no bookkeeping of its own.
* A row with no original pair recorded is UNKNOWN, not "original": a table saved before
  the pair existed, or a merge whose first fragment was hand-made (it carries that
  fragment's blank pair). It gets no mark, which the guide says, with ONE exception: a
  merge that took in a hand-made row, whose span is the merge a person asked for, reads
  corrected on both edges. When its pair IS recorded (a detected fragment came first) it
  is compared like any detected row, so only the edge the merge moved is marked.
  ``source == "mixed"`` alone does not say "merged with a hand-made row":
  `events.combine_sources` writes it for any two sources, and a reconciled row where two
  DETECTORS agreed is mixed too. A merge ORs ``added``, so a mixed row with ``added`` set
  is the one that took a hand-made row in.
* A RECONCILED row is its canonical member copied whole -- start, end and original pair
  (`reconcile._assemble_frame`) -- while its ``source`` and ``added`` describe the cluster.
  So for those rows the canonical member's own source says who made the edges: a hand-made
  canonical (a curator's boundary wins the cluster) is a hand-made span, a mixed one a merge
  with a hand-made row, and a detector's is compared through the pair it carried.

Vectorised over the frame (this runs on every curation redraw); nothing here imports Qt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .events import ORIG_END_COL, ORIG_START_COL, SOURCE_COL, SOURCE_MANUAL, SOURCE_MIXED
from .reconcile import CONSENSUS_CANONICAL_SOURCE_COL
from .scoring import added_mask, hand_made_mask

__all__ = ["BoundaryMarks", "boundary_marks"]


@dataclass(frozen=True)
class BoundaryMarks:
    """Per-row booleans, aligned with the frame they were read from."""

    #: Every edge of these rows was drawn by a person.
    hand: np.ndarray
    #: The start edge of a non-hand-made row is not where the detector put it.
    start_corrected: np.ndarray
    #: ...and the end edge.
    end_corrected: np.ndarray


def _orig(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """``(values int64, known bool)`` for one original-bound column (all unknown if absent)."""
    n = len(df)
    if col not in df.columns:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    s = pd.to_numeric(df[col], errors="coerce")
    known = s.notna().to_numpy(dtype=bool)
    vals = s.fillna(0).to_numpy(dtype=np.float64)
    known &= np.isfinite(vals)
    return np.where(known, vals, 0.0).astype(np.int64), known


def _token_is(df: pd.DataFrame, col: str, token: str) -> np.ndarray | None:
    """Per row: does ``col`` read ``token``? ``None`` when the column is absent. Stripped, as
    `events._normalise_token` reads a source cell; a blank or missing cell is never a match."""
    if col not in df.columns:
        return None
    return (df[col].astype("string").str.strip().eq(token)
            .fillna(False).to_numpy(dtype=bool))


def boundary_marks(df: pd.DataFrame) -> BoundaryMarks:
    """Classify every row's two edges. See the module docstring for the rules."""
    n = len(df)
    if n == 0:
        z = np.zeros(0, dtype=bool)
        return BoundaryMarks(z, z, z)
    hand = np.asarray(hand_made_mask(df), dtype=bool)
    start = df["start_idx"].to_numpy(dtype=np.int64)
    end = df["end_idx"].to_numpy(dtype=np.int64)
    o_start, k_start = _orig(df, ORIG_START_COL)
    o_end, k_end = _orig(df, ORIG_END_COL)
    # A half-recorded pair states no span (the rule `events.original_bounds` applies), so a
    # row is compared only where BOTH of its cells are known.
    known = k_start & k_end
    start_corr = known & (start != o_start)
    end_corr = known & (end != o_end)
    canonical_hand = _token_is(df, CONSENSUS_CANONICAL_SOURCE_COL, SOURCE_MANUAL)
    if canonical_hand is not None:
        # Reconciled: the canonical member's source, not the cluster's (module docstring).
        hand = hand | canonical_hand
        merged_with_hand = _token_is(df, CONSENSUS_CANONICAL_SOURCE_COL, SOURCE_MIXED)
    else:
        mixed = _token_is(df, SOURCE_COL, SOURCE_MIXED)
        merged_with_hand = (np.zeros(n, dtype=bool) if mixed is None
                            else mixed & added_mask(df))
    merged_with_hand = merged_with_hand & ~known
    start_corr |= merged_with_hand
    end_corr |= merged_with_hand
    return BoundaryMarks(hand=hand, start_corrected=start_corr & ~hand,
                         end_corrected=end_corr & ~hand)
