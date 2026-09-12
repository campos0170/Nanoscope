"""The Event contract: what a detected translocation *is*.

`EventSet` wraps a pandas DataFrame (one row per event) plus the provenance
needed to reproduce it. The DataFrame is the product -- users can always drop to
`.df` and do their own thing. Nothing here hides the data.

Sign convention, stated once and obeyed everywhere:
    dev = polarity * (current - baseline)
is **positive during an event**, for both blockades (polarity=-1) and
enhancement (polarity=+1). So `depth_pA` is always a positive magnitude, and
`frac_depth` is always positive. The direction lives in the detected polarity,
not in the sign of the features.

Column NAMES keep their ``_pA`` suffix as the frozen cross-module contract even
though the magnitudes carry the source channel's own units (which may be nA or
blank), threaded through separately (PORT_PLAN R9).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Column -> (dtype, unit, description). This IS the schema; readers of the
# EventSet should consult it rather than hardcoding assumptions.
SCHEMA: dict[str, tuple[str, str, str]] = {
    "event_id":    ("int64",   "-",    "index within its source recording"),
    "recording":   ("string",  "-",    "name of the source Recording"),
    "pore_id":     ("string",  "-",    "which physical pore produced this; never pool across pores"),
    "start_idx":   ("int64",   "sample", "first sample of the event"),
    "end_idx":     ("int64",   "sample", "one past the last sample"),
    "t_start_s":   ("float64", "s",    "event start, seconds from recording start"),
    "dwell_s":     ("float64", "s",    "duration (end_idx-start_idx)/fs"),
    "baseline_pA": ("float64", "pA",   "local open-pore current I0 at the event"),
    "depth_pA":    ("float64", "pA",   "excursion magnitude, positive (see module docstring)"),
    "frac_depth":  ("float64", "-",    "depth_pA / |baseline_pA|, i.e. dI/I0"),
    "ecd_pA_s":    ("float64", "pA*s", "integrated charge deficit, sum(dev)*dt"),
    "max_dev_pA":  ("float64", "pA",   "peak excursion within the event"),
    "mean_dev_pA": ("float64", "pA",   "mean excursion within the event"),
    "rise_s":      ("float64", "s",    "10-90% entry time"),
    "fall_s":      ("float64", "s",    "90-10% exit time"),
    "snr":         ("float64", "-",    "depth_pA / baseline noise RMS"),
    "n_levels":    ("int64",   "-",    "resolved sub-states (1 = simple square event)"),
    "filter_limited": ("bool", "-",    "dwell is within ~2x filter rise time; depth is attenuated and dwell is biased long"),
    "accepted":    ("bool",    "-",    "survives curation; exports honour this"),
    "flag":        ("string",  "-",    "';'-joined set of reasons, or '' (see FLAG_SEP)"),
}

REQUIRED = list(SCHEMA.keys())

#: The measured columns. NaN is this app's "not measured" (an unreachable 90 %
#: level, an undefined ``frac_depth``, a blanked baseline); see `demote_infinities`.
FLOAT_COLUMNS = tuple(c for c, (d, _, _) in SCHEMA.items() if d == "float64")


def empty_df() -> pd.DataFrame:
    """An empty frame with the right columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=d) for c, (d, _, _) in SCHEMA.items()})


def demote_infinities(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """``(frame with no +/-Inf in any float column, how many cells were demoted)``.

    NO SCHEMA COLUMN EVER HOLDS +/-Inf. A magnitude that overflowed reads ``NaN``,
    which is this app's own "not measured" sentinel and is already what the same row
    carries in ``snr`` and (for a zero baseline) ``frac_depth``.

    Why the value is changed rather than merely flagged. ``characterize``'s
    ``non_finite`` rule already catches the row -- it is flagged and it is NOT
    accepted, and every shipped export honours ``accepted``, so the ``-inf`` was not
    leaving the machine. What it was doing is worse than leaving: **+/-Inf is the only
    non-value that survives ``dropna()``**. ``df['ecd_pA_s'].dropna().mean()`` is
    ``-inf``; ``describe()``, ``median()``, a histogram, every plotting library and
    every downstream notebook treat NaN as missing and ``Inf`` as a number of infinite
    magnitude. Measured, one ``np.inf`` inside one event of a three-event trace:
    ``ecd_pA_s`` and ``mean_dev_pA`` both ``-inf`` on a row whose ``depth_pA`` read a
    perfectly plausible 20.135567. Untick "Use accepted events only" in the Plots panel
    and that pair reaches `~nanoscope.analysis.stats` intact.

    Only NON-FINITE cells are touched, and only in the float columns; a finite number
    on a poisoned row is left exactly as measured, because it is still what the
    instrument recorded and the ``non_finite`` flag is what says not to trust it.

    Applied at the `EventSet` door, which is the ONE place every producer and every
    restore passes through: `characterize`, `recharacterize_span` (hence the merge, the
    Event Inspector resize and a hand-added event), both detectors, `remeasure`, the
    undo snapshot rebuild and a project reopened from disk. A frame with no infinities
    -- every well-formed recording -- is returned UNCHANGED, same object, so the common
    path pays one ``np.isinf`` per float column and nothing else.
    """
    cols = [c for c in FLOAT_COLUMNS if c in df.columns]
    if not cols or not len(df):
        return df, 0
    bad = {}
    for c in cols:
        v = df[c].to_numpy(dtype=np.float64, copy=False)
        m = np.isinf(v)
        if m.any():
            bad[c] = m
    if not bad:
        return df, 0
    out = df.copy()
    n = 0
    for c, m in bad.items():
        rows = np.flatnonzero(m)
        n += int(rows.size)
        # POSITIONAL, not label-based: an event table's index is not guaranteed unique
        # (a subset, a concat without ``ignore_index``), and ``.loc`` with a repeated
        # label would blank every row that shares it.
        out.iloc[rows, out.columns.get_loc(c)] = np.nan
    return out, n


# ===========================================================================
# THE FLAG VOCABULARY -- a flag is a SET, not a slot
# ===========================================================================
# ``flag`` used to be a single-value slot that every writer OVERWROTE. That is
# what made the defect class possible: a row can be rejected for two reasons at
# once, and whichever writer ran last erased the other. Measured on the owner's
# -600 mV recording, a 2 ms ``max_dwell_s`` limit, and a 5 ms gap merge: 240 rows
# came out of the merge ACCEPTED with dwell > 2 ms -- the worst 24.88 ms, 12.4x
# the declared limit -- because ``merge`` rebuilt each row through
# ``characterize`` (flag always ''), OR-ed ``accepted``, and so un-rejected every
# fragment that had been flagged. Those rows flowed into the table, both
# histograms, the censored dwell fit (whose ``t_max`` IS ``max_dwell_s``) and
# every export.
#
# So the column is a SET of tokens joined by ``FLAG_SEP``, and there are exactly
# two operations on it, both defined here and used by every writer:
#
#   * :func:`combine_flags`      -- union, for a row rebuilt from other rows
#   * :func:`inherited_curation` -- what a rebuilt row must CARRY OVER
#
# Read it with :func:`has_flag`, never with ``df['flag'] == token``: the equality
# silently misses a row that carries the token alongside another one.
FLAG_SEP = ";"

#: What separates a user LABEL's key from its value inside one token (``molecule=apo``).
#: Defined here rather than in :mod:`nanoscope.analysis.labels` because
#: :func:`inherited_curation` has to tell a label from a rejection REASON, and it cannot
#: import labels (labels imports this module).
LABEL_SEP = "="

#: ``|baseline_pA|`` was 0, so ``frac_depth`` is undefined for this row.
FLAG_ZERO_BASELINE = "zero_baseline"
#: Longer than the declared ``max_dwell_s``: a clog, kept but not accepted.
FLAG_MAX_DWELL = "dwell>max_dwell_s"
#: Still above threshold at the first/last analysed sample -- cut off by the edge
#: of the span rather than by the pore, so dwell/depth/baseline are not
#: measurements of the event. See ``characterize(clip_bounds=...)``.
FLAG_CLIPPED = "clipped_by_boundary"
#: A sample this row was measured from -- or a magnitude computed out of one --
#: is NaN or +/-Inf, so the row's numbers are not measurements. Measured, one
#: ``np.inf`` inside one event of a 3-event trace::
#:
#:     clean               depth=20.140960  ecd= 0.12069  mean_dev= 20.115
#:     one +inf inside ev0 depth=20.135567  ecd=-inf      mean_dev=-inf
#:
#: -- and the row came back ``accepted=True, flag=''``, so a plausible-looking
#: ``depth_pA`` entered the table beside two ``-inf`` columns, exported literal
#: ``-inf`` into the frozen schema and detonated hundreds of steps later in
#: ``stats._clean_1d`` or ``figures._column`` with an error that named no row.
FLAG_NON_FINITE = "non_finite"

#: THE CURATOR'S OWN MARK — "I looked at this one and want to find it again". The only
#: token in this vocabulary a PERSON writes (the ``f`` key on the event table); every
#: other one is a statement the app derived. It is deliberately NOT in
#: :data:`DERIVED_FLAGS`, so a merge, a mask resize or a hand-added event carries it
#: through :func:`inherited_curation` verbatim — and so a re-detection that would erase
#: it is counted as destroying hand curation, which is what puts the warning and the
#: undo step on screen.
#:
#: It lives here rather than as a string literal in the GUI because that is what the
#: rest of this vocabulary is for: the literal spelling was in ``gui/panels.py``, written
#: straight over the cell instead of combined into it, so marking a boundary-clipped
#: event would have erased ``clipped_by_boundary`` and left a rejected row with no reason
#: — C21's failure, at the one call site C21 did not reach.
FLAG_CURATED = "flagged"

#: Tokens that are statements about the CURRENT measurement and are therefore
#: RE-DERIVED whenever a row is rebuilt, rather than inherited from the rows it
#: replaces. Everything else (a curator's own note) is inherited verbatim.
#:
#: This split is what makes both halves of the guarantee honest at once: a merged
#: row that is still too long is re-flagged on its own merits, and a resized row
#: that no longer touches the boundary stops claiming it does -- instead of C21's
#: "rejected with NO REASON", which is invisible to every filter and count.
DERIVED_FLAGS = frozenset(
    {FLAG_ZERO_BASELINE, FLAG_MAX_DWELL, FLAG_CLIPPED, FLAG_NON_FINITE}
)

#: Tokens that EXPLAIN AN AUTOMATIC REJECTION -- the three rules
#: :func:`~nanoscope.analysis.characterize.characterize` actually rejects on
#: (``accepted = ~(too_long | clipped | non_finite)``), and no others.
#:
#: THIS IS A DIFFERENT QUESTION FROM :data:`DERIVED_FLAGS`, and one set answering
#: both is what this pair exists to end. "Will a rebuild re-derive this token?" and
#: "does this token say why the app rejected the row?" have the same answer for
#: three tokens and OPPOSITE answers for ``zero_baseline``: it is re-derived (so it
#: belongs in ``DERIVED_FLAGS``) but it never rejected anything (so it explains no
#: rejection). Read as one set, a row a PERSON rejected that also happened to carry
#: ``zero_baseline`` looked automatic, and:
#:
#:   * :func:`inherited_curation` un-rejected it on every merge, mask resize and
#:     re-measure -- the hand rejection silently gone;
#:   * ``core.history._hand_curation`` counted it as **0** hand rejections, so the
#:     "this will destroy N hand rejections" warning -- the thing that exists to
#:     announce exactly this loss -- stayed silent about it.
#:
#: The trigger (``|baseline_pA| == 0.0`` exactly) is not reachable on a real ionic
#: current, but the miscount is real today whenever a rejected row carries any
#: derived-but-non-rejecting token beside its reason. Both readers now ask this
#: question through :func:`is_automatic_rejection`, so there is one answer.
AUTOMATIC_REJECTION_FLAGS = frozenset(
    {FLAG_MAX_DWELL, FLAG_CLIPPED, FLAG_NON_FINITE}
)


def is_automatic_rejection(
    flag: Any,
    *,
    derived: frozenset[str] = DERIVED_FLAGS,
    rejecting: frozenset[str] = AUTOMATIC_REJECTION_FLAGS,
) -> bool:
    """Is a rejection carrying ``flag`` fully explained by rules a rebuild re-applies?

    THE one definition of "the app rejected this row, not a person". Called by
    :func:`inherited_curation` (which drops such a rejection so the new span can be
    re-judged), by ``core.history._hand_curation`` (which must not count it as a
    person's work) and by
    :func:`~nanoscope.analysis.remeasure.merge_late_curation` (which must not read
    it as a curator's late decision). Three readers, one rule.

    A token counts as a REASON here unless it is
    * a label (``molecule=apo`` says what an event is, never why it was rejected), or
    * a derived token that rejects nothing (``zero_baseline`` is a caveat about
      ``frac_depth``; ``characterize`` accepts the row anyway).

    Neither can make a person's rejection look automatic, and neither can hide one
    that is: a row flagged ``zero_baseline;clipped_by_boundary`` is still automatic,
    because ``clipped_by_boundary`` is a real and re-derivable reason.

    Args:
        flag: One flag cell.
        derived: Tokens this rebuild will re-derive. Narrowed by
            :func:`~nanoscope.analysis.characterize.recharacterize_span` when
            ``max_dwell_s`` is unknown -- a reason that cannot be re-derived has to
            be inherited, rejection and all.
        rejecting: Tokens that explain a rejection at all.

    Returns:
        ``True`` only when there is at least one real reason and every one of them
        will be re-derived. ``False`` for no reason at all (a bare Reject click) and
        for any curator's own token.
    """
    caveats = derived - rejecting
    reasons = [
        t for t in flag_tokens(flag) if LABEL_SEP not in t and t not in caveats
    ]
    return bool(reasons) and all(t in derived for t in reasons)


def flag_tokens(flag: Any) -> tuple[str, ...]:
    """The reason tokens in one ``flag`` cell: split, stripped, de-duplicated.

    ``''``, ``None``, ``NaN`` and ``pd.NA`` all mean "no reason" and give ``()``.
    """
    if flag is None or flag is pd.NA:
        return ()
    if isinstance(flag, float) and np.isnan(flag):
        return ()
    text = str(flag)
    if not text or text == "<NA>":
        return ()
    out: list[str] = []
    for part in text.split(FLAG_SEP):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return tuple(out)


def combine_flags(*flags: Any) -> str:
    """Union of every argument's tokens, in first-appearance order.

    This is the ONLY way two flags are ever put together. ``combine_flags('a',
    'b;a')`` is ``'a;b'``; combining with ``''`` is the identity.
    """
    out: list[str] = []
    for flag in flags:
        for token in flag_tokens(flag):
            if token not in out:
                out.append(token)
    return FLAG_SEP.join(out)


def has_flag(flags: Any, token: str) -> Any:
    """Does this flag cell (or column) carry ``token``?

    Scalar in -> ``bool``; array-like in -> ``np.ndarray`` of bool. Use this
    instead of ``df['flag'] == token``, which misses ``'zero_baseline;'+token``.
    """
    if isinstance(flags, str) or flags is None or flags is pd.NA:
        return token in flag_tokens(flags)
    if isinstance(flags, float):
        return token in flag_tokens(flags)
    values = list(np.asarray(flags, dtype=object).ravel())
    return np.array([token in flag_tokens(v) for v in values], dtype=bool)


def inherited_curation(
    flags: Any,
    accepted: Any,
    *,
    derived: frozenset[str] = DERIVED_FLAGS,
) -> tuple[str, bool]:
    """What a REBUILT row carries over from the one-or-many rows it replaces.

    A merge, an Event Inspector mask resize and a hand-added event all destroy a
    row and build a new one. Two things must survive that, and one must not:

    * **Curator's reasons survive.** Any token outside ``derived`` is unioned in.
    * **A hand rejection survives.** A row with ``accepted=False`` whose reasons
      are NOT fully explained by ``derived`` tokens (including the no-reason-at-all
      case, which is what a click on the Reject checkbox looks like) was rejected
      by a person; the rebuilt row stays rejected. ``accepted`` is combined
      CONSERVATIVELY -- AND, never OR. One rejected fragment rejects the merge.
    * **An automatic rejection does NOT survive as a rejection.** A row rejected
      solely because it was too long, or clipped, or measured from a non-finite
      sample, or had a zero baseline, is
      re-judged from the new span by :func:`~nanoscope.analysis.characterize.characterize`,
      which owns all four rules. Inheriting it as well would leave a resized event
      rejected with its reason erased -- rejected, invisible to the very filter
      that is supposed to find it.

    Args:
        flags: One flag cell, or an iterable of them (the run being merged).
        accepted: The matching ``accepted`` value(s).
        derived: Tokens the rebuild re-derives. Narrow it when a rule cannot be
            re-derived -- e.g. ``max_dwell_s`` is unknown, so its token has to be
            inherited or the reason is lost.

    Returns:
        ``(flag, accepted)`` to hand to ``characterize`` as ``inherit_flag`` /
        ``inherit_accepted``.
    """
    if isinstance(flags, str) or flags is None or flags is pd.NA:
        flag_list: list[Any] = [flags]
        accept_list: list[Any] = [accepted]
    else:
        flag_list = list(np.asarray(flags, dtype=object).ravel())
        accept_list = list(np.asarray(accepted, dtype=object).ravel())
    if len(flag_list) != len(accept_list):
        raise ValueError(
            f"flags ({len(flag_list)}) and accepted ({len(accept_list)}) differ in length"
        )

    carried: list[str] = []
    keep_accepted = True
    for flag, ok in zip(flag_list, accept_list):
        tokens = flag_tokens(flag)
        for token in tokens:
            if token not in derived and token not in carried:
                carried.append(token)
        if not bool(ok) and not is_automatic_rejection(flag, derived=derived):
            # Fully explained by re-derivable REJECTION rules => automatic =>
            # re-judged, not inherited. Anything else (a bare reject, a curator's own
            # token, a caveat like `zero_baseline` that rejected nothing) is a
            # decision this rebuild has no standing to reverse.
            #
            # The rule itself lives in `is_automatic_rejection` rather than here
            # because `core.history._hand_curation` and
            # `remeasure.merge_late_curation` ask the SAME question, and asked it
            # two other ways: history counted a `zero_baseline` hand rejection as
            # zero hand rejections, and a label beside a derived reason as one.
            keep_accepted = False
    return FLAG_SEP.join(carried), keep_accepted


# ===========================================================================
# THE ANALYSED SPAN -- the thing ``clipped_by_boundary`` is judged AGAINST
# ===========================================================================
# ``clipped_by_boundary`` is not a property of a row on its own. It is a
# relation between the row and the span the detector actually looked at, and a
# row that does not carry that span cannot be re-judged: on the first rebuild it
# is measured against the WHOLE RECORDING, finds itself nowhere near an edge,
# and -- because the token is a DERIVED flag -- drops it and un-rejects itself.
#
# Measured before this existed, a 600-sample event with the ML range starting
# mid-event, the row carrying ``flag='clipped_by_boundary', accepted=False``::
#
#     as detected    [200300,200600): accepted=True  flag=''  dwell=6.000 ms
#     1-sample nudge [200300,200599): accepted=True  flag=''  dwell=5.980 ms
#
# No edit was required -- a plain ``merge_by_gap`` lost it -- and a dwell that is
# purely an artifact of where the user drew the zoom rectangle entered the table,
# both histograms, the censored dwell fit and every export as a fully-measured,
# accepted event while provenance still claimed ``n_clipped_boundary=1``.
#
# So the span TRAVELS WITH THE ROWS, in the EventSet's own provenance, under the
# two keys below. There is one writer (:func:`span_provenance`, used by both
# detectors) and one reader (:func:`analysed_span`, used by
# :func:`~nanoscope.analysis.characterize.recharacterize_span`, which every
# rebuild path already goes through). A fourth rebuild path cannot forget the
# span, because there is no argument for it to forget.

#: First sample of the analysed span, in the SAME coordinates as the rows'
#: ``start_idx`` (absolute in the document the rows belong to -- segment-local for
#: a segment, since a segment is its own document with its own indices).
ANALYSED_START_KEY = "analysed_start_idx"
#: One past the last analysed sample, same coordinates.
ANALYSED_STOP_KEY = "analysed_stop_idx"

#: The ML detector stamped its zoom rectangle under these names before the span
#: became part of the contract, and nothing read them. Read as a fallback so an ML
#: result saved by an older build still knows the span it was detected in.
_LEGACY_SPAN_KEYS = ("range_start", "range_stop")


def span_provenance(start: int, stop: int) -> dict[str, int]:
    """The provenance entries recording the analysed span. The ONE writer.

    Both detectors stamp their span through this, so the two keys are spelled
    once and a reader can rely on them travelling together.

    Args:
        start: First analysed sample (0 for a whole-recording detection).
        stop: One past the last analysed sample.

    Raises:
        ValueError: if the span is empty or reversed -- a detection that analysed
            nothing has no rows to judge, and a reversed span would silently flag
            every row.
    """
    start, stop = int(start), int(stop)
    if stop <= start:
        raise ValueError(
            f"analysed span [{start}, {stop}) is empty or reversed; a span that "
            "analysed no samples cannot have produced events"
        )
    return {ANALYSED_START_KEY: start, ANALYSED_STOP_KEY: stop}


def analysed_span(provenance: Any, n_fallback: int) -> tuple[int, int]:
    """``(start, stop)`` of the span the rows were detected in. The ONE reader.

    Resolution order, and why:

    1. :data:`ANALYSED_START_KEY` / :data:`ANALYSED_STOP_KEY` -- what both
       detectors stamp today.
    2. ``range_start`` / ``range_stop`` -- what the ML detector stamped before
       this contract existed, so a stored ML sub-range result restored from a
       project is still judged against its zoom rectangle.
    3. ``(0, n_fallback)`` -- a legacy EventSet carries no span at all. The
       whole-recording span is the right answer for every pre-existing SCHMITT
       result (``detect_events`` has always been handed the whole analysed array)
       and it is exactly what the code did before the span existed, so no legacy
       row changes its judgement. The one legacy case it gets wrong -- an ML
       sub-range result -- is covered by rule 2.

    Args:
        provenance: The detection's provenance mapping, or None.
        n_fallback: Length of the signal the indices refer to, used by rule 3.

    Returns:
        A half-open ``(start, stop)`` with ``stop > start``. A stored pair that is
        empty, reversed or not integral is ignored rather than trusted -- it would
        flag every row in the table.
    """
    n_fallback = int(n_fallback)
    if isinstance(provenance, Mapping):
        for lo_key, hi_key in (
            (ANALYSED_START_KEY, ANALYSED_STOP_KEY),
            _LEGACY_SPAN_KEYS,
        ):
            lo, hi = provenance.get(lo_key), provenance.get(hi_key)
            if lo is None or hi is None:
                continue
            try:
                lo_i, hi_i = int(lo), int(hi)
            except (TypeError, ValueError):
                continue
            # A sample index is an integer. ``int()`` TRUNCATES, so a stored 1.5
            # would silently become the clip bound 1 and be trusted -- the one case
            # the paragraph below claimed was ignored and was not. Nothing writes a
            # float today (``span_provenance`` casts both), so this closes the
            # contract rather than a live path.
            try:
                if float(lo) != lo_i or float(hi) != hi_i:
                    continue
            except (TypeError, ValueError):
                continue
            if hi_i > lo_i:
                return lo_i, hi_i
    return 0, n_fallback


#: Provenance key holding how many rows are boundary-clipped. Spelled once so the
#: producers and :class:`EventSet`'s reconciliation cannot drift apart.
CLIPPED_COUNT_KEY = "n_clipped_boundary"


def n_clipped_boundary(df: pd.DataFrame) -> int:
    """How many rows say they are boundary-clipped. The count's ONE definition.

    ``n_clipped_boundary`` in provenance is not an independent fact -- it is a
    statement about the rows, and the hunt's tell was provenance claiming a
    clipped event that zero rows admitted to. Everything that reports the number
    derives it from here, and :class:`EventSet` re-derives it at construction so
    the disagreement cannot be built in the first place.
    """
    if "flag" not in df.columns or not len(df):
        return 0
    return int(np.asarray(has_flag(df["flag"], FLAG_CLIPPED), dtype=bool).sum())


def n_filter_limited(df: pd.DataFrame) -> int:
    """How many rows say they are filter-limited. The count's ONE definition
    (the sibling of :func:`n_clipped_boundary`, for the ``filter_limited`` column)."""
    if "filter_limited" not in df.columns or not len(df):
        return 0
    return int(np.asarray(df["filter_limited"], dtype=bool).sum())


#: The provenance keys that are STATEMENTS ABOUT THE ROWS beside them, each with the
#: one reader that derives it from the frame. ``n_events`` is the size of the table
#: (what `subpop_report.rows_in_hand` subtracts the rows in hand from to say how
#: many were curated OUT); the other two are the counts `modules._build_warnings`
#: composes into "3 of 251 event(s) ... filter-limited" and "1 of 317 event(s) were
#: still above threshold at the first or last analysed sample".
ROW_COUNT_KEYS: dict[str, Callable[[pd.DataFrame], int]] = {
    "n_events": lambda df: int(len(df)),
    "n_filter_limited": n_filter_limited,
    CLIPPED_COUNT_KEY: n_clipped_boundary,
}


def reconcile_row_counts(prov: Mapping[str, Any] | None, df: pd.DataFrame) -> dict[str, Any]:
    """``prov`` with every row-derived count re-read from ``df`` -- THE ONE RULE for a
    frame that has changed shape under its provenance.

    Sweep 7 measured the same defect at three doors, each of which had fixed ONE key
    at ITS call site and left the others stale:

      * ``Merge all`` (317 -> 247 rows on the owner's -600 mV file) re-read
        ``n_clipped_boundary`` and carried ``n_events = 317`` forward, so the
        on-screen warning read "1 of 317 event(s)" over a 247-row table and the
        subpopulation report printed "CURATION: 71 of the detector's 317 region(s)
        were curated OUT" for a user who had curated out exactly one -- a fabricated
        number in the one line that exists to stop a fabricated number (S7-DET4);
      * ``Re-measure depths`` re-read ``n_clipped_boundary`` (remeasure.py) and not
        ``n_filter_limited``, so provenance said 3 over a table whose rows flagged 2,
        and the exported ``.meta.json`` carried the disagreement (S7 MISSED-2);
      * a hand-added missed event grew the table by one and restamped nothing.

    ``detect.py`` states the rule the producers already follow -- "Report what the
    ROWS say, not what a parallel computation predicted" -- and this is that rule for
    every path that INSTALLS a changed frame: the merge engine, the re-measure engine
    and `modules.reconcile_result` all call it, so the next door cannot fix one key
    and forget the rest.

    Only a key the provenance ALREADY carries is corrected -- a set that never had
    one does not grow it (the same additive-by-choice rule `EventSet.provenance`
    follows for ``n_clipped_boundary``), and a synthetic provenance in a test stays
    the dict the test wrote. Always a new dict; the caller's object is never mutated.
    """
    out = dict(prov or {})
    for key, read in ROW_COUNT_KEYS.items():
        if key in out:
            out[key] = read(df)
    return out


# ===========================================================================
# THE DETECTOR'S OWN BOUNDS -- what "revert to the model's regions" reverts TO
# ===========================================================================
# ``start_idx`` / ``end_idx`` are EDITED IN PLACE. The Event Inspector's mask drag
# writes them straight onto the shared frame (that sharing is the whole design: an
# edit in the browser is an edit in the table), a merge rebuilds them, and the undo
# recorder snapshots the frame AFTER the fact. So the moment a curator drags an edge,
# the number the DETECTOR produced is gone from every durable surface in the app --
# the frame, the result store and the project file all hold the edited value and
# nothing holds the original. "Revert to the model's regions" therefore needs
# somewhere to revert TO, and this is it.
#
# TWO COLUMNS, WRITTEN ONCE, NEVER BY AN EDIT. :func:`stamp_original_bounds` is the
# ONE writer and it is called at DETECTION, on a frame whose bounds are by definition
# the detector's own. Every later path -- the mask drag, the merge, the undo restore,
# a project round-trip -- carries the pair through untouched or not at all; none of
# them may write it, because a value written after an edit is not a record of what the
# detector said, it is a record of what the last edit said, wearing the detector's name.
#
# *** THEY ARE NOT IN :data:`SCHEMA`, AND THAT IS THE COMPATIBILITY MECHANISM. ***
# ``SCHEMA`` is REQUIRED (``EventSet.__post_init__`` raises on a missing column) and
# ``characterize`` ends with ``df[list(SCHEMA)]``, which DROPS anything outside it. Put
# here, the pair would (a) make every detection saved before this change unloadable and
# (b) be erased by the very rebuild the Inspector's edit path runs. Declared separately,
# they are additive: a frame that has them keeps them (the Inspector writes measured
# columns BY NAME, so an untouched column survives an edit), and a frame that does not
# have them reads "unknown" -- which is the honest answer for a detection saved before
# this existed.
#
# NULLABLE ``Int64``, NOT ``int64``. "Unknown" has to be representable PER ROW: a table
# restored from an old project and then given a hand-added event has rows of both kinds,
# and int64 has no value that means "not recorded". ``pd.NA`` does, and it does not
# collide with a real sample index the way a -1 sentinel would once somebody subtracts.

#: ``start_idx`` as the detector wrote it. See the block above.
ORIG_START_COL = "orig_start_idx"
#: ``end_idx`` as the detector wrote it.
ORIG_END_COL = "orig_end_idx"

#: Column -> (dtype, unit, description), the same shape as :data:`SCHEMA` so a reader can
#: consult one or the other without a second convention. Deliberately NOT merged into
#: ``SCHEMA``; see the block above for why that is the whole point.
ORIGINAL_BOUNDS_SCHEMA: dict[str, tuple[str, str, str]] = {
    ORIG_START_COL: ("Int64", "sample",
                     "first sample of the event AS THE DETECTOR PRODUCED IT, never "
                     "rewritten by an edit; pd.NA when the detection predates this column"),
    ORIG_END_COL:   ("Int64", "sample",
                     "one past the last sample as the detector produced it, never "
                     "rewritten by an edit; pd.NA when unknown"),
}


def stamp_original_bounds(df: pd.DataFrame, *, overwrite: bool = False) -> pd.DataFrame:
    """Record ``start_idx``/``end_idx`` as the detector's own bounds. The ONE writer.

    Call this at DETECTION and nowhere else. On a frame that has already been edited it
    would write the EDIT under the detector's name, which is a fabricated provenance
    claim and strictly worse than the honest "unknown" an unstamped frame reads as -- so
    an existing pair is left exactly alone unless ``overwrite`` is passed, and the only
    caller entitled to pass it is a re-detection that genuinely replaced the rows.

    Returns a COPY with the two columns appended; the input frame is not touched. A frame
    missing ``start_idx``/``end_idx`` is returned unchanged, because there is nothing to
    record.
    """
    if not {"start_idx", "end_idx"} <= set(df.columns):
        return df
    if not overwrite and has_original_bounds(df):
        return df
    out = df.copy()
    for col, src in ((ORIG_START_COL, "start_idx"), (ORIG_END_COL, "end_idx")):
        out[col] = pd.array(
            np.asarray(out[src], dtype=np.int64) if len(out) else np.empty(0, np.int64),
            dtype="Int64")
    return out


def has_original_bounds(df: Any) -> bool:
    """Does this frame carry the detector's own bounds at all?

    ``True`` only when BOTH columns are present -- a half-stamped frame states no span
    and must read exactly as unstamped. Says nothing about any individual row: a stamped
    frame can still hold ``pd.NA`` rows (a hand-added event has no detector bounds), which
    is :func:`original_bounds`' question.
    """
    cols = getattr(df, "columns", None)
    return cols is not None and ORIG_START_COL in cols and ORIG_END_COL in cols


def original_bounds(row: Any) -> tuple[int, int] | None:
    """The detector's ``(start, end)`` for ONE row, or ``None`` when it is unknown.

    The ONE reader, so "is Revert available for this event?" has a single answer and the
    GUI cannot invent a second. ``None`` for every way the record can be absent or
    unusable -- the columns were never written (a detection from before this change), the
    cell is ``pd.NA``/NaN (a hand-added event, which no detector produced), the value is
    not integral, or the recorded span is empty or reversed. A caller that gets ``None``
    must DISABLE the revert, never fall back to the current bounds: reverting an edited
    event to its own edited bounds is a no-op dressed up as a restoration.

    Args:
        row: One row -- a ``pd.Series``, a mapping, or anything with ``[]`` access.
    """
    try:
        lo_raw, hi_raw = row[ORIG_START_COL], row[ORIG_END_COL]
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    out: list[int] = []
    for value in (lo_raw, hi_raw):
        if value is None or value is pd.NA:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number) or number != int(number):
            return None
        out.append(int(number))
    lo, hi = out
    # An empty or reversed span is not a span the detector can have produced, and handing
    # it to a revert would collapse the event. Refuse it the way `analysed_span` refuses a
    # stored pair it cannot trust, rather than repairing it into something plausible.
    return (lo, hi) if hi > lo else None

# ===========================================================================
# WHO FOUND IT, AND WHO HAS RULED ON IT -- ``source`` + ``review``
# ===========================================================================
# Section 7 of the owner's notes: "Automatic detection should generate candidate
# events; user curation should determine the final dataset", and "each event should
# store -- Source: conventional detector / ML detector / manual. Status: unreviewed /
# accepted / rejected."
#
# NEITHER FACT IS PER-ROW TODAY, measured on this tree:
#
#   * WHICH detector authored a row is knowable only from the ``module_id`` on the
#     enclosing `~nanoscope.core.analysis.AnalysisResult` -- i.e. per TABLE. That is
#     enough right up until a table stops being one detector's output, which it does
#     the first time somebody hand-adds a missed event (``AnalysisPanel`` appends the
#     row to the SAME result under the SAME module id) and again the moment two
#     detections are compared or pooled. ``scoring.ADDED_COLUMN`` already marks the
#     hand-added rows, so the manual/automatic split exists; the automatic side has
#     no per-row name at all.
#   * ``accepted`` is a BOOLEAN, so "I looked at this and kept it" and "nobody has
#     ever looked at this" are the same cell. A detection arrives with every row
#     already ``accepted=True``, which is exactly why "curation determines the final
#     dataset" is not true of this app yet: the final dataset is the detector's,
#     minus whatever the user happened to reject.
#
# *** THE SHAPE: REVIEWEDNESS IS STORED, STATUS IS DERIVED. ***
# ``accepted`` is read by every export, every figure, every fit and `EventSet.accepted`
# itself, and it keeps its exact present meaning: does this row belong to the final
# dataset. The NEW column stores only the half that is missing -- has a person ruled on
# this row at all -- and the owner's three-state status is COMPUTED from the pair by
# :func:`event_status`. Storing a second three-valued column beside ``accepted`` was the
# alternative, and it is how this app already earned a provenance count that disagreed
# with its own table (see `reconcile_row_counts` and the ``n_clipped_boundary`` note on
# `EventSet.provenance`): two writers for one fact drift, and the drift is silent.
#
# *** NOT IN :data:`SCHEMA`, FOR THE SAME TWO REASONS AS THE BOUNDS ABOVE. ***
# ``SCHEMA`` is ``REQUIRED`` -- `EventSet.__post_init__` raises on a missing column, so
# a project saved before this change would stop loading -- and ``characterize`` ends
# with ``df[list(SCHEMA)]``, which DROPS anything outside it, so every rebuild path
# (the Inspector's mask drag, the merge, the re-measure) must re-apply these two
# deliberately. Declared here instead, they are additive: a frame that has them keeps
# them, and a frame that does not reads UNKNOWN, which is the honest answer for a
# detection saved before this existed. The owner was promised he would not have to
# migrate every saved project, and that promise is this declaration.
#
# *** BOTH COLUMNS ARE pandas ``string``, AND THE DTYPE IS LOAD-BEARING. ***
# Reviewedness is a yes/no with an "unknown", so ``boolean`` (pandas' nullable bool) is
# the obvious dtype and it is the wrong one. RE-MEASURED ON THIS TREE (pandas 2.2.3),
# because the first version of this paragraph named a crash that this very change has
# since removed, and a stale reason is worse than none:
#
#   * The pandas fact is unchanged: ``pd.api.types.is_bool_dtype`` is True for
#     ``boolean``, and ``pd.Series([True, pd.NA], dtype='boolean').to_numpy(dtype=bool)``
#     raises ``ValueError: cannot convert to 'bool'-dtype NumPy array with missing
#     values``.
#   * ``merge._merged_row``'s generic carry -- ``if is_bool_dtype(series):
#     series.to_numpy(dtype=bool).any()`` -- STILL raises that ValueError out of a Qt
#     slot, measured just now on a merged run carrying a ``boolean`` extra column with
#     one ``pd.NA``. That branch is live and it is the hazard.
#   * It is no longer the hazard FOR THESE TWO COLUMNS, and only because this change
#     put ``if col in (SOURCE_COL, REVIEW_COL)`` ahead of it. Measured: a merge over a
#     ``boolean``-dtyped ``review`` holding ``pd.NA`` does not raise -- it comes back
#     ``<NA>``, still ``boolean``. So "a boolean review would crash Merge all" is true
#     of the tree this design was written against and FALSE of the tree it ships in.
#     The name-routing is the only thing standing between them, i.e. one rename away.
#
# What is true here and now, and what the dtype actually rests on: **the vocabulary is
# words, and it is three-valued**. ``review`` holds 'reviewed' / 'unreviewed' / ``pd.NA``
# with ``pd.NA`` meaning "nobody recorded which" -- a state a ``boolean`` can only spell
# as the same missing value it would use for False. And every reader here refuses a token
# outside :data:`REVIEW_VALUES` (`review_of`: an unrecognised spelling is not evidence
# that a person ruled), so a ``boolean`` column does not fail loudly, it reads
# ALL-UNKNOWN: measured, ``event_status`` over a ``boolean`` ``review`` of
# ``[True, pd.NA]`` returns ``['unknown', 'unknown']`` -- the reviewedness is gone and
# nothing says so. Parquet keeps ``boolean`` as ``boolean`` (measured), so such a column
# persists and stays invisible; that is why `core.project._settle_source_status_dtypes`
# exists and why it keeps the VALUES verbatim. ``string`` round-trips losslessly through
# the project's parquet member (``string`` in, ``string`` out, ``pd.NA`` preserved),
# takes ``merge``'s non-boolean branch even if the name-routing is ever removed, and
# cannot be read as a truthy bool by accident the way ``Int64`` 0/1 invites. The cost is
# that the cells spell their values out, which is also what the event table wants to
# display.
#
# *** ``df['source']`` IS NOT ``provenance['source']``, AND THE COLLISION IS REAL. ***
# Detection provenance has carried a key called ``source`` since before this column
# existed and it holds the RECORDING NAME (``detect.py``: ``"source": recording or
# None``) -- measured present on a live detection of the owner's -1000 mV file. This
# COLUMN answers a different question: which detector authored this ROW. The name is the
# owner's own word for it ("Source: conventional detector / ML detector / manual") and is
# kept for that reason, but the two land side by side in the events export -- the CSV gets
# the column, the ``.meta.json`` sidecar gets the provenance key -- so anything that
# renders either one must say WHICH source it means. ``test_src_events_contract.py``
# pins the distinction so a later reader cannot quietly fold them together.

#: Which detector authored this row -- verbatim the ``module_id`` of the
#: `~nanoscope.core.analysis.AnalysisResult` that produced it, or :data:`SOURCE_MANUAL`.
SOURCE_COL = "source"
#: Has a person ruled on this row: :data:`REVIEW_DONE`, :data:`REVIEW_NONE`, or ``pd.NA``.
REVIEW_COL = "review"

#: A row a person put there by hand. NOT a module id, and it cannot collide with one:
#: every detector id this app registers is dotted (``detect.schmitt``) and this is not.
SOURCE_MANUAL = "manual"
#: A row assembled from fragments with more than one source (a merge across a detector
#: run and a hand-added event). Named rather than resolved, because picking one of the
#: two contributors would be a claim nobody made -- see :func:`combine_sources`.
SOURCE_MIXED = "mixed"

#: A person has ruled on this row; ``accepted`` says which way.
REVIEW_DONE = "reviewed"
#: A candidate nobody has ruled on yet. This is what a fresh detection stamps.
REVIEW_NONE = "unreviewed"
#: The two tokens :data:`REVIEW_COL` may hold. ``pd.NA`` -- not in this set -- is the
#: third state, and it means the reviewedness was never recorded, not that it is False.
REVIEW_VALUES = (REVIEW_DONE, REVIEW_NONE)

#: The owner's three source categories ("conventional detector / ML detector / manual"),
#: plus the two honest non-answers. :func:`source_kind` is the only reader.
KIND_CONVENTIONAL = "conventional"
KIND_ML = "ml"
KIND_MANUAL = "manual"
KIND_MIXED = "mixed"
KIND_UNKNOWN = "unknown"

#: module id -> which KIND of detector it is. THE one place this project says that
#: Schmitt and CUSUM are threshold detectors and Keras is the model, and the only new
#: vocabulary this change introduces. The ids themselves are NOT re-spelled here as a
#: source token: a detector row stores its ``module_id`` verbatim, so
#: `~nanoscope.core.sources.detector_name` -- already the one writer of "Schmitt" /
#: "CUSUM" / "Keras" for a legend -- reads a stored ``source`` cell with no second
#: mapping to keep in step. ``tests/analysis/test_src_events_contract.py`` asserts this
#: covers every id in ``sources.DETECTION_MODULE_IDS``, so a fourth detector cannot be
#: registered without landing here.
SOURCE_KIND_BY_MODULE: dict[str, str] = {
    "detect.schmitt": KIND_CONVENTIONAL,
    "detect.cusum": KIND_CONVENTIONAL,
    "detect.keras": KIND_ML,
    # A derived table may contain conventional, ML and manual contributors. Calling
    # the result itself conventional or ML would erase that fact; ``mixed`` is the
    # existing honest category for exactly this case. Individual rows still carry
    # their precise ``source`` and reconciliation membership provenance.
    "detect.reconciled": KIND_MIXED,
}

#: Column -> (dtype, unit, description), the same shape as :data:`SCHEMA` and
#: :data:`ORIGINAL_BOUNDS_SCHEMA` so a reader can consult any of the three without a
#: second convention. Deliberately NOT merged into ``SCHEMA``; see the block above.
SOURCE_STATUS_SCHEMA: dict[str, tuple[str, str, str]] = {
    SOURCE_COL: ("string", "-",
                 "which detector authored this row: the producing AnalysisResult's "
                 "module_id verbatim, 'manual' for a hand-added event, 'mixed' for a "
                 "merge across sources; pd.NA when the row predates this column"),
    REVIEW_COL: ("string", "-",
                 "'reviewed' once a person has ruled on this row, 'unreviewed' for a "
                 "candidate nobody has judged; pd.NA when nobody recorded which. "
                 "Reviewedness ONLY -- which way they ruled is 'accepted'"),
}

#: The derived three-state the owner asked for, plus the two states the app can
#: actually be in and he did not name. See :func:`event_status`.
STATUS_UNREVIEWED = "unreviewed"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
#: The DETECTOR rejected this row, not a person: ``accepted`` is False and the ``flag``
#: is fully explained by :data:`AUTOMATIC_REJECTION_FLAGS`. Not one of the owner's three
#: because it is not a curation outcome at all -- `scoring.ScoreResult.n_auto_rejected`
#: already holds these rows out of TP/FP/FN for exactly that reason ("the detector
#: emitted them and then said itself that they are not measurements of an event").
STATUS_AUTO_REJECTED = "auto-rejected"
#: Nobody recorded whether a person ruled on this row -- a detection saved before this
#: column existed. NOT "unreviewed": that would claim a fact about a curator's attention
#: that no one ever wrote down.
STATUS_UNKNOWN = "unknown"


def _normalise_token(value: Any) -> str | None:
    """ONE normalisation for a ``source`` / ``review`` cell: the stripped text, or
    ``None`` for every shape of "nothing was recorded here".

    THE single normaliser, and it exists because there were two. ``review_of`` stripped
    its cell and the vectorised ``event_status`` did not, so a ``review`` of
    ``' reviewed'`` -- one leading space, which is all it takes to arrive from a CSV
    round-trip or a hand-edited project -- read ACCEPTED per row and UNKNOWN in the same
    table's column; ``source_of`` stripped while ``source_disagrees_with_added``
    compared raw, so ``' manual '`` on an ``added`` row was reported as a contradiction
    that is not there. A per-row read-out, a table column and a summary count are three
    shapes of one question, and they may not each decide for themselves what a cell says.

    ``None`` (never ``''``) is the one spelling of unknown: ``pd.NA``, ``None``, ``NaN``,
    a whitespace-only cell and an absent column all mean nobody recorded this, and a
    blank string must never read as a value -- :func:`ensure_source_status` refuses to
    WRITE one for the same reason.
    """
    if value is None or value is pd.NA:
        return None
    try:
        if not isinstance(value, str) and pd.isna(value):
            return None
    except (TypeError, ValueError):          # a non-scalar is not a missing value
        return None
    text = str(value).strip()
    return text or None


def _normalise_review(value: Any) -> str | None:
    """A ``review`` cell as one of :data:`REVIEW_VALUES`, or ``None``.

    A token this build does not know reads ``None`` -- an unrecognised spelling is not
    evidence that a person ruled -- and that judgement is made HERE, once, so the row
    reader and the frame reader cannot disagree about it. See :func:`_normalise_token`.
    """
    text = _normalise_token(value)
    return text if text in REVIEW_VALUES else None


def source_for_module(module_id: Any) -> str:
    """The ``source`` token for rows authored by ``module_id``. THE one mapping.

    Identity by design -- a detector row stores its module id verbatim -- so there is no
    second spelling to keep in step with `~nanoscope.core.sources.DETECTOR_NAMES`, and an
    id this project does not ship reads back AS ITS ID rather than as a guess, exactly as
    `~nanoscope.core.sources.detector_name` already resolves an unknown id. It is a
    function and not a bare cast so that every producer goes through one door, and so
    that the door can refuse the one input that must never reach the column.

    Raises:
        ValueError: for an empty/``None`` module id. A detection always knows its own
            id, so a blank one is a bug at the call site; stamping ``pd.NA`` for it would
            record "no detector authored this row", which is what a MANUAL row means, and
            silently mislabelling a detector's output as hand-added would move the
            detector's own score (``added`` counts as a false negative in
            `~nanoscope.analysis.scoring.score_detection`).
    """
    text = "" if module_id is None else str(module_id).strip()
    if not text:
        raise ValueError(
            "source_for_module() needs the producing AnalysisResult's module_id "
            "(e.g. 'detect.schmitt'); pass events.SOURCE_MANUAL for a hand-added row.")
    return text


def source_kind(value: Any) -> str:
    """``value`` as one of the owner's categories: conventional / ml / manual / mixed /
    unknown.

    The display-and-grouping question ("final analyses should normally use all accepted
    events, regardless of how they were detected") asked of a stored ``source`` cell. An
    id that is not in :data:`SOURCE_KIND_BY_MODULE` reads :data:`KIND_UNKNOWN` rather than
    being guessed into a bucket: a detector this build does not know about might be
    either kind, and a wrong bucket is worse than an admitted gap.

    Normalised through :func:`_normalise_token`, the one place these two columns decide
    what a cell says.
    """
    text = _normalise_token(value)
    if text is None:
        return KIND_UNKNOWN
    if text == SOURCE_MANUAL:
        return KIND_MANUAL
    if text == SOURCE_MIXED:
        return KIND_MIXED
    return SOURCE_KIND_BY_MODULE.get(text, KIND_UNKNOWN)


def has_source_status(df: Any) -> bool:
    """Does this frame carry the pair at all?

    ``True`` only when BOTH columns are present -- a half-stamped frame states nothing
    and must read exactly as unstamped, the same rule :func:`has_original_bounds`
    follows. Says nothing about any individual row: a stamped frame can still hold
    ``pd.NA`` cells, which is :func:`status_of`'s question.
    """
    cols = getattr(df, "columns", None)
    return cols is not None and SOURCE_COL in cols and REVIEW_COL in cols


def _added_mask(df: pd.DataFrame) -> np.ndarray:
    """``scoring.added_mask`` without the import cycle (``scoring`` imports this module).

    One writer for "which rows were hand-added" -- this delegates rather than re-reading
    the column, so ``source`` cannot disagree with the marker the detector score and the
    training export already read.
    """
    from .scoring import added_mask          # local: scoring imports events at module scope
    return added_mask(df)


def _string_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """``df[col]`` as an object array, materialising a missing column as all-``pd.NA``."""
    if col not in df.columns:
        return np.full(len(df), pd.NA, dtype=object)
    return df[col].astype("string").to_numpy(dtype=object)


def ensure_source_status(
    df: pd.DataFrame,
    *,
    source: Any = pd.NA,
    review: Any = pd.NA,
) -> pd.DataFrame:
    """Give ``df`` the two columns IN PLACE, filling only what the caller can honestly
    state. Returns the same frame, like `scoring.ensure_added_column`.

    IN PLACE and not a copy, deliberately: the event table, the Event Inspector and the
    undo recorder all hold the SAME ``DataFrame`` object (that sharing is the design --
    an edit in the browser is an edit in the table), so a materialisation that returned a
    copy would give the column to nobody. This is why `scoring.ensure_added_column` is
    in-place too, and why the columns must be materialised when a detection is first
    DISPLAYED rather than at first use: `core.history._snapshot` compares whole frames,
    and a column that appears midway reports a mutation that never happened.

    BOTH DEFAULTS ARE ``pd.NA``, WHICH IS THE POINT. An old project's rows have a
    reviewedness nobody recorded, and that is UNKNOWN -- not "unreviewed" (a claim that
    the row is a fresh candidate) and not "reviewed" (a claim that somebody looked). The
    only way to get either token onto a row is to pass it, so a caller who does not know
    cannot accidentally assert. A caller who DOES know -- the panel materialising a
    detection it has just run, which knows both the module id and that nobody has
    curated it yet -- passes both.

    A cell that already holds a value is NEVER overwritten, so this is idempotent and
    safe to call on every display. ``added`` wins over ``source`` where they would
    conflict: a hand-added row is stamped :data:`SOURCE_MANUAL` whatever ``source`` says,
    because ``added`` is the existing recorded fact (the detector score reads it as a
    false negative and the training export ships it) and ``source`` must agree with it
    rather than contradict it.

    *** ``added`` WINS FOR ``source`` AND SAYS NOTHING ABOUT ``review``. THE TWO ARE
    DELIBERATELY SEPARATED, AND THIS IS THE CORRECTION OF A REAL PROMOTION. *** This
    function used to write :data:`REVIEW_DONE` for every ``added`` row on the reasoning
    that creating an event IS a curation gesture -- which is true of the gesture and
    false of the flag. ``added`` is not only set by the person who drew the span:
    ``merge._merged_row`` ORs it across a run, so a merged row inherits ``added=True``
    from one hand-added fragment. Measured, on ordinary gestures: a project saved before
    these columns existed (every ``review`` unknown) + one hand-added event + "Merge
    all" -> ``merge`` routes ``review`` through :func:`combine_reviews`, which returns
    ``pd.NA`` on purpose ("a merged event nobody has looked at as a whole must not
    inherit 'reviewed' from the one piece of it that somebody did") -- and then this
    door, one call later in ``modules.reconcile_result``, saw ``added=True`` and wrote
    ``reviewed`` back over it. The conservative rule was undone by the very next
    function. So: CREATING an event records reviewedness, INHERITING an ``added`` flag
    through a merge does not, and since this door cannot tell those two apart, it no
    longer guesses. The creating gesture says so itself, by calling :func:`mark_reviewed`
    on the row it just made (`Args` below: pass ``review`` when you know, and the panel
    that just added a row does).

    Args:
        df: The live event frame. Mutated.
        source: A module id (see :func:`source_for_module`), :data:`SOURCE_MANUAL`, or
            ``pd.NA`` to leave un-stamped rows unknown.
        review: :data:`REVIEW_DONE`, :data:`REVIEW_NONE`, or ``pd.NA``.

    Raises:
        ValueError: for a ``review`` outside :data:`REVIEW_VALUES`, or an empty
            ``source`` string -- a blank is not the same as unknown and must not be
            written as if it were.
    """
    if review is not pd.NA and review is not None and review not in REVIEW_VALUES:
        raise ValueError(
            f"review must be one of {REVIEW_VALUES} or pd.NA (unknown), got {review!r}")
    token = None
    if source is not pd.NA and source is not None:
        token = str(source).strip()
        if not token:
            raise ValueError(
                "source='' is not 'unknown': pass pd.NA to leave a row un-stamped.")

    src = _string_column(df, SOURCE_COL)
    rev = _string_column(df, REVIEW_COL)
    added = _added_mask(df)
    missing_src = pd.isna(src)
    missing_rev = pd.isna(rev)

    if token is not None:
        src[missing_src & ~added] = token
    src[missing_src & added] = SOURCE_MANUAL
    if review is not pd.NA and review is not None:
        # ``~added`` still: the caller's ``review`` describes the rows the SOURCE it
        # named produced ("this detection is a pile of candidates"), and it has no
        # standing to say what a person did about a row that detection did not author.
        # An ``added`` row's reviewedness is left exactly as recorded -- unknown until
        # the gesture that created it records it.
        rev[missing_rev & ~added] = str(review)

    df[SOURCE_COL] = pd.array(src, dtype="string")
    df[REVIEW_COL] = pd.array(rev, dtype="string")
    return df


def stamp_source_status(
    df: pd.DataFrame,
    module_id: Any,
    *,
    review: str = REVIEW_NONE,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Record ``module_id`` as the author of every row. The DETECTION writer.

    The sibling of :func:`stamp_original_bounds`, called at the same place and for the
    same reason: a detector's own statement about its own output, made once, on a frame
    whose rows are by definition its own. It stamps :data:`REVIEW_NONE` -- every row a
    detector produces is a CANDIDATE, which is the half of section 7 the app is missing
    ("automatic detection should generate candidate events; user curation should
    determine the final dataset"). It does NOT touch ``accepted``: a detection still
    arrives with its rows accepted, every existing export and figure still reads exactly
    the rows it read before, and the new column is what lets a reader tell that nobody
    has confirmed them.

    An already-stamped frame is returned UNCHANGED unless ``overwrite`` is passed, for
    the reason `stamp_original_bounds` refuses too: re-stamping after curation would
    write the current table's rows under the detector's name. The only caller entitled to
    ``overwrite`` is a re-detection that genuinely replaced the rows. Even then, rows
    marked ``added`` keep :data:`SOURCE_MANUAL` -- a hand-added event is not the
    detector's, no matter who is stamping -- and their ``review`` reads ``pd.NA``, not
    :data:`REVIEW_NONE` and not :data:`REVIEW_DONE`. A detector cannot state whether a
    person has ruled on a row it did not author: calling it unreviewed would erase a
    ruling, and calling it reviewed would invent one (and ``added`` is ORed across a
    merge, so it is not by itself evidence that anybody looked -- see
    :func:`ensure_source_status`). Unknown is the only honest cell, and
    :func:`mark_reviewed` is what puts a token back.

    Returns a COPY with the two columns appended; the input frame is not touched.
    """
    token = source_for_module(module_id)
    if review not in REVIEW_VALUES:
        raise ValueError(f"review must be one of {REVIEW_VALUES}, got {review!r}")
    if has_source_status(df) and not overwrite:
        return df
    out = df.copy()
    n = len(out)
    out[SOURCE_COL] = pd.array([token] * n, dtype="string")
    out[REVIEW_COL] = pd.array([review] * n, dtype="string")
    added = _added_mask(out)
    if added.any():
        src = out[SOURCE_COL].to_numpy(dtype=object)
        rev = out[REVIEW_COL].to_numpy(dtype=object)
        src[added] = SOURCE_MANUAL
        rev[added] = pd.NA
        out[SOURCE_COL] = pd.array(src, dtype="string")
        out[REVIEW_COL] = pd.array(rev, dtype="string")
    return out


def _cell(row: Any, col: str) -> Any:
    """One cell of one row, or ``None`` when it is absent or missing.

    Tolerant of every row shape the GUI hands around -- a ``pd.Series``, a dict, a
    ``namedtuple``-ish record -- because `original_bounds` above is, and a caller should
    not have to know which one it holds.
    """
    try:
        value = row[col]
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if value is None or value is pd.NA:
        return None
    try:
        if not isinstance(value, str) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return value


def source_of(row: Any) -> str | None:
    """The ``source`` of ONE row, or ``None`` when it was never recorded.

    ``None`` is not "unknown detector" as a value a caller may print -- it is the signal
    that this row predates the column, and a caller must say "unknown" rather than
    attributing it to whichever detector the enclosing result happens to name. That
    attribution is available and it is exactly the lie this column exists to stop: the
    result's ``module_id`` is a statement about the TABLE, and this row may be the one
    somebody added by hand.
    """
    return _normalise_token(_cell(row, SOURCE_COL))


def review_of(row: Any) -> str | None:
    """The ``review`` token of ONE row (:data:`REVIEW_DONE` / :data:`REVIEW_NONE`), or
    ``None`` when nobody recorded it. A token this build does not know reads ``None``
    too: an unrecognised spelling is not evidence that a person ruled.

    :func:`_normalise_review` is the rule, shared with the vectorised reader."""
    return _normalise_review(_cell(row, REVIEW_COL))


def is_reviewed(row: Any) -> bool | None:
    """``True`` / ``False`` / ``None`` for "has a person ruled on this row?".

    THREE-VALUED ON PURPOSE, and the reason this is not a ``bool``: ``None`` means the
    answer was never recorded, and a caller that collapses it to ``False`` has just told
    the user that an old project is a pile of unreviewed candidates. Use
    ``is_reviewed(row) is True`` when you need the affirmative.
    """
    token = review_of(row)
    return None if token is None else token == REVIEW_DONE


_MISSING_ACCEPTED = (
    "needs the 'accepted' column: the status is derived from (review, accepted), "
    "and there is nothing to derive it from.")


def _status_rule(accepted: bool, auto_rejected: bool, review: str | None) -> str:
    """THE derivation. Both public readers are callers of this and of nothing else.

    ``auto_rejected`` is :func:`is_automatic_rejection` of the row's ``flag``, already
    computed; ``review`` is :func:`_normalise_review` of its ``review`` cell. Everything
    that could be normalised two ways has been normalised once by the time it gets here,
    which is the entire reason this function exists rather than two implementations of
    "the same four rules".

    The rules, in order, and the order is the design:

    1. Not accepted, and :func:`is_automatic_rejection` explains the flag ->
       :data:`STATUS_AUTO_REJECTED`. First, because that function is already THE one
       definition of "the app rejected this row, not a person" (`inherited_curation`,
       ``core.history._hand_curation`` and `remeasure.merge_late_curation` are its three
       existing readers), and reporting a boundary-clipped row as a curator's rejection
       would attribute to him a decision the app made -- including for a hand-added event
       whose selection ran off the end of the recording, which arrives reviewed AND
       auto-rejected.
    2. Not accepted, and nothing automatic explains it -> :data:`STATUS_REJECTED`. **A
       PERSON DID THAT, WHATEVER ``review`` SAYS**, because a row is only ever not
       accepted for one of two reasons and rule 1 has already taken the other one. This
       is not a new opinion: ``core.history._hand_curation`` reads the IDENTICAL row
       through the IDENTICAL authority -- ``(~accepted) & ~is_automatic_rejection(flag)``
       -- and counts it as a hand rejection, in the warning that tells the user what a
       re-detection is about to destroy. Reporting the same row as "unreviewed" (which
       this did, before: nothing in the app calls :func:`mark_reviewed` yet, so no
       rejection carries a ``review`` token) made the app answer one question two ways,
       and the already-wired ``curation.by_status`` sidecar published the wrong one.
       Storage is untouched: ``review`` still says only what somebody WROTE, and this
       says what the pair MEANS.
    3. Accepted, and a person has ruled -> :data:`STATUS_ACCEPTED`.
    4. Accepted, and nobody has ruled yet -> :data:`STATUS_UNREVIEWED`.
    5. Accepted, and nobody recorded whether anybody ruled -> :data:`STATUS_UNKNOWN`.

    Rules 1, 2 and 5 need no new column, so a legacy row still reads honestly: the
    detector's own rejects are named as such, a person's rejects are named as such, and
    what is left admits it does not know. ``accepted`` is untouched throughout -- it
    remains the single answer to "is this row in the final dataset", which is what every
    export, figure and fit reads.
    """
    if not accepted:
        return STATUS_AUTO_REJECTED if auto_rejected else STATUS_REJECTED
    if review is None:
        return STATUS_UNKNOWN
    return STATUS_ACCEPTED if review == REVIEW_DONE else STATUS_UNREVIEWED


def status_of(row: Any) -> str:
    """The owner's review status for ONE row -- DERIVED, never stored. See
    :func:`_status_rule`, which is the definition; this is its single-row shape and adds
    nothing to it.

    Raises:
        ValueError: when the row carries no ``accepted``, for the reason
            :func:`event_status` refuses the same frame -- status is a statement ABOUT
            ``accepted``, and assuming True for a row that does not have one is how the
            per-row read-out came to say "accepted" about a frame the table column
            refused to describe at all. Every EventSet row has the column;
            ``accepted`` is in :data:`SCHEMA` and :data:`REQUIRED`.
    """
    accepted = _cell(row, "accepted")
    if accepted is None:
        raise ValueError(f"status_of() {_MISSING_ACCEPTED}")
    is_acc = bool(accepted)
    auto = (not is_acc) and is_automatic_rejection(_cell(row, "flag") or "")
    return _status_rule(is_acc, auto, review_of(row))


def event_status(df: pd.DataFrame) -> pd.Series:
    """:func:`status_of` for a whole frame, as a ``string`` Series on ``df``'s index.

    The vectorised shape of :func:`_status_rule`, which is the ONE definition: this
    normalises the three inputs column-wise and then calls it, so a table column, a
    summary count and a per-row read-out cannot answer differently. They could before,
    and did: this path compared ``review`` cells raw where `review_of` stripped them, so
    ``' reviewed'`` read UNKNOWN here and ACCEPTED there. A frame without the pair reads
    all-:data:`STATUS_UNKNOWN` except for rejections -- the detector's own, which are
    visible in ``flag``, and a person's, which are visible in ``accepted``.

    Raises:
        ValueError: when ``accepted`` is missing. Status is a statement ABOUT
            ``accepted``; computing it without one would invent the very fact it
            describes. :func:`status_of` refuses the same row for the same reason.
    """
    if "accepted" not in df.columns:
        raise ValueError(f"event_status() {_MISSING_ACCEPTED}")
    n = len(df)
    accepted = df["accepted"].to_numpy(dtype=bool) if n else np.zeros(0, dtype=bool)
    flags = (df["flag"].to_numpy(dtype=object) if "flag" in df.columns
             else np.full(n, "", dtype=object))
    rev = _string_column(df, REVIEW_COL)

    # Comprehensions, not ``rev == REVIEW_DONE``: ``rev`` is an object array that may
    # hold ``pd.NA``, and ``pd.NA == 'reviewed'`` is ``pd.NA``, not False -- a numpy
    # comparison over it yields an object array whose ``&`` raises the "boolean value of
    # NA is ambiguous" that `remeasure._differs` documents. The per-cell call is
    # `_normalise_review`, the same one `review_of` makes.
    tokens = [_normalise_review(r) for r in rev]
    # ``is_automatic_rejection`` parsed once per DISTINCT flag cell rather than once per
    # row -- a 100 000-event table holds a handful of distinct flag strings, and
    # ``core.history._hand_curation`` already memoises the identical question this way.
    auto_by_flag: dict[Any, bool] = {}
    autos = []
    for f in flags:
        key = "" if f is None or f is pd.NA else f
        try:
            hit = auto_by_flag[key]
        except (KeyError, TypeError):
            hit = is_automatic_rejection(key)
            try:
                auto_by_flag[key] = hit
            except TypeError:                # an unhashable cell: measured, not cached
                pass
        autos.append(hit)

    out = np.array(
        [_status_rule(bool(a), bool(au), t) for a, au, t in zip(accepted, autos, tokens)],
        dtype=object)
    if not n:
        out = np.empty(0, dtype=object)
    return pd.Series(pd.array(out, dtype="string"), index=df.index, name="status")


def mark_reviewed(
    df: pd.DataFrame,
    rows: Any = None,
    *,
    reviewed: bool = True,
) -> int:
    """Record that a person has ruled on ``rows`` (default: all of them). IN PLACE.

    THE CURATION GESTURE, and deliberately only half of one: this writes reviewedness
    and NOTHING else. Which way the ruling went is ``accepted``, which the panel already
    sets, and giving this function an ``accepted=`` argument would make two writers for
    the column the whole app reads. Accept/Reject in the UI is therefore "set accepted,
    then mark_reviewed"; a bare "mark as reviewed" (a curator who looked and kept the
    detector's answer) is this call alone.

    Args:
        df: The live frame. Mutated; the columns are materialised if absent, with every
            OTHER row left ``pd.NA`` rather than defaulted.
        rows: A boolean mask over positions, an array of integer positions, or ``None``
            for every row.
        reviewed: ``False`` un-marks -- the undo path for a review, not a rejection.

    Returns:
        How many cells this call changed, **counting the ones the materialisation door
        changed**, so that ``0`` means the frame is exactly as it was found and a caller
        may skip the undo checkpoint on that word alone.

        IT DID NOT MEAN THAT, AND THAT WAS THE DEFECT. The count covered the ``review``
        writes only, while the first statement of the function -- ``ensure_source_status``
        -- materialises two columns IN PLACE and fills ``source`` for every ``added`` row.
        Measured: ``mark_reviewed(df, rows=[])`` on a frame with one hand-added row
        returned 0 while two columns appeared and a row's derived status moved. A caller
        that trusts the count then skips the checkpoint, and ``core.history._snapshot``
        compares WHOLE frames -- so the frame on the undo stack and the frame on screen
        differ with nothing recorded in between, which is precisely what
        `ensure_source_status`'s own docstring warns a mid-session column change does.
        Two things fix it: an empty selection now returns before touching the frame at
        all (nothing to record, so nothing is changed), and the ``source`` cells the door
        fills are counted with the ``review`` cells it writes.
    """
    n = len(df)
    if rows is None:
        idx = np.arange(n)
    else:
        arr = np.asarray(rows)
        idx = np.flatnonzero(arr) if arr.dtype == bool else arr.astype(np.int64)
    if not len(idx):
        # NOTHING TO MARK IS NOTHING TO DO -- no materialisation, no ``source`` fill, no
        # column appearing behind the caller's back. A no-op call must leave a frame a
        # snapshot cannot tell apart from the one before it, or the ``0`` it returns is
        # a false statement rather than a spare checkpoint.
        return 0

    before_src = _string_column(df, SOURCE_COL)
    before_rev = _string_column(df, REVIEW_COL)
    ensure_source_status(df)
    rev = df[REVIEW_COL].to_numpy(dtype=object)
    token = REVIEW_DONE if reviewed else REVIEW_NONE
    wrote = False
    for i in idx:
        i = int(i)
        if rev[i] is pd.NA or rev[i] != token:
            rev[i] = token
            wrote = True
    if wrote:
        # POSITIONAL assignment through a rebuilt array, not ``.loc``: an event table's
        # index is not guaranteed unique (a subset, a concat without ``ignore_index``),
        # and a label write would touch every row that shares the label -- the same trap
        # `demote_infinities` documents.
        df[REVIEW_COL] = pd.array(rev, dtype="string")
    return (_cells_changed(before_src, _string_column(df, SOURCE_COL))
            + _cells_changed(before_rev, _string_column(df, REVIEW_COL)))


def _cells_changed(before: np.ndarray, after: np.ndarray) -> int:
    """How many cells of one column moved value, ``pd.NA`` included.

    An ABSENT column reads as all-``pd.NA`` (that is what :func:`_string_column` hands
    back), because absent and unrecorded say the same thing here and a materialised
    all-``pd.NA`` column asserts nothing new about any row. It is safe to count it as no
    change ONLY because :func:`mark_reviewed` returns before materialising anything when
    its selection is empty: with a non-empty selection the target cells were ``pd.NA``
    and become a token, so a materialisation always brings at least one real change with
    it and can never hide inside a ``0``.

    ``pd.NA`` is compared as a value, not as a number: ``pd.NA != pd.NA`` is ``pd.NA``,
    which is neither True nor False, so this asks :func:`pandas.isna` first.
    """
    changed = 0
    for old, new in zip(before, after):
        old_na, new_na = pd.isna(old), pd.isna(new)
        if old_na != new_na or (not old_na and old != new):
            changed += 1
    return changed


def carry_source_status(rebuilt: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Put ``prior``'s two columns back onto a frame ``characterize`` has just rebuilt.

    **NOT "THE REBUILD DOOR", AND THE DOCSTRING THAT SAID SO WAS DISPROVED BY ITS OWN
    TREE.** It named itself the door every rebuild path goes through, and no production
    code called it -- measured, ``grep -rn carry_source_status nanoscope/``: zero hits
    outside this module, and the only mention is a sentence in `characterize` pointing
    at it. A door nothing walks through is a claim the next reader will act on, so here
    is what the rebuild paths ACTUALLY do today, each with a test that pins it:

    * ``merge._merged_row`` carries every non-schema column per run, and routes THESE
      TWO by name through :func:`combine_sources` / :func:`combine_reviews` -- a merged
      row has more than one author, so a per-row copy would be the wrong answer here and
      this function cannot serve it.
    * ``remeasure.remeasure_events`` copies every non-schema column per row from the row
      it is replacing, which is exactly this function's ``event_id`` alignment done
      generically for all four extra columns at once.
    * The Event Inspector's mask drag writes the MEASURED columns back BY NAME onto the
      shared frame, so an untouched column is never dropped in the first place.
    * ``modules.reconcile_result`` re-materialises the pair through
      :func:`ensure_source_status` for everything that reaches it without them.

    So this is the NAMED, TESTED form of the same re-application for a caller that has a
    bare ``characterize`` product and the frame it came from -- the shape a fourth
    rebuild path would otherwise write for itself as a ``reindex`` and get subtly wrong.
    It is public and supported for that; it is not a claim about the paths that exist.
    ``characterize`` ends with ``df[list(SCHEMA)]``, so anything outside the frozen 20 is
    gone by the time a rebuilt row comes back -- the same mechanism that makes an old
    project loadable and the reason a rebuild path has to re-apply this pair on purpose.

    Alignment is by ``event_id`` when both frames carry it and ``prior``'s ids are
    unique, because the identity of a row survives a re-measure by design ("the row's
    IDENTITY is not a measurement"). Positional alignment is the fallback and only when
    the lengths match exactly. Anything else RAISES rather than guessing: a mis-aligned
    provenance column is worse than an absent one, because it reads as a fact.

    A ``prior`` without the pair returns ``rebuilt`` unchanged (same object), so this is
    safe to call unconditionally on a legacy table. Otherwise a COPY comes back; rows
    with no match read ``pd.NA``, which is the honest "unknown" this whole section rests
    on.
    """
    if not has_source_status(prior):
        return rebuilt
    src = _string_column(prior, SOURCE_COL)
    rev = _string_column(prior, REVIEW_COL)
    n = len(rebuilt)

    take: np.ndarray | None = None
    if ("event_id" in rebuilt.columns and "event_id" in prior.columns
            and not prior["event_id"].duplicated().any()):
        where = {int(v): i for i, v in enumerate(prior["event_id"].to_numpy())}
        ids = rebuilt["event_id"].to_numpy()
        if all(int(v) in where for v in ids):
            take = np.array([where[int(v)] for v in ids], dtype=np.int64)
    if take is None:
        if n != len(prior):
            raise ValueError(
                f"carry_source_status(): cannot align {n} rebuilt row(s) to "
                f"{len(prior)} prior row(s) -- no shared event_id and no equal length. "
                "A merge reassigns event_id and changes the row count; combine the "
                "columns per group with combine_sources()/combine_reviews() instead.")
        take = np.arange(n, dtype=np.int64)

    out = rebuilt.copy()
    out[SOURCE_COL] = pd.array(src[take] if n else np.empty(0, object), dtype="string")
    out[REVIEW_COL] = pd.array(rev[take] if n else np.empty(0, object), dtype="string")
    return out


def combine_sources(values: Any) -> Any:
    """The ``source`` of one row assembled from several -- a merged run.

    One distinct source survives as itself; two or more become :data:`SOURCE_MIXED`,
    which is a real state and not a hedge: merging a detector's fragment with a
    hand-added event produces a row that BOTH authored, and picking either name would
    put words in somebody's mouth (and, for ``manual``, would move the detector's score,
    which counts hand-added rows as false negatives). All-unknown stays ``pd.NA``;
    unknown mixed with a known source takes the known one, because "at least one
    fragment came from here and the rest were never recorded" is better described by the
    name than by a blank.

    ``merge._merged_row`` today carries a non-boolean extra column as ``series.iloc[0]``
    -- the first fragment's value, i.e. whichever event started earliest. That is not
    wrong so much as arbitrary; wiring this in is a one-line change there.
    """
    seen: list[str] = []
    for v in values:
        if v is None or v is pd.NA:
            continue
        try:
            if not isinstance(v, str) and pd.isna(v):
                continue
        except (TypeError, ValueError):
            continue
        text = str(v).strip()
        if text and text not in seen:
            seen.append(text)
    if not seen:
        return pd.NA
    if len(seen) == 1:
        return seen[0]
    return SOURCE_MIXED


def combine_reviews(values: Any) -> Any:
    """The ``review`` token of one row assembled from several -- a merged run.

    An UNKNOWN fragment makes the whole row unknown, and a single unreviewed fragment
    makes it unreviewed: a merged event is only "reviewed" when a person has ruled on
    every piece of it. The conservative direction is the load-bearing one -- claiming a
    merged row is reviewed because one of its fragments was is how a curator ends up
    with a "final dataset" containing spans nobody ever looked at.
    """
    tokens: list[Any] = list(values)
    if not tokens:
        return pd.NA
    normalised = []
    for v in tokens:
        if v is None or v is pd.NA:
            return pd.NA
        try:
            if not isinstance(v, str) and pd.isna(v):
                return pd.NA
        except (TypeError, ValueError):
            return pd.NA
        text = str(v).strip()
        if text not in REVIEW_VALUES:
            return pd.NA
        normalised.append(text)
    return REVIEW_DONE if all(t == REVIEW_DONE for t in normalised) else REVIEW_NONE


def source_disagrees_with_added(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask of rows where ``source`` and ``added`` contradict each other.

    ``added`` is NOT replaced by ``source`` and is not derived from it. It stays exactly
    what it is -- the marker `scoring.score_detection` reads as a false negative and the
    training export ships -- because deriving it would change a number the detector's
    score already reports, and this change is description, not measurement. ``source``
    is stamped to AGREE with it (see :func:`ensure_source_status`, which is the one
    place that rule lives), and this function is how a caller checks rather than
    assumes.

    A disagreement is a bug in whoever wrote the columns, so it is REPORTED and not
    repaired: silently rewriting one of two recorded facts to match the other is how
    the true one gets lost. ``pd.NA`` never disagrees with anything -- an unstamped row
    makes no claim.
    """
    added = _added_mask(df)
    # Read through the SAME normaliser as `source_of` and `source_kind`. Comprehensions,
    # not ``src == SOURCE_MANUAL``: see the note in `event_status` -- ``pd.NA ==
    # 'manual'`` is ``pd.NA`` and poisons the mask into an object array. And normalised
    # rather than compared raw, because it was not: ``' manual '`` on an ``added`` row
    # read as a contradiction here while `source_of` called the same cell ``manual``.
    src = [_normalise_token(v) for v in _string_column(df, SOURCE_COL)]
    known = np.array([v is not None for v in src], dtype=bool)
    manual = np.array([v == SOURCE_MANUAL for v in src], dtype=bool)
    # :data:`SOURCE_MIXED` NAMES a person among the authors instead of denying one, so it
    # does not contradict ``added`` either. Measured on the live merge: a run stitched
    # from two detector fragments and one hand-added fragment comes back
    # ``source='mixed', added=True`` -- ``merge._combined_provenance_cell`` no longer
    # overwrites ``mixed`` with ``manual`` (doing so dropped a detector-found event out of
    # the sidecar's conventional-vs-ML split). Without this clause every such merge was
    # reported here as a contradiction, and a wolf-crier is no use for finding the real
    # thing: a source that says a DETECTOR alone authored a hand-added row.
    mixed = np.array([v == SOURCE_MIXED for v in src], dtype=bool)
    return (added & known & ~manual & ~mixed) | (~added & manual)



def clipped_boundary_warning(n_clipped: int, n_events: int) -> str:
    """The user-facing sentence for ``n_clipped`` boundary-clipped rows.

    Lives here rather than in either detector so the Schmitt path and the ML path
    say the same thing -- ``ml_modules`` cannot import ``modules`` (``modules``
    registers it), and a second copy of this text is how the two drift apart.
    """
    return (
        f"{n_clipped} of {n_events} event(s) were still above threshold at the "
        "first or last analysed sample, so they are cut off by the edge of the "
        "span, not by the pore: dwell is a lower bound and depth/baseline are read "
        f"from a partial event. They are flagged '{FLAG_CLIPPED}' and NOT accepted. "
        "If this is a segment or a zoomed range, widen it to take the whole event in."
    )


#: A ``frac_depth`` above this is not a blockade fraction. See :func:`frac_depth_note`.
FRAC_DEPTH_ENHANCEMENT = 1.0


def frac_depth_note(value: Any) -> str:
    """The warning that must travel with a ``dI/I0`` above 1, or ``""``.

    ``frac_depth`` is ``depth_pA / |baseline_pA|`` and the whole app reads it as a
    blockade fraction -- a number between 0 and 1. On the owner's apoferritin
    recordings it is **1.81 to 2.57**, and that is real: the current MAGNITUDE grows
    during an event (the detector calls this "enhancement (down)"), and an I-V analysis
    showed the amplifier's ~26 pA DC offset does not explain it away. The Summary row
    read as a 250% blockade, which is exactly the number a co-author lifts off a
    screenshot (report High-5).

    WHY IT LIVES IN ``analysis.events`` AND NOT IN A PANEL. It began as a classmethod
    on ``gui.panels.StatsPanel``, whose docstring said "the note belongs to the QUANTITY
    and not to this table" -- and then had exactly two consumers, the Summary row and
    the CSV sidecar, because the cross-panel import guard means no other panel can reach
    it there. The Event Inspector went on printing ``dI/I0 2.36`` bare on every event,
    the Measure panel's live read-out likewise, and ``plot/figures.py`` captioned its
    histogram "Fractional blockade ΔI/I₀" -- the figure a co-author is most likely to
    receive. A note about a COLUMN belongs beside the column's schema.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v) or v <= FRAC_DEPTH_ENHANCEMENT:
        return ""
    return (f"dI/I0 = {v:.4g} is ABOVE 1, so this is not a blockade fraction: the "
            f"current magnitude GREW during the event, to about {v:.2f}× the open "
            f"pore. A blockade would be between 0 and 1. Check the polarity and "
            f"the amplifier's DC offset before quoting this as a blockade depth.")


def frac_depth_flag(value: Any) -> str:
    """``" ⚠>1"`` for an enhancement, ``""`` otherwise — the inline form.

    A per-event read-out has no room for the sentence, and a glyph is what survives a
    greyscale print and a colour-blind reader where an orange cell does not. Same
    threshold, same function, so the mark and the sentence cannot disagree.
    """
    return " ⚠>1" if frac_depth_note(value) else ""


@dataclass
class EventSet:
    """Detected events plus the provenance that produced them."""

    df: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"EventSet missing required columns: {missing}")
        # The one door every producer and every restore passes through. See
        # `demote_infinities`: a well-formed table is returned unchanged, same object.
        self.df, n_demoted = demote_infinities(self.df)
        if n_demoted:
            self.provenance = {**(self.provenance or {}),
                               "n_infinite_cells_demoted": int(n_demoted)}

    # -- basics -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, k):
        return self.df[k]

    @property
    def accepted(self) -> "EventSet":
        """Only curated-in events. This is what figures and fits should use."""
        return EventSet(self.df[self.df["accepted"]].copy(), {**self.provenance, "filtered": "accepted"})

    @property
    def n_pores(self) -> int:
        return int(self.df["pore_id"].nunique())

    def rate_hz(self) -> float:
        """Event rate over the analysed duration (not the recorded duration)."""
        dur = self.provenance.get("analyzed_duration_s")
        if not dur:
            raise ValueError(
                "analyzed_duration_s missing from provenance; event rate is undefined. "
                "Detection must record how much signal it actually analysed."
            )
        return len(self.df) / dur

    # -- the guardrail ----------------------------------------------------
    def pool(self, *others: "EventSet", allow_multi_pore: bool = False) -> "EventSet":
        """Concatenate event sets.

        Raises if the result spans multiple pores unless you say so explicitly.
        Pooling events across pores and then quoting 1/sqrt(N) error bars treats
        the event as the unit of inference when the *pore* is -- pore geometry
        varies far more than counting noise. If you really want this, pass
        allow_multi_pore=True and report per-pore variance yourself.
        """
        merged = pd.concat([self.df, *(o.df for o in others)], ignore_index=True)
        pores = merged["pore_id"].nunique()
        if pores > 1 and not allow_multi_pore:
            raise ValueError(
                f"pool() would span {pores} pores: {sorted(merged['pore_id'].unique())}.\n"
                "Pore-to-pore variability dominates counting error, so a pooled fit's "
                "error bars would be wrong (too small).\n"
                "Fit per pore and combine the fits, or pass allow_multi_pore=True if you "
                "have a reason and will report per-pore spread."
            )
        return EventSet(merged, {**self.provenance, "pooled_from": [self.provenance.get("recording")] +
                                 [o.provenance.get("recording") for o in others]})

    # -- display ----------------------------------------------------------
    def describe(self) -> pd.DataFrame:
        cols = ["dwell_s", "frac_depth", "depth_pA", "ecd_pA_s", "snr"]
        return self.df[[c for c in cols if c in self.df]].describe()

    def __repr__(self) -> str:
        if not len(self.df):
            return "<EventSet empty>"
        acc = int(self.df["accepted"].sum())
        fl = int(self.df["filter_limited"].sum())
        d = self.df["dwell_s"]
        warn = f", {fl} filter-limited" if fl else ""
        return (
            f"<EventSet n={len(self.df)} ({acc} accepted{warn}), {self.n_pores} pore(s), "
            f"dwell {d.min() * 1e6:.0f}-{d.max() * 1e6:.0f} us "
            f"(median {d.median() * 1e6:.0f} us), "
            f"depth {self.df['frac_depth'].median():.3f} dI/I0>"
        )

    def _repr_html_(self) -> str:
        if not len(self.df):
            return "<i>EventSet: empty</i>"
        fl = int(self.df["filter_limited"].sum())
        banner = ""
        if fl:
            pct = 100 * fl / len(self.df)
            banner = (
                f"<div style='background:#5a3d00;color:#ffd479;padding:6px 8px;"
                f"border-radius:4px;margin-bottom:6px;font-size:12px'>"
                f"⚠ {fl} of {len(self.df)} events ({pct:.0f}%) are filter-limited: their "
                f"depths are attenuated and their dwell times biased long. Raise the filter "
                f"cutoff or exclude them before fitting.</div>"
            )
        return banner + self.df.head(10).to_html(max_cols=12)


# ---------------------------------------------------------------------------
# ``provenance`` is READ-DERIVED for the one key that duplicates the rows
# ---------------------------------------------------------------------------
# ``n_clipped_boundary`` is not an independent fact -- it is a statement about the
# ``flag`` column sitting right next to it, and the hunt's tell was provenance
# claiming a clipped event that zero rows admitted to. Fixing that in each
# PRODUCER is the shape of fix this project has twice watched come back: the
# detectors and the merge each re-read the count, and it went stale anyway,
# because the Event Inspector edits the shared frame IN PLACE (measured: after a
# mask resize that legitimately un-clips a row, the stored count stayed 1 with
# zero rows flagged, and that number is what a project save writes to disk).
#
# So the count is derived where it is READ. Every construction and every mutation
# of the frame is covered by one rule, and a disagreement is not representable
# rather than merely not produced. Only a set that ALREADY carries the key is
# corrected -- one that never had it does not grow one, so provenance stays
# additive by its own choice. Measured cost: 0.16 ms on a 1738-event table,
# 7.7 ms on 100 k events, and there are 35 read sites in the package, none of them
# inside a per-event loop.
def _get_provenance(self: "EventSet") -> dict[str, Any]:
    prov = self._provenance
    if CLIPPED_COUNT_KEY in prov:
        actual = n_clipped_boundary(self.df)
        if prov[CLIPPED_COUNT_KEY] != actual:
            prov[CLIPPED_COUNT_KEY] = actual
    # Consumers export this dict directly. Once geometry/measurement edits make an
    # advisory stale, persist that fact once, keeping the original report for audit.
    # Do not add QC to legacy data or repeatedly wrap already-stale reports.
    quality = prov.get("measurement_quality")
    if isinstance(quality, dict) and quality.get("state") == "current":
        from .measurement_quality import validated_measurement_quality
        checked = validated_measurement_quality(self.df, prov, copy_report=False)
        if checked.get("state") != "current":
            prov["measurement_quality"] = checked
    return prov


def _set_provenance(self: "EventSet", value: Any) -> None:
    # ALWAYS a copy, never the caller's object. The getter above WRITES the
    # reconciled count back into the stored dict, so keeping the caller's dict by
    # reference made a bare read of one EventSet silently rewrite
    # ``n_clipped_boundary`` in a dict somebody else still owns. Measured: with
    # ``df, prov = detect_events(...)`` (prov says 0) and a three-row subset whose
    # rows include one clipped, ``EventSet(subset, prov).provenance`` moved the
    # DETECTOR's own ``prov['n_clipped_boundary']`` from 0 to 1 -- and that dict is
    # what a project save writes to disk. Nothing in the package shares a dict
    # across differing frames today; copying makes it un-shareable instead of
    # merely unshared.
    self._provenance = dict(value or {})


#: Assigned after the ``@dataclass`` decorator has built ``__init__``: the
#: generated ``self.provenance = provenance`` then routes through the setter, and
#: the field keeps its name, its default and its place in the signature.
EventSet.provenance = property(_get_provenance, _set_provenance)  # type: ignore[assignment]


def concat(sets: list[EventSet], allow_multi_pore: bool = False) -> EventSet:
    """Concatenate many EventSets (same guardrail as EventSet.pool)."""
    if not sets:
        return EventSet(empty_df(), {})
    return sets[0].pool(*sets[1:], allow_multi_pore=allow_multi_pore)
