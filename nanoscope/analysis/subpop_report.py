"""The Zn2-insulin subpopulation pipeline, composed end to end and made reportable.

Three modules landed separately, each validated against the owner's own run:
:mod:`~nanoscope.analysis.event_features` (item 1-2, the gate and the thirteen
features), :mod:`~nanoscope.analysis.subpop_cluster` (items 3-7, the per-scope
exploratory clustering) and :mod:`~nanoscope.analysis.subpop_gmm` (items 8-12,
the mixture). None of them had ever been run *into* another: the clustering was
verified by feeding it the owner's stored feature CSV, and the mixture by
feeding it the same CSV. This module is the composition -- a trace and an
``EventSet`` in, a report out -- and the file that pins the seams shut.

THE COMPOSITION REPRODUCES THE ORACLE
-------------------------------------
Run on the four ``CSVData/*.csv.gz`` traces with the owner's own event catalogue,
the chain ``sustained_gate -> event_features -> cluster_scopes ->
quantify_subpopulations`` returns, with nothing tuned:

* all 14 feature columns bit-exact against ``event_clustering_features.csv``
  (``max|diff| == 0.0``, 1488 rows), ``time_half`` included;
* all ten stored label columns exact -- ``hdbscan``, ``hdbscan_{2,10,18,30}``,
  ``kmeans3``, ``kmeans3_{2,10,18,30}``, NaN pattern included;
* ``n_PCs = 11``, ``K = 4``, held-out 16.4090 nats/event, BIC argmin ``K = 8``,
  0.9861 of events assigned above posterior 0.9, sizes 387/295/269/125;
* the 15-column population table equal to his ``subpop_zn_30h_table.csv``, and
  the GMM x ``hdbscan_30`` crosstab equal to his.

That is the whole pipeline, from float32 samples to the reported fractions.

TWO SEAMS WERE ACTUALLY LOOSE
-----------------------------
1. **``CLUST_FEATS`` was written out twice.** ``event_features`` (which produces
   the columns) and ``subpop_cluster`` (which consumes them) each held their own
   thirteen-name literal. Equal on the day, and nothing anywhere would have
   noticed them diverging: the PCA reads columns *by name*, so a reordering of
   one literal produces a differently-ordered loading table and -- through the
   SVD's sign convention -- a differently-signed basis, with no exception raised.
   ``subpop_cluster`` now re-exports the one in ``event_features``, so
   ``subpop_cluster.CLUST_FEATS is event_features.CLUST_FEATS is
   subpop_gmm.SUBPOP_FEATURES``.
2. **Nothing long could be watched.** ``cluster_scopes``, ``select_pcs_and_k``,
   ``bootstrap_fractions`` and ``quantify_subpopulations`` all polled the cancel
   token but had no progress channel, so a minute of cross-validation could only
   present a bar that never moved -- which is what a hang looks like. Each grew
   an optional ``progress`` sink (default ``None``, never read back, so it cannot
   move a number), and this module apportions the run's 0-1 by MEASURED cost.

WHAT THIS MODULE DOES *NOT* DO
------------------------------
``i0_pA`` is the whole recording's ``abs(median(raw))`` and the two baselines are
whole-trace medians (see ``event_features``'s FLOAT FIDELITY note). Every one of
them is therefore a property of the *recording*, not of the events being
analysed: computing them from a filtered event table, or from a time slice, gives
a different feature table under the same column names. So this module takes the
trace and derives them once, and refuses to accept a pre-cut feature matrix.

The two embeddings (items 6-7) are EXPORTED and never printed. t-SNE has no
oracle and UMAP is not bit-reproducible across numba versions; both are figure
aids whose coordinates nothing should gate on, and a coordinate on a page invites
exactly the reading they cannot support. So they ride in the events table, ready
to be plotted, and the text says what was embedded rather than what it was. UMAP
is off by default for the reproducibility reason alone -- see :class:`SubpopConfig`.

ITEM 13 IS HALF HERE, AND THE HALF THAT IS HERE IS THE HALF THAT EXPORTS.
:func:`subpopulation_waveforms` gives the *substance* of the per-subpopulation
trace overlay -- each subpopulation's duration-normalised median waveform on the
raw trace, on a common axis, as a table a plot can be drawn straight from. The
galleries and the rendered overlays are drawn by the GUI (``gui.event_gallery``
filtered by class, and the trace's span overlays coloured by class) from the
``subpop`` column of :meth:`SubpopReport.event_table`; they are VIEWS, so nothing
image-like leaves in the exported files, and the report text says exactly that
rather than leaving the reader to notice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.atomic_write import staged_write_set
from . import event_features as EF
from . import subpop_cluster as SC
from . import subpop_gmm as SG
from .cancel import CancelFlag, OperationCancelled, check_cancelled
from .event_shape import event_ensemble
from .events import EventSet, analysed_span

__all__ = [
    "SubpopConfig",
    "SubpopReport",
    "META_COLUMNS",
    "analysed_span_s",
    "rows_in_hand",
    "DETECTED_NOUN",
    "ACCEPTED_NOUN",
    "MIN_GMM_EVENTS",
    "SOFT_GMM_EVENTS",
    "SLOW_GMM_EVENTS",
    "slow_gmm_note",
    "WAVEFORM_POINTS",
    "feature_table",
    "polarity_for_features",
    "scope_embeddings",
    "run_subpopulations",
    "subpopulation_waveforms",
    "report_paths",
    "write_report",
]


# ===========================================================================
# contract
# ===========================================================================

#: Columns :func:`feature_table` adds beside :data:`event_features.FEATURE_COLUMNS`.
#: ``scope`` is what :func:`~nanoscope.analysis.subpop_cluster.cluster_scopes` cuts
#: on; the owner's notebook cut on ``incubation_h``, which is one recording-level
#: key among several, so the column is named for its ROLE and the caller says what
#: goes in it.
META_COLUMNS: tuple[str, ...] = (
    "event_id",
    "recording",
    "scope",
    "onset_s",
    "time_half",
)

#: Below this the mixture stage is refused outright. It is
#: :data:`subpop_cluster.MIN_SCOPE_N` on purpose: if a scope is too small to be
#: worth one HDBSCAN fit, it is certainly too small for a search over
#: full-covariance mixtures up to K=8 in 11 dimensions. Refusing is the honest
#: outcome -- ``select_pcs_and_k`` would happily return an argmax computed from
#: singular covariances, and nothing in the number would look wrong.
MIN_GMM_EVENTS: int = SC.MIN_SCOPE_N

#: Above :data:`MIN_GMM_EVENTS` but below this, the mixture runs and the report
#: says the selection is not to be trusted. A full-covariance component in ``p``
#: dimensions carries ``p + p(p+1)/2`` parameters -- at the owner's ``n_PCs = 11``
#: that is 77 per component, so ``K = 4`` is 311 free parameters. His 1076 events
#: give ~3.5 events per parameter, which is already thin; a few hundred events
#: would be fitting noise, and the held-out likelihood cannot tell you so because
#: every candidate is equally over-parameterised.
SOFT_GMM_EVENTS: int = 400

#: Above this many events the mixture stage stops being a menu item you wait for.
#: The cost is NOT in the features or the clustering -- those stay seconds into
#: the tens of thousands. It is the two refit loops: :func:`select_pcs_and_k` runs
#: ``n_splits x |p_grid| x |k_grid|`` = 480 full-covariance EM fits, and
#: :func:`bootstrap_fractions` another 200, and each EM needs MORE iterations to
#: converge as n grows, so the wall clock climbs faster than n does. Worse, the
#: selected K climbs too, and cost is linear in K: at n=1076 the CV picks K=4, at
#: n=2000 it picks K=8, the top of the grid.
#:
#: Measured here at ``n_PCs = 11``, selection + bootstrap: n=500 -> 8 s;
#: n=1076 (his zn_30h) -> ~2 min on a quiet machine, 4.5 min under load;
#: n=2000 -> ~7 min quiet, 42 min under load; n~11000 -- reachable on that same
#: recording just by lowering the gate -- did not finish in 45 min.
#:
#: The threshold sits BELOW the 2000 that measured in minutes-to-tens-of-minutes
#: and ABOVE his own 1076, so the owner's own analysis is never interrupted and
#: the first run that could outlast a coffee break says so.
#:
#: Nothing here SUBSAMPLES above the threshold; doing so silently would change
#: his numbers. The report warns, the GUI asks first, and cancellation stays
#: responsive throughout (the loops poll the token every fit).
SLOW_GMM_EVENTS: int = 1500

#: Above this many events the embeddings are skipped and the report says so.
#: Barnes-Hut t-SNE is O(n log n) but with a large constant -- a few seconds at
#: the owner's 1076 events, minutes at 20 k -- and unlike every other stage here
#: it exposes no progress hook and produces nothing anything downstream reads. A
#: figure aid must not become the reason a report takes ten minutes. Raise it
#: deliberately if the picture is what you came for.
EMBEDDING_MAX_EVENTS: int = 5000

#: Points on the common axis of :func:`subpopulation_waveforms`. Enough to show
#: the entry/exit shape a subpopulation is distinguished by; small enough that the
#: exported table stays readable.
WAVEFORM_POINTS: int = 200

# Progress budget, apportioned by MEASURED cost on the owner's zn_30h (1076
# events): features ~1 s, clustering ~2 s, the mixture ~75 s, the waveforms <1 s.
# Splitting a bar evenly across four stages would park it at 25 % for the whole
# minute that matters.
_P_FEATURES = 0.04
_P_CLUSTER = 0.10
_P_GMM = 0.96


@dataclass(frozen=True)
class SubpopConfig:
    """Every knob and every seed the run takes, in one value object.

    Defaults are the notebook's, so a default-constructed config is the
    configuration that reproduces the owner's numbers. Frozen and recorded on
    the report, which is what makes :meth:`SubpopReport.text` a self-describing
    document rather than a snapshot of settings nobody wrote down.

    Attributes:
        min_depth_sigma: The sustained-depth gate, in event-excluded MADs.
            **This is the parameter that decides the population**: it kept
            1076 of 11004 fired regions on zn_30h.
        polarity: ``+1`` for a downward blockade (every Zn2 recording),
            ``-1`` to mirror the trace first.
        min_scope_n: Scopes below this are recorded and skipped by the
            exploratory clustering.
        k_grid: Component counts the mixture search covers.
        n_splits: Cross-validation folds.
        n_boot: Bootstrap resamples for the fraction intervals; ``0`` skips.
        random_state: THE model seed -- fold split, every PCA, every mixture,
            k-means. One knob, deliberately.
        bootstrap_seed: Seed for the resampling draws.
        waveform_points: Common-axis length for :func:`subpopulation_waveforms`.
        with_waveforms: Compute item 13's per-subpopulation median waveforms.
        with_tsne: Compute item 6's t-SNE of the pooled scope. On by default:
            seeded, a second or two, and the coordinates are EXPORTED but never
            printed and never gated on.
        with_umap: Compute item 7's UMAP. **Off** by default, and not because it
            is slow: ``random_state`` forces it single-threaded but the result
            is still not bit-reproducible across ``umap-learn``/numba/platform
            versions, so a run and a re-run of the same report can disagree
            about a column an unwary reader will assume is data. Turn it on
            deliberately, for a figure.
    """

    min_depth_sigma: float = EF.MIN_DEPTH_SIGMA
    polarity: int = 1
    min_scope_n: int = SC.MIN_SCOPE_N
    k_grid: tuple[int, ...] = SG.DEFAULT_K_GRID
    n_splits: int = SG.DEFAULT_N_SPLITS
    n_boot: int = SG.DEFAULT_N_BOOT
    random_state: int = SC.RANDOM_STATE
    bootstrap_seed: int = 0
    waveform_points: int = WAVEFORM_POINTS
    with_waveforms: bool = True
    with_tsne: bool = True
    with_umap: bool = False

    def as_dict(self) -> dict[str, Any]:
        """A JSON-writable copy, for the report's metadata sidecar."""
        return {
            "min_depth_sigma": float(self.min_depth_sigma),
            "polarity": int(self.polarity),
            "min_scope_n": int(self.min_scope_n),
            "k_grid": [int(k) for k in self.k_grid],
            "n_splits": int(self.n_splits),
            "n_boot": int(self.n_boot),
            "random_state": int(self.random_state),
            "bootstrap_seed": int(self.bootstrap_seed),
            "waveform_points": int(self.waveform_points),
            "with_waveforms": bool(self.with_waveforms),
            "with_tsne": bool(self.with_tsne),
            "with_umap": bool(self.with_umap),
        }


# ===========================================================================
# the sign convention -- THE TWO IN THIS TREE ARE MIRROR IMAGES
# ===========================================================================


def polarity_for_features(app_polarity: Any) -> int:
    """Translate this app's event polarity into ``event_features``' own.

    **They point opposite ways, and both are right.** This app's convention
    (``nanoscope.analysis.events``, ``characterize``, ``baseline.derive_polarity``)
    is ``dev = polarity * (current - baseline)``, so ``+1`` means excursions run
    ABOVE the open-pore level -- which at negative bias is what an ordinary
    blockade does. ``event_features`` measures ``d = baseline - sample`` and calls
    ``+1`` the case where excursions run BELOW, because that is the sign the
    notebook's ``current_detr_pA`` had.

    So the translation is a negation, and it is not optional: feeding the app's
    ``+1`` straight through mirrors the trace, which turns every depth into its
    own negative and drops every event through the ``>= 4-sigma`` gate. The
    symptom would be an empty report, not a wrong one -- but on a recording where
    the polarity is genuinely ``-1`` the symptom is a report of the *baseline
    noise*, which looks entirely plausible.

    Args:
        app_polarity: ``+1`` / ``-1`` in this app's convention. Anything else
            (``None``, a missing provenance key) yields ``+1`` here, i.e. the
            downward-blockade reading the whole Zn2 corpus has.

    Returns:
        ``+1`` or ``-1``, for ``event_features``.
    """
    try:
        pol = int(app_polarity)
    except (TypeError, ValueError):
        return 1
    return -1 if pol == 1 else 1


# ===========================================================================
# the clock -- resolved once, for every statistic that has halves in it
# ===========================================================================


def analysed_span_s(
    events: EventSet | pd.DataFrame | None,
    n_samples: int,
    fs: float,
) -> tuple[float, float]:
    """``(t0, t1)`` in seconds of the span the DETECTOR looked in. The one clock.

    THE INVARIANT: **every statistic in a report is computed on the span the
    report says it describes.** ``time_half`` used to be cut at the midpoint of
    ``len(detrended)/fs`` -- the whole raw trace -- while the rows under it came
    from whatever span the detector was pointed at. The ML detector's range combo
    DEFAULTS to "Visible view", so the two clocks differ on a default path, and a
    recording whose 700 events are spread perfectly evenly over 22 s, detected in
    its own last 8.8 s, reported ``early_frac`` 0.00 and printed "THE RECORDING IS
    NOT STATIONARY: 100 % of all gated events fall in the second half of it". The
    events were evenly spread; the claim was false, and because the pooled
    fraction collapsed to 0.00 the per-class check underneath it returned early,
    so the false sentence was the *only* thing the reader got.

    The span travels with the rows, in the EventSet's provenance, under the two
    keys :func:`~nanoscope.analysis.events.span_provenance` writes and
    :func:`~nanoscope.analysis.events.analysed_span` reads -- the same pair the
    boundary-clipping rule already judges rows against. Reading it here means the
    halves cannot disagree with the flags, and no caller has an argument to
    forget.

    Nothing on the Schmitt path moves: ``detect_events`` stamps ``(0, n)``, which
    is the whole trace, which is what this returned before. A bare frame carries
    no provenance and falls back to ``(0, n_samples)`` for the same reason.

    Args:
        events: The detected regions. Only ``.provenance`` is read; a bare frame
            (or ``None``) has none and takes the whole-trace fallback.
        n_samples: Length of the trace the onsets index into -- the fallback span
            and the clamp. A stored span reaching past it belongs to a different
            array and is not trusted.
        fs: Sample rate, Hz.

    Returns:
        ``(t0, t1)`` in seconds, on the same clock as ``onset_s``. Degenerate only
        where the trace itself is: an empty trace gives ``(0.0, 0.0)``, exactly as
        the whole-trace expression did, and the caller downstream objects to it.

    Raises:
        ValueError: if ``fs`` is not a positive, finite rate -- a clock with no
            rate is not a clock.
    """
    rate = float(fs)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(
            f"fs={fs!r} is not a positive sample rate; the analysed span cannot be "
            "expressed in seconds without one"
        )
    n = int(n_samples)
    # `analysed_span` is the ONE reader of these keys and already ignores anything
    # that is not a mapping (a bare frame's `.provenance` attribute does not exist;
    # a frame with a `provenance` COLUMN hands back a Series). Passing it straight
    # through keeps that rule spelled once, where it belongs.
    start, stop = analysed_span(getattr(events, "provenance", None), n)
    # CLAMPED TO THE TRACE WE WERE HANDED, and reverted whole if the clamp empties
    # it. `analysed_span` guarantees `stop > start` but not that either end is
    # inside this array: a stored span from a longer document, or a trace sliced by
    # a caller, would otherwise put the midpoint outside every onset and make every
    # event "early". The whole trace is the honest fallback -- it is what the
    # expression this replaced always used.
    lo = max(0, min(start, n))
    hi = max(0, min(stop, n))
    if hi <= lo:
        lo, hi = 0, n
    return lo / rate, hi / rate


def _span_pair(span_s: Any) -> tuple[float, float]:
    """``span_s`` as ``(t0, t1)`` floats, or a ValueError naming what was wrong.

    ``time_half`` accepts a bare DURATION as its ``span`` as well as a pair, so a
    caller reaching for this override is quite likely to hand one over -- and
    indexing it produced ``'float' object is not subscriptable``, which names
    neither the argument nor the two forms it takes. A clock handed in wrong is
    the one thing this whole repair is about; it must not fail obscurely.
    """
    try:
        t0, t1 = (float(v) for v in span_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"span_s={span_s!r} is not a (t0, t1) pair of seconds; a bare duration "
            "is not accepted here -- say (0.0, duration) so the two ends are "
            "explicit"
        ) from exc
    return t0, t1


#: The mark :attr:`~nanoscope.analysis.events.EventSet.accepted` stamps on the rows it
#: selects (``provenance["filtered"]``). Curation goes through that one door, so this is
#: the flag that says "these rows are a SUBSET somebody chose", not a detection.
FILTERED_ACCEPTED = "accepted"

#: What the report calls the rows it was handed when nothing says they were filtered.
DETECTED_NOUN = "detected region"
#: ...and when their own provenance says they are the curated-in subset.
ACCEPTED_NOUN = "accepted region"

#: What the report calls the count in ``meta["n_detected"]`` -- the denominator of the
#: "curated out" line -- keyed by what has reshaped the table since the detector stamped
#: it. ``n_events`` is re-read from the rows every time the frame changes shape (see
#: ``events.reconcile_row_counts``), so after a merge it is the MERGED table's size and
#: after a hand-added event it is larger still; the word must follow the number. Resolved
#: once, by :func:`_table_shape`, and printed everywhere the count appears.
TABLE_NOUNS: Mapping[str, str] = {
    "detected": "detected region(s)",
    "merged": "region(s) in the merged table",
    "edited": "region(s) in the table, merged and edited since detection",
}


def _table_shape(
    prov: Mapping[str, Any], n_detected: int | None
) -> tuple[str, int | None, int | None]:
    """What the rows' ``n_events`` is a count OF, from the merge on record -- decided ONCE.

    Returns ``(shape, n_before, n_after)``: ``shape`` is a key of :data:`TABLE_NOUNS`
    and the two counts are the last merge's, or ``None`` when no merge is on record.

    ``n_events`` is re-derived by ``events.reconcile_row_counts`` every time the frame
    changes shape, so after a merge it is the size of the MERGED table and subtracting
    the rows in hand counts rows the user curated out -- not rows a merge folded away.
    Before sweep 7 it was the detector's original stamp, and a 317 -> 247 merge printed
    "71 ... curated OUT" for a user who had curated out one (S7-DET4). The wave-1 fix
    made the CURATION caveat say what the denominator IS -- and left the summary line
    two rows above it saying ``(of 247 detected region(s); ...)``: same number, two
    nouns, in the file written to be read without the app (S7 E3 / X7, measured on the
    owner's -600 mV file after the real Merge all button: 317 detected, 247 merged,
    and 248 after one hand-added event). Both lines now phrase from THIS decision:

      * ``"merged"``   -- a merge is on record and ``n_events`` is still its ``n_after``;
      * ``"edited"``   -- a merge is on record but the table has changed shape since
        (a hand-added event, say): still not the detector's table;
      * ``"detected"`` -- nothing on record says the count is not the detector's.

    ``provenance['merged']`` describes the LAST merge only (each merge overwrites it),
    which is why the words say "the last merge". A ``None`` ``n_detected`` makes the
    word moot -- no line prints a count it does not have -- and it is resolved the same
    way so ``meta["table_noun"]`` never calls a merged table the detection.
    """
    merged = prov.get("merged")
    n_before = n_after = None
    if isinstance(merged, Mapping):
        try:
            n_before = int(merged.get("n_before"))
            n_after = int(merged.get("n_after"))
        except (TypeError, ValueError):
            n_before = n_after = None
    if n_before is None:
        return "detected", None, None
    if n_after == n_detected:
        return "merged", n_before, n_after
    return "edited", n_before, n_after


def rows_in_hand(events: Any, n_rows: int) -> tuple[dict[str, Any], str | None]:
    """Name what the rows a run was handed ACTUALLY are, from their own provenance.

    THE REPORT MUST NOT CALL A CURATED SUBSET THE DETECTED SET. ``run_subpopulations``
    is documented to take ``events.accepted`` -- and the app's entry point passes
    exactly that -- so the number this report has always printed under the words
    "detected regions" is, on every run the app produces, the count of events the user
    curated IN. On a 300-event detection with 180 curated out the exported file read
    ``detected regions        : 120`` and headlined *"120 of 120 detected region(s)
    passed the gate"*, while the app's own status bar for the same run said "Grouping
    120 accepted event(s) into subpopulations". The file is written to be read WITHOUT
    the app beside it, and it itemises every other row the pipeline drops -- the
    >=sigma gate and the too-short-to-measure floor, each with a count and a named
    mechanism. The largest removal of all was the one it never named, and the count it
    did print carried the word for a different number.

    NOTHING IS MEASURED OR GUESSED HERE. Both facts are read off the provenance that
    TRAVELS WITH THE ROWS -- the same discipline :func:`analysed_span_s` follows for the
    clock, reading what the rows carry instead of re-deriving it from the caller:

      * ``provenance["filtered"] == "accepted"`` -- stamped by ``EventSet.accepted``;
      * ``provenance["n_events"]`` -- stamped by the DETECTOR (``detect.py``) with the
        size of the table it produced, which ``accepted`` copies forward. The difference
        between it and the rows in hand is the number curated out.

    A row set carrying neither mark is the detection itself, and is still called that:
    this changes no report that was already accurate.

    Args:
        events: The rows handed to the run. Only ``.provenance`` is read.
        n_rows: How many rows are in hand -- what the gate is about to be asked.

    Returns:
        ``(meta, note)``: ledger keys for :attr:`SubpopReport.meta`, and the caveat the
        run should carry, or ``None`` when the rows have nothing to disclose.
    """
    prov = getattr(events, "provenance", None) or {}
    filtered = str(prov.get("filtered") or "")
    try:
        n_detected: int | None = (None if prov.get("n_events") is None
                                  else int(prov["n_events"]))
    except (TypeError, ValueError):
        n_detected = None
    # A count SMALLER than the rows in hand cannot describe a removal, so it is not
    # used as one. The detector stamps `n_events` once and the table can legitimately
    # grow afterwards (a hand-added missed event), and a stale stamp subtracted from a
    # larger table would print a negative "curated out" -- a fabricated number in the
    # one line that exists to stop a fabricated number.
    if n_detected is not None and n_detected < n_rows:
        n_detected = None
    # THE WORD FOR THE DENOMINATOR IS DECIDED HERE, ONCE. The CURATION caveat and the
    # "curated out (not below)" summary line both print `n_detected`; the noun beside
    # it is read off the merge on record (see `_table_shape`) and carried in `meta`,
    # so the two lines cannot disagree about what the same number counts.
    shape, n_before, n_after = _table_shape(prov, n_detected)
    table_noun = TABLE_NOUNS[shape]

    curated = filtered == FILTERED_ACCEPTED
    # "detected" IS CLAIMED ONLY BY AN UNMARKED ROW SET. ``accepted`` is the only filter
    # mark anything stamps today, but a report that treated an UNRECOGNISED mark as "the
    # detection" would be this same defect returning the moment a second filter is added
    # -- so a mark this function does not know is still not the detection: it is named
    # with the mark's own word, and disclosed.
    if filtered and not curated:
        return (
            {"rows_filtered": filtered, "n_detected": n_detected,
             "region_noun": f"{filtered} region", "table_noun": table_noun},
            f"FILTERED ROWS: these {n_rows} region(s) reached this run carrying the mark "
            f"'{filtered}', so they are a SUBSET of what was detected and every count "
            f"below is a count of that subset. This report does not know what that mark "
            f"selects on and cannot say what it removed.",
        )
    meta: dict[str, Any] = {
        "rows_filtered": filtered,
        "n_detected": n_detected,
        "region_noun": ACCEPTED_NOUN if curated else DETECTED_NOUN,
        "table_noun": table_noun,
    }
    if not curated:
        return meta, None
    if n_detected is None:
        # Honest about the hole: the rows say they were curated, and do not say from
        # how many. Saying so is worth more than the silence it replaces -- a reader
        # who knows a subset was taken can go and look, and one who was told "detected"
        # has no reason to.
        note = (
            f"CURATION: the {n_rows} region(s) below are the events curated IN, not the "
            f"detection. Whatever the detector found and the user then removed is in "
            f"nothing below -- not in the gate counts, not in the mixture, not in any "
            f"fraction. How many were removed is not recorded in these rows' "
            f"provenance, so this report cannot state it."
        )
    elif n_detected == n_rows:
        note = None          # curated, but nothing was actually taken out
    else:
        # When a merge is on record, say what the denominator IS, so "the detector's"
        # is not claimed for a table the detector never produced. The CASE is the one
        # `_table_shape` decided above -- the same one the summary line's noun comes
        # from -- so this caveat and that line describe the number with one word.
        if shape == "merged":
            whose = (
                f"the {n_detected} region(s) left in the table after a merge (the "
                f"last merge folded {n_before} rows into {n_after})"
            )
            of_what = f"the {n_detected} in the merged table"
        elif shape == "edited":
            whose = (
                f"the {n_detected} region(s) in the table (merged and edited since "
                f"detection; the last merge folded {n_before} rows into {n_after})"
            )
            of_what = f"the {n_detected} in the table"
        else:
            whose = f"the detector's {n_detected} region(s)"
            of_what = f"the {n_detected} detected"
        note = (
            f"CURATION: {n_detected - n_rows} of {whose} "
            f"were curated OUT before this run started, and the {n_rows} region(s) "
            f"below are the ones curated IN. Every count and every fraction in this "
            f"report is over the curated-in set: the gate line below says how many of "
            f"THOSE {n_rows} passed, not how many of {of_what}."
        )
    return meta, note


# ===========================================================================
# stage 1 -- the gated feature table
# ===========================================================================


def feature_table(
    detrended,
    raw,
    fs: float,
    events: EventSet | pd.DataFrame,
    *,
    recording: str = "",
    scope: str | Sequence[Any] | None = None,
    span_s: tuple[float, float] | None = None,
    config: SubpopConfig = SubpopConfig(),
    i0_pA: float | None = None,
    cancel: CancelFlag | None = None,
) -> tuple[pd.DataFrame, EF.GateResult]:
    """Gate the detected regions and measure the thirteen features on the survivors.

    The two stages of ``event_features`` in the order they must run, plus the
    per-event metadata the downstream scopes and the stability check need. The
    per-recording scalars are derived HERE, from the whole trace, and never from
    the events -- see the module docstring. ``time_half`` follows the same rule
    but on a different clock: its boundary is the midpoint of the span the rows
    were DETECTED in (:func:`analysed_span_s`, off ``events.provenance``), not the
    median of the onsets themselves (see
    :func:`~nanoscope.analysis.event_features.time_half` for why that
    distinction is the whole of the stability check) and not the length of the
    file, which is a different span whenever the detector was pointed at a
    sub-range.

    Args:
        detrended: The trace the events were detected on, in pA. In the app this
            is :func:`~nanoscope.analysis.modules.detection_signal`, which rebuilds
            it from the detection's own provenance; in the notebook it is
            ``current_detr_pA``. **Pass the whole trace the onsets index into,
            not a slice of it** -- the onsets are absolute, and its length is the
            fallback clock for an ``events`` that carries no provenance.
        raw: The recording's raw current in pA, used only for
            ``i0_pA = abs(median(raw))``. Ignored when ``i0_pA`` is given.
        fs: Sample rate, Hz.
        events: The detected regions -- an :class:`EventSet` or a frame with
            ``start_idx``/``end_idx``. **Every row is gated**, so pass the set you
            mean: an already-curated set is gated again on depth, which is a
            second, different filter.
        recording: Name recorded on every row.
        scope: The clustering scope. A string (or ``None``, meaning ``recording``)
            puts every event in one scope; a per-event sequence -- aligned with
            ``events`` BEFORE gating -- cuts them, which is how the notebook's
            four incubation times would be reproduced from a pooled table.
            Recorded verbatim: this function is the composable primitive, and a
            caller building a pooled table one recording at a time needs its own
            names kept. (:func:`run_subpopulations` is where a *single* scope is
            collapsed onto the pooled one.)
        span_s: ``(t0, t1)`` in seconds to cut ``time_half`` at, overriding what
            :func:`analysed_span_s` reads off ``events``. For a caller whose rows'
            provenance does not describe the span they were found in -- a table
            assembled by hand, or one replayed onto a different trace. **The app
            never passes it**: the provenance is the authority precisely so that
            no caller has to remember, and a wrong span here is a wrong clock.
        config: Gate threshold, polarity and seeds.
        i0_pA: Explicit open-pore magnitude, overriding ``raw``.
        cancel: Cancel token; ambient when ``None``.

    Returns:
        ``(table, gate)``. The table carries
        :data:`~nanoscope.analysis.event_features.FEATURE_COLUMNS` plus
        :data:`META_COLUMNS`, **indexed by the surviving rows' own labels** from
        ``events`` -- not re-indexed, so a caller can join back to its own table
        by label. ``gate`` is the decision on every row, including the rejected
        ones, because "how many were thrown away" is half the finding. The clock
        the halves were cut at is on ``table.attrs["analysed_span_s"]``, so a
        caller reporting on the table cannot resolve a second, different one.

    Raises:
        ValueError: from ``event_features`` on a bad trace, if ``scope`` is a
            sequence of the wrong length, or from ``time_half`` if ``span_s``
            does not span a positive interval.
    """
    check_cancelled(cancel)
    # THE CLOCK IS RESOLVED FIRST, before a minute of gating and feature work: a
    # span handed in wrong is a wrong report, and the caller should hear about it
    # while the answer is still ahead of them rather than after.
    span = (analysed_span_s(events, len(detrended), fs) if span_s is None
            else _span_pair(span_s))
    df = getattr(events, "df", events)
    gate = EF.sustained_gate(
        detrended,
        events,
        min_depth_sigma=config.min_depth_sigma,
        polarity=config.polarity,
        cancel=cancel,
    )

    scalars = EF.feature_scalars(
        detrended, raw, i0_pA=i0_pA, polarity=config.polarity
    )
    kept = df.loc[gate.kept_index]
    feats = EF.event_features(
        detrended, fs, kept, scalars=scalars, polarity=config.polarity, cancel=cancel
    )

    # `event_features` DROPS rows shorter than its `min_samples` and says so in its
    # own docstring -- "rows dropped for min_samples are missing from the index, so
    # events.df.loc[result.index] recovers the matching metadata and no positional
    # assignment is ever needed". Every assignment below WAS positional, so the
    # first gated event under three samples turned the whole action into
    # "Analysis failed: Length of values (720) does not match length of index
    # (710)". Re-align once, here, and every column below is by label.
    #
    # This is a no-op on any recording where nothing is dropped -- `feats.index`
    # is then `kept.index`, in order -- which is every recording the frozen
    # numbers come from.
    if len(feats) != len(kept):
        kept = kept.loc[feats.index]

    feats["event_id"] = (
        kept["event_id"].to_numpy()
        if "event_id" in kept.columns
        else np.asarray(kept.index, dtype=object)
    )
    feats["recording"] = str(recording)

    if scope is None or isinstance(scope, str):
        feats["scope"] = str(scope if scope is not None else recording)
    else:
        per_event = np.asarray(scope, dtype=object).ravel()
        if per_event.size != len(df):
            raise ValueError(
                f"scope has {per_event.size} entries for {len(df)} detected region(s); "
                "a per-event scope is aligned with the regions BEFORE gating, so that "
                "the caller does not have to know which ones survived"
            )
        feats["scope"] = pd.Series(per_event, index=df.index).loc[feats.index].to_numpy()

    onset = (
        kept["t_start_s"].to_numpy(dtype=np.float64)
        if "t_start_s" in kept.columns
        else kept["start_idx"].to_numpy(dtype=np.float64) / float(fs)
    )
    feats["onset_s"] = onset
    # THE HALF IS THE ANALYSED SPAN'S, NOT THE EVENTS' AND NOT THE FILE'S.
    # `time_half`'s default is a split at each scope's own median onset, which
    # halves its own sample by definition -- so `early_frac` pooled to 0.50 on
    # every recording ever analysed and the non-stationarity caveat built on it
    # could not fire. The clock that replaced it, `len(detrended)/fs`, was the
    # whole FILE's, and the detector is not always pointed at the whole file: the
    # ML detector's range combo defaults to "Visible view", so the two clocks
    # differed on a default path and the report split a 22.000 s clock three lines
    # under a header printing `duration : 8.800 s`. `analysed_span_s` reads the
    # span the rows were actually detected in, off the provenance that travels
    # with them; on the Schmitt path that IS `(0, len(detrended))` and nothing
    # moves. EVERY scope in this table shares that one boundary, which is what
    # makes a scope's `early_frac` comparable with its neighbour's -- and what
    # makes a scope confined to one stretch of the span read as confined instead
    # of as a comfortable 50/50.
    feats["time_half"] = EF.time_half(onset, feats["scope"].to_numpy(), span=span)
    # THE CLOCK TRAVELS WITH THE TABLE. `run_subpopulations` prints this span in
    # the report header and gates its SUB-RANGE caveat on it; reading it back off
    # the table means the printed clock cannot be a second, independently-resolved
    # one -- which is the exact shape of the defect this repair is about.
    feats.attrs["analysed_span_s"] = span
    return feats, gate


# ===========================================================================
# item 13 (half of it) -- per-subpopulation median waveforms
# ===========================================================================


def subpopulation_waveforms(
    signal,
    fs: float,
    events: EventSet | pd.DataFrame,
    labels: pd.Series,
    *,
    n_points: int = WAVEFORM_POINTS,
    pad_frac: float = 0.25,
) -> pd.DataFrame:
    """Each subpopulation's duration-normalised MEDIAN waveform, on a common axis.

    The exportable substance of item 13's trace overlay: what a subpopulation
    actually looks like, rather than where it sits in a PCA. Duration-normalised
    (``T`` runs ``-pad_frac`` to ``1 + pad_frac`` across the event), so classes
    that differ in dwell -- and on the owner's zn_30h two of the four differ by
    a factor of 18 at the median -- overlay by SHAPE.

    Measured on the raw/calibrated trace, deliberately: the detrended trace is
    what the features were computed on, but "what does this class of event look
    like" is a question about the recording, and a reader comparing this against
    a screenshot of the trace should be looking at the same current.

    Args:
        signal: The recording's current (whole channel), pA.
        fs: Sample rate, Hz.
        events: The event set the labels index into. An ``EventSet``, or a bare
            frame with ``start_idx``/``end_idx`` -- the same two forms
            :func:`feature_table` takes, so a caller composing the primitives
            can pass one object to both without learning which stage wants
            which. A bare frame simply carries no provenance.
        labels: Subpopulation id per event, indexed by the event frame's labels.
        n_points: Common-axis length.
        pad_frac: Fraction of each dwell included either side, so the entry and
            exit are visible rather than clipped at the event edges.

    Returns:
        A frame with ``T`` and one ``subpop_<k>`` column per subpopulation, plus
        an ``n_events`` attr per column recorded in ``df.attrs["n_events"]``.
        Empty (0 rows) when there is nothing to average.

    Raises:
        ValueError: if ``labels`` names rows that are not in the event frame.
    """
    ev_df = getattr(events, "df", events)
    provenance = dict(getattr(events, "provenance", None) or {})
    missing = pd.Index(labels.index).difference(ev_df.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} labelled event(s) are not in the event set; "
            f"first few: {list(missing[:5])}"
        )
    out: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    axis: np.ndarray | None = None
    for k in sorted(pd.unique(labels.to_numpy())):
        rows = labels.index[labels.to_numpy() == k]
        sub = EventSet(ev_df.loc[rows].copy(), dict(provenance))
        T, matrix, _mean, median = event_ensemble(
            signal, fs, sub, n_points=int(n_points), pad_frac=float(pad_frac)
        )
        axis = T
        out[f"subpop_{int(k)}"] = median
        counts[f"subpop_{int(k)}"] = int(matrix.shape[0])
    if axis is None:
        return pd.DataFrame({"T": np.empty(0, dtype=np.float64)})
    frame = pd.DataFrame({"T": axis, **out})
    frame.attrs["n_events"] = counts
    return frame


# ===========================================================================
# items 6-7 -- the two embeddings, exported and never printed
# ===========================================================================


def _pca_concentration_hint(pca: SC.ScopePCA) -> str:
    """One clause naming *why* a scope's PCA space came out too narrow to embed.

    Only for the case that actually happens on real recordings: a scope whose
    PCA kept a single component. That is not a malfunction -- it is one axis
    reaching the 95 % variance target on its own -- but "kept 1 component" is
    not yet something the reader can act on, so this names the share PC1 holds
    and the feature it loads on, which is the thing to go and look at. Returns
    ``""`` when the scope is not in that shape, so the caller can join it
    unconditionally.
    """
    evr = np.asarray(pca.explained_variance_ratio, dtype=np.float64)
    if int(pca.n_kept) >= SC.EMBEDDING_MIN_COMPONENTS or not evr.size:
        return ""
    comps = np.asarray(pca.components, dtype=np.float64)
    loads = ""
    if comps.ndim == 2 and comps.shape[0] and comps.shape[1] == len(pca.columns):
        j = int(np.argmax(np.abs(comps[0])))
        loads = f", loading most on {str(pca.columns[j])!r} ({comps[0][j]:+.2f})"
    return (
        f"PC1 alone carries {evr[0]:.1%} of this scope's scaled variance{loads}. "
        f"One feature dominating that far is usually a handful of extreme events: "
        f"the robust scaler divides by the IQR, so a single spiky event can push "
        f"one feature orders of magnitude past the rest. Look at that feature's "
        f"extremes -- or exclude them and re-run -- if you want the picture back."
    )


def _embedding_failure_note(
    label: str,
    kind: str,
    scope: str,
    res: SC.ScopeClustering,
    exc: BaseException,
) -> str:
    """The note that stands in for an embedding that could not be fitted.

    Items 6-7 are figure aids -- exported, never printed, depended on by nothing
    else in the report -- so an embedding that cannot be computed must cost the
    reader the picture and NOT the run. What it must never cost them is the
    knowledge that it is missing, or the reason: this note says which scope, why
    in the module's own words, and what still stands.

    :func:`~nanoscope.analysis.subpop_cluster.embedding_refusal` is re-asked here
    rather than the exception text being trusted, so that a refusal we can
    explain in terms of the DATA is reported that way, and only a failure we did
    not anticipate is quoted from the solver -- verbatim and named as such,
    because an unexplained failure paraphrased is worse than one quoted.
    """
    scores = np.asarray(res.pca.scores, dtype=np.float64)
    perp = SC.tsne_perplexity(scores.shape[0]) if kind == "t-SNE" else None
    reason = SC.embedding_refusal(scores, kind=kind, perplexity=perp)
    if reason is None:
        reason = f"{kind} failed inside its solver -- {type(exc).__name__}: {exc}"
    hint = _pca_concentration_hint(res.pca)
    return " ".join(
        part
        for part in (
            f"NO {label}: scope {scope!r} was not embedded: {reason}.",
            hint,
            "Items 6-7 are figure aids, exported and never printed, and nothing "
            "else in this report depends on them -- the gate, the features, the "
            "clustering and the mixture are unaffected by the missing picture.",
        )
        if part
    )


def scope_embeddings(
    clustering: SC.SubpopClustering,
    *,
    scope: str = SC.SCOPE_ALL,
    tsne: bool = True,
    umap: bool = False,
    max_events: int = EMBEDDING_MAX_EVENTS,
    random_state: int = SC.RANDOM_STATE,
) -> tuple[pd.DataFrame | None, tuple[str, ...]]:
    """2-D embeddings of one scope's PCA space, as a table a plot is drawn from.

    Items 6-7. They are FIGURE AIDS: both consume the scope's own PCA scores and
    neither ever sees a cluster label, so the picture cannot be accused of drawing
    the clusters it is used to display -- but by the same token nothing may be
    concluded from the coordinates. They are exported and deliberately never
    printed into the report text, because a number on a page invites exactly the
    reading these two cannot support.

    Args:
        clustering: The sweep to embed a scope of.
        scope: Which scope. Defaults to the pooled one, which for a
            single-recording run is the only one there is.
        tsne: Compute the t-SNE columns.
        umap: Compute the UMAP columns. See :class:`SubpopConfig`.
        max_events: Above this the scope is not embedded at all, with a note.
            See :data:`EMBEDDING_MAX_EVENTS`.
        random_state: Seed for both.

    Returns:
        ``(table, notes)``. ``table`` is indexed by event label with ``tsne_x`` /
        ``tsne_y`` and/or ``umap_x`` / ``umap_y``, or ``None`` when nothing was
        asked for or the scope is absent. ``notes`` explains any absence -- an
        embedding that silently did not happen is the one failure mode a figure
        aid can actually cause.

        A fit that RAISES is an absence too, and is reported the same way. A
        scope whose PCA kept a single component has no second axis to place
        events along, and scikit-learn ends such a run with "n_components=2 must
        be between 1 and min(n_samples, n_features)=1" -- a solver sentence,
        arriving as the total failure of the whole pipeline, for a picture
        nothing else here depends on. Both embeddings are therefore fitted
        inside a guard: whatever they raise becomes a ``NO T-SNE:`` / ``NO
        UMAP:`` note naming the scope, the reason in this package's own words
        (see :func:`~nanoscope.analysis.subpop_cluster.embedding_refusal`) and
        what the run still stands on. Cancellation is re-raised, because a
        cancelled run is not a report with a missing figure.
    """
    notes: list[str] = []
    res = clustering.scopes.get(str(scope))
    if res is None:
        if tsne or umap:
            notes.append(
                f"NO EMBEDDING: scope {scope!r} was not clustered, so items 6-7 "
                f"have no PCA space to embed."
            )
        return None, tuple(notes)
    if (tsne or umap) and res.n > int(max_events):
        notes.append(
            f"NO EMBEDDING: scope {scope!r} has {res.n} events, above the "
            f"{int(max_events)}-event ceiling. t-SNE would dominate the run time of a "
            f"report whose actual findings are above, and it reports no progress while "
            f"it does. Nothing else in this report depends on it."
        )
        return None, tuple(notes)

    cols: dict[str, np.ndarray] = {}
    if tsne:
        try:
            xy = SC.tsne_embedding(res.pca.scores, random_state=random_state)
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - a figure aid may not end the run
            notes.append(_embedding_failure_note("T-SNE", "t-SNE", scope, res, exc))
        else:
            cols["tsne_x"], cols["tsne_y"] = xy[:, 0], xy[:, 1]
    if umap:
        try:
            xy = SC.umap_embedding(res.pca.scores, random_state=random_state)
        except ImportError as exc:
            notes.append(f"NO UMAP: {exc}")
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - a figure aid may not end the run
            notes.append(_embedding_failure_note("UMAP", "UMAP", scope, res, exc))
        else:
            cols["umap_x"], cols["umap_y"] = xy[:, 0], xy[:, 1]
    if not cols:
        return None, tuple(notes)
    return pd.DataFrame(cols, index=res.index), tuple(notes)


# ===========================================================================
# the report
# ===========================================================================


@dataclass
class SubpopReport:
    """Everything the run produced: the stages, the tables and the caveats.

    Attributes:
        features: The gated per-event feature table (:func:`feature_table`).
        gate: The decision on every detected region, survivors and rejects.
        clustering: Items 3-5 per scope, or ``None`` when every scope was too
            small. Never silently absent: the reason is in :attr:`warnings`.
        quantification: Items 8-12, or ``None`` when the mixture was refused.
        waveforms: Item 13's median waveforms, or ``None``.
        embeddings: Items 6-7's 2-D coordinates for the pooled scope, or
            ``None``. Exported, never printed -- see :func:`scope_embeddings`.
        labels: The joined exploratory label table (item 4-5 columns).
        meta: Recording identity, counts, timings, seeds and the config used.
        warnings: Every caveat from every stage, de-duplicated in order. READ
            THIS ONE -- the sub-results each see only their own.
    """

    features: pd.DataFrame
    gate: EF.GateResult
    clustering: SC.SubpopClustering | None = None
    quantification: SG.SubpopQuantification | None = None
    waveforms: pd.DataFrame | None = None
    embeddings: pd.DataFrame | None = None
    labels: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    # -- derived views -------------------------------------------------------
    @property
    def n_fired(self) -> int:
        """The regions the gate was asked about -- i.e. the rows the run was HANDED.

        Not necessarily the rows the detector found: the app passes ``events.accepted``.
        :attr:`region_noun` is the word this number may be printed with, and
        ``meta["n_detected"]`` is the detection's own count when the rows carry it.
        """
        return int(len(self.gate.keep))

    @property
    def region_noun(self) -> str:
        """The noun every surface must use for :attr:`n_fired` -- "accepted region" when
        the rows say they are the curated-in subset, "detected region" otherwise.

        Resolved once, in :func:`rows_in_hand`, and carried in :attr:`meta`, so no
        surface picks a word of its own. See that function for what went wrong when
        they did.
        """
        return str(self.meta.get("region_noun") or DETECTED_NOUN)

    @property
    def table_noun(self) -> str:
        """The noun every surface must print beside ``meta["n_detected"]`` -- "detected
        region(s)" only when nothing has reshaped the table since the detector stamped
        it, "region(s) in the merged table" after a merge, and so on.

        Resolved once, in :func:`rows_in_hand` (via :func:`_table_shape`), and carried
        in :attr:`meta` beside the CURATION caveat it was phrased with, so the summary
        line cannot call the merged table "detected" two rows above a caveat that says
        it is not (S7 E3 / X7).
        """
        return str(self.meta.get("table_noun") or TABLE_NOUNS["detected"])

    @property
    def n_kept(self) -> int:
        """Regions that passed the sustained-depth gate."""
        return int(self.gate.n_kept)

    def event_table(self) -> pd.DataFrame:
        """The per-event table with every label column joined on, for export.

        The one table a reader needs: the thirteen features, the metadata, the
        exploratory labels and (when the mixture ran) the subpopulation id and
        the posterior it was assigned with.
        """
        out = self.features.copy()
        if self.labels is not None:
            for col in self.labels.columns:
                out[col] = self.labels[col]
        if self.embeddings is not None:
            for col in self.embeddings.columns:
                out[col] = self.embeddings[col]
        if self.quantification is not None:
            fit = self.quantification.fit
            out["subpop"] = fit.labels
            out["subpop_confidence"] = pd.Series(fit.confidence, index=fit.labels.index)
        return out

    def tables(self) -> dict[str, pd.DataFrame]:
        """Name -> table, in the order :func:`write_report` writes them."""
        out: dict[str, pd.DataFrame] = {"events": self.event_table()}
        if self.clustering is not None:
            for scope, res in self.clustering.scopes.items():
                out[f"loadings_{scope}"] = res.loadings
        if self.quantification is not None:
            q = self.quantification
            out["populations"] = q.table
            out["cv_grid"] = pd.DataFrame(
                q.selection.grid,
                index=pd.Index(q.selection.p_grid, name="n_pcs"),
                columns=pd.Index(q.selection.k_grid, name="k"),
            )
            if q.crosstab is not None:
                out["crosstab"] = q.crosstab
        if self.waveforms is not None and len(self.waveforms):
            out["waveforms"] = self.waveforms
        return out

    def embedding_names(self) -> tuple[str, ...]:
        """Which of items 6-7 actually produced coordinates."""
        if self.embeddings is None:
            return ()
        return tuple(
            name for name, prefix in (("t-SNE", "tsne_"), ("UMAP", "umap_"))
            if any(c.startswith(prefix) for c in self.embeddings.columns)
        )

    def headline(self) -> str:
        """The one line every surface may state about this run, qualifier included.

        THE SINGLE SOURCE OF THE CLAIM. Before this existed, three surfaces each
        built their own sentence out of ``quantification.fit.k`` -- the report's
        own header, the report window's header and the app's status bar -- and
        all three said a class count while the statistic that says whether the
        classes are apart sat as caveat 2 of 6 inside a nine-thousand-character
        document. The two strings a reader carried away both stated a count and
        neither stated the evidence.

        So the count is not available as a formatted string anywhere except
        here, and here it is welded to
        :meth:`~nanoscope.analysis.subpop_gmm.SubpopQuantification.claim` and to
        the continuum verdict from the independent clustering. A caller that
        wants the bare integer can still read ``quantification.fit.k``; what it
        cannot do is print a sentence about it that this method did not write.
        """
        # The NOUN is part of the claim, not decoration: "120 of 120 detected region(s)"
        # over a run whose 300 detections were curated down to 120 states a false thing
        # about the detection in the one sentence every surface repeats.
        bits = [f"{self.n_kept} of {self.n_fired} {self.region_noun}(s) passed the gate"]
        q = self.quantification
        if q is None:
            bits.append("no mixture was fitted")
        else:
            bits.append(q.claim())
            verdict = self.continuum_verdict()
            if verdict:
                bits.append(verdict)
        n = len(self.warnings)
        bits.append("no caveats" if not n else f"{n} caveat{'s' if n != 1 else ''}")
        return " · ".join(bits)

    def continuum_verdict(self) -> str:
        """What the *independent* method says about the cloud the mixture split.

        Empty when there is nothing to say -- no clustering, no mixture, or a
        density clusterer that did find clusters. Non-empty exactly when the two
        methods disagree about whether there is anything there, which is the
        strongest single line the run can offer and which lived, before this, in
        a caveat whose last clause was "not a failure".
        """
        q = self.quantification
        if q is None or self.clustering is None or q.fit.k <= 1:
            return ""
        scope = str(self.meta.get("gmm_scope", SC.SCOPE_ALL))
        res = self.clustering.scopes.get(scope)
        if res is None or res.n_hdbscan_clusters != 0:
            return ""
        return (
            f"HDBSCAN on the same {res.n} events finds 0 clusters and calls "
            f"{100 * res.hdbscan_noise_fraction:.0f} % of them noise: reads as ONE CONTINUUM"
        )

    def text(self) -> str:
        """The whole report as plain text -- items 3-13, plus what is missing."""
        return _report_text(self)

    def __repr__(self) -> str:
        k = self.quantification.fit.k if self.quantification is not None else None
        return (
            f"<SubpopReport {self.meta.get('recording', '')!r} "
            f"kept={self.n_kept}/{self.n_fired} k={k} "
            f"warnings={len(self.warnings)}>"
        )


def _dedup(items: Sequence[str]) -> tuple[str, ...]:
    """De-duplicate while preserving order -- the same caveat can arrive twice."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def _stage_progress(
    progress: Callable[[float, str], None] | None, lo: float, hi: float
) -> Callable[[float, str], None] | None:
    """Map a stage's own 0-1 onto ``[lo, hi]``; ``None`` stays ``None``."""
    if progress is None:
        return None

    def _emit(frac: float, msg: str) -> None:
        progress(lo + (hi - lo) * float(frac), msg)

    return _emit


def _gmm_note(n: int, n_pcs: int, k: int) -> str | None:
    """The over-parameterisation caveat, or None when there is room for the fit."""
    if n >= SOFT_GMM_EVENTS:
        return None
    per_component = n_pcs + n_pcs * (n_pcs + 1) // 2
    n_params = k * per_component + (k - 1)
    return (
        f"THIN FIT: {n} events for a {k}-component full-covariance mixture in "
        f"{n_pcs} dimensions -- {n_params} free parameters, "
        f"{n / max(n_params, 1):.1f} events per parameter. The held-out likelihood "
        f"cannot flag this, because every candidate in the search is "
        f"over-parameterised together. Treat K and the fractions as a hypothesis, "
        f"not a measurement, below ~{SOFT_GMM_EVENTS} events."
    )


def slow_gmm_note(n: int) -> str | None:
    """The cost warning for a mixture stage that is about to take many minutes.

    The counterpart to :func:`_gmm_note`'s floor: that one says the fit is too
    THIN to trust, this one says it is too BIG to wait for. Public, and pure,
    because the useful moment to say so is BEFORE the 480 cross-validation fits
    start -- a caller that only learns it from ``report.warnings`` has already
    paid. :data:`SLOW_GMM_EVENTS` carries the measured cost curve.

    Args:
        n: Number of gated events entering the mixture stage.

    Returns:
        The warning, or None at or below :data:`SLOW_GMM_EVENTS`.
    """
    if int(n) <= SLOW_GMM_EVENTS:
        return None
    return (
        f"SLOW: {int(n)} events is well past the ~{SLOW_GMM_EVENTS} this stage "
        f"stays interactive at. The cross-validated selection is 480 "
        f"full-covariance EM fits and the bootstrap another 200, and both the "
        f"iterations per fit and the selected K climb with n. Measured: 1076 "
        f"events (the owner's zn_30h) ~2 min, 2000 events ~7 min, ~11000 events "
        f"did not finish in 45. The features and the clustering are unaffected -- "
        f"they stay seconds into the tens of thousands. Nothing is subsampled, so "
        f"the numbers are the real ones; cancelling is safe and takes effect "
        f"within about a second."
    )


def run_subpopulations(
    detrended,
    raw,
    fs: float,
    events: EventSet,
    *,
    recording: str = "",
    scope: str | Sequence[Any] | None = None,
    span_s: tuple[float, float] | None = None,
    duration_s: float | None = None,
    gmm_scope: str | None = None,
    config: SubpopConfig = SubpopConfig(),
    i0_pA: float | None = None,
    signal_for_waveforms=None,
    cancel: CancelFlag | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> SubpopReport:
    """The whole pipeline: gate, features, per-scope clustering, mixture, waveforms.

    THE ONE ENTRY POINT the app calls. Every stage is also public, so a caller
    that already has a feature table can start further in.

    Determinism: ``config.random_state`` seeds the fold split, every PCA, every
    mixture and k-means; ``config.bootstrap_seed`` seeds the resampling draws.
    HDBSCAN takes no seed and is deterministic given its input. Two runs with the
    same arguments give identical output down to the labels -- pinned by a test,
    not asserted here.

    Args:
        detrended: The trace the events were detected on, pA.
        raw: The recording's raw current, pA (for ``i0_pA``).
        fs: Sample rate, Hz.
        events: The detected events. Pass ``eventset.accepted`` to honour
            curation -- the app's entry point does, so the counts in this report
            and the counts on screen are about the same rows.
        recording: Name recorded on every row and in the report header.
        scope: Clustering scope; see :func:`feature_table`.
        span_s: ``(t0, t1)`` in seconds to cut ``time_half`` at; see
            :func:`feature_table`. ``None`` -- what the app passes -- reads the
            span off the rows' own provenance, and the header prints whichever of
            the two the column was actually cut at.
        duration_s: Recording duration, for ``rate_per_s``. ``None`` leaves the
            rate column NaN and says so -- a rate needs a denominator, and taking
            the last event's onset for one understates it.
        gmm_scope: Which scope the mixture is fit on. ``None`` fits the pooled
            table, which is right for the app's single-recording case and wrong
            for a pooled multi-recording one (the owner fit zn_30h alone).
        config: Knobs and seeds.
        i0_pA: Explicit open-pore magnitude, overriding ``raw``.
        signal_for_waveforms: Trace for item 13's overlays; defaults to ``raw``,
            falling back to ``detrended`` when ``raw`` is not given.
        cancel: Cancel token; ambient when ``None``.
        progress: ``(fraction, message)`` sink for the whole run.

    Returns:
        A :class:`SubpopReport`. Stages that could not run are ``None`` with the
        reason in :attr:`SubpopReport.warnings` -- never silently absent.

    Raises:
        ValueError: propagated from the stage that objected.
        OperationCancelled: if the cancel token fires.
    """
    notes: list[str] = []
    if progress is not None:
        progress(0.0, f"Measuring features on {len(getattr(events, 'df', events))} events")

    feats, gate = feature_table(
        detrended, raw, fs, events,
        recording=recording, scope=scope, span_s=span_s, config=config, i0_pA=i0_pA,
        cancel=cancel,
    )
    # ONE SCOPE MEANS THE POOLED SCOPE, spelled as such. `iter_scopes` always yields
    # the pooled scope first and then one per distinct value, so a single-recording
    # table cut on its own name produces two scopes with identical membership and
    # identical numbers -- `ALL` and `zn_30h` -- and a reader has no way to tell that
    # is a tautology rather than a corroboration. Naming the single scope `ALL`
    # collapses them into the one entry they are. (The redundant fit is still
    # computed and then overwritten by an identical one: a single PCA, HDBSCAN and
    # k-means on a table the mixture stage then spends a minute on, so buying the
    # clarity with it is the right trade.)
    if len(feats) and feats["scope"].nunique(dropna=False) <= 1:
        feats["scope"] = SC.SCOPE_ALL
        if gmm_scope is not None and str(gmm_scope) != SC.SCOPE_ALL:
            gmm_scope = None

    n = int(len(feats))
    # WHAT THE ROWS IN HAND ARE, BEFORE ANY COUNT IS PRINTED ABOUT THEM. Resolved from
    # the rows' OWN provenance (see :func:`rows_in_hand`), and first in the caveat list
    # because curation is the first thing that happened to them -- before the gate,
    # before the waveform floor. Every count below is a count of THIS set.
    rows_meta, curation_note = rows_in_hand(events, int(gate.keep.size))
    noun = str(rows_meta["region_noun"])
    if curation_note:
        notes.append(curation_note)
    if n < gate.keep.size:
        notes.append(
            f"GATE: {gate.n_kept} of {gate.keep.size} {noun}(s) reached "
            f"{config.min_depth_sigma:g}x the event-excluded noise in SUSTAINED depth "
            f"and were kept. The rest are not in anything below."
        )
    if n < int(gate.n_kept):
        # The gate is not the only thing that removes a row: an event shorter
        # than `event_features`' `min_samples` has no waveform to measure and is
        # missing from the feature table. The counts on the header come from
        # different stages, so a reader comparing them needs the difference named
        # rather than left as an off-by-ten between two lines of the same report.
        notes.append(
            f"TOO SHORT TO MEASURE: {int(gate.n_kept) - n} of the {int(gate.n_kept)} "
            f"gated region(s) are shorter than the {EF.MIN_EVENT_SAMPLES}-sample floor "
            f"the waveform features need and carry no row in anything below. The "
            f"mixture, the tables and every fraction are over {n} events, not "
            f"{int(gate.n_kept)}."
        )

    # THE CLOCK THE HALVES WERE CUT AT, READ BACK OFF THE TABLE THAT WAS CUT.
    # NOT re-resolved: a second resolution from the same inputs is the exact shape
    # of the defect this repair is about, and two clocks in one document is what
    # made the report split a 22.000 s span three lines under `duration : 8.800 s`.
    span_t0, span_t1 = feats.attrs["analysed_span_s"]
    file_t1 = float(len(detrended)) / float(fs)
    if (span_t0, span_t1) != (0.0, file_t1):
        # A SUB-RANGE IS NOT THE RECORDING, and every sentence below that says
        # "the recording" means this span. Said once, at the top of the caveats,
        # because the alternative is a reader carrying "second half of it" back to
        # a file two and a half times longer than the thing that was measured.
        # Cannot fire on a whole-recording detection -- both detectors stamp
        # `(0, n)` there -- so no existing report grows a caveat.
        notes.append(
            f"SUB-RANGE: the detector analysed {span_t0:.3f}-{span_t1:.3f} s of a "
            f"{file_t1:.3f} s recording ({100 * (span_t1 - span_t0) / file_t1:.1f} % "
            f"of it). Every early/late half below is cut at the midpoint of THAT "
            f"span, {0.5 * (span_t0 + span_t1):.3f} s, and every stationarity "
            f"sentence built on them says 'the recording' where it means this span. "
            f"Nothing outside it was looked for, so no fraction below is evidence "
            f"about the rest of the file."
        )
    meta: dict[str, Any] = {
        "recording": str(recording),
        "fs_hz": float(fs),
        "duration_s": (None if duration_s is None else float(duration_s)),
        "span_start_s": float(span_t0),
        "span_stop_s": float(span_t1),
        "n_fired": int(gate.keep.size),
        "n_kept": n,
        "gate_baseline_pA": float(gate.scalars.baseline_pA),
        "gate_noise_pA": float(gate.scalars.noise_pA),
        "gate_duty": float(gate.scalars.duty),
        "min_depth_sigma": float(gate.min_depth_sigma),
        "config": config.as_dict(),
        "features": list(EF.CLUST_FEATS),
    }
    # The ledger carries the noun and the detection's own count, so every surface that
    # states a number about this run reads them from here rather than each choosing a
    # word for itself -- the same reason the analysed span rides in this dict.
    meta.update(rows_meta)

    # -- items 3-5: per-scope exploratory clustering -------------------------
    clustering: SC.SubpopClustering | None = None
    labels: pd.DataFrame | None = None
    embeddings: pd.DataFrame | None = None
    if n < 2:
        notes.append(
            f"NO CLUSTERING: {n} event(s) survived the gate. Nothing below the "
            f"gate line in this report was computed."
        )
    else:
        check_cancelled(cancel)
        if progress is not None:
            progress(_P_FEATURES, f"Clustering {n} events")
        clustering = SC.cluster_scopes(
            feats,
            scope_column="scope",
            min_scope_n=config.min_scope_n,
            random_state=config.random_state,
            cancel=cancel,
            progress=_stage_progress(progress, _P_FEATURES, _P_CLUSTER),
        )
        labels = clustering.label_table()
        embeddings, emb_notes = scope_embeddings(
            clustering,
            tsne=config.with_tsne,
            umap=config.with_umap,
            random_state=config.random_state,
        )
        notes.extend(emb_notes)
        for scope_name, count in clustering.skipped.items():
            notes.append(
                f"SCOPE {scope_name!r} NOT CLUSTERED: {count} event(s), below the "
                f"{config.min_scope_n}-event floor. It is absent from the label "
                f"columns, which is not the same as it having no structure."
            )
        for scope_name, res in clustering.scopes.items():
            if res.n_hdbscan_clusters == 0:
                notes.append(
                    f"SCOPE {scope_name!r}: HDBSCAN found no cluster at all -- every "
                    f"event is noise. That reads as a continuum, and it is the "
                    f"NORMAL outcome for a within-recording scope here, not a failure."
                )

    # -- items 8-12: the mixture ---------------------------------------------
    quant: SG.SubpopQuantification | None = None
    sub = feats if gmm_scope is None else feats[feats["scope"].astype(str) == str(gmm_scope)]
    if gmm_scope is not None and not len(sub):
        notes.append(
            f"NO MIXTURE: scope {gmm_scope!r} has no events in the gated table."
        )
    elif len(sub) < MIN_GMM_EVENTS:
        notes.append(
            f"NO MIXTURE: {len(sub)} event(s), below the {MIN_GMM_EVENTS}-event floor. "
            f"A search over full-covariance mixtures up to K={max(config.k_grid)} in up "
            f"to {len(EF.CLUST_FEATS)} dimensions on this many events returns an argmax "
            f"computed from singular covariances, and nothing about the number it "
            f"returns would look wrong. Items 8-12 were not computed."
        )
    else:
        check_cancelled(cancel)
        if progress is not None:
            progress(_P_CLUSTER, f"Fitting subpopulations on {len(sub)} events")
        expl = None
        expl_name = "hdbscan"
        if labels is not None:
            expl_name = "hdbscan" if gmm_scope is None else f"hdbscan_{gmm_scope}"
            if expl_name in labels.columns:
                col = labels.loc[sub.index, expl_name]
                # Integer labels where they are complete: a float column crosstabs
                # to '0.0'/'-1.0' headers, and -1 (noise) is the column a reader
                # most needs to recognise.
                expl = col.astype("int64") if col.notna().all() else col
            else:
                notes.append(
                    f"NO CROSSTAB: no exploratory labels for scope "
                    f"{gmm_scope or SC.SCOPE_ALL!r} -- item 12 was not computed."
                )
        # Said BEFORE the 480 fits start, so it reaches a progress sink while the
        # waiting is still ahead of the caller rather than behind them.
        slow = slow_gmm_note(len(sub))
        if slow:
            notes.append(slow)
            if progress is not None:
                progress(
                    _P_CLUSTER,
                    f"Mixture search on {len(sub)} events — this will take many "
                    f"minutes; cancelling is safe",
                )
        quant = SG.quantify_subpopulations(
            sub,
            duration_s=duration_s,
            exploratory_labels=expl,
            exploratory_name=expl_name,
            k_grid=config.k_grid,
            n_splits=config.n_splits,
            n_boot=config.n_boot,
            random_state=config.random_state,
            bootstrap_seed=config.bootstrap_seed,
            cancel=cancel,
            progress=_stage_progress(progress, _P_CLUSTER, _P_GMM),
        )
        notes.extend(quant.warnings)
        thin = _gmm_note(len(sub), quant.fit.n_pcs, quant.fit.k)
        if thin:
            notes.append(thin)
        # THE TWO METHODS DISAGREE ABOUT WHETHER THERE IS ANYTHING THERE, and
        # that has to be a caveat of its own rather than an inference a reader
        # is left to draw by putting the clustering note and the mixture's K
        # side by side. The clustering note above ends "not a failure", which is
        # true of HDBSCAN and stops being the whole story the moment a mixture
        # reports several classes on the same events.
        scope_key = SC.SCOPE_ALL if gmm_scope is None else str(gmm_scope)
        res = None if clustering is None else clustering.scopes.get(scope_key)
        if res is not None and res.n_hdbscan_clusters == 0 and quant.fit.k > 1:
            notes.append(
                f"DENSITY SAYS CONTINUUM, MIXTURE SAYS {quant.fit.k} CLASSES: on the same "
                f"{res.n} events, HDBSCAN found 0 clusters and called "
                f"{100 * res.hdbscan_noise_fraction:.0f} % of them noise, while the mixture "
                f"reports {quant.fit.k} subpopulations. The crosstab below therefore has one "
                f"column and corroborates nothing. Two methods that share no machinery "
                f"agreeing is evidence; one of them finding no cluster at all while the "
                f"other reports {quant.fit.k} is the shape of a continuum being subdivided."
            )
        meta["gmm_scope"] = SC.SCOPE_ALL if gmm_scope is None else str(gmm_scope)
        meta["gmm_n_events"] = int(len(sub))

    # -- item 13 (half): the median waveforms --------------------------------
    waves: pd.DataFrame | None = None
    if config.with_waveforms and quant is not None:
        check_cancelled(cancel)
        if progress is not None:
            progress(_P_GMM, "Averaging per-subpopulation waveforms")
        src = signal_for_waveforms
        if src is None:
            src = raw if raw is not None else detrended
        waves = subpopulation_waveforms(
            src, fs, events, quant.fit.labels, n_points=config.waveform_points
        )
    # The caveat has to describe what this run actually produced. Stated
    # unconditionally it promises a waveforms table that is absent from
    # rep.tables() and from the export whenever with_waveforms is off.
    notes.append(
        (
            "ITEM 13 IS PARTIAL IN THESE FILES: the per-subpopulation MEDIAN "
            "WAVEFORMS are here (and exported as a table). The representative "
            "per-subpopulation galleries and the coloured full-recording map / "
            "zoom overlays are drawn in the app and are VIEWS, so no picture of "
            "them leaves in this export."
            if waves is not None
            else "ITEM 13 IS ABSENT from this run: the per-subpopulation median "
            "waveforms were NOT computed (with_waveforms is off"
            + ("" if quant is not None else " / no mixture was fitted")
            + "), so this export carries no waveform table; the galleries and "
            "trace overlays are views and never did."
        )
        + " The events table's `subpop` column is what both are drawn from — "
        "export it to rebuild them elsewhere, or use the report window's "
        "'Event gallery…' and 'Colour the trace' buttons to see them here."
    )

    if progress is not None:
        progress(1.0, "Subpopulation report ready")

    return SubpopReport(
        features=feats,
        gate=gate,
        clustering=clustering,
        quantification=quant,
        waveforms=waves,
        embeddings=embeddings,
        labels=labels,
        meta=meta,
        warnings=_dedup(notes),
    )


# ===========================================================================
# the text
# ===========================================================================

_RULE = "=" * 78
_THIN = "-" * 78


def _fmt(value: Any, nd: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(f) else f"{f:.{nd}f}"


def _report_text(rep: SubpopReport) -> str:
    m = rep.meta
    out: list[str] = [
        _RULE,
        "EVENT SUBPOPULATIONS -- waveform features, exploratory clustering, mixture",
        _RULE,
        f"recording      : {m.get('recording') or '(unnamed)'}",
        f"sample rate    : {_fmt(m.get('fs_hz'), 1)} Hz",
        f"duration       : {_fmt(m.get('duration_s'), 3)} s",
    ]
    # THE CLOCK, BESIDE THE DURATION IT IS NOT. `duration` is the analysed
    # duration -- the rate column's denominator -- and the halves are cut on the
    # analysed SPAN. On a whole-recording detection they describe the same
    # stretch; on a sub-range they do not, and this line is where a reader sees
    # which one every `early_frac` below was measured against.
    if m.get("span_start_s") is not None and m.get("span_stop_s") is not None:
        out.append(
            f"analysed span  : {_fmt(m.get('span_start_s'), 3)} - "
            f"{_fmt(m.get('span_stop_s'), 3)} s  (the clock every early/late half "
            f"below is cut at)"
        )
    out.append("")
    # THE CLAIM AND ITS EVIDENCE, AT THE TOP, IN ONE PLACE. This block exists
    # because the class count used to be stated three times in prominent places
    # and qualified once, in a caveat two thirds of the way down. It is written
    # by `SubpopReport.headline`, the same method the window header calls, so
    # the two cannot drift apart.
    out += _headline_block(rep)
    out += [
        "ITEMS 1-2  GATE AND FEATURES",
        _THIN,
        # THE NUMBER IS LABELLED WITH THE WORD FOR ITSELF. `region_noun` is "accepted
        # regions" whenever the rows say they are the curated-in subset (which is what
        # the app hands in), and only "detected regions" when they really are the
        # detection. Both nouns are 16 characters, so the column does not move.
        f"{rep.region_noun + 's':<24}: {rep.n_fired}",
    ]
    # THE REMOVAL THAT HAPPENED BEFORE THIS REPORT, WITH ITS COUNT, DIRECTLY UNDER the
    # count it explains and above the removals that happened inside the run. The
    # detection's own total is the number the reader has in hand from the Analysis panel;
    # without this line the two documents disagree by the curation and neither says so.
    # Printed only when the rows carried the detector's count -- never reconstructed.
    # The noun after the count is `rep.table_noun`, resolved with the caveat this line
    # points at: after a merge the count is the MERGED table's, and this line used to
    # call it "detected region(s)" two rows above a caveat saying it was not.
    n_detected = m.get("n_detected")
    if n_detected is not None and int(n_detected) > rep.n_fired:
        out.append(
            f"{'curated out (not below)':<24}: {int(n_detected) - rep.n_fired}"
            f"  (of {int(n_detected)} {rep.table_noun}; see the CURATION caveat)"
        )
    out += [
        f"passed the >={_fmt(m.get('min_depth_sigma'), 1)}-sigma sustained gate : "
        f"{rep.n_kept}"
        + (f"  ({100 * rep.n_kept / rep.n_fired:.1f} %)" if rep.n_fired else ""),
        f"event-excluded baseline : {_fmt(m.get('gate_baseline_pA'), 3)} pA",
        f"event-excluded noise    : {_fmt(m.get('gate_noise_pA'), 4)} pA (1.4826 x MAD)",
        f"event-like duty cycle   : {_fmt(m.get('gate_duty'), 4)}",
        f"features                : {', '.join(m.get('features', ()))}",
    ]
    out.append("")

    if rep.clustering is None:
        out += ["ITEMS 3-7  NOT COMPUTED (see the caveats below).", ""]
    else:
        out += _clustering_text(rep.clustering)
        out += _embedding_text(rep)

    if rep.quantification is None:
        out += ["ITEMS 8-12  NOT COMPUTED (see the caveats below).", ""]
    else:
        out += _quantification_text(rep.quantification, rep.meta)

    out += _waveform_text(rep)

    out += [_RULE, "CAVEATS -- every one of them, from every stage", _RULE]
    if not rep.warnings:
        out.append("(none)")
    for i, w in enumerate(rep.warnings, 1):
        out.append(f"{i:2d}. {w}")
    out.append("")
    return "\n".join(out)


def _wrap(text: str, width: int = 78, indent: str = "  ") -> list[str]:
    """Fold one long sentence into the report's column width."""
    words = text.split()
    lines: list[str] = []
    cur = indent
    for w in words:
        if len(cur) + len(w) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += w + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def _headline_block(rep: SubpopReport) -> list[str]:
    """What this run found, and in the same breath how well it is supported."""
    out = [_RULE, "WHAT THIS RUN FOUND", _RULE]
    out += _wrap(rep.headline(), indent="")
    q = rep.quantification
    if q is not None and q.separation is not None and not q.separation.substantial:
        out += [""]
        out += _wrap(
            "READ THAT SILHOUETTE BEFORE THE POPULATION TABLE. Held-out likelihood, "
            "BIC and the posteriors are properties of the DENSITY and are all "
            "satisfied by one cloud cut into contiguous wedges; the silhouette is "
            "the only number here that says whether the classes are apart, and it "
            "says they are not. The table below is a profile of regions of one "
            "cloud unless something outside this report says otherwise.",
            indent="",
        )
    out.append("")
    return out


def _k_margin_lines(sel: SG.PCSelection) -> list[str]:
    """By how much the chosen K beat its nearest rival, in the profile's own units."""
    prof = sel.k_profile()
    rivals = {k: v for k, v in prof.items() if k != sel.k and np.isfinite(v)}
    if not rivals:
        return []
    runner = max(rivals, key=lambda k: rivals[k])
    gap = float(prof[sel.k]) - float(rivals[runner])
    finite = [v for v in prof.values() if np.isfinite(v)]
    span = (max(finite) - min(finite)) if len(finite) > 1 else float("nan")
    span_txt = (
        "" if not np.isfinite(span) or span <= 0
        else f", i.e. {100 * gap / span:.1f} % of the {span:.2f}-nat spread of this profile"
    )
    return [f"{f'margin over K={runner}':<17}: {gap:+.4f} nats/event{span_txt}"]


def _separation_lines(q: SG.SubpopQuantification) -> list[str]:
    """The mixture's own separation, beside the one already printed for k-means."""
    sep = q.separation
    if sep is None:
        return [
            "separation       : not measured — a silhouette is undefined at K = "
            f"{q.fit.k} (nothing to be apart from)."
        ]
    ref = (
        "n/a"
        if sep.reference_silhouette is None
        else f"{sep.reference_silhouette:.4f}"
    )
    lines = [
        f"separation       : silhouette {sep.silhouette:.4f} over the reported "
        f"{q.fit.k} class(es),",
        f"                   measured in the {sep.n_pcs}-PC space the mixture was FITTED in"
        + (f" (on {sep.n_used} of {sep.n_events} events)" if sep.subsampled else "")
        + f" — {sep.verdict}",
        f"                   {100 * sep.neg_fraction:.1f} % of events have a NEGATIVE "
        f"silhouette; {100 * sep.nearer_other_fraction:.1f} % sit nearer another class's centre",
        f"                   reference: k-means K={sep.reference_k} on the same "
        f"coordinates scores {ref}",
    ]
    # THE CEILING, AND WHY IT IS NOT A RIVAL RESULT. k-means minimises the
    # within-class scatter a silhouette penalises, so it out-scores a
    # full-covariance mixture at the same K on real data as a matter of course;
    # and the mean silhouette falls with K on a continuum, so it out-scores it
    # across K as a matter of arithmetic. Printed as a BOUND on how separated
    # these coordinates can be made to look -- which is the one reading neither
    # bias undermines -- rather than as a partition the reader should have got.
    ceil, ceil_k = sep.ceiling_silhouette, sep.ceiling_k
    if ceil is not None and sep.reference_profile:
        row = "  ".join(f"K={k}:{v:.3f}" for k, v in sorted(sep.reference_profile.items()))
        lines += [
            f"                   ceiling: the best plain k-means at ANY searched K scores "
            f"{ceil:.4f} (at K={ceil_k})",
            f"                   k-means silhouette by K: {row}",
            "                   (that is a BOUND on separability, measured with the "
            "method the silhouette",
            "                    favours — not a rival result and not a suggested K)",
        ]
    lines += [
        "                   (bands: <=0.25 no substantial structure, <=0.50 weak, "
        "<=0.70 reasonable)",
        "per-class        : " + "  ".join(
            f"{c}:n={n_c},sil={s:.3f}" for c, (n_c, s, _) in sorted(sep.per_subpop.items())
        ),
    ]
    return lines


def _clustering_text(cl: SC.SubpopClustering) -> list[str]:
    out = [
        "ITEMS 3-5  PER-SCOPE PCA, HDBSCAN AND k-MEANS",
        _THIN,
        "Scaling and PCA are RE-FIT WITHIN EACH SCOPE, so a scope's components are",
        "its own -- PC1 of one scope is not PC1 of another and they must not be",
        "compared coordinate for coordinate.",
        "",
    ]
    for scope, res in cl.scopes.items():
        pca = res.pca
        evr = ", ".join(f"{v:.3f}" for v in pca.explained_variance_ratio[: pca.n_kept])
        sil_h = (
            "n/a (fewer than two clusters)"
            if res.hdbscan_silhouette is None
            else f"{res.hdbscan_silhouette:.3f}"
        )
        sil_k = (
            "n/a" if res.kmeans3_silhouette is None else f"{res.kmeans3_silhouette:.3f}"
        )
        out += [
            f"scope {scope!r}: n = {res.n}",
            f"  PCs kept          : {pca.n_kept} of {len(pca.explained_variance_ratio)} "
            f"({100 * pca.variance_captured:.2f} % of variance)",
            f"  variance per PC   : {evr}",
            f"  HDBSCAN           : {res.n_hdbscan_clusters} cluster(s), "
            f"{100 * res.hdbscan_noise_fraction:.1f} % noise "
            f"(min_cluster_size={res.meta.get('min_cluster_size')}, "
            f"min_samples={res.meta.get('min_samples')}), silhouette {sil_h}",
            f"  k-means (k={SC.KMEANS_K})     : sizes "
            f"{np.bincount(res.kmeans3, minlength=SC.KMEANS_K).tolist()}, "
            f"silhouette {sil_k}",
            "  loading table (feature x PC, ranked by info share):",
        ]
        out += ["    " + line for line in res.loadings.to_string().splitlines()]
        out.append("")
    for scope, count in cl.skipped.items():
        out.append(f"scope {scope!r}: SKIPPED, {count} event(s) below the floor.")
    out.append("")
    return out


def _embedding_text(rep: SubpopReport) -> list[str]:
    """Items 6-7: say what was embedded, and refuse to print a coordinate."""
    done = rep.embedding_names()
    out = [
        "ITEMS 6-7  t-SNE / UMAP",
        _THIN,
        "Both consume the scope's own PCA space and NEVER a cluster label, so the",
        "picture cannot be accused of drawing the clusters it is used to display.",
        "By the same token nothing may be concluded from the coordinates, and none",
        "is printed here: they are in the exported events table, to be plotted.",
        "",
    ]
    if not done:
        out += [
            "Neither was computed for this run.",
            "",
        ]
        return out
    out += [
        f"computed for scope {SC.SCOPE_ALL}: {', '.join(done)} "
        f"({len(rep.embeddings)} events)",
        "",
    ]
    if "UMAP" not in done:
        out += [
            "UMAP is OFF by default, and not because it is slow: `random_state`",
            "forces it single-threaded but the result is still not bit-reproducible",
            "across umap-learn / numba / platform versions, so a run and a re-run of",
            "this report can disagree about a column that looks like data. Turn it on",
            "deliberately (SubpopConfig(with_umap=True)) when you want the figure.",
            "",
        ]
    return out


def _quantification_text(q: SG.SubpopQuantification, meta: Mapping[str, Any]) -> list[str]:
    sel, fit, ci = q.selection, q.fit, q.ci
    out = [
        "ITEMS 8-10  MIXTURE: (n_PCs, K) BY CROSS-VALIDATION, CONFIDENCE, BOOTSTRAP",
        _THIN,
        f"scope fit        : {meta.get('gmm_scope', SC.SCOPE_ALL)}  "
        f"(n = {q.n_events})",
        f"chosen           : n_PCs = {sel.n_pcs}, K = {sel.k}",
        f"held-out logL    : {sel.loglik:.4f} nats/event, {sel.n_splits}-fold, "
        f"PPCA residual density on the discarded PCs",
        f"K profile at n_PCs = {sel.n_pcs}:",
        "  " + "  ".join(f"K={k}:{v:.2f}" for k, v in sel.k_profile().items()),
        f"PC profile at K = {sel.k}:",
        "  " + "  ".join(f"p={p}:{v:.2f}" for p, v in sel.p_profile().items()),
    ]
    # THE MARGIN IS PRINTED WHETHER OR NOT IT TRIPS A CAVEAT. "K=7 won" and
    # "K=7 won by 0.11 nats/event out of a 3.5-nat profile" are different
    # claims, and the profile line above states the second one only to a reader
    # who does the subtraction. `_selection_warnings`' FLAT OPTIMUM threshold is
    # left exactly where it was: a number that fires below 0.1 nats is a
    # judgement about when to shout, and this line is the number itself.
    out += _k_margin_lines(sel)
    if fit.bic:
        out += [
            f"BIC cross-check  : argmin at K = {fit.bic_k}"
            + ("  (AGREES)" if fit.bic_k == sel.k else "  (DISAGREES -- see caveats)"),
            "  " + "  ".join(f"K={k}:{v:.0f}" for k, v in sorted(fit.bic.items())),
        ]
    out += _separation_lines(q)
    out += [
        f"assignment       : {100 * fit.conf_frac:.2f} % of events above posterior "
        f"{SG.CONFIDENCE_FLOOR:g} (median {fit.conf_median:.4f})",
        "                   (a posterior is how sure the mixture is WHICH component "
        "made an event;",
        "                    it is not evidence the components are apart — that is the "
        "line above)",
        f"bootstrap        : {ci.n_ok}/{ci.n_boot} resamples converged, "
        f"Hungarian-matched, seed {ci.seed}"
        if ci.computed
        else "bootstrap        : not computed (skipped or point-fits-only)",
        "",
        "ITEM 11  POPULATION TABLE  (every profile column is a MEDIAN;",
        "         early_frac is the share of a class's events falling before the",
        "         midpoint of the ANALYSED SPAN'S CLOCK, printed in the header —",
        "         not before the median event,",
        "         which would be 0.50 for every class by construction. ~0.5 means",
        "         present throughout; near 0 or 1 means drift or fouling, not a",
        "         class. Read each one against the pooled value, not against 0.50)",
        _THIN,
    ]
    out += q.table.to_string(index=False).splitlines()
    out.append("")
    if q.crosstab is not None:
        out += [
            "ITEM 12  MIXTURE x EXPLORATORY LABELS",
            _THIN,
            "Two methods that share no machinery. Column -1 is HDBSCAN's NOISE label:",
            "a subpopulation living mostly under it is the one built from the sparse tail.",
            "",
        ]
        out += q.crosstab.to_string().splitlines()
        out.append("")
    return out


def _waveform_text(rep: SubpopReport) -> list[str]:
    out = ["ITEM 13  PHYSICAL VALIDATION", _THIN]
    if rep.waveforms is None or not len(rep.waveforms):
        out += [
            "Not computed (no mixture, or waveforms disabled).",
            "",
        ]
        return out
    counts = rep.waveforms.attrs.get("n_events", {})
    out += [
        "Per-subpopulation MEDIAN waveform, duration-normalised on a common axis",
        f"({len(rep.waveforms)} points, T from {rep.waveforms['T'].iloc[0]:.2f} to "
        f"{rep.waveforms['T'].iloc[-1]:.2f}), measured on the raw trace. Exported in",
        "full as the 'waveforms' table; the depth extremes are:",
        "",
    ]
    for col in rep.waveforms.columns:
        if col == "T":
            continue
        v = rep.waveforms[col].to_numpy(dtype=np.float64)
        if not np.any(np.isfinite(v)):
            out.append(f"  {col}: no event could be averaged (windows off the trace)")
            continue
        out.append(
            f"  {col}: n={counts.get(col, '?')}, median waveform spans "
            f"{np.nanmin(v):.2f} to {np.nanmax(v):.2f} pA"
        )
    out += [
        "",
        # This sentence used to read "are NOT in this pass -- see the caveats."
        # while the caveat twenty lines away, and two enabled buttons on the same
        # window, said the opposite. Both exist; what is true is that they are
        # VIEWS, so a text export carries the `subpop` column they are drawn from
        # and not a picture. Pinned by a test that greps the report TEXT, not the
        # warning -- the previous guard only read the warning and passed while
        # this line went on contradicting it.
        "The raw-event GALLERIES and the rendered trace overlays (full map + zoom)",
        "are drawn in the app, from the events table's `subpop` column: open them",
        "with the report window's 'Event gallery…' and 'Colour the trace' buttons.",
        "They are VIEWS, so no picture of them leaves in this export -- export the",
        "events table to rebuild them elsewhere.",
        "",
    ]
    return out


# ===========================================================================
# export
# ===========================================================================


#: Tables whose row index is a value, not a row number. See :func:`write_report`.
_INDEXED_TABLES = frozenset({"cv_grid", "crosstab"})


def report_paths(path: str | Path, names: Sequence[str] = ()) -> tuple[Path, ...]:
    """Every file :func:`write_report` will write for ``path``, in write order.

    The app's overwrite confirmation asks this BEFORE the run's tables exist, so
    ``names`` is optional: with it, the exact set; without it, the two files that
    are always written. Callers that can name the tables should, so the
    confirmation covers what will actually be overwritten.
    """
    p = Path(path)
    stem = p.with_suffix("") if p.suffix else p
    out = [stem.with_suffix(".txt"), stem.with_suffix(".meta.json")]
    out += [stem.parent / f"{stem.name}_{n}.csv" for n in names]
    return tuple(out)


def write_report(
    rep: SubpopReport, path: str | Path, *, notes: list[str] | None = None
) -> dict[str, Path]:
    """Write the report text, its metadata sidecar and every table as CSV.

    The text is the document; the CSVs are the numbers behind it. Both, always: a
    text report alone cannot be re-analysed, and a pile of CSVs alone does not say
    what was run or what was wrong with it -- and *this* pipeline's caveats are not
    decoration. A ``populations.csv`` of fractions sitting in a folder without the
    label-switching warning beside it is precisely the artefact that must not
    escape, which is why the sidecar is written on the same terms as the numbers.

    ATOMIC AS A SET (``core.atomic_write``, PORT_PLAN rule 21). Eight-ish files land
    under one stem, and a folder holding today's ``populations.csv`` next to last
    week's ``events.csv`` is worse than no export at all: it is indistinguishable
    from a consistent one and the disagreement between them is silent. Every member
    is staged and the set commits together or not at all.

    Args:
        rep: The report.
        path: Base path. The suffix is replaced -- this writes several files.
        notes: Optional sink for the staging layer's identity notes (a symlinked
            target, a case-folding filesystem). Passed straight through.

    Returns:
        ``name -> path`` for everything written (``"text"``, ``"meta"``, and one
        entry per table).
    """
    p = Path(path)
    stem = p.with_suffix("") if p.suffix else p

    written: dict[str, Path] = {}
    text_path = stem.with_suffix(".txt")
    meta_path = stem.with_suffix(".meta.json")
    meta = dict(rep.meta)
    meta["warnings"] = list(rep.warnings)
    # THE SIDECAR IS THE THIRD PLACE THE CLAIM ESCAPES, after the window header
    # and the report text, and it is the one that outlives the session. It
    # already carried `gmm_scope` and the caveats; it now carries the same
    # qualified sentence the other two say, and the separation statistic as
    # numbers, so a script reading the sidecar cannot pick the class count out of
    # it without the evidence beside it.
    meta["headline"] = rep.headline()
    q = rep.quantification
    if q is not None and q.separation is not None:
        sep = q.separation
        meta["separation"] = {
            "silhouette": sep.silhouette,
            "verdict": sep.verdict,
            "substantial": sep.substantial,
            "neg_fraction": sep.neg_fraction,
            "nearer_other_fraction": sep.nearer_other_fraction,
            "reference_k": sep.reference_k,
            "reference_silhouette": sep.reference_silhouette,
            # The ceiling travels with the claim into the sidecar too: a script
            # that reads `silhouette` alone cannot tell "this partition is not the
            # structure" from "there is no separated structure at any K".
            "reference_profile": {str(k): v
                                  for k, v in sorted(sep.reference_profile.items())},
            "ceiling_k": sep.ceiling_k,
            "ceiling_silhouette": sep.ceiling_silhouette,
            "ceiling_substantial": sep.ceiling_substantial,
            "n_used": sep.n_used,
            "n_events": sep.n_events,
            "n_pcs": sep.n_pcs,
            "per_subpop": {str(c): list(v) for c, v in sorted(sep.per_subpop.items())},
        }
    tables = rep.tables()

    with staged_write_set(notes=notes) as group:
        group.stage(text_path).write_text(rep.text(), encoding="utf-8")
        group.stage(meta_path).write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        for name, table in tables.items():
            out = stem.parent / f"{stem.name}_{name}.csv"
            # The index is written only where it CARRIES something: the feature name
            # on a loading table, the PC count on the CV grid, the subpopulation on
            # the crosstab. Everywhere else it is a RangeIndex, and an unnamed
            # integer column in an exported CSV is one more thing for a reader to
            # mistake for data.
            table.to_csv(
                group.stage(out),
                index=name in _INDEXED_TABLES or name.startswith("loadings_"),
            )
            written[name] = out

    written["text"] = text_path
    written["meta"] = meta_path
    return written
