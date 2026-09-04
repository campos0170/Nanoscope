"""The multi-source model: every per-event table an analysis window may overlay.

Before this module the Plots window was offered exactly TWO frames -- the CURRENT
detection and the manual-measurement table -- so the two comparisons a nanopore
paper is actually built out of were impossible: *this pore vs that pore*, and
*this segment vs that segment of the same recording*. This is the list that makes
them possible.

PURE by construction (no Qt, no :class:`~nanoscope.core.session.Session`): it takes
the documents and a result lookup and returns value objects, so the whole discovery
rule is testable without a window. :class:`~nanoscope.gui.main_window.MainWindow`
stays the wirer; it just calls :func:`discover_plot_sources`.

Three properties this file exists to guarantee:

**Identity is the persistent key, never the display name.** A segment may carry the
same user-typed name as its parent (nothing stops "baseline" being cut out of
"baseline"), and two files in different folders share a basename all the time. So
:attr:`PlotSource.key` is built from :func:`nanoscope.core.segment.persistent_key`
-- the identity the project format already round-trips -- and only the *label* is
the display name. Two same-named sources therefore overlay as two distinct curves
instead of one silently clobbering the other.

**A label a human can act on.** Colliding display names are disambiguated (by
channel first, then by a stable ordinal), because a legend that reads
"baseline / baseline" is not a legend.

**Provenance travels WITH the frame.** Each source carries its own detection
provenance, so a per-source dwell fit resolves ``t_min`` from *its own* detector's
``min_dwell_s`` (see :func:`nanoscope.analysis.modules.dwell_fit_inputs`). Applying
one recording's detection floor to another recording's dwell times is exactly the
bias the provenance work removed; the overlay must not re-introduce it by pooling.

**Gain comparability is a source property.** The current gain in this project is
UNCALIBRATED by default, so absolute pA is NOT comparable across recordings while
dwell and ``frac_depth`` (dI/I0) are. Each source records its calibration
fingerprint so a window about to plot an absolute-pA column across two recordings
can say so out loud (:func:`gain_comparability`).

**And so is BASELINE comparability, which is the same kind of rule.** A depth is a
distance from an ESTIMATED baseline, so two recordings probed under different
baseline windows report different measurements of the same pore -- measured on the
owner's apoferritin ladder, re-detecting the -700 mV recording under the -600 mV
recording's windows moved its median ``frac_depth`` by 11.3 %. That rule was written
inside ``analysis/voltage_dep.py`` and therefore existed on exactly one of the two
surfaces that pool recordings. It lives here now (:func:`pool_comparability`,
:func:`column_depends_on_baseline`, :func:`merge_state`), next to the gain rule,
with ``voltage_dep`` and the Plots window as its two callers and NO second copy.

**And the gain rule is INSIDE that one call, not beside it.** Extracting the
baseline/merge half left :func:`pool_comparability` answering two thirds of the
question, and a two-thirds answer is something a caller can take for the whole one:
Plots asked this AND :func:`gain_comparability` and got the full verdict, while the
voltage-dependence window asked only this and went silent on a fitted law over
``median depth`` in absolute pA across three uncalibrated pores -- measured
2026-08-22, ``V₀ = -0.432 V, R² = 0.9157``, no caveat anywhere in the window, while
Plots on ``depth_pA`` over the identical three sources printed the gain CAUTION.
So :func:`pool_comparability` now asks all three halves and reconciles them (
``frac_depth`` survives a gain disagreement but NOT a baseline one, and the gain
advice used to recommend it regardless), and :func:`gain_comparability` is left as
what it always was -- one third, for the tests and the PSD window's sibling
question, never the entry point for a surface that pools recordings.

**And the SIGNAL-DOMAIN half: is this pA at all?** (Not to be confused with
``voltage_dep.duration_comparability``, which calls itself "the fourth half" and
argues -- correctly -- that it does NOT belong here: an analysed duration is a
property of one metric's arithmetic, not of how a detection was measured. This one
is the latter, so it does.) The three above all assume the numbers on
the axis are currents, and two ordinary settings make that false without moving any
of them. ``Preprocess ▸ Normalize`` (``preprocess_for_detection`` stage 7) rescales
the whole trace onto a dimensionless axis before either detector sees it, so
``depth_pA`` carries sigmas -- or, in ``model_input`` mode, a BOUNDED [-1, +1] model
input -- while keeping its "pA" spelling; the only disclosure was a live banner in
the Detection panel, which does not travel with the table, so a normalized run and
an ordinary one pooled onto one ``depth_pA`` axis in silence. And
``ChannelCalibration.target_units`` is deliberately outside ``fingerprint()``
("target_units is a label"), so a pA recording and an nA one agree on every half of
the rule and overlay a thousand-fold apart. :func:`signal_domain_of` reads both off
the provenance that already records them, and :func:`pool_comparability` groups on
it. It is NOT folded into the gain half: a bounded clip is a different failure from
a scale factor, and the escapes differ -- ``frac_depth`` survives a change of gain
or of unit (both pure scales) but NOT Normalize, which is affine and puts the dI/I0
denominator at ~0 (measured, 872x-1048x; see :data:`_SIGNAL_DOMAIN_FREE_COLUMNS`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

from .document import Document
from .layer import ColourRegistry, colour_key
from .segment import SegmentDocument, persistent_key

__all__ = [
    "ABSOLUTE_CURRENT_COLUMNS",
    "DEFAULT_SIGNAL_DOMAIN",
    "DEFAULT_DETECTION_MODULE_ID",
    "DETECTION_MODULE_IDS",
    "DETECTOR_NAMES",
    "GAIN_ALTERNATIVES",
    "GAIN_SAFE_ALTERNATIVES",
    "GAIN_UNSTATED",
    "MANUAL_SOURCE_KEY",
    "MANUAL_SOURCE_LABEL",
    "NORMALIZED_DOMAIN_PREFIX",
    "SOURCE_COLOURS",
    "UNKNOWN_BASELINE",
    "UNMERGED",
    "PlotSource",
    "PoolVerdict",
    "PooledItem",
    "SourceVerdict",
    "baseline_fingerprint_of",
    "column_depends_on_baseline",
    "column_depends_on_gain",
    "column_depends_on_signal_domain",
    "detector_name",
    "disambiguate_labels",
    "discover_plot_sources",
    "gain_comparability",
    "gain_label",
    "gains_comparable",
    "is_normalized_domain",
    "merge_state",
    "pool_comparability",
    "signal_domain_of",
    "source_colour",
    "source_comparability",
    "source_key",
]

#: EVERY detector result id discovery looks for, in the order the sources are
#: offered. NOT a preference list any more, and that is the whole point.
#:
#: It used to be ("detect.schmitt", "detect.keras") and it was read BEST-FIRST: the
#: first id that had a stored result on a channel won and the rest were dropped, one
#: source per (doc, channel). Two things followed, both measured on the owner's
#: -1000 mV recording (749k samples) on 2026-09-02:
#:
#:   * A CHANNEL'S OTHER DETECTIONS WERE UNREACHABLE. Schmitt (110 events) and CUSUM
#:     (250) were held simultaneously by ``ResultStore`` -- which has ALWAYS been keyed
#:     (doc_id, channel, module_id) -- the Analysis panel showed CUSUM, and the Plots
#:     window offered ONE source, drew Schmitt's 110 rows, flagged it ``current=True``
#:     and labelled it with the filename alone. A figure exported from that window is
#:     of a table the user was not looking at, with nothing on it that says so.
#:   * ``detect.cusum`` WAS MISSING ENTIRELY, so a CUSUM-only recording produced zero
#:     plot sources: 250 events on screen and "Run detection first" on a greyed
#:     Analysis > Plots. (``gui/workflow_status.py`` kept the correct three-id copy
#:     ten files away, which is how the two drifted.)
#:
#: Discovery now yields one source per (doc, channel, module_id), so the whole store
#: is offered and the window can overlay two detectors on one recording -- which is
#: detector comparison, and it costs no new surface.
DETECTION_MODULE_IDS: tuple[str, ...] = (
    "detect.schmitt", "detect.cusum", "detect.keras", "detect.reconciled"
)

#: The detector whose source key carries NO module suffix -- see :func:`source_key`.
#: Schmitt is the app's default engine, so every key already written into a project
#: file, a saved Plots template or a QSettings label override names a Schmitt source;
#: anchoring the unsuffixed form to it keeps those keys resolving after this change.
DEFAULT_DETECTION_MODULE_ID: str = "detect.schmitt"

#: Short human name per detector, for a legend entry and a source row. THE one place
#: this project spells them for a plot; ``gui/main_window.py`` still keeps a private
#: copy (``_DETECTOR_DISPLAY_NAMES``, which says "ML" where this says "Keras") for its
#: report-window sentence, and should import this instead -- see the report.
DETECTOR_NAMES: Mapping[str, str] = MappingProxyType({
    "detect.schmitt": "Schmitt",
    "detect.cusum": "CUSUM",
    "detect.keras": "Keras",
    "detect.reconciled": "Reconciled",
})


def detector_name(module_id: Any) -> str:
    """The short name a legend calls ``module_id`` by ("Schmitt", "CUSUM", "Keras").

    An id this project does not run reads back AS ITS ID rather than as a guess or a
    blank: a source labelled with a module id is ugly, and a source labelled with
    nothing is the defect this module exists to close.
    """
    if not module_id:
        return ""
    return DETECTOR_NAMES.get(str(module_id), str(module_id))


def source_key(doc_key: str, channel: int | None, module_id: Any = None) -> str:
    """THE one spelling of a plot source's identity.

    ``colour_key(doc_key, channel)`` for the DEFAULT detector, and that form suffixed
    with ``"@<module id>"`` for any other. Both halves are load-bearing:

    * A channel can now carry SEVERAL sources at once (one per detector), so the
      identity has to name the detector or the second one silently clobbers the first
      -- exactly the collision :func:`~nanoscope.core.segment.persistent_key` was
      introduced to stop between a segment and its parent.
    * The DEFAULT detector keeps the bare, historical key, so every colour, legend
      label override and template that was stored against a Schmitt source before this
      change still resolves to it -- and so a Schmitt source is still the same ink in
      the trace view and in the plots (``core.layer.colour_key`` is that ink's key).

    The suffix is stable across a reopen (it is a module id, never a display name) and
    does not move when a sibling detection appears or is dropped -- the anchor is
    :data:`DEFAULT_DETECTION_MODULE_ID`, not "whichever detector was found first".
    """
    base = colour_key(doc_key, channel)
    mid = str(module_id or "")
    if not mid or mid == DEFAULT_DETECTION_MODULE_ID:
        return base
    return f"{base}@{mid}"

#: The manual-measurement table's stable key + label. It is not a detection: it has
#: no document, no channel and no provenance, and the windows say so rather than
#: inventing a detection floor for it.
MANUAL_SOURCE_KEY = "manual-measurements"
MANUAL_SOURCE_LABEL = "Manual measurements"

# Okabe-Ito, the established categorical palette (plot.style.OKABE_ITO). Imported
# lazily-by-value so this module stays free of the matplotlib import that
# plot.style pulls in -- the colours are a contract, not an implementation detail.
#
# *** THIS IS THE PALETTE COPY THE GUARDS DO NOT COVER. ***
# ``tests/gui/test_subpop_views.py::test_neither_caller_hard_codes_a_categorical_palette``
# reads the AST of ``gui.subpop_window`` and ``gui.plots_panel`` and fails on any
# Okabe-Ito hex written as a literal. It is parametrised over those two GUI modules
# only, so this tuple -- eight Okabe-Ito hexes spelled out by hand, one import level
# BELOW the palette it copies -- has never been in its scope. The copy is deliberate
# (the matplotlib import is real) but "deliberate" is not "self-checking", so
# ``tests/core/test_sources.py::test_the_source_palette_has_not_drifted_from_okabe_ito``
# now pins it BY VALUE against ``plot.style.OKABE_ITO``. Value-pinning rather than
# extending the AST guard, because extending the guard would only forbid the copy
# this module needs; what actually matters is that the two never disagree.
# The copy disappears entirely the day ``plot.style`` imports its palette from here
# instead (one line, the right direction in the tree) -- see the report.
SOURCE_COLOURS: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

#: A calibration that multiplies by 1 and adds 0 has not been calibrated at all.
_IDENTITY_FINGERPRINT = (1.0, 0.0)


def source_colour(i: int) -> str:
    """Categorical colour ``i``, cycling the Okabe-Ito palette.

    POSITIONAL, and that is its whole problem: ``i`` is a source's index in the
    discovery list, so closing one recording, re-detecting on another channel or
    reordering the Files list silently repaints EVERY source. Kept because the
    fallback path below still needs it (and because a caller with no registry is a
    legitimate caller -- the pure-function tests are three of them), but a surface
    the user compares things in should hand :func:`discover_plot_sources` a
    :class:`~nanoscope.core.layer.ColourRegistry` and get identity-keyed inks
    instead.
    """
    return SOURCE_COLOURS[int(i) % len(SOURCE_COLOURS)]


@dataclass(frozen=True, eq=False)
class PlotSource:
    """One per-event table an analysis window can draw, with everything needed to
    draw it *honestly* alongside another one.

    ``eq=False`` (so identity equality/hashing) on purpose: a value-based ``__eq__``
    would compare DataFrames -- which raises rather than returning a bool -- and the
    generated ``__hash__`` would try to hash one. Sources are compared by
    :attr:`key`, which is what the key exists for.

    Attributes:
        key: STABLE, UNIQUE identity. Built from
            :func:`~nanoscope.core.segment.persistent_key` (plus the channel), so a
            segment named identically to its parent never collides with it, and a
            refresh re-identifies the same source across a project round-trip.
        doc_key: The document's persistent key on its own (the project identity).
        label: What the legend says -- the display name, disambiguated by channel
            (or a stable ordinal) only when it would otherwise be a duplicate.
        df: The FULL per-event frame (not pre-filtered to accepted): the window's
            own *accepted only* control does that filtering.
        provenance: THIS source's detection provenance. Never another source's --
            ``min_dwell_s``/``filter_rise_s``/``threshold_out_pA`` are properties of
            the run that produced these rows.
        colour: The source's Okabe-Ito ink. Identity-keyed when
            :func:`discover_plot_sources` was given a
            :class:`~nanoscope.core.layer.ColourRegistry` -- the same recording is
            then the same colour here, in the trace view and after a reopen --
            and only otherwise assigned by discovery POSITION, which does not
            survive the set changing. :attr:`key` is exactly the registry key
            (:func:`~nanoscope.core.layer.colour_key`), so any surface holding a
            source can ask the registry itself.
        doc_id: In-memory document id (None for the manual table).
        channel: Channel index within that document (None for the manual table).
        module_id: The detector that produced the rows (None for the manual table).
        kind: ``"detection"`` or ``"manual"``.
        is_segment: Whether the document is a derived segment.
        display_name: The raw, undisambiguated display name.
        calibration_fingerprint: ``(gain, offset)`` for the source channel, or None
            when unknown. ``(1.0, 0.0)`` means the gain was never calibrated, which
            is what makes absolute pA incomparable across recordings.
        current: Whether this is the app's currently-selected detection (the window
            pre-selects it, so opening Plots shows what the user was looking at).
        warnings: The DETECTION's own caveats -- ``AnalysisResult.warnings``, whole
            and unfiltered, the same tuple the Summary panel renders. A table
            measured at the unprobed default baseline carries one, and a plotting
            surface that drops it shows the same number with fewer caveats than the
            table it came from. Carried per source because it is a property of the
            run that produced these rows, exactly like :attr:`provenance`.
    """

    key: str
    label: str
    df: Any
    doc_key: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    colour: str = SOURCE_COLOURS[0]
    doc_id: str | None = None
    channel: int | None = None
    module_id: str | None = None
    kind: str = "detection"
    is_segment: bool = False
    display_name: str = ""
    calibration_fingerprint: tuple[float, float] | None = None
    current: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        try:
            return int(len(self.df))
        except TypeError:            # pragma: no cover - a frame always has a length
            return 0

    @property
    def has_provenance(self) -> bool:
        return bool(self.provenance)

    @property
    def detector_label(self) -> str:
        """The short name of the detector that produced these rows ("" for the manual
        table). What a source row, a tooltip and a legend entry name a detector by --
        see :func:`detector_name`; a figure that pools two detectors has to say which
        curve is which, and ``module_id`` is not a thing to print at a reader."""
        return detector_name(self.module_id)

    def events(self) -> tuple[Any, Mapping[str, Any]]:
        """The ``(frame, provenance)`` pair :mod:`nanoscope.analysis.modules` takes.

        Handing this to ``dwell_fit_inputs`` is what makes a per-source fit use
        THIS source's detection floor rather than a neighbour's.
        """
        return self.df, dict(self.provenance)


def _is_calibrated(fp: tuple[float, float] | None) -> bool:
    if fp is None:
        return False
    return tuple(float(x) for x in fp) != _IDENTITY_FINGERPRINT


#: What the gain advice may name as a safe alternative, mapped to whether that
#: alternative ALSO survives a BASELINE disagreement. ``dwell_s`` does (the Schmitt
#: edges are set by the noise ladder); ``frac_depth`` does NOT -- it is
#: ``depth_pA / |baseline_pA|``, so it moves with the baseline window just as
#: ``depth_pA`` does, and offering it to a user whose recordings disagree about
#: their baselines walks them off one cliff onto another.
#:
#: A MAPPING rather than two lists because the drop rule is then a property of the
#: alternative itself: :func:`pool_comparability` filters on the flag. It used to
#: live at ONE call site as ``not a.startswith("frac_depth")`` -- a string sniff that
#: silently answered "keep it" for every alternative a second surface might name.
GAIN_ALTERNATIVES: Mapping[str, bool] = MappingProxyType({
    "dwell_s": True,
    "frac_depth (dI/I0)": False,
})

#: The same alternatives as bare names, in the order they are said. Derived from
#: :data:`GAIN_ALTERNATIVES` so the two cannot drift.
GAIN_SAFE_ALTERNATIVES: tuple[str, ...] = tuple(GAIN_ALTERNATIVES)

#: A pooled item that states NO calibration at all. DISTINCT from ``None``, which
#: states "this recording's gain is not known", a fact :func:`gains_comparable`
#: judges and warns about. ``GAIN_UNSTATED`` means the item is not an amplified
#: recording whose gain could disagree with a neighbour's -- the manual-measurement
#: table is the real case -- so it takes no part in the gain question.
GAIN_UNSTATED = "gain-unstated"


def gain_label(fingerprint: Any) -> str:
    """One display label for a calibration, for a per-source table or a plot title.

    ``(1.0, 0.0)`` is not a calibration, it is the absence of one, and it is the
    state every recording in this project is in by default -- so it says
    ``"uncalibrated"`` rather than printing an identity gain that reads like a
    measurement. An unknown or unreadable fingerprint says ``"unknown"``: not the
    same claim, and not a match for anything.
    """
    if fingerprint is None or fingerprint == GAIN_UNSTATED:
        return "unknown"
    try:
        gain, offset = (float(x) for x in fingerprint)
    except (TypeError, ValueError):
        return "unknown"
    if (gain, offset) == _IDENTITY_FINGERPRINT:
        return "uncalibrated"
    return f"gain {gain:g}, offset {offset:g}"


def _advice(alternatives: Sequence[str]) -> str:
    """``"a and b ARE comparable — plot those instead."`` / the singular of it.

    The sentence that redirects the user. It is generated rather than written out
    because the safe list is not constant: when the sources disagree on their
    baseline windows ``frac_depth`` leaves it, and a fixed sentence would go on
    recommending the column this module exists to protect.
    """
    alts = [a for a in alternatives if a]
    if not alts:
        # A caller with nothing it can vouch for gets SILENCE, not a dangling clause
        # and not a guess. The PSD window passes ``()`` for exactly that reason: it
        # holds calibrations, not detections, so it cannot know whether ``frac_depth``
        # is safe, and it offers the spectrum's SHAPE in its own words instead.
        return ""
    if len(alts) == 1:
        return f" {alts[0]} IS comparable — plot that instead."
    return f" {' and '.join(alts)} ARE comparable — plot those instead."


def gains_comparable(
    fingerprints: Sequence,
    *,
    alternatives: Sequence[str] = GAIN_SAFE_ALTERNATIVES,
) -> tuple[bool, str]:
    """Can absolute currents be compared across these calibrations? ``(ok, why)``.

    Comparable only when every recording shares the SAME non-identity calibration:
    a shared identity gain means nobody has calibrated anything (raw ADC-scaled
    units, pore to pore), and different gains mean the numbers are in different
    units even if both are labelled pA. Times and ratios (dwell, dI/I0, and the
    SHAPE of a noise spectrum) are unaffected by the GAIN either way -- which is
    what ``alternatives`` names.

    Takes bare ``(gain, offset)`` fingerprints so both the Plots window (which holds
    :class:`PlotSource` objects) and the PSD window (which holds recordings) ask the
    same question and get the same answer.

    Args:
        fingerprints: One ``(gain, offset)`` (or None) per recording.
        alternatives: The columns the closing advice may name as comparable
            *instead*. Defaults to :data:`GAIN_SAFE_ALTERNATIVES`. A caller that
            has ALSO found the sources' baselines to disagree must pass the list
            with ``frac_depth`` removed: dI/I0 divides by the baseline, so a
            baseline disagreement makes it exactly as incomparable as the depth,
            and recommending it would walk the user off one cliff onto another.
    """
    if len(fingerprints) < 2:
        return True, ""
    fps = {None if fp is None else tuple(float(x) for x in fp) for fp in fingerprints}
    if len(fps) == 1 and _is_calibrated(next(iter(fps))):
        return True, ""
    if len(fps) > 1:
        return False, (
            "These recordings do NOT share a current calibration, so absolute-current "
            "quantities (depth_pA, amplitude, ecd_pA_s, noise magnitude, …) are in "
            "different units here and must not be compared across them."
            + _advice(alternatives)
        )
    return False, (
        "The current gain is UNCALIBRATED for these recordings, so absolute pA is not "
        "comparable across them (each pore's raw scale is its own)."
        + _advice(alternatives)
    )


def gain_comparability(
    sources: Sequence[PlotSource],
    *,
    alternatives: Sequence[str] = GAIN_SAFE_ALTERNATIVES,
) -> tuple[bool, str]:
    """:func:`gains_comparable` over the DETECTION sources' calibrations.

    The manual-measurement table is skipped: it is a hand-measured table, not a
    second pore whose amplifier gain could disagree.

    NOT the entry point for a surface that pools recordings. This is one THIRD of
    the comparability rule, and a surface that asks only this (or only
    :func:`pool_comparability`'s other two halves) is how the two pooling windows
    came to disagree. Ask :func:`pool_comparability` -- it asks this for you, with
    the baseline verdict in hand so the advice stays honest.
    """
    real = [s for s in sources if s.kind == "detection"]
    return gains_comparable([s.calibration_fingerprint for s in real],
                            alternatives=alternatives)


# ---------------------------------------------------------------------------
# was this pool measured the same way?  (the ONE baseline/merge rule)
# ---------------------------------------------------------------------------
#
# This rule was written for ``Analysis ▸ Voltage dependence`` -- the surface built
# to put several recordings on one axis -- and it lived inside
# ``analysis/voltage_dep.py``. But ``Analysis ▸ Plots`` pools recordings too, and a
# rule that lives inside ONE consumer is a rule the other consumer does not have:
# Plots shipped with no baseline check at all, and its gain guard cheerfully
# redirected the user onto ``frac_depth``, which moves with the baseline.
#
# So it lives HERE, next to :func:`gains_comparable`, the cross-source comparability
# rule Plots already consumed. ``voltage_dep`` imports it; ``plots_panel`` imports it
# through :func:`source_comparability`. There is no second copy, and adding one
# should be treated as a bug: the two surfaces must not be able to disagree about
# whether the same two recordings may be pooled.

#: The merge state of a detection that has not been merged, in the detector or after.
UNMERGED = "unmerged"

#: Stand-in for a detection whose provenance carries no ``baseline_fingerprint``:
#: a pre-fingerprint row, a hand-measured table, or a detector handed a baseline
#: array and so with no estimator to name. NOT a fingerprint -- two of these are
#: not "the same".
UNKNOWN_BASELINE = "unknown"

#: The EventSet columns that do NOT move when the baseline windows move: the
#: DETECTION GEOMETRY (the Schmitt edges, as a sample index or a time) and the
#: curation flags derived from it. Everything else in the 20-column schema is read
#: against ``baseline`` and moves with it.
#:
#: MEASURED, not reasoned. The owner's -500 mV apo detection (1196 events) was
#: re-measured through :func:`~nanoscope.analysis.remeasure.remeasure_events` --
#: same rows, same edges, ONLY the baseline window moved, from the -500's own
#: probed 1.0/0.30 s to the -600's 2.0/0.50 s, i.e. exactly the "re-detect them
#: under ONE baseline window" the warning asks for. Rows whose value changed:
#:
#:     event_id 0    start_idx 0     end_idx 0     t_start_s 0     dwell_s 0
#:     baseline_pA 1181   depth_pA 1126   frac_depth 1181   ecd_pA_s 1181
#:     max_dev_pA 1121    mean_dev_pA 1181   snr 1126
#:     rise_s 838    fall_s 845    n_levels 3
#:
#: ``rise_s``/``fall_s``/``n_levels`` were on THIS list until that was measured, and
#: they do not belong on it. ``rise_fall_over_set`` takes ``baseline`` as an argument
#: and reads its crossings at 0.9 and 0.1 OF THAT EVENT'S OWN DEPTH
#: (``characterize.py``), and ``_count_levels`` segments the deviation FROM the
#: baseline -- so all three are baseline-relative by construction, and the
#: population medians move further than the depth does (rise 1.49 %, fall 2.60 %,
#: against depth_pA 0.33 % and frac_depth 0.09 % on that pair of windows).
#: ``analysis.remeasure`` had this right all along: ``rise_s`` and ``fall_s`` are in
#: its ``REMEASURED_COLUMNS``, the columns a re-measure rewrites, and its
#: ``GEOMETRY_COLUMNS`` is the list below minus the flags. The two are pinned to
#: each other by ``tests/core/test_pool_comparability.py`` so this cannot drift
#: apart again.
#:
#: Stated as the EXCEPTION list on purpose. A column added to the schema and
#: forgotten here is treated as baseline-DEPENDENT, i.e. the comparability check
#: fires for it until somebody proves it should not. The opposite default would let
#: the next amplitude column onto a pooled plot with the silence this exists to
#: remove -- which is precisely what ``rise_s`` was doing.
_BASELINE_FREE_COLUMNS: frozenset[str] = frozenset({
    "event_id", "start_idx", "end_idx", "t_start_s", "dwell_s",
    "filter_limited", "accepted",
})

#: Columns whose values are ABSOLUTE currents (or current-time integrals). With an
#: uncalibrated gain these are pore-specific raw scales, so comparing them ACROSS
#: recordings is meaningless -- unlike ``dwell_s`` (a time) and ``frac_depth`` (a
#: ratio), which survive a gain disagreement (but not a baseline one).
ABSOLUTE_CURRENT_COLUMNS: tuple[str, ...] = (
    "baseline_pA", "depth_pA", "ecd_pA_s", "max_dev_pA", "mean_dev_pA",
    "amplitude", "rms",
)


def column_depends_on_baseline(column: str) -> bool:
    """Does this event column's value move when the baseline windows move?

    The per-column half of the rule, and the only place the question is answered.
    ``depth_pA`` / ``frac_depth`` / ``snr`` are distances from (or ratios to) the
    estimated baseline and answer True, and so -- measured, see
    :data:`_BASELINE_FREE_COLUMNS` -- do ``rise_s``, ``fall_s`` and ``n_levels``,
    which read levels expressed as fractions of that event's own depth. Only the
    detection GEOMETRY answers False: the edges, and the times and flags computed
    from them. Those pool cleanly across recordings probed under their own windows
    -- which is the owner's rule for probing.

    The question is about ONE ROW'S VALUE: "would this number be different had the
    baseline window been the other one?". It is not about which events a
    re-detection would have found (the Schmitt thresholds are k*sigma about the
    baseline, so that set moves too) -- that is a property of the detection, and
    the merge/fingerprint halves of the warning are what say so.

    Fails CLOSED: an unknown column answers True.
    """
    return str(column) not in _BASELINE_FREE_COLUMNS


def column_depends_on_gain(column: str) -> bool:
    """Is this event column an ABSOLUTE current, so it moves with the current gain?

    The per-column half of the GAIN rule, and -- like
    :func:`column_depends_on_baseline` -- the only place the question is answered.
    Both pooling surfaces ask it here, so ``Analysis ▸ Plots`` on ``depth_pA`` and
    ``Analysis ▸ Voltage dependence`` on ``median depth`` cannot classify the same
    column differently.

    ``True`` exactly for :data:`ABSOLUTE_CURRENT_COLUMNS`. Ratios (``frac_depth``,
    ``snr``) and times (``dwell_s``) divide the gain out and answer False; they may
    still be blocked by the BASELINE half, which is a different question.
    """
    return str(column) in ABSOLUTE_CURRENT_COLUMNS


#: The columns a change of SIGNAL DOMAIN cannot touch: the detection geometry and the
#: flags derived from it. Identical to :data:`_BASELINE_FREE_COLUMNS` today, and
#: pinned to it by ``tests/core/test_signal_domain.py`` -- a rescale of the trace
#: moves no sample index and no dwell -- but stated as its OWN name because the two
#: answer different questions and a future column could separate them.
#:
#: ``frac_depth`` is deliberately NOT here, and that is the whole reason this
#: predicate is not just :func:`column_depends_on_gain`. dI/I0 survives a change of
#: GAIN because gain is a scale that divides out of a ratio. It does NOT survive
#: Normalize, because Normalize is AFFINE, not a scale: it subtracts a location too,
#: which puts the open-pore level -- the DENOMINATOR -- at approximately zero.
#: MEASURED here on a synthetic 100 pA blockade in a 1000 pA pore with 10 pA noise,
#: 200 events, through ``preprocess_for_detection`` + ``detect_events``:
#:
#:     raw           baseline_pA  999.97    depth_pA 100.1    frac_depth  0.10012
#:     zscore                       0.0690           7.097               87.328  (872x)
#:     robust                       0.0095           9.903              104.96  (1048x)
#:     minmax                       0.2218           0.5029               2.2658  (22.6x)
#:     model_input                 -0.0001           0.3351             104.53  (1044x)
#:
#: ``snr`` (depth / noise_rms) came through all four at 9.8, unchanged -- both terms
#: scale together and neither carries the offset. It is still on the closed side
#: anyway: ``model_input`` CLIPS, and a clipped depth is not monotone in the real one,
#: so ``snr`` is only unharmed for the three affine modes and this list must not
#: promise it for the fourth.
_SIGNAL_DOMAIN_FREE_COLUMNS: frozenset[str] = frozenset(_BASELINE_FREE_COLUMNS)


def column_depends_on_signal_domain(column: str) -> bool:
    """Does this event column's value move when the trace's SCALE moves?

    The per-column half of the signal-domain rule, and -- like its two siblings --
    the only place the question is answered.

    Fails CLOSED, for the same reason :func:`column_depends_on_baseline` does: a
    column added to the schema and forgotten here is treated as domain-DEPENDENT
    until somebody proves otherwise. Getting that default backwards is what let
    ``depth_pA`` onto a pooled axis carrying sigmas.
    """
    return str(column) not in _SIGNAL_DOMAIN_FREE_COLUMNS


def merge_state(provenance: Mapping[str, Any] | None) -> str:
    """One canonical, comparable label for how (and whether) a detection was merged.

    Reads the two places a merge is recorded, both of which already exist:
    ``merge_gap_s`` (the detector merged while detecting) and ``merged`` (a
    post-hoc :func:`nanoscope.analysis.merge.merge_by_gap` /
    :func:`~nanoscope.analysis.merge.merge_region`, which stamps
    ``{"rule", "window_s"|"region_s", "n_before", "n_after"}``). Both can be set;
    both are named, in that order, so the label is a function of the provenance
    and two equal labels mean two equally-merged sets.

    Args:
        provenance: An :class:`~nanoscope.analysis.events.EventSet`'s provenance
            (``None`` / empty is fine -- it reads as :data:`UNMERGED`).

    Returns:
        :data:`UNMERGED`, or ``"merged: ..."`` naming every merge applied.
    """
    prov = provenance or {}
    parts: list[str] = []
    gap = prov.get("merge_gap_s")
    if gap is not None:
        try:
            parts.append(f"detector gap {float(gap) * 1e3:g} ms")
        except (TypeError, ValueError):
            parts.append(f"detector gap {gap!r}")
    note = prov.get("merged")
    if isinstance(note, Mapping):
        rule = str(note.get("rule", "?"))
        window = note.get("window_s")
        if rule == "gap" and window is not None:
            try:
                parts.append(f"gap {float(window) * 1e3:g} ms")
            except (TypeError, ValueError):
                parts.append(f"gap {window!r}")
        elif rule == "region":
            parts.append("region")
        else:
            parts.append(rule)
    elif note:
        # A truthy note of some other shape still means "this was merged"; say so
        # rather than reading as unmerged because the shape was unexpected.
        parts.append("merged")
    return UNMERGED if not parts else "merged: " + " + ".join(parts)


def baseline_fingerprint_of(provenance: Mapping[str, Any] | None) -> str:
    """A detection's ``baseline_fingerprint``, or :data:`UNKNOWN_BASELINE`."""
    fp = (provenance or {}).get("baseline_fingerprint")
    return str(fp) if fp else UNKNOWN_BASELINE


# --- what SCALE is this table's amplitude column on? ------------------------
#
# The gain half above asks "which pA?". This one asks the question UNDER it:
# "is this pA at all?". Two ways the answer is no, and neither one moves a single
# byte of the three properties the rule already read:
#
#   * ``Preprocess ▸ Normalize`` (``preprocess_for_detection`` stage 7) rescales the
#     WHOLE trace onto a dimensionless axis before either detector sees it, and the
#     detector characterises on that trace. Its own docstring states the consequence
#     -- "NORMALIZE IS LAST, AND IT TAKES THE PICOAMPS WITH IT ... depth_pA, the
#     baseline, thresholds and current-derived pA/pA·s export fields retain physical-
#     unit labels while carrying sigmas or span-fractions" -- and the only disclosure was a LIVE BANNER
#     in the Detection panel, which does not travel with the table. Overlay a
#     normalized run on an ordinary one and Plots pooled them onto one ``depth_pA``
#     axis in silence.
#   * ``ChannelCalibration.target_units``. It is deliberately NOT part of
#     ``fingerprint()`` ("target_units is a label"), so two recordings calibrated to
#     pA and to nA carry the SAME gain fingerprint, agree on every half of the rule,
#     and pool onto one axis a thousand-fold apart.
#
# Both are already in provenance (``normalize_enabled`` / ``normalize_mode`` /
# ``normalize_clip_sigma`` from ``stamp_preprocess_provenance``, ``units`` from
# ``detect_events``); nothing read them. One reader, one label, one half.
#
# AND WHAT DELIBERATELY IS NOT A HALF: ``pore_id``. It is a real confound -- pore
# geometry varies far more than counting noise, which is why ``EventSet.pool()``
# REFUSES a multi-pore concatenation -- and ``Analysis > Plots`` never reads it, so a
# cross-pore ``depth_pA`` overlay is unguarded here. It still must not be grouped on,
# because in this app ``pore_id`` is not a pore identity: nothing sets it by hand.
# ``analysis_panel`` and ``analysis_worker`` both pass ``doc.meta.stem_key``, which is
# the FILE STEM, lowercased. So:
#
#   * five recordings of ONE pore at five voltages -- the owner's apoferritin ladder,
#     the exact figure this window exists to draw -- are five ``pore_id``s, and a
#     warning saying they are different pores would be FALSE;
#   * two SEGMENTS cut from one recording are two ``pore_id``s as well, and that one
#     is by construction: ``segment._segment_path`` folds the segment uid into the
#     stem specifically so "the parent's ``stem_key`` never accidentally matches".
#
# Grouping on it would therefore fire on essentially every overlay this window is
# opened for, while asserting something the provenance cannot support -- and "a guard
# that fires on everything trains the user to ignore it" is this rule's own standard.
# The place to close it is upstream: a pore id the USER sets, distinguishable from the
# auto-derived stem. When that exists, it belongs here as a warning (never a refusal --
# Plots draws and says so) and this is the function to add it to.

#: The signal domain of a trace nothing rescaled. A detection that states no ``units``
#: reads as this rather than as a distinct "unknown": ``pA`` is what the whole app
#: assumes when a channel does not say (``ch.units or "pA"`` in the Inspector, the
#: measure panel and the status chip), and the manual-measurement table -- which
#: carries no provenance at all -- really is in picoamps. A second "unknown" domain
#: here would fire this warning on every hand-measured overlay.
DEFAULT_SIGNAL_DOMAIN = "pA"

#: Prefix every normalized domain label starts with, so :func:`is_normalized_domain`
#: is a property of the label and not a re-read of the provenance.
NORMALIZED_DOMAIN_PREFIX = "normalized"

#: ``filters.MODEL_INPUT_MODE``, restated. ``core`` must not import ``analysis``
#: (``analysis.voltage_dep`` imports THIS module), so the one string crosses as a
#: literal -- and ``tests/core/test_signal_domain.py`` pins it to the real constant
#: so the two cannot drift.
_MODEL_INPUT_MODE = "model_input"

#: What each normalize mode leaves on the axis, in the SAME words the Detection
#: panel's live banner uses (``DetectionPanel._refresh_normalize_units_note``). The
#: banner and this label describe one stage to one user; two vocabularies for it is
#: how a user decides they are two different problems. The banner's parenthetical is
#: flattened to a comma here and nowhere else: this phrase is rendered as the tail of
#: a dash-joined label that itself sits inside a warning's group line, and a bracket
#: three levels in reads as a typo rather than as an aside.
_NORMALIZE_UNITS: Mapping[str, str] = MappingProxyType({
    "zscore": "standard deviations of the whole trace",
    "robust": "baseline-noise sigmas, 1.4826·MAD",
    "minmax": "fractions of the trace's full peak-to-peak span",
})


def is_normalized_domain(domain: str) -> bool:
    """Was this :func:`signal_domain_of` label produced by the Normalize stage?

    The two domain mismatches have DIFFERENT remedies, so the warning has to know
    which one it is looking at: a units mismatch is a label to fix in Calibration and
    ``frac_depth`` survives it, while a normalize mismatch is a re-detect and
    ``frac_depth`` does NOT survive it (see :func:`_domain_remedy`).
    """
    return str(domain).startswith(NORMALIZED_DOMAIN_PREFIX)


def signal_domain_of(provenance: Mapping[str, Any] | None) -> str:
    """What scale this detection's amplitude columns are actually on.

    :data:`DEFAULT_SIGNAL_DOMAIN` (or the recording's own ``units``) for an ordinary
    run; an explicit descriptor naming the MODE -- and, for ``model_input``, the CLIP
    -- when ``Preprocess ▸ Normalize`` was on.

    The mode is in the label because the modes are not interchangeable: ``'robust'``
    divides by the baseline noise sigma, ``'minmax'`` by the trace's full span (which
    the one largest spike in the recording sets), and pooling a run of each is as
    wrong as pooling either with picoamps. The CLIP is in the label for a second
    reason -- ``model_input`` is BOUNDED, so two runs clipped at different sigmas
    disagree about where depth stops being monotone, and the remedy sentence has to
    name a number.
    """
    prov = provenance or {}
    if not prov.get("normalize_enabled"):
        # ``units`` is ``cal.target_units or info.raw_units`` (core.session), stamped
        # by detect_events. Compared verbatim, not case-folded: "nA" and "NA" are not
        # reliably the same claim, and a rule that guesses at unit spellings is a rule
        # that goes quiet on the one pair it was written for.
        return str(prov.get("units") or "").strip() or DEFAULT_SIGNAL_DOMAIN
    mode = str(prov.get("normalize_mode", "zscore"))
    if mode == _MODEL_INPUT_MODE:
        try:
            clip = float(prov["normalize_clip_sigma"])
        except (KeyError, TypeError, ValueError):
            # Only 'model_input' stamps the clip, so a missing one means the
            # provenance is hand-built or pre-dates the key. Say the mode without
            # inventing a bound -- a number this rule guessed would read as measured.
            return f"{NORMALIZED_DOMAIN_PREFIX}: {mode} — dimensionless, CLIPPED"
        return (f"{NORMALIZED_DOMAIN_PREFIX}: {mode} — event-excluded noise sigmas "
                f"/ {clip:g}, CLIPPED to [-1, +1] beyond {clip:g} sigma")
    return (f"{NORMALIZED_DOMAIN_PREFIX}: {mode} — "
            f"{_NORMALIZE_UNITS.get(mode, 'dimensionless units')}")


class PooledItem(NamedTuple):
    """One thing being pooled, reduced to WHAT MEASURED IT.

    The whole input to :func:`pool_comparability`: a name to blame in the warning,
    and the FOUR properties that decide poolability. Both consumers build these --
    a voltage-dependence point from its ``VDepPoint``, a Plots source from its
    ``PlotSource`` -- so the rule never sees either surface's types.

    Attributes:
        label: The name the warnings blame.
        baseline_fingerprint: What estimated the baseline every amplitude here is a
            distance from (:data:`UNKNOWN_BASELINE` when the provenance names none).
        merge_state: :func:`merge_state` of the detection's provenance.
        calibration_fingerprint: The ``(gain, offset)`` this item's currents are in,
            ``None`` when the recording's gain is not known, and
            :data:`GAIN_UNSTATED` when the item is not an amplified recording at all
            (the manual-measurement table) and so takes no part in the gain question.
            Carried HERE, on the item, so that a pooled point cannot exist without
            the units it is in -- the same reason the baseline fingerprint is here.
        signal_domain: :func:`signal_domain_of` the detection's provenance -- WHAT
            SCALE the amplitude columns are on, which the calibration fingerprint
            does not answer (``target_units`` is deliberately outside it, and the
            Normalize stage is outside it entirely). Carried on the item for exactly
            the reason above: a pooled point must not be able to exist without the
            scale its numbers are in. Defaults to :data:`DEFAULT_SIGNAL_DOMAIN` so a
            caller that has not been widened yet says "picoamps" -- which is what
            every caller meant before this field existed -- rather than a fourth
            unknown that would warn about nothing.
    """

    label: str
    baseline_fingerprint: str = UNKNOWN_BASELINE
    merge_state: str = UNMERGED
    calibration_fingerprint: Any = GAIN_UNSTATED
    signal_domain: str = DEFAULT_SIGNAL_DOMAIN


class PoolVerdict(NamedTuple):
    """Whether these things were measured comparably, and why not. ALL FOUR HALVES.

    Attributes:
        depends_on_baseline: Whether the baseline groups are load-bearing for the
            quantity asked about, or merely informational.
        baseline_groups: ``((fingerprint, (label, ...)), ...)``, first-seen order.
            :data:`UNKNOWN_BASELINE` is its own group.
        merge_groups: ``((merge_state, (label, ...)), ...)``, first-seen order.
        warnings: EVERY reason this pool is not comparable -- merge, then baseline,
            then signal domain, then gain -- each naming the offending groups AND
            what to do about it. Empty when it is comparable. This is the list a
            surface renders; there is no second question left to ask afterwards.
        gain_groups: ``((gain_label, (label, ...)), ...)``, first-seen order, over
            the items that STATE a calibration (see :data:`GAIN_UNSTATED`).
        gain_warnings: The subset of :attr:`warnings` that came from the gain half.
            A surface that must name WHICH quantity is blocked needs to know whether
            only the gain fired; nobody has to re-run the rule to find out.
        domain_groups: ``((signal_domain, (label, ...)), ...)``, first-seen order.
        domain_warnings: The subset of :attr:`warnings` that came from the signal
            domain half, for the same reason ``gain_warnings`` exists -- and it is a
            SEPARATE tuple from that one because the two block the same columns for
            different reasons and offer different escapes (a normalize mismatch
            rules out ``frac_depth``; a units mismatch does not).
    """

    # No defaults on the gain OR domain fields, deliberately. A default would let a
    # future constructor produce a verdict that LOOKS complete while silently saying
    # nothing about the gain (or the scale) -- which is the exact shape of the defect
    # this object was widened, twice, to close.
    depends_on_baseline: bool
    baseline_groups: tuple[tuple[str, tuple[str, ...]], ...]
    merge_groups: tuple[tuple[str, tuple[str, ...]], ...]
    warnings: tuple[str, ...]
    gain_groups: tuple[tuple[str, tuple[str, ...]], ...]
    gain_warnings: tuple[str, ...]
    domain_groups: tuple[tuple[str, tuple[str, ...]], ...]
    domain_warnings: tuple[str, ...]

    @property
    def comparable(self) -> bool:
        """True when nothing about how these were measured blocks pooling them."""
        return not self.warnings

    @property
    def baselines_agree(self) -> bool:
        """True when every pooled item names the SAME, KNOWN baseline configuration.

        An absent fingerprint is not agreement: there is nothing to check against.
        This is what makes the gain advice honest -- see
        :data:`GAIN_SAFE_ALTERNATIVES`.
        """
        return (len(self.baseline_groups) == 1
                and self.baseline_groups[0][0] != UNKNOWN_BASELINE)


def _group(items, key) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((key, (label, ...)), ...)`` in first-seen order (dicts preserve it)."""
    groups: dict[str, list[str]] = {}
    for it in items:
        groups.setdefault(str(key(it)), []).append(it.label)
    return tuple((k, tuple(v)) for k, v in groups.items())


def _render(groups: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    """The groups as indented ``key -- label, label`` lines, for a warning body."""
    return "\n".join(f"    {k} -- {', '.join(labels)}" for k, labels in groups)


def _domain_remedy(
    groups: tuple[tuple[str, tuple[str, ...]], ...], alternatives: str
) -> str:
    """The closing sentence of the signal-domain warning. TWO of them, on purpose.

    A domain disagreement has two causes with two different fixes, and one sentence
    covering both would be wrong for whichever one the user is actually in:

    * a NORMALIZE mismatch is fixed by re-detecting, and ``frac_depth`` is not an
      escape from it (see :data:`_SIGNAL_DOMAIN_FREE_COLUMNS` for the measurement);
    * a UNITS mismatch is a calibration label, and ``frac_depth`` *is* an escape --
      pA-to-nA is a pure scale, so a ratio of two of them is unchanged.

    Offering the normalize user ``frac_depth`` would repeat the exact defect the gain
    guard had before :data:`GAIN_ALTERNATIVES` existed: advice that walks the user off
    one cliff onto another.
    """
    if any(is_normalized_domain(k) for k, _ in groups):
        # A caller with nothing it can vouch for gets no closing offer, exactly as
        # :func:`_advice` does for the gain half -- a dangling "… survives it." with
        # no subject reads as a truncated sentence, not as silence.
        offer = f" What survives it: {alternatives}." if alternatives else ""
        return (
            "Preprocess ▸ Normalize rescales the WHOLE trace before either detector "
            "sees it, and every column downstream is still spelled \"pA\" while "
            "carrying those units. Re-detect with Normalize OFF for every recording "
            "on this axis (or ON, identically, for all of them). frac_depth is NOT a "
            "way out: the rescale subtracts a location as well as dividing, so "
            "|baseline_pA| lands at ~0 and dI/I0 blows up (measured on a 100 pA "
            "blockade in a 1000 pA pore: 0.100 raw against 105 through 'robust')."
            + offer
        )
    return (
        "Every \"_pA\" column is spelled from the channel's units rather than "
        "measured, so these differ by a factor nothing on this axis states. Give "
        "them ONE target unit (Calibration ▸ Units) and re-detect, or plot "
        "frac_depth or dwell_s, which a change of unit divides out."
    )


def pool_comparability(
    items: Iterable[PooledItem],
    *,
    quantity: str,
    depends_on_baseline: bool,
    depends_on_gain: bool = True,
    depends_on_signal_domain: bool = True,
    domain_quantity: str = "",
    gain_quantity: str = "",
    gain_alternatives: Mapping[str, bool] = GAIN_ALTERNATIVES,
    noun: str = "recordings",
    merge_remedy: str = "Merge (or un-merge) them all the same way, then Refresh.",
    reading: str = "before reading this trend",
    alternatives: str = "capture rate, median dwell",
) -> PoolVerdict:
    """Judge whether these pooled things were measured comparably. The ONE decision.

    ALL FOUR halves of the rule live here, and only here. There is deliberately no
    way to obtain three of them: a pooling surface that asks this question gets the
    whole answer, because the failure this module exists to prevent is two surfaces
    reaching different verdicts about the same two recordings, and the way that
    happened was one caller reaching for a subset of the rule.

    * **merge state**, for every quantity. Merging changes the count, the rate, the
      dwell and the depth alike, so pooling a merged set with an unmerged one is
      a comparison of two different tables;
    * **baseline fingerprint**, when ``depends_on_baseline``. Different baseline
      windows are different amplitude measurements; an ABSENT fingerprint is not a
      match either -- there is then nothing to check against, so it warns rather
      than assuming agreement;
    * **signal domain**, when ``depends_on_signal_domain``. The question UNDER the
      gain question: is this pA at all? ``Preprocess ▸ Normalize`` hands the detector
      a dimensionless trace and every column downstream keeps its "pA" spelling
      (:func:`signal_domain_of`), and ``target_units`` lets one recording be in nA
      while the next is in pA without either one's gain fingerprint moving. Arrives
      after the baseline half and before the gain one, which is where it belongs on
      the "how fundamental" ladder: the baseline decides what the number is measured
      AGAINST, the domain decides what UNIT it is in, and the gain only decides
      WHICH pA -- a question that is moot once the answer to the domain one is
      "sigmas";
    * **current calibration**, when ``depends_on_gain``. An absolute pA is a raw
      per-pore scale until somebody calibrates it (:func:`gains_comparable`), so
      pooling ``depth_pA`` -- or a median of it -- across uncalibrated recordings
      compares numbers that are not in the same units. This half arrives LAST in
      ``warnings`` because it is the least fundamental: two recordings measured
      under different baseline windows are not comparable even in the same units.

    The gain advice is reconciled with the baseline verdict here rather than at a
    call site: an alternative that a baseline disagreement also rules out is dropped
    from the sentence (see :data:`GAIN_ALTERNATIVES`), so the user is never
    redirected onto a column just as incomparable as the one they were on.

    A single item cannot disagree with anything, so a pool of one warns about
    nothing. Nothing is dropped and no fit is refused: the verdict is advisory, and
    every caller is expected to draw the pool anyway and SAY this.

    Args:
        items: The :class:`PooledItem` list (any iterable of them).
        quantity: What is being pooled, as a sentence noun -- ``"median depth"``,
            ``"depth_pA"``. Named in the baseline warning.
        depends_on_baseline: Whether ``quantity`` is measured against the baseline
            (:func:`column_depends_on_baseline` answers this for a raw column).
        depends_on_gain: Whether ``quantity`` is an ABSOLUTE current
            (:func:`column_depends_on_gain` answers this for a raw column).
            Defaults True -- fails CLOSED, like every other classification here.
            The check still needs two items that STATE a calibration, so a pool
            of items that state none (the default :data:`GAIN_UNSTATED`) is silent
            on gain: there is nothing to compare.
        depends_on_signal_domain: Whether ``quantity`` is read off the trace's own
            scale (:func:`column_depends_on_signal_domain` answers this for a raw
            column). Defaults True -- fails CLOSED, like every other classification
            here. NOT the same gate as ``depends_on_gain``: ``frac_depth`` divides
            the gain out but does not survive Normalize, which is affine rather than
            a pure scale (see :data:`_SIGNAL_DOMAIN_FREE_COLUMNS`).
        domain_quantity: What to name in the signal-domain warning, when that is not
            ``quantity``. Defaults to ``quantity``.
        gain_quantity: What to name in the gain warning, when that is not
            ``quantity`` -- a Plots overlay of ``depth_pA, snr`` is blocked on the
            baseline for both columns but on the gain for ``depth_pA`` alone.
            Defaults to ``quantity``.
        gain_alternatives: ``{name: survives_a_baseline_disagreement}``, in the
            order the advice says them. Each surface names its own quantities
            (``dwell_s`` in Plots, ``median dwell`` in a voltage series).
        noun: Plural noun for the pooled things in the warning bodies.
        merge_remedy: Closing sentence of the merge warning (each surface has its
            own re-gather control to name).
        reading: The phrase completing "…under ONE baseline window … <reading>".
        alternatives: Baseline-free quantities this surface can offer instead.

    Returns:
        A :class:`PoolVerdict` whose ``warnings`` is empty iff the pool is comparable.
    """
    pool = tuple(items)
    baseline_groups = _group(pool, lambda p: p.baseline_fingerprint)
    merge_groups = _group(pool, lambda p: p.merge_state)
    domain_groups = _group(pool, lambda p: p.signal_domain)

    warnings: list[str] = []
    domain_warns: list[str] = []
    if len(pool) >= 2:
        if len(merge_groups) > 1:
            warnings.append(
                f"NOT COMPARABLE: these {len(pool)} {noun} are not in the same "
                f"merge state, and a gap merge changes the event count, the capture "
                f"rate, the dwell AND the depth alike:\n"
                f"{_render(merge_groups)}\n"
                f"  {merge_remedy}"
            )
        if depends_on_baseline:
            known = tuple(g for g in baseline_groups if g[0] != UNKNOWN_BASELINE)
            unknown = tuple(g for g in baseline_groups if g[0] == UNKNOWN_BASELINE)
            if len(baseline_groups) > 1:
                warnings.append(
                    f"NOT COMPARABLE: {quantity} is measured against each "
                    f"recording's own estimated baseline, and these {len(pool)} "
                    f"{noun} were detected under "
                    f"{len(baseline_groups)} DIFFERENT baseline configurations:\n"
                    f"{_render(baseline_groups)}\n"
                    "  Re-detect them under ONE baseline window (Preprocess ▸ "
                    f"Baseline) {reading}, or plot a "
                    f"baseline-independent metric ({alternatives})."
                )
            elif unknown and not known:
                warnings.append(
                    f"NOT COMPARABLE: {quantity} is measured against each "
                    "recording's own estimated baseline, but no baseline_fingerprint "
                    f"is recorded for any of these {len(pool)} detections:\n"
                    f"{_render(unknown)}\n"
                    "  There is nothing to check them against — re-detect to stamp "
                    f"one, or plot a baseline-independent metric ({alternatives})."
                )
        # The SIGNAL DOMAIN half. Unlike the gain half below there is no "unstated"
        # exemption: every item states a domain, because a table with no provenance
        # at all (the manual-measurement one) really is in picoamps and belongs in
        # the pA group rather than in an abstention. Groups DISAGREEING is the whole
        # test -- a pool that is uniformly normalized says nothing here, and the
        # per-run disclosure that its columns are mislabelled "pA" is
        # ``AnalysisResult.warnings``' job (``modules._build_warnings``), which
        # every surface that renders a detection already prints.
        if depends_on_signal_domain and len(domain_groups) > 1:
            domain_warns.append(
                f"NOT COMPARABLE: {domain_quantity or quantity} is read off whatever "
                f"scale the detector measured on, and these {len(pool)} {noun} were "
                f"NOT measured on the same one:\n"
                f"{_render(domain_groups)}\n"
                f"  {_domain_remedy(domain_groups, alternatives)}"
            )
        warnings.extend(domain_warns)

    # The gain half. Only items that STATE a calibration take part: the manual
    # measurement table is not a second amplifier whose gain could disagree, and
    # saying so by omitting its fingerprint keeps that fact on the ITEM rather than
    # in a kind-check the next surface would have to remember to write.
    gain_items = [p for p in pool if p.calibration_fingerprint != GAIN_UNSTATED]
    verdict = PoolVerdict(
        depends_on_baseline=depends_on_baseline,
        baseline_groups=baseline_groups,
        merge_groups=merge_groups,
        warnings=tuple(warnings),
        gain_groups=_group(gain_items, lambda p: gain_label(p.calibration_fingerprint)),
        gain_warnings=(),
        domain_groups=domain_groups,
        domain_warnings=tuple(domain_warns),
    )
    if depends_on_gain and len(gain_items) >= 2:
        # ``baselines_agree`` is read off the verdict just built, so the advice and
        # the baseline warning cannot be answering two different questions.
        alts = [name for name, baseline_free in gain_alternatives.items()
                if verdict.baselines_agree or baseline_free]
        ok, why = gains_comparable(
            [p.calibration_fingerprint for p in gain_items], alternatives=alts)
        if not ok:
            # "across recordings" is literal, not ``noun``: this sentence is about
            # the AMPLIFIERS that produced the numbers, and a Plots "source" is one
            # channel of one recording. The wording is also what the -2026-08-22
            # battery measured on Analysis > Plots, and it must not drift.
            gain_warning = (f"CAUTION — comparing {gain_quantity or quantity} across "
                            f"recordings: {why}")
            verdict = verdict._replace(warnings=verdict.warnings + (gain_warning,),
                                       gain_warnings=(gain_warning,))
    return verdict


# --- the Plots-window adapter: ONE verdict over ONE set of columns ----------

#: What the Plots window's baseline warning offers instead. ``dwell_s`` is the only
#: event column that is BOTH gain-free and baseline-free, so it is the one answer
#: that is right whichever guard fired.
_PLOTS_ALTERNATIVES = "dwell_s, or a count"


class SourceVerdict(NamedTuple):
    """The single answer to "may I pool these sources for THESE columns?".

    One object rather than two independent banners, because the two guards are not
    independent: a gain disagreement and a baseline disagreement rule out
    overlapping sets of columns, and the ADVICE each would give alone is wrong in
    the presence of the other (the gain guard used to recommend ``frac_depth``,
    which a baseline disagreement makes just as incomparable as ``depth_pA``).

    Attributes:
        columns: The columns asked about, in the order given.
        pool: The whole :class:`PoolVerdict` over the drawn sources -- baseline,
            merge AND gain.
        warnings: Every reason these columns must not be pooled, most fundamental
            first (merge, then baseline, then signal domain, then gain). Empty when
            they may be. Identical to ``pool.warnings``: this object adds a headline,
            not a verdict.
        headline: A one-line verdict for a plot TITLE -- the surface a saved PNG
            keeps. ``""`` exactly when ``warnings`` is empty.
    """

    columns: tuple[str, ...]
    pool: PoolVerdict
    warnings: tuple[str, ...]
    headline: str

    @property
    def comparable(self) -> bool:
        return not self.warnings


def source_comparability(
    sources: Sequence[PlotSource],
    columns: Sequence[str],
) -> SourceVerdict:
    """May these Plots sources be pooled for these columns? The ONE verdict.

    Asks the SAME rule the voltage-dependence window asks -- all of it, in one call
    to :func:`pool_comparability`, which reconciles the baseline, merge, signal-domain
    and gain halves so the user gets one honest answer instead of banners that contradict
    each other. This function is a TRANSLATION (columns to quantities, sources to
    :class:`PooledItem`), not a second decision; there is no second call here for a
    future edit to lose.

    Per column, exactly as the voltage-dependence window is per metric: the
    baseline check fires only for the columns :func:`column_depends_on_baseline`
    calls True, so a ``dwell_s`` overlay across differently-probed recordings stays
    silent -- the Schmitt edges are set by the noise ladder, not by the baseline
    level. The gain check fires only for :data:`ABSOLUTE_CURRENT_COLUMNS`, and the
    signal-domain check for the columns :func:`column_depends_on_signal_domain`
    calls True -- a WIDER set than the gain's, because a normalized trace takes
    ``frac_depth`` with it and a change of gain does not.

    Args:
        sources: The sources actually being drawn (already filtered to the ticked
            ones by the caller).
        columns: The event columns on the axes.

    Returns:
        A :class:`SourceVerdict`. Never refuses and never hides a source: an
        incomparable overlay is still drawn, and said out loud.
    """
    cols = tuple(str(c) for c in columns)
    base_cols = tuple(c for c in cols if column_depends_on_baseline(c))
    gain_cols = tuple(c for c in cols if column_depends_on_gain(c))
    domain_cols = tuple(c for c in cols if column_depends_on_signal_domain(c))

    # The pool is judged over EVERY drawn source, the manual table included: a
    # hand-measured depth was measured against whatever the hand picked, which is
    # not the detector's window either. The GAIN half still skips it -- and says so
    # by stating no calibration for it, rather than by a second kind-check.
    pool = pool_comparability(
        [PooledItem(label=s.label,
                    baseline_fingerprint=baseline_fingerprint_of(s.provenance),
                    merge_state=merge_state(s.provenance),
                    calibration_fingerprint=(s.calibration_fingerprint
                                             if s.kind == "detection" else GAIN_UNSTATED),
                    # The manual table takes the DEFAULT domain rather than an
                    # abstention: a hand-measured depth really is in picoamps, so it
                    # is evidence AGAINST a normalized neighbour, not a party that
                    # has no opinion. (Contrast the gain fingerprint just above,
                    # where it genuinely has none.)
                    signal_domain=signal_domain_of(s.provenance))
         for s in sources],
        quantity=", ".join(base_cols) if base_cols else ", ".join(cols) or "this column",
        depends_on_baseline=bool(base_cols),
        depends_on_gain=bool(gain_cols),
        depends_on_signal_domain=bool(domain_cols),
        domain_quantity=", ".join(domain_cols),
        gain_quantity=", ".join(gain_cols),
        noun="sources",
        merge_remedy="Merge (or un-merge) them all the same way, then re-open Plots.",
        reading="before pooling these",
        alternatives=_PLOTS_ALTERNATIVES,
    )

    headline = ""
    if pool.warnings:
        # Which columns to NAME. When only the gain fired, the baseline-dependent
        # columns are not the ones blocked -- ``frac_depth`` is fine here. A SIGNAL
        # DOMAIN warning is not in that exemption even though it blocks the same
        # absolute-current columns the gain does: it also takes ``frac_depth`` with
        # it (Normalize is affine, so the dI/I0 denominator goes to ~0), so the
        # headline must name the wider set, which is what ``base_cols`` already is.
        gain_only = len(pool.gain_warnings) == len(pool.warnings)
        blocked = gain_cols if gain_only else (base_cols or domain_cols)
        headline = f"⚠ NOT COMPARABLE across {len(sources)} sources: {', '.join(blocked or cols)}"
    return SourceVerdict(columns=cols, pool=pool,
                         warnings=pool.warnings, headline=headline)


def disambiguate_labels(entries: list[dict]) -> None:
    """Give every entry a label no other entry shares, mutating ``entries``.

    Display names collide for real reasons -- a segment may be named after its
    parent, and two pores' files often share a basename across folders -- and a
    legend with two identical entries is worse than no legend. Channel name first
    (it is meaningful: "current" vs "current 2"), then a stable ordinal.

    ``entries`` are dicts with a ``label`` and optionally ``detector`` /
    ``channel_name`` / ``is_segment``. Shared by the Plots source list and the PSD
    recording list, so both windows name the same recording the same way.

    DETECTOR FIRST, added 2026-09-02 with one-source-per-detector discovery. A
    recording can now be offered twice -- once per detector that has run on it -- and
    a legend reading "apo.abf / apo.abf" is the same unreadable figure a segment named
    after its parent used to produce. The detector name is prepended ONLY inside a
    colliding group whose members disagree about it ("Schmitt · apo.abf",
    "CUSUM · apo.abf"); a recording that carries one detection is still named by
    itself, because there is nothing there to tell it apart from. Entries with no
    ``detector`` key -- every PSD entry, and the manual-measurement table -- are
    untouched, so the pass costs the other caller nothing.
    """
    from collections import Counter

    counts = Counter(e["label"] for e in entries)
    if all(c == 1 for c in counts.values()):
        return

    # Pass 0: name the DETECTOR where a group disagrees about it.
    by_label: dict[str, list[dict]] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    for label, group in by_label.items():
        if len(group) == 1:
            continue
        dets = [str(e.get("detector") or "") for e in group]
        if all(dets) and len(set(dets)) > 1:
            for e, det in zip(group, dets):
                e["label"] = f"{det} · {label}"

    # Pass 1: append the channel name where that alone separates the collisions.
    by_label = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    for label, group in by_label.items():
        if len(group) == 1:
            continue
        chan_names = [e.get("channel_name") or "" for e in group]
        if len(set(chan_names)) == len(group) and all(chan_names):
            for e, name in zip(group, chan_names):
                e["label"] = f"{label} — {name}"

    # Pass 2: whatever still collides gets a stable ordinal in discovery order.
    counts = Counter(e["label"] for e in entries)
    seen: Counter = Counter()
    for e in entries:
        label = e["label"]
        if counts[label] == 1:
            continue
        seen[label] += 1
        suffix = " (segment)" if e.get("is_segment") else ""
        e["label"] = f"{label}{suffix} #{seen[label]}"


def discover_plot_sources(
    documents: Iterable[Document],
    get_result: Callable[[str, int, str], Any],
    *,
    manual_df: Any = None,
    current: tuple[str, int] | tuple[str, int, str] | None = None,
    calibration_fingerprint: Callable[[str, int], tuple[float, float]] | None = None,
    module_ids: Sequence[str] = DETECTION_MODULE_IDS,
    colours: ColourRegistry | None = None,
) -> list[PlotSource]:
    """Every plottable per-event table across the OPEN documents, plus the manual table.

    One source per ``(document, channel, MODULE ID)`` that has a stored detection
    carrying an ``EventSet``-shaped payload (``.df`` + ``.provenance``) with at least
    one row -- segments included, because a segment IS an ordinary document and
    comparing two segments of one recording is half the reason segments exist.

    PER DETECTOR, not per channel, since 2026-09-02. ``ResultStore`` has always been
    keyed (doc_id, channel, module_id) and genuinely holds a channel's detections side
    by side; this function used to read :data:`DETECTION_MODULE_IDS` best-first and
    keep the FIRST hit, so every other detection on that channel was invisible to
    every window built on this list. Measured on the owner's -1000 mV recording:
    Schmitt 110 events and CUSUM 250 both stored, the Analysis panel showing CUSUM,
    and Plots offering one source -- Schmitt's -- flagged as the current one and
    labelled with the filename alone. Offering all of them is also what makes DETECTOR
    COMPARISON work: tick two and the window's existing multi-source overlay draws
    them together, each with its own colour, its own provenance and its own dwell fit.

    Args:
        documents: The open documents, in Files-list order.
        get_result: ``(doc_id, channel, module_id) -> AnalysisResult | None``; in the
            app this is ``session.results.get``.
        manual_df: The manual-measurement frame, appended as a source when non-empty.
        current: The app's ACTIVE detection, as ``(doc_id, channel, module_id)``; that
            one source is flagged :attr:`PlotSource.current` so a window opens showing
            what the user is looking at. The two-element ``(doc_id, channel)`` form is
            still accepted for a caller that does not know which detector is active,
            and then the channel's FIRST discovered detection is flagged -- the
            pre-2026-09-02 behaviour, which is a GUESS: on a channel carrying two
            detections it names Schmitt while the panel may be showing CUSUM. Pass the
            three-element form wherever the active detector is known.
        calibration_fingerprint: ``(doc_id, channel) -> (gain, offset)``; in the app
            ``session._calibration.fingerprint``-equivalent. Used only to tell the
            user whether absolute pA is comparable across the chosen sources.
        module_ids: Detector result ids to look for. Every one that has a stored,
            non-empty result becomes its own source; the order is the order the
            sources are offered in.
        colours: The project's :class:`~nanoscope.core.layer.ColourRegistry`. When
            given, each source's ink is looked up by its IDENTITY, so it survives a
            neighbour closing, a re-detection, a Files-list reorder and a project
            reopen, and matches the colour the same recording has in the trace view.
            When ``None`` the inks fall back to the old positional assignment --
            the pure-function callers (the tests, and anything that just wants the
            list) have no project to key against, and refusing them a colour would
            be worse than giving them an unstable one. MUTATES the registry: a
            source seen for the first time is assigned its slot here (see
            :meth:`~nanoscope.core.layer.ColourRegistry.slot` for why asking is
            registering).

    Returns:
        The sources in a stable order (documents in Files order, channels ascending,
        detectors in ``module_ids`` order, the manual table last), each with its own
        Okabe-Ito colour.
    """
    entries: list[dict] = []

    cur_doc, cur_ch, cur_mid = None, None, None
    if current is not None:
        cur = tuple(current)
        if len(cur) >= 3:
            cur_doc, cur_ch, cur_mid = cur[0], cur[1], str(cur[2])
        elif len(cur) == 2:
            cur_doc, cur_ch = cur
    flagged_current = False

    for doc in documents:
        meta = getattr(doc, "meta", None)
        if meta is None:
            continue
        doc_key = persistent_key(doc)
        for info in meta.channels:
            fp = None
            if calibration_fingerprint is not None:
                try:
                    fp = tuple(float(x) for x in calibration_fingerprint(doc.doc_id, info.index))
                except (KeyError, TypeError, ValueError):
                    fp = None
            for module_id in module_ids:
                res = get_result(doc.doc_id, info.index, module_id)
                if res is None:
                    continue
                df = _event_frame(res)
                if df is None or len(df) == 0:
                    continue
                on_current_channel = (cur_doc is not None
                                      and (doc.doc_id, int(info.index)) == (cur_doc, cur_ch))
                if cur_mid is not None:
                    is_current = on_current_channel and module_id == cur_mid
                else:
                    # No active detector was named: the channel's FIRST detection takes
                    # the flag, once. Flagging every detection on the channel would tell
                    # a window that two different tables are both "what is on screen".
                    is_current = on_current_channel and not flagged_current
                flagged_current = flagged_current or is_current
                entries.append({
                    # ONE spelling of the identity, shared with Layer.layer_id and with
                    # the colour registry -- see core.layer.colour_key. Per DETECTOR,
                    # because a channel can now be offered more than once; the default
                    # detector keeps the bare channel key (source_key says why).
                    "key": source_key(doc_key, info.index, module_id),
                    "doc_key": doc_key,
                    "label": doc.display_name,
                    "display_name": doc.display_name,
                    "channel_name": info.name,
                    # The detector's short name, read by disambiguate_labels when two
                    # sources of one recording would otherwise share a legend entry.
                    "detector": detector_name(module_id),
                    "df": df,
                    "provenance": dict(getattr(res.payload, "provenance", {}) or {}),
                    "doc_id": doc.doc_id,
                    "channel": int(info.index),
                    "module_id": module_id,
                    "kind": "detection",
                    "is_segment": isinstance(doc, SegmentDocument),
                    "calibration_fingerprint": fp,
                    "current": is_current,
                    # The detection's own caveats, WHOLE and unfiltered -- the same
                    # tuple the Summary panel shows. Filtering here would decide, in a
                    # discovery function, which of a detector's advisories a plot is
                    # allowed to inherit.
                    "warnings": tuple(str(w) for w in (getattr(res, "warnings", ()) or ())),
                })

    disambiguate_labels(entries)

    def ink(i: int, key: str) -> str:
        """Identity-keyed when there is a registry to key against; positional otherwise."""
        if colours is None:
            return source_colour(i)
        return colours.ink(key, SOURCE_COLOURS)

    sources = [
        PlotSource(
            key=e["key"], doc_key=e["doc_key"], label=e["label"], df=e["df"],
            provenance=e["provenance"], colour=ink(i, e["key"]), doc_id=e["doc_id"],
            channel=e["channel"], module_id=e["module_id"], kind=e["kind"],
            is_segment=e["is_segment"], display_name=e["display_name"],
            calibration_fingerprint=e["calibration_fingerprint"], current=e["current"],
            warnings=e["warnings"],
        )
        for i, e in enumerate(entries)
    ]

    if manual_df is not None and len(manual_df):
        # The manual table has no document and no channel, so its identity is the bare
        # key -- see colour_key's `channel=None` case. It is the one source whose ink
        # was ALWAYS unstable under the positional rule: its index is `len(sources)`,
        # so opening one more recording moved the hand-measured points to a new colour.
        sources.append(PlotSource(
            key=MANUAL_SOURCE_KEY, doc_key=MANUAL_SOURCE_KEY, label=MANUAL_SOURCE_LABEL,
            df=manual_df, provenance={},
            colour=ink(len(sources), colour_key(MANUAL_SOURCE_KEY)),
            kind="manual", display_name=MANUAL_SOURCE_LABEL,
        ))
    return sources


def _event_frame(result: Any) -> Any:
    """The per-event frame carried by an ``AnalysisResult``, or None.

    Duck-typed on purpose: ``analysis.events.EventSet`` lives above ``core`` in the
    import order, and a structural check (``.df`` that is frame-shaped) keeps this
    module free of that dependency without accepting arbitrary payloads.
    """
    payload = getattr(result, "payload", None)
    df = getattr(payload, "df", None)
    if df is None or not hasattr(df, "columns"):
        return None
    return df
