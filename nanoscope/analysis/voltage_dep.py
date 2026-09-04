"""Voltage-dependence analysis: how event metrics vary with applied voltage.

This is the classic nanopore experiment. The user records the same pore (or the
same analyte) at a ladder of applied voltages -- the corpus carries
-1000/-700/-600/-500/-200 mV runs -- detects translocations in each, and asks how
a per-recording summary of the events changes with the command voltage:

* the **capture rate** (events per second) rises roughly exponentially with |V|,
  ``R = A·exp(V/V0)`` -- the electrophoretic/­electro-osmotic capture barrier is
  lowered linearly by the field, so its Arrhenius-like rate is log-linear in V;
* the **dwell time** falls with |V| as the stronger field drives the molecule
  through faster;
* the **depth** / **dI/I0** shift as the blockade responds to the drive.

PURE FUNCTIONS LIVE HERE (numpy + :mod:`.populations`, no Qt):

* :func:`vdep_points` gathers one ``(voltage, metric)`` sample from every open
  document that has BOTH a known applied voltage AND a stored detection result --
  the aggregate the voltage-dependence plot needs, one point per recording; and
* :func:`vdep_fit` fits that ``(V, value)`` cloud with either a straight line
  (:func:`nanoscope.analysis.populations.linear_fit`) or an exponential
  ``value = A·exp(V/V0)`` (a line through ``(V, ln value)``, for capture-rate-like
  positive metrics), reporting the recovered parameters, R² and n.

The voltage of each recording is taken from exactly one place -- the same
:func:`nanoscope.analysis.iv.iv_point` extraction the I-V window uses (measured
VOLTAGE-channel median, else nominal) -- so a recording's applied voltage means
the same thing here as it does there. The detection metrics come from the stored
:class:`~nanoscope.analysis.events.EventSet` (``detect.schmitt`` by default),
honouring curation via ``accepted_only``.

Robustness is the contract: a document missing either half of the pair (no known
voltage, or no detection result) does not contribute a point. Nothing here
fabricates a voltage or an event -- and nothing here drops a recording in silence
either: :func:`vdep_series` returns the points TOGETHER with the
:class:`VDepExclusion` list naming every open recording that could not become one
and why, in the same words the I-V window uses for the same refusal. A pooled
series that cannot say what it lost cannot be checked.

A POOLED COMPARISON CARRIES THE PROVENANCE OF WHAT IT POOLED
------------------------------------------------------------
This is the one surface in the app built to put several recordings on the same
axis, so it is the one surface that has to answer "were these measured the same
way?". THREE properties of a stored detection decide it; two travel in the
:class:`~nanoscope.analysis.events.EventSet`'s provenance and the third is the
channel's calibration:

* the ``baseline_fingerprint``. Every amplitude the detector reports is a distance
  from an ESTIMATED baseline, so a ``depth_pA`` measured under a 0.03 s base
  window and one measured under a 0.50 s base window are different measurements of
  the same pore. Measured on the owner's apoferritin ladder: re-detecting the
  -700 mV recording under the -600 mV recording's windows moved its median
  ``frac_depth`` by 11.3 %, and reordered the three-point trend. Dwell and rate do
  NOT move with it -- the Schmitt edges are set by the noise ladder, not by the
  baseline level -- which is why :func:`depends_on_baseline` is asked PER METRIC
  instead of blanket-disabling the plot;
* the MERGE STATE. A gap merge stitches fragments back together, so it changes the
  event count, the rate, the dwell and the depth alike. On the same ladder the
  fitted capture-rate ``V0`` moved from 0.653 V to 0.441 V -- 33 %, the headline
  number of the experiment -- purely by pressing Merge, with nothing on screen
  recording which of the two numbers was on show;
* the CURRENT CALIBRATION. ``depth_pA`` is an absolute current, and until somebody
  calibrates the gain it is a raw per-pore scale, so a median depth pooled across
  three uncalibrated pores is a fit over three different units. This half was here
  in NAME only: the rule lived in ``core.sources`` but this module asked for two of
  its three answers, so on 2026-08-22, with all three apo recordings re-detected
  under ONE baseline window -- the state the app's own warning tells you to reach --
  ``Median depth`` fitted ``V0 = -0.432 V, R² = 0.9157`` and said NOTHING, while
  ``Analysis > Plots`` on ``depth_pA`` over the identical three recordings printed
  the gain CAUTION. The cleanest-looking number the battery produced, on the one
  surface that never mentioned the standing caveat that produced it.

So :class:`VDepPoint` carries all three, structurally: a caller cannot hold a pooled
point without also holding what measured it -- and, since the same reasoning applies
one level down, it carries that detection's own ``detection_warnings`` too (named
apart from the SERIES' ``warnings``, which are the pooled verdict: the two are
different claims and must not read as one), because a caveat
about ONE recording's baseline is at its most consequential where that recording
becomes one term in a fitted law. The rule that turns those properties into a
verdict is :func:`nanoscope.core.sources.pool_comparability` -- ONE call, ALL THREE
halves. A FOURTH property is judged here rather than there, because it is a property
of the metric's arithmetic and not of the detection: the ANALYSED DURATION. A raw
``n_events`` is a capture rate times a duration, so counts pooled over a 59.876 s
recording and a 250.053 s one fit a trend in file length as much as in voltage --
measured on the owner's ladder, the fitted slope REVERSES SIGN. Every point carries
its ``analyzed_duration_s`` and :func:`duration_comparability` is the per-metric
rule; it lives in ``core.sources`` because ``Analysis > Plots`` pools recordings
too and must reach the SAME verdict, and it is no longer possible to ask it for a
subset. :func:`vdep_comparability` is the thin translation from a ``VDepPoint`` and
a metric id onto it, and :func:`vdep_series` gathers and judges in ONE call, so a
plotting surface cannot obtain the points without also obtaining the verdict.
Neither hides a point nor refuses a fit -- an incomparable pool is still plotted,
and said out loud.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import numpy as np

from ..core.sources import (
    GAIN_UNSTATED,
    UNKNOWN_BASELINE,
    UNMERGED,
    PooledItem,
    baseline_fingerprint_of,
    column_depends_on_baseline,
    column_depends_on_gain,
    DEFAULT_SIGNAL_DOMAIN,
    column_depends_on_signal_domain,
    merge_state,
    pool_comparability,
    signal_domain_of,
)
from . import iv
from .events import EventSet
from .iv import iv_point, iv_point_refusal
from .populations import linear_fit

if TYPE_CHECKING:  # avoid importing Qt (Session pulls in PySide6); duck-typed at runtime.
    from ..core.analysis import AnalysisResult
    from ..core.session import Session

__all__ = [
    "METRICS",
    "GAIN_UNSTATED",
    "UNMERGED",
    "UNKNOWN_BASELINE",
    "VDepPoint",
    "VDepSeries",
    "VDepExclusion",
    "depends_on_baseline",
    "depends_on_gain",
    "depends_on_duration",
    "depends_on_signal_domain",
    "merge_state",
    "vdep_points",
    "vdep_exclusions",
    "vdep_comparability",
    "vdep_series",
    "vdep_fit",
    "vdep_predict",
]

# The Schmitt detector's module id -- the detection whose events these metrics
# summarise. Kept as the default so the voltage-dependence plot reads the same
# results the batch detector writes.
_DETECT_MODULE_ID = "detect.schmitt"

# Metric ids the plot offers. "capture_rate_hz" and "n_events" are whole-EventSet
# reductions (rate / count); the "median_*" ids are the median of one event column.
METRICS: tuple[str, ...] = (
    "capture_rate_hz",
    "median_dwell_s",
    "median_depth",
    "median_frac_depth",
    "n_events",
)

# median-metric id -> the EventSet column whose median it is.
_MEDIAN_COLUMN: dict[str, str] = {
    "median_dwell_s": "dwell_s",
    "median_depth": "depth_pA",
    "median_frac_depth": "frac_depth",
}

# Plain-text names for the warning bodies -- a sentence noun, not a UI label
# (the window keeps its own typeset labels; nothing here imports Qt).
_METRIC_NAME: dict[str, str] = {
    "capture_rate_hz": "capture rate",
    "median_dwell_s": "median dwell",
    "median_depth": "median depth",
    "median_frac_depth": "median dI/I0",
    "n_events": "event count",
}

# Whole-EventSet reductions -- a count and a rate. Neither reads an amplitude
# column, so neither is measured against the estimated baseline.
_WHOLE_SET_METRICS: frozenset[str] = frozenset({"capture_rate_hz", "n_events"})

# What the GAIN warning may offer this window's user instead, mapped to whether the
# alternative also survives a BASELINE disagreement -- the shape
# ``nanoscope.core.sources.GAIN_ALTERNATIVES`` defines. The Plots window names
# columns ("dwell_s", "frac_depth (dI/I0)"); this window names ITS metrics, because
# telling someone looking at a metric combo to plot "frac_depth" names nothing they
# can click. Both lists say the same two quantities, and each entry is asserted
# gain-free (its column is not an ABSOLUTE_CURRENT_COLUMN) by
# tests/test_fix_vdep_gain_alternatives.
_GAIN_ALTERNATIVES: Mapping[str, bool] = MappingProxyType({
    _METRIC_NAME["median_dwell_s"]: True,
    _METRIC_NAME["median_frac_depth"]: False,
})

# Which EventSet columns move with the baseline, the merge-state label, and the two
# provenance sentinels all come from :mod:`nanoscope.core.sources` -- imported above,
# NOT restated here. That module is the single home of the rule; ``UNMERGED`` and
# ``UNKNOWN_BASELINE`` are re-exported through this module's ``__all__`` because
# they are part of :class:`VDepPoint`'s published contract, but they are the same
# objects the Plots window compares against, so the two surfaces cannot drift.


def depends_on_baseline(metric: str) -> bool:
    """Does this metric's value move when the baseline windows move?

    The per-metric half of the comparability rule. ``median_depth`` and
    ``median_frac_depth`` are distances from the estimated baseline and answer
    True; ``capture_rate_hz``, ``median_dwell_s`` and ``n_events`` are set by the
    Schmitt edges and answer False, so a voltage series of those pools cleanly
    across recordings probed under their own windows (which is the owner's rule
    for probing).

    Fails CLOSED: a metric that is neither a known whole-set reduction nor a
    median of a known baseline-free column answers True.

    Args:
        metric: One of :data:`METRICS`.

    Returns:
        True when the metric is an amplitude measured against the baseline.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    col = _MEDIAN_COLUMN.get(metric)
    if col is not None:
        return column_depends_on_baseline(col)
    return metric not in _WHOLE_SET_METRICS


#: Metrics whose VALUE is a count of events rather than a rate, a time or an
#: amplitude. A count is a rate TIMES a duration, so it is the one metric here whose
#: number cannot be compared between two recordings of different length -- see
#: :func:`depends_on_duration`.
_COUNT_METRICS: frozenset[str] = frozenset({"n_events"})

#: How far two analysed durations may differ before a count pooled across them stops
#: meaning one thing. 1 % is a rounding allowance (a detector's analysed span is
#: derived from sample counts), NOT a tolerance for "close enough": anything a user
#: would call two different recording lengths is above it. On the owner's own
#: apoferritin ladder the spread is 4.18x.
_DURATION_TOLERANCE = 0.01


def depends_on_duration(metric: str) -> bool:
    """Does this metric's value scale with how long the recording was analysed?

    The FOURTH per-metric question, and the one the shared
    :func:`~nanoscope.core.sources.pool_comparability` rule cannot answer: merge
    state, baseline fingerprint and current calibration are properties of the stored
    DETECTION, and this is a property of the metric's own arithmetic. ``n_events``
    answers True -- a raw count is a capture rate times an analysed duration, so
    pooling counts from a 59.876 s recording and a 250.053 s one fits a trend in file
    length as much as in voltage (measured on the owner's ladder: the fitted slope
    reverses sign, +85.47 counts/V as shown against -2142 counts/V at one duration).
    Everything else answers False: ``capture_rate_hz`` has already divided the
    duration out -- it is the honest sibling of the count, the same detections per
    second -- and a median dwell, depth or dI/I₀ never multiplied by it.

    Args:
        metric: One of :data:`METRICS`.

    Returns:
        True when the metric's value is proportional to the analysed duration.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return metric in _COUNT_METRICS


def depends_on_gain(metric: str) -> bool:
    """Is this metric an ABSOLUTE current, so it moves with the current gain?

    The per-metric half of the GAIN rule, and the twin of
    :func:`depends_on_baseline` -- ``median_depth`` is the median of ``depth_pA``,
    so the two windows that can plot that number must agree about whether it is
    poolable, and they do because both ask
    :func:`nanoscope.core.sources.column_depends_on_gain` about the same column.

    ``median_depth`` answers True. ``median_frac_depth`` is dI/I0 and
    ``median_dwell_s`` is a time -- both divide the gain out -- and a rate and a
    count never touched an amplitude, so all four answer False. (``frac_depth``
    still moves with the BASELINE; that is the other question.)

    Args:
        metric: One of :data:`METRICS`.

    Returns:
        True when the metric is an absolute current.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return column_depends_on_gain(_MEDIAN_COLUMN.get(metric, ""))


def depends_on_signal_domain(metric: str) -> bool:
    """Does this metric move when the trace's SCALE moves -- the Normalize stage, or
    two recordings calibrated to different units?

    The per-metric half of the signal-domain rule and the twin of
    :func:`depends_on_gain`, delegated to
    :func:`nanoscope.core.sources.column_depends_on_signal_domain` for the same
    reason the gain half is: the two windows that can plot ``median_depth`` must
    agree about whether it is poolable, and they agree by asking ONE function about
    the same column.

    IT IS NOT THE SAME QUESTION AS THE GAIN, and assuming it was is how this nearly
    shipped silent. A gain is a pure SCALE, so a ratio divides it out and
    ``median_frac_depth`` answers False there. Normalize is AFFINE: it subtracts a
    location, and that location is the open-pore level -- the dI/I0 DENOMINATOR.
    Measured through the real chain (a 100 pA blockade in a 1000 pA pore, 10 pA
    noise): ``frac_depth`` 0.10012 raw becomes 87.3 under zscore and 105.0 under
    robust, a 872x and a 1048x error. So ``median_frac_depth`` answers TRUE here
    while answering False for the gain, and gating this half on the gain -- which is
    what the first draft of the rule did -- would have left the WORST case quiet.

    Args:
        metric: One of :data:`METRICS`.

    Returns:
        True when the metric's value moves with the signal domain.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    col = _MEDIAN_COLUMN.get(metric)
    if col is not None:
        return column_depends_on_signal_domain(col)
    # A rate and a count never touched an amplitude, so no rescaling of the trace
    # can move them. Asked the same way :func:`depends_on_baseline` asks it, and for
    # the same reason: ``column_depends_on_signal_domain`` fails CLOSED on a name it
    # does not know, which is right for a real column and wrong for "this metric has
    # no column at all" -- passing "" would have made every capture-rate series warn
    # about units it does not use.
    return metric not in _WHOLE_SET_METRICS


class VDepPoint(NamedTuple):
    """One ``(voltage, metric value, recording)`` sample for the plot, WITH its provenance.

    The first three fields are the plotted point. The rest are what measured it,
    carried on the point itself so a pooled comparison cannot lose them:
    :func:`vdep_comparability` reads exactly these and nothing else.

    Attributes:
        v_volts: Applied voltage in **volts** (signed), from
            :func:`nanoscope.analysis.iv.iv_point`.
        value: The chosen metric's value for this recording's detection
            (curated when ``accepted_only``).
        recording: The document's display name, for the per-recording table.
        baseline_fingerprint: The detection's ``baseline_fingerprint``
            (``sha256:...``), or :data:`UNKNOWN_BASELINE` when its provenance
            carries none. Amplitude metrics are only poolable across points that
            agree on this.
        merge_state: :func:`merge_state` of the detection's provenance --
            :data:`UNMERGED` or ``"merged: ..."``. Every metric is only poolable
            across points that agree on this.
        calibration_fingerprint: The ``(gain, offset)`` the recording's current
            channel is in, or None when it cannot be read. ABSOLUTE-current metrics
            (``median_depth``) are only poolable across points that share ONE
            NON-IDENTITY calibration -- an uncalibrated pA is a per-pore raw scale.
            :data:`GAIN_UNSTATED` (the default) means no gain was stated at all, and
            the pool is then not judged on it; :func:`vdep_points` always states one.
        detection_warnings: THIS detection's own caveats -- ``AnalysisResult.warnings``, whole
            and unfiltered, the same tuple the Summary panel renders and the same
            one ``PlotSource`` carries. A per-recording caveat that changes the
            depths ("the tracked baseline sits 4.20 pA INSIDE the states … a voltage
            series measured this way puts one recording out of line with the rest")
            is at its most load-bearing HERE, where that recording becomes one point
            in a fitted law -- so it travels with the point, like the provenance.
        analyzed_duration_s: How much signal the detection actually analysed, in
            seconds (``provenance["analyzed_duration_s"]``), or None when it recorded
            none. The fourth property of a pooled point, here for the same structural
            reason as the other three: ``n_events`` is a rate TIMES this number, so a
            count plotted without it is a voltage trend mixed with a file-length
            trend. :func:`depends_on_duration` is the per-metric half.
        voltage_source: Where this point's x-coordinate CAME FROM -- ``"measured"``
            (the command channel's median, through the calibration) or ``"nominal"``
            (``nominal_voltage_mV``, parsed from the filename or header), straight
            from :func:`nanoscope.analysis.iv.iv_point`, which the I-V window already
            prints against every point it lists. A filename-derived bias is a much
            weaker claim than a measured one, and the window that turns these
            voltages into a fitted LAW needs to say which kind each one is.
        doc_id: Stable source-document identity. Display names may collide when two
            folders contain the same basename; ``""`` is the explicit unavailable
            value for older callers and restored legacy analyses.
    """

    v_volts: float
    value: float
    recording: str
    baseline_fingerprint: str = UNKNOWN_BASELINE
    merge_state: str = UNMERGED
    calibration_fingerprint: Any = GAIN_UNSTATED
    detection_warnings: tuple[str, ...] = ()
    analyzed_duration_s: float | None = None
    voltage_source: str = ""
    #: WHAT UNIT this point's amplitude is in. ``"pA"`` (or the recording's own
    #: stated units) for an ordinary detection; a ``"normalized: ..."`` descriptor
    #: when the Preprocess chain's Normalize stage ran, because that stage leaves
    #: every amplitude dimensionless while the columns keep their picoamp names.
    #: The FOURTH poolable property -- see
    #: :func:`nanoscope.core.sources.signal_domain_of`.
    #:
    #: APPENDED after the original positional fields ON PURPOSE. A NamedTuple's
    #: field ORDER is API: this type is
    #: built POSITIONALLY in the corpus (tests/test_fix_vdep_gain_rule.py passes
    #: seven positional arguments), so inserting a field mid-list silently slides
    #: every later argument into the wrong slot -- a detection's own warnings
    #: landed in this field and vanished from ``detection_warnings``, with no error.
    #: Defaulted, so a caller that states no domain is read as picoamps rather than
    #: as a fourth "unknown" -- the same convention ``signal_domain_of`` uses.
    signal_domain: str = DEFAULT_SIGNAL_DOMAIN
    #: APPENDED LAST for stable source identity without shifting any positional
    #: caller. Empty means unavailable; it is never reconstructed from ``recording``.
    doc_id: str = ""


class VDepExclusion(NamedTuple):
    """A recording that is OPEN but is not a point in this series, and why.

    The counterpart of :class:`VDepPoint`, and the reason this module now gathers in
    one call: a pooled surface that cannot say what it lost is a pooled surface whose
    "over 3 recording(s)" is a count of something the reader cannot check. The I-V
    window has listed its own refusals since CF-5; this is the same disclosure for the
    window that fits the voltage law.

    Attributes:
        doc_id: The excluded document.
        recording: Its display name, for the window's list.
        reason: Why it is not in the series, in one sentence a reader can act on --
            :func:`nanoscope.analysis.iv.iv_point_refusal`'s words when the applied
            voltage is what blocked it (so this window and the I-V window cannot
            disagree about the same recording), or this module's own words when the
            detection or the metric is what blocked it.
        detail: The value in its honest form when there is one worth showing (the
            measured voltage with its undeclared units, the way the I-V window prints
            it), else ``""``.
    """

    doc_id: str
    recording: str
    reason: str
    detail: str = ""


# ---------------------------------------------------------------------------
# metric extraction
# ---------------------------------------------------------------------------


def _metric_value(events: EventSet, metric: str, accepted_only: bool) -> float | None:
    """The chosen metric of one detection, or ``None`` when it is undefined.

    ``None`` (never a fabricated number) is returned when the metric cannot be
    computed for this recording -- an empty event set for a median, or a missing
    ``analyzed_duration_s`` for the rate -- so the caller drops the point rather
    than plotting a NaN. :func:`_metric_refusal` is the twin that says WHY.

    Args:
        events: The recording's detected events.
        metric: One of :data:`METRICS`.
        accepted_only: Summarise only the curated-in (accepted) events.

    Returns:
        The metric value as a float, or ``None`` when undefined.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    return _metric_gate(events, metric, accepted_only)[0]


def _metric_gate(events: EventSet, metric: str,
                 accepted_only: bool) -> tuple[float | None, str]:
    """``(value, "")`` when the metric is defined here, else ``(None, reason)``.

    ONE walk for the number and for the reason there is no number, so a series that
    LISTS what it dropped cannot describe a recording differently from the gather that
    dropped it.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

    ev = events.accepted if accepted_only else events
    df = ev.df
    curated = "accepted " if accepted_only else ""
    name = _METRIC_NAME.get(metric, metric)

    if metric == "n_events":
        return float(len(df)), ""

    if metric == "capture_rate_hz":
        # rate_hz needs analyzed_duration_s in provenance; a set without it
        # (or with zero accepted events over a real duration -> 0.0) is handled here.
        try:
            return float(ev.rate_hz()), ""
        except (ValueError, ZeroDivisionError):
            return None, ("the detection did not record how much signal it analysed "
                          "(no analyzed_duration_s), so a capture rate is undefined")

    if len(df) == 0:
        return None, f"no {curated}events, so a {name} is undefined"
    col = _MEDIAN_COLUMN[metric]
    vals = np.asarray(df[col], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, f"no finite {col} values, so a {name} is undefined"
    return float(np.median(vals)), ""


def _analyzed_duration(prov: Mapping[str, Any] | None) -> float | None:
    """``provenance["analyzed_duration_s"]`` as a finite float, or ``None``.

    Never raises and never guesses: a detection that recorded no analysed span
    contributes an unknown duration, which the duration rule treats as "not known",
    which fails closed into a warning rather than into a silent assumption of
    agreement -- the same shape as an absent baseline fingerprint.
    """
    try:
        dur = float((prov or {}).get("analyzed_duration_s"))     # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return dur if np.isfinite(dur) and dur > 0.0 else None


def _calibration_of(session: "Session", res: "AnalysisResult") -> Any:
    """The ``(gain, offset)`` this detection's channel is in, or None when unreadable.

    The LIVE calibration of the channel the detection ran on -- the same question
    ``Analysis ▸ Plots`` asks (``discover_plot_sources`` is handed
    ``session.calibration(doc, ch).fingerprint()``), asked the same way, so the two
    pooling surfaces cannot answer it differently for the same recording. Never
    raises and never drops a point: an unreadable calibration reads as None, which
    the rule treats as "not known", which fails closed into a warning.

    NOT ``res.calibration_fingerprint``, which is the gain in force WHEN THE RESULT
    WAS COMPUTED. The two differ exactly when the stored result is STALE, and
    staleness is a different alarm with its own surface -- reading it here would
    have this window quietly disagree with Plots about the same two recordings,
    which is the failure this module is shaped to prevent.
    """
    try:
        return session.calibration(res.doc_id, int(res.channel)).fingerprint()
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _detection_result(
    session: "Session", doc_id: str, module_id: str
) -> "AnalysisResult | None":
    """The first stored detection (an ``EventSet`` payload) for a document.

    Searches across the document's channels so the detection is found regardless
    of which current channel it ran on; ``None`` when the document has no such
    result.
    """
    for res in session.results.for_doc(doc_id):
        if res.module_id == module_id and isinstance(res.payload, EventSet):
            return res
    return None


# ---------------------------------------------------------------------------
# gather points across the session
# ---------------------------------------------------------------------------


def vdep_points(
    session: "Session",
    *,
    metric: str,
    accepted_only: bool = True,
    module_id: str = _DETECT_MODULE_ID,
) -> list[VDepPoint]:
    """One ``(voltage, metric)`` point per open document that can contribute one.

    A document contributes a :class:`VDepPoint` only when it has BOTH a known
    applied voltage (via :func:`nanoscope.analysis.iv.iv_point`) AND a stored
    detection result whose chosen ``metric`` is defined. Everything else is
    skipped -- no voltage, no detection, or an empty event set for a median metric
    -- so the returned list is exactly the recordings that place on the plot.

    Args:
        session: The owning :class:`~nanoscope.core.session.Session` (duck-typed;
            not imported, to keep this module Qt-free).
        metric: One of :data:`METRICS`.
        accepted_only: Summarise only the curated-in (accepted) events (default).
        module_id: The detection module whose events to read (default the Schmitt
            detector, ``detect.schmitt``).

    Returns:
        A list of :class:`VDepPoint`, in :meth:`Session.documents` order. Each
        carries the detection's ``baseline_fingerprint`` and
        :func:`merge_state`; :func:`vdep_series` is the call that also judges
        them.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    return _gather(session, metric=metric, accepted_only=accepted_only,
                   module_id=module_id)[0]


def vdep_exclusions(
    session: "Session",
    *,
    metric: str,
    accepted_only: bool = True,
    module_id: str = _DETECT_MODULE_ID,
) -> list[VDepExclusion]:
    """Every OPEN recording that is not a point in this series, with the reason.

    The complement of :func:`vdep_points` over the same walk, so the two lists add up
    to the open recordings and neither can drift from the other. Same arguments; see
    :class:`VDepExclusion` for where each reason comes from.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    return _gather(session, metric=metric, accepted_only=accepted_only,
                   module_id=module_id)[1]


def _gather(
    session: "Session",
    *,
    metric: str,
    accepted_only: bool,
    module_id: str,
) -> tuple[list[VDepPoint], list[VDepExclusion]]:
    """One walk of the session: the points, AND what did not become one.

    NOTHING LEAVES A POOLED SERIES UNNAMED. Every ``continue`` below used to be
    silent, so a recording the user had opened, calibrated and detected on simply
    was not on the plot -- while the I-V window, consuming the identical refusal from
    the identical resolver (:func:`nanoscope.analysis.iv.iv_point`), listed it with
    its value, its cause and its remedy. Gathering both halves in one function is what
    makes the two lists impossible to disagree: a document leaves here as a point or
    as a named exclusion, never as neither.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

    out: list[VDepPoint] = []
    dropped: list[VDepExclusion] = []
    for doc in session.documents():
        doc_id = doc.doc_id
        pt = iv_point(session, doc_id)          # voltage (+ current channel) or None
        if pt is None:
            # WHY there is no voltage, in the I-V window's own words -- one resolver,
            # one refusal, two windows that now say the same thing about it.
            reason = iv_point_refusal(session, doc_id)
            if reason == iv.IV_NOT_A_TRACE:
                continue        # not a recording at all; there was nothing to lose
            dropped.append(VDepExclusion(
                doc_id=doc_id, recording=doc.display_name, reason=reason,
                detail=_voltage_detail(session, doc_id)))
            continue
        res = _detection_result(session, doc_id, module_id)
        if res is None:
            dropped.append(VDepExclusion(
                doc_id=doc_id, recording=doc.display_name,
                reason=f"no stored {module_id} detection — detect events on this "
                       f"recording (batch-detect the whole series), then Refresh",
                detail=f"V = {pt.v_volts * 1000.0:.4g} mV  [{pt.voltage_source}]"))
            continue
        value, why = _metric_gate(res.payload, metric, accepted_only)
        if value is None or not np.isfinite(value):
            dropped.append(VDepExclusion(
                doc_id=doc_id, recording=doc.display_name,
                reason=why or f"the {_METRIC_NAME.get(metric, metric)} is not a finite "
                              f"number for this detection",
                detail=f"V = {pt.v_volts * 1000.0:.4g} mV  [{pt.voltage_source}]"))
            continue
        # Read off the SAME EventSet the value came from, here rather than at any
        # call site: a point and the provenance of that point are built together
        # or the comparison downstream has nothing to check.
        prov = getattr(res.payload, "provenance", None) or {}
        out.append(VDepPoint(
            v_volts=pt.v_volts,
            value=float(value),
            recording=doc.display_name,
            baseline_fingerprint=baseline_fingerprint_of(prov),
            merge_state=merge_state(prov),
            calibration_fingerprint=_calibration_of(session, res),
            # WHAT UNIT this point's amplitude is in -- picoamps, the recording's own
            # stated units, or the dimensionless scale the Normalize stage leaves
            # behind. Read from the same provenance as the other three, because a
            # point whose units are not carried with it cannot be compared later.
            signal_domain=signal_domain_of(prov),
            # The detection's own caveats, WHOLE and unfiltered, exactly as
            # ``discover_plot_sources`` carries them onto a PlotSource. Deciding
            # here which of a detector's advisories a voltage series may inherit
            # would be the same silence, one level down.
            detection_warnings=tuple(
                str(w) for w in (getattr(res, "warnings", ()) or ())),
            analyzed_duration_s=_analyzed_duration(prov),
            voltage_source=pt.voltage_source,
            doc_id=doc_id,
        ))
    return out, dropped


def _voltage_detail(session: "Session", doc_id: str) -> str:
    """The refused recording's voltage in its HONEST form, or ``""``.

    Printed exactly the way the I-V window prints the same read-out for the same
    recording -- value first, and "(units unknown)" when the file declared none --
    because a refusal the user is meant to act on should show them the number they
    are about to give units to.
    """
    try:
        got = iv.applied_voltage_readout(session, doc_id)
    except Exception:       # noqa: BLE001 - a disclosure must never break the gather
        return ""
    if got is None:
        return ""
    value = f"{got.value:.4g} {got.units}" if got.units else f"{got.value:.4g} (units unknown)"
    return f"V = {value}  [{got.source}]"


# ---------------------------------------------------------------------------
# is this pool comparable? (the per-metric verdict)
# ---------------------------------------------------------------------------


class VDepSeries(NamedTuple):
    """A pooled ``(V, metric)`` series together with the verdict on pooling it.

    What :func:`vdep_series` returns and what a plotting surface should render:
    the points, the provenance they were grouped by, and the warnings that must
    not be swallowed. ``warnings`` is empty exactly when the pool is comparable.

    Attributes:
        metric: The metric id these points are of.
        points: The gathered :class:`VDepPoint` list, unfiltered and in gather
            order -- an incomparable pool is still plotted.
        depends_on_baseline: :func:`depends_on_baseline` of ``metric``; True when
            the baseline groups below are load-bearing rather than informational.
        baseline_groups: ``((fingerprint, (recording, ...)), ...)``, in first-seen
            order. :data:`UNKNOWN_BASELINE` is its own group.
        merge_groups: ``((merge_state, (recording, ...)), ...)``, in first-seen
            order.
        warnings: Human-readable reasons this pool is not comparable -- merge, then
            baseline, then gain -- each naming the offending groups AND what to do
            about it. Empty when it is comparable.
        depends_on_gain: :func:`depends_on_gain` of ``metric``; True when the
            calibration groups below are load-bearing rather than informational.
        gain_groups: ``((gain label, (recording, ...)), ...)``, in first-seen order
            -- ``"uncalibrated"`` for the identity gain every recording here carries
            until somebody calibrates it.
        excluded: The OPEN recordings that are NOT points here, each with the reason
            (:class:`VDepExclusion`). Empty when every open recording placed. A
            pooled series that cannot say what it lost cannot be checked.
        duration_warnings: The DURATION verdict, separate from ``warnings`` because it
            is a different claim about a different thing -- see
            :func:`duration_comparability`. Empty when the metric does not read the
            duration, or when the pooled recordings agree on it.
        duration_groups: ``((duration label, (recording, ...)), ...)`` in first-seen
            order -- what the "Duration" row of the pooled-provenance block renders,
            for every metric, so the reader can see the spans a trend was fitted over
            whether or not this metric multiplies by them.
        voltage_source_groups: ``((source, (recording, ...)), ...)`` in first-seen
            order -- ``"measured"`` (the command channel's median) against
            ``"nominal"`` (a bias read off the filename or header). The x-axis of a
            fitted LAW is made of these, and the two are not equally strong claims;
            the I-V window has printed the distinction against every point it lists
            since it was written, and this window is where those voltages become a
            trend.
    """

    # No defaults on the gain fields, deliberately -- see PoolVerdict. A series that
    # can be built without stating whether the metric reads the gain is a series
    # whose title can silently drop the caveat again.
    metric: str
    points: tuple[VDepPoint, ...]
    depends_on_baseline: bool
    depends_on_gain: bool
    baseline_groups: tuple[tuple[str, tuple[str, ...]], ...]
    merge_groups: tuple[tuple[str, tuple[str, ...]], ...]
    gain_groups: tuple[tuple[str, tuple[str, ...]], ...]
    warnings: tuple[str, ...]
    # Defaulted, unlike the gain fields above, and for the opposite reason: these two
    # describe what is NOT in the pool and what measured its x-axis, so a caller that
    # hand-builds a point list (a test, a notebook) has nothing to report and must not
    # be forced to invent an answer. Every SESSION-gathered series goes through
    # `vdep_series`, which always fills them.
    excluded: tuple[VDepExclusion, ...] = ()
    duration_warnings: tuple[str, ...] = ()
    duration_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    voltage_source_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def comparable(self) -> bool:
        """True when nothing about how these were measured blocks pooling them."""
        return not self.warnings

    @property
    def detection_warnings(self) -> tuple[tuple[str, str], ...]:
        """``((recording, warning), ...)`` -- every pooled detection's own caveats.

        The per-recording half of the honesty, which the pooled verdict cannot
        express: a warning about ONE recording's baseline does not make the three
        of them incomparable, it makes ONE of the three points wrong, and this is
        the surface where that point becomes a term in a fitted law. Flattened in
        point order so a window can render it without touching the points itself.
        """
        return tuple((p.recording, w)
                     for p in self.points for w in p.detection_warnings)


def _duration_label(seconds: float | None) -> str:
    """How one analysed duration is spelled, everywhere it is shown."""
    return "unknown" if seconds is None else f"{seconds:.3f} s"


def _group_by(points, key) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((label, (recording, ...)), ...)`` in first-seen order, grouped by ``key(p)``.

    The same shape ``pool_comparability`` returns for merge / baseline / gain, so the
    two properties judged HERE render through the panel's one group renderer instead
    of growing a second way of saying the same thing.
    """
    groups: dict[str, list[str]] = {}
    for p in points:
        groups.setdefault(key(p), []).append(p.recording)
    return tuple((k, tuple(v)) for k, v in groups.items())


def _duration_groups(points) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((label, (recording, ...)), ...)`` in first-seen order, one group per span."""
    return _group_by(points, lambda p: _duration_label(p.analyzed_duration_s))


def _voltage_source_groups(points) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((source, (recording, ...)), ...)`` -- measured vs read off a filename."""
    return _group_by(points, lambda p: p.voltage_source or "unstated")


def duration_comparability(points, *, metric: str) -> tuple[str, ...]:
    """Warn when a COUNT is pooled across recordings of unequal analysed duration.

    THE FOURTH HALF, AND WHY IT LIVES HERE RATHER THAN IN ``core.sources``. The shared
    :func:`~nanoscope.core.sources.pool_comparability` rule judges three properties of
    a stored DETECTION -- merge state, baseline fingerprint, current calibration --
    and answers "were these measured the same way?". The analysed duration is not one
    of them: it is a property of the METRIC's arithmetic, and it makes exactly one
    metric incomparable. ``n_events`` is a capture rate TIMES a duration, so a count
    of 1196 over 250.053 s and a count of 275 over 59.876 s are not two measurements
    of one quantity; every other metric here has already divided the duration out or
    never multiplied by it (:func:`depends_on_duration`).

    Named apart from the pooled verdict for the same reason ``detection_warnings`` is:
    they are different claims and must not read as one. This one does not say the
    recordings were measured differently -- they may be impeccable -- it says the
    NUMBER being plotted is part physics and part file length.

    Args:
        points: The :class:`VDepPoint` list to judge (any iterable of them).
        metric: One of :data:`METRICS`.

    Returns:
        One warning, or ``()`` when the metric does not read the duration, when
        fewer than two points are pooled (nothing can disagree), or when every
        pooled recording was analysed over the same span.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    if not depends_on_duration(metric):
        return ()
    pts = tuple(points)
    if len(pts) < 2:
        return ()
    known = [p.analyzed_duration_s for p in pts if p.analyzed_duration_s is not None]
    unknown = [p.recording for p in pts if p.analyzed_duration_s is None]
    lo, hi = (min(known), max(known)) if known else (None, None)
    # FAILS CLOSED, like the absent baseline fingerprint: a detection that recorded no
    # analysed span is not evidence that it agrees with the others.
    if not unknown and lo is not None and hi <= lo * (1.0 + _DURATION_TOLERANCE):
        return ()

    name = _METRIC_NAME.get(metric, metric)
    spans = ", ".join(f"{rec} {label}" for label, recs in _duration_groups(pts)
                      for rec in recs)
    head = (f"COUNTS ACROSS UNEQUAL ANALYSED DURATIONS — the {name} is a capture rate "
            f"TIMES the analysed duration, and these {len(pts)} recording(s) were not "
            f"analysed over the same span")
    if lo is not None and hi is not None and hi > lo * (1.0 + _DURATION_TOLERANCE):
        head += f" ({lo:.3f} s to {hi:.3f} s, {hi / lo:.2f}x)"
    if unknown:
        head += (f"; {len(unknown)} of them recorded no analysed duration at all "
                 f"({', '.join(unknown)})")
    return (f"{head}. A trend fitted through these values is part voltage dependence "
            f"and part file length. Plot '{_METRIC_NAME['capture_rate_hz']}' instead — "
            f"the SAME detections, each divided by its own recording's duration. "
            f"Durations: {spans}.",)


def vdep_comparability(points, *, metric: str, excluded=()) -> VDepSeries:
    """Judge whether these pooled points were measured comparably. The ONE decision.

    Both halves of the rule live here, and only here:

    * **merge state**, for every metric. Merging changes the count, the rate, the
      dwell and the depth alike, so pooling a merged set with an unmerged one is
      a comparison of two different tables;
    * **baseline fingerprint**, for the metrics :func:`depends_on_baseline` calls
      True. Different baseline windows are different amplitude measurements; an
      ABSENT fingerprint is not a match either -- there is then nothing to check
      against, so it warns rather than assuming agreement.

    A single point cannot disagree with anything, so a series of one warns about
    nothing. Nothing is dropped and no fit is refused: the pool is returned whole,
    with the reasons attached.

    Args:
        points: The :class:`VDepPoint` list to pool (any iterable of them).
        metric: One of :data:`METRICS` -- which metric these values are.
        excluded: The :class:`VDepExclusion` list to carry alongside -- what the
            gather could NOT turn into a point. Defaulted to empty because a caller
            pooling a hand-built point list has no session to have lost anything
            from; :func:`vdep_series` always passes what it dropped.

    Returns:
        A :class:`VDepSeries` whose ``warnings`` is empty iff the pool is comparable.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

    pts = tuple(points)
    reads_baseline = depends_on_baseline(metric)
    reads_gain = depends_on_gain(metric)
    reads_domain = depends_on_signal_domain(metric)
    # The decision itself is nanoscope.core.sources.pool_comparability -- the ONE
    # implementation, shared with the Plots window, and it answers ALL THREE halves
    # in this single call. This function's job is the translation: a VDepPoint
    # carries the three provenance properties already, and the metric NAME (not the
    # column) is what a voltage-series warning should say.
    verdict = pool_comparability(
        [PooledItem(label=p.recording,
                    baseline_fingerprint=p.baseline_fingerprint,
                    merge_state=p.merge_state,
                    calibration_fingerprint=p.calibration_fingerprint,
                    signal_domain=p.signal_domain)
         for p in pts],
        quantity=_METRIC_NAME.get(metric, metric),
        depends_on_baseline=reads_baseline,
        depends_on_gain=reads_gain,
        # The FOURTH half. Not implied by the gain: a gain is a pure scale that a
        # ratio divides out, while Normalize is affine and moves the dI/I0
        # denominator itself, so ``median_frac_depth`` is gain-free and
        # domain-DEPENDENT. See :func:`depends_on_signal_domain`.
        depends_on_signal_domain=reads_domain,
        gain_alternatives=_GAIN_ALTERNATIVES,
    )
    return VDepSeries(
        metric=metric,
        points=pts,
        depends_on_baseline=verdict.depends_on_baseline,
        depends_on_gain=reads_gain,
        baseline_groups=verdict.baseline_groups,
        merge_groups=verdict.merge_groups,
        gain_groups=verdict.gain_groups,
        warnings=verdict.warnings,
        excluded=tuple(excluded),
        # The duration half rides along on the SAME call, for the same structural
        # reason as the other three: a surface cannot obtain the points without also
        # obtaining what qualifies them.
        duration_warnings=duration_comparability(pts, metric=metric),
        duration_groups=_duration_groups(pts),
        voltage_source_groups=_voltage_source_groups(pts),
    )


def vdep_series(
    session: "Session",
    *,
    metric: str,
    accepted_only: bool = True,
    module_id: str = _DETECT_MODULE_ID,
) -> VDepSeries:
    """Gather the points AND judge them, in one call. The plotting entry point.

    :func:`vdep_points` followed by :func:`vdep_comparability`, joined so that a
    surface which pools recordings cannot obtain the points without also obtaining
    the verdict on pooling them -- the failure this module is shaped to prevent. The
    same joint holds what the gather could NOT place (``excluded``): a surface cannot
    obtain the points without also obtaining the list of recordings that are missing
    from them.

    Args:
        session: The owning :class:`~nanoscope.core.session.Session`.
        metric: One of :data:`METRICS`.
        accepted_only: Summarise only the curated-in (accepted) events (default).
        module_id: The detection module whose events to read.

    Returns:
        The :class:`VDepSeries`.

    Raises:
        ValueError: on an unknown ``metric`` id.
    """
    points, excluded = _gather(session, metric=metric, accepted_only=accepted_only,
                               module_id=module_id)
    return vdep_comparability(points, metric=metric, excluded=excluded)


# ---------------------------------------------------------------------------
# fit metric vs voltage
# ---------------------------------------------------------------------------


def vdep_fit(voltages, values, *, model: str = "linear") -> dict[str, Any]:
    """Fit a metric against applied voltage; report the recovered parameters.

    Two models, matching the physics of the different metrics:

    * ``"linear"`` -- ``value = slope·V + intercept``, a straight
      least-squares line via
      :func:`nanoscope.analysis.populations.linear_fit`. ``params`` carries
      ``slope`` and ``intercept``.
    * ``"exponential"`` -- ``value = A·exp(V / V0)``, fit as a straight line
      through ``(V, ln value)`` (so ``slope = 1/V0`` and ``intercept = ln A``);
      ``params`` carries ``A`` and ``V0``. This is the capture-rate model and
      needs strictly positive ``values`` -- non-positive points are dropped
      before the fit (they have no logarithm), and REPORTED in ``dropped`` /
      ``drop_reason`` so no caller can pass the drop on in silence.

    ``voltages`` is used verbatim, in whatever sign/units the caller plots on the
    x-axis (signed volts, or |V| -- the fit follows the axis so its overlay lands
    on the scatter). ``r2`` is the coefficient of determination in the fitted
    space (log space for the exponential), and ``n`` is the number of points
    actually fit.

    Args:
        voltages: Applied voltages (array-like), as plotted on the x-axis.
        values: The metric values (array-like, same length as ``voltages``).
        model: ``"linear"`` or ``"exponential"``.

    Returns:
        ``{"model", "params", "r2", "n", "n_input", "dropped", "drop_reason"}`` --
        ``params`` is ``{"slope", "intercept"}`` for the linear model and
        ``{"A", "V0"}`` for the exponential. ``n_input`` is how many ``(V, value)``
        pairs were HANDED IN, ``dropped`` the indices of the ones this fit could not
        use, and ``drop_reason`` why (``""`` when nothing was dropped).

        THE FIT SAYS WHAT IT LEFT OUT. ``n`` has always been the count AFTER the
        exponential model discards its non-positive points, and the window printed
        that ``n`` ten lines below its own "over N recording(s)" with no sentence
        connecting the two: a 0 Hz recording -- a real, load-bearing low-|V| point,
        or one whose events the user rejected -- silently left the fit while staying
        on the plot, in the table and in the count. A caller cannot report the drop
        it does not receive, so the drop now travels with the fit.

    Raises:
        ValueError: on an unknown ``model``; or, propagated from ``linear_fit``,
            fewer than 2 usable points or a degenerate voltage axis (every
            recording at the same voltage). For the exponential model "usable"
            also means strictly positive.
    """
    model = str(model).lower()
    if model not in ("linear", "exponential"):
        raise ValueError(f"unknown model {model!r}; expected 'linear' or 'exponential'")

    v = np.asarray(voltages, dtype=np.float64).ravel()
    y = np.asarray(values, dtype=np.float64).ravel()

    finite = np.isfinite(v) & np.isfinite(y)
    if model == "linear":
        lf = linear_fit(v, y, xlabel="V (V)", ylabel="metric")
        return {
            "model": "linear",
            "params": {"slope": lf.slope, "intercept": lf.intercept},
            "r2": lf.r2,
            "n": lf.n,
            "n_input": int(v.size),
            # linear_fit drops non-finite pairs before fitting; say which.
            "dropped": tuple(int(i) for i in np.flatnonzero(~finite)),
            "drop_reason": ("not a finite (V, value) pair"
                            if int(np.sum(~finite)) else ""),
        }

    # exponential: line through (V, ln value); value = A·exp(V/V0).
    good = finite & (y > 0.0)
    vg, yg = v[good], y[good]
    if vg.size < 2:
        raise ValueError(
            "exponential fit needs >= 2 strictly-positive (V, value) points; "
            f"got {vg.size}"
        )
    lf = linear_fit(vg, np.log(yg), xlabel="V (V)", ylabel="ln(metric)")
    if lf.slope == 0.0:
        raise ValueError("degenerate exponential fit: the metric does not vary with V")
    v0 = 1.0 / lf.slope
    amp = float(np.exp(lf.intercept))
    # Two causes can drop a point here, and a refusal names the one that applies:
    # a non-finite pair (the same cause the linear model has) and a non-positive
    # value (this model's own).
    causes = []
    if int(np.sum(~finite)):
        causes.append("not a finite (V, value) pair")
    if int(np.sum(finite & ~(y > 0.0))):
        causes.append("the exponential model is fitted as a line through "
                      "(V, ln value), and a value <= 0 has no logarithm")
    return {
        "model": "exponential",
        "params": {"A": amp, "V0": v0},
        "r2": lf.r2,
        "n": lf.n,
        "n_input": int(v.size),
        "dropped": tuple(int(i) for i in np.flatnonzero(~good)),
        "drop_reason": "; ".join(causes),
    }


def vdep_predict(fit: dict[str, Any], voltages) -> np.ndarray:
    """Evaluate a :func:`vdep_fit` result over ``voltages`` (in the fit's x-units).

    Used by the plot to draw the fitted curve on the same axis the points sit on:
    the straight line ``slope·V + intercept`` for the linear model, or the
    exponential ``A·exp(V/V0)`` for the exponential model.

    Args:
        fit: A dict returned by :func:`vdep_fit`.
        voltages: The voltages to evaluate at (array-like), same convention as the
            ones passed to :func:`vdep_fit`.

    Returns:
        The predicted metric values as a ``float64`` array.

    Raises:
        ValueError: on a ``fit`` whose ``model`` is neither linear nor exponential.
    """
    v = np.asarray(voltages, dtype=np.float64)
    params = fit.get("params", {})
    model = fit.get("model")
    if model == "linear":
        return params["slope"] * v + params["intercept"]
    if model == "exponential":
        return params["A"] * np.exp(v / params["V0"])
    raise ValueError(f"cannot predict for model {model!r}")
