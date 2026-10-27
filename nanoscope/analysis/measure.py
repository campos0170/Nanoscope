"""Manual, cursor-selected event measurement (pure, no Qt).

Detection + `characterize` answer "where are the events and what are they" for the
*whole* recording. This module answers a different question the user asks by hand:
"I have put two cursors around THIS excursion -- tell me everything about it, live,
and let me save it." It is the compute core behind the GUI's Measure tool.

It deliberately does NOT reuse the frozen 20-column event SCHEMA. A hand-picked
selection is a richer, looser object: the user chooses how the amplitude level is
defined, and wants the full descriptive-statistics set (including kurtosis, which
is exactly the thing you look at to decide whether a blockade is a clean square or
a noisy multi-level mess). So it carries its OWN schema, `MEASUREMENT_SCHEMA`.

SIGN CONVENTION (LOCAL TO THIS MODULE -- read this)
---------------------------------------------------
`characterize` defines ``dev = polarity*(current - baseline)`` positive-in-event.
Here the headline number is the *amplitude*, and the user's mental model is
"how far did the current move away from the open pore". So this module's polarity
is defined the other way round, off the level:

    polarity = sign(baseline - level)          # +1 when the selection sits below I0
    amplitude = polarity * (baseline - level)  # always a positive magnitude

That makes ``amplitude`` a positive blockade/enhancement depth regardless of the
sign of I0. The event-positive per-sample deviation consistent with it is
``polarity*(baseline - current)`` -- which is numerically identical to
`characterize`'s ``dev`` (because this polarity is the negative of that one), so
``area_ecd = sum(polarity*(baseline-current))/fs`` is the same positive charge
deficit `characterize` reports, just reached through the local sign convention.

All currents are in the channel's own units (pA by convention); all times seconds.

IDENTITY (why a row carries `doc_key` as well as `recording`)
-------------------------------------------------------------
A saved row must stay navigable: the GUI has to be able to put the cursors back on the
exact span it describes. `recording` is a DISPLAY NAME — two files with the same basename
in different folders share one, and a segment's name is free user text that can collide
with its parent's. So a row also carries `doc_key`, the same persistent key the project
persists results/calibrations/layers under (`core.segment.persistent_key`: the resolved
source path for a file recording, the segment uid for a segment), plus `channel_index`
(the stable channel key; `channel` is only the on-disk name). Rows written before those
columns existed load with the documented defaults and degrade to display-name matching.

BOUNDED PREVIEW (`max_samples`) — WHAT A SUBSAMPLED READING IS AND IS NOT
------------------------------------------------------------------------
`measure_selection` is O(n) over the selection, and a hand selection can be the whole
recording: measured on the dev machine it costs ~8 ms at 200 k samples, ~80 ms at 2 M and
~1.1 s at 15 M. A GUI that recomputes it once per drag frame therefore freezes on exactly
the recordings this app exists for. So the *caller* may bound the cost by passing
`max_samples`: above that the statistics are computed over a STRIDED SUBSAMPLE
(`current[start:stop:stride]`, ``stride = ceil(n_samples/max_samples)``) instead of every
sample. Default `None` = no bound = today's exact behaviour, bit for bit.

The contract of an approximate reading:

* **Geometry is never approximated.** `start_idx`, `end_idx`, `n_samples`, `t_start_s` and
  `dwell_s` always describe the TRUE span, never the subsample. `n_measured`/`stride` say
  how much of it was actually read.
* **The reference is exact for any selection with a flank, given `fs`.** `baseline`, the
  polarity it implies and `snr`'s noise denominator are read from windows OUTSIDE the
  selection that are already capped at 3 ms / 5 ms — a few hundred samples — so
  subsampling never touches them. Two cases do fall back and are then estimated: `fs=None`
  (no time base, so no cap on the flanks — the preview clamps them to `max_samples`), and
  a selection spanning the WHOLE trace, which has no flanks at all and stands in the whole
  trace as its own reference — the preview strides that too. The second is not exotic:
  "snap to view" on a fully zoomed-out recording is exactly it, and left unbounded it
  costs a ~450 ms whole-trace median no matter how small the subsample.
* **Converging statistics are estimated and flagged.** All are consistent estimators with
  error O(1/sqrt(n_measured)). Worst relative error measured over ten seeds of a 2.5 M-sample
  FLAT BLOCKADE at the 200 k bound (stride 13) — the fixture, and the bounds, that
  `tests/analysis/test_measure_preview.py` pins over the same ten seeds:

  =========================================  ==========================================
  `mean` `median` `rms` `level`                1.5e-4   (~sigma/sqrt(m))
  `amplitude` `frac_amplitude` `snr`           3.6e-4   (all affine in the level)
  `area_ecd`                                   3.6e-4   (estimated as
                                               ``pol*n_samples*(baseline-mean)/fs``, i.e.
                                               over the TRUE span, NOT as a sum over the
                                               subsample — that would be `stride` times low)
  `std`                                        4.0e-3   (~sqrt(2/m))
  `var`                                        8.0e-3   (twice std's, being its square)
  `level`/`amplitude` under 'mode'             2.0e-2 / 5.0e-2  (the histogram bin edges
                                               come from the observed min/max, which move,
                                               so the level can jump a whole bin)
  `skewness` `kurtosis`                        <1 % when the tail is bulk structure — the
                                               "clean square vs multi-level mess" verdict
                                               they exist for. Up to ~20 % when the tail
                                               comes from ONE rare burst of a few dozen
                                               samples, of which the subsample catches a
                                               handful. Noisy, but unbiased, so reported.
  =========================================  ==========================================

  Read these as bounds on the ABSOLUTE error, expressed relative to a headline amplitude
  that is really there. They are NOT a promise about the printed ratio when `amplitude` is
  ~0 — which is what a selection whose level sits ON the open pore gives you (a span of
  mostly-baseline containing scattered short events). The absolute error is the same few
  hundredths of a pA; it is the denominator that has collapsed. Judge a preview by the pA,
  not by the percentage, whenever the amplitude itself is near zero.
* **Statistics striding CANNOT estimate are blanked, not guessed.** An order statistic at
  the tail is biased inward by construction (you cannot find the deepest sample in a set
  you did not look at), so `min`, `max` and `ptp` come back NaN and are listed in
  `suppressed`; with `amplitude_method` 'min'/'max' the level IS that order statistic, so
  `level`/`amplitude`/`frac_amplitude`/`snr` are blanked too. A blank the caller can
  render as "—" is honest; a number that is quietly ~1 sigma shallow is not.
* **`approximate=True` is in the returned dict** so no caller can consume a preview
  without seeing that it is one. These diagnostics (`MEASUREMENT_DIAGNOSTIC_KEYS`) are
  deliberately NOT `MEASUREMENT_SCHEMA` columns: a SAVED row is always exact, so there is
  no column in which "this one was estimated" could ever be true. Anything that persists a
  measurement must re-measure with `max_samples=None` first.

CALIBRATION + CLOCK PROVENANCE (why a saved row carries `units`/`cal_*`/`t0_s`)
-------------------------------------------------------------------------------
A saved row is a RECORD of what was measured, and both of the things that give its
numbers meaning can change after it is saved:

* the **calibration** (gain/offset/target units) of the channel it was measured on —
  the documented workflow is to defer the gain question and revise it later, exactly
  as detections anticipate with their ``calibration_fingerprint`` stamp. A 30 pA row
  saved under gain 1 and a 0.03 nA row saved after a pA→nA change are DIFFERENT
  NUMBERS for the same physical event, and without a stamp they pool silently.
* the **clock**: ``t_start_s`` is on the RECORDING clock (0 = first loaded sample)
  while the trace x-axis is on the instrument clock (`core.clock`), and this table
  pools rows from recordings with different ``t0``.

So every saved row stamps ``units`` (the channel's units label when measured),
``cal_gain``/``cal_offset`` (the calibration in force — compare against
``ChannelCalibration.fingerprint()``) and ``t0_s`` (the recording's instrument-clock
origin — `clock.instrument_origin_s`, so ``t_start_s + t0_s`` is the row's start in the
EXPERIMENT). Two things are deliberately not in that number, because a stamp is a record
and neither of them is: a layer's on-screen alignment nudge, which the user can change
after the fact; and a segment's rebase to ``t0_s = 0``, which is a display convenience of
the cut — a segment's rows carry the parent recording's origin, so a parent row and a
row measured through a segment of it land on the same clock. The stamps make
a superseded row DETECTABLE (:func:`calibration_stamp_state`); the numbers themselves
are never rescaled — a silently "corrected" record is indistinguishable from data
corruption. The remedy for a stale row is to re-measure the span.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..core.atomic_write import staged_path
from . import filters

__all__ = [
    "MEASUREMENT_SCHEMA",
    "MEASUREMENT_META_COLUMNS",
    "MEASUREMENT_DIAGNOSTIC_KEYS",
    "AMPLITUDE_METHODS",
    "EXTREME_AMPLITUDE_METHODS",
    "LIVE_PREVIEW_MAX_SAMPLES",
    "UNKNOWN_CHANNEL_INDEX",
    "CAL_STAMP_MATCH",
    "CAL_STAMP_STALE",
    "CAL_STAMP_UNKNOWN",
    "calibration_stamp_state",
    "empty_measurements_df",
    "measure_selection",
    "NonFiniteSelection",
    "preview_stride",
    "append_measurement",
    "coerce_measurements",
    "schema_default",
    "write_measurements_csv",
]

# Amplitude level estimators the user can pick between. 'min'/'max' are the extreme
# in-selection current levels (the peak of a blockade lands on one of them);
# 'mode' is the histogram-peak level, robust to the rise/fall shoulders.
AMPLITUDE_METHODS: tuple[str, ...] = ("median", "mean", "min", "max", "mode")

# The level estimators that ARE a tail order statistic. Under a strided preview these
# cannot be estimated (see the module docstring): the deepest sample of a subsample is
# systematically shallower than the deepest sample of the whole selection, and the bias
# grows with the stride. Everything derived from the level goes with them.
EXTREME_AMPLITUDE_METHODS: frozenset[str] = frozenset({"min", "max"})


class NonFiniteSelection(ValueError):
    """The selection, or the open-pore window it is measured against, carries NaN/Inf.

    A ``ValueError`` subclass, so every caller that already writes ``except ValueError``
    around :func:`measure_selection` keeps behaving exactly as it did. It exists so a
    caller can tell this refusal apart from the ORDINARY ones (a collapsed drag, a
    1-sample selection, an unknown amplitude method), which are user gestures rather
    than a defect in the data and must not raise a banner. The GUI catches this one
    specifically and prints ``str(exc)`` in the panel — see
    ``gui/measure_panel.MeasurePanel.refusal_note``.
    """


# Below this a "selection" is a click, not a region: std/skew/kurtosis have nothing
# to describe. Hard floor, raised as ValueError.
_MIN_SELECTION_SAMPLES = 2

# Default bound for a LIVE readout, in samples. Chosen off the measured cost curve of this
# function (dev machine, float32 input): 200 k -> ~8 ms exact / ~5 ms strided, 2 M -> ~80 ms,
# 15 M -> ~1.1 s. At 200 k a preview lands inside one 60 Hz frame even with the most
# expensive level estimator ('mode', which adds a percentile + histogram, ~10 ms total),
# and 200 k is also 2 s at 100 kHz — i.e. the point where the exact computation was already
# fast enough that subsampling buys nothing. It is a LATENCY policy, not a correctness one:
# nothing in this module applies it by default, callers pass it explicitly.
LIVE_PREVIEW_MAX_SAMPLES: int = 200_000

# Statistics a strided preview blanks (returns NaN for) rather than misreport. Extended with
# the level-derived keys when amplitude_method is itself an extreme — see `_suppressed_keys`.
_SUPPRESSED_ORDER_STATS: tuple[str, ...] = ("min", "max", "ptp")
_SUPPRESSED_WITH_EXTREME_LEVEL: tuple[str, ...] = (
    "level", "amplitude", "frac_amplitude", "snr",
)

# Local open-pore baseline (and noise) are read from a symmetric window just OUTSIDE
# the selection, sized to the selection but capped so a very long selection does not
# drag the reference across real drift. A few ms is longer than any single event.
_BASELINE_CAP_S = 3e-3
_NOISE_CAP_S = 5e-3

# A `channel_index` this app could not resolve (a row written before the column existed).
# int64 cannot hold NA, so absence is a SENTINEL, not a null — see `schema_default`.
UNKNOWN_CHANNEL_INDEX: int = -1

# Column -> (dtype, unit, description). This IS the manual-measurement schema; it is
# intentionally NOT the event SCHEMA. 'signal' unit means "the channel's own units".
#
# EXTENDING IT: columns may be added ADDITIVELY (see `coerce_measurements`, which fills a
# missing column with that dtype's default so an older CSV/project still loads). Never
# rename or retype an existing column: `t_start_s` etc. are already out in exported CSVs.
MEASUREMENT_SCHEMA: dict[str, tuple[str, str, str]] = {
    "measurement_id":   ("int64",   "-",      "index within this measurement session"),
    "recording":        ("string",  "-",      "name of the source recording/document"),
    "doc_key":          ("string",  "-",      "persistent document identity (source path, or segment uid); '' when unknown"),
    "channel":          ("string",  "-",      "channel the selection was taken on"),
    "channel_index":    ("int64",   "-",      f"channel INDEX on that document ({UNKNOWN_CHANNEL_INDEX} when unknown)"),
    "start_idx":        ("int64",   "sample", "first sample of the selection"),
    "end_idx":          ("int64",   "sample", "one past the last sample of the selection"),
    "t_start_s":        ("float64", "s",      "selection start, seconds from recording start (NaN if no time base)"),
    "dwell_s":          ("float64", "s",      "duration (end_idx-start_idx)/fs; NaN when fs is undefined"),
    "n_samples":        ("int64",   "sample", "number of samples in the selection"),
    "amplitude_method": ("string",  "-",      "how the in-selection level was estimated (see AMPLITUDE_METHODS)"),
    "level":            ("float64", "signal", "the in-selection current level, by amplitude_method"),
    "baseline":         ("float64", "signal", "local open-pore reference I0, median just outside the selection"),
    "amplitude":        ("float64", "signal", "polarity*(baseline-level); a positive excursion magnitude"),
    "frac_amplitude":   ("float64", "-",      "amplitude / |baseline|, i.e. dI/I0"),
    "mean":             ("float64", "signal", "mean of the in-selection current"),
    "median":           ("float64", "signal", "median of the in-selection current"),
    "std":              ("float64", "signal", "population standard deviation (ddof=0)"),
    "var":              ("float64", "signal", "population variance (ddof=0)"),
    "min":              ("float64", "signal", "minimum in-selection current"),
    "max":              ("float64", "signal", "maximum in-selection current"),
    "ptp":              ("float64", "signal", "peak-to-peak (max-min)"),
    "rms":              ("float64", "signal", "root-mean-square of the in-selection current"),
    "skewness":         ("float64", "-",      "Fisher-Pearson skewness of the in-selection current"),
    "kurtosis":         ("float64", "-",      "excess (Fisher) kurtosis; 0 for a Gaussian"),
    "area_ecd":         ("float64", "signal*s", "integrated charge deficit sum(polarity*(baseline-current))/fs"),
    "snr":              ("float64", "-",      "amplitude / local baseline noise RMS"),
    "label":            ("string",  "-",      "free-text annotation the user typed on the row"),
    # -- provenance stamps (additive; see the module docstring). A row saved before these
    #    columns existed loads with '' / NaN, which reads as "unknown", never as "matches".
    "units":            ("string",  "-",      "the channel's units label when this row was measured ('' when the channel declared none)"),
    "cal_gain":         ("float64", "-",      "calibration gain in force when this row was measured (NaN when unknown — a row saved before this column existed)"),
    "cal_offset":       ("float64", "signal", "calibration offset in force when this row was measured (NaN when unknown)"),
    "t0_s":             ("float64", "s",      "instrument-clock origin of the row's recording — the experimental time of its first sample: t_start_s + t0_s is this instant on the INSTRUMENT clock (for a segment, its parent recording's clock, not the segment's own rebased axis; a layer alignment nudge is a display shift and is never included). NaN when unknown, or when the recording has no time base"),
}

# Columns the CALLER stamps — the row's identity (which recording/channel it came from),
# the user's annotation, and the provenance stamps. `measure_selection` computes none of
# them: it is handed a bare array and returns pure geometry/statistics (the one nuance is
# `units`, which the caller passes IN and the function echoes back unchanged). Public so a
# caller (or a test) can ask which columns are metadata instead of hard-coding a list that
# goes stale on every additive schema change.
MEASUREMENT_META_COLUMNS: tuple[str, ...] = (
    "measurement_id", "recording", "doc_key", "channel", "channel_index", "label",
    # The provenance stamps are the caller's too: `measure_selection` is handed a bare
    # array and cannot know which gain/offset produced it or what the recording's t0
    # is. `units` is caller-SUPPLIED as well (the function merely echoes it back in the
    # returned dict, as it always has, so a caller that stamps nothing still saves the
    # label it measured under).
    "units", "cal_gain", "cal_offset", "t0_s",
)

# Keys `measure_selection` returns that describe the MEASUREMENT rather than the signal:
# the preview diagnostics (`approximate`/`stride`/`n_measured`/`suppressed`). They are
# deliberately NOT schema columns — a saved row is always exact, so there is no row on
# which `approximate` could be True — and `append_measurement` drops them because it
# builds its row from MEASUREMENT_SCHEMA. Public so a caller that copies the dict before
# stamping it can strip them by name instead of guessing. (`units` used to be dropped
# here too; it is now a schema column, because a pooled table mixing pA and nA rows with
# no units column was permanently ambiguous — see the module docstring.)
MEASUREMENT_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "approximate", "stride", "n_measured", "suppressed",
)

# Keys that come purely from the geometry/statistics of a selection (everything
# except the caller-supplied metadata). measure_selection returns exactly these.
_STAT_KEYS = tuple(c for c in MEASUREMENT_SCHEMA if c not in MEASUREMENT_META_COLUMNS)


def schema_default(column: str):
    """The value that stands for "not recorded" in `column`, per its declared dtype.

    ``pd.NA`` cannot be cast into a numpy ``int64``/``float64`` column — ``astype`` raises
    ``TypeError``. So a missing column is filled with a dtype-appropriate default instead:
    ``""`` for ``string``, ``NaN`` for floats, and a documented sentinel for ints
    (``channel_index`` -> :data:`UNKNOWN_CHANNEL_INDEX`, other ints -> -1). THIS is what
    makes the schema genuinely additive: adding a numeric column must not make every older
    project raise mid-restore.
    """
    dtype = MEASUREMENT_SCHEMA[column][0]
    if dtype == "string":
        return ""
    if dtype.startswith("float"):
        return float("nan")
    return UNKNOWN_CHANNEL_INDEX if column == "channel_index" else -1


# ---------------------------------------------------------------------------
# the calibration stamp
# ---------------------------------------------------------------------------

#: The row's stamp equals the live calibration: its numbers mean today what they meant
#: when they were saved.
CAL_STAMP_MATCH = "match"
#: The stamp differs: the row was measured under a calibration that has since been
#: superseded. The row is a record and is never rescaled — re-measure to update it.
CAL_STAMP_STALE = "stale"
#: The stamp cannot be compared (a row saved before the stamp columns existed). Per the
#: project-IO rule, "a fingerprint that cannot be compared must read as 'unknown', never
#: as 'matches'".
CAL_STAMP_UNKNOWN = "unknown"


def calibration_stamp_state(
    row_gain, row_offset, row_units,
    live_gain: float, live_offset: float, live_units: str,
) -> str:
    """Is a saved row's calibration stamp current? One of the ``CAL_STAMP_*`` strings.

    The comparison is EXACT, like every other fingerprint comparison in this app
    (``AnalysisResult.calibration_fingerprint != live.fingerprint()``): the stamp is a
    content key copied from the calibration in force, not a measurement with error bars,
    so any difference means a different calibration was applied. ``units`` participates
    because a pA→nA change with a matching gain still changes what every signal-unit
    number in the row MEANS.

    A ``NaN`` gain/offset is the documented "not recorded" default of a row saved before
    the stamp columns existed (see :func:`schema_default`): such a row is
    :data:`CAL_STAMP_UNKNOWN` — it must never be presented as matching, and marking it
    stale would claim knowledge nobody has.
    """
    try:
        rg = float(row_gain)
        ro = float(row_offset)
    except (TypeError, ValueError):
        return CAL_STAMP_UNKNOWN
    if not (np.isfinite(rg) and np.isfinite(ro)):
        return CAL_STAMP_UNKNOWN
    ru = "" if row_units is None or pd.isna(row_units) else str(row_units)
    same = (rg == float(live_gain) and ro == float(live_offset)
            and ru == str(live_units or ""))
    return CAL_STAMP_MATCH if same else CAL_STAMP_STALE


# ---------------------------------------------------------------------------
# the time-base check
# ---------------------------------------------------------------------------

#: The row's seconds were computed at the recording's LIVE rate.
RATE_STAMP_MATCH = "match"
#: They were not: the recording's time base has been re-declared since. The row is a record
#: and is never rescaled — re-measure the span to update it.
RATE_STAMP_STALE = "stale"
#: Nothing to compare: the row carries no seconds (measured on a rate-less recording), or
#: no span.
RATE_STAMP_UNKNOWN = "unknown"


def rate_stamp_state(row_start_idx, row_end_idx, row_dwell_s, live_fs: float | None) -> str:
    """Was this row's ``dwell_s`` computed at the recording's live sample rate?

    A manual row carries no rate stamp and needs none: it stores its span in SAMPLES
    (``start_idx``/``end_idx``) beside its duration in SECONDS (``dwell_s``), and the two
    agree only under the rate the row was measured at. After the recording's time base is
    re-declared (``Session.set_sample_rate_declaration``) every seconds column of such a row
    is a record of the old clock, so it is marked — never rescaled — exactly as the event
    table's time columns are. The tolerance is round-off only: the stamp is a content key,
    not a measurement.
    """
    try:
        start, end = int(row_start_idx), int(row_end_idx)
        dwell = float(row_dwell_s)
    except (TypeError, ValueError):
        return RATE_STAMP_UNKNOWN
    if not np.isfinite(dwell) or end <= start:
        return RATE_STAMP_UNKNOWN
    if live_fs is None or not float(live_fs) > 0:
        return RATE_STAMP_STALE            # seconds on a recording that now has no time base
    expected = (end - start) / float(live_fs)
    return RATE_STAMP_MATCH if abs(dwell - expected) <= 1e-6 * expected else RATE_STAMP_STALE


# ---------------------------------------------------------------------------
# level estimators
# ---------------------------------------------------------------------------


def _mode_level(x: np.ndarray) -> float:
    """Histogram-peak current level, with Freedman-Diaconis binning.

    The mode is the most-occupied level, so on a selection whose interior is one
    flat blockade it lands on that blockade even when the rise/fall shoulders drag
    the mean. FD bin width scales with the noise (IQR) and the sample count, so it
    does not need a magic bin count.
    """
    n = x.size
    rng = float(x.max() - x.min())
    if n < 2 or rng <= 0.0:
        return float(x[0])
    q75, q25 = np.percentile(x, [75.0, 25.0])
    iqr = float(q75 - q25)
    if iqr > 0.0:
        bw = 2.0 * iqr / np.cbrt(n)
        nbins = int(np.clip(np.ceil(rng / bw), 16, 512)) if bw > 0.0 else 64
    else:
        nbins = 64
    counts, edges = np.histogram(x, bins=nbins)
    k = int(np.argmax(counts))
    return float(0.5 * (edges[k] + edges[k + 1]))


def _level(samples: np.ndarray, method: str) -> float:
    """The in-selection current level for `method`. See AMPLITUDE_METHODS."""
    if method == "median":
        return float(np.median(samples))
    if method == "mean":
        return float(np.mean(samples))
    if method == "min":
        return float(np.min(samples))
    if method == "max":
        return float(np.max(samples))
    if method == "mode":
        return _mode_level(samples)
    raise ValueError(
        f"amplitude_method must be one of {AMPLITUDE_METHODS}, got {method!r}"
    )


# ---------------------------------------------------------------------------
# local reference (baseline + noise) from just outside the selection
# ---------------------------------------------------------------------------


def _flank_windows(
    current: np.ndarray, start: int, stop: int, fs: float | None, cap_s: float,
    max_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric left/right windows just outside [start, stop).

    Width tracks the selection but is capped to `cap_s` seconds when a time base
    exists (a long selection must not pull the open-pore reference across drift).
    Edge-safe: either side may come back empty near the trace boundaries.

    `max_width` is the PREVIEW cost bound and normally does nothing: with a time base the
    `cap_s` cap is a few hundred samples, far below it, which is why a bounded preview's
    baseline/noise stay bit-identical to the exact ones. It bites only when `fs is None` —
    there is no cap then, so a 15 M-sample selection would take the median of 15 M flank
    samples on every drag frame, which is precisely the freeze the bound exists to stop.
    """
    n = current.size
    w = stop - start
    if fs is not None and np.isfinite(fs) and fs > 0:
        cap = max(1, int(round(cap_s * fs)))
        w = min(w, cap)
    if max_width is not None:
        w = min(w, max(1, int(max_width)))
    w = max(1, int(w))
    left = current[max(0, start - w) : start]
    right = current[stop : min(n, stop + w)]
    return left, right


def _refuse_non_finite(x: np.ndarray, what: str, tail: str, *,
                       offset: int = 0, stride: int = 1) -> None:
    """Raise :class:`NonFiniteSelection` if ``x`` carries NaN/Inf; else return.

    The ONE place this module words a non-finite refusal, so the selection and the
    open-pore window it is measured against cannot end up described differently.
    ``offset``/``stride`` map an index in ``x`` back to a sample number in the
    recording, which is the only part of the message the user can act on.
    """
    finite = np.isfinite(x)
    if finite.all():
        return
    from ..core.types import nonfinite_reason

    bad = np.flatnonzero(~finite)
    first = int(bad[0]) * int(stride) + int(offset) if bad.size else -1
    raise NonFiniteSelection(
        f"{what}: {nonfinite_reason(int(bad.size), int(finite.size), first)} {tail}"
    )


def _check_reference_finite(
    current: np.ndarray, start: int, stop: int, fs: float | None,
    max_width: int | None, *, need_baseline: bool, need_noise: bool,
) -> None:
    """Refuse a selection whose OPEN-PORE WINDOW is poisoned, not just its interior.

    The interior check upstream is not enough, and the reason is the same asymmetry
    that :func:`~nanoscope.analysis.characterize.recharacterize_span` has to handle: a
    flank ``NaN`` propagates into the median and the readout goes blank, but a flank
    ``+Inf`` does NOT -- the flank median is an order statistic and reads straight
    through it, so ``baseline`` came back correct-looking while ``filters.noise_rms``
    refused the same window and the panel showed nothing, with the raised text naming
    an internal variable (``"x contains non-finite samples"``) rather than the flank.
    ``NaN`` in a flank was worse still: it reached the level estimator and surfaced as
    ``"cannot convert float NaN to integer"``.

    Both are now one refusal that names the window, because the baseline, the polarity
    AND the SNR denominator are all read from it -- an amplitude measured against a
    reference that is not a number is not a measurement.

    Only the windows actually READ are checked (``baseline``/``noise_rms`` supplied by
    the caller are not re-derived), and they are the same few hundred capped samples
    the reference itself reads, so this costs nothing measurable. A selection with no
    flanks at all stands in the whole trace as its own reference -- those samples ARE
    the selection, already checked upstream.
    """
    caps = ([_BASELINE_CAP_S] if need_baseline else []) + \
           ([_NOISE_CAP_S] if need_noise else [])
    tail = ("The baseline, the polarity and the SNR denominator are all read from that "
            "window, so every amplitude here would be measured against a reference that "
            "is not a number. Move the selection away from it, or clean the trace.")
    for cap in caps:
        left, right = _flank_windows(current, start, stop, fs, cap, max_width)
        for part, base in ((left, start - left.size), (right, stop)):
            if part.size:
                _refuse_non_finite(
                    part, f"the open-pore window beside selection [{start}, {stop})",
                    tail, offset=base)


def _local_baseline(
    current: np.ndarray, start: int, stop: int, fs: float | None,
    max_width: int | None = None, fallback: np.ndarray | None = None,
) -> float:
    """Local open-pore current: median of a symmetric window on each side.

    Averaging the two flank medians makes it straddle the selection, so slow drift
    cancels to first order. Falls back to the global median only when the selection
    is the entire trace (no open pore to reference) — `fallback` overrides what stands in
    for "the whole trace" there; see `_whole_trace_fallback`.
    """
    left, right = _flank_windows(current, start, stop, fs, _BASELINE_CAP_S, max_width)
    parts = [p for p in (left, right) if p.size]
    if not parts:
        return float(np.median(current if fallback is None else fallback))
    return float(np.mean([np.median(p) for p in parts]))


def _local_noise(
    current: np.ndarray, start: int, stop: int, fs: float | None,
    max_width: int | None = None, fallback: np.ndarray | None = None,
) -> float:
    """Baseline noise RMS from the out-of-selection neighbourhood (1.4826*MAD).

    Measured off the open pore, not the selection, so the excursion itself never
    inflates the SNR denominator. Falls back to the whole trace when the selection has no
    neighbours, with the same `fallback` override as `_local_baseline`.
    """
    left, right = _flank_windows(current, start, stop, fs, _NOISE_CAP_S, max_width)
    if left.size or right.size:
        neigh = np.concatenate([p for p in (left, right) if p.size])
    else:
        neigh = current if fallback is None else fallback
    if neigh.size == 0:
        return float("nan")
    return filters.noise_rms(neigh)


def _whole_trace_fallback(x: np.ndarray, approximate: bool) -> np.ndarray | None:
    """The stand-in reference a FLANK-LESS selection should use, or None for "the trace".

    Both references fall back to the whole trace exactly when the selection has no
    neighbours on either side — i.e. when it IS the whole trace (`start == 0` and
    `stop == len(current)`; the flank width is at least 1 sample, so nothing else empties
    both). And in that case the already-materialised in-selection array `x` holds
    precisely those samples, so on the preview path handing it back costs nothing and
    saves two more strided passes over the entire recording — which is the difference
    between ~15 ms and ~30 ms on a 15 M-sample "snap to view", the exact gesture the bound
    exists to make survivable. Returns None on the exact path, where `x` is a float32
    view's float64 copy rather than the float32 array the reference has always been read
    from, and changing that would move the exact numbers.
    """
    return x if approximate else None


# ---------------------------------------------------------------------------
# the live-preview bound
# ---------------------------------------------------------------------------


def preview_stride(n_samples: int, max_samples: int | None) -> int:
    """The sample stride :func:`measure_selection` would use for this bound.

    ``1`` means "no subsampling — the result is EXACT", and is what a caller gets for
    ``max_samples=None`` or any selection that already fits inside the bound. Otherwise
    ``ceil(n_samples/max_samples)``, so the subsample is never larger than the bound.

    Exposed so a GUI can decide whether a click is about to be cheap (and skip a busy
    indicator) without measuring anything, and so a test can predict the stride instead of
    re-deriving the arithmetic.

    Raises:
        ValueError: if `max_samples` is below `_MIN_SELECTION_SAMPLES` — a bound that
            cannot hold two samples cannot describe a selection.
    """
    n = int(n_samples)
    if max_samples is None:
        return 1
    m = int(max_samples)
    if m < _MIN_SELECTION_SAMPLES:
        raise ValueError(
            f"max_samples must be at least {_MIN_SELECTION_SAMPLES} when given, got {m!r}"
        )
    return 1 if n <= m else -(-n // m)


def _suppressed_keys(stride: int, amplitude_method: str) -> tuple[str, ...]:
    """Which statistics this preview must blank instead of estimating.

    Empty when `stride == 1` (nothing was approximated). Otherwise the tail order
    statistics, plus everything derived from the level when the level estimator IS one.
    """
    if stride <= 1:
        return ()
    keys = _SUPPRESSED_ORDER_STATS
    if amplitude_method in EXTREME_AMPLITUDE_METHODS:
        keys = keys + _SUPPRESSED_WITH_EXTREME_LEVEL
    return keys


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def measure_selection(
    current: np.ndarray,
    fs: float | None,
    start: int,
    stop: int,
    *,
    amplitude_method: str = "median",
    polarity: int | None = None,
    baseline: float | None = None,
    noise_rms: float | None = None,
    units: str = "",
    max_samples: int | None = None,
) -> dict:
    """Measure one hand-selected span [start, stop) and return the full stat set.

    Args:
        current: The current trace (1-D array, channel units). float32 is fine.
        fs: Sample rate in Hz, or None. When None, times (dwell, t_start, area_ecd)
            are undefined and come back NaN; every amplitude/shape statistic is
            still valid and n_samples is reported.
        start: First sample of the selection (inclusive).
        stop: One past the last sample of the selection (exclusive). Must be > start
            and within [0, len(current)].
        amplitude_method: Which level estimator to use, one of AMPLITUDE_METHODS.
        polarity: The local sign (+1/-1). None derives it from
            ``sign(baseline - median(selection))`` -- +1 when the selection sits
            below the open pore.
        baseline: The open-pore reference I0. None computes it locally from a
            symmetric window just outside the selection.
        noise_rms: Baseline noise RMS for the SNR denominator. None estimates it
            from the out-of-selection neighbourhood via filters.noise_rms.
        units: The channel's units label, echoed back in the result. It IS a schema
            column (``units``): a saved row must say what its signal numbers are in,
            or a pooled pA/nA table is permanently ambiguous.
        max_samples: Cost bound for a LIVE readout. ``None`` (the default) reads EVERY
            sample of the selection and is what anything that persists a value must use.
            A number caps the work: a longer selection is measured over
            ``current[start:stop:stride]`` with ``stride = ceil(n_samples/max_samples)``,
            and the result comes back with ``approximate=True``. See the module docstring
            for exactly which statistics that estimates, which it blanks, and why the
            geometry and (with a time base) the baseline stay exact either way.
            :data:`LIVE_PREVIEW_MAX_SAMPLES` is the measured default for a GUI.

    Returns:
        A dict keyed by the per-selection MEASUREMENT_SCHEMA columns (everything
        except the caller-stamped MEASUREMENT_META_COLUMNS — identity, label, and
        the calibration/clock stamps; the ``units`` echo is included), plus the
        MEASUREMENT_DIAGNOSTIC_KEYS: the preview state —
        ``approximate`` (bool), ``stride`` (1 when exact), ``n_measured`` (samples
        actually read) and ``suppressed`` (the keys returned as NaN because striding
        cannot estimate them). On the exact path ``approximate`` is False, ``stride``
        is 1, ``n_measured == n_samples`` and ``suppressed`` is empty.

    Raises:
        ValueError: on stop <= start, an out-of-range span, a selection below
            `_MIN_SELECTION_SAMPLES`, an unknown amplitude_method, or a `max_samples`
            below `_MIN_SELECTION_SAMPLES`. These are user GESTURES, not defects in
            the data.
        NonFiniteSelection: (a ValueError) when the samples read carry NaN/Inf --
            either inside the selection (an order statistic reads straight through an
            Inf and would print a confident amplitude) or in the open-pore window just
            outside it, from which the baseline, the polarity and the SNR denominator
            all come. Raised as its own type so a GUI can tell "your data is poisoned,
            here is where" apart from "you have not selected anything yet".
    """
    x_full = np.asarray(current)
    if x_full.ndim != 1:
        raise ValueError(f"current must be 1-D, got shape {x_full.shape}")
    n_total = x_full.size
    start = int(start)
    stop = int(stop)

    if stop <= start:
        raise ValueError(f"stop ({stop}) must be greater than start ({start})")
    if start < 0 or stop > n_total:
        raise ValueError(
            f"selection [{start}, {stop}) falls outside the trace [0, {n_total})"
        )
    n_samples = stop - start
    if n_samples < _MIN_SELECTION_SAMPLES:
        raise ValueError(
            f"selection has {n_samples} sample(s); need at least "
            f"{_MIN_SELECTION_SAMPLES} to compute statistics"
        )
    if amplitude_method not in AMPLITUDE_METHODS:
        raise ValueError(
            f"amplitude_method must be one of {AMPLITUDE_METHODS}, got {amplitude_method!r}"
        )

    # `stride == 1` is the exact path and MUST stay byte-for-byte what it always was:
    # the plain contiguous slice, no reformulated arithmetic anywhere below.
    stride = preview_stride(n_samples, max_samples)
    x = (x_full[start:stop] if stride == 1 else x_full[start:stop:stride]) \
        .astype(np.float64, copy=False)
    n_measured = int(x.size)
    approximate = stride > 1
    suppressed = _suppressed_keys(stride, amplitude_method)

    # A selection carrying NaN/Inf is refused rather than measured. It is not the
    # empty case: the ROBUST statistics survive an Inf (the median, and therefore
    # `level`, `amplitude`, `frac_amplitude` and `snr`, are order statistics), so the
    # readout prints a confident amplitude beside a `mean` of `inf` — the same shape
    # of failure as `characterize`'s ``flag='non_finite'`` row, which is where a
    # selection saved as an event ends up. Checked over the samples actually READ,
    # which is the whole selection on the exact path (`max_samples=None` — what
    # anything that persists a value must use) and the subsample on a live preview,
    # in keeping with this module's rule that a preview reports what it read.
    _refuse_non_finite(
        x, f"selection [{start}, {stop})",
        "Every statistic over them is either NaN or a robust number quietly measured on "
        "a poisoned window. Crop the selection or clean the trace.",
        offset=start, stride=stride)

    has_fs = fs is not None and np.isfinite(fs) and fs > 0
    if fs is not None and not has_fs:
        raise ValueError(f"fs must be positive and finite when given, got {fs!r}")

    # --- reference: baseline + polarity + noise --------------------------------
    # The flanks are already capped to a few ms when fs is known, so a preview reads the
    # SAME few hundred samples the exact path does and the reference is identical. Only the
    # fs=None case (no cap) needs the preview bound applied — see _flank_windows — and the
    # flank-LESS whole-trace case, which reuses the subsample (_whole_trace_fallback).
    flank_cap = max_samples if (approximate and not has_fs) else None
    fallback = _whole_trace_fallback(x, approximate)
    if baseline is None or noise_rms is None:
        _check_reference_finite(
            x_full, start, stop, fs if has_fs else None, flank_cap,
            need_baseline=baseline is None, need_noise=noise_rms is None)
    if baseline is None:
        baseline_v = _local_baseline(
            x_full, start, stop, fs if has_fs else None, flank_cap, fallback)
    else:
        baseline_v = float(baseline)

    median_v = float(np.median(x))
    if polarity is None:
        s = np.sign(baseline_v - median_v)
        pol = int(s) if s != 0 else 1
    else:
        pol = int(polarity)
        if pol not in (-1, 1):
            raise ValueError(f"polarity must be +1 or -1, got {polarity!r}")

    if noise_rms is None:
        noise_v = _local_noise(
            x_full, start, stop, fs if has_fs else None, flank_cap, fallback)
    else:
        noise_v = float(noise_rms)

    # --- amplitude -------------------------------------------------------------
    level_v = _level(x, amplitude_method)
    amplitude_v = pol * (baseline_v - level_v)
    abs_base = abs(baseline_v)
    frac_amp = amplitude_v / abs_base if abs_base > 0 else float("nan")

    # --- descriptive statistics ------------------------------------------------
    mean_v = float(np.mean(x))
    var_v = float(np.var(x))  # population variance (ddof=0)
    std_v = float(np.sqrt(var_v))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    ptp_v = max_v - min_v
    rms_v = float(np.sqrt(np.mean(x * x)))

    # skew/kurtosis are undefined on a constant slice (0/0); report NaN rather than
    # let scipy divide by zero and warn. Gated on what was actually READ (identical to
    # n_samples on the exact path) — a 2-element subsample has no third moment either.
    if var_v > 0.0 and n_measured >= 3:
        skew_v = float(stats.skew(x))
        kurt_v = float(stats.kurtosis(x, fisher=True))  # excess kurtosis; Gaussian -> 0
    else:
        skew_v = float("nan")
        kurt_v = float("nan")

    # --- time-dependent quantities ---------------------------------------------
    # dwell/t_start are pure geometry: the TRUE span, never the subsample (invariant).
    if has_fs:
        dwell_s = n_samples / float(fs)
        t_start_s = start / float(fs)
        # Event-positive deviation (identical to characterize's dev under this
        # module's polarity), integrated over the selection.
        if stride == 1:
            area_ecd = float(np.sum(pol * (baseline_v - x)) / float(fs))
        else:
            # The exact sum is algebraically pol*n*(baseline-mean), so scaling the
            # SUBSAMPLE's mean by the TRUE n_samples is the same estimator the mean is,
            # over the true span — not a sum over however many samples happened to be read
            # (which would under-report the integral by a factor of `stride`).
            area_ecd = float(pol * n_samples * (baseline_v - mean_v) / float(fs))
    else:
        dwell_s = float("nan")
        t_start_s = float("nan")
        area_ecd = float("nan")

    snr = amplitude_v / noise_v if (np.isfinite(noise_v) and noise_v > 0) else float("nan")

    out = {
        "start_idx": start,
        "end_idx": stop,
        "t_start_s": t_start_s,
        "dwell_s": dwell_s,
        "n_samples": n_samples,
        "amplitude_method": amplitude_method,
        "level": level_v,
        "baseline": baseline_v,
        "amplitude": float(amplitude_v),
        "frac_amplitude": float(frac_amp),
        "mean": mean_v,
        "median": median_v,
        "std": std_v,
        "var": var_v,
        "min": min_v,
        "max": max_v,
        "ptp": ptp_v,
        "rms": rms_v,
        "skewness": skew_v,
        "kurtosis": kurt_v,
        "area_ecd": area_ecd,
        "snr": float(snr),
        "units": units,
        "approximate": approximate,
        "stride": stride,
        "n_measured": n_measured,
        "suppressed": suppressed,
    }
    # Blank (never guess) the statistics striding is structurally unable to estimate.
    for key in suppressed:
        out[key] = float("nan")
    return out


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


def empty_measurements_df() -> pd.DataFrame:
    """An empty measurement table with the schema columns and dtypes."""
    return pd.DataFrame(
        {c: pd.Series(dtype=d) for c, (d, _, _) in MEASUREMENT_SCHEMA.items()}
    )


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the schema dtypes and column order (columns must all be present)."""
    return df.astype({c: d for c, (d, _, _) in MEASUREMENT_SCHEMA.items()})[
        list(MEASUREMENT_SCHEMA)
    ]


def coerce_measurements(df: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce any measurement frame onto the CURRENT schema — the back-compat seam.

    Missing columns are filled with :func:`schema_default` (dtype-appropriate, so a
    numeric addition never raises on an older frame), extra columns are dropped, and the
    result carries the schema's column order + dtypes. ``None``/empty yields the canonical
    empty table. This is what lets a ``.nanoscope`` project or a CSV written before
    ``doc_key``/``channel_index``/``label`` existed load unchanged.
    """
    if df is None or len(df) == 0:
        return empty_measurements_df()
    out = df.copy()
    for col in MEASUREMENT_SCHEMA:
        if col not in out.columns:
            out[col] = pd.Series([schema_default(col)] * len(out), index=out.index)
        elif MEASUREMENT_SCHEMA[col][0] == "int64":
            # An int column that arrived with holes (a hand-edited CSV) cannot be cast
            # either; fill the holes with the same documented sentinel.
            if out[col].isna().any():
                out[col] = out[col].fillna(schema_default(col))
    return _cast(out)


def append_measurement(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Append one measurement `row` to `df`, returning a new schema-typed frame.

    `row` is typically ``measure_selection(...)`` augmented with the caller-stamped
    identity (`recording`/`doc_key`/`channel`/`channel_index`) and any `label`. Missing
    `measurement_id` is auto-assigned to the next index; every other absent column takes
    its :func:`schema_default` (NOT ``None``, which an int64/float64 cast rejects). Extra
    keys (e.g. the preview diagnostics) are dropped.
    """
    row = dict(row)
    row.setdefault("measurement_id", len(df))
    new = pd.DataFrame([{
        c: (row[c] if row.get(c) is not None else schema_default(c))
        for c in MEASUREMENT_SCHEMA
    }])
    out = pd.concat([coerce_measurements(df) if len(df) else empty_measurements_df(), new],
                    ignore_index=True)
    return _cast(out)


def write_measurements_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the measurement table to CSV (no index). Returns the path written.

    Coerced through :func:`coerce_measurements`, so an older in-memory frame (one loaded
    from a project written before a column existed) exports with the CURRENT header
    rather than raising.
    """
    p = Path(path)
    # Staged, like every other write over user data: a truncate-in-place export that dies
    # part way leaves a CSV with a plausible header and half the measurements, which is
    # worse than no file at all because it opens. See nanoscope.core.atomic_write.
    with staged_path(p) as staged:
        coerce_measurements(df).to_csv(staged, index=False)
    return p
