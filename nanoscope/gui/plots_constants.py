"""The constants and pure helpers of the Plots window, in a leaf of their own.

``plots_panel`` is the one place the window's configuration, saved-file format and
``PlotsWindow`` itself live; this module holds what both it and the ``plots_*_mixin``
parts only READ: column and plot-type names, fit, normalisation, colour and unit
constants, ``FrozenMap`` and the text/legend key helpers, the quantity helpers, and the
small value and pyqtgraph classes the window draws with.

``plots_panel`` re-imports every name defined here, so ``nanoscope.gui.plots_panel.X``
still resolves for each of them (tests, ``figure_adapter_*`` and ``figure_recipes`` import
them from there, and a monkeypatch of a name on ``plots_panel`` is read from there). This
module imports nothing from ``plots_panel`` or from a ``plots_*_mixin`` part: it is a leaf.

Every definition below was moved out of ``plots_panel.py`` without an edit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace as _dc_replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis import events, scoring, stats
from ..core.sources import PlotSource, source_colour
from ..plot.style import OKABE_ITO, PALETTE
from .categorical import category_colour
from .combos import choose_combo_item
from .figure_text import GATE, HIDDEN_TEXT, plain_text
from .font_picker import DEFAULT_ROW_LABEL

# Columns the detector's own dwell provenance (min_dwell_s / max_dwell_s /
# filter_rise_s) actually describes. A histogram of rise_s or ecd_pA_s is not a
# dwell distribution, so it must not borrow the dwell detection floor.
_DWELL_COLUMNS = ("dwell_s", "dwell", "dwell_time_s")
#: Time columns that are an INSTANT rather than an interval, i.e. the ones that mean
#: nothing without a time origin. ``dwell_s``/``rise_s``/``fall_s`` are durations and
#: are origin-free; ``t_start_s``/``t_end_s`` are "seconds since THIS recording's first
#: loaded sample" (:mod:`nanoscope.core.clock`) and are the only ones this window can
#: pool across recordings that do not share a clock.
#:
#: TAKEN FROM THE MODULE THAT DECLARES IT, not restated here. ``analysis.scoring``
#: already owns this list (``RECORDING_CLOCK_COLUMNS``, "columns in the tables this
#: module writes that are INSTANTS on the recording clock") and ``gui/panels`` reads
#: the same set to badge the event-table header. A private copy is how two surfaces
#: end up disagreeing about which columns are on a clock -- the same reason the
#: absolute-current and baseline-dependent lists live in ``core.sources``.
_ORIGIN_COLUMNS: tuple[str, ...] = tuple(scoring.RECORDING_CLOCK_COLUMNS)
# ... and the columns the detector's depth floor (threshold_out_pA) describes.
_DEPTH_COLUMNS = ("frac_depth", "depth_pA", "max_dev_pA", "mean_dev_pA")
# NOTE: the absolute-current column list and the baseline-dependent column list both
# live in core.sources -- the module that owns the cross-source comparability rule
# (:func:`nanoscope.core.sources.source_comparability` reads them). Restating either
# here is how the two surfaces drifted apart in the first place.

# Plot-type labels (also the QComboBox item texts / QStackedWidget page order).
HISTOGRAM = "Histogram"
SCATTER = "Scatter"
PCA = "PCA"
KDE = "KDE (density)"
#: The name Event waveforms already gives the same picture (``event_plot_panel.
#: DENSITY_LABEL``), so one quantity has one name across the app -- and it says COUNT,
#: because the guide's own rule is that a count heatmap is not a KDE density.
HEATMAP = "Event-count heatmap"
_PLOT_TYPES = (HISTOGRAM, SCATTER, KDE, HEATMAP, PCA)
#: What the status line calls a plot (quantity_reset), with its two columns filled in.
_SHOWS_PLOT = {HISTOGRAM: "the histogram of {x}", SCATTER: "the scatter of {y} against {x}",
               KDE: "the density of {y} against {x}",
               HEATMAP: "the event-count heatmap of {y} against {x}", PCA: "the PCA scores"}

#: The heatmap's colour maps: Event waveforms' list, all perceptually uniform and
#: sequential, so a count reads as a brightness and no hue boundary invents a level.
_HEATMAP_COLORMAPS = ("viridis", "magma", "plasma", "inferno", "cividis")

# Fit-overlay labels for the histogram.
_FIT_NONE = "none"
_FIT_EXP = "exponential"
_FIT_GMM = "gaussian-mixture"
#: Dwell families with a MODE. An exponential mixture is monotonically decreasing at every
#: component count, so it cannot describe a histogram that peaks away from the origin --
#: see stats.DWELL_PEAKED_MODELS. Truncated identically, so BIC compares directly.
_FIT_GAMMA = "gamma (peaked)"
_FIT_INVGAUSS = "inverse-gaussian (first passage)"
#: A fast escape AND a slow peaked passage in one model. Neither of the families above can
#: describe a BIMODAL dwell distribution whose slow lobe has a mode: an exponential mixture
#: is monotone in linear density, and a single gamma forced to explain a fast population as
#: well collapses to k ~ 1 (= an exponential). Measured on the owner's -600 mV apo file:
#: exp-mixture BIC -2062, gamma -2040, this -2682.
_FIT_EXP_GAMMA = "exponential + gamma (bimodal)"
#: The fits whose component count is a free choice. ``gamma``, ``inverse-gaussian`` and
#: ``exponential + gamma`` have a FIXED parameterisation, so a component spin — and an
#: Auto-BIC over it — means nothing for them.
_MULTI_COMPONENT_FITS = (_FIT_EXP, _FIT_GMM)

# Colour-by labels for the scatter AND (new) the histogram. "Source" is implicit:
# with >1 source checked the colour channel belongs to the source, always.
_COLOUR_NONE = "None"
_COLOUR_CLUSTER = "Cluster"

# Histogram y-quantity. Raw counts cannot be overlaid across recordings of
# different length; density and probability-per-bin can.
#
# WHY THIS ITEM IS NO LONGER CALLED "density (area = 1)" (report S6-03). The heights are
# ``counts / (n * bin_width)`` with the width in the column's STORED unit -- seconds for a
# dwell -- and ``_axis_label_for`` re-scales the x TICKS for display without touching them.
# It must not touch them: the exponential fit takes its detection floor from the detector's
# ``min_dwell_s`` in seconds, out of provenance, so scaling the sample would bias tau
# silently. Measured on the owner's -700 mV apo file (982 events, dwell_s, 30 bins) by
# driving this window offscreen, the curve as DRAWN integrates to
#
#     linear x:  1 (Time unit = s)   1000 (ms, the default)   1e6 (us)
#     log x:     23.3011485 -- IN ALL THREE UNITS
#
# and the log row is why no rescale was taken. On a log axis the abscissa is log10(s):
# ``AxisItem.scale`` stays 1, ``_DecadeLabelAxis.display_factor`` carries the unit, and the
# drawn area is a property of the DATA's range, not of any display factor. Dividing the
# heights by the display factor -- the obvious repair, prototyped and measured -- moves the
# log-x area to 0.0233011485 at ms and 2.33e-05 at us, i.e. FURTHER from 1, and costs a us
# y axis whose ticks render 0.00000 / 0.00005 / 0.00010 / 0.00015 on a publication figure.
# The shipped "Dwell-time histogram (log-x, exponential)" template below selects log x AND
# this mode AND ms, so the app's own preset lands on the unfixable case: the name asserted,
# on screen, something the same screen contradicted.
#
# The values are deliberately left alone; the FIGURE states its unit instead (see
# :meth:`PlotsWindow._y_axis_label`, which labels this axis ``density (1/s)`` /
# ``density (1/pA)``, and :meth:`PlotsWindow._density_unit_notes`, which says in Details
# where the drawn area is not 1 and by what factor -- and, on a log axis, that it is not a
# factor at all). "density" alone promises nothing that can be false, and costs nothing:
# measured, the combo's sizeHint stays 139 px and the window minimum 775 px, because
# "probability per bin" is still the widest item. ("density (area = 1 in stored units)" is
# also true and measures 221 px / 802 px -- it would make this the widest control here.)
_NORM_COUNT = "count"
_NORM_DENSITY = "density"
_NORM_PROB = "probability per bin"
_NORMS = (_NORM_COUNT, _NORM_DENSITY, _NORM_PROB)
#: What the status line says the histogram's y axis is after a Normalize pick (quantity_reset).
_NORM_PHRASE = {_NORM_COUNT: "counts", _NORM_DENSITY: "density", _NORM_PROB: "probability per bin"}


# Time display units. The event schema stores every time in SECONDS, which is rarely
# the unit anyone reads a dwell in -- ms is.
#
# THIS IS A DISPLAY SCALE, NOT A DATA CHANGE. It is applied through
# ``AxisItem.setScale``, which multiplies the tick LABELS and leaves the plotted values
# untouched. That distinction is load-bearing: the exponential dwell fit takes its
# detection floor from the detector's own ``min_dwell_s`` -- in seconds, out of
# provenance, see this module's header -- so scaling the SAMPLE without scaling that
# floor would bias tau silently, which is precisely the failure the header warns about.
# An axis scale cannot do that, because the fit never sees it.
#
# EVERY TIME COLUMN, DURATIONS *AND* INSTANTS, DERIVED RATHER THAN LISTED. This tuple
# is what makes a column read in the Time-unit combo's unit -- the axis label and its
# scale (:meth:`PlotsWindow._axis_label_for`), the printed parameters
# (:meth:`_display_time`) and the Fit-range boxes, which interpret what the user types
# in that same unit. Spelling the four names out left `t_end_s` behind on all three:
# it is an INSTANT this window already annotates with its clock in Details
# (:data:`_ORIGIN_COLUMNS`), yet its axis read a bare "t_end_s" -- no t₀ and no unit --
# while `t_start_s` beside it read "t_start (ms, from recording start (t₀ = 61.3034 s))".
# The axis label is the half an exported image keeps, so that is the half that mattered.
# Written as durations + :data:`_ORIGIN_COLUMNS` so an instant column added to the
# shared declaration in ``analysis.scoring`` arrives here already carrying its unit and
# its clock, instead of inheriting this hole in silence.
_DURATION_COLUMNS = ("dwell_s", "rise_s", "fall_s")
_TIME_COLUMNS: tuple[str, ...] = _DURATION_COLUMNS + _ORIGIN_COLUMNS
_TIME_UNITS: dict[str, tuple[float, str]] = {
    "s": (1.0, "s"), "ms": (1e3, "ms"), "µs": (1e6, "µs"),
}

#: What each EVENT COLUMN IS IN, taken from the schema that declares it rather than
#: guessed from its name. ``events.SCHEMA`` has carried a unit per column since the
#: file was written; this window was reading none of them (see ``_column_quantity``).
#:
#: Two translations, both because these strings are pasted straight onto a number:
#: the schema's ``"-"`` for dimensionless becomes ``""`` (" -" beside a value reads as
#: a minus sign), and ``"pA*s"`` becomes the ``pA·s`` this window already prints.
#: Everything else is passed through exactly as declared, ``"sample"`` included -- an
#: index IS in samples, and saying so is the difference between a labelled number and
#: a bare one.
_SCHEMA_UNITS: dict[str, str] = {
    col: ("" if unit == "-" else "pA·s" if unit == "pA*s" else unit)
    for col, (_dtype, unit, _desc) in events.SCHEMA.items()
}
_DEFAULT_TIME_UNIT = "ms"

#: Where a legend that has never been DRAGGED sits: 12 px in from the top-right corner,
#: which is where this window has always put it.
#:
#: THE ONLY TIME THIS IS APPLIED IS WHEN ``StyleSettings.legend_pos`` IS NONE, and that
#: restriction is the whole of round-4 Q4. pyqtgraph 0.14's ``LegendItem`` is draggable
#: out of the box -- ``LegendItem.mouseDragEvent`` calls ``autoAnchor`` on every mouse
#: move -- so the user could already put the legend anywhere they liked. A previous
#: round added a four-corner combo and implemented it by calling ``setOffset`` inside
#: :meth:`PlotsWindow._ensure_legend`, i.e. on EVERY draw; ``setOffset`` re-runs
#: ``anchor()``, so the next redraw threw the dragged position away. Measured on the
#: owner's two-source dwell overlay: a drag moved the legend from (487, 12) to
#: (287, 192) and the very next redraw put it back at (486.7, 12).
#:
#: So a corner is a STARTING POINT now, not a setting: once the user has moved the
#: legend, this constant is never consulted again until they press *Reset position*.
_LEGEND_DEFAULT_OFFSET: tuple[int, int] = (-12, 12)

#: The controls column's minimum width, and the width the splitter opens at. Both are
#: MEASURED quantities rather than round numbers: the first is the widest
#: ``minimumSizeHint`` the six control groups produce once the histogram page's Fit row
#: is unstacked, and the second is that plus a little air so nothing is at its limit on
#: a fresh window. ``tests/gui/test_plots_layout_room.py`` asserts the relationship the
#: old code got backwards -- the maximum was 24 px NARROWER than the content minimum --
#: so it cannot come back.
_CONTROLS_MIN_PX = 240
#: Air between the column's minimum and the width it OPENS at, so a fresh window is not
#: sitting on its own limit (which is what made the 420/444 mismatch invisible: the
#: column was already as wide as it could go).
_CONTROLS_SLACK_PX = 24

#: How long a combo has to be able to show, in characters, before it is allowed to stop
#: growing. ``QComboBox`` sizes itself to its WIDEST ITEM by default, so a single
#: 33-character entry ("inverse-gaussian (first passage)") set the width of the whole
#: controls column and, through it, of the horizontal scrollbar that would not go away.
#: The full text is still readable -- the popup is as wide as it needs to be, and the
#: tooltip spells every option out.
_COMBO_CHARS = 14

#: What an un-run window says on its canvas (round-4 Q1). The instruction is the whole
#: point: an empty plot with no words on it reads as a bug, and this window's empty
#: states have always said what to do next ("No data yet — run detection…", "Tick a
#: source on the left…") rather than merely being empty.
_PLOT_PROMPT = ("Choose your data and plot type, then update the plot.")

#: Free parameters in EITHER scatter best fit: (slope, intercept) for the straight
#: line, (coefficient, exponent) for the log-log one. What
#: :func:`nanoscope.analysis.stats.r2_dof_tag` subtracts from n to decide whether the
#: printed R² has any residual degrees of freedom left to be evidence over.
_SCATTER_FIT_PARAMS = 2

# Best-fit-line labels for the scatter (GUI text -> populations.linear_fit mode).
_SFIT_NONE = "none"
_SFIT_LINEAR = "linear"
_SFIT_POWER = "power law (log-log)"

# Clustering method labels (GUI text -> populations method id).
_METHOD_GMM = "GMM"
_METHOD_KMEANS = "k-means"
_METHOD_IDS = {_METHOD_GMM: "gmm", _METHOD_KMEANS: "kmeans"}

# k-selection modes for GMM.
_K_AUTO = "Auto (BIC)"
_K_FIXED = "Fixed k"

# Columns worth reaching for first when seeding the pickers, in priority order.
_PREFER_DWELL = ("dwell_s", "dwell", "rise_s", "fall_s")
_PREFER_AMP = ("amplitude", "depth_pA", "frac_depth", "frac_amplitude", "max_dev_pA")
_PREFER_FEATURES = (
    "dwell_s", "depth_pA", "frac_depth", "amplitude", "frac_amplitude",
    "max_dev_pA", "ecd_pA_s", "snr", "rise_s", "fall_s", "kurtosis", "skewness",
    "rms",
)
#: Row identifiers, not measurements: still offered as features (a saved configuration
#: may use them), but listed LAST under an "Identifiers" caption and never in Default.
#: ``t_start_s`` is not one of them -- a time of arrival is a legitimate drift feature.
_IDENTIFIER_FEATURES = ("event_id", "start_idx", "end_idx")

# Drawing colours (Okabe-Ito, reused for colour-blind-safe cluster colours).
_DATA_HEX = OKABE_ITO[0]        # blue -- histogram bars / plain scatter
_FIT_HEX = OKABE_ITO[1]         # vermillion -- fitted overlay
#: Orange -- the NOT-COMPARABLE verdict in the plot title. The same ink the
#: voltage-dependence window's banner uses, so one verdict reads as one verdict
#: wherever the user meets it.
_WARN_HEX = OKABE_ITO[4]
_EXCLUDED_HEX = PALETTE["excluded"]   # below-detection-limit shading (same ink as figures.py)
_DROPPED_HEX = "#9A9A9A"        # rows dropped from a clustering (label -1)
_MSG_HEX = "#8b9199"            # muted -- empty/invalid messages (on the dark theme)
_MSG_LIGHT_HEX = "#555b63"      # the same, darkened so it reads on a white background

# Fill alphas: the data colour is opaque as a hex, but the histogram fill and the plain
# scatter/PCA points draw it translucent (matching the old hard-coded mkBrush alphas).
_HIST_FILL_ALPHA = 110
_POINT_ALPHA = 160

# Theme foreground/background. "dark" reproduces the app theme exactly (gui.app), so the default
# style is byte-for-byte today's look; "light" is a white publication canvas with dark ink.
_BG_DARK = "#111316"
_FG_DARK = "#b9bfc7"
_BG_LIGHT = "#ffffff"
_FG_LIGHT = "#1a1a1a"


class FrozenMap(Mapping):
    """A read-only, hashable mapping -- the collection type a frozen style may hold.

    WHY THIS EXISTS RATHER THAN A PLAIN ``dict``. ``StyleSettings`` is frozen so a
    snapshot of it is a VALUE: hand one out, change the window, and the snapshot still
    describes the moment it was taken. ``frozen=True`` alone does not deliver that,
    because it only forbids rebinding the FIELDS -- a ``dict`` field stays shared, and
    ``dataclasses.replace()`` copies the reference, not the mapping. Measured on the
    pre-freeze class: ``replace(s, title="x").label_overrides is s.label_overrides`` was
    True, so writing one override through the copy also wrote it into the "old" style,
    and an undo stack built on that would have had every entry aliasing one live dict.

    Reading is a plain mapping (``m["k"]``, ``"k" in m``, ``m.get(k, default)``,
    ``m == {}``), which is why the ~6 read sites and the tests that compare against a
    literal dict did not have to change. Writing is by descendant: :meth:`set` and
    :meth:`remove` return a NEW map and leave this one alone.
    """
    __slots__ = ("_data",)

    def __init__(self, data: Mapping | Iterable = ()) -> None:
        # COPIED, never adopted. Building on the caller's dict would leave them a handle
        # that edits this "immutable" map from outside -- the same aliasing bug one
        # level down. ``__slots__`` above keeps the copy the only state there is.
        self._data: dict = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        # Compare equal to an ordinary dict with the same items, so a test may keep
        # asserting ``style.label_overrides == {}`` and a caller may keep using dicts.
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        # ``Mapping`` sets ``__hash__ = None``; a frozen dataclass holding this field
        # needs it back, and order-independence is what ``frozenset`` buys here.
        return hash(frozenset(self._data.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"

    def set(self, key, value) -> "FrozenMap":
        """This map plus ``key -> value``, as a new map (this one is untouched)."""
        return FrozenMap({**self._data, key: value})

    def remove(self, key) -> "FrozenMap":
        """This map without ``key``, as a new map. A missing key is not an error --
        the call sites are "forget this override", which is idempotent by nature."""
        if key not in self._data:
            return self
        rest = dict(self._data)
        del rest[key]
        return FrozenMap(rest)


#: The one empty map every default style shares. Safe to share precisely because it
#: cannot be written to -- which is the whole point of the type.
_NO_ENTRIES: FrozenMap = FrozenMap()


def _deep_frozen(value):
    """A JSON value as one a frozen style may hold: maps become :class:`FrozenMap`,
    lists become tuples, all the way down (a FrozenMap hashes its items)."""
    if isinstance(value, Mapping):
        return FrozenMap({str(k): _deep_frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_frozen(v) for v in value)
    return value


def _thawed(value):
    """:func:`_deep_frozen` undone: plain dicts and lists, for JSON and ``from_mapping``."""
    if isinstance(value, Mapping):
        return {str(k): _thawed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thawed(v) for v in value]
    return value


#: The fit label's element key (``figure_text``'s ``annotation.fit``).
FIT_LABEL_KEY = "annotation.fit"
#: The elements whose TEXT is a StyleSettings field of its own.
_TEXT_FIELD_OF = {"title": "title", "axis.bottom.label": "x_label", "axis.left.label": "y_label"}
#: The detection-floor line's words ("below detection limit"), an annotation of the figure.
LIMIT_LABEL_KEY = "annotation.limit"
_LIMIT_LABEL_AUTO = "below detection limit"


def _text_kept_in_styles(key: str) -> bool:
    """Whether ``text_styles[key]`` holds the element's TEXT too. An annotation's words
    have no field of their own (the fit label's live in ``fit_label``); every other
    element's text is a field (title, axis labels) or a source's name (``label_overrides``)."""
    return key.startswith("annotation.") and key != FIT_LABEL_KEY


#: The legend rows that are not a source's own, by the kind ``PlotsWindow._legend_id`` gives them.
_LEGEND_ROW_KINDS = ("fit", "cluster", "kde")


def _legend_style_key(kind: str, key: str) -> str:
    """The element key of one legend row's Text style: ``legend.entry.<source key>`` for a
    source's row (the key its Rename has always had), ``legend.entry.<kind>:<key>`` else."""
    return f"legend.entry.{key}" if kind == "series" else f"legend.entry.{kind}:{key}"


def _legend_entry_id(element_key: str, source_keys=()) -> str:
    """:func:`_legend_style_key` undone, as the ``"<kind>:<key>"`` entry id. A source key
    that itself reads ``fit:…`` is a source's row."""
    rest = element_key[len("legend.entry."):]
    kind, sep, key = rest.partition(":")
    if sep and kind in _LEGEND_ROW_KINDS and rest not in source_keys:
        return f"{kind}:{key}"
    return f"series:{rest}"


def _legend_styles(styles, legend: bool) -> FrozenMap:
    """``styles`` (a ``text_styles`` map) with only the legend rows' entries, or without them."""
    return FrozenMap({key: value for key, value in dict(styles or {}).items()
                      if key.startswith("legend.entry.") == legend})


def _style_text(value: str) -> str | None:
    """A Title / X / Y box value as ``TextStyle.text``: "" automatic, spaces hidden."""
    if not value:
        return None
    return value.strip() if value.strip() else HIDDEN_TEXT


def _unit_factor(time_unit: str) -> float:
    """Data-units-per-display-unit for the Time-unit combo's choice (s -> 1.0, ms -> 1e3)."""
    return _TIME_UNITS.get(time_unit, (1.0, "s"))[0]


def _display_factor(col: str, time_unit: str) -> float:
    """The factor ``col``'s axis is DISPLAYED in: the time unit for a time column, else 1.

    A non-time column's axis shows data units, so its typed limits need no conversion --
    which is exactly why a number typed for a dwell axis is nonsense over a picoamp one
    and :meth:`PlotConfig.rebased` clears it rather than converting it.
    """
    return _unit_factor(time_unit) if col in _TIME_COLUMNS else 1.0


def _rescaled(text: str, old_factor: float, new_factor: float) -> str:
    """A typed bound, rewritten from ``old_factor``'s display unit into ``new_factor``'s.

    ``repr`` rather than a format spec because these are round-trip values, not a
    readout: "0.1" ms -> "100.0" µs has to come back as exactly 1e-4 s when the box is
    read again. Anything that is not a finite number is left verbatim -- a half-typed
    "0." is not a bound yet, and "nan"/"inf" are reported to the user as typed by
    :meth:`PlotsWindow._fit_range_input_notes`.
    """
    if old_factor == new_factor or not text.strip():
        return text
    try:
        value = float(text)
    except ValueError:
        return text
    if not np.isfinite(value):
        return text
    return repr(value * new_factor / old_factor)



def _plot_column_quantity(column, display_unit="ms"):
    from ..core.quantity_display import quantity_for
    # This finite native classification predates this presentation migration; unknown
    # columns are deliberately not assigned seconds from spelling or magnitude.
    return quantity_for(column or "unselected", schema=events.SCHEMA,
        declared_unit="s" if column in _TIME_COLUMNS else None, display_unit=display_unit)


def _plot_density_quantity(column, display_unit="ms"):
    from ..core.quantity_display import density_quantity, QuantityPresentation
    quantity = _plot_column_quantity(column, display_unit)
    key = "histogram.density:" + (column or "unselected")
    if quantity.known and quantity.quantity not in ("categorical", "count", "sample"):
        return density_quantity([quantity], key=key)
    # Existing histograms also accept integer measurements and explicitly undeclared
    # numeric columns. Their units stay unchanged; never infer time or refuse their data.
    unit = quantity.stored_unit
    per = "" if unit in ("", "-") else (f"1/({unit})" if any(c in unit for c in "·*/ ") else "1/" + unit)
    return QuantityPresentation(key, "Density", "density", per, per, known=quantity.known)


def _plot_quantity_bases(cfg, *, legacy=False):
    from ..core.quantity_display import AxisDisplayBasis, quantity_for, density_quantity
    ident = cfg.identity
    x = _plot_column_quantity(ident.x_column, cfg.axes.time_unit)
    if ident.density_y:
        # Histograms normalize by LINEAR raw bin width even with log-spaced bins.
        # The log axis does not change the statistical reference measure.
        y = _plot_density_quantity(ident.x_column, cfg.axes.time_unit)
        if legacy:
            y = _dc_replace(y, display_unit=y.stored_unit, factor=1.)
    elif cfg.plot_type == HISTOGRAM:
        y = quantity_for("histogram." + cfg.histogram.normalization, declared_unit="-",
            quantity="count" if cfg.histogram.normalization == _NORM_COUNT else "dimensionless")
    else:
        y = _plot_column_quantity(ident.y_column, cfg.axes.time_unit)
    return {side: AxisDisplayBasis(quantity, "log10" if log else "linear")
            for side, quantity, log in zip(("bottom", "left"), (x, y), cfg.log_state)}


def migrate_plot_quantity(cfg):
    """Re-express legacy display drafts once; raw applied ranges/results never move.

    Incomplete legacy numerical text is refused at this boundary, retaining the
    original artifact. Existing witnessed drafts remain legal and verbatim.
    """
    from ..core.figure_quantity_basis import quantity_witness, read_quantity_bases
    from ..core.quantity_display import convert_pin, QuantityError
    old = read_quantity_bases(cfg.axes.quantity_display)
    target = _dc_replace(cfg, axes=_dc_replace(cfg.axes, time_unit="ms"))
    desired = _plot_quantity_bases(target)
    legacy = _plot_quantity_bases(cfg, legacy=True)
    changes = {}
    for side, prefix in (("bottom", "x"), ("left", "y")):
        previous, new = old.get(side, legacy[side]), desired[side]
        values = {"min": getattr(cfg.axes, prefix + "_min"), "max": getattr(cfg.axes, prefix + "_max")}
        if previous.quantity != new.quantity and any(values.values()):
            values = convert_pin(values, previous.quantity, new.quantity)
        changes[prefix + "_min"], changes[prefix + "_max"] = values["min"], values["max"]
    fit = cfg.fit
    if fit.range_unit != "ms" and cfg.histogram.column in _TIME_COLUMNS:
        values = convert_pin({"min": fit.range_lo, "max": fit.range_hi},
            _plot_column_quantity(cfg.histogram.column, fit.range_unit),
            _plot_column_quantity(cfg.histogram.column, "ms"))
        fit = _dc_replace(fit, range_lo=values["min"], range_hi=values["max"], range_unit="ms")
    axes = _dc_replace(target.axes, **changes, quantity_display=_deep_frozen(
        quantity_witness(desired, original=cfg.axes.quantity_display)))
    return _dc_replace(target, axes=axes, fit=fit)


# ================================================================================================
# UNDO / REDO -- a LOCAL, bounded stack of config values (owner item A2)
# ================================================================================================

#: How many past configurations one Plots window keeps.
#:
#: MEASURED, not guessed. A ``PlotConfig`` is 1,446 bytes fully expanded (every nested
#: dataclass and every string counted) and 1,584 with a typed title and two axis labels,
#: so this depth costs about 320 kB -- three orders of magnitude under
#: :data:`nanoscope.core.history.MAX_BYTES`, which is why this stack needs no byte ceiling
#: where the event-table one needs one first. That IS the difference between the two: an
#: event snapshot is a deep-copied DataFrame (38.7 MiB on a 100k-event table), a config
#: snapshot is forty scalars, and comparing two of them costs 0.4 us.
#:
#: Bounded anyway, because this window is constructed ONCE and re-shown for the whole
#: session (see :meth:`PlotsWindow.reset_state`), so an unbounded stack is a leak that
#: grows with every colour click of every figure made that day. 200 is deliberately far
#: above :data:`nanoscope.core.history.MAX_DEPTH` (30 per event table): walking back
#: thirty curation decisions is unusual, walking back thirty control moves while tuning
#: one figure is an ordinary afternoon.
_HISTORY_DEPTH = 200

#: How many origins' records the two import disclosures (:class:`_NormRefusal`,
#: :class:`_ConfigLosses`) keep before the oldest is dropped. A record is spoken only
#: while its origin is the one on screen, and the history can reach at most the base,
#: ``_HISTORY_DEPTH`` past entries, ``_HISTORY_DEPTH`` future entries and one pending
#: origin -- so this bound can never drop a record an undo or a redo could still land
#: on. It is a memory ceiling on the INERT ones: a template recalled five hundred times
#: opens five hundred origins and pushes no step at all.
_ORIGINS_KEPT = 2 * _HISTORY_DEPTH + 2


def _drop_oldest(records: dict) -> None:
    """Hold a per-origin record dict to :data:`_ORIGINS_KEPT` entries, oldest first
    (dicts keep insertion order; both records re-insert a re-taken origin at the end)."""
    while len(records) > _ORIGINS_KEPT:
        del records[next(iter(records))]

#: How long a run of KEYSTROKES may stay open before it becomes one undo entry. See
#: :meth:`PlotsWindow._history_touch_text` for the whole grain rule; this number only
#: decides how long the user must pause MID-TITLE, without leaving the box, before
#: "figure" and "figure 3b" become two entries instead of one.
_HISTORY_COALESCE_MS = 700


#: The symbol a fitted parameter is WRITTEN as, by the name ``stats`` stores it under.
#: ``FitResult.params`` keys are ASCII (``tau_1``, ``mu_2``, ``sigma_1``, ``w_2``)
#: because they travel through JSON into a saved project; the plot title has always
#: rendered the headline one as Greek, and the fit label has to use the SAME
#: alphabet or one figure names a quantity two ways in two corners of itself.
_PARAM_SYMBOLS: dict[str, str] = {
    "tau": "τ", "mu": "μ", "sigma": "σ", "lambda": "λ", "theta": "θ",
    "alpha": "α", "beta": "β", "w": "w", "k": "k",
}


def _param_symbol(key: str) -> str:
    """``"tau_1"`` -> ``"τ_1"``; anything unrecognised is passed through untouched."""
    head, sep, tail = str(key).partition("_")
    return f"{_PARAM_SYMBOLS.get(head, head)}{sep}{tail}"


#: Concise rejection tag for Details and saved diagnostic context, not artwork.
_KS_REJECTED_TAG = "CDF mismatch (descriptive)"


class _FitSummary(str):
    """A fit's one-line summary, carrying the FULL parameter set alongside it.

    Owner note #13 wants every fitted number on the plot, and owner note #14's box is
    where they go -- but the one-line form is still what the title and the legend
    take, because a title has one line's worth of room and a legend entry less. So one
    fit produces two strings.

    A ``str`` SUBCLASS rather than a third element of ``_overlay_fit``'s return tuple:
    the summary is consumed as text at five sites (the title, the legend, Details, the
    overlay list, the pending-fit placeholder) and every one of them keeps working
    unchanged, including callers outside this module that unpack ``(summary, notes)``.
    The richer form is read off ``.full`` by the one caller that needs it.

    ``rejected`` and ``plain`` keep the diagnostic verdict separate from artwork.
    The string includes the rejection tag for Details; the title and legend use
    ``plain``. ``full`` contains fitted parameters and numerical statistics (the old
    on-plot values box's text), without warning prose.
    """

    full: str = ""
    rejected: bool = False
    plain: str = ""
    #: The fit label's tokens (``fit_label.FIT_FIELDS`` -> text), set by the fitter.
    parts: dict | None = None

    def __new__(cls, text: str, full: str = "", *, rejected: bool = False,
                plain: str = ""):
        obj = super().__new__(cls, text)
        obj.full = full or text
        obj.rejected = bool(rejected)
        obj.plain = plain or text
        return obj


class _LegendText(str):
    """Keep the committed count separate from its optional presentation."""
    def __new__(cls, text, name, count):
        result = super().__new__(cls, text)
        result.name, result.count = str(name), count
        return result


class _FitLegendText(str):
    """A fit's legend row with its numbers (In legend instead): the head is elided like
    any row, the numbers are a second line that never is."""
    def __new__(cls, head, numbers):
        result = super().__new__(cls, f"{head}: {numbers}")
        result.head, result.numbers = str(head), str(numbers)
        return result


class _LimitLabel(pg.InfLineLabel):
    """The detection-floor line's words ("below detection limit"): figure text the user
    may retype, style or hide (``annotation.limit``). Like every annotation
    ``figure_text`` edits, it says when it is double-clicked (``editRequested``) or
    right-clicked (``menuRequested``, the screen position), from pyqtgraph's own dispatch.

    pyqtgraph re-sets an InfLineLabel's PLAIN format text whenever it is shown or its line
    moves (``valueChanged``); this one redraws the window's rich text instead, so the next
    show cannot drop the user's words or style."""

    editRequested = QtCore.Signal()
    menuRequested = QtCore.Signal(object)
    #: A left or right click landed on the words: the Format tab's selection (Phase 1). It
    #: is raised before the click is dispatched and changes nothing here; the figure-text
    #: controller connects any target item that has it.
    selectRequested = QtCore.Signal()
    #: The window's rich text, "" when hidden; None until the window first draws it.
    rich: str | None = None

    def show_rich(self, html: str) -> None:
        self.rich = html
        pg.TextItem.setVisible(self, bool(plain_text(html).strip()))
        self.valueChanged()

    def valueChanged(self):  # noqa: N802 - pyqtgraph API, our own subclass
        if self.rich is None:
            super().valueChanged()
        elif self.isVisible():
            self.setHtml(self.rich)
            self.updatePosition()

    def hoverEvent(self, ev):  # noqa: N802 - pyqtgraph API
        if not ev.isExit():
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)

    def mouseClickEvent(self, ev):  # noqa: N802 - pyqtgraph API
        if ev.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton):
            self.selectRequested.emit()
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and GATE(ev, ("limit", id(self))):
            ev.accept()
            self.editRequested.emit()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.menuRequested.emit(ev.screenPos())
        else:
            ev.ignore()


def _narrow_combo(combo: QtWidgets.QComboBox, chars: int = _COMBO_CHARS) -> None:
    """Stop a combo from sizing the whole controls column to its longest item.

    ``QComboBox.minimumSizeHint`` is the width of its WIDEST entry, which is why five
    fit families -- one of them "inverse-gaussian (first passage)" -- were between them
    responsible for a 444 px minimum on a 420 px column and the permanent horizontal
    scrollbar that followed. ``AdjustToMinimumContentsLengthWithIcon`` caps the CLOSED
    width at ``chars`` characters; the popup still opens at full width, so no option
    becomes unreadable.
    """
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)


#: What the Font row's first entry is called: the figure default, Arial. It stores ``""`` --
#: ``StyleSettings.font_family``'s "no family chosen" -- so a saved window keeps saying
#: "default" and a figure draws in ``figure_fonts.DEFAULT_FAMILY`` until one is chosen.
_FONT_DEFAULT_LABEL = DEFAULT_ROW_LABEL


def _swatch_text_color(hex_color: str) -> str:
    """Black or white ink for a colour swatch button, whichever is legible on it."""
    c = QtGui.QColor(hex_color)
    luma = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#000000" if luma > 140 else "#ffffff"


#: Colour for a cluster id; the dropped-row sentinel (-1) renders grey.
#:
#: THE ALIAS IS THE POINT. This used to be a private four-line function here, and
#: the subpopulation report window needed the identical map for its embedding
#: scatter -- a second copy of which would have been this project's signature
#: failure in colour. There is now one map, in ``gui.categorical``, and a test
#: asserts this name *is* that function rather than a copy of it.
_cluster_color = category_colour


def _numeric_columns(df: pd.DataFrame | None) -> list[str]:
    """Numeric, non-boolean columns of ``df`` that carry at least one finite value.

    Booleans (``accepted``, ``filter_limited``) are excluded -- they are flags,
    not quantities to histogram or cluster on. All-NaN numeric columns are
    dropped too, so the pickers never offer a column that can only draw an empty
    plot.
    """
    if df is None or df.shape[1] == 0:
        return []
    out: list[str] = []
    for c in df.columns:
        if str(c).startswith("amplitude_") and c != "amplitude_pA":
            # Audit coordinates/reference levels are not alternate event amplitudes.
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(v).any():
            out.append(str(c))
    return out


def _detection_key(result) -> tuple | None:
    """``(doc_id, channel, module_id)`` of a stored ``AnalysisResult``, or None.

    Duck-typed, because the only caller (:meth:`PlotsWindow._active_source_key`) is
    reaching across a window boundary at something it must not assume the shape of.
    """
    if result is None:
        return None
    doc_id = getattr(result, "doc_id", None)
    channel = getattr(result, "channel", None)
    module_id = getattr(result, "module_id", None)
    if doc_id is None or channel is None or not module_id:
        return None
    return (str(doc_id), int(channel), str(module_id))


def _normalize_sources(sources, provenance: dict | None = None) -> dict[str, PlotSource]:
    """Coerce either accepted ``set_data`` shape into ``{key: PlotSource}``.

    The owner pushes real :class:`~nanoscope.core.sources.PlotSource` objects (each
    already carrying its own provenance, colour and stable key). The legacy
    ``{label: DataFrame}`` mapping is still accepted -- it is how a bare frame gets
    explored in a test or a notebook-style session -- and is wrapped, keyed by its
    label, with a palette colour assigned in order.
    """
    if not sources:
        return {}
    if isinstance(sources, dict):
        prov = {str(k): dict(v) for k, v in (provenance or {}).items() if v}
        out: dict[str, PlotSource] = {}
        for i, (label, df) in enumerate(sources.items()):
            if df is None:
                continue
            key = str(label)
            out[key] = PlotSource(
                key=key, doc_key=key, label=key, df=df, provenance=prov.get(key, {}),
                colour=source_colour(i), display_name=key,
                kind="manual" if not prov.get(key) else "detection",
            )
        return out
    return {s.key: s for s in sources if s is not None and s.df is not None}


def _prefer(cols: Sequence[str], prefs: Sequence[str], fallback: int = 0) -> str | None:
    """First of ``prefs`` present in ``cols``; else the ``fallback``-th column."""
    for p in prefs:
        if p in cols:
            return p
    if cols:
        return cols[min(fallback, len(cols) - 1)]
    return None


def _column_quantity(col: str) -> tuple[str, str]:
    """``(quantity, unit)`` for a fitted column: what it measures, and in what.

    Fed to :func:`nanoscope.analysis.stats.fit_depth_gaussians`, which uses the first to
    NAME the model and the second to caption every ``mu``/``sigma`` it returns. Both used
    to be the constant "dI/I0", so a mixture fit to ``depth_pA`` came back labelled with
    the dimensionless ratio and printed picoamps beside it (report High-4).

    ``frac_depth`` keeps the dimensionless reading because that is what the column IS --
    and on the owner's apo-ferritin data it is 1.81-2.57, i.e. an enhancement, not a
    blockade fraction, which is precisely the number that must never be confused with a
    picoamp.

    THE HOLE THIS CLOSED. Everything not matched above used to answer ``""`` -- "this
    number has no unit" -- which for the TIME columns is false, and false by a factor of
    1000 on the surface where it shows. A ``gaussian-mixture`` fit to ``dwell_s`` printed

        mu_1 [0.01987, 0.02027]

    on the same window, the same axis and the same events where the ``exponential``
    printed ``tau_1 [19.85, 20.27] ms``: same quantity, same axis, a thousand-fold apart,
    and nothing on the page to say which was which. The estimate and CI lines convert a
    parameter whose unit is ``"s"`` into the axis's own display unit
    (:meth:`PlotsWindow._display_time`) and leave anything else alone -- so a time column
    that answers ``""`` here opts itself out of the one mechanism that would have made
    the two agree.

    Answered by SUFFIX, not by the enumerated time columns
    (:data:`_TIME_COLUMNS`), because the next ``*_s`` column added to the schema would
    otherwise inherit exactly this hole in silence. ``_pA_s`` is tested first: ``ecd_pA_s``
    also ends in ``_s`` and is a charge, not a time.

    The remaining ``""`` are true dimensionless readings -- ``snr`` and ``n_levels``.

    THE SCHEMA IS ASKED FIRST, and that is the whole point of this second pass. The
    suffix rules below are a GUESS from a column's spelling; ``events.SCHEMA`` is the
    project's own declaration of what each column is in, and it already carried the
    answer for every column in the frozen twenty. Guessing where a declaration exists
    is how ``start_idx``/``end_idx`` -- ``"sample"`` in the schema, since 0.1 -- went
    on answering "dimensionless" here: nothing in their spelling says otherwise, and
    a mixture fit to one printed ``mu = 1.196e+06`` with no unit beside it. Reading
    the schema also means a column ADDED to it arrives here already labelled, which
    the suffix table could only manage for names that happen to end the right way.

    The suffix rules stay, underneath, because the frames this window plots are not
    only schema frames: a merged table, a hand-measured table or a derived segment can
    carry a column the schema has never heard of, and ``*_pA`` / ``*_s`` is a better
    answer for those than silence. ``_pA_s`` is tested first: ``ecd_pA_s`` also ends
    in ``_s`` and is a charge, not a time.

    Two spellings are translated on the way out of the schema: ``"-"`` (its way of
    writing "dimensionless") becomes ``""``, because this string is CONCATENATED onto
    a number and " -" would read as a minus sign; and ``"pA*s"`` becomes the ``pA·s``
    this window already prints.
    """
    name = str(col)
    if name == "frac_depth":
        return stats.DEFAULT_DEPTH_QUANTITY, ""
    declared = _SCHEMA_UNITS.get(name)
    if declared is not None:
        return name, declared
    if name.endswith("_pA_s"):          # ecd_pA_s -- checked first, it also ends in _s
        return name, "pA·s"
    if name.endswith("_pA"):
        return name, "pA"
    if name.endswith("_s") or name in _TIME_COLUMNS:
        return name, "s"
    return name, ""


#: THE INVARIANT LIVES IN :mod:`nanoscope.gui.combos`, not here.
#:
#: ``QComboBox.setCurrentText`` on a non-editable combo is a no-op for unmatched text --
#: no exception, no signal, no change -- so every setter built on it fails silently and
#: the window reports the PREVIOUS selection's numbers. The first fix put this function in
#: this file and wired it into 2 of this window's 7 combo setters and 0 of the sibling
#: windows; the same defect then bit `VDepWindow.set_metric` during verification. It is a
#: shared module now, and `test_plots_panel.py` asserts no setter here bypasses it.
_choose_combo_item = choose_combo_item
