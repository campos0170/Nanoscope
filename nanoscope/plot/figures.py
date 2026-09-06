"""Publication figures for nanopore event statistics.

Every function returns ``(fig, ax)`` and attaches the numbers behind the figure
to the figure object, so `save_figure` can export them as CSV alongside the
vector art. Figures are drawn inside `nanoscope.plot.style.nanoscope_style`, so
they look right without the caller opting in, and nothing leaks into the
caller's rcParams.

Two things in here are easy to get wrong and are the reason this module exists:

1. **Log bins.** Dwell times span decades. Linear bins on log-distributed data
   turn an exponential into something that looks like a power law. Default is
   `np.logspace`.
2. **The log-bin Jacobian.** A density fitted in linear time, evaluated at a
   bin centre and plotted over a *log* bin, is wrong by a factor of
   ``t * ln(10) * dlog10(t)`` -- which varies by orders of magnitude across the
   axis, yet still produces a curve that hugs the data closely enough to look
   plausible. See `_expected_counts_per_bin`.

Data-model note (v2 port): the four event-statistics figures + ``save_figure``
are data-model-agnostic -- they eat an ``EventSet``, a raw event ``DataFrame``,
or an ``AnalysisResult.payload`` via the duck-typed ``_unpack``. Only
``trace_preview`` touches the signal, and it takes the target
``CalibratedChannel`` + ``XAxis`` (never a legacy ``Recording``), threading the
channel's own ``units`` into the axis label instead of hard-coding pA
(PORT_PLAN R9).
"""

from __future__ import annotations

import copy
import json
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D, AffineDeltaTransform
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..analysis import events as events_mod
from ..analysis.events import FLAG_NON_FINITE, FLAG_ZERO_BASELINE
from ..analysis.stats import _row_labels
from ..core import clock
from ..core.atomic_write import StagedSet
from .style import (OKABE_ITO, PALETTE, SEQUENTIAL_CMAP, color, figure_size,
                    nanoscope_style, rc_params)

__all__ = [
    "dwell_histogram",
    "depth_histogram",
    "scatter_dwell_depth",
    "joint_density",
    "event_rate_plot",
    "trace_preview",
    "save_figure",
    "figure_details",
    "save_plot_image",
    "image_write_failure_text",
    "ImageNotRendered",
    "figure_output_paths",
    "FIGURE_OUTPUT_EXTENSIONS",
    "export_plot_pdf",
    "plot_source_data",
    "plot_data_paths",
    "save_plot_data",
    "PLOT_DATA_PARTS",
]

_DATA_ATTR = "_nanoscope_source_data"
_DETAILS_ATTR = "_nanoscope_plot_details"


def figure_details(fig: Figure) -> dict[str, Any]:
    """Detached scientific context for Details/export metadata, never plot artwork.

    A figure may contain several report panels, so each panel keeps its own source
    provenance, fit parameters/diagnostics, and full advisory text. Annotation
    visibility must not affect this record.
    """
    return copy.deepcopy(getattr(fig, _DETAILS_ATTR, {"schema_version": 1, "plots": []}))


def _attach_details(fig: Figure, kind: str, prov: dict, *, fit: Any = None,
                    notes: Sequence[str] = (), **extra: Any) -> None:
    details = figure_details(fig)
    fit_data = None if fit is None else {
        "params": getattr(fit, "params", {}), "ci": getattr(fit, "ci", {}),
        "n": getattr(fit, "n", None), "meta": getattr(fit, "meta", {}),
        "warnings": getattr(fit, "warnings", ()),
    }
    text: list[str] = []
    for values in (prov.get("warnings", ()), prov.get("baseline_warnings", ()),
                   getattr(fit, "warnings", ()), notes):
        if isinstance(values, str):
            values = [values]
        text.extend(str(value) for value in (values or ()) if value is not None and str(value).strip())
    if fit is not None:
        text.extend(_fit_caveats(fit))
        gof = _gof(fit)
        if gof and gof.get("note"):
            text.append(str(gof["note"]))
    details["plots"].append(copy.deepcopy({
        "kind": kind, "provenance": prov, "fit": fit_data,
        "details": list(dict.fromkeys(text)), **extra,
    }))
    setattr(fig, _DETAILS_ATTR, details)


def _details_json(fig: Figure) -> str:
    """Strict JSON for PDF Subject / PNG Description; source arrays stay untouched."""
    def normal(value):
        if isinstance(value, dict):
            return {str(key): normal(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normal(item) for item in value]
        if isinstance(value, np.ndarray):
            return normal(value.tolist())
        if isinstance(value, np.generic):
            return normal(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
    return json.dumps(normal(figure_details(fig)), ensure_ascii=False, allow_nan=False)


# --------------------------------------------------------------------------
# input plumbing
# --------------------------------------------------------------------------
def _unpack(events: Any) -> tuple[pd.DataFrame, dict]:
    """Return (accepted-only event frame, provenance dict) from an EventSet or DataFrame.

    Figures always show curated-in events only; a figure that quietly includes
    events the user rejected is a lie about what they curated.
    """
    provenance: dict = dict(getattr(events, "provenance", {}) or {})
    df = getattr(events, "df", events)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected an EventSet or DataFrame, got {type(events).__name__}")
    if "accepted" in df.columns:
        df = df[df["accepted"].astype(bool)]
    return df.reset_index(drop=True), provenance


def _column(df: pd.DataFrame, name: str) -> np.ndarray:
    """One column as float, refusing to plot anything that is not a measurement.

    THE REFUSAL NAMES THE ROWS. "column 'frac_depth' contains non-finite values" tells a
    user who has just unticked *Use accepted events only* nothing they can act on: the
    Plots panel handed a whole column over, the offending rows are the automatically
    rejected ``non_finite`` / ``zero_baseline`` ones, and nothing on screen said which.
    ``stats._row_labels`` is the helper that answers that, in the frame's OWN index, and
    it is used here for the same reason ``stats._clean_1d`` uses it — the two are the same
    refusal on the same rows, reached through the two different doors (a fit, a figure),
    and a rule that holds at one consumer and not the other is not a rule.

    Source sanitisation (``events.demote_infinities``) turns +/-Inf into NaN, so what
    still reaches here is a genuinely unmeasured row — a blanked baseline, an unreachable
    90 % level — which is exactly the case worth naming.
    """
    if name not in df.columns:
        raise KeyError(f"events frame has no column {name!r}; got {list(df.columns)}")
    series = df[name]
    v = np.asarray(series, dtype=float)
    if v.size == 0:
        raise ValueError(f"no events to plot (column {name!r} is empty)")
    if not np.all(np.isfinite(v)):
        bad = ~np.isfinite(v)
        index = series.index if isinstance(series, pd.Series) and len(series.index) == v.size \
            else None
        raise ValueError(
            f"column {name!r} contains {int(bad.sum())} non-finite value(s), "
            f"{_row_labels(index, bad)}; refusing to plot a wrong figure. Those rows carry "
            f"a 'flag' saying why they were not measured ({FLAG_NON_FINITE}, "
            f"{FLAG_ZERO_BASELINE}); ticking 'Use accepted events only' excludes them, "
            f"which is what every export already does."
        )
    return v


def _n_bins(n: int, lo: int = 12, hi: int = 48) -> int:
    return int(np.clip(int(np.sqrt(max(n, 1))), lo, hi))


def _fig_of(ax: Axes | None, size: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        return fig, ax
    return ax.figure, ax


def _attach(fig: Figure, data: pd.DataFrame) -> None:
    setattr(fig, _DATA_ATTR, data)


# --------------------------------------------------------------------------
# the Jacobian
# --------------------------------------------------------------------------
def _bin_width_at(t: np.ndarray, edges: np.ndarray, log_bins: bool) -> np.ndarray:
    """Local bin width in **linear** units at position `t`.

    This is the whole ballgame for overlaying a fitted density on a log-binned
    histogram of counts. For bins uniform in u = log10(t) with width du, a bin
    at t covers dt = t * ln(10) * du of linear time -- so a pdf f(t) in units of
    1/s predicts ``N * f(t) * t * ln(10) * du`` counts there, not ``N * f(t) * du``
    and not ``N * f(t)``. The factor t*ln(10) spans decades across the axis,
    which is exactly why omitting it produces a curve that still looks vaguely
    right and is nonetheless wrong.
    """
    if log_bins:
        du = float(np.log10(edges[1]) - np.log10(edges[0]))
        return np.asarray(t, dtype=float) * np.log(10.0) * du
    return np.full(np.shape(t), float(edges[1] - edges[0]))


def _expected_counts_per_bin(
    pdf: Any,
    t: np.ndarray,
    edges: np.ndarray,
    n_fit: int,
    log_bins: bool,
) -> np.ndarray:
    """Expected counts at positions `t` for a histogram with `edges`.

    `pdf` is a callable returning the linear-time probability density (units
    1/s), normalized over the fitted support -- i.e. FitResult.predict.
    """
    dens = np.asarray(pdf(t), dtype=float)
    if dens.shape != np.shape(t):
        raise ValueError(f"fit.predict returned shape {dens.shape} for input of shape {np.shape(t)}")
    return n_fit * dens * _bin_width_at(t, edges, log_bins)


def _bin_centers(edges: np.ndarray, log_bins: bool) -> np.ndarray:
    """Midpoint of each bin, geometric for log bins (the midpoint rule in log space)."""
    if log_bins:
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------
# fits (owned by nanoscope.analysis.stats)
# --------------------------------------------------------------------------
def _stats_module():
    try:
        from ..analysis import stats
    except ImportError as e:
        raise ImportError(
            "fitting requires nanoscope.analysis.stats, which is unavailable "
            f"({e}). Pass fit=False to plot the histogram alone, or pass a "
            "pre-computed FitResult as fit=."
        ) from e
    return stats


def resolve_t_min(prov: dict, observed_min: float, *, notes: list[str] | None = None) -> float:
    """The detection limit for a dwell figure, warning when it had to be guessed.

    Delegates to :func:`nanoscope.analysis.modules.resolve_t_min` -- the SAME
    resolution the censored fit and the interactive Plots window use, so a figure
    can never be drawn against a different detection floor than the fit printed
    in the GUI (that divergence was worth ~25% of tau). Anything short of
    ``provenance['min_dwell_s']`` raises a RuntimeWarning naming what it fell
    back to.
    """
    try:
        from ..analysis.modules import resolve_t_min as _resolve
    except ImportError:  # pragma: no cover - analysis package always ships
        return float(observed_min)
    t_min, _source, note = _resolve(prov, observed_min=float(observed_min))
    if note:
        if notes is not None:
            notes.append(str(note))
        warnings.warn(note, RuntimeWarning, stacklevel=3)
    return float(t_min)


def _resolve_fit(fit: Any, fitter, *args, **kwargs):
    """`fit=True` -> call the stats fitter; `fit=<FitResult>` -> use it as given."""
    if fit is True:
        return fitter(*args, **kwargs)
    for attr in ("predict", "params"):
        if not hasattr(fit, attr):
            raise TypeError(f"fit= must be True, False, or a FitResult with .{attr}; got {type(fit).__name__}")
    return fit


def _gof(fit: Any) -> dict | None:
    """The goodness-of-fit verdict ``analysis.stats._stamp_gof`` left on the fit, or None.

    ``meta['gof']`` is ``{'ks', 'ks_critical_001', 'describes_data', ...}`` on every fit
    the stats module returns; a stand-in fit without one gets no verdict and no
    rejection, exactly as ``FitResult.summary`` treats it.
    """
    meta = getattr(fit, "meta", None) or {}
    gof = meta.get("gof") if isinstance(meta, dict) else None
    if not isinstance(gof, dict) or "ks" not in gof:
        return None
    return gof


def _fit_rejected(fit: Any) -> bool:
    """True when the app's own KS test threw this curve out (``describes_data is False``)."""
    gof = _gof(fit)
    return gof is not None and gof.get("describes_data") is False


def _gof_verdict(fit: Any) -> str | None:
    """The existing KS verdict for Details and image metadata, never artwork."""
    gof = _gof(fit)
    if gof is None:
        return None
    ks = float(gof["ks"])
    crit = gof.get("ks_critical_001")
    # Two visual lines each: the box on a single-column figure holds ~42 characters a
    # line (see _BOX_LINE_CHARS), and the verdict must not squeeze the histogram.
    if _fit_rejected(fit):
        against = f" > {float(crit):.3f} (1%)" if crit is not None else ""
        return (f"KS D = {ks:.3f}{against}: fit REJECTED\n"
                "— parameters are not measurements")
    against = f" ≤ {float(crit):.3f} (1%)" if crit is not None else ""
    return f"KS D = {ks:.3f}{against}:\nconsistent with these data"


def _param_items(fit: Any, needle: str) -> list[tuple[str, float, tuple[float, float] | None]]:
    """Pull params whose name mentions `needle`, with CI if the FitResult has one.

    NEVER A CONFIDENCE INTERVAL FOR A REJECTED FIT. An interval is a stronger claim
    than a point estimate -- "this is a measurement, and this is its precision" -- and
    this is the one place every figure in this module gets its intervals from, so a
    fit whose own KS verdict is ``describes_data: False`` hands out ``None`` bounds
    here, whatever its ``ci`` dict holds. The point value stays (labelled by
    :func:`_param_text`) because the curve is drawn and a reader must be able to name
    what was drawn; its precision is withheld because it is not a measurement.
    """
    params = dict(getattr(fit, "params", {}) or {})
    ci = {} if _fit_rejected(fit) else dict(getattr(fit, "ci", {}) or {})
    out = []
    for k, v in params.items():
        if needle not in k.lower():
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        bounds = ci.get(k)
        if bounds is not None:
            try:
                bounds = (float(bounds[0]), float(bounds[1]))
            except (TypeError, ValueError, IndexError):
                bounds = None
        out.append((k, val, bounds))
    return out


def _param_text(symbol: str, val: float, ci: tuple[float, float] | None, *,
                rejected: bool, scale: float = 1.0, fmt: str = ".1f",
                unit: str = "") -> str:
    """One annotation line for a fitted parameter: ``symbol = value [lo, hi] unit``.

    Rejected-fit intervals remain withheld as before; the verdict and original
    intervals are retained in figure details, not appended to the artwork.
    """
    text = f"{symbol} = {val * scale:{fmt}}"
    if ci is not None:
        text += f" [{ci[0] * scale:{fmt}}, {ci[1] * scale:{fmt}}]"
    text += unit
    return text


def _depth_wording(depth: np.ndarray) -> tuple[str, str]:
    """Quantity label and median summary; warnings are retained separately in details."""
    median = float(np.median(depth))
    enhanced = bool(events_mod.frac_depth_note(median))
    # Do not call an out-of-range excursion a blockade. The explanatory warning
    # belongs to figure_details, while the axis retains its scientific quantity.
    label = (r"$\Delta I / I_0$" if enhanced
             else r"blockade depth $\Delta I / I_0$")
    median_line = rf"median $\Delta I / I_0$ = {median:.3f}"
    return label, median_line


def _fit_caveats(fit: Any) -> list[str]:
    """Complete existing fit caveats for Details and image metadata, not the canvas."""
    meta = dict(getattr(fit, "meta", {}) or {})
    out: list[str] = []
    n_excl = int(meta.get("n_excluded_filter_limited", 0) or 0)
    if n_excl:
        out.append(f"{n_excl} filter-limited event(s) excluded")
    t_min = meta.get("t_min")
    if t_min is not None:
        src = str(meta.get("t_min_source", "") or "")
        label = rf"$t_{{min}}$ = {float(t_min) * 1e6:.1f} $\mu$s"
        out.append(label if src in ("", "min_dwell_s") else f"{label} ({src})")
    t_max = meta.get("t_max")
    if t_max is not None:
        out.append(rf"right-truncated at {float(t_max) * 1e6:.1f} $\mu$s (modelled)")
    if meta.get("floor_biased"):
        floor = meta.get("depth_floor")
        out.append(f"floor-biased: $\\mu_1$ cut off at {float(floor):.3f}")
    verdict = _gof_verdict(fit)
    if verdict:
        out.append(verdict)
    return out


def _fit_support(fit: Any) -> list[str]:
    """Scientific model support labels, separate from warning/verdict text."""
    meta = dict(getattr(fit, "meta", {}) or {})
    out = []
    if meta.get("t_min") is not None:
        out.append(rf"$t_{{min}}$ = {float(meta['t_min']) * 1e6:.1f} $\mu$s")
    if meta.get("t_max") is not None:
        out.append(rf"$t_{{max}}$ = {float(meta['t_max']) * 1e6:.1f} $\mu$s")
    return out


#: Characters per visual line of the annotation box on a single-column figure. Measured
#: on the exported -600 mV dwell figure: the box's longest line, "tau = 20411.6 [17554.2,
#: 25035.2] us" (33 rendered characters at 7 pt), spans 55% of the 2.6-inch axes; a
#: 75-character verdict written as ONE line made ``figure.autolayout`` shrink the axes
#: to a third of the figure to fit the text inside the canvas. Every line this module
#: writes is either under this budget or split with an explicit newline, and
#: :func:`_annotate` wraps any math-free line that is not.
_BOX_LINE_CHARS = 42


def _wrap_line(line: str) -> str:
    """Fold a math-free line longer than the box budget; leave mathtext lines alone.

    A ``$...$`` span cannot be broken across lines (mathtext parses each line on its
    own), so lines carrying one are written pre-split by their authors instead. An
    author's own fold (an explicit newline) is kept: each segment is wrapped on its
    own, so "fit REJECTED / — parameters are not measurements" folds where it was
    written and not where ``textwrap`` would re-flow it.
    """
    if "$" in line:
        return line
    import textwrap
    return "\n".join(
        "\n".join(textwrap.wrap(seg, _BOX_LINE_CHARS, break_long_words=False,
                                break_on_hyphens=False) or [seg])
        if len(seg) > _BOX_LINE_CHARS else seg
        for seg in line.split("\n"))


#: Set (truthy) on an Axes to make :func:`_annotate` write ABOVE it -- right-aligned in
#: the title band, beside a left-aligned title -- instead of inside it. For a figure
#: whose data fill the axes edge to edge (the trace preview), where an in-axes box
#: would sit on the trace. A marker on the axes rather than a keyword because
#: ``_annotate(ax, lines)`` is the documented single call site for every string this
#: module writes onto exported axes, and the sweep-7 repro wraps it with a
#: two-argument lambda: a third argument raised TypeError inside that wrapper and the
#: exporter's best-effort loop silently dropped the whole trace figure.
_ANNOTATE_ABOVE_ATTR = "_nanoscope_annotate_above"

#: Where the in-axes box is anchored: its top edge and its inset from either side. The
#: right-hand corner is the pre-sweep anchor (0.97, 0.95); the box leaves it only when
#: the other corner has less drawn under it (:func:`_place_clear_of_data`).
_BOX_TOP = 0.95
_BOX_RIGHT = 0.97
_BOX_LEFT = 0.03
#: Clearance (axes fraction) the box keeps between its bottom edge and the highest thing
#: drawn beneath it, and between its side and a guide it had to step past. Wider than
#: the box's own background pad (0.25 em at 7 pt = 1.75 pt against ~4 pt here) and than
#: the shift a changed y tick label makes to the axes at the export draw.
_BOX_CLEARANCE = 0.03
#: The most of the axes' width the box may take before its font is reduced to fit. The
#: scatter's main axes are ~1.75 in wide (marginals and colour bar take the rest of the
#: column) and its 34-character enhancement line at 7 pt was 104 % of that: the box
#: began left of the y spine, over the tick labels (X-SURF-1).
_BOX_MAX_WIDTH = _BOX_RIGHT - _BOX_LEFT
_BOX_MIN_FONT_PT = 5.0
#: A guide (a patch not on ``transData``) spanning at least this much of the axes'
#: height is a wall -- the "below detection limit" shading, a filter-limited band --
#: which no headroom clears: the box steps sideways past it instead.
_SKY_BLOCKED = 0.9
_SKY_COLUMNS = 240
#: Headroom is not taken when it would leave the data less than this share of the axes:
#: a histogram squeezed into a quarter of its axes to fit a box is a worse figure than a
#: box over the data, and the reader still has the box's background to read it by.
_MIN_DATA_SHARE = 0.25

#: The dpi the figure family is rendered at -- and therefore the ONLY dpi at which
#: measuring the box means anything. Taken from the shared style so that
#: :func:`save_figure` and the placer cannot drift apart; ``save_figure`` passes it to
#: ``savefig``, and :func:`_renderer_of` draws at it before anything is measured.
#:
#: A BOX MEASURED AT THE SCREEN DPI IS NOT THE BOX THAT IS EXPORTED. ``figure.dpi`` is
#: 110 and ``savefig.dpi`` is 300; at 110 the glyph advances are hinted to whole pixels
#: at 7 pt (9.7 px/em) and ``figure.autolayout`` reserves less room for the tick labels,
#: so the same seven-line box measured 67.6 % of the axes' width at 110 dpi and 77.1 %
#: of it in the 300 dpi PNG the user opens -- and the scatter's box, 93.7 % wide at 110
#: dpi (inside :data:`_BOX_MAX_WIDTH`, so its font was left alone), was 107.7 % wide in
#: the export, i.e. it began LEFT of the y spine. Measured on the owner's -700 mV
#: recording through the real window; the dwell box placed clear of the peak at 110 dpi
#: had two 0.75-height bars under it at 300 dpi (A2-NEW-1 / X-SURF-1, second cut).
_EXPORT_DPI = float(rc_params()["savefig.dpi"])

#: Point size of the annotation box. The literal 7 the box has always used, named because
#: :func:`_box_font` and :data:`_BOX_LINE_CHARS` are both measured against it.
_BOX_FONT_PT = 7.0


def _box_font(size_pt: float = _BOX_FONT_PT) -> FontProperties:
    """The box's font, resolved to a FILE here so it is DRAWN with what it was MEASURED with.

    THE BOX WAS LAID OUT IN ONE FONT AND PRINTED IN ANOTHER (A2-NEW-1 / X-SURF-1, second
    cut). A figure is built inside :func:`~nanoscope.plot.style.nanoscope_style`, where
    ``font.sans-serif`` is Helvetica; ``save_figure`` is a separate call the exporter
    makes OUTSIDE that context, where the family resolves to matplotlib's own DejaVu
    Sans -- and a Text keeps its family as the unresolved word "sans-serif", so
    ``findfont`` runs again at draw time and answers differently. Measured at 300 dpi on
    one verdict line: laid out 0.718 of the axes wide, drawn 0.778 -- 8 % more ink than
    the layout reserved, ending at x = 1.034, i.e. past the right spine. The placer is
    only as good as its measurement, so the depth box printed over the fitted curve and
    the scatter box began outside the axes even after it had been placed clear of both.
    Resolving the family to a path AT BUILD TIME pins the drawing to the font the
    measurement used: same line, laid out 0.718 and drawn 0.713, ending at 0.967.

    Only the annotation box is pinned. The rest of the figure's text has the same split
    (the axis labels are positioned in Helvetica and printed in DejaVu, and the exported
    PDF gets the default Type-3 glyphs rather than the style's ``pdf.fonttype`` 42),
    which is ``save_figure`` rendering outside the style the module docstring promises --
    a wider change than this box, and not this finding's.
    """
    return FontProperties(fname=findfont(FontProperties(size=size_pt)), size=size_pt)


def _renderer_of(fig: Figure):
    """The renderer the figure is drawn with, at :data:`_EXPORT_DPI`, after a draw.

    A draw and not a layout call, because two things are only settled at draw time:
    ``figure.autolayout`` runs tight_layout then, and the scatter's marginal axes are
    positioned by an axes locator then. Measured before a draw, the axes are their
    un-laid-out size and the box's share of them is understated by a fifth.

    The dpi is the caller's business to restore (:func:`_place_clear_of_data` does it in
    a ``finally``): every measurement taken off this renderer is a fraction of the axes,
    and those fractions move with the dpi -- see :data:`_EXPORT_DPI`.
    """
    fig.set_dpi(_EXPORT_DPI)
    fig.canvas.draw()
    get = getattr(fig.canvas, "get_renderer", None)
    return get() if get is not None else fig._get_renderer()


def _paint(sky: np.ndarray, x0: Any, x1: Any, y: Any) -> None:
    """Raise the skyline to ``y`` over the columns spanning ``[x0, x1]`` (axes fractions)."""
    x0 = np.asarray(x0, dtype=float).ravel()
    x1 = np.asarray(x1, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = sky.size
    lo, hi = np.minimum(x0, x1), np.maximum(x0, x1)
    keep = (np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
            & (hi >= 0.0) & (lo <= 1.0) & (y > 0.0))
    if not keep.any():
        return
    a = np.clip(np.floor(lo[keep] * n), 0, n - 1).astype(int)
    b = np.clip(np.floor(hi[keep] * n), 0, n - 1).astype(int)
    for i, j, h in zip(a, b, np.minimum(y[keep], 1.0)):
        seg = sky[i:j + 1]
        np.maximum(seg, h, out=seg)


def _paint_polyline(sky: np.ndarray, pts: np.ndarray) -> None:
    """Paint a polyline given as (N, 2) axes fractions; each segment at its higher end."""
    p = np.asarray(pts, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        return
    if p.shape[0] == 1:
        _paint(sky, p[:, 0], p[:, 0], p[:, 1])
        return
    _paint(sky, p[:-1, 0], p[1:, 0], np.maximum(p[:-1, 1], p[1:, 1]))


def _paint_collection(sky: np.ndarray, ax: Axes, art: Any, ax_bb: Any) -> None:
    """Paint a collection's cells (hexbin) or markers (scatter) as drawn, or its areas."""
    def frac(px):
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        return np.column_stack(((px[:, 0] - ax_bb.x0) / ax_bb.width,
                                (px[:, 1] - ax_bb.y0) / ax_bb.height))

    offsets = np.asarray(art.get_offsets(), dtype=float)
    paths = art.get_paths()
    if offsets.size == 0 or not len(paths):
        # A collection of areas (fill_between): its polygons, as drawn.
        for path in paths:
            for poly in path.to_polygons(transform=art.get_transform(), closed_only=False):
                _paint_polyline(sky, frac(poly))
        return
    otr = art.get_offset_transform()
    master = art.get_transform()
    if isinstance(otr, AffineDeltaTransform):
        # hexbin's convention: the offsets are in the axis's SCALED units (log10 on a log
        # axis) and are placed by the affine part of the data transform -- what the
        # delta transform is the delta of -- while the one hexagon is stored in data
        # units so that the non-affine part of `master` returns it to scaled units.
        # Offsets pushed through the full transData landed at x = -10 of the axes.
        centres_px = master.get_affine().transform(offsets.reshape(-1, 2))
    else:
        centres_px = otr.transform(offsets.reshape(-1, 2))
    centres = frac(centres_px)
    per = np.asarray(art.get_transforms())
    half_w = half_h = 0.0
    for i, path in enumerate(paths):
        # The full transform (get_extents handles a non-affine one): a scatter marker's
        # per-path size in pixels, a hexbin cell's size in scaled units.
        tr = (Affine2D(per[i % len(per)]) + master) if len(per) else master
        ext = path.get_extents(tr)
        half_w = max(half_w, float(ext.width) / 2.0)
        half_h = max(half_h, float(ext.height) / 2.0)
    hw, hh = half_w / ax_bb.width, half_h / ax_bb.height
    _paint(sky, centres[:, 0] - hw, centres[:, 0] + hw, centres[:, 1] + hh)


def _skyline(ax: Axes, ax_bb: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(sky, blocked)``: the highest drawn point per column of the axes' width, and the
    columns a wall stands in -- both in axes fractions, ``_SKY_COLUMNS`` wide.

    Everything that carries data is painted as drawn -- histogram bars bin by bin (not
    their one bounding box, which on a full-width histogram is the whole axes below the
    peak), fitted curves point by point, hexbin cells and scatter markers by their
    centres and size. Guides are told apart by their transform (``get_data_transform``
    is not ``transData``): an ``axvspan`` spanning the axes' height (the "below
    detection limit" shading, a filter-limited band) is a WALL -- a column no headroom
    clears, marked in ``blocked`` for :func:`_place_clear_of_data` to step past, and
    never mistaken for data: a peak bar that reaches 0.95 of the axes is data the box
    must clear by headroom, and a rule that read walls off height took that peak for a
    wall and left it under the box. An ``axvline`` (a fitted mean, the detection floor)
    paints nothing, because a dotted guide passing behind a legible box is what every
    reader expects.

    Paths are transformed with ``Transform.transform_path`` and only then converted to
    polygons: ``Path.to_polygons(transform=...)`` keeps just the affine part of a
    transform, which on a log x axis put every bar at the wrong column.
    """
    sky = np.zeros(_SKY_COLUMNS)
    blocked = np.zeros(_SKY_COLUMNS, dtype=bool)

    def frac(px):
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        return np.column_stack(((px[:, 0] - ax_bb.x0) / ax_bb.width,
                                (px[:, 1] - ax_bb.y0) / ax_bb.height))

    for art in ax.patches:
        if not art.get_visible():
            continue
        try:
            polys = [frac(p) for p in
                     art.get_transform().transform_path(art.get_path()).to_polygons(
                         closed_only=False)]
            if art.get_data_transform() is ax.transData:
                for poly in polys:
                    _paint_polyline(sky, poly)
                continue
            pts = np.concatenate([p for p in polys if len(p)]) if polys else None
            if pts is None or pts.size == 0:
                continue
            ys = pts[:, 1]
            if np.nanmax(ys) - np.nanmin(ys) < _SKY_BLOCKED:
                continue                      # a band, not a wall: a guide to sit over
            n = sky.size
            a = int(np.clip(np.floor(np.nanmin(pts[:, 0]) * n), 0, n - 1))
            b = int(np.clip(np.floor(np.nanmax(pts[:, 0]) * n), 0, n - 1))
            blocked[a:b + 1] = True
        except Exception:
            continue
    for art in ax.lines:
        if not art.get_visible() or art.get_transform() is not ax.transData:
            continue
        try:
            _paint_polyline(sky, frac(ax.transData.transform(art.get_xydata())))
        except Exception:
            continue
    for art in ax.collections:
        if not art.get_visible():
            continue
        try:
            _paint_collection(sky, ax, art, ax_bb)
        except Exception:
            continue
    return sky, blocked


def _box_size(box: Any, renderer: Any, ax_bb: Any) -> tuple[float, float]:
    """``(width, height)`` of the box AS DRAWN, in axes fractions -- background included.

    ``Text.get_window_extent`` measures the glyphs; what the reader sees is the glyphs
    plus the white background patch, which ``boxstyle="square,pad=0.25"`` inflates by
    0.25 em on every side (1.75 pt at :data:`_BOX_FONT_PT` -- 7.3 px each side at
    :data:`_EXPORT_DPI`, 1.8 % of a single-column axes' width). Placed on the glyph
    extent alone, the depth box's background edge landed 0.011 of the axes left of where
    the placer thought it ended, and there the fitted curve is near vertical: 540 px of
    it were printed over (measured on the owner's -600 mV events, exported PNG, box ink
    diffed against the same figure with the box painted in "none").

    The pad is COMPUTED from the boxstyle, not read off the patch: the patch's extent is
    only refreshed when the text is drawn, so after :func:`_place_clear_of_data` shrinks
    the font to fit the axes the patch still reports its 7 pt size (414.7 px against the
    text's 286.4 px, measured). ``pad`` is a multiple of the font size, and matched the
    drawn patch to 0.01 px at both 110 and 300 dpi.
    """
    tb = box.get_window_extent(renderer)
    w_px, h_px = float(tb.width), float(tb.height)
    patch = box.get_bbox_patch()
    if patch is not None:
        try:
            pad_px = (float(patch.get_boxstyle().pad) * float(box.get_fontsize())
                      * float(box.figure.dpi) / 72.0)
            w_px += 2.0 * pad_px
            h_px += 2.0 * pad_px
        except Exception:
            pass
    return w_px / ax_bb.width, h_px / ax_bb.height


def _place_clear_of_data(ax: Axes, box: Any) -> None:
    """Put the annotation box where the data is not: the emptier top corner, with headroom.

    THE VERDICT WAS PRINTED ON TOP OF THE DATA IT JUDGES (A2-NEW-1, X-SURF-1). The
    sweep-7 lines grew the box on the exported dwell figure from 3 to 7 lines -- 24 % to
    51 % of the axes' height and 66 % to 77 % of its width, measured through the real
    window on the owner's -700 mV recording (1,179 events) -- and on the depth figure
    from 2 to 6 lines (14 % to 44 %). Anchored at the top-right corner with no
    background, "KS D = 0.511 > 0.047 (1%): fit REJECTED" and "open-pore baseline not
    certified" ran through 2 bars each on the dwell histogram and 6/6/7/8 bars on the
    depth histogram, whose peak stood at 0.97 of the axes under a box whose bottom was
    at 0.52, in the same ink as the fitted curve they crossed. On the scatter the
    34-character enhancement line was 104 % of the main axes' width, so the box began
    left of the y spine, over the tick labels. The lines were right; the placement made
    them the least legible text on the page.

    ONE PLACER, THREE RULES, NOTHING PER FIGURE:

    1. **Width.** A box wider than :data:`_BOX_MAX_WIDTH` of the axes has its font
       reduced to fit (never below :data:`_BOX_MIN_FONT_PT`). Lines are already folded
       at :data:`_BOX_LINE_CHARS`; this is for an axes narrower than the one that budget
       was measured on.
    2. **Corner.** Of the two top corners, the one with less drawn under the box's
       footprint (:func:`_skyline`, bin by bin) wins; a tie keeps the pre-sweep
       right-hand anchor. A guide drawn edge to edge (an ``axvspan``) is a wall the box
       steps past, not something it sits on. On the owner's -600 mV dwell figure the
       events sit at 12.6-26 ms on an axis that starts at 10 µs, so the top-left is blank
       canvas: the box goes there and nothing else changes.
    3. **Headroom.** If what is under the chosen corner still reaches the box, the
       y-limit is raised -- only upward, and only as far as it takes for the box to clear
       it by :data:`_BOX_CLEARANCE`. The depth histogram's bars span its full width, so
       there the box keeps its corner and the histogram sits below it; the bars, their
       counts and the source-data CSV are unchanged, only the scale they are drawn on.
       Not taken when it would leave the data under :data:`_MIN_DATA_SHARE` of the axes.

    The box also has a light background (0.85 white, no edge) and sits above the
    curves, so whatever guide does pass behind it -- shading, a dotted mean -- does not
    print through the verdict. Placement needs a renderer; without one (or on any error
    while measuring) the box keeps the corner it was given, which is exactly what the
    figure did before, and never raises into the exporter's best-effort loop.

    EVERY FRACTION HERE IS MEASURED AT :data:`_EXPORT_DPI`, not at the screen dpi the
    figure was created with -- the box's share of the axes differs by a tenth between
    the two, which is the difference between clearing the peak and standing on it. The
    figure's own dpi is restored before returning, so nothing the caller holds is left
    changed; ``figure.autolayout`` re-runs at whatever dpi the next draw uses, and the
    export's draw is at this one.
    """
    fig = ax.figure
    dpi0 = float(fig.get_dpi())
    try:
        renderer = _renderer_of(fig)
        ax_bb = ax.get_window_extent(renderer)
        if not (ax_bb.width > 0 and ax_bb.height > 0):
            return
        w, h = _box_size(box, renderer, ax_bb)
        if w > _BOX_MAX_WIDTH:
            box.set_fontsize(max(_BOX_MIN_FONT_PT,
                                 float(box.get_fontsize()) * _BOX_MAX_WIDTH / w))
            w, h = _box_size(box, renderer, ax_bb)

        sky, blocked = _skyline(ax, ax_bb)
        n = sky.size
        free_lo, free_hi = _BOX_LEFT, _BOX_RIGHT
        if not blocked.all():
            run = int(np.argmin(blocked))            # first free column from the left
            if run > 0:
                free_lo = max(free_lo, run / n + _BOX_CLEARANCE)
            run = int(np.argmin(blocked[::-1]))      # ...and from the right
            if run > 0:
                free_hi = min(free_hi, 1.0 - run / n - _BOX_CLEARANCE)

        def under(x0: float, x1: float) -> float:
            a = int(np.clip(np.floor(x0 * n), 0, n - 1))
            b = int(np.clip(np.ceil(x1 * n), a + 1, n))
            seg, ok = sky[a:b], ~blocked[a:b]
            return float(seg[ok].max()) if ok.any() else 0.0

        fits = (free_hi - free_lo) >= w
        right = (free_hi - w, free_hi) if fits else (_BOX_RIGHT - w, _BOX_RIGHT)
        left = (free_lo, free_lo + w) if fits else (_BOX_LEFT, _BOX_LEFT + w)
        u_right, u_left = under(*right), under(*left)
        if u_left < u_right - 0.01:
            box.set_ha("left")
            box.set_position((left[0], _BOX_TOP))
            beneath = u_left
        else:
            box.set_ha("right")
            box.set_position((right[1], _BOX_TOP))
            beneath = u_right

        floor = _BOX_TOP - h - _BOX_CLEARANCE
        if beneath > floor >= _MIN_DATA_SHARE:
            y0, y1 = ax.get_ylim()
            if y1 > y0:
                # In the axis's own scale (log-safe): the point now at `beneath` of the
                # axes must land at `floor` of it.
                tr = ax.yaxis.get_transform()
                s0, s1 = (float(v) for v in tr.transform(np.array([y0, y1], dtype=float)))
                s_top = s0 + beneath * (s1 - s0) / floor
                y_top = float(tr.inverted().transform(np.array([s_top], dtype=float))[0])
                if np.isfinite(y_top) and y_top > y1:
                    ax.set_ylim(y0, y_top)
    except Exception:
        return
    finally:
        try:
            fig.set_dpi(dpi0)
        except Exception:
            pass


def _annotate(ax: Axes, lines: list[str]) -> None:
    """The one call site for every string this module writes onto exported axes.

    Keep the signature at exactly ``(ax, lines)`` -- see :data:`_ANNOTATE_ABOVE_ATTR`.
    Call it LAST in a figure, after every limit and label is final: the box is placed
    around what is drawn (:func:`_place_clear_of_data`), so it must see the figure the
    reader will see.
    """
    if not lines:
        return
    text = "\n".join(_wrap_line(str(l)) for l in lines)
    if getattr(ax, _ANNOTATE_ABOVE_ATTR, False):
        ax.text(1.0, 1.02, text, transform=ax.transAxes, ha="right", va="bottom",
                fontproperties=_box_font(), color=PALETTE["fit"], linespacing=1.35)
        return
    box = ax.text(
        _BOX_RIGHT,
        _BOX_TOP,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontproperties=_box_font(),
        color=PALETTE["fit"],
        linespacing=1.35,
        zorder=6,
        bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none",
                  alpha=0.85),
    )
    _place_clear_of_data(ax, box)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def dwell_histogram(
    events: Any,
    *,
    ax: Axes | None = None,
    fit: Any = True,
    log_bins: bool = True,
    annotate: bool = True,
) -> tuple[Figure, Axes]:
    """Dwell-time histogram with an optional fitted exponential overlay.

    Units: dwell in seconds, plotted in microseconds on a log axis. The y axis
    is counts per bin.

    Gotchas handled here:
      * Bins are log-spaced by default (`np.logspace`). Dwell times span
        decades; linear bins on them are a classic way to mistake an
        exponential for a power law.
      * The fitted curve carries the log-bin Jacobian (see `_bin_width_at`).
      * The region below the detection limit `min_dwell_s` is shaded: those
        events were never detectable, so the histogram's left edge is an
        artefact of the detector, not of the physics. The fit is a *censored*
        MLE for exactly this reason.

    Args:
        events: EventSet (accepted events are used) or an event DataFrame.
        ax: Axes to draw into; a new figure is made if None.
        fit: True to fit via `analysis.stats.fit_dwell_exponential`, False for no
            overlay, or a pre-computed FitResult (whose `.predict` must return
            the linear-time density in 1/s over the fitted support). PREFER the
            pre-computed one whenever the user has already run a fit: `True`
            refits from scratch at n_components=1, so a figure exported after a
            two-component fit would carry a different tau from the one the app
            displayed. Its complete metadata, caveats and verdict remain in
            :func:`figure_details` and the saved PDF/PNG metadata.
        log_bins: log-spaced bins (default) vs linear.
        annotate: write fitted parameters, n and model-support limits onto the
            axes. Set False for a shared-caption panel. Warning/verdict text is
            always confined to figure details; rejected-fit intervals remain
            withheld without changing the fit or its original confidence bounds.

    Returns:
        (fig, ax).
    """
    df, prov = _unpack(events)
    dwell = _column(df, "dwell_s")
    if np.any(dwell <= 0):
        raise ValueError("dwell_s contains non-positive values; a dwell time cannot be <= 0")

    limit_notes: list[str] = []
    t_min = resolve_t_min(prov, float(dwell.min()), notes=limit_notes)
    fitted = None

    with nanoscope_style():
        fig, ax = _fig_of(ax, figure_size("single"))
        lo, hi = float(dwell.min()), float(dwell.max())
        if hi <= lo:
            hi = lo * 1.5 if log_bins else lo + 1e-9
        nb = _n_bins(dwell.size)
        edges = np.logspace(np.log10(lo), np.log10(hi), nb + 1) if log_bins else np.linspace(lo, hi, nb + 1)
        # logspace endpoints can round inside [min, max] and drop the extreme
        # events out of the histogram; pull both edges back to enclose the data.
        edges[0] = np.nextafter(min(edges[0], lo), -np.inf)
        edges[-1] = np.nextafter(max(edges[-1], hi), np.inf)
        counts, edges = np.histogram(dwell, bins=edges)
        centers = _bin_centers(edges, log_bins)

        ax.stairs(counts, edges * 1e6, fill=True, color=PALETTE["data"], alpha=0.75, linewidth=0)
        ax.stairs(counts, edges * 1e6, fill=False, color=PALETTE["data"], linewidth=0.8)
        if log_bins:
            ax.set_xscale("log")

        fit_at_centers = np.full(centers.shape, np.nan)
        if fit is not False:
            fitted = _resolve_fit(
                fit,
                lambda: _stats_module().fit_dwell_exponential(dwell, t_min=t_min),
            )
            # Scale the curve by the sample the FIT saw, not by what the histogram
            # shows. They are the same number for a fit made here; they differ when
            # a pre-computed fit excluded filter-limited events, and then scaling by
            # the histogram's n would draw a curve the fit never claimed.
            #
            # Same reasoning for the FLOOR: a supplied fit knows the t_min it was
            # actually conditioned on, which is NOT always resolve_t_min(prov) -- a
            # fit that dropped filter-limited events raised its floor to
            # 2*filter_rise_s. Shading at the provenance floor while the curve is
            # conditioned on a higher one draws the dashed "below detection limit"
            # line in the wrong place and leaves the excluded bars unmarked. Adopt
            # the fit's own floor so the shading, the curve and the annotation all
            # describe one number.
            fit_t_min = (getattr(fitted, "meta", None) or {}).get("t_min")
            if fit_t_min is not None and np.isfinite(fit_t_min) and fit_t_min > 0:
                t_min = float(fit_t_min)

            n_fit = int(getattr(fitted, "n", 0) or 0) or int(np.count_nonzero(dwell >= t_min))
            fit_at_centers = _expected_counts_per_bin(fitted.predict, centers, edges, n_fit, log_bins)

            grid_lo = max(t_min, lo)
            grid = (
                np.logspace(np.log10(grid_lo), np.log10(hi), 400)
                if log_bins
                else np.linspace(grid_lo, hi, 400)
            )
            curve = _expected_counts_per_bin(fitted.predict, grid, edges, n_fit, log_bins)
            ax.plot(grid * 1e6, curve, color=PALETTE["fit"], lw=1.4, zorder=5, label="exponential fit")

            rejected = _fit_rejected(fitted)
            texts = [
                _param_text(r"$\tau$", val, ci, rejected=rejected, scale=1e6,
                            unit=r" $\mu$s")
                for _name, val, ci in _param_items(fitted, "tau")
            ]
            texts.append(f"n = {n_fit}")
            texts.extend(_fit_support(fitted))
            box = texts if annotate else []
        else:
            box = []

        # The user must see what they cannot see.
        left = min(t_min, lo) * 0.5
        if t_min > left:
            ax.axvspan(left * 1e6, t_min * 1e6, color=PALETTE["excluded"], alpha=0.3, lw=0, zorder=0)
            ax.axvline(t_min * 1e6, color=PALETTE["excluded"], lw=0.8, ls="--", zorder=1)
            ax.text(
                t_min * 1e6,
                0.02,
                " below detection limit",
                transform=ax.get_xaxis_transform(),
                ha="left" if log_bins else "left",
                va="bottom",
                fontsize=6,
                color="#5A5A5A",
                rotation=90,
            )
        ax.set_xlim(left * 1e6, hi * 1.3e6)
        ax.set_xlabel(r"dwell time ($\mu$s)")
        ax.set_ylabel("events per bin")
        # Last: the box is placed around what is drawn, at the limits the reader sees.
        _annotate(ax, box)

    _attach(
        fig,
        pd.DataFrame(
            {
                "bin_left_s": edges[:-1],
                "bin_right_s": edges[1:],
                "bin_center_s": centers,
                "count": counts,
                "fit_expected_count": fit_at_centers,
            }
        ),
    )
    _attach_details(fig, "dwell_histogram", prov, fit=fitted, notes=limit_notes,
                    detection_limit_s=t_min, annotate=bool(annotate))
    return fig, ax


def depth_histogram(
    events: Any,
    *,
    ax: Axes | None = None,
    fit: Any = True,
    n_components: int = 1,
    annotate: bool = True,
) -> tuple[Figure, Axes]:
    """Blockade-depth (dI/I0) histogram with a fitted Gaussian mixture overlay.

    Units: `frac_depth` is dimensionless and always positive (direction lives in
    the detected polarity, never in the sign of the feature). The y axis is a
    probability density, so the mixture pdf overlays directly with no Jacobian
    -- bins are linear here.

    Args:
        events: EventSet (accepted events) or event DataFrame.
        ax: Axes to draw into.
        fit: True to fit via `analysis.stats.fit_depth_gaussians`, False, or a
            pre-computed FitResult whose `.predict` returns the density.
        n_components: mixture components to fit.
        annotate: write fitted means (with applicable intervals) and n onto the
            axes. Full caveats/verdicts remain in figure details and metadata.

    Returns:
        (fig, ax).
    """
    df, prov = _unpack(events)
    depth = _column(df, "frac_depth")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    fitted = None

    with nanoscope_style():
        fig, ax = _fig_of(ax, figure_size("single"))
        lo, hi = float(depth.min()), float(depth.max())
        if hi <= lo:
            hi = lo + max(abs(lo) * 0.1, 1e-6)
        edges = np.linspace(lo, hi, _n_bins(depth.size) + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        dens, edges = np.histogram(depth, bins=edges, density=True)
        centers = _bin_centers(edges, False)

        ax.stairs(dens, edges, fill=True, color=PALETTE["data"], alpha=0.75, linewidth=0)
        ax.stairs(dens, edges, fill=False, color=PALETTE["data"], linewidth=0.8)

        fit_at_centers = np.full(centers.shape, np.nan)
        if fit is not False:
            fitted = _resolve_fit(
                fit,
                lambda: _stats_module().fit_depth_gaussians(depth, n_components=n_components),
            )
            grid = np.linspace(lo, hi, 400)
            ax.plot(grid, np.asarray(fitted.predict(grid), dtype=float),
                    color=PALETTE["fit"], lw=1.4, zorder=5, label="Gaussian mixture")
            fit_at_centers = np.asarray(fitted.predict(centers), dtype=float)

            texts = []
            rejected = _fit_rejected(fitted)
            means = _param_items(fitted, "mu") or _param_items(fitted, "mean")
            for _, val, ci in means:
                ax.axvline(val, color=PALETTE["fit"], lw=0.8, ls=":", zorder=4)
                texts.append(_param_text(r"$\mu$", val, ci, rejected=rejected, fmt=".3f"))
            texts.append(f"n = {int(getattr(fitted, 'n', 0) or 0) or depth.size}")
            box = texts if annotate else []
        else:
            box = []

        # A FIGURE LEAVES THE APP. If the median of this very histogram is above 1 the
        # x axis is not a blockade depth at all -- the current MAGNITUDE grew -- and a
        # co-author reading the exported PNG has nothing else to go on. On the owner's
        # apoferritin recordings the median is 1.81-2.57 (report High-5), so the axis
        # said "blockade depth" over a distribution centred at 2.4. Same threshold and
        # same function as the Summary row and the CSV sidecar -- and the same helper
        # as the scatter's y axis, which drifted from this one once (MISSED-1).
        x_label, _median_line = _depth_wording(depth)
        ax.set_xlabel(x_label)
        ax.set_ylabel("probability density")
        ax.set_xlim(lo, hi)
        # Last: the box is placed around what is drawn, at the limits the reader sees.
        _annotate(ax, box)

    _attach(
        fig,
        pd.DataFrame(
            {
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "bin_center": centers,
                "density": dens,
                "fit_density": fit_at_centers,
            }
        ),
    )
    _attach_details(fig, "depth_histogram", prov, fit=fitted,
                    notes=[events_mod.frac_depth_note(float(np.median(depth)))],
                    annotate=bool(annotate))
    return fig, ax


def scatter_dwell_depth(
    events: Any,
    *,
    ax: Axes | None = None,
    density: bool = True,
    annotate: bool = True,
) -> tuple[Figure, Axes]:
    """The canonical nanopore figure: dwell (log x) vs blockade depth (y).

    Marginal histograms sit on both axes. Filter-limited events are drawn as
    open markers over a shaded band: their dwell is within ~2x the filter rise
    time, so the filter never let the current reach the true blockade level --
    their depths are attenuated (biased shallow) and their dwells biased long.
    Those points lie, and the reader has to be told which ones.

    Args:
        events: EventSet (accepted events) or event DataFrame.
        ax: main Axes; marginals are appended around it.
        density: True -> hexbin coloured by counts (a scatter of 5000 points is
            an unreadable blob); False -> transparent scatter.
        annotate: write this population's summary numbers onto the axes, the way
            the two histograms write their fitted ones. WHAT numbers differs, and
            the difference is not an oversight: the histograms have a MODEL, so
            they print its parameters; this figure fits nothing, so it prints what
            it actually measured -- n, the two medians, and how many of the points
            are filter-limited and therefore lying about their depth. A saved PNG
            that names neither is a picture a reader cannot quote. The median dI/I0
            and the y axis name the measured quantity without calling values above
            one blockades. Full warnings are retained in figure details/metadata.

    Returns:
        (fig, ax) where ax is the main (joint) axes.
    """
    df, prov = _unpack(events)
    dwell = _column(df, "dwell_s")
    depth = _column(df, "frac_depth")
    if np.any(dwell <= 0):
        raise ValueError("dwell_s contains non-positive values; cannot use a log axis")
    limited = (
        np.asarray(df["filter_limited"], dtype=bool)
        if "filter_limited" in df.columns
        else np.zeros(dwell.size, dtype=bool)
    )

    with nanoscope_style():
        fig, ax = _fig_of(ax, figure_size("single", aspect=0.95))
        # The divider owns the layout once marginals exist.
        fig.set_layout_engine("none")
        div = make_axes_locatable(ax)
        ax_top = div.append_axes("top", size="20%", pad=0.06, sharex=ax)
        ax_right = div.append_axes("right", size="20%", pad=0.06, sharey=ax)

        x = dwell * 1e6
        ax.set_xscale("log")
        if density:
            hb = ax.hexbin(x, depth, xscale="log", gridsize=32, mincnt=1,
                           cmap=SEQUENTIAL_CMAP, linewidths=0.0)
            cax = div.append_axes("right", size="5%", pad=0.32)
            cb = fig.colorbar(hb, cax=cax)
            cb.set_label("events per bin", fontsize=7)
            cb.ax.tick_params(labelsize=6)
        else:
            ax.scatter(x[~limited], depth[~limited], s=8, alpha=0.35,
                       color=PALETTE["event"], linewidths=0, label="events")

        if limited.any():
            band_hi = float(dwell[limited].max())
            rise = prov.get("filter_rise_s")
            if rise is not None:
                band_hi = max(band_hi, 2.0 * float(rise))
            ax.axvspan(x.min() * 0.5, band_hi * 1e6, color=PALETTE["filter_limited"],
                       alpha=0.10, lw=0, zorder=0)
            ax.scatter(x[limited], depth[limited], s=14, facecolors="none",
                       edgecolors=PALETTE["filter_limited"], linewidths=0.7, zorder=4,
                       label="filter-limited")
            ax.legend(loc="lower right", fontsize=6)

        nb = _n_bins(dwell.size)
        xe = np.logspace(np.log10(dwell.min()), np.log10(dwell.max()), nb + 1) * 1e6
        ye = np.linspace(depth.min(), depth.max(), nb + 1)
        ax_top.hist(x, bins=xe, color=PALETTE["data"], alpha=0.8, linewidth=0)
        ax_right.hist(depth, bins=ye, orientation="horizontal",
                      color=PALETTE["data"], alpha=0.8, linewidth=0)
        for a in (ax_top, ax_right):
            a.tick_params(labelsize=6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)
        ax_top.set_yticks([])
        ax_right.set_xticks([])
        ax_top.spines["left"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)

        # The y axis and the median line come from the SAME guard the depth histogram
        # uses (MISSED-1): this figure said "blockade depth" and "median dI/I0 = 2.524"
        # bare over a cloud in which 236 of 275 events were enhancements.
        y_label, median_line = _depth_wording(depth)
        if annotate:
            # Placed by _annotate like the histograms' box. The top-right is NOT reliably
            # the sparse corner here: for a translocation population (dwell rising to
            # the right, depth to the top) it is, and for the owner's enhancement
            # population (median dI/I0 2.5, cloud at 2-4.5 over 12-26 ms) the box sat on
            # the hexbin cloud -- so the placer measures rather than assumes.
            texts = [f"n = {dwell.size}",
                     rf"median dwell = {float(np.median(dwell)) * 1e6:.1f} $\mu$s",
                     median_line]
            n_limited = int(np.count_nonzero(limited))
            if n_limited:
                texts.append(f"filter-limited: {n_limited}")
            box = texts
        else:
            box = []

        ax.set_xlabel(r"dwell time ($\mu$s)")
        ax.set_ylabel(y_label)
        # Last: the box is placed around what is drawn, at the limits the reader sees.
        _annotate(ax, box)

    _attach(
        fig,
        pd.DataFrame({"dwell_s": dwell, "frac_depth": depth, "filter_limited": limited}),
    )
    _attach_details(fig, "scatter_dwell_depth", prov,
                    notes=[events_mod.frac_depth_note(float(np.median(depth))),
                           "Filter-limited event depths may be attenuated and dwells biased long."
                           if limited.any() else ""],
                    n_filter_limited=int(np.count_nonzero(limited)), annotate=bool(annotate))
    return fig, ax


def event_rate_plot(
    runset_df: pd.DataFrame,
    *,
    x: str = "voltage_mV",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Event rate vs voltage (or concentration), one line per pore.

    Pores are plotted individually and never averaged into a single line with
    error bars: pore-to-pore spread -- geometry, wetting, surface charge --
    dominates counting noise by a lot, so the spread *between* pores is the real
    uncertainty. Averaging it away manufactures precision that isn't there.

    Args:
        runset_df: one row per (pore, condition) with columns `x`, 'pore_id',
            and an event-rate column ('rate_hz', 'event_rate_hz' or 'rate').
        x: column for the x axis, e.g. 'voltage_mV' or 'concentration'.
        ax: Axes to draw into.

    Returns:
        (fig, ax).
    """
    if not isinstance(runset_df, pd.DataFrame):
        raise TypeError(f"runset_df must be a DataFrame, got {type(runset_df).__name__}")
    if x not in runset_df.columns:
        raise KeyError(f"runset_df has no column {x!r}; got {list(runset_df.columns)}")
    rate_col = next((c for c in ("rate_hz", "event_rate_hz", "rate") if c in runset_df.columns), None)
    if rate_col is None:
        raise KeyError(
            f"runset_df has no event-rate column (looked for 'rate_hz', 'event_rate_hz', 'rate'); "
            f"got {list(runset_df.columns)}"
        )
    if "pore_id" not in runset_df.columns:
        raise KeyError(
            "runset_df has no 'pore_id' column. Event rate must be shown per pore -- "
            "pore-to-pore spread is the error bar, so an unlabelled pool cannot be plotted honestly."
        )
    d = runset_df.dropna(subset=[x, rate_col]).copy()
    if d.empty:
        raise ValueError(f"no rows with both {x!r} and {rate_col!r} present")

    with nanoscope_style():
        fig, ax = _fig_of(ax, figure_size("single"))
        for i, (pore, g) in enumerate(d.groupby("pore_id", sort=True)):
            g = g.sort_values(x)
            ax.plot(g[x], g[rate_col], marker="o", ls="-", lw=0.9, ms=4,
                    color=color(i), alpha=0.9, label=str(pore))
        n_pores = int(d["pore_id"].nunique())
        if n_pores > 1:
            ax.legend(title=f"pore (n={n_pores})", loc="best", fontsize=6, title_fontsize=6)
        if "conc" in x.lower() and np.all(np.asarray(d[x], dtype=float) > 0):
            ax.set_xscale("log")
        ax.set_xlabel({"voltage_mV": "applied voltage (mV)"}.get(x, x))
        ax.set_ylabel("event rate (Hz)")
        ax.margins(x=0.08)

    _attach(fig, d[["pore_id", x, rate_col]].reset_index(drop=True))
    return fig, ax


def _minmax_envelope(y: np.ndarray, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-column min/max envelope of `y`, preserving temporal order.

    Returns (sample_indices, values), both length ~2*n_cols. Naive subsampling
    would drop the sub-millisecond spikes this figure exists to show; the
    envelope keeps every extremum, so a 20 us event survives at any zoom.
    """
    n = y.size
    block = max(1, n // max(1, n_cols))
    if block == 1:
        return np.arange(n), y
    n_full = n // block
    head = y[: n_full * block].reshape(n_full, block)
    starts = np.arange(n_full) * block
    lo = starts + head.argmin(axis=1)
    hi = starts + head.argmax(axis=1)
    idx = np.sort(np.stack([lo, hi], axis=1), axis=1).ravel()
    tail = y[n_full * block :]
    if tail.size:
        t0 = n_full * block
        idx = np.concatenate([idx, np.sort([t0 + tail.argmin(), t0 + tail.argmax()])])
    return idx, y[idx]


def _axis_clock(x_start: float, meta: Any) -> tuple[str, float | None]:
    """The x-axis label + the instrument-clock origin for the source-data CSV.

    THE SECOND HALF OF THE CLOCK DEFECT, and the one a derived origin cannot answer.
    Placement can be read off the axis (see :func:`trace_preview`), but WHICH CLOCK the
    axis's origin belongs to cannot: ``core.segment`` makes REBASED documents whose
    ``meta.t0_s`` is a zero *this app manufactured* while the cut's first sample was
    really recorded at ``rebased_from_t0_s`` (1155.7909 s on the owner's quick apo cut).
    An axis that starts at 0 is therefore indistinguishable, from inside this module,
    between a file that genuinely starts at 0 and a cut of a recording 19 minutes in —
    and a layer alignment nudge makes a NON-zero origin just as ambiguous.

    So there are exactly two honest answers, and which one this returns depends on
    whether the caller handed over the document's ``meta``:

    * ``meta`` given -- the full sentence :func:`nanoscope.core.clock.axis_label`
      writes, built from the same :class:`~nanoscope.core.clock.ClockOffset` that
      ``gui/trace_view.py`` passes it, so the exported figure's x-label is
      byte-identical to the on-screen trace axis for that document (a rebased cut
      included), and the CSV's ``t0_s`` is
      :func:`~nanoscope.core.clock.instrument_origin_s` -- the SAME number, computed by
      the SAME function, that ``measure.MEASUREMENT_SCHEMA``'s ``t0_s`` column carries
      for a row from this recording.
    * ``meta`` withheld -- the figure states the one thing it can prove (the origin it
      added to every event time) and says outright that the clock was not supplied. It
      does NOT fall back to naming a clock: before this, a derived origin of 0 took
      ``axis_label``'s "t₀ = 0, so both clocks read alike here" branch and printed that
      on a rebased cut, contradicting the on-screen axis about the same document.
      Silence would be the older bug; a guess is the worse one.
    """
    if meta is None:
        return (
            f"s (axis origin {clock.format_t0(x_start)} s, added to every event time; "
            f"which clock that origin is on was not supplied to this figure)",
            None,
        )
    # ``Session.x_axis`` builds ``arange(n)/fs + (meta.t0_s + layer.time_offset_s)``, so
    # ``x_start`` IS ``clock.clock_offset_s(meta, time_offset_s)`` numerically -- the
    # nudge included, which is right for a LABEL (it describes what is drawn). The rebase
    # origin is the half only ``meta`` carries, and it is what stops the false branch.
    offset = clock.ClockOffset(x_start, clock.rebased_origin_from_meta(meta))
    return (clock.axis_label("s", clock.INSTRUMENT, offset),
            clock.instrument_origin_s(meta))


def trace_preview(
    channel: Any,
    xaxis: Any,
    *,
    events: Any = None,
    baseline: np.ndarray | None = None,
    meta: Any = None,
    t0: float | None = None,
    t1: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Static current-trace figure with events shaded and the baseline overlaid.

    The trace is decimated with a **min/max envelope per pixel column**, not by
    subsampling: subsampling hides the very spikes this figure exists to show,
    and plotting 5M raw points takes minutes and yields a 200 MB PDF.

    Signal-first / data-model-adapted (v2 port): this takes the target
    :class:`~nanoscope.core.document.CalibratedChannel` plus the matching
    :class:`~nanoscope.core.document.XAxis` (from ``Session.x_axis``) instead of
    a legacy ``Recording``. The y-axis label is threaded from
    ``channel.units`` -- never hard-coded to pA (PORT_PLAN R9) -- and the x-axis
    honours ``xaxis.unit`` ("s" or "sample"). ``t0``/``t1`` are positions on the
    x-axis (seconds when ``xaxis.is_time``, sample indices otherwise).

    ONE CLOCK ON THIS FIGURE, AND IT IS THE AXIS'S OWN
    --------------------------------------------------
    The x-axis handed in is ``Session.x_axis`` -- ``arange(n)/fs + meta.t0_s``, the
    **instrument** clock (:mod:`nanoscope.core.clock`). Every ``t_start_s`` in an
    event table is on the **recording** clock (0 = the first LOADED sample). This
    function used to shade ``axvspan(t_start_s, ...)`` straight onto that axis, so on
    the owner's quick apoferritin ABF (``t0_s`` = 1154.99 s) all 111 bands landed on
    blank canvas ~1155 s from the events they annotate, and on a 60 s ``t0`` they
    landed on a stretch of open pore 60 s early -- worse than absent, and the figure
    still reported ``_nanoscope_n_events_shown`` as though every one had been drawn.
    ``gui/overlays.py`` never had this bug because it maps spans by SAMPLE INDEX.

    So the conversion happens HERE, once, against the origin read off the axis
    itself (``xvals[0]`` == ``meta.t0_s + time_offset_s`` by construction), and the
    default view window is that same origin. A ``t0`` default of ``0.0`` was itself
    part of the defect: a default that has to be overridden to be correct puts the
    clock in the caller's hands, and the caller (``MainWindow._export_figures``)
    passed nothing. Nothing about placement can now be got wrong from outside.

    The axis LABEL says which clock it is on -- but only ``meta`` can settle WHICH,
    because a rebased cut's axis origin of 0 is a zero this app made (see
    :func:`_axis_clock`). Hand ``meta`` over and the label is
    :func:`nanoscope.core.clock.axis_label`'s sentence, byte-identical to the on-screen
    trace axis for that same document; withhold it and the figure names its origin and
    says the clock was not supplied rather than guessing one.

    The source-data CSV carries BOTH origins, and they are different questions:
    ``x_axis_origin_s`` is what this figure added to every event time (always known),
    and ``t0_s`` is the recording's instrument-clock origin in
    ``measure.MEASUREMENT_SCHEMA``'s spelling and by its rule -- from
    :func:`~nanoscope.core.clock.instrument_origin_s`, so a segment carries its parent's
    clock and not its own rebased 0, and NaN when ``meta`` was not supplied ("NaN when
    unknown" is that schema's own convention).

    Args:
        channel: a CalibratedChannel (``.data``, ``.units``, ``.info``).
        xaxis: the matching XAxis (``.values`` full-length float64, ``.unit``,
            ``.is_time``).
        events: EventSet or DataFrame with 't_start_s' and 'dwell_s'; shaded at
            ``t_start_s + <axis origin>``, i.e. converted onto the axis's clock.
            Only meaningful on a time axis: on a sample axis the event seconds
            cannot be placed at all, and this warns and shades nothing rather than
            painting them at a number that means something else.
        baseline: full-length baseline array (as from
            ``analysis.baseline.estimate_baseline``).
        meta: the ``RecordingMeta`` of the document this channel/axis came from
            (``Session.document(doc_id).meta``). Optional, and it changes NOTHING about
            what is drawn -- placement is derived from the axis and cannot be got wrong
            from outside. It is what lets the figure NAME the clock its seconds are on:
            without it a rebased cut and a genuine t₀ = 0 file are the same axis. See
            :func:`_axis_clock`.
        t0, t1: view window ON THE X-AXIS'S OWN CLOCK; ``None`` (the default for
            both) means the axis's own extent -- its first and last value.
        ax: Axes to draw into.

    Returns:
        (fig, ax).
    """
    current = np.asarray(channel.data)
    xvals = np.asarray(xaxis.values, dtype=float)
    if xvals.size != current.size:
        raise ValueError(
            f"xaxis length {xvals.size} != channel length {current.size}; "
            "pass the XAxis for this exact document/channel"
        )
    units = getattr(channel, "units", "") or "pA"
    x_is_time = bool(getattr(xaxis, "is_time", True))

    # THE ORIGIN OF THE AXIS WE WERE HANDED. On a time axis this is
    # ``meta.t0_s + layer.time_offset_s`` exactly (Session.x_axis builds
    # ``arange(n)/fs + that``), which is `clock.clock_offset_s` for this document --
    # derived from the axis rather than asked for, because a caller cannot pass what
    # it does not know it owes.
    x_start = float(xvals[0]) if xvals.size else 0.0
    t_end = float(xvals[-1]) if xvals.size else 0.0

    # A meta from a DIFFERENT document would describe this axis's clock confidently and
    # wrongly -- the one failure mode the optional argument introduces, so it is checked
    # rather than trusted. ``Session.x_axis`` is exactly ``meta.n_samples`` long.
    if meta is not None:
        n_meta = getattr(meta, "n_samples", None)
        if n_meta is not None and int(n_meta) != int(xvals.size):
            raise ValueError(
                f"meta describes {int(n_meta)} samples but this XAxis has {xvals.size}; "
                "pass the meta of the document this axis and channel came from "
                "(Session.document(doc_id).meta) -- a meta from another document would "
                "name the wrong clock on this figure"
            )
    t0 = x_start if t0 is None else float(t0)
    t1 = t_end if t1 is None else float(t1)
    if t1 <= t0:
        raise ValueError(f"empty window: t0={t0} t1={t1}")
    # Positions on the x-axis -> sample indices (the axis is monotone increasing).
    i0 = int(np.searchsorted(xvals, t0, side="left"))
    i1 = int(np.searchsorted(xvals, t1, side="right"))
    i0 = max(0, i0)
    i1 = min(current.size, i1)
    if i1 <= i0:
        raise ValueError(f"window [{t0}, {t1}) contains no samples")
    seg = current[i0:i1]

    with nanoscope_style():
        fig, ax = _fig_of(ax, figure_size("double", aspect=0.28))
        n_cols = 2000
        idx, vals = _minmax_envelope(seg, n_cols)
        t = xvals[i0 + idx]
        ax.plot(t, vals, color=PALETTE["event"], lw=0.6, zorder=2,
                label=f"current (min/max envelope, {seg.size:,} samples)")

        if baseline is not None:
            b = np.asarray(baseline, dtype=float)
            if b.size != current.size:
                raise ValueError(
                    f"baseline length {b.size} != channel length {current.size}; "
                    "estimate_baseline returns a full-length array"
                )
            step = max(1, (i1 - i0) // n_cols)
            bt = xvals[i0:i1:step]
            ax.plot(bt, b[i0:i1:step], color=PALETTE["baseline"], lw=1.0, ls="--",
                    zorder=3, label="baseline")

        n_shown = 0
        # The detection's baseline advisory travels with its events (S7-JRN3 part 2):
        # this figure draws the very baseline the advisory is about.
        eprov: dict = {}
        trace_details: list[str] = []
        if events is not None:
            edf, eprov = _unpack(events)
            if len(edf):
                for c in ("t_start_s", "dwell_s"):
                    if c not in edf.columns:
                        raise KeyError(f"events frame has no column {c!r}")
                if not x_is_time:
                    # Samples are on neither clock, so there is no number on this axis
                    # that ``t_start_s`` means. Shading it anyway (which is what this
                    # did) paints a band at "0.25" on an axis where 0.25 is the 0.25th
                    # SAMPLE. Nothing is drawn, the count says zero, and the reason is
                    # said out loud instead of being left for the reader to notice.
                    note = (
                        "trace_preview: events were not shaded because this XAxis is in "
                        "samples, not seconds — an event's t_start_s is on the recording "
                        "CLOCK and cannot be placed on a sample axis. Pass a time axis "
                        "(Session.x_axis on a document with a known sample rate)."
                    )
                    trace_details.append(note)
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
                else:
                    # THE CONVERSION. ``t_start_s`` is recording-clock (0 = first loaded
                    # sample); this axis is instrument-clock. This is
                    # ``clock.to_instrument_s(x_start, t_start_s)`` -- that function's
                    # arithmetic, over a whole column instead of one scalar.
                    s = np.asarray(edf["t_start_s"], dtype=float) + x_start
                    e = s + np.asarray(edf["dwell_s"], dtype=float)
                    vis = (e > t0) & (s < t1)
                    n_shown = int(vis.sum())
                    for a, b_ in zip(s[vis], e[vis]):
                        ax.axvspan(a, b_, color=PALETTE["accent"], alpha=0.25, lw=0,
                                   zorder=1)

        ax.set_xlim(t0, t1)
        # ...and the axis SAYS what it is on: core.clock's own sentence when the caller
        # handed over the document (then this reads exactly like the trace the user
        # curated on, rebased cuts included), and an explicit "the clock was not
        # supplied" when it did not. Never a guess -- see _axis_clock.
        if x_is_time:
            x_label, instrument_origin_s = _axis_clock(x_start, meta)
        else:
            # Samples are on neither clock; annotating them would be a second falsehood.
            x_label, instrument_origin_s = "sample", None
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"current ({units})")
        title = getattr(getattr(channel, "info", None), "name", "") or ""
        if title:
            ax.set_title(f"{title}", loc="left")
        ax.legend(loc="lower right", fontsize=6)
        # The source's baseline advisories are attached as details, never drawn.

    source = {"x": t, "current": vals}
    if x_is_time:
        # The numbers behind the figure travel with their clock. TWO origins, because
        # they answer two questions and are the same number only on an un-nudged,
        # un-rebased recording:
        #   x_axis_origin_s -- what this figure ADDED to every event time to place it,
        #       i.e. ``x - x_axis_origin_s`` is the clock the events CSV is on. Always
        #       known: it is read off the axis that was drawn.
        #   t0_s -- where this recording's first sample sits in the EXPERIMENT, spelled
        #       and computed exactly as ``measure.MEASUREMENT_SCHEMA``'s column is
        #       (``clock.instrument_origin_s``: a segment carries its parent's clock,
        #       a layer nudge is a display shift and is never included), so a row in
        #       this CSV and a row in measurements.csv from the same folder mean the
        #       same thing by ``t0_s``. NaN when ``meta`` was withheld -- that schema's
        #       own "NaN when unknown", never a 0 that would read as a real origin.
        # Both omitted on a sample axis, where there is no clock to name at all.
        shape = np.asarray(t).shape
        source["x_axis_origin_s"] = np.full(shape, x_start, dtype=float)
        source["t0_s"] = np.full(
            shape, np.nan if instrument_origin_s is None else float(instrument_origin_s),
            dtype=float)
    _attach(fig, pd.DataFrame(source))
    _attach_details(fig, "trace_preview", eprov, notes=trace_details, n_events_shown=n_shown,
                    x_axis_origin_s=x_start if x_is_time else None,
                    instrument_origin_s=instrument_origin_s)
    setattr(fig, "_nanoscope_n_events_shown", n_shown)
    # The origin every band on this figure was placed through — None when the axis is
    # samples and no shading was possible. A caller (or a test) can read back WHICH
    # clock the figure is on without re-deriving it.
    setattr(fig, "_nanoscope_clock_origin_s", x_start if x_is_time else None)
    # ...and where that origin sits in the EXPERIMENT, when the caller said. None means
    # "not supplied", which is why it is a separate attribute from the one above rather
    # than the same number twice: on a rebased cut they differ by 1155.79 s.
    setattr(fig, "_nanoscope_instrument_origin_s", instrument_origin_s)
    return fig, ax


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------
def _harvest_lines(fig: Figure) -> pd.DataFrame:
    """Last-resort source data: whatever is in the figure's Line2D artists."""
    rows = []
    for ai, ax in enumerate(fig.axes):
        for li, art in enumerate(ax.lines):
            if not isinstance(art, Line2D):
                continue
            xd, yd = np.asarray(art.get_xdata(), dtype=float), np.asarray(art.get_ydata(), dtype=float)
            if xd.size == 0 or xd.size != yd.size:
                continue
            label = art.get_label()
            if label.startswith("_"):
                label = f"line{li}"
            rows.append(pd.DataFrame({"axes": ai, "series": label, "x": xd, "y": yd}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


#: Every extension :func:`save_figure` may write for one figure. Declared ONCE and used
#: by both the writer and :func:`figure_output_paths`, so the "what will this overwrite"
#: question can never answer differently from what the write actually does.
FIGURE_OUTPUT_EXTENSIONS: tuple[str, ...] = ("pdf", "png", "csv")


def figure_output_paths(path: str | Path) -> tuple[Path, ...]:
    """Every file :func:`save_figure` may write for ``path`` — NOT just ``path`` itself.

    ``save_figure`` ignores the extension it is handed and writes a whole family of
    siblings: ``<stem>.pdf``, ``<stem>.png`` and ``<stem>.csv``. A caller that confirms
    an overwrite of the single name the user typed is therefore confirming a file that
    is frequently never written, while silently clobbering up to three that are — the
    ``.csv`` (the source data a user may well have hand-edited) most damagingly.

    Callers ask THIS function what is at stake. The ``.csv`` is included even though
    ``save_figure`` skips it for an empty frame: an overwrite prompt must name every
    path the write MAY touch, and over-naming is a false alarm while under-naming is
    data loss.
    """
    stem = Path(path).with_suffix("")
    return tuple(stem.with_suffix("." + ext) for ext in FIGURE_OUTPUT_EXTENSIONS)


def save_figure(fig: Figure, path: str | Path, *, data: Any = None,
                group: StagedSet | None = None) -> dict[str, Path]:
    """Save PDF + PNG + source CSV, keeping warning/provenance text out of artwork.

    :func:`figure_details` is embedded as JSON in PDF Subject and PNG Description.
    This preserves the existing three-file overwrite/atomic-write contract and the
    numerical CSV unchanged while carrying the full scientific context with both
    image formats. The UI can also consume that detached details record directly.

    Writes the family of siblings enumerated by :func:`figure_output_paths`; a GUI
    caller must confirm THAT set, not the single path it was handed.

    Researchers always need to restyle a figure later, usually in another tool,
    usually the week of submission. Exporting the numbers behind every figure is
    the single kindest thing this app can do -- and it is what makes the figure
    checkable by a reviewer.

    THE FAMILY LANDS TOGETHER OR NOT AT ALL. Injecting ENOSPC on the PNG left
    ``figures_depth.pdf`` from today beside ``figures_depth.png`` and
    ``figures_depth.csv`` from a previous export, under one name, with nothing on disk
    saying so -- a figure and the table of numbers behind it that disagree, which is worse
    than either being missing. See :class:`nanoscope.core.atomic_write.StagedSet`.

    Args:
        fig: figure to save.
        path: output path; any extension is replaced (.pdf/.png/.csv siblings).
        data: DataFrame / EventSet / dict of arrays to export. If None, the data
            the figure function attached is used, falling back to the plotted
            line data.
        group: an OPEN :class:`~nanoscope.core.atomic_write.StagedSet` to stage into
            instead of committing on our own. A caller exporting several figures under
            one stem passes one group for all of them, so the whole *export* -- not
            merely each family -- is all-or-nothing; it commits the group itself. With
            ``None`` (the default, and every non-GUI caller) this function stages and
            commits its own three files.

    Returns:
        {'pdf': ..., 'png': ..., 'csv': ...} -- the paths the figure family occupies
        ('csv' absent if there was genuinely nothing to export). With a ``group`` these
        are the paths the files WILL occupy when the caller commits.
    """
    p = Path(path)
    stem = p.with_suffix("")
    # THE FOLDER IS MADE BY ``stage``, NOT HERE (EXP3-03). This function used to run
    # ``stem.parent.mkdir(parents=True, exist_ok=True)`` at exactly this point -- before a
    # single member was staged. :meth:`StagedSet.stage` creates the parent too, and records
    # the levels it had to create in ``_made_dirs`` so :meth:`StagedSet.abandon` can undo
    # exactly them ("'Nothing on disk changes' is the promise; an empty folder named after
    # an export that never happened is a change, and a confusing one -- the user goes
    # looking in it for the twelve files the status line said were not written"). Making
    # the tree first meant ``stage`` walked upward and found nothing missing, ``_made_dirs``
    # stayed empty, and an export into a new ``paper/figures/v3`` that then failed on every
    # figure left all three levels behind -- under a dialog that says "Nothing was written
    # -- every file already in the folder is exactly as it was". The guard was never broken;
    # it simply never saw the figure export. Removing this line connects the two.
    # The declaration is CHECKED against the write, not trusted: a caller has already used it
    # to decide what the user consented to overwrite, so the two drifting apart is data loss.
    declared = set(figure_output_paths(p))

    if data is None:
        data = getattr(fig, _DATA_ATTR, None)
    if data is None:
        data = _harvest_lines(fig)
    df = getattr(data, "df", data)
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"data must be a DataFrame, EventSet or dict of arrays, got {type(data).__name__}")

    out: dict[str, Path] = {}
    owned = StagedSet() if group is None else group
    mine: list[Path] = []    # the members THIS call staged — all this call may touch
    try:
        for ext in ("pdf", "png"):
            f = stem.with_suffix("." + ext)
            assert f in declared
            mine.append(f)
            metadata = {"Subject" if ext == "pdf" else "Description": _details_json(fig)}
            fig.savefig(owned.stage(f), dpi=_EXPORT_DPI, bbox_inches="tight", metadata=metadata)
            out[ext] = f
        if len(df):
            csv = stem.with_suffix(".csv")
            assert csv in declared
            mine.append(csv)
            df.to_csv(owned.stage(csv), index=False)
            out["csv"] = csv
    except BaseException:
        if group is None:
            owned.abandon()
        else:
            # A SHARED group holds OTHER figures' staged families. ``stage`` runs before
            # the write, so a failure INSIDE the write (``fig.savefig`` raises at draw
            # time — e.g. invalid mathtext in a title carried from the data file) leaves
            # this family staged-but-never-written; commit would then refuse over the
            # orphan and sink every healthy figure with it, misreported as a placement
            # failure. This figure's whole family is withdrawn — never a partial one,
            # which would commit a PDF whose PNG and CSV are a previous export's — and
            # only its own members: the rest of the export is the caller's to finish.
            owned.unstage(mine)
        raise
    if group is None:
        owned.commit()

    if not len(df):
        warnings.warn(
            f"no source data to export next to {stem.name}; the figure was saved without a .csv",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


# ==========================================================================
# The interactive panels' "Save image…" -- one staged write, one wording
# ==========================================================================
#
# WHY THIS LIVES HERE, BESIDE ``save_figure``. Five panels (Plots, PSD, I-V,
# voltage-dependence, event-plot) each export the plot the user is looking at
# through pyqtgraph's ``ImageExporter``. All five had written the user's chosen
# path DIRECTLY -- ``exporter.export(str(path))`` -- and all five reported every
# failure with one copied sentence, "The file was not created". Two harms came
# out of that pair (EXP3-06):
#
#   * ``QImage.save`` opens the TARGET and truncates it. A format it cannot
#     write (a ``.pdf``/``.svg``/``.eps`` name reached through the dialog's
#     "All files (*)" filter, over a PDF an earlier *Export figure…* left there)
#     left the user's file at zero bytes -- and then said it "was not created".
#   * Even where nothing was destroyed (a target the user had chmod-ed 444),
#     "The file was not created" is simply false: the file of that name IS in
#     the folder. It is the PREVIOUS export, and the researcher who goes and
#     looks finds the right name with a plausible timestamp and takes it for
#     the figure they just saved.
#
# ``docs/USER_GUIDE.md`` §12 already promises these five the opposite: "Every
# export replaces its target the same way a project save does: staged beside it
# and moved into place, so a failure leaves the previous file whole rather than
# truncated." So the fix is not a better sentence on top of an unstaged write --
# it is the staged write that MAKES the better sentence true. It is written once,
# here in the plot layer next to ``save_figure`` (which does the same job for
# matplotlib), rather than five times: one copied sentence in five files is how
# this defect got to be five defects.


class ImageNotRendered(RuntimeError):
    """The renderer produced no image, and nothing on disk was touched.

    Raised by :func:`save_plot_image` when the render call reports failure --
    ``ImageExporter.export`` ends in ``QImage.save``, which signals failure by
    RETURNING ``False`` and raising nothing -- or when it returns happily having
    written no bytes. Distinct from an :class:`OSError`, because the two need
    different advice: this one is the app's own inability to draw that format,
    not anything about the folder.
    """


def save_plot_image(target: str | Path, render) -> list[str]:
    """Stage ``render``'s image beside ``target`` and move it into place.

    Args:
        target: the file the user chose. Untouched unless the render succeeds.
        render: called with the STAGED path (a sibling of ``target``, keeping its
            suffix so the renderer still picks the format from the extension).
            Anything it returns other than ``False`` counts as "it says it wrote".

    Returns:
        The file-identity sentences for this write, drained from the
        ``core.atomic_write`` ledger and scoped to ``target`` alone -- a symlink
        that was followed to somewhere else, hard-linked names now holding the
        previous contents. Normally empty. THIS IS THE HALF THE CALLER MUST NOT
        DROP (EXP3-02): the disclosure is not obtainable anywhere afterwards,
        because an atomic replace leaves no trace of the link it broke, and
        ``docs/USER_GUIDE.md`` §11 promises it for "the project, the calibration
        sidecar, exported events, exported figures" without scoping it to the
        project save. The ledger is drained for ``target`` and no other file, so
        a panel never prints a sentence about a write it did not make.

    Raises:
        ImageNotRendered: the renderer declined; ``target`` is exactly as it was.
        OSError: staging or the replace failed. Already re-addressed to ``target``
            by ``atomic_write`` rather than naming the hidden staging sibling.
    """
    from ..core import atomic_write

    out = Path(target)
    with atomic_write.staged_path(out) as staged:
        ok = render(staged)
        # Both halves matter. ``ok is False`` is QImage's own "I could not write
        # this"; the size check catches a renderer that returns something truthy
        # having produced nothing, which ``staged_path`` would otherwise turn into
        # a bare FileNotFoundError about a path the user never saw.
        wrote = staged.is_file() and staged.stat().st_size > 0
        if ok is False or not wrote:
            # Raising here abandons the staging: the temporary is discarded and
            # ``target`` is never replaced. That is the whole point -- the failure
            # now costs the user nothing, so the message may say so truthfully.
            raise ImageNotRendered(
                f"{out.name} could not be rendered to an image in this format")
    return atomic_write.take_identity_notes(out)


def image_write_failure_text(target: str | Path, exc: BaseException,
                             *, formats: Sequence[str] = (".png",)) -> str:
    """The dialog body for a failed *Save image…* -- said once, for all five panels.

    Two things have to be true of it, and were not:

    * it may not claim the name is absent when a file of that name is there. The
      staged write above guarantees the stronger statement -- that file is the
      EARLIER export, byte-for-byte as it was, and it is now OLDER than the figure
      on screen -- so say exactly that, with the date it carries, the way
      ``MainWindow._report_unreplaced_figures`` dates the figures it could not
      replace;
    * where the name really is absent the old sentence was right, and it stays.

    Args:
        formats: the extensions THIS panel can actually render, named in the advice.
            Callers pass the formats offered by their panel. Analysis panels now share
            PNG, SVG and PDF export; the single-format default remains conservative for
            older callers that have not declared broader support.

            One shared sentence with a per-caller list, rather than a second sentence
            in one panel: five copies of one paragraph is how the defect this function
            was written for got to be five defects (EXP3-06).
    """
    out = Path(target)
    if isinstance(exc, ImageNotRendered):
        names = list(formats) or [".png"]
        if len(names) == 1:
            offer = f"a {names[0]} name"
        else:
            offer = f"a {', '.join(names[:-1])} or {names[-1]} name"
        head = (f"Could not write the image:\n{out}\n\nThe plot could not be rendered "
                f"into a file of this type — try {offer}.")
    else:
        # An OSError from the staging seam already names the file the user chose and
        # says what to do about it ("… is marked read-only and was not changed…"),
        # so it is shown as it is rather than paraphrased.
        head = f"Could not write the image:\n{exc}"
    try:
        when = time.strftime("%d %b %Y %H:%M", time.localtime(out.stat().st_mtime))
    except OSError:
        # Nothing of that name is there: the original sentence, which was only ever
        # wrong for the other branch.
        return (f"{head}\n\nThe file was not created — check that the folder still "
                f"exists and that you can write to it (permissions, read-only "
                f"location, disk space), then try again.")
    return (f"{head}\n\nNothing was written, so the {out.name} already in that folder "
            f"is untouched — but it is the EARLIER export (last written {when}), not "
            f"this one, and it is now older than the figure on screen. Do not take it "
            f"for the image you just saved.")


# ==========================================================================
# Joint KDE density -- several molecules on one pair of axes.
# ==========================================================================
def joint_density(
    groups: "dict[str, Any]",
    *,
    xcol: str = "dwell_s",
    ycol: str = "depth_pA",
    x_scale: float = 1e3,
    xlabel: str = "Dwell Time (ms)",
    ylabel: str = "Amplitude (pA)",
    log_x: bool = True,
    levels: int = 8,
    title: str | None = None,
    gridsize: int = 160,
    line_width: float = 0.7,
    legend_loc: str = "upper right",
    legend_fontsize: float = 6.0,
    legend_title: str | None = "molecule",
    show_counts: bool = True,
    font_scale: float = 1.0,
) -> "tuple[Figure, Axes]":
    """Overlaid 2-D KDE contours per group, with marginal KDEs on both axes.

    The comparison figure of a host-guest experiment: host, guest and the complex on ONE
    pair of axes, so the reader sees that the pair occupies a region neither component
    reaches. A scatter cannot show that -- three clouds of thousands of points overplot
    into mud, and whichever group is drawn last appears dominant. Contours make the
    populations comparable at a glance and independent of draw order.

    KDE on LOG DWELL when ``log_x``, not on the raw seconds. Dwell spans decades, so a
    kernel of fixed width in linear time is far too wide at 0.1 ms and far too narrow at
    10 ms -- the same reason the dwell histogram bins logarithmically. The density is
    estimated in log10 space and the axis is labelled in the display unit, so the contours
    mean what they look like.

    Each group is normalised to ITSELF: contour levels are fractions of that group's own
    peak density, so an n=200 population and an n=17000 one are the same SHAPE side by
    side rather than one being invisible. Counts belong in the legend, and are put there.

    Args:
        groups: ``{label: EventSet | DataFrame}``, drawn in insertion order.
        xcol, ycol: columns to cross. Defaults are the canonical dwell-vs-depth pair.
        x_scale: multiply x by this for display (1e3 -> seconds to ms).
        log_x: estimate and plot x on a log axis.
        levels: number of contour lines per group.
        gridsize: evaluation grid per axis; 160 is smooth without being slow.
        line_width: contour line width.
        legend_loc: any Matplotlib location, ``"outside right"`` to lift it clear of the
            data entirely, or ``"none"`` to drop it.
        legend_fontsize: point size of the legend text.
        legend_title: heading above the entries; ``None`` removes it.
        show_counts: append ``(n = ...)`` to each entry. The counts are the longest part
            of a label and a legend is sized by its widest entry, so turning them off is
            the single biggest thing that shrinks it -- the numbers are still in the
            exported source CSV either way.
        font_scale: multiplies axis label and tick sizes.

    A legend is an INDEX, not content. Overlaying three recordings puts it on top of the
    very contours it names, and the defaults here are deliberately small for that reason;
    ``legend_loc="outside right"`` moves it off the data completely when the populations
    fill the axes.

    Returns:
        ``(fig, ax)`` with the joint axes; marginals are attached around it.
    """
    from scipy.stats import gaussian_kde

    prepared: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for i, (label, src) in enumerate(groups.items()):
        df, _prov = _unpack(src)
        x = _column(df, xcol) * float(x_scale)
        y = _column(df, ycol)
        ok = np.isfinite(x) & np.isfinite(y)
        if log_x:
            ok &= x > 0
        x, y = x[ok], y[ok]
        if x.size >= 5:      # a KDE of four points is a decoration, not an estimate
            prepared.append((label, x, y, OKABE_ITO[i % len(OKABE_ITO)]))
    if not prepared:
        raise ValueError("no group has enough finite points to estimate a density")

    with nanoscope_style():
        fig, ax = _fig_of(None, figure_size("single", aspect=1.05))
        fig.set_layout_engine("none")
        div = make_axes_locatable(ax)
        ax_top = div.append_axes("top", size="18%", pad=0.06, sharex=ax)
        ax_right = div.append_axes("right", size="18%", pad=0.06, sharey=ax)

        xs = np.concatenate([np.log10(x) if log_x else x for _l, x, _y, _c in prepared])
        ys = np.concatenate([y for _l, _x, y, _c in prepared])
        xpad = 0.05 * (xs.max() - xs.min() or 1.0)
        ypad = 0.05 * (ys.max() - ys.min() or 1.0)
        gx = np.linspace(xs.min() - xpad, xs.max() + xpad, gridsize)
        gy = np.linspace(ys.min() - ypad, ys.max() + ypad, gridsize)
        MX, MY = np.meshgrid(gx, gy)
        flat = np.vstack([MX.ravel(), MY.ravel()])

        handles = []
        for label, x, y, colour in prepared:
            u = np.log10(x) if log_x else x
            try:
                dens = gaussian_kde(np.vstack([u, y]))(flat).reshape(MX.shape)
            except (ValueError, np.linalg.LinAlgError):
                continue          # degenerate group (all one value) -- skip, keep the rest
            peak = float(dens.max())
            if not (peak > 0):
                continue
            # Levels as fractions of THIS group's peak: shapes stay comparable across
            # groups of wildly different n.
            cs = np.linspace(0.1, 0.95, levels) * peak
            ax.contour(10 ** MX if log_x else MX, MY, dens, levels=cs,
                       colors=colour, linewidths=line_width)
            handles.append(Line2D(
                [], [], color=colour, lw=max(1.0, line_width * 1.6),
                label=f"{label}  (n = {x.size:,})" if show_counts else str(label)))
            # Marginals, each on its own scale for the same reason.
            for axis, vals, horiz in ((ax_top, u, False), (ax_right, y, True)):
                try:
                    k = gaussian_kde(vals)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                grid = gx if not horiz else gy
                d = k(grid)
                if horiz:
                    axis.plot(d, grid, color=colour, lw=0.9)
                else:
                    axis.plot(10 ** grid if log_x else grid, d, color=colour, lw=0.9)

        if log_x:
            ax.set_xscale("log")
            ax_top.set_xscale("log")
        base = float(plt.rcParams.get("font.size", 8.0))
        ax.set_xlabel(xlabel, fontsize=base * font_scale)
        ax.set_ylabel(ylabel, fontsize=base * font_scale)
        ax.tick_params(labelsize=base * font_scale * 0.9)
        for axis in (ax_top, ax_right):
            axis.set_axis_off()
        if title:
            ax_top.set_title(title, fontsize=base * font_scale * 1.15)
        if handles and str(legend_loc).lower() != "none":
            kw: dict[str, Any] = dict(handles=handles, fontsize=legend_fontsize,
                                      framealpha=0.9)
            if legend_title:
                kw.update(title=legend_title, title_fontsize=legend_fontsize)
            if str(legend_loc) == "outside right":
                kw.update(loc="upper left", bbox_to_anchor=(1.24, 1.0), borderaxespad=0.0)
            else:
                kw.update(loc=legend_loc)
            ax.legend(**kw)

        data = pd.concat(
            [pd.DataFrame({"group": l, xcol: x, ycol: y}) for l, x, y, _c in prepared],
            ignore_index=True)
        _attach(fig, data)
    return fig, ax


# ==========================================================================
# The interactive panels' VECTOR export -- PDF, and where SVG comes from
# ==========================================================================
#
# WHY THIS IS HERE AND NOT IN ``plot/pg_svg.py``. That module is the *workaround*
# for a pyqtgraph defect and is meant to disappear when pyqtgraph fixes it; this is
# a format the app offers on its own account. It sits beside ``save_plot_image``
# for the same reason that function does: the five pyqtgraph panels share one
# export seam, and a second copy of it in a panel is how the last export defect got
# to be five defects.
#
# WHY PDF AT ALL, WHEN SVG NOW WORKS. Measured this session on the Plots window
# (640x480 canvas, the same three pictures rendered three ways):
#
#     picture                     PDF              SVG              PNG
#     scatter, n = 10 000     0.151 MB  0.10 s   5.282 MB  3.06 s   0.083 MB
#     KDE, 8 contours          0.042 MB  0.03 s   0.085 MB  0.02 s   0.077 MB
#     histogram, 30 bins       0.023 MB  0.01 s   0.028 MB  0.01 s   0.026 MB
#
# The scatter row is the whole argument. pyqtgraph's SVG exporter emits one
# ``<path>`` per symbol (``pg_svg`` limitation (b)), so a real event census is a
# 5 MB file that vector editors are slow to open; Qt's PDF backend compresses its
# content stream and writes the same picture 35x smaller and 30x faster, still as
# vector art. So the honest answer to "SVG from here, PDF from the publication
# figures" turned out to be neither: PDF from here TOO, and it is the better of the
# two vector formats for exactly the plot type this app produces most.
#
# THIS IS NOT A REPLACEMENT FOR ``save_figure``. The six publication figures are
# matplotlib and keep their own PDF+PNG+CSV family with the typography
# ``plot.style`` gives them. This renders the LIVE view -- the thing on screen,
# with the styling the user tuned in the Style group.

def _paint_items(item) -> list:
    """Every QGraphicsItem at or below *item*, itself included.

    Only used to hand ``setExportMode`` to the items that implement it, so the
    painting ORDER pyqtgraph's own ``Exporter.getPaintItems`` is careful about does
    not matter here -- the scene still paints itself, in its own order.
    """
    out: list = []
    stack = [item]
    while stack:
        node = stack.pop()
        out.append(node)
        children = getattr(node, "childItems", None)
        if children is not None:
            stack.extend(children())
    return out


def export_plot_pdf(item, path: str | Path, *, width: float = 0, height: float = 0) -> Path:
    """Render a live pyqtgraph item to a single-page vector PDF.

    Args:
        item: the ``PlotItem`` on screen.
        path: destination. Pass the STAGED path when going through
            :func:`save_plot_image`, exactly as the PNG and SVG exports do.
        width, height: page size in POINTS (1 pt = 1 px of the on-screen canvas, so
            the panel's *Export size* boxes mean the same thing here as they do for
            the PNG). ``0`` on both means the canvas's own size; ``0`` on one means
            "follow the aspect ratio", which is the rule ``ImageExporter`` already
            uses for its width/height pair.

    Returns:
        ``Path(path)``.

    Raises:
        ImageNotRendered: the item is not in a scene, or has no area to draw --
            the same refusal a PNG of the same item would produce, so
            :func:`image_write_failure_text` can speak for both.

    THREE THINGS THIS COPIES FROM ``ImageExporter`` ON PURPOSE, because a vector
    file that does not match the raster one of the same picture is a trap:

    * THE BACKGROUND. ``QGraphicsScene.render`` paints items and nothing else -- the
      canvas colour lives on the VIEW. Without the fill below, a PDF of the app's
      dark-themed plot came out with white paper behind light-grey tick labels
      (measured: ink coverage 0.155 against the PNG's 0.231, the whole difference
      being the missing ground). The brush is taken from the same
      ``views()[0].backgroundBrush()`` ``ImageExporter`` reads, so "what the PNG
      would have" and "what the PDF has" cannot drift apart.
    * ``setExportMode``. pyqtgraph items change how they draw for an export
      (``AxisItem`` rescales its pen by ``resolutionScale``, scatter symbols switch
      off the pixel-cache). It is turned off again in a ``finally``: an item left in
      export mode would keep drawing that way on screen.
    * ``scene.render``, NOT ``item.paint``. pyqtgraph's ``GraphicsScene.render``
      begins with ``prepareForPaint()``, which flushes each ViewBox's deferred
      ``updateAutoRange``/``updateMatrix``. That is precisely the pass
      ``SVGExporter`` skips and ``pg_svg.refresh_for_export`` has to put back
      (limitation (a): a never-shown KDE exported at ~3.1e6 px on a 640 px canvas).
      Going through the scene means PDF never had that bug and cannot acquire it.
    """
    from PySide6 import QtCore, QtGui

    out = Path(path)
    scene = item.scene() if hasattr(item, "scene") else None
    if scene is None:
        raise ImageNotRendered(
            f"{out.name} could not be rendered: the plot is not on a canvas yet")
    source = item.sceneBoundingRect()
    if source.width() <= 0 or source.height() <= 0:
        raise ImageNotRendered(
            f"{out.name} could not be rendered: the plot has no area to draw")

    w, h = float(width), float(height)
    if w > 0 and h <= 0:
        h = w * source.height() / source.width()
    elif h > 0 and w <= 0:
        w = h * source.width() / source.height()
    elif w <= 0:
        w, h = source.width(), source.height()

    writer = QtGui.QPdfWriter(str(out))
    # 72 dpi makes one PDF point one scene pixel, so the page is the canvas at 1:1
    # and a width typed into *Export size* lands as that many points. Vector output
    # has no resolution of its own; this only fixes the unit.
    writer.setResolution(72)
    writer.setPageSize(QtGui.QPageSize(QtCore.QSizeF(w, h), QtGui.QPageSize.Point,
                                       "plot", QtGui.QPageSize.ExactMatch))
    writer.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))

    background = None
    views = scene.views()
    if views:
        brush = views[0].backgroundBrush()
        if brush.style() != QtCore.Qt.BrushStyle.NoBrush:
            background = brush

    painter = QtGui.QPainter(writer)
    if not painter.isActive():
        raise ImageNotRendered(
            f"{out.name} could not be opened for writing as a PDF")
    target = QtCore.QRectF(0, 0, writer.width(), writer.height())
    opts = {"antialias": True, "painter": painter, "resolutionScale": 1.0}
    if background is not None:
        opts["background"] = background.color()
    touched = [n for n in _paint_items(item) if hasattr(n, "setExportMode")]
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        if background is not None:
            painter.fillRect(target, background)
        for node in touched:
            node.setExportMode(True, opts)
        scene.render(painter, target, source)
    finally:
        for node in touched:
            node.setExportMode(False)
        painter.end()
    return out


# ==========================================================================
# The interactive panels' SOURCE DATA -- the numbers behind the live plot
# ==========================================================================
#
# ``save_figure`` has exported the numbers behind every publication figure since it
# was written, for the reason stated there: "Researchers always need to restyle a
# figure later, usually in another tool, usually the week of submission." Nothing on
# any INTERACTIVE panel did -- and the interactive panel is where the user actually
# decides what the figure is. A PNG of a histogram whose bin edges exist only inside
# a running Qt process is a picture nobody can re-plot, re-bin or check.
#
# THE SHAPE FOLLOWS ``joint_density``'s CSV, which is the precedent for a figure with
# SEVERAL populations on one pair of axes: LONG form, one row per drawn value, with a
# label column naming which series the row belongs to (``group`` there, ``series``
# here) rather than one column per source. Wide form cannot survive two sources with
# different point counts, and the panel's whole purpose is overlays.
#
# ...and it follows ``dwell_histogram``'s CSV for the columns of a binned figure --
# ``bin_left``/``bin_right``/centre/height, so a reader can re-draw the bars exactly
# rather than guess the edges back out of the centres.
#
# PROVENANCE TRAVELS IN THE TABLE, NOT IN A COMMENT HEADER. That is this module's own
# convention (``trace_preview`` writes ``x_axis_origin_s`` and ``t0_s`` as constant
# columns beside x and current, so the clock cannot be separated from the numbers it
# dates), and it is what keeps the file a plain ``pd.read_csv`` away from being
# usable -- no ``comment=``, no ``skiprows``, no hand-editing before it opens in
# Excel. Here the provenance is carried by the COLUMN NAMES: the x and y columns are
# named for the event-table columns that were plotted, so a scatter of amplitude
# against dwell writes ``dwell_s,amplitude`` and never ``x,y``.
#
# EVERY NUMBER IS IN DATA UNITS -- seconds, picoamps -- never the display unit the
# axis happens to be showing, and never log10 of anything. The panel draws a dwell
# axis in ms via a tick multiplier and hands its scatter points to pyqtgraph already
# log10'd (it must: ``setLogMode`` does not reach a ScatterPlotItem). Exporting what
# the item literally holds would write ``-3.7`` under a column called ``dwell_s``.
# So the log is undone here, and the column name states the unit the stored data is
# in. This is the same rule ``save_figure``'s CSVs follow (``bin_left_s``,
# ``dwell_s``), for the same reason: a number in a file has no axis beside it.

#: What :func:`plot_source_data` puts in the ``part`` column, and what each means.
#: A reader filters on this; a plot type uses two or three of them.
PLOT_DATA_PARTS: dict[str, str] = {
    "bins":  "one histogram bar: bin_left/bin_right and the drawn height",
    "fit":   "a sample of a fitted curve or line, at the resolution it was drawn",
    "points": "one plotted marker (scatter, PCA score)",
    "grid":  "one node of a 2-D KDE evaluation grid, with its density",
    "level": "one contour level of a 2-D KDE, as a density value (x and y are blank)",
    "loading": "one feature's loading on the two principal components",
}


def plot_data_paths(path: str | Path) -> tuple[Path, ...]:
    """Every file :func:`save_plot_data` writes for ``path`` -- exactly one CSV.

    Declared as a tuple, and by the writer's own module, so the *Export plot data…*
    dialog confirms what is really at stake through the same
    :func:`nanoscope.gui.export_paths.ask_export_path` seam every other export uses.
    A single-file export needs no enumerator today; having one means the seam does
    not have to be re-plumbed the day this grows a sidecar.
    """
    return (Path(path).with_suffix(".csv"),)


def _undo_log(values: np.ndarray, is_log: bool) -> np.ndarray:
    """``10**v`` when the panel drew *v* as log10 of the data, else *v* unchanged."""
    v = np.asarray(values, dtype=float)
    return np.power(10.0, v) if is_log else v


def plot_source_data(
    item,
    *,
    x_name: str,
    y_name: str,
    log_x: bool = False,
    log_y: bool = False,
    series_labels: Sequence[str] = (),
    fit_labels: Sequence[str] = (),
) -> pd.DataFrame:
    """The numbers behind a live pyqtgraph plot, as a long DataFrame.

    Walks the items the panel added, in the order it added them, and reads back what
    each one holds. Reading the ITEMS rather than re-deriving the numbers from the
    controls is the whole point: a CSV built by re-running the binning could disagree
    with the picture it claims to describe, and the user would have no way to tell.

    Args:
        item: the ``PlotItem``.
        x_name, y_name: what to call the two value columns. These are the EVENT-TABLE
            column names for a scatter/KDE (``dwell_s``, ``frac_depth``), the binned
            column and the normalisation for a histogram (``dwell_s``, ``count``),
            and ``PC1``/``PC2`` for a PCA.
        log_x, log_y: the panel's log-axis toggles, so the log it applied by hand to
            scatter points and KDE grids can be undone. See the module comment above.
        series_labels: names for the drawn series, in draw order -- what goes in the
            ``series`` column for ``bins``/``points``/``grid`` rows.
        fit_labels: the same for ``fit`` rows, which are added after every series.

    Returns:
        A DataFrame with ``series``, ``part`` (see :data:`PLOT_DATA_PARTS`), the two
        value columns, and ``bin_left``/``bin_right`` (a histogram) or ``density`` (a
        KDE) where the plot type has them. The density column is called
        ``kde_density`` in the one case where "density" would collide with ``x_name``
        or ``y_name``. EMPTY (no columns, no rows) when nothing is drawn -- the caller
        refuses to write a file rather than writing a header with no numbers under it.

    WHY THE LABELS ARE POSITIONAL AND NOT READ OFF THE LEGEND. The legend's texts are
    ELIDED (``_LEGEND_LABEL_CHARS`` = 34, so a recording filename comes back as
    "500 pM Apo ferri…-500 mV.csv"), and a legend the user has switched off has no
    entries at all -- neither is a name a data file should carry. The caller passes
    the full texts instead, in the order the draw offered them. Where it passes too
    few, the remaining series are numbered rather than mislabelled.
    """
    import pyqtgraph as pg

    rows: list[pd.DataFrame] = []
    series_i = fit_i = 0
    grids: dict[int, str] = {}          # id(z) -> the series label it was given
    # The KDE's third column normally has the obvious name -- but a histogram of a
    # column under *density* names its HEIGHT column "density" too, and an
    # events table is free to contain a column called ``density`` of its own. Two
    # columns of one name in one CSV is silent data loss (pandas writes both and reads
    # one), so the collision is resolved here, once, rather than left to chance.
    density_name = "density"
    if density_name in (x_name, y_name):
        density_name = "kde_density"

    def _series_label(i: int) -> str:
        if i < len(series_labels) and str(series_labels[i]).strip():
            return str(series_labels[i])
        return f"series {i + 1}"

    def _fit_label(i: int) -> str:
        if i < len(fit_labels) and str(fit_labels[i]).strip():
            return str(fit_labels[i])
        return f"fit {i + 1}"

    for node in list(getattr(item, "items", []) or []):
        if isinstance(node, pg.IsocurveItem):
            # A 2-D KDE. Every contour of one source shares the SAME density array --
            # ``_draw_kde`` evaluates ``z`` once and hands the identical object to
            # each level -- so identity groups the levels by source exactly, with no
            # dependence on how many levels the user asked for or on which sources
            # were skipped as degenerate.
            z = getattr(node, "data", None)
            if z is None:
                continue
            level = float(getattr(node, "level", float("nan")))
            key = id(z)
            if key not in grids:
                grids[key] = _series_label(series_i)
                series_i += 1
                zz = np.asarray(z, dtype=float)
                nx, ny = zz.shape
                # The item's own transform is what maps grid index -> axis
                # coordinates (``_draw_kde`` builds it: translate to the grid origin,
                # scale by the step). Read from the item so the exported grid is the
                # one that was contoured, not one recomputed from the controls.
                tr = node.transform()
                gx = np.array([tr.map(float(i), 0.0)[0] for i in range(nx)])
                gy = np.array([tr.map(0.0, float(j))[1] for j in range(ny)])
                # x-major, matching ``z[i, j]``: i -> x, j -> y.
                rows.append(pd.DataFrame({
                    "series": grids[key],
                    "part": "grid",
                    x_name: np.repeat(_undo_log(gx, log_x), ny),
                    y_name: np.tile(_undo_log(gy, log_y), nx),
                    density_name: zz.ravel(),
                }))
            rows.append(pd.DataFrame({
                "series": [grids[key]], "part": ["level"],
                x_name: [np.nan], y_name: [np.nan], density_name: [level],
            }))
            continue

        if isinstance(node, pg.ScatterPlotItem):
            data = getattr(node, "data", None)
            if data is None or len(data) == 0:
                continue
            # Display coordinates: ``_draw_scatter`` log10s them by hand because
            # ``setLogMode`` cannot reach a ScatterPlotItem. Undone here.
            rows.append(pd.DataFrame({
                "series": _series_label(series_i),
                "part": "points",
                x_name: _undo_log(data["x"], log_x),
                y_name: _undo_log(data["y"], log_y),
            }))
            series_i += 1
            continue

        if isinstance(node, pg.PlotDataItem):
            # ``xData``/``yData``, never ``getData()``: the latter returns the DISPLAY
            # arrays, which ``setLogMode`` has already taken log10 of. A PlotDataItem
            # keeps the data it was given, so this branch needs no undo at all.
            xd, yd = getattr(node, "xData", None), getattr(node, "yData", None)
            if xd is None or yd is None or len(xd) == 0:
                continue
            xd = np.asarray(xd, dtype=float)
            yd = np.asarray(yd, dtype=float)
            if node.opts.get("stepMode") and xd.size == yd.size + 1:
                # A histogram: N+1 edges against N heights.
                rows.append(pd.DataFrame({
                    "series": _series_label(series_i),
                    "part": "bins",
                    "bin_left": xd[:-1],
                    "bin_right": xd[1:],
                    x_name: 0.5 * (xd[:-1] + xd[1:]),
                    y_name: yd,
                }))
                series_i += 1
            else:
                if xd.size != yd.size:
                    continue
                rows.append(pd.DataFrame({
                    "series": _fit_label(fit_i),
                    "part": "fit",
                    x_name: xd,
                    y_name: yd,
                }))
                fit_i += 1

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # A stable column order, with the optional columns only where a row uses them.
    # ``dict.fromkeys`` de-duplicates while keeping the order: ``y_name`` is "density"
    # for a density-normalised histogram, and listing it twice made pandas write the
    # column twice.
    order = list(dict.fromkeys(
        c for c in ("series", "part", "bin_left", "bin_right", x_name, y_name,
                    density_name) if c in out.columns))
    return out[order]


def save_plot_data(target: str | Path, data: pd.DataFrame) -> list[str]:
    """Stage ``data`` as CSV beside ``target`` and move it into place.

    The CSV half of what ``save_plot_image`` does for the picture, and staged for the
    same reason (USER_GUIDE §12: "Every export replaces its target the same way a
    project save does… so a failure leaves the previous file whole rather than
    truncated"). ``DataFrame.to_csv`` opens and truncates the name it is handed, so
    an unstaged write over yesterday's export that then ran out of disk would leave a
    half-table under a name the user trusts.

    Returns:
        The file-identity sentences for this write, scoped to ``target`` -- the half
        the caller must not drop (see :func:`save_plot_image`). Normally empty.

    Raises:
        ValueError: ``data`` has no rows. Refused rather than written, because a CSV
            with a header and nothing under it is indistinguishable, a month later,
            from a plot that genuinely had no events in it.
        OSError: staging or the replace failed; ``target`` is exactly as it was.
    """
    from ..core import atomic_write

    out = Path(target)
    if data is None or not len(data):
        raise ValueError(
            f"nothing is plotted, so there is no source data to write to {out.name}")
    with atomic_write.staged_path(out) as staged:
        data.to_csv(staged, index=False)
    return atomic_write.take_identity_notes(out)
