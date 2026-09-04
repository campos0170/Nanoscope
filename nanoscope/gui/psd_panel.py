"""The standalone power-spectrum / noise window: the pore quality-control plot.

Opened from ``Analysis ▸ Power spectrum (noise / PSD)…``, this is the GUI face of
:mod:`nanoscope.analysis.psd` — the log-log current-noise spectrum every nanopore
paper uses to grade a pore. It sources the **calibrated current trace directly**
(PSD is raw-signal characterisation and needs no detection), computes a Welch PSD
over a chosen quiet baseline segment, and separates the three noise regimes:

* the low-frequency **1/f (flicker)** power law — its exponent ``alpha`` and 1 Hz
  magnitude are fit and the line is overlaid;
* the mid-decade **thermal / white floor**;
* the high-frequency **dielectric (~f) / capacitance (~f²) rise** toward Nyquist.

It also reports the **band-integrated RMS noise**: a total figure, the default
``<1 kHz / 1-10 kHz / 10 kHz-Nyquist`` decomposition, and a live read-out over a
user-settable band.

Design mirrors :class:`~nanoscope.gui.iv_panel.IVWindow` /
:class:`~nanoscope.gui.plots_panel.PlotsWindow`: a single, lazily-built top-level
window owned by :class:`~nanoscope.gui.main_window.MainWindow`, rendered with
pyqtgraph so it inherits the app's dark theme. It reads the calibrated current +
``fs`` + units for the selected recording straight from the
:class:`~nanoscope.core.session.Session` — the SAME path
:class:`~nanoscope.gui.measure_panel.MeasurePanel` and ``IVWindow`` use
(``session.calibrated(doc_id, ch)`` → ``.data`` / ``.sample_rate_hz`` /
``.units``) — so a pA channel yields a pA²/Hz spectrum and a pA RMS, and an nA
channel the nA-scaled equivalents, with no unit conversion in the app. A channel
whose units the file never declared (every fresh HDF5/CSV) yields a spectrum
labelled "(units unknown)" on every read-out and on the axis: this window reports
the units it was given and never supplies one of its own.

The spectrum defaults to the RAW calibrated trace: noise characterisation wants
the unfiltered signal, so no filter chain is applied here.

COMPARING PORES (why this window holds more than one curve)
-----------------------------------------------------------
"Is this pore quieter than that one?" is the single most common PSD question in
the lab, and a window that can only hold one curve cannot answer it -- you are
left flipping between two screenshots. So besides the primary recording the
window takes a **checkable comparison list**: every other open recording (segments
included) can be overlaid on the same axes, each in its own Okabe-Ito ink with a
legend entry, all computed with the SAME Welch parameters so the comparison is
like-for-like. The noise summary and the band-RMS read-out are reported **per
curve**, never merged: a single "RMS" number over two pores would be a number
that describes neither.

A comparison recording is analysed over the same time window as the primary,
clipped to its own duration (and the clipping is stated). Its own sample rate,
units and Nyquist are its own -- they are printed per curve rather than assumed
to match.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from PySide6.QtCore import QRectF

from ..analysis import allpoints as ap_mod
from ..analysis import psd as psd_mod
from ..analysis.filters import noise_rms
from ..core import clock
from ..core.segment import SegmentDocument, persistent_key
from ..core.session import Session
from ..core.sources import disambiguate_labels, gains_comparable, source_colour
from ..plot.style import OKABE_ITO
from .axes import ReadableLogAxis
from .combos import choose_combo_item
from .export_paths import ask_export_path

__all__ = ["PSDCurve", "PSDWindow", "VIEWS", "VIEW_PSD", "VIEW_FFT", "VIEW_SPECTROGRAM",
           "VIEW_ALLPOINTS"]

# Welch segment-length choices offered in the combo (plus "Auto").
_AUTO = "Auto"

#: The segment spins' tooltip before a recording is selected. Once one is, the offset
#: is known and :meth:`PSDWindow._label_segment_clock` replaces this with
#: ``clock.field_note(clock.RECORDING, offset)``, which names t₀ in numbers.
#:
#: THIS IS THE FIELD THE 5.7x ERROR WAS TYPED INTO (report High-6). The trace x-axis
#: and the event table both name their clock now; these two spins were the third
#: surface the finding lists and the only one where the wrong number is ENTERED rather
#: than merely read. A quiet stretch read off the -500 mV trace (instrument clock,
#: t₀ = 10.2526 s) and typed here analyses a different 1.4 s and returns total
#: RMS 22.66 pA / alpha 1.822 in place of 3.99 pA / 0.985 -- four translocations
#: measured as noise. `Analysis > Power spectrum of the measured region` hands the
#: selection over as sample indices and cannot make the mistake; a user typing can.
_SEG_SPIN_TIP = (
    "{edge} of the analysed segment, in seconds FROM THE START OF THIS RECORDING "
    "(0 = first sample). This is the same clock as the event table's t_start_s and "
    "NOT necessarily the clock the trace x-axis is drawn on.")

_NPERSEG_CHOICES = ("1024", "2048", "4096", "8192", "16384", "32768", "65536")
# Window functions (valid scipy.signal.get_window names).
_WINDOWS = ("hann", "hamming", "blackman", "bartlett", "blackmanharris", "flattop", "boxcar")

# Drawing colours (Okabe-Ito, colour-blind-safe — matching the other analysis windows).
_DATA_HEX = OKABE_ITO[0]        # blue -- the spectrum
_FIT_HEX = OKABE_ITO[1]         # vermillion -- the fitted 1/f line
_MSG_HEX = "#8b9199"
_MSG_LIGHT_HEX = "#555b63"

# Comparison curves start at Okabe-Ito index 2, NOT 1: index 0 is the primary spectrum
# and index 1 is the vermillion the primary's fitted 1/f line already uses, so starting at
# 1 would paint the first comparison the same colour as the fit it is meant to be read
# against.
_COMPARE_COLOUR_START = 2

# Theme foreground/background (mirrors PlotsWindow so light mode is publication-ready).
_BG_DARK = "#111316"
_FG_DARK = "#b9bfc7"
_BG_LIGHT = "#ffffff"
_FG_LIGHT = "#1a1a1a"

# ================================================================================================
# THE FOUR VIEWS OF ONE SPAN
# ================================================================================================
#
# This window already owned everything the three added views need — the channel pull, the
# clock-safe segment selection, the window/nperseg controls, the log-log axis class and the
# light/dark styling — so they are pages of it rather than three more windows. The SPAN is
# the shared fact: every view answers a different question about the same
# ``[Start, End]`` stretch of the same recording, and switching views never re-picks it.
#
# NO CONTROL MOVES WHEN THE VIEW CHANGES. Both the plot area and the per-view options are
# QStackedWidgets, whose size hint is the maximum over all pages, so the controls below the
# view selector sit at the same y in all four views. Each ADDED view's caption is a
# fixed-height read-only box for the same reason: its text changes with the data (a reduced
# FFT says so, a clipped histogram says so) and a caption that grows would move the plot
# above it under the pointer.
#
# THE PSD PAGE HAS NO CAPTION BOX, and that is deliberate rather than an omission. This
# window shipped before the three added views did, and "Save image…" on the Welch PSD
# exports that plot AT ITS NATIVE SIZE — so a 96 px box under it silently shortened every
# exported spectrum by 100 px (measured: a 691x488 PNG became 691x388). The PSD view also
# has nothing a caption would carry that is not already in the Noise details box below,
# which is on screen in all four views. A reserved box that never fills is worse than no
# box: it costs a fifth of the plot and says nothing. So the PSD page IS the plot widget,
# and its export is byte-for-byte the one it was before this window grew three views.
VIEW_PSD = "psd"
VIEW_FFT = "fft"
VIEW_SPECTROGRAM = "spectrogram"
VIEW_ALLPOINTS = "allpoints"

#: ``(key, combo label)`` in the order they are offered. The PSD is first and default: it
#: is the measurement, and the other three are readings taken beside it.
VIEWS: tuple[tuple[str, str], ...] = (
    (VIEW_PSD, "Welch PSD (averaged noise spectrum)"),
    (VIEW_FFT, "FFT (single-shot magnitude)"),
    (VIEW_SPECTROGRAM, "Spectrogram (PSD vs time)"),
    (VIEW_ALLPOINTS, "All-points histogram (levels)"),
)

# Spectrogram block lengths offered. Capped at 16384 rather than the Welch combo's 65536
# because the row count is the IMAGE HEIGHT: 65536 would be a 32 769-row image, and no
# screen and no exporter has any use for it. The Welch combo's own choice is deliberately
# NOT reused — the two autos optimise opposite ends of the same trade (see
# psd.default_spectrogram_nperseg).
_SPEC_NPERSEG_CHOICES = ("512", "1024", "4096", "16384")

#: Columns the spectrogram image is allowed. Above it neighbouring columns are AVERAGED
#: (never dropped), so every sample of the span is still on some pixel — see
#: :func:`nanoscope.analysis.psd.spectrogram_psd`.
_SPEC_MAX_COLUMNS = 900

_SPEC_COLORMAPS = ("viridis", "inferno", "magma", "cividis")

_HIST_BIN_CHOICES = ("128", "256", "512", "1024", "2048", "4096")

#: Quantile the all-points histogram clips to when "Clip outliers" is ticked: 0.01% at each
#: end. One saturating sample sets the full range otherwise and collapses the whole pore
#: signal into a single bin.
_HIST_CLIP_Q = 1e-4

#: Height of every per-view caption box, in px. Fixed so a longer message never resizes the
#: plot above it; it scrolls instead.
_CAPTION_H = 96


@dataclass
class PSDStyle:
    """The user-tunable look of the spectrum, mirroring PlotsWindow's StyleSettings.

    A blank ``title`` means "use the automatic title"; ``export_w``/``export_h`` of
    0 mean "export at the widget's native size".
    """
    title: str = ""
    data_color: str = _DATA_HEX
    fit_color: str = _FIT_HEX
    background: str = "dark"          # "dark" | "light"
    export_w: int = 0
    export_h: int = 0


@dataclass
class _Span:
    """The slice of ONE recording the window is currently looking at.

    Produced by :meth:`PSDWindow._span_for` and consumed by every view, so that "which
    samples" is decided exactly once per redraw rather than four times.
    """
    chan: object          # the CalibratedChannel (units / issues ride on it)
    seg: np.ndarray       # the samples themselves — a VIEW into the channel, never a copy
    fs: float
    i0: int               # first sample index within the recording
    i1: int               # one past the last
    note: str             # "" or why the requested window was clipped


@dataclass
class PSDCurve:
    """One recording's spectrum on the plot, with everything needed to report it
    WITHOUT borrowing another curve's numbers.

    Its own ``fs``, ``units`` and Nyquist travel with it, because two pores are not
    obliged to share a sample rate and a pA channel and an nA channel do not share
    a y unit. ``note`` carries anything that had to be adjusted for this curve (a
    time window clipped to a shorter recording, most often).
    """

    key: str                     # persistent key (+ channel) — stable, segment-safe
    doc_id: str
    channel: int
    label: str
    colour: str
    freqs: np.ndarray            # raw Welch frequencies (Hz), including DC
    psd: np.ndarray              # raw Welch PSD (units²/Hz)
    summary: dict
    #: The channel's OWN units, verbatim — ``""`` when the file declared none and the
    #: user has not set any in Calibration. Never defaulted to a real unit: a spectrum
    #: labelled pA is a claim about the amplifier, and this window does not have one.
    units: str = ""
    fs: float = 0.0
    nperseg: int = 0
    noverlap: int = 0
    nseg: int = 0
    n_samples: int = 0
    #: Index of the analysed window's FIRST sample in this recording. The segment
    #: read-out states the window on both clocks (report High-6) and cannot do that
    #: from a length alone -- it used to be told (0, n, n) and would have named the
    #: wrong stretch of the trace.
    i0: int = 0
    note: str = ""
    #: What is wrong with the CALIBRATED array this spectrum was computed from — the
    #: ``CalibratedChannel.issues`` of the channel, verbatim. Empty on a healthy channel.
    #: Carried on the curve because the panel that prints "unavailable" is the one place a
    #: user actually asks why, and the answer used to be "see the Issues box", which was
    #: filled from the RAW arrays and therefore said "(none)".
    channel_issues: tuple[str, ...] = ()
    #: True when the analysed window has zero variance. ``band_rms`` returns a literal,
    #: correct 0.0 for a constant signal, and that number is indistinguishable on screen
    #: from a flawless pore, so the panel has to say which of the two it is.
    constant: bool = False
    #: True when the analysed window VARIES but every PSD bin is exactly 0.0: the
    #: squared fluctuations underflowed float64 inside Welch (anything below ~1.5e-162
    #: squares to under the smallest subnormal, ~5e-324). ``band_rms`` then returns a
    #: literal 0.0 that is representation, not measurement — the signal is live and
    #: its noise is real, merely below what a float64 spectrum can hold — so every
    #: RMS surface must say "below-representable", never a confident zero (NE-07).
    #: The same duty ``constant`` carries for the flat-line case; the panel is the one
    #: place that can tell them apart, because it still has the samples.
    underflow: bool = False
    #: The analysed window's robust time-domain noise RMS
    #: (:func:`~nanoscope.analysis.filters.noise_rms` — MAD-based, so it survives the
    #: very underflow that blanks the spectrum). Computed only for the underflow case:
    #: it is the honest tiny number the spectrum cannot carry. 0.0 otherwise.
    time_rms: float = 0.0
    x: np.ndarray | None = None  # what was actually drawn (DC dropped, log-filtered)
    y: np.ndarray | None = None
    item: object = None          # the pg.PlotDataItem, for tests + incremental clears

    def band_rms(self, f_lo: float, f_hi: float) -> float:
        return psd_mod.band_rms(self.freqs, self.psd, f_lo, f_hi)


#: How a channel whose file declared no units is named on screen — the Calibration
#: panel's own wording (``_fmt_voltage`` there, the CF-5 contract) and the I-V window's.
#: A spectrum inherits its channel's units and MUST NOT invent one: an HDF5/CSV declares
#: none (docs/USER_GUIDE.md line 88: "reported as unknown and you set them in
#: Calibration"), and this window used to fill the blank with "pA", printing
#: "0.004204 pA" — an impossible 4.2 fA noise floor — for a file whose own loader note
#: says the magnitudes suggest nA (JM-06).
_UNKNOWN_UNITS = "(units unknown)"


def _unit_or_unknown(units: str) -> str:
    """The channel's units, or the named undeclared state. Never a guess."""
    return units or _UNKNOWN_UNITS


def _sq_units(units: str) -> str:
    """The PSD's own units, ``pA²/Hz`` — undeclared stays undeclared, squared."""
    return f"{_unit_or_unknown(units)}²/Hz"


def _level_boundaries(levels: list[dict]) -> list[float]:
    """Midpoints between neighbouring level centres — where one level's share stops."""
    cs = sorted(float(lv["center"]) for lv in levels)
    return [0.5 * (a + b) for a, b in zip(cs[:-1], cs[1:])]


def _level_occupancy(hist: ap_mod.AllPointsHistogram, levels: list[dict]) -> list[float]:
    """How much of the record each level actually holds — NOT one bin's share.

    :func:`~nanoscope.analysis.allpoints.level_peaks` reports a level's PEAK BIN count,
    which is what the picture is drawn from and what ranks the levels; its ``fraction``
    is that one bin over the binned samples. Printed as "the fraction of the record at
    this level" it is wrong by whatever the bin count happens to be: on a trace that sits
    at the open level 99.2% of the time, the peak bin's share reads 6.6% at 128 bins and
    0.21% at 4 096 — the same two physical levels, a 32x spread, and neither number is
    the occupancy.

    So the occupancy is INTEGRATED here: every bin is assigned to the nearest level
    centre (boundaries at the midpoints — the half-amplitude rule single-channel analysis
    has always used), and the level's share is that region's counts over
    ``hist.n_binned``. The shares therefore sum to exactly 1 and are stable in the bin
    count: the same synthetic reads 99.2% / 0.8% at every binning from 128 to 2 048,
    against a truth of 99.2% / 0.8%.

    Returns one fraction per entry of ``levels``, in the order given.
    """
    if not levels:
        return []
    centers = np.asarray([float(lv["center"]) for lv in levels], dtype=np.float64)
    binned = float(max(hist.n_binned, 1))
    counts = np.asarray(hist.counts, dtype=np.float64)
    if not counts.size:
        return [0.0] * len(levels)
    nearest = np.abs(hist.centers[:, None] - centers[None, :]).argmin(axis=1)
    return [float(counts[nearest == k].sum()) / binned for k in range(centers.size)]


def _fmt_band_edge(f: float) -> str:
    """A band edge in Hz, EXACTLY, always in plain notation.

    The shortest positional spelling that round-trips the value — ``0.5`` stays
    ``0.5``, ``25000`` stays ``25000`` (never ``2.5e+04``: the band label sits beside
    spinboxes that show plain decimals, and a read-out must not restate their values
    in a different notation) and ``12345.6`` stays ``12345.6``. It never rounds. The
    old ``:.0f`` rounded the edges to integers while the spinboxes deliberately accept
    0.1 Hz (``setDecimals(1)``), which is how a genuine 0–0.5 Hz band was labelled
    ``(0–0 Hz)`` (NE-03); its replacement, four SIGNIFICANT digits, re-rounded the
    edges independently of :func:`~nanoscope.analysis.psd.band_support` and above
    9,999 Hz misstated the band the integral came from — 12345–12355 Hz was labelled
    ``(12340–12360 Hz)`` and 24999–25000 Hz ``(25000–25000 Hz)``, a zero-width band
    :func:`~nanoscope.analysis.psd.band_rms` itself defines as holding nothing,
    printed beside a non-zero RMS (S7-PSD1). A label that restates the integrated
    band may not carry less precision than the band; see :func:`_fmt_band_edges` for
    the two-edge form that also tidies a float-noisy Nyquist.
    """
    return np.format_float_positional(float(f), unique=True, trim="-")


#: The RMS-band spinboxes' resolution in decimals — 0.1 Hz — and, BY THE SAME
#: CONSTANT, the resolution the band label restates the integrated edges at. One
#: number serving both is what stops the label re-rounding the edges on its own
#: terms: with the spins at 0.1 Hz and the label at "the fewest decimals within half
#: a PSD bin", the owner's -1000 mV excerpt (0.76 Hz bins, half = 0.38) printed a
#: typed 12345.3–12355.7 Hz as ``(12345–12356 Hz)`` and 1000.3–2000.0 as
#: ``(1000–2000 Hz)``, while band_support had handed the integral exactly (12345.3,
#: 12355.7) — the S7-PSD1 divergence again, at sub-bin magnitude (F-SK-4).
_BAND_SPIN_DECIMALS = 1


def _fmt_band_edges(lo: float, hi: float) -> tuple[str, str]:
    """Both edges of a NON-EMPTY integrated band, as the label restates them.

    Each edge rounded to the spinboxes' own resolution (:data:`_BAND_SPIN_DECIMALS`,
    0.1 Hz) and then spelt exactly, so a typed edge — which
    :func:`~nanoscope.analysis.psd.band_support` passes to the integral unchanged —
    prints back exactly as typed at every frequency, and a Nyquist that ``rfftfreq``
    delivers as 24999.999999999996 reads ``25000``, the number the spinbox beside it
    shows. The previous rule, the fewest decimals within half a PSD bin, made the
    label's precision depend on the recording: on the owner's -1000 mV excerpt (0.76 Hz
    bins) a typed 12345.3–12355.7 Hz read ``(12345–12356 Hz)`` and 1000.3–2000.0 read
    ``(1000–2000 Hz)``; on a 20 000-sample trace (24 Hz bins) 12345.0–12345.5 read
    ``(12345–12346 Hz)`` and 10000.4–10012.6 read ``(10000–10013 Hz)``. The integral
    had used the typed values; a label beside it may not restate them at less than the
    resolution they were entered at (F-SK-4).

    The two edges must also print DIFFERENTLY: a band narrower than the spin
    resolution is still a non-empty band (its trapezoid is non-zero) and a zero-width
    label beside a non-zero RMS is the S7-PSD1 contradiction, so the decimals grow
    (to 6) until the edges separate; past that the exact spelling stands.
    """
    for decimals in range(_BAND_SPIN_DECIMALS, 7):
        a, b = round(float(lo), decimals), round(float(hi), decimals)
        if a != b:
            return (_fmt_band_edge(a), _fmt_band_edge(b))
    return (_fmt_band_edge(lo), _fmt_band_edge(hi))


def _fmt_q(value: float | None, units: str) -> str:
    """A measured quantity for the read-out, or an honest "unavailable".

    NEVER let a non-finite value reach a bare ``:.4g``. ``band_rms`` returns NaN when a
    band cannot be integrated (a single non-finite sample poisons every Welch bin), and
    that has to read as *unmeasurable*, not as "nan" and above all not as the ``0.0`` it
    used to be — "Total RMS noise: 0 pA" is indistinguishable from a flawless pore.

    Deliberately does NOT name a cause: a NaN white floor can also mean "no positive PSD
    bins", which is not the same thing. The cause is stated once, by the caller, when the
    whole spectrum is non-finite.

    A blank ``units`` is NOT dropped to a bare number either: an unlabelled figure is
    read in whatever unit the rest of the window is in, so it says :data:`_UNKNOWN_UNITS`.
    """
    if value is None or not np.isfinite(value):
        return "unavailable"
    return f"{value:.4g} {_unit_or_unknown(units)}"


def _swatch_ink(hex_color: str) -> str:
    """Black or white ink for a colour swatch button, whichever is legible on it."""
    c = QtGui.QColor(hex_color)
    luma = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#000000" if luma > 140 else "#ffffff"


# Width the monospace Details pane is laid out for, and the wrapper that keeps a
# multi-sentence caution inside it. An unwrapped caution is a single long line the
# reader has to scroll sideways for, which is a good way to have it not read at all.
_DETAILS_COLS = 92


def _plural(n: int, noun: str) -> str:
    """``4 FFT bins`` / ``1 FFT bin`` — a count is never printed without its noun."""
    return f"{int(n)} {noun}" + ("" if int(n) == 1 else "s")


def _wrap(text: str, *, indent: int = 0) -> list[str]:
    """``text`` wrapped to the Details pane, continuation lines hanging by ``indent``."""
    return textwrap.wrap(text, width=_DETAILS_COLS,
                         subsequent_indent=" " * int(indent)) or [text]


# Superscript digits for the power-of-ten tick labels.
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _pow10(exponent: int) -> str:
    """``10ⁿ`` — ONE notation for every decade on a log axis."""
    return "10" + str(int(exponent)).translate(_SUPERSCRIPT)


class _ReadableAxis(ReadableLogAxis):
    """The spectrum's log-log axis: :class:`~nanoscope.gui.axes.ReadableLogAxis` in
    power-of-ten notation.

    The readability invariant -- no two drawn labels touching, decades ranked ahead of
    minor ticks, no pyqtgraph SI prefix inventing a ``(x0.001)`` multiplier on a log
    axis -- moved into the shared base after it was found implemented twice, in two
    files, differently: this window dropped only COLLIDING labels while the Plots
    window blanked every non-decade one, and blanking empties an axis zoomed inside a
    single decade. What is left here is this window's NOTATION, which is deliberately
    not shared: a spectrum spans ~10 decades and reads as ``10⁻⁵ … 10⁴``, while a
    5-decade dwell axis reads better as ``0.01 0.1 1 10 100``.

    The behaviour on this window's own axes is unchanged by the move -- the shipped
    smear (``4·10¹5·10⁶·10⁷·10⁸·10⁹·10¹00²``), the mixed ``0.01, 0.1, 1, 10¹``
    notation and the ``PSD (pA²/Hz) (x0.001)`` heading are all still gone, and the tick
    values are still the numbers the label says they are.
    """

    def choose_notation(self, exponents):
        return False          # always 10ⁿ / m·10ⁿ on a spectrum


class PSDWindow(QtWidgets.QMainWindow):
    """Log-log current-noise spectrum + band-RMS read-out for one recording.

    Fed nothing directly beyond the :class:`~nanoscope.core.session.Session`: it
    lists every open recording that has a live current channel and a known sample
    rate, pulls the selected one's calibrated current on demand, and recomputes
    the Welch PSD over the chosen segment. The owner builds it once, calls
    :meth:`set_target` to point it at the active recording, and shows it;
    :meth:`refresh` re-gathers the recording list (picking up any opened since).
    """

    workspaceChanged = Signal()

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None, *,
                 defer_initial_refresh: bool = False) -> None:
        """Build the window, optionally deferring every numerical analysis kernel.

        ``defer_initial_refresh`` is the project-restore path: the source list is still
        populated so durable identities can resolve, but the workspace adapter installs
        and paints the saved PSD/FFT/spectrogram/histogram before any recomputation.
        """
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("NanoScope — Power spectrum (noise / PSD)")
        self.resize(1080, 720)

        self._style = PSDStyle()
        self._updating = False

        # Live results (kept for tests + redraws).
        self._freqs: np.ndarray | None = None       # raw Welch frequencies (Hz), inc. DC
        self._psd: np.ndarray | None = None          # raw Welch PSD (units²/Hz)
        self._summary: dict | None = None
        self._units = ""              # the primary curve's units; "" == undeclared
        self._fs: float | None = None
        #: instrument_s - recording_s for the selected recording; None = no clock.
        #: See :meth:`_label_segment_clock` -- the segment spins TAKE a time, so they
        #: must say which clock it is in (report High-6).
        self._clock_offset_s: float | None = None
        self._nperseg_used = 0
        self._noverlap_used = 0
        self._nseg_used = 0
        self._plot_x: np.ndarray | None = None       # data-unit x/y actually plotted
        self._plot_y: np.ndarray | None = None
        self._curve_item: pg.PlotDataItem | None = None
        self._fit_item: pg.PlotDataItem | None = None
        self._message_item: pg.TextItem | None = None
        # The overlay. curves[0] is the PRIMARY recording, and the single-curve
        # accessors above mirror it so a one-pore session reads exactly as before.
        self._curves: list[PSDCurve] = []
        self._recordings: list[dict] = []            # every offerable recording
        #: The ``(doc_id, channel)`` the segment + RMS spin boxes were last DEFAULTED for,
        #: with the duration and Nyquist those defaults were derived from. See
        #: :meth:`_reconfigure_for_recording`: the four control values are the user's
        #: choice, and they are only overwritten when the recording under them changes.
        self._configured_for: tuple[str, int, float, float] | None = None

        #: Last result per added view, with the signature it was computed for, so flipping
        #: between the four pages is free and only a real parameter change recomputes.
        #: Keyed by view name -> ``(signature, payload)``.
        self._view_cache: dict[str, tuple[tuple, object]] = {}
        self._fft: psd_mod.FFTSpectrum | None = None
        self._spec: psd_mod.Spectrogram | None = None
        self._hist: ap_mod.AllPointsHistogram | None = None
        self._hist_levels: list[dict] = []

        self._build_ui()
        if defer_initial_refresh:
            self._repopulate(None)
        else:
            self.refresh()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        controls = self._build_controls()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(430)
        splitter.addWidget(scroll)

        right = QtWidgets.QSplitter(Qt.Vertical)
        # Both axes are _ReadableAxis: a log-log spectrum is this window's primary
        # output and it was unreadable — overlapping tick labels, three number
        # formats, and an invented "(x0.001)" multiplier on a log axis. The axis
        # class owns all three, so no draw path can reintroduce them.
        self._plot_widget = pg.PlotWidget(
            axisItems={"bottom": _ReadableAxis(orientation="bottom"),
                       "left": _ReadableAxis(orientation="left")})
        self._plot_widget.setMinimumSize(460, 360)
        self._plot: pg.PlotItem = self._plot_widget.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.15)

        # One page per view. The stack's size hint is the max over its pages, so switching
        # view never resizes the stack itself or shifts the details box below it. The PSD
        # page is the bare plot widget — see "THE PSD PAGE HAS NO CAPTION BOX" at the top of
        # this section: a caption box there would shorten every exported spectrum.
        self._view_stack = QtWidgets.QStackedWidget()
        self._captions: dict[str, QtWidgets.QPlainTextEdit] = {}
        self._view_stack.addWidget(self._plot_widget)
        self._view_stack.addWidget(self._page(VIEW_FFT, self._build_fft_plot()))
        self._view_stack.addWidget(self._page(VIEW_SPECTROGRAM, self._build_spec_plot()))
        self._view_stack.addWidget(self._page(VIEW_ALLPOINTS, self._build_hist_plot()))
        right.addWidget(self._view_stack)

        # The title names WHICH span and WHICH estimator these numbers came from, because
        # three of the four views are something else on screen above it and a box labelled
        # only "Noise details" beside a histogram reads as the histogram's numbers.
        details_box = QtWidgets.QGroupBox("Noise details — Welch PSD of the same span")
        dlay = QtWidgets.QVBoxLayout(details_box)
        dlay.setContentsMargins(6, 6, 6, 6)
        self._details = QtWidgets.QPlainTextEdit()
        self._details.setReadOnly(True)
        # Tall enough for the whole read-out INCLUDING a caution: a warning the reader
        # has to scroll to find is a warning that does not arrive.
        self._details.setMaximumHeight(260)
        self._details.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        dlay.addWidget(self._details)
        right.addWidget(details_box)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Wire every change signal AFTER the controls exist, so building them never
        # triggers a recompute mid-construction (the established panel pattern).
        self._wire_signals()

    # -- the four view pages ---------------------------------------------------------------------
    def _page(self, view: str, plot: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """One ADDED view's page: its plot above a FIXED-height caption box.

        The caption is where each added view says what it is and what it is not — the FFT
        that it is unaveraged and that its envelope draws the floor high, the spectrogram
        whether its columns average to the PSD view at the current settings, the histogram
        how many samples the drawn window excluded. It is a read-only text box of constant
        height rather than a growing label because its text changes with the data, and a
        caption that resizes moves the plot above it (the failure shape this campaign is
        under standing orders to avoid).

        NOT used for :data:`VIEW_PSD`: that page is the plot widget itself, so the
        pre-existing "Save image…" export of the spectrum keeps its native size. Every
        caption here is text this view cannot be read correctly without; a box reserved
        for a view that has nothing to add would only shrink the picture.
        """
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(plot, 1)
        cap = QtWidgets.QPlainTextEdit()
        cap.setReadOnly(True)
        cap.setFixedHeight(_CAPTION_H)
        cap.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        cap.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lay.addWidget(cap, 0)
        self._captions[view] = cap
        return page

    def _build_fft_plot(self) -> pg.PlotWidget:
        """Log-log magnitude spectrum — the same axis class as the PSD, a different y."""
        w = pg.PlotWidget(axisItems={"bottom": _ReadableAxis(orientation="bottom"),
                                     "left": _ReadableAxis(orientation="left")})
        w.setMinimumSize(460, 360)
        self._fft_widget = w
        self._fft_plot: pg.PlotItem = w.getPlotItem()
        self._fft_plot.showGrid(x=True, y=True, alpha=0.15)
        self._fft_item: pg.PlotDataItem | None = None
        return w

    def _build_spec_plot(self) -> pg.PlotWidget:
        """Image + colour bar. LINEAR in frequency, because the image rows are.

        An :class:`~pyqtgraph.ImageItem` maps its rows onto a uniform rectangle; a log
        frequency axis would leave every row where a linear axis put it and silently
        mislabel the whole picture. So the spectrogram is linear in f and logarithmic in
        POWER (the colour scale), which is the convention every spectrogram uses and the
        one that keeps the drawn thing and the axis in agreement.
        """
        w = pg.PlotWidget()
        w.setMinimumSize(460, 360)
        self._spec_widget = w
        self._spec_plot: pg.PlotItem = w.getPlotItem()
        self._spec_image = pg.ImageItem()
        self._spec_plot.addItem(self._spec_image)
        self._spec_bar: pg.ColorBarItem | None = None
        self._spec_msg: pg.TextItem | None = None
        return w

    def _build_hist_plot(self) -> pg.PlotWidget:
        """Counts (log) vs amplitude — the classic all-points view.

        The COUNT axis is the log one: translocations are rare, so on a linear count axis
        a 0.1%-occupancy blocked level is a flat line on the floor beside an open-pore
        peak a thousand times taller, and the picture says nothing the mean did not.
        """
        w = pg.PlotWidget(axisItems={"left": _ReadableAxis(orientation="left")})
        w.setMinimumSize(460, 360)
        self._hist_widget = w
        self._hist_plot: pg.PlotItem = w.getPlotItem()
        self._hist_plot.showGrid(x=True, y=True, alpha=0.15)
        self._hist_item: pg.PlotDataItem | None = None
        self._hist_marks: list[pg.InfiniteLine] = []
        return w

    def _build_controls(self) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(host)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self._build_recording_group())
        root.addWidget(self._build_view_group())
        root.addWidget(self._build_segment_group())
        root.addWidget(self._build_welch_group())
        root.addWidget(self._build_display_group())
        root.addWidget(self._build_rms_group())
        root.addWidget(self._build_style_group())

        self._save_btn = QtWidgets.QPushButton("Save image…")
        self._save_btn.setToolTip(
            "Export the view currently on screen as a PNG — the spectrum, the FFT, the "
            "spectrogram or the histogram, whichever is showing.")
        self._save_btn.clicked.connect(lambda: self.save_image())
        root.addWidget(self._save_btn)
        root.addStretch(1)
        return host

    def _build_recording_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Recording")
        form = QtWidgets.QFormLayout(box)
        self._rec_combo = QtWidgets.QComboBox()
        self._rec_combo.setToolTip(
            "The PRIMARY recording: it sets the segment window, the RMS band range "
            "and the y units of the plot.")
        form.addRow("Source", self._rec_combo)

        self._compare_list = QtWidgets.QListWidget()
        self._compare_list.setMaximumHeight(110)
        self._compare_list.setToolTip(
            "Tick other open recordings to overlay their spectra on the same axes — the "
            "two-pore noise comparison. Every curve uses the same Welch settings, and the "
            "noise summary and band-RMS are reported per curve.")
        self._compare_list.itemChanged.connect(self._on_compare_changed)
        form.addRow("Compare with", self._compare_list)

        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip(
            "Re-gather the open recordings (pick up any opened since) and recompute.")
        self._refresh_btn.clicked.connect(self._refresh_from_button)
        form.addRow(self._refresh_btn)
        return box

    def _build_view_group(self) -> QtWidgets.QGroupBox:
        """The view selector + the options that belong to the selected view only.

        The options live in a QStackedWidget rather than in groups that are shown and
        hidden, so the Segment / Welch / Display / RMS / Style groups below never move: a
        stack reserves the height of its TALLEST page in every page. Controls a view does
        not use are DISABLED rather than removed, for the same reason.
        """
        box = QtWidgets.QGroupBox("View")
        lay = QtWidgets.QVBoxLayout(box)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self._view_combo = QtWidgets.QComboBox()
        for key, label in VIEWS:
            self._view_combo.addItem(label, key)
        self._view_combo.setToolTip(
            "Four readings of the SAME span of the SAME recording:\n"
            "• Welch PSD — the averaged noise spectrum a pore is graded on;\n"
            "• FFT — one unaveraged transform of the whole span, for reading a mains line "
            "at the full 1/T resolution (noisier, not better);\n"
            "• Spectrogram — that PSD as a function of time, which is how a burst or a "
            "mains change that starts partway through becomes visible;\n"
            "• All-points histogram — how many samples sat at each current level.")
        form.addRow("Show", self._view_combo)
        lay.addLayout(form)

        self._view_options = QtWidgets.QStackedWidget()
        pages = [self._psd_options_page(), self._fft_options_page(),
                 self._spec_options_page(), self._hist_options_page()]
        for page in pages:
            self._view_options.addWidget(page)
        # PINNED to the tallest page, measured. A QStackedWidget's size HINT is already the
        # maximum over its pages, but a hint is only a request: the first build of this
        # group used word-wrapped notes, whose one-line sizeHint became three laid-out
        # lines, and the Segment / Welch / Display / RMS groups below sat 20 px LOWER in
        # the FFT view than in the spectrogram view (measured on the 1512x900 window,
        # y = 395 vs 375). Every note is now single-line and the height is fixed, so the
        # groups below cannot move when the view changes.
        self._view_options.setFixedHeight(max(p.sizeHint().height() for p in pages))
        lay.addWidget(self._view_options)
        return box

    @staticmethod
    def _options_page(rows: list[tuple[str, QtWidgets.QWidget]]) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)
        for label, widget in rows:
            if label:
                form.addRow(label, widget)
            else:
                form.addRow(widget)
        return page

    @staticmethod
    def _note(text: str) -> QtWidgets.QLabel:
        """A single-line-per-line note. Word wrap is OFF on purpose — see _build_view_group."""
        note = QtWidgets.QLabel(text)
        note.setWordWrap(False)
        note.setStyleSheet("color:#8b9199;")
        return note

    def _psd_options_page(self) -> QtWidgets.QWidget:
        return self._options_page([("", self._note(
            "Uses the Welch estimate settings\nbelow."))])

    def _fft_options_page(self) -> QtWidgets.QWidget:
        return self._options_page([("", self._note(
            "One transform of the whole span,\ntapered with the Window setting below.\n"
            "Segment length and overlap do not apply."))])

    def _spec_options_page(self) -> QtWidgets.QWidget:
        self._spec_nperseg_combo = QtWidgets.QComboBox()
        self._spec_nperseg_combo.addItem(_AUTO)
        self._spec_nperseg_combo.addItems(_SPEC_NPERSEG_CHOICES)
        self._spec_nperseg_combo.setToolTip(
            "Samples per time column (the FFT size). Short = fine in TIME and coarse in "
            "frequency; long = the reverse. Auto targets ~400 columns across the span, "
            "which is a different target from the Welch Auto below (~8 averages) because "
            "a spectrogram with 8 columns has no time axis worth drawing.")
        self._spec_cmap_combo = QtWidgets.QComboBox()
        self._spec_cmap_combo.addItems(_SPEC_COLORMAPS)
        self._spec_cmap_combo.setToolTip(
            "Colour map for log₁₀ power. All four are perceptually uniform, so equal "
            "colour steps are equal power steps.")
        return self._options_page([("Time resolution", self._spec_nperseg_combo),
                                   ("Colour map", self._spec_cmap_combo)])

    def _hist_options_page(self) -> QtWidgets.QWidget:
        self._hist_bins_combo = QtWidgets.QComboBox()
        self._hist_bins_combo.addItems(_HIST_BIN_CHOICES)
        self._hist_bins_combo.setCurrentText(str(ap_mod.DEFAULT_BINS))
        self._hist_bins_combo.setToolTip(
            "Amplitude bins across the drawn range. More bins resolve two close levels; "
            "fewer average away counting noise.")
        self._hist_clip_check = QtWidgets.QCheckBox("Clip outliers (0.01–99.99%)")
        self._hist_clip_check.setToolTip(
            "Window on the 0.01–99.99% quantiles instead of the full min–max, so one "
            "saturating sample cannot collapse the whole pore signal into a single bin. "
            "Excluded samples are COUNTED and reported in the caption, never dropped "
            "silently.")
        self._hist_levels_check = QtWidgets.QCheckBox("Mark the populated levels")
        self._hist_levels_check.setChecked(True)
        self._hist_levels_check.setToolTip(
            "Draw a dashed line at each level the histogram is peaked on and print it in "
            "the caption — the open-pore current and each blocked level, read off before "
            "any detector has run.")
        return self._options_page([("Bins", self._hist_bins_combo),
                                   ("", self._hist_clip_check),
                                   ("", self._hist_levels_check)])

    def _build_segment_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Segment (quiet open-pore baseline)")
        form = QtWidgets.QFormLayout(box)
        self._start_spin = QtWidgets.QDoubleSpinBox()
        self._end_spin = QtWidgets.QDoubleSpinBox()
        for sb in (self._start_spin, self._end_spin):
            sb.setDecimals(4)
            sb.setRange(0.0, 1.0)
            sb.setSingleStep(0.1)
            sb.setSuffix(" s")
        # The clock is named in :meth:`_label_segment_clock`, per recording, because the
        # offset is a property of the file. Until one is selected, say the general rule.
        self._start_spin.setToolTip(_SEG_SPIN_TIP.format(edge="Start"))
        self._end_spin.setToolTip(_SEG_SPIN_TIP.format(edge="End"))
        form.addRow("Start", self._start_spin)
        form.addRow("End", self._end_spin)
        self._full_btn = QtWidgets.QPushButton("Use full trace")
        self._full_btn.setToolTip("Reset the segment to span the whole recording.")
        self._full_btn.clicked.connect(self._use_full_trace)
        form.addRow(self._full_btn)
        self._seg_label = QtWidgets.QLabel("")
        self._seg_label.setStyleSheet("color:#8b9199;")
        form.addRow(self._seg_label)
        return box

    def _build_welch_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Welch estimate")
        form = QtWidgets.QFormLayout(box)
        self._nperseg_combo = QtWidgets.QComboBox()
        self._nperseg_combo.addItem(_AUTO)
        self._nperseg_combo.addItems(_NPERSEG_CHOICES)
        self._nperseg_combo.setToolTip(
            "Welch segment length (FFT size). Auto balances variance vs frequency "
            "resolution (≈8+ averaged segments). Longer = finer low-frequency detail.")
        form.addRow("Segment length", self._nperseg_combo)

        self._window_combo = QtWidgets.QComboBox()
        self._window_combo.addItems(_WINDOWS)
        self._window_combo.setToolTip("Taper applied to each segment before the FFT.")
        form.addRow("Window", self._window_combo)

        self._overlap_spin = QtWidgets.QSpinBox()
        self._overlap_spin.setRange(0, 95)
        self._overlap_spin.setValue(50)
        self._overlap_spin.setSuffix(" %")
        self._overlap_spin.setToolTip("Overlap between consecutive segments.")
        form.addRow("Overlap", self._overlap_spin)
        return box

    def _build_display_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Display")
        lay = QtWidgets.QVBoxLayout(box)
        self._loglog_check = QtWidgets.QCheckBox("Log-log axes")
        self._loglog_check.setChecked(True)
        self._loglog_check.setToolTip("Plot log10(PSD) vs log10(f) — the standard noise view.")
        lay.addWidget(self._loglog_check)
        self._asd_check = QtWidgets.QCheckBox("Amplitude spectral density (√PSD)")
        self._asd_check.setToolTip(
            "Plot the ASD √PSD in units/√Hz instead of the PSD in units²/Hz.")
        lay.addWidget(self._asd_check)
        return box

    def _build_rms_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("RMS noise band")
        form = QtWidgets.QFormLayout(box)
        self._rms_lo_spin = QtWidgets.QDoubleSpinBox()
        self._rms_hi_spin = QtWidgets.QDoubleSpinBox()
        for sb in (self._rms_lo_spin, self._rms_hi_spin):
            sb.setDecimals(_BAND_SPIN_DECIMALS)   # the label prints at this resolution too
            sb.setRange(0.0, 1.0)
            sb.setSingleStep(100.0)
            sb.setSuffix(" Hz")
        self._rms_lo_spin.setToolTip("Low edge of the RMS-noise integration band.")
        self._rms_hi_spin.setToolTip("High edge of the RMS-noise integration band.")
        form.addRow("f low", self._rms_lo_spin)
        form.addRow("f high", self._rms_hi_spin)
        self._rms_label = QtWidgets.QLabel("—")
        self._rms_label.setStyleSheet("font-weight:600; font-size:13px;")
        form.addRow("RMS in band", self._rms_label)
        return box

    def _build_style_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Style")
        form = QtWidgets.QFormLayout(box)
        self._title_edit = QtWidgets.QLineEdit()
        self._title_edit.setPlaceholderText("auto")
        self._title_edit.setToolTip("Override the plot title; blank uses the automatic title.")
        form.addRow("Title", self._title_edit)

        self._data_swatch = self._make_swatch(self._style.data_color, self._pick_data_color,
                                              "Colour of the spectrum curve.")
        form.addRow("Data colour", self._data_swatch)
        self._fit_swatch = self._make_swatch(self._style.fit_color, self._pick_fit_color,
                                             "Colour of the fitted 1/f line.")
        form.addRow("Fit colour", self._fit_swatch)

        self._bg_combo = QtWidgets.QComboBox()
        self._bg_combo.addItems(["Dark", "Light"])
        self._bg_combo.setToolTip("Light gives a white canvas with dark ink for publication.")
        form.addRow("Background", self._bg_combo)

        erow = QtWidgets.QHBoxLayout()
        self._export_w = QtWidgets.QSpinBox()
        self._export_h = QtWidgets.QSpinBox()
        for sp in (self._export_w, self._export_h):
            sp.setRange(0, 20000)
            sp.setValue(0)
            sp.setToolTip("Saved-PNG size in px (0 = native / follow aspect).")
        erow.addWidget(self._export_w, 1)
        erow.addWidget(QtWidgets.QLabel("×"))
        erow.addWidget(self._export_h, 1)
        form.addRow("Export size", erow)
        return box

    def _make_swatch(self, hex_color: str, on_click, tip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton()
        btn.setToolTip(tip)
        btn.setFixedHeight(24)
        btn.clicked.connect(on_click)
        self._paint_swatch(btn, hex_color)
        return btn

    @staticmethod
    def _paint_swatch(btn: QtWidgets.QPushButton, hex_color: str) -> None:
        btn.setText(hex_color)
        btn.setStyleSheet(
            f"background-color: {hex_color}; color: {_swatch_ink(hex_color)}; "
            "border: 1px solid #555; padding: 2px;")

    def _wire_signals(self) -> None:
        # Recording / segment / Welch params change the PSD itself -> full recompute.
        self._rec_combo.currentIndexChanged.connect(self._on_recording_changed)
        self._start_spin.valueChanged.connect(self._on_param_changed)
        self._end_spin.valueChanged.connect(self._on_param_changed)
        self._nperseg_combo.currentTextChanged.connect(self._on_param_changed)
        self._window_combo.currentTextChanged.connect(self._on_param_changed)
        self._overlap_spin.valueChanged.connect(self._on_param_changed)
        # Display toggles only change the drawn quantity -> redraw (no recompute).
        self._loglog_check.toggled.connect(self._on_display_changed)
        self._asd_check.toggled.connect(self._on_display_changed)
        # The view selector swaps pages and computes the newly-shown view if its inputs
        # moved since it was last drawn; the per-view options invalidate only their own.
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        self._spec_nperseg_combo.currentTextChanged.connect(self._on_view_option_changed)
        self._spec_cmap_combo.currentTextChanged.connect(self._on_view_option_changed)
        self._hist_bins_combo.currentTextChanged.connect(self._on_view_option_changed)
        self._hist_clip_check.toggled.connect(self._on_view_option_changed)
        self._hist_levels_check.toggled.connect(self._on_view_option_changed)
        # RMS band only re-integrates the stored PSD -> cheap read-out update.
        self._rms_lo_spin.valueChanged.connect(self._on_rms_changed)
        self._rms_hi_spin.valueChanged.connect(self._on_rms_changed)
        # Style is display-only.
        self._title_edit.textChanged.connect(self._on_style_changed)
        self._bg_combo.currentTextChanged.connect(self._on_style_background)
        self._export_w.valueChanged.connect(self._on_export_size)
        self._export_h.valueChanged.connect(self._on_export_size)

    # -- recording gathering ---------------------------------------------------------------------
    def _gather_recordings(self) -> list[dict]:
        """Every open recording PSD can run on: a live current channel + known rate.

        One dict per recording -- ``doc_id``, ``channel``, ``key``, ``label`` -- with
        the channel being the recording's first live current channel as
        :meth:`~nanoscope.core.session.Session.resolve_current_channel` finds it: the
        EFFECTIVE kind (the user's Calibration-panel override, else the header), the
        one resolver the Analysis-menu gate, Batch detect, the workflow rail and the
        calibration chip all ask. This window used to read ``ChannelInfo.kind`` off
        the header instead, so on a 2 s CSV of the owner's -700 mV recording with
        neutral column headers, corrected to "current" + pA + gain 2.5 in the
        Calibration panel and detected on (15 events on ch0, chip "✓ Gain set: ×2.5 →
        pA", rail "Events detected."), Analysis ▸ Power spectrum was ENABLED -- the
        menu asks the effective kind -- and opened a window whose Source combo was
        empty and whose canvas said "No recording with a current channel and a known
        sample rate. Open one, then Refresh." (sweep 7, X1). The reverse override (a
        header-current channel re-declared as voltage) empties the list here for the
        same reason it disables the menu item, so the two never disagree again.

        ``key`` is built from :func:`~nanoscope.core.segment.persistent_key`, so a
        segment named after its parent gets its own identity instead of shadowing it,
        and labels are disambiguated for the same reason a legend needs distinct names.
        """
        out: list[dict] = []
        for doc in self._session.documents():
            meta = doc.meta
            if meta.sample_rate_hz is None:
                continue
            index = self._session.resolve_current_channel(doc.doc_id, None)
            if index is None:
                continue
            cur = meta.channels[index]
            out.append({
                "doc_id": doc.doc_id,
                "channel": int(cur.index),
                "key": f"{persistent_key(doc)}#{int(cur.index)}",
                "label": doc.display_name,
                "channel_name": cur.name,
                "is_segment": isinstance(doc, SegmentDocument),
            })
        disambiguate_labels(out)
        return out

    def _repopulate(self, target: tuple[str, int] | None) -> None:
        """Refill the primary combo + the comparison list, preserving ticks by key."""
        prev_compare = set(self.compare_keys())
        recs = self._gather_recordings()
        self._recordings = recs
        self._updating = True
        self._rec_combo.clear()
        for rec in recs:
            self._rec_combo.addItem(rec["label"], (rec["doc_id"], rec["channel"]))
        self._select_combo_target(target)
        self._rebuild_compare_list(prev_compare)
        self._updating = False

    def _rebuild_compare_list(self, keep: set[str]) -> None:
        """List every recording EXCEPT the primary, re-ticking the ones still open.

        Each row carries the colour its curve will be drawn in, so the list reads as
        the legend key before the plot is even looked at."""
        primary = self._current_target()
        self._compare_list.clear()
        colour_index = _COMPARE_COLOUR_START
        for rec in self._recordings:
            if primary is not None and (rec["doc_id"], rec["channel"]) == primary:
                continue
            item = QtWidgets.QListWidgetItem(rec["label"])
            item.setData(Qt.UserRole, rec["key"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if rec["key"] in keep else Qt.Unchecked)
            item.setForeground(QtGui.QBrush(QtGui.QColor(source_colour(colour_index))))
            self._compare_list.addItem(item)
            colour_index += 1

    def refresh(self) -> None:
        """Re-gather the open recordings and recompute, keeping the current selection."""
        self._repopulate(self._current_target())
        self._reconfigure_for_recording()
        self._recompute()

    def _refresh_from_button(self) -> None:
        """A deliberate refresh replaces the durable numerical snapshot."""
        self.refresh()
        self.workspaceChanged.emit()

    def set_target(self, doc_id: str, channel: int) -> None:
        """Point the window at ``(doc_id, channel)`` — the active recording.

        Re-gathers first (so a just-opened recording is listed), then selects it.
        A target with no current channel / no rate is ignored (the combo keeps its
        selection); the owner only calls this for a current-channel recording.
        """
        recs = self._gather_recordings()
        # Resolve to the recording's own current channel if the passed channel is not it.
        target = next(((r["doc_id"], r["channel"]) for r in recs if r["doc_id"] == doc_id), None)
        self._repopulate(target)
        self._reconfigure_for_recording()
        self._recompute()

    def _select_combo_target(self, target: tuple[str, int] | None) -> bool:
        """Select the combo entry whose ``(doc_id, channel)`` equals ``target``.

        A manual scan, not ``QComboBox.findData``: Qt cannot match a Python tuple
        stored as user-data by value, so ``findData`` silently misses and the combo
        keeps index 0 — the recording would never switch (see the follow-selection
        test). Returns whether a match was found.
        """
        if target is None:
            return False
        for i in range(self._rec_combo.count()):
            if self._rec_combo.itemData(i) == target:
                self._rec_combo.setCurrentIndex(i)
                return True
        return False

    def _current_target(self) -> tuple[str, int] | None:
        data = self._rec_combo.currentData()
        if data is None:
            return None
        doc_id, channel = data
        return (doc_id, channel) if self._session.has_document(doc_id) else None

    # -- reconfigure spin ranges when the recording changes --------------------------------------
    def _reconfigure_for_recording(self) -> None:
        """Set the segment + RMS spin ranges/defaults **when the recording changes**.

        *** THE FOUR CONTROL VALUES BELONG TO THE USER, NOT TO THE REFRESHER. ***

        The analysis segment and the RMS integration band are the two things a user
        actually configures in this window: the band-integrated RMS is the number a pore is
        graded on, and it is only meaningful over the stretch and the band she chose. This
        method DEFAULTS them, which is right exactly once per recording — its own section
        header says "when the recording changes" — and wrong every other time it runs.

        It ran every other time. ``set_target`` had no same-target short-circuit and
        ``refresh`` calls it outright, so every entry in
        ``MainWindow.SURFACE_MUTATION_SIGNALS`` reached it: one accept/reject in the events
        table, a mask drag, a label, a hand measurement, a merge, opening or closing ANY
        document, a calibration edit, or re-selecting the SAME recording in the File dock.
        Measured on the owner's 15 s apo run with the window open: a segment of
        [2.000, 6.000] s and a band of 100–10000 Hz reading **52.24 pA** became
        [0.000, 14.993] s / 0–25000 Hz reading **228.9 pA** — a 4.4x different figure,
        presented identically, on a recording that never changed and with nothing said.
        With the window closed the reset still landed, so reopening it showed the reset
        state rather than the user's.

        So the defaults are applied only when the target this window is pointed at is not
        the one the current values were derived for. The duration and Nyquist are part of
        that identity: if the recording under the same key genuinely re-declares its rate
        or length, the defaults are the honest answer again. The rule the app already keeps
        one window over — ``PlotsWindow.set_data``: "Sources still present after a refresh
        keep their checked state, so re-pulling data doesn't yank the user off their view."
        """
        target = self._current_target()
        self._updating = True
        try:
            if target is None:
                for sb in (self._start_spin, self._end_spin, self._rms_lo_spin, self._rms_hi_spin):
                    sb.setRange(0.0, 1.0)
                    sb.setValue(0.0)
                self._seg_label.setText("")
                self._clock_offset_s = None
                self._label_segment_clock()
                self._configured_for = None
                return
            doc_id, channel = target
            meta = self._session.document(doc_id).meta
            self._clock_offset_s = clock.clock_offset_s(meta)
            self._label_segment_clock()
            fs = meta.sample_rate_hz or 0.0
            dur = meta.duration_s or 0.0
            nyq = fs / 2.0 if fs else 0.0
            # The ranges are re-asserted unconditionally — they are a property of the
            # recording, they are idempotent, and a value outside them would be a lie.
            for sb in (self._start_spin, self._end_spin):
                sb.setRange(0.0, float(dur))
            for sb in (self._rms_lo_spin, self._rms_hi_spin):
                sb.setRange(0.0, float(nyq))
            stamp = (doc_id, int(channel), float(dur), float(nyq))
            if stamp == self._configured_for:
                return          # same recording: the user's segment and band stand
            self._configured_for = stamp
            # Segment spans [0, duration]; default = whole trace.
            self._start_spin.setValue(0.0)
            self._end_spin.setValue(float(dur))
            # RMS band spans [0, Nyquist]; default = whole band (read-out = total RMS).
            self._rms_lo_spin.setValue(0.0)
            self._rms_hi_spin.setValue(float(nyq))
        finally:
            self._updating = False

    def _use_full_trace(self) -> None:
        target = self._current_target()
        if target is None:
            return
        meta = self._session.document(target[0]).meta
        dur = meta.duration_s or 0.0
        self._updating = True
        self._start_spin.setValue(0.0)
        self._end_spin.setValue(float(dur))
        self._updating = False
        self._recompute()
        self.workspaceChanged.emit()

    # -- change slots ----------------------------------------------------------------------------
    def _on_recording_changed(self, *_a) -> None:
        if self._updating:
            return
        # The primary moved: it must leave the comparison list (a recording cannot be
        # compared with itself), and whichever recording it displaced must join it.
        keep = set(self.compare_keys())
        self._updating = True
        self._rebuild_compare_list(keep)
        self._updating = False
        self._reconfigure_for_recording()
        self._recompute()
        self.workspaceChanged.emit()

    def _on_compare_changed(self, *_a) -> None:
        if not self._updating:
            self._recompute()
            self.workspaceChanged.emit()

    def _on_param_changed(self, *_a) -> None:
        if not self._updating:
            self._recompute()
            self.workspaceChanged.emit()

    def _on_display_changed(self, *_a) -> None:
        if not self._updating:
            self._redraw_and_views()
            self.workspaceChanged.emit()

    def _on_rms_changed(self, *_a) -> None:
        if not self._updating:
            self._update_rms_readout()
            self.workspaceChanged.emit()

    def _on_style_changed(self, *_a) -> None:
        self._style.title = self._title_edit.text().strip()
        if not self._updating:
            self._redraw_and_views()
            self.workspaceChanged.emit()

    def _on_style_background(self, text: str) -> None:
        self._style.background = "light" if text == "Light" else "dark"
        if not self._updating:
            self._redraw_and_views()
            self.workspaceChanged.emit()

    def _on_export_size(self, *_a) -> None:
        self._style.export_w = int(self._export_w.value())
        self._style.export_h = int(self._export_h.value())
        if not self._updating:
            self.workspaceChanged.emit()

    def _pick_data_color(self) -> None:
        self._pick_color("data_color", self._data_swatch)

    def _pick_fit_color(self) -> None:
        self._pick_color("fit_color", self._fit_swatch)

    def _pick_color(self, attr: str, swatch: QtWidgets.QPushButton) -> None:
        col = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(getattr(self._style, attr)), self, "Pick colour")
        if col.isValid():
            setattr(self._style, attr, col.name())
            self._paint_swatch(swatch, col.name())
            self._redraw_and_views()
            self.workspaceChanged.emit()

    # -- public setters (owner + tests) ----------------------------------------------------------
    def set_segment(self, start_s: float, end_s: float) -> None:
        self._updating = True
        self._start_spin.setValue(float(start_s))
        self._end_spin.setValue(float(end_s))
        self._updating = False
        self._recompute()

    def set_nperseg(self, nperseg: int | str) -> None:
        """Choose the Welch segment length: an int (matched to a combo entry, else
        appended) or the string ``"Auto"``."""
        if isinstance(nperseg, str) and nperseg.lower() == "auto":
            choose_combo_item(self._nperseg_combo, _AUTO, "nperseg")
            return
        text = str(int(nperseg))
        if self._nperseg_combo.findText(text) < 0:
            self._nperseg_combo.addItem(text)
        choose_combo_item(self._nperseg_combo, text, "nperseg")

    def set_window(self, name: str) -> None:
        """Welch taper (``hann``, ``flattop``, ...). RAISES on an unknown name.

        ``QComboBox.setCurrentText`` on a non-editable combo is a no-op for unmatched
        text, so this used to keep the previous taper and report its spectrum under the
        requested one's name -- report Low-21, in a third window (see
        :func:`nanoscope.gui.combos.choose_combo_item`).
        """
        choose_combo_item(self._window_combo, name, "window")

    def set_overlap_percent(self, pct: int) -> None:
        self._overlap_spin.setValue(int(pct))

    def set_log_log(self, on: bool) -> None:
        self._loglog_check.setChecked(bool(on))

    def set_asd(self, on: bool) -> None:
        self._asd_check.setChecked(bool(on))

    def is_asd(self) -> bool:
        return self._asd_check.isChecked()

    def set_rms_band(self, f_lo: float, f_hi: float) -> None:
        self._updating = True
        self._rms_lo_spin.setValue(float(f_lo))
        self._rms_hi_spin.setValue(float(f_hi))
        self._updating = False
        self._update_rms_readout()

    def set_title(self, title: str) -> None:
        self._title_edit.setText(str(title))

    def set_background(self, mode: str) -> None:
        choose_combo_item(
            self._bg_combo, "Light" if str(mode).lower() == "light" else "Dark",
            "background")

    def set_export_size(self, w: int, h: int) -> None:
        self._export_w.setValue(int(w))
        self._export_h.setValue(int(h))

    # -- comparison overlay (owner + tests) ------------------------------------------------------
    def compare_keys(self) -> list[str]:
        """The ticked comparison-recording keys, in list order."""
        if not hasattr(self, "_compare_list"):
            return []
        return [self._compare_list.item(i).data(Qt.UserRole)
                for i in range(self._compare_list.count())
                if self._compare_list.item(i).checkState() == Qt.Checked]

    def comparable_labels(self) -> list[str]:
        """Every recording offered for comparison (i.e. every one but the primary)."""
        return [self._compare_list.item(i).text() for i in range(self._compare_list.count())]

    def set_compare(self, names) -> None:
        """Overlay exactly these recordings alongside the primary.

        Accepts persistent keys, ``doc_id``s or display labels, so a caller can name
        a recording however it happens to hold it. Unknown names are ignored."""
        want = {str(n) for n in names}
        self._updating = True
        for i in range(self._compare_list.count()):
            item = self._compare_list.item(i)
            key = item.data(Qt.UserRole)
            rec = next((r for r in self._recordings if r["key"] == key), None)
            hit = (key in want or item.text() in want
                   or (rec is not None and rec["doc_id"] in want))
            item.setCheckState(Qt.Checked if hit else Qt.Unchecked)
        self._updating = False
        self._recompute()

    def curves(self) -> list[PSDCurve]:
        """Every drawn spectrum, primary first — each with its OWN summary, units and
        Welch parameters. This is the per-curve report the overlay exists for."""
        return list(self._curves)

    def curve_labels(self) -> list[str]:
        return [c.label for c in self._curves]

    def curve_summary(self, name: str) -> dict | None:
        """One curve's noise summary, by key or label (None when it is not drawn)."""
        for c in self._curves:
            if name in (c.key, c.label, c.doc_id):
                return dict(c.summary)
        return None

    def band_rms_per_curve(self) -> dict[str, float]:
        """``{label: band RMS}`` over the current band, one entry per drawn curve."""
        lo, hi = self._rms_lo_spin.value(), self._rms_hi_spin.value()
        return {c.label: c.band_rms(lo, hi) for c in self._curves}

    # -- public accessors (owner + tests) --------------------------------------------------------
    def has_data(self) -> bool:
        return self._psd is not None and self._freqs is not None

    def psd_result(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The raw Welch ``(freqs, psd)`` (including the DC bin), or None."""
        if self._freqs is None or self._psd is None:
            return None
        return self._freqs.copy(), self._psd.copy()

    def plotted_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The ``(x, y)`` actually drawn, in DATA units (PSD or √PSD; DC dropped)."""
        if self._plot_x is None or self._plot_y is None:
            return None
        return self._plot_x.copy(), self._plot_y.copy()

    def summary(self) -> dict | None:
        return None if self._summary is None else dict(self._summary)

    def rms_value(self) -> float:
        """The band RMS currently shown in the read-out (0.0 when unavailable)."""
        if self._freqs is None or self._psd is None:
            return 0.0
        return psd_mod.band_rms(self._freqs, self._psd,
                                self._rms_lo_spin.value(), self._rms_hi_spin.value())

    def rms_readout_text(self) -> str:
        return self._rms_label.text()

    def units(self) -> str:
        return self._units

    def segment_info(self) -> tuple[int, int, int]:
        """``(nperseg, noverlap, n_segments)`` used in the last compute."""
        return (self._nperseg_used, self._noverlap_used, self._nseg_used)

    def plot_item(self) -> pg.PlotItem:
        return self._plot

    def curve_item(self) -> pg.PlotDataItem | None:
        return self._curve_item

    def details_text(self) -> str:
        return self._details.toPlainText()

    def style(self) -> PSDStyle:
        return self._style

    # -- compute ---------------------------------------------------------------------------------
    def _resolve_welch_params(self, n: int) -> tuple[int, int]:
        """Concrete ``(nperseg, noverlap)`` for a segment of ``n`` samples."""
        choice = self._nperseg_combo.currentText()
        if choice == _AUTO:
            nps = psd_mod.default_nperseg(n)
        else:
            nps = min(int(choice), n)
        nps = max(int(nps), 1)
        nov = int(round(nps * self._overlap_spin.value() / 100.0))
        nov = int(np.clip(nov, 0, nps - 1))
        return nps, nov

    def _span_for(self, rec: dict) -> _Span | str:
        """The samples of ONE recording inside the current ``[Start, End]`` window.

        THE ONE WRITER of "which samples is this window looking at". All four views ask
        it, so a spectrum, an FFT, a spectrogram and a histogram drawn from the same
        controls are drawn from the identical array; when the Segment spins move, they
        all move together or none of them do.

        It slices BEFORE it computes and never materialises more than the window: a
        2 s look at a 12 M-sample recording hands 100 000 samples to the kernel, not
        12 M. (``Session.calibrated`` is itself lazy and cached, and at an identity
        calibration it returns the document's own array by reference, so no copy of the
        recording is made here either.)

        Returns the span, or a human-readable reason there is none.
        """
        label = rec["label"]
        try:
            chan = self._session.calibrated(rec["doc_id"], rec["channel"])
        except (KeyError, TypeError, ValueError):
            return f"{label}: could not read the calibrated current."
        fs = chan.sample_rate_hz
        if not fs:
            return f"{label}: no known sample rate — a spectrum needs a time base."

        data = np.asarray(chan.data)
        n_total = data.size
        want0, want1 = self._start_spin.value(), self._end_spin.value()
        i0 = int(np.clip(int(round(want0 * fs)), 0, n_total))
        i1 = int(np.clip(int(round(want1 * fs)), 0, n_total))
        seg = data[i0:i1]
        if seg.size < 2:
            return (f"{label}: the selected {want0:.4g}–{want1:.4g} s window holds "
                    f"{seg.size} sample(s) of this recording — nothing to transform.")
        note = ""
        if i1 < int(round(want1 * fs)):
            note = (f"window clipped to this recording's length "
                    f"({i0 / fs:.4g}–{i1 / fs:.4g} s)")
        return _Span(chan=chan, seg=seg, fs=float(fs), i0=int(i0), i1=int(i1), note=note)

    def _compute_curve(self, rec: dict, colour: str) -> PSDCurve | str:
        """Welch-estimate ONE recording over the current time window.

        Returns the curve, or a human-readable reason it could not be computed.
        The window is taken from the primary's spin boxes and clipped to this
        recording's own length -- a shorter comparison pore is analysed over what it
        has, and the curve says so rather than silently spanning a different span.
        """
        span = self._span_for(rec)
        if isinstance(span, str):
            return span
        chan, seg, fs, i0, note = span.chan, span.seg, span.fs, span.i0, span.note
        doc_id, channel = rec["doc_id"], rec["channel"]
        label = rec["label"]

        nps, nov = self._resolve_welch_params(seg.size)
        try:
            freqs, psd = psd_mod.welch_psd(
                seg, float(fs), nperseg=nps, noverlap=nov,
                window=self._window_combo.currentText())
        except ValueError as exc:
            return f"{label}: could not compute the spectrum: {exc}"

        constant = bool(np.isfinite(seg).all() and seg.size and seg.min() == seg.max())
        # A live window whose every PSD bin is exactly 0 lost its variance to float64
        # representation inside Welch (see PSDCurve.underflow) — the same detection
        # welch_psd itself warns on, done here so the flag rides the curve to every
        # read-out. The robust time-domain RMS is computed ONLY then: it is the honest
        # number the spectrum cannot hold, and MAD needs no squaring so it survives.
        underflow = bool(not constant and np.isfinite(seg).all()
                         and psd.size and not psd.any())
        time_rms = float(noise_rms(np.asarray(seg, dtype=np.float64))) if underflow else 0.0

        return PSDCurve(
            key=rec["key"], doc_id=doc_id, channel=channel, label=label, colour=colour,
            freqs=freqs, psd=psd, summary=psd_mod.noise_summary(freqs, psd, float(fs)),
            # `chan.units` verbatim: "" means undeclared and is reported as such
            # everywhere below (JM-06 — this used to read `chan.units or "pA"`).
            units=chan.units, fs=float(fs), nperseg=nps, noverlap=nov,
            nseg=1 if nps >= seg.size else 1 + (seg.size - nps) // max(nps - nov, 1),
            n_samples=int(seg.size), i0=int(i0), note=note,
            channel_issues=tuple(chan.issues),
            constant=constant, underflow=underflow, time_rms=time_rms,
        )

    def _recompute(self) -> None:
        """Welch-estimate the primary recording and every ticked comparison.

        The primary's result is mirrored onto the single-curve fields (``_freqs`` /
        ``_psd`` / ``_summary`` / ``_fs`` / ``_units``) so a one-pore session — and
        every accessor built on it — behaves exactly as before.
        """
        self._freqs = self._psd = self._summary = None
        self._nperseg_used = self._noverlap_used = self._nseg_used = 0
        self._curves = []
        self._compute_notes: list[str] = []

        target = self._current_target()
        if target is None:
            self._message("No recording with a current channel and a known sample rate.\n"
                          "Open one, then Refresh.")
            self._update_rms_readout()
            return

        primary = next((r for r in self._recordings
                        if (r["doc_id"], r["channel"]) == target), None)
        if primary is None:
            self._message("The selected recording is no longer open — press Refresh.")
            self._update_rms_readout()
            return

        wanted = [(primary, OKABE_ITO[0])]
        compare_keys = self.compare_keys()
        colour_index = _COMPARE_COLOUR_START
        for rec in self._recordings:
            if rec["key"] in compare_keys and (rec["doc_id"], rec["channel"]) != target:
                wanted.append((rec, source_colour(colour_index)))
                colour_index += 1

        curves: list[PSDCurve] = []
        for rec, colour in wanted:
            out = self._compute_curve(rec, colour)
            if isinstance(out, str):
                if rec is primary:
                    self._message(out)
                    self._update_rms_readout()
                    return
                self._compute_notes.append(out)
                continue
            curves.append(out)

        head = curves[0]
        self._curves = curves
        self._freqs, self._psd, self._summary = head.freqs, head.psd, head.summary
        self._units, self._fs = head.units, head.fs
        self._nperseg_used, self._noverlap_used = head.nperseg, head.noverlap
        self._nseg_used = head.nseg

        self._update_seg_label(head.i0, head.i0 + head.n_samples, head.n_samples)
        self._redraw()
        # The three added views read the same span; they are refreshed AFTER the PSD so a
        # failure in one of them can never leave the spectrum half-drawn.
        self._refresh_view()

    # -- redraw ----------------------------------------------------------------------------------
    def _redraw(self) -> None:
        """Draw every stored spectrum (PSD or ASD, log-log or linear) + flicker lines.

        With one curve this is the original single-spectrum view, drawn in the
        Style group's data colour. With several it becomes the pore-comparison
        overlay: one coloured curve per recording, named in a legend, and only the
        PRIMARY's 1/f line is drawn — n dashed fit lines over n spectra is a thicket
        rather than a comparison, and every curve's alpha is printed in Details.
        """
        self._clear_plot()
        if not self._curves:
            return

        asd = self._asd_check.isChecked()
        log_log = self._loglog_check.isChecked()
        multi = len(self._curves) > 1
        if multi:
            self._ensure_legend()

        drawn = 0
        for i, curve in enumerate(self._curves):
            xs, ys = psd_mod.drop_dc(curve.freqs, curve.psd)  # DC has no log; drop it
            if asd:
                ys = psd_mod.asd_from_psd(ys)
            good = np.isfinite(xs) & np.isfinite(ys)
            if log_log:
                good &= (xs > 0.0) & (ys > 0.0)               # log10 needs positive data
            xs, ys = xs[good], ys[good]
            curve.x, curve.y = xs, ys
            if xs.size == 0:
                curve.item = None
                continue
            colour = self._style.data_color if i == 0 else curve.colour
            item = pg.PlotDataItem(x=xs, y=ys, pen=pg.mkPen(colour, width=1))
            self._plot.addItem(item)
            curve.item = item
            drawn += 1
            if i == 0:
                self._curve_item = item
                self._plot_x, self._plot_y = xs, ys
            if multi:
                self._legend_add(item, self._curve_legend_text(curve, asd))

        if drawn == 0:
            self._message(self._nothing_drawn_reason())
            self._update_rms_readout()
            return

        self._draw_flicker_line(asd, log_log)

        self._plot.setLogMode(x=log_log, y=log_log)
        yq = "ASD" if asd else "PSD"
        # The y axis carries the PRIMARY curve's units. Undeclared units are named on
        # the axis too -- "PSD ()" or "PSD (²/Hz)" reads as a formatting slip and lets
        # the viewer supply whatever unit they expected.
        if self._units:
            yunit = f"{self._units}/√Hz" if asd else f"{self._units}²/Hz"
            left = f"{yq} ({yunit})"
        else:
            left = f"{yq} (units unknown — set them in Calibration)"
        self._plot.setLabel("bottom", "Frequency (Hz)", **self._label_style())
        self._plot.setLabel("left", left, **self._label_style())
        self._apply_title(self._auto_title())
        self._apply_background()
        self._plot.autoRange()

        self._fill_details()
        self._update_rms_readout()

    def _redraw_and_views(self) -> None:
        """Redraw the spectrum AND the visible added view.

        Title, background and colour belong to all four pages, and log-log belongs to two
        of them, so a display change that reached only the PSD would leave a light-mode
        spectrum beside a dark-mode spectrogram. Cheap: the added views redraw from their
        cached kernel results and only recompute when the SPAN or their own options move.
        """
        self._redraw()
        self._refresh_view()

    def _nothing_drawn_reason(self) -> str:
        """Why there is no spectrum on screen — the REAL reason, not a guess.

        "No positive spectral points to plot (check the log-log toggle)" sent the user to
        fiddle with an axis setting when their recording contained a NaN. Welch averages
        segment periodograms, so one bad sample makes every bin NaN and nothing is
        plottable on any axis. Say that, and point at where the count is.
        """
        if any(not np.isfinite(c.psd).any() for c in self._curves):
            head = ("This spectrum is not finite, so there is nothing to plot and no noise "
                    "figure can be read from it.\n\n"
                    "Welch averages segment periodograms, so a single NaN or inf sample "
                    "anywhere in the recording poisons every frequency bin.")
            # Name the CAUSE when we have it. Pointing at the File info panel's Issues box
            # was misdirection whenever the calibration was what broke the channel: that box
            # is filled from the RAW arrays at load, so a gain that overflows float32 leaves
            # it reading "(none)" while every bin here is NaN.
            named = [line for c in self._curves for line in c.channel_issues]
            if named:
                return head + "\n\n" + "\n\n".join(named)
            return head + ("\n\nThe File info panel's Issues box names the channel and "
                           "counts the bad samples.")
        # A CONSTANT window is the other way to reach an empty plot, and it is the one that
        # used to be reported as an axis problem. Every PSD bin of a signal with no variance
        # is 0, which log-log cannot draw — but "check the log-log toggle" sends the user to
        # fiddle with a setting when their channel is a flat line. It is the same misdirection
        # the NaN branch above exists to correct, and the same sentence is owed here: a total
        # RMS of exactly 0 is arithmetic, not a measurement of a perfect pore.
        flat = [c for c in self._curves if c.constant]
        if flat and len(flat) == len(self._curves):
            names = ", ".join(c.label for c in flat)
            return (f"Every sample of the analysed window is the same value, so this "
                    f"recording has no spectrum: the total RMS is exactly 0 by arithmetic, "
                    f"not because the pore is quiet.\n\n"
                    f"A channel that never moves is usually an unset or wrong gain, a "
                    f"disconnected headstage, or a dead ADC channel. Check the Calibration "
                    f"panel's raw-vs-calibrated read-out for {names}, or choose a different "
                    f"time window.")
        # A LIVE signal can also blank the plot: fluctuations so small their squares
        # underflow float64 inside Welch leave every bin exactly 0 — nothing positive to
        # draw, and the same "check the log-log toggle" misdirection waiting below.
        low = [c for c in self._curves if c.underflow]
        if low and len(low) == len(self._curves):
            names = ", ".join(
                f"{c.label} (time-domain RMS ≈ {c.time_rms:.4g} {_unit_or_unknown(c.units)})"
                if c.time_rms > 0.0 else c.label for c in low)
            return (f"The signal is live but its fluctuations are so small that they "
                    f"square to below float64's smallest representable number inside the "
                    f"Welch estimate: every PSD bin is exactly 0, which no axis can draw "
                    f"and no RMS can be integrated from.\n\n"
                    f"This is a representation limit, not a quiet pore — measured in the "
                    f"time domain: {names}. A magnitude this small usually means the "
                    f"calibration gain is off by many orders; rescale in the Calibration "
                    f"panel to measure this noise.")
        return "No positive spectral points to plot (check the log-log toggle)."

    @staticmethod
    def _curve_legend_text(curve: PSDCurve, asd: bool) -> str:
        """A legend entry that carries the number the comparison is actually about:
        the curve's total RMS noise (with its own units), not just its name."""
        rms = curve.summary.get("total_rms")
        if rms is None or not np.isfinite(rms):
            return curve.label
        if curve.underflow and rms == 0.0:
            # the 0 is float64 underflow, not a measurement — the Details pane and the
            # RMS read-out both say so, and a legend cropped into a slide may not be
            # the one surface still claiming "0 pA RMS" (NE-07).
            return f"{curve.label}  (RMS below-representable)"
        return f"{curve.label}  ({rms:.4g} {_unit_or_unknown(curve.units)} RMS)"

    def _draw_flicker_line(self, asd: bool, log_log: bool) -> None:
        """Overlay the PRIMARY curve's fitted 1/f power law over its fit band."""
        if self._summary is None:
            return
        fl = self._summary.get("flicker", {})
        alpha, a1hz = fl.get("alpha"), fl.get("A1Hz")
        lo, hi = self._summary.get("flicker_band", (0.0, 0.0))
        if (alpha is None or a1hz is None or not np.isfinite(alpha) or not np.isfinite(a1hz)
                or not (hi > lo > 0.0)):
            return
        grid = np.geomspace(lo, hi, 60)
        line = a1hz * np.power(grid, -alpha)               # PSD units
        if asd:
            line = np.sqrt(np.clip(line, 0.0, None))
        if log_log:
            keep = (grid > 0.0) & (line > 0.0)
            grid, line = grid[keep], line[keep]
        if grid.size < 2:
            return
        self._fit_item = pg.PlotDataItem(
            x=grid, y=line, pen=pg.mkPen(self._style.fit_color, width=2, style=Qt.DashLine))
        self._plot.addItem(self._fit_item)
        if len(self._curves) > 1:
            # Every surface that prints α prints what it rests on — a legend entry is
            # the one that ends up cropped into a slide on its own.
            n = int(fl.get("n", 0))
            mark = ", MARGINAL" if fl.get("status") == "marginal" else ""
            self._legend_add(
                self._fit_item,
                f"{self._curves[0].label} 1/f fit (α = {alpha:.3f}, {n} bins{mark})")

    # -- legend ----------------------------------------------------------------------------------
    def _drop_legend(self) -> None:
        """Empty + hide the legend. ``PlotItem.clear()`` leaves ``self.legend`` alive
        holding its rows, so without this every redraw's entries pile up; and
        ``LegendItem.setParentItem(None)`` raises ("Cannot anchor; parent is not
        set"), so detaching one is not an option."""
        legend = getattr(self._plot, "legend", None)
        if legend is None:
            return
        legend.clear()
        legend.setVisible(False)

    def _ensure_legend(self):
        """The legend, created if needed and re-inked for the current theme — it has
        to stay readable on the white publication canvas as well as the dark one."""
        fg = QtGui.QColor(self._fg_hex())
        backing = QtGui.QColor(self._bg_spec())
        backing.setAlpha(200)
        border = QtGui.QColor(fg)
        border.setAlpha(90)
        legend = self._plot.addLegend(offset=(-12, 12))
        legend.setLabelTextColor(fg)
        legend.setBrush(pg.mkBrush(backing))
        legend.setPen(pg.mkPen(border))
        legend.setVisible(True)
        return legend

    def _legend_add(self, item, label: str) -> None:
        legend = self._ensure_legend()
        legend.addItem(item, label)
        legend.setVisible(True)

    def legend_labels(self) -> list[str]:
        """The legend entry texts currently on the plot (empty when there is none)."""
        legend = getattr(self._plot, "legend", None)
        if legend is None:
            return []
        return [label.text for _sample, label in legend.items]

    # -- read-outs -------------------------------------------------------------------------------
    def _update_rms_readout(self) -> None:
        """The band RMS, PER CURVE — labelled with the band that was actually integrated.

        Merging two pores into one RMS figure would produce a number that describes
        neither, so with an overlay up the read-out names each curve and gives its
        own value in its own units.

        The band in brackets is :func:`~nanoscope.analysis.psd.band_support` of the
        spins — the SAME clip ``band_rms`` integrates over — printed at the precision
        the edges need, never rounded to integers: the old ``:.0f`` paired a genuine
        0–0.5 Hz figure with "(0–0 Hz)", a zero-width band the app itself defines as
        holding nothing, and a 9.5–10.5 Hz request with "(10–10 Hz)" (NE-03). When the
        clipped band IS empty, the label says so instead of dressing the honest 0.0 up
        as an integral over a band that held nothing."""
        if not self._curves:
            self._rms_label.setText("—")
            return
        lo, hi = self._rms_lo_spin.value(), self._rms_hi_spin.value()

        def value_text(c: PSDCurve) -> str:
            v = c.band_rms(lo, hi)
            if c.underflow and v == 0.0:
                # a live signal whose spectrum underflowed float64 — see PSDCurve.underflow
                return "below-representable"
            return _fmt_q(v, c.units)

        def band_text(c: PSDCurve) -> str:
            sup = psd_mod.band_support(c.freqs, lo, hi)
            if sup is None:
                return (f"({_fmt_band_edge(lo)}–{_fmt_band_edge(hi)} Hz — "
                        f"empty band, nothing integrated)")
            # The SAME (lo, hi) the integral used, printed at the spinboxes' own
            # resolution and never as one number twice — see _fmt_band_edges.
            a, b = _fmt_band_edges(sup[0], sup[1])
            return f"({a}–{b} Hz)"

        if len(self._curves) == 1:
            c = self._curves[0]
            self._rms_label.setText(f"{value_text(c)}   {band_text(c)}")
            return
        bands = {band_text(c) for c in self._curves}
        if len(bands) == 1:
            head = next(iter(bands))
            parts = [f"{c.label}: {value_text(c)}" for c in self._curves]
        else:
            # Curves at different sample rates clip the same request differently, so
            # the integrated band goes on each line — one shared header would misstate
            # every curve but one.
            head = f"(requested {_fmt_band_edge(lo)}–{_fmt_band_edge(hi)} Hz, clipped per curve)"
            parts = [f"{c.label}: {value_text(c)} {band_text(c)}" for c in self._curves]
        self._rms_label.setText(f"{head}\n" + "\n".join(parts))

    def _label_segment_clock(self) -> None:
        """Name the clock the two segment spins are in, on the spins themselves.

        ``clock``'s own rule is *one clock, or every control that shows or takes a
        time says which clock it is in*, and these spins TAKE one. The suffix stays
        the bare ``s`` -- a QDoubleSpinBox suffix sits inside the editable field and a
        sentence there is unusable -- so the naming goes in the tooltip, where
        :func:`clock.field_note` also spells out the arithmetic in this recording's
        own numbers ("subtract 10.2526 s from a number read off the trace").
        """
        offset = getattr(self, "_clock_offset_s", None)
        if offset is None:
            self._start_spin.setToolTip(_SEG_SPIN_TIP.format(edge="Start"))
            self._end_spin.setToolTip(_SEG_SPIN_TIP.format(edge="End"))
            return
        note = clock.field_note(clock.RECORDING, offset)
        self._start_spin.setToolTip(_SEG_SPIN_TIP.format(edge="Start") + "\n\n" + note)
        self._end_spin.setToolTip(_SEG_SPIN_TIP.format(edge="End") + "\n\n" + note)

    def _update_seg_label(self, i0: int, i1: int, n: int) -> None:
        """``N samples · D s`` plus, when the clocks differ, where that is on the trace.

        The read-out under the spins is where a user checks that the numbers they
        typed mean what they think. Stating the same span on the instrument clock
        turns "did I read this off the trace or off the table?" into something
        answerable without leaving the window.
        """
        fs = self._fs or 0.0
        dur = (i1 - i0) / fs if fs else 0.0
        text = f"{n:,} samples · {dur:.4g} s"
        offset = getattr(self, "_clock_offset_s", None)
        if fs and offset is not None and not clock.clocks_agree(offset):
            a = clock.to_instrument_s(offset, i0 / fs)
            b = clock.to_instrument_s(offset, i1 / fs)
            text += (f"\n{i0 / fs:.4f}–{i1 / fs:.4f} s into the recording "
                     f"(= {a:.4f}–{b:.4f} s on the trace's instrument clock)")
        self._seg_label.setText(text)

    def _fill_details(self) -> None:
        """The full noise report, PER CURVE — one block each, never merged."""
        if not self._curves:
            return
        blocks: list[str] = []
        for curve in self._curves:
            blocks.append(self._curve_details(curve, header=len(self._curves) > 1))
        text = "\n\n".join(blocks)
        notes = list(getattr(self, "_compute_notes", [])) + self._comparison_cautions()
        if notes:
            text += "\n\n" + "\n".join(f"NOTE  {n}" for n in notes)
        self._details.setPlainText(text)

    def _comparison_cautions(self) -> list[str]:
        """What makes two overlaid spectra NOT directly comparable in absolute terms.

        A PSD's y value is a squared current, so it inherits the amplifier gain — and
        this project's gain is uncalibrated by default. Two pores' spectra can always
        be compared in SHAPE (alpha, where the floor sits, where the capacitive rise
        starts); their absolute magnitudes can only be compared once they share a real
        calibration. Different y UNITS (a pA channel against an nA one) are worse
        still, because the axis is labelled with the primary's."""
        if len(self._curves) < 2:
            return []
        out: list[str] = []
        units = {c.units for c in self._curves}
        if len(units) > 1:
            named = ', '.join(sorted(_unit_or_unknown(u) for u in units))
            primary = _unit_or_unknown(self._units)
            out.append(
                f"these curves are in DIFFERENT units ({named}); the y axis "
                f"is labelled in the primary's ({primary}), so their heights are not "
                f"comparable as drawn. Calibrate them to a common unit first.")
        fps = []
        for curve in self._curves:
            try:
                fps.append(self._session.calibration(curve.doc_id, curve.channel).fingerprint())
            except (KeyError, TypeError, ValueError):
                fps.append(None)
        # `alternatives=()`: this window is looking at NOISE SPECTRA, and it does not
        # hold the detections' provenance, so it cannot vouch for an event column. The
        # default advice ("dwell_s and frac_depth (dI/I0) ARE comparable — plot those
        # instead") is the sentence the Plots window had to stop saying: frac_depth
        # divides BY the baseline, so it is only safe while the detections agree on
        # their baseline windows, and NOTHING here knows whether they do. Its own
        # alternative — the SHAPE of the spectrum — follows, and is one this surface
        # can actually stand behind.
        ok, why = gains_comparable(fps, alternatives=())
        if not ok:
            out.append(
                f"CAUTION — comparing absolute noise magnitude across recordings: {why} "
                f"The SHAPE of each spectrum (α, the white floor's position, the capacitive "
                f"rise) is unaffected and is comparable as drawn.")
        return out

    def _curve_details(self, curve: PSDCurve, *, header: bool) -> str:
        s = curve.summary
        u = curve.units
        lines: list[str] = []
        if header:
            lines += [f"── {curve.label} ──", ""]
        def q(val: float | None) -> str:
            # An underflowed curve's 0.0 is representation, not measurement (see
            # PSDCurve.underflow): the one thing this read-out may NOT do with it is
            # print the confident "0 pA" that is indistinguishable from a perfect pore.
            if curve.underflow and val == 0.0:
                return "below-representable (see note)"
            return _fmt_q(val, u)

        lines += [
            f"Total RMS noise:   {q(s['total_rms'])}",
            "",
            "Band RMS:",
        ]
        for label, val in s["bands"].items():
            lines.append(f"  {label:<16s} {q(val)}")
        wf_lo, wf_hi = s["white_floor_band"]
        lines += [
            "",
            f"White floor:       {_fmt_q(s['white_floor'], _sq_units(u))} "
            f"(median {wf_lo:.4g}–{wf_hi:.4g} Hz)",
        ]
        if not np.isfinite(s.get("total_rms", np.nan)):
            lines += [
                "",
                "The spectrum is not finite, so no RMS can be read from it. Welch averages "
                "segment periodograms, so a single NaN/inf sample in the recording makes "
                "every bin NaN.",
            ]
            # The calibration's own damage is named here, because the File info panel's
            # Issues box is filled from the raw arrays and would say "(none)".
            lines += ["", *curve.channel_issues] if curve.channel_issues else [
                "See the recording's issues in the File info panel."]
        elif curve.constant:
            # A literal, correct zero — and the most convincing-looking QC result there is.
            # `band_rms` cannot distinguish "no noise in this band" from "this channel is
            # not a measurement"; the panel can, because it still has the samples.
            level = "the same value"
            lines += [
                "",
                f"Every sample in the analysed window is {level}, so this 0 is arithmetic, "
                f"not a noise measurement. A channel that never moves is usually an unset "
                f"or wrong gain, a disconnected headstage, or a dead ADC channel — check "
                f"the Calibration panel's raw-vs-calibrated read-out.",
            ]
        elif curve.underflow:
            # The signal is LIVE — its noise is real, just below what a float64 spectrum
            # can represent — so the honest tiny number (or an explicit
            # "below-representable") is owed, never a bare confident zero (NE-07).
            measured = (
                f"its robust time-domain RMS is ≈ {curve.time_rms:.4g} "
                f"{_unit_or_unknown(u)}" if curve.time_rms > 0.0 else
                "its noise sits below even the robust time-domain estimate")
            lines += ["", *_wrap(
                f"NOTE  the signal in this window is live — {measured} — but fluctuations "
                f"this small square to below float64's smallest representable number "
                f"inside the Welch estimate, so every PSD bin reads exactly 0 and no RMS "
                f"can be integrated from the spectrum. 'Below-representable' above means "
                f"unmeasurably small HERE, never a quiet pore. Rescale the signal (e.g. a "
                f"larger calibration gain) to measure it.", indent=6)]
        lines += self._flicker_lines(curve)
        lines += [
            "",
            f"fs = {curve.fs:,.6g} Hz   ·   Nyquist = {s['nyquist_hz']:,.6g} Hz   ·   "
            f"Δf = {s['df_hz']:.4g} Hz",
            f"Welch: nperseg = {curve.nperseg}, noverlap = {curve.noverlap}, "
            f"segments ≈ {curve.nseg}, window = {self._window_combo.currentText()}",
        ]
        lines += self._welch_support_lines(curve)
        if not curve.units:
            # Every figure above is then a bare number in the file's own scale. Saying
            # "(units unknown)" beside each one is the honest label; saying WHY, once,
            # is what lets the user fix it -- and it is the same instruction the loader
            # note and the Calibration panel give for this state.
            lines += ["", *_wrap(
                "NOTE  this recording declares no units for the current channel and none "
                "have been set in Calibration, so every magnitude above is a bare number "
                "in the file's own scale, not a current in any named unit. Set the "
                "channel's units (and gain) in the Calibration panel to make these noise "
                "figures physical.", indent=6)]
        if curve.note:
            lines.append(f"NOTE  {curve.note}")
        return "\n".join(lines)

    # -- the 1/f read-out ------------------------------------------------------------------------
    def _flicker_lines(self, curve: PSDCurve) -> list[str]:
        """α, and WHAT IT RESTS ON.

        An exponent's whole credibility is the number of FFT bins under it, and R²
        cannot carry that: a straight line has two free parameters, so four points
        leave two residual degrees of freedom and an R² that is high almost by
        construction. This panel printed "α 1.207, R² 0.980" from exactly four bins —
        the fewest the fit accepts — with nothing on screen to say so (report §7,
        Medium-14). So the bin count now travels with every exponent this window
        prints, the ``marginal`` verdict comes straight from
        :func:`~nanoscope.analysis.psd.flicker_status` (never re-derived here), and
        an unavailable fit says how many bins it found instead of only that there
        were too few.
        """
        fl = curve.summary["flicker"]
        u = curve.units
        n, n_band = int(fl.get("n", 0)), int(fl.get("n_band", 0))
        lo, hi = fl.get("f_lo", 0.0), fl.get("f_hi", 0.0)
        df = curve.summary.get("df_hz", float("nan"))
        status = fl.get("status") or psd_mod.flicker_status(n)

        if not (n and np.isfinite(fl.get("alpha", np.nan))):
            if not (hi > lo):
                # The edges CROSSED: each was clipped on its own (the low edge up to
                # the fixed band / first bin, the high edge down to Nyquist) and the
                # clips emptied the band. A refusal must name a band that exists or
                # say WHY none does at this fs — never render the crossed pair with a
                # dash between the numbers, which printed "the 1–0.5 Hz flicker band"
                # at fs = 1 Hz and "the 3815–100 Hz flicker band" at 250 MHz (NE-04);
                # neither band exists, and each remedy below is the one that actually
                # helps its case.
                tgt_lo, tgt_hi = psd_mod.FLICKER_BAND_HZ
                nyq = float(curve.summary.get("nyquist_hz", float("nan")))
                # The bin count still travels with the refusal — "an unavailable fit
                # says how many bins it found" is this pane's standing rule, and here
                # the count is 0 WITH its reason.
                need = f"holds {_plural(0, 'FFT bin')} (this fit needs " \
                       f"{psd_mod.FLICKER_MIN_POINTS})"
                if np.isfinite(nyq) and nyq < tgt_lo:
                    return _wrap(
                        f"1/f exponent α:    unavailable — no flicker band exists at "
                        f"fs = {curve.fs:.4g} Hz: the fit's fixed {tgt_lo:.4g}–{tgt_hi:.4g} "
                        f"Hz band lies entirely above Nyquist ({nyq:.4g} Hz), so it "
                        f"{need}. No Welch setting can reach it — resolving flicker needs "
                        f"a sample rate above {2.0 * tgt_lo:.4g} Hz.",
                        indent=19)
                if np.isfinite(df) and df > tgt_hi:
                    return _wrap(
                        f"1/f exponent α:    unavailable — the frequency resolution "
                        f"Δf = {df:.4g} Hz is coarser than the whole {tgt_lo:.4g}–"
                        f"{tgt_hi:.4g} Hz flicker band, which therefore {need}. Raise the "
                        f"Welch segment length — Δf = fs/nperseg — until "
                        f"Δf ≤ {tgt_hi:.4g} Hz.",
                        indent=19)
                return _wrap(
                    f"1/f exponent α:    unavailable — the {tgt_lo:.4g}–{tgt_hi:.4g} Hz "
                    f"flicker band, clipped to this spectrum's own grid "
                    f"(Δf = {df:.4g} Hz) and Nyquist ({nyq:.4g} Hz), {need} at this "
                    f"sample rate.",
                    indent=19)
            found = n_band or n
            return _wrap(
                f"1/f exponent α:    unavailable — the {lo:.4g}–{hi:.4g} Hz flicker band "
                f"holds {_plural(found, 'FFT bin')} and this fit needs "
                f"{psd_mod.FLICKER_MIN_POINTS} (Δf = {df:.4g} Hz). Analyse a longer quiet "
                f"segment, or raise the Welch segment length — Δf = fs/nperseg, so a finer "
                f"grid needs more samples, not fewer.",
                indent=19)

        marginal = status == "marginal"
        out = [
            f"1/f exponent α:    {fl['alpha']:.3f}   "
            f"(magnitude at 1 Hz: {fl['A1Hz']:.4g} {_sq_units(u)}, R²={fl['r2']:.3f})",
            f"                   fit over {lo:.4g}–{hi:.4g} Hz on {_plural(n, 'FFT bin')}"
            + ("  ← MARGINAL" if marginal else ""),
        ]
        if marginal:
            floor_note = (", the fewest this fit accepts"
                          if n == psd_mod.FLICKER_MIN_POINTS else "")
            out += _wrap(
                f"CAUTION  α rests on {_plural(n, 'FFT bin')}{floor_note}, under the "
                f"{psd_mod.FLICKER_MARGINAL_POINTS} this window reads as enough. A log-log "
                f"line has two free parameters, so R² = {fl['r2']:.3f} over "
                f"{_plural(n, 'point')} is nearly free and is not evidence the power law "
                f"holds. Δf = {df:.4g} Hz is what makes the band this coarse: a longer quiet "
                f"segment (or a larger nperseg) is what refines it.",
                indent=9)
        return out

    def _welch_support_lines(self, curve: PSDCurve) -> list[str]:
        """Say when the analysed window is short for the Welch parameters in force.

        ``segments ≈ 1`` is printed above and reads like a setting, not a warning —
        setting nperseg = 65536 on a 1.4 s window silently turns the estimate into a
        single periodogram whose every bin carries ~100 % standard error, and the α
        and R² that come out of it still look like measurements (report §7 Medium-14;
        per-recording P8). The automatic choice aims for
        :data:`~nanoscope.analysis.psd.TARGET_SEGMENTS` averages, so falling under
        that is the honest trigger.
        """
        target = psd_mod.TARGET_SEGMENTS
        nseg = int(curve.nseg)
        if nseg >= target or nseg <= 0:
            return []
        if nseg == 1:
            head = (f"CAUTION  this is a single periodogram, not a Welch average: nperseg = "
                    f"{curve.nperseg:,} covers the whole {curve.n_samples:,}-sample window. ")
        else:
            head = (f"CAUTION  the {curve.n_samples:,}-sample window supports only ≈{nseg} "
                    f"Welch averages (the automatic segment length aims for {target}). ")
        return _wrap(
            head + "Each bin's standard error is ~1/√(averages), so the spectrum — and every "
                   "number read off it, α included — is correspondingly noisy. A longer quiet "
                   "segment, or a shorter Welch segment length, buys averages back.",
            indent=9)

    # -- style helpers ---------------------------------------------------------------------------
    def _fg_hex(self) -> str:
        return _FG_LIGHT if self._style.background == "light" else _FG_DARK

    def _bg_spec(self) -> str:
        return _BG_LIGHT if self._style.background == "light" else _BG_DARK

    def _msg_hex(self) -> str:
        return _MSG_LIGHT_HEX if self._style.background == "light" else _MSG_HEX

    def _label_style(self) -> dict:
        return {"color": self._fg_hex()}

    def _auto_title(self) -> str:
        """The drawn quantity plus WHICH recordings are on the plot.

        Naming only the primary while a second spectrum sits beside it is the kind of
        caption that survives into a slide and misattributes a curve, so an overlay
        names every curve (past three, it counts the rest)."""
        yq = "ASD" if self._asd_check.isChecked() else "PSD"
        labels = [c.label for c in self._curves]
        if not labels:
            target = self._current_target()
            if target is not None and self._session.has_document(target[0]):
                labels = [self._session.document(target[0]).display_name]
        if not labels:
            return yq
        if len(labels) > 3:
            shown = " vs ".join(labels[:3]) + f" (+{len(labels) - 3} more)"
        else:
            shown = " vs ".join(labels)
        return f"{yq} — {shown}"

    def _apply_title(self, auto: str | None) -> None:
        text = self._style.title or auto
        if not text:
            self._plot.setTitle(None)
            return
        self._plot.setTitle(text, color=self._fg_hex())

    def _apply_background(self) -> None:
        self._plot_widget.setBackground(self._bg_spec())
        pen = pg.mkPen(self._fg_hex())
        for name in ("bottom", "left"):
            ax = self._plot.getAxis(name)
            ax.setPen(pen)
            ax.setTextPen(pen)

    # -- plot state ------------------------------------------------------------------------------
    def _clear_plot(self) -> None:
        self._plot.clear()
        self._plot.setTitle(None)
        self._drop_legend()
        self._curve_item = self._fit_item = self._message_item = None
        self._plot_x = self._plot_y = None
        for curve in self._curves:
            curve.item = None

    def _message(self, text: str) -> None:
        self._clear_plot()
        self._curves = []
        self._freqs = self._psd = self._summary = None
        self._apply_background()
        self._plot.setLogMode(x=False, y=False)
        self._plot.setLabel("bottom", "")
        self._plot.setLabel("left", "")
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=self._msg_hex())
        self._plot.addItem(item)
        self._plot.setXRange(0, 1, padding=0)
        self._plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)
        self._message_item = item
        self._details.setPlainText(text)
        self._seg_label.setText("")
        # The other three views are looking at the same nothing, and a stale spectrogram
        # under a "no recording" spectrum is the worst of the four possible screens.
        self._clear_views(text)

    # -- the FFT / spectrogram / all-points views -------------------------------------------------
    #
    # All three read the SAME span object the Welch PSD does (:meth:`_span_for`), compute a
    # pure kernel from :mod:`nanoscope.analysis.psd` / :mod:`nanoscope.analysis.allpoints`,
    # and draw. None of them touches ``_curves``, ``_freqs``, ``_psd`` or ``_summary``, so
    # every existing PSD read-out, export and test sees exactly the numbers it saw before.
    def current_view(self) -> str:
        """Which of :data:`VIEWS` is on screen."""
        return str(self._view_combo.currentData() or VIEW_PSD)

    def set_view(self, view: str) -> None:
        """Switch views by key (:data:`VIEW_PSD` … :data:`VIEW_ALLPOINTS`)."""
        for i in range(self._view_combo.count()):
            if self._view_combo.itemData(i) == view:
                self._view_combo.setCurrentIndex(i)
                return
        raise ValueError(f"unknown view {view!r}; expected one of {[k for k, _ in VIEWS]}")

    def caption_text(self, view: str | None = None) -> str:
        """What the named view (default: the visible one) says about itself.

        Empty for :data:`VIEW_PSD`, which carries no caption box (its numbers are in the
        Noise details box, and a box under it would shorten the exported spectrum).
        """
        cap = self._captions.get(view or self.current_view())
        return cap.toPlainText() if cap is not None else ""

    def fft_spectrum(self) -> psd_mod.FFTSpectrum | None:
        """The last computed single-shot spectrum, or None."""
        return self._fft

    def spectrogram(self) -> psd_mod.Spectrogram | None:
        return self._spec

    def all_points(self) -> ap_mod.AllPointsHistogram | None:
        return self._hist

    def hist_levels(self) -> list[dict]:
        """The amplitude levels marked on the all-points histogram (may be empty)."""
        return list(self._hist_levels)

    def active_plot_item(self) -> pg.PlotItem:
        """The plot the visible view draws on — what *Save image…* exports."""
        return {VIEW_PSD: self._plot, VIEW_FFT: self._fft_plot,
                VIEW_SPECTROGRAM: self._spec_plot,
                VIEW_ALLPOINTS: self._hist_plot}[self.current_view()]

    def _on_view_changed(self, *_a) -> None:
        if self._updating:
            return
        self._refresh_view()
        self.workspaceChanged.emit()

    def _on_view_option_changed(self, *_a) -> None:
        if self._updating:
            return
        # A per-view option invalidates only its own view's cache: changing the histogram
        # bin count must not throw away a spectrogram that took 160 ms to compute.
        self._view_cache.pop(self.current_view(), None)
        self._refresh_view()
        self.workspaceChanged.emit()

    def _sync_view_enablement(self, view: str) -> None:
        """Grey the two Display toggles that do not describe the drawn picture.

        Nothing is hidden and nothing is re-laid-out — a disabled checkbox occupies the
        same pixels as an enabled one, so the Display group's neighbours never move. Only
        these two are touched: the Welch and RMS groups drive the Noise details box, which
        is on screen in ALL FOUR views, so greying them would be a lie about which control
        does what.
        """
        self._loglog_check.setEnabled(view in (VIEW_PSD, VIEW_FFT))
        self._asd_check.setEnabled(view == VIEW_PSD)

    def _clear_views(self, message: str) -> None:
        """Blank the three added views and say why, without disturbing the PSD state."""
        self._fft = self._spec = self._hist = None
        self._hist_levels = []
        self._view_cache.clear()
        self._fft_plot.clear()
        self._fft_item = None
        self._spec_image.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)
        self._clear_hist_marks()
        self._hist_plot.clear()
        self._hist_item = None
        for view in (VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS):
            self._captions[view].setPlainText(message)

    def _clear_hist_marks(self) -> None:
        for line in self._hist_marks:
            self._hist_plot.removeItem(line)
        self._hist_marks = []

    def _primary_rec(self) -> dict | None:
        target = self._current_target()
        if target is None:
            return None
        return next((r for r in self._recordings
                     if (r["doc_id"], r["channel"]) == target), None)

    def _refresh_view(self) -> None:
        """Show the selected view, computing it only if its inputs moved.

        Called at the end of every :meth:`_recompute` and on every view / view-option
        change. The PSD page returns immediately — it was already drawn by
        :meth:`_redraw`, and recomputing it here would double the cost of every keystroke.
        """
        view = self.current_view()
        index = next((i for i, (key, _) in enumerate(VIEWS) if key == view), 0)
        self._view_stack.setCurrentIndex(index)
        self._view_options.setCurrentIndex(index)
        self._sync_view_enablement(view)
        if view == VIEW_PSD:
            return

        rec = self._primary_rec()
        if rec is None:
            self._clear_views("No recording with a current channel and a known sample "
                              "rate.\nOpen one, then Refresh.")
            return
        span = self._span_for(rec)
        if isinstance(span, str):
            self._clear_views(span)
            return

        sig = self._view_signature(view, rec, span)
        cached = self._view_cache.get(view)
        payload = cached[1] if (cached is not None and cached[0] == sig) else None
        if view == VIEW_FFT:
            self._draw_fft(rec, span, payload)
        elif view == VIEW_SPECTROGRAM:
            self._draw_spectrogram(rec, span, payload)
        else:
            self._draw_allpoints(rec, span, payload)

    def _render_workspace_snapshot(self) -> None:
        """Draw the installed project snapshot without reading data or running kernels.

        The analytical-workspace adapter installs ``_curves`` and the three auxiliary
        result objects before calling this method.  Reopening a project must show those
        exact arrays even when the source recording has moved or is unavailable; only a
        deliberate Refresh or parameter edit is allowed to compute a replacement.
        """
        self._redraw()  # Stored PSD curves only; this path never calls Welch.

        view = self.current_view()
        index = next((i for i, (key, _) in enumerate(VIEWS) if key == view), 0)
        self._view_stack.setCurrentIndex(index)
        self._view_options.setCurrentIndex(index)
        self._sync_view_enablement(view)
        if view == VIEW_PSD:
            return

        head = self._curves[0] if self._curves else None
        live = self._primary_rec()
        label = (head.label if head is not None else
                 str(live.get("label")) if live is not None else "saved analysis")
        rec = live or {"label": label, "key": head.key if head is not None else ""}

        # The drawing helpers need only units and span metadata when a cached result is
        # supplied.  ``seg`` is deliberately empty: an accidental kernel call then fails a
        # focused test instead of silently analysing live/current project data.
        result = {VIEW_FFT: self._fft, VIEW_SPECTROGRAM: self._spec,
                  VIEW_ALLPOINTS: self._hist}.get(view)
        result_fs = float(getattr(result, "fs", 0.0) or 0.0)
        fs = float(head.fs if head is not None and head.fs else result_fs or self._fs or 1.0)
        i0 = int(head.i0 if head is not None else 0)
        n_samples = int(head.n_samples if head is not None else
                        getattr(result, "n_samples", 0) or 0)
        span = _Span(
            chan=SimpleNamespace(units=(head.units if head is not None else self._units)),
            seg=np.empty(0, dtype=np.float64), fs=fs, i0=i0, i1=i0 + n_samples,
            note=head.note if head is not None else "")

        if view == VIEW_FFT and isinstance(self._fft, psd_mod.FFTSpectrum):
            self._draw_fft(rec, span, self._fft)
        elif view == VIEW_SPECTROGRAM and isinstance(self._spec, psd_mod.Spectrogram):
            self._draw_spectrogram(rec, span, self._spec)
        elif view == VIEW_ALLPOINTS and isinstance(self._hist, ap_mod.AllPointsHistogram):
            saved_levels = list(self._hist_levels)
            self._draw_allpoints(rec, span, self._hist, restored_levels=saved_levels)
        else:
            # Do not call `_clear_views`: it would delete the other saved results and make
            # an unavailable recording lossy on the next project save.
            message = "No saved result is available for this view."
            if view == VIEW_FFT:
                self._fft_plot.clear()
                self._fft_item = None
                self._view_message(self._fft_plot, message)
                self._captions[VIEW_FFT].setPlainText(message)
            elif view == VIEW_SPECTROGRAM:
                self._spec_image.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)
                self._spec_image.setVisible(False)
                self._view_message(self._spec_plot, message)
                self._captions[VIEW_SPECTROGRAM].setPlainText(message)
            else:
                self._clear_hist_marks()
                self._hist_plot.clear()
                self._hist_item = None
                self._view_message(self._hist_plot, message)
                self._captions[VIEW_ALLPOINTS].setPlainText(message)

    def _view_signature(self, view: str, rec: dict, span: _Span) -> tuple:
        """Everything a view's KERNEL result depends on — the cache key.

        Style and colour map are deliberately absent: they change the drawing, not the
        numbers, so a colour-map change redraws from cache instead of re-transforming
        8 M samples.
        """
        # The calibration fingerprint is in the key because a gain edit changes the SAMPLES
        # under an unchanged span, and a cached view drawn from the old numbers would be
        # the same silent-stale-figure failure _reconfigure_for_recording documents.
        cal = self._session.calibration(rec["doc_id"], rec["channel"]).fingerprint()
        base = (rec["key"], span.i0, span.i1, span.fs, cal)
        if view == VIEW_FFT:
            return base + (self._window_combo.currentText(),)
        if view == VIEW_SPECTROGRAM:
            return base + (self._window_combo.currentText(),
                           self._spec_nperseg_combo.currentText())
        return base + (self._hist_bins_combo.currentText(),
                       self._hist_clip_check.isChecked())

    def _apply_view_background(self, widget: pg.PlotWidget, plot: pg.PlotItem) -> None:
        widget.setBackground(self._bg_spec())
        pen = pg.mkPen(self._fg_hex())
        for name in ("bottom", "left"):
            ax = plot.getAxis(name)
            ax.setPen(pen)
            ax.setTextPen(pen)

    def _view_message(self, plot: pg.PlotItem, text: str) -> None:
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=self._msg_hex())
        plot.addItem(item)
        plot.setXRange(0, 1, padding=0)
        plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)

    # -- (1) single-shot FFT ----------------------------------------------------------------------
    def _draw_fft(self, rec: dict, span: _Span, cached: object | None) -> None:
        """The unaveraged magnitude spectrum, drawn with its own honesty printed under it."""
        spec = cached if isinstance(cached, psd_mod.FFTSpectrum) else None
        if spec is None:
            try:
                spec = psd_mod.fft_magnitude(
                    span.seg, span.fs, window=self._window_combo.currentText())
            except ValueError as exc:
                self._clear_views(f"{rec['label']}: could not transform this span: {exc}")
                return
            self._view_cache[VIEW_FFT] = (self._view_signature(VIEW_FFT, rec, span), spec)
        self._fft = spec

        self._fft_plot.clear()
        self._fft_item = None
        log_log = self._loglog_check.isChecked()
        xs, ys = spec.freqs, spec.mag
        good = np.isfinite(xs) & np.isfinite(ys)
        if log_log:
            good &= (xs > 0.0) & (ys > 0.0)
        xs, ys = xs[good], ys[good]
        self._apply_view_background(self._fft_widget, self._fft_plot)
        if xs.size == 0:
            self._fft_plot.setLogMode(x=False, y=False)
            self._view_message(self._fft_plot, self._fft_empty_reason(spec, log_log))
        else:
            item = pg.PlotDataItem(x=xs, y=ys, pen=pg.mkPen(self._style.data_color, width=1))
            self._fft_plot.addItem(item)
            self._fft_item = item
            self._fft_plot.setLogMode(x=log_log, y=log_log)
        units = _unit_or_unknown(span.chan.units)
        self._fft_plot.setLabel("bottom", "Frequency (Hz)", **self._label_style())
        self._fft_plot.setLabel("left", f"|X(f)| — peak amplitude ({units})",
                                **self._label_style())
        self._fft_plot.setTitle(self._style.title or f"FFT — {rec['label']}",
                                color=self._fg_hex())
        self._fft_plot.showGrid(x=True, y=True, alpha=0.15)
        self._fft_plot.autoRange()
        self._captions[VIEW_FFT].setPlainText(self._fft_caption(spec, span))

    @staticmethod
    def _fft_empty_reason(spec: psd_mod.FFTSpectrum, log_log: bool) -> str:
        if spec.n_nonfinite:
            return (f"{spec.n_nonfinite:,} non-finite sample(s) in this span make every "
                    f"one of the {spec.n_bins:,} bins NaN.\nA single NaN poisons a whole "
                    f"transform — fix the span, not the axes.")
        return ("Nothing positive to draw"
                + (" on log-log axes." if log_log else "."))

    def _fft_caption(self, spec: psd_mod.FFTSpectrum, span: _Span) -> str:
        """WHAT IT IS, AND WHAT THE VIEW BESIDE IT IS BETTER AT.

        The spikier curve looks like the more informative one and it is not, so the
        comparison is stated on the surface with the two numbers that decide it — this
        view's ONE segment against the Welch view's ``nseg``.
        """
        dur = spec.n_samples / spec.fs
        nseg = self._nseg_used or 1
        lines = [
            f"Single-shot FFT: ONE {spec.window}-tapered transform of {spec.n_samples:,} "
            f"samples ({dur:.4g} s). Resolution {spec.df_hz:.4g} Hz over "
            f"{spec.n_bins:,} bins.",
        ]
        if spec.reduced:
            per_point = spec.n_bins / max(spec.freqs.size, 1)
            lines.append(
                f"Drawn as {spec.freqs.size:,} points: a per-bin PEAK envelope on a "
                f"log-frequency grid — each point is the LARGEST of the {per_point:.0f} raw "
                f"bins it stands for on average (far more per point at high frequency, "
                f"where the log grid is coarse). A narrow line therefore survives the "
                f"reduction at its true height (a mean or a decimation would lose it), and "
                f"each point carries the frequency of the bin its peak was in, so a line's "
                f"frequency is still exact.")
            lines.append(
                f"THE PRICE IS THE FLOOR: a bucket's MAXIMUM sits above the level its bins "
                f"average to, so the broadband grass here is drawn HIGHER than it is, and "
                f"by more the harsher the reduction — measured on Gaussian noise, the drawn "
                f"median floor is 1.7x the unreduced one at ~11 raw bins per point and 2.0x "
                f"at ~48. Read a NOISE FLOOR from the Welch PSD view, which averages; read "
                f"line frequencies and line heights here.")
        lines.append(
            f"THIS IS A PERIODOGRAM — an UNAVERAGED estimate. Its scatter never falls with "
            f"record length: about 52% of the mean on this magnitude axis (100% in power — "
            f"the chi-square-with-2-dof figure, which is quoted for the periodogram's POWER "
            f"and is not what this axis draws). More samples buy finer bins, never a "
            f"smoother curve. For noise characterisation the Welch PSD view is the better "
            f"tool ({nseg} averaged segment(s) here, so ~{nseg}x less variance). Use this "
            f"view to read a coherent line — mains and its harmonics — at the full 1/T "
            f"resolution, where a tone concentrates into one bin and does not average down.")
        if spec.n_nonfinite:
            lines.append(f"WARNING: {spec.n_nonfinite:,} non-finite sample(s) — every bin "
                         f"is NaN.")
        if span.note:
            lines.append(span.note)
        return "\n".join(lines)

    # -- (2) spectrogram --------------------------------------------------------------------------
    def _draw_spectrogram(self, rec: dict, span: _Span, cached: object | None) -> None:
        """PSD vs time as an image: the view that shows WHEN the noise changed."""
        sg = cached if isinstance(cached, psd_mod.Spectrogram) else None
        if sg is None:
            choice = self._spec_nperseg_combo.currentText()
            nps = None if choice == _AUTO else int(choice)
            try:
                sg = psd_mod.spectrogram_psd(
                    span.seg, span.fs, nperseg=nps,
                    window=self._window_combo.currentText(),
                    max_columns=_SPEC_MAX_COLUMNS)
            except ValueError as exc:
                self._clear_views(f"{rec['label']}: could not compute a spectrogram: {exc}")
                return
            self._view_cache[VIEW_SPECTROGRAM] = (
                self._view_signature(VIEW_SPECTROGRAM, rec, span), sg)
        self._spec = sg

        if self._spec_msg is not None:
            self._spec_plot.removeItem(self._spec_msg)
            self._spec_msg = None
        self._apply_view_background(self._spec_widget, self._spec_plot)
        self._spec_plot.setLogMode(x=False, y=False)   # the image rows are linear in f

        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log10(sg.psd)
        finite = np.isfinite(z)
        if not finite.any():
            self._spec_image.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)
            # A failed target must not retain the previous target's image chrome. Before
            # this branch hid nothing and returned before rewriting the title/ranges: after
            # switching good.dat -> an all-NaN bad.dat, the caption named the failure while
            # the plot still said "Spectrogram — good.dat" and kept good.dat's colour bar.
            self._spec_image.setVisible(False)
            if self._spec_bar is not None:
                self._spec_bar.setVisible(False)
            reason = self._fft_empty_reason(  # the same failure, said the same way
                psd_mod.FFTSpectrum(freqs=np.empty(0), mag=np.empty(0), fs=sg.fs,
                                    n_samples=sg.n_samples, n_bins=int(sg.psd.size),
                                    df_hz=0.0, window="", reduced=False,
                                    n_nonfinite=sg.n_nonfinite),
                False)
            self._spec_msg = pg.TextItem(
                reason, anchor=(0.5, 0.5), color=self._msg_hex())
            self._spec_plot.addItem(self._spec_msg)
            self._spec_plot.setXRange(0.0, 1.0, padding=0)
            self._spec_plot.setYRange(0.0, 1.0, padding=0)
            self._spec_msg.setPos(0.5, 0.5)
            self._spec_plot.setLabel("bottom", "Time (s from the start of this recording)",
                                     **self._label_style())
            self._spec_plot.setLabel("left", "Frequency (Hz)", **self._label_style())
            self._spec_plot.setTitle(self._style.title or f"Spectrogram — {rec['label']}",
                                     color=self._fg_hex())
            self._captions[VIEW_SPECTROGRAM].setPlainText(self._spec_caption(sg, span, None))
            return

        self._spec_image.setVisible(True)
        lo, hi = (float(v) for v in np.percentile(z[finite], [2.0, 99.8]))
        if hi <= lo:
            hi = lo + 1.0
        # col-major (pyqtgraph's default image axis order) wants image[x, y] = [time, freq].
        self._spec_image.setImage(np.asarray(z.T, dtype=np.float32), autoLevels=False)
        self._spec_image.setLevels((lo, hi))
        cmap = pg.colormap.get(self._spec_cmap_combo.currentText())
        self._spec_image.setColorMap(cmap)

        t0 = span.i0 / sg.fs                       # RECORDING seconds: the segment spins' clock
        dt = float(sg.times[1] - sg.times[0]) if sg.times.size > 1 else sg.nperseg / sg.fs
        df = float(sg.freqs[1] - sg.freqs[0]) if sg.freqs.size > 1 else sg.fs / 2.0
        self._spec_image.setRect(QRectF(
            t0 + float(sg.times[0]) - dt / 2.0, float(sg.freqs[0]) - df / 2.0,
            dt * sg.times.size, df * sg.freqs.size))

        if self._spec_bar is None:
            self._spec_bar = pg.ColorBarItem(values=(lo, hi), colorMap=cmap, interactive=False)
            self._spec_bar.setImageItem(self._spec_image, insert_in=self._spec_plot)
        else:
            self._spec_bar.setVisible(True)
            self._spec_bar.setColorMap(cmap)
            self._spec_bar.setLevels((lo, hi))
        self._spec_bar.getAxis("right").setLabel(f"log₁₀ PSD ({_sq_units(span.chan.units)})",
                                                 **self._label_style())
        self._spec_plot.setLabel("bottom", "Time (s from the start of this recording)",
                                 **self._label_style())
        self._spec_plot.setLabel("left", "Frequency (Hz)", **self._label_style())
        self._spec_plot.setTitle(self._style.title or f"Spectrogram — {rec['label']}",
                                 color=self._fg_hex())
        self._spec_plot.autoRange()
        self._captions[VIEW_SPECTROGRAM].setPlainText(self._spec_caption(sg, span, (lo, hi)))

    def _half_block_overlap(self) -> bool:
        """Is the PSD view hopping by half a block, the way the spectrogram always does?

        Compared as a FRACTION of that view's OWN block length: ``_noverlap_used`` is in
        samples, so testing it against the spectrogram's ``nperseg // 2`` calls a 50%
        Welch overlap a disagreement whenever the two block lengths differ — which is the
        usual case and exactly the state this caption is describing.
        """
        return self._nperseg_used > 0 and self._noverlap_used == self._nperseg_used // 2

    def _spec_match_hint(self, sg: psd_mod.Spectrogram) -> str:
        """How to make the two views comparable — naming only settings that EXIST.

        The two block-length combos are not the same list (this view stops at 16 384
        because the block length is the image height; the Welch combo starts at 1 024),
        so "set Time resolution to 65 536" would be advice the window cannot take. The
        hint names whichever side can actually be moved, and says so plainly when the
        current pair has no common value on offer.
        """
        n_psd, n_spec = self._nperseg_used, sg.nperseg
        ov = "" if self._half_block_overlap() else " with Overlap 50%"
        if str(n_psd) in _SPEC_NPERSEG_CHOICES:
            return (f"Set Time resolution to {n_psd:,}{ov} and the two agree bin for bin.")
        if str(n_spec) in _NPERSEG_CHOICES:
            ov2 = "" if self._half_block_overlap() else " and Overlap to 50%"
            return (f"Set the Welch view's Block length to {n_spec:,}{ov2} and the two "
                    f"agree bin for bin.")
        return (f"To compare them, put both on a block length the two combos share — "
                f"1024, 4096 or 16384: Time resolution offers "
                f"{', '.join(_SPEC_NPERSEG_CHOICES)} and cannot reach the PSD view's "
                f"current {n_psd:,}.")

    def _spec_caption(self, sg: psd_mod.Spectrogram, span: _Span,
                      levels: tuple[float, float] | None) -> str:
        """What this image is, and WHETHER it reduces to the PSD view at these settings.

        The bin-for-bin claim is conditional and the condition is checked here, not
        assumed: :func:`~nanoscope.analysis.psd.spectrogram_psd` IS Welch with the time
        average not taken, so the column mean equals the PSD view's curve exactly — but
        only at the same block length, the same 50% overlap and with no column merging.
        The two views ship with DIFFERENT block lengths on purpose (this one buys time
        resolution, that one buys variance), so at the shipped defaults the two spectra do
        not even have the same number of bins and the caption says so instead. Measured on
        a 250 000-sample synthetic: matched nperseg gives max |relative difference| 0.0
        across every bin; the shipped defaults give 257 bins against 8 193.

        The units clause is conditional for the same reason: the colour bar is always
        log₁₀ POWER, while the PSD view draws the ASD — the square root — whenever its
        "Amplitude spectral density" box is ticked.
        """
        t0, t1 = span.i0 / sg.fs, span.i1 / sg.fs
        units = _unit_or_unknown(span.chan.units)
        # Welch's own segment layout: same blocks, same taper, same 50% hop, one image
        # column per block. Any of those three differing breaks the equality.
        psd_bins = int(self._freqs.size) if self._freqs is not None else 0
        same_blocks = (self._nperseg_used == sg.nperseg
                       and self._half_block_overlap()
                       and sg.columns_merged == 1
                       and psd_bins > 0)
        lines = [
            f"{sg.freqs.size:,} frequency bins x {sg.times.size:,} time columns over "
            f"{t0:.4g}–{t1:.4g} s. One column = {sg.nperseg:,} samples "
            f"({sg.time_resolution_s * 1e3:.4g} ms); bin spacing "
            f"{sg.fs / sg.nperseg:.4g} Hz.",
        ]
        if same_blocks:
            lines.append(
                f"Same estimator AND the same blocks as the PSD view ({sg.nperseg:,} "
                f"samples, 50% overlap): averaging these columns returns that spectrum bin "
                f"for bin. What the average hides is WHEN: a burst, a pump, or a mains "
                f"change that starts partway through is a vertical or horizontal feature "
                f"here and a modest lift there.")
        elif psd_bins:
            why = []
            if self._nperseg_used != sg.nperseg:
                why.append(f"one column is {sg.nperseg:,} samples against that view's "
                           f"{self._nperseg_used:,}-sample Welch segment, so this image has "
                           f"{sg.freqs.size:,} frequency bins to its {psd_bins:,}")
            if not self._half_block_overlap():
                why.append(f"this view always hops by half a block, that one is set to "
                           f"{self._overlap_spin.value():g}% overlap")
            if sg.columns_merged > 1:
                why.append(f"the DRAWN columns are groups of {sg.columns_merged} blocks, and "
                           f"their unweighted mean is the block mean only when the last "
                           f"group is full")
            if self._nperseg_used == sg.nperseg and self._half_block_overlap():
                # The only thing between them is the merge, so "change the block length"
                # would send the reader to a combo that is already right.
                hint = (f"The blocks already match: it is the {sg.columns_merged}-way column "
                        f"merge that stands between them, so compare on a span short enough "
                        f"to stay under {_SPEC_MAX_COLUMNS} columns.")
            else:
                hint = self._spec_match_hint(sg)
                if sg.columns_merged > 1:
                    # Matching the blocks alone would not do it while the DRAWN columns are
                    # groups: the hint must not promise an agreement the merge can deny.
                    hint += (f" That also needs the columns unmerged — a span of fewer than "
                             f"{_SPEC_MAX_COLUMNS} blocks.")
            lines.append(
                f"Same estimator as the PSD view (Welch, {self._window_combo.currentText()} "
                f"taper, per-block detrend) but not the same settings, so averaging these "
                f"columns does NOT return that spectrum: " + "; ".join(why) + ". " + hint
                + f" What a spectrum hides either way is WHEN: a burst, a pump, "
                f"or a mains change that starts partway through is a vertical or horizontal "
                f"feature here and a modest lift there.")
        else:
            lines.append(
                f"Welch's estimator with the time average not taken — one column per "
                f"{sg.nperseg:,}-sample block. There is no PSD curve to compare it with "
                f"right now, so this caption makes no claim about the two agreeing.")
        if self._asd_check.isChecked():
            lines.append(
                f"UNITS DIFFER FROM THE PSD VIEW RIGHT NOW: colour is log₁₀ power in "
                f"{_sq_units(span.chan.units)}, while that view is drawing the ASD — its "
                f"square root — in {units}/√Hz.")
        if sg.columns_merged > 1:
            lines.append(
                f"Time columns were AVERAGED in groups of {sg.columns_merged} to fit "
                f"{_SPEC_MAX_COLUMNS} on screen — averaged, never dropped, so every sample "
                f"of the span is still on some pixel.")
        if levels is not None:
            lines.append(f"Colour spans log₁₀ PSD {levels[0]:.3g} … {levels[1]:.3g} "
                         f"(2nd–99.8th percentile of the drawn bins).")
        if sg.n_nonfinite:
            lines.append(f"WARNING: {sg.n_nonfinite:,} non-finite sample(s) in this span.")
        if span.note:
            lines.append(span.note)
        return "\n".join(lines)

    # -- (3) all-points histogram -------------------------------------------------------------
    def _draw_allpoints(self, rec: dict, span: _Span, cached: object | None,
                        *, restored_levels: list[dict] | None = None) -> None:
        """Counts per current level, log count axis — the levels before any detector."""
        hist = cached if isinstance(cached, ap_mod.AllPointsHistogram) else None
        if hist is None:
            try:
                hist = ap_mod.all_points_histogram(
                    np.asarray(span.seg), bins=int(self._hist_bins_combo.currentText()),
                    clip_quantile=_HIST_CLIP_Q if self._hist_clip_check.isChecked() else 0.0)
            except ValueError as exc:
                self._clear_views(f"{rec['label']}: {exc}")
                return
            self._view_cache[VIEW_ALLPOINTS] = (
                self._view_signature(VIEW_ALLPOINTS, rec, span), hist)
        self._hist = hist
        if restored_levels is not None:
            # Project restore owns the exact frozen peak report.  Re-running `level_peaks`
            # here could move markers under a newer algorithm and violates no-auto-run.
            self._hist_levels = [dict(level) for level in restored_levels]
        else:
            self._hist_levels = (ap_mod.level_peaks(hist)
                                 if self._hist_levels_check.isChecked() else [])

        self._clear_hist_marks()
        self._hist_plot.clear()
        self._hist_item = None
        self._apply_view_background(self._hist_widget, self._hist_plot)
        counts = hist.counts.astype(np.float64)
        # An EMPTY bin is drawn as a gap, not as a floor value: on a log count axis there
        # is no honest place to put a zero, and substituting 0.5 or 1 would invent samples
        # at a level the pore never visited. `connect="finite"` breaks the line instead.
        ys = np.where(counts > 0, counts, np.nan)
        item = pg.PlotDataItem(x=hist.centers, y=ys, connect="finite",
                               pen=pg.mkPen(self._style.data_color, width=1))
        self._hist_plot.addItem(item)
        self._hist_item = item
        self._hist_plot.setLogMode(x=False, y=True)
        for level in self._hist_levels:
            line = pg.InfiniteLine(
                pos=level["center"], angle=90,
                pen=pg.mkPen(self._style.fit_color, width=1, style=Qt.DashLine),
                label=f"{level['center']:.4g}", labelOpts={"color": self._style.fit_color,
                                                           "position": 0.9})
            self._hist_plot.addItem(line)
            self._hist_marks.append(line)
        units = _unit_or_unknown(span.chan.units)
        self._hist_plot.setLabel("bottom", f"Current ({units})", **self._label_style())
        self._hist_plot.setLabel("left", "Samples per bin (log)", **self._label_style())
        self._hist_plot.setTitle(self._style.title or f"All-points histogram — {rec['label']}",
                                 color=self._fg_hex())
        self._hist_plot.showGrid(x=True, y=True, alpha=0.15)
        self._hist_plot.autoRange()
        self._captions[VIEW_ALLPOINTS].setPlainText(self._hist_caption(hist, span))

    def _hist_caption(self, hist: ap_mod.AllPointsHistogram, span: _Span) -> str:
        units = _unit_or_unknown(span.chan.units)
        lines = [
            f"{hist.n_binned:,} of {hist.n_total:,} samples in {hist.counts.size:,} bins "
            f"of {hist.bin_width:.4g} {units} over {hist.lo:.6g}–{hist.hi:.6g} {units}. "
            f"No detection needed: this is every sample's VALUE.",
        ]
        if self._hist_levels:
            occ = _level_occupancy(hist, self._hist_levels)
            parts = [f"{lv['center']:.6g} {units} ({100.0 * f:.3g}%)"
                     for lv, f in zip(self._hist_levels, occ)]
            edges = _level_boundaries(self._hist_levels)
            where = ("the whole drawn range" if not edges else
                     "boundaries at " + ", ".join(f"{e:.6g} {units}" for e in edges))
            lines.append(
                "Levels, most populated first: " + "; ".join(parts) + ". Each percentage "
                f"INTEGRATES every bin nearer that level than any other ({where}) — the "
                f"half-amplitude rule — over the {hist.n_binned:,} binned samples, so the "
                f"shares sum to 100% and do not change when the bin count does. The gap "
                f"between two levels is the blockade depth in {units}.")
        else:
            lines.append(f"Mode {hist.mode_center:.6g} {units} "
                         f"({hist.mode_count:,} samples in one bin).")
        if hist.n_below or hist.n_above:
            lines.append(f"CLIPPED: {hist.n_below:,} sample(s) below and {hist.n_above:,} "
                         f"above the drawn range are counted here and NOT in the picture.")
        if hist.n_nonfinite:
            lines.append(f"{hist.n_nonfinite:,} non-finite sample(s) have no amplitude and "
                         f"are excluded from every bin rather than piled on an edge.")
        if span.note:
            lines.append(span.note)
        return "\n".join(lines)

    # -- export ----------------------------------------------------------------------------------
    def save_image(self, path: str | None = None) -> Path | None:
        """Export the VISIBLE view as a PNG (dialog when ``path`` is None).

        Returns the written path, or None when the user cancelled OR the write
        failed (a failed write is reported in a dialog, never returned as a path).
        """
        if path is None:
            chosen = ask_export_path(
                self, title="Save spectrum image", suggestion="noise_psd.png",
                filters="PNG image (*.png);;All files (*)")   # writes exactly this file
            if chosen is None:
                return None
            path = str(chosen)
        # The VISIBLE view, not always the PSD: pressing "Save image…" with a spectrogram on
        # screen and receiving a spectrum is the kind of surprise that ends up in a paper.
        #
        # The spectrogram exports its SCENE rather than its PlotItem, because the colour bar
        # is a sibling item in that scene: exporting the PlotItem alone rendered the bar
        # folded over the image's left edge with its ticks on top of the frequency axis
        # (measured at 640x480 — the axis showed two labels, "0" and "1000", for a 0-25 kHz
        # range). A spectrogram without a legible colour scale is not a saved spectrogram.
        if self.current_view() == VIEW_SPECTROGRAM:
            exporter = ImageExporter(self._spec_widget.scene())
        else:
            exporter = ImageExporter(self.active_plot_item())
        w, h = self._style.export_w, self._style.export_h
        params = exporter.parameters()
        if w > 0 and h > 0:
            params.param("width").setValue(w, blockSignal=exporter.widthChanged)
            params.param("height").setValue(h, blockSignal=exporter.heightChanged)
        elif w > 0:
            params["width"] = w
        elif h > 0:
            params["height"] = h
        out = Path(path)
        # STAGED, like every other export (USER_GUIDE §12). pyqtgraph's ImageExporter
        # ends in ``QImage.save``, which OPENS AND TRUNCATES its target and reports
        # failure by RETURNING False — writing ``out`` directly destroyed the previous
        # export for any format QImage cannot write and then said "The file was not
        # created". Rendering into a staged sibling costs the user nothing on failure,
        # which is what lets the message below say so truthfully (EXP3-06).
        from ..plot.figures import image_write_failure_text, save_plot_image

        try:
            notes = save_plot_image(out, lambda staged: exporter.export(str(staged)))
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Save spectrum image", image_write_failure_text(out, exc))
            return None
        # What the replace changed about the FILE (symlink followed, hard-linked names
        # left holding the previous contents) — unrecoverable after the fact, so it is
        # disclosed here the way File ▸ Save Project discloses it (EXP3-02).
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Save spectrum image", "\n\n".join(notes))
        self._details.setPlainText(self.details_text() + f"\n\nSaved image → {out.name}")
        return out
