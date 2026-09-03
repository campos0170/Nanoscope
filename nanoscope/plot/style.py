"""Publication style for nanoscope figures: colours, sizes, rcParams.

Opt-in only. Importing this module changes nothing about your matplotlib --
that would be rude to notebook users who already have a style. To use it::

    with nanoscope_style():
        fig, ax = dwell_histogram(events)

The categorical palette is Okabe-Ito, which stays distinguishable under all
three common forms of colour-vision deficiency and in greyscale print. The
sequential map for density is viridis (perceptually uniform); do not swap in
jet/rainbow, which invents structure that isn't in the data.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import matplotlib as mpl

# Okabe & Ito (2008) colour-universal palette, reordered so the first few are
# the ones you actually reach for first. Every pair is separable under
# deuteranopia, protanopia and tritanopia.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

#: Named roles, so figure code says what it means instead of picking hex codes.
PALETTE: dict[str, str] = {
    "data": OKABE_ITO[0],
    "fit": OKABE_ITO[1],
    "accent": OKABE_ITO[2],
    "alt": OKABE_ITO[3],
    "warn": OKABE_ITO[4],
    "baseline": OKABE_ITO[1],
    "event": OKABE_ITO[0],
    "filter_limited": OKABE_ITO[1],
    "selected": OKABE_ITO[1],
    # ^^ THE ONE EVENT THE USER IS LOOKING AT -- the marker the Analysis panel puts on the
    # trace when an event is dived into. It was the literal ``#56B4E9`` in that panel, which
    # is this same entry spelled by hand; the panel now draws categorical CLASSES too (the
    # subpopulation spans), so a palette colour written out beside a call to the shared map is
    # exactly the drift ``gui.categorical`` exists to prevent, and an AST guard now fails on
    # it. Same ink as ``fit`` and ``baseline``, deliberately: nothing that draws a marker also
    # draws a fit line, and giving the role its own name is what lets the ink move later
    # without hunting for hexes.
    "excluded": "#9A9A9A",
    "grid": "#D8D8D8",
}

#: The fitted CUSUM sub-state levels' ink, everywhere a level is drawn (the main-trace
#: STEP overlay, the Event Inspector's segments, the gallery's tile bars). NOT an
#: Okabe-Ito member: it is the accent hue (#009E73) raised to full HSV value, because
#: the palette green that reads perfectly in a pane of its own disappears into a live
#: trace inside a dark pane — the hue keeps Okabe-Ito's colour-vision separation, the
#: value buys the contrast the trace needs. THIS is the one spelling; ``gui.overlays``
#: re-exports it as ``STEP_LEVEL_HEX`` and every GUI surface imports one of the two
#: names. It lives here rather than in ``gui.overlays`` because the Event Gallery is
#: architecturally barred from importing GUI siblings (see
#: ``test_the_gallery_still_reaches_no_sibling_panel``) but may reach ``plot.style``.
LEVEL_INK = "#24F0B8"

#: The estimated-baseline rail's default ink.  This lives below ``gui`` for the
#: same architectural reason as :data:`LEVEL_INK`: both the trace view and the
#: overlay renderer need the value, but neither GUI sibling should import the
#: other merely to share a colour.  White was selected from a real -1000 mV
#: recording: before the rail was drawn the dark trace pane contained no pixels
#: within 24/255 of white, and white provides 18.6:1 contrast against ``#111316``.
#: ``gui.overlays`` re-exports this as ``BASELINE_TRACK_HEX`` for compatibility.
BASELINE_TRACK_INK = "#FFFFFF"

#: Perceptually-uniform sequential map for 2-D density / counts.
SEQUENTIAL_CMAP = "viridis"

# Journal column widths in inches. Nature/Science single ~3.35", double ~6.9".
SINGLE_COLUMN_IN = 3.35
DOUBLE_COLUMN_IN = 6.9


def color(i: int) -> str:
    """Categorical colour `i`, cycling through the Okabe-Ito palette."""
    return OKABE_ITO[int(i) % len(OKABE_ITO)]


def figure_size(width: str | float = "single", aspect: float = 0.72) -> tuple[float, float]:
    """Figure size in inches.

    Args:
        width: 'single' (3.35"), 'double' (6.9"), or an explicit width in inches.
        aspect: height / width.

    Returns:
        (width_in, height_in).
    """
    if isinstance(width, str):
        try:
            w = {"single": SINGLE_COLUMN_IN, "double": DOUBLE_COLUMN_IN}[width]
        except KeyError:
            raise ValueError(f"width must be 'single', 'double' or a number in inches, got {width!r}") from None
    else:
        w = float(width)
    if w <= 0 or aspect <= 0:
        raise ValueError(f"width and aspect must be positive, got width={w}, aspect={aspect}")
    return (w, w * float(aspect))


def rc_params(**overrides: Any) -> dict[str, Any]:
    """The nanoscope rcParams as a plain dict (merge it into your own style).

    fonttype 42 embeds real TrueType glyphs in PDF/PS instead of drawing type-3
    outlines, which is what makes the text editable in Illustrator/Inkscape --
    the difference between a figure a reviewer can ask you to retouch and a
    figure you have to regenerate.
    """
    rc: dict[str, Any] = {
        "figure.figsize": figure_size("single"),
        "figure.dpi": 110,
        "figure.autolayout": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "axes.titleweight": "regular",
        "axes.labelpad": 2.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO)),
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.0,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.3,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "lines.solid_capstyle": "round",
        "image.cmap": SEQUENTIAL_CMAP,
        "errorbar.capsize": 2.0,
    }
    rc.update(overrides)
    return rc


@contextmanager
def nanoscope_style(**overrides: Any) -> Iterator[None]:
    """Context manager applying the nanoscope publication style.

    Scoped: rcParams are restored on exit, so this never leaks into a user's
    session. All figure functions in `nanoscope.plot.figures` already wrap
    themselves in this, so you only need it for your own axes.

    Example:
        >>> with nanoscope_style(**{'font.size': 10}):
        ...     fig, ax = plt.subplots()
    """
    with mpl.rc_context(rc_params(**overrides)):
        yield
