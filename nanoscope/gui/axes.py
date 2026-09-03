"""A log axis whose tick labels can be READ. One implementation, every plot window.

THE DEFECT, AND THEN THE SECOND DEFECT
======================================
The report found the same log axis broken in two windows (§7 Medium-13 and High-7):

* labels overprinting into a smear (``4·10¹5·10⁶·10⁷·10⁸·10⁹·10¹00²`` across the whole
  PSD frequency axis);
* two notations on one axis (``0.01, 0.1, 1, 10¹, 10²`` -- so ``1`` and ``10¹`` sit two
  ticks apart looking like different KINDS of number);
* and a pyqtgraph automatic SI prefix inventing a ``(x0.001)`` multiplier that scaled
  every tick VALUE by 1000 on a log axis, where a multiplier means nothing at all.

Two agents then fixed it twice, differently, in two files. ``psd_panel`` dropped only
labels that would COLLIDE; ``plots_panel`` blanked every non-decade label. Blanking has
a failure the collision rule does not: **an axis zoomed inside one decade goes blank.**
Measured on the owner's -600 mV dwell histogram with x pinned to 15-25 ms -- an ordinary
zoom onto the median dwell of all three recordings -- the drawn axis carried ZERO tick
labels. The same axis at full extent painted each of its five decade labels TWICE, at
identical coordinates, because pyqtgraph emits the decades on more than one tick level
and nothing deduplicated them.

So the invariant lives here, and both windows subclass it. What is shared is
READABILITY -- no two labels touching, no invented multiplier, decades never lost to a
minor label. What is NOT shared is NOTATION: a 5-decade dwell axis reads better as
``0.01 0.1 1 10 100`` and a 10-decade spectrum reads better as ``10⁻⁵ … 10⁴``, and each
window chooses by overriding :meth:`ReadableLogAxis.format_decade` /
:meth:`ReadableLogAxis.format_minor`. One notation *per axis* is enforced here; which
notation is the window's business.

Imports nothing from the app: numpy, pyqtgraph, and this module's own arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg

__all__ = ["ReadableLogAxis", "ReadableLinearAxis", "SUPERSCRIPT",
           "exponent_decade", "decimal_decade"]

#: Unicode superscripts, so one decade label is one glyph run.
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def exponent_decade(e: float) -> str:
    """``-3`` -> ``10⁻³``."""
    return "10" + f"{int(round(e))}".translate(SUPERSCRIPT)


def decimal_decade(e: float) -> str:
    """``-3`` -> ``0.001``, ``2`` -> ``100``. Exact over the |e| <= 4 range it is used on."""
    n = int(round(e))
    return f"{10.0 ** n:.0f}" if n >= 0 else f"{10.0 ** n:.{-n}f}"


class ReadableLogAxis(pg.AxisItem):
    """A pyqtgraph axis that holds three properties instead of asserting them per draw.

    1. **No two drawn labels touch.** :meth:`generateDrawSpecs` post-filters the text
       specs, dropping any label whose (gap-padded) rectangle intersects one already
       kept. pyqtgraph's own crowding heuristic works per tick LEVEL, on the summed
       width of a whole level against the axis length, so a level that is legible over
       most of the axis and illegible inside each decade sails through it -- and it does
       nothing at all about the same decade being emitted on two levels and painted
       twice at identical coordinates.

    2. **Priority is by label form, not emission order.** Decades are placed first
       (:meth:`label_rank`), so a crowded axis loses ``9·10¹`` before it loses ``10²``.
       Necessary, not cosmetic: over ~3.6 decades pyqtgraph emits ONE flat level of all
       31 ticks, so "decades come first" is false as an ordering assumption -- naive
       left-to-right thinning keeps ``9·10¹`` and drops ``10²``, ``10³``, ``10⁴``.

    3. **No invented multiplier, and one notation.** The automatic SI prefix is off from
       construction: with the unit written into the label TEXT there is no unit string
       for it to attach to, so pyqtgraph appended ``(x0.001)`` and silently scaled every
       tick -- on a LOG axis. :attr:`display_factor` replaces it with a multiplier the
       WINDOW owns and sets together with the unit name, so the two cannot disagree.

    Linear mode holds the same three properties. It used to keep pyqtgraph's default
    ``tickStrings``, whose ``"%g"`` fallback (6 significant digits for any
    ``|value| >= 10000``) breaks both halves of (3) at once: a trace axis a day into a
    recording (t0 = 86,400 s) zoomed to a 100 µs window labelled EVERY tick ``86400``
    — six identical labels across the one surface whose job is saying WHERE you are
    (NE-05) — and a 10⁶ pA autorange printed ``800000`` beside ``1e+06`` on one axis
    (NE-06), the exact two-notations defect this class refuses in log mode. The
    :meth:`tickStrings` override below derives the label precision from the tick
    SPACING (the span being labelled), never from the tick magnitude, and chooses ONE
    notation for the whole axis.
    """

    #: Clearance in pixels required between two drawn labels ALONG the axis. Rectangles
    #: that merely fail to intersect still read as one string.
    LABEL_GAP_PX = 6.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: LOG-mode tick multiplier, e.g. 1e3 to show seconds as milliseconds without
        #: touching the plotted log10 values. 1.0 = "the data's own unit".
        #: ``AxisItem.setScale`` cannot do this on a log axis (``tickValues`` multiplies
        #: the log10 RANGE by it, which is arithmetic on an exponent).
        self.display_factor: float = 1.0
        self.enableAutoSIPrefix(False)

    def set_display_factor(self, factor: float) -> None:
        """Set the log-mode tick multiplier, redrawing if it changed."""
        f = float(factor)
        if f != self.display_factor:
            self.display_factor = f
            self.picture = None
            self.update()

    # -- notation: the subclass's choice, made once for the whole axis ---------------------------
    def choose_notation(self, exponents: list[float]) -> bool:
        """True to use the plain-decimal forms for THIS axis. Default: never.

        Called once per :meth:`logTickStrings` with the base-10 exponent of EVERY tick
        the axis is about to label, so the answer is a property of the axis rather than
        of each tick -- which is what stops ``0.01, 0.1, 1, 10¹, 10²`` happening again.

        Every tick, not only the decades: an axis zoomed inside one decade has no
        decade to decide from, and a rule that fell back to a default there would draw
        ``2·10¹`` on the same axis that draws ``20`` one scroll-wheel click away.
        """
        return False

    def format_decade(self, e: float, plain: bool) -> str:
        return decimal_decade(e) if plain else exponent_decade(e)

    def format_minor(self, mantissa: float, e: float, plain: bool) -> str:
        """A non-decade tick, e.g. ``2·10⁻³`` or ``0.002``.

        Minor labels are EMITTED and then thinned by collision, never blanked. Blanking
        is what empties an axis zoomed inside one decade.
        """
        if plain:
            # Fixed decimals chosen from the exponent, NOT ``%g``. ``%g`` would render a
            # minor tick at 2.5e4 as ``2.5e+04`` beside a decade written ``10000`` --
            # the mixed notation this class exists to prevent, reintroduced on the one
            # tick nobody looks at. It also rounds off the float noise pyqtgraph's tick
            # values carry (0.199986 -> 0.2).
            mant = round(float(mantissa), 2)
            digits = max(0, -int(e))
            return f"{mant * (10.0 ** e):.{digits}f}"
        return f"{mantissa:.3g}·{exponent_decade(e)}"

    def logTickStrings(self, values, scale, spacing):
        """Every tick labelled, in ONE notation, in the unit the axis label names."""
        vals: list[float] = []
        for v in values:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        # `scale` is pyqtgraph's own (autoSIPrefixScale * setScale); with the prefix off
        # and setScale(1.0) on a log axis it is 1.0. `display_factor` is the window's.
        factor = float(scale) * float(self.display_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            return ["" for _ in vals]
        shift = math.log10(factor)
        exponents = [v + shift for v in vals if np.isfinite(v)]
        plain = self.choose_notation(exponents)
        out: list[str] = []
        for v in vals:
            if not np.isfinite(v):
                out.append("")
                continue
            e = v + shift
            k = round(e)
            if abs(e - k) < 1e-6:
                out.append(self.format_decade(k, plain))
                continue
            floor_e = math.floor(e)
            out.append(self.format_minor(10.0 ** (e - floor_e), floor_e, plain))
        return out

    # -- linear mode: precision follows the SPAN, one notation per axis ---------------------------

    #: Most significant digits a plain fixed-notation label may carry before the whole
    #: axis switches to mantissa·10ⁿ. 14 keeps every plain label inside float64's exact
    #: range (~15-16 significant digits) with margin, and comfortably covers the real
    #: cases: ``86400.00002`` (a day-deep t0 under a 100 µs zoom) is 10, a 10⁶ pA
    #: autorange is 7.
    MAX_PLAIN_DIGITS = 14

    def tickStrings(self, values, scale, spacing):
        """Linear-mode labels whose precision follows the tick SPACING, in ONE notation.

        pyqtgraph's default formats each tick independently and falls to ``"%g"``
        (6 significant digits) whenever ``|value| >= 10000``, so the precision follows
        the MAGNITUDE of the tick instead of the span between ticks. Any zoomed view
        needing more than 6 significant digits then renders every tick as the same
        string — at t0 = 86,400 s a 100 µs window read ``86400`` six times over — and a
        single tick crossing 10⁶ flips to exponent notation beside its plain
        neighbours. Both are the failure class this module exists for; the digits come
        from ``spacing`` here (the smallest difference the axis must express), and the
        fixed-vs-``m·10ⁿ`` choice is made once from every tick at hand, exactly as
        :meth:`choose_notation` does for log mode.

        Log mode is dispatched to :meth:`logTickStrings` unchanged, as the base class
        does.
        """
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)

        factor = float(scale)
        vals: list[float] = []
        for v in values:
            try:
                vals.append(float(v) * factor)
            except (TypeError, ValueError):
                vals.append(float("nan"))
        step = abs(float(spacing) * factor)
        finite = [v for v in vals if math.isfinite(v)]
        if not finite or not (math.isfinite(step) and step > 0.0):
            # No finite ticks, or no spacing to derive a precision from. pyqtgraph
            # always passes a real positive spacing alongside its tick values, so this
            # is a guard against a degenerate direct call, not a drawn surface; float's
            # own shortest-repr is the honest fallback because inventing a span to
            # round to would be a guess.
            return [f"{v:g}" if math.isfinite(v) else "" for v in vals]

        # Decimals that RESOLVE one tick step: a 2e-5 s spacing needs 5 decimals
        # (86400.00000 / 86400.00002), a 2e5 pA spacing needs none. One more when the
        # step's own mantissa needs it — pyqtgraph emits 2.5×10ⁿ spacings, and rounding
        # a 0.25 step to 1 decimal would label two different ticks 0.2 and 0.5 apart
        # while claiming 0.3.
        places = max(0, -math.floor(math.log10(step)))
        scaled = step * 10.0 ** places
        if abs(scaled - round(scaled)) > 1e-9 * max(1.0, abs(scaled)):
            places += 1

        vmax = max(abs(v) for v in finite)
        int_digits = 1 if vmax < 1.0 else int(math.floor(math.log10(vmax))) + 1
        if int_digits + places <= self.MAX_PLAIN_DIGITS:
            out = []
            for v in vals:
                if not math.isfinite(v):
                    out.append("")
                    continue
                s = f"{v:.{places}f}"
                if float(s) == 0.0:
                    # never "-0.00": a value that rounds to zero at this precision IS
                    # zero, and a signed zero reads as a different kind of number.
                    s = f"{0.0:.{places}f}"
                out.append(s)
            return out

        # The plain form would need more significant digits than anyone can read at a
        # glance (an astronomically zoomed-out span, or a value beyond float64's exact
        # integers), so the WHOLE axis switches to mantissa·10ⁿ. Never per tick: a
        # per-tick switch is exactly the mixed notation the default reintroduced.
        step_e = math.floor(math.log10(step))
        out = []
        for v in vals:
            if not math.isfinite(v):
                out.append("")
            elif v == 0.0:
                out.append("0")   # zero has no exponent; the one conventional exception
            else:
                e = math.floor(math.log10(abs(v)))
                # Mantissa decimals that still resolve one tick step at this exponent.
                d = max(0, min(15, e - step_e))
                mant = round(v / 10.0 ** e, d)
                if abs(mant) >= 10.0:            # 9.99… rounded up a decade
                    mant /= 10.0
                    e += 1
                    d = max(0, min(15, e - step_e))
                out.append(f"{mant:.{d}f}·{exponent_decade(e)}")
        return out

    # -- readability: never two labels in the same place ------------------------------------------
    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs is None:
            return specs
        axis_spec, tick_specs, text_specs = specs
        return axis_spec, tick_specs, self.without_collisions(text_specs)

    @staticmethod
    def label_rank(text: str) -> int:
        """0 for a decade, 1 for a minor tick — the drop order.

        Read off the label's own VALUE, not its position in the emission order and not
        its glyphs, so it is correct for both notations this class can produce. A
        decade in exponent form is ``10ⁿ`` with no mantissa; in decimal form it is a
        number whose log10 is an integer.
        """
        s = str(text).strip()
        if not s:
            return 1
        if "·" in s:                                   # m·10ⁿ -- a minor by construction
            return 1
        if s.startswith("10"):
            body = s[2:]
            if body and all(ch in "⁻⁰¹²³⁴⁵⁶⁷⁸⁹" for ch in body):
                return 0
        try:
            x = float(s)
        except ValueError:
            return 1
        if x <= 0.0:
            return 1
        e = math.log10(x)
        return 0 if abs(e - round(e)) < 1e-9 else 1

    def without_collisions(self, text_specs):
        """Drop every label that would touch one already kept.

        Decades are placed first (see :meth:`label_rank`) and, within a rank, labels are
        considered in axis order, so what survives is the decade skeleton plus whatever
        minor labels genuinely fit between them. An axis zoomed inside a single decade
        has no decade to keep and fills with minors instead of going blank.

        Exact duplicates -- the same decade emitted on two tick levels and painted at
        identical coordinates -- collide with themselves and are dropped by the same
        rule, which is the whole of that half of the fix.
        """
        horizontal = self.orientation in ("bottom", "top")
        gap = float(self.LABEL_GAP_PX) / 2.0
        order = sorted(range(len(text_specs)),
                       key=lambda i: (self.label_rank(text_specs[i][2]), i))
        kept: set[int] = set()
        kept_rects: list = []
        for i in order:
            rect, _flags, text = text_specs[i]
            if not str(text).strip():
                continue          # a blank label draws nothing; it may not block a real one
            padded = (rect.adjusted(-gap, 0.0, gap, 0.0) if horizontal
                      else rect.adjusted(0.0, -gap, 0.0, gap))
            if any(padded.intersects(other) for other in kept_rects):
                continue
            kept.add(i)
            kept_rects.append(padded)
        return [spec for i, spec in enumerate(text_specs) if i in kept]


class ReadableLinearAxis(ReadableLogAxis):
    """A linear-only axis with the readable tick labels: the trace panes' axis.

    Nothing beyond :class:`ReadableLogAxis` — the span-aware, one-notation linear
    ``tickStrings``, the collision filter and the disabled SI prefix all live there —
    but a linear surface (the trace view's time and current axes, which are never put
    in log mode) should not have to subclass a class named "log" to get them. Pass
    instances to ``PlotItem``/``addPlot`` via ``axisItems``::

        axisItems={"bottom": ReadableLinearAxis(orientation="bottom"),
                   "left": ReadableLinearAxis(orientation="left")}
    """
