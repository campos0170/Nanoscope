"""Sweep 7 fix-up — the annotation box must not be printed over the data it judges.

The sweep-7 fix grew the box on the exported histograms from 2-3 lines to 6-7 ("KS
D = 0.511 > 0.047 (1%): fit REJECTED", "— parameters are not measurements", "open-pore
baseline not certified", the pointer), and left it anchored at the top-right corner with
no background. On the owner's −700 mV recording (1,179 events) the added lines ran
through the peak bars of both histograms, and the scatter's box began LEFT of the y
spine (A2-NEW-1, X-SURF-1) — the verdict about a curve, printed across the curve.

Everything here is measured the way the reader sees it: at the dpi ``save_figure``
exports at, on the box's DRAWN footprint (its glyphs plus the white background patch),
against bar tops and hexbin cells taken off ``ax.transData`` rather than off the
placer's own skyline — a test that asked the placer where the data is would agree with
it by construction.

Three separate things have to hold, and each was broken on its own:

* the box's footprint clears the bars (the placer: corner choice + headroom);
* the box is DRAWN with the font it was MEASURED with -- a figure is built inside
  ``nanoscope_style`` (Helvetica) and saved outside it (DejaVu Sans), so an unpinned
  box drew 8 % more ink than its layout reserved and overshot the right spine;
* the baseline pointer names a file that exists (X-SURF-2): Export figure… writes
  ``fig_<name>.pdf`` + ``.png`` + ``.csv``, and that CSV is the figure's own source
  data, not an advisory.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from nanoscope.plot import figures
from nanoscope.plot.style import nanoscope_style

T_MIN = 20e-6

#: The dpi ``save_figure`` renders at. Read through ``getattr`` so that swapping in a
#: pre-fix ``figures.py`` (which has no such constant) still fails these tests on the
#: geometry they are about, rather than erroring on a missing name.
EXPORT_DPI = float(getattr(figures, "_EXPORT_DPI", 300.0))

NOT_CERTIFIED = ("The open pore here is NOT CERTIFIED, only chosen. This recording has "
                 "TWO conductance levels: the tracked one at -34.6 pA, which holds 90% "
                 "of the recording, and another at -120.7 pA holding 5%. ...")


class _Events:
    """The duck ``figures._unpack`` eats: ``.df`` + ``.provenance``."""

    def __init__(self, df: pd.DataFrame, provenance: dict):
        self.df, self.provenance = df, provenance


def _owner_shaped_events(n: int = 1179, seed: int = 7) -> _Events:
    """An event set shaped like the owner's −700 mV apoferritin run.

    The two properties that put the box on the data are both here: the dwell mass sits
    at 12-26 ms on a log axis whose left edge is the 20 µs detection limit (so the
    histogram is a spike in the RIGHT third — the pre-sweep top-right corner), and
    ΔI/I₀ is an ENHANCEMENT centred near 2 whose bars span the full width of their own
    axes (so no corner is free there and the box has to make headroom instead).
    """
    rng = np.random.default_rng(seed)
    dwell = np.concatenate([rng.uniform(1.0e-4, 3.5e-3, n // 8),
                            rng.normal(1.9e-2, 3.0e-3, n - n // 8)])
    dwell = np.clip(dwell, T_MIN * 1.5, None)
    depth = np.concatenate([rng.normal(0.45, 0.05, n // 8),
                            rng.normal(2.1, 0.35, n - n // 8)])
    df = pd.DataFrame({
        "t_start_s": np.sort(rng.uniform(0.0, 9.0, n)),
        "dwell_s": dwell,
        "frac_depth": np.abs(depth),
        "filter_limited": np.zeros(n, dtype=bool),
        "accepted": np.ones(n, dtype=bool),
    })
    return _Events(df, {"analyzed_duration_s": 10.0, "min_dwell_s": T_MIN,
                        "baseline_warnings": [NOT_CERTIFIED]})


# ------------------------------------------------------------------------------------
# geometry, measured off the figure and not off the placer
# ------------------------------------------------------------------------------------
def _box_of(ax):
    """The annotation box on ``ax`` — the one text that carries the fitted numbers."""
    for t in ax.texts:
        s = t.get_text()
        if "n = " in s or "KS D" in s or "median" in s:
            return t
    return None


def _footprint(fig, ax, box) -> tuple[float, float, float, float]:
    """``(x0, x1, y0, y1)`` of the box AS DRAWN, in axes fractions, at the export dpi.

    The white background patch counts: it is opaque enough (alpha 0.85) that whatever it
    covers is no longer the figure's ink. Its pad is a multiple of the font size, which
    is how the box itself computes it, and matched the drawn patch to 0.01 px.
    """
    fig.set_dpi(EXPORT_DPI)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.get_window_extent(r)
    tb = box.get_window_extent(r)
    pad = 0.0
    patch = box.get_bbox_patch()
    if patch is not None:
        pad = float(patch.get_boxstyle().pad) * float(box.get_fontsize()) * fig.dpi / 72.0
    return ((tb.x0 - pad - ab.x0) / ab.width, (tb.x1 + pad - ab.x0) / ab.width,
            (tb.y0 - pad - ab.y0) / ab.height, (tb.y1 + pad - ab.y0) / ab.height)


def _bars(ax) -> list[tuple[float, float, float]]:
    """``(x_left, x_right, top)`` per non-empty histogram bar, in axes fractions."""
    from matplotlib.patches import StepPatch

    out = []
    ab = ax.get_window_extent(ax.figure.canvas.get_renderer())
    for p in ax.patches:
        if not isinstance(p, StepPatch):
            continue
        vals = np.asarray(p._values, dtype=float)
        edges = np.asarray(p._edges, dtype=float)
        for i, v in enumerate(vals):
            if not np.isfinite(v) or v <= 0:
                continue
            (lx, _), (rx, ty) = ax.transData.transform([(edges[i], 0.0), (edges[i + 1], v)])
            out.append(((lx - ab.x0) / ab.width, (rx - ab.x0) / ab.width,
                        (ty - ab.y0) / ab.height))
    return out


def _overlapping_bars(fig, ax) -> list[tuple[float, float, float]]:
    box = _box_of(ax)
    assert box is not None, "the figure has no annotation box to place"
    x0, x1, y0, _ = _footprint(fig, ax, box)
    return [b for b in _bars(ax) if b[1] > x0 and b[0] < x1 and b[2] > y0]


# ------------------------------------------------------------------------------------
# the box clears the data
# ------------------------------------------------------------------------------------
def test_no_line_of_the_dwell_box_is_printed_across_a_histogram_bar():
    ev = _owner_shaped_events()
    fig, ax = figures.dwell_histogram(ev)
    over = _overlapping_bars(fig, ax)
    assert not over, f"{len(over)} bar(s) reach into the box: {over[:3]}"


def test_no_line_of_the_depth_box_is_printed_across_a_histogram_bar():
    ev = _owner_shaped_events()
    fig, ax = figures.depth_histogram(ev)
    over = _overlapping_bars(fig, ax)
    assert not over, f"{len(over)} bar(s) reach into the box: {over[:3]}"


def test_the_headroom_the_box_takes_leaves_the_bars_their_own_counts():
    """Placement may change the SCALE the bars are drawn on, never the bars.

    The depth histogram's bars span its full width, so the box cannot step sideways and
    the y limit is raised instead. That is a change to the axis, and this pins that it
    is ONLY a change to the axis: the same counts, the same edges, and the same source
    data the CSV sidecar is written from.
    """
    ev = _owner_shaped_events()
    fig_a, ax_a = figures.depth_histogram(ev)
    fig_b, ax_b = figures.depth_histogram(ev, annotate=False)
    data_a = getattr(fig_a, figures._DATA_ATTR)
    data_b = getattr(fig_b, figures._DATA_ATTR)
    pd.testing.assert_frame_equal(data_a, data_b)
    from matplotlib.patches import StepPatch
    a = [p for p in ax_a.patches if isinstance(p, StepPatch)][0]
    b = [p for p in ax_b.patches if isinstance(p, StepPatch)][0]
    assert np.array_equal(np.asarray(a._values), np.asarray(b._values))
    assert np.array_equal(np.asarray(a._edges), np.asarray(b._edges))
    assert ax_a.get_ylim()[1] >= ax_b.get_ylim()[1], "headroom only ever raises the top"


def test_the_scatter_box_stays_inside_the_axes_and_off_the_hexbin_cloud():
    """The top-right corner is NOT the sparse corner for an enhancement population.

    The scatter's docstring used to assume it was — true for a translocation cloud
    (dwell rising right, depth rising up), false for the owner's, whose cloud sits at
    ΔI/I₀ 1.5-3 over 12-26 ms, i.e. exactly under that corner. The box's widest line,
    "(ENHANCEMENT: > 1, not a blockade)", is also wider than the scatter's main axes,
    which is how it came to start left of the y spine.
    """
    ev = _owner_shaped_events()
    fig, ax = figures.scatter_dwell_depth(ev)
    box = _box_of(ax)
    assert box is not None
    x0, x1, y0, y1 = _footprint(fig, ax, box)
    assert x0 >= 0.0 and x1 <= 1.0, f"box spans x[{x0:.3f},{x1:.3f}] of the axes"
    assert y0 >= 0.0 and y1 <= 1.0, f"box spans y[{y0:.3f},{y1:.3f}] of the axes"
    ab = ax.get_window_extent(fig.canvas.get_renderer())
    inside = 0
    for coll in ax.collections:
        offs = np.asarray(coll.get_offsets(), dtype=float)
        if offs.size == 0:
            continue
        px = coll.get_transform().get_affine().transform(offs.reshape(-1, 2))
        fx = (px[:, 0] - ab.x0) / ab.width
        fy = (px[:, 1] - ab.y0) / ab.height
        inside += int(np.count_nonzero((fx > x0) & (fx < x1) & (fy > y0) & (fy < y1)))
    assert inside == 0, f"{inside} hexbin cell(s) under the box"


# ------------------------------------------------------------------------------------
# the box is drawn with the font it was measured with
# ------------------------------------------------------------------------------------
def test_the_box_draws_no_more_ink_than_the_layout_it_was_placed_by_reserved(tmp_path):
    """Layout and glyphs must come from one font, or placing the box means nothing.

    ``figures`` builds inside ``nanoscope_style``, where ``font.sans-serif`` resolves to
    Helvetica, and the exporter calls ``save_figure`` outside it, where the same
    unresolved family resolves to DejaVu Sans. The Text keeps the WORD "sans-serif", so
    it is laid out in one and printed in the other: measured at 300 dpi on one verdict
    line, 0.718 of the axes reserved against 0.778 drawn, ending at x = 1.034 — past the
    right spine, on a box the placer had just put inside it.
    """
    ev = _owner_shaped_events()
    with nanoscope_style():                        # …measured where the placer measures it
        fig, ax = figures.dwell_histogram(ev)
        box = _box_of(ax)
        assert box is not None
        fig.set_dpi(EXPORT_DPI)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        ab = ax.get_window_extent(r)
        tb = box.get_window_extent(r)
        laid_out = tb.width / ab.width

    png = tmp_path / "dwell.png"
    fig.set_layout_engine("none")                  # …and rendered outside it, as the app does
    fig.savefig(png, dpi=EXPORT_DPI, bbox_inches=None)
    img = mpimg.imread(png)[..., :3]
    # the box's own colour, and the only ink of it above the histogram's mass
    ink = (img[..., 0] > 0.6) & (img[..., 1] < 0.75) & (img[..., 2] < 0.55)
    ys, xs = np.nonzero(ink)
    top = ys < img.shape[0] * 0.45
    assert top.any(), "no box ink found in the exported PNG"
    drawn = (xs[top].max() - xs[top].min()) / ab.width
    assert drawn <= laid_out * 1.02, (
        f"the box drew {drawn:.3f} of the axes where its layout reserved {laid_out:.3f}")


def test_the_box_font_is_resolved_to_a_file_so_it_cannot_change_at_draw_time():
    """The source-level half of the test above: one resolver, and it returns a path."""
    ev = _owner_shaped_events()
    _, ax = figures.dwell_histogram(ev)
    box = _box_of(ax)
    assert box is not None
    assert box.get_fontproperties().get_file() is not None, \
        "the box's family is still the unresolved word, so findfont runs again at draw time"


# ------------------------------------------------------------------------------------
# X-SURF-2 — the pointer names something the reader can open
# ------------------------------------------------------------------------------------
def test_the_baseline_caveat_does_not_point_at_a_sidecar_the_figure_export_never_writes(tmp_path):
    ev = _owner_shaped_events()
    fig, ax = figures.dwell_histogram(ev)
    text = "\n".join(t.get_text() for t in ax.texts)
    assert "open-pore baseline not certified" in text
    assert "events CSV sidecar" not in text, \
        "the figure still points at an events CSV that Export figure… does not write"

    figures.save_figure(fig, tmp_path / "fig_dwell")
    wrote = sorted(p.name for p in tmp_path.iterdir())
    assert wrote == ["fig_dwell.csv", "fig_dwell.pdf", "fig_dwell.png"], wrote
    body = (tmp_path / "fig_dwell.csv").read_text()
    assert "CERTIFIED" not in body.upper(), \
        "the CSV beside the figure is its source data; nothing may imply it carries the advisory"
    assert "Export events" in text, "the pointer must name the command that writes the paragraph"


def test_both_baseline_caveats_point_at_the_same_place():
    """One spelling of the pointer, so the certified and the counted line cannot drift."""
    certified = figures._baseline_caveats({"baseline_warnings": [NOT_CERTIFIED]})
    counted = figures._baseline_caveats({"baseline_warnings": ["drift on this segment"]})
    assert len(certified) == 1 and len(counted) == 1
    pointer = getattr(figures, "_ADVISORY_POINTER", "")
    assert pointer, "the two caveat lines must share one spelling of the pointer"
    assert certified[0].endswith(pointer)
    assert counted[0].endswith(pointer)
    assert "sidecar" not in pointer


def test_the_pointer_fits_the_box_without_being_folded():
    """A folded pointer reads as two half-sentences; it is written under the budget."""
    line = figures._ADVISORY_POINTER
    assert len(line) <= figures._BOX_LINE_CHARS
    assert figures._wrap_line(line) == line
