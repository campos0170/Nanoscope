"""Tests for the batched overlay renderers (gui/overlays.py, PORT_PLAN §5).

The anchor assertion: N event spans render as exactly ONE SpanOverlayItem, NOT N
LinearRegionItems. Also checks the int64-index -> x mapping, the hline/marker
renderers, registration wiring, and the TraceView.add_overlay/remove_overlay
lifecycle end to end. Driven headless (offscreen).
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.core.analysis import OverlayHint, OverlayKind
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.overlays import (
    STEP_LEVEL_HEX,
    SpanOverlayItem,
    StepOverlayItem,
    register_all_renderers,
    render_hline,
    render_marker,
    render_spans,
    render_step,
)
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
N = 200_000


# ================================================================================================
# Fixtures: a synthetic single-current-channel document wired into a Session + TraceView.
# ================================================================================================

def _make_document() -> TraceDocument:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(N).astype(np.float32)
    info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=Path("/tmp/synthetic_overlays.dat"),
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(data,))


@pytest.fixture
def session_view(qapp):
    doc = _make_document()
    session = Session()
    session.add_document(doc)
    session.add_layer(doc.doc_id, 0)
    view = TraceView(session)
    view.resize(1200, 600)   # a real extent so the viewbox has a range to clip against
    return session, view, doc.doc_id


@pytest.fixture(autouse=True)
def _renderers_registered(qapp):
    # register_all_renderers is idempotent; ensure it ran even if the qapp guard skipped it.
    register_all_renderers()


def _span_hint(starts, stops, **kw) -> OverlayHint:
    return OverlayHint(
        kind=OverlayKind.SPAN,
        starts=np.asarray(starts, dtype=np.int64),
        stops=np.asarray(stops, dtype=np.int64),
        **kw,
    )


# ================================================================================================
# render_spans: N events -> ONE SpanOverlayItem (the anti-10k-LinearRegionItem contract).
# ================================================================================================

def test_span_overlay_does_not_drive_y_autorange(session_view):
    """Regression: a SpanOverlayItem paints full-pane-height spans that track the view, so its
    boundingRect equals the current view range. If it participated in autorange, every autorange
    pass would union an ever-larger view rect and ratchet the y-axis wide (observed: a pA pane going
    from ~[-320,8] to ±50000), crushing the signal to a flat line. It must be added ignoreBounds=True.
    """
    session, view, doc_id = session_view
    pane = view.panes()[0]
    vb = pane.plot.vb
    vb.enableAutoRange(y=True)
    view.add_overlay(doc_id, 0, _span_hint(np.linspace(100, N - 100, 1500).astype(np.int64),
                                           np.linspace(100, N - 100, 1500).astype(np.int64) + 20))
    y0 = tuple(vb.viewRange()[1])
    for _ in range(4):                       # repeated autoranges must NOT ratchet the range wider
        vb.enableAutoRange(y=True)
        vb.updateAutoRange()
    y1 = tuple(vb.viewRange()[1])
    assert y1 == pytest.approx(y0, rel=0.05), f"span overlay expanded y-autorange: {y0} -> {y1}"
    # The signal is ~unit-normal (std 1); the pane must still be at signal scale, not blown up.
    assert abs(y1[0]) < 20 and abs(y1[1]) < 20, f"y-axis blown up by overlay: {y1}"


def test_render_spans_returns_single_batched_item(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    starts = np.arange(0, 100_000, 10_000, dtype=np.int64)   # 10 events
    stops = starts + 2_000
    items = render_spans(plot, session, doc_id, _span_hint(starts, stops))

    assert len(items) == 1, "spans must batch into ONE item, not one per event"
    item = items[0]
    assert isinstance(item, SpanOverlayItem)
    assert item.n_spans == len(starts)


def test_render_spans_never_creates_linearregionitems(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    starts = np.arange(0, 150_000, 1_000, dtype=np.int64)    # 150 events
    stops = starts + 200
    before = [it for it in plot.items if isinstance(it, pg.LinearRegionItem)]
    render_spans(plot, session, doc_id, _span_hint(starts, stops))
    after = [it for it in plot.items if isinstance(it, pg.LinearRegionItem)]
    assert len(after) == len(before), "spans must NOT spawn LinearRegionItems"
    added = [it for it in plot.items if isinstance(it, SpanOverlayItem)]
    assert len(added) == 1


def test_render_spans_index_to_x_mapping(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    values = session.x_axis(doc_id).values
    starts = np.array([0, 25_000, 100_000], dtype=np.int64)
    stops = np.array([1_000, 30_000, 120_000], dtype=np.int64)
    item = render_spans(plot, session, doc_id, _span_hint(starts, stops))[0]

    # x maps through the session x-axis at native t0 (offset 0): idx/fs.
    np.testing.assert_allclose(item._x0, values[starts])
    np.testing.assert_allclose(item._x1, values[stops])
    lo, hi = item.x_bounds()
    assert lo == pytest.approx(values[0])
    assert hi == pytest.approx(values[120_000])


def test_render_spans_clips_out_of_range_indices(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    values = session.x_axis(doc_id).values
    # end_idx == N (half-open) and a stray huge index must clip to the last sample, not raise.
    starts = np.array([N - 10], dtype=np.int64)
    stops = np.array([N], dtype=np.int64)
    item = render_spans(plot, session, doc_id, _span_hint(starts, stops))[0]
    assert item._x1[0] == pytest.approx(values[-1])


def test_span_bounding_rect_tracks_viewbox(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    starts = np.array([1_000, 50_000], dtype=np.int64)
    stops = np.array([3_000, 60_000], dtype=np.int64)
    item = render_spans(plot, session, doc_id, _span_hint(starts, stops))[0]
    # boundingRect follows the current viewbox so full-height rects stay inside it.
    (vx0, vx1), (vy0, vy1) = plot.vb.viewRange()
    br = item.boundingRect()
    assert br.left() == pytest.approx(vx0)
    assert br.top() == pytest.approx(vy0)
    assert br.width() == pytest.approx(vx1 - vx0)
    assert br.height() == pytest.approx(vy1 - vy0)


def test_empty_span_hint_is_harmless(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    empty = np.array([], dtype=np.int64)
    item = render_spans(plot, session, doc_id, _span_hint(empty, empty))[0]
    assert item.n_spans == 0
    assert item.x_bounds() is None
    assert item.boundingRect() == item.boundingRect()  # does not raise


# ================================================================================================
# hline + marker renderers.
# ================================================================================================

def test_render_hline_curve(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    values = session.x_axis(doc_id).values
    knots = np.array([0, 50_000, 100_000, 150_000], dtype=np.int64)
    base = np.array([1.0, 1.2, 0.9, 1.1], dtype=np.float64)
    hint = OverlayHint(kind=OverlayKind.HLINE, starts=knots, values=base, color="#81C784")
    items = render_hline(plot, session, doc_id, hint)
    assert len(items) == 2, "the visible baseline is a dark casing plus its dashed ink"
    for item in items:
        x, y = item.getData()
        np.testing.assert_allclose(x, values[knots])
        np.testing.assert_allclose(y, base)


def test_render_marker_region(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    values = session.x_axis(doc_id).values
    hint = OverlayHint(
        kind=OverlayKind.MARKER,
        starts=np.array([40_000], dtype=np.int64),
        stops=np.array([42_000], dtype=np.int64),
    )
    items = render_marker(plot, session, doc_id, hint)
    assert len(items) == 1
    region = items[0]
    assert isinstance(region, pg.LinearRegionItem)
    lo, hi = region.getRegion()
    assert lo == pytest.approx(values[40_000])
    assert hi == pytest.approx(values[42_000])


def test_render_marker_line_when_no_stop(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    values = session.x_axis(doc_id).values
    hint = OverlayHint(kind=OverlayKind.MARKER, starts=np.array([12_345], dtype=np.int64))
    items = render_marker(plot, session, doc_id, hint)
    assert len(items) == 1
    assert isinstance(items[0], pg.InfiniteLine)
    assert items[0].value() == pytest.approx(values[12_345])


# ================================================================================================
# Registration + TraceView.add_overlay / remove_overlay lifecycle.
# ================================================================================================

def test_registration_dispatches_through_trace_view(session_view):
    session, view, doc_id = session_view
    starts = np.arange(0, 60_000, 6_000, dtype=np.int64)     # 10 events
    stops = starts + 1_500
    overlay_id = view.add_overlay(doc_id, 0, _span_hint(starts, stops))

    items = view._overlays[overlay_id]
    assert len(items) == 1
    assert isinstance(items[0], SpanOverlayItem)
    assert items[0].n_spans == len(starts)
    # the item is actually parented into the pane's plot.
    assert items[0] in view.anchor_plot().items


def test_add_then_remove_overlay(session_view):
    session, view, doc_id = session_view
    starts = np.array([1_000, 2_000], dtype=np.int64)
    stops = np.array([1_500, 2_500], dtype=np.int64)
    overlay_id = view.add_overlay(doc_id, 0, _span_hint(starts, stops))
    item = view._overlays[overlay_id][0]
    assert item.scene() is not None

    view.remove_overlay(overlay_id)
    assert overlay_id not in view._overlays
    assert item.scene() is None   # detached from the scene


# ================================================================================================
# Overlay id uniqueness. Regression for the intermittent
# tests/gui/test_measure_navigation.py::test_markers_are_redrawn_after_a_trace_view_rebuild.
# ================================================================================================

def test_overlay_ids_are_unique_back_to_back(session_view):
    """Two add_overlay calls back to back must return DIFFERENT ids -- every call is a new overlay.

    The id used to end in ``id(hint)``, a CPython memory ADDRESS, which collides two ways:

    * passing the SAME hint object twice collides 100% of the time (asserted here);
    * passing two short-lived hints collides whenever the allocator hands the second one the
      address the first just freed -- which is what made the measure-panel redraw flaky rather
      than broken. That shape is covered by the loop test below.

    Either way the second overlay evicts the first from ``_overlays`` under the shared id, and the
    first id becomes the "stranded dead reference" that
    test_markers_are_redrawn_after_a_trace_view_rebuild exists to catch.
    """
    session, view, doc_id = session_view
    starts = np.array([1_000], dtype=np.int64)
    hint = _span_hint(starts, starts + 500)   # retained, so id(hint) cannot differ between calls

    first = view.add_overlay(doc_id, 0, hint)
    second = view.add_overlay(doc_id, 0, hint)

    assert first != second
    # ...and both overlays are tracked, rather than the second silently evicting the first
    # and leaking its graphics items.
    assert first in view._overlays and second in view._overlays
    assert view._overlays[first][0] is not view._overlays[second][0]


def test_overlay_ids_never_repeat_across_many_creations(session_view):
    """Repetition converts the flaky collision into a deterministic failure.

    A single pair of short-lived hints collides only when the allocator happens to hand the second
    one the address the first just freed -- through add_overlay, with the renderer allocating
    graphics items in between, that was ~20% per pair, which is why the measure-panel redraw failed
    intermittently instead of always. Over 200 creations it is a certainty, so any id generator
    that can repeat gets caught here rather than in a full-suite run three weeks from now.
    """
    session, view, doc_id = session_view
    starts = np.array([2_000], dtype=np.int64)

    ids = []
    for _ in range(200):
        # The hint is a temporary that dies at the end of the call, exactly as in
        # MeasurePanel._refresh_markers -- this is what frees the address for reuse.
        ids.append(view.add_overlay(doc_id, 0, _span_hint(starts, starts + 100)))

    assert len(set(ids)) == len(ids), "overlay ids must be unique by construction, not by luck"


def test_register_all_renderers_is_idempotent():
    # Calling twice must not raise (register_overlay_renderer just re-binds the dict entry).
    register_all_renderers()
    register_all_renderers()
    from nanoscope.gui.trace_view import _OVERLAY_RENDERERS
    assert _OVERLAY_RENDERERS[OverlayKind.SPAN] is render_spans
    assert _OVERLAY_RENDERERS[OverlayKind.HLINE] is render_hline
    assert _OVERLAY_RENDERERS[OverlayKind.MARKER] is render_marker
    assert _OVERLAY_RENDERERS[OverlayKind.STEP] is render_step


# ================================================================================================
# render_step — the CUSUM sub-state levels on the MAIN trace.
#
# Reported twice by the owner: "the Cusum sublevels show on the curate window but not on the main
# panel", then "the sublevels still don't show on the main panel, and cannot be read easily on the
# curate panel. Either the color has to be boldened/changed or another way of representing the
# sublevels is needed." Both halves are READABILITY, not wiring — the two PlotDataItems were
# always added to the pane. A level of 1 ms inside a 250 s recording is 4e-6 of the pane width,
# i.e. a zero-length stroke, which Qt paints as the pen's square cap and nothing else. These pin
# the two things that make it a mark instead of a speck: the per-zoom pixel floor and the ink.
# ================================================================================================

def _step_hint(starts, stops, values, **kw) -> OverlayHint:
    return OverlayHint(
        kind=OverlayKind.STEP,
        starts=np.asarray(starts, dtype=np.int64),
        stops=np.asarray(stops, dtype=np.int64),
        values=np.asarray(values, dtype=np.float64),
        **kw,
    )


def _drawn_widths(item) -> np.ndarray:
    """The x-extent of each drawn segment, out of the NaN-separated triplets."""
    x, _y = item.getData()
    return x[1::3] - x[0::3]


def test_render_step_returns_one_cased_pair_not_one_item_per_level(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    starts = np.arange(0, 100_000, 10_000, dtype=np.int64)      # 10 levels
    items = render_step(plot, session, doc_id,
                        _step_hint(starts, starts + 400, np.linspace(-40, -30, starts.size)))
    assert len(items) == 2, "a dark casing and the ink — never one item per level"
    assert all(isinstance(it, StepOverlayItem) for it in items)
    assert [it.zValue() for it in items] == [5.0, 6.0]
    assert all(it.n_levels == starts.size for it in items)


def test_a_sub_pixel_level_still_paints_a_visible_bar(session_view):
    """*** THE ROUND-2 DEFECT. *** At full-recording zoom a short level is far narrower than a
    device pixel, and a zero-length stroke is a square cap: the level was DRAWN and invisible,
    which from the user's chair is the same thing as not drawn. SpanOverlayItem has had a
    one-device-pixel floor since the same complaint was made about the event spans; this is that
    floor, expressed for a stroke rather than a fill.
    """
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    view.reset_zoom()                       # the full-recording map, where the defect lives
    # 20 samples at 50 kHz == 0.4 ms, ~1/10000 of the recording: sub-pixel at any real width.
    items = render_step(plot, session, doc_id,
                        _step_hint([100_000], [100_020], [-42.0]))
    for it in items:
        px = it.pixel_width()
        assert px > 0.0, "the pane has a viewport, so a pixel has a size"
        native = session.x_axis(doc_id).values[100_020] - session.x_axis(doc_id).values[100_000]
        assert native < px, "premise: this level is narrower than one device pixel"
        drawn = _drawn_widths(it)
        assert drawn.size == 1
        assert drawn[0] / px == pytest.approx(3.0, rel=1e-6), (
            f"a sub-pixel level must still paint _STEP_MIN_PX px wide, got {drawn[0] / px} px")


def test_the_pixel_floor_grows_a_level_about_its_own_centre(session_view):
    """Grown about the CENTRE, exactly as SpanOverlayItem grows a narrow span: widening from the
    left edge would walk every level of a burst rightwards by half a floor and put the fitted
    sub-states beside the event they were fitted to."""
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    view.reset_zoom()
    values = session.x_axis(doc_id).values
    starts = np.array([20_000, 120_000], dtype=np.int64)
    stops = starts + 10
    item = render_step(plot, session, doc_id,
                       _step_hint(starts, stops, [-40.0, -38.0]))[1]
    x, _y = item.getData()
    drawn_mid = 0.5 * (x[0::3] + x[1::3])
    native_mid = 0.5 * (values[starts] + values[stops])
    np.testing.assert_allclose(drawn_mid, native_mid, rtol=0, atol=1e-12)


def test_a_level_wider_than_the_floor_is_left_exactly_alone(session_view):
    """The floor changes no geometry the user can zoom into: once a level is wider than
    _STEP_MIN_PX px it must be drawn at its own extent, to the last float."""
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    values = session.x_axis(doc_id).values
    starts = np.array([10_000], dtype=np.int64)
    stops = np.array([60_000], dtype=np.int64)     # 1 s of a 4 s recording — hugely wide
    item = render_step(plot, session, doc_id, _step_hint(starts, stops, [-41.0]))[1]
    x, y = item.getData()
    assert x[0] == pytest.approx(values[10_000])
    assert x[1] == pytest.approx(values[60_000])
    assert y[0] == y[1] == -41.0


def test_the_floor_follows_the_zoom(session_view):
    """The floor is in DEVICE PIXELS, so it is a different number of seconds at every zoom.
    Diving into the level must hand back its true extent; zooming back out must widen it again."""
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    view.reset_zoom()
    values = session.x_axis(doc_id).values
    item = render_step(plot, session, doc_id,
                       _step_hint([100_000], [100_400], [-42.0]))[1]
    native = float(values[100_400] - values[100_000])
    wide_out = float(_drawn_widths(item)[0])
    assert wide_out > native, "premise: widened at the full-recording extent"

    view.zoom_to_span(values[100_000], values[100_400])   # dive in
    item.refresh_widths()
    assert float(_drawn_widths(item)[0]) == pytest.approx(native)

    view.reset_zoom()                                        # and back out
    item.refresh_widths()
    assert float(_drawn_widths(item)[0]) == pytest.approx(wide_out)


def test_the_level_ink_is_bold_and_does_not_collide_with_the_other_overlays(session_view):
    """The other half of the complaint: "the color has to be boldened/changed".

    The ink is this module's named constant, not ``hint.color`` — how a level reads is a
    property of the trace it is drawn on, and the panel that declares the hint passes the
    inspector's darker green, which is what disappeared into the curve. Pinned here because the
    two things that make it legible are exactly the two the user asked for: a value bright
    enough for a dark pane, and a stroke wider than the 3 px it used to be.
    """
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    under, item = render_step(
        plot, session, doc_id,
        _step_hint([1_000], [4_000], [-40.0], color="#009E73"))   # the inspector's own green

    ink = item.opts["pen"].color()
    assert ink.name().lower() == STEP_LEVEL_HEX.lower(), "the surface owns the level ink"
    assert ink.value() >= 200, f"too dark to read on a dark pane: value {ink.value()}"
    # Never the event-span amber and never the selected-event vermillion — a level that reads
    # as either of those is a level that says the wrong thing about the event it sits on.
    assert ink.name().lower() not in {"#ffb74d", "#d55e00"}

    assert item.opts["pen"].width() >= 4, "boldened, the user's own word for what was missing"
    assert under.opts["pen"].width() > item.opts["pen"].width(), (
        "the casing is a border around the ink, not a second line beside it")
    assert under.opts["pen"].color().value() < 60, "the casing is the dark half of the contrast"


def test_step_levels_do_not_drive_y_autorange(session_view):
    """Same rule as the span overlay: an annotation, added ignoreBounds. A level lies inside the
    curve's own range, so including it can only ratchet — never inform — the pane."""
    session, view, doc_id = session_view
    pane = view.panes()[0]
    vb = pane.plot.vb
    vb.enableAutoRange(y=True)
    y0 = tuple(vb.viewRange()[1])
    render_step(pane.plot, session, doc_id,
                _step_hint([1_000], [2_000], [500.0]))   # far outside the ~unit-normal signal
    for _ in range(4):
        vb.enableAutoRange(y=True)
        vb.updateAutoRange()
    y1 = tuple(vb.viewRange()[1])
    assert y1 == pytest.approx(y0, rel=0.05), f"step overlay expanded y-autorange: {y0} -> {y1}"


def test_empty_step_hint_renders_nothing(session_view):
    session, view, doc_id = session_view
    plot = view.anchor_plot()
    empty = np.array([], dtype=np.int64)
    assert render_step(plot, session, doc_id,
                       _step_hint(empty, empty, np.array([], dtype=np.float64))) == []
