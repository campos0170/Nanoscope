"""Tests for the zoomed-in pan scrollbar (gui/pan_scrollbar.py).

The ask: "once the user zooms into an event, sometimes the event might not be centred — place a
horizontal scroller (appears only once zoomed) to centre events." So the contract under test is:

  - VISIBILITY: hidden at full extent, shown the moment the view is narrower, hidden again on
    reset — and with no document / a degenerate (single-sample) extent / a view wider than the
    recording, it stays hidden rather than dividing by zero;
  - PAN, NOT ZOOM: dragging the handle translates the view and preserves the zoom width EXACTLY,
    including across twenty successive pans (the integer<->float mapping must not accumulate);
  - TWO-WAY: every other way of moving the view (box zoom, navigator drag, Full reset,
    zoom back/forward, an event-table jump) restates the bar, and neither side re-triggers
    the other — the emission counts are asserted, not assumed;
  - AGREEMENT: the Navigator (the OTHER controller bound to the same x-range) and the bar always
    describe the same window;
  - ARCHITECTURE: the widget imports no sibling gui panel; MainWindow places it between the trace
    and the navigator.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pathlib
from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.navigator import Navigator
from nanoscope.gui.pan_scrollbar import PAN_STEPS, PanScrollBar
from nanoscope.gui.trace_view import TraceView

from .test_measure_navigation import _canonical_gui_imports

FS = 50_000.0
N = 200_000          # 4 s at 50 kHz


def _make_document(n: int = N, name: str = "synthetic_pan.dat") -> TraceDocument:
    rng = np.random.default_rng(3)
    data = rng.standard_normal(n).astype(np.float32)
    info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=Path(f"/tmp/{name}"),
        format_name="synthetic",
        n_samples=n,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(data,))


def _build(n: int = N, name: str = "synthetic_pan.dat"):
    """Session + TraceView + a bound PanScrollBar, framed at full extent (a fresh load)."""
    session = Session()
    doc = _make_document(n, name)
    session.add_document(doc)
    session.add_layer(doc.doc_id, 0)
    view = TraceView(session)
    view.resize(1200, 600)
    bar = PanScrollBar()
    bar.bind(session, view)
    return session, view, bar, doc.doc_id


@pytest.fixture
def rig(qapp):
    return _build()


def _xrange(view) -> tuple[float, float]:
    lo, hi = view.anchor_plot().vb.viewRange()[0]
    return float(lo), float(hi)


def _width(view) -> float:
    lo, hi = _xrange(view)
    return hi - lo


def _zoom_middle(view, half_width: float) -> None:
    e0, e1 = view.full_x_extent()
    c = 0.5 * (e0 + e1)
    view.zoom_x_to(c - half_width, c + half_width)


# ================================================================================================
# 1. visibility — "appears only once zoomed"
# ================================================================================================

def test_hidden_at_full_extent(rig):
    _session, view, bar, _doc = rig
    assert view.full_x_extent() is not None
    assert _width(view) == pytest.approx(view.full_x_extent()[1] - view.full_x_extent()[0], rel=1e-6)
    assert bar.is_active() is False
    assert bar.scrollbar().isHidden() is True
    assert bar._full_trace_button.isEnabled() is False


def test_shown_as_soon_as_the_view_is_narrower(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.05)                 # 100 ms of a 4 s recording
    assert bar.is_active() is True
    assert bar.scrollbar().isHidden() is False
    assert bar.scrollbar().maximum() > 0     # there is somewhere to pan to


def test_hidden_again_on_reset(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.05)
    assert bar.is_active() is True
    view.reset_zoom()                        # the Full / Ctrl+0 path
    assert bar.is_active() is False
    assert bar.scrollbar().isHidden() is True


def test_a_hairs_breadth_off_full_extent_does_not_flicker_the_bar(rig):
    """The not-zoomed test is a RELATIVE tolerance, so a rounding-sized difference (far below one
    pixel of a 1200 px plot) must not make the bar blink into existence."""
    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    span = e1 - e0
    view.zoom_x_to(e0 + span * 1e-6, e1 - span * 1e-6)
    assert bar.is_active() is False
    view.zoom_x_to(e0 + span * 0.02, e1)     # 2% in: a real zoom
    assert bar.is_active() is True


def test_the_visibility_boundary_is_exactly_one_tolerance_below_the_extent(rig):
    """Pin the predicate itself, not a comfortable distance either side of it: the bar is active
    iff the visible width is strictly under ``span * (1 - FULL_EXTENT_TOL)`` — i.e. under 99.9%
    of the recording. 99.9% and 100.1% are BOTH hidden (the second is zoomed out past the data);
    99.85% is the first width that shows it."""
    from nanoscope.gui.pan_scrollbar import FULL_EXTENT_TOL

    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    span = e1 - e0
    centre = 0.5 * (e0 + e1)

    def active_at(frac: float) -> bool:
        half = 0.5 * span * frac
        view.zoom_x_to(centre - half, centre + half)
        return bar.is_active()

    assert FULL_EXTENT_TOL == 1e-3
    assert active_at(1.001) is False        # wider than the recording
    assert active_at(1.0) is False
    assert active_at(0.999) is False        # exactly at the threshold: still "full extent"
    assert active_at(0.9995) is False
    assert active_at(0.9985) is True        # first side of the boundary that counts as zoomed
    assert active_at(0.95) is True


def test_hidden_when_the_view_is_wider_than_the_recording(rig):
    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    span = e1 - e0
    view.zoom_x_to(e0 - span, e1 + span)     # zoomed OUT past the data
    assert bar.is_active() is False


def test_no_document_is_inert(qapp):
    """No panes at all: refresh(), a stray value change and centre_on() must all be no-ops."""
    session = Session()
    view = TraceView(session)
    bar = PanScrollBar()
    bar.bind(session, view)
    assert view.anchor_plot() is None
    assert bar.is_active() is False
    assert bar._full_trace_button.isEnabled() is False
    bar.refresh()                            # must not raise
    bar.scrollbar().setValue(500)            # must not raise, must not activate
    assert bar.is_active() is False
    assert bar.centre_on(1.0) is False


def test_degenerate_single_sample_extent_is_inert(qapp):
    """A one-sample recording has no extent to pan across (full_x_extent() -> None)."""
    _session, view, bar, _doc = _build(n=1, name="one_sample.dat")
    assert view.full_x_extent() is None
    assert bar.is_active() is False
    bar.scrollbar().setValue(bar.scrollbar().maximum())
    assert bar.is_active() is False


def test_closing_the_document_hides_the_bar(rig):
    session, view, bar, doc_id = rig
    _zoom_middle(view, 0.05)
    assert bar.is_active() is True
    session.close_document(doc_id)
    assert bar.is_active() is False
    assert view.anchor_plot() is None


# ================================================================================================
# 2. panning preserves the zoom width EXACTLY
# ================================================================================================

def test_dragging_the_bar_pans_without_changing_the_zoom_width(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.05)
    before_lo, _before_hi = _xrange(view)
    width_before = _width(view)

    sb = bar.scrollbar()
    sb.setValue(sb.value() + sb.pageStep())          # page right by one screenful

    after_lo, _after_hi = _xrange(view)
    assert after_lo > before_lo                       # it actually moved
    assert _width(view) == pytest.approx(width_before, rel=1e-12)


def test_twenty_successive_pans_do_not_drift_the_zoom_width(rig):
    """The whole point of taking the width from the stored float rather than from pageStep."""
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.02)                          # 40 ms window
    width0 = _width(view)
    sb = bar.scrollbar()

    seen = []
    for _ in range(20):
        sb.setValue(min(sb.maximum(), sb.value() + sb.singleStep()))
        seen.append(_width(view))

    for w in seen:
        assert w == pytest.approx(width0, rel=1e-12)
    # ...and the LAST one, compared to the FIRST, is the drift test that a per-step tolerance
    # would let through if every step lost a little.
    assert seen[-1] == pytest.approx(width0, rel=1e-12)


def test_twenty_alternating_left_right_pans_do_not_drift_the_zoom_width(rig):
    """Successive pans in ONE direction can only accumulate error one way; alternating pans are
    the case where a per-step round trip through the integer bar would show up as a width that
    breathes. Every intermediate width is asserted, not just the last."""
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.02)
    width0 = _width(view)
    sb = bar.scrollbar()

    for i in range(20):
        delta = sb.singleStep() if i % 2 == 0 else -sb.singleStep()
        sb.setValue(max(sb.minimum(), min(sb.maximum(), sb.value() + delta)))
        assert _width(view) == pytest.approx(width0, rel=1e-12), f"width drifted at pan {i}"


def test_pans_do_not_drift_the_width_at_a_three_sample_zoom(rig):
    """The deepest zoom the data itself allows, where a quantisation error would be a whole
    screenful rather than a rounding artefact."""
    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    dx = 1.0 / FS
    centre = 0.5 * (e0 + e1)
    view.zoom_x_to(centre, centre + 3 * dx)
    assert bar.is_active() is True
    width0 = _width(view)

    sb = bar.scrollbar()
    for _ in range(20):
        sb.setValue(min(sb.maximum(), sb.value() + sb.singleStep()))
        assert _width(view) == pytest.approx(width0, rel=1e-12)
    lo, hi = _xrange(view)
    assert e0 - 1e-12 <= lo and hi <= e1 + 1e-12


def test_panning_back_and_forth_returns_to_the_same_window(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.02)
    lo0, hi0 = _xrange(view)
    sb = bar.scrollbar()
    start = sb.value()

    for _ in range(10):
        sb.setValue(sb.value() + sb.singleStep())
    for _ in range(10):
        sb.setValue(sb.value() - sb.singleStep())
    assert sb.value() == start

    lo1, hi1 = _xrange(view)
    step = (view.full_x_extent()[1] - view.full_x_extent()[0]) / PAN_STEPS
    assert lo1 == pytest.approx(lo0, abs=2 * step)
    assert hi1 - lo1 == pytest.approx(hi0 - lo0, rel=1e-12)


# ================================================================================================
# 3. clamping at both ends
# ================================================================================================

def test_clamps_at_the_left_end(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.03)
    width = _width(view)
    sb = bar.scrollbar()
    sb.setValue(sb.minimum())
    e0, _e1 = view.full_x_extent()
    lo, hi = _xrange(view)
    assert lo == pytest.approx(e0, abs=1e-12)
    assert hi - lo == pytest.approx(width, rel=1e-12)


def test_clamps_at_the_right_end(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.03)
    width = _width(view)
    sb = bar.scrollbar()
    sb.setValue(sb.maximum())
    e0, e1 = view.full_x_extent()
    lo, hi = _xrange(view)
    step = (e1 - e0) / PAN_STEPS
    assert hi == pytest.approx(e1, abs=4 * step)
    assert hi <= e1 + 1e-12                            # never past the end of the recording
    assert lo >= e0 - 1e-12
    assert hi - lo == pytest.approx(width, rel=1e-12)


def test_qt_itself_refuses_a_value_past_the_ends(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.03)
    sb = bar.scrollbar()
    sb.setValue(sb.maximum() + 10_000_000)
    assert sb.value() == sb.maximum()
    _e0, e1 = view.full_x_extent()
    assert _xrange(view)[1] <= e1 + 1e-12


# ================================================================================================
# 4. the mapping: fixed resolution, pageStep == one screenful
# ================================================================================================

def test_page_step_equals_the_visible_width(rig):
    _session, view, bar, _doc = rig
    for half in (1.0, 0.1, 0.01, 0.001):
        _zoom_middle(view, half)
        if not bar.is_active():
            continue
        e0, e1 = view.full_x_extent()
        span = e1 - e0
        sb = bar.scrollbar()
        assert sb.pageStep() / PAN_STEPS * span == pytest.approx(_width(view), rel=1e-3)
        assert sb.maximum() == PAN_STEPS - sb.pageStep()
        assert sb.minimum() == 0
        assert sb.singleStep() == max(1, sb.pageStep() // 10)


def test_the_mapping_resolution_is_finer_than_one_sample(rig):
    """A fixed PAN_STEPS across the extent: one step must be below the data's own resolution,
    otherwise the finest possible nudge would jump whole screens at a deep zoom."""
    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    assert (e1 - e0) / PAN_STEPS < 1.0 / FS


def test_bar_position_reflects_where_in_the_recording_the_view_is(rig):
    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    span = e1 - e0
    values = []
    for frac in (0.05, 0.25, 0.5, 0.75, 0.9):
        lo = e0 + span * frac
        view.zoom_x_to(lo, lo + 0.02)
        values.append(bar.scrollbar().value())
        # the handle's position decodes back to the view's left edge
        decoded = e0 + bar.scrollbar().value() / PAN_STEPS * span
        assert decoded == pytest.approx(lo, abs=2 * span / PAN_STEPS)
    assert values == sorted(values)


# ================================================================================================
# 5. two-way binding: every other way of moving the view restates the bar
# ================================================================================================

def test_programmatic_zoom_to_span_updates_the_bar(rig):
    """The event-table 'zoom to event' and the measurement Go-to both land in zoom_to_span."""
    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    view.zoom_to_span(e0 + 0.8 * (e1 - e0), e0 + 0.8 * (e1 - e0) + 0.001)
    assert bar.is_active() is True
    lo, _hi = _xrange(view)
    decoded = e0 + bar.scrollbar().value() / PAN_STEPS * (e1 - e0)
    assert decoded == pytest.approx(lo, abs=2 * (e1 - e0) / PAN_STEPS)


def test_zoom_back_and_forward_update_the_bar(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.05)
    view._settle.stop()
    view._commit_view()                       # settle the deep view into the history
    first = bar.scrollbar().value()

    e0, e1 = view.full_x_extent()
    view.zoom_x_to(e0 + 0.9 * (e1 - e0) - 0.05, e0 + 0.9 * (e1 - e0) + 0.05)
    view._settle.stop()
    view._commit_view()
    second = bar.scrollbar().value()
    assert second != first

    view.view_back()
    assert bar.scrollbar().value() == first
    view.view_forward()
    assert bar.scrollbar().value() == second


def test_navigator_drag_updates_the_bar(rig):
    """The OTHER controller drives the view; the bar must follow it, not fight it."""
    session, view, bar, _doc = rig
    nav = Navigator()
    nav.bind(session, view)
    _zoom_middle(view, 0.05)
    before = bar.scrollbar().value()

    lo, hi = _xrange(view)
    nav._region.setRegion((lo + 0.5, hi + 0.5))     # a genuine user drag (not _syncing)
    assert bar.scrollbar().value() > before
    assert _xrange(view)[0] == pytest.approx(lo + 0.5, rel=1e-9)


def test_bar_and_navigator_always_describe_the_same_window(rig):
    session, view, bar, _doc = rig
    nav = Navigator()
    nav.bind(session, view)
    _zoom_middle(view, 0.05)

    sb = bar.scrollbar()
    sb.setValue(sb.value() + 3 * sb.pageStep())

    lo, hi = _xrange(view)
    r_lo, r_hi = nav._region.getRegion()
    assert float(r_lo) == pytest.approx(lo, rel=1e-9)
    assert float(r_hi) == pytest.approx(hi, rel=1e-9)

    e0, e1 = view.full_x_extent()
    decoded = e0 + sb.value() / PAN_STEPS * (e1 - e0)
    assert decoded == pytest.approx(lo, abs=2 * (e1 - e0) / PAN_STEPS)


# ================================================================================================
# 6. no feedback loop — counted, not assumed
# ================================================================================================

def test_one_bar_move_produces_exactly_one_view_update(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.05)

    view_emits: list[tuple[float, float]] = []
    bar_emits: list[int] = []
    view.sigVisibleRangeChanged.connect(lambda a, b: view_emits.append((a, b)))
    bar.scrollbar().valueChanged.connect(bar_emits.append)

    sb = bar.scrollbar()
    sb.setValue(sb.value() + sb.pageStep())

    assert len(bar_emits) == 1, "the bar must not be re-driven by its own pan"
    assert len(view_emits) == 1, "one pan, one view update"


def test_one_plot_zoom_produces_exactly_one_bar_update(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.05)

    bar_emits: list[int] = []
    view_emits: list[tuple[float, float]] = []
    bar.scrollbar().valueChanged.connect(bar_emits.append)
    view.sigVisibleRangeChanged.connect(lambda a, b: view_emits.append((a, b)))

    e0, e1 = view.full_x_extent()
    view.zoom_x_to(e0 + 0.3 * (e1 - e0), e0 + 0.3 * (e1 - e0) + 0.02)

    assert len(view_emits) == 1
    assert len(bar_emits) == 1, "the bar mirrored the view and stopped there"


def test_a_hundred_alternating_moves_terminate(rig):
    """A crude but decisive loop check: if either side re-triggered the other this would not
    return (or would drift the width away)."""
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.02)
    width0 = _width(view)
    sb = bar.scrollbar()
    e0, e1 = view.full_x_extent()

    for i in range(50):
        sb.setValue(min(sb.maximum(), sb.value() + sb.singleStep()))
        lo = e0 + (0.1 + 0.01 * (i % 50)) * (e1 - e0)
        view.zoom_x_to(lo, lo + width0)
    assert _width(view) == pytest.approx(width0, rel=1e-9)


# ================================================================================================
# 7. centring — the user's stated goal
# ================================================================================================

def test_centre_on_pans_without_zooming(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.02)
    width = _width(view)
    e0, e1 = view.full_x_extent()
    target = e0 + 0.7 * (e1 - e0)

    assert bar.centre_on(target) is True
    lo, hi = _xrange(view)
    assert 0.5 * (lo + hi) == pytest.approx(target, rel=1e-9)
    assert hi - lo == pytest.approx(width, rel=1e-12)


def test_centre_on_clamps_near_the_ends(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.02)
    width = _width(view)
    e0, e1 = view.full_x_extent()

    bar.centre_on(e0)                                  # cannot centre on the very first sample
    lo, hi = _xrange(view)
    assert lo == pytest.approx(e0, abs=1e-12)
    assert hi - lo == pytest.approx(width, rel=1e-12)

    bar.centre_on(e1)
    lo, hi = _xrange(view)
    assert hi == pytest.approx(e1, abs=1e-12)
    assert hi - lo == pytest.approx(width, rel=1e-12)


def test_centre_on_does_not_move_a_degenerate_view(qapp):
    """A one-sample recording has NO extent, so there is nothing to clamp against — centring
    would fling the view to wherever ``x`` is, off the data, and still report success. It must
    refuse and leave the view exactly where it was.

    (Found adversarially: ``centre_on(12345.0)`` on a one-sample document moved the view from
    [-0.5, 0.5] to [12344.5, 12345.5] and returned True, contradicting the method's own
    docstring — "Returns False when there is nothing to pan (no document, degenerate view)".)"""
    _session, view, bar, _doc = _build(n=1, name="one_sample_centre.dat")
    assert view.full_x_extent() is None
    before = tuple(view.anchor_plot().vb.viewRange()[0])

    assert bar.centre_on(12_345.0) is False
    assert tuple(view.anchor_plot().vb.viewRange()[0]) == before
    assert bar.is_active() is False


def test_centre_on_refuses_when_the_view_is_wider_than_the_recording(rig):
    """No width-preserving view can both fit and centre an arbitrary point in a shorter target."""
    _session, view, bar, _doc = rig
    e0, e1 = view.full_x_extent()
    span = e1 - e0
    view.zoom_x_to(e0 - span, e1 + span)          # wider than the recording; bar is hidden
    assert bar.is_active() is False
    before = _xrange(view)

    assert bar.centre_on(e0 + 0.5 * span) is False
    assert _xrange(view) == pytest.approx(before)


def test_centre_on_keeps_the_bar_in_agreement(rig):
    _session, view, bar, _doc = rig
    _zoom_middle(view, 0.02)
    e0, e1 = view.full_x_extent()
    bar.centre_on(e0 + 0.4 * (e1 - e0))
    lo, _hi = _xrange(view)
    decoded = e0 + bar.scrollbar().value() / PAN_STEPS * (e1 - e0)
    assert decoded == pytest.approx(lo, abs=2 * (e1 - e0) / PAN_STEPS)


# ================================================================================================
# 8. layout: the height is reserved, so show/hide never resizes the trace
# ================================================================================================

def test_the_container_height_is_fixed_whether_the_bar_shows_or_not(rig):
    """The chosen anti-jitter behaviour: the CONTAINER reserves its height forever and only the
    QScrollBar inside it hides. If the container collapsed instead, every wheel notch across the
    full-extent threshold would resize the plot mid-gesture."""
    _session, view, bar, _doc = rig
    h = bar.minimumHeight()
    assert h == bar.maximumHeight() > 0          # fixed by construction, in BOTH states
    _zoom_middle(view, 0.05)
    assert bar.is_active() is True
    assert (bar.minimumHeight(), bar.maximumHeight()) == (h, h)
    view.reset_zoom()
    assert bar.is_active() is False
    assert (bar.minimumHeight(), bar.maximumHeight()) == (h, h)


def test_showing_the_bar_does_not_resize_the_trace_beside_it(qapp):
    """The behavioural version of the same claim, through a real layout: a sibling widget's
    geometry must be identical with the bar shown and hidden."""
    from PySide6 import QtWidgets

    _session, view, bar, _doc = _build(name="jitter.dat")
    host = QtWidgets.QWidget()
    host.resize(1200, 700)
    lay = QtWidgets.QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(view, 1)
    lay.addWidget(bar)
    lay.activate()
    hidden_geom = view.geometry()

    _zoom_middle(view, 0.05)
    lay.activate()
    assert bar.is_active() is True
    assert view.geometry() == hidden_geom

    view.reset_zoom()
    lay.activate()
    assert bar.is_active() is False
    assert view.geometry() == hidden_geom


# ================================================================================================
# 9. architecture
# ================================================================================================

def test_pan_scrollbar_imports_no_sibling_gui_panel():
    """It is bound to the view by MainWindow, exactly like the Navigator; it reaches for nothing
    in gui/ itself, by ANY of the five import spellings."""
    import nanoscope.gui.pan_scrollbar as mod

    assert _canonical_gui_imports(pathlib.Path(mod.__file__).read_text()) == set()


def test_main_window_places_the_bar_between_the_trace_and_the_navigator(qapp, qtbot):
    from nanoscope.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    lay = win.centralWidget().layout()
    order = [lay.itemAt(i).widget() for i in range(lay.count())]
    assert order.index(win._pan_scrollbar) == order.index(win._trace_view) + 1
    assert order.index(win._navigator) == order.index(win._pan_scrollbar) + 1
    assert win._pan_scrollbar.is_active() is False       # empty session: nothing to pan


def test_main_window_centre_action_is_gated_on_a_selection(qapp, qtbot):
    from nanoscope.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    assert win._act_centre_selection.isEnabled() is False
    win._centre_selection_in_view()                       # no selection: a status message, no crash
