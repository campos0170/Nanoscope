"""Tests for the manual-measurement cursor tool on TraceView (PORT_PLAN Measure, Phase 2).

The Measure tool is a single movable pg.LinearRegionItem whose two edges are the boundary cursors,
living on the current-channel pane. These tests drive it headless (offscreen) and assert:

  - set_measure_active on -> exactly ONE movable LinearRegionItem on the current pane, added
    ignoreBounds=True (so it is NOT in the ViewBox's autorange set);
  - moving/placing the region emits sigMeasureChanged with the correct half-open sample indices;
  - measure_bounds() mirrors the last emitted indices;
  - set_measure_active off removes the region and measure_bounds() -> None;
  - (regression) adding the region does NOT blow up the pane's y-autorange.
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

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.trace_view import MEASURE_HANDLE_SIZE_PX, TraceView

FS = 50_000.0
N = 200_000


def _make_document() -> TraceDocument:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(N).astype(np.float32)
    info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=Path("/tmp/synthetic_measure.dat"),
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
    view.resize(1200, 600)   # a real extent so the viewbox has a range to seed/clip against
    return session, view, doc.doc_id


def _regions(plot: pg.PlotItem) -> list[pg.LinearRegionItem]:
    return [it for it in plot.items if isinstance(it, pg.LinearRegionItem)]


# ================================================================================================
# activation: one movable, ignoreBounds region on the current pane.
# ================================================================================================

def test_activate_adds_single_movable_region(session_view):
    session, view, doc_id = session_view
    pane = view.pane_for_layer(f"{doc_id}#0")
    assert _regions(pane.plot) == []          # nothing before

    view.set_measure_active(doc_id, 0, True)

    regions = _regions(pane.plot)
    assert len(regions) == 1, "Measure must add exactly one LinearRegionItem"
    region = regions[0]
    assert region.movable is True
    # ignoreBounds=True means the item is NOT registered in the ViewBox autorange set...
    assert region not in pane.plot.vb.addedItems
    # ...but it IS parented into the plot (visible, removable).
    assert region in pane.plot.items


def test_activate_does_not_blow_up_y_autorange(session_view):
    """Regression: a full-height LinearRegionItem must be added ignoreBounds=True, or repeated
    autorange passes union its boundingRect and ratchet the pA pane's y-range ever wider (crushing
    the ~unit-normal signal to a flat line)."""
    session, view, doc_id = session_view
    pane = view.pane_for_layer(f"{doc_id}#0")
    vb = pane.plot.vb
    vb.enableAutoRange(y=True)
    vb.updateAutoRange()
    y0 = tuple(vb.viewRange()[1])

    view.set_measure_active(doc_id, 0, True)
    for _ in range(4):
        vb.enableAutoRange(y=True)
        vb.updateAutoRange()
    y1 = tuple(vb.viewRange()[1])

    assert y1 == pytest.approx(y0, rel=0.05), f"measure region expanded y-autorange: {y0} -> {y1}"
    assert abs(y1[0]) < 20 and abs(y1[1]) < 20, f"y-axis blown up by measure region: {y1}"


def test_seeded_region_lands_inside_the_current_view(session_view):
    session, view, doc_id = session_view
    view.set_measure_active(doc_id, 0, True)
    (vx0, vx1) = view.anchor_plot().vb.viewRange()[0]
    lo, hi = view._measure_region.getRegion()
    assert vx0 <= lo < hi <= vx1
    # ~20% of the view width (middle fifth).
    assert (hi - lo) == pytest.approx(0.20 * (vx1 - vx0), rel=1e-6)


def test_measure_edges_have_device_pixel_handles_that_survive_deep_zoom(session_view, qapp):
    """The edge grips must enlarge the real mouse target, not only thicken the paint."""
    _session, view, doc_id = session_view
    view.show()
    view.set_measure_active(doc_id, 0, True)
    qapp.processEvents()
    region = view._measure_region
    assert region is not None and region.movable is True
    assert len(region.lines) == 2

    for line in region.lines:
        assert line.movable is True
        assert any(size >= MEASURE_HANDLE_SIZE_PX for _path, _position, size in line.markers)
        assert line.sceneBoundingRect().width() >= 12.0

    lo, hi = region.getRegion()
    centre = 0.5 * (lo + hi)
    view.zoom_x_to(centre - 1e-5, centre + 1e-5)
    qapp.processEvents()
    for line in region.lines:
        assert line.sceneBoundingRect().width() >= 12.0


# ================================================================================================
# emission + index mapping.
# ================================================================================================

def test_move_emits_correct_sample_indices(session_view):
    session, view, doc_id = session_view
    values = session.x_axis(doc_id).values

    emitted: list[tuple[int, int]] = []
    view.sigMeasureChanged.connect(lambda a, b: emitted.append((a, b)))

    view.set_measure_active(doc_id, 0, True)
    emitted.clear()                                   # drop the initial seeded emit

    start_idx, stop_idx = 40_000, 60_000
    view.set_measure_region(float(values[start_idx]), float(values[stop_idx]))

    assert emitted, "moving the region must emit sigMeasureChanged"
    assert emitted[-1] == (start_idx, stop_idx)
    assert view.measure_bounds() == (start_idx, stop_idx)


def test_measure_region_edges_clamp_to_trace(session_view):
    session, view, doc_id = session_view
    values = session.x_axis(doc_id).values
    view.set_measure_active(doc_id, 0, True)

    # x-edges beyond both ends of the trace clamp to [0, N], never raising or going out of range.
    view.set_measure_region(float(values[0]) - 10.0, float(values[-1]) + 10.0)
    start, stop = view.measure_bounds()
    assert start == 0
    assert stop == N


def test_set_region_swaps_reversed_edges(session_view):
    session, view, doc_id = session_view
    values = session.x_axis(doc_id).values
    view.set_measure_active(doc_id, 0, True)
    # Pass hi, lo reversed: the region normalises and bounds come back ordered.
    view.set_measure_region(float(values[90_000]), float(values[10_000]))
    start, stop = view.measure_bounds()
    assert (start, stop) == (10_000, 90_000)


# ================================================================================================
# deactivation + rebuild.
# ================================================================================================

def test_deactivate_removes_region(session_view):
    session, view, doc_id = session_view
    pane = view.pane_for_layer(f"{doc_id}#0")
    view.set_measure_active(doc_id, 0, True)
    region = _regions(pane.plot)[0]
    assert region.scene() is not None

    view.set_measure_active(doc_id, 0, False)

    assert _regions(pane.plot) == []
    assert region.scene() is None                     # detached from the scene
    assert view.measure_bounds() is None


def test_bounds_none_when_inactive(session_view):
    session, view, doc_id = session_view
    assert view.measure_bounds() is None


def test_region_reattaches_across_rebuild(session_view):
    session, view, doc_id = session_view
    values = session.x_axis(doc_id).values
    view.set_measure_active(doc_id, 0, True)
    view.set_measure_region(float(values[40_000]), float(values[60_000]))

    view.rebuild()                                    # panes (and the old region) are recreated

    pane = view.pane_for_layer(f"{doc_id}#0")
    assert len(_regions(pane.plot)) == 1, "region must reattach after a rebuild"
    assert view.measure_bounds() == (40_000, 60_000)  # at the same place


def test_region_dropped_when_channel_closes(session_view):
    session, view, doc_id = session_view
    view.set_measure_active(doc_id, 0, True)
    assert view.measure_bounds() is not None

    session.remove_layer(f"{doc_id}#0")               # channel goes away -> rebuild with no panes
    assert view.measure_bounds() is None
