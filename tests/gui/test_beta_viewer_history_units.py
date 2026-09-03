"""Regression tests for three beta-report viewer defects, all in gui/trace_view.py:

  VIEWER-001  Zoom Back/Forward pressed inside the 250 ms settle window discarded the
              in-flight view: Back landed TWO framings back and the dive was silently
              dropped from history (Forward could never return to it). Fixed by flushing
              the pending view into the history before stepping (_commit_pending).
  VIEWER-002  A calibration units change (pA -> nA) re-set curve data only: the pane's
              y-axis label (set in ChannelPane.__init__) and the minimum-y-span floor
              (apply_y_policy) went stale — axis said pA beside a crosshair printing nA,
              and the 5 pA floor made nA data unzoomable. Fixed by refresh_units().
  VIEWER-003  Every rebuild (layer toggle, Overlay/Stacked switch) reseeded the zoom
              history to one entry, so a still-enabled Zoom Back silently did nothing.
              Fixed by preserving the trail whenever the SHOWN documents are unchanged —
              the same test rebuild() already uses to preserve the zoom itself.

The two history fixes interact, so the cross cases are tested too: a rebuild mid-debounce
must neither drop the pending dive nor commit it twice.

Driven offscreen through the real Session + TraceView on synthetic documents (the
behaviours are size- and file-independent), mirroring tests/gui/test_pan_scrollbar.py.
Settling is simulated deterministically by firing the debounce by hand
(view._settle.stop(); view._commit_view()) — the suite's existing idiom — instead of
sleeping out the 250 ms interval.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
N = 200_000          # 4 s at 50 kHz


def _make_document(name: str = "synthetic_viewer.dat", *,
                   current_units: str = "pA") -> TraceDocument:
    """Two channels — a noisy current and a flat command voltage — so a visibility toggle
    of the voltage layer (the VIEWER-003 trigger) leaves the current pane standing."""
    rng = np.random.default_rng(7)
    cur = rng.standard_normal(N).astype(np.float32)
    volt = np.full(N, -1000.0, dtype=np.float32)
    infos = (
        ChannelInfo(index=0, name="IN15", raw_units=current_units, kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
    )
    meta = RecordingMeta(
        path=Path(f"/tmp/{name}"),
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=infos,
    )
    return TraceDocument(meta=meta, channels=(cur, volt))


def _build(*, current_units: str = "pA"):
    """Session + TraceView with both channels layered, framed at full extent (a fresh load)."""
    session = Session()
    doc = _make_document(current_units=current_units)
    session.add_document(doc)
    session.add_layer(doc.doc_id, 0)
    session.add_layer(doc.doc_id, 1)
    view = TraceView(session)
    view.resize(1200, 600)
    return session, view, doc.doc_id


@pytest.fixture
def rig(qapp):
    return _build()


def _xr(view) -> tuple[float, float]:
    lo, hi = view.anchor_plot().vb.viewRange()[0]
    return float(lo), float(hi)


def _settle(view) -> None:
    """The 250 ms debounce fires: commit the current view into the history, deterministically."""
    view._settle.stop()
    view._commit_view()


def _approx(rng: tuple[float, float]):
    return (pytest.approx(rng[0], rel=1e-9, abs=1e-9),
            pytest.approx(rng[1], rel=1e-9, abs=1e-9))


def _y_floor(pane) -> float | None:
    return pane.plot.vb.state["limits"]["yRange"][0]


# ================================================================================================
# VIEWER-001 — a quick Back/Forward must treat the in-flight framing as real
# ================================================================================================

def test_quick_back_returns_to_the_previous_framing_not_two(rig):
    """Browse at B, dive to C, press Back inside the settle window: land on B (one step),
    with C still reachable via Forward. The bug landed on A and lost C entirely."""
    _session, view, _did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 2.0, c + 2.0); _settle(view)     # framing A (settled)
    view.zoom_x_to(c - 0.5, c + 0.5); _settle(view)     # framing B (settled)
    frame_b = _xr(view)
    view.zoom_x_to(c - 0.01, c + 0.01)                   # dive to C — do NOT settle
    frame_c = _xr(view)
    assert view._settle.isActive()                       # the dive really is mid-debounce

    view.view_back()
    assert _xr(view) == _approx(frame_b)                 # one step back, not two
    view.view_forward()
    assert _xr(view) == _approx(frame_c)                 # the dive survived into history


def test_first_zoom_quick_back_is_not_a_silent_noop(rig):
    """Degenerate case: the very first zoom after a load, Back pressed immediately. The bug
    left can_go_back() False (nothing committed yet) and Back did nothing at all."""
    _session, view, _did = rig
    initial = _xr(view)
    lo, hi = initial
    c = (lo + hi) / 2
    view.zoom_x_to(c - 0.01, c + 0.01)                   # quick dive, un-settled
    dive = _xr(view)
    assert view._settle.isActive()

    view.view_back()
    assert _xr(view) == _approx(initial)                 # returned to the load framing
    assert view.can_go_forward()
    view.view_forward()
    assert _xr(view) == _approx(dive)


def test_quick_forward_prunes_like_a_slow_forward(rig):
    """A new zoom prunes the forward branch; the debounce must not resurrect it. Quick zoom
    then Forward behaves exactly like waiting 250 ms then Forward: no step, branch gone."""
    _session, view, _did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 1.8, c + 1.8); _settle(view)     # A (distinct from the full extent)
    frame_a = _xr(view)
    view.zoom_x_to(c - 0.5, c + 0.5); _settle(view)     # B
    view.view_back()                                     # back to A; B is the forward branch
    view.zoom_x_to(c - 1.2, c + 1.2)                     # new zoom A' — un-settled
    frame_a2 = _xr(view)

    view.view_forward()                                  # must NOT restore B
    assert _xr(view) == _approx(frame_a2)
    assert not view.can_go_forward()                     # branch pruned, same as settled path
    view.view_back()                                     # ...and A' was committed, so Back
    assert _xr(view) == _approx(frame_a)                 # steps to A, not to A's predecessor


# ================================================================================================
# VIEWER-002 — calibration units change re-keys the y-axis label and the min-y-span floor
# ================================================================================================

def test_units_change_relabels_axis_and_rekeys_y_floor(rig):
    """pA -> nA: the axis label and the floor must follow the data, or the axis disagrees
    1000x with the crosshair readout beside it and the pA floor makes nA data unzoomable."""
    session, view, did = rig
    pane = view.pane_for_layer(f"{did}#0")
    assert pane.plot.getAxis("left").labelText == "pA"
    assert _y_floor(pane) == 5.0                         # the pA policy, installed at build

    session.set_calibration(did, 0,
                            ChannelCalibration(gain=0.001, offset=0.0, target_units="nA"),
                            write_sidecar=False)

    pane = view.pane_for_layer(f"{did}#0")
    assert pane.plot.getAxis("left").labelText == "nA"   # label agrees with the data
    assert session.calibrated(did, 0).units == "nA"      # ...which really is in nA now
    assert _y_floor(pane) == 0.005                       # the nA floor, not the stale 5.0


def test_units_change_to_unknown_releases_the_floor(qapp):
    """The other direction: a floor installed for a known unit must be RELEASED (not left
    lying) when the effective units go back to unknown — apply_y_policy always sets."""
    session, view, did = _build(current_units="")        # undeclared raw units
    pane = view.pane_for_layer(f"{did}#0")
    assert pane.plot.getAxis("left").labelText == "current"   # kind fallback, no unit claim
    assert _y_floor(pane) is None

    session.set_calibration(did, 0, ChannelCalibration(target_units="pA"),
                            write_sidecar=False)
    assert pane.plot.getAxis("left").labelText == "pA"
    assert _y_floor(pane) == 5.0

    session.set_calibration(did, 0, ChannelCalibration(target_units=""),
                            write_sidecar=False)
    assert pane.plot.getAxis("left").labelText == "current"
    assert _y_floor(pane) is None                        # released, not left at 5.0


def test_units_change_does_not_rebuild_or_move_the_zoom(rig):
    """refresh_channel's contract: no rebuild, no zoom change — the same pane objects
    survive and the x-window the user is inspecting stays put."""
    session, view, did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 0.01, c + 0.01); _settle(view)
    dive = _xr(view)
    panes_before = view.panes()

    session.set_calibration(did, 0, ChannelCalibration(gain=0.001, target_units="nA"),
                            write_sidecar=False)
    assert view.panes() == panes_before                  # refreshed in place, not rebuilt
    assert _xr(view) == _approx(dive)


def test_overlay_pane_with_mixed_units_stops_claiming_either(qapp):
    """Overlay grouping guarantees one unit per pane at build; a calibration edit can break
    that in place. Until the next rebuild regroups, the pane must not keep asserting a unit
    that is now false for half its curves — it falls back to the ChannelKind label and
    releases the unit-keyed floor."""
    session = Session()
    doc_a = _make_document("overlay_a.dat")
    doc_b = _make_document("overlay_b.dat")
    session.add_document(doc_a)
    session.add_document(doc_b)
    session.add_layer(doc_a.doc_id, 0)
    session.add_layer(doc_b.doc_id, 0)
    session.set_view_mode(ViewMode.OVERLAY)
    view = TraceView(session)
    view.resize(1200, 600)

    pane = view.pane_for_layer(f"{doc_a.doc_id}#0")
    assert pane is view.pane_for_layer(f"{doc_b.doc_id}#0")   # grouped: one (CURRENT, pA) pane
    assert pane.plot.getAxis("left").labelText == "pA"

    session.set_calibration(doc_b.doc_id, 0,
                            ChannelCalibration(gain=0.001, target_units="nA"),
                            write_sidecar=False)
    assert pane.plot.getAxis("left").labelText == "current"   # neither pA nor nA is true here
    assert _y_floor(pane) is None


# ================================================================================================
# VIEWER-003 — the Back/Forward trail survives rebuilds that keep the same documents
# ================================================================================================

def test_layer_toggle_keeps_zoom_history(rig):
    """Untick the voltage channel while dived in: the zoom is kept (it already was) and the
    trail is too — Back steps to the pre-dive framing instead of silently doing nothing."""
    session, view, did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 1.0, c + 1.0); _settle(view)     # browse (settled)
    browse = _xr(view)
    view.zoom_x_to(c - 0.01, c + 0.01); _settle(view)   # dive (settled)
    dive = _xr(view)
    assert len(view._view_hist) == 3

    session.set_layer_visible(f"{did}#1", False)         # the Files-dock visibility toggle
    assert _xr(view) == _approx(dive)                    # zoom preserved (pre-existing)
    assert len(view._view_hist) == 3                     # trail preserved (the fix)
    assert not view._settle.isActive()                   # the restore itself was not recorded

    view.view_back()
    assert _xr(view) == _approx(browse)
    view.view_forward()
    assert _xr(view) == _approx(dive)


def test_view_mode_switch_keeps_zoom_history(rig):
    """The Overlay/Stacked switch is the other rebuild trigger a user hits mid-inspection."""
    session, view, _did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 1.0, c + 1.0); _settle(view)
    browse = _xr(view)
    view.zoom_x_to(c - 0.01, c + 0.01); _settle(view)

    session.set_view_mode(ViewMode.STACKED)
    assert len(view._view_hist) == 3
    view.view_back()
    assert _xr(view) == _approx(browse)


def test_new_document_reseeds_history(rig):
    """The preservation must NOT outlive the documents it describes: when the shown set
    changes, the old trail's framings are about a view that no longer exists — reseed."""
    session, view, _did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 1.0, c + 1.0); _settle(view)
    assert view.can_go_back()

    doc2 = _make_document("second_recording.dat")
    session.add_document(doc2)
    session.add_layer(doc2.doc_id, 0)                    # shown documents changed -> reseed
    assert len(view._view_hist) == 1
    assert not view.can_go_back()


# ================================================================================================
# The cross cases — a rebuild mid-debounce meets a quick Back
# ================================================================================================

def test_toggle_mid_debounce_then_quick_back_lands_on_predive(rig):
    """Dive quickly, toggle a layer before the debounce fires, press Back immediately: the
    preserved trail must not resurrect VIEWER-001 — the pending dive is committed by the
    flush, Back lands one framing back and Forward recovers the dive."""
    session, view, did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 1.0, c + 1.0); _settle(view)     # browse (settled)
    browse = _xr(view)
    view.zoom_x_to(c - 0.01, c + 0.01)                   # dive — un-settled
    dive = _xr(view)

    session.set_layer_visible(f"{did}#1", False)         # rebuild while mid-debounce
    assert view._settle.isActive()                       # the pending dive survived the rebuild

    view.view_back()
    assert _xr(view) == _approx(browse)                  # one step, through the flushed dive
    view.view_forward()
    assert _xr(view) == _approx(dive)


def test_pending_dive_commits_once_after_rebuild(rig):
    """...and when the user does NOT press Back, the debounce firing after the rebuild
    commits the dive into the preserved trail exactly once."""
    session, view, did = rig
    lo, hi = _xr(view)
    c = (lo + hi) / 2
    view.zoom_x_to(c - 1.0, c + 1.0); _settle(view)     # browse (settled)
    browse = _xr(view)
    view.zoom_x_to(c - 0.01, c + 0.01)                   # dive — un-settled
    hist_len = len(view._view_hist)

    session.set_layer_visible(f"{did}#1", False)         # rebuild while mid-debounce
    assert view._settle.isActive()
    _settle(view)                                        # the pending debounce fires
    assert len(view._view_hist) == hist_len + 1          # committed once, not lost/doubled
    view.view_back()
    assert _xr(view) == _approx(browse)
