"""S2 regression — VIEWER2-001: a units-changing recalibration while box-zoomed.

A RectMode drag (the app's default drag mode) ends in ``ViewBox.showAxRect``, which
turns y-autorange OFF. ``refresh_channel`` then re-set the curve data and
``refresh_units`` relabelled the axis, but the viewbox kept the OLD numeric y-window:
after pA -> nA (gain 0.001) the pane framed -85..-45 under an axis reading "nA" while
the signal now lived at -0.08..-0.04 — a blank pane whose axis shows numbers the
signal never had, disagreeing 1000x with the crosshair readout beside it.

The fix carries the kept y-window through the same old->new affine map the data went
through (``ChannelPane.remap_y_window``), derived from the curve data itself because
the Session no longer holds the old gain/offset when ``calibrationChanged`` fires.
Pinned here:

  * the box-zoomed window is rescaled so the same signal stays framed (with and
    without an offset in the new calibration), y-autorange stays off;
  * a SAME-unit gain change keeps the numeric window untouched — exactly what it
    does today (the window is still meaningful in the same unit);
  * a pane whose y-autorange is still on (no manual y framing) is left to refit itself;
  * a constant trace (affine map underdetermined) falls back to re-enabling
    y-autorange — re-derive, never guess.

Driven offscreen through the real Session + TraceView on synthetic documents,
mirroring tests/gui/test_beta_viewer_history_units.py.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
N = 100_000          # 2 s at 50 kHz


def _make_document(name: str = "s2_units_zoom.dat", *, constant: bool = False) -> TraceDocument:
    """One current channel around -65 pA (or exactly constant for the fallback case)."""
    if constant:
        cur = np.full(N, -65.0, dtype=np.float32)
    else:
        rng = np.random.default_rng(11)
        cur = (-65.0 + rng.normal(0.0, 5.0, N)).astype(np.float32)
    info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=Path(f"/tmp/{name}"),
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(cur,))


def _build(*, constant: bool = False):
    session = Session()
    doc = _make_document(constant=constant)
    session.add_document(doc)
    session.add_layer(doc.doc_id, 0)
    view = TraceView(session)
    view.resize(1200, 600)
    return session, view, doc.doc_id


def _box_zoom(view, did, *, x0=0.5, y0=-85.0, w=0.5, h=40.0):
    """End a RectMode drag exactly as pyqtgraph does: showAxRect on the pane's viewbox.
    This is what turns y-autorange off — the precondition the defect needed."""
    pane = view.pane_for_layer(f"{did}#0")
    pane.plot.vb.showAxRect(QtCore.QRectF(x0, y0, w, h))
    assert not bool(pane.plot.vb.autoRangeEnabled()[1]), "box-zoom must disable y-autorange"
    return pane


def _y_window(pane) -> tuple[float, float]:
    lo, hi = pane.plot.vb.viewRange()[1]
    return float(lo), float(hi)


# ================================================================================================
# the defect: pA -> nA while box-zoomed
# ================================================================================================

def test_units_change_rescales_the_boxzoomed_y_window(qapp):
    """After pA -> nA (gain 0.001) the kept window must be the OLD window mapped through
    the calibration, so the same signal stays framed — not the old pA numbers under a
    nA label with the signal entirely off-screen."""
    session, view, did = _build()
    pane = _box_zoom(view, did)          # y window: ~-85 .. -45 pA (plus viewbox padding)
    lo0, hi0 = _y_window(pane)           # the ACTUAL kept window, padding included

    session.set_calibration(did, 0, ChannelCalibration(gain=0.001, target_units="nA"),
                            write_sidecar=False)

    lo, hi = _y_window(pane)
    assert lo == pytest.approx(lo0 * 0.001, rel=1e-4)
    assert hi == pytest.approx(hi0 * 0.001, rel=1e-4)
    # The signal itself is framed again: its median lies inside the kept window.
    med = float(np.median(session.calibrated(did, 0).data))
    assert lo < med < hi, (
        f"signal median {med} nA is outside the y-window ({lo}, {hi}) — "
        "the blank-pane defect")
    # The user's zoom is preserved (rescaled), not thrown away to autorange.
    assert not bool(pane.plot.vb.autoRangeEnabled()[1])
    assert pane.plot.getAxis("left").labelText == "nA"


def test_units_change_with_offset_maps_the_window_affinely(qapp):
    """The map is affine, not a bare ratio: calibrated = raw*gain + offset. A units
    change that also shifts the level must land the window where the signal is."""
    session, view, did = _build()
    pane = _box_zoom(view, did)          # ~-85 .. -45 pA
    lo0, hi0 = _y_window(pane)

    session.set_calibration(did, 0,
                            ChannelCalibration(gain=0.001, offset=1.0, target_units="nA"),
                            write_sidecar=False)

    lo, hi = _y_window(pane)
    # old calibrated v (gain 1, offset 0) -> raw v -> new v*0.001 + 1.0
    assert lo == pytest.approx(lo0 * 0.001 + 1.0, rel=1e-4)
    assert hi == pytest.approx(hi0 * 0.001 + 1.0, rel=1e-4)
    med = float(np.median(session.calibrated(did, 0).data))
    assert lo < med < hi


# ================================================================================================
# the behaviours the fix must NOT change
# ================================================================================================

def test_same_unit_gain_change_keeps_the_numeric_window(qapp):
    """A gain tweak that stays in pA keeps doing what it does today: the numeric
    window is still meaningful in the same unit and is left exactly where the
    user put it."""
    session, view, did = _build()
    pane = _box_zoom(view, did)
    before = _y_window(pane)

    session.set_calibration(did, 0, ChannelCalibration(gain=2.0, target_units="pA"),
                            write_sidecar=False)

    assert _y_window(pane) == pytest.approx(before)
    assert not bool(pane.plot.vb.autoRangeEnabled()[1])
    assert pane.plot.getAxis("left").labelText == "pA"


def test_autoranging_pane_is_left_to_refit_itself(qapp):
    """Wheel-only zoom never disables y-autorange; a units change must not start
    imposing explicit windows on a pane that will refit itself."""
    session, view, did = _build()
    pane = view.pane_for_layer(f"{did}#0")
    assert bool(pane.plot.vb.autoRangeEnabled()[1])      # fresh load: y-autorange on

    session.set_calibration(did, 0, ChannelCalibration(gain=0.001, target_units="nA"),
                            write_sidecar=False)

    assert bool(pane.plot.vb.autoRangeEnabled()[1]), (
        "y-autorange was on and must stay on — autorange refits to the new units by itself")


def test_constant_trace_falls_back_to_autorange(qapp):
    """A constant trace gives no two distinct samples to derive the affine map from:
    the honest fallback is to re-enable y-autorange (re-derive the frame from the
    data), never to keep the old-unit numbers or guess a map."""
    session, view, did = _build(constant=True)
    pane = _box_zoom(view, did)

    session.set_calibration(did, 0, ChannelCalibration(gain=0.001, target_units="nA"),
                            write_sidecar=False)

    assert bool(pane.plot.vb.autoRangeEnabled()[1]), (
        "underdetermined map: the pane must re-derive its frame via y-autorange")
