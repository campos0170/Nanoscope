"""Regression tests for VIEWER-004 (gui/navigator.py).

The Navigator overview strip used to bind to the FIRST visible layer, full stop. ABF channel
order puts the command voltage (IN14) before the current (IN15), so on every standard
multi-channel ABF the "where am I" map drew a constant -1000 mV line — no events, nothing to
scrub to — while the current channel was never shown.

Contract under test (Navigator._overview_layer):

  - PREFER CURRENT: with a voltage layer ahead of a current layer (the real ABF shape), the
    overview draws the CURRENT channel, not the flat command voltage;
  - FALLBACK: with no visible current layer (voltage-only recording, or current unticked),
    the overview falls back to the first visible layer instead of going blank;
  - REBIND ON VISIBILITY: unticking / re-ticking the current layer moves the overview to the
    fallback and back — the pre-existing rebind-on-layersChanged behaviour must keep working.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.navigator import Navigator
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
N = 20_000           # 0.4 s at 50 kHz — enough samples, cheap to build


def _abf_shaped_doc(tmp_path, name: str = "abf_shaped.dat") -> TraceDocument:
    """Voltage at index 0, current at index 1 — the channel ORDER that triggered VIEWER-004."""
    rng = np.random.default_rng(4)
    cur = rng.standard_normal(N).astype(np.float32) - 50.0     # current with visible texture
    volt = np.full(N, -1000.0, dtype=np.float32)               # flat command voltage
    v = ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    c = ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=Path(tmp_path) / name, format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(v, c))
    return TraceDocument(meta=meta, channels=(volt, cur))


def _voltage_only_doc(tmp_path, name: str = "volt_only.dat") -> TraceDocument:
    volt = np.full(N, -600.0, dtype=np.float32)
    v = ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=Path(tmp_path) / name, format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(v,))
    return TraceDocument(meta=meta, channels=(volt,))


def _build(doc):
    """Session + TraceView + a bound Navigator — the same wiring MainWindow performs.
    Layers are added in channel order, exactly as _create_default_layers would."""
    session = Session()
    session.add_document(doc)
    for info in doc.meta.channels:
        session.add_layer(doc.doc_id, info.index)
    view = TraceView(session)
    view.resize(1200, 600)
    nav = Navigator()
    nav.bind(session, view)
    return session, view, nav


def _drawn_channel(session: Session, nav: Navigator, doc_id: str) -> int | None:
    """Which channel index the overview curve's y-data equals, via the same calibrated door
    the Navigator itself reads through. None when nothing (or something else) is drawn."""
    if nav._curve is None:
        return None
    ydata = np.asarray(nav._curve.yData)
    for info in session.document(doc_id).meta.channels:
        chan = np.asarray(session.calibrated(doc_id, info.index).data)
        if ydata.size == chan.size and np.array_equal(ydata, chan):
            return info.index
    return None


# ================================================================================================
# 1. prefer current — the VIEWER-004 shape itself
# ================================================================================================

def test_overview_draws_current_not_command_voltage(qapp, tmp_path):
    """Voltage layer FIRST (ABF order): the overview must still draw the current channel."""
    doc = _abf_shaped_doc(tmp_path)
    session, _view, nav = _build(doc)
    assert _drawn_channel(session, nav, doc.doc_id) == 1  # IN15, CURRENT — not the flat IN14


def test_overview_curve_wears_the_current_layers_color(qapp, tmp_path):
    """The curve must carry the CURRENT layer's colour — proof it bound to that layer, not
    merely to data that happens to match."""
    doc = _abf_shaped_doc(tmp_path)
    session, _view, nav = _build(doc)
    current_layer = next(l for l in session.layers() if l.channel == 1)
    assert nav._curve.opts["pen"].color().name().lower() == current_layer.color.lower()


# ================================================================================================
# 2. fallback — no visible current channel
# ================================================================================================

def test_voltage_only_recording_still_gets_an_overview(qapp, tmp_path):
    """No current channel at all: fall back to the first visible layer rather than a blank strip."""
    doc = _voltage_only_doc(tmp_path)
    session, _view, nav = _build(doc)
    assert _drawn_channel(session, nav, doc.doc_id) == 0
    assert nav._region.isVisible() is True


# ================================================================================================
# 3. rebind on visibility toggles — the behaviour that already worked must keep working
# ================================================================================================

def test_unticking_current_falls_back_then_reticking_restores(qapp, tmp_path):
    doc = _abf_shaped_doc(tmp_path)
    session, _view, nav = _build(doc)
    current_layer = next(l for l in session.layers() if l.channel == 1)

    session.set_layer_visible(current_layer.layer_id, False)   # untick current in the Files dock
    assert _drawn_channel(session, nav, doc.doc_id) == 0, \
        "overview did not fall back to the remaining visible layer"

    session.set_layer_visible(current_layer.layer_id, True)    # re-tick
    assert _drawn_channel(session, nav, doc.doc_id) == 1, \
        "overview did not rebind to the current channel on re-tick"


def test_all_layers_hidden_hides_the_region(qapp, tmp_path):
    doc = _abf_shaped_doc(tmp_path)
    session, _view, nav = _build(doc)
    for layer in session.layers():
        session.set_layer_visible(layer.layer_id, False)
    assert nav._curve is None
    assert nav._region.isVisible() is False
