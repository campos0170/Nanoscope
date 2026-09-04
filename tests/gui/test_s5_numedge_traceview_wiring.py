"""Sweep-5 NE-05 / NE-06 — the WIRING half: the trace panes actually use the fixed axis.

``tests/gui/test_s5_numedge_linear_ticks.py`` pins the formatter itself
(:class:`nanoscope.gui.axes.ReadableLinearAxis`: precision from tick SPACING, one
notation per axis). That is necessary but not sufficient — both defects shipped
because ``TraceView.rebuild`` constructed pyqtgraph's DEFAULT ``AxisItem``, so the
fixed formatter never ran on the one surface the dossier quotes. These tests pin the
wiring end-to-end: a real :class:`TraceView` (and the same panes inside a wired
:class:`MainWindow`), a real paint via ``grab()``, and the labels read from the very
axis objects the panes draw with.

Idiom note: tick labels are asserted through ``tickValues``/``tickStrings`` on the
pane's own axis AFTER a paint, not through ``generateDrawSpecs`` — the collision
filter legitimately thins the drawn set by pane width, and these tests are about
what the formatter says, not how many labels fit at 1200 px.
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
from nanoscope.gui.axes import ReadableLinearAxis
from nanoscope.gui.trace_view import TraceView


def _day_deep_session(n: int = 250_000, fs: float = 250_000.0) -> Session:
    """One current channel whose instrument clock starts a day in (t0 = 86,400 s)."""
    rng = np.random.default_rng(5)
    arr = np.ascontiguousarray(100.0 + rng.normal(0, 2, n), dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=Path("day.dat"), format_name="synthetic", n_samples=n,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META,
                         channels=(info,), t0_s=86_400.0)
    sess = Session()
    did = sess.add_document(TraceDocument(meta=meta, channels=(arr,)))
    sess.add_layer(did, 0)
    return sess


def _shown_labels(axis, lo: float, hi: float, px: int = 800) -> list[str]:
    """The major-tick labels the axis formatter produces for the view [lo, hi]."""
    spacing, vals = axis.tickValues(lo, hi, px)[0]
    return [s for s in axis.tickStrings(vals, 1.0, spacing) if s]


@pytest.fixture
def view(qapp, qtbot):
    v = TraceView(_day_deep_session())
    qtbot.addWidget(v)
    v.resize(1200, 600)
    v.show()
    qapp.processEvents()
    v.grab()          # a real offscreen paint, so axis geometry is live
    return v


# ================================================================================================
# The wiring itself: every pane's axes are ReadableLinearAxis, not pyqtgraph's default
# ================================================================================================

def test_every_pane_axis_is_the_readable_linear_axis(view):
    """The seam fix is dead code on this surface unless rebuild() passes axisItems=.

    Checked per pane and per side, because rebuild() constructs the panes in a loop —
    a regression that wires only the first pane (or only "bottom") must fail here.
    """
    assert view._panes, "the fixture session must produce at least one pane"
    for pane in view._panes:
        for side in ("bottom", "left"):
            axis = pane.plot.getAxis(side)
            assert isinstance(axis, ReadableLinearAxis), (
                f"trace pane {side} axis is {type(axis).__name__}, not "
                f"ReadableLinearAxis — NE-05/NE-06 regress the moment the default "
                f"AxisItem comes back")


def test_rebuild_keeps_the_wiring(view, qapp):
    """rebuild() tears every pane down and re-creates them; the axes must survive it."""
    view.rebuild()
    qapp.processEvents()
    view.grab()
    for pane in view._panes:
        for side in ("bottom", "left"):
            assert isinstance(pane.plot.getAxis(side), ReadableLinearAxis)


# ================================================================================================
# NE-05 on the shipped surface: a day-deep 100 us zoom labels WHERE, not "86400" x6
# ================================================================================================

def test_ne05_day_deep_zoom_labels_are_distinct_on_the_trace_pane(view, qapp):
    pane = view._panes[-1]
    # Box zoom and every programmatic x-framing gesture land in this ViewBox range set.
    pane.plot.vb.setXRange(86_400.0, 86_400.0001, padding=0)
    qapp.processEvents()
    view.grab()
    labels = _shown_labels(pane.plot.getAxis("bottom"), 86_400.0, 86_400.0001)
    assert len(labels) >= 3
    assert len(set(labels)) == len(labels), (
        f"time-axis labels repeat on a 100 us window a day in: {labels}")
    assert "86400" not in labels, (
        f"a bare '86400' means precision followed magnitude again: {labels}")


# ================================================================================================
# NE-06 on the shipped surface: a 1e6 pA span keeps ONE notation on the left axis
# ================================================================================================

def test_ne06_megapicoamp_span_uses_one_notation_on_the_trace_pane(view, qapp):
    pane = view._panes[-1]
    pane.plot.vb.setYRange(-44_387.0, 1.04439e+06, padding=0)
    qapp.processEvents()
    view.grab()
    labels = _shown_labels(pane.plot.getAxis("left"), -44_387.0, 1.04439e+06, px=600)
    assert labels
    exp = [s for s in labels if ("e+" in s or "e-" in s or "·10" in s)]
    plain = [s for s in labels if s not in exp and s != "0"]
    assert not (exp and plain), (
        f"two notations on one linear axis: plain {plain} beside exponent {exp}")


# ================================================================================================
# Flip side: an ordinary window keeps ordinary labels (no precision spam, no regressions)
# ================================================================================================

def test_ordinary_window_labels_stay_ordinary(view, qapp):
    """A full-trace view (1 s of data from t0=86,400) must not grow spurious digits."""
    pane = view._panes[-1]
    pane.plot.vb.setXRange(86_400.0, 86_401.0, padding=0)
    qapp.processEvents()
    view.grab()
    labels = _shown_labels(pane.plot.getAxis("bottom"), 86_400.0, 86_401.0)
    assert len(labels) >= 3 and len(set(labels)) == len(labels)
    # Spacing here is >= 0.1 s, so one decimal place suffices; the formatter must not
    # print micro-second digits on a one-second window.
    assert all(len(s.split(".", 1)[1]) <= 2 for s in labels if "." in s), labels
