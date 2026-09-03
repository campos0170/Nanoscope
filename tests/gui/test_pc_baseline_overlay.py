"""The estimated baseline, made visible: the rail on the trace and the residual view.

The owner asked for two pictures — "the original recording with the estimated baseline
overlaid" and "the baseline-subtracted signal" — and the audit found the parts for the
first already built and never called: ``analysis.baseline.estimated_wander`` says in its
own docstring that it exists "so a caller can OVERLAY it on the raw trace", and
``gui.overlays.render_hline`` was registered for ``OverlayKind.HLINE`` and commented
"# baseline track". Neither had a caller outside tests.

WHY THIS FILE LEANS ON PIXELS RATHER THAN ON THE SCENE GRAPH. Wiring the renderer as it
stood would have shipped the feature and not the picture: it drew ``width=1``, dashed, at
``zValue=-4`` — UNDER the trace curve — so the auditor's screenshot had the overlay in the
scene, the values right, and nothing on screen. "The item is on the plot" is the assertion
that would have passed. So the load-bearing test here grabs the pane and COUNTS INK, on
the owner's own -1000 mV apo recording, against a control rendered without the rail: an
overlay that cannot be seen fails it no matter how correct its numbers are.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.analysis.baseline import (
    baseline_subtracted,
    baseline_track,
    characterisation_baseline,
    decimate_track,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.overlays import BASELINE_TRACK_HEX, register_all_renderers
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
REAL_ABF = (
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
    "0004_process _ -1000 mV_ 385-400 s time trace.abf"
)


# ================================================================================================
# builders
# ================================================================================================
def _drifting_trace(n: int = 200_000, i0: float = -600.0, seed: int = 3) -> np.ndarray:
    """A synthetic pore: open-pore ``i0``, a slow 15 pA sag, noise, and five blockades.

    Synthetic on purpose for the arithmetic assertions — the answers are exact and
    chosen, so "centred on zero" can be a number rather than an impression. The
    VISIBILITY assertion uses the owner's real recording instead; nothing about ink over
    real trace noise can be established with a trace this clean.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    drift = 15.0 * np.sin(2 * np.pi * t / (n / FS))
    x = i0 + drift + rng.normal(0.0, 2.0, n)
    for start in (20_000, 60_000, 100_000, 140_000, 180_000):
        x[start:start + 300] -= 80.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, data: np.ndarray, name: str = "pore.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=data.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(data,))


@pytest.fixture
def view(qapp, qtbot, tmp_path):
    """A real TraceView showing one synthetic current channel, sized like a real pane."""
    register_all_renderers()
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, _drifting_trace()))
    session.add_layer(doc_id, 0)
    tv = TraceView(session)
    qtbot.addWidget(tv)
    tv.resize(1000, 400)
    tv.show()
    qtbot.waitExposed(tv)
    return tv, session, doc_id


def _curve(view: TraceView, doc_id: str, channel: int = 0):
    pane = view.pane_for_layer(f"{doc_id}#{channel}")
    return pane._curves[f"{doc_id}#{channel}"]


def _rails(view: TraceView, doc_id: str, channel: int = 0):
    """The rail's items, in the order the renderer added them (casing, then ink)."""
    pane = view.pane_for_layer(f"{doc_id}#{channel}")
    ink = QtGui.QColor(BASELINE_TRACK_HEX).name()
    out = []
    for item in pane.plot.items:
        pen = getattr(item, "opts", {}).get("pen") if hasattr(item, "opts") else None
        if pen is None or not isinstance(pen, QtGui.QPen):
            continue
        if item is _curve(view, doc_id, channel):
            continue
        if pen.color().name() in (ink, "#10151b"):
            out.append(item)
    return sorted(out, key=lambda it: it.zValue())


# ================================================================================================
# A. The trap: drawn is not the same as visible.
# ================================================================================================
def test_the_rail_is_drawn_above_the_trace_and_not_underneath_it(view):
    """z-order and width, pinned as numbers, because they are the whole feature.

    ``render_hline`` shipped at ``zValue=-4``, width 1, one item. The trace curve is at
    z 0, so every pixel of the rail that mattered — the pixels where it crosses the
    signal — was painted over by the signal.
    """
    tv, _session, doc_id = view
    idx, values = baseline_track(np.asarray(_curve(tv, doc_id).yData), FS, polarity=-1)
    assert tv.set_baseline_track(doc_id, 0, idx, values) is True

    rails = _rails(tv, doc_id)
    assert len(rails) == 2, "a cased rail is two strokes: a dark rail and the ink on it"
    casing, ink = rails
    curve_z = _curve(tv, doc_id).zValue()
    assert casing.zValue() > curve_z and ink.zValue() > casing.zValue(), (
        f"the rail must sit ON the trace, not under it: casing z={casing.zValue()}, "
        f"ink z={ink.zValue()}, curve z={curve_z}")
    assert ink.zValue() < 5, "...and still under the fitted levels (5/6) and marker (10)"
    assert casing.opts["pen"].width() > ink.opts["pen"].width() >= 2, (
        "the casing is a border around a rail at least 2 px wide, not a hairline")
    assert ink.opts["pen"].style() == QtCore.Qt.DashLine, (
        "dashed: it is an annotation, not a second signal")


@pytest.mark.skipif(not __import__("pathlib").Path(REAL_ABF).exists(),
                    reason="the owner's -1000 mV apo recording is not on this machine")
def test_the_rail_paints_ink_over_the_owners_own_recording(qapp, qtbot):
    """THE LOAD-BEARING ONE: pixels of rail counted on his real trace, against a control.

    The auditor's screenshot is the reason this is a pixel test. Everything about the
    old renderer was correct except that you could not see it, and no assertion about
    the scene graph can tell those two apart. So: grab the pane with the rail and
    without it, and require that the rail's own ink actually appears — thousands of
    pixels of it, not the handful a hairline under the curve leaked through.
    """
    from nanoscope.io.abf import AbfReader

    register_all_renderers()
    doc = AbfReader(__import__("pathlib").Path(REAL_ABF)).load()
    fs = float(doc.meta.sample_rate_hz)
    session = Session()
    doc_id = session.add_document(doc)
    session.add_layer(doc_id, 2)          # ch2 is the current channel in this file
    tv = TraceView(session)
    qtbot.addWidget(tv)
    # THE APP'S OWN PANE COLOUR (``gui.app.configure_pyqtgraph``), set on this widget
    # rather than through the global pyqtgraph option: a bare TraceView inherits
    # pyqtgraph's default WHITE, and counting white rail pixels on a white pane would
    # measure the background. Set here and not process-wide because the option is global
    # and this file is not the only test in the session.
    tv.setBackground("#111316")
    tv.resize(1000, 400)
    tv.show()
    qtbot.waitExposed(tv)
    qtbot.wait(50)

    def ink_pixels() -> int:
        img = tv.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(
            img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]
        want = np.array([QtGui.QColor(BASELINE_TRACK_HEX).blue(),
                         QtGui.QColor(BASELINE_TRACK_HEX).green(),
                         QtGui.QColor(BASELINE_TRACK_HEX).red()], dtype=np.int16)
        return int((np.abs(arr.astype(np.int16) - want).max(axis=2) <= 24).sum())

    before = ink_pixels()
    idx, values = baseline_track(np.asarray(session.calibrated(doc_id, 2).data), fs,
                                 polarity=-1)
    assert tv.set_baseline_track(doc_id, 2, idx, values) is True
    qtbot.wait(60)
    after = ink_pixels()

    assert before == 0, f"the rail's ink is not already on the pane ({before} px)"
    assert after >= 1500, (
        f"the baseline is ON SCREEN over his own recording: {after} px of rail ink "
        f"(a hairline under the curve reads in the low hundreds at best)")


# ================================================================================================
# B. The second picture: the signal minus its baseline.
# ================================================================================================
def test_the_subtracted_view_centres_the_trace_on_zero_and_the_axis_says_so(view):
    """The residual is centred, and the pane admits it is not showing raw current.

    The one real hazard of a MODE is forgetting you are in it: a subtracted trace looks
    exactly like a well-behaved recording with no drift, so a user who walks away and
    comes back would read "0 pA open pore" off a -600 pA pore. The y-axis label is the
    surface that is always beside the curve, so that is where it is said — and it is
    taken back the moment the mode is.
    """
    tv, session, doc_id = view
    pane = tv.pane_for_layer(f"{doc_id}#0")
    raw = np.asarray(session.calibrated(doc_id, 0).data)
    assert pane.plot.getAxis("left").labelText == "pA"
    assert abs(float(np.median(raw)) + 600.0) < 25.0, "the synthetic pore sits near -600 pA"

    base = characterisation_baseline(raw, FS, polarity=-1, health=False).baseline
    assert tv.set_baseline_subtracted(doc_id, 0, base) is True

    shown = np.asarray(_curve(tv, doc_id).yData)
    assert abs(float(np.median(shown))) < 1.0, (
        f"the open pore is now AT zero, not near it: median {float(np.median(shown)):.3f} pA")
    assert float(np.abs(shown).max()) < 120.0, (
        "and the whole trace fits inside one blockade depth of zero")
    assert "Δ vs baseline" in pane.plot.getAxis("left").labelText, (
        f"the axis must say what it is showing: {pane.plot.getAxis('left').labelText!r}")

    tv.set_baseline_subtracted(doc_id, 0, None)
    back = np.asarray(_curve(tv, doc_id).yData)
    assert np.array_equal(back, raw), "unticking restores the recording's own samples"
    assert pane.plot.getAxis("left").labelText == "pA", "...and its own axis"


def test_two_recordings_at_different_open_pore_currents_end_up_on_one_y_axis(
        qapp, qtbot, tmp_path):
    """WHY THE RESIDUAL IS A MODE ON THIS PANE AND NOT A PANE OF ITS OWN.

    A -600 pA run and a -1000 pA run overlaid share a pane and therefore a y axis, and
    that axis has to span 400 pA of DIFFERENCE before it can show either signal — the
    two traces are two flat lines at opposite ends of it. Subtracted, both sit on zero
    and the axis spans one blockade. A second pane could not do this: it would give the
    residuals a second axis, which is the thing being removed.
    """
    register_all_renderers()
    session = Session()
    a = session.add_document(_doc(tmp_path, _drifting_trace(i0=-600.0), "a.dat"))
    b = session.add_document(_doc(tmp_path, _drifting_trace(i0=-1000.0, seed=7), "b.dat"))
    session.add_layer(a, 0)
    session.add_layer(b, 0)
    session.set_view_mode(ViewMode.OVERLAY)
    tv = TraceView(session)
    qtbot.addWidget(tv)
    tv.resize(1000, 400)
    tv.show()
    qtbot.waitExposed(tv)
    assert len(tv.panes()) == 1, "same kind and units: the overlay rule puts them in one pane"

    def span() -> float:
        ys = np.concatenate([np.asarray(_curve(tv, d).yData) for d in (a, b)])
        return float(ys.max() - ys.min())

    raw_span = span()
    for doc_id in (a, b):
        raw = np.asarray(session.calibrated(doc_id, 0).data)
        tv.set_baseline_subtracted(
            doc_id, 0, characterisation_baseline(raw, FS, polarity=-1, health=False).baseline)
    shared_span = span()
    assert raw_span > 400.0, f"raw, the axis must span the gap between the runs: {raw_span:.0f} pA"
    assert shared_span < 130.0, (
        f"subtracted, it spans one blockade instead: {shared_span:.0f} pA")
    label = tv.panes()[0].plot.getAxis("left").labelText
    assert "Δ vs baseline" in label and "mixed" not in label


def test_the_rail_lies_flat_at_zero_once_the_trace_is_subtracted(view):
    """The two controls COMPOSE, and the composition is the proof the subtraction worked.

    Ticking both is not a contradiction. The producer hands over the same numbers either
    way — the baseline as measured — and the view, which is the only thing that knows
    what its own y axis currently means, puts them where the curve is. Drawn absolute on
    a residual, the rail would hang 600 pA off the top of the pane.
    """
    tv, session, doc_id = view
    raw = np.asarray(session.calibrated(doc_id, 0).data)
    base = characterisation_baseline(raw, FS, polarity=-1, health=False).baseline
    idx, values = decimate_track(base)
    tv.set_baseline_track(doc_id, 0, idx, values)
    drawn = np.asarray(_rails(tv, doc_id)[1].yData)
    assert abs(float(np.median(drawn)) + 600.0) < 25.0, "raw: the rail is at the open pore"

    tv.set_baseline_subtracted(doc_id, 0, base)
    flat = np.asarray(_rails(tv, doc_id)[1].yData)
    assert float(np.abs(flat).max()) < 1e-6, (
        f"subtracted: the rail is the zero line ({float(np.abs(flat).max()):.4g} pA off)")
    assert len(_rails(tv, doc_id)) == 2, "and there is still ONE rail, not two stacked"

    tv.set_baseline_subtracted(doc_id, 0, None)
    again = np.asarray(_rails(tv, doc_id)[1].yData)
    assert np.allclose(again, drawn), (
        "toggling the mode twice returns the rail to where it started — it is re-issued "
        "from the values as MEASURED, never from the values last drawn")


def test_the_rail_and_the_residual_both_survive_a_stacked_overlay_rebuild(view):
    """A managed item torn down by a rebuild is put back by the rebuild.

    Switching View ▸ Stacked/Overlay destroys every pane. Both of these are standing
    REQUESTS, on the same rule as the measure region and the filter preview, or the
    checkbox goes on claiming a line that is no longer anywhere.
    """
    tv, session, doc_id = view
    raw = np.asarray(session.calibrated(doc_id, 0).data)
    base = characterisation_baseline(raw, FS, polarity=-1, health=False).baseline
    idx, values = decimate_track(base)
    tv.set_baseline_track(doc_id, 0, idx, values)
    tv.set_baseline_subtracted(doc_id, 0, base)

    session.set_view_mode(ViewMode.STACKED)

    assert len(_rails(tv, doc_id)) == 2, "the rail is back on the new pane"
    assert abs(float(np.median(np.asarray(_curve(tv, doc_id).yData)))) < 1.0, (
        "and the new pane's curve is still the residual")
    assert "Δ vs baseline" in tv.pane_for_layer(f"{doc_id}#0").plot.getAxis("left").labelText


def test_clearing_the_views_leaves_nothing_the_checkbox_no_longer_claims(view):
    """One writer: the owner clears everything and re-applies to the current selection.

    These are per-LAYER requests behind a per-SELECTION checkbox. Tick it on one run,
    select another, untick — and without this the first run's rail stays up, claimed by
    nothing on screen.
    """
    tv, session, doc_id = view
    raw = np.asarray(session.calibrated(doc_id, 0).data)
    base = characterisation_baseline(raw, FS, polarity=-1, health=False).baseline
    tv.set_baseline_track(doc_id, 0, *decimate_track(base))
    tv.set_baseline_subtracted(doc_id, 0, base)

    tv.clear_baseline_views()

    assert _rails(tv, doc_id) == [], "no rail is left on any layer"
    assert tv.has_baseline_track(doc_id, 0) is False
    assert tv.baseline_subtracted_for(doc_id, 0) is None
    assert np.array_equal(np.asarray(_curve(tv, doc_id).yData), raw)


def test_a_baseline_of_the_wrong_length_is_refused_by_name(view):
    """A residual against another recording's baseline is a picture of nothing, and
    NumPy would broadcast a length-1 baseline across the whole trace without a word."""
    tv, _session, doc_id = view
    with pytest.raises(ValueError, match="cannot be subtracted"):
        tv.set_baseline_subtracted(doc_id, 0, np.zeros(17, dtype=np.float64))


# ================================================================================================
# C. Nothing measured may move.
# ================================================================================================
def test_drawing_the_baseline_moves_no_number_anyone_measures(view):
    """Hashed before and after, because "it is only a view" is exactly what someone says
    the day a view mutates the array it was handed.

    The two things every measured column comes from are the calibrated samples and the
    baseline estimated from them. Both are hashed with both views on and both off.
    """
    tv, session, doc_id = view
    raw = np.asarray(session.calibrated(doc_id, 0).data)

    def fingerprint() -> tuple[str, str]:
        data = np.asarray(session.calibrated(doc_id, 0).data)
        base = characterisation_baseline(data, FS, polarity=-1, health=False,
                                         cache=False).baseline
        return (hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
                hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest())

    before = fingerprint()
    base = characterisation_baseline(raw, FS, polarity=-1, health=False).baseline
    tv.set_baseline_track(doc_id, 0, *decimate_track(base))
    tv.set_baseline_subtracted(doc_id, 0, base)
    during = fingerprint()
    tv.clear_baseline_views()
    after = fingerprint()

    assert before == during == after, (
        f"samples/baseline hashes moved: {before} -> {during} -> {after}")


def test_the_residual_preserves_every_depth_it_is_drawn_from(view):
    """A constant offset per sample cancels in ``open - blocked``: the picture changes,
    the physics does not. Measured on the five injected 80 pA blockades."""
    tv, session, doc_id = view
    raw = np.asarray(session.calibrated(doc_id, 0).data)
    base = np.asarray(characterisation_baseline(raw, FS, polarity=-1, health=False).baseline)
    resid = baseline_subtracted(raw, base)
    for start in (20_000, 60_000, 100_000, 140_000, 180_000):
        sl = slice(start + 50, start + 250)
        # depth = median(local open pore - sample). Per-sample, not a difference of two
        # medians: the estimator's baseline moves inside the window, and only the
        # per-sample form is the quantity `characterize` actually measures.
        raw_depth = float(np.median(base[sl] - raw[sl]))
        res_depth = float(np.median(0.0 - resid[sl]))
        assert abs(raw_depth - res_depth) < 1e-3, (
            f"depth at {start}: {raw_depth:.6f} pA raw vs {res_depth:.6f} pA subtracted")
        assert abs(raw_depth - 80.0) < 1.0, f"...and it is the 80 pA injected: {raw_depth:.3f}"


# ================================================================================================
# D. Driven in the real window, at the size of the owner's display.
# ================================================================================================
def _window_row(win, w) -> tuple[int, int, int]:
    """``(x0, x1, painted)`` of a widget's row IN WINDOW COORDINATES.

    ``painted`` is what actually reaches glass: the widget's own visible region (Qt has
    already clipped it to every scroll viewport it sits inside) intersected with the
    window. Three named buttons on this very panel measured ``painted=0`` at 1512 px in
    the audit — present, laid out, and off the right-hand edge of the owner's display —
    so "the control exists" and "the control is reachable" are separate questions and
    this answers the second.
    """
    region = w.visibleRegion()
    if region.isEmpty():
        return 0, 0, 0
    rect = region.boundingRect()
    top_left = w.mapTo(win, rect.topLeft())
    x0, x1 = top_left.x(), top_left.x() + rect.width()
    painted = max(0, min(x1, win.width()) - max(x0, 0))
    return x0, x1, painted


@pytest.mark.skipif(not __import__("pathlib").Path(REAL_ABF).exists(),
                    reason="the owner's -1000 mV apo recording is not on this machine")
def test_both_boxes_are_reachable_and_draw_the_baseline_in_the_real_window(qapp, qtbot):
    """THE WHOLE FEATURE, driven: a 1512x900 window, his recording, two clicks.

    Reachability is measured rather than assumed, because on this exact panel the audit
    found 'Save as…', 'Rename' and 'Delete' painting ZERO pixels at 1512 px — laid out
    past the right edge of a 420 px dock viewport, invisible until a user notices a
    horizontal scrollbar inside a dock and drags it. A control the owner cannot click is
    not a control.

    The connection from these boxes to the trace view lives in ``gui/analysis_panel.py``,
    which this pass does not own, so the test makes that connection itself — the exact
    five lines handed over as wiring — and then drives the CHECKBOX, not the slot.
    """
    from pathlib import Path

    from nanoscope.gui.app import create_app
    from nanoscope.gui.main_window import MainWindow

    create_app([])
    register_all_renderers()
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1512, 900)
    win.show()
    qtbot.waitExposed(win)
    win.open_paths([Path(REAL_ABF)])
    qtbot.waitUntil(lambda: len(win._session.layers()) > 0, timeout=20_000)
    qtbot.wait(100)

    panel = win._analysis_panel
    win._inspector_tabs.setCurrentWidget(panel)
    panel._detection.mode_tabs.setCurrentIndex(0)          # the Preprocess page
    qtbot.wait(50)
    bp = panel.baseline_panel

    # -- THE WIRING, as it will be applied in AnalysisPanel ------------------------
    view = win._trace_view

    def _apply_baseline_views() -> None:
        view.clear_baseline_views()
        sel = panel._sel
        if sel is None or not (bp.show_baseline() or bp.subtract_baseline()):
            return
        doc_id, channel = sel
        data = np.asarray(win._session.calibrated(doc_id, channel).data)
        fs = win._session.calibrated(doc_id, channel).sample_rate_hz
        pol = panel._detection.params().get("polarity")
        result = characterisation_baseline(
            data, float(fs), config=bp.config(),
            polarity=None if pol is None else int(pol), health=False)
        if bp.subtract_baseline():
            view.set_baseline_subtracted(doc_id, channel, result.baseline)
        if bp.show_baseline():
            view.set_baseline_track(doc_id, channel, *decimate_track(result.baseline))

    bp.showBaselineToggled.connect(lambda _on: _apply_baseline_views())
    bp.subtractBaselineToggled.connect(lambda _on: _apply_baseline_views())

    # -- 1. reachable at his own display size -------------------------------------
    rows = {}
    for name, w in (("show", bp.show_track), ("subtract", bp.subtract_view),
                    ("caption", bp.which_baseline_label)):
        x0, x1, painted = _window_row(win, w)
        rows[name] = (x0, x1, painted)
        assert w.isVisible(), f"{name} is not visible at 1512x900"
        assert painted >= 120, (
            f"{name} paints only {painted} px at 1512x900 (x {x0}..{x1}); the audit found "
            f"three buttons on this very panel painting 0 px the same way")
        assert x1 <= win.width(), (
            f"{name} ends at x={x1}, past the right edge of a {win.width()} px window")

    # -- 2. clicked, not called ----------------------------------------------------
    doc_id, channel = panel._sel
    for box in (bp.show_track, bp.subtract_view):
        QtWidgets.QApplication.processEvents()
        from PySide6.QtTest import QTest
        QTest.mouseClick(box, QtCore.Qt.LeftButton, pos=QtCore.QPoint(8, box.height() // 2))
        qtbot.wait(200)

    assert bp.show_baseline() is True and bp.subtract_baseline() is True, (
        f"the clicks landed on the boxes: {rows}")
    assert view.has_baseline_track(doc_id, channel) is True, "the rail is up on his trace"
    assert view.baseline_subtracted_for(doc_id, channel) is not None
    pane = view.pane_for_layer(f"{doc_id}#{channel}")
    shown = np.asarray(pane._curves[f"{doc_id}#{channel}"].yData)
    assert abs(float(np.median(shown))) < 2.0, (
        f"the trace on screen is the residual, centred on zero: median "
        f"{float(np.median(shown)):.3f} pA")
    assert "Δ vs baseline" in pane.plot.getAxis("left").labelText

    # -- 3. and the boxes did not move anything under the pointer -------------------
    for name, w in (("show", bp.show_track), ("subtract", bp.subtract_view),
                    ("caption", bp.which_baseline_label)):
        assert _window_row(win, w) == rows[name], (
            f"{name} moved when the boxes were ticked: {_window_row(win, w)} != {rows[name]}")
