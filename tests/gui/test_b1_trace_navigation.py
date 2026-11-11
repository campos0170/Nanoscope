"""Plot Studio phase 1, track B1 — navigating the main trace.

Three gaps from the plotting comparison (section 5: Zoom history, Crosshair and data cursor,
Event navigation), each pinned here on synthetic data:

* **View ▸ Zoom Back / Zoom Forward were never greyed out.** ``TraceView`` knew
  (``can_go_back`` / ``can_go_forward``) and the menu was never told, so a fresh load offered a
  Zoom Back that did nothing. The view now announces ``sigZoomHistoryChanged``. The state it
  announces counts a zoom still inside the 250 ms debounce as a step Back will take, because a
  disabled QAction drops its shortcut as well: following the committed trail alone would have
  swallowed the quick Ctrl+← after a dive that ``_commit_pending`` exists to honour.
* **The crosshair gave a time and a value but no sample index.** It now names the snapped
  sample in the event table's ``start_idx`` / ``end_idx`` numbering (zero at the first LOADED
  sample), and where that is not the file's numbering it adds the file's (a partial load) or
  the parent's (a segment).
* **No jump to an event ID or a time.** View ▸ Go to event… (Ctrl+J) and Go to time…
  (Ctrl+Shift+J), and an ID… door in the Event inspector, whose Event box is a POSITION in the
  filter and stays one.

SYNTHETIC ONLY: every recording is arrays in a ``TraceDocument`` whose path does not exist.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta, SampleRange
from nanoscope.gui.trace_view import TraceView

from ._inspector_hosting import (  # noqa: F401  (fixtures are used by name)
    SPANS, build_window, close_window, docked, private_settings, settle)
from .trace_rig import drag, free_mouse, point_at, rig, zoom  # noqa: F401  (fixtures)

Qt = QtCore.Qt


def _settle_history(view: TraceView) -> None:
    """Let the 250 ms history debounce commit, without waiting for it."""
    if view._settle.isActive():
        view._settle.stop()
        view._commit_view()


def _x_range(view: TraceView) -> tuple[float, float]:
    lo, hi = view.anchor_plot().vb.viewRange()[0]
    return float(lo), float(hi)


# ================================================================================================
# B1a — the menu follows the zoom history
# ================================================================================================
def test_a_fresh_load_has_nowhere_to_go(rig):
    assert rig.view.zoom_history_state() == (False, False)


def test_an_unsettled_dive_already_offers_back_and_the_trail_then_moves_both_ways(rig):
    view = rig.view
    seen: list[tuple[bool, bool]] = []
    view.sigZoomHistoryChanged.connect(lambda b, f: seen.append((b, f)))
    lo, hi = _x_range(view)
    c = 0.5 * (lo + hi)
    view.zoom_x_to(c - 0.5, c + 0.5)                    # dive — still debouncing
    assert view._settle.isActive()
    assert not view.can_go_back()                       # the committed trail has not moved...
    assert view.zoom_history_state() == (True, False)   # ...but Back would, so the menu says so
    assert seen == [(True, False)]                      # announced once, not once per frame
    _settle_history(view)
    assert view.zoom_history_state() == (True, False)
    view.view_back()
    assert view.zoom_history_state() == (False, True)
    view.view_forward()
    assert view.zoom_history_state() == (True, False)
    view.view_back()
    view.zoom_x_to(c - 0.2, c + 0.2)                    # a new zoom prunes the forward branch
    assert view.zoom_history_state() == (True, False)
    assert seen[-1] == (True, False)


def test_a_zoom_that_returns_to_the_current_entry_offers_nothing_new(rig):
    view = rig.view
    lo, hi = _x_range(view)
    view.zoom_x_to(lo + 1.0, hi - 1.0)
    view.zoom_x_to(lo, hi)                              # back where the trail already is
    assert view.zoom_history_state() == (False, False)


def test_closing_the_recording_greys_both_out(rig):
    view = rig.view
    lo, hi = _x_range(view)
    view.zoom_x_to(lo + 1.0, hi - 1.0)
    _settle_history(view)
    assert view.zoom_history_state() == (True, False)
    rig.session.close_document(rig.doc_id)
    QtWidgets.QApplication.processEvents()
    assert view.zoom_history_state() == (False, False)


@pytest.fixture
def window(qtbot, private_settings):
    win = build_window(qtbot, size=(1280, 800))
    yield win
    close_window(win)


def test_the_view_menu_items_are_greyed_out_and_the_quick_key_still_works(window):
    view = window._trace_view
    back, forward = window._act_zoom_back, window._act_zoom_forward
    assert (back.isEnabled(), forward.isEnabled()) == (False, False)
    start = _x_range(view)
    c = 0.5 * (start[0] + start[1])
    view.zoom_x_to(c - 0.05, c + 0.05)
    assert view._settle.isActive()
    assert (back.isEnabled(), forward.isEnabled()) == (True, False)
    window.activateWindow()
    view.setFocus()
    QTest.keyClick(view, Qt.Key_Left, Qt.ControlModifier)   # inside the 250 ms debounce
    settle()
    assert _x_range(view) == pytest.approx(start, rel=1e-6)
    assert (back.isEnabled(), forward.isEnabled()) == (False, True)


# ================================================================================================
# B1b — the crosshair names the sample index
# ================================================================================================
def _crosshair_texts(view: TraceView) -> list[str]:
    return [label.textItem.toPlainText() for _pane, _line, label in view.crosshair._items]


def _doc(n: int = 20_000, *, loaded: SampleRange | None = None, name: str = "b1.dat"):
    rng = np.random.default_rng(5)
    data = (100.0 + rng.normal(0.0, 1.0, n)).astype(np.float32)
    meta = RecordingMeta(
        path=Path("/nonexistent") / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=10_000.0, rate_provenance=Provenance.FILE_META, loaded_range=loaded,
        channels=(ChannelInfo(index=0, name="I", raw_units="pA", kind=ChannelKind.CURRENT),))
    return TraceDocument(meta=meta, channels=(data,))


def _view_of(qapp, doc) -> tuple[Session, TraceView]:
    session = Session()
    session.add_document(doc)
    session.add_layer(doc.doc_id, 0)
    view = TraceView(session)
    view.resize(780, 500)
    view.show()
    qapp.processEvents()
    return session, view


def test_the_crosshair_names_the_event_table_index_of_the_snapped_sample(rig):
    xs = rig.xs()
    idx = 12_345
    rig.session.set_cursor(float(xs[idx]))
    texts = _crosshair_texts(rig.view)
    assert texts and all("sample 12,345 (event-table index)" in t for t in texts), texts
    assert rig.session.time_to_index(rig.doc_id, 1, float(xs[idx])) == idx
    assert not any("file sample" in t or "parent sample" in t for t in texts)


def _assert_readouts_inside(view: TraceView) -> None:
    """Every SHOWN readout lies wholly inside its pane, where the ViewBox does not clip it."""
    shown = 0
    for pane, _line, label in view.crosshair._items:
        if not label.isVisible():
            continue
        shown += 1
        vb = pane.plot.vb
        box = label.mapRectToItem(vb, label.boundingRect())
        assert -0.5 <= box.top() and box.bottom() <= vb.height() + 0.5, (box, vb.height())
        assert -0.5 <= box.left() and box.right() <= vb.width() + 0.5, (box, vb.width())
    assert shown, "no readout is shown"


@pytest.mark.parametrize("fraction", [0.2, 0.8])
def test_the_readout_is_drawn_inside_its_pane(rig, fraction):
    """It was anchored by its bottom edge at the top of the view — wholly above the ViewBox,
    which clips its children, so the readout this track extends was never drawn at all."""
    lo, hi = rig.x_range()
    rig.session.set_cursor(lo + fraction * (hi - lo))
    QtWidgets.QApplication.processEvents()
    _assert_readouts_inside(rig.view)


def test_the_readout_follows_a_y_zoom_made_under_a_still_cursor(rig):
    """Placed in data coordinates at the view's top y, it used to be placed only when the
    cursor moved: a y zoom left it at the OLD top edge, wholly above the pane (measured:
    top -567 px on a 240 px pane)."""
    lo, hi = rig.x_range()
    rig.session.set_cursor(lo + 0.3 * (hi - lo))
    QtWidgets.QApplication.processEvents()
    vb = rig.target_vb
    y0, y1 = vb.viewRange()[1]
    vb.setYRange(y0, y0 + 0.3 * (y1 - y0), padding=0)          # the lower 30 %
    QtWidgets.QApplication.processEvents()
    _assert_readouts_inside(rig.view)


def test_the_readout_follows_a_pan_into_the_right_half(rig):
    """...and a pan under a still cursor could carry it off the pane's right edge."""
    zoom(rig, 2.0, 4.0)
    rig.session.set_cursor(2.4)                         # left half: hangs right of the line
    QtWidgets.QApplication.processEvents()
    _assert_readouts_inside(rig.view)
    rig.view.set_x_range(1.0, 3.0)                      # the still cursor is now at 70 %
    QtWidgets.QApplication.processEvents()
    _assert_readouts_inside(rig.view)
    for _pane, _line, label in rig.view.crosshair._items:
        assert tuple(label.anchor) == (1, 0)            # ...and now hangs left of it


def test_an_empty_readout_draws_nothing(rig):
    """Past the end of the data there is no sample under the line and so no text — and the
    translucent backing alone would still draw a small dark box at the top of the pane."""
    lo, hi = rig.view.full_x_extent()
    rig.session.set_cursor(hi + 5.0)
    QtWidgets.QApplication.processEvents()
    assert all(not label.isVisible() for _p, _l, label in rig.view.crosshair._items)
    rig.session.set_cursor(0.5 * (lo + hi))
    QtWidgets.QApplication.processEvents()
    assert all(label.isVisible() for _p, _l, label in rig.view.crosshair._items)


def test_a_refresh_before_the_first_hover_draws_no_readout(rig):
    """Session.cursor_x starts at 0.0, which nobody chose. A calibration change refreshes
    the readout; before any hover that must not put a full readout at x = 0."""
    rig.view.crosshair.refresh()
    QtWidgets.QApplication.processEvents()
    assert all(not label.isVisible() for _p, _l, label in rig.view.crosshair._items)
    assert _crosshair_texts(rig.view) == ["", ""]


def test_on_a_recording_without_a_rate_the_index_is_given_once(qapp):
    """The x axis there is arange(n): the x IS the event-table index, so a separate index
    line under "1234 sample" would print one number twice."""
    n = 20_000
    rng = np.random.default_rng(5)
    meta = RecordingMeta(
        path=Path("/nonexistent") / "b1-norate.dat", format_name="synthetic", n_samples=n,
        sample_rate_hz=None, rate_provenance=Provenance.UNKNOWN,
        loaded_range=SampleRange(1_000_000, 1_000_000 + n),
        channels=(ChannelInfo(index=0, name="I", raw_units="pA", kind=ChannelKind.CURRENT),))
    doc = TraceDocument(meta=meta, channels=((100.0 + rng.normal(0.0, 1.0, n)).astype(np.float32),))
    session, view = _view_of(qapp, doc)
    try:
        session.set_cursor(12_345.0)
        (text,) = _crosshair_texts(view)
        lines = text.split("\n")
        assert lines[0] == "sample 12,345 (event-table index)", lines
        assert lines[-1] == "file sample 1,012,345", lines
        assert len(lines) == 3 and "12345 sample" not in text, lines
    finally:
        view.hide()   # not close(): see trace_rig's rig fixture


def test_a_drag_that_starts_on_the_readout_still_draws_a_zoom_box(rig):
    """The readout now sits INSIDE the pane, at its top edge, where a box zoom often starts. It
    is transparent to the mouse, as the target caption is, so it cannot eat that drag."""
    view, vb = rig.view, rig.target_vb
    lo, hi = rig.x_range()
    x0, x1 = lo + 0.3 * (hi - lo), lo + 0.5 * (hi - lo)
    p0 = point_at(view, vb, x0, 0.02)
    drag(view, p0, point_at(view, vb, x1, 0.9))
    QtWidgets.QApplication.processEvents()
    pane = view.pane_for_layer(f"{rig.doc_id}#1")
    label = next(lb for pn, _ln, lb in view.crosshair._items if pn is pane)
    assert label.acceptedMouseButtons() == Qt.MouseButton.NoButton
    new_lo, new_hi = rig.x_range()
    assert new_lo == pytest.approx(x0, abs=0.02 * (hi - lo))
    assert new_hi == pytest.approx(x1, abs=0.02 * (hi - lo))


def test_a_partial_load_also_names_the_files_own_sample(qapp):
    """The event table counts from the window's first sample; the file counts from its own."""
    doc = _doc(loaded=SampleRange(30_000, 50_000))
    session, view = _view_of(qapp, doc)
    try:
        xs = session.x_axis(doc.doc_id).values
        session.set_cursor(float(xs[250]))
        (text,) = _crosshair_texts(view)
        assert "sample 250 (event-table index)" in text, text
        assert "file sample 30,250" in text, text
    finally:
        view.hide()   # not close(): see trace_rig's rig fixture


def test_a_segment_also_names_its_parents_sample(qapp):
    doc = _doc()
    session, view = _view_of(qapp, doc)
    try:
        seg_id = session.create_segment(doc.doc_id, 4_000, 9_000, name="cut")
        for layer in list(session.layers()):
            session.remove_layer(layer.layer_id)
        session.add_layer(seg_id, 0)
        qapp.processEvents()
        xs = session.x_axis(seg_id).values
        session.set_cursor(float(xs[100]))
        (text,) = _crosshair_texts(view)
        assert "sample 100 (event-table index)" in text, text
        assert "parent sample 4,100" in text, text
        # The time line agrees with the parent sample: 100 / 10 kHz = 10.0 ms into the cut,
        # and the cut's origin (4,000 / 10 kHz = 400.0 ms) added for the instrument clock.
        # 2026-11-11: the readout is in ms with one decimal since 8661d35 (it read
        # "0.0100 s in" / "0.4100 s instrument"). The whole line is compared, not two
        # substrings: "10.0 ms in" is itself a substring of "410.0 ms instrument", so the
        # first half of a substring pair could not fail.
        assert text.split("\n")[0] == "10.0 ms in  ·  410.0 ms instrument", text
    finally:
        view.hide()   # not close(): see trace_rig's rig fixture


# ================================================================================================
# B1c — Go to event…, Go to time…
# ================================================================================================
def _span_x(window, event_id: int) -> tuple[float, float]:
    df = window._analysis_panel.current_detection().payload.df
    row = df.loc[df["event_id"] == event_id].iloc[0]
    xs = window._trace_view.x_axis_for(*window._analysis_panel.current_target()).values
    return float(xs[int(row["start_idx"])]), float(xs[int(row["end_idx"])])


def test_go_to_event_selects_it_and_frames_the_trace_on_it(window):
    assert window._go_to_event_id(3)
    settle()
    assert window._analysis_panel.selected_event_id() == 3
    x0, x1 = _span_x(window, 3)
    lo, hi = _x_range(window._trace_view)
    assert lo < x0 < x1 < hi and hi - lo < 4 * (x1 - x0) + 1e-2


def test_go_to_the_already_selected_event_frames_it_again(window):
    """A table click on the current row changes no selection, so nothing re-zooms; Go to
    event must not inherit that and leave the view where the user panned it."""
    assert window._go_to_event_id(2)
    settle()
    framed = _x_range(window._trace_view)
    window._trace_view.set_x_range(0.0, 0.05)           # pan well away from event 2
    settle()
    assert window._go_to_event_id(2)
    settle()
    assert _x_range(window._trace_view) == pytest.approx(framed, rel=1e-6)


def test_an_unknown_id_moves_nothing_and_says_so(window):
    before = _x_range(window._trace_view)
    assert not window._go_to_event_id(99)
    assert _x_range(window._trace_view) == pytest.approx(before)
    message = window.statusBar().currentMessage()
    assert "No event 99" in message and "0–5" in message, message


def test_the_menu_action_asks_for_an_id_prefilled_with_the_selection(window, monkeypatch):
    asked = []

    def answer(value, lo, hi):
        asked.append((value, lo, hi))
        return 4

    monkeypatch.setattr(window, "_ask_event_id", answer)
    window._analysis_panel.select_event(1)
    settle()
    window._act_go_to_event.trigger()
    settle()
    assert asked == [(1, 0, len(SPANS) - 1)]
    assert window._analysis_panel.selected_event_id() == 4


def test_go_to_time_centres_the_view_and_places_the_crosshair_on_a_sample(window):
    view = window._trace_view
    view.zoom_x_to(0.1, 0.2)                            # a 0.1 s window
    settle()
    target = 0.9
    assert window._go_to_time(target + 3e-6)            # off-sample: snapped to the nearest
    settle()
    lo, hi = _x_range(view)
    assert hi - lo == pytest.approx(0.1, rel=1e-6)      # the zoom is kept
    assert 0.5 * (lo + hi) == pytest.approx(target, abs=1e-4)
    assert window._session.cursor_x == pytest.approx(target, abs=1e-9)


def test_go_to_time_outside_the_trace_is_refused(window):
    view = window._trace_view
    view.zoom_x_to(0.1, 0.2)
    settle()
    before = _x_range(view)
    assert not window._go_to_time(50.0)
    assert _x_range(view) == pytest.approx(before)
    assert "outside the trace" in window.statusBar().currentMessage()


def test_the_time_prompt_quotes_the_axis_clock(window, monkeypatch):
    asked = []
    monkeypatch.setattr(window, "_ask_trace_time",
                        lambda label, value, lo, hi, decimals: asked.append(
                            (label, lo, hi, decimals)) or None)
    window._act_go_to_time.trigger()
    (label, lo, hi, decimals), = asked
    assert window._trace_view.x_axis_label() in label
    assert "instrument clock" in label and "t_start_s" in label
    assert (lo, hi) == pytest.approx(window._trace_view.full_x_extent())
    assert decimals == 6


def test_the_time_prompt_says_the_table_reads_alike_on_a_zero_t0(window, monkeypatch):
    """t₀ = 0 and no alignment shift: the axis and t_start_s are one number, and the prompt
    must not tell the user to convert."""
    asked = []
    monkeypatch.setattr(window, "_ask_trace_time",
                        lambda label, *_a: asked.append(label) or None)
    window._act_go_to_time.trigger()
    (label,) = asked
    assert "reads the same number here" in label and "add " not in label, label


def test_the_time_prompt_gives_an_alignment_shift_to_add(window, monkeypatch):
    """A layer drawn shifted for alignment moves the axis away from t_start_s even on a
    t₀ = 0 file; "add t₀" alone would be wrong by the shift."""
    layer = window._trace_axis_layer()
    window._session.set_layer_time_offset(layer.layer_id, 0.25)
    settle()
    asked = []
    monkeypatch.setattr(window, "_ask_trace_time",
                        lambda label, *_a: asked.append(label) or None)
    window._act_go_to_time.trigger()
    (label,) = asked
    assert "add 0.25 s" in label, label


def test_the_time_prompt_on_a_segment_says_its_own_clock_reads_alike(window, monkeypatch):
    """A segment's axis is the cut's own clock, rebased to t₀ = 0, as its label says — the
    same numbers as its event table, so there is nothing to add (whatever the instrument
    clock behind the cut reads)."""
    parent = window._trace_axis_layer().doc_id
    seg_id = window._session.create_segment(parent, 10_000, 40_000, name="cut")
    for layer in list(window._session.layers()):
        window._session.remove_layer(layer.layer_id)
    window._session.add_layer(seg_id, 0)
    settle()
    assert window._trace_axis_layer().doc_id == seg_id
    asked = []
    monkeypatch.setattr(window, "_ask_trace_time",
                        lambda label, *_a: asked.append(label) or None)
    window._act_go_to_time.trigger()
    (label,) = asked
    assert "rebased" in label and "reads the same number here" in label, label
    assert "add " not in label, label


def test_go_to_time_at_the_full_zoom_says_only_the_crosshair_moved(window):
    """Nothing pans at the full-trace zoom: the status line must not claim it centred."""
    view = window._trace_view
    view.reset_zoom()
    settle()
    before = _x_range(view)
    assert window._go_to_time(0.4)
    settle()
    assert _x_range(view) == pytest.approx(before)
    message = window.statusBar().currentMessage()
    assert "already in view" in message and "Centred" not in message, message
    assert window._session.cursor_x == pytest.approx(0.4)
    _assert_readouts_inside(view)


def test_go_to_time_near_the_start_says_it_is_in_view_but_not_centred(window):
    view = window._trace_view
    view.zoom_x_to(0.5, 0.6)
    settle()
    assert window._go_to_time(0.01)
    settle()
    lo, hi = _x_range(view)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi - lo == pytest.approx(0.1, rel=1e-6)
    message = window.statusBar().currentMessage()
    assert "not centred" in message and "Centred" not in message, message


def test_after_go_to_time_the_readout_is_inside_its_pane(window):
    """Go to time pans and then places the cursor; the y autorange that follows the pan
    used to leave the readout clipped by a few pixels at the top."""
    view = window._trace_view
    view.zoom_x_to(0.1, 0.2)
    settle(300)
    assert window._go_to_time(0.9)
    settle(300)
    _assert_readouts_inside(view)


def _window_with(qtbot, **meta):
    """A shown MainWindow on one synthetic 100,000-sample current trace, no detection."""
    from nanoscope.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1280, 800)
    win.show()
    settle()
    n = 100_000
    data = (np.random.default_rng(1).normal(0.0, 1.0, n) - 1000.0).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    doc = TraceDocument(meta=RecordingMeta(
        path=Path("/nonexistent") / "b1-clock.dat", format_name="synthetic", n_samples=n,
        channels=(info,), **meta), channels=(data,))
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    settle()
    return win


def test_go_to_time_takes_the_axis_clock_on_a_recording_whose_t0_is_not_zero(
        qtbot, private_settings, monkeypatch):
    """The brief's point: the prompt takes the AXIS's clock. On t₀ = 100 s the axis reads
    100.9 where the event table reads t_start_s 0.9, and 0.9 typed here is not that sample."""
    win = _window_with(qtbot, sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
                       t0_s=100.0)
    try:
        asked = []
        monkeypatch.setattr(win, "_ask_trace_time",
                            lambda label, value, lo, hi, decimals: asked.append(
                                (label, lo, hi, decimals)) or None)
        win._act_go_to_time.trigger()
        (label, lo, hi, decimals), = asked
        assert "add 100 s" in label and (lo, hi) == pytest.approx((100.0, 101.99998)), asked
        win._trace_view.zoom_x_to(100.1, 100.2)
        settle()
        assert win._go_to_time(100.9)
        settle()
        assert win._session.cursor_x == pytest.approx(100.9)
        assert any("sample 45,000 (event-table index)" in t
                   for t in _crosshair_texts(win._trace_view))
        assert not win._go_to_time(0.9)                     # a t_start_s, typed raw
        assert "outside the trace" in win.statusBar().currentMessage()
    finally:
        close_window(win)


def test_go_to_time_on_a_recording_without_a_rate_asks_for_a_sample(
        qtbot, private_settings, monkeypatch):
    win = _window_with(qtbot, sample_rate_hz=None, rate_provenance=Provenance.UNKNOWN)
    try:
        asked = []
        monkeypatch.setattr(win, "_ask_trace_time",
                            lambda label, value, lo, hi, decimals: asked.append(
                                (label, decimals)) or None)
        win._act_go_to_time.trigger()
        (label, decimals), = asked
        assert "sample number" in label and decimals == 0, asked
        win._trace_view.zoom_x_to(1_000.0, 2_000.0)
        settle()
        assert win._go_to_time(45_000.4)                    # snapped to the nearest sample
        settle()
        assert win._session.cursor_x == 45_000.0
        (text,) = _crosshair_texts(win._trace_view)
        assert text.split("\n")[0] == "sample 45,000 (event-table index)", text
    finally:
        close_window(win)


def _shortcut_owners(window, key: str) -> list[str]:
    seq = QtGui.QKeySequence(key)
    owners = [a.text() for a in window.findChildren(QtGui.QAction)
              if any(s == seq for s in a.shortcuts())]
    owners += [repr(s) for s in window.findChildren(QtGui.QShortcut)
               if s.key() == seq and s.isEnabled()]
    return owners


@pytest.mark.parametrize("key", ["Ctrl+J", "Ctrl+Shift+J"])
def test_the_new_keys_are_bound_once_in_the_main_window(window, key):
    owners = _shortcut_owners(window, key)
    assert len(owners) == 1, owners


# ================================================================================================
# B1c — the Event inspector goes to an event by ID; its Event box stays a position
# ================================================================================================
@pytest.fixture
def inspector(window):
    window._analysis_panel.set_event_accepted(1, False)
    window._analysis_panel.set_event_accepted(3, False)
    window._open_event_inspector()
    settle()
    insp = window._event_inspector
    assert insp is not None
    return insp


def test_the_event_box_says_it_is_a_position_not_an_id(inspector):
    assert "not the event ID" in inspector._index_spin.toolTip()
    assert "event_id" in inspector._goto_id_btn.toolTip()
    assert inspector._goto_id_btn.isEnabled()


def test_an_id_inside_the_filter_keeps_the_filter(inspector):
    inspector.set_filter("Rejected only")
    settle()
    assert inspector.go_to_event_id(3)
    assert inspector.current_event_id() == 3
    assert inspector.current_filter() == "Rejected only"
    assert inspector._index_spin.value() == 2           # position 2 of the two rejected


def test_an_id_outside_the_filter_moves_to_all_and_says_why(inspector):
    inspector.set_filter("Rejected only")
    settle()
    assert inspector.go_to_event_id(4)
    assert inspector.current_event_id() == 4
    assert inspector.current_filter() == "All"
    assert inspector._index_spin.value() == 5           # ID 4 is the fifth event of All
    assert "Filter → All" in inspector.progress_text()


def test_an_unknown_id_moves_nothing_and_the_progress_line_says_so(inspector):
    shown = inspector.current_event_id()
    assert not inspector.go_to_event_id(42)
    assert inspector.current_event_id() == shown
    assert "No event ID 42" in inspector.progress_text()


def test_going_to_the_event_on_screen_clears_an_earlier_miss(inspector):
    """show_event returns early for the event already shown, so nothing cleared the note:
    "No event ID 42" stayed beside the success it no longer describes."""
    shown = inspector.current_event_id()
    assert not inspector.go_to_event_id(42)
    assert "No event ID 42" in inspector.progress_text()
    assert inspector.go_to_event_id(shown)
    assert inspector.current_event_id() == shown
    assert "No event ID" not in inspector.progress_text(), inspector.progress_text()


def test_the_id_button_asks_through_its_seam(inspector, monkeypatch):
    monkeypatch.setattr(inspector, "_ask_event_id", lambda value, lo, hi: 5)
    inspector._goto_id_btn.click()
    settle()
    assert inspector.current_event_id() == 5


def test_main_window_go_to_event_moves_an_open_inspector_by_id(window, inspector):
    inspector.set_filter("Rejected only")
    settle()
    assert window._go_to_event_id(2)
    settle()
    assert inspector.current_event_id() == 2
    assert inspector.current_filter() == "All"


def test_docked_the_inspector_gives_ctrl_j_to_the_main_window(window):
    insp, _dock = docked(window)
    assert insp._goto_id_shortcut.key().isEmpty()
    assert len(_shortcut_owners(window, "Ctrl+J")) == 1
    window.set_curation_host("event_inspector", "window", present=True)
    settle()
    assert insp._goto_id_shortcut.key() == QtGui.QKeySequence("Ctrl+J")


# -- the keys, pressed: a registration count cannot see a focused widget that takes the press --
def _count_prompts(monkeypatch, window, inspector=None) -> dict[str, int]:
    """Replace every Go to prompt with a counter that cancels, so a press opens no modal."""
    calls = {"main_id": 0, "main_time": 0, "inspector_id": 0}

    def counter(name):
        def ask(*_args):
            calls[name] += 1
            return None
        return ask

    monkeypatch.setattr(window, "_ask_event_id", counter("main_id"))
    monkeypatch.setattr(window, "_ask_trace_time", counter("main_time"))
    if inspector is not None:
        monkeypatch.setattr(inspector, "_ask_event_id", counter("inspector_id"))
    return calls


def _press(widget, key, modifiers) -> None:
    widget.window().activateWindow()
    widget.setFocus()
    settle()
    QTest.keyClick(widget, key, modifiers)
    settle()


def test_ctrl_j_and_ctrl_shift_j_reach_the_menu_from_the_trace(window, monkeypatch):
    calls = _count_prompts(monkeypatch, window)
    _press(window._trace_view, Qt.Key_J, Qt.ControlModifier)
    _press(window._trace_view, Qt.Key_J, Qt.ControlModifier | Qt.ShiftModifier)
    assert (calls["main_id"], calls["main_time"]) == (1, 1), calls


def test_ctrl_j_and_ctrl_shift_j_reach_the_menu_from_the_event_table(window, monkeypatch):
    """The table binds bare j / k; the Ctrl chords must still reach View ▸ Go to."""
    calls = _count_prompts(monkeypatch, window)
    window._analysis_panel.select_event(2)
    settle()
    table = window._analysis_panel._table.view
    _press(table, Qt.Key_J, Qt.ControlModifier)
    _press(table, Qt.Key_J, Qt.ControlModifier | Qt.ShiftModifier)
    assert (calls["main_id"], calls["main_time"]) == (1, 1), calls
    assert window._analysis_panel.selected_event_id() == 2      # j did not step the table


def test_ctrl_j_in_the_floating_inspector_is_its_own_id_door(window, monkeypatch):
    window._open_event_inspector()
    settle()
    insp = window._event_inspector
    calls = _count_prompts(monkeypatch, window, insp)
    _press(insp, Qt.Key_J, Qt.ControlModifier)
    assert (calls["inspector_id"], calls["main_id"]) == (1, 0), calls


def test_ctrl_j_in_the_docked_inspector_is_view_go_to_event(window, monkeypatch):
    """Docked, the inspector's own shortcut is off, so the press is the main window's — from
    its Event box (a spin box, which takes keys) as from the inspector itself."""
    insp, _dock = docked(window)
    calls = _count_prompts(monkeypatch, window, insp)
    _press(insp._index_spin, Qt.Key_J, Qt.ControlModifier)
    _press(insp, Qt.Key_J, Qt.ControlModifier)
    assert (calls["main_id"], calls["inspector_id"]) == (2, 0), calls
