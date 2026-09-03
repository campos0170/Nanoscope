"""Sweep-3 regression chaos3-001 — "show processed trace on the plot" must never be
ticked over a plot with no curve and nothing said.

THE RULE, one line: **a managed item torn down by a rebuild is put back by the rebuild,
and when it cannot be put back the surface that asked for it says why.**

``TraceView.rebuild`` runs on gestures the Analysis panel never hears — View ▸
Stacked/Overlay (``Session.viewModeChanged``) and every channel show/hide in the File
Manager (``Session.layersChanged``). It tears down all three managed items. Two were
already re-applied there (``_reattach_measure``, and the measure markers via their own
listener); the filter preview was not: the code claimed "the panel re-applies" while
``AnalysisPanel`` connected to neither signal. The orange curve vanished with the
checkbox still ticked, the "Active: Low-pass" label still up and the preview error line
EMPTY — a state indistinguishable from "the processed trace happens to lie on the raw
one", on the only surface that shows what the detector will actually run on. Only
un-tick + re-tick repaired it; switching the view mode back did not.

A second silence, same shape: ``set_filter_preview`` returned early on ``pane is None``
("channel not currently layered") without a word, so ticking the box for a channel that
is hidden drew nothing and said nothing.

Both are asserted here through real widgets and a real :class:`MainWindow`, and the
oracle for "is the curve on the plot" is the PLOT — the orange ``#e69f00`` items
actually parented to a pane — not the bookkeeping attribute a rebuild nulls.

THE SAME RULE, ONE ITEM ALONG (not a filed defect; found while generalising this one):
the event SPAN overlays were lost by the identical mechanism — measured, a 20-event
detection put one ``SpanOverlayItem`` on the pane and View ▸ Stacked left ZERO, while
``AnalysisPanel._overlay_ids`` still held a live-looking id for it, so the trace and
the event table disagreed about which stretches of the recording are events. Overlay
REQUESTS now outlive the pane too (``TraceView._overlay_reqs`` →
``_reattach_overlays``), and the last test here pins that.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.trace_view import PREVIEW_CHANNEL_NOT_SHOWN

FS = 50_000.0
N = 100_000
PREVIEW_COLOUR = "#e69f00"


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for i in range(12):
        s = 2_000 + i * 8_000
        x[s:s + 250] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _two_channel_doc(tmp_path, name: str = "preview_rebuild.dat") -> TraceDocument:
    """A CURRENT channel and a VOLTAGE channel, the shape of the owner's own ABFs — so
    a channel can be hidden in the File Manager without hiding the previewed one."""
    current = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    voltage = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(current, voltage))
    volts = np.full(N, -1000.0, dtype=np.float32)
    return TraceDocument(meta=meta, channels=(_trace(), volts))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _orange_curves(window: MainWindow) -> int:
    """How many preview-coloured curves are ACTUALLY on the panes right now.

    Walks the live pyqtgraph items rather than asking ``has_filter_preview()``, which
    reads the very attribute ``rebuild()`` nulls — the independent verification of this
    defect used the same oracle for the same reason."""
    n = 0
    for pane in window._trace_view.panes():
        for item in pane.plot.items:
            pen = getattr(item, "opts", {}).get("pen") if hasattr(item, "opts") else None
            colour = getattr(pen, "color", None)
            if callable(colour):
                colour = colour()
            if colour is not None and colour.name().lower() == PREVIEW_COLOUR:
                n += 1
    return n


def _state(window: MainWindow) -> dict:
    det = window._analysis_panel._detection
    return {
        "checkbox": det.filter_preview.isChecked(),
        "curve": _orange_curves(window),
        "error": det.preview_error_label.text(),
    }


def _turn_preview_on(window: MainWindow, qtbot) -> None:
    det = window._analysis_panel._detection
    det.detrend_enabled.setChecked(True)
    det.detrend_window_s.setValue(0.2)
    det.filter_preview.setChecked(True)
    qtbot.wait(30)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def previewing(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _two_channel_doc(tmp_path))
    _turn_preview_on(window, qtbot)
    state = _state(window)
    assert state["checkbox"] and state["curve"] == 1 and state["error"] == "", (
        f"setup: the preview never appeared ({state})")
    return doc_id, channel


# ================================================================================================
# the rebuild paths
# ================================================================================================

def test_the_preview_survives_a_view_mode_switch(window, qtbot, previewing):
    """View ▸ Stacked, then back to Overlay: the curve is there each time."""
    window._act_stacked.trigger()
    qtbot.wait(50)
    stacked = _state(window)
    assert stacked["curve"] == 1, (
        f"View ▸ Stacked dropped the preview while the box stayed ticked: {stacked}")
    assert stacked["error"] == ""

    window._act_overlay.trigger()
    qtbot.wait(50)
    overlay = _state(window)
    assert overlay["curve"] == 1, f"View ▸ Overlay dropped the preview: {overlay}"
    assert overlay["error"] == ""


def test_the_preview_survives_hiding_another_channel(window, qtbot, previewing):
    """The File Manager's visibility checkbox on the VOLTAGE channel rebuilds the view;
    the previewed CURRENT channel is untouched, so its curve must still be drawn."""
    doc_id, channel = previewing
    other = next(lay for lay in window._session.layers()
                 if lay.doc_id == doc_id and lay.channel != channel)
    window._session.set_layer_visible(other.layer_id, False)
    qtbot.wait(50)

    state = _state(window)
    assert state["curve"] == 1, (
        f"hiding an unrelated channel dropped the preview curve: {state}")
    assert state["error"] == ""


def test_the_preview_comes_back_when_a_second_recording_is_opened(
        window, qtbot, tmp_path, previewing):
    """Opening another recording re-layers the view. The selection does not move, so
    nothing withdraws the request — and the curve must still be on the plot."""
    doc_b = window._session.add_document(_two_channel_doc(tmp_path, name="second.dat"))
    window._create_default_layers(window._session.document(doc_b))
    qtbot.wait(50)

    state = _state(window)
    assert state["curve"] == 1, f"opening a second recording dropped the preview: {state}"
    assert state["error"] == ""


# ================================================================================================
# the channel that is not on the plot — refuse OUT LOUD
# ================================================================================================

def test_hiding_the_previewed_channel_says_why_instead_of_going_quiet(
        window, qtbot, previewing):
    doc_id, channel = previewing
    layer = next(lay for lay in window._session.layers()
                 if lay.doc_id == doc_id and lay.channel == channel)

    window._session.set_layer_visible(layer.layer_id, False)
    qtbot.wait(50)
    hidden = _state(window)
    assert hidden["curve"] == 0, "setup: the previewed channel is still on the plot"
    assert hidden["checkbox"], (
        "the request is still the user's — the box must not silently untick")
    # The label's own fixed prefix ("processed-trace preview unavailable: ") plus the
    # view's reason — the same line every other preview refusal is written on.
    assert hidden["error"].endswith(PREVIEW_CHANNEL_NOT_SHOWN), (
        f"a ticked box drew nothing and said nothing: {hidden}")
    assert hidden["error"].startswith("processed-trace preview unavailable:")

    # ...and showing it again brings the curve back AND clears the reason.
    window._session.set_layer_visible(layer.layer_id, True)
    qtbot.wait(50)
    shown = _state(window)
    assert shown["curve"] == 1, f"the preview did not come back with its channel: {shown}"
    assert shown["error"] == "", (
        f"the reason outlived the condition it described: {shown}")


def test_ticking_the_box_for_a_hidden_channel_refuses_out_loud(window, qtbot, tmp_path):
    """The second silent path: with the previewed channel hidden, the FIRST tick drew
    nothing and said nothing (``set_filter_preview`` returned early on ``pane is None``)."""
    doc_id, channel = _install(window, _two_channel_doc(tmp_path))
    layer = next(lay for lay in window._session.layers()
                 if lay.doc_id == doc_id and lay.channel == channel)
    window._session.set_layer_visible(layer.layer_id, False)
    qtbot.wait(30)

    _turn_preview_on(window, qtbot)

    state = _state(window)
    assert state["checkbox"] and state["curve"] == 0
    assert state["error"].endswith(PREVIEW_CHANNEL_NOT_SHOWN), (
        f"ticking the box for a hidden channel drew nothing in silence: {state}")


# ================================================================================================
# the request is withdrawn when the user withdraws it — a rebuild must not resurrect it
# ================================================================================================

def test_unticking_withdraws_the_request_across_a_rebuild(window, qtbot, previewing):
    det = window._analysis_panel._detection
    det.filter_preview.setChecked(False)
    qtbot.wait(30)
    assert _orange_curves(window) == 0

    window._act_stacked.trigger()
    qtbot.wait(50)
    state = _state(window)
    assert state["curve"] == 0, (
        f"a rebuild redrew a preview nobody is asking for: {state}")
    assert state["error"] == ""


def test_closing_the_recording_unticks_the_box_it_belonged_to(window, qtbot, previewing):
    """Nothing is open, the plot reads "Open a recording" — the request cannot still be
    the user's, and a box left ticked over that is a control claiming a state it no
    longer has (the same rule as the selection-change untick)."""
    doc_id, _channel = previewing
    window._session.close_document(doc_id)
    qtbot.wait(50)

    det = window._analysis_panel._detection
    assert window._analysis_panel._sel is None
    assert not det.filter_preview.isChecked(), (
        "the preview box is still ticked with no recording open")
    assert _orange_curves(window) == 0
    assert det.preview_error_label.text() == ""
    assert not window._trace_view.filter_preview_requested()


def test_selecting_another_recording_does_not_carry_the_preview_over(
        window, qtbot, tmp_path, previewing):
    """The documented auto-untick still stands: a selection change withdraws the
    request, so the incoming recording gets no curve and no error line."""
    _install(window, _two_channel_doc(tmp_path, name="other.dat"))
    qtbot.wait(50)

    det = window._analysis_panel._detection
    assert not det.filter_preview.isChecked()
    assert _orange_curves(window) == 0
    assert det.preview_error_label.text() == ""
    assert not window._trace_view.filter_preview_requested()


# ================================================================================================
# the same rule for the OTHER managed item a rebuild tore down: the event spans
# ================================================================================================

def _span_items(window: MainWindow) -> int:
    """Batched event-span overlays actually on the panes (one item per overlay, by design)."""
    from nanoscope.gui.overlays import SpanOverlayItem
    return sum(1 for pane in window._trace_view.panes()
               for item in pane.plot.items if isinstance(item, SpanOverlayItem))


def test_the_event_spans_survive_a_rebuild(window, qtbot, tmp_path):
    """A detection's spans are a standing annotation on a channel. A view-mode switch
    must not leave the table saying "20 events" over a trace with none marked, and it
    must not strand the ids their owner is still holding."""
    doc_id, channel = _install(window, _two_channel_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(50)

    rows = panel._table._model.rowCount()
    assert rows > 0, "setup: nothing was detected"
    assert _span_items(window) == 1, "setup: the detection drew no spans"
    tracked = len(panel._overlay_ids)

    window._act_stacked.trigger()
    qtbot.wait(50)
    assert _span_items(window) == 1, (
        "View ▸ Stacked left the event spans off the trace while the table still "
        f"shows {rows} events")
    assert len(panel._overlay_ids) == tracked

    other = next(lay for lay in window._session.layers()
                 if lay.doc_id == doc_id and lay.channel != channel)
    window._session.set_layer_visible(other.layer_id, False)
    qtbot.wait(50)
    assert _span_items(window) == 1, "a channel-visibility toggle dropped the spans"


def test_removing_an_overlay_is_not_undone_by_a_rebuild(window, qtbot, tmp_path):
    """The withdrawal half: an overlay its owner removed must stay removed."""
    doc_id, channel = _install(window, _two_channel_doc(tmp_path))
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(50)
    assert _span_items(window) == 1

    for oid in list(panel._overlay_ids):
        window._trace_view.remove_overlay(oid)
    panel._overlay_ids = []
    qtbot.wait(20)
    assert _span_items(window) == 0

    window._act_stacked.trigger()
    qtbot.wait(50)
    assert _span_items(window) == 0, "a rebuild resurrected an overlay that was removed"
