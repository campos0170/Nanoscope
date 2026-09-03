"""T1 — "I SWITCHED FROM OVERLAY TO STACKED AND THEN OVERLAY DID NOT COME BACK."

The round trip is functionally correct, and no test said so. What is actually wrong is
that with ONE recording shown the two modes produce an IDENTICAL pane set — Overlay's
rule is "group comparable channels ACROSS recordings" (``plan_panes``), and with one
recording there is nothing to group across. One recording is also the DEFAULT state:
``_show_document`` calls ``Session.show_only``. So the user's Overlay click left the
trace visibly unchanged, with nothing on screen connecting that to the multi-recording
view she had never built — and the honest reading of "it did not come back" is "it never
looked like anything in the first place".

Two things are pinned here:

* the round trip itself, on a REAL two-recording view: OVERLAY gives one pane holding
  both curves with distinct pens and real data, STACKED gives two panes, and OVERLAY
  again gives back the first arrangement — not a pane that merely exists, but one that is
  actually drawing both traces;
* the missing sentence: choosing Overlay while only one recording is shown posts what
  Overlay is for and the gesture that supplies the missing ingredient. The control is NOT
  disabled — the mode really is set, and it decides the layout the moment a second
  recording joins the view.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 5_000


def _doc(tmp_path, name: str, level: float) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    data = np.full(N, level, dtype=np.float64) + rng.normal(0.0, 2.0, N)
    return TraceDocument(meta=meta, channels=(np.ascontiguousarray(data, dtype=np.float32),))


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _open(win, tmp_path, name: str, level: float) -> str:
    doc_id = win._session.add_document(_doc(tmp_path, name, level))
    win._create_default_layers(win._session.document(doc_id))
    return doc_id


def _add_to_view(win, doc_id: str) -> None:
    """What the Files dock's "Add to view (compare)" does, at the session level: make this
    recording's layers visible WITHOUT hiding the others (which is what ``show_only`` —
    the default one-recording path — would do)."""
    for layer in win._session.layers():
        if layer.doc_id == doc_id:
            win._session.set_layer_visible(layer.layer_id, True)


def _curves(pane) -> list:
    return list(pane._curves.values())


# ================================================================================================
# the round trip, on a view that actually has two recordings in it
# ================================================================================================

def test_overlay_stacked_overlay_returns_the_overlaid_pane(win, tmp_path, qtbot):
    doc_a = _open(win, tmp_path, "a.dat", -1000.0)
    doc_b = _open(win, tmp_path, "b.dat", -900.0)
    win._show_document(doc_a)          # show_only(a) — the app's default
    _add_to_view(win, doc_b)           # ...and now the comparison view
    qtbot.wait(10)

    win._session.set_view_mode(ViewMode.OVERLAY)
    qtbot.wait(10)
    panes = win._trace_view.panes()
    assert len(panes) == 1, "two pA channels of two recordings belong on one pane"
    assert len(_curves(panes[0])) == 2

    win._session.set_view_mode(ViewMode.STACKED)
    qtbot.wait(10)
    assert len(win._trace_view.panes()) == 2

    win._session.set_view_mode(ViewMode.OVERLAY)
    qtbot.wait(10)
    panes = win._trace_view.panes()
    assert len(panes) == 1, "Overlay did not come back"
    curves = _curves(panes[0])
    assert len(curves) == 2
    # ...and both are DRAWN: a pane holding two empty items would satisfy every count above.
    for curve in curves:
        x, y = curve.getData()
        assert x is not None and y is not None and len(y) > 0
    pens = {curve.opts["pen"].color().name() for curve in curves}
    assert len(pens) == 2, "the two overlaid traces must be told apart by colour"


def test_the_round_trip_is_driven_from_the_menu_actions(win, tmp_path, qtbot):
    """Through the actual controls, not the session API — the complaint is about the menu."""
    doc_a = _open(win, tmp_path, "a.dat", -1000.0)
    doc_b = _open(win, tmp_path, "b.dat", -900.0)
    win._show_document(doc_a)
    _add_to_view(win, doc_b)
    qtbot.wait(10)

    win._act_stacked.trigger()
    qtbot.wait(10)
    assert win._session.view_mode is ViewMode.STACKED
    assert len(win._trace_view.panes()) == 2

    win._act_overlay.trigger()
    qtbot.wait(10)
    assert win._session.view_mode is ViewMode.OVERLAY
    assert len(win._trace_view.panes()) == 1
    assert win._act_overlay.isChecked() and not win._act_stacked.isChecked()


# ================================================================================================
# ...and the sentence for the case where the round trip cannot show
# ================================================================================================

def test_overlay_with_one_recording_shown_explains_itself(win, tmp_path, qtbot):
    """The default state, and the one the complaint was actually made from."""
    doc_a = _open(win, tmp_path, "a.dat", -1000.0)
    _open(win, tmp_path, "b.dat", -900.0)      # open, but not IN THE VIEW
    win._show_document(doc_a)
    qtbot.wait(10)

    win._act_overlay.trigger()
    msg = win.statusBar().currentMessage()
    assert "only one recording is shown" in msg.lower()
    assert "Add to view" in msg
    # The control is not disabled and the mode really is set — it decides the layout the
    # moment a second recording joins the view.
    assert win._session.view_mode is ViewMode.OVERLAY
    assert win._act_overlay.isEnabled()


def test_overlay_says_nothing_when_it_actually_did_something(win, tmp_path, qtbot):
    """The note is for the case that cannot show, not a lecture on every Overlay click."""
    doc_a = _open(win, tmp_path, "a.dat", -1000.0)
    doc_b = _open(win, tmp_path, "b.dat", -900.0)
    win._show_document(doc_a)
    _add_to_view(win, doc_b)
    win._act_stacked.trigger()
    qtbot.wait(10)
    win.statusBar().clearMessage()

    win._act_overlay.trigger()
    assert win.statusBar().currentMessage() == ""
    assert len(win._trace_view.panes()) == 1


def test_stacked_never_posts_the_note(win, tmp_path, qtbot):
    """Stacked is honest with one recording — one layer, one pane, exactly as advertised."""
    doc_a = _open(win, tmp_path, "a.dat", -1000.0)
    win._show_document(doc_a)
    win.statusBar().clearMessage()

    win._act_stacked.trigger()
    assert win.statusBar().currentMessage() == ""
