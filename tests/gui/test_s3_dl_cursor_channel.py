"""JM-03 — the status-bar cursor readout describes the channel the user is on.

``MainWindow._on_cursor`` took ``next(lay for lay in session.layers() if lay.visible)`` —
the first visible layer in layer order — with no preference for kind and no reference to
the selection. ABF channel order puts the command voltage BEFORE the current (the navigator
already prefers the current-kind layer for exactly this reason), so on the owner's
recordings the ``y`` beside the cursor was always the command voltage. Measured on his quick
ABF with IN15 selected and the current trace on screen: the cursor at sample 200 000 sat on
**-168.457 pA** and the status bar read ``t = 1158.99 s     y = -999.878 mV`` — naming no
channel — and identically in stacked mode, where the channels are in separate panes and the
cursor is demonstrably in the current pane.

The window already holds this rule for analysis ("THE APP NEVER ANALYSES A RECORDING IT IS
NOT DRAWING"). Nothing in ``docs/USER_GUIDE.md`` blesses reading another channel, and no
test covered the readout at all (grep for ``_cursor_label`` / ``_on_cursor`` over ``tests/``:
zero hits) — which is why this file exists.

The second half is the honest fallback: when the selection is NOT drawn there is still only
one candidate, and then the readout has to name it rather than let the number pass for the
selection's.
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

FS = 50_000.0
N = 20_000
#: The owner's ABF layout: the command voltage FIRST, the current after it.
VOLTS = -999.878
AMPS = -168.457


@pytest.fixture
def win(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    volt = ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    curr = ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "abf_order.dat", format_name="synthetic",
                         n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(volt, curr))
    doc = TraceDocument(meta=meta, channels=(
        np.full(N, VOLTS, dtype=np.float32), np.full(N, AMPS, dtype=np.float32)))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    yield w, doc_id
    w._analysis_controller.abort()


def _readout(w, doc_id, index: int = 10_000) -> str:
    w._session.set_cursor(w._session.x_axis(doc_id).values[index])
    return w._cursor_label.text()


def test_the_readout_describes_the_selected_channel_not_the_first_layer(win, qtbot):
    w, doc_id = win
    w._on_selection(doc_id, 1)                  # the CURRENT channel, as a user would
    text = _readout(w, doc_id)
    assert "pA" in text and f"{AMPS:.6g}" in text, text
    assert "mV" not in text, (
        f"the status bar reported the command voltage while the current channel was "
        f"selected and drawn: {text!r}")


def test_it_holds_in_stacked_mode_too(win, qtbot):
    """The verifier's second measurement: in stacked mode the channels are in separate panes
    and the cursor is visibly in the current's, which made the wrong number look authoritative."""
    from nanoscope.core.layer import ViewMode

    w, doc_id = win
    w._on_selection(doc_id, 1)
    w._session.set_view_mode(ViewMode.STACKED)
    qtbot.wait(20)
    text = _readout(w, doc_id)
    assert "pA" in text and "mV" not in text, text


def test_selecting_the_voltage_channel_reports_the_voltage(win, qtbot):
    """CONTROL. The rule is "the channel the user is on", not "always the current" — a fix
    that hard-coded a preference for CURRENT-kind layers would be wrong the moment somebody
    is deliberately reading the command waveform."""
    w, doc_id = win
    w._on_selection(doc_id, 0)
    text = _readout(w, doc_id)
    assert "mV" in text and f"{VOLTS:.6g}" in text, text


def test_when_the_selection_is_not_drawn_the_readout_names_whose_number_it_is(win, qtbot):
    """The honest fallback. Hiding the selected channel leaves one candidate, and an
    unlabelled number then reads as the selection's — the original defect one step over."""
    w, doc_id = win
    w._on_selection(doc_id, 1)
    for lay in w._session.layers():
        if lay.channel == 1:
            w._session.set_layer_visible(lay.layer_id, False)
    qtbot.wait(20)
    text = _readout(w, doc_id)
    assert "IN14" in text, (
        f"the readout fell back to another channel without saying so: {text!r}")


def test_nothing_drawn_leaves_the_readout_blank(win, qtbot):
    """Unchanged behaviour, pinned: a readout with nothing behind it says nothing."""
    w, doc_id = win
    w._on_selection(doc_id, 1)
    for lay in w._session.layers():
        w._session.set_layer_visible(lay.layer_id, False)
    qtbot.wait(20)
    w._session.set_cursor(0.05)
    assert w._cursor_label.text() == ""
