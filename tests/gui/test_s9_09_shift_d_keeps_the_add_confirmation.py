"""S9-09 — Shift+D with no further window keeps the add's confirmation.

MEASURED DEFECT (sweep 9, finding 9; a 50 kHz recording of 120 000 samples). Shift+D adds the
mask and then steps to the next window. ``TraceKeyController.add`` called ``step_window`` after the
add, and when no step was possible -- the view on the recording's last window, or the whole
recording in view, the state a recording opens in -- the step's own sentence, "That is the end of
the recording.", replaced the add's "Added event 0 to Manual. ⌘Z undoes it." for three seconds.
The row was added; the status line no longer said where it went or how to undo it. Mid-recording
the same Shift+D kept the confirmation. Design 3.4: after an add "the status line says 'Added
event 318 to Schmitt. ⌘Z undoes'".

THE FIX. When the step after an add cannot move, the status line keeps the add's sentence and adds
that there is no next window ("... ⌘Z undoes it. There is no next window: that is the end of the
recording."). G and Shift+G on their own say what they said before.

These tests drive a real MainWindow on a synthetic recording: the C and Shift+D keys pressed on
the shown, focused trace, and their slots with the pointer stood in (as ``test_trace_keys``
does), on the last window, fully zoomed out, and mid-recording.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6.QtTest import QTest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 120_000
I0 = -1000.0
EARLY = (20_000, 20_400)
LAST = (115_000, 115_400)           # inside the recording's last window
NO_NEXT = "There is no next window: that is the end of the recording."


def _signal(seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0) + rng.normal(0.0, 8.0, N)
    for s, e in (EARLY, LAST):
        x[s:e] += 300.0
    return x.astype(np.float32)


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "s9_09.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    return win, doc_id


def _point(win, doc_id, sample, px=10.0):
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _rows(win, doc_id) -> int:
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return 0 if res is None else len(res.payload.df)


def _view(win):
    return tuple(win._trace_view.anchor_plot().vb.viewRange()[0])


def _zoom(win, doc_id, lo: int, hi: int) -> None:
    tv = win._trace_view
    tv.zoom_x_to(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, hi))


def test_shift_d_on_the_last_window_keeps_the_add_and_says_there_is_no_next(rig, qtbot):
    win, doc_id = rig
    win.resize(1_200, 800)
    win.show()
    win.activateWindow()
    qtbot.waitUntil(win.isActiveWindow, timeout=5_000)
    tv = win._trace_view
    _zoom(win, doc_id, 110_000, N - 1)                         # the recording's last window
    _point(win, doc_id, LAST[0] + 200)
    tv.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
    assert tv.hasFocus()
    QTest.keyClick(tv, QtCore.Qt.Key_C)
    qtbot.wait(50)
    assert tv.measure_bounds() is not None, win.statusBar().currentMessage()
    before = _view(win)

    QTest.keyClick(tv, QtCore.Qt.Key_D, QtCore.Qt.ShiftModifier)
    qtbot.wait(50)

    assert _rows(win, doc_id) == 1
    said = win.statusBar().currentMessage()
    assert said.startswith("Added event 0 to Manual.") and "undoes it." in said, said
    assert said.endswith(NO_NEXT), said
    assert _view(win) == pytest.approx(before)                  # there was nowhere to go


def test_shift_d_with_the_whole_recording_in_view_keeps_the_add(rig):
    """The state a recording opens in: fully zoomed out, no window to step to."""
    win, doc_id = rig
    _point(win, doc_id, EARLY[0] + 200)
    win._trace_keys.catch()
    assert win._trace_view.measure_bounds() is not None
    win._trace_keys.add_and_next()                             # Shift+D
    assert _rows(win, doc_id) == 1
    said = win.statusBar().currentMessage()
    assert said.startswith("Added event 0 to Manual.") and said.endswith(NO_NEXT), said


def test_shift_d_mid_recording_still_steps_and_keeps_the_add(rig):
    win, doc_id = rig
    _zoom(win, doc_id, 105_000, 117_000)
    _point(win, doc_id, LAST[0] + 200)
    win._trace_keys.catch()
    before = _view(win)
    win._trace_keys.add_and_next()
    assert _rows(win, doc_id) == 1 and _view(win)[0] > before[0]   # it went on
    said = win.statusBar().currentMessage()
    assert said.startswith("Added event 0 to Manual.") and NO_NEXT not in said, said


def test_g_on_its_own_still_says_the_end_of_the_recording(rig):
    win, doc_id = rig
    _zoom(win, doc_id, 110_000, N - 1)
    win._trace_keys.step_window(+1)                            # G
    assert win.statusBar().currentMessage() == "That is the end of the recording."
