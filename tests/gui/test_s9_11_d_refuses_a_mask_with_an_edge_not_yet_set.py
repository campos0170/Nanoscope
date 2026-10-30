"""S9-11 — D on the half-built mask S or F alone leaves files nothing and says which edge to set.

MEASURED DEFECT (sweep 9, finding 11; a clean 50 kHz blockade at samples 20 000 - 20 400). With
no mask, S fits the start and shows a one-sample stand-in mask ``[start, start + 1)`` while it
waits for F ("Start edge set. Point at the finish and press F."). D right then -- the next key
along -- filed that stand-in: ``[20 000, 20 001)``, a 20 us event, accepted and counted in every
statistic, recorded as ``end=drawn`` with ``drawn_end_idx`` 20 001, an end nobody put anywhere.
``TraceKeyController.add`` held back only an edge that was ``placed``; an edge never set at all
passed, and ``_edge_record`` called it drawn. Design 3.3: "S then F builds one"; design 3.2:
unfitted edges are "recorded as placed, never drawn".

THE FIX. ``add`` refuses a mask with one edge still unset, keeps the half-built mask and files
nothing, with a sentence naming the key that sets it: "Nothing was added: the finish edge is not
set yet. Point at it and press F, or Shift+F to set it exactly." (S / Shift+S for a start). Ctrl+M,
Return, Enter and the Add event button add through the same ``add`` and refuse the same way. The
stand-in's free edge counts as set once a drag pulls the stand-in wider; a rebuild that puts the
stand-in back sets nothing. The Measure tab's preview says the same before any press
(``MeasurePanel.set_add_hold``): "Would not be added to Manual: the finish edge is not set yet.
...", where it would otherwise show the one-sample stand-in as the row on offer.

These tests drive a real MainWindow on a synthetic recording: the S, F and D keys' slots with the
pointer stood in (as ``test_trace_keys`` does), the Analysis menu action Ctrl+M triggers, a
view-mode change (a rebuild), and the mask moved the way a handle drag leaves it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 120_000
I0 = -1000.0
DEPTH = 300.0
EVENT = (20_000, 20_400)            # a clean blockade no detector was asked about
NOT_SET = {
    "finish": "Nothing was added: the finish edge is not set yet. Point at it and press F, or "
              "Shift+F to set it exactly.",
    "start": "Nothing was added: the start edge is not set yet. Point at it and press S, or "
             "Shift+S to set it exactly.",
}


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    x[EVENT[0]:EVENT[1]] += DEPTH
    return x.astype(np.float32)


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "s9_11.dat", format_name="synthetic", n_samples=N,
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


def _half_built(win, doc_id, which: str) -> tuple[int, int]:
    """S (or F) alone, with no mask yet: one edge fitted, a one-sample stand-in for the other."""
    keys = win._trace_keys
    if which == "start":
        _point(win, doc_id, EVENT[0] + 12)
        keys.place_edge("start", fit=True)
        assert "Point at the finish and press F" in win.statusBar().currentMessage()
    else:
        _point(win, doc_id, EVENT[1] - 9)
        keys.place_edge("end", fit=True)
        assert "Point at the start and press S" in win.statusBar().currentMessage()
    bounds = win._trace_view.measure_bounds()
    assert bounds is not None and bounds[1] - bounds[0] == 1
    return bounds


@pytest.mark.parametrize("which, missing", [("start", "finish"), ("end", "start")])
def test_d_before_the_other_edge_files_nothing_keeps_the_mask_and_names_the_key(rig, which,
                                                                               missing):
    win, doc_id = rig
    shown = _half_built(win, doc_id, which)

    win._trace_keys.add()                                      # D before F (or before S)

    assert _rows(win, doc_id) == 0
    assert win.statusBar().currentMessage() == NOT_SET[missing]
    assert win._trace_view.measure_bounds() == shown           # the half-built mask is kept


def test_ctrl_m_refuses_the_half_built_mask_the_same_way(rig):
    win, doc_id = rig
    shown = _half_built(win, doc_id, "start")
    win._act_add_event.trigger()                               # Analysis ▸ Add event from cursors
    assert _rows(win, doc_id) == 0
    assert win.statusBar().currentMessage() == NOT_SET["finish"]
    assert win._trace_view.measure_bounds() == shown


def test_after_the_refusal_f_then_d_files_the_event(rig):
    win, doc_id = rig
    _half_built(win, doc_id, "start")
    win._trace_keys.add()
    assert _rows(win, doc_id) == 0
    _point(win, doc_id, EVENT[1] - 9)
    win._trace_keys.place_edge("end", fit=True)                # F
    win._trace_keys.add()                                      # D
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    row = res.payload.df.iloc[0]
    assert abs(int(row["start_idx"]) - EVENT[0]) <= 3 and abs(int(row["end_idx"]) - EVENT[1]) <= 3
    assert row["bounds_source"] == "start=fit:step;end=fit:step"


def test_the_preview_says_so_before_the_click_and_the_add_button_gets_the_same_answer(rig,
                                                                                    qtbot):
    """The preview shows the row an add would store, or why there is none, before the click: the
    half-built mask is no one-sample row on offer. The Add button still asks, as it does where
    the door would refuse, and is answered with the same sentence."""
    win, doc_id = rig
    mp = win._analysis_panel.measure_panel
    label = win._trace_keys._label_timer
    _half_built(win, doc_id, "start")
    qtbot.waitUntil(lambda: not label.isActive(), timeout=2_000)
    text = mp.event_preview_text()
    assert text.startswith("Would not be added to Manual: the finish edge is not set yet. "
                           "Point at it and press F"), text
    assert mp._add_button.isEnabled()
    mp._add_button.click()
    assert _rows(win, doc_id) == 0
    assert win.statusBar().currentMessage() == NOT_SET["finish"]

    _point(win, doc_id, EVENT[1] - 9)
    win._trace_keys.place_edge("end", fit=True)                # F: now there is a row to show
    qtbot.waitUntil(lambda: not label.isActive(), timeout=2_000)
    assert mp.event_preview_text().startswith("→ Manual: dwell"), mp.event_preview_text()


def test_a_rebuild_does_not_set_the_edge_still_to_come(rig):
    win, doc_id = rig
    shown = _half_built(win, doc_id, "start")
    win._session.set_view_mode(ViewMode.STACKED)              # the trace is rebuilt...
    QtWidgets.QApplication.processEvents()
    assert win._trace_view.measure_bounds() == shown           # ...and the stand-in put back
    win._trace_keys.add()
    assert _rows(win, doc_id) == 0
    assert win.statusBar().currentMessage() == NOT_SET["finish"]


def test_dragging_the_stand_in_wider_sets_the_edge_and_d_then_adds(rig):
    win, doc_id = rig
    tv = win._trace_view
    lo, _hi = _half_built(win, doc_id, "start")
    # the finish handle pulled out to 20 450 (a drag release ends in exactly this placement)
    tv.set_measure_region(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, 20_450))
    win._trace_keys.add()
    row = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID).payload.df.iloc[0]
    assert (int(row["start_idx"]), int(row["end_idx"])) == (lo, 20_450)
    assert row["bounds_source"] == "start=fit:step;end=drawn"
    assert int(row["drawn_end_idx"]) == 20_450
