"""S9-10 — A mask edge the trace keys put at the recording's end stays there, and the row with it.

MEASURED DEFECT (sweep 9, finding 10; 120 000 samples at 50 kHz, an event still in progress when
the recording stops at sample 119 600). ``TraceKeyController._show_mask`` placed both edges
through ``TraceView.x_for_index``, which clamps its index to ``[0, n-1]``, so an end at ``n`` --
where C bounds an event the recording's end cuts off, and where Shift+F or a draw release puts it
-- landed on the last sample's x and read back as ``n - 1``. C on that event, then D twice (its
end is placed, not fitted), stored ``[119 600, 119 999)`` with ``drawn_end_idx`` 120 000: one
sample short, disagreeing with its own record, and no longer touching the edge of the analysed
span, so it was accepted and unflagged. Shift+S then Shift+F at the last sample stored
``[119 500, 119 999)`` the same way, and so did a draw that ran off the end. The add door, given
the span C caught (``add_manual_event(doc, 0, 119 600, 120 000)``), rejects it automatically as
``clipped_by_boundary``, the rule design section 3 names for a span cut by the recording edge.
``MeasurePanel._x_span`` documents this exact case and puts the end one sample past the last x.

THE FIX. ``TraceKeyController._x`` places an edge at ``n`` one sample interval past the last
sample's x, as ``MeasurePanel._x_span`` does, which ``measure_bounds`` reads back as ``n``. The
row ends where C, Shift+F or the draw put it, ``drawn_end_idx`` equals ``end_idx``, and the door
rejects it as ``clipped_by_boundary`` exactly as it does the same span handed to it directly.

These tests drive a real MainWindow on a synthetic recording with the end of the trace in view:
the C, Shift+S, Shift+F and D keys' slots with the pointer stood in (as ``test_trace_keys`` does;
``pointer_sample`` clamps to ``[0, n]``), and a draw release that reached the end of the trace.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 120_000
I0 = -1000.0
DEPTH = 300.0
TAIL = (119_600, N)                 # an event still in progress when the recording stops


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    x[TAIL[0]:TAIL[1]] += DEPTH
    x[20_000:20_400] += DEPTH       # an ordinary event, so the polarity vote has company
    return x.astype(np.float32)


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "s9_10.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    win._trace_view.set_x_range(2.30, 2.40)      # the last 0.1 s on screen, the end included
    assert win._analysis_panel.visible_sample_range()[1] == N
    return win, doc_id


def _point(win, doc_id, sample, px=1.0):
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _row(win, doc_id):
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    assert res is not None and len(res.payload.df) == 1, win.statusBar().currentMessage()
    return res.payload.df.iloc[0]


def _clipped(row) -> bool:
    return "clipped_by_boundary" in str(row["flag"]) and not bool(row["accepted"])


def test_c_on_an_event_cut_off_by_the_end_stores_the_span_it_caught_and_rejects_it(rig):
    win, doc_id = rig
    keys = win._trace_keys
    _point(win, doc_id, 119_800)
    keys.catch()
    assert keys._end.pos == N, win.statusBar().currentMessage()     # C bounded it at the end
    assert win._trace_view.measure_bounds()[1] == N                 # ...and shows it there

    keys.add()
    keys.add()                                   # the end is placed, not fitted: D asks twice

    row = _row(win, doc_id)
    assert int(row["end_idx"]) == N, (int(row["start_idx"]), int(row["end_idx"]))
    assert int(row["drawn_end_idx"]) == int(row["end_idx"])
    assert row["bounds_source"] == "start=fit:step;end=placed"
    assert _clipped(row), (row["flag"], bool(row["accepted"]))
    assert "rejected automatically (clipped_by_boundary)" in win.statusBar().currentMessage()


def test_shift_s_then_shift_f_at_the_last_sample_sets_the_end_exactly_there(rig):
    win, doc_id = rig
    keys = win._trace_keys
    _point(win, doc_id, 119_500)
    keys.place_edge("start", fit=False)
    _point(win, doc_id, float(N))                # pointer_sample clamps to [0, n]
    keys.place_edge("end", fit=False)
    assert win._trace_view.measure_bounds() == (119_500, N)

    keys.add()

    row = _row(win, doc_id)
    assert (int(row["start_idx"]), int(row["end_idx"])) == (119_500, N)
    assert row["bounds_source"] == "start=drawn;end=drawn"
    assert int(row["drawn_end_idx"]) == N and _clipped(row)


def test_a_draw_that_ran_off_the_end_is_filed_to_the_end(rig):
    win, doc_id = rig
    tv = win._trace_view
    keys = win._trace_keys
    keys.set_draw_mode(True)
    # where a draw drag that ran off the end of the trace leaves the mask (the legal extent)
    tv.set_measure_region(tv.x_for_index(doc_id, 0, 119_595),
                          tv.measure_cursor_extent(doc_id, 0)[1])
    assert tv.measure_bounds() == (119_595, N)
    tv.sigDrawReleased.emit(doc_id, 0, 119_595, N, 1.0)           # the release: fit both edges
    assert tv.measure_bounds()[1] == N

    keys.add()
    keys.add()

    row = _row(win, doc_id)
    assert int(row["end_idx"]) == N and int(row["drawn_end_idx"]) == N
    assert _clipped(row)


def test_the_door_given_the_same_span_rejects_it_the_same_way(rig):
    """What the keys now file is what the door says about the span itself."""
    win, doc_id = rig
    out = win._file_hand_made_event(doc_id, 0, *TAIL)
    assert out.ok
    row = _row(win, doc_id)
    assert (int(row["start_idx"]), int(row["end_idx"])) == TAIL and _clipped(row)
