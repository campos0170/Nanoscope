"""S9-18 — The "press again" asked about one mask is not spent on the next one.

MEASURED DEFECT (on e2c92cb for D; found while integrating the sweep-9 fixes, because S9-05 gave
Ctrl+M and the Add event button the same second press). ``TraceKeyController.add`` asks for a
second press when an edge is ``placed`` (not fitted) and arms a ten-second clock. Only ``_reset``
(X, an add, Measure off) disarmed it, so a new mask made inside those ten seconds -- a catch, a
draw release -- was filed by its FIRST press, although its own status line had just said "... or D
twice to add it as placed." Design 3.2: "D on an unfitted edge asks for a confirming second press".

THE FIX. The armed press records the mask it was asked about (its target and bounds), and only a
press on that same mask within the ten seconds adds it; a different mask asks again.

A wired MainWindow on a synthetic recording, the pointer stood in at a sample, the keys called as
their shortcuts call them, and a draw's release emitted as TraceView emits it. Controls: with X in
between, or with no earlier question, the new mask's first D asks.
"""


from __future__ import annotations

import os

import nanoscope  # noqa: F401
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
TAIL = (119_600, N)                 # an event the recording's end cuts off: C leaves its end placed
EVENT = (20_000, 20_400)            # a clean blockade: a draw [19 990, 25 000) fits only its start
FIRST = ["D", "Ctrl+M", "Add event button"]


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    x[TAIL[0]:TAIL[1]] += DEPTH
    x[EVENT[0]:EVENT[1]] += DEPTH
    return x.astype(np.float32)


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "fk1.dat", format_name="synthetic", n_samples=N,
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


def _said(win) -> str:
    return win.statusBar().currentMessage()


def _press(win, how: str) -> None:
    if how == "D":
        win._trace_keys.add()
    elif how == "Ctrl+M":
        win._act_add_event.trigger()                           # Analysis > Add event from cursors
    else:
        win._analysis_panel.measure_panel._add_button.click()   # the Measure tab's Add event


def _ask_about_an_open_pore_mask(win, doc_id, lo, hi, how, px=10.0):
    """S at lo, F at hi on open pore (neither edge has a step: both placed), then ONE press."""
    keys = win._trace_keys
    _point(win, doc_id, hi, px)
    keys.place_edge("end", fit=True)
    _point(win, doc_id, lo, px)
    keys.place_edge("start", fit=True)
    assert win._trace_view.measure_bounds() == (lo, hi)
    assert (keys._start.source, keys._end.source) == ("placed", "placed")
    _press(win, how)
    assert _rows(win, doc_id) == 0, f"{how}: the first press on an unfitted mask filed it ({_said(win)!r})"
    assert "again" in _said(win), _said(win)


def _catch_the_tail(win, doc_id) -> str:
    keys = win._trace_keys
    _point(win, doc_id, 119_800, px=1.0)
    keys.catch()
    said = _said(win)
    assert keys._end.source == "placed" and keys._start.source == "fit:step", said
    assert win._trace_view.measure_bounds()[1] >= N - 1     # N on the fixed tree (S9-10), N - 1 before
    assert "not fitted" in said, said
    return said


def _draw_over_the_event(win, doc_id) -> str:
    tv = win._trace_view
    keys = win._trace_keys
    keys.set_draw_mode(True)
    lo, hi = 19_990, 25_000
    tv.preview_measure_region(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, hi))
    tv.sigDrawReleased.emit(doc_id, 0, lo, hi, 10.0)             # the release: fit both edges
    said = _said(win)
    assert (keys._start.source, keys._end.source) == ("fit:step", "placed"), said
    assert "not fitted" in said, said
    return said


# ================================================================================================
# the defect: the new mask's first D files it
# ================================================================================================
@pytest.mark.parametrize("first", FIRST)
def test_a_catch_that_says_d_twice_is_not_filed_by_one_d(rig, first):
    win, doc_id = rig
    win._trace_view.set_x_range(2.30, 2.40)      # the end of the recording in view (as S9-10)
    _ask_about_an_open_pore_mask(win, doc_id, 116_000, 116_300, first, px=1.0)

    caught = _catch_the_tail(win, doc_id)        # another event, another mask
    win._trace_keys.add()                        # the FIRST D on it

    assert _rows(win, doc_id) == 0, (
        f"one D filed the caught mask; the catch said {caught!r}; after D: {_said(win)!r}")


@pytest.mark.parametrize("first", FIRST)
def test_a_draw_that_says_d_twice_is_not_filed_by_one_d(rig, first):
    win, doc_id = rig
    _ask_about_an_open_pore_mask(win, doc_id, 90_000, 90_300, first)

    drawn = _draw_over_the_event(win, doc_id)    # another event, another mask
    win._trace_keys.add()                        # the FIRST D on it

    assert _rows(win, doc_id) == 0, (
        f"one D filed the drawn mask; the release said {drawn!r}; after D: {_said(win)!r}")


# ================================================================================================
# controls: with the question cleared, or never asked, the same mask's first D asks
# ================================================================================================
def test_control_with_x_in_between_the_catch_asks_again(rig):
    win, doc_id = rig
    win._trace_view.set_x_range(2.30, 2.40)
    _ask_about_an_open_pore_mask(win, doc_id, 116_000, 116_300, "D", px=1.0)
    win._trace_keys.discard()                    # X
    _catch_the_tail(win, doc_id)
    win._trace_keys.add()
    assert _rows(win, doc_id) == 0 and "again" in _said(win), _said(win)
    win._trace_keys.add()
    assert _rows(win, doc_id) == 1, _said(win)


def test_control_a_fresh_catch_asks_on_its_first_d(rig):
    win, doc_id = rig
    win._trace_view.set_x_range(2.30, 2.40)
    _catch_the_tail(win, doc_id)
    win._trace_keys.add()
    assert _rows(win, doc_id) == 0 and "again" in _said(win), _said(win)


def test_control_a_fresh_draw_asks_on_its_first_d(rig):
    win, doc_id = rig
    win._analysis_panel.measure_panel.set_active(True, seed=False)
    _draw_over_the_event(win, doc_id)
    win._trace_keys.add()
    assert _rows(win, doc_id) == 0 and "again" in _said(win), _said(win)
