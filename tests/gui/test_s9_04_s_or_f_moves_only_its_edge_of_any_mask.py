"""S9-04 — S or F on a mask placed some other way moves only that edge.

MEASURED DEFECT (sweep 9, finding 4; a clean 50 kHz blockade at samples 20 000 - 20 400).
``TraceKeyController.place_edge`` took the other edge only from a mask the controller had built
itself ("a mask that was seeded or dragged in some other way is not an instruction"). For a mask
placed any other way -- a double-click on the trace, Snap to view, Snap to selected event, Go to
measurement, a drag with the Measure tool -- S or F threw the mask's other edge away and began
an S-then-F build. Zoomed to 19 000 - 22 000 and double-clicked, the mask was [19 900, 20 500); S
at the event's start collapsed it to the one-sample [20 000, 20 001) with "Start edge set. Point
at the finish and press F.", F collapsed it to [20 399, 20 400), and S then D filed the one-sample
event [20 000, 20 001) as ``start=fit:step;end=drawn``, for an end nobody drew. After an earlier C
the same S on the same mask kept the finish edge, so what S did depended on history the person
cannot see. The guide's key table says S / F "Put the start / finish edge at the pointer", and
design 3.3 says S "moves the start edge to the pointer's x and fits it there".

THE FIX. A mask the controller did not place is what the person sees: when it settles
(``TraceKeyController._on_mask_settled``) both its edges are recorded as drawn, so S or F moves
only its own edge and the other stays exactly where it was placed (a drawn edge follows every
settle, even a one-sample nudge), whatever was built before. Recording costs no measurement: a
settle repaints the label, which measures the span again, only when an edge changes how it was
made. Measure's own seed, the middle fifth of the view that turning Measure on lays down (it
settles before Measure reports itself on), is still no instruction: S then F builds a mask over
it as before, and a rebuild that puts the seed back does not make it one. Snap to view and a
double-click that turn Measure on themselves turn it on without the seed
(``MeasurePanel._place_mask_in_view``): Snap to view lays its mask on the seed's own middle fifth,
the trace reports nothing for a mask placed where one already is, and the placed mask was taken
for the seed.

These tests drive a real MainWindow on a synthetic recording. Two of them are all real input on a
shown window: a double-click on the trace, the pointer moved onto the event (focus follows it onto
the trace), and the S, F and D keys. The others stand the pointer in with a patched
``TraceView.pointer_sample`` (a real pointer is a screen coordinate), as ``test_trace_keys`` does,
and place the mask through the Measure tab's Snap to selected event button, the trace's
double-click relay and the Measure toggle; a drag release is ``set_measure_region``, the
placement a drag ends in.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

import nanoscope.gui.measure_panel as mp_mod
from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

from ._inspector_hosting import settle
from .test_trace_rig import double_click, free_mouse, hover, point_at  # noqa: F401

FS = 50_000.0
N = 120_000
I0 = -1000.0
DEPTH = 300.0
EVENT = (20_000, 20_400)            # a clean blockade no detector was asked about
VIEW = (19_000, 22_000)             # the zoom the defect was measured at
CLICKED = (19_900, 20_500)          # what a double-click at 20 200 places there (20 % of the view)


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    x[EVENT[0]:EVENT[1]] += DEPTH
    return x.astype(np.float32)


def _open(win, tmp_path) -> str:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "s9_04.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    return doc_id


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    return win, _open(win, tmp_path)


@pytest.fixture
def shown(qapp, qtbot, tmp_path, free_mouse):
    """The same, on screen: the pointer helpers need the trace inside the 800 x 800 offscreen
    screen, and focus follows the pointer only into an active window."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(780, 700)
    win.move(0, 0)
    win.show()
    settle()
    doc_id = _open(win, tmp_path)
    win.activateWindow()
    qtbot.waitUntil(win.isActiveWindow, timeout=5_000)
    settle()
    return win, doc_id


def _zoom(win, doc_id, view=VIEW) -> None:
    tv = win._trace_view
    tv.zoom_x_to(tv.x_for_index(doc_id, 0, view[0]), tv.x_for_index(doc_id, 0, view[1]))
    settle()


def _point(win, doc_id, sample, px=10.0):
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _table(win, doc_id):
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return None if res is None else res.payload.df


def _target_vb(win, doc_id):
    return win._trace_view.pane_for_layer(f"{doc_id}#0").plot.vb


def _move_onto(win, doc_id, sample: int) -> None:
    """Move the real pointer onto the trace at ``sample``, from just left of the plot, so the
    pointer ARRIVES (focus follows it onto the trace)."""
    tv, vb = win._trace_view, _target_vb(win, doc_id)
    r = vb.sceneBoundingRect()
    hover(tv, QtCore.QPointF(tv.mapFromScene(QtCore.QPointF(r.left() - 12.0, r.center().y()))))
    hover(tv, point_at(tv, vb, tv.x_for_index(doc_id, 0, sample)))
    settle(10)
    assert tv.hasFocus(), type(QtWidgets.QApplication.focusWidget()).__name__
    ps = tv.pointer_sample()
    assert ps is not None and abs(ps[2] - sample) < 40, ps


def _double_click(win, doc_id) -> tuple[int, int]:
    """A real double-click on the event, outside draw mode: the Measure tool places a mask."""
    tv = win._trace_view
    double_click(tv, point_at(tv, _target_vb(win, doc_id), tv.x_for_index(doc_id, 0, 20_200)))
    settle()
    bounds = tv.measure_bounds()
    assert bounds is not None, win.statusBar().currentMessage()
    assert bounds[0] < EVENT[0] and bounds[1] > EVENT[1]          # it brackets the whole event
    return bounds


# ================================================================================================
# real input: double-click, point, S / F, D
# ================================================================================================
def test_a_real_double_click_then_s_fits_the_start_keeps_the_finish_and_d_files_that(shown):
    win, doc_id = shown
    tv = win._trace_view
    _zoom(win, doc_id)
    lo, hi = _double_click(win, doc_id)

    _move_onto(win, doc_id, EVENT[0] + 12)
    QTest.keyClick(tv, QtCore.Qt.Key_S)
    settle()

    new_lo, new_hi = tv.measure_bounds()
    said = win.statusBar().currentMessage()
    assert abs(new_lo - EVENT[0]) <= 3, (new_lo, new_hi, said)       # the start went to its step
    assert new_hi == hi, f"S turned the mask [{lo}, {hi}) into [{new_lo}, {new_hi}); {said!r}"
    assert "Point at the finish" not in said and "D adds the event" in said

    QTest.keyClick(tv, QtCore.Qt.Key_D)
    settle()
    df = _table(win, doc_id)
    assert df is not None and len(df) == 1, win.statusBar().currentMessage()
    row = df.iloc[0]
    assert (int(row["start_idx"]), int(row["end_idx"])) == (new_lo, hi)
    assert row["bounds_source"] == "start=fit:step;end=drawn"
    assert int(row["drawn_end_idx"]) == hi                          # the finish as it was placed
    assert abs(int(row["drawn_start_idx"]) - (EVENT[0] + 12)) < 40   # where the start was pointed
    assert json.loads(row["fit_json"])["edges"]["start"]["fitted"] is True


def test_a_real_double_click_then_f_fits_the_finish_and_keeps_the_start(shown):
    win, doc_id = shown
    tv = win._trace_view
    _zoom(win, doc_id)
    lo, hi = _double_click(win, doc_id)

    _move_onto(win, doc_id, EVENT[1] - 9)
    QTest.keyClick(tv, QtCore.Qt.Key_F)
    settle()

    new_lo, new_hi = tv.measure_bounds()
    assert abs(new_hi - EVENT[1]) <= 3, (new_lo, new_hi, win.statusBar().currentMessage())
    assert new_lo == lo, f"F turned the mask [{lo}, {hi}) into [{new_lo}, {new_hi})"


# ================================================================================================
# every other way a mask is placed, and the history that no longer matters
# ================================================================================================
def test_s_on_a_mask_snapped_to_the_selected_event_keeps_its_finish(rig):
    win, doc_id = rig
    mp = win._analysis_panel.measure_panel
    _zoom(win, doc_id)
    mp.set_selected_event(doc_id, 0, 19_900, 20_600)
    mp._snap_event_btn.click()                                  # Measure ▸ Snap to selected event
    assert win._trace_view.measure_bounds() == (19_900, 20_600)

    _point(win, doc_id, EVENT[0] + 12)
    win._trace_keys.place_edge("start", fit=True)                # the S key's slot

    lo, hi = win._trace_view.measure_bounds()
    assert abs(lo - EVENT[0]) <= 3 and hi == 20_600, (lo, hi)


def test_s_on_a_mask_moved_by_a_drag_keeps_the_edge_the_drag_left(rig):
    """Measure on with a mask moved the way a handle drag leaves it (``set_measure_region`` with
    nothing of the controller's armed is what a drag release ends in)."""
    win, doc_id = rig
    tv = win._trace_view
    _zoom(win, doc_id)
    win._analysis_panel.measure_panel.set_active(True)            # the seed...
    tv.set_measure_region(tv.x_for_index(doc_id, 0, 19_950),        # ...dragged onto the event
                          tv.x_for_index(doc_id, 0, 20_450))
    _point(win, doc_id, EVENT[1] - 9)
    win._trace_keys.place_edge("end", fit=True)                  # the F key's slot
    lo, hi = tv.measure_bounds()
    assert lo == 19_950 and abs(hi - EVENT[1]) <= 3, (lo, hi)


@pytest.mark.parametrize("caught_first", [False, True])
def test_the_same_s_on_the_same_mask_does_the_same_whatever_was_built_before(rig, caught_first):
    win, doc_id = rig
    tv = win._trace_view
    _zoom(win, doc_id)
    if caught_first:
        _point(win, doc_id, EVENT[0] + 200)
        win._trace_keys.catch()                                  # an earlier C, now replaced
    tv._request_measure_placement(tv.x_for_index(doc_id, 0, 20_200))   # the double-click relay
    assert tv.measure_bounds() == CLICKED

    _point(win, doc_id, EVENT[0] + 12)
    win._trace_keys.place_edge("start", fit=True)

    lo, hi = tv.measure_bounds()
    assert abs(lo - EVENT[0]) <= 3 and hi == CLICKED[1], (caught_first, lo, hi)


@pytest.mark.parametrize("how", ["Snap to view", "double-click at the view centre"])
def test_s_on_a_mask_placed_while_measure_is_off_keeps_its_finish(rig, how):
    """Both turn Measure on themselves and both lay their mask exactly where Measure's seed goes
    (the middle fifth of the view). Turned on WITH its seed, Measure left the trace nothing to
    report, so S built a new mask instead of moving this one's start."""
    win, doc_id = rig
    tv = win._trace_view
    mp = win._analysis_panel.measure_panel
    _zoom(win, doc_id, (18_700, 21_700))                       # its middle fifth brackets EVENT
    assert not mp.is_active()
    if how == "Snap to view":
        mp._snap_view_btn.click()
    else:
        tv._request_measure_placement(tv.x_for_index(doc_id, 0, 20_200))   # the double-click relay
    placed = tv.measure_bounds()
    assert placed is not None and placed[0] < EVENT[0] and placed[1] > EVENT[1], placed

    _point(win, doc_id, EVENT[0] + 12)
    win._trace_keys.place_edge("start", fit=True)

    lo, hi = tv.measure_bounds()
    said = win.statusBar().currentMessage()
    assert abs(lo - EVENT[0]) <= 3 and hi == placed[1], (how, placed, lo, hi, said)
    assert "Point at the finish" not in said


def test_s_keeps_the_finish_exactly_where_a_one_sample_nudge_left_it(rig):
    win, doc_id = rig
    tv = win._trace_view
    _zoom(win, doc_id)
    tv._request_measure_placement(tv.x_for_index(doc_id, 0, 20_200))   # the double-click relay
    lo, hi = tv.measure_bounds()
    tv.set_measure_region(tv.x_for_index(doc_id, 0, lo),                # the finish handle
                          tv.x_for_index(doc_id, 0, hi + 1))            # nudged one sample out

    _point(win, doc_id, EVENT[0] + 12)
    win._trace_keys.place_edge("start", fit=True)

    new_lo, new_hi = tv.measure_bounds()
    assert abs(new_lo - EVENT[0]) <= 3 and new_hi == hi + 1, (lo, hi, new_lo, new_hi)


def test_a_later_drag_of_a_recorded_mask_is_measured_once(rig, monkeypatch, qtbot):
    """Recording the mask must not cost a measurement: Measure measures a settled span once,
    and the label (whose repaint measures it again) has nothing new to say about drawn edges."""
    win, doc_id = rig
    tv = win._trace_view
    label_timer = win._trace_keys._label_timer
    _zoom(win, doc_id)
    tv._request_measure_placement(tv.x_for_index(doc_id, 0, 20_200))   # the double-click relay
    qtbot.waitUntil(lambda: not label_timer.isActive(), timeout=2_000)
    calls = []
    real = mp_mod.measure_selection

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mp_mod, "measure_selection", counted)
    for lo, hi in ((19_950, 20_450), (19_960, 20_460)):                 # two drag releases
        tv.set_measure_region(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, hi))
        qtbot.waitUntil(lambda: not label_timer.isActive(), timeout=2_000)
    assert len(calls) == 2, f"{len(calls)} measurements for two settles"


# ================================================================================================
# Measure's own seed is still no instruction
# ================================================================================================
@pytest.mark.parametrize("rebuilt", [False, True])
def test_measures_own_seed_still_lets_s_then_f_build_the_mask(rig, rebuilt):
    win, doc_id = rig
    tv = win._trace_view
    _zoom(win, doc_id, (19_000, 25_000))
    win._analysis_panel.measure_panel.set_active(True)            # the middle fifth of the view
    seed = tv.measure_bounds()
    assert seed is not None and seed[0] > EVENT[1]                 # nowhere near the event
    if rebuilt:
        win._session.set_view_mode(ViewMode.STACKED)              # a rebuild puts the seed back
        settle()
        assert tv.measure_bounds() == seed

    _point(win, doc_id, EVENT[0] + 12)
    win._trace_keys.place_edge("start", fit=True)
    assert "Point at the finish and press F" in win.statusBar().currentMessage()
    _point(win, doc_id, EVENT[1] - 9)
    win._trace_keys.place_edge("end", fit=True)

    lo, hi = tv.measure_bounds()
    assert abs(lo - EVENT[0]) <= 3 and abs(hi - EVENT[1]) <= 3, (lo, hi)
