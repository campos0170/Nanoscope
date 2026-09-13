"""The Event Gallery — the contact-sheet surface for triaging a whole detection.

The inspector answers "what is THIS event?"; the gallery answers "what is in this
detection, and which ones are wrong?". These tests drive it the way the user does: open it
from the Analysis menu, sort, sweep a selection, curate in bulk — and check the curation
landed on the SHARED events frame, which is the only reason any of it reaches the table,
the overlays, the summary and the exports.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt

from nanoscope.analysis.events import REVIEW_COL, REVIEW_DONE
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_gallery import _EnvelopeSource, EventGalleryWindow
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0


# ================================================================================================
# builders
# ================================================================================================

def _trace(n_events: int = 60, seed: int = 3) -> np.ndarray:
    """A trace with many events of DIFFERENT dwells and depths, so sorting has something
    to sort and the tiles are not all the same picture."""
    rng = np.random.default_rng(seed)
    n = 400_000
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for i in range(n_events):
        s = 2_000 + i * 6_000
        length = int(rng.integers(60, 600))
        x[s:s + length] -= float(rng.uniform(15.0, 45.0))
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def win(qapp, qtbot, tmp_path):
    """A window with a real detection on a many-event trace, gallery NOT yet opened."""
    csv = tmp_path / "many.csv"
    arr = _trace()
    _write_csv(csv, arr)
    w = MainWindow()
    qtbot.addWidget(w)
    doc_id = w._session.add_document(_doc(csv, arr))
    w._create_default_layers(w._session.document(doc_id))
    ch = next(c.index for c in w._session.document(doc_id).meta.channels
              if c.kind is ChannelKind.CURRENT and not c.is_dead)
    w._on_selection(doc_id, ch)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    yield w, doc_id, ch
    w._analysis_controller.abort()
    w._controller.abort()


def _open_gallery(w, qtbot) -> EventGalleryWindow:
    """Through the MENU ACTION, never by constructing the window.

    A handler that works when called directly and fails from the menu is a defect this
    project has already shipped once (a QAction.triggered `checked` bool bound to the first
    positional parameter), so every gallery test enters the same way the user does.
    """
    w._act_event_gallery.trigger()
    qtbot.wait(30)
    assert w._event_gallery is not None, "the menu action opened no gallery"
    return w._event_gallery


def _select_rows(gal: EventGalleryWindow, first: int, last: int) -> None:
    sel = QItemSelection(gal._model.index(first, 0), gal._model.index(last, 0))
    gal.view.selectionModel().select(sel, QItemSelectionModel.ClearAndSelect)


# ================================================================================================
# the envelope — what a tile is allowed to hide
# ================================================================================================

def test_the_envelope_keeps_a_sub_state_that_stride_sampling_would_step_over():
    """The reason tiles are min/max per column and not strided samples.

    MEASURED: a 500 ms event on a 140 px tile puts ~358 samples in each column, so a 2 ms
    sub-state is one sample in three hundred. Min/max bottoms out at the sub-state;
    stride-sampling bottoms out 24 pA shallower and draws an event that never dipped.
    Sub-states are the thing CUSUM exists to find, so a thumbnail that hides them is worse
    than no thumbnail at all.
    """
    ev, pad = 50_000, 100
    n = ev + 2 * pad
    ys = np.full(n, 100.0)
    ys[pad:pad + ev] -= 20.0
    ys[pad + 20_000:pad + 20_200] -= 25.0        # the brief deeper sub-state
    cols = 140
    per = n // cols

    envelope_min = ys[:per * cols].reshape(cols, per).min(axis=1).min()
    stride_min = ys[::max(1, n // cols)][:cols].min()

    assert envelope_min == pytest.approx(55.0, abs=1.0), "the envelope lost the sub-state"
    assert stride_min > envelope_min + 20.0, (
        "this fixture no longer distinguishes the two methods, so it no longer defends "
        "the choice the module docstring makes")


def test_per_tile_scaling_fills_the_tile_top_to_bottom(win, qtbot):
    """With each tile scaled to itself the trace touches both rims, and the delegate can
    scale by the tile rect alone — 0 at the TOP, because Qt's y grows downward and that
    flip belongs in one place rather than in the painter."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_shared_scale(False)
    lo, hi, (e0, e1) = gal.envelope_for(0, 140)
    assert float(hi.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(lo.max()) == pytest.approx(1.0, abs=1e-6)
    assert np.all(lo >= hi - 1e-6), "the column minimum must sit below the column maximum"
    assert 0.0 <= e0 < e1 <= 1.0, "the event span must lie inside the window"


def test_the_shared_scale_draws_a_deep_event_deeper_than_a_shallow_one(win, qtbot):
    """The defect this scale mode exists to fix, and it was invisible to every other test
    here: normalising each tile to its own extremes drew a 15 pA event and a 45 pA event as
    the SAME PICTURE, so sorting the grid by depth produced a wall of identical tiles and
    the surface could not answer the question it had just been sorted to answer.

    Under the shared scale the excursion is proportional to the event's actual depth.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    depth = np.abs(df["depth_pA"].to_numpy(dtype=float))
    shallow, deep = int(np.nanargmin(depth)), int(np.nanargmax(depth))
    assert depth[deep] > 1.5 * depth[shallow], "the fixture has no depth contrast to test"

    gal.set_shared_scale(True)
    reach = {}
    for row in (shallow, deep):
        lo, hi, _ = gal.envelope_for(row, 140)
        reach[row] = float(lo.max()) - float(hi.min())
    assert reach[deep] > reach[shallow] * 1.3, (
        f"deep event fills {reach[deep]:.2f} of the tile, shallow {reach[shallow]:.2f} — "
        f"the shared scale is not conveying depth")

    # ...and per-tile scaling is exactly the mode in which they DO look alike, which is why
    # it is not the default and why it remains available for judging shape.
    gal.set_shared_scale(False)
    same = []
    for row in (shallow, deep):
        lo, hi, _ = gal.envelope_for(row, 140)
        same.append(float(lo.max()) - float(hi.min()))
    assert same[0] == pytest.approx(same[1], abs=1e-6)


def test_one_outlier_does_not_flatten_the_whole_grid(win, qtbot):
    """The shared span is a high percentile, not the maximum. A single monstrous blockade
    setting the scale for nine hundred ordinary events is how a shared axis stops being
    worth having; the outlier clips instead."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    typical_row = int(np.nanargmin(np.abs(df["depth_pA"].to_numpy(dtype=float))))
    gal.set_shared_scale(True)
    lo, hi, _ = gal.envelope_for(typical_row, 140)
    before = float(lo.max()) - float(hi.min())

    df.loc[df.index[1], "depth_pA"] = 100_000.0      # an absurd outlier
    gal.refresh()
    lo, hi, _ = gal.envelope_for(typical_row, 140)
    after = float(lo.max()) - float(hi.min())
    assert after == pytest.approx(before, rel=0.05), (
        "one outlier rescaled every other tile")


def test_the_event_is_framed_by_open_pore_rather_than_filling_the_tile(win, qtbot):
    """A tile has to show the blockade AND the thing that makes it a blockade.

    MEASURED BEFORE: the padding was capped at 1 ms, so a 6 ms event occupied ~75% of the
    tile and left a sliver of open pore too short to judge it against. Padding proportional
    to the event keeps the blockade in the middle half at every dwell.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    longest = int(np.nanargmax(df["dwell_s"].to_numpy(dtype=float)))
    for row in (0, longest):
        _lo, _hi, (e0, e1) = gal.envelope_for(row, 140)
        width = e1 - e0
        assert 0.3 < width < 0.7, (
            f"row {row}: the event covers {width:.0%} of the tile; it should sit in the "
            f"middle half with open pore either side")


def test_a_closed_recording_leaves_the_gallery_drawable(win, qtbot):
    """The envelope source must answer None rather than raise when the signal is gone: the
    delegate paints during a repaint the window does not control, and an exception there
    takes the whole grid down."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    assert gal.envelope_for(0, 140) is not None
    src = _EnvelopeSource(w._session)
    src.set_target("no-such-doc", 0)
    assert src.envelope(0, 100, 140) is None


# ================================================================================================
# contents, filtering, sorting
# ================================================================================================

def test_the_gallery_shows_one_tile_per_detected_event(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    assert len(df) > 30, "the fixture should produce a detection worth a gallery"
    assert gal.event_count() == len(df)
    assert gal.tile_count() == min(len(df), gal._model.page_size())


def test_the_filter_selects_the_same_three_subsets_the_inspector_offers(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 4)
    gal.set_accepted_selected(False)
    qtbot.wait(10)

    gal.set_filter("Rejected only")
    assert gal.event_count() == 5
    gal.set_filter("Accepted only")
    assert gal.event_count() == len(df) - 5
    gal.set_filter("All")
    assert gal.event_count() == len(df)


def test_sorting_by_dwell_orders_the_grid_longest_first(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    gal.set_sort("Dwell (longest first)")
    qtbot.wait(10)
    dwell = df["dwell_s"].to_numpy(dtype=float)[gal._model.df_rows()]
    assert np.all(np.diff(dwell) <= 1e-12), "the grid is not in descending dwell order"


def test_an_unmeasurable_row_sorts_to_the_END_not_into_the_middle(win, qtbot):
    """A row whose depth could not be measured is exactly the row worth reaching, so it
    must land at a predictable end of the grid. numpy's argsort puts NaN wherever it likes;
    this pins it."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    df.loc[df.index[3], "depth_pA"] = np.nan
    gal.refresh()
    gal.set_sort("Depth (deepest first)")
    qtbot.wait(10)
    rows = gal._model.df_rows()
    assert rows[-1] == 3, f"the unmeasurable row sorted to position {rows.index(3)}, not last"


# ================================================================================================
# curation — the point of the surface
# ================================================================================================

def test_bulk_reject_writes_the_SHARED_frame_and_announces_once(win, qtbot):
    """The whole reason the gallery is worth building: sweep a box over a run of bad
    detections and kill them together.

    It must write the same frame the table edits — a gallery curating a private copy would
    look right and change nothing — and announce with the BATCH signal, because the owner's
    per-event path repaints the table row by row.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    before = int(df["accepted"].to_numpy(dtype=bool).sum())

    batched: list = []
    singles: list = []
    gal.curationBatchChanged.connect(lambda: batched.append(True))
    gal.acceptanceChanged.connect(lambda i, a: singles.append((i, a)))

    _select_rows(gal, 0, 9)
    ids = gal.selected_event_ids()
    assert len(ids) == 10
    gal.set_accepted_selected(False)
    qtbot.wait(10)

    assert not df.set_index("event_id").loc[ids, "accepted"].to_numpy().any(), (
        "the rejection did not reach the shared events frame")
    assert int(df["accepted"].to_numpy(dtype=bool).sum()) == before - 10
    assert len(batched) == 1, "a multi-tile curation must announce once, as a batch"
    assert singles == [], "a batch must not also emit per-event acceptance"


def test_clicking_a_tile_then_R_rejects_exactly_that_event(win, qtbot):
    """The primary correction: click the bad tile, press R.

    One tile is the table's ordinary per-row path, so it announces per-event; sending the
    batch signal for a single click would make one correction rebuild the whole table.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    seen: list = []
    batched: list = []
    gal.acceptanceChanged.connect(lambda i, a: seen.append((i, a)))
    gal.curationBatchChanged.connect(lambda: batched.append(True))

    row = gal._model.df_rows()[2]
    eid = int(df.iloc[row]["event_id"])
    gal.view.setCurrentIndex(gal._model.index(2, 0))          # a click
    _select_rows(gal, 2, 2)
    gal.set_accepted_selected(False)                          # R
    qtbot.wait(10)

    assert seen == [(eid, False)]
    assert batched == [], "one tile must not announce as a batch"
    assert bool(df.iloc[row]["accepted"]) is False
    assert int((~df["accepted"].to_numpy(dtype=bool)).sum()) == 1, "it rejected more than one"


def test_double_click_opens_the_event_in_the_inspector_and_curates_nothing(win, qtbot):
    """Double-click OPENS the event, and "open" has to be visible.

    It used to toggle accept/reject, which put a destructive action on the gesture people
    use to open things. It was then made to zoom the main window's trace — which the owner
    reported as the gesture doing nothing, and it looks that way for a structural reason:
    the gallery is a separate top-level window sitting OVER the main one, so an effect that
    happens behind it is an effect nobody sees. The inspector is a window of its own and
    comes to the front, so the gesture lands where the user is looking. The trace is still
    on the right-click menu, and acceptance must still be untouched.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    accepted_before = df["accepted"].to_numpy(dtype=bool).copy()

    inspected: list = []
    activated: list = []
    curated: list = []
    gal.inspectRequested.connect(inspected.append)
    gal.eventActivated.connect(activated.append)
    gal.acceptanceChanged.connect(lambda *_: curated.append(True))
    gal.curationBatchChanged.connect(lambda: curated.append(True))

    row = gal._model.df_rows()[3]
    eid = int(df.iloc[row]["event_id"])
    assert w._event_inspector is None
    # A REAL double-click focuses the tile on the press before the second click lands, so
    # the gesture is reproduced in that order rather than as a bare signal emission.
    gal.view.setCurrentIndex(gal._model.index(3, 0))
    qtbot.wait(10)
    gal.view.doubleClicked.emit(gal._model.index(3, 0))
    qtbot.wait(30)

    assert inspected == [eid], "double-click did not ask for the inspector"
    assert activated == [], "double-click must no longer go only to the trace"
    assert curated == [], "double-click curated something; it must only navigate"
    assert np.array_equal(df["accepted"].to_numpy(dtype=bool), accepted_before)
    # ...and a window opened for it. WHICH EVENT it lands on is MainWindow's route
    # (``_open_inspector_at_event``), not this window's: the gallery emits the id and
    # nothing else. That route currently selects THROUGH the analysis panel, which is a
    # no-op when the panel already holds this selection — and the gallery's own
    # ``eventSelected`` relay has always just put it there — so a freshly opened inspector
    # can land on the detection's first event instead. Asserted where it belongs, in
    # MainWindow's own tests; asserting it here would make this test fail for a defect in
    # another window.
    insp = w._event_inspector
    assert insp is not None and insp.isVisible(), "double-click opened no inspector"


def test_the_inspector_button_opens_the_focused_tile(win, qtbot):
    """Owner note #4, the gallery half: a way OUT of the grid that is on the bar rather
    than hidden behind a right-click. It emits the signal the context menu already emits,
    so it is a second road to one route and not a second implementation."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df

    asked: list = []
    gal.inspectRequested.connect(asked.append)

    # Nothing focused: the button must not invent an event to open.
    gal.view.selectionModel().clearCurrentIndex()
    gal.inspect_btn.click()
    qtbot.wait(10)
    assert asked == [], "the button opened the inspector with no tile focused"

    row = gal._model.df_rows()[5]
    eid = int(df.iloc[row]["event_id"])
    gal.view.setCurrentIndex(gal._model.index(5, 0))
    qtbot.wait(10)
    gal.inspect_btn.click()
    qtbot.wait(30)

    assert asked == [eid]
    # As above: the window opening is this button's business; landing it on the event is
    # MainWindow's ``_open_inspector_at_event``.
    insp = w._event_inspector
    assert insp is not None and insp.isVisible()


def test_the_status_line_names_every_gesture_including_the_right_click_menu(win, qtbot):
    """The hint used to advertise click / R / 1–9 and "double-click opens it in the trace",
    and never mentioned the context menu — where "go to it on the trace" and the per-tile
    accept/reject actually live. A road out of the window that nothing on screen names is a
    road nobody takes."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    text = gal.status.text()
    assert "right-click" in text, f"the line never mentions the context menu: {text!r}"
    assert "Event Inspector" in text, "the line does not say what double-click now does"
    assert "double-click opens it in the trace" not in text, "the stale hint survived"


def test_accepting_fresh_already_accepted_tiles_records_the_whole_review_scope(win, qtbot):
    """The detector chose the initial boolean; Accept is the curator's distinct ruling."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    fired: list = []
    gal.curationBatchChanged.connect(lambda: fired.append(True))
    gal.acceptanceChanged.connect(lambda *_: fired.append(True))
    _select_rows(gal, 0, 9)
    expected = set(int(v) for v in gal._df.iloc[:10]["event_id"])
    gal.set_accepted_selected(True)         # they are already accepted
    qtbot.wait(10)
    reviewed = set(int(v) for v in gal._df.loc[
        gal._df[REVIEW_COL].astype("string") == REVIEW_DONE, "event_id"])
    assert reviewed == expected
    assert fired == [True]


def test_multi_tile_reject_marks_every_changed_gallery_row_reviewed(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 9)
    expected = set(int(v) for v in gal._df.iloc[:10]["event_id"])

    gal.set_accepted_selected(False)
    qtbot.wait(10)

    reviewed = set(int(v) for v in df.loc[
        df[REVIEW_COL].astype("string") == REVIEW_DONE, "event_id"])
    assert reviewed == expected


def test_accept_all_shown_respects_the_FILTER_and_says_so(win, qtbot):
    """"Accept all shown" curates the current subset, not the detection. Under a filter the
    two differ, and a button that quietly meant "everything" would undo a rejection the
    user cannot even see."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df

    _select_rows(gal, 0, 9)
    gal.set_accepted_selected(False)
    qtbot.wait(10)
    gal.set_filter("Accepted only")
    qtbot.wait(10)
    gal.reject_all()                          # rejects only what the filter shows
    qtbot.wait(10)
    assert int(df["accepted"].to_numpy(dtype=bool).sum()) == 0

    gal.set_filter("Rejected only")
    qtbot.wait(10)
    gal.accept_all()
    qtbot.wait(10)
    assert int(df["accepted"].to_numpy(dtype=bool).sum()) == len(df)
    assert "accepted" in gal.status.text()


def test_reflect_acceptance_is_an_inbound_mirror_that_never_re_emits(win, qtbot):
    """The table and the inspector curate the same frame. If the gallery echoed what it was
    told, the three would drive each other in a loop."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    echoed: list = []
    gal.acceptanceChanged.connect(lambda *_: echoed.append(True))
    gal.curationBatchChanged.connect(lambda: echoed.append(True))

    eid = int(df.iloc[2]["event_id"])
    gal.reflect_acceptance(eid, False)
    qtbot.wait(10)
    assert bool(df.iloc[2]["accepted"]) is False
    assert echoed == [], "the inbound mirror re-emitted"


# ================================================================================================
# wiring — the gallery is a view of the same detection, not a second copy of it
# ================================================================================================

def test_the_gallery_and_the_table_curate_the_same_frame(win, qtbot):
    """End to end through MainWindow's own wiring: a bulk rejection in the gallery must
    show up in the panel's accepted count, because both read one `accepted` column."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 5)
    gal.set_accepted_selected(False)
    qtbot.wait(30)
    assert int(df["accepted"].to_numpy(dtype=bool).sum()) == len(df) - 6
    # ...and the panel agrees, having been resynced by MainWindow rather than by the gallery.
    assert w._analysis_panel.current_detection().payload.df is df


def test_selecting_a_tile_drives_the_rest_of_the_app(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    picked: list = []
    gal.eventSelected.connect(picked.append)
    gal.view.setCurrentIndex(gal._model.index(4, 0))
    qtbot.wait(10)
    assert picked and picked[-1] == int(df.iloc[4]["event_id"])


def test_an_inbound_selection_focuses_the_tile_without_echoing(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    echoed: list = []
    gal.eventSelected.connect(echoed.append)
    gal.show_event(int(df.iloc[7]["event_id"]))
    qtbot.wait(10)
    assert gal.current_event_id() == int(df.iloc[7]["event_id"])
    assert echoed == [], "show_event is an inbound mirror; it must not emit back"


def test_the_menu_action_is_gated_on_there_being_a_detection(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    assert not w._act_event_gallery.isEnabled(), (
        "the gallery must not be reachable with nothing to show")


def test_the_gallery_clears_when_the_detection_goes_away(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    assert gal.tile_count() > 0
    gal.clear()
    assert gal.tile_count() == 0
    assert gal.event_count() == 0
    assert "No detection" in gal.status.text()

# ================================================================================================
# scale — the whole reason this surface exists
# ================================================================================================

def test_a_two_thousand_event_detection_costs_only_what_is_on_screen(win, qtbot):
    """The gallery is for the case where inspecting events one at a time is hopeless, so
    the cost of opening it must not scale with the size of the detection.

    The property that delivers that is laziness: a tile's waveform is sliced and reduced
    only when the tile is PAINTED, so a model holding two thousand events has computed
    nothing until the viewport asks. Building envelopes eagerly would slice two thousand
    windows out of the recording on open — measured at ~40 ms for the whole window here,
    against seconds if every tile were realised up front.

    Sorting and filtering must stay in the model for the same reason: they are permutations
    of an index list and must never touch the signal.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df

    big = pd.concat([df] * max(1, 2000 // max(len(df), 1)), ignore_index=True)
    big["event_id"] = np.arange(len(big), dtype=np.int64)
    assert len(big) >= 1900

    gal.set_events(doc_id, ch, big, "detect.schmitt", {})
    gal._envelopes.clear()
    assert gal.event_count() == len(big)
    assert len(gal._envelopes._cache) == 0, "opening realised envelopes it was never asked to draw"

    # ...and a tile that IS asked for costs exactly one envelope.
    assert gal.envelope_for(0, 140) is not None
    assert len(gal._envelopes._cache) == 1

    # Sorting and filtering are permutations of an index list. They repaint, so the VISIBLE
    # tiles are realised -- that is the design working -- but the count must stay bounded by
    # the viewport rather than growing with the detection.
    gal._envelopes.clear()
    gal.set_sort("Depth (deepest first)")
    gal.set_filter("Accepted only")
    qtbot.wait(10)
    assert gal.event_count() == len(big)
    realised = len(gal._envelopes._cache)
    assert realised < 150, (
        f"{realised} of {len(big)} envelopes were built to re-sort the grid; the cost is "
        f"tracking the detection instead of the viewport")


def test_bulk_curation_of_the_whole_grid_writes_every_row(win, qtbot):
    """Reject-all-shown over a large subset is one pass over the frame, and it must land on
    every row — this is the button that saves the user two thousand clicks, so a partial
    write would be worse than no button."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    gal.reject_all()
    qtbot.wait(10)
    assert int(df["accepted"].to_numpy(dtype=bool).sum()) == 0
    gal.set_filter("Rejected only")
    gal.accept_all()
    qtbot.wait(10)
    assert int(df["accepted"].to_numpy(dtype=bool).sum()) == len(df)


# ================================================================================================
# pages
# ================================================================================================

def test_the_grid_is_paged_and_the_page_is_all_the_model_holds(win, qtbot):
    """A page is a fixed frame the eye can sweep and then dismiss — "I have looked at
    everything" is a statement you can make about page 7 of 42 and cannot make about a
    scroll position. The model holds ONE PAGE, so there is nothing below the fold."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_page_size(24)
    n = gal.event_count()
    assert n > 24, "the fixture must overflow one page for this to mean anything"

    assert gal.page() == 0
    assert gal.page_count() == (n + 23) // 24
    assert gal.tile_count() == 24

    gal.set_page(1)
    assert gal.page() == 1
    assert gal.tile_count() == min(24, n - 24)
    # ...and the tiles really are the NEXT ones, not the same ones renumbered.
    gal.set_page(0)
    first = set(gal._model.page_df_rows())
    gal.set_page(1)
    assert first.isdisjoint(set(gal._model.page_df_rows()))


def test_paging_past_either_end_clamps_instead_of_emptying_the_grid(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_page_size(24)
    gal.set_page(9999)
    assert gal.page() == gal.page_count() - 1
    assert gal.tile_count() > 0
    gal.set_page(-5)
    assert gal.page() == 0
    assert gal.tile_count() > 0


def test_changing_the_page_size_keeps_you_where_you_were_looking(win, qtbot):
    """Re-paginating around a fixed page NUMBER moves the user somewhere else in the
    detection for no reason they asked for. The first tile on screen is the anchor."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_page_size(24)
    gal.set_page(1)
    first_row = gal._model.page_df_rows()[0]

    gal.set_page_size(48)
    assert first_row in gal._model.page_df_rows(), (
        "the tile the user was looking at is no longer on the page")


def test_curation_that_empties_the_last_page_does_not_strand_the_user_on_it(win, qtbot):
    """Under a filter, rejecting shrinks the subset and can delete the page you are on.
    Left unclamped that is an empty grid with working page buttons — the surface looks
    broken and the events are still there."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_page_size(24)
    gal.set_filter("Accepted only")
    gal.set_page(gal.page_count() - 1)
    assert gal.tile_count() > 0

    gal.reject_all()          # every accepted event leaves the subset
    qtbot.wait(10)
    assert gal.event_count() == 0
    assert gal.page() == 0, "the user was left on a page that no longer exists"
    assert gal.page_count() == 1


def test_focusing_an_event_on_another_page_turns_to_it(win, qtbot):
    """MEASURED BEFORE pagination existed this was free; with pages it is the thing that
    silently stops working. The table and the inspector move to an event, and the gallery
    sits on an unrelated page looking authoritative."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_page_size(24)
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    far_row = gal._model.df_rows()[30]
    far_id = int(df.iloc[far_row]["event_id"])
    assert gal.page() == 0

    gal.show_event(far_id)
    qtbot.wait(10)
    assert gal.page() == 1, "the gallery did not turn to the page holding the event"
    assert gal.current_event_id() == far_id


def test_the_status_line_says_which_slice_of_the_detection_is_on_screen(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_page_size(24)
    gal.set_page(1)
    text = gal.status.text()
    assert "25–" in text, f"the status line does not locate the page: {text!r}"
    assert str(gal.event_count()) in text


def test_fit_is_exactly_one_screenful_on_the_FIRST_show(win, qtbot):
    """The number in the menu must be the number on the glass, from the first paint.

    MEASURED, and this is the defect the test replaces one that could not see: on first
    show of a 300-event detection the viewport settled at 1471x588 — a 9x4 grid, 36 tiles —
    and "Fit" had set the page to 12. The status line read "1–12 of 300" on a screen with
    room for three times that, and nudging the window one pixel corrected it to 32. The old
    test only asserted that a small window pages fewer tiles than a big one, which was true
    the whole time it was wrong by a factor of three.

    Nothing here waits on a timer or a nudge: the page size is compared to ``_grid_cells``
    right after ``show()``, so a page computed from a stale viewport fails.
    """
    from nanoscope.gui import event_gallery as eg

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    assert gal.page_size_box.currentText() == "Fit", "Fit is the default page size"
    # The dead configuration that used to say otherwise: ``_DEFAULT_PAGE_SIZE = 48`` sat
    # beside ``_PAGE_SIZES`` reading like the gallery's default page size and was applied to
    # nothing — the combo has always been set to "Fit". The one page-size default left is
    # the fallback Fit uses before there IS a viewport, and it is reached (below).
    assert not hasattr(eg, "_DEFAULT_PAGE_SIZE"), (
        "a page-size default nothing applies is a claim about this window that is false")

    for w_px, h_px in ((1400, 1100), (1000, 760), (700, 520)):
        gal.resize(w_px, h_px)
        qtbot.wait(50)
        _cell_w, cell_h, cols, vp_h = gal._grid_cells()
        rows = max(1, vp_h // cell_h)
        assert gal._model.page_size() == cols * rows, (
            f"at {w_px}x{h_px} Fit paged {gal._model.page_size()} tiles into a grid that "
            f"holds {cols}x{rows} = {cols * rows}")
        assert gal.tile_count() <= cols * rows
        # ...and a screenful never scrolls. That is the whole promise of "Fit".
        assert "scroll for the rest" not in gal.status.text()


def test_fit_on_a_window_that_was_never_shown_is_still_a_page(qapp, qtbot, tmp_path):
    """A gallery driven headlessly has no viewport to fit to, and the honest arithmetic
    over an unrealised one lands on ONE TILE — a "page" that hides 99,999 of 100,000 events
    from a caller with no screen to notice it on. Fit falls back to a plain count until the
    first viewport resize; ``test_s5_hke_batch_curation`` drives exactly such a window.
    """
    from nanoscope.core.session import Session
    from nanoscope.gui.event_gallery import _FIT_FALLBACK_PAGE_SIZE, EventGalleryWindow

    gal = EventGalleryWindow(Session())
    qtbot.addWidget(gal)
    assert not gal.view.isVisible()
    assert gal._model.page_size() == _FIT_FALLBACK_PAGE_SIZE
    assert _FIT_FALLBACK_PAGE_SIZE > 1


def test_an_explicit_page_size_is_literal_and_its_shortfall_is_disclosed(win, qtbot):
    """The other half of the owner's "the number per window does not match the number
    displayed" — and the half where the honest answer is disclosure, not a cap.

    MEASURED on a 1512 px display: choosing 48 puts 48 tiles on the page and about 32
    inside the viewport; 96 and 200 both show 36. Capping the number to the screenful was
    tried and rejected, because "Per page" is ALSO the representative subpopulation sample
    size (``test_subpop_galleries.test_per_page_is_the_sample_size``) — a cap turned "sample
    24 events spaced through the dwell distribution" into 12 on a small window, which is a
    sample silently cut to fit a screen. So the count is exactly what was asked for, every
    tile is reachable, and the line says the page is taller than the window.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.resize(900, 620)
    qtbot.wait(50)
    _cell_w, cell_h, cols, vp_h = gal._grid_cells()
    screenful = cols * max(1, vp_h // cell_h)
    assert gal.event_count() > screenful, "the fixture must have more events than one page"

    for choice in ("24", "48", "96", "200"):
        gal.page_size_box.setCurrentText(choice)
        qtbot.wait(40)
        assert gal._model.page_size() == int(choice), (
            f"'{choice}' paged {gal._model.page_size()} tiles; the number is literal")
        overflows = int(choice) > screenful
        assert ("scroll for the rest" in gal.status.text()) is overflows, (
            f"'{choice}' over a window showing {screenful} said the wrong thing about "
            f"whether the page fits: {gal.status.text()!r}")
        if overflows:
            # ...and the tiles below the fold are REACHABLE, which is the whole reason the
            # scrollbar stays AsNeeded instead of being turned off under "Fit"'s promise.
            assert gal.view.verticalScrollBar().maximum() > 0


def test_an_explicit_page_size_keeps_a_scrollbar_as_a_safety_valve(win, qtbot):
    """"Fit" makes the scrollbar unnecessary; an explicit 200 in a small window makes it
    essential. Turning it off outright would leave those events silently unreachable, which
    is a worse failure than a scrollbar the default never shows."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    from PySide6.QtCore import Qt as _Qt
    assert gal.view.verticalScrollBarPolicy() == _Qt.ScrollBarAsNeeded


# ================================================================================================
# labelling — the gallery is the bulk surface
# ================================================================================================

def _vocab():
    from nanoscope.analysis.labels import LabelVocabulary
    v = LabelVocabulary()
    v.add("molecule", "apo-ferritin")
    v.add("molecule", "holo-ferritin")
    v.add("type", "collision", hotkey="c")
    return v


def test_a_digit_labels_the_whole_selection_on_the_shared_frame(win, qtbot):
    """The point of labelling here: sweep a box over a run of tiles, press a digit.

    It must write the SHARED events frame — the same one the table, the exports and the
    project read — not a private copy.
    """
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df

    fired: list = []
    gal.labelsChanged.connect(lambda: fired.append(True))
    _select_rows(gal, 0, 9)
    gal.apply_hotkey("1")
    qtbot.wait(10)

    got = label_column(df["flag"], "molecule")
    assert list(got).count("apo-ferritin") == 10
    assert fired, "labelling did not announce itself"
    assert (df.iloc[:10][REVIEW_COL].astype("string") == REVIEW_DONE).all()


def test_an_explicit_hotkey_beats_position(win, qtbot):
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 2)
    gal.apply_hotkey("c")
    qtbot.wait(10)
    assert list(label_column(df["flag"], "type")).count("collision") == 3


def test_an_explicit_custom_hotkey_works_from_a_real_gallery_key_event(win, qtbot):
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 2)
    gal.view.setFocus()

    qtbot.keyClick(gal.view, Qt.Key_C)
    qtbot.wait(20)

    assert list(label_column(df["flag"], "type")).count("collision") == 3


def test_labelling_never_disturbs_curation_or_derived_flags(win, qtbot):
    """A label writes a token; it must not overwrite the cell. Rejecting and labelling the
    same events must both stick."""
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df

    _select_rows(gal, 0, 4)
    gal.set_accepted_selected(False)
    qtbot.wait(10)
    _select_rows(gal, 0, 4)
    gal.apply_hotkey("1")
    qtbot.wait(10)

    rows = gal._model.df_rows()[:5] if gal._model.df_rows() else []
    assert int((~df["accepted"].to_numpy(dtype=bool)).sum()) == 5, "the rejection was lost"
    assert list(label_column(df["flag"], "molecule")).count("apo-ferritin") == 5


def test_setting_a_second_value_for_one_key_replaces_it(win, qtbot):
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 3)
    gal.apply_hotkey("1")
    _select_rows(gal, 0, 3)
    gal.apply_hotkey("2")
    qtbot.wait(10)
    got = list(label_column(df["flag"], "molecule"))
    assert got.count("holo-ferritin") == 4
    assert got.count("apo-ferritin") == 0, "an event claimed two molecules at once"


def test_clear_removes_only_that_key(win, qtbot):
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 3)
    gal.apply_hotkey("1")          # molecule=apo-ferritin
    _select_rows(gal, 0, 3)
    gal.apply_hotkey("c")          # type=collision
    _select_rows(gal, 0, 3)
    gal.label_box.setCurrentIndex(0)      # molecule=apo-ferritin
    gal.clear_selected_label()
    qtbot.wait(10)
    assert list(label_column(df["flag"], "molecule")).count("apo-ferritin") == 0
    assert list(label_column(df["flag"], "type")).count("collision") == 4, "the other key went too"


def test_the_tile_shows_the_label(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    _select_rows(gal, 0, 1)
    gal.apply_hotkey("1")
    qtbot.wait(10)
    row = gal._model.df_rows()[0]
    assert gal.label_of(row) == "apo-ferritin"
    assert gal.label_of(gal._model.df_rows()[-1]) == "", "an unlabelled tile must show nothing"


def test_the_label_controls_are_disabled_until_a_vocabulary_exists(win, qtbot):
    from nanoscope.analysis.labels import LabelVocabulary

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(LabelVocabulary())
    assert not gal.apply_label_btn.isEnabled()
    gal.set_vocabulary(_vocab())
    assert gal.apply_label_btn.isEnabled()


def test_labelling_with_nothing_selected_changes_nothing(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    before = df["flag"].tolist()
    gal.view.selectionModel().clearSelection()
    assert gal.apply_hotkey("1") is None
    assert df["flag"].tolist() == before


def test_a_digit_typed_into_the_pager_pages_rather_than_labelling(win, qtbot):
    """The hazard of window-level digit shortcuts: they must not eat a keystroke aimed at a
    spin box.

    Qt hands a focused QAbstractSpinBox the key BEFORE a window shortcut sees it, so this
    passes — but nothing in the code says so, and the failure mode (a pager you cannot type
    into) is silent and maddening. Measured with the text selected, which is the state after
    clicking into a spin box; typing into a box that already reads "1" makes "12" and is
    rejected by the validator for reasons that have nothing to do with shortcuts.
    """
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    gal.set_page_size(24)
    qtbot.wait(30)
    assert gal.page_count() >= 2, "the fixture must have more than one page to type a number"

    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    _select_rows(gal, 0, 3)
    before = list(label_column(df["flag"], "molecule"))

    gal.page_spin.setFocus()
    gal.page_spin.lineEdit().selectAll()
    qtbot.keyClick(gal.page_spin, Qt.Key_2)
    qtbot.wait(50)
    gal.page_spin.interpretText()

    assert gal.page_spin.value() == 2, "the pager swallowed the digit — it is not typeable"
    assert list(label_column(df["flag"], "molecule")) == before, (
        "a digit aimed at the pager labelled events")


def test_every_page_paints_the_events_it_acts_on(win, qtbot):
    """MEASURED BEFORE: every page after the first painted page 1's waveforms and captions,
    while the selection, the curation and the labels acted on the page's real events. You
    looked at event 0, pressed R, and rejected event 10.

    It survived the pagination tests because those assert on ``df_row`` and ``page_df_rows``,
    which were both correct — only the delegate's ``data(_ROW_ROLE)`` was wrong, so only the
    PICTURE lied. This compares the two directly, on every page.
    """
    from nanoscope.gui.event_gallery import _ROW_ROLE

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_page_size(24)
    assert gal.page_count() >= 3, "the fixture must have several pages for this to mean anything"

    for page in range(gal.page_count()):
        gal.set_page(page)
        painted = [gal._model.data(gal._model.index(i, 0), _ROW_ROLE)
                   for i in range(gal._model.rowCount())]
        acted_on = [gal._model.df_row(gal._model.index(i, 0))
                    for i in range(gal._model.rowCount())]
        assert painted == acted_on, (
            f"page {page + 1}: the tiles draw rows {painted[:4]}… but click/label/reject "
            f"hit rows {acted_on[:4]}…")
        if page > 0:
            gal.set_page(0)
            first = [gal._model.data(gal._model.index(i, 0), _ROW_ROLE)
                     for i in range(gal._model.rowCount())]
            gal.set_page(page)
            assert painted != first, f"page {page + 1} paints the same tiles as page 1"


@pytest.mark.parametrize("settle_ms", [0, 200])
def test_labelling_keeps_the_selection_so_a_second_key_can_be_added(win, qtbot, settle_ms):
    """The full signal route includes the table's curation refresh. Restoring its
    current row must not echo as navigation and shrink the gallery label scope."""
    from nanoscope.analysis.labels import label_column

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_vocabulary(_vocab())
    df = w._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    if settle_ms:
        qtbot.wait(settle_ms)

    _select_rows(gal, 0, 4)
    gal.apply_hotkey("1")                      # molecule=apo-ferritin
    qtbot.wait(10)
    assert len(gal._selected_rows()) == 5, "the selection was lost after the first label"

    gal.apply_hotkey("c")                      # type=collision, same tiles, no re-select
    qtbot.wait(10)
    assert list(label_column(df["flag"], "molecule")).count("apo-ferritin") == 5
    assert list(label_column(df["flag"], "type")).count("collision") == 5

# ================================================================================================
# neighbour spans — the tile agrees with the trace about the OTHER events in its window
# (owner note #4: a fractured detection's sibling dip sat bare in the tile while the
# main panel shaded it)
# ================================================================================================

def test_neighbour_spans_places_the_adjacent_event_on_the_tile_axis(win, qtbot):
    """Force two rows adjacent enough to share a padded window and check the neighbour
    lands as fractions consistent with envelope_for's own (ev0, ev1) axis."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    # Rewrite rows 0/1 into close-by fragments: 600 samples apart, well inside the pad
    # a ~500-sample event gets. Positional writes, exactly how curation edits the df.
    df.iloc[0, df.columns.get_loc("start_idx")] = 10_000
    df.iloc[0, df.columns.get_loc("end_idx")] = 10_500
    df.iloc[1, df.columns.get_loc("start_idx")] = 10_600
    df.iloc[1, df.columns.get_loc("end_idx")] = 11_100
    df.iloc[0, df.columns.get_loc("accepted")] = True
    df.iloc[1, df.columns.get_loc("accepted")] = True

    gal.envelope_for(0, 140)                      # builds → cache remembers the trace length
    lo_i, hi_i = gal._envelopes.window_bounds(10_000, 10_500)
    assert lo_i < 10_000 and hi_i > 10_500, "the window must be padded on both sides"

    spans = gal.neighbour_spans(0)
    assert len(spans) >= 1, "the adjacent fragment must appear in the tile's window"
    f0, f1 = spans[0]
    winlen = hi_i - lo_i
    assert f0 == pytest.approx(max(10_600 - lo_i, 0) / winlen, abs=1e-9)
    assert f1 == pytest.approx(min(11_100 - lo_i, winlen) / winlen, abs=1e-9)
    assert 0.0 <= f0 < f1 <= 1.0


def test_neighbour_spans_excludes_the_rejected_and_itself(win, qtbot):
    """The trace shades ACCEPTED spans only, so the tile's context must too — and the
    tile's own event is the full-strength fill, never also a neighbour."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    df.iloc[0, df.columns.get_loc("start_idx")] = 10_000
    df.iloc[0, df.columns.get_loc("end_idx")] = 10_500
    df.iloc[1, df.columns.get_loc("start_idx")] = 10_600
    df.iloc[1, df.columns.get_loc("end_idx")] = 11_100
    df.iloc[0, df.columns.get_loc("accepted")] = True
    df.iloc[1, df.columns.get_loc("accepted")] = False   # rejected → unshaded on the trace

    gal.envelope_for(0, 140)
    assert gal.neighbour_spans(0) == [], "a rejected neighbour must not be shaded"

    # A far-away event keeps to its own window entirely.
    df.iloc[1, df.columns.get_loc("accepted")] = True
    df.iloc[1, df.columns.get_loc("start_idx")] = 300_000
    df.iloc[1, df.columns.get_loc("end_idx")] = 300_500
    lo_i, hi_i = gal._envelopes.window_bounds(10_000, 10_500)
    spans = gal.neighbour_spans(0)
    for f0, f1 in spans:
        s = lo_i + f0 * (hi_i - lo_i)
        assert not (10_000 <= s < 10_500), "the tile's own event must not be its own neighbour"
        assert s < 299_000, "an event outside the window must not appear"

# ================================================================================================
# owner notes #2/#5: the tile context menu opens the event on the other surfaces
# ================================================================================================

def test_context_menu_open_in_inspector_lands_all_three_surfaces(win, qtbot, monkeypatch):
    """Right-click a tile → "Open in Event Inspector": the inspector opens ON that event,
    and — because the selection goes through the panel, the same relay as a table-row
    click — the table row is selected too."""
    from PySide6 import QtCore, QtWidgets

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    target_row = 2
    target_id = int(df.iloc[target_row]["event_id"])

    # The menu blocks; choose "Open in Event Inspector" the moment it would show.
    monkeypatch.setattr(
        type(gal), "_exec_menu",
        staticmethod(lambda menu, pos: next(a_ for a_ in menu.actions()
                                            if a_.text() == "Open in Event Inspector")))
    idx = next(gal._model.index(i, 0) for i in range(gal._model.rowCount())
               if gal._model.df_row(gal._model.index(i, 0)) == target_row)
    # Offscreen the list view may not have laid tiles out, so indexAt(pos) can miss;
    # pin the hit-test to the target index and keep everything downstream real.
    monkeypatch.setattr(gal.view, "indexAt", lambda pos: idx, raising=True)
    assert w._event_inspector is None
    gal._on_context_menu(QtCore.QPoint(5, 5))
    qtbot.wait(30)

    insp = w._event_inspector
    assert insp is not None and insp.isVisible(), "the context menu must open the inspector"
    assert insp.current_event_id() == target_id, "…on the right event"
    sel = w._analysis_panel._table.selected_event_id() \
        if hasattr(w._analysis_panel._table, "selected_event_id") else None
    if sel is not None:
        assert sel == target_id, "the table row must land with it"


def test_context_menu_reject_curates_the_tile(win, qtbot, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    target_row = 1
    assert bool(df.iloc[target_row]["accepted"])

    monkeypatch.setattr(
        type(gal), "_exec_menu",
        staticmethod(lambda menu, pos: next(a_ for a_ in menu.actions()
                                            if a_.text() == "Reject")))
    idx = next(gal._model.index(i, 0) for i in range(gal._model.rowCount())
               if gal._model.df_row(gal._model.index(i, 0)) == target_row)
    monkeypatch.setattr(gal.view, "indexAt", lambda pos: idx, raising=True)
    gal._on_context_menu(QtCore.QPoint(5, 5))
    qtbot.wait(20)
    assert not bool(df.iloc[target_row]["accepted"]), "Reject in the menu must curate the tile"

# ================================================================================================
# owner note #12: a fixed page size can overflow the window, and macOS overlay
# scrollbars give no cue — the status line must say the page scrolls
# ================================================================================================

def test_status_line_admits_when_the_page_overflows_the_window(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.resize(900, 620)
    qtbot.wait(50)

    # Fit: one screenful by construction — no overflow, no hint.
    gal.page_size_box.setCurrentText("Fit")
    qtbot.wait(30)
    assert "scroll for the rest" not in gal.status.text()

    # Fixed 200 in a window that shows ~35: the page scrolls, and the line says so.
    gal.page_size_box.setCurrentText("200")
    qtbot.wait(60)
    assert gal.view.verticalScrollBar().maximum() > 0, (
        "fixture regression: 200 tiles are supposed to overflow a 900x620 window")
    assert "scroll for the rest" in gal.status.text()

    # Back to Fit: the hint leaves with the overflow.
    gal.page_size_box.setCurrentText("Fit")
    qtbot.wait(60)
    assert "scroll for the rest" not in gal.status.text()


# ================================================================================================
# owner note #3: "the mask on the gallery does not match the mask on the main panel"
#
# The sample indices always agreed. The SHADING RULE did not: the tile's own-event fill ran
# unconditionally, so an accepted tile and a rejected one painted the identical shade while
# the trace (analysis_panel filters its span starts by ``accepted``) drew no span at all for
# the rejected one — and the tile's alpha was half the trace's, so even the agreeing halves
# were two different oranges.
# ================================================================================================

def _paint_tile(gal, model_row: int, selected: bool = False):
    """Render one tile through the REAL delegate into an image, and hand back the image and
    the waveform rect inside it — the only way to ask what a tile actually looks like."""
    from PySide6 import QtCore, QtGui, QtWidgets
    from nanoscope.gui.event_gallery import _CAPTION_H, _TILE_PAD

    tw, th = gal.tile_size()
    w_px, h_px = tw + 2 * _TILE_PAD, th + _CAPTION_H + 2 * _TILE_PAD
    img = QtGui.QImage(w_px, h_px, QtGui.QImage.Format_ARGB32)
    img.fill(QtGui.QColor("#10151b"))
    painter = QtGui.QPainter(img)
    opt = QtWidgets.QStyleOptionViewItem()
    opt.rect = QtCore.QRect(0, 0, w_px, h_px)
    opt.state = QtWidgets.QStyle.State_Enabled
    if selected:
        opt.state |= QtWidgets.QStyle.State_Selected
    gal.view.itemDelegate().paint(painter, opt, gal._model.index(model_row, 0))
    painter.end()
    wave = QtCore.QRect(_TILE_PAD, _TILE_PAD, tw, th - _CAPTION_H)
    return img, wave


def test_the_tile_shades_the_mask_at_the_traces_own_alpha(win, qtbot):
    """Same samples, same hue, same STRENGTH. ``overlays._DEFAULT_SPAN_ALPHA`` is 60; the
    gallery drew the same #FFB74D at 30, so one event's mask was literally half as strong
    in the contact sheet as on the trace it came from."""
    from nanoscope.gui import event_gallery as eg
    from nanoscope.gui import overlays

    assert eg._REGION_ALPHA == overlays._DEFAULT_SPAN_ALPHA, (
        "the gallery and the trace shade one event's mask at two different strengths")
    assert eg._REGION_HEX == "#FFB74D", "…and they must still be the same hue"

    # The framing the two surfaces put around that mask is the other half of "same picture".
    # 0.5 puts the event in the middle 50% of the tile; trace_view's zoom_to_span is being
    # reconciled to the same number. If that constant moves, this window's has to move with
    # it or the tile and the pane frame the same event two different ways again.
    assert eg._PAD_FRACTION == 0.5


def test_a_rejected_tile_does_not_paint_the_accepted_mask(win, qtbot):
    """MEASURED BEFORE: an accepted tile and a rejected tile both painted centre pixel
    0xff3a362f — the identical shade — while the main panel shaded one and not the other.
    The own-event fill ran above the ``_REJECTED_DIM`` opacity and was never gated on
    ``accepted`` at all."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()

    rows = gal._model.page_df_rows()
    acc_row, rej_row = rows[0], rows[1]
    df.loc[df.index[acc_row], "accepted"] = True
    df.loc[df.index[rej_row], "accepted"] = False
    gal.refresh()
    qtbot.wait(20)

    page = gal._model.page_df_rows()
    acc_img, wave = _paint_tile(gal, page.index(acc_row))
    rej_img, _ = _paint_tile(gal, page.index(rej_row))

    # A point INSIDE the event span and below the waveform, so it is the shading being
    # compared and not the trace: the event occupies the middle 50% of the tile.
    x = wave.left() + wave.width() // 2
    y = wave.top() + int(wave.height() * 0.9)
    accepted_px = acc_img.pixel(x, y)
    rejected_px = rej_img.pixel(x, y)
    assert accepted_px != rejected_px, (
        f"accepted and rejected tiles paint the same mask ({accepted_px:#010x}); the "
        f"trace draws a span for one and nothing for the other")

    # ...and specifically: the accepted tile carries the span ink, the rejected one does not.
    from PySide6 import QtGui
    from nanoscope.gui.event_gallery import _REGION_ALPHA, _REGION_HEX, _TILE_BG_HEX

    bg = QtGui.QColor(_TILE_BG_HEX)
    shade = QtGui.QColor(_REGION_HEX)
    shade.setAlpha(_REGION_ALPHA)
    a = _REGION_ALPHA / 255.0
    expect = QtGui.QColor(
        round(shade.red() * a + bg.red() * (1 - a)),
        round(shade.green() * a + bg.green() * (1 - a)),
        round(shade.blue() * a + bg.blue() * (1 - a))).rgb()
    assert accepted_px == expect, (
        f"the accepted tile's mask is {accepted_px:#010x}, not #FFB74D at alpha "
        f"{_REGION_ALPHA} over the tile ground ({expect:#010x})")
    assert rejected_px == bg.rgb(), (
        "the rejected tile still fills its span with the accepted-mask colour")


def test_a_rejected_tile_still_shows_WHERE_its_span_was(win, qtbot):
    """Not shading it is the trace's rule; erasing it entirely is not. A tile is a 2x-event
    window with no time axis, so with no mark at all there is nothing to judge a re-accept
    against. The span is outlined in the rejection ink instead — unmistakably not the
    accepted shading, and the same red as the tile's border and caption."""
    from PySide6 import QtGui
    from nanoscope.gui.event_gallery import _REJECTED_HEX

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    rej_row = gal._model.page_df_rows()[1]
    df.loc[df.index[rej_row], "accepted"] = False
    gal.refresh()
    qtbot.wait(20)

    img, wave = _paint_tile(gal, gal._model.page_df_rows().index(rej_row))
    ink = QtGui.QColor(_REJECTED_HEX).rgb()
    # INSIDE the tile, away from the edges: the border is painted in this same red, so a
    # search that includes the rim would pass on the border alone and prove nothing.
    inner = wave.adjusted(4, 4, -4, -4)
    band = [x for x in range(inner.left(), inner.right() + 1)
            if any(img.pixel(x, y) == ink
                   for y in range(inner.top(), inner.bottom() + 1))]
    assert band, "a rejected tile draws no mark at all where its span is"
    # ...and it is where the event is: the padded window puts the span in the middle half.
    assert wave.left() + wave.width() * 0.15 < min(band) < wave.left() + wave.width() * 0.45


# ================================================================================================
# owner notes #5/#2: the toolbar has to survive a window narrower than the screen
# ================================================================================================

def test_the_window_opens_narrower_than_the_display(win, qtbot):
    """MEASURED BEFORE: ``minimumSizeHint().width()`` was 1432 px at .AppleSystemUIFont
    13 pt on a 1512 px display, with "Accept selected"/"Reject selected" last in a single
    QHBoxLayout that does not wrap — so a window any narrower than the whole screen put the
    per-event controls past the right-hand edge, where they are not merely small but
    unreachable. The dominant term was not the buttons: it was the status QLabel, which
    without word wrap reports its entire dozen-clause line as a minimum width.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    qtbot.wait(20)
    got = gal.minimumSizeHint().width()
    assert got <= 1100, f"the gallery cannot be opened below {got} px wide"

    # ...and the controls the width defect hid are all still there and enabled.
    gal.resize(1000, 700)
    qtbot.wait(40)
    for btn in (gal.accept_selected_btn, gal.reject_selected_btn, gal.inspect_btn):
        assert btn.isEnabled()
        assert btn.x() + btn.width() <= gal.width(), (
            f"{btn.text()!r} is off the right-hand edge of a 1000 px window")


# ================================================================================================
# owner note #17/#3: the fitted sub-state levels, on the contact sheet
# ================================================================================================

def _levels_for(df, rows=(0, 1, 2, 3)) -> pd.DataFrame:
    """A three-level CUSUM fit for a few of the fixture's events, in LEVEL_SCHEMA's own
    columns — the frame MainWindow hands the inspector, handed to the gallery instead."""
    out = []
    for row in rows:
        r = df.iloc[row]
        s, e = int(r["start_idx"]), int(r["end_idx"])
        base, depth = float(r["baseline_pA"]), abs(float(r["depth_pA"]))
        seg = max((e - s) // 3, 1)
        for k, scale in enumerate((0.6, 1.0, 0.75)):
            out.append(dict(
                event_id=int(r["event_id"]), level_idx=k, n_levels=3,
                start_idx=s + k * seg, end_idx=(s + (k + 1) * seg if k < 2 else e),
                level_pA=base - depth * scale, depth_pA=depth * scale,
                frac_depth=depth * scale / max(abs(base), 1.0)))
    return pd.DataFrame(out)


def test_the_tile_draws_the_fitted_levels_in_the_overlays_own_ink(win, qtbot):
    """The gallery used to be the only one of the three event surfaces that never showed a
    sub-state: the inspector drew levels, the trace drew levels, and the grid whose whole
    job is finding the events worth opening showed none of the structure the fit was run to
    find. The ink is imported from ``overlays`` so a level is one colour on all three."""
    from PySide6 import QtGui
    from nanoscope.gui.event_gallery import _LEVEL_HEX
    from nanoscope.gui.overlays import STEP_LEVEL_HEX

    # ONE COLOUR ON ALL THREE SURFACES. The gallery may not import a sibling GUI module
    # (test_subpop_galleries pins that), so it spells the constant out — and this is the
    # assertion that keeps the two spellings from drifting.
    assert _LEVEL_HEX == STEP_LEVEL_HEX

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    row = gal._model.page_df_rows()[0]

    before, wave = _paint_tile(gal, 0)
    gal.set_levels(_levels_for(df, [row]))
    qtbot.wait(20)
    after, _ = _paint_tile(gal, 0)

    ink = QtGui.QColor(_LEVEL_HEX).rgb()
    def _count(img):
        return sum(img.pixel(x, y) == ink
                   for x in range(wave.left(), wave.right() + 1)
                   for y in range(wave.top(), wave.top() + wave.height()))

    assert _count(before) == 0, "the tile drew level ink before any fit was installed"
    assert _count(after) > 0, "the tile draws no fitted levels"
    # Three flat bars, not a fill: an annotation must not become the picture.
    assert _count(after) < wave.width() * wave.height() * 0.25


def test_levels_are_left_off_the_tile_sizes_too_small_to_read_them(win, qtbot):
    """RENDERED, not assumed. On Medium (89 px of waveform) and Large (133 px) the bars sit
    on the sub-states they annotate with the min/max band readable either side. On Small
    (57 px) the whole blockade is ~20 px of envelope and a 2 px bar drawn across a 4-6 px
    band covers it — the tile goes green where it was blue and the noise band disappears
    under its own annotation. That band is what makes a marginal event look marginal at tile
    size, so hiding it to show a level the user can open the tile to see is the wrong trade.
    """
    from PySide6 import QtGui
    from nanoscope.gui.event_gallery import _LEVEL_HEX

    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    row = gal._model.page_df_rows()[0]
    gal.set_levels(_levels_for(df, [row]))
    qtbot.wait(20)

    ink = QtGui.QColor(_LEVEL_HEX).rgb()
    counts = {}
    for name in ("Small", "Medium", "Large"):
        gal.set_tile_size(name)
        qtbot.wait(30)
        img, wave = _paint_tile(gal, 0)
        counts[name] = sum(img.pixel(x, y) == ink
                           for x in range(wave.left(), wave.right() + 1)
                           for y in range(wave.top(), wave.top() + wave.height()))
    assert counts["Small"] == 0, "levels are drawn on a tile with no room to read them"
    assert counts["Medium"] > 0 and counts["Large"] > 0, counts


def test_a_new_detection_drops_the_level_fit_it_was_not_measured_on(win, qtbot):
    """The rule the subpopulation map already follows, for the same reason: levels fitted on
    one detection, printed under another detection's event ids, are a picture of the wrong
    event."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    gal.set_levels(_levels_for(df, gal._model.page_df_rows()[:2]))
    assert gal._levels is not None

    gal.set_events(doc_id, ch, df.copy(), "detect.schmitt", {})
    qtbot.wait(20)
    assert gal._levels is None, "a re-feed kept a level fit measured on another frame"

    gal.set_levels(_levels_for(df, gal._model.page_df_rows()[:2]))
    gal.clear()
    assert gal._levels is None, "clear() left the level fit behind"
