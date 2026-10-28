"""S8-XS1 — the Results dock's Table selector re-feeds the Event inspector and the gallery.

MEASURED DEFECT (sweep 8, owner's -600 mV recording, Schmitt 317 rows + CUSUM 370 rows both
stored). The Event inspector sits on Schmitt event 159 ("Reviewed 0 / 317"); the user moves
the Results dock's Table selector to CUSUM. The dock, its summary and the trace overlays say
CUSUM; the inspector (docked or in its own window) and the Event gallery are still holding the
SCHMITT frame, because ``AnalysisPanel._show_result_choice`` shows the chosen table and emits
nothing that ``MainWindow`` re-feeds a surface on -- ``_refresh_event_inspector`` only ran on
the mutation signals. One R keypress then wrote TWO tables: the surface rejected its own
(Schmitt) frame in place, and its ``acceptanceChanged(event_id, accepted)`` -- which names no
table -- was relayed to ``AnalysisPanel.set_event_accepted``, which rejects whatever table the
dock shows *by event id* (CUSUM event 159, a different event). Edit > Undo took back the
second and left the first: Schmitt kept a "reviewed" rejection that rode into the saved
project and every export, and the menu then read "Undo -- nothing on this table; the
threshold detector (1 step) on this channel". Clicking a CUSUM row afterwards showed Schmitt's
event under the same id ("sync by id, not by event"). The mirror (surface on CUSUM, dock on
Schmitt), the inspector in its own window and the gallery's R on a tile did the same. The
detector combo on the Detection tab is the second door that swaps the table on screen with
no announcement, and had the same hole.

THE FIX. ``AnalysisPanel`` announces the swap (``resultSwitched``, emitted by the Table
selector's and the detector combo's handlers -- the two doors that change the table on screen
without writing a result), and ``MainWindow._on_result_switched`` re-feeds the two curation
surfaces and the subpopulation colouring that rides on the gallery's frame, the way a
recording switch does (``_on_selection``'s tail). The rule for a table the surface was not
holding is the one ``set_events`` already applies to any different detection, with nothing
announced: the inspector goes back to All and its first event, the gallery to its first tile
(``quiet=True``) with its Show filter kept, so a pure view switch moves neither the main
trace nor the Results selection. ``_levels_for_detection`` no longer hands the fit measured on
one table to the surfaces showing another. The acceptance relay (``_relay_acceptance``) is the
belt to those braces: a surface whose frame is not the table the dock shows can no longer have
its decision mirrored by event id onto the dock's table.

These tests drive the real ``MainWindow``: the Table selector (mouse-index and Key_Down), the
detector combo, R through ``QTest.keyClick`` on the inspector in each host and on the gallery's
tile view, a Results row click, Edit > Undo as the menu offers it. The data is synthetic: ten
events of the same recording stored as a Schmitt table (all ten) and a CUSUM table that keeps
only the long five, so an event id names a DIFFERENT event in the two tables and no coincidence
can hide the defect.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, reviewed_mask
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, private_settings, settle)

FS = 50_000.0
I0 = -1_000.0
S, C = "detect.schmitt", "detect.cusum"
NAMES = {S: "Schmitt", C: "CUSUM"}
#: Ten events, alternately short (400-480 samples, 8-10 ms) and long (820-1000 samples,
#: 16-20 ms), every one a different length. A minimum dwell of 10 ms keeps the long five.
_SPANS = []
_pos = 6_000
for _i in range(10):
    _len = 400 + 10 * _i if _i % 2 == 0 else 800 + 20 * _i
    _SPANS.append((_pos, _pos + _len))
    _pos += 10_000
N = _SPANS[-1][1] + 6_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(8)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 7.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


# ================================================================================================
# the rig
# ================================================================================================

@dataclass
class Rig:
    window: object
    doc_id: str

    @property
    def panel(self):
        return self.window._analysis_panel

    def frame(self, module_id: str) -> pd.DataFrame:
        return self.window._session.results.get(self.doc_id, 0, module_id).payload.df

    def show(self, module_id: str) -> None:
        """What the user does: pick the table in the Results dock's selector."""
        sel = self.panel._result_selector
        sel.setCurrentIndex(sel.findData(module_id))
        settle()
        assert self.panel.current_detection().module_id == module_id

    def shown(self) -> str:
        return self.panel.current_detection().module_id

    def open_inspector(self, host: str, *, on: str | None = None):
        """Analysis > Event inspector, in ``host``, fed with the table on screen (or with
        ``on``, which is put on screen first)."""
        if on is not None:
            self.show(on)
        self.window.set_curation_host("event_inspector", host, present=False)
        self.window._open_event_inspector()
        settle()
        insp = self.window._event_inspector
        insp.set_advance_after_decision(False)
        assert insp._df is self.frame(self.shown())
        return insp

    def snapshots(self) -> dict[str, pd.DataFrame]:
        return {mid: self.frame(mid).copy(deep=True) for mid in (S, C)}

    def assert_untouched(self, module_id: str, before: dict[str, pd.DataFrame]) -> None:
        """Not one accept flag, review stamp or column differs from the snapshot."""
        after = self.frame(module_id)
        moved = _changed_event_ids(after, before[module_id])
        assert moved == [], f"the {NAMES[module_id]} table was written: events {moved} changed"
        pd.testing.assert_frame_equal(after, before[module_id])

    def name_of(self, frame) -> str:
        for module_id in (S, C):
            if frame is self.frame(module_id):
                return f"the {NAMES[module_id]} table ({len(frame)} rows)"
        return "no table" if frame is None else "a table that is no longer stored"

    def assert_holds(self, surface, module_id: str) -> None:
        """The surface holds the very frame of ``module_id``'s stored table -- the table the
        Results dock is showing -- and says so in the failure when it does not."""
        held = surface._df
        who = "Event inspector" if "Inspector" in type(surface).__name__ else "Event gallery"
        assert held is self.frame(module_id), (
            f"the {who} holds {self.name_of(held)}; the Results dock shows "
            f"{self.name_of(self.frame(module_id))}")

    def undo_text(self) -> str:
        """Edit > Undo as the menu offers it (the Edit menu opening commits and re-labels)."""
        self.window._on_edit_menu_about_to_show()
        return self.window._act_undo.text()

    def summary(self) -> str:
        return self.panel._results_summary.text()


@pytest.fixture
def make_rig(private_settings, qtbot, tmp_path):
    made = []

    def make(*, with_cusum: bool = True) -> Rig:
        window = build_window(qtbot, detection=False)
        data = _signal()
        info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
        meta = RecordingMeta(path=tmp_path / f"xs1_{len(made)}.dat", format_name="synthetic",
                             n_samples=N, sample_rate_hz=FS,
                             rate_provenance=Provenance.FILE_META, channels=(info,))
        doc = TraceDocument(meta=meta, channels=(data,))
        doc_id = window._session.add_document(doc)
        window._create_default_layers(doc)
        window._on_selection(doc_id, 0)
        cal = window._session.calibration(doc_id, 0).fingerprint()
        tables = [(S, 1e-4)] + ([(C, 1e-2)] if with_cusum else [])
        for module_id, min_dwell in tables:
            frame, prov = detect_events(data, FS, baseline=I0, min_dwell_s=min_dwell)
            window._session.add_result(AnalysisResult(
                module_id=module_id, module_version="1.0", doc_id=doc_id, channel=0,
                params={"min_dwell_s": min_dwell}, calibration_fingerprint=cal,
                payload=EventSet(frame, prov)))
            settle()
        rig = Rig(window, doc_id)
        if with_cusum:
            assert (len(rig.frame(S)), len(rig.frame(C))) == (10, 5)
            rig.show(C)               # every frame is displayed once before any snapshot, so
        rig.show(S)                   # the display-time columns are in place on both
        made.append(window)
        return rig

    yield make
    for window in made:
        close_window(window)


@pytest.fixture
def rig(make_rig) -> Rig:
    return make_rig()


def _press(widget, key=Qt.Key_R) -> None:
    QTest.keyClick(widget, key)
    settle()


def _row(frame: pd.DataFrame, event_id: int) -> pd.Series:
    return frame.loc[frame["event_id"] == event_id].iloc[0]


def _is_reviewed(frame: pd.DataFrame, event_id: int) -> bool:
    return bool(reviewed_mask(frame)[(frame["event_id"] == event_id).to_numpy()][0])


def _changed_event_ids(after: pd.DataFrame, before: pd.DataFrame) -> list[int]:
    """The event ids whose accepted flag or review stamp differ between two frames."""
    assert list(after["event_id"]) == list(before["event_id"])
    moved = ((after["accepted"].to_numpy() != before["accepted"].to_numpy())
             | (reviewed_mask(after) != reviewed_mask(before)))
    return [int(e) for e in after["event_id"].to_numpy()[moved]]


def _click_results_row(rig: Rig, event_id: int) -> None:
    """A mouse click on the Results table's row for ``event_id`` (column 2: a click that
    selects a row and cannot land on the accepted checkbox)."""
    table = rig.panel._table
    df = table._model.df
    src = int(np.nonzero(df["event_id"].to_numpy() == event_id)[0][0])
    idx = table._proxy.mapFromSource(table._model.index(src, 2))
    table.view.scrollTo(idx)
    QTest.mouseClick(table.view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     table.view.visualRect(idx).center())
    settle()


# ================================================================================================
# 1. the inspector follows the table the dock shows
# ================================================================================================

@pytest.mark.parametrize("host", ["dock", "window"])
def test_the_inspector_follows_the_table_selector_in_both_hosts(rig, host):
    insp = rig.open_inspector(host)
    assert re.search(r"Reviewed 0 / 10\b", insp.progress_text()), insp.progress_text()
    rig.show(C)
    assert rig.summary().startswith("5 events"), rig.summary()
    assert re.search(r"Reviewed 0 / 5\b", insp.progress_text()), (
        "the dock says CUSUM, the inspector still counts Schmitt: " + insp.progress_text())
    assert insp.event_count() == 5
    rig.assert_holds(insp, C)
    assert insp._module_id == C
    rig.show(S)
    assert re.search(r"Reviewed 0 / 10\b", insp.progress_text()), insp.progress_text()
    rig.assert_holds(insp, S)
    assert insp.event_count() == 10


def test_a_keyboard_pick_on_the_table_combo_is_the_same_door(rig):
    insp = rig.open_inspector("dock")
    sel = rig.panel._result_selector
    sel.setFocus()
    _press(sel, Qt.Key_Down)
    assert rig.shown() == C
    rig.assert_holds(insp, C)
    _press(sel, Qt.Key_Up)
    assert rig.shown() == S
    rig.assert_holds(insp, S)


@pytest.mark.parametrize("host", ["dock", "window"])
def test_the_inspector_is_refed_with_its_filter_and_place_reset_and_the_trace_left_alone(
        rig, host):
    """The rule for a table the surface was not holding, stated once: the one ``set_events``
    applies to any different detection. Filter back to All, the first event, and NOTHING
    announced -- a pure view switch moves neither the main trace nor the Results selection.
    (There is no per-table memory: going back to the first table starts it from the top.)"""
    insp = rig.open_inspector(host)
    insp.show_event(5)
    insp.set_accepted_current(False)
    insp.set_filter("Accepted only")
    settle()
    assert insp.current_filter() == "Accepted only"
    announced: list[int] = []
    insp.eventSelected.connect(announced.append)
    view_before = rig.window._trace_view.visible_x_range_for(rig.doc_id, 0)
    rig.show(C)
    assert insp.current_filter() == "All"
    assert insp.current_event_id() == int(rig.frame(C)["event_id"].iloc[0])
    assert rig.panel.selected_event_id() is None
    assert announced == [], "re-feeding a table must not yank the trace or the dock's selection"
    assert rig.window._trace_view.visible_x_range_for(rig.doc_id, 0) == view_before
    rig.show(S)
    assert insp.current_filter() == "All"
    assert insp.current_event_id() == int(rig.frame(S)["event_id"].iloc[0])


def test_a_hidden_inspector_dock_is_refed_when_it_is_shown_again(rig):
    """The path that was already sound (a surface that is not on screen is re-fed on show);
    pinned so the new re-feed cannot replace it with something that skips it."""
    insp = rig.open_inspector("dock")
    dock = rig.window._inspector_dock
    dock.hide()
    settle()
    rig.show(C)
    assert insp._df is rig.frame(S), "a hidden surface is not re-fed (the opener does it)"
    dock.show()
    settle()
    rig.assert_holds(insp, C)


# ================================================================================================
# 2. one R keypress writes one table, as one undo step
# ================================================================================================

@pytest.mark.parametrize("host", ["dock", "window"])
@pytest.mark.parametrize("fed_on, shown", [(S, C), (C, S)])
def test_one_R_keypress_curates_exactly_the_table_on_screen(rig, host, fed_on, shown):
    """The finder's repro and its mirror (V5), in both hosts. The inspector is fed while
    ``fed_on`` is on screen, the user moves the selector to ``shown`` and presses R on the
    event the inspector is looking at."""
    insp = rig.open_inspector(host, on=fed_on)
    rig.show(shown)
    other = fed_on
    before = rig.snapshots()
    eid = insp.current_event_id()
    assert eid is not None and eid in set(rig.frame(shown)["event_id"])
    n_shown = len(rig.frame(shown))
    insp.focus_curation_surface()
    _press(insp, Qt.Key_R)

    # the table on screen took the decision -- and only that event of it
    assert _changed_event_ids(rig.frame(shown), before[shown]) == [eid]
    assert not bool(_row(rig.frame(shown), eid)["accepted"])
    assert _is_reviewed(rig.frame(shown), eid)
    assert rig.summary() == f"{n_shown} events · {n_shown - 1} accepted", rig.summary()
    assert re.search(rf"Reviewed 1 / {n_shown}\b.*accepted {n_shown - 1}\b",
                     insp.progress_text()), insp.progress_text()
    # ...and the table that is NOT on screen was not touched at all (no rejection, no
    # "reviewed" stamp, not even a new column)
    rig.assert_untouched(other, before)

    # Edit > Undo: ONE step, which puts back the only thing that moved
    assert rig.undo_text() == "Undo Curate event"
    rig.window._act_undo.trigger()
    settle()
    rig.assert_untouched(shown, before)
    rig.assert_untouched(other, before)
    text = rig.undo_text()
    assert not rig.window._act_undo.isEnabled(), text
    assert "nothing on this table" not in text, (
        "the menu says another table still holds history: " + text)


def test_a_keyboard_pick_then_R_then_undo_leaves_nothing_behind_on_either_table(rig):
    """V1/V3: the selector moved with the keyboard, R, Edit > Undo. Whatever a project save
    would then carry (a 'reviewed' rejection, a flag) is back to what was stored."""
    insp = rig.open_inspector("dock", on=S)
    sel = rig.panel._result_selector
    sel.setFocus()
    _press(sel, Qt.Key_Down)
    assert rig.shown() == C
    before = rig.snapshots()
    insp.focus_curation_surface()
    _press(insp, Qt.Key_R)
    assert rig.undo_text() == "Undo Curate event"
    rig.window._act_undo.trigger()
    settle()
    for module_id in (S, C):
        rig.assert_untouched(module_id, before)
        assert not reviewed_mask(rig.frame(module_id)).any()
        assert rig.frame(module_id)["accepted"].all()


@pytest.mark.parametrize("host", ["dock", "window"])
def test_a_results_row_click_after_the_switch_shows_that_tables_event(rig, host):
    """'Sync by id, not by event': the row of CUSUM event 1 is not Schmitt's event 1."""
    insp = rig.open_inspector(host, on=S)
    rig.show(C)
    eid = 1
    assert _row(rig.frame(S), eid)["start_idx"] != _row(rig.frame(C), eid)["start_idx"]
    _click_results_row(rig, eid)
    assert rig.panel.selected_event_id() == eid
    dwell_us = int(round(float(_row(rig.frame(C), eid)["dwell_s"]) * 1e6))
    other_us = int(round(float(_row(rig.frame(S), eid)["dwell_s"]) * 1e6))
    assert dwell_us != other_us
    readout = insp._readout.text()
    assert f"{dwell_us} µs" in readout, (
        f"the row clicked is CUSUM's event {eid} ({dwell_us} µs); "
        "the inspector shows:\n" + readout)
    assert f"{other_us} µs" not in readout, readout
    assert "of 5" in readout, readout
    rig.assert_holds(insp, C)
    assert insp.current_event_id() == eid


# ================================================================================================
# 3. the detector combo is the second door
# ================================================================================================

def test_the_detector_combo_moves_the_inspector_too(rig):
    insp = rig.open_inspector("dock", on=S)
    rig.panel.detection_panel.set_params({"detector": C})
    settle()
    assert rig.shown() == C
    rig.assert_holds(insp, C)
    before = rig.snapshots()
    eid = insp.current_event_id()
    insp.focus_curation_surface()
    _press(insp, Qt.Key_R)
    assert _changed_event_ids(rig.frame(C), before[C]) == [eid]
    rig.assert_untouched(S, before)


def test_a_detector_with_no_stored_run_empties_the_surfaces_and_nothing_can_be_written(
        make_rig):
    rig = make_rig(with_cusum=False)
    insp = rig.open_inspector("dock", on=S)
    rig.window._open_event_gallery()
    settle()
    gal = rig.window._event_gallery
    before = rig.frame(S).copy(deep=True)
    rig.panel.detection_panel.set_params({"detector": C})
    settle()
    assert rig.panel.current_detection() is None
    assert insp._df is None and insp.current_event_id() is None
    assert gal.frame() is None
    insp.focus_curation_surface()
    _press(insp, Qt.Key_R)
    pd.testing.assert_frame_equal(rig.frame(S), before)


# ================================================================================================
# 4. the gallery
# ================================================================================================

def _gallery_on(rig: Rig):
    rig.window._open_event_gallery()
    settle()
    gal = rig.window._event_gallery
    gal.set_advance_after_decision(False)
    return gal


def test_the_gallery_follows_the_table_selector(rig):
    gal = _gallery_on(rig)
    rig.assert_holds(gal, S)
    assert gal.event_count() == 10
    rig.show(C)
    assert re.search(r"Reviewed 0 / 5\b", gal.progress_text()), (
        "the dock says CUSUM, the gallery still counts Schmitt: " + gal.progress_text())
    rig.assert_holds(gal, C)
    assert gal.event_count() == 5
    rig.show(S)
    rig.assert_holds(gal, S)
    assert gal.event_count() == 10


def test_a_results_row_click_after_the_switch_moves_the_gallery_to_that_tables_tile(rig):
    gal = _gallery_on(rig)
    rig.show(C)
    eid = 1
    _click_results_row(rig, eid)
    rig.assert_holds(gal, C)
    assert gal.selected_event_ids() == [eid]
    assert gal.current_event_id() == eid


@pytest.mark.parametrize("fed_on, shown", [(S, C), (C, S)])
def test_R_on_a_gallery_tile_after_the_switch_curates_exactly_one_table(rig, fed_on, shown):
    """V6 and its mirror: the gallery open on ``fed_on`` with a tile selected, the selector
    moved to ``shown``, R on the tile that is selected then."""
    rig.show(fed_on)
    gal = _gallery_on(rig)
    gal.show_event(2)
    settle()
    assert gal.selected_event_ids() == [2]
    rig.show(shown)
    before = rig.snapshots()
    selected = gal.selected_event_ids()
    assert len(selected) == 1, selected
    eid = selected[0]
    assert eid in set(rig.frame(shown)["event_id"])
    gal.activateWindow()
    gal.view.setFocus()
    _press(gal.view, Qt.Key_R)
    assert _changed_event_ids(rig.frame(shown), before[shown]) == [eid]
    rig.assert_untouched(fed_on, before)
    assert rig.undo_text() == "Undo Curate event"
    rig.window._act_undo.trigger()
    settle()
    rig.assert_untouched(shown, before)
    rig.assert_untouched(fed_on, before)
    assert not rig.window._act_undo.isEnabled()


def test_a_multi_tile_reject_after_the_switch_writes_the_table_on_screen_only(rig):
    gal = _gallery_on(rig)
    rig.show(C)
    before = rig.snapshots()
    gal.view.selectAll()
    settle()
    assert len(gal.selected_event_ids()) == 5
    gal.set_accepted_selected(False)
    settle()
    assert _changed_event_ids(rig.frame(C), before[C]) == [0, 1, 2, 3, 4]
    rig.assert_untouched(S, before)


def test_the_gallery_keeps_its_show_filter_across_a_switch_and_applies_it_to_the_new_table(rig):
    """Its Show box is a standing choice about how to look, not part of one table's place
    (the inspector's own filter, a browsing position, goes back to All on a different table):
    a filter the user set is a decision, and a switch does not take it back."""
    rig.panel.set_event_accepted(3, False)               # one rejected event in Schmitt
    rig.show(C)
    rig.panel.set_event_accepted(2, False)               # ...and one in CUSUM
    rig.show(S)
    gal = _gallery_on(rig)
    gal.filter_box.setCurrentText("Accepted only")
    settle()
    assert gal.filter_box.currentText() == "Accepted only" and gal.event_count() == 9
    rig.show(C)                                          # nothing is written from here on
    rig.assert_holds(gal, C)
    assert gal.filter_box.currentText() == "Accepted only"
    assert gal.event_count() == 4, "the filter was dropped (5) or not applied to the new table"


def test_a_switch_with_both_surfaces_open_moves_neither_the_trace_nor_the_selection(rig):
    """The rule for the gallery too: a different frame starts on its first tile, but the
    tile is not announced -- a view switch must not jump the main trace (the changelog's
    rule for a hand-add or a merge with the gallery open)."""
    insp = rig.open_inspector("dock", on=S)
    gal = _gallery_on(rig)
    rig.panel.select_event(3)
    settle()
    announced: list[tuple[str, int]] = []
    insp.eventSelected.connect(lambda e: announced.append(("inspector", e)))
    gal.eventSelected.connect(lambda e: announced.append(("gallery", e)))
    trace_before = rig.window._trace_view.visible_x_range_for(rig.doc_id, 0)
    rig.window.statusBar().clearMessage()
    rig.show(C)
    rig.assert_holds(insp, C)
    rig.assert_holds(gal, C)
    assert announced == [], announced
    assert rig.window._trace_view.visible_x_range_for(rig.doc_id, 0) == trace_before
    assert rig.panel.selected_event_id() is None
    assert "selected" not in rig.window.statusBar().currentMessage()
    # the gallery still starts on a tile, so its keys have something to act on
    assert len(gal.selected_event_ids()) == 1


# ================================================================================================
# 5. a surface that is somehow stale still cannot write the other table
# ================================================================================================

@pytest.mark.parametrize("surface", ["inspector", "gallery"])
def test_a_stale_surface_cannot_have_its_decision_mirrored_onto_the_other_table(rig, surface):
    """Defence in depth. Re-feeding on every switch is the fix; this pins the property that
    does not depend on every present and future door remembering to: a surface holding a
    frame that is not the table the dock shows has its decision applied to ITS OWN table and
    to no other -- the relay by event id is refused, and the surface is brought back in step."""
    rig.show(S)
    if surface == "inspector":
        win_surface = rig.open_inspector("dock")
    else:
        win_surface = _gallery_on(rig)
    # a door that swaps the dock's table without announcing it: the surface is left behind
    rig.panel._set_active_detection(C)
    rig.panel._show_result(rig.window._session.results.get(rig.doc_id, 0, C))
    settle()
    assert rig.shown() == C
    stale_frame = win_surface._df if surface == "inspector" else win_surface.frame()
    assert stale_frame is rig.frame(S), "setup: the surface must be the stale one"
    before = rig.snapshots()
    if surface == "inspector":
        eid = win_surface.current_event_id()
        win_surface.focus_curation_surface()
        _press(win_surface, Qt.Key_R)
    else:
        win_surface.show_event(2)
        settle()
        eid = win_surface.selected_event_ids()[0]
        win_surface.activateWindow()
        win_surface.view.setFocus()
        _press(win_surface.view, Qt.Key_R)
    rig.assert_untouched(C, before)
    assert _changed_event_ids(rig.frame(S), before[S]) == [eid]
    status = rig.window.statusBar().currentMessage()
    assert "Results" in status and "Schmitt" in status and "CUSUM" in status, status
    # ...and the surface was brought back in step with the dock
    rig.assert_holds(win_surface, C)


def test_emitting_the_switch_announcement_refeeds_every_surface_on_screen(rig):
    """The wiring itself: ``resultSwitched`` is what re-feeds, not the selector's index."""
    insp = rig.open_inspector("window", on=S)
    gal = _gallery_on(rig)
    rig.panel._set_active_detection(C)
    rig.panel._show_result(rig.window._session.results.get(rig.doc_id, 0, C))
    rig.assert_holds(insp, S)
    rig.assert_holds(gal, S)
    rig.panel.resultSwitched.emit()
    settle()
    rig.assert_holds(insp, C)
    rig.assert_holds(gal, C)


# ================================================================================================
# 6. what the re-feed must not drag along
# ================================================================================================

def test_a_level_fit_belongs_to_the_table_it_was_measured_on(rig, qtbot):
    """Re-feeding the inspector with another table of the same recording must not print the
    sub-state levels fitted on the first one under the second one's events (matched by event
    id, which means nothing across tables). The fit stays parked beside its own table and
    comes back with it."""
    window = rig.window
    insp = rig.open_inspector("window", on=S)
    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=20_000)
    qtbot.waitUntil(lambda: window._level_fit is not None, timeout=20_000)
    settle()
    fit = window._level_fit
    assert fit is not None and not fit.payload.empty
    assert insp._levels is not None, "setup: the inspector shows the levels on their table"
    rig.show(C)
    rig.assert_holds(insp, C)
    assert insp._levels is None, "Schmitt's levels are being printed under CUSUM's events"
    assert window._level_fit is fit, "a view switch must not touch the fit"
    assert window._session.results.get(rig.doc_id, 0, "fit.levels") is not None
    rig.show(S)
    rig.assert_holds(insp, S)
    assert insp._levels is not None and len(insp._levels) == len(fit.payload)


class _Report:
    """What ``MainWindow._adopt_subpop_assignment`` reads off a finished subpopulation report."""

    meta = {"recording": "xs1"}
    warnings = ()

    def __init__(self, ids):
        self._table = pd.DataFrame({"event_id": list(ids),
                                    "subpop": [i % 2 for i in ids]})
        self.n_fired = len(self._table)

    def event_table(self):
        return self._table

    def headline(self):
        return "2 subpopulations"


def test_the_subpopulation_colouring_rides_the_gallery_across_a_round_trip(rig):
    """The gallery drops its map when it is handed a different frame; the re-feed has to
    withdraw it while another table is on screen and hand it back with its own table."""
    window = rig.window
    gal = _gallery_on(rig)
    target = window._session.results.get(rig.doc_id, 0, S)
    ids = [int(e) for e in rig.frame(S)["event_id"]]
    assert window._adopt_subpop_assignment(_Report(ids), target) == ""
    assert gal.subpopulations() is not None, "setup: the gallery carries the map"
    rig.show(C)
    rig.assert_holds(gal, C)
    assert gal.subpopulations() is None, "a map fitted on Schmitt is colouring CUSUM's tiles"
    rig.show(S)
    rig.assert_holds(gal, S)
    assert gal.subpopulations() is not None, "the map did not come back with its table"
