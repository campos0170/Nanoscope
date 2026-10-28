"""S8-DATA1 -- a table the amber line calls stale is not edited in place.

MEASURED DEFECT (sweep 8, a 2 s, 50 kHz slice of the -600 mV recording, detected at 50 kHz and
then declared to be 40 kHz through Inspector > File Info > Sample rate / time unit...). The
declaration marks the ten detections stale -- "every start time, dwell, rise/fall time and ECD
in this table was computed at the old rate" -- and the documented contract is that stored
detections keep the rate they were measured at and are never silently re-scaled. Three
in-place doors did not honour it. They read the LIVE rate and wrote ``start_idx / fs`` and
``(end - start) / fs`` into the table whose provenance still said fs = 50 kHz: Shift+Cmd+M
(``add_current_selection_as_event``: a hand-added event of 300 samples got dwell_s = 7.5 ms),
a span edge dragged in the Event inspector (``_apply_edited_bounds`` -> ``_recharacterize_row``),
and Merge all (``merge_by_gap`` on ``_active_signal``). After the three the table held four
old-clock rows and four new-clock ones, with the amber line still up above them. "Merge events
in selection" and the inspector's "Merge with next" are the same write by another button, and
Compare baseline estimators -- the one door that re-measures -- DID refuse the stale table, but
as a raw ``ValueError`` modal (S8-DATA4) addressed to a caller of a Python function.

THE FIX. ``analysis.events.stale_rate_hz`` / ``stale_rate_reason`` are the one rule and the one
sentence (the amber line's own); ``AnalysisPanel._sample_rate_staleness`` -- the question the
banner asks -- and every door ask there, so a door can neither refuse where the banner offers no
Re-run nor edit where the banner is up. Hand-add, the three merges and the re-measure entry
points refuse BEFORE anything is read or written ("Nothing was added." / "Nothing was merged." /
"Nothing was re-measured." + the amber line's sentence, on the status bar); the inspector's
single write path for bounds refuses a drag or a Revert, puts the handles back and says it
through the new ``editRefused`` signal. No row, no stamp, no undo step moves. A table that is not
stale -- the declaration taken back, a fresh run under the declared rate, a table that carries no
stamp -- edits exactly as before.

These tests drive the real ``MainWindow``: the declaration through ``SampleRateDialog``, the
Analysis menu action behind Shift+Cmd+M and the Measure tab's button, a span edge and Revert in
the Event inspector (docked and in its own window), the Results dock's Merge all / Merge events
in selection buttons, the inspector's Merge with next, Analysis > Compare baseline estimators, the
amber line's own Re-run, and a project saved and reopened. The data is synthetic: three isolated
events, one three-fragment translocation and one short blockade the detector drops (the one the
user marks by hand).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.sample_rate_dialog import SampleRateDialog

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, private_settings, settle)

FS = 50_000.0
DECLARED = 40_000.0
N = 100_000
BASELINE, DEPTH = 100.0, 30.0
_ISOLATED = [(6_000, 6_400), (30_000, 30_300), (60_000, 60_450)]
#: One translocation the detector split into three, 20 samples (0.4 ms) apart.
_FRAGMENTS = [(12_000, 12_200), (12_220, 12_500), (12_520, 13_000)]
#: 1.2 ms: under the 2 ms minimum dwell, so the detector drops it -- the user marks it.
_MISSED = (45_000, 45_060)
_MIN_DWELL_S = 2e-3
_MERGE_WINDOW_S = 5e-3
_MODAL = []            # every QMessageBox this module's rigs opened: (title, text)


def _signal() -> np.ndarray:
    rng = np.random.default_rng(8)
    x = np.full(N, BASELINE) + rng.normal(0.0, 1.5, N)
    for start, end in _ISOLATED + _FRAGMENTS + [_MISSED]:
        x[start:end] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


def _document(path) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(_signal(),))


# ================================================================================================
# the rig
# ================================================================================================

@dataclass
class Rig:
    window: object
    doc_id: str
    qtbot: object

    @property
    def panel(self):
        return self.window._analysis_panel

    def df(self) -> pd.DataFrame:
        return self.panel.current_detection().payload.df

    def snapshot(self) -> pd.DataFrame:
        return self.df().copy(deep=True)

    def stamp(self) -> float:
        return float(self.panel.current_detection().payload.provenance["fs"])

    def status(self) -> str:
        return self.window.statusBar().currentMessage()

    def undo_text(self) -> str:
        """Edit > Undo as the menu offers it (opening it commits any unannounced edit)."""
        self.window._on_edit_menu_about_to_show()
        return self.window._act_undo.text()

    def _dialog(self, mode: str, hz: float | None = None) -> None:
        dlg = SampleRateDialog(self.window._session, self.doc_id, self.window._file_info_panel)
        dlg.show()
        dlg.mode.setCurrentIndex(dlg.mode.findData(mode))
        if hz is not None:
            dlg.rate.setText(format(hz, ".17g"))
        dlg.apply()
        dlg.close()
        dlg.deleteLater()
        settle(50)

    def declare(self, hz: float = DECLARED) -> None:
        """Inspector > File Info > Sample rate / time unit... : declare the rate, Apply."""
        self._dialog("sample_rate", hz)

    def take_the_declaration_back(self) -> None:
        """...and "Use the file's own time base": the rates match again."""
        self._dialog("file")

    def open_inspector(self, host: str = "window"):
        self.window.set_curation_host("event_inspector", host, present=True)
        settle()
        return self.window._event_inspector

    def bracket(self, span: tuple[int, int]) -> None:
        """Turn the Measure cursors on and drop them on ``span`` (sample indices)."""
        self.panel.measure_panel.set_active(True)
        vals = self.window._session.x_axis(self.doc_id).values
        self.window._trace_view.set_measure_region(float(vals[span[0]]), float(vals[span[1]]))
        settle()
        assert tuple(int(i) for i in self.window._trace_view.measure_bounds()) == span

    def drag_edge(self, insp, event_id: int, by: int = 25) -> None:
        """The trailing edge of the shaded span dragged ``by`` samples outwards; the gesture
        ends on the region's own ``sigRegionChangeFinished``."""
        df = self.df()
        row = df.index[df["event_id"] == event_id][0]
        start, end = int(df.at[row, "start_idx"]), int(df.at[row, "end_idx"])
        xs = self.window._session.x_axis(self.doc_id).values
        insp.show_event(event_id)
        settle()
        insp._region.setRegion((float(xs[start]), float(xs[end + by])))
        insp._region.sigRegionChangeFinished.emit(insp._region)
        settle()

    def merge_all(self) -> None:
        """Press Merge all, and wait for whatever it started to land."""
        self.panel.merge_controls.set_window_s(_MERGE_WINDOW_S)
        self.panel.merge_controls.merge_all_btn.click()
        self.qtbot.waitUntil(
            lambda: self.panel._pending is None and self.panel._merge_job is None, timeout=30_000)
        settle()

    def ids_by_start(self) -> list[int]:
        df = self.df().sort_values("start_idx")
        return [int(i) for i in df["event_id"]]


def rows_on_clock(df: pd.DataFrame, fs: float) -> int:
    """Rows whose dwell_s and t_start_s are the sample counts divided by ``fs``."""
    n = (df["end_idx"] - df["start_idx"]).to_numpy(float)
    dwell = np.isclose(df["dwell_s"].to_numpy(float) * fs, n, rtol=1e-6)
    start = np.isclose(df["t_start_s"].to_numpy(float) * fs,
                       df["start_idx"].to_numpy(float), rtol=1e-6, atol=1e-6)
    return int((dwell & start).sum())


@pytest.fixture
def rig(private_settings, qtbot, tmp_path, monkeypatch):
    """A recording detected at 50 kHz through the real Detect, not yet declared."""
    _MODAL.clear()
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec",
        lambda self: _MODAL.append((self.windowTitle(), self.text())) or 0)
    window = build_window(qtbot, detection=False)
    doc = _document(tmp_path / "data1.dat")
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(_MIN_DWELL_S)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    settle(80)
    made = Rig(window, doc_id, qtbot)
    assert len(made.df()) == 6, made.df()[["event_id", "start_idx", "end_idx"]]
    assert made.stamp() == FS and rows_on_clock(made.df(), FS) == 6
    yield made
    close_window(window)


@pytest.fixture
def stale(rig) -> Rig:
    """The rig after the declaration: the amber line is up over a table entirely on 50 kHz."""
    rig.declare()
    assert rig.window._session.document(rig.doc_id).meta.sample_rate_hz == DECLARED
    assert not rig.panel._stale.isHidden() and rig.panel._stale.kind == "sample_rate"
    assert rig.stamp() == FS
    return rig


def assert_refused(rig: Rig, before: pd.DataFrame, undo_before: str, outcome: str) -> None:
    """The outcome of every refusal, whichever door it came through: said on the status bar in
    the amber line's own words, with the way forward; the table, its stamp and the undo menu
    exactly as they were; the amber line still up."""
    assert _MODAL == [], f"no modal, least of all a Python exception: {_MODAL}"
    pd.testing.assert_frame_equal(rig.df(), before)
    assert rows_on_clock(rig.df(), FS) == len(before), "one clock: the table's own"
    assert rig.stamp() == FS
    assert rig.undo_text() == undo_before, "a refusal is not an undo step"
    assert not rig.panel._stale.isHidden()
    said = rig.status()
    assert said.startswith(outcome), said
    assert rig.panel._stale.message() in said, said
    assert "Re-run detection under the declared rate" in said


# ================================================================================================
# 1. hand-add (Shift+Cmd+M, the Measure tab's button)
# ================================================================================================

def test_the_shortcut_refuses_to_hand_add_an_event_into_a_stale_table(stale):
    stale.bracket(_MISSED)
    before, undo = stale.snapshot(), stale.undo_text()
    assert stale.window._act_mark_missed.shortcut().toString() == "Ctrl+Shift+M"   # Shift+Cmd+M
    assert stale.window._act_mark_missed.isEnabled()
    stale.window._act_mark_missed.trigger()
    settle()
    assert_refused(stale, before, undo, "Nothing was added.")


def test_the_measure_tabs_button_refuses_the_same_way(stale):
    stale.bracket(_MISSED)
    before, undo = stale.snapshot(), stale.undo_text()
    button = stale.panel.measure_panel._mark_missed_btn
    assert button.isEnabled()
    button.click()
    settle()
    assert_refused(stale, before, undo, "Nothing was added.")
    assert stale.panel.last_missed_refusal.startswith("Nothing was added.")


# ================================================================================================
# 2. the Event inspector's span edge and Revert
# ================================================================================================

@pytest.mark.parametrize("host", ["window", "dock"])
def test_an_edge_drag_in_the_inspector_is_refused_and_the_handles_go_back(stale, host):
    insp = stale.open_inspector(host)
    eid = stale.ids_by_start()[4]
    insp.show_event(eid)
    settle()
    handles = insp._region.getRegion()
    edited = []
    insp.eventEdited.connect(edited.append)
    before, undo = stale.snapshot(), stale.undo_text()
    stale.drag_edge(insp, eid)
    assert_refused(stale, before, undo, "The event's bounds were not changed.")
    assert edited == [], "no edit was announced"
    assert insp._region.getRegion() == pytest.approx(handles), "the handles went back"
    # the window itself carries the reason too (a floating inspector can cover the status bar)
    assert insp.stale_note() == stale.panel._stale.message()


def test_revert_to_the_detectors_bounds_is_refused_too(rig):
    """Revert re-measures through the same writer as a drag. A drag made while the table was
    current stands; once the rate is declared, Revert cannot rewrite the row on the new clock."""
    insp = rig.open_inspector()
    eid = rig.ids_by_start()[4]
    rig.drag_edge(insp, eid)
    assert rows_on_clock(rig.df(), FS) == 6 and insp._revert_btn.isEnabled()
    rig.declare()
    before, undo = rig.snapshot(), rig.undo_text()
    insp._revert_btn.click()
    settle()
    assert_refused(rig, before, undo, "The event's bounds were not changed.")


# ================================================================================================
# 3. the three merges
# ================================================================================================

def test_merge_all_is_refused_before_any_job_starts(stale):
    before, undo = stale.snapshot(), stale.undo_text()
    stale.merge_all()
    assert stale.panel._merge_job is None
    assert_refused(stale, before, undo, "Nothing was merged.")


def test_merge_events_in_selection_is_refused(stale):
    stale.bracket((11_500, 13_500))
    before, undo = stale.snapshot(), stale.undo_text()
    stale.panel.merge_controls.merge_region_btn.click()
    settle()
    assert_refused(stale, before, undo, "Nothing was merged.")


def test_the_inspectors_merge_with_next_is_refused(stale):
    insp = stale.open_inspector()
    insp.show_event(stale.ids_by_start()[1])
    settle()
    stale.panel.merge_controls.set_window_s(_MERGE_WINDOW_S)
    before, undo = stale.snapshot(), stale.undo_text()
    insp._merge_next_btn.click()
    settle()
    assert_refused(stale, before, undo, "Nothing was merged.")
    assert insp.current_event_id() == stale.ids_by_start()[1], "the browser did not advance"


# ================================================================================================
# 4. the door that re-measures says it in the same voice (S8-DATA4)
# ================================================================================================

def test_compare_baseline_estimators_says_why_instead_of_opening_a_python_exception(stale):
    before, undo = stale.snapshot(), stale.undo_text()
    assert stale.window._act_compare_baseline.isEnabled()
    stale.window._act_compare_baseline.trigger()
    stale.qtbot.waitUntil(lambda: stale.panel._pending is None, timeout=30_000)
    settle()
    assert_refused(stale, before, undo, "Nothing was re-measured.")


def test_re_measure_depths_refuses_with_the_same_token_and_sentence(stale):
    before, undo = stale.snapshot(), stale.undo_text()
    assert stale.panel.remeasure_active_detection() == "stale_rate"
    stale.qtbot.waitUntil(lambda: stale.panel._pending is None, timeout=30_000)
    settle()
    assert_refused(stale, before, undo, "Nothing was re-measured.")


# ================================================================================================
# 5. a table that is not stale edits exactly as before
# ================================================================================================

def test_with_the_declaration_taken_back_hand_add_writes_a_row_on_the_tables_clock(stale):
    stale.take_the_declaration_back()
    assert stale.panel._stale.isHidden(), "the rates match again: nothing is stale"
    stale.bracket(_MISSED)
    stale.window._act_mark_missed.trigger()
    settle()
    df = stale.df()
    assert len(df) == 7 and rows_on_clock(df, FS) == 7
    assert "Added missed event" in stale.status() and _MODAL == []


def test_with_the_declaration_taken_back_a_drag_and_merge_all_edit_as_before(stale):
    stale.take_the_declaration_back()
    insp = stale.open_inspector()
    eid = stale.ids_by_start()[4]
    edited = []
    insp.eventEdited.connect(edited.append)
    stale.drag_edge(insp, eid)
    assert edited == [eid]
    row = stale.df().set_index("event_id").loc[eid]
    assert int(row["end_idx"]) - int(row["start_idx"]) == 325
    stale.merge_all()
    df = stale.df()
    assert len(df) == 4 and rows_on_clock(df, FS) == 4
    assert stale.status().startswith("Merged 2 fragment(s): 6 → 4 events.")


def test_a_rerun_under_the_declared_rate_makes_every_door_edit_on_the_new_clock(stale):
    """The amber line's own button is the way forward, and it leaves nothing stale."""
    stale.panel._stale._btn.click()
    stale.qtbot.waitUntil(lambda: stale.panel._pending is None and stale.stamp() == DECLARED,
                          timeout=60_000)
    settle(80)
    assert stale.panel._stale.isHidden()
    assert rows_on_clock(stale.df(), DECLARED) == len(stale.df())
    n = len(stale.df())
    stale.bracket(_MISSED)
    stale.window._act_mark_missed.trigger()
    settle()
    df = stale.df()
    assert len(df) == n + 1 and rows_on_clock(df, DECLARED) == n + 1
    assert "Added missed event" in stale.status() and _MODAL == []


def test_a_reopened_project_is_still_stale_and_still_refuses(stale, qtbot, tmp_path):
    """The stamp travels with the table and the declaration with the project: reopened, the
    table is stale again, and the doors still refuse."""
    from nanoscope.core.project import load_project, save_project

    saved = save_project(tmp_path / "data1.nanoscope", stale.window._gather_project_state())
    window = build_window(qtbot, detection=False)
    try:
        doc_id = window._session.add_document(_document(tmp_path / "data1.dat"))
        window._restore_state = load_project(saved)
        window._apply_restore_for_doc(doc_id)
        window._create_default_layers(window._session.document(doc_id))
        window._on_selection(doc_id, 0)
        settle(80)
        reopened = Rig(window, doc_id, qtbot)
        assert window._session.document(doc_id).meta.sample_rate_hz == DECLARED
        assert reopened.stamp() == FS and reopened.panel._stale.kind == "sample_rate"
        before, undo = reopened.snapshot(), reopened.undo_text()
        reopened.bracket(_MISSED)
        window._act_mark_missed.trigger()
        settle()
        assert_refused(reopened, before, undo, "Nothing was added.")
        reopened.merge_all()
        assert_refused(reopened, before, undo, "Nothing was merged.")
    finally:
        window._restore_state = None
        close_window(window)


def test_a_table_that_carries_no_rate_stamp_is_judged_by_nothing_and_edits_as_before(rig):
    """An older table has no ``fs`` to compare: the amber line stays silent, and so do the
    doors (the conservative reading the banner has always taken)."""
    rig.panel.current_detection().payload.provenance.pop("fs")
    rig.declare()
    # (a baseline-estimator line may be up -- the rate moved its derived decimation -- but the
    # RATE line is not: there is no stamp to compare)
    assert rig.panel._stale.isHidden() or rig.panel._stale.kind != "sample_rate"
    n = len(rig.df())
    rig.bracket(_MISSED)
    rig.window._act_mark_missed.trigger()
    settle()
    assert len(rig.df()) == n + 1 and "Added missed event" in rig.status()
