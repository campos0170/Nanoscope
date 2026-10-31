"""Each table's un-run form edits are saved with the project: design §5.2 and §8, the owner's
decision 4.

The forms keep an edit the user has not run yet for every event table, so a switch of recording
or table never discards it (``AnalysisPanel.form_drafts``). Those drafts go into the project as
one versioned ``workspace.extra`` key, ``analysis_form_drafts_v1``, keyed by each recording's
PERSISTENT key (``nanoscope/core/form_draft_state.py``), and come back on reopen BEFORE the
restore's selection settles, so the table that comes back on screen comes back with its draft.

What is pinned here, through the real save (``MainWindow._write_project``) and the real reopen
(``MainWindow._open_project`` until ``projectRestored``):

* drafts survive a save and a reopen, re-keyed to the reopened recordings' new ``doc_id``;
* the table that had a draft gets it back in its form when it comes back on screen;
* a draft the reopened session has since settled is not written again, while a draft about a
  recording that did not reopen is kept for the session that can apply it;
* an unreadable drafts section is quarantined: never applied, said in the restore banner, and
  preserved verbatim beside a fresh section by the next save;
* edits TYPED into the forms (not handed to ``set_form_drafts``) come through the same trip, the
  one on screen included: the reopen arrives at the draft AND then restores the project's saved
  recipe over it, and the two are one form, so the reopened form is what was saved, counts as
  edited, creates no undo step and leaves the project clean (see the section on typed edits);
* a recipe that puts the detector combo on a detector the recording has no table for moves the
  table under the form without a switch, and neither draft is lost by it (the last section);
* a table the form did not make, landing on a recording whose form stood for it, leaves no
  record of a draft for no table in the saved project.

Synthetic recordings only: CSVs written to pytest's ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.form_draft_state import UNREADABLE_KEY, WORKSPACE_KEY
from nanoscope.core.project import load_project, save_project
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
S, C = "detect.schmitt", "detect.cusum"
_SPANS = [(10_000, 10_300), (20_000, 20_400), (30_000, 30_300)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] += 300.0
    return x.astype(np.float32)


def _settle() -> None:
    QtWidgets.QApplication.processEvents()


def _recording(win: MainWindow, tmp_path: Path, name: str) -> tuple[str, Path]:
    """A CSV on disk (so the project can reopen it), opened as a document and selected."""
    arr = _signal()
    csv = tmp_path / name
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": arr.astype(np.float64)}).to_csv(csv, index=False)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=csv, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, t0_s=0.0, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    _settle()
    return doc_id, csv


def _land(win: MainWindow, doc_id: str, module_id: str, *, min_dwell_s: float) -> None:
    """A finished run's table, installed the way a finished run installs it: with the whole
    form it ran with, so showing it seeds every setting (a partial dict seeds only its keys
    and leaves the rest of the form as it was)."""
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=min_dwell_s)
    params = {**win._analysis_panel.detection_params(), "detector": module_id,
              "k_in": 5.0, "min_dwell_s": min_dwell_s}
    win._session.add_result(AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id=doc_id, channel=0,
        params=params,
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    _settle()


def _show_table(w: MainWindow, module_id: str) -> None:
    selector = w._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData(module_id))
    QTest.qWait(10)
    assert w._analysis_panel.current_detection().module_id == module_id


def _reopen(qtbot, proj: Path) -> MainWindow:
    """File ▸ Open Project after its dialog: the real restore, until it says it has settled."""
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(50)
    return w


def _live(w: MainWindow, csv: Path) -> str:
    doc_id = w._session.doc_id_for_source(str(csv.resolve()))
    assert doc_id is not None, f"{csv.name} did not reopen"
    return doc_id


def _drafted(state: dict, *, k_in: float, rough_s: float) -> dict:
    """A form state with an edit the user has not run: what a draft holds."""
    out = copy.deepcopy(state)
    out["detection"]["k_in"] = k_in
    out["baseline"]["rough_s"] = rough_s
    return out


@pytest.fixture
def saved(qapp, qtbot, tmp_path):
    """Recording A with a Schmitt and a CUSUM table, Schmitt on screen; recording B with no
    table. Drafts on A's CUSUM table and on B's channel, neither on screen; saved."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc_b, csv_b = _recording(win, tmp_path, "rec_b.csv")
    doc_a, csv_a = _recording(win, tmp_path, "rec_a.csv")
    _land(win, doc_a, S, min_dwell_s=1e-4)
    _land(win, doc_a, C, min_dwell_s=2e-3)
    _show_table(win, S)
    base = win._analysis_panel.form_state()
    drafts = {(doc_a, 0, C): _drafted(base, k_in=7.0, rough_s=1.5),
              (doc_b, 0, None): _drafted(base, k_in=8.0, rough_s=2.5)}
    win._analysis_panel.set_form_drafts(drafts)
    proj = tmp_path / "drafts.nanoscope"
    assert win._write_project(str(proj))
    yield {"proj": proj, "csv_a": csv_a, "csv_b": csv_b,
           "a": drafts[(doc_a, 0, C)], "b": drafts[(doc_b, 0, None)]}
    win._analysis_controller.abort()


# ================================================================================================
# the round trip
# ================================================================================================

def test_the_save_writes_each_draft_under_its_recordings_persistent_key(saved):
    records = load_project(saved["proj"]).workspace.extra.get(WORKSPACE_KEY)
    assert isinstance(records, list) and len(records) == 2, records
    by_table = {(r["doc_id"], r["module_id"]): r for r in records}
    a = by_table[(str(saved["csv_a"].resolve()), C)]
    b = by_table[(str(saved["csv_b"].resolve()), None)]
    assert "#load-" not in a["doc_id"], "a per-load doc_id was written into the project"
    assert (a["channel"], a["channel_name"]) == (0, "current")
    assert a["state"] == saved["a"] and b["state"] == saved["b"]


def test_drafts_survive_a_project_save_and_reopen(saved, qtbot):
    w2 = _reopen(qtbot, saved["proj"])
    doc_a, doc_b = _live(w2, saved["csv_a"]), _live(w2, saved["csv_b"])
    drafts = w2._analysis_panel.form_drafts()
    assert drafts.get((doc_a, 0, C)) == saved["a"], sorted(drafts)
    assert drafts.get((doc_b, 0, None)) == saved["b"], sorted(drafts)
    assert w2._analysis_panel.current_detection().module_id == S, (
        "setup: the reopened screen is not the table it was saved on")


def test_a_reopened_table_comes_back_with_its_draft_in_the_form(qapp, qtbot, tmp_path):
    """Edit the CUSUM table's form, look at Schmitt's, save, reopen, look at CUSUM again: the
    edit is in the form, as it was before the save."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id, csv = _recording(win, tmp_path, "rec.csv")
    _land(win, doc_id, S, min_dwell_s=1e-4)
    _land(win, doc_id, C, min_dwell_s=2e-3)
    _show_table(win, C)
    det = win._analysis_panel.detection_panel
    k_in = det.k_in.value() + 1.5
    det.k_in.setValue(k_in)
    _show_table(win, S)
    assert det.k_in.value() != pytest.approx(k_in), "setup: Schmitt's form shows CUSUM's edit"
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    _live(w2, csv)
    _show_table(w2, C)
    assert w2._analysis_panel.detection_panel.k_in.value() == pytest.approx(k_in), (
        "the reopened CUSUM table came back without the edit it was saved with")
    win._analysis_controller.abort()


# ================================================================================================
# the next save
# ================================================================================================

def test_a_draft_the_reopened_session_settled_is_not_saved_again(saved, qtbot, tmp_path):
    """A run drops the draft of the form that produced it. Merged over the opened file the way
    the table choices are, the old record would come back on the next reopen and put back a
    form its own run had already settled."""
    w2 = _reopen(qtbot, saved["proj"])
    w2._analysis_panel.set_form_drafts({})       # what the two runs that settle them leave
    proj2 = tmp_path / "again.nanoscope"
    assert w2._write_project(str(proj2))
    assert load_project(proj2).workspace.extra.get(WORKSPACE_KEY) == []


def test_a_draft_for_a_recording_that_did_not_reopen_is_kept(saved, qtbot, tmp_path):
    """B's file is gone when the project reopens: its draft cannot be applied, and nothing in
    this session can speak for it, so the next save carries it unchanged."""
    before = load_project(saved["proj"]).workspace.extra[WORKSPACE_KEY]
    b_record = next(r for r in before if r["module_id"] is None)
    saved["csv_b"].rename(saved["csv_b"].with_suffix(".moved"))
    w2 = _reopen(qtbot, saved["proj"])
    assert w2._session.doc_id_for_source(str(saved["csv_b"].resolve())) is None
    proj2 = tmp_path / "again.nanoscope"
    assert w2._write_project(str(proj2))
    after = load_project(proj2).workspace.extra[WORKSPACE_KEY]
    assert b_record in after, "the draft of a recording that did not reopen was dropped"


# ================================================================================================
# an unreadable section
# ================================================================================================

def test_an_unreadable_drafts_section_is_quarantined_not_applied(saved, qtbot, tmp_path):
    state = load_project(saved["proj"])
    broken = {"written by": "something else", "drafts": 3}
    state.workspace.extra[WORKSPACE_KEY] = broken
    proj = save_project(tmp_path / "broken.nanoscope", state)

    w2 = _reopen(qtbot, proj)
    assert w2._analysis_panel.form_drafts() == {}, "an unreadable section was applied"
    banner = w2._info_bar._label.text()
    assert "analysis-form edits could not be restored" in banner, banner

    proj2 = tmp_path / "resaved.nanoscope"
    assert w2._write_project(str(proj2))
    extra = load_project(proj2).workspace.extra
    assert extra.get(UNREADABLE_KEY) == broken, "the unreadable section was not preserved"
    assert extra.get(WORKSPACE_KEY) == [], extra.get(WORKSPACE_KEY)


# ================================================================================================
# the user's own edits, saved from the screens they are on
# ================================================================================================
#
# The fixtures above hand drafts to the panel. A user's session holds them differently: typed into
# the form, with the edit of the table ON SCREEN never having left it. That one is written by the
# save from the live form (``AnalysisPanel.form_drafts`` includes the key on screen when it holds
# edits), and on reopen it reaches the form twice: the selection's arrival restores the draft,
# then the project's analysis-panel recipe is written over it (under the forms' writing guard).
# Both were taken from the same live form, so they must agree, and the guard must keep that
# second write from being an edit of the user's.

def _typed_session(win: MainWindow, tmp_path: Path) -> dict:
    """A user's session at the moment of Save: recording B (no table yet) edited and left;
    recording A with its Schmitt table on screen and edits of its own that nobody ran.

    Every edit is typed into a form widget: a chain stage, a threshold and a baseline window."""
    panel = win._analysis_panel
    det, base = panel.detection_panel, panel.baseline_panel
    doc_b, csv_b = _recording(win, tmp_path, "rec_b.csv")
    det.k_in.setValue(8.0)
    base.rough_s.setValue(2.5)
    doc_a, csv_a = _recording(win, tmp_path, "rec_a.csv")
    _land(win, doc_a, S, min_dwell_s=1e-4)
    det.k_in.setValue(7.0)
    det.notch_enabled.setChecked(not det.notch_enabled.isChecked())
    base.rough_s.setValue(1.5)
    return {"csv_a": csv_a, "csv_b": csv_b, "a": panel.form_state(),
            "b": panel.form_drafts()[(doc_b, 0, None)]}


def test_edits_typed_on_screen_and_off_screen_survive_a_real_save_and_reopen(
        qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    typed = _typed_session(win, tmp_path)
    proj = tmp_path / "typed.nanoscope"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    doc_a, doc_b = _live(w2, typed["csv_a"]), _live(w2, typed["csv_b"])
    panel = w2._analysis_panel
    assert panel.current_target() == (doc_a, 0) and panel.current_detection().module_id == S, (
        "setup: the reopened screen is not the table it was saved on")
    assert panel.form_state() == typed["a"], (
        "the recipe the project restores over the draft is not the form that was saved")
    assert panel.form_drafts().get((doc_b, 0, None)) == typed["b"], sorted(panel.form_drafts())

    assert not w2._act_undo.isEnabled() and w2._act_undo.text() == "Undo", (
        "the reopen is no edit of the user's: " + w2._act_undo.text())
    assert w2._settings_history.scopes_with_history() == ()
    assert not w2._dirty, "a reopen left the project unsaved"

    # The reopened form counts as edited, so the edit survives a look at B and back, and
    # B's own comes with B: a form read as clean would be re-seeded from its run on the way out.
    w2._on_selection(doc_b, 0)
    qtbot.wait(20)
    assert panel.form_state() == typed["b"], "B's draft did not come back on B"
    w2._on_selection(doc_a, 0)
    qtbot.wait(20)
    assert panel.form_state() == typed["a"], "A's edit was lost to a look at B"
    assert w2._settings_history.scopes_with_history() == (), "a look at B recorded a step"
    assert not w2._dirty

    # ...and a second save writes what the first did.
    proj2 = tmp_path / "again.nanoscope"
    assert w2._write_project(str(proj2))

    def by_table(path):
        records = load_project(path).workspace.extra[WORKSPACE_KEY]
        return sorted(records, key=lambda r: (r["doc_id"], r["module_id"] or ""))

    assert by_table(proj2) == by_table(proj)
    win._analysis_controller.abort()


def test_the_first_edit_after_a_reopen_is_a_step_of_its_own(qapp, qtbot, tmp_path):
    """The reopened form is adopted as the starting point of its history, without a step. A form
    that was not, would take the user's next edit as its baseline and leave nothing to undo."""
    win = MainWindow()
    qtbot.addWidget(win)
    typed = _typed_session(win, tmp_path)
    proj = tmp_path / "typed.nanoscope"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    panel = w2._analysis_panel
    k_in = panel.detection_panel.k_in.value()
    panel.detection_panel.k_in.setValue(k_in + 2.0)
    w2._checkpoint_everything()
    assert w2._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 2.0:g}", (
        w2._act_undo.text())
    w2._act_undo.trigger()
    assert panel.form_state() == typed["a"], "Undo did not return to the form that was reopened"
    win._analysis_controller.abort()


def test_a_form_nobody_edited_is_not_an_edited_form_after_a_reopen(qapp, qtbot, tmp_path):
    """The recipe the restore writes over the form is not an edit: a table on screen that was
    saved with its form as its run left it comes back clean, so the next save writes no draft
    for it. Written as an edit, it would be filed as a draft the moment the user looked away."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id, csv = _recording(win, tmp_path, "rec.csv")
    _land(win, doc_id, S, min_dwell_s=1e-4)
    assert win._analysis_panel.form_drafts() == {}, "setup: the seeded form counts as edited"
    proj = tmp_path / "clean.nanoscope"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    assert w2._analysis_panel.form_drafts() == {}, (
        "the restore of the saved recipe marked the form on screen as edited")
    proj2 = tmp_path / "again.nanoscope"
    assert w2._write_project(str(proj2))
    assert load_project(proj2).workspace.extra.get(WORKSPACE_KEY, []) == []
    win._analysis_controller.abort()


def test_open_project_over_an_edited_session_leaves_the_edit_behind(
        qapp, qtbot, tmp_path, monkeypatch):
    """The window that opens a project may hold an un-run edit of its own. The reset that hands
    the Analysis panel its defaults back, and the restore that follows it, are neither of them
    an edit of the user's: the forms are written without one ``formEdited``, and come up as the
    project saved them, with no draft carried over, no step and no unsaved mark.

    The signal is what pins the reset's replay of the constructor defaults, which writes the
    outgoing project's form before the session is cleared: whatever it marked would be gone by
    the end of the open, so only the edit it announced says it was taken for the user's."""
    saved = MainWindow()
    qtbot.addWidget(saved)
    doc_id, csv = _recording(saved, tmp_path, "rec.csv")
    _land(saved, doc_id, S, min_dwell_s=1e-4)
    proj = tmp_path / "clean.nanoscope"
    assert saved._write_project(str(proj))

    win = MainWindow()
    qtbot.addWidget(win)
    _recording(win, tmp_path, "other.csv")
    win._analysis_panel.detection_panel.k_in.setValue(9.0)
    assert win._analysis_panel.form_drafts(), "setup: the edit on screen is not a draft"
    monkeypatch.setattr(win, "_confirm_close_choice", lambda: "discard")
    edited: list[int] = []
    win._analysis_panel.formEdited.connect(lambda: edited.append(1))
    with qtbot.waitSignal(win.projectRestored, timeout=30_000):
        win._open_project(str(proj))
    qtbot.wait(50)

    assert edited == [], "opening a project wrote a form as if the user had edited it"
    assert win._analysis_panel.form_drafts() == {}, "the project opened with an edit it never had"
    assert win._settings_history.scopes_with_history() == ()
    assert not win._act_undo.isEnabled()
    assert not win._dirty
    saved._analysis_controller.abort()
    win._analysis_controller.abort()


# ================================================================================================
# the table the recipe lands on is not the table the arrival landed on
# ================================================================================================
#
# The reopen's selection arrives at one table of the recording (and takes its draft off the
# shelf); the saved panel recipe then puts the detector combo where it was saved, and a detector
# the recording has no table for leaves NO table on screen. That moves the form from its table's
# key to ``(doc, ch, None)`` with no switch, and the form the arrival had put on screen was
# re-filed under the key it landed on -- the table's draft was nowhere, and the next save dropped
# it from the file.

def _no_table_session(win: MainWindow, tmp_path: Path, *,
                      edit_the_table: bool) -> tuple[str, Path, float]:
    """One recording holding only a CUSUM table, whose form is edited when ``edit_the_table``;
    the detector combo is then moved to Schmitt, which has no table on it. Returns
    ``(doc_id, csv, cusum_k)``, the CUSUM threshold the table's form held when it was left."""
    doc_id, csv = _recording(win, tmp_path, "rec.csv")
    _land(win, doc_id, C, min_dwell_s=2e-3)
    det = win._analysis_panel.detection_panel
    if edit_the_table:
        det.cusum_k.setValue(det.cusum_k.value() + 0.75)     # an edit nobody ran
    cusum_k = det.cusum_k.value()
    det.detector_combo.setCurrentIndex(det.detector_combo.findData(S))
    QTest.qWait(20)
    assert win._analysis_panel.form_key() == (doc_id, 0, None), "setup: Schmitt shows no table"
    return doc_id, csv, cusum_k


def _saved_modules(path: Path) -> set:
    return {r["module_id"] for r in load_project(path).workspace.extra[WORKSPACE_KEY]}


def test_a_tables_draft_survives_a_reopen_whose_recipe_moves_the_form_off_its_table(
        qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    _doc_id, csv, cusum_k = _no_table_session(win, tmp_path, edit_the_table=True)
    win._analysis_panel.detection_panel.k_in.setValue(7.0)   # an edit of the no-table form too
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))
    assert _saved_modules(proj) == {None, C}, "setup: the file holds both drafts"

    w2 = _reopen(qtbot, proj)
    doc_id = _live(w2, csv)
    panel = w2._analysis_panel
    assert (doc_id, 0, C) in panel.form_drafts(), (
        f"the CUSUM table's draft was lost by the reopen: {sorted(panel.form_drafts())}")
    proj2 = tmp_path / "again.nanoscope"
    assert w2._write_project(str(proj2))
    assert C in _saved_modules(proj2), "...and the next save dropped it from the file"

    _show_table(w2, C)
    assert panel.detection_panel.cusum_k.value() == pytest.approx(cusum_k), (
        "the table came back without the edit it was saved with")
    win._analysis_controller.abort()


def test_an_edit_of_the_no_table_form_survives_a_look_at_the_other_table_after_a_reopen(
        qapp, qtbot, tmp_path):
    """The reopened screen showed the edit, but counted it as no edit: the first look away
    dropped the draft the reopen had restored, and the form read its table's number again."""
    win = MainWindow()
    qtbot.addWidget(win)
    _doc_id, csv, _cusum_k = _no_table_session(win, tmp_path, edit_the_table=False)
    det = win._analysis_panel.detection_panel
    det.k_in.setValue(7.0)
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    _live(w2, csv)
    det2 = w2._analysis_panel.detection_panel
    assert det2.k_in.value() == pytest.approx(7.0), "setup: the reopened form shows the edit"
    _show_table(w2, C)
    det2.detector_combo.setCurrentIndex(det2.detector_combo.findData(S))
    QTest.qWait(20)
    assert det2.k_in.value() == pytest.approx(7.0), (
        "the edit of the no-table form was lost to a look at the other table")
    win._analysis_controller.abort()


# ================================================================================================
# a first table the form did not make
# ================================================================================================

def test_a_first_table_the_form_did_not_make_is_saved_with_no_draft_for_no_table(
        qapp, qtbot, tmp_path):
    """A batch run lands the first table of the recording on screen, whose form was tuned.
    That form stood for the table; written into the project as a record for no table, it
    came back on the form of the first detector that had none."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id, _csv = _recording(win, tmp_path, "rec.csv")
    win._analysis_panel.detection_panel.k_in.setValue(7.0)
    _land(win, doc_id, S, min_dwell_s=1e-4)

    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))
    records = load_project(proj).workspace.extra.get(WORKSPACE_KEY, [])
    assert [r for r in records if r["module_id"] is None] == [], records
    win._analysis_controller.abort()
