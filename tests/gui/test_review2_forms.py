"""The second integrated review of the owner's decisions, group ``forms``: three defects where
13.4 Q3 (the detector combo's re-seed is no step) and 13.4 Q6 (a first table keeps an un-run
edit) met each other or the batch.

What is pinned here:

* a draft of no table that moves onto a first table takes that table's min dwell and merge gap
  where it held the OTHER detector's re-seeded defaults, and so do the steps that move with it:
  a Schmitt table no longer shows CUSUM's knobs as "Settings changed" that no Undo takes back,
  and a draft whose only difference from the batch was that re-seed was run;
* a step the re-seed's carry leaves moving nothing goes, with its journal entries, and a step
  whose difference it changed is named for the difference it now makes;
* a first table ran a draft only when a batch was launched from that very form, every part of
  it: an un-run ML edit is no longer dropped because the detection settings happen to agree;
* only the knobs the app put at a detector's defaults (the combo's re-seed, a table's seed, the
  panel's start) follow a first table: a value typed that equals the other detector's default
  stays, and a draft a project put back, which cannot say, takes the combo's rule by value;
* that record describes the form a project's restore leaves with no draft (its knobs at the
  shown detector's defaults, as at the panel's start), and goes with a draft that moves onto a
  first table;
* a first table another door made from the form on screen as it stands (a script's
  ``add_result``) ran it, as Detect does: no draft, and a reference it restores reads restored.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest

from nanoscope.core.history import SettingsHistory
from nanoscope.gui.main_window import _UNDO_EVENTS, _UNDO_RUN, _UNDO_SETTINGS, MainWindow

from .test_editing_drafts_persistence import _live, _recording, _reopen
from .test_editing_form_undo import C, S, _commit, _det, _install, _redo, _scope, _undo
from .test_owner_decisions_forms import _combo, _edited_k_in, _knobs, _landed

SCHMITT_KNOBS, CUSUM_KNOBS = (pytest.approx(2e-5), pytest.approx(0.0)), (
    pytest.approx(1e-3), pytest.approx(0.5))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _batch(w: MainWindow, qtbot) -> None:
    """Analysis ▸ Batch detect from the form on screen, until its tables have landed."""
    with qtbot.waitSignal(w._batch_controller.finished, timeout=30_000):
        w._batch_detect()
    qtbot.wait(50)


def _look(w: MainWindow, doc_id: str) -> None:
    w._on_selection(doc_id, 0)
    QTest.qWait(20)


def _chain(w: MainWindow):
    return w._analysis_panel.refresh_chain_status()


# ================================================================================================
# reseed-knobs-ride-kept-first-table-draft
# ================================================================================================

def test_a_kept_first_table_draft_takes_its_tables_knobs_not_the_other_detectors(
        window, tmp_path, qtbot):
    """b: k_in typed, then CUSUM picked (min dwell and merge gap re-seeded, no step). a: back on
    Schmitt, k_in as before. A batch from a gives b its first Schmitt table, at 20 µs / off.
    b keeps the k_in edit; CUSUM's knobs used to ride along, named by the chain notice, and the
    Undo of the k_in step left them there with nothing more to undo."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    k_in, typed = _edited_k_in(window)
    _combo(window, C)
    assert _knobs(window) == CUSUM_KNOBS, "setup: no re-seed"
    _look(window, doc_a)
    _combo(window, S)
    _det(window).k_in.setValue(k_in)
    _batch(window, qtbot)
    res = window._session.results.get(doc_b, 0, S)
    assert res.params["min_dwell_s"] == pytest.approx(2e-5) and res.params["merge_gap_s"] is None

    draft = window._analysis_panel.form_drafts()[(doc_b, 0, S)]["detection"]
    assert (draft["k_in"], draft["min_dwell_s"], draft["merge_gap_s"]) == (
        pytest.approx(typed), pytest.approx(2e-5), None)
    _look(window, doc_b)
    assert _det(window).detector_choice() == S
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == SCHMITT_KNOBS, "CUSUM's re-seed rode the kept draft onto Schmitt"
    status = _chain(window)
    assert status.kind == "differs" and "k_in" in status.sentence, status.sentence
    assert "dwell" not in status.sentence and "merge" not in status.sentence, status.sentence

    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {typed:g}"
    _undo(window)
    assert _det(window).k_in.value() == pytest.approx(k_in)
    assert _knobs(window) == SCHMITT_KNOBS, "the moved step's Undo put CUSUM's knobs back"
    assert _chain(window).kind != "differs", _chain(window).sentence
    _redo(window)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == SCHMITT_KNOBS


def test_a_first_table_landing_on_screen_moves_the_reseeded_knobs_with_the_draft(window, tmp_path):
    doc_id = _install(window, tmp_path)
    k_in, typed = _edited_k_in(window)
    _combo(window, C)
    _landed(window, doc_id, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)

    panel = window._analysis_panel
    assert panel.form_key() == (doc_id, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed) and panel._form_dirty
    assert _knobs(window) == SCHMITT_KNOBS
    _undo(window)
    assert _det(window).k_in.value() == pytest.approx(k_in)
    assert _knobs(window) == SCHMITT_KNOBS


def test_a_draft_that_differs_from_the_batch_only_by_the_reseed_was_run(window, tmp_path, qtbot):
    """b: k_in typed, CUSUM picked. a shows b's numbers (it has no table), is put back on
    Schmitt, and launches the batch: the batch ran b's draft as Schmitt runs it. b's first
    table seeds the form as before Q6, and its moved step cannot put CUSUM's knobs back."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    k_in, typed = _edited_k_in(window)
    _combo(window, C)
    _look(window, doc_a)
    _combo(window, S)
    _batch(window, qtbot)
    assert window._session.results.get(doc_b, 0, S).params["k_in"] == pytest.approx(typed)

    assert not [key for key in window._analysis_panel.form_drafts() if key[0] == doc_b]
    _look(window, doc_b)
    assert not window._analysis_panel._form_dirty
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == SCHMITT_KNOBS
    assert _chain(window).kind != "differs", _chain(window).sentence
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {typed:g}"
    _undo(window)
    assert _knobs(window) == SCHMITT_KNOBS, "the moved step's Undo put CUSUM's knobs back"


# ================================================================================================
# carry-leaves-noop-and-misnamed-form-steps
# ================================================================================================

def test_a_redo_the_reseed_has_reached_is_withdrawn(window, tmp_path):
    """Min dwell typed to 1 ms (CUSUM's default) and undone; CUSUM picked re-seeds 20 µs to
    1 ms. The Redo stayed named, said "Redone", and moved nothing."""
    _install(window, tmp_path)
    det = _det(window)
    det.min_dwell_s.setValue(1e-3)
    _commit(window)
    _undo(window)
    assert window._act_redo.text() == "Redo Change min dwell 20 µs → 1 ms"

    _combo(window, C)
    assert det.min_dwell_s.value() == pytest.approx(1e-3), "setup: no re-seed"
    _commit(window)
    assert not window._act_redo.isEnabled(), window._act_redo.text()
    assert window._settings_history.redo_depth(_scope(window)) == 0
    assert (_UNDO_SETTINGS, _scope(window)) not in window._redo_journal


def test_an_undo_step_the_reseed_emptied_goes_with_its_journal_entry(window, tmp_path):
    """Min dwell typed to 1 ms on Schmitt, then CUSUM and back: the way back re-seeds 1 ms to
    20 µs, and the step 20 µs → 1 ms moved nothing any more."""
    _install(window, tmp_path)
    det = _det(window)
    det.min_dwell_s.setValue(1e-3)
    _commit(window)
    _combo(window, C)
    _combo(window, S)
    assert _knobs(window) == SCHMITT_KNOBS, "setup: no re-seed on the way back"
    _commit(window)
    assert not window._act_undo.isEnabled(), window._act_undo.text()
    assert window._settings_history.undo_depth(_scope(window)) == 0
    assert (_UNDO_SETTINGS, _scope(window)) not in window._undo_journal


def test_a_step_the_reseed_changed_is_named_for_what_it_now_moves(window, tmp_path):
    """20 µs → 2 ms → 20 µs, then CUSUM (20 µs re-seeded to 1 ms). The second Undo was named
    "20 µs → 2 ms" and put the form on 1 ms."""
    _install(window, tmp_path)
    det = _det(window)
    det.min_dwell_s.setValue(2e-3)
    _commit(window)
    det.min_dwell_s.setValue(2e-5)
    _commit(window)
    _combo(window, C)
    assert window._act_undo.text() == "Undo Change min dwell 2 ms → 1 ms"
    _undo(window)
    assert det.min_dwell_s.value() == pytest.approx(2e-3)
    assert window._act_undo.text() == "Undo Change min dwell 1 ms → 2 ms"
    _undo(window)
    assert det.min_dwell_s.value() == pytest.approx(1e-3)
    assert window.statusBar().currentMessage().startswith("Undone: Change min dwell 1 ms → 2 ms")


def test_carry_drops_the_steps_it_empties_and_renames_only_derived_labels():
    store = {"v": {"a": 1, "b": 0}}
    history = SettingsHistory(
        lambda _s: dict(store["v"]), lambda _s, state: store.update(v=state),
        lambda _s, before, after: "Change " + ", ".join(
            f"{k} {before[k]} → {after[k]}" for k in sorted(after) if before[k] != after[k]))
    scope = ("form", "doc", 0, None)
    history.rebase(scope)
    for state, label in (({"a": 2, "b": 0}, None), ({"a": 2, "b": 5}, "Recall protocol ‘X’"),
                         ({"a": 2, "b": 0}, None), ({"a": 2, "b": 9}, None),
                         ({"a": 3, "b": 9}, None)):
        store["v"] = state
        assert history.checkpoint(scope, label=label)
    history.undo(scope)                                  # a 2 → 3 waits in Redo

    def amend(state):
        if state["b"] == 0:
            state["b"] = 9
        return state

    store["v"] = {"a": 2, "b": 9}
    assert history.carry(scope, amend) == ((0,), ())     # b 0 → 9 moves nothing now
    assert history.undo_depth(scope) == 3 and history.redo_depth(scope) == 1
    assert history.undo_label(scope) == "Change b 5 → 9", "a derived label was kept"
    history.undo(scope)
    assert history.undo_label(scope) == "Recall protocol ‘X’", "an action's name was replaced"
    history.undo(scope)
    assert history.undo_label(scope) == "Change a 1 → 2" and store["v"] == {"a": 2, "b": 9}


def test_the_journal_loses_exactly_the_entries_of_the_steps_a_carry_dropped(window):
    scope, other = ("form", "d", 0, S), ("form", "d", 0, None)
    journal = [(_UNDO_SETTINGS, scope), (_UNDO_RUN, scope, 2), (_UNDO_SETTINGS, other),
               (_UNDO_SETTINGS, scope)]
    # Counted from the newest end: 0 is the last entry, 1 and 2 the run's two steps, 3 the first.
    window._drop_journal_steps(journal, scope, (0, 2))
    assert journal == [(_UNDO_SETTINGS, scope), (_UNDO_RUN, scope, 1), (_UNDO_SETTINGS, other)]
    window._drop_journal_steps(journal, scope, (0,))
    assert journal == [(_UNDO_SETTINGS, scope), (_UNDO_EVENTS, ("d", 0, S)),
                       (_UNDO_SETTINGS, other)]


# ================================================================================================
# q6-unrun-ml-edit-dropped
# ================================================================================================

def _ml_enter(w: MainWindow, value: float) -> None:
    _det(w).ml_enter.setValue(value)
    _commit(w)


def test_an_un_run_ml_edit_is_kept_over_a_first_table_a_batch_from_another_form_gives(
        window, tmp_path, qtbot):
    """a: ML enter edited, no table. b: another ML enter, detection untouched; Batch detect
    from b. a's detection settings equal what the batch ran, and its ML edit was dropped."""
    doc_a = _install(window, tmp_path, "a.dat")
    _ml_enter(window, 0.85)
    _install(window, tmp_path, "b.dat")
    _ml_enter(window, 0.95)
    _batch(window, qtbot)
    assert window._session.results.get(doc_a, 0, S) is not None, "setup: a got no table"

    drafts = window._analysis_panel.form_drafts()
    assert drafts[(doc_a, 0, S)]["ml"]["enter"] == pytest.approx(0.85), "the ML edit was dropped"
    _look(window, doc_a)
    assert _det(window).ml_enter.value() == pytest.approx(0.85)
    assert window._analysis_panel._form_dirty


def test_a_baseline_reference_edit_is_kept_over_a_first_table_too(window, tmp_path, qtbot):
    """The builder's documented edge: a reference-only edit, which a Schmitt pass records
    nowhere, counted as run."""
    doc_a = _install(window, tmp_path, "a.dat")
    window._analysis_panel.baseline_panel.reference_editor.notes.setText(
        "from the 1 M KCl calibration")
    _commit(window)
    assert window._analysis_panel._form_dirty, "setup: the reference edit is no edit"
    _install(window, tmp_path, "b.dat")
    window._analysis_panel.baseline_panel.reference_editor.notes.setText("")
    _batch(window, qtbot)

    drafts = window._analysis_panel.form_drafts()
    assert drafts[(doc_a, 0, S)]["baseline"]["reference"]["notes"] == (
        "from the 1 M KCl calibration")


# ================================================================================================
# typed-value-knob-translation
# ================================================================================================

@pytest.mark.parametrize("knob", ["merge_gap", "min_dwell"])
def test_a_typed_value_equal_to_the_other_detectors_default_stays_on_a_first_table(
        window, tmp_path, qtbot, knob):
    """b: Schmitt all along, the combo never moved; the user types a merge gap of 0.5 ms (or a
    min dwell of 1 ms), CUSUM's default, and k_in. A batch from a gives b its first Schmitt
    table at 20 µs / off. The typed value is the user's, an un-run edit the table keeps (13.4
    Q6); it used to be taken for the combo's re-seed and replaced, in the draft and its step."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    det = _det(window)
    assert det.detector_choice() == S
    if knob == "merge_gap":
        det.merge_gap_ms.setValue(0.5)
    else:
        det.min_dwell_s.setValue(1e-3)
    _commit(window)
    k_in, typed = _edited_k_in(window)
    knobs = _knobs(window)
    _look(window, doc_a)
    det.min_dwell_s.setValue(2e-5)
    det.merge_gap_ms.setValue(0.0)
    det.k_in.setValue(k_in)
    _batch(window, qtbot)
    res = window._session.results.get(doc_b, 0, S)
    assert res.params["min_dwell_s"] == pytest.approx(2e-5) and res.params["merge_gap_s"] is None

    _look(window, doc_b)
    assert _knobs(window) == (pytest.approx(knobs[0]), pytest.approx(knobs[1])), (
        "the typed value was taken for the combo's re-seed")
    status = _chain(window)
    assert status.kind == "differs" and ("merge" if knob == "merge_gap" else "dwell") in (
        status.sentence), status.sentence
    _undo(window)
    _undo(window)
    assert _knobs(window) == SCHMITT_KNOBS, "the typed value's step lost it"


def test_only_the_knob_the_combo_reseeded_follows_the_first_table(window, tmp_path, qtbot):
    """b: a merge gap of 0.5 ms typed on Schmitt, then CUSUM picked: the combo re-seeds the min
    dwell (1 ms) and leaves the typed merge gap. A Schmitt first table takes back the min dwell
    the combo put there and keeps the merge gap the user typed."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    _det(window).merge_gap_ms.setValue(0.5)
    _commit(window)
    _combo(window, C)
    assert _knobs(window) == (pytest.approx(1e-3), pytest.approx(0.5)), "setup: no re-seed"
    _look(window, doc_a)
    _combo(window, S)
    assert _knobs(window) == SCHMITT_KNOBS, "setup: a is not at Schmitt's defaults"
    _batch(window, qtbot)

    draft = window._analysis_panel.form_drafts()[(doc_b, 0, S)]["detection"]
    assert (draft["min_dwell_s"], draft["merge_gap_s"]) == (
        pytest.approx(2e-5), pytest.approx(5e-4))
    _look(window, doc_b)
    assert _knobs(window) == (pytest.approx(2e-5), pytest.approx(0.5))
    status = _chain(window)
    assert "merge" in status.sentence and "dwell" not in status.sentence, status.sentence
    _undo(window)
    assert _knobs(window) == SCHMITT_KNOBS


def test_a_form_at_schmitts_defaults_follows_a_first_cusum_table(window, tmp_path):
    """The other way round: knobs the user never touched, at Schmitt's defaults since the panel
    started, follow a first CUSUM table like the re-seed's."""
    doc_id = _install(window, tmp_path)
    k_in, typed = _edited_k_in(window)
    _landed(window, doc_id, C, k_in=k_in, min_dwell_s=1e-3, merge_gap_s=5e-4)

    assert window._analysis_panel.form_key() == (doc_id, 0, C)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == CUSUM_KNOBS, "Schmitt's defaults rode the draft onto CUSUM"


def test_a_cusum_tables_defaults_a_look_away_carries_follow_a_first_schmitt_table(
        window, tmp_path):
    """a's CUSUM table at its defaults seeds the knobs; b, with no table, shows them (the
    look-away keeps the widgets) and the user types k_in. A first Schmitt table on b takes
    them, as the re-seed's: the user typed neither."""
    doc_a = _install(window, tmp_path, "a.dat")
    _landed(window, doc_a, C, min_dwell_s=1e-3, merge_gap_s=5e-4)
    assert _det(window).detector_choice() == C and _knobs(window) == CUSUM_KNOBS, (
        "setup: a's form did not follow its table")
    doc_b = _install(window, tmp_path, "b.dat")
    assert _knobs(window) == CUSUM_KNOBS, "setup: the look-away did not carry a's knobs"
    k_in, typed = _edited_k_in(window)
    _landed(window, doc_b, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)

    assert window._analysis_panel.form_key() == (doc_b, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == SCHMITT_KNOBS


def test_a_draft_a_project_put_back_follows_the_combos_rule_by_value(window, tmp_path):
    """A project does not record which knobs of a draft the app put there: a no-table draft
    put back by `set_form_drafts` holding CUSUM's defaults takes a first Schmitt table's, as
    the combo's own rule would."""
    doc_id = _install(window, tmp_path)
    panel = window._analysis_panel
    k_in = _det(window).k_in.value()
    state = panel.form_state()
    state["detection"].update(k_in=k_in + 2.0, min_dwell_s=1e-3, merge_gap_s=5e-4)
    panel.set_form_drafts({(doc_id, 0, None): state})
    assert _knobs(window) == CUSUM_KNOBS and panel._form_dirty, "setup: the draft is not back"
    _landed(window, doc_id, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)

    assert panel.form_key() == (doc_id, 0, S)
    assert _det(window).k_in.value() == pytest.approx(k_in + 2.0)
    assert _knobs(window) == SCHMITT_KNOBS


# ================================================================================================
# defaulted-record-after-reopen
# ================================================================================================

def test_a_reseed_a_reopen_restores_with_no_draft_follows_a_first_schmitt_table(
        qapp, qtbot, tmp_path):
    """b, with no table: CUSUM picked, the knobs re-seeded and nothing typed, so no draft; saved
    and reopened. The recipe puts CUSUM's knobs back. k_in typed on b, then a batch from a, back
    on Schmitt, gives b its first Schmitt table: CUSUM's knobs follow it. The record kept the
    panel's start over the restored form, read them as typed, and the table showed "min dwell
    20 µs → 1 ms; merge gap off → 500 µs" with no Undo to take them (the original reseed defect)."""
    win = MainWindow()
    qtbot.addWidget(win)
    _doc_a, csv_a = _recording(win, tmp_path, "a.csv")
    _doc_b, csv_b = _recording(win, tmp_path, "b.csv")
    _combo(win, C)
    assert _knobs(win) == CUSUM_KNOBS and not win._analysis_panel.form_drafts(), "setup"
    proj = tmp_path / "p.nsproj"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    doc_a, doc_b = _live(w2, csv_a), _live(w2, csv_b)
    _look(w2, doc_b)
    assert _det(w2).detector_choice() == C and _knobs(w2) == CUSUM_KNOBS, (
        "setup: the reopen did not restore the re-seed")
    k_in, typed = _edited_k_in(w2)
    _look(w2, doc_a)
    _combo(w2, S)
    _det(w2).k_in.setValue(k_in)
    _batch(w2, qtbot)
    res = w2._session.results.get(doc_b, 0, S)
    assert res.params["min_dwell_s"] == pytest.approx(2e-5) and res.params["merge_gap_s"] is None

    _look(w2, doc_b)
    assert _det(w2).k_in.value() == pytest.approx(typed)
    assert _knobs(w2) == SCHMITT_KNOBS, "the re-seed the recipe restored rode the draft"
    status = _chain(w2)
    assert "dwell" not in status.sentence and "merge" not in status.sentence, status.sentence
    _undo(w2)
    assert _chain(w2).kind != "differs", _chain(w2).sentence


def test_a_value_typed_elsewhere_that_a_reopen_restores_stays_on_a_first_table(
        qapp, qtbot, tmp_path):
    """a: a merge gap of 0.5 ms typed on Schmitt. b, with no table, shows it (the look-away
    keeps the widgets) and is saved on screen with no draft. The reopen's recipe puts it back:
    not at the shown detector's default, so not the app's, and b's first Schmitt table keeps it.
    (Pins the record over "unknown", which takes the combo's rule by value and drops it.)"""
    win = MainWindow()
    qtbot.addWidget(win)
    _doc_a, _csv_a = _recording(win, tmp_path, "a.csv")
    _det(win).merge_gap_ms.setValue(0.5)
    _commit(win)
    _doc_b, csv_b = _recording(win, tmp_path, "b.csv")
    typed_gap = (pytest.approx(2e-5), pytest.approx(0.5))
    assert _knobs(win) == typed_gap, "setup: the look-away did not carry the typed gap"
    proj = tmp_path / "p.nsproj"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    doc_b = _live(w2, csv_b)
    _look(w2, doc_b)
    assert _knobs(w2) == typed_gap and (doc_b, 0, None) not in w2._analysis_panel.form_drafts()
    k_in, typed = _edited_k_in(w2)
    _landed(w2, doc_b, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)

    assert w2._analysis_panel.form_key() == (doc_b, 0, S)
    assert _det(w2).k_in.value() == pytest.approx(typed)
    assert _knobs(w2) == typed_gap, "the typed merge gap was taken for the app's"


def test_a_draft_moved_onto_a_first_table_keeps_its_record_of_the_apps_knobs(window, tmp_path):
    """b: k_in typed on the knobs the panel started with; b's first Schmitt table lands and the
    draft moves onto it. A look away to d, with no table, carries those knobs; k_in typed
    there; d's first CUSUM table takes CUSUM's knobs, as the app put Schmitt's there. The
    moved draft came back with an empty record, which read every knob as typed."""
    doc_b = _install(window, tmp_path, "b.dat")
    k_in, typed = _edited_k_in(window)
    _landed(window, doc_b, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)
    assert window._analysis_panel.form_key() == (doc_b, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed), "setup: the draft did not move"
    doc_d = _install(window, tmp_path, "d.dat")
    assert _knobs(window) == SCHMITT_KNOBS, "setup: the look-away did not carry b's knobs"
    k_in, typed = _edited_k_in(window)
    _landed(window, doc_d, C, k_in=k_in, min_dwell_s=1e-3, merge_gap_s=5e-4)

    assert window._analysis_panel.form_key() == (doc_d, 0, C)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == CUSUM_KNOBS, "Schmitt's knobs rode d's draft onto CUSUM"


def test_a_typed_value_a_moved_draft_carries_stays_on_the_next_first_table(window, tmp_path):
    """b: a merge gap of 0.5 ms typed on Schmitt, and k_in; b's first Schmitt table keeps them
    and the draft moves onto it. A look away to d carries them; d's first Schmitt table keeps
    the typed gap too. The moved draft's record has to come with it: unknown, the rule by
    value took the gap for CUSUM's re-seed."""
    doc_b = _install(window, tmp_path, "b.dat")
    _det(window).merge_gap_ms.setValue(0.5)
    _commit(window)
    k_in, typed = _edited_k_in(window)
    typed_gap = (pytest.approx(2e-5), pytest.approx(0.5))
    _landed(window, doc_b, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)
    assert window._analysis_panel.form_key() == (doc_b, 0, S) and _knobs(window) == typed_gap, (
        "setup: b's first table did not keep the typed gap")
    doc_d = _install(window, tmp_path, "d.dat")
    assert _knobs(window) == typed_gap, "setup: the look-away did not carry b's knobs"
    k_in, typed = _edited_k_in(window)
    _landed(window, doc_d, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)

    assert window._analysis_panel.form_key() == (doc_d, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == typed_gap, "the typed merge gap was taken for the app's"


def test_the_knobs_a_first_table_moved_count_as_the_apps_on_the_next_first_table(
        window, tmp_path):
    """b: k_in typed, then CUSUM picked (re-seeded); b's first Schmitt table takes the knobs
    back to its own as the draft moves onto it. A look away to d carries them; k_in typed
    there; d's first CUSUM table takes CUSUM's again. The record still held the re-seed's
    values, read the table's as typed, and let them ride onto CUSUM."""
    doc_b = _install(window, tmp_path, "b.dat")
    k_in, typed = _edited_k_in(window)
    _combo(window, C)
    _landed(window, doc_b, S, k_in=k_in, min_dwell_s=2e-5, merge_gap_s=None)
    assert window._analysis_panel.form_key() == (doc_b, 0, S) and _knobs(window) == (
        SCHMITT_KNOBS), "setup: the first table did not move the re-seed"
    doc_d = _install(window, tmp_path, "d.dat")
    assert _knobs(window) == SCHMITT_KNOBS, "setup: the look-away did not carry b's knobs"
    k_in, typed = _edited_k_in(window)
    _landed(window, doc_d, C, k_in=k_in, min_dwell_s=1e-3, merge_gap_s=5e-4)

    assert window._analysis_panel.form_key() == (doc_d, 0, C)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == CUSUM_KNOBS, "Schmitt's knobs rode d's draft onto CUSUM"


# ================================================================================================
# first-table-made-from-the-form-as-it-stands
# ================================================================================================

def _constant_reference(w: MainWindow, current_pA: float) -> None:
    editor = w._analysis_panel.baseline_panel.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData("constant"))
    editor.current.setText(str(current_pA))
    _commit(w)


def test_a_first_table_made_from_the_form_as_it_stands_ran_it(window, tmp_path):
    """k_in and a constant reference set on a, then a first table made from that very form by a
    door that is neither Detect nor a batch (a script's ``add_result``, here `_landed` with
    nothing laid over the form). It ran every edit the form holds, so the form follows it and
    keeps no draft, as after Detect. Kept as an un-run edit, the reference came back from a look
    away as typed ('user'), not restored from the table (test_open_pore_reference_workflow)."""
    doc_a = _install(window, tmp_path, "a.dat")
    panel = window._analysis_panel
    _k_in, typed = _edited_k_in(window)
    _constant_reference(window, 100)
    res = _landed(window, doc_a, S)
    assert res.params["k_in"] == pytest.approx(typed), "setup: the table is not the form's"

    assert panel.form_key() == (doc_a, 0, S)
    assert not panel._form_dirty and not panel.form_drafts(), "a run of the form kept it as a draft"
    _install(window, tmp_path, "b.dat")
    assert panel.baseline_panel.reference_mode() == "automatic"
    _look(window, doc_a)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert panel._live_baseline_config().open_pore_reference.current_pA == 100
    assert panel.baseline_panel.param_source() == "restored"
    assert not panel._form_dirty


def test_an_ml_edit_keeps_the_draft_over_a_first_table_made_from_the_form(window, tmp_path):
    """The other side: the ML form, which no threshold table records, edited before such a table
    lands. The table did not run that edit, which stays as its draft (13.4 Q6)."""
    doc_a = _install(window, tmp_path, "a.dat")
    _ml_enter(window, 0.85)
    _landed(window, doc_a, S)

    panel = window._analysis_panel
    assert panel.form_key() == (doc_a, 0, S) and panel._form_dirty
    _install(window, tmp_path, "b.dat")
    assert panel.form_drafts()[(doc_a, 0, S)]["ml"]["enter"] == pytest.approx(0.85)
