"""Edit ▸ Undo's journal with both kinds of entry in it: an event step that names its TABLE
and a run's press that names its table AND its form (design §3.1, §3.5, §3.6; the owner's
decisions 1, 2 and 3).

The journal decides which stack a press acts on. Two changes to it were made at the same time
and meet here. One made every event entry name its table, so that Ctrl+Z on the recording on
screen takes back that recording's newest step and not a newer curation made on another one.
The other folded the form steps that led to a run into the run's own entry, so that "change
settings, run, Ctrl+Z" restores table and form in one press. A run that journalled its step
under the old, unnamed spelling would have been dead for the first rule and out of order for
the second.

What is pinned here:

* every journal entry names what it is a step of, so a curation made after a run is taken
  before that run's press, not after it;
* a run's press acts only while its table and its form are the ones on screen: another
  recording's newer step neither takes it nor stands in for it;
* a pending press (redo) keeps each half its own stack still holds: a step on another table,
  another recording or another setting leaves it whole, a new step of its form ends the form's
  half only, and a curation of its table ends the table's half only (decision 2);
* closing a recording takes its runs and the breaks behind its tables out of both journals,
  and leaves everyone else's;
* a press that cannot write a form back changes nothing, in either direction, and says so;
* when the bounded journal has run out the stacks are asked in order, table first, then the
  settings behind it;
* "Undo — nothing on this recording" is never said while the recording on screen holds steps,
  another table's form steps included;
* an edit still in its pause is OLDER than any step that lands inside it, so it is journalled
  first, and Redo is not offered for a form while an edit waits to end its branch;
* a value typed after the redo of a press is not "a form an undo walked back to the seed", so
  the next event undo or redo does not re-seed it away.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.document import TraceDocument
from nanoscope.core.history import MAX_DEPTH
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui.main_window import MainWindow

from .test_editing_form_undo import (
    FS, N, S, _commit, _det, _form, _install, _land, _redo, _run, _table, _trace, _undo,
)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _open_two_channels(w: MainWindow, tmp_path, name: str = "two.dat") -> str:
    """A recording with two current channels, channel 0 on screen."""
    infos = tuple(ChannelInfo(index=i, name=f"current {i}", raw_units="pA",
                              kind=ChannelKind.CURRENT) for i in range(2))
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=infos)
    doc = TraceDocument(meta=meta, channels=(_trace(0), _trace(1)))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    return doc_id


def _reject_first(w: MainWindow) -> None:
    """One curation click on the table on screen, committed."""
    df = w._analysis_panel.current_detection().payload.df
    assert w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    _commit(w)


def _press(w: MainWindow, qtbot) -> tuple:
    """Detect, change k_in, detect again: the second run takes the edit with it as one press.
    Returns ``(first table, first form, k_in before the edit)``."""
    det = _det(w)
    _run(w, qtbot)
    first_table, first_form = _table(w), _form(w)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _run(w, qtbot)
    assert w._act_undo.text() == "Undo Re-run detection and 1 setting change", (
        "setup: the second run did not take its edit into one press")
    return first_table, first_form, k_in


def _named(w: MainWindow) -> bool:
    """Every entry in both journals says what it is a step of: a tuple, bare only for the one
    measurement table."""
    return all(entry == mw._UNDO_MEASUREMENTS or isinstance(entry, tuple)
               for journal in (w._undo_journal, w._redo_journal) for entry in journal)


# ================================================================================================
# every entry names what it is a step of
# ================================================================================================

def test_a_curation_after_a_run_is_taken_before_the_runs_press(window, tmp_path, qtbot):
    """The newest step first, whatever kind it is. A curation journalled without its table is
    dead to the table-on-screen rule, so the journal walked past it to the older press and the
    curation came back only when the journal had nothing left to say."""
    _install(window, tmp_path)
    _press(window, qtbot)
    _reject_first(window)
    assert _named(window), (window._undo_journal, window._redo_journal)

    assert window._act_undo.text() == "Undo Curate event"
    _undo(window)
    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"


def test_a_runs_press_acts_only_while_its_table_and_form_are_on_screen(window, tmp_path, qtbot):
    """Decision 1: another recording's newer step neither takes A's press nor stands in for
    it, and the press is there for A when A is back."""
    doc_a = _install(window, tmp_path, "a.dat")
    first_table, first_form, _k_in = _press(window, qtbot)
    second_table = _table(window)
    doc_b = _install(window, tmp_path, "b.dat")
    _land(window, doc_b, S, min_dwell_s=1e-4)
    b_before = _table(window)
    _reject_first(window)                          # the newest step of all, on B
    assert _named(window)

    assert window._act_undo.text() == "Undo Curate event"
    _undo(window)
    assert _table(window).equals(b_before)
    assert not window._act_undo.isEnabled(), "A's press was offered on B"
    window._on_selection(doc_a, 0)
    assert _table(window).equals(second_table), "Ctrl+Z on B changed A's table"

    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"
    _undo(window)
    assert _table(window).equals(first_table) and _form(window) == first_form
    assert not window._act_undo.isEnabled(), "B's curation was offered on A"
    window._on_selection(doc_b, 0)
    assert _table(window).equals(b_before), "Ctrl+Z on A changed B's table"
    assert window._act_redo.text() == "Redo Curate event"


# ================================================================================================
# a pending press keeps each half its stack still holds (decision 2)
# ================================================================================================

def test_a_step_on_another_recording_leaves_a_pending_press_whole(window, tmp_path, qtbot):
    doc_a = _install(window, tmp_path, "a.dat")
    _press(window, qtbot)
    second_table, second_form = _table(window), _form(window)
    _undo(window)
    assert window._act_redo.text() == "Redo Re-run detection and 1 setting change"

    doc_b = _install(window, tmp_path, "b.dat")
    _land(window, doc_b, S, min_dwell_s=1e-4)
    _reject_first(window)                          # a table step, but not on A's table
    window._on_selection(doc_a, 0)
    assert window._act_redo.text() == "Redo Re-run detection and 1 setting change", (
        "a curation on another recording split A's pending press")
    _redo(window)
    assert _table(window).equals(second_table) and _form(window) == second_form
    assert not window._act_redo.isEnabled()


def test_a_step_of_another_setting_leaves_a_pending_press_whole(window, tmp_path, qtbot):
    doc_id = _install(window, tmp_path)
    _press(window, qtbot)
    second_form = _form(window)
    _undo(window)

    window._session.set_acquisition_bandwidth(doc_id, 0, AcquisitionBandwidth(
        cutoff_hz=10_000.0, status="reported", source="user"))
    _commit(window)
    assert window._act_undo.text() == "Undo Change acquisition bandwidth"
    assert window._act_redo.text() == "Redo Re-run detection and 1 setting change"
    _redo(window)
    assert _form(window) == second_form


def test_a_new_step_of_the_form_ends_only_the_forms_half_of_a_pending_press(
        window, tmp_path, qtbot):
    _install(window, tmp_path)
    _press(window, qtbot)
    second_table = _table(window)
    _undo(window)

    det = _det(window)
    det.k_out.setValue(det.k_out.value() + 0.5)    # a step of the same form
    _commit(window)
    assert window._act_redo.text() == "Redo Re-run detection", (
        "the form's new step took the table's redo with it")
    _redo(window)
    assert _table(window).equals(second_table)
    assert not window._act_redo.isEnabled()


def test_a_curation_of_the_table_ends_only_the_tables_half_of_a_pending_press(
        window, tmp_path, qtbot):
    _install(window, tmp_path)
    _first_table, _first_form, k_in = _press(window, qtbot)
    _undo(window)

    _reject_first(window)                          # a step on the press's own table
    step = f"Change k_in {k_in:g} → {k_in + 1.0:g}"
    assert window._act_redo.text() == f"Redo {step}", (
        "the table's new step took the form's redo with it, or left a press that cannot run")
    _redo(window)
    assert _det(window).k_in.value() == pytest.approx(k_in + 1.0)
    assert not window._act_redo.isEnabled()


def test_a_hand_measurement_ends_only_the_tables_half_of_a_pending_press(
        window, tmp_path, qtbot):
    """The cross-stack rule meets the press: a hand measurement taken on the recording ends
    the redo of its event tables (test_redo_branch), and what is left of the press is the
    form's steps, redoable on their own. No press is left naming a table that cannot come
    back."""
    _install(window, tmp_path)
    _first_table, _first_form, k_in = _press(window, qtbot)
    _undo(window)
    measure = window._analysis_panel.measure_panel
    measure.set_active(True)
    window._trace_view.set_measure_region(25_000 / FS, 25_400 / FS)
    assert measure.add_current() is True
    qtbot.wait(5)

    assert [e for e in window._redo_journal if window._run_of(e) is not None] == []
    assert window._act_redo.text() == f"Redo Change k_in {k_in:g} → {k_in + 1.0:g}"


# ================================================================================================
# closing a recording
# ================================================================================================

def test_closing_a_recording_takes_its_runs_and_breaks_with_it(window, tmp_path, qtbot):
    """B1's lifecycle rule for the entries that came after it: a closed recording's press and
    the break behind its first table name a recording that is gone, and the redo journal is
    never trimmed. Everyone else's stay."""
    def settings_then_first_run(name: str) -> str:
        doc_id = _install(window, tmp_path, name)
        det = _det(window)
        det.k_in.setValue(det.k_in.value() + 1.0)
        _commit(window)
        _run(window, qtbot)                        # the first table is born from those settings
        return doc_id

    doc_a = settings_then_first_run("a.dat")
    det = _det(window)
    det.k_out.setValue(det.k_out.value() + 0.5)
    _run(window, qtbot)                            # A's second run, with its edit: a press
    _undo(window)                                  # ...now pending in the redo journal
    doc_b = settings_then_first_run("b.dat")
    kinds = {entry[0] for entry in window._undo_journal + window._redo_journal
             if isinstance(entry, tuple)}
    assert {mw._UNDO_RUN, mw._UNDO_BREAK} <= kinds, f"setup: {kinds}"

    window._session.close_document(doc_a)
    held = [entry for journal in (window._undo_journal, window._redo_journal)
            for entry in journal if doc_a in repr(entry)]
    assert held == [], "the journal kept entries of a recording that is closed"
    assert any(window._break_of(entry) and window._break_of(entry)[0] == doc_b
               for entry in window._undo_journal), "B's break went with A's recording"
    assert _named(window)


# ================================================================================================
# a press that cannot write back
# ================================================================================================

@pytest.mark.parametrize("verb", ["Undone", "Redone"])
def test_a_press_that_cannot_write_a_form_back_changes_nothing(
        window, tmp_path, qtbot, monkeypatch, verb):
    """All or nothing, as for any settings step (B1): the table and the form stay where they
    were, the journals stay as they were, and the status bar says why."""
    _install(window, tmp_path)
    det = _det(window)
    _run(window, qtbot)
    det.k_in.setValue(det.k_in.value() + 1.0)
    _commit(window)
    det.k_out.setValue(det.k_out.value() + 0.5)
    _run(window, qtbot)                            # commits the second edit, then runs: n = 2
    assert window._act_undo.text() == "Undo Re-run detection and 2 setting changes"
    if verb == "Redone":
        _undo(window)
    table, form = _table(window), _form(window)
    undo_journal, redo_journal = list(window._undo_journal), list(window._redo_journal)

    panel = window._analysis_panel
    real, calls = panel.restore_form_state, []

    def fail_on_the_second_write(state):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the form refused")
        real(state)

    monkeypatch.setattr(panel, "restore_form_state", fail_on_the_second_write)
    (_undo if verb == "Undone" else _redo)(window)
    monkeypatch.undo()

    assert len(calls) >= 2, "setup: the press never reached its second write"
    assert _table(window).equals(table) and _form(window) == form
    assert (window._undo_journal, window._redo_journal) == (undo_journal, redo_journal)
    said = window.statusBar().currentMessage()
    assert said.startswith(f"Nothing was {verb.lower()} — “Re-run detection and 2 setting "
                           f"changes” could not be put back"), said
    assert "the form refused" in said


# ================================================================================================
# the bounded journal runs out
# ================================================================================================

def test_the_journal_stays_within_its_bound_and_the_stacks_carry_the_rest(window, tmp_path):
    """Sixty-one curation steps on one table: the journal keeps its bound, the stack keeps its
    own 30, and the press that walks them back stops where the stack does."""
    doc_id = _install(window, tmp_path)
    _land(window, doc_id, S, min_dwell_s=1e-4)
    row = int(_table(window)["event_id"].iloc[0])
    for i in range(2 * MAX_DEPTH + 1):
        assert window._analysis_panel._table.set_accepted_by_id(row, i % 2 == 1)
        _commit(window)
    assert len(window._undo_journal) == 2 * MAX_DEPTH
    assert _named(window)
    for _ in range(MAX_DEPTH):
        assert window._act_undo.text() == "Undo Curate event"
        _undo(window)
    assert not window._act_undo.isEnabled()


def test_a_press_the_journal_no_longer_holds_is_carried_by_its_stacks(window, tmp_path, qtbot):
    """Past the bound only the stacks are asked, in order: the table on screen first, then
    the settings behind it. The press comes apart into those two steps."""
    _install(window, tmp_path)
    _first_table, first_form, k_in = _press(window, qtbot)
    window._undo_journal.clear()
    window._sync_undo_actions()

    assert window._act_undo.text() == "Undo Re-run detection"
    _undo(window)
    step = f"Change k_in {k_in:g} → {k_in + 1.0:g}"
    assert window._act_undo.text() == f"Undo {step}"
    _undo(window)
    assert _form(window) == first_form
    assert not window._act_undo.isEnabled()


# ================================================================================================
# the menu
# ================================================================================================

def test_nothing_on_this_recording_is_never_said_while_this_recording_holds_steps(
        window, tmp_path):
    """A form's steps wait for ITS table, so the recording on screen can hold steps that are
    not the target: here the form of channel 0's table while channel 1 is on screen. The line
    "nothing on this recording" is about steps of OTHER recordings."""
    doc_id = _open_two_channels(window, tmp_path)
    _land(window, doc_id, S, min_dwell_s=1e-4)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    step = f"Undo Change k_in {k_in:g} → {k_in + 1.0:g}"
    assert window._act_undo.text() == step

    window._on_selection(doc_id, 1)
    assert window._analysis_panel.current_target() == (doc_id, 1), "setup: channel 1 is shown"
    assert not window._act_undo.isEnabled(), "channel 0's form step was offered on channel 1"
    assert window._act_undo.text() == "Undo — nothing to undo on this event table"
    assert window._act_redo.text() == "Redo"

    window._on_selection(doc_id, 0)
    assert window._act_undo.text() == step
    _undo(window)
    window._on_selection(doc_id, 1)
    assert window._act_redo.text() == "Redo — nothing to redo on this event table", (
        "the redo of channel 0's form step is this recording's too")


def test_steps_of_another_recording_alone_still_read_nothing_on_this_recording(
        window, tmp_path):
    doc_a = _install(window, tmp_path, "a.dat")
    _land(window, doc_a, S, min_dwell_s=1e-4)
    det = _det(window)
    det.k_in.setValue(det.k_in.value() + 1.0)
    _commit(window)
    _install(window, tmp_path, "b.dat")            # B holds no step of its own
    assert window._act_undo.text() == "Undo — nothing on this recording"


# ================================================================================================
# an edit still in its pause is older than the step that lands inside it
# ================================================================================================
#
# A form edit becomes a step 700 ms after the last keystroke, or at a run, a switch, Ctrl+Z, the
# Edit menu, a save or a quit; every other step is journalled the moment it is pushed. Anything
# done inside the pause was therefore journalled BEFORE the older edit it found waiting, and the
# journal's one rule -- the newest step first -- ran on the wrong order. The pause is widened in
# each test, so a loaded machine cannot let it run out and make a test pass for the wrong reason.

def _hold_the_pause(w: MainWindow) -> None:
    w._form_pause.setInterval(60_000)


def _apply_gain(w: MainWindow, gain: str) -> None:
    panel = w._calibration_panel
    panel._confirm_sidecar_removal = lambda path, **kw: True
    panel._confirm_note_loss = lambda path, lost, **kw: True
    panel._gain.setText(gain)
    panel._apply()


def _first_event_accepted(w: MainWindow) -> bool:
    df = w._analysis_panel.current_detection().payload.df
    return bool(df["accepted"].iloc[0])


def test_ctrl_z_takes_a_curation_made_inside_the_pause_before_the_edit_it_found_waiting(
        window, tmp_path, qtbot):
    """Measured: type k_in, reject an event 100 ms later, Ctrl+Z. The menu said "Undo Curate
    event" and the press took the edit: the rejection stayed in force."""
    _install(window, tmp_path)
    _run(window, qtbot)
    _hold_the_pause(window)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)                  # an edit, still in its pause
    assert window._form_pause.isActive(), "setup: the edit is no step yet"
    df = window._analysis_panel.current_detection().payload.df
    assert window._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    assert not _first_event_accepted(window)

    assert window._act_undo.text() == "Undo Curate event"
    _undo(window)
    assert _first_event_accepted(window), "the press named the curation and took the edit"
    assert det.k_in.value() == pytest.approx(k_in + 1.0)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 1.0:g}"
    _undo(window)
    assert det.k_in.value() == pytest.approx(k_in)


def test_ctrl_z_takes_a_calibration_applied_inside_the_pause_before_the_edit_it_found_waiting(
        window, tmp_path):
    doc_id = _install(window, tmp_path)
    _hold_the_pause(window)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _apply_gain(window, "2")
    assert window._session.calibration(doc_id, 0).gain == 2.0

    assert window._act_undo.text() == "Undo Apply calibration"
    _undo(window)
    assert window._session.calibration(doc_id, 0).gain == 1.0, (
        "the press named the calibration and took the edit")
    assert det.k_in.value() == pytest.approx(k_in + 1.0)


def test_an_edit_that_a_curation_separated_from_a_run_is_not_part_of_its_press(
        window, tmp_path, qtbot):
    """Design §3.6's last bullet: a setting changed before other work stays a step of its own.
    A run commits an edit still in its pause, and committing it AFTER the curation put it
    right in front of the run, where the press folded it in."""
    _install(window, tmp_path)
    _run(window, qtbot)
    _hold_the_pause(window)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)                  # edit...
    df = window._analysis_panel.current_detection().payload.df
    assert window._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    _run(window, qtbot)                            # ...curate...run, all inside the pause

    assert window._act_undo.text() == "Undo Re-run detection"
    _undo(window)
    assert window._act_undo.text() == "Undo Curate event"
    _undo(window)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 1.0:g}"


def test_redo_is_not_offered_while_a_typed_edit_waits_to_end_its_branch(window, tmp_path):
    """A new edit ends the form's redo the moment it is committed, so a press during the pause
    committed the edit first and then had nothing to redo: the item was enabled, named a step
    and took nothing."""
    _install(window, tmp_path)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    _undo(window)
    step = f"Change k_in {k_in:g} → {k_in + 1.0:g}"
    assert window._act_redo.text() == f"Redo {step}"

    _hold_the_pause(window)
    det.k_in.setValue(k_in + 3.0)                  # a new edit, its pause running
    assert window._form_pause.isActive()
    assert not window._act_redo.isEnabled(), f"Redo offered: {window._act_redo.text()!r}"
    assert step not in window._act_redo.text()

    _commit(window)                                # the edit is a step; the redo is gone
    assert not window._act_redo.isEnabled()
    assert det.k_in.value() == pytest.approx(k_in + 3.0)


def test_a_pending_press_is_named_without_its_form_half_while_an_edit_waits(
        window, tmp_path, qtbot):
    """The press's form half dies when the edit is committed, so the label names the table's
    half alone, and the press does exactly that."""
    _install(window, tmp_path)
    _press(window, qtbot)
    second_table = _table(window)
    _undo(window)
    assert window._act_redo.text() == "Redo Re-run detection and 1 setting change"

    _hold_the_pause(window)
    det = _det(window)
    det.k_in.setValue(det.k_in.value() + 3.0)
    assert window._act_redo.text() == "Redo Re-run detection"
    _redo(window)
    assert window.statusBar().currentMessage() == "Redone: Re-run detection."
    assert _table(window).equals(second_table)


# ================================================================================================
# the form a redone press leaves behind
# ================================================================================================
#
# The redo of a press puts its settings back first and lands its table second, over a form that
# is by then edited and no longer the seed, so the form is KEPT (design §13.2: a same-table result
# does not re-seed edits nobody ran). What `_fill_form` held as "the form the last seed left" was
# still the OLDER table's, and a form that equalled it read as one an undo had walked back to the
# seed.

def test_a_value_typed_after_a_redone_press_survives_an_event_undo(window, tmp_path, qtbot):
    """Measured in the review: run at k_in 5, change it to 3 and run again, Ctrl+Z, Ctrl+Shift+Z,
    type 5 ("let me try 5 again"), reject an event, Ctrl+Z on the curation: the form read 3, and
    the menu still offered "Undo Change k_in 3 → 5" for an edit no longer on the form."""
    _install(window, tmp_path)
    det = _det(window)
    _first_table, _first_form, k_in = _press(window, qtbot)
    _undo(window)
    _redo(window)                                  # the newer run is on screen again
    assert det.k_in.value() == pytest.approx(k_in + 1.0)
    det.k_in.setValue(k_in)                        # the older run's value, typed and not run
    _commit(window)
    _reject_first(window)
    assert window._act_undo.text() == "Undo Curate event"
    _undo(window)                                  # an event undo lands a new result object

    assert det.k_in.value() == pytest.approx(k_in), (
        "the value typed after the redo was replaced by the table's run")
    assert window._act_undo.text() == f"Undo Change k_in {k_in + 1.0:g} → {k_in:g}"
    assert window._analysis_panel.refresh_chain_status().kind == "differs", (
        "the chain notice says nothing about an edit nobody ran")


def test_undo_all_then_redo_all_returns_the_form_it_started_from(window, tmp_path, qtbot):
    _install(window, tmp_path)
    det = _det(window)
    _first_table, _first_form, k_in = _press(window, qtbot)
    det.k_in.setValue(k_in)                        # an edit nobody ran, the first run's value
    _commit(window)
    _reject_first(window)
    before = _form(window)["detection"]

    for _ in range(10):
        if not window._act_undo.isEnabled():
            break
        _undo(window)
    for _ in range(10):
        if not window._act_redo.isEnabled():
            break
        _redo(window)

    assert _form(window)["detection"] == before
