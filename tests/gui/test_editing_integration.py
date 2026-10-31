"""The editing foundation, merged: what only the whole tree can show.

Wave C's parts were built and tested apart: the Analysis panel's form history and drafts, the
main window's form scopes and its one journal, the stage controls and the settings notice, the
dependency wiring. Each part's tests ran against a stand-in for the others or through a spy on a
signal. What is pinned here is the seams between them, through the real MainWindow, its Signal
Conditioning window and Edit ▸ Undo / Redo (``QAction.trigger``, the door a key press and a
menu click both reach):

* **A Revert of the chain or of the settings, a Revert stage and a Reset stage are each ONE named
  step**, taken back by one Undo and put back by one Redo. The stage controls pinned only the
  label each action names its step by (``formStepNamed``). A Revert writes many keys, each of
  which starts the edit pause, so what the user sees is whether the step it leaves is a single
  one, and whether a second lands when the pause ends.
* **⌘Z in the Signal Conditioning window takes back the chain edit made there.** Until chain
  edits were steps, the press took back whatever came last, curation and calibration included.
* **⌘Z reaches a form's first edit inside its pause.** An edit is a step 700 ms after the last one,
  and the key press is one of the moments that commit it earlier, but a disabled Undo never receives
  the key: with nothing else to take back it was dead until the pause ran out.
* **A table born by a landing is announced as born, and nothing else is.** The steps made before
  a recording and channel had a table move onto its first table only when the arriving table is
  the one the session is announcing; a switch to a table that already existed moves nothing.
* **The chain notice and the amber line's Details list judge a legacy table alike** (``params``
  empty, provenance only), Schmitt and CUSUM, for every stage and threshold the notice names.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``; the detectors run
synchronously through their modules, so every table carries a real run's record.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis import conditioning_recipes as cr
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt, register_all
from nanoscope.gui.main_window import _FORM_COALESCE_MS, MainWindow

from .test_editing_stage_resets import (
    C, RUN_CHAIN, RUN_CUSUM_H, S, _open, _run, _settle, _show_table,
)


@pytest.fixture
def window(qapp, qtbot):
    register_all()
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def runs(window, tmp_path):
    """A recording with a Schmitt and a CUSUM table made with ``RUN_CHAIN``, the Schmitt table
    on screen with its form as the run left it, and the Signal Conditioning window open.

    The chain goes into the runs and not through the form, so the form holds no edit and the
    form's history no step before a test starts."""
    doc_id = _open(window, tmp_path)
    window._session.add_result(_run(window, doc_id, DetectSchmitt(), **RUN_CHAIN))
    window._session.add_result(_run(window, doc_id, DetectCusum(), detector=C,
                                    cusum_h=RUN_CUSUM_H, **RUN_CHAIN))
    _show_table(window, C)
    _show_table(window, S)
    window._open_signal_conditioning()
    _settle(window)
    ap = window._analysis_panel
    assert ap.chain_status().kind == "match", ap.chain_status().sentence
    assert window._settings_history.scopes_with_history() == (), "setup: the form holds a step"
    return window, doc_id


@pytest.fixture
def shown(window, tmp_path):
    """A recording with a Schmitt table made with ``RUN_CHAIN`` on screen: a form with no step
    and no Signal Conditioning window."""
    doc_id = _open(window, tmp_path)
    window._session.add_result(_run(window, doc_id, DetectSchmitt(), **RUN_CHAIN))
    _settle(window)
    assert window._settings_history.scopes_with_history() == (), "setup: the form holds a step"
    return window, doc_id


def _commit(w: MainWindow) -> None:
    """Opening the Edit menu: one of the four moments that commit an edit in its pause."""
    w._on_edit_menu_about_to_show()


def _depth(w: MainWindow) -> int:
    """How many steps the form on screen holds."""
    return w._settings_history.undo_depth(w._form_scope())


def _after_the_pause(w: MainWindow) -> None:
    """Let the edit pause run out, as it does for a user who has stopped typing: a step that
    was going to land late lands now."""
    QTest.qWait(_FORM_COALESCE_MS + 100)
    _commit(w)


# ================================================================================================
# a Revert and a Reset are one step end to end
# ================================================================================================

@pytest.mark.parametrize("edit, label", [
    ({"notch_q": 20.0, "filter_cutoff_hz": 5_000.0}, "Revert chain to Schmitt's run"),
    ({"notch_q": 20.0, "k_in": 6.0}, "Revert settings to Schmitt's run"),
], ids=["chain", "settings"])
def test_a_revert_is_one_named_step_that_undo_and_redo_take_back(runs, edit, label):
    """The window's Revert writes every key that differs, and each write starts the edit
    pause. One press of Undo must take all of it back, and no second step may be left behind
    for the pause to record. The edit before it stays a step of its own."""
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    run = det.params()

    det.set_params(edit)
    _settle(window)                             # the window is pushed the verdict it enables on
    _commit(window)
    edited, edit_step = det.params(), window._act_undo.text()
    assert edit_step.startswith("Undo Change"), edit_step
    depth = _depth(window)

    assert ws.revert_button.isEnabled()
    ws.revert_button.click()
    _after_the_pause(window)
    assert det.params() == run, "Revert did not put the run's settings back"
    assert _depth(window) == depth + 1, "a Revert left more than one step"
    assert window._act_undo.text() == f"Undo {label}"

    window._act_undo.trigger()
    assert det.params() == edited, "one Undo did not take the whole Revert back"
    assert window._act_undo.text() == edit_step, "the Revert took more than its own step"
    assert window._act_redo.text() == f"Redo {label}"
    _after_the_pause(window)
    assert window._act_redo.text() == f"Redo {label}", "the undo recorded a step of its own"

    window._act_redo.trigger()
    assert det.params() == run, "one Redo did not put the whole Revert back"
    assert window._act_undo.text() == f"Undo {label}"
    assert _depth(window) == depth + 1


@pytest.mark.parametrize("button, label, writes", [
    ("stage_revert_button", "Revert Notch stage", lambda before: {"notch_q": 30.0}),
    ("stage_reset_button", "Reset Notch stage", lambda before: {
        key: value for key, value in cr.reset_stage_params("notch").items()}),
], ids=["revert", "reset"])
def test_a_stage_revert_and_a_stage_reset_are_one_named_step_each(runs, button, label, writes):
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel

    det.set_params({"notch_q": 20.0, "filter_cutoff_hz": 5_000.0})
    _settle(window)
    _commit(window)
    before = det.params()
    depth = _depth(window)

    det.select_stage("notch")
    assert getattr(ws, button).isEnabled()
    getattr(ws, button).click()
    _after_the_pause(window)
    after = det.params()
    expected = {**before, **writes(before)}
    assert after == expected, "the stage action wrote more or less than its own stage"
    assert after["filter_cutoff_hz"] == 5_000.0, "the other stage was touched"
    assert _depth(window) == depth + 1, "a stage action left more than one step"
    assert window._act_undo.text() == f"Undo {label}"

    window._act_undo.trigger()
    assert det.params() == before, "one Undo did not take the whole stage action back"
    assert window._act_undo.text().startswith("Undo Change"), "it took more than its own step"

    window._act_redo.trigger()
    assert det.params() == after
    assert window._act_undo.text() == f"Undo {label}"


def test_ctrl_z_in_the_signal_conditioning_window_takes_back_the_chain_edit_made_there(runs):
    """The press from that window is Edit ▸ Undo's, and since a chain edit is a step of the
    table's form it is the newest step on the recording on screen: the first press takes the
    edit back and says so, the next puts it back."""
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    q = det.params()["notch_q"]

    det.notch_q.setValue(q - 10.0)              # a stage's own box, in the window
    ws.act_undo.trigger()
    assert det.params()["notch_q"] == q, "⌘Z in the window did not take back the edit made there"
    assert window.statusBar().currentMessage().startswith("Undone: Change notch Q"), (
        window.statusBar().currentMessage())
    ws.act_redo.trigger()
    assert det.params()["notch_q"] == q - 10.0


def test_ctrl_z_inside_the_pause_of_a_forms_first_edit_takes_the_edit_back(shown):
    """The key press itself, as a user makes it: the edit is not a step yet, and the press commits
    it and takes it back, with the menu reading a bare "Undo" until it is committed."""
    window, _ = shown
    det = window._analysis_panel.detection_panel
    q = det.params()["notch_q"]

    det.notch_q.setValue(q + 5.0)
    assert window._form_pause.isActive()
    assert window._settings_history.scopes_with_history() == (), "setup: the edit is a step"
    assert window._act_undo.isEnabled() and window._act_undo.text() == "Undo", (
        "Undo was not offered for an edit in its pause: " + window._act_undo.text())

    window.activateWindow()
    QTest.keyClick(window, Qt.Key_Z, Qt.ControlModifier)
    assert det.params()["notch_q"] == q, "Ctrl+Z inside the pause did nothing"
    assert window._act_redo.text().startswith("Redo Change notch Q"), window._act_redo.text()
    assert not window._act_undo.isEnabled()


def test_an_edit_that_came_to_nothing_stops_offering_undo_when_its_pause_ends(shown):
    """Undo is offered for an edit while it waits and for no longer than that: put back by hand,
    the edit leaves no step, and the menu says so once the pause is over."""
    window, _ = shown
    det = window._analysis_panel.detection_panel
    q = det.params()["notch_q"]

    det.notch_q.setValue(q + 5.0)
    det.notch_q.setValue(q)
    assert window._act_undo.isEnabled()
    QTest.qWait(_FORM_COALESCE_MS + 100)
    assert not window._form_pause.isActive()
    assert not window._act_undo.isEnabled() and window._act_undo.text() == "Undo"
    assert window._settings_history.scopes_with_history() == ()


# ================================================================================================
# a table born by a landing
# ================================================================================================

def test_a_table_born_by_a_landing_is_announced_while_the_session_names_it(window, tmp_path):
    """What ``_on_form_arrived`` decides a first table by: the form of the table that leaves
    is still named by ``form_key`` while ``formLeaving`` is announced, and ``formArrived`` comes
    while ``Session.announcing`` is the table that has just landed."""
    doc_id = _open(window, tmp_path)
    ap = window._analysis_panel
    stood_for_it = (doc_id, 0, None)
    assert ap.form_key() == stood_for_it
    leaving, arrived = [], []
    ap.formLeaving.connect(lambda key: leaving.append((key, ap.form_key())))
    ap.formArrived.connect(
        lambda old, new: arrived.append((old, new, window._session.announcing)))

    window._session.add_result(_run(window, doc_id, DetectSchmitt()))
    born = (doc_id, 0, S)
    assert leaving == [(stood_for_it, stood_for_it)], (
        "the form key had already moved when its form was announced as leaving")
    assert arrived == [(stood_for_it, born, born)], (
        "a table's birth was not announced while the session named it")


def test_a_switch_to_a_table_that_already_exists_takes_over_no_steps(window, tmp_path):
    """The detector combo can move the form to a detector with no table here, ``(doc, ch,
    None)``, and back to one that has: the second is a switch, not a birth. The steps made on
    the form with no table stay with it, and come back when that form does."""
    doc_id = _open(window, tmp_path)
    ap = window._analysis_panel
    det = ap.detection_panel
    window._session.add_result(_run(window, doc_id, DetectSchmitt(), **RUN_CHAIN))
    none_scope, schmitt_scope = ("form", doc_id, 0, None), ("form", doc_id, 0, S)

    det.set_detector(C)                         # CUSUM has no table on this recording
    assert ap.form_key() == (doc_id, 0, None)
    det.notch_q.setValue(det.params()["notch_q"] + 5.0)
    _commit(window)
    assert window._settings_history.undo_depth(none_scope) == 1
    step = window._act_undo.text()
    assert step.startswith("Undo Change notch Q"), step

    det.set_detector(S)
    assert ap.form_key() == (doc_id, 0, S)
    assert window._settings_history.undo_depth(none_scope) == 1, "the step left its form"
    assert window._settings_history.undo_depth(schmitt_scope) == 0, (
        "a switch to Schmitt's table took over the steps made where there was none")
    assert not window._act_undo.isEnabled(), "Ctrl+Z reached a form that is not on screen"

    det.set_detector(C)
    assert window._act_undo.text() == step


# ================================================================================================
# a legacy table: the notice and the Details list read one record
# ================================================================================================

#: Each edit of the form that the chain notice can name, as ``set_params`` takes it. ``k_in`` is
#: Schmitt's threshold and ``cusum_h`` / ``cusum_k`` CUSUM's: each detector judges only its own.
_EDITS = {
    "k_in": {"k_in": 6.0},
    "cusum_h": {"cusum_h": RUN_CUSUM_H + 4.0},
    "cusum_k": {"cusum_k": 1.5},
    "min dwell": {"min_dwell_s": 3e-3},
    "merge gap": {"merge_gap_s": 2e-3},
    "low-pass": {"filter_cutoff_hz": 5_000.0},
    "notch": {"notch_q": 20.0},
    "median": {"median_enabled": True, "median_window_s": 5e-4},
}


@pytest.mark.parametrize("module_id", [S, C])
def test_the_notice_and_the_details_list_judge_a_legacy_table_alike(runs, module_id):
    """A v1/v2 project restores a table with ``params={}``: its settings are the detector's
    stamp in the provenance, read by the chain notice through the panel and by the Details
    list through ``analysis.lineage``. For every edit they must name the same differences, the
    thresholds a CUSUM table was run with among them."""
    window, doc_id = runs
    ap = window._analysis_panel
    det = ap.detection_panel
    legacy = replace(window._session.results.get(doc_id, 0, module_id), params={})
    window._session.add_result(legacy)
    _show_table(window, module_id)
    shown = det.params()

    for name, edit in _EDITS.items():
        det.set_params(shown)
        det.set_params(edit)
        status = ap.refresh_chain_status()
        rows = {line[2:].split(":", 1)[0]: line
                for line in ap._stale_inputs_text(legacy).splitlines()[1:]}
        chain = [d for d in status.diffs if d.stage != "thresholds"]
        thresholds = [d for d in status.diffs if d.stage == "thresholds"]
        for title, found in (("Signal chain", chain), ("Thresholds", thresholds)):
            row = rows[title]
            assert ("→" in row) == bool(found), (
                f"{name}: the notice says {[d.text for d in status.diffs]}, "
                f"the Details list says {row!r}")
            for diff in found:
                named = (cr.STAGE_TITLES[diff.stage] if diff.stage in cr.STAGE_TITLES
                         else diff.text.partition(" → ")[0])
                assert named.lower() in row.lower(), (name, diff.text, row)
    det.set_params(shown)
    assert ap.refresh_chain_status().kind == "match"
