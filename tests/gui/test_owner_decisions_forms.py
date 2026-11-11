"""The owner's answers to design 13.4 Q6 and Q3 and to 13.6's batch detector stamp.

What is pinned here:

* a recording's first table that the form did not make (a batch run, a hand-made event, a
  project restore) keeps an edit typed before it: the edit stays on the form, edited, the chain
  notice names it, its draft is the new table's, and Undo/Redo still work on it -- on screen and
  off. A run from the form still consumes the edit that made it (13.4 Q6);
* the detector combo's re-seed of min dwell and merge gap is no undo step: Undo goes to the
  edit before it and leaves the knobs where the detector put them, an edit still in its pause
  becomes a step of its own, and an edit after the switch is a step (13.4 Q3);
* a batch run records the detector it ran. Batch detect now runs the detector the combo names,
  so in the window the two agree; at the controller the stored params name the module that
  ran whatever the params dict it was handed said. A stored table's Re-run replays its own
  detector whatever its params say (13.6).

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt
from nanoscope.core.history import SettingsHistory
from nanoscope.gui.main_window import MainWindow

from .test_editing_form_undo import (
    C, S, _commit, _det, _form, _install, _redo, _run, _undo,
)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _combo(w: MainWindow, module_id: str) -> None:
    """The detector combo, as the user moves it."""
    combo = _det(w).detector_combo
    combo.setCurrentIndex(combo.findData(module_id))
    QTest.qWait(20)


def _landed(w: MainWindow, doc_id: str, module_id: str = S, **over):
    """A table installed the way a batch pass installs one: the module run with the form's
    params, ``over`` laid on top, through ``Session.add_result``."""
    session = w._session
    params = {**w._analysis_panel._form_params(), "detector": module_id, **over}
    module = DetectSchmitt() if module_id == S else DetectCusum()
    res = module.run(SimpleAnalysisContext(
        session.calibrated(doc_id, 0), session.document(doc_id).meta, params))
    session.add_result(res)
    QTest.qWait(10)
    return res


def _edited_k_in(w: MainWindow) -> tuple[float, float]:
    """k_in moved by one, committed as a step. Returns (before, after)."""
    det = _det(w)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 2.0)
    _commit(w)
    assert w._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 2.0:g}"
    return k_in, k_in + 2.0


# ================================================================================================
# 13.4 Q6: a first table keeps the edit typed before it
# ================================================================================================

def test_a_first_table_landing_on_screen_keeps_the_edit(window, tmp_path):
    """A batch run gives the recording on screen its first table, made with k_in 5. The form
    used to be seeded from it (k_in back to 5, not edited, no draft); the edit was reachable
    only as the table's first undo step."""
    doc_id = _install(window, tmp_path)
    k_in, typed = _edited_k_in(window)
    panel = window._analysis_panel

    _landed(window, doc_id, S, k_in=k_in)
    assert panel.form_key() == (doc_id, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed), "the first table seeded the edit away"
    assert panel._form_dirty, "the edit is not marked edited"
    assert panel.form_drafts()[(doc_id, 0, S)]["detection"]["k_in"] == pytest.approx(typed)
    assert (doc_id, 0, None) not in panel.form_drafts()
    status = panel.refresh_chain_status()
    assert status.kind == "differs" and "k_in" in status.sentence, status.sentence

    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {typed:g}"
    _undo(window)
    assert _det(window).k_in.value() == pytest.approx(k_in)
    _redo(window)
    assert _det(window).k_in.value() == pytest.approx(typed)


def test_a_first_table_landing_off_screen_takes_the_draft(window, tmp_path):
    doc_a = _install(window, tmp_path, "a.dat")
    k_in, typed = _edited_k_in(window)
    _install(window, tmp_path, "b.dat")                # A's edit is filed as its draft
    assert (doc_a, 0, None) in window._analysis_panel.form_drafts()

    _landed(window, doc_a, S, k_in=k_in)               # A's FIRST table, off screen
    drafts = window._analysis_panel.form_drafts()
    assert (doc_a, 0, None) not in drafts
    assert drafts[(doc_a, 0, S)]["detection"]["k_in"] == pytest.approx(typed), (
        "the table that the edit stood for did not take its draft")

    window._on_selection(doc_a, 0)
    QTest.qWait(20)
    assert window._analysis_panel.form_key() == (doc_a, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert window._analysis_panel._form_dirty
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {typed:g}"
    _undo(window)
    assert _det(window).k_in.value() == pytest.approx(k_in)


def test_the_next_run_takes_a_kept_edit_in_one_press(window, tmp_path, qtbot):
    """The kept edit did not make the first table; the next run is the one it leads to, so
    "change settings, run, Ctrl+Z" is one press (3.6), as on a table that already existed."""
    doc_id = _install(window, tmp_path)
    k_in, typed = _edited_k_in(window)
    first = _landed(window, doc_id, S, k_in=k_in)

    _run(window, qtbot)
    assert window._session.results.get(doc_id, 0, S).params["k_in"] == pytest.approx(typed)
    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"
    _undo(window)
    assert window._session.results.get(doc_id, 0, S).params == first.params
    assert _det(window).k_in.value() == pytest.approx(k_in)


def test_a_batch_launched_from_the_tuned_form_ran_the_edit(window, tmp_path, qtbot):
    """The common batch: the recording on screen tuned, then Batch detect. The batch ran that
    form, so its first table holds no un-run edit: the form is not edited, nothing is kept as a
    draft, and the steps that made it stay out of the next run's press."""
    doc_id = _install(window, tmp_path)
    k_in, typed = _edited_k_in(window)
    with qtbot.waitSignal(window._batch_controller.finished, timeout=30_000):
        window._batch_detect()
    qtbot.wait(100)
    assert window._session.results.get(doc_id, 0, S).params["k_in"] == pytest.approx(typed)

    panel = window._analysis_panel
    assert panel.form_key() == (doc_id, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert not panel._form_dirty and panel.form_drafts() == {}
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {typed:g}"
    k_out = _det(window).k_out.value()
    _det(window).k_out.setValue(k_out + 0.5)
    _run(window, qtbot)
    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"
    _undo(window)
    assert _det(window).k_in.value() == pytest.approx(typed)


def test_a_hand_made_first_table_keeps_the_edit(window, tmp_path):
    """Ctrl+M on a recording with no table makes its Manual table: the form keeps the edit."""
    doc_id = _install(window, tmp_path)
    k_in, typed = _edited_k_in(window)

    out = window._analysis_panel.add_manual_event(doc_id, 0, 10_000, 10_300)
    assert out.ok and out.created_table, out.message
    assert window._analysis_panel.form_key() == (doc_id, 0, MANUAL_MODULE_ID)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert window._analysis_panel.form_drafts()[(doc_id, 0, MANUAL_MODULE_ID)][
        "detection"]["k_in"] == pytest.approx(typed)


def test_a_run_from_the_form_still_consumes_the_edit_that_made_its_first_table(
        window, tmp_path, qtbot):
    doc_id = _install(window, tmp_path)
    _k_in, typed = _edited_k_in(window)

    _run(window, qtbot)
    assert window._analysis_panel.form_key() == (doc_id, 0, S)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert not window._analysis_panel._form_dirty
    assert window._analysis_panel.form_drafts() == {}


def test_a_run_from_the_form_that_lands_after_a_look_away_leaves_no_draft(
        window, tmp_path, qtbot):
    """The form's own run, landing off screen, is not a table the edit is kept over."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    _det(window).k_in.setValue(_det(window).k_in.value() + 2.0)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._analysis_panel.trigger_detection()
        window._on_selection(doc_a, 0)
    qtbot.wait(20)
    assert window._session.results.get(doc_b, 0, S) is not None, "setup: the run did not land"
    assert not [key for key in window._analysis_panel.form_drafts() if key[0] == doc_b]


# ================================================================================================
# 13.4 Q3: the detector combo's re-seed is no step
# ================================================================================================

def _knobs(w: MainWindow) -> tuple[float, float]:
    det = _det(w)
    return det.min_dwell_s.value(), det.merge_gap_ms.value()


def test_a_detector_switch_with_no_other_table_is_no_step(window, tmp_path):
    _install(window, tmp_path)
    assert _knobs(window) == (pytest.approx(2e-5), pytest.approx(0.0))

    _combo(window, C)
    assert _knobs(window) == (pytest.approx(1e-3), pytest.approx(0.5)), "setup: no re-seed"
    _commit(window)
    assert not window._act_undo.isEnabled(), window._act_undo.text()
    assert not window._settings_history.scopes_with_history()


def test_undo_after_a_switch_takes_the_edit_before_it_and_keeps_the_detectors_knobs(
        window, tmp_path):
    _install(window, tmp_path)
    k_in, typed = _edited_k_in(window)
    _combo(window, C)
    _commit(window)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {typed:g}"

    _undo(window)
    assert _det(window).k_in.value() == pytest.approx(k_in)
    assert _knobs(window) == (pytest.approx(1e-3), pytest.approx(0.5)), (
        "the undo of an older edit put Schmitt's knobs back under CUSUM")
    _redo(window)
    assert _det(window).k_in.value() == pytest.approx(typed)
    assert _knobs(window) == (pytest.approx(1e-3), pytest.approx(0.5))


def test_an_edit_in_its_pause_at_a_switch_is_its_own_step(window, tmp_path):
    """The 700 ms pause is still running when the combo moves: the edit becomes a step as it
    stood, without the re-seed folded in, and is not lost."""
    _install(window, tmp_path)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 2.0)
    assert window._form_pause.isActive(), "setup: the edit is not in its pause"

    _combo(window, C)
    _commit(window)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 2.0:g}"
    scope = ("form", *window._analysis_panel.form_key())
    assert window._settings_history.undo_depth(scope) == 1
    _undo(window)
    assert det.k_in.value() == pytest.approx(k_in)
    assert _knobs(window) == (pytest.approx(1e-3), pytest.approx(0.5))
    assert not window._act_undo.isEnabled()


def test_an_edit_after_a_switch_is_a_step(window, tmp_path):
    _install(window, tmp_path)
    _combo(window, C)
    det = _det(window)
    cusum_k = det.cusum_k.value()
    det.cusum_k.setValue(cusum_k + 1.0)
    _commit(window)
    assert window._act_undo.text() == f"Undo Change CUSUM k {cusum_k:g} → {cusum_k + 1:g}"
    before = _form(window)
    _undo(window)
    assert det.cusum_k.value() == pytest.approx(cusum_k)
    assert _knobs(window) == (pytest.approx(1e-3), pytest.approx(0.5))
    _redo(window)
    assert _form(window) == before


def test_carry_amends_every_state_of_one_scope_and_keeps_the_labels():
    store = {"v": {"detection": {"a": 1, "b": 0}}}
    history = SettingsHistory(lambda _s: {"detection": dict(store["v"]["detection"])},
                              lambda _s, state: store.update(v=state))
    scope, other = ("form", "doc", 0, None), ("form", "doc", 1, None)
    history.rebase(scope)
    history.rebase(other)
    store["v"] = {"detection": {"a": 2, "b": 0}}
    assert history.checkpoint(scope) and history.checkpoint(other)
    store["v"] = {"detection": {"a": 3, "b": 0}}
    assert history.checkpoint(scope)
    history.undo(scope)

    def amend(state):
        if state["detection"]["b"] == 0:
            state["detection"]["b"] = 9
        return state

    store["v"] = {"detection": {"a": 2, "b": 9}}        # the app moves b, then carries it
    labels = history.undo_label(scope), history.redo_label(scope)
    history.carry(scope, amend)
    assert not history.checkpoint(scope), "the carried state is not the one the scope holds"
    assert (history.undo_label(scope), history.redo_label(scope)) == labels
    assert history.redo(scope) and store["v"] == {"detection": {"a": 3, "b": 9}}
    history.undo(scope)
    history.undo(scope)
    assert store["v"] == {"detection": {"a": 1, "b": 9}}
    history.undo(other)
    assert store["v"] == {"detection": {"a": 2, "b": 0}}, "another scope was carried"


# ================================================================================================
# 13.6: a batch run records the detector it ran
# ================================================================================================

def test_a_batch_run_records_the_detector_it_ran(window, tmp_path, qtbot):
    """Batch detect now runs the detector the combo names (it ran Schmitt whatever the combo
    said when 13.6 was answered), so the detector a table records is still the one that made
    it: CUSUM under a CUSUM combo, Schmitt under a Schmitt one."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    _combo(window, C)                                  # the combo names CUSUM
    with qtbot.waitSignal(window._batch_controller.finished, timeout=30_000):
        window._batch_detect()
    qtbot.wait(100)

    cusum = {}
    for doc_id in (doc_a, doc_b):
        res = cusum[doc_id] = window._session.results.get(doc_id, 0, C)
        assert res is not None, "the batch made no CUSUM table"
        assert res.module_id == C and res.params["detector"] == C, res.params["detector"]
        assert window._session.results.get(doc_id, 0, S) is None
    assert _det(window).detector_choice() == C
    assert window._analysis_panel.current_detection().module_id == C

    _combo(window, S)                                  # ...and now Schmitt
    with qtbot.waitSignal(window._batch_controller.finished, timeout=30_000):
        window._batch_detect()
    qtbot.wait(100)
    for doc_id in (doc_a, doc_b):
        res = window._session.results.get(doc_id, 0, S)
        assert res is not None, "the batch made no Schmitt table"
        assert res.module_id == S and res.params["detector"] == S, res.params["detector"]
        assert window._session.results.get(doc_id, 0, C) is cusum[doc_id]
    assert _det(window).detector_choice() == S
    assert window._analysis_panel.current_detection().module_id == S


@pytest.mark.parametrize("ran, named", [(C, S), (S, C)])
def test_a_batch_tables_params_name_the_detector_that_ran_whatever_the_dict_said(
        window, tmp_path, qtbot, ran, named):
    """13.6 at the controller, where it is implemented: the pass is handed a params dict that
    names the OTHER detector, and every table it stores records the module that ran. (In the
    window the combo and the module agree, so the test above cannot see this line go.) A
    table stamped with a detector that did not make it sends its seed and the stale banner's
    Re-run to the wrong detector."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    params = {**window._analysis_panel.batch_detection_params(), "detector": named}
    how = {} if ran == S else {"module": ran}          # Schmitt is the controller's default
    with qtbot.waitSignal(window._batch_controller.finished, timeout=30_000):
        assert window._batch_controller.start([(doc_a, 0), (doc_b, 0)], params, **how)
    qtbot.wait(100)

    for doc_id in (doc_a, doc_b):
        res = window._session.results.get(doc_id, 0, ran)
        assert res is not None, "the pass stored no table of the module it ran"
        assert res.module_id == ran
        assert res.params["detector"] == ran, (
            f"a {ran} table records detector {res.params['detector']}")
        assert window._session.results.get(doc_id, 0, named) is None


def test_a_stored_tables_rerun_replays_its_own_detector(window, tmp_path):
    """A Schmitt table a batch made before this fix records ``detector: detect.cusum``; its
    stale banner's Re-run would have run CUSUM over it."""
    doc_id = _install(window, tmp_path)
    res = _landed(window, doc_id, S, detector=C)
    assert res.module_id == S and res.params["detector"] == C, "setup: no mislabelled table"
    assert window._analysis_panel._rerun_params(res)["detector"] == S
