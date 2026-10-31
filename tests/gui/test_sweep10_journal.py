"""Sweep 10's journal and form-draft defects: S10-FORM2, S10-LIFE1, S10-UNDO1, S10-UNDO2 and
S10-FORM1 (design §3.4, §3.5, §3.6, §5.2 and 13.4 Q6).

What is pinned here:

* a re-run of a table that already existed is no birth: the recording's no-table form keeps its
  un-run edit and its undo steps, on screen and off (FORM2);
* a run that lands after Ctrl+Z took its form back withdraws the Redo it has made empty, and a
  Redo the landing did not reach stays offered (LIFE1);
* a step on another recording leaves this recording's redo order as Undo left it, for a settings
  step and for the hand-measurement table alike (UNDO1);
* Detect after a Baseline-only edit is one press that takes the table and the setting back,
  whatever the frame diff would have called it (UNDO2);
* a saved draft holding a value of the wrong type is never applied, and a draft the form cannot
  take is dropped without leaving the form half written (FORM1).

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtTest import QTest

from nanoscope.core.document import TraceDocument
from nanoscope.core.form_draft_state import restore_form_drafts
from nanoscope.core.history import SettingsHistory
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

from .test_editing_form_undo import (
    FS, C, S, _commit, _det, _form, _install, _land, _redo, _run, _table, _undo,
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


def _reject_first(w: MainWindow) -> None:
    df = w._analysis_panel.current_detection().payload.df
    assert w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    _commit(w)


def _rejected(w: MainWindow) -> int:
    df = w._analysis_panel.current_detection().payload.df
    return int((~df["accepted"].astype(bool)).sum())


def _menu(w: MainWindow) -> tuple[str, str]:
    _commit(w)
    return w._act_undo.text(), w._act_redo.text()


def _drafts_of(w: MainWindow, doc_id: str) -> dict:
    return {key[1:]: state["detection"].get("cusum_k")
            for key, state in w._analysis_panel.form_drafts().items() if key[0] == doc_id}


def _cusum_edit_on_a_schmitt_recording(w: MainWindow, tmp_path) -> tuple[str, float]:
    """A with a Schmitt table, the combo on CUSUM (no table: key (A, 0, None)) and CUSUM k
    changed without a run. Returns A and the value typed."""
    doc_a = _install(w, tmp_path, "a.dat")
    _land(w, doc_a, S, min_dwell_s=1e-4)
    _combo(w, C)
    assert w._analysis_panel.form_key() == (doc_a, 0, None)
    det = _det(w)
    typed = det.cusum_k.value() + 1.0
    det.cusum_k.setValue(typed)
    _commit(w)
    assert w._act_undo.text().startswith("Undo Change CUSUM k"), w._act_undo.text()
    return doc_a, typed


# ================================================================================================
# S10-FORM2: only a table born takes the no-table form
# ================================================================================================

def test_a_rerun_of_an_existing_table_off_screen_keeps_the_no_table_edit(window, tmp_path):
    """13.4 Q6: a landing on a table that already exists keeps the edit. The Schmitt re-run
    (a batch run, say) used to drop A's CUSUM draft and move its step onto the Schmitt table,
    behind "Undo Re-run detection"."""
    doc_a, typed = _cusum_edit_on_a_schmitt_recording(window, tmp_path)
    doc_b = _install(window, tmp_path, "b.dat")
    assert _drafts_of(window, doc_a) == {(0, None): typed}

    _land(window, doc_a, S, min_dwell_s=2e-4)          # the existing Schmitt table, re-run
    assert _drafts_of(window, doc_a) == {(0, None): typed}, "the re-run dropped the CUSUM edit"

    window._on_selection(doc_a, 0)
    QTest.qWait(20)
    assert window._analysis_panel.current_detection().module_id == S
    _undo(window)                                      # the re-run, alone
    assert _menu(window)[0] != f"Undo Change CUSUM k {typed - 1:g} → {typed:g}", (
        "the CUSUM step was moved onto the Schmitt table")
    _combo(window, C)
    assert _det(window).cusum_k.value() == pytest.approx(typed)
    assert window._act_undo.text() == f"Undo Change CUSUM k {typed - 1:g} → {typed:g}"
    _undo(window)
    assert _det(window).cusum_k.value() == pytest.approx(typed - 1.0)
    del doc_b


def test_a_rerun_of_an_existing_table_on_screen_keeps_the_no_table_edit(window, tmp_path):
    """The same landing while A's CUSUM form is on screen: the edit is kept for the no-table
    key, so a save keeps it and the combo brings it back."""
    doc_a, typed = _cusum_edit_on_a_schmitt_recording(window, tmp_path)

    _land(window, doc_a, S, min_dwell_s=2e-4)
    panel = window._analysis_panel
    drafts = {key[1:]: state for key, state in panel.form_drafts().items() if key[0] == doc_a}
    assert (0, None) in drafts, "the re-run left the CUSUM edit out of the next save"
    assert drafts[(0, None)]["detection"]["cusum_k"] == pytest.approx(typed)

    if panel.form_key()[2] is not None:
        _combo(window, C)
    assert panel.form_key() == (doc_a, 0, None)
    assert _det(window).cusum_k.value() == pytest.approx(typed)
    assert window._act_undo.text() == f"Undo Change CUSUM k {typed - 1:g} → {typed:g}"


def test_a_table_born_off_screen_still_takes_the_first_form(window, tmp_path):
    """The first-table behaviour of 13.4 Q6 is unchanged: a recording's first table, landing
    while another recording is shown, takes the form that stood for it and its steps."""
    doc_a = _install(window, tmp_path, "a.dat")
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    _install(window, tmp_path, "b.dat")
    assert (doc_a, 0, None) in window._analysis_panel.form_drafts()

    _land(window, doc_a, S, min_dwell_s=1e-4)          # A's FIRST table
    assert (doc_a, 0, None) not in window._analysis_panel.form_drafts()
    assert ("form", doc_a, 0, S) in window._settings_history.scopes_with_history()
    assert ("form", doc_a, 0, None) not in window._settings_history.scopes_with_history()


# ================================================================================================
# S10-LIFE1: a landing withdraws the Redo it has made empty
# ================================================================================================

def test_undo_during_a_run_leaves_no_redo_that_does_nothing(window, tmp_path, qtbot):
    _install(window, tmp_path)
    _run(window, qtbot)
    det, panel = _det(window), window._analysis_panel
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
        _undo(window)                                   # Ctrl+Z while the run is in flight
        assert det.k_in.value() == pytest.approx(k_in)
    qtbot.wait(60)
    assert det.k_in.value() == pytest.approx(k_in + 1.0), "setup: the landing re-seeded the form"

    undo_text, redo_text = _menu(window)
    assert undo_text == "Undo Re-run detection"
    assert not redo_text.startswith("Redo Change k_in"), redo_text
    assert not window._act_redo.isEnabled()


def test_a_redo_the_landing_did_not_reach_stays_offered(window, tmp_path, qtbot):
    """Only what the landing already holds is withdrawn: an undone edit the run was not launched
    from is still the user's to redo."""
    _install(window, tmp_path)
    _run(window, qtbot)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    _undo(window)
    assert _menu(window)[1] == f"Redo Change k_in {k_in:g} → {k_in + 1:g}"

    _run(window, qtbot)                                 # a run from the form as it stands
    assert _menu(window)[1] == f"Redo Change k_in {k_in:g} → {k_in + 1:g}"
    _redo(window)
    assert det.k_in.value() == pytest.approx(k_in + 1.0)


def test_drop_redo_reached_drops_down_to_the_state_reached_and_no_further():
    store = {"v": {"a": 1, "b": 1}}
    history = SettingsHistory(lambda _s: dict(store["v"]),
                              lambda _s, state: store.update(v=dict(state)))
    scope = ("form", "doc", 0, S)
    history.rebase(scope)
    for change in ({"a": 2, "b": 1}, {"a": 2, "b": 2}, {"a": 3, "b": 2}):
        store["v"] = dict(change)
        assert history.checkpoint(scope)
    for _ in range(3):
        history.undo(scope)
    assert store["v"] == {"a": 1, "b": 1} and history.redo_depth(scope) == 3

    store["v"] = {"a": 2, "b": 2}                       # a landing put the second step's state
    history.rebase(scope)
    assert history.drop_redo_reached(scope) == 2
    assert history.redo_depth(scope) == 1
    assert history.redo(scope) and store["v"] == {"a": 3, "b": 2}

    assert history.drop_redo_reached(scope) == 0        # nothing left that it reaches


# ================================================================================================
# S10-UNDO1: a step on another recording leaves this recording's redo order alone
# ================================================================================================

def _two_recordings_with_tables(w: MainWindow, tmp_path, qtbot) -> tuple[str, str]:
    doc_a = _install(w, tmp_path, "a.dat")
    _run(w, qtbot)
    doc_b = _install(w, tmp_path, "b.dat")
    _run(w, qtbot)
    w._on_selection(doc_a, 0)
    QTest.qWait(20)
    return doc_a, doc_b


def test_a_curation_on_another_recording_keeps_this_recordings_redo_order(
        window, tmp_path, qtbot):
    """On A: curate, then change k_in; Undo twice. A curation on B used to put A's settings
    step in front: Redo re-applied the older k_in edit before the curation undone after it."""
    doc_a, doc_b = _two_recordings_with_tables(window, tmp_path, qtbot)
    det = _det(window)
    k_in = det.k_in.value()
    _reject_first(window)
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    _undo(window)
    _undo(window)
    assert _rejected(window) == 0 and _menu(window)[1] == "Redo Curate event"

    window._on_selection(doc_b, 0)
    QTest.qWait(20)
    _reject_first(window)
    window._on_selection(doc_a, 0)
    QTest.qWait(20)

    assert _menu(window)[1] == "Redo Curate event"
    _redo(window)
    assert _rejected(window) == 1 and det.k_in.value() == pytest.approx(k_in)
    assert _menu(window)[1] == f"Redo Change k_in {k_in:g} → {k_in + 1:g}"
    _redo(window)
    assert det.k_in.value() == pytest.approx(k_in + 1.0)
    assert _menu(window)[0] == f"Undo Change k_in {k_in:g} → {k_in + 1:g}"


def test_a_curation_on_another_recording_keeps_a_measurement_redo_first(
        window, tmp_path, qtbot):
    """The same inversion with the two tables the app has always had (the skeptic's base
    case): add a hand measurement, curate, Undo twice; a curation on B put the curation
    first."""
    doc_a, doc_b = _two_recordings_with_tables(window, tmp_path, qtbot)
    measure = window._analysis_panel.measure_panel
    measure.set_active(True)
    window._trace_view.set_measure_region(45_000 / FS, 45_400 / FS)
    assert measure.add_current() is True
    measure.set_active(False)
    _commit(window)
    _reject_first(window)
    _undo(window)
    _undo(window)
    assert _menu(window)[1] == "Redo Add measurement"

    window._on_selection(doc_b, 0)
    QTest.qWait(20)
    _reject_first(window)
    window._on_selection(doc_a, 0)
    QTest.qWait(20)

    assert _menu(window)[1] == "Redo Add measurement"
    _redo(window)
    assert _menu(window)[1] == "Redo Curate event"


def _measure(w: MainWindow) -> None:
    measure = w._analysis_panel.measure_panel
    measure.set_active(True)
    w._trace_view.set_measure_region(45_000 / FS, 45_400 / FS)
    assert measure.add_current() is True
    measure.set_active(False)
    _commit(w)


def test_a_step_on_another_recording_ends_only_its_own_measurement_entry(
        window, tmp_path, qtbot):
    """The measurement table is one stack for every recording, and its journal entries name
    none. A curation on B ends B's undone measurement; B's entry used to stay in the redo
    journal while A also had one, so Redo on A replayed A's measurement ahead of the k_in edit
    undone after it (the cross-group review of S10-UNDO1)."""
    doc_a, doc_b = _two_recordings_with_tables(window, tmp_path, qtbot)
    window._on_selection(doc_b, 0)
    QTest.qWait(20)
    _measure(window)                                   # B's measurement
    window._on_selection(doc_a, 0)
    QTest.qWait(20)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)                                    # A's k_in
    _measure(window)                                   # A's measurement
    _undo(window)
    _undo(window)
    window._on_selection(doc_b, 0)
    QTest.qWait(20)
    _undo(window)                                      # B's measurement
    _reject_first(window)                              # ends B's measurement redo
    window._on_selection(doc_a, 0)
    QTest.qWait(20)

    assert _menu(window)[1] == f"Redo Change k_in {k_in:g} → {k_in + 1:g}"
    _redo(window)
    assert det.k_in.value() == pytest.approx(k_in + 1.0)
    assert _menu(window)[1] == "Redo Add measurement"


# ================================================================================================
# S10-UNDO2: a run is one press whatever its frame diff is called
# ================================================================================================

_N_FILTERED = 300_000


def _filtered_trace(seed: int) -> np.ndarray:
    """24 events of mixed depth and dwell with every edge smoothed (a 6-sample boxcar at 50 kHz,
    as a filtered amplifier gives finite rise times), slow drift, white noise: a Baseline-only
    change then moves a few spans and the diff names the run "Resize N event masks"."""
    rng = np.random.default_rng(seed)
    t = np.arange(_N_FILTERED)
    x = np.full(_N_FILTERED, 100.0) + 3.0 * np.sin(2 * np.pi * t / 120_000.0 + seed)
    starts = np.sort(rng.choice(np.arange(5_000, _N_FILTERED - 10_000, 11_000), 24,
                                replace=False))
    for s in starts:
        dwell = int(rng.integers(40, 2_000))
        x[s:s + dwell] -= float(rng.uniform(15.0, 45.0))
    x = np.convolve(x, np.ones(6) / 6.0, mode="same")
    x += rng.normal(0.0, 1.2, _N_FILTERED)
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.mark.parametrize("knob,factor", [("rough_s", 4.0), ("base_s", 2.0)])
def test_detect_after_a_baseline_only_edit_is_one_press(window, tmp_path, qtbot, knob, factor):
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "filtered.dat", format_name="synthetic",
                         n_samples=_N_FILTERED, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = window._session.add_document(TraceDocument(meta=meta, channels=(_filtered_trace(0),)))
    window._create_default_layers(window._session.document(doc_id))
    window._on_selection(doc_id, 0)
    _run(window, qtbot)
    first_table, first_form = _table(window), _form(window)
    box = getattr(window._analysis_panel.baseline_panel, knob)
    before = box.value()
    box.setValue(before * factor)
    _run(window, qtbot)
    assert window._history.last_step_run

    assert _menu(window)[0] == "Undo Re-run detection and 1 setting change"
    _undo(window)
    assert _table(window).equals(first_table)
    assert box.value() == pytest.approx(before), "the setting needed a second press"
    assert _form(window) == first_form


# ================================================================================================
# S10-FORM1: a draft the form cannot take is never half applied
# ================================================================================================

def _record(key: str, detection: dict) -> dict:
    return {"doc_id": key, "channel": 0, "channel_name": "current", "module_id": S,
            "state": {"detection": {"k_in": 6.0, "k_out": 1.5, "max_dwell_s": None,
                                    "polarity": None, "notch_enabled": False,
                                    "filter_kind": "bessel", **detection},
                      "baseline": {"rough_s": 0.5, "base_s": 0.1, "source": "user"}}}


@pytest.mark.parametrize("bad", ["abc", None, [1, 2], True])
def test_a_wrong_typed_draft_value_is_not_applied(tmp_path, bad):
    path = tmp_path / "pore.csv"
    path.write_text("time,current\n0,1\n")
    key = str(path.resolve())

    drafts, said = restore_form_drafts([_record(key, {"k_in": bad})], key, {0: "current"})
    assert drafts == {}, "a value the form cannot take was applied"
    assert any("not ones this version can read" in m for m in said), said

    good, said = restore_form_drafts([_record(key, {})], key, {0: "current"})
    assert set(good) == {(0, S)} and said == (), "the None sentinels are values the form takes"


def test_a_draft_the_form_cannot_take_is_dropped_whole(window, tmp_path):
    """Past the record check, the form's own restore is the last guard: a draft it refuses
    half way leaves the form as an arrival with no draft would, and says so."""
    doc_id = _install(window, tmp_path)
    _land(window, doc_id, S, min_dwell_s=1e-4)
    _land(window, doc_id, C, min_dwell_s=2e-3)
    panel = window._analysis_panel
    panel._result_selector.setCurrentIndex(panel._result_selector.findData(S))
    QTest.qWait(20)
    seeded = _form(window)
    draft = {**seeded, "detection": {**seeded["detection"], "k_in": 9.0},
             "baseline": {**seeded["baseline"], "accepted_condemned": 5}}   # not iterable

    panel._result_selector.setCurrentIndex(panel._result_selector.findData(C))
    QTest.qWait(20)
    panel.set_form_drafts({(doc_id, 0, S): draft})
    window.statusBar().clearMessage()
    panel._result_selector.setCurrentIndex(panel._result_selector.findData(S))   # must not raise
    QTest.qWait(20)

    assert panel.form_key() == (doc_id, 0, S)
    assert _form(window) == seeded, "the refused draft left part of itself on the form"
    assert not panel._form_dirty
    assert (doc_id, 0, S) not in panel.form_drafts()
    assert "could not be put back" in window.statusBar().currentMessage()
