"""Edits stay with their table: the Analysis panel's side of the form history (design §3.4,
§5.2 and §5.3; the owner's decision 4).

The Detection, ML and Baseline forms are ONE set of widgets showing whichever event table is
on screen. On ``27ec0eb`` every switch re-seeded them from the stored run of the table
arriving, so an edit nobody had run yet was gone after a look elsewhere: a low-pass typed on
A came back as A's run's low-pass after a look at B, and when A had no run the form kept B's
numbers. A Table-selector switch threw the edit away the same way.

What is pinned here:

* each form key -- the table on screen, ``(doc_id, channel, module_id)``, or
  ``(doc_id, channel, None)`` with no table -- keeps the form the user left as its draft,
  and gets it back on return, for a switch of recording, of table and of detector;
* the draft is taken BEFORE anything moves the form: a 20 kHz typed on a 50 kHz recording
  comes back after a 25 kHz one, and a clamp the draft brings back is still announced;
* a run this panel started from the form clears the draft of the key it lands on; a table
  written by anything else leaves edits nobody ran alone;
* the signals MainWindow builds the history from: ``formLeaving`` before the form changes
  and ``formArrived`` after, with the right keys; seeds, draft restores and
  ``restore_form_state`` are never ``formEdited``, and a seed is ``formWritten``; a recall,
  a preset load and both baseline resets name their step (``formStepNamed``);
* ``restore_form_state(form_state())`` is an exact round trip, the frequency boxes' set
  values included -- the undo of an edit made to a clamped box puts the typed value back;
* a closed recording's drafts go with it.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``. Driven through a
real MainWindow, the way a user switches: ``_on_selection``, the Table selector, the
detector combo.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6.QtCore import Qt

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS, SLOW = 50_000.0, 25_000.0        # Nyquist 25 kHz and 12.5 kHz
I0 = -1_000.0
N = 60_000
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600), (30_000, 30_500)]
S, C = "detect.schmitt", "detect.cusum"
HOST_GUEST = "Host-guest (CB7 + adamantyl)"
_CLAMP = ("Low-pass 20 kHz is above this recording's Nyquist (12.5 kHz); the form now "
          "reads 12.4999 kHz.")


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _open(w: MainWindow, tmp_path, name: str, *, fs: float = FS) -> str:
    """Open a synthetic one-channel recording at ``tmp_path / name`` and put it on screen."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc_id = w._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    w._create_default_layers(w._session.document(doc_id))
    w._on_selection(doc_id, 0)
    return doc_id


def _detect(w: MainWindow, doc_id: str, module_id: str = S, **params):
    """Store a table on ``doc_id`` the way a batch pass does -- through the module and
    ``Session.add_result``, NOT from this panel -- run with the form's settings and
    ``params`` over them."""
    session = w._session
    run = {**w._analysis_panel._form_params(), "detector": module_id, **params}
    module = DetectSchmitt() if module_id == S else DetectCusum()
    res = module.run(SimpleAnalysisContext(
        session.calibrated(doc_id, 0), session.document(doc_id).meta, run))
    session.add_result(res)
    return res


def _type(qtbot, box, text: str) -> None:
    """Type ``text`` into a spin box and commit it with Return, as a user does."""
    box.selectAll()
    qtbot.keyClicks(box, text)
    qtbot.keyClick(box, Qt.Key_Return)


def _low_pass(qtbot, det, text: str) -> None:
    det.filter_enabled.setChecked(True)
    _type(qtbot, det.filter_cutoff_hz, text)


def _status(w: MainWindow) -> str:
    return w.statusBar().currentMessage()


def _run_detect(w: MainWindow, qtbot) -> None:
    """Detect, through this panel, with the form on screen; wait for the table to land."""
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)


# ================================================================================================
# an edit stays with its table (design §5.2)
# ================================================================================================

@pytest.mark.parametrize("a_run, b_run", [(True, True), (True, False), (False, True),
                                          (False, False)],
                         ids=["both-run", "A-run", "B-run", "no-run"])
def test_a_low_pass_edit_on_a_survives_a_look_at_b_and_back(
        window, tmp_path, qtbot, a_run, b_run):
    """On ``27ec0eb`` the three cases with a stored run lose the edit: the return to A
    re-seeds A from its run, or, with no run on A, the form keeps B's. With neither, the
    edit survived only because the form carried it across (the control)."""
    a = _open(window, tmp_path, "a.dat")
    if a_run:
        _detect(window, a, filter_cutoff_hz=None)
    b = _open(window, tmp_path, "b.dat")
    if b_run:
        _detect(window, b, filter_cutoff_hz=8_000.0)
    det = window._analysis_panel.detection_panel

    window._on_selection(a, 0)
    _low_pass(qtbot, det, "5000")
    window._on_selection(b, 0)
    if b_run:
        assert det.filter_cutoff_hz.value() == pytest.approx(8_000.0), (
            "B did not show its own run's settings")
    window._on_selection(a, 0)

    assert det.filter_enabled.isChecked()
    assert det.filter_cutoff_hz.value() == pytest.approx(5_000.0), (
        "the low-pass typed on A was lost by a look at B")


def test_a_table_selector_switch_schmitt_cusum_schmitt_keeps_schmitts_edit(
        window, tmp_path, qtbot):
    doc = _open(window, tmp_path, "a.dat")
    _detect(window, doc, S, filter_cutoff_hz=None, k_in=5.0)
    _detect(window, doc, C, filter_cutoff_hz=8_000.0, min_dwell_s=1e-3, merge_gap_s=5e-4)
    ap = window._analysis_panel
    det = ap.detection_panel
    selector = ap._result_selector
    selector.setCurrentIndex(selector.findData(S))
    assert ap.current_detection().module_id == S
    assert not det.filter_enabled.isChecked()

    _low_pass(qtbot, det, "5000")
    _type(qtbot, det.k_in, "7")
    selector.setCurrentIndex(selector.findData(C))
    assert ap.current_detection().module_id == C
    assert det.filter_cutoff_hz.value() == pytest.approx(8_000.0), (
        "CUSUM's table did not bring CUSUM's own settings")
    assert det.min_dwell_s.value() == pytest.approx(1e-3)

    selector.setCurrentIndex(selector.findData(S))
    assert ap.current_detection().module_id == S
    assert det.filter_enabled.isChecked()
    assert det.filter_cutoff_hz.value() == pytest.approx(5_000.0)
    assert det.k_in.value() == pytest.approx(7.0), "Schmitt's edit did not survive the switch"
    assert det.min_dwell_s.value() == pytest.approx(2e-5), (
        "Schmitt's form came back with CUSUM's minimum dwell")


def test_a_detector_combo_switch_keeps_the_edit_and_not_the_combos_own_reseed(
        window, tmp_path, qtbot):
    """The combo re-seeds the two knobs the detectors share as it moves, each one that
    still holds the other detector's default. That is the app's doing, not the user's:
    Schmitt's draft keeps Schmitt's minimum dwell, and a merge gap typed on Schmitt that
    happens to be CUSUM's default is not re-seeded over once Schmitt's form is back."""
    doc = _open(window, tmp_path, "a.dat")
    _detect(window, doc, S, k_in=5.0)                  # CUSUM has never run here
    ap = window._analysis_panel
    det = ap.detection_panel
    assert det.min_dwell_s.value() == pytest.approx(2e-5)
    _type(qtbot, det.k_in, "7")
    _type(qtbot, det.merge_gap_ms, "0.5")               # CUSUM's default, typed on Schmitt

    det.set_detector(C)
    assert ap.current_detection() is None
    assert det.min_dwell_s.value() == pytest.approx(1e-3), (
        "CUSUM's first form did not get CUSUM's own default minimum dwell")
    det.set_detector(S)
    assert ap.current_detection().module_id == S
    assert det.k_in.value() == pytest.approx(7.0), "Schmitt's edit did not survive the combo"
    assert det.min_dwell_s.value() == pytest.approx(2e-5), (
        "Schmitt's form came back with the combo's re-seed in it")
    assert det.merge_gap_ms.value() == pytest.approx(0.5), (
        "the combo re-seeded the merge gap over Schmitt's form after it came back")


def test_a_combo_switch_files_the_form_as_it_stood_before_the_combo_moved(
        window, tmp_path, qtbot):
    """What leaves is filed -- in the draft, and in what the form reads while `formLeaving`
    is announced -- with the shared knobs the user left, not the combo's re-seed of them."""
    doc = _open(window, tmp_path, "a.dat")
    _detect(window, doc, S, k_in=5.0)
    ap = window._analysis_panel
    det = ap.detection_panel
    _type(qtbot, det.k_in, "7")
    filed: list[tuple] = []
    ap.formLeaving.connect(lambda key: filed.append(
        (key, ap.form_key(), ap.form_state()["detection"]["min_dwell_s"])))

    det.set_detector(C)
    assert filed == [((doc, 0, S), (doc, 0, S), pytest.approx(2e-5))]
    assert ap.form_key() == (doc, 0, None)
    draft = ap.form_drafts()[(doc, 0, S)]
    assert draft["detection"]["k_in"] == pytest.approx(7.0)
    assert draft["detection"]["min_dwell_s"] == pytest.approx(2e-5)
    assert draft["detection"]["merge_gap_s"] is None
    assert det.min_dwell_s.value() == pytest.approx(1e-3)


def test_a_run_from_the_form_clears_the_draft_of_the_key_it_lands_on(window, tmp_path, qtbot):
    doc = _open(window, tmp_path, "a.dat")
    _detect(window, doc, S)
    ap = window._analysis_panel
    det = ap.detection_panel
    key = (doc, 0, S)
    _low_pass(qtbot, det, "5000")
    assert key in ap.form_drafts(), "an edit nobody ran is a draft of the table on screen"

    edited: list[int] = []
    written: list[int] = []
    ap.formEdited.connect(lambda: edited.append(1))
    ap.formWritten.connect(lambda: written.append(1))
    _run_detect(window, qtbot)

    assert ap.current_detection().params["filter_cutoff_hz"] == pytest.approx(5_000.0)
    assert ap.form_drafts() == {}, "the form produced the run, so it is no draft any more"
    assert edited == [] and written, "the landing re-seed must be the app's write"


def test_a_table_written_by_anything_else_leaves_edits_nobody_ran_alone(
        window, tmp_path, qtbot):
    """A batch pass, an event undo, a merge: a new table under the form re-seeded it, and
    the edit was gone with nothing on screen saying so. It stays, and the chain notice
    says how it differs from the table now shown."""
    doc = _open(window, tmp_path, "a.dat")
    _detect(window, doc, S)
    ap = window._analysis_panel
    det = ap.detection_panel
    _low_pass(qtbot, det, "5000")

    _detect(window, doc, S, filter_cutoff_hz=8_000.0)   # written by something else
    assert ap.current_detection().params["filter_cutoff_hz"] == pytest.approx(8_000.0)
    assert det.filter_cutoff_hz.value() == pytest.approx(5_000.0), (
        "a table written elsewhere threw away the edit on the form")
    assert ap.refresh_chain_status().kind == "differs"
    assert (doc, 0, S) in ap.form_drafts()


def test_a_form_an_undo_walked_back_follows_a_new_table(window, tmp_path, qtbot):
    """Edited, but back to what the last seed put there (an undo writes through
    ``restore_form_state``): nothing nobody ran is lost, so the form follows the table."""
    doc = _open(window, tmp_path, "a.dat")
    _detect(window, doc, S)
    ap = window._analysis_panel
    det = ap.detection_panel
    seeded = ap.form_state()
    _low_pass(qtbot, det, "5000")
    ap.restore_form_state(seeded)                        # what Edit ▸ Undo writes
    assert not det.filter_enabled.isChecked()

    _detect(window, doc, S, filter_cutoff_hz=8_000.0)
    assert det.filter_enabled.isChecked()
    assert det.filter_cutoff_hz.value() == pytest.approx(8_000.0)


def test_a_table_written_under_an_edited_form_does_not_make_an_older_seed_current(
        window, tmp_path, qtbot):
    """A form kept over a new table is the user's, and the seed it was compared with belongs
    to an older table. A value typed back to that older seed is an edit like any other, not a
    form an undo walked back to it: the next table written under it must not re-seed it away."""
    doc = _open(window, tmp_path, "a.dat")
    det = window._analysis_panel.detection_panel
    _run_detect(window, qtbot)
    first = det.k_in.value()
    det.k_in.setValue(first + 2.0)                       # an edit nobody ran
    _detect(window, doc, S, k_in=first - 1.0)            # a table with other settings lands
    assert det.k_in.value() == pytest.approx(first + 2.0)

    det.k_in.setValue(first)                             # the first table's value, typed back
    _detect(window, doc, S, k_in=first + 5.0)            # ...and another lands under it
    assert det.k_in.value() == pytest.approx(first), (
        "the form was re-seeded from a table that landed under it")


def test_a_table_landing_under_an_edited_form_shows_with_an_infinite_ml_notch_entry(
        window, tmp_path, qtbot):
    """A new table of the same key reads the form before it is shown (`_fill_form` asks whether
    it holds edits nobody ran). An ML notch entry that made ``ml_params()`` raise stopped that
    read half way, and the panel went on showing the table it had: measured in the review, four
    rows on screen and three stored."""
    doc = _open(window, tmp_path, "a.dat")
    _detect(window, doc, S, min_dwell_s=1e-3)
    ap = window._analysis_panel
    det = ap.detection_panel
    first_rows = ap._table.view.model().rowCount()
    det.k_in.setValue(det.k_in.value() + 1.0)             # the form holds an edit nobody ran
    det.ml_notch_freqs.setText("60, inf")
    det.ml_notch_freqs.editingFinished.emit()

    _detect(window, doc, S, min_dwell_s=11e-3)            # a batch pass lands another table
    stored = len(ap.current_detection().payload.df)
    assert stored != first_rows, "setup: the two tables must differ in length"
    assert ap._table.view.model().rowCount() == stored, (
        "the panel kept showing the table it had")


# ================================================================================================
# a clamp is captured before it lands, and announced after a draft comes back (design §5.3)
# ================================================================================================

def test_the_typed_20_khz_comes_back_after_a_25_khz_recording(window, tmp_path, qtbot):
    """FAST has a stored run, so without a draft its return would re-seed the run's
    low-pass; the draft has to hold what was typed, taken before the clamp."""
    fast = _open(window, tmp_path, "fast.dat")
    _detect(window, fast, S)
    slow = _open(window, tmp_path, "slow.dat", fs=SLOW)
    ap = window._analysis_panel
    det = ap.detection_panel

    window._on_selection(fast, 0)
    _low_pass(qtbot, det, "20000")
    window._on_selection(slow, 0)
    assert det.filter_cutoff_hz.value() < SLOW / 2
    said = _status(window)

    window._on_selection(fast, 0)
    assert det.filter_enabled.isChecked()
    assert det.filter_cutoff_hz.value() == pytest.approx(20_000.0), (
        "the 20 kHz typed on the 50 kHz recording did not come back")
    assert said == _CLAMP

    window._on_selection(slow, 0)
    draft = ap.form_drafts()[(fast, 0, S)]
    assert draft["detection"]["filter_cutoff_hz"] == pytest.approx(20_000.0), (
        "the draft was taken after the clamp")


def test_a_clamp_a_draft_brings_back_is_announced(window, tmp_path, qtbot):
    """The clamp notice is read once the form is in place, so it names a clamp the
    arriving draft holds even when the form that was on screen fitted."""
    fast = _open(window, tmp_path, "fast.dat")
    slow = _open(window, tmp_path, "slow.dat", fs=SLOW)
    det = window._analysis_panel.detection_panel
    window._on_selection(fast, 0)
    _low_pass(qtbot, det, "20000")
    window._on_selection(slow, 0)                        # carried over and clamped
    _type(qtbot, det.k_in, "6")                          # SLOW is edited: its draft keeps 20 kHz
    window._on_selection(fast, 0)
    _type(qtbot, det.filter_cutoff_hz, "10000")          # fits on SLOW
    window.statusBar().showMessage("what the user was reading")

    window._on_selection(slow, 0)
    assert det.k_in.value() == pytest.approx(6.0)
    assert _status(window) == _CLAMP, "the clamp SLOW's draft brought back was not said"


# ================================================================================================
# the five signals
# ================================================================================================

def test_leaving_fires_before_the_form_changes_and_arriving_after(window, tmp_path, qtbot):
    a = _open(window, tmp_path, "a.dat")
    _detect(window, a, S, filter_cutoff_hz=None)
    _detect(window, a, C, filter_cutoff_hz=8_000.0)
    b = _open(window, tmp_path, "b.dat")
    _detect(window, b, S, filter_cutoff_hz=6_000.0)
    ap = window._analysis_panel
    det = ap.detection_panel
    window._on_selection(a, 0)
    selector = ap._result_selector
    selector.setCurrentIndex(selector.findData(S))
    _low_pass(qtbot, det, "5000")

    seen: list[tuple] = []

    def reads() -> tuple:
        return (ap.form_key(), det.filter_cutoff_hz.value() if det.filter_enabled.isChecked()
                else None)

    ap.formLeaving.connect(lambda key: seen.append(("leaving", key, *reads())))
    ap.formArrived.connect(lambda old, new: seen.append(("arrived", (old, new), *reads())))

    window._on_selection(b, 0)                           # a switch of recording
    assert seen == [("leaving", (a, 0, S), (a, 0, S), 5_000.0),
                    ("arrived", ((a, 0, S), (b, 0, S)), (b, 0, S), 6_000.0)]

    seen.clear()
    window._on_selection(a, 0)
    selector.setCurrentIndex(selector.findData(C))       # a switch of table
    assert seen[-2:] == [("leaving", (a, 0, S), (a, 0, S), 5_000.0),
                         ("arrived", ((a, 0, S), (a, 0, C)), (a, 0, C), 8_000.0)]

    seen.clear()
    window._session.close_document(a)                    # nothing on screen
    assert seen[:2] == [("leaving", (a, 0, C), (a, 0, C), 8_000.0),
                        ("arrived", ((a, 0, C), None), *seen[1][2:])]


def test_a_first_table_arrives_from_the_key_that_stood_for_it(window, tmp_path, qtbot):
    """``(doc, ch, None)`` stands for the table a recording does not have yet. Its first
    table arrives FROM it; a run the form made there consumes the edits that made it, and
    a table that came from anywhere else is seeded like any arrival -- and leaves no draft
    behind for a table that no longer is missing (the edit stays reachable as the undo
    step that moves onto the table: ``test_editing_form_undo``)."""
    doc = _open(window, tmp_path, "a.dat")
    ap = window._analysis_panel
    det = ap.detection_panel
    assert ap.form_key() == (doc, 0, None)
    _low_pass(qtbot, det, "5000")
    arrived: list[tuple] = []
    ap.formArrived.connect(lambda old, new: arrived.append((old, new)))

    _run_detect(window, qtbot)
    assert arrived == [((doc, 0, None), (doc, 0, S))]
    assert ap.form_drafts() == {}
    assert det.filter_cutoff_hz.value() == pytest.approx(5_000.0)

    other = _open(window, tmp_path, "b.dat")
    _type(qtbot, det.k_in, "7")
    _detect(window, other, S, k_in=5.0)                  # written by something else
    assert det.k_in.value() == pytest.approx(5.0), "a first table must seed the form"
    assert (other, 0, None) not in ap.form_drafts(), (
        "the form that stood for the table was left filed as a draft of no table")


def test_seeds_drafts_and_restores_are_never_edits_and_a_seed_is_a_write(
        window, tmp_path, qtbot):
    a = _open(window, tmp_path, "a.dat")
    _detect(window, a, S, filter_cutoff_hz=None)
    b = _open(window, tmp_path, "b.dat")
    _detect(window, b, S, filter_cutoff_hz=6_000.0)
    ap = window._analysis_panel
    det = ap.detection_panel
    window._on_selection(a, 0)
    edited: list[int] = []
    written: list[int] = []
    ap.formEdited.connect(lambda: edited.append(1))
    ap.formWritten.connect(lambda: written.append(1))

    window._on_selection(b, 0)                           # a seed
    assert edited == [] and len(written) >= 1

    _low_pass(qtbot, det, "5000")                        # the control: a user's edit
    assert len(edited) >= 1
    n_edits = len(edited)
    window._on_selection(a, 0)
    window._on_selection(b, 0)                           # a draft coming back
    assert len(edited) == n_edits
    assert det.filter_cutoff_hz.value() == pytest.approx(5_000.0)

    state = ap.form_state()
    _type(qtbot, det.k_in, "7")
    n_edits, n_writes = len(edited), len(written)
    ap.restore_form_state(state)
    assert len(edited) == n_edits, "restore_form_state read as a user's edit"
    assert len(written) == n_writes + 1, "restore_form_state must say formWritten exactly once"
    assert det.k_in.value() == pytest.approx(5.0)


def test_each_named_action_names_its_step(window, tmp_path, qtbot):
    _open(window, tmp_path, "a.dat")
    ap = window._analysis_panel
    det = ap.detection_panel
    bp = ap.baseline_panel
    named: list[str] = []
    ap.formStepNamed.connect(named.append)

    det.protocol_combo.setCurrentText(HOST_GUEST)
    det.ml_preset.setCurrentIndex(det.ml_preset.findData("notebook1"))
    bp.reset_button.click()
    editor = bp.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData("constant"))
    editor.current.setText("-120")
    assert editor.clearable()
    editor.clear_button.click()

    assert named == [f"Recall protocol ‘{HOST_GUEST}’",
                     f"Load ML preset ‘{det._ml_preset_label('notebook1')}’",
                     "Reset baseline estimator",
                     "Clear open-pore reference"]

    # The app's own writes pass presets and resets through the same forms, and name nothing.
    state = ap.form_state()
    det.ml_preset.setCurrentIndex(det.ml_preset.findData("notebook2"))
    named.clear()
    ap.restore_form_state(state)
    assert named == []
    assert det.ml_preset_name() == "notebook1"
    with ap._writing_form():
        det.protocolRecalled.emit(HOST_GUEST)
        bp.resetPerformed.emit("Reset baseline estimator")
    assert named == [], "a named action inside the writing guard was taken for a step"


# ================================================================================================
# form_state / restore_form_state
# ================================================================================================

def test_restore_form_state_is_an_exact_round_trip_set_values_included(
        window, tmp_path, qtbot):
    fast = _open(window, tmp_path, "fast.dat")
    slow = _open(window, tmp_path, "slow.dat", fs=SLOW)
    ap = window._analysis_panel
    det = ap.detection_panel
    bp = ap.baseline_panel
    window._on_selection(fast, 0)
    _low_pass(qtbot, det, "20000")
    _type(qtbot, det.k_in, "6.5")
    det.notch_enabled.setChecked(True)
    det.apply_ml_preset("notebook2", user_choice=True)
    det.ml_enter.setValue(0.7)
    bp.rough_s.setValue(2.0)
    bp.base_s.setValue(0.5)
    window._on_selection(slow, 0)                        # the low-pass clamps here

    state = ap.form_state()
    assert "detector" not in state["detection"] and "model_path" not in state["ml"]
    assert state["detection"]["filter_cutoff_hz"] < SLOW / 2
    assert state["intents"]["set"]["filter_cutoff_hz"] == pytest.approx(20_000.0)
    assert json.loads(json.dumps(state)) == state, "the snapshot is not plain data"
    ap.restore_form_state(ap.form_state())
    assert ap.form_state() == state

    # Change all of it -- the low-pass by hand ON the clamped box, which is the write-back
    # that used to keep the clamp as the value set -- and the snapshot puts all of it back.
    _type(qtbot, det.filter_cutoff_hz, "10000")
    _type(qtbot, det.k_in, "4")
    det.notch_enabled.setChecked(False)
    det.apply_ml_preset("notebook1", user_choice=True)
    bp.rough_s.setValue(1.0)
    assert det.frequency_intents()["set"]["filter_cutoff_hz"] == pytest.approx(10_000.0)

    ap.restore_form_state(state)
    assert ap.form_state() == state
    assert det.clamp_notice() == _CLAMP, "the undone edit left the clamp as the value set"
    det.set_recording(FS)                                # the faster recording's range
    assert det.filter_cutoff_hz.value() == pytest.approx(20_000.0)


def test_an_off_low_pass_gets_its_box_back_too(window, tmp_path, qtbot):
    """``set_params`` leaves an OFF low-pass's box alone; the set values put it back."""
    _open(window, tmp_path, "a.dat")
    ap = window._analysis_panel
    det = ap.detection_panel
    _low_pass(qtbot, det, "7000")
    det.filter_enabled.setChecked(False)
    state = ap.form_state()
    _low_pass(qtbot, det, "3000")
    det.filter_enabled.setChecked(False)

    ap.restore_form_state(state)
    assert det.filter_cutoff_hz.value() == pytest.approx(7_000.0)
    assert ap.form_state() == state


# ================================================================================================
# drafts as a whole: the project's view of them, and a closed recording's
# ================================================================================================

def test_a_closed_recordings_drafts_are_dropped(window, tmp_path, qtbot):
    a = _open(window, tmp_path, "a.dat")
    _detect(window, a, S)
    b = _open(window, tmp_path, "b.dat")
    _detect(window, b, S)
    ap = window._analysis_panel
    det = ap.detection_panel
    window._on_selection(a, 0)
    _low_pass(qtbot, det, "5000")
    window._on_selection(b, 0)
    _low_pass(qtbot, det, "6000")
    assert set(ap.form_drafts()) == {(a, 0, S), (b, 0, S)}

    window._session.close_document(a)                    # a recording not on screen
    assert set(ap.form_drafts()) == {(b, 0, S)}
    window._session.close_document(b)                    # the one on screen, edited
    assert ap.form_drafts() == {}


def test_set_form_drafts_replaces_them_and_shows_the_key_on_screen_at_once(
        window, tmp_path, qtbot):
    a = _open(window, tmp_path, "a.dat")
    _detect(window, a, S)
    b = _open(window, tmp_path, "b.dat")
    _detect(window, b, S)
    ap = window._analysis_panel
    det = ap.detection_panel
    window._on_selection(a, 0)
    _low_pass(qtbot, det, "5000")
    window._on_selection(b, 0)
    _low_pass(qtbot, det, "6000")
    saved = ap.form_drafts()

    window._on_selection(a, 0)
    ap.set_form_drafts({})                               # what a fresh project holds
    assert ap.form_drafts() == {}
    window._on_selection(b, 0)
    assert not det.filter_enabled.isChecked(), "a replaced draft came back"

    ap.set_form_drafts(saved)
    assert det.filter_cutoff_hz.value() == pytest.approx(6_000.0), (
        "the key on screen did not show its draft at once")
    window._on_selection(a, 0)
    assert det.filter_cutoff_hz.value() == pytest.approx(5_000.0)
    assert set(ap.form_drafts()) == {(a, 0, S), (b, 0, S)}
