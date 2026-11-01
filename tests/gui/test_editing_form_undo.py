"""Edit ▸ Undo takes back the Detection, ML and Baseline forms, one form per event table: design
§3.2-3.6, the owner's decisions 2 and 3.

Before this, the forms had no history at all. A chain or threshold edit, a recalled protocol, a
loaded ML preset or a baseline reset could only be put back by hand, and one Ctrl+Z after
"change settings, run" restored the table and re-seeded the form from it -- right by accident,
since nothing recorded the settings steps the re-seed jumped over.

What is pinned here:

* a chain edit, a threshold edit, a protocol recall, an ML preset load and a baseline reset are
  each ONE named step that Ctrl+Z takes back and Ctrl+Shift+Z puts forward, the form restored
  exactly;
* a burst of edits inside the 700 ms pause is one step; edits further apart are two;
* "change settings, run, Ctrl+Z" restores the table AND the form exactly, in one press named
  "Undo Re-run detection and 2 setting changes", and Ctrl+Shift+Z replays both (decision 3);
* a settings step separated from the run by a curation step stays its own step;
* the app's own writes push no step: a table switch, a recording switch, and the re-seed that
  undoing a run causes (design §3.9);
* the first table a recording and channel gets takes over the settings steps made before it
  (§3.4), and they do not join that table's NEXT run -- wherever it is born: on screen, or
  while another recording is shown (a batch run, or a Detect that lands after a look away);
* the Edit menu names the step of the table on screen, and Ctrl+Z never reaches another
  recording's form (decision 1).

The Analysis panel's half of the form protocol (wave C contract: ``form_key``, ``form_state``,
``restore_form_state`` and the five form signals) is what these tests drive end to end; the
label tests at the bottom are pure.

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

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 100_000
S, C = "detect.schmitt", "detect.cusum"
_EVENTS = [(10_000, 10_300), (20_000, 20_250), (30_000, 30_400), (60_000, 60_350)]
PROTOCOL = "Zn2-insulin (zn_30h)"


def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _install(w: MainWindow, tmp_path, name: str = "pore.dat") -> str:
    """A synthetic one-channel recording at ``tmp_path / name``, opened and on screen."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_trace(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    return doc_id


def _land(w: MainWindow, doc_id: str, module_id: str, *, min_dwell_s: float) -> None:
    """A finished run's table installed the way the controller installs one."""
    session = w._session
    frame, prov = detect_events(_trace(), FS, baseline=100.0, min_dwell_s=min_dwell_s)
    session.add_result(AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": min_dwell_s},
        calibration_fingerprint=session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(frame, prov)))
    QTest.qWait(10)


def _run(w: MainWindow, qtbot) -> None:
    """Detect with the form as it stands, through the panel's own door, until it lands."""
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)


def _det(w: MainWindow):
    return w._analysis_panel.detection_panel


def _form(w: MainWindow) -> dict:
    return w._analysis_panel.form_state()


def _scope(w: MainWindow) -> tuple:
    return ("form", *w._analysis_panel.form_key())


def _commit(w: MainWindow) -> None:
    """Opening the Edit menu: one of the four moments that commit an edit in its pause."""
    w._on_edit_menu_about_to_show()


def _undo(w: MainWindow) -> None:
    w._act_undo.trigger()


def _redo(w: MainWindow) -> None:
    w._act_redo.trigger()


def _table(w: MainWindow):
    return w._analysis_panel.current_detection().payload.df.copy(deep=True)


def _show_table(w: MainWindow, module_id: str) -> None:
    """The Results dock's Table selector, as the user picks a table."""
    selector = w._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData(module_id))
    QTest.qWait(10)
    assert w._analysis_panel.current_detection().module_id == module_id


def _settings_steps(w: MainWindow) -> int:
    history = w._settings_history
    return sum(history.undo_depth(s) for s in history.scopes_with_history())


# ================================================================================================
# each kind of form change is one named step
# ================================================================================================

def test_a_chain_edit_is_one_named_step_that_undoes_and_redoes(window, tmp_path):
    _install(window, tmp_path)
    det = _det(window)
    det.filter_enabled.setChecked(True)
    det.filter_cutoff_hz.setValue(10_000.0)
    _commit(window)
    before = _form(window)

    det.notch_enabled.setChecked(not det.notch_enabled.isChecked())
    _commit(window)
    turned = "Turn on notch" if det.notch_enabled.isChecked() else "Turn off notch"
    assert window._act_undo.text() == f"Undo {turned}"
    det.filter_cutoff_hz.setValue(5_000.0)
    _commit(window)
    assert window._act_undo.text() == "Undo Change low-pass 10000 → 5000 Hz"
    after = _form(window)

    _undo(window)
    _undo(window)
    assert _form(window) == before, "two presses did not put the form back exactly"
    assert window.statusBar().currentMessage() == f"Undone: {turned}."
    assert window._act_redo.text() == f"Redo {turned}"
    _redo(window)
    _redo(window)
    assert _form(window) == after
    assert det.filter_cutoff_hz.value() == pytest.approx(5_000.0)


def test_a_threshold_edit_is_one_named_step(window, tmp_path):
    _install(window, tmp_path)
    det = _det(window)
    before, k_in = _form(window), det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 1.0:g}"
    after = _form(window)

    _undo(window)
    assert det.k_in.value() == pytest.approx(k_in) and _form(window) == before
    _redo(window)
    assert det.k_in.value() == pytest.approx(k_in + 1.0) and _form(window) == after


def test_a_protocol_recall_is_one_step_named_after_the_protocol(window, tmp_path):
    """A recall writes a whole chain, a dozen widgets: one step, under the protocol's name,
    not "Change 12 settings"."""
    _install(window, tmp_path)
    det = _det(window)
    before = _form(window)
    det.protocol_combo.setCurrentText(PROTOCOL)
    assert window._act_undo.text() == f"Undo Recall protocol ‘{PROTOCOL}’"
    recalled = _form(window)
    assert recalled != before, "setup: the protocol changed nothing"

    _undo(window)
    assert _form(window)["detection"] == before["detection"]
    assert window._act_redo.text() == f"Redo Recall protocol ‘{PROTOCOL}’"
    _redo(window)
    assert _form(window)["detection"] == recalled["detection"]
    assert _settings_steps(window) == 1


def test_an_ml_preset_load_is_one_step_named_after_the_preset(window, tmp_path):
    from nanoscope.analysis.ml_preprocess import PRESET_NAMES

    _install(window, tmp_path)
    det = _det(window)
    before = _form(window)
    preset = next(p for p in PRESET_NAMES if p != det.ml_preset_name())
    det.apply_ml_preset(preset, user_choice=True)
    label = f"Load ML preset ‘{det._ml_preset_label(preset)}’"
    assert window._act_undo.text() == f"Undo {label}"
    loaded = _form(window)

    _undo(window)
    assert _form(window)["ml"] == before["ml"]
    _redo(window)
    assert _form(window)["ml"] == loaded["ml"] and det.ml_preset_name() == preset


def test_a_baseline_reset_is_one_named_step(window, tmp_path):
    _install(window, tmp_path)
    baseline = window._analysis_panel.baseline_panel
    baseline.rough_s.setValue(2.0)
    _commit(window)
    assert window._act_undo.text().startswith("Undo Change baseline rough window"), (
        window._act_undo.text())
    edited = _form(window)

    baseline._on_reset_clicked()                  # Back to default
    assert window._act_undo.text() == "Undo Reset baseline estimator"
    assert baseline.rough_s.value() != pytest.approx(2.0)
    _undo(window)
    assert baseline.rough_s.value() == pytest.approx(2.0)
    assert _form(window)["baseline"] == edited["baseline"]
    _redo(window)
    assert baseline.rough_s.value() != pytest.approx(2.0)


# ================================================================================================
# the pause
# ================================================================================================

def test_a_burst_of_edits_inside_the_pause_is_one_step(window, tmp_path, qtbot):
    _install(window, tmp_path)
    det = _det(window)
    before = _form(window)
    det.k_in.setValue(det.k_in.value() + 1.0)
    det.k_out.setValue(det.k_out.value() + 0.5)
    det.notch_enabled.setChecked(not det.notch_enabled.isChecked())
    qtbot.wait(mw._FORM_COALESCE_MS + 300)          # the pause runs out on its own
    assert _settings_steps(window) == 1
    assert window._act_undo.text() == "Undo Change 3 settings"
    _undo(window)
    assert _form(window) == before


def test_edits_further_apart_than_the_pause_are_two_steps(window, tmp_path, qtbot):
    """CONTROL for the burst: the same two edits with the pause between them."""
    _install(window, tmp_path)
    det = _det(window)
    det.k_in.setValue(det.k_in.value() + 1.0)
    qtbot.wait(mw._FORM_COALESCE_MS + 300)
    det.k_out.setValue(det.k_out.value() + 0.5)
    qtbot.wait(mw._FORM_COALESCE_MS + 300)
    assert _settings_steps(window) == 2


def test_the_pause_is_the_plots_windows_pause():
    from nanoscope.gui.plots_panel import _HISTORY_COALESCE_MS

    assert mw._FORM_COALESCE_MS == _HISTORY_COALESCE_MS


# ================================================================================================
# one press for a run (design §3.6, decision 3)
# ================================================================================================

def test_change_settings_run_then_undo_restores_table_and_form_in_one_press(
        window, tmp_path, qtbot):
    _install(window, tmp_path)
    det = _det(window)
    _run(window, qtbot)                            # the first table
    first_table, first_form = _table(window), _form(window)

    det.k_in.setValue(det.k_in.value() + 1.0)
    _commit(window)
    det.notch_enabled.setChecked(not det.notch_enabled.isChecked())
    _commit(window)
    _run(window, qtbot)                            # replaces it
    second_table, second_form = _table(window), _form(window)
    assert not second_table.equals(first_table) or second_form != first_form

    assert window._act_undo.text() == "Undo Re-run detection and 2 setting changes"
    _undo(window)
    assert _table(window).equals(first_table), "the press did not restore the table"
    assert _form(window) == first_form, "the press left the settings that led to the run"
    assert window.statusBar().currentMessage() == (
        "Undone: Re-run detection and 2 setting changes.")
    assert window._act_redo.text() == "Redo Re-run detection and 2 setting changes"

    _redo(window)
    assert _table(window).equals(second_table)
    assert _form(window) == second_form
    assert window._act_undo.text() == "Undo Re-run detection and 2 setting changes"


def test_one_setting_change_is_named_in_the_singular(window, tmp_path, qtbot):
    _install(window, tmp_path)
    det = _det(window)
    _run(window, qtbot)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _run(window, qtbot)                            # Detect commits the edit "at once before a run"
    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"
    _undo(window)
    assert det.k_in.value() == pytest.approx(k_in)


def test_change_the_baseline_re_measure_then_undo_is_one_press_too(window, tmp_path, qtbot):
    """Re-measure installs a new run on the table as Re-run does; the baseline edit that led
    to it goes with it, even when the click comes inside the edit's pause."""
    _install(window, tmp_path)
    _run(window, qtbot)
    first_table, first_form = _table(window), _form(window)
    baseline = window._analysis_panel.baseline_panel
    baseline.rough_s.setValue(baseline.rough_s.value() * 2.0)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert window._analysis_panel.remeasure_active_detection() == "started"
    qtbot.wait(20)
    remeasured_form = _form(window)

    assert window._act_undo.text() == "Undo Re-measure depths and 1 setting change"
    _undo(window)
    assert _table(window).equals(first_table) and _form(window) == first_form
    _redo(window)
    assert _form(window) == remeasured_form


def test_a_settings_step_separated_from_the_run_by_curation_stays_its_own(
        window, tmp_path, qtbot):
    """Design §3.6's last bullet: a setting changed before other work is not part of the run
    that followed the work. Undoing that run restores its table, and the form follows the
    restored table (the CF-4 rule); the setting stays a step of its own, behind the curation,
    and every step replays in order."""
    _install(window, tmp_path)
    det = _det(window)
    _run(window, qtbot)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    df = window._analysis_panel.current_detection().payload.df
    assert window._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    _run(window, qtbot)
    rerun_form = _form(window)

    assert window._act_undo.text() == "Undo Re-run detection"
    _undo(window)
    assert window._act_undo.text() == "Undo Curate event", (
        "the setting behind the curation went with the run")
    _undo(window)
    step = f"Change k_in {k_in:g} → {k_in + 1.0:g}"
    assert window._act_undo.text() == f"Undo {step}"
    _undo(window)
    assert det.k_in.value() == pytest.approx(k_in)
    assert not window._act_undo.isEnabled()

    assert window._act_redo.text() == f"Redo {step}"
    _redo(window)
    assert det.k_in.value() == pytest.approx(k_in + 1.0)
    assert window._act_redo.text() == "Redo Curate event"
    _redo(window)
    assert window._act_redo.text() == "Redo Re-run detection"
    _redo(window)
    assert _form(window) == rerun_form


# ================================================================================================
# the app's own writes are not steps (design §3.9)
# ================================================================================================

def test_switching_recordings_and_tables_pushes_no_step(window, tmp_path):
    doc_a = _install(window, tmp_path, "a.dat")
    _land(window, doc_a, S, min_dwell_s=1e-4)
    _land(window, doc_a, C, min_dwell_s=2e-3)
    doc_b = _install(window, tmp_path, "b.dat")
    _land(window, doc_b, S, min_dwell_s=5e-3)

    for _round in range(2):
        window._on_selection(doc_a, 0)
        _show_table(window, S)
        _show_table(window, C)
        window._on_selection(doc_b, 0)
    _commit(window)
    assert _settings_steps(window) == 0, (
        f"a seed was recorded as an edit: {window._act_undo.text()!r}")
    assert not window._act_undo.isEnabled()


def test_the_re_seed_of_an_undone_run_pushes_no_step(window, tmp_path, qtbot):
    """Undoing a run re-seeds the form from the restored table. A seed recorded as an edit
    would be the newest step, and the next press would undo IT; it would also end the redo
    branch, so Ctrl+Shift+Z could not replay the run."""
    _install(window, tmp_path)
    det = _det(window)
    _run(window, qtbot)
    det.k_in.setValue(det.k_in.value() + 1.0)
    _run(window, qtbot)
    scope = _scope(window)
    held = window._settings_history.undo_depth(scope)

    _undo(window)
    _commit(window)                                # the menu opening must find nothing new
    assert window._settings_history.undo_depth(scope) == held - 1
    assert window._act_redo.text() == "Redo Re-run detection and 1 setting change"
    _redo(window)
    _commit(window)
    assert window._settings_history.undo_depth(scope) == held
    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"


# ================================================================================================
# the first table takes over the steps made before it (design §3.4)
# ================================================================================================

def test_the_first_table_takes_over_the_settings_steps_made_before_it(window, tmp_path, qtbot):
    doc_id = _install(window, tmp_path)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    assert window._analysis_panel.form_key() == (doc_id, 0, None)
    assert window._settings_history.undo_depth(("form", doc_id, 0, None)) == 1

    _run(window, qtbot)                            # the recording's first table
    assert window._analysis_panel.form_key() == (doc_id, 0, S)
    assert window._settings_history.undo_depth(("form", doc_id, 0, None)) == 0
    assert window._settings_history.undo_depth(("form", doc_id, 0, S)) == 1
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 1.0:g}"

    # ...and they produced THAT table, so the table's next run does not take them with it.
    k_out = det.k_out.value()
    det.k_out.setValue(k_out + 0.5)
    _run(window, qtbot)
    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"
    _undo(window)
    assert det.k_out.value() == pytest.approx(k_out)
    assert det.k_in.value() == pytest.approx(k_in + 1.0), (
        "the first run's settings went with the second run")
    _undo(window)
    assert det.k_in.value() == pytest.approx(k_in)


# ================================================================================================
# ...and a first table born where nobody is looking takes them over the same way
# ================================================================================================
#
# The panel arrives at a table born on screen (`formArrived`), and that is where the steps used
# to change hands. A table that lands while another recording is shown -- a batch run, or a
# Detect that lands after a look away -- never reaches the panel's arrival, so its recording's
# steps stayed under ``(doc, ch, None)`` for good: Edit ▸ Undo on the new table read "nothing to
# undo", and its un-run edit stayed filed under no table, to be saved with the project.

def test_a_table_born_off_screen_takes_over_the_steps_made_before_it(window, tmp_path, qtbot):
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")    # on screen, and no table yet
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 2.0)
    _commit(window)
    history = window._settings_history
    assert history.undo_depth(("form", doc_b, 0, None)) == 1, "setup: the edit is a step"
    window._on_selection(doc_a, 0)

    _land(window, doc_b, S, min_dwell_s=1e-4)      # a batch run lands B's first table
    assert history.undo_depth(("form", doc_b, 0, None)) == 0, (
        "the steps made before the table were left filed under no table")
    assert history.undo_depth(("form", doc_b, 0, S)) == 1
    window._on_selection(doc_b, 0)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 2.0:g}"

    # ...they did NOT produce that table: the edit stays on its form as its draft (the owner's
    # answer to 13.4 Q6), so the next run is the one they lead to, and takes them with it.
    k_out = det.k_out.value()
    det.k_out.setValue(k_out + 0.5)
    _run(window, qtbot)
    assert window._act_undo.text() == "Undo Re-run detection and 2 setting changes"
    _undo(window)
    assert det.k_in.value() == pytest.approx(k_in) and det.k_out.value() == pytest.approx(k_out)


def test_a_table_born_off_screen_leaves_no_draft_under_no_table(window, tmp_path):
    """The table-less form's draft was for the table that now exists. Left filed, it was
    written into the next project save as a record for no table, and came back on the form as
    soon as the detector combo named a detector with no table."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    det = _det(window)
    det.k_in.setValue(det.k_in.value() + 2.0)
    window._on_selection(doc_a, 0)
    assert (doc_b, 0, None) in window._analysis_panel.form_drafts(), "setup: B's edit is filed"

    _land(window, doc_b, S, min_dwell_s=1e-4)
    assert (doc_b, 0, None) not in window._analysis_panel.form_drafts()


def test_a_run_that_lands_after_a_look_away_hands_its_form_over_too(window, tmp_path, qtbot):
    """Detect on B, then a look at A before the run lands: the steps and the edit that made
    the table are B's table's, and nothing is left filed under no table."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 2.0)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._analysis_panel.trigger_detection()
        window._on_selection(doc_a, 0)             # the look away, before the run lands
    qtbot.wait(20)
    assert window._session.results.get(doc_b, 0, S) is not None, "setup: the run did not land"

    assert window._settings_history.undo_depth(("form", doc_b, 0, S)) == 1
    assert window._settings_history.undo_depth(("form", doc_b, 0, None)) == 0
    assert (doc_b, 0, None) not in window._analysis_panel.form_drafts()
    window._on_selection(doc_b, 0)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 2.0:g}"


def test_a_table_born_off_screen_does_not_split_the_press_of_the_recording_on_screen(
        window, tmp_path, qtbot):
    """The break that keeps a new table's first steps out of its next run's press belongs
    behind THOSE steps. Put behind another recording's, it stood between them and the run they
    led to, and "change settings, run, Ctrl+Z" took two presses."""
    doc_a = _install(window, tmp_path, "a.dat")
    _run(window, qtbot)                            # A's table
    doc_b = _install(window, tmp_path, "b.dat")
    det = _det(window)
    det.k_in.setValue(det.k_in.value() + 2.0)      # B's edit, before it has a table
    _commit(window)
    window._on_selection(doc_a, 0)
    det.k_out.setValue(det.k_out.value() + 0.5)    # A's own edit: the newest step of all
    _commit(window)
    _land(window, doc_b, S, min_dwell_s=1e-4)      # B's first table lands, off screen

    _run(window, qtbot)                            # Detect on A
    assert window._act_undo.text() == "Undo Re-run detection and 1 setting change"


def test_the_real_batch_detect_hands_over_the_steps_of_a_recording_not_on_screen(
        window, tmp_path, qtbot):
    """Analysis ▸ Batch detect, not a stand-in for it."""
    doc_a = _install(window, tmp_path, "a.dat")
    doc_b = _install(window, tmp_path, "b.dat")
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 2.0)
    window._on_selection(doc_a, 0)                 # a look at A, and a batch over every recording
    with qtbot.waitSignal(window._batch_controller.finished, timeout=30_000):
        window._batch_detect()
    qtbot.wait(100)
    assert window._session.results.get(doc_b, 0, S) is not None, "setup: no table on B"

    stranded = [s for s in window._settings_history.scopes_with_history()
                if s[1] == doc_b and s[3] is None]
    assert not stranded, f"B's steps are filed under {stranded}, unreachable from its table"
    window._on_selection(doc_b, 0)
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 2.0:g}", (
        window._act_undo.text())


def test_a_table_born_on_screen_by_a_batch_hands_over_without_a_draft_left_behind(
        window, tmp_path, qtbot):
    """The table on screen, tuned and then batch-detected: its steps move to the table, and
    the form it is seeded with is no draft of a table that does not exist any more."""
    doc_id = _install(window, tmp_path)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 2.0)
    _commit(window)

    _land(window, doc_id, S, min_dwell_s=1e-4)
    assert window._settings_history.undo_depth(("form", doc_id, 0, S)) == 1
    assert (doc_id, 0, None) not in window._analysis_panel.form_drafts()
    assert window._act_undo.text() == f"Undo Change k_in {k_in:g} → {k_in + 2.0:g}"


# ================================================================================================
# the Edit menu names the table on screen (decision 1)
# ================================================================================================

def test_the_edit_menu_names_the_step_of_the_table_on_screen(window, tmp_path):
    doc_a = _install(window, tmp_path, "a.dat")
    _land(window, doc_a, S, min_dwell_s=1e-4)
    det = _det(window)
    k_in = det.k_in.value()
    det.k_in.setValue(k_in + 1.0)
    _commit(window)
    a_step = f"Undo Change k_in {k_in:g} → {k_in + 1.0:g}"
    assert window._act_undo.text() == a_step

    doc_b = _install(window, tmp_path, "b.dat")
    _land(window, doc_b, S, min_dwell_s=1e-4)
    det.notch_enabled.setChecked(not det.notch_enabled.isChecked())
    _commit(window)
    b_step = window._act_undo.text()
    assert b_step in ("Undo Turn on notch", "Undo Turn off notch")

    window._on_selection(doc_a, 0)
    assert window._act_undo.text() == a_step
    window._on_selection(doc_b, 0)
    assert window._act_undo.text() == b_step
    _undo(window)
    assert window._act_undo.text() == "Undo — nothing on this recording", (
        "B's form history is spent; A's step is not B's to undo")
    window._undo_events()                          # what ⌘Z from another window reaches
    window._on_selection(doc_a, 0)
    assert det.k_in.value() == pytest.approx(k_in + 1.0), (
        "Ctrl+Z on B changed A's form")
    assert window._act_undo.text() == a_step


def test_another_tables_form_steps_are_named_where_they_are(window, tmp_path):
    """On the same channel, another table's form steps are one table switch away; the menu
    points at them as it points at that table's curation."""
    doc_id = _install(window, tmp_path)
    _land(window, doc_id, S, min_dwell_s=1e-4)
    _land(window, doc_id, C, min_dwell_s=2e-3)
    _show_table(window, C)
    det = _det(window)
    det.k_in.setValue(det.k_in.value() + 1.0)
    _commit(window)
    _show_table(window, S)
    assert window._act_undo.text() == (
        "Undo — nothing on this table; the CUSUM detector (1 step) on this channel")


# ================================================================================================
# labels, pure (design §3.2)
# ================================================================================================

def _panel_state(window) -> dict:
    """A real form state, from the Analysis panel the window built."""
    return window._analysis_panel.form_state()


def _with(state: dict, part: str, **changes) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in state.items()}
    out[part] = {**out[part], **changes}
    return out


def test_one_change_is_named_in_the_chain_notices_words(window):
    state = _panel_state(window)
    det = state["detection"]
    on = _with(state, "detection", filter_cutoff_hz=10_000.0)
    assert mw._form_step_label(on, _with(on, "detection", filter_cutoff_hz=5_000.0),
                               fs=FS) == "Change low-pass 10000 → 5000 Hz"
    off = _with(state, "detection", filter_cutoff_hz=None)
    assert mw._form_step_label(off, on, fs=FS) == "Turn on low-pass"
    notch_off = _with(state, "detection", notch_enabled=False)
    notch_on = _with(state, "detection", notch_enabled=True)
    assert mw._form_step_label(notch_off, notch_on, fs=FS) == "Turn on notch"
    assert mw._form_step_label(notch_on, notch_off, fs=FS) == "Turn off notch"
    k_in = float(det["k_in"])
    assert mw._form_step_label(state, _with(state, "detection", k_in=k_in + 1.0),
                               module_id=S) == f"Change k_in {k_in:g} → {k_in + 1.0:g}"
    cusum_h = float(det["cusum_h"])
    assert mw._form_step_label(state, _with(state, "detection", cusum_h=cusum_h + 1.0),
                               module_id=S) == (
        f"Change CUSUM h {cusum_h:g} → {cusum_h + 1.0:g}"), (
        "the other detector's threshold, on the form while Schmitt's table is shown")
    q = float(det["notch_q"])
    assert mw._form_step_label(notch_off, _with(notch_off, "detection", notch_q=q + 5.0),
                               fs=FS) == f"Change notch Q {q:g} → {q + 5.0:g}", (
        "a setting of a stage that is off is still named in that stage's words")


def test_baseline_ml_and_several_changes_are_named(window):
    state = _panel_state(window)
    rough = float(state["baseline"]["rough_s"])
    assert mw._form_step_label(state, _with(state, "baseline", rough_s=rough + 1.5)) == (
        f"Change baseline rough window {rough:g} → {rough + 1.5:g} s")
    reference = {**state["baseline"]["reference"], "current": "123.4"}
    assert mw._form_step_label(state, _with(state, "baseline", reference=reference)) == (
        "Change open-pore reference")
    enter = 0.95 if state["ml"]["ml_decode"]["enter_threshold"] != 0.95 else 0.85
    decode = {**state["ml"]["ml_decode"], "enter_threshold": enter}
    assert mw._form_step_label(state, _with(state, "ml", ml_decode=decode, enter=enter,
                                            ml_preset_customized=True)) == (
        "Change ML enter threshold"), "the derived ML keys count as nothing"
    det = state["detection"]
    several = _with(_with(state, "detection", k_in=float(det["k_in"]) + 1.0,
                          notch_enabled=not det["notch_enabled"]),
                    "baseline", rough_s=rough + 1.5)
    assert mw._form_step_label(state, several) == "Change 3 settings"
    assert mw._form_step_label(state, state) == "Change settings"
