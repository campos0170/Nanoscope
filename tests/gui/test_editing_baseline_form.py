"""The Baseline form gets a real SET for undo, and resets that reset one thing.

Editing foundation, design §3.9 and the "Baseline ▸ Reset automatic windows" row of §6 (approved
by the owner 2026-10-30):

* ``BaselinePanel.set_params`` is a SEED, and a seed deliberately steps aside for a
  confirmed form ("the app proposes, the user confirms"). Edit ▸ Undo is the user's own
  act, so it cannot go through it: ``form_state()`` is the snapshot the settings history
  keeps, ``restore_form_state()`` the set it writes back through.
* Reset automatic windows was hidden outside Automatic, so with a user-provided reference chosen
  the estimator's windows -- which still travel in ``params`` and in the baseline
  fingerprint -- had no reset at all. It now resets them in every mode, and never the
  reference. Under a reference it stands without Measure and Use proposed windows, beside a
  caption that names the estimator and quotes the windows it resets.
* The reference had no reset of its own. "Use automatic reference" is it, and it keeps the
  windows.
* Each reset names its own step (``resetPerformed``), because only the button knows
  which part of the form it reset.

The panel is a pure view: every claim here is made on a bare ``BaselinePanel``, as the
panel's own tests make theirs, and nothing reads a recording.
"""
from __future__ import annotations

import copy
import json
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

from nanoscope.analysis import baseline as B  # noqa: E402
from nanoscope.analysis import baseline_probe as P  # noqa: E402
from nanoscope.gui.baseline_panel import _DIM_BUTTON, BaselinePanel  # noqa: E402

FS = 1000.0
DEFAULT_WINDOWS = (B.DEFAULT_BASELINE.rough_s, B.DEFAULT_BASELINE.base_s)
#: A proposal-scope health block that condemns the windows on offer, so an accepted
#: proposal carries the standing "ACCEPTED OVER" mark.
CONDEMNING = {"checks": [
    {"name": "base_vs_state", "level": "warn", "label": "base window vs state dwell",
     "value": 1.38, "threshold": ">= 3.0", "message": "the base window rides the states"},
    {"name": "overshoot", "level": "ok", "label": "overshoot",
     "value": 0.09, "threshold": "<= 0.3", "message": "0.09"},
]}


@pytest.fixture
def panel(qapp, qtbot):
    widget = BaselinePanel()
    qtbot.addWidget(widget)
    widget.set_recording(FS, ("recording-a", 0), duration_s=10.0)
    widget.set_gate(True)
    return widget


def choose(panel, mode):
    editor = panel.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData(mode))
    return editor


def known(value=-120.0):
    return {"baseline_open_pore_reference": {
        "version": 1, "mode": "constant", "current_pA": value,
        "intervals_s": [], "interpolation": "constant"}}


def _proposal(rough=2.0, base=0.5, source="proposed"):
    probe = P.BaselineProbe(
        fs=50_000.0, n=3_000_000, polarity=-1, sigma_pA=4.60, noise_pA=4.44,
        longest_state_s=0.496, n_states=489, mask_fraction=0.0931, gating=False,
        gating_score=0.51, level_dwell_s=None, probe_rough_s=5.0, usable=True,
        elapsed_s=0.23)
    cfg = B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)
    return P.BaselineProposal(
        config=cfg, param_source=source, probe=probe,
        reason="Longest state 496 ms (489 states, 9.3% of the trace) -> rough 2 s.")


def _accept_condemned(panel, rough=2.0, base=0.5):
    """Accept a condemned proposal the way the override does, leaving the mark up."""
    panel.set_proposal(_proposal(rough, base))
    panel.set_health(CONDEMNING, measured_for="proposal")
    assert panel.apply_proposal(override=True)
    assert panel.accepted_condemned() == ("base window vs state dwell",)


def _windows(panel):
    return float(panel.rough_s.value()), float(panel.base_s.value())


def _record(panel):
    """Every emission of the two form signals, in order."""
    seen = []
    panel.paramsChanged.connect(lambda params: seen.append(("params", params)))
    panel.resetPerformed.connect(lambda name: seen.append(("reset", name)))
    return seen


# ===========================================================================
# 1. form_state / restore_form_state: a SET that leaves exactly the snapshot
# ===========================================================================
def test_an_automatic_form_comes_back_exactly_with_its_mark(panel):
    """The windows, where they came from, and the "ACCEPTED OVER" mark: the mark says
    the depths measured with these windows are not certifiable, so an undo that put the
    windows back without it would put back the panel's most reassuring sentence over
    condemned numbers."""
    _accept_condemned(panel)
    state = panel.form_state()
    params = panel.params()
    assert state["rough_s"] == 2.0 and state["base_s"] == 0.5
    assert state["source"] == "proposed"
    assert state["accepted_condemned"] == ["base window vs state dwell"]

    panel.reset_button.click()
    assert panel.form_state() != state

    panel.restore_form_state(state)
    assert panel.form_state() == state
    assert panel.params() == params
    assert panel.param_source() == "proposed"
    assert "ACCEPTED OVER" in panel.source_label.text()


def test_a_constant_reference_comes_back_as_typed(panel):
    editor = choose(panel, "constant")
    editor.current.setText("−120.25")          # a pasted Unicode minus, kept AS TYPED
    editor.notes.setText("Open-pore sweep before the run, notebook 8")
    state = panel.form_state()
    params = panel.params()
    assert params["baseline_open_pore_reference"]["current_pA"] == -120.25

    editor.clear_button.click()
    assert panel.reference_mode() == "automatic"

    panel.restore_form_state(state)
    assert panel.form_state() == state
    assert panel.params() == params
    assert editor.current.text() == "−120.25"
    assert panel.automatic_controls.isHidden()
    assert not editor.known_box.isHidden()


def test_interval_rows_come_back_as_typed_including_an_invalid_draft(panel):
    """An invalid draft is shown, refused and KEPT by this form, never repaired -- so the
    snapshot holds every row as text, and an undo brings back the draft the user left,
    with the same refusal on it."""
    panel.add_reference_interval(0.5, 1.0)
    panel.add_reference_interval(8.0, 9.0)
    editor = panel.reference_editor
    editor.method.setCurrentIndex(editor.method.findData("linear"))
    assert panel.params()["baseline_open_pore_reference"]["intervals_s"] == [[.5, 1.], [8., 9.]]
    editor.table.item(1, 1).setText("9.5x")
    error = panel.reference_error()
    assert "Interval 2" in error
    state = panel.form_state()
    params = panel.params()
    assert state["reference"]["intervals"] == [["500.0", "1000.0"], ["8000.0", "9.5x"]]
    assert state["reference"]["interval_display"] == {"version": 1, "units": [["ms", "ms"], ["ms", "ms"]]}
    assert state["reference"]["interpolation"] == "linear"

    editor.clear_button.click()
    assert editor.table.rowCount() == 0

    panel.restore_form_state(state)
    assert panel.form_state() == state
    assert panel.params() == params
    assert panel.reference_error() == error
    assert editor.validation_label.text() == error


@pytest.mark.parametrize("stored", [
    {"baseline_open_pore_reference": {"mode": "future"}},
    {"baseline_reference_applied": True, "baseline_open_pore_reference": None},
], ids=["unsupported-recipe", "missing-recipe"])
def test_a_stored_reference_that_could_not_be_restored_comes_back_blocking(panel, stored):
    """The refusal, the recipe it refused and the flags that must keep travelling with it
    are part of the form: they are what blocks detection until someone chooses."""
    panel.set_params(stored)
    error = panel.reference_error()
    assert error
    state = panel.form_state()
    params = panel.params()

    panel.reference_editor.clear_button.click()
    assert panel.reference_error() == ""

    panel.restore_form_state(state)
    assert panel.form_state() == state
    assert panel.reference_error() == error
    assert panel.params() == params


def test_a_confirmed_form_is_set_back_where_a_seed_steps_aside(panel):
    """THE REASON THIS DOOR EXISTS. A hand edit confirms the windows and the reference
    alike, and `set_params` then refuses to seed over either -- that is the rule, and it
    is why undo cannot be a seed."""
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)
    editor = choose(panel, "constant")
    editor.current.setText("-120")
    assert panel.form_state()["source"] == "user"
    assert panel.form_state()["reference"]["confirmed"] is True
    confirmed = panel.form_state()

    choose(panel, "automatic")
    panel.rough_s.setValue(3.0)
    choose(panel, "constant")
    editor.current.setText("-200")

    # The seed steps aside for the confirmation...
    panel.set_params({**known(-120.0), "baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": "user"})
    assert _windows(panel)[0] == 3.0
    assert editor.current.text() == "-200"

    # ...the set does not.
    panel.restore_form_state(confirmed)
    assert panel.form_state() == confirmed
    assert _windows(panel) == (2.0, 0.5)
    assert editor.current.text() == "-120"
    # A muted write moves the spins under a reference too, and the caption quoting them
    # follows it.
    assert panel.estimator_windows_label.text().endswith(": 2000 ms / 500 ms")


def test_an_unconfirmed_form_comes_back_unconfirmed(panel):
    """Exact in the other direction too. A restore that left the form confirmed would
    quietly turn every later seed away -- the stored run of the table on screen could
    no longer fill a form the user never touched."""
    panel.set_params({**known(-100.0), "baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": "restored"})
    seeded = panel.form_state()
    assert seeded["source"] == "restored"
    assert seeded["reference"]["confirmed"] is False
    assert panel.estimator_windows_label.text().endswith(": 2000 ms / 500 ms")

    panel.reference_editor.current.setText("-150")       # confirms the reference
    panel.restore_form_state(seeded)
    assert panel.form_state() == seeded

    panel.set_params(known(-90.0))
    assert panel.params()["baseline_open_pore_reference"]["current_pA"] == -90.0


def test_a_restore_emits_params_changed_once_after_the_whole_write(panel):
    """Every widget it writes would otherwise speak for itself: a mode, a table, two
    boxes and two spins would each re-judge the detection on screen against a form that
    is half old and half new, and each would stamp the source ``'user'``."""
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)
    panel.add_reference_interval(1.0, 2.0)
    panel.add_reference_interval(5.0, 6.0)
    panel.reference_editor.notes.setText("notebook 8")
    target = panel.form_state()

    panel.reference_editor.clear_button.click()
    panel.reset_button.click()
    choose(panel, "constant").current.setText("-80")

    seen = _record(panel)
    states = []
    panel.paramsChanged.connect(lambda _params: states.append(panel.form_state()))
    panel.restore_form_state(target)
    assert seen == [("params", panel.params())]
    assert states == [target], "it spoke before the write was complete"
    assert panel.param_source() == "user"


def test_the_snapshot_is_plain_data_and_a_copy(panel):
    """The history keeps snapshots for as long as the session runs: they must survive
    JSON and ``deepcopy`` unchanged, and must not be views of the live form."""
    _accept_condemned(panel)
    panel.set_params({"baseline_open_pore_reference": {"mode": "future", "nested": [1, 2]}})
    state = panel.form_state()
    assert json.loads(json.dumps(state)) == state
    assert copy.deepcopy(state) == state

    # A snapshot is not a view of the form...
    state["reference"]["invalid_recipe"]["nested"].append(3)
    state["accepted_condemned"].append("tampered")
    assert panel.params()["baseline_open_pore_reference"] == {"mode": "future", "nested": [1, 2]}
    assert panel.accepted_condemned() == ("base window vs state dwell",)

    # ...and a form set from one does not hold on to it.
    pristine = json.loads(json.dumps(panel.form_state()))
    panel.reference_editor.clear_button.click()
    panel.reset_button.click()
    panel.restore_form_state(pristine)
    assert panel.form_state() == pristine
    pristine["reference"]["invalid_recipe"]["nested"].append(4)
    assert panel.params()["baseline_open_pore_reference"] == {"mode": "future", "nested": [1, 2]}


def test_the_snapshot_leaves_out_the_proposal_and_the_view_boxes(panel):
    """A proposal is a MEASUREMENT of the recording, not a setting, and the two boxes
    ask for a drawing. Neither is in the snapshot, and a restore leaves both alone."""
    state = panel.form_state()
    panel.set_proposal(_proposal())
    panel.set_health(CONDEMNING, measured_for="proposal")
    panel.show_track.setChecked(True)
    panel.subtract_view.setChecked(True)
    assert panel.form_state() == state

    panel.rough_s.setValue(3.0)
    panel.restore_form_state(state)
    assert panel.proposal() is not None
    assert panel.condemned_checks() == ("base_vs_state",)
    assert panel.show_track.isChecked() and panel.subtract_view.isChecked()


def test_a_missing_key_leaves_its_part_alone(panel):
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)
    editor = choose(panel, "constant")
    editor.current.setText("-120")

    panel.restore_form_state({"reference": {"current": "-80"}})
    assert _windows(panel) == (2.0, 0.5)
    assert panel.reference_mode() == "constant"
    assert editor.current.text() == "-80"

    panel.restore_form_state({"rough_s": 1.0})
    assert _windows(panel) == (1.0, 0.5)
    assert editor.current.text() == "-80"
    assert panel.param_source() == "user"


# ===========================================================================
# 2. Reset automatic windows: the estimator, in every mode, never the reference
# ===========================================================================
@pytest.mark.parametrize("mode", ["constant", "intervals"])
def test_back_to_default_is_on_screen_and_works_under_a_reference(panel, mode):
    """Hidden outside Automatic until this change, though the windows it resets still
    travel in ``params`` and in the baseline fingerprint while a reference is chosen."""
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)
    editor = choose(panel, mode)
    if mode == "constant":
        editor.current.setText("-120")
    else:
        panel.add_reference_interval(1.0, 2.0)

    assert panel.reset_button.isVisibleTo(panel), (
        "Reset automatic windows is hidden while a user-provided reference is chosen")
    assert panel.reset_button.isEnabled()
    caption = panel.estimator_windows_label
    assert caption.isVisibleTo(panel)
    assert caption.text() == "Automatic estimator (not in use): 2000 ms / 500 ms"

    reference = editor.form_state()
    recipe = panel.params()["baseline_open_pore_reference"]
    seen = _record(panel)
    panel.reset_button.click()
    assert [kind for kind, _ in seen] == ["params", "reset"]
    assert seen[0][1] == panel.params()
    assert seen[1][1] == "Reset baseline estimator"

    assert _windows(panel) == DEFAULT_WINDOWS
    # The spins are hidden in this mode; without the caption the press would change
    # nothing anyone could see.
    assert caption.text() == "Automatic estimator (not in use): 500 ms / 100 ms"
    assert editor.form_state() == reference, "Reset automatic windows touched the reference"
    assert panel.params()["baseline_open_pore_reference"] == recipe
    assert panel.param_source() == "user", "the reference's own confirmation moved"
    choose(panel, "automatic")
    assert panel.param_source() == "default"


def test_back_to_default_in_automatic_names_its_step_and_drops_the_mark(panel):
    _accept_condemned(panel)
    seen = _record(panel)
    panel.reset_button.click()
    assert seen == [("params", panel.params()), ("reset", "Reset baseline estimator")]
    assert _windows(panel) == DEFAULT_WINDOWS
    assert panel.param_source() == "default"
    assert panel.accepted_condemned() == ()


def test_back_to_default_under_a_reference_does_not_raise_the_run_windows_line(panel):
    """The line quotes the spins "above", which a reference hides. Reset automatic windows moves
    those spins in every mode now, and its refresh used to put the line back up."""
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)
    panel.set_run_windows({"baseline_rough_s": 2.0, "baseline_base_s": 0.5})
    assert not panel.run_windows_label.isVisibleTo(panel)
    choose(panel, "constant").current.setText("-120")

    panel.reset_button.click()
    assert not panel.run_windows_label.isVisibleTo(panel)

    # Back in Automatic the contrast is true, and said.
    choose(panel, "automatic")
    assert panel.run_windows_label.isVisibleTo(panel)
    assert "not the 500 ms / 100 ms above" in panel.run_windows_label.text()


def test_under_a_reference_back_to_default_stands_alone_beside_what_it_resets(panel):
    """Of the estimator's three actions, only the reset stays under a reference: Measure
    and Use proposed windows propose and confirm windows for the estimator a reference
    replaces. Apply's cell names that estimator instead, so the button left beside it
    does not read as the reference form's reset."""
    panel.set_proposal(_proposal(0.04, 0.01))
    panel.set_health(CONDEMNING, measured_for="proposal")
    panel.apply_button.click()                       # refused; armed
    assert panel.apply_button.text() == "Use proposed windows anyway"
    caption = panel.estimator_windows_label
    assert not caption.isVisibleTo(panel)

    choose(panel, "constant")
    assert panel.reset_button.isVisibleTo(panel)
    assert not panel.probe_button.isVisibleTo(panel)
    assert not panel.apply_button.isVisibleTo(panel)
    assert caption.isVisibleTo(panel)
    assert caption.text().startswith("Automatic estimator")

    # Back in Automatic the row is the one it always was, and the override the user was
    # shown is still armed.
    choose(panel, "automatic")
    assert panel.probe_button.isVisibleTo(panel) and panel.apply_button.isVisibleTo(panel)
    assert not caption.isVisibleTo(panel)
    assert panel.apply_button.text() == "Use proposed windows anyway"


# ===========================================================================
# 3. Use automatic reference: the reference, never the windows
# ===========================================================================
def test_clear_reference_resets_the_reference_and_keeps_the_windows(panel):
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)
    editor = choose(panel, "intervals")
    panel.add_reference_interval(1.0, 2.0)
    panel.add_reference_interval(5.0, 6.0)
    editor.method.setCurrentIndex(editor.method.findData("linear"))
    editor.notes.setText("notebook 8")
    assert editor.clear_button.styleSheet() == ""        # live: there is a reference

    seen = _record(panel)
    editor.clear_button.click()
    assert seen == [("params", panel.params()), ("reset", "Clear open-pore reference")]

    assert panel.reference_mode() == "automatic"
    assert "baseline_open_pore_reference" not in panel.params()
    assert editor.table.rowCount() == 0
    assert editor.current.text() == editor.notes.text() == ""
    assert editor.method.currentData() == "constant"
    assert not panel.automatic_controls.isHidden()
    assert _windows(panel) == (2.0, 0.5), "Use automatic reference touched the windows"
    assert panel.param_source() == "user"

    # A reset, not a choice: the confirmation went with it, so a stored run may seed the
    # reference again -- as Reset automatic windows's 'default' lets one seed the windows.
    assert editor.form_state()["confirmed"] is False
    panel.set_params(known(-80.0))
    assert panel.reference_mode() == "constant"


def test_clear_reference_with_nothing_to_clear_says_so_and_changes_nothing(panel, qtbot):
    """"Locked" is styling here, never ``setEnabled(False)``: the button stays clickable
    and answers the click, as every button in this panel does."""
    editor = panel.reference_editor
    assert editor.clear_button.isVisibleTo(panel)
    assert editor.clear_button.isEnabled()
    assert editor.clear_button.styleSheet() == _DIM_BUTTON
    before = panel.form_state()
    with qtbot.assertNotEmitted(panel.paramsChanged), \
            qtbot.assertNotEmitted(panel.resetPerformed):
        editor.clear_button.click()
    assert panel.form_state() == before
    assert "no open-pore reference to clear" in panel.gate_label.text()

    # Drafts typed into a mode no longer chosen contribute nothing to a run.
    choose(panel, "constant").current.setText("-120")
    assert editor.clear_button.styleSheet() == ""
    choose(panel, "automatic")
    assert editor.clear_button.styleSheet() == _DIM_BUTTON


def test_clear_reference_is_one_click_out_of_a_reference_that_could_not_be_restored(panel):
    """The mode never moved -- it still says Automatic -- yet detection is blocked until
    someone chooses. Until now the only way out was to pick another mode and come back."""
    panel.set_params({"baseline_reference_applied": True,
                      "baseline_open_pore_reference": None})
    editor = panel.reference_editor
    assert panel.reference_mode() == "automatic"
    assert "missing its recipe" in panel.reference_error()
    assert editor.clear_button.styleSheet() == ""

    editor.clear_button.click()
    assert panel.reference_error() == ""
    assert "baseline_reference_applied" not in panel.params()
    assert B.baseline_config_from_params(panel.params())[0].open_pore_reference is None


def test_each_reset_says_what_it_keeps(panel):
    clear_tip = panel.reference_editor.clear_button.toolTip()
    assert "rough and base windows are kept" in clear_tip
    assert "Reset automatic windows" in clear_tip
    reset_tip = panel.reset_button.toolTip()
    assert "every mode" in reset_tip and "Use automatic reference" in reset_tip


def test_neither_reset_moves_out_from_under_the_pointer(qapp, qtbot):
    """Hosted the way the page hosts it (a column ending in a stretch). Use automatic reference
    changes the mode, and everything under the mode's row changes with it -- which is
    why the button rides that row. Reset automatic windows re-words the caption beside it, and
    at this width the new words take a line fewer: measured, a button centred in its row
    then moved 8 px, which is why it is top-aligned."""
    host = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(host)
    panel = BaselinePanel()
    column.addWidget(panel)
    column.addStretch(1)
    qtbot.addWidget(host)
    host.resize(340, 1000)
    host.show()
    qtbot.wait(50)
    panel.set_recording(FS, ("doc-A", 0), duration_s=10.0)
    panel.set_gate(True)
    y = lambda w: w.mapTo(host, QtCore.QPoint(0, 0)).y()  # noqa: E731
    editor = panel.reference_editor
    panel.rough_s.setValue(12.345)
    panel.base_s.setValue(0.1234)
    panel.add_reference_interval(1.0, 2.0)
    qtbot.wait(50)

    before = y(panel.reset_button)
    qtbot.mouseClick(panel.reset_button, QtCore.Qt.LeftButton)
    qtbot.wait(50)
    assert _windows(panel) == DEFAULT_WINDOWS
    assert y(panel.reset_button) == before

    before = y(editor.clear_button)
    qtbot.mouseClick(editor.clear_button, QtCore.Qt.LeftButton)
    qtbot.wait(50)
    assert panel.reference_mode() == "automatic"
    assert y(editor.clear_button) == before
