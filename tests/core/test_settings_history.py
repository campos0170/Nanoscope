"""SettingsHistory — Edit ▸ Undo over the analysis settings (design §3.3-3.5, decision 2).

Before this class no undo reached a single analysis setting: a chain edit, a calibration Apply
or Reset, a time-base declaration and an acquisition-bandwidth override were all one-way
(design §1.3). It is the third sibling in ``core/history.py``, and these tests hold it to the
rule it shares with the other two and to the two things it adds:

* **the shared rule** — a pristine copy per scope that is COMPARED, so first sight is adopted
  silently and a state equal to the last one (a fresh NaN, a tuple for a list, a numpy float
  for a float) pushes nothing; recording muted while suspended and while an undo writes;
  derived labels; a depth ceiling that drops the oldest step, counts it and says so;
* **redo per scope** (owner decision 2) — a new step ends its own scope's redo branch, no other;
* **move_scope** — the settings a recording's form gathered before its first event table
  become that table's history.

The class holds no widget, so a dict stands in for the app on the far side of ``read`` and
``write``, and no QApplication is needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pytest

from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.history import MAX_DEPTH, SettingsHistory, _settings_equal
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.types import ChannelKind, Provenance

#: The four scope shapes MainWindow builds (``core/history.py``, ``SettingsScope``).
FORM = ("form", "doc-A", 0, "detect.schmitt")
FORM_NO_TABLE = ("form", "doc-A", 0, None)
CAL = ("calibration", "doc-A", 0)
TIME_BASE = ("time_base", "doc-A")
BANDWIDTH = ("acquisition", "doc-A", 0)
FORM_B = ("form", "doc-B", 0, "detect.schmitt")

DECLARED_50K = {"declaration": SampleRateDeclaration(mode="sample_rate", sample_rate_hz=50_000.0)}
OVERRIDE_10K = {"override": AcquisitionBandwidth(cutoff_hz=10_000.0, status="reported",
                                                 source="user")}


# ================================================================================================
# the far side of read/write
# ================================================================================================

class _App:
    """One live state per scope, read and written the way a careless app would.

    ``read`` hands out the LIVE object, never a copy, and ``write`` installs the very object it
    is given: the two habits that would let an in-place edit reach a history that stored
    references. ``writes`` keeps a copy of every state written; ``refuse`` makes the next
    writes raise.
    """

    def __init__(self) -> None:
        self.states: dict = {}
        self.writes: list = []
        self.readable = True
        self.refuse: Exception | None = None

    def read(self, scope):
        return self.states.get(scope) if self.readable else None

    def write(self, scope, state) -> None:
        if self.refuse is not None:
            raise self.refuse
        self.writes.append((scope, copy.deepcopy(state)))
        self.states[scope] = state


def _form(k_in: float = 5.0, **extra) -> dict:
    """A cut-down form state with the real one's shapes: floats, a ``None`` 'off' sentinel,
    a list of notch lines and a nested ML block."""
    state = {"detector": "detect.schmitt", "k_in": k_in, "k_out": 2.0, "max_dwell_s": None,
             "notch_freqs": [50.0, 150.0],
             "ml": {"enter": 0.6, "exit": 0.4, "ml_preset_deviations": []}}
    state.update(extra)
    return state


@pytest.fixture
def app() -> _App:
    return _App()


def _history(app: _App, **kw) -> SettingsHistory:
    return SettingsHistory(app.read, app.write, **kw)


def _listen(history: SettingsHistory) -> list:
    """Every signal, in order: ``("changed",)``, ``("step", scope, label)`` and
    ``("dropped", scope, n)``."""
    heard: list = []
    history.changed.connect(lambda: heard.append(("changed",)))
    history.stepRecorded.connect(lambda scope, label: heard.append(("step", scope, label)))
    history.stepsDepthDropped.connect(lambda scope, n: heard.append(("dropped", scope, n)))
    return heard


def _record(history: SettingsHistory, app: _App, scope, state, **kw) -> bool:
    """Put ``state`` on screen for ``scope`` and checkpoint it."""
    app.states[scope] = state
    return history.checkpoint(scope, **kw)


@dataclass(frozen=True)
class _Declared:
    rate_hz: float
    lines: tuple = ()
    cache: object = field(default=None, compare=False)


@dataclass(frozen=True)
class _AlsoDeclared:
    rate_hz: float
    lines: tuple = ()
    cache: object = field(default=None, compare=False)


# ================================================================================================
# the comparison — what counts as no change
# ================================================================================================

@pytest.mark.parametrize("a, b", [
    pytest.param(float("nan"), float("nan"), id="two NaN objects"),
    pytest.param({"ref": float("nan")}, {"ref": float("nan")}, id="NaN in a dict"),
    pytest.param([50.0, 150.0], (50.0, 150.0), id="list and tuple"),
    pytest.param({"intervals_s": [[0.1, 0.2], [0.5, 0.9]]},
                 {"intervals_s": ((0.1, 0.2), (0.5, 0.9))}, id="nested pairs"),
    pytest.param(np.float64(5.0), 5.0, id="np.float64"),
    pytest.param(np.float32(0.5), 0.5, id="np.float32 of an exact value"),
    pytest.param(np.int64(3), 3, id="np.int64"),
    pytest.param(np.bool_(True), True, id="np.bool_"),
    pytest.param(np.float64("nan"), float("nan"), id="numpy NaN"),
    pytest.param(np.array([50.0, np.nan]), [50.0, float("nan")], id="array and list"),
    pytest.param(np.array([1.0, 2.0]), np.array([1.0, 2.0]), id="two arrays"),
    pytest.param({"ml": {"pre": {"window": [1, 2]}, "enter": np.float64(0.6)}},
                 {"ml": {"pre": {"window": (1, 2)}, "enter": 0.6}}, id="nested"),
    pytest.param(5, 5.0, id="int and float"),
    pytest.param(_Declared(float("nan"), (50.0,)), _Declared(float("nan"), [50.0]),
                 id="dataclass field by field"),
    pytest.param(_Declared(1.0, cache=object()), _Declared(1.0, cache=object()),
                 id="compare=False field skipped"),
])
def test_these_pairs_are_one_setting(a, b):
    assert _settings_equal(a, b) and _settings_equal(b, a)


@pytest.mark.parametrize("a, b", [
    pytest.param(5.0, 6.0, id="a value"),
    pytest.param(None, float("nan"), id="None is not NaN"),
    pytest.param(None, 0.0, id="None is not 0"),
    pytest.param(None, "", id="None is not empty"),
    pytest.param([50.0, 150.0], [150.0, 50.0], id="order"),
    pytest.param([50.0], [50.0, 150.0], id="length"),
    pytest.param({"k_in": 5.0}, {"k_in": 5.0, "k_out": 2.0}, id="a key"),
    pytest.param({"ml": {"enter": 0.6}}, {"ml": {"enter": 0.7}}, id="one level down"),
    pytest.param([50.0], {0: 50.0}, id="container kind"),
    pytest.param(np.float32(0.1), 0.1, id="np.float32 of an inexact value"),
    pytest.param(np.array([[1.0, 2.0]]), [1.0, 2.0], id="array shape"),
    pytest.param(_Declared(1.0), _Declared(2.0), id="a dataclass field"),
    pytest.param(_Declared(1.0), _AlsoDeclared(1.0), id="dataclass type"),
    pytest.param(_Declared(1.0), {"rate_hz": 1.0, "lines": ()}, id="dataclass and dict"),
    pytest.param("detect.schmitt", "detect.cusum", id="a string"),
])
def test_everything_else_is_exact(a, b):
    assert not _settings_equal(a, b) and not _settings_equal(b, a)


@pytest.mark.parametrize("state", [
    pytest.param(_form(5.0, ref=float("nan")), id="form"),
    pytest.param(ChannelCalibration(gain=0.1, offset=-2.5, target_units="mV",
                                    provenance=Provenance.USER, note="rig B",
                                    kind=ChannelKind.VOLTAGE), id="calibration"),
    pytest.param(DECLARED_50K, id="time base"),
    pytest.param({"declaration": None}, id="file's own time base"),
    pytest.param({"override": AcquisitionBandwidth(
        cutoff_hz=10_000.0, status="reported", source="user", filter_type="bessel",
        filter_order=8, header_fields=(("lpf", "10 kHz"),))}, id="bandwidth override"),
])
def test_every_real_kind_of_state_equals_its_own_copy(state):
    """THE NO-PHANTOM INVARIANT. A checkpoint stores a deep copy; when nothing changed, the
    next read must compare equal to it, or every checkpoint pushes a step nobody made."""
    assert _settings_equal(state, copy.deepcopy(state))


def test_a_comparison_that_cannot_answer_reads_as_a_change_and_never_raises():
    class Stubborn:
        def __eq__(self, other):
            raise TypeError("no answer")
        __hash__ = object.__hash__

    one = Stubborn()
    assert _settings_equal({"x": one}, {"x": Stubborn()}) is False
    assert _settings_equal(one, one) is True       # identity still answers


# ================================================================================================
# recording
# ================================================================================================

def test_first_sight_is_adopted_silently(app):
    h = _history(app)
    heard = _listen(h)
    assert _record(h, app, FORM, _form(5.0)) is False
    assert not h.can_undo(FORM) and not h.has_any() and heard == []
    # ...and it is the baseline the first real edit is measured from.
    assert _record(h, app, FORM, _form(6.0)) is True
    assert h.undo(FORM)
    assert app.states[FORM] == _form(5.0)


def test_a_state_equal_to_the_last_one_pushes_nothing(app):
    h = _history(app)
    heard = _listen(h)
    _record(h, app, FORM, _form(5.0, ref=float("nan")))
    for same in (_form(5.0, ref=float("nan")),                          # a fresh NaN object
                 _form(np.float64(5.0), ref=np.float64("nan")),          # numpy scalars
                 _form(5.0, ref=float("nan"), notch_freqs=(50.0, 150.0)),  # a tuple for the list
                 copy.deepcopy(app.states[FORM])):
        assert _record(h, app, FORM, same) is False
    assert h.checkpoint_all() is False
    assert not h.has_any() and heard == []


def test_one_edit_checkpointed_twice_is_one_step(app):
    """Wiring ``checkpoint`` to two signals for one edit cannot invent a step (design §3.3)."""
    h = _history(app)
    _record(h, app, CAL, ChannelCalibration(gain=1.0))
    app.states[CAL] = ChannelCalibration(gain=0.1, target_units="mV")
    assert h.checkpoint(CAL) is True
    assert h.checkpoint(CAL) is False
    assert h.undo_depth(CAL) == 1


def test_a_change_pushes_one_step_and_announces_it(app):
    h = _history(app)
    heard = _listen(h)
    _record(h, app, TIME_BASE, {"declaration": None})
    assert _record(h, app, TIME_BASE, DECLARED_50K, label="Declare 50 kHz") is True
    assert heard == [("step", TIME_BASE, "Declare 50 kHz"), ("changed",)]
    assert h.last_step_scope == TIME_BASE
    assert h.can_undo(TIME_BASE) and h.undo_label(TIME_BASE) == "Declare 50 kHz"
    assert h.undo_depth(TIME_BASE) == 1 and not h.can_redo(TIME_BASE)
    assert h.scopes_with_history() == (TIME_BASE,)


def test_checkpoint_all_commits_every_scope_it_has_seen(app):
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, CAL, ChannelCalibration(gain=1.0))
    _record(h, app, BANDWIDTH, {"override": None})
    app.states[FORM] = _form(6.0)
    app.states[BANDWIDTH] = OVERRIDE_10K
    assert h.checkpoint_all() is True
    assert h.undo_depth(FORM) == 1 and h.undo_depth(BANDWIDTH) == 1
    assert not h.can_undo(CAL)
    assert set(h.scopes_with_history()) == {FORM, BANDWIDTH}


def test_no_scope_on_screen_is_a_scope_with_nothing(app):
    h = _history(app)
    assert h.checkpoint(None) is False and h.undo(None) == "" and h.redo(None) == ""
    assert not h.can_undo(None) and not h.can_redo(None)
    assert h.undo_label(None) == "" and h.undo_depth(None) == 0 and h.depth_dropped(None) == 0


# ================================================================================================
# labels
# ================================================================================================

def test_describe_names_the_step_from_both_ends(app):
    asked = []

    def describe(scope, before, after):
        asked.append((scope, copy.deepcopy(before), copy.deepcopy(after)))
        return f"Change k_in {before['k_in']:g} → {after['k_in']:g}"

    h = SettingsHistory(app.read, app.write, describe)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0))
    assert asked == [(FORM, _form(5.0), _form(6.0))]
    assert h.undo_label(FORM) == "Change k_in 5 → 6"
    assert h.undo(FORM) == "Change k_in 5 → 6"
    assert h.redo_label(FORM) == "Change k_in 5 → 6"


def test_an_explicit_label_wins_and_describe_is_not_asked(app):
    asked = []
    h = SettingsHistory(app.read, app.write, lambda *a: asked.append(a) or "derived")
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0), label="Recall protocol ‘HG student’")
    assert h.undo_label(FORM) == "Recall protocol ‘HG student’"
    assert asked == []


def test_without_describe_a_step_is_called_change_settings(app):
    h = _history(app)
    _record(h, app, CAL, ChannelCalibration(gain=1.0))
    _record(h, app, CAL, ChannelCalibration(gain=0.1))
    assert h.undo_label(CAL) == "Change settings"


@pytest.mark.parametrize("describe", [
    pytest.param(lambda *a: 1 / 0, id="raises"),
    pytest.param(lambda *a: "", id="empty"),
    pytest.param(lambda *a: None, id="None"),
])
def test_a_failing_describe_costs_the_label_never_the_step(app, describe):
    h = SettingsHistory(app.read, app.write, describe)
    _record(h, app, FORM, _form(5.0))
    assert _record(h, app, FORM, _form(6.0)) is True
    assert h.undo_label(FORM) == "Change settings"


# ================================================================================================
# undo and redo
# ================================================================================================

def test_undo_writes_the_state_before_and_redo_the_state_after(app):
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0), label="Change k_in 5 → 6")
    _record(h, app, FORM, _form(7.0), label="Change k_in 6 → 7")
    heard = _listen(h)

    assert h.undo(FORM) == "Change k_in 6 → 7"
    assert app.writes[-1] == (FORM, _form(6.0))
    assert h.undo(FORM) == "Change k_in 5 → 6"
    assert app.writes[-1] == (FORM, _form(5.0))
    assert h.undo(FORM) == "" and not h.can_undo(FORM)
    assert h.redo_depth(FORM) == 2 and h.redo_label(FORM) == "Change k_in 5 → 6"

    # Redo clears nothing: the second step is still there after the first is redone.
    assert h.redo(FORM) == "Change k_in 5 → 6"
    assert app.writes[-1] == (FORM, _form(6.0))
    assert h.redo(FORM) == "Change k_in 6 → 7"
    assert app.writes[-1] == (FORM, _form(7.0))
    assert h.redo(FORM) == "" and len(app.writes) == 4
    # A move is announced by `changed` alone: undoing is not a new step.
    assert heard == [("changed",)] * 4


def test_undo_takes_back_an_edit_not_yet_committed(app):
    """The 700 ms pause has not fired: Ctrl+Z must undo THAT edit, not jump past it."""
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0))
    heard = _listen(h)
    app.states[FORM] = _form(7.0)                  # typed, never checkpointed
    h.undo(FORM)
    assert app.states[FORM] == _form(6.0), "Ctrl+Z jumped past the edit that was pending"
    assert heard[0] == ("step", FORM, "Change settings"), (
        "the pending edit was not pushed as a step of its own, so MainWindow never journals it")
    assert h.undo_depth(FORM) == 1 and h.redo_depth(FORM) == 1
    h.redo(FORM)
    assert app.states[FORM] == _form(7.0)


def test_what_a_write_settles_on_is_adopted_not_recorded(app):
    """A writer may clamp, as the frequency boxes clamp to a slower recording's Nyquist. That
    is the app's doing: recording it would push a phantom step and end the redo branch."""
    def clamping_write(scope, state):
        app.write(scope, {**state, "k_in": min(state["k_in"], 8.0)})

    h = SettingsHistory(app.read, clamping_write)
    _record(h, app, FORM, _form(9.0))
    _record(h, app, FORM, _form(4.0))
    assert h.undo(FORM)
    assert app.states[FORM]["k_in"] == 8.0
    assert h.checkpoint(FORM) is False, "the writer's clamp was recorded as the user's edit"
    assert h.can_redo(FORM)


# ================================================================================================
# redo per scope (owner decision 2)
# ================================================================================================

def test_a_new_step_ends_only_its_own_scopes_redo(app):
    h = _history(app)
    for scope, (old, new) in {
        FORM: (_form(5.0), _form(6.0)),
        FORM_B: (_form(5.0), _form(7.0)),
        CAL: (ChannelCalibration(gain=1.0), ChannelCalibration(gain=0.1)),
    }.items():
        _record(h, app, scope, old)
        _record(h, app, scope, new)
        h.undo(scope)
    assert h.can_redo(FORM) and h.can_redo(FORM_B) and h.can_redo(CAL)

    _record(h, app, FORM, _form(8.0))
    assert not h.can_redo(FORM)
    assert h.can_redo(FORM_B) and h.can_redo(CAL), (
        "a step on one table's form ended another scope's redo branch")


# ================================================================================================
# recording muted: suspended, and while an undo writes
# ================================================================================================

def test_while_suspended_a_change_is_adopted_not_recorded(app):
    h = _history(app)
    heard = _listen(h)
    _record(h, app, CAL, ChannelCalibration(gain=1.0))
    h.set_suspended(True)                          # a project restore
    assert _record(h, app, CAL, ChannelCalibration(gain=0.1, note="restored")) is False
    h.set_suspended(False)
    assert h.checkpoint(CAL) is False, "the restored state was not adopted"
    assert not h.can_undo(CAL) and heard == []
    # The restored state is the baseline the next edit returns to.
    _record(h, app, CAL, ChannelCalibration(gain=0.2, note="restored"))
    h.undo(CAL)
    assert app.states[CAL] == ChannelCalibration(gain=0.1, note="restored")


def test_a_checkpoint_fired_from_inside_write_pushes_nothing(app):
    """A write emits the app's own signals (``calibrationChanged``, the form's
    ``paramsChanged``), and MainWindow's checkpoint slots hang off them."""
    fired = []

    def chatty_write(scope, state):
        app.write(scope, state)
        fired.append((h.is_applying(), h.checkpoint(scope), h.checkpoint_all()))

    h = SettingsHistory(app.read, chatty_write)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0))
    assert h.undo(FORM)
    assert fired == [(True, False, False)]
    assert (h.undo_depth(FORM), h.redo_depth(FORM)) == (0, 1) and not h.is_applying()
    assert h.redo(FORM)
    assert fired[-1] == (True, False, False)
    assert (h.undo_depth(FORM), h.redo_depth(FORM)) == (1, 0)


def test_what_a_write_moves_in_another_scope_is_adopted_there(app):
    """Muting ADOPTS, it does not merely decline. Undoing a declaration clamps the form's
    low-pass to the file's slower Nyquist, and MainWindow's checkpoint slot runs inside that
    write. A clamp that was only declined would be recorded by the form's next checkpoint as
    an edit nobody made, offered back by Ctrl+Z, and would end the form's redo branch."""
    def declaring_write(scope, state):
        app.write(scope, state)
        if scope == TIME_BASE and state["declaration"] is None:    # the file's own 25 kHz
            app.states[FORM] = {**app.states[FORM], "lowpass_hz": 12_500.0}
            h.checkpoint_all()                     # the slot on sampleRateChanged

    h = SettingsHistory(app.read, declaring_write)
    _record(h, app, TIME_BASE, {"declaration": None})
    _record(h, app, TIME_BASE, DECLARED_50K, label="Declare 50 kHz")
    _record(h, app, FORM, _form(5.0, lowpass_hz=20_000.0))
    _record(h, app, FORM, _form(6.0, lowpass_hz=20_000.0), label="Change k_in 5 → 6")
    h.undo(FORM)                                   # the form's own redo branch
    assert h.undo(TIME_BASE) == "Declare 50 kHz"
    assert app.states[FORM] == _form(5.0, lowpass_hz=12_500.0)
    assert h.checkpoint(FORM) is False, "the app's clamp was recorded as the user's edit"
    assert not h.can_undo(FORM) and h.redo_label(FORM) == "Change k_in 5 → 6"
    # The clamped form is the baseline the next edit returns to.
    _record(h, app, FORM, _form(7.0, lowpass_hz=12_500.0))
    h.undo(FORM)
    assert app.states[FORM] == _form(5.0, lowpass_hz=12_500.0)


# ================================================================================================
# the depth ceiling
# ================================================================================================

def test_the_depth_ceiling_drops_the_oldest_step_counts_it_and_says_so(app):
    h = _history(app, depth=3)
    heard = _listen(h)
    _record(h, app, FORM, _form(0.0))
    for k in range(1, 6):
        _record(h, app, FORM, _form(float(k)), label=f"step {k}")
    assert h.undo_depth(FORM) == 3 and h.depth_dropped(FORM) == 2
    assert [e for e in heard if e[0] == "dropped"] == [("dropped", FORM, 1), ("dropped", FORM, 2)]
    # Announced before the step that caused it, as the event stack does.
    assert heard[-3:] == [("dropped", FORM, 2), ("step", FORM, "step 5"), ("changed",)]
    # The far end went: Ctrl+Z walks back to step 2's result and no further.
    assert [h.undo(FORM) for _ in range(4)] == ["step 5", "step 4", "step 3", ""]
    assert app.states[FORM] == _form(2.0)


def test_the_default_ceiling_is_max_depth_and_belongs_to_one_scope(app):
    h = _history(app)
    _record(h, app, FORM, _form(0.0))
    _record(h, app, FORM, _form(1.0))              # one step on the form...
    _record(h, app, CAL, ChannelCalibration(gain=1.0))
    for k in range(1, MAX_DEPTH + 2):              # ...and MAX_DEPTH + 1 on the calibration
        _record(h, app, CAL, ChannelCalibration(gain=1.0 + k))
    assert h.undo_depth(CAL) == MAX_DEPTH and h.depth_dropped(CAL) == 1
    assert h.undo_depth(FORM) == 1 and h.depth_dropped(FORM) == 0, (
        "one scope's sprint cost another scope a step")


# ================================================================================================
# move_scope — the steps made before a recording's first event table
# ================================================================================================

def test_move_scope_gives_a_fresh_table_everything_the_form_gathered(app):
    h = _history(app, depth=2)
    _record(h, app, FORM_NO_TABLE, _form(5.0))
    for k in (6.0, 7.0, 8.0):
        _record(h, app, FORM_NO_TABLE, _form(k), label=f"k_in {k:g}")
    h.undo(FORM_NO_TABLE)                          # 7 on screen, "k_in 8" redoable
    heard = _listen(h)

    app.states[FORM] = app.states.pop(FORM_NO_TABLE)   # the form now answers to the table's key
    h.move_scope(FORM_NO_TABLE, FORM)
    assert heard == [("changed",)]
    assert not h.can_undo(FORM_NO_TABLE) and not h.can_redo(FORM_NO_TABLE)
    assert h.depth_dropped(FORM_NO_TABLE) == 0 and h.depth_dropped(FORM) == 1
    assert h.last_step_scope == FORM
    assert h.undo_label(FORM) == "k_in 7" and h.redo_label(FORM) == "k_in 8"
    assert h.redo(FORM) == "k_in 8" and app.states[FORM] == _form(8.0)


def test_move_scope_carries_the_pristine_copy(app):
    """A form seen but never edited keeps its baseline under the table's key, so the first
    edit after the detection is measured from it rather than adopted as a first sight."""
    h = _history(app)
    heard = _listen(h)
    _record(h, app, FORM_NO_TABLE, _form(5.0))
    h.move_scope(FORM_NO_TABLE, FORM)
    assert heard == [], "moving a scope with no steps announced a change"
    assert _record(h, app, FORM, _form(6.0)) is True
    h.undo(FORM)
    assert app.states[FORM] == _form(5.0)


def test_move_scope_keeps_the_new_scopes_own_pristine_copy(app):
    h = _history(app)
    _record(h, app, FORM_NO_TABLE, _form(5.0))
    _record(h, app, FORM, _form(6.0))              # the table's key, rebased onto what it showed
    h.move_scope(FORM_NO_TABLE, FORM)
    assert h.checkpoint(FORM) is False, "the older pristine copy replaced the newer one"


def test_move_scope_puts_the_old_steps_before_the_new_ones(app):
    h = _history(app, depth=3)
    heard = _listen(h)
    _record(h, app, FORM_NO_TABLE, _form(0.0))
    _record(h, app, FORM_NO_TABLE, _form(1.0), label="old 1")
    _record(h, app, FORM_NO_TABLE, _form(2.0), label="old 2")
    _record(h, app, FORM, _form(2.0))
    _record(h, app, FORM, _form(3.0), label="new 3")
    _record(h, app, FORM, _form(4.0), label="new 4")
    heard.clear()

    h.move_scope(FORM_NO_TABLE, FORM)
    # Four steps against a ceiling of three: the oldest of them, "old 1", is the one that goes.
    assert heard == [("dropped", FORM, 1), ("changed",)]
    assert [h.undo(FORM) for _ in range(4)] == ["new 4", "new 3", "old 2", ""]
    assert app.states[FORM] == _form(1.0)


@pytest.mark.parametrize("new_has", ["redo", "undo"])
def test_move_scope_never_splices_two_redo_branches(app, new_has):
    """A redo branch is the future of the newest step. Once the old steps sit under the new
    ones, the old branch would rewrite the form from behind them, so it is dropped."""
    h = _history(app)
    _record(h, app, FORM_NO_TABLE, _form(0.0))
    _record(h, app, FORM_NO_TABLE, _form(1.0), label="old 1")
    h.undo(FORM_NO_TABLE)                          # the old scope's branch: "old 1"
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0), label="new 6")
    if new_has == "redo":
        h.undo(FORM)                               # the new scope's own branch: "new 6"
    h.move_scope(FORM_NO_TABLE, FORM)
    assert h.redo_label(FORM) == ("new 6" if new_has == "redo" else "")
    assert h.redo_depth(FORM) == (1 if new_has == "redo" else 0)


# ================================================================================================
# lifetime
# ================================================================================================

def test_drop_doc_forgets_every_scope_of_that_recording_and_no_other(app):
    h = _history(app, depth=1)
    _record(h, app, FORM_B, _form(5.0))
    _record(h, app, FORM_B, _form(7.0))
    for scope, (old, new) in {
        FORM: (_form(5.0), _form(6.0)),
        CAL: (ChannelCalibration(gain=1.0), ChannelCalibration(gain=0.1)),
        TIME_BASE: ({"declaration": None}, DECLARED_50K),
        BANDWIDTH: ({"override": None}, OVERRIDE_10K),
    }.items():
        _record(h, app, scope, old)
        _record(h, app, scope, new)
    _record(h, app, FORM, _form(9.0))              # past the ceiling of 1: one dropped
    h.undo(CAL)                                    # and a redo entry on doc-A too
    assert h.depth_dropped(FORM) == 1 and h.last_step_scope == FORM
    heard = _listen(h)

    h.drop_doc("doc-A")
    assert heard == [("changed",)]
    assert h.scopes_with_history() == (FORM_B,)
    assert h.depth_dropped(FORM) == 0 and h.last_step_scope is None
    # The pristine copies went too: an edit now is a first sight, not a step.
    assert _record(h, app, FORM, _form(10.0)) is False
    assert h.can_undo(FORM_B)

    h.drop_doc("doc-Z")                            # nothing of it held: silent
    assert heard == [("changed",)]


def test_clear_forgets_everything(app):
    h = _history(app, depth=1)
    heard = _listen(h)
    h.clear()
    assert heard == [], "clearing an empty history announced a change"

    for k in (5.0, 6.0, 7.0):
        _record(h, app, FORM, _form(k))
    _record(h, app, CAL, ChannelCalibration(gain=1.0))
    _record(h, app, CAL, ChannelCalibration(gain=0.1))
    h.undo(CAL)
    heard.clear()

    h.clear()
    assert heard == [("changed",)]
    assert not h.has_any() and h.scopes_with_history() == ()
    assert h.depth_dropped(FORM) == 0 and h.last_step_scope is None
    assert _record(h, app, FORM, _form(8.0)) is False   # first sight again


def test_rebase_adopts_what_arrived_without_a_step(app):
    """A switch, a project restore or a seed from the stored run puts a state on screen that
    the user did not type. Undo must return to it, never past it to a state they never had."""
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    heard = _listen(h)
    app.states[FORM] = _form(9.0)                  # the seed
    h.rebase(FORM)
    assert h.checkpoint(FORM) is False and heard == []
    _record(h, app, FORM, _form(10.0))
    h.undo(FORM)
    assert app.states[FORM] == _form(9.0)


def test_forget_drops_the_pristine_copy_and_keeps_the_steps(app):
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0))
    h.forget(FORM)
    assert _record(h, app, FORM, _form(7.0)) is False   # adopted, as on first sight
    assert h.undo_depth(FORM) == 1


# ================================================================================================
# isolation, and the far side failing
# ================================================================================================

def test_editing_the_live_state_in_place_cannot_edit_the_history(app):
    """A reader that hands out the dict it goes on editing: the history must hold its own copy,
    or the edit compares equal to itself and is never recorded."""
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    live = app.states[FORM]
    live["k_in"] = 6.0                             # the same object, edited in place...
    live["ml"]["enter"] = 0.9                      # ...one level down too
    assert h.checkpoint(FORM) is True
    h.undo(FORM)
    assert app.states[FORM] == _form(5.0)


def test_editing_what_write_installed_cannot_edit_the_history(app):
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0))
    h.undo(FORM)
    handed = app.states[FORM]
    handed["k_in"] = 99.0                          # the app scribbles on what it was handed...
    handed["notch_freqs"].append(250.0)
    app.states[FORM] = _form(5.0)                  # ...and shows a fresh copy instead
    assert h.redo(FORM) and h.undo(FORM)           # the round trip rewrites the step's 'before'
    assert app.states[FORM] == _form(5.0), (
        "write was handed the step's own state, and the app's scribble rewrote history")


def test_read_returning_none_leaves_everything_as_it_was(app):
    h = _history(app)
    _record(h, app, CAL, ChannelCalibration(gain=1.0))
    _record(h, app, CAL, ChannelCalibration(gain=0.1))
    heard = _listen(h)
    app.readable = False                           # the scope cannot be read now
    assert h.checkpoint(CAL) is False and h.checkpoint_all() is False
    h.rebase(CAL)
    assert heard == [] and h.undo_depth(CAL) == 1
    # An undo still writes, and with nothing to read back it adopts the state it wrote.
    assert h.undo(CAL) == "Change settings"
    assert app.writes[-1] == (CAL, ChannelCalibration(gain=1.0))
    app.readable = True
    assert h.checkpoint(CAL) is False, (
        "the written state was not adopted while read() had no answer")
    assert h.redo(CAL) and app.states[CAL] == ChannelCalibration(gain=0.1)


def test_a_read_that_raises_reads_as_cannot_be_read(app):
    """``Session.sample_rate_declaration`` raises KeyError for a doc_id that has closed;
    a save's ``checkpoint_all`` must not die on it."""
    def closed(scope):
        raise KeyError(scope[1])

    h = SettingsHistory(closed, app.write)
    assert h.checkpoint(TIME_BASE) is False and h.checkpoint_all() is False
    h.rebase(TIME_BASE)
    assert not h.has_any()


@pytest.mark.parametrize("direction", ["undo", "redo"])
def test_a_write_that_raises_leaves_the_stacks_as_they_were(app, direction):
    h = _history(app)
    _record(h, app, FORM, _form(5.0))
    _record(h, app, FORM, _form(6.0), label="k_in 6")
    _record(h, app, FORM, _form(7.0), label="k_in 7")
    if direction == "redo":
        h.undo(FORM)
    stacks = (h.undo_depth(FORM), h.redo_depth(FORM), h.undo_label(FORM), h.redo_label(FORM))
    on_screen = copy.deepcopy(app.states[FORM])
    heard = _listen(h)

    app.refuse = RuntimeError("the panel refused the state")
    with pytest.raises(RuntimeError, match="refused"):
        getattr(h, direction)(FORM)
    assert (h.undo_depth(FORM), h.redo_depth(FORM),
            h.undo_label(FORM), h.redo_label(FORM)) == stacks
    assert heard == [] and not h.is_applying()
    assert app.states[FORM] == on_screen

    # The step is still on top, and recording is not stuck muted.
    app.refuse = None
    assert getattr(h, direction)(FORM) == "k_in 7"
    assert _record(h, app, FORM, _form(11.0)) is True
