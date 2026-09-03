"""H-3 — a new step on EITHER undo stack must end the redo branch on BOTH.

There is one Edit ▸ Redo and two stacks (the event tables and the hand-measurement table;
``core/history.py`` explains why they cannot be one). ``MainWindow._note_step``'s own
docstring states the rule — *"A new step invalidates the redo branch, exactly as it does
inside each stack"* — and it cleared only the JOURNAL. The entries stayed alive inside
whichever stack the journal was not naming, and ``_redo_target``'s fallback reaches
straight past an empty journal into them.

Measured on the tree as found, through a real MainWindow:

* curate 2 rejections → re-detect (wiped) → Ctrl+Z (restored) → add ONE hand measurement →
  the Edit menu still read **"Redo Re-run detection"**, enabled → press it → the 2
  rejections were destroyed again;
* 3 hand measurements → Remove → Ctrl+Z → reject one event → **"Redo Delete 3
  measurements"** → press it → all 3 gone.

The stale branch survives arbitrarily many LEGITIMATE undo/redo cycles on the other stack:
it is never consumed or aged out. The loss is recoverable by one further Undo, so what is
wrong is that it is unannounced BY CONTEXT — the user's last action was on the other table,
and a project save or a CSV export taken before they notice persists the reverted
``accepted`` column.

Deleting the ``_redo_target`` fallback would hide the entries and leave them in place,
which is why the tests below assert the STACKS, not only the menu.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600), (30_000, 30_500)]


def _signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "redo.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(_signal(),))


@pytest.fixture
def win(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    doc = _doc(tmp_path)
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    yield w, doc_id
    w._analysis_controller.abort()


def _detect(w, doc_id, *, min_dwell_s=1e-4):
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=min_dwell_s)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": min_dwell_s},
        calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))


def _rejections(w, doc_id) -> int:
    df = w._session.results.get(doc_id, 0, "detect.schmitt").payload.df
    return int((~df["accepted"].to_numpy()).sum())


def _add_measurement(w, a: int, b: int) -> None:
    """One hand measurement, through the real cursor tool."""
    mp = w._analysis_panel.measure_panel
    mp.set_active(True)
    w._trace_view.set_measure_region(a / FS, b / FS)
    assert mp.add_current() is True


def _menu(w) -> tuple[str, bool]:
    w._sync_undo_actions()
    return w._act_redo.text(), w._act_redo.isEnabled()


def _event_redo_depth(w) -> int:
    return len(getattr(w._history, "_redo"))


def _measure_redo_depth(w) -> int:
    return len(getattr(w._measure_history, "_redo"))


# ================================================================================================
# A2 — an EVENT redo survives a MEASUREMENT step
# ================================================================================================

def test_a_measurement_step_ends_the_event_redo_branch(win, qtbot):
    w, doc_id = win
    _detect(w, doc_id)
    df = w._session.results.get(doc_id, 0, "detect.schmitt").payload.df
    for row in (0, 1):
        w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[row]), False)
    w._history.checkpoint_all()
    assert _rejections(w, doc_id) == 2

    _detect(w, doc_id, min_dwell_s=1.2e-4)          # a re-detection wipes the curation
    qtbot.wait(5)
    assert _rejections(w, doc_id) == 0
    w._undo_events()
    qtbot.wait(5)
    assert _rejections(w, doc_id) == 2, "premise: Ctrl+Z brought the curation back"
    assert _menu(w)[1] is True, "premise: a redo is genuinely on offer at this point"
    assert _event_redo_depth(w) == 1

    _add_measurement(w, 25_000, 25_400)             # a step on the OTHER table
    qtbot.wait(5)

    assert _event_redo_depth(w) == 0, (
        "the event redo entry outlived a step on the measurement table — deleting the "
        "_redo_target fallback would hide this, not end it")
    text, enabled = _menu(w)
    assert enabled is False and "Re-run detection" not in text, text

    w._redo_events()
    qtbot.wait(5)
    assert _rejections(w, doc_id) == 2, "Redo destroyed the curation the user had restored"


# ================================================================================================
# A3 — a MEASUREMENT redo survives an EVENT step
# ================================================================================================

def test_an_event_step_ends_the_measurement_redo_branch(win, qtbot):
    w, doc_id = win
    _detect(w, doc_id)
    for k, (a, b) in enumerate([(25_000, 25_400), (35_000, 35_400), (45_000, 45_400)]):
        _add_measurement(w, a, b)
    qtbot.wait(5)
    mp = w._analysis_panel.measure_panel
    assert len(mp.measurements_df) == 3

    mp.set_measurements_df(mp.measurements_df.iloc[:0].copy())
    mp.measurementsChanged.emit()
    qtbot.wait(5)
    assert len(mp.measurements_df) == 0
    w._undo_events()
    qtbot.wait(5)
    assert len(mp.measurements_df) == 3, "premise: Ctrl+Z brought the rows back"
    assert _measure_redo_depth(w) == 1

    df = w._session.results.get(doc_id, 0, "detect.schmitt").payload.df
    w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    w._history.checkpoint_all()
    qtbot.wait(5)
    assert _rejections(w, doc_id) == 1, "premise: the event step landed"

    assert _measure_redo_depth(w) == 0, "the measurement redo entry outlived an event step"
    text, enabled = _menu(w)
    assert "Delete" not in text, text

    w._redo_events()
    qtbot.wait(5)
    assert len(mp.measurements_df) == 3, "Redo deleted the measurements again"
    assert _rejections(w, doc_id) == 1, "...and it must not have undone the event step either"


# ================================================================================================
# the properties the verifier measured, which a menu-only fix would not close
# ================================================================================================

def test_the_stale_branch_cannot_be_revived_by_cycling_the_other_stack(win, qtbot):
    """The verifier's sharpest measurement: the entry survived a full legitimate
    undo→redo cycle on the other stack and re-surfaced the instant the journal emptied
    again. It is never consumed or aged out, so hiding it is not ending it."""
    w, doc_id = win
    _detect(w, doc_id)
    df = w._session.results.get(doc_id, 0, "detect.schmitt").payload.df
    w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    w._history.checkpoint_all()
    _detect(w, doc_id, min_dwell_s=1.2e-4)
    qtbot.wait(5)
    w._undo_events()
    qtbot.wait(5)
    assert _rejections(w, doc_id) == 1

    _add_measurement(w, 25_000, 25_400)
    qtbot.wait(5)
    w._undo_events()                # undo the measurement (journal now names measurements)
    qtbot.wait(5)
    w._redo_events()                # redo it (journal empties again)
    qtbot.wait(5)

    assert _event_redo_depth(w) == 0
    w._redo_events()
    qtbot.wait(5)
    assert _rejections(w, doc_id) == 1


def test_a_redo_chain_on_one_stack_is_not_broken_by_its_own_replays(win, qtbot):
    """CONTROL. The invalidation is about NEW steps, not about applying old ones — undo,
    undo, redo, redo on one table has to keep working, or the fix has cost the feature."""
    w, doc_id = win
    _detect(w, doc_id)
    df = w._session.results.get(doc_id, 0, "detect.schmitt").payload.df
    w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    w._history.checkpoint_all()
    w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[1]), False)
    w._history.checkpoint_all()
    assert _rejections(w, doc_id) == 2

    w._undo_events(); qtbot.wait(5)
    w._undo_events(); qtbot.wait(5)
    assert _rejections(w, doc_id) == 0
    w._redo_events(); qtbot.wait(5)
    w._redo_events(); qtbot.wait(5)
    assert _rejections(w, doc_id) == 2


def test_the_measurement_stack_announces_its_own_pushes(win, qtbot):
    """The wiring, not the outcome. The journal used to hang off one WRAPPER, and two real
    pushes never pass through it (``_move``'s leading checkpoint, and the restore's
    adoption call) — an unjournalled step is a step that invalidated nothing."""
    w, _doc_id = win
    assert hasattr(w._measure_history, "stepRecorded"), (
        "MeasurementHistory announces no step, so the journal can only learn about the "
        "pushes that happen to come through _checkpoint_measurements")
    seen: list[str] = []
    w._measure_history.stepRecorded.connect(seen.append)
    _add_measurement(w, 25_000, 25_400)
    qtbot.wait(5)
    assert seen == ["Add measurement"], seen


def test_both_stacks_expose_the_same_invalidation_door():
    """The invariant is one method name on both classes, called from one place. Two
    different spellings is how the rule ends up applied to one of them again."""
    from nanoscope.core.history import EventHistory, MeasurementHistory

    for cls in (EventHistory, MeasurementHistory):
        assert callable(getattr(cls, "clear_redo", None)), (
            f"{cls.__name__} has no clear_redo: the redo branch cannot be ended from outside")

    import inspect
    import re

    # MATCH THE CALL, NOT THE OLD SPELLING. This used to assert the literal text
    # "_history.clear_redo()" / "_measure_history.clear_redo()". Those calls now take a
    # scope argument -- clear_redo(doc_ids) / clear_redo(doc_keys) -- because the
    # unscoped pair was a data-loss bug: one curation click on recording B ended
    # recording A's redo branch, making an undone hand curation on A unrecoverable. The
    # old assertion kept passing only because the comment explaining that bug quotes the
    # old spelling verbatim, i.e. it was matching a COMMENT while the code beneath it had
    # changed. The invariant this test exists for is unchanged and is what is checked
    # here: both stacks are invalidated from this one door, whatever the arguments.
    src = inspect.getsource(MainWindow._note_step)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    for attr in ("_history", "_measure_history"):
        assert re.search(rf"\b{attr}\.clear_redo\(", code), (
            f"_note_step must end the redo branch on BOTH stacks ({attr}.clear_redo is "
            f"missing); it is the one door every push on either goes through")
