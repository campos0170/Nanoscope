"""JM-02 / JM-05 — THE UNDO HISTORY OF ONE RECORDING IS NOT SPENDABLE BY ANOTHER.

Two data-loss defects with one shape: a per-recording promise implemented over a
session-wide list. ``docs/USER_GUIDE.md`` §9b — *"The history is per event table"* — and
*"Closing a recording is the one remaining one-way door … Nothing else in the app is
one-way."* Both were false across recordings, measured through a real MainWindow:

* **JM-02 (redo).** ``EventHistory.clear_redo()`` dropped the WHOLE redo list — every key —
  and ``MainWindow._note_step`` called it on every step recorded on EITHER stack. So:
  reject 2 events on recording A, Ctrl+Z (Edit reads *"Redo Curate event"*, enabled),
  switch to recording B and reject one event, come back to A — *"Redo — nothing to redo on
  this event table"*, ``can_redo`` False, and the undone rejection is in no stack at all.
  Nothing in the app could put it back and nothing announced the loss. The docstring
  justified the wipe by the CROSS-STACK case (a hand measurement invalidating the event
  redo), which is real — but only within one recording.
* **JM-05 (undo).** ``MAX_DEPTH`` (30) capped the whole session and ``_trim()`` deleted
  ``self._undo[0]``, the globally oldest entry, regardless of key — while every read is per
  key. Measured on the owner's quick ABF: *Merge all* took 115 events to 87 and Edit read
  *"Undo Merge events (115 → 87)"*; 31 ordinary accept/reject clicks on a DIFFERENT
  recording evicted it, after which Edit read *"Undo — nothing to undo on this event
  table"* — the wording used for a table nobody has ever edited.

The suite that should have caught both had each axis and never crossed them:
``tests/gui/test_redo_branch.py``'s fixture installs ONE document, and
``tests/gui/test_undo_events.py::test_the_stack_is_bounded`` pushes 30+ steps on ONE key
and asserts only the global depth. So every test below has TWO recordings.

The controls matter as much as the defects: the cross-stack invalidation the H-3 fix added
is still required WITHIN a recording (``test_redo_branch.py`` owns those), and the memory
ceiling still has to bind — a correctness fix that traded the byte ceiling for unbounded
RAM would be a worse defect than the one it closed (``core/history.py``: one events table
can be 1.1 GB).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.history import MAX_BYTES, MAX_DEPTH
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600),
          (30_000, 30_500), (40_000, 40_400), (50_000, 50_700)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(_signal(),))


@pytest.fixture
def two(qapp, qtbot, tmp_path):
    """TWO recordings, both detected — the configuration neither prior suite had."""
    w = MainWindow()
    qtbot.addWidget(w)
    ids = []
    for name in ("pore_A.dat", "pore_B.dat"):
        doc = _doc(tmp_path, name)
        doc_id = w._session.add_document(doc)
        w._create_default_layers(doc)
        ids.append(doc_id)
    for doc_id in ids:
        w._on_selection(doc_id, 0)
        _detect(w, doc_id)
    yield w, ids[0], ids[1]
    w._analysis_controller.abort()


def _detect(w, doc_id, *, min_dwell_s=1e-4) -> None:
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=min_dwell_s)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": min_dwell_s},
        calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _rejections(w, doc_id) -> int:
    return int((~_frame(w, doc_id)["accepted"].to_numpy(dtype=bool)).sum())


def _reject_row(w, doc_id, row: int) -> None:
    """One accept/reject click, through the real table model on the selected recording."""
    w._on_selection(doc_id, 0)
    df = _frame(w, doc_id)
    w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[row]), False)
    w._history.checkpoint_all()


def _toggle_row(w, doc_id, row: int) -> None:
    """FLIP one row's acceptance through the real table model, so every call is a genuine
    step (a click that re-writes the value already there moves nothing and — correctly —
    records nothing)."""
    w._on_selection(doc_id, 0)
    df = _frame(w, doc_id)
    row = row % len(df)
    now = bool(df["accepted"].iloc[row])
    w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[row]), not now)
    w._history.checkpoint_all()


def _menu(w) -> tuple[str, str]:
    w._sync_undo_actions()
    return w._act_undo.text(), w._act_redo.text()


def _key(doc_id):
    return (doc_id, 0, "detect.schmitt")


# ================================================================================================
# JM-02 — a step on recording B leaves recording A's redo branch alone
# ================================================================================================
def test_a_curation_click_on_another_recording_leaves_this_redo_branch_alone(two, qtbot):
    """THE MEASURED DEFECT, end to end and through the real Edit actions."""
    w, a, b = two
    _reject_row(w, a, 0)
    _reject_row(w, a, 1)
    assert _rejections(w, a) == 2

    w._on_selection(a, 0)
    w._undo_events()
    qtbot.wait(5)
    assert _rejections(w, a) == 1, "premise: Ctrl+Z took one rejection back"
    assert w._history.can_redo(_key(a)) is True
    assert "Redo Curate event" in _menu(w)[1]

    _reject_row(w, b, 0)                       # one ordinary click on the OTHER recording
    qtbot.wait(5)
    assert _rejections(w, b) == 1, "premise: the step on B landed"

    w._on_selection(a, 0)
    assert w._history.can_redo(_key(a)) is True, (
        "recording B's curation destroyed recording A's redo branch — the loss JM-02 "
        "measured, with no dialog, no status line and no way back")
    assert "Redo Curate event" in _menu(w)[1]
    w._redo_events()
    qtbot.wait(5)
    assert _rejections(w, a) == 2, "Redo restored nothing: the undone rejection is gone"


def test_the_loss_is_symmetric_so_the_test_is_too(two, qtbot):
    """The verifier measured it in both directions; a fix that only protected the first
    recording opened would pass the test above and still lose work."""
    w, a, b = two
    _reject_row(w, b, 0)
    w._on_selection(b, 0)
    w._undo_events()
    qtbot.wait(5)
    assert w._history.can_redo(_key(b)) is True

    _reject_row(w, a, 0)
    qtbot.wait(5)
    w._on_selection(b, 0)
    assert w._history.can_redo(_key(b)) is True
    w._redo_events()
    qtbot.wait(5)
    assert _rejections(w, b) == 1


def test_a_hand_measurement_on_one_recording_spares_the_others_event_redo(two, qtbot):
    """The CROSS-STACK half, scoped. A step on the hand-measurement table still ends the
    event redo branch of the recording it was measured on (``test_redo_branch.py`` owns
    that case) — but only that one. The old code ended every recording's."""
    w, a, b = two
    for doc_id in (a, b):
        _reject_row(w, doc_id, 0)
        w._on_selection(doc_id, 0)
        w._undo_events()
        qtbot.wait(5)
        assert w._history.can_redo(_key(doc_id)) is True

    # A hand measurement drawn on recording B, through the real cursor tool.
    w._on_selection(b, 0)
    mp = w._analysis_panel.measure_panel
    mp.set_active(True)
    w._trace_view.set_measure_region(25_000 / FS, 25_400 / FS)
    assert mp.add_current() is True
    qtbot.wait(5)

    assert w._history.can_redo(_key(b)) is False, (
        "the cross-stack rule still has to hold within one recording")
    assert w._history.can_redo(_key(a)) is True, (
        "a measurement on recording B ended recording A's event redo branch")


def test_a_step_on_one_recording_spares_another_recordings_measurement_redo(two, qtbot):
    """The same rule on the other stack. The hand-measurement table is ONE table for the
    session, but its rows name their recording (``doc_key``), so a curation click on
    recording A must not end a measurement redo branch that only rewrites B's rows."""
    w, a, b = two
    w._on_selection(b, 0)
    mp = w._analysis_panel.measure_panel
    mp.set_active(True)
    w._trace_view.set_measure_region(25_000 / FS, 25_400 / FS)
    assert mp.add_current() is True
    qtbot.wait(5)
    assert len(mp.measurements_df) == 1

    mp.set_measurements_df(mp.measurements_df.iloc[:0].copy())
    mp.measurementsChanged.emit()
    qtbot.wait(5)
    w._undo_events()                            # bring the row back; the redo branch is armed
    qtbot.wait(5)
    assert len(mp.measurements_df) == 1
    assert w._measure_history.can_redo() is True

    _reject_row(w, a, 0)                        # an event step on the OTHER recording
    qtbot.wait(5)
    assert w._measure_history.can_redo() is True, (
        "curating recording A ended a measurement redo branch that belongs to B")


def test_two_detectors_on_one_channel_do_not_end_each_others_redo(two, qtbot):
    """The finest boundary of the same rule: a stack is per (recording, channel, DETECTOR),
    so an ML step must not reach into the Schmitt table's redo branch either."""
    w, a, _b = two
    w._on_selection(a, 0)
    _reject_row(w, a, 0)
    w._undo_events()
    qtbot.wait(5)
    assert w._history.can_redo(_key(a)) is True

    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1.3e-4)
    w._session.add_result(AnalysisResult(
        module_id="detect.ml", module_version="1.0", doc_id=a, channel=0, params={},
        calibration_fingerprint=w._session.calibration(a, 0).fingerprint(),
        payload=EventSet(df, prov)))
    w._history.checkpoint((a, 0, "detect.ml"))
    qtbot.wait(5)

    assert w._history.can_redo(_key(a)) is True, (
        "a step on the ML table ended the Schmitt table's redo branch")


# ================================================================================================
# JM-05 — the depth budget is per event table
# ================================================================================================
def test_curation_on_one_recording_cannot_evict_anothers_undo_history(two, qtbot):
    """31 ordinary clicks on recording B used to push the shared 30-entry stack past its cap
    and delete recording A's only entry — the merge that had removed 28 rows."""
    w, a, b = two
    _reject_row(w, a, 0)
    assert w._history.can_undo(_key(a)) is True
    label_before = _menu(w)[0]

    for row in range(MAX_DEPTH + 1):
        _toggle_row(w, b, row)
    qtbot.wait(5)
    assert sum(e.key == _key(b) for e in w._history._undo) >= MAX_DEPTH, (
        "premise: recording B really did fill the budget")

    w._on_selection(a, 0)
    assert w._history.can_undo(_key(a)) is True, (
        "recording B's curation evicted recording A's undo entry from the shared stack")
    assert _menu(w)[0] == label_before
    w._undo_events()
    qtbot.wait(5)
    assert _rejections(w, a) == 0, "the evicted step could not be undone"


def test_the_depth_cap_still_binds_and_binds_on_the_table_that_overflowed(two, qtbot):
    """The bound is not abandoned, only re-scoped: a table over budget loses its OWN oldest
    step (the far end, per ``MAX_BYTES``'s contract), and nobody else's."""
    w, a, b = two
    _reject_row(w, a, 0)
    for i in range(MAX_DEPTH + 12):
        _toggle_row(w, b, i)
    qtbot.wait(5)

    per_key = {}
    for entry in w._history._undo:
        per_key[entry.key] = per_key.get(entry.key, 0) + 1
    assert per_key[_key(b)] == MAX_DEPTH, per_key
    assert per_key[_key(a)] == 1, "the overflowing table spent someone else's budget"


def test_the_byte_ceiling_still_binds(two, qtbot):
    """THE CONTROL THAT MATTERS MOST. Depth is now per table, so the only thing standing
    between the session and 1.1 GB of snapshots is ``MAX_BYTES`` — a correctness fix that
    left memory unbounded would be the worse defect. Driven by shrinking the ceiling rather
    than by building a 128 MiB table."""
    import nanoscope.core.history as history

    w, a, b = two
    _reject_row(w, a, 0)
    _reject_row(w, b, 0)
    real = history.MAX_BYTES
    history.MAX_BYTES = 1                       # every entry is over the ceiling
    try:
        w._history._trim()
    finally:
        history.MAX_BYTES = real
    assert w._history.nbytes() == 0 or len(w._history._undo) == 0
    assert MAX_BYTES == 128 * 1024 * 1024, "the shipped ceiling is unchanged"


def test_an_eviction_is_never_reported_as_a_table_nobody_edited(two, qtbot):
    """The second half of JM-05: the loss was not only unrecoverable, it was described with
    the same disabled wording an untouched table gets. Two surfaces, one fact — the Edit
    menu and the status bar both have to say the history was dropped."""
    import nanoscope.core.history as history

    w, a, b = two
    said: list[str] = []
    w.statusBar().messageChanged.connect(lambda m: said.append(m))
    _reject_row(w, a, 0)
    _reject_row(w, b, 0)

    real = history.MAX_BYTES
    history.MAX_BYTES = 1
    try:
        w._history._trim()
    finally:
        history.MAX_BYTES = real
    qtbot.wait(5)

    w._on_selection(a, 0)
    undo_text, _redo_text = _menu(w)
    assert w._history.evicted(_key(a)) >= 1
    assert "nothing to undo on this event table" not in undo_text, undo_text
    assert "memory ceiling" in undo_text, undo_text
    assert any("memory ceiling" in m for m in said), said


def test_nothing_to_undo_still_reads_plainly_when_nothing_was_evicted(two, qtbot):
    """CONTROL for the wording above — the new sentence must not appear on the ordinary
    empty table, or it becomes the noise the real one is drowned in."""
    w, a, b = two
    _reject_row(w, b, 0)
    w._on_selection(a, 0)
    undo_text, _ = _menu(w)
    assert "nothing to undo on this event table" in undo_text, undo_text
    assert "memory ceiling" not in undo_text, undo_text
