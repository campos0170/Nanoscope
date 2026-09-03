"""CD-01 — Edit ▸ Undo names the GESTURE, never the state it happened to leave behind.

``docs/USER_GUIDE.md`` §9b: *"the menu names the step it will undo — 'Undo Re-run
detection', 'Undo Merge events', 'Undo Resize event mask', 'Undo Accept all events'"*.

``MainWindow._checkpoint_batch_curation`` was the application's one label HINT, and it
chose the word by reading the whole ``accepted`` column AFTER the fact. Its docstring
justified that with *"the batch signal means exactly one thing, so it may say so"* — true
of the Event Inspector, whose ``curationBatchChanged`` has a single emit site and always
covers the whole frame, and false of the Event Gallery, which routes ANY gesture that wrote
more than one row into the same signal (``event_gallery.py``: ``if len(changed) == 1:
acceptanceChanged … else: curationBatchChanged.emit()``). Both directions were wrong, and
both were measured through the real widgets:

* **A subset gesture was promoted to a batch.** Rejecting the last 2 accepted tiles of 11
  with the gallery's "Reject selected" left the column uniformly False, so Edit read
  *"Undo Reject all events"* — the rows were safe (the undo restores exactly those 2), but
  the menu claimed a scope the user never asked for. The gallery's own button read
  "Reject all 3" and its status line "3 of 10 accepted" for the filter-scoped case, so two
  surfaces disagreed about one gesture.
* **The real batch was demoted.** A whole-frame Accept-all can never reach ``accepted.all()``
  on a frame holding any unmeasurable row — those stay rejected by design, and the owner's
  −600 mV run has three — so the documented *"Undo Accept all events"* was unreachable on
  every real recording, both from the inspector and from the gallery.

The test that should have caught it, ``tests/gui/test_undo_events.py::
test_accept_all_is_one_undoable_step``, drives only the inspector's whole-frame
``accept_all()`` on a frame with nothing unmeasurable. Everything below is driven through
the real gallery buttons, keys and filters, and through the real inspector.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(4_000, 4_400), (10_000, 10_500), (16_000, 16_600), (22_000, 22_500),
          (28_000, 28_400), (34_000, 34_700), (40_000, 40_500), (46_000, 46_400),
          (52_000, 52_600)]


def _signal() -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "cd01.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    w._open_event_gallery()
    qtbot.wait(20)
    gal = w._event_gallery
    gal.page_size_box.setCurrentText("200")
    qtbot.wait(40)
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _label(w) -> str:
    w._checkpoint_everything()
    w._sync_undo_actions()
    return w._act_undo.text()


def _flag_unmeasurable(df, rows) -> None:
    """Stamp the detector's own "I could not measure this" token — the state every real
    recording is in, and the one the old state-reading hint could not survive."""
    col = df.columns.get_loc("flag")
    acc = df.columns.get_loc("accepted")
    for r in rows:
        df.iat[r, col] = "dwell>max_dwell_s"
        df.iat[r, acc] = False


def _select_rows(gal, rows) -> None:
    sm = gal.view.selectionModel()
    sm.clearSelection()
    for r in rows:
        idx = gal._model.index_for_df_row(r)
        assert idx.isValid(), f"df row {r} is not on the page"
        sm.select(idx, QItemSelectionModel.Select)


# ================================================================================================
# (a) a subset gesture is offered as the subset it was
# ================================================================================================
def test_rejecting_two_selected_tiles_is_not_offered_as_reject_all(wired, qtbot):
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    # Reject everything by hand except the last two, one row at a time, so the multi-tile
    # gesture below is the only batch on the stack.
    for r in range(len(df) - 2):
        w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[r]), False)
        qtbot.wait(2)
    accepted = [i for i in range(len(df)) if bool(df["accepted"].iloc[i])]
    assert len(accepted) == 2, accepted

    _select_rows(gal, accepted)
    qtbot.wait(20)
    gal.reject_selected_btn.click()
    qtbot.wait(60)

    label = _label(w)
    assert "Reject all events" not in label, (
        f"a 2-of-{len(df)} hand gesture is offered as a whole-frame reject: {label!r}")
    assert "Curate 2 events" in label, label
    w._undo_events()
    qtbot.wait(20)
    after = _frame(w, doc_id)                   # the undo INSTALLS a frame; re-read it
    assert [i for i in range(len(after)) if bool(after["accepted"].iloc[i])] == accepted, (
        "and the undo restores exactly those two rows — which is why the label mattered: "
        "a whole-frame reject would have brought every measurable row back")


def test_accepting_two_selected_tiles_is_not_offered_as_accept_all(wired, qtbot):
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.reject_all_btn.click()
    qtbot.wait(60)
    assert not df["accepted"].to_numpy(dtype=bool).any()

    _select_rows(gal, [0, 1])
    qtbot.wait(20)
    gal.accept_selected_btn.click()
    qtbot.wait(60)

    label = _label(w)
    assert "Accept all events" not in label, label
    assert "Curate 2 events" in label, label


def test_a_filter_scoped_all_shown_is_not_a_whole_frame_batch(wired, qtbot):
    """"Accept/Reject all shown" is filter-scoped by its own tooltip — *"not the whole
    detection unless the filter is All"* — so a press under *Accepted only* is a gesture
    over 3 of 9 rows and the button says so ("Reject all 3")."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    for r in range(len(df) - 3):
        w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[r]), False)
        qtbot.wait(2)
    gal.set_filter("Accepted only")
    qtbot.wait(40)
    assert gal.event_count() == 3, gal.event_count()

    gal.reject_all_btn.click()
    qtbot.wait(60)
    label = _label(w)
    assert "Reject all events" not in label, (
        f"a 3-of-{len(df)} filter-scoped gesture was recorded as a whole-frame reject: "
        f"{label!r}")
    assert "Curate 3 events" in label, label


def test_the_keyboard_reject_over_a_selection_is_a_subset_gesture_too(wired, qtbot):
    """R over two tiles reaches the same signal as the buttons and must reach the same
    honest label — the verifier's second leg."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    for r in range(len(df) - 2):
        w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[r]), False)
        qtbot.wait(2)
    accepted = [i for i in range(len(df)) if bool(df["accepted"].iloc[i])]
    _select_rows(gal, accepted)
    qtbot.wait(20)
    gal.set_accepted_selected(False)            # the A / R / Space path
    qtbot.wait(60)

    label = _label(w)
    assert "Reject all events" not in label, label
    assert "Curate 2 events" in label, label


# ================================================================================================
# (b) ...and the real whole-frame batch is offered as the batch it was
# ================================================================================================
def test_a_whole_frame_gallery_accept_all_is_named_accept_all(wired, qtbot):
    """The documented label, on a frame that HAS unmeasurable rows — i.e. every real
    recording. ``accepted.all()`` is False here and always will be."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    _flag_unmeasurable(df, [0, 4])
    w._analysis_panel.resync_curation()
    qtbot.wait(40)
    gal.reject_all_btn.click()
    qtbot.wait(60)
    assert "Reject all events" in _label(w), _label(w)

    gal.accept_all_btn.click()
    qtbot.wait(60)
    label = _label(w)
    assert "Accept all events" in label, (
        f"a genuine whole-frame Accept all is offered as {label!r}; the documented "
        f"'Undo Accept all events' is unreachable on any real recording")
    acc = df["accepted"].to_numpy(dtype=bool)
    assert acc.sum() == len(df) - 2 and not acc.all(), (
        "premise: the unmeasurable rows are still rejected, which is why the state-reading "
        "hint could never fire")


def test_the_inspectors_accept_all_is_named_accept_all(wired, qtbot):
    """The other whole-frame surface, same frame, same word. The inspector's signal really
    does mean one thing — it has a single emit site — so this is where the hint survives."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    _flag_unmeasurable(df, [0, 4])
    w._open_event_inspector()
    qtbot.wait(20)
    insp = w._event_inspector
    insp.reject_all()
    qtbot.wait(60)
    assert "Reject all events" in _label(w), _label(w)
    insp.accept_all()
    qtbot.wait(60)
    assert "Accept all events" in _label(w), _label(w)


def test_a_whole_frame_reject_all_under_the_all_filter_is_still_reject_all(wired, qtbot):
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    assert gal.event_count() == len(df), "premise: the All filter shows the whole detection"
    gal.reject_all_btn.click()
    qtbot.wait(60)
    assert "Reject all events" in _label(w), _label(w)


def test_the_gesture_marker_does_not_outlive_its_own_click(wired, qtbot):
    """The marker that says "this was an all-shown press" is armed on ``pressed`` and dies
    on ``clicked`` — otherwise an "Accept all shown" that moved no row (it emits no batch
    signal at all) would leave it standing and rename the NEXT subset gesture."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.accept_all_btn.click()                  # everything is already accepted: no rows move
    qtbot.wait(60)
    assert w._gallery_whole_frame is None

    _select_rows(gal, [0, 1])
    qtbot.wait(20)
    gal.set_accepted_selected(False)
    qtbot.wait(60)
    label = _label(w)
    assert "Reject all events" not in label, label
    assert "Curate 2 events" in label, label
