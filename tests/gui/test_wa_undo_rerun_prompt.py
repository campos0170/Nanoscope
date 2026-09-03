"""WA-UNDO-2 — RE-RUNNING THE SAME DETECTOR DESTROYED CURATION WITH NOTHING TO STOP IT.

Measured on the -1000 mV apo-ferritin recording through a real ``MainWindow`` at 1512x900:
hand-reject 10 events (105 accepted of 115), press Detect again, and 115 came back
accepted. The press was reachable *by accident* — with two detectors on one channel, the
Detection tab's combo could be moved to CUSUM and back to Schmitt while ``_active_detection``
stayed on CUSUM, so the table on screen belonged to a detector the combo did not name and
the only door back to the Schmitt table was Run: the one action that destroys it.

Three things are held here, and they are three different guarantees:

1. **The prompt.** A gesture that would replace hand curation asks first, naming what is at
   stake. Cancel is the default AND the escape button, and the answer is read
   ``clickedButton() is replace`` — by IDENTITY. ``buttons()`` returns LAYOUT order, not
   the order the buttons were added (asserted below, because that ordering is exactly what
   made a destructive button the Enter default once already on this project).
2. **The counts are the ones a person can recognise.** An automatic rejection the next run
   re-derives costs nothing and must not raise a prompt. The panel and
   ``core.history._hand_curation`` are held to the same answer here, because they cannot be
   held to it by sharing an import: no GUI module may import the undo stack's module.
3. **The combo tells the truth.** Selecting a detector shows that detector's stored run,
   and coming back finds the curated table intact — without pressing Run at all.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, stamp_original_bounds
from nanoscope.analysis.scoring import ADDED_COLUMN, ensure_added_column
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.history import _hand_curation
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600),
          (30_000, 30_500), (40_000, 40_400), (50_000, 50_700)]
SCHMITT = "detect.schmitt"
CUSUM = "detect.cusum"


# ================================================================================================
# builders
# ================================================================================================

def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


def _install(window, tmp_path):
    sig = _signal()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa_rerun.dat", format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(sig,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, sig


def _detect_into(window, doc_id, signal, module_id=SCHMITT):
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    events = EventSet(stamp_original_bounds(df), prov)   # as analysis/modules.py does
    window._session.add_result(AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id=doc_id, channel=0,
        params={"detector": module_id},
        calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        payload=events))
    return events


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def curated(window, tmp_path):
    """One recording, one Schmitt detection, two rejections a PERSON made."""
    doc_id, sig = _install(window, tmp_path)
    _detect_into(window, doc_id, sig)
    panel = window._analysis_panel
    df = panel.current_detection().payload.df
    ensure_added_column(df)
    for r in (1, 3):
        panel._table.set_accepted_by_id(int(df["event_id"].iloc[r]), False)
    return window, doc_id, sig, panel


def _press_detect(panel):
    """The gesture the audit measured: a click on the Detection tab's Detect button."""
    panel._detection.detect_button.click()


# ================================================================================================
# 1. the prompt
# ================================================================================================

def test_pressing_detect_over_hand_curation_asks_before_replacing_it(curated, qtbot,
                                                                     blocked_modals):
    """The headline. The suppressed modal answers as a cancelling user, so the table lives.

    ``blocked_modals`` is the harness's record of what the box WOULD have said, which is
    the assertion that matters: the user has to be told what is at stake by name.
    """
    window, _doc_id, _sig, panel = curated
    before = panel.current_detection().payload.df.copy(deep=True)
    _press_detect(panel)
    qtbot.wait(50)

    assert len(blocked_modals) == 1, blocked_modals
    said = blocked_modals[0].text
    assert "2 events you rejected by hand" in said, said
    after = panel.current_detection().payload.df
    assert int((~after["accepted"].to_numpy(bool)).sum()) == 2, (
        "the run went ahead over a dialog nobody answered")
    assert list(after["event_id"]) == list(before["event_id"])


def test_cancel_is_the_default_and_the_escape_button_and_is_matched_by_identity(curated,
                                                                                qtbot):
    """Escape must reach Cancel BY IDENTITY, never by position.

    ``buttons()`` returns LAYOUT order, so the destructive button can be first in that list
    while Cancel was added second — which is how a Replace button became the Enter default
    on this project once before. The assertion below pins the layout order specifically so
    the identity check cannot be "simplified" into an index.
    """
    window, _doc_id, _sig, panel = curated
    seen = {}

    def _spy(self):
        seen["default"] = self.defaultButton()
        seen["escape"] = self.escapeButton()
        seen["layout_order"] = [b.text() for b in self.buttons()]
        seen["roles"] = {b.text(): self.buttonRole(b) for b in self.buttons()}
        return QtWidgets.QMessageBox.Rejected

    old = QtWidgets.QMessageBox.__dict__.get("exec")
    QtWidgets.QMessageBox.exec = _spy
    try:
        _press_detect(panel)
        qtbot.wait(50)
    finally:
        if old is None:
            del QtWidgets.QMessageBox.exec
        else:
            QtWidgets.QMessageBox.exec = old

    assert seen, "no dialog was raised"
    assert seen["default"] is seen["escape"], "Escape and Enter must land on one button"
    assert seen["default"].text() == "Cancel"
    assert seen["layout_order"][0] == "Cancel", (
        "buttons() is LAYOUT order — this is why the answer is read by identity")
    assert seen["roles"]["Replace the table"] == QtWidgets.QMessageBox.DestructiveRole


def test_answering_replace_runs_and_the_curated_table_is_one_undo_away(curated, qtbot):
    """Going ahead is allowed — and recoverable, which is the other half of the promise."""
    window, _doc_id, _sig, panel = curated
    panel.willReplaceEvents.connect(window._checkpoint_everything)   # see needs_wiring
    key = window._undo_key()

    def _accept(self):
        for b in self.buttons():
            if b.text().startswith("Replace"):
                b.click()
                return QtWidgets.QMessageBox.Accepted
        return QtWidgets.QMessageBox.Rejected

    old = QtWidgets.QMessageBox.__dict__.get("exec")
    QtWidgets.QMessageBox.exec = _accept
    try:
        with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
            _press_detect(panel)
    finally:
        if old is None:
            del QtWidgets.QMessageBox.exec
        else:
            QtWidgets.QMessageBox.exec = old
    qtbot.wait(30)

    assert int((~panel.current_detection().payload.df["accepted"].to_numpy(bool)).sum()) == 0
    assert window._history.can_undo(key)
    window._history.undo(key)
    qtbot.wait(10)
    assert int((~panel.current_detection().payload.df["accepted"].to_numpy(bool)).sum()) == 2


def test_a_run_over_an_uncurated_table_asks_nothing(window, tmp_path, qtbot,
                                                    blocked_modals):
    """The prompt must stay rare, or it becomes a click-through."""
    doc_id, sig = _install(window, tmp_path)
    _detect_into(window, doc_id, sig)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        _press_detect(panel)
    qtbot.wait(20)
    assert blocked_modals == [], blocked_modals


def test_a_run_starting_announces_that_it_will_replace_the_table(window, tmp_path, qtbot):
    """``willReplaceEvents`` is the seam MainWindow hangs the pre-replace checkpoint on."""
    doc_id, sig = _install(window, tmp_path)
    _detect_into(window, doc_id, sig)
    panel = window._analysis_panel
    fired = []
    panel.willReplaceEvents.connect(lambda: fired.append(1))
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        _press_detect(panel)
    assert fired, "a launched run said nothing about the table it is about to replace"


# ================================================================================================
# 2. the counts
# ================================================================================================

def test_an_automatic_rejection_is_not_treated_as_a_persons_decision(curated):
    """A ``clipped_by_boundary`` row is re-derived by the next run; it is not at stake."""
    _w, _d, _s, panel = curated
    df = panel.current_detection().payload.df.copy(deep=True)
    df["accepted"] = True
    df[ADDED_COLUMN] = False
    df.iat[0, df.columns.get_loc("accepted")] = False
    df.iat[0, df.columns.get_loc("flag")] = "clipped_by_boundary"
    assert AnalysisPanel.hand_curation_at_stake(df) == (0, 0)


def test_the_prompts_counts_are_the_edit_menus_counts(curated):
    """The drift guard. Two surfaces quoting different numbers is how a warning dies.

    They cannot share an import — ``test_undo_events`` forbids a GUI module from importing
    ``core.history`` — so they are held to the same answer here, over the same frames.
    """
    _w, _d, _s, panel = curated
    base = panel.current_detection().payload.df.copy(deep=True)
    ensure_added_column(base)
    cases = [base]
    for flag in ("", "operator_note", "clipped_by_boundary", "non_finite"):
        df = base.copy(deep=True)
        df.iat[2, df.columns.get_loc("accepted")] = False
        df.iat[2, df.columns.get_loc("flag")] = flag
        df.iat[4, df.columns.get_loc(ADDED_COLUMN)] = True
        cases.append(df)
    for df in cases:
        theirs = _hand_curation(df)
        assert AnalysisPanel.hand_curation_at_stake(df) == (
            theirs["n_hand_rejected"], theirs["n_added"])


def test_an_empty_table_is_nothing_at_stake():
    """The degenerate frame must not raise on the way to a dialog."""
    assert AnalysisPanel.hand_curation_at_stake(pd.DataFrame()) == (0, 0)


# ================================================================================================
# 3. the combo
# ================================================================================================

def test_selecting_a_detector_shows_that_detectors_stored_run(curated, qtbot):
    """Both runs are already in the store, keyed (doc, channel, module). Show the one named."""
    window, doc_id, sig, panel = curated
    _detect_into(window, doc_id, sig, module_id=CUSUM)
    qtbot.wait(20)
    combo = panel._detection.detector_combo

    combo.setCurrentIndex(combo.findData(SCHMITT))
    qtbot.wait(20)
    assert panel._active_detection == SCHMITT
    df = panel.current_detection().payload.df
    assert int((~df["accepted"].to_numpy(bool)).sum()) == 2, (
        "the curated Schmitt table was not restored by naming its detector")


def test_the_combo_and_the_table_on_screen_name_the_same_detector(curated, qtbot):
    """The measured lie: combo reading Schmitt over CUSUM's rows, and no way back but Run."""
    window, doc_id, sig, panel = curated
    _detect_into(window, doc_id, sig, module_id=CUSUM)
    qtbot.wait(20)
    combo = panel._detection.detector_combo
    assert combo.currentData() == panel._active_detection == CUSUM, (
        "a run that installed a table left the combo naming the other detector")

    combo.setCurrentIndex(combo.findData(SCHMITT))
    qtbot.wait(20)
    assert combo.currentData() == panel._active_detection == SCHMITT


def test_a_detector_that_never_ran_here_shows_nothing_rather_than_the_other_ones_rows(
        curated, qtbot):
    """Honest emptiness. Nothing is destroyed: the Schmitt run comes back with the combo."""
    window, _doc_id, _sig, panel = curated
    combo = panel._detection.detector_combo
    combo.setCurrentIndex(combo.findData(CUSUM))
    qtbot.wait(20)
    assert panel._active_detection == CUSUM
    assert panel.current_detection() is None

    combo.setCurrentIndex(combo.findData(SCHMITT))
    qtbot.wait(20)
    df = panel.current_detection().payload.df
    assert int((~df["accepted"].to_numpy(bool)).sum()) == 2
