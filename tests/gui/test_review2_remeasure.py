"""A Re-measure with the estimator unchanged was named a re-run that lost the curation.

Analysis ▸ Compare estimators keeps "Re-measure now" enabled when nothing changed. Pressed on a
curated table, the history saw a new table whose fingerprint had not moved (and, for a table whose
params named no run, params that moved only into the stamp), so it named the step "Re-run
detection", Edit read "Undo Re-run detection", and the status bar said "Detection replaced the
previous table … Your curation is not carried over" while the rejection was kept. The same held
for a table stored with no estimator stamp. The Re-measure now names its own step
(``AnalysisPanel.landing_remeasure``), and a step named a run is one for the fold of design 3.6.

Synthetic recordings only, each under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.modules import register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
_SPANS = [(5_000, 5_400), (15_000, 15_500), (25_000, 25_600), (40_000, 40_500)]
RUN = {"filter_cutoff_hz": 10_000.0, "detrend_enabled": False, "k_in": 6.0}
BLOCKADE = 1
_FALSE_LOSS = "Your curation is not carried over"


def _signal() -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(N, -1_000.0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    register_all()
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    yield w
    w._analysis_controller.abort()


def _detected(w: MainWindow, qtbot, tmp_path: Path) -> None:
    """A synthetic recording detected through the panel's Detect: a v3 table."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, t0_s=0.0,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    ap = w._analysis_panel
    ap.detection_panel.set_params(RUN)
    ap.detection_panel.polarity_override.setCurrentIndex(BLOCKADE)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        ap.trigger_detection()
    QtWidgets.QApplication.processEvents()
    assert len(ap.current_detection().payload.df), "the synthetic events were found"


def _restored_as(w: MainWindow, shape: str) -> None:
    """Put the table back in the shape a project can restore it in (no history step)."""
    cur = w._analysis_panel.current_detection()
    if shape == "baseline_only":     # the estimator's keys alone, as an older Re-measure stored
        cur = replace(cur, params={k: v for k, v in cur.params.items()
                                   if str(k).startswith("baseline_")})
    elif shape == "no_fingerprint":  # a table stored with no estimator stamp
        cur = replace(cur, baseline_fingerprint=None)
    else:
        return
    w._history.set_suspended(True)
    try:
        w._session.add_result(cur)
    finally:
        w._history.set_suspended(False)
    QtWidgets.QApplication.processEvents()


def _reject_first(w: MainWindow) -> None:
    df = w._analysis_panel.current_detection().payload.df
    assert w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    w._on_edit_menu_about_to_show()


def _rejected(w: MainWindow) -> int:
    df = w._analysis_panel.current_detection().payload.df
    return int((~df["accepted"].astype(bool)).sum())


def _undo_text(w: MainWindow) -> str:
    w._on_edit_menu_about_to_show()
    return w._act_undo.text()


def _compare_then_remeasure_now(w: MainWindow, qtbot, monkeypatch) -> None:
    """Analysis ▸ Compare estimators, and the report's "Re-measure now"."""
    from nanoscope.gui import panels

    ap = w._analysis_panel
    monkeypatch.setattr(panels.EstimatorCompareDialog, "exec",
                        lambda self: QtWidgets.QDialog.Accepted)
    before = ap.current_detection()
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._act_compare_baseline.trigger()
    qtbot.waitUntil(lambda: ap._pending is None and ap.current_detection() is not before,
                    timeout=30_000)
    QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("shape", ["v3", "baseline_only", "no_fingerprint"])
def test_a_re_measure_with_the_estimator_unchanged_is_named_a_re_measure(
        window, qtbot, tmp_path, monkeypatch, shape):
    _detected(window, qtbot, tmp_path)
    _restored_as(window, shape)
    fingerprint = window._analysis_panel.current_detection().baseline_fingerprint
    _reject_first(window)
    seen: list[str] = []
    window.statusBar().messageChanged.connect(seen.append)

    _compare_then_remeasure_now(window, qtbot, monkeypatch)
    after = window._analysis_panel.current_detection()
    if shape != "no_fingerprint":
        assert after.baseline_fingerprint == fingerprint, "precondition: the estimator is unchanged"
    assert _undo_text(window) == "Undo Re-measure depths"            # was "Undo Re-run detection"
    assert not [m for m in seen if m and _FALSE_LOSS in m], seen
    assert _rejected(window) == 1, "the curation was kept, as the step's name now says"
    assert window._analysis_panel.landing_remeasure is None


def test_the_fold_takes_the_baseline_steps_of_a_re_measure_that_moved_no_fingerprint(
        window, qtbot, tmp_path):
    """A Baseline edit and its return are two steps that leave the estimator as it was; the
    Re-measure that follows still takes them in one press (design 3.6), as Detect does."""
    _detected(window, qtbot, tmp_path)
    _reject_first(window)
    ap = window._analysis_panel
    box = ap.baseline_panel.rough_s
    start = box.value()
    box.setValue(start * 4.0)
    window._on_edit_menu_about_to_show()
    box.setValue(start)
    window._on_edit_menu_about_to_show()
    fingerprint = ap.current_detection().baseline_fingerprint

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert ap.remeasure_active_detection() == "started"
    QtWidgets.QApplication.processEvents()
    assert ap.current_detection().baseline_fingerprint == fingerprint
    assert window._history.last_step_run, "a step named a run is one"
    assert _undo_text(window) == "Undo Re-measure depths and 2 setting changes"

    window._act_undo.trigger()
    QtWidgets.QApplication.processEvents()
    assert _undo_text(window) == "Undo Curate event", "one press took the run and both steps"
    assert _rejected(window) == 1


def test_detect_pressed_again_is_still_named_a_re_run(window, qtbot, tmp_path):
    """The control: the name is the Re-measure's own, not every new table with an equal
    fingerprint. Detect with the same settings drops the rejection and says so."""
    _detected(window, qtbot, tmp_path)
    _reject_first(window)
    seen: list[str] = []
    window.statusBar().messageChanged.connect(seen.append)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._analysis_panel.trigger_detection()
    QtWidgets.QApplication.processEvents()
    assert _undo_text(window) == "Undo Re-run detection"
    assert [m for m in seen if m and _FALSE_LOSS in m], seen
    assert _rejected(window) == 0
