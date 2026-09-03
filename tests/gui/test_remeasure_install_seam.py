"""The re-measure install, driven through the panel that installs it.

The engine half is in ``tests/analysis/test_remeasure_install_seam.py``. This is the
half a user reaches: a real ``MainWindow``, a real detection, the real off-thread
controller, and an edit made while the job runs.

FAILING BEFORE -- an Undo (or a mask drag) during the job was silently overwritten by
the answer, under the status line "Re-measured N events… Detection geometry and
curation unchanged." On the owner's own file that was a 24.900 ms event coming back as
24.100 ms; on a synthetic, 12.000 ms -> 24.000 ms and 300.19 pA -> 150.62 pA. Edit ▸
Redo was disabled afterwards, so nothing on screen pointed at what had happened.

FAILING BEFORE -- a rejection RETRACTED during the job was discarded, because the
carry went through the rebuild's inheritance rule instead of a diff.

FAILING BEFORE -- a re-measure that raised inside the worker emitted no terminal
message at all, leaving "Re-measuring 800 events…" in the status bar for the rest of
the session: a discarded job still reading as running.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. ----------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from nanoscope.analysis.baseline import BASELINE_PARAM_KEYS, BaselineConfig
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, flag_tokens
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
OLD = BaselineConfig(method="block_median", window_s=0.01)


def _signal(n=200_000, dwell=250, period=2000, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, -100.0) + rng.normal(0.0, 2.0, n)
    t = 500
    while t + dwell < n:
        x[t:t + dwell] -= 60.0
        t += period
    return x.astype(np.float32)


def _doc(tmp_path, signal, name="seam.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA",
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _window(qtbot, tmp_path, signal, *, config=OLD):
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)

    df, prov = detect_events(signal, FS, baseline_config=config, polarity=-1,
                             min_dwell_s=1e-4, recording="seam", pore_id="pore")
    block = config.to_provenance()
    params = {"min_dwell_s": 1e-4,
              **{k: block[k] for k in BASELINE_PARAM_KEYS if k in block}}
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params=params,
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov),
        baseline_fingerprint=prov.get("baseline_fingerprint")))
    return win, doc_id


def _pump(qtbot, panel, timeout=60_000):
    qtbot.waitUntil(lambda: panel._pending is None, timeout=timeout)
    QtWidgets.QApplication.processEvents()


# ===========================================================================
# 1. GEOMETRY THAT MOVED UNDER THE JOB
# ===========================================================================
def test_a_span_edited_during_the_job_is_not_overwritten_by_the_answer(
        qapp, qtbot, tmp_path):
    """FAILING BEFORE -- C-4's own sequence, with the mask drag standing in for the
    Ctrl+Z (both reach the store the same way, and the drag needs no history).

    The answer measured the dispatch-time span. The store holds a different one. The
    install must drop the answer and SAY so, because the alternative is writing a
    dwell the user can see is wrong under a sentence that denies having written it.
    """
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))
    assert panel.remeasure_active_detection() == "started"

    live = panel.current_detection().payload.df    # the SHARED, live frame
    row = 2
    was = (int(live.loc[row, "start_idx"]), int(live.loc[row, "end_idx"]))
    live.loc[row, "start_idx"] = was[0] - 60
    live.loc[row, "end_idx"] = was[1] + 60
    live.loc[row, "t_start_s"] = (was[0] - 60) / FS
    live.loc[row, "dwell_s"] = (was[1] + 60 - (was[0] - 60)) / FS
    edited_dwell = float(live.loc[row, "dwell_s"])
    _pump(qtbot, panel)

    after = panel.current_detection().payload.df
    assert float(after.loc[row, "dwell_s"]) == edited_dwell, (
        "the install put the dispatch-time span back over the user's edit")
    assert (int(after.loc[row, "start_idx"]), int(after.loc[row, "end_idx"])) == (
        was[0] - 60, was[1] + 60)
    assert any(t == "superseded" for t, _ in said), said
    assert any("geometry changed" in m for _, m in said), said


def test_the_status_line_never_claims_geometry_unchanged_when_it_dropped_the_answer(
        qapp, qtbot, tmp_path):
    """The sentence is the defect as much as the number is. "Detection geometry and
    curation unchanged" was emitted by the very install that had just changed the
    geometry."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))
    assert panel.remeasure_active_detection() == "started"
    live = panel.current_detection().payload.df
    live.loc[1, "end_idx"] = int(live.loc[1, "end_idx"]) + 40
    live.loc[1, "dwell_s"] = (int(live.loc[1, "end_idx"])
                              - int(live.loc[1, "start_idx"])) / FS
    _pump(qtbot, panel)

    terminal = [(t, m) for t, m in said if t != "started"]
    assert terminal, said
    assert not any("geometry and curation unchanged" in m for _, m in terminal), said
    # ...and the table is still marked stale, which is still true.
    assert panel._stale.isVisibleTo(panel)


def test_an_ordinary_re_measure_still_installs_and_says_so(qapp, qtbot, tmp_path):
    """A CONTROL that passed before and passes now, and the one the widened
    precondition could plausibly have broken: nothing moved, so nothing is refused."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))
    depth_before = float(np.median(panel.current_detection().payload.df["depth_pA"]))
    assert panel.remeasure_active_detection() == "started"
    _pump(qtbot, panel)

    assert any(t == "remeasured" for t, _ in said), said
    depth_after = float(np.median(panel.current_detection().payload.df["depth_pA"]))
    assert depth_after > 1.5 * depth_before


# ===========================================================================
# 2. CURATION MADE WHILE THE JOB RUNS
# ===========================================================================
def test_the_worker_is_handed_a_frame_the_user_cannot_reach(
        qapp, qtbot, tmp_path, monkeypatch):
    """FAILING BEFORE, and it is the half of the seam a GUI test can pin
    deterministically.

    ``_launch_remeasure`` handed the worker ``res.payload`` ITSELF -- the frame the
    results table, the Event Inspector and Undo/Redo write into. Two threads, one
    pandas frame, and whether the job saw an edit or not came down to scheduling: the
    same click produced a correct install, a silently reverted one, or a modal reading
    "re-measure moved 'start_idx'", with nothing to tell them apart. The spy below
    records the object the job actually received, so "detached" is asserted rather
    than inferred from an outcome that a fast synthetic run reaches by luck.

    With the frame detached, the retraction the user makes during the job can only
    arrive through the install's DIFF -- which is what the second half checks.
    """
    import pandas as pd

    from nanoscope.analysis import remeasure as R

    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    df = panel.current_detection().payload.df
    rows = [i for i in df.index
            if bool(df.loc[i, "accepted"]) and not flag_tokens(df.loc[i, "flag"])][:12]
    assert len(rows) >= 5
    df.loc[rows, "accepted"] = False
    df.loc[rows[0], "flag"] = "noise"

    seen: dict = {}
    real = R.remeasure_events

    def spy(eventset, *a, **kw):
        seen["given"] = eventset.df
        seen["at_dispatch"] = eventset.df.copy(deep=True)
        return real(eventset, *a, **kw)

    monkeypatch.setattr("nanoscope.analysis.remeasure.remeasure_events", spy)

    assert panel.remeasure_active_detection() == "started"
    live = panel.current_detection().payload.df
    live.loc[rows, "accepted"] = True                 # "Accept all", mid-job
    live.loc[rows[0], "flag"] = ""                    # ...and the note cleared
    _pump(qtbot, panel)

    assert "given" in seen, "the job never ran"
    assert seen["given"] is not live, (
        "the worker was handed the live table: one pandas frame, two threads")
    pd.testing.assert_frame_equal(seen["given"], seen["at_dispatch"])
    assert not bool(seen["given"].loc[rows[0], "accepted"]), (
        "the user's mid-job edit reached the frame the job was measuring")

    after = panel.current_detection().payload.df
    assert all(bool(after.loc[i, "accepted"]) for i in rows), (
        "the retractions were reverted by the install")
    assert "noise" not in flag_tokens(after.loc[rows[0], "flag"])


def test_a_rejection_made_during_the_job_still_carries(qapp, qtbot, tmp_path):
    """A CONTROL that passed before and passes now: the tightening direction, which
    was the only one under test and the only one that worked."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    assert panel.remeasure_active_detection() == "started"
    live = panel.current_detection().payload.df
    live.loc[3, "accepted"] = False
    live.loc[7, "flag"] = "typed_during_the_job"
    _pump(qtbot, panel)

    after = panel.current_detection().payload.df
    assert not bool(after.loc[3, "accepted"])
    assert "typed_during_the_job" in flag_tokens(after.loc[7, "flag"])


# ===========================================================================
# 3. ONE TERMINAL MESSAGE PER STARTED JOB -- ON THE FAILURE PATH TOO
# ===========================================================================
def test_a_re_measure_that_raises_says_so_instead_of_reading_as_still_running(
        qapp, qtbot, tmp_path):
    """FAILING BEFORE. ``_on_controller_failed`` cleared its state and returned
    without emitting, so the status bar kept the "Re-measuring N events…" it had been
    given at dispatch -- indistinguishable from a job still running, for the rest of
    the session."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))
    assert panel.remeasure_active_detection() == "started"
    assert said and said[-1][0] == "started"

    panel._on_controller_failed("ValueError: boom")

    terminal = [(t, m) for t, m in said if t != "started"]
    assert terminal, "the job failed and the status line was left claiming it was running"
    assert all(m for _, m in terminal)
    assert panel._remeasure_target is None


def test_a_failure_with_no_re_measure_in_flight_stays_quiet(qapp, qtbot, tmp_path):
    """A CONTROL. The guarantee is one message per STARTED job, not a message per
    failure of any kind: a probe or a detection that fails is MainWindow's to report,
    and a second sentence about it on this channel would be the duplicate story
    ``baselineJobStatus`` exists to prevent."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))
    panel._on_controller_failed("ValueError: boom")
    assert said == []
