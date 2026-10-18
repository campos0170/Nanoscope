"""§4.2 in the app: record, mark stale, offer — never silently, never refuse.

Two of these reproduce failures that existed before this change, and each says so:

* **FAILING BEFORE — the stamp did not survive a project round-trip.**
  ``AnalysisResult.baseline_fingerprint`` was written by the detectors and stored in
  the EventSet's provenance, but ``MainWindow._restore_results`` never read it back.
  Every reopened project therefore came back with ``baseline_fingerprint is None`` —
  indistinguishable from a genuinely pre-change table, so a project measured by the
  configured estimator this morning read as stale this afternoon and the banner could
  never clear. Same defect class as C6 (the calibration stamp re-derived on load),
  one level worse: not re-derived, dropped.
* **FAILING BEFORE — the staleness check never looked at the baseline.**
  ``_check_staleness`` compared the calibration fingerprint and nothing else, so a
  table measured by the 10 ms block median sat next to a Preprocess tab configured
  for the masked estimator with no banner, no badge, and depths understated by more
  than 10x on the owner's own recording.

Everything else here pins behaviour that had no implementation at all: the badge, the
two different repairs behind one button, the off-thread re-measure, the curation
inheritance through the GUI, and the report that installs nothing.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.baseline import (
    BASELINE_PARAM_KEYS,
    DEFAULT_BASELINE,
    BaselineConfig,
)
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, has_flag
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS, compare_estimators
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import _STALE_BASELINE, _STALE_CALIBRATION
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import EstimatorCompareDialog

FS = 50_000.0
N = 400_000
I0 = -100.0
DEPTH = 60.0

#: The estimator every pre-change row was measured by.
OLD = BaselineConfig(method="block_median", window_s=0.01)


# ================================================================================================
# builders
# ================================================================================================
def _signal(*, dwell=250, period=2000, n=N, seed=3) -> np.ndarray:
    """The `tests/analysis/test_remeasure.py` fixture, same reasoning: events long
    enough against half a 10 ms block that the block median lands inside them, but
    not so long that the old estimator loses the events outright."""
    rng = np.random.default_rng(seed)
    x = np.full(n, I0) + rng.normal(0.0, 2.0, n)
    t = 500
    while t + dwell < n:
        x[t:t + dwell] -= DEPTH
        t += period
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name="mig.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _window(qtbot, tmp_path, signal, *, config=OLD, select=True):
    """A window with a recording open, selected, and ONE detection measured by
    ``config`` — the state a user reaches by opening a project made before this
    change. ``select=False`` skips the selection, leaving the panel pointed at
    nothing — the true reopened-project ordering, where the results are installed
    BEFORE the panel is pointed at their target. (Installing a detection on the
    CURRENT selection seeds the form immediately: ``_on_results_changed`` enforces
    "the settings on the tab are the settings that produced the table on screen"
    on that path too — the CF-4 beta fix.)"""
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    if select:
        win._on_selection(doc_id, 0)

    df, prov = detect_events(signal, FS, baseline_config=config, polarity=-1,
                             min_dwell_s=1e-4, recording="mig", pore_id="pore")
    # The params a real run stores: the detection form's, WITH the baseline block
    # merged in exactly as `AnalysisPanel._form_params` merges it.
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
    """Wait for the off-thread baseline job to land."""
    qtbot.waitUntil(lambda: panel._pending is None, timeout=timeout)
    QtWidgets.QApplication.processEvents()


def _strip_stamp(win, doc_id):
    """Make the shown detection look like a genuinely pre-change table: no stamp on
    the result and none in its provenance."""
    res = win._analysis_panel.current_detection()
    prov = {k: v for k, v in res.payload.provenance.items()
            if not str(k).startswith("baseline_")}
    from dataclasses import replace
    win._session.add_result(replace(
        res, payload=EventSet(res.payload.df, prov), baseline_fingerprint=None))
    QtWidgets.QApplication.processEvents()


# ================================================================================================
# 1. THE STAMP SURVIVES A PROJECT ROUND-TRIP  (FAILING BEFORE)
# ================================================================================================
def test_the_baseline_stamp_survives_save_and_reopen(qapp, qtbot, tmp_path):
    """FAILING BEFORE: ``_restore_results`` built its ``AnalysisResult`` without
    ``baseline_fingerprint``, so every reopened project read as "no estimator named"
    — the same signal a genuinely pre-change table gives — and a freshly measured
    project came back permanently stale."""
    signal = _signal()
    win, doc_id = _window(qtbot, tmp_path, signal, config=DEFAULT_BASELINE)
    stamped = win._analysis_panel.current_detection().baseline_fingerprint
    assert stamped == DEFAULT_BASELINE.fingerprint(FS)

    # A REAL round-trip through the project file: the stamp rides inside the
    # EventSet's provenance, which `ResultRec` persists verbatim, so no project
    # VERSION bump is involved and this proves the whole path rather than the
    # in-memory dataclass.
    from nanoscope.core.project import load_project, save_project
    path = tmp_path / "p.nanoscope"
    save_project(path, win._gather_project_state())
    state = load_project(path)
    assert state.results[0].provenance["baseline_fingerprint"] == stamped

    win2 = MainWindow()
    qtbot.addWidget(win2)
    live_id = win2._session.add_document(_doc(tmp_path, signal))
    win2._restore_state = state
    win2._apply_restore_for_doc(live_id)

    res = win2._session.results.get(live_id, 0, "detect.schmitt")
    assert res is not None
    assert res.baseline_fingerprint == stamped, (
        "the stamp was dropped on restore: every reopened project would read as "
        "pre-change and the staleness banner could never clear")


def test_a_v1_project_with_no_stamp_still_loads_and_is_marked_not_refused(
        qapp, qtbot, tmp_path):
    """§4.2.5 + ``MIN_READ_VERSION``. The oldest project this build reads carries no
    ``baseline_*`` key at all; it must come back readable, with the absence itself
    read as the signal."""
    from nanoscope.core.project import MIN_READ_VERSION, load_project, save_project

    signal = _signal()
    win, _ = _window(qtbot, tmp_path, signal)
    _strip_stamp(win, win._session.documents()[0].doc_id)
    path = tmp_path / "old.nanoscope"
    save_project(path, win._gather_project_state())
    state = load_project(path)
    assert MIN_READ_VERSION == 1
    assert "baseline_fingerprint" not in state.results[0].provenance

    win2 = MainWindow()
    qtbot.addWidget(win2)
    live_id = win2._session.add_document(_doc(tmp_path, signal))
    win2._create_default_layers(win2._session.document(live_id))
    win2._restore_state = state
    win2._apply_restore_for_doc(live_id)
    win2._on_selection(live_id, 0)

    res = win2._session.results.get(live_id, 0, "detect.schmitt")
    assert res is not None and res.baseline_fingerprint is None
    assert len(res.payload.df)                       # readable
    panel = win2._analysis_panel
    assert panel._stale.kind == _STALE_BASELINE      # marked
    assert panel._table.stale_columns == REMEASURED_COLUMNS


def test_a_malformed_stored_stamp_reads_as_absent_not_as_a_stamp(qapp, tmp_path):
    """A fingerprint that cannot be compared must read as "unknown", never as
    "matches" — the rule ``_fingerprint_to_json`` already applies to the calibration
    pair, for the same reason."""
    from nanoscope.gui.main_window import _stored_baseline_fingerprint

    assert _stored_baseline_fingerprint({"baseline_fingerprint": "sha256:abc"}) == "sha256:abc"
    assert _stored_baseline_fingerprint({}) is None
    assert _stored_baseline_fingerprint(None) is None
    assert _stored_baseline_fingerprint({"baseline_fingerprint": 7}) is None
    assert _stored_baseline_fingerprint({"baseline_fingerprint": ""}) is None


# ================================================================================================
# 2. MARKING  (FAILING BEFORE)
# ================================================================================================
def test_a_pre_change_table_is_marked_stale_and_its_columns_badged(qapp, qtbot, tmp_path):
    """FAILING BEFORE: ``_check_staleness`` compared the calibration stamp and
    nothing else, so a table measured by the 10 ms block median showed no banner and
    no badge beside a Preprocess tab configured for the masked estimator."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel

    # isHidden(), not isVisibleTo(panel): the stale line lives in the Results dock now, so the
    # panel is not its ancestor and isVisibleTo would depend on the window being shown.
    assert not panel._stale.isHidden()
    assert panel._stale.kind == _STALE_BASELINE
    assert panel._stale.action_text() == "Re-measure depths"
    msg = panel._stale.message()
    assert "block_median" in msg and "masked median" in msg
    # `baseline_pA` is badged too, not just §4.2.5's six: it moves as well, and it
    # is what the other six are computed from.
    assert panel._table.stale_columns == REMEASURED_COLUMNS
    assert "baseline_pA" in panel._table.stale_columns


def test_a_table_with_no_stamp_at_all_says_it_predates_the_estimator(qapp, qtbot, tmp_path):
    """Absence IS the signal, and the message says which estimator the absence
    implies rather than "unknown"."""
    win, doc_id = _window(qtbot, tmp_path, _signal())
    _strip_stamp(win, doc_id)
    panel = win._analysis_panel
    assert panel.current_detection().baseline_fingerprint is None
    assert panel._stale.kind == _STALE_BASELINE
    assert "predates" in panel._stale.message()
    assert "Compare baseline estimators" in panel._stale.message()


def test_a_matching_stamp_is_not_marked(qapp, qtbot, tmp_path):
    """The control. Without it the two tests above pass on a banner that is always up."""
    win, _ = _window(qtbot, tmp_path, _signal(), config=DEFAULT_BASELINE)
    panel = win._analysis_panel
    assert panel._stale.isHidden()
    assert panel._table.stale_columns == ()


def test_a_stale_calibration_outranks_a_stale_baseline(qapp, qtbot, tmp_path):
    """A gain change moved the thresholds the detector ran with, so the EVENTS are in
    question and only a re-detection answers it. Re-measuring on top of a stale
    calibration would produce confidently-wrong depths against the right open-pore
    reference."""
    win, doc_id = _window(qtbot, tmp_path, _signal())
    from nanoscope.core.calibration import ChannelCalibration
    # write_sidecar=False: a test must never write into the owner's data directory.
    win._session.set_calibration(
        doc_id, 0, ChannelCalibration(gain=2.0, offset=0.0, target_units="pA",
                                      provenance=Provenance.MEASURED),
        write_sidecar=False)
    QtWidgets.QApplication.processEvents()
    panel = win._analysis_panel
    assert panel._stale.kind == _STALE_CALIBRATION
    assert panel._stale.action_text() == "Re-run"
    # ...and the amplitude columns are badged either way: the badge is about the
    # NUMBERS, not about which stamp moved.
    assert panel._table.stale_columns == REMEASURED_COLUMNS


def test_editing_the_estimator_re_judges_the_shown_detection_at_once(qapp, qtbot, tmp_path):
    """A stamp only compared when a result is first displayed leaves the user looking
    at depths measured by an estimator he has just replaced, with nothing saying so
    until the next selection change."""
    win, _ = _window(qtbot, tmp_path, _signal(), config=DEFAULT_BASELINE)
    panel = win._analysis_panel
    assert panel._stale.isHidden()
    panel.baseline_panel.rough_s.setValue(2.0)
    QtWidgets.QApplication.processEvents()
    assert not panel._stale.isHidden()
    assert panel._stale.kind == _STALE_BASELINE


def test_the_badge_is_readable_without_colour(qapp, qtbot, tmp_path):
    """Colour is the first thing lost to a colour-blind reader, a screenshot or a
    printout, and this badge is the only thing distinguishing a number that is right
    from one that is 10x wrong."""
    win, _ = _window(qtbot, tmp_path, _signal())
    model = win._analysis_panel._table._model
    cols = list(model._columns)
    depth = cols.index("depth_pA")
    dwell = cols.index("dwell_s")
    assert "⚠" in model.headerData(depth, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
    assert "⚠" not in model.headerData(dwell, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
    tip = model.data(model.index(0, depth), QtCore.Qt.ToolTipRole)
    assert tip and "re-measured" in tip
    assert model.data(model.index(0, dwell), QtCore.Qt.ToolTipRole) is None
    # ...and the value itself is shown EXACTLY as stored. "Never silently re-measure"
    # includes never quietly correcting what is on screen.
    df = win._analysis_panel.current_detection().payload.df
    assert model.data(model.index(0, depth), QtCore.Qt.DisplayRole) == \
        f"{df.iloc[0]['depth_pA']:.4g}"


# ================================================================================================
# 3. THE OFFER — one click, off-thread, curation intact
# ================================================================================================
def test_the_banner_button_re_measures_and_never_re_detects(qapp, qtbot, tmp_path):
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    before = panel.current_detection().payload.df.copy(deep=True)

    # curate first, so the thing that must survive exists before the click
    live = panel.current_detection().payload.df
    live.loc[2, "accepted"] = False
    live.loc[4, "flag"] = "curator_note"

    panel._stale._btn.click()
    _pump(qtbot, panel)

    after = panel.current_detection().payload.df
    assert len(after) == len(before)
    for col in ("start_idx", "end_idx", "dwell_s", "t_start_s", "event_id"):
        assert (after[col].to_numpy() == before[col].to_numpy()).all(), col
    assert float(np.median(after["depth_pA"])) > 1.5 * float(np.median(before["depth_pA"]))
    assert not bool(after.loc[2, "accepted"])
    assert has_flag(after.loc[4, "flag"], "curator_note")


def test_a_re_measure_clears_the_banner_and_the_badge(qapp, qtbot, tmp_path):
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    assert not panel._stale.isHidden()
    panel.remeasure_active_detection()
    _pump(qtbot, panel)
    assert panel.current_detection().baseline_fingerprint == DEFAULT_BASELINE.fingerprint(FS)
    assert panel._stale.isHidden()
    assert panel._table.stale_columns == ()


def test_nothing_re_measures_without_a_click(qapp, qtbot, tmp_path):
    """§4.2.4. A number that changes 15x between two openings of the same project,
    with no prompt, is indistinguishable from data corruption — so opening,
    selecting, editing the estimator and displaying the table must all leave the
    stored values ALONE."""
    win, doc_id = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    stored = panel.current_detection().payload.df["depth_pA"].to_numpy().copy()

    win._on_selection(doc_id, 0)
    panel.baseline_panel.rough_s.setValue(2.0)
    panel.baseline_panel.rough_s.setValue(0.5)
    win._sync_export_action()
    QtWidgets.QApplication.processEvents()

    assert panel._pending is None, "displaying a stale table must launch no job"
    assert np.array_equal(
        panel.current_detection().payload.df["depth_pA"].to_numpy(), stored)


def test_it_runs_off_the_gui_thread(qapp, qtbot, tmp_path):
    """1641 events on the owner's 3 M-sample file cost ~3.3 s. A frozen window during
    an operation that looks destructive is how a user concludes the app has hung."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    assert panel.remeasure_active_detection() == "started"
    assert panel._pending == "remeasure"           # still in flight, GUI responsive
    _pump(qtbot, panel)


def test_an_answer_landing_after_the_user_moved_on_still_reaches_its_own_table(
        qapp, qtbot, tmp_path):
    """The job belongs to the RESULT it was launched for, not to whatever is selected
    when it lands. Gating it on the selection would drop the answer silently — the
    table stays marked stale, the click looks ignored, and that is indistinguishable
    from a failure."""
    signal = _signal()
    win, doc_id = _window(qtbot, tmp_path, signal)
    panel = win._analysis_panel
    other = win._session.add_document(_doc(tmp_path, signal, name="other.dat"))
    win._create_default_layers(win._session.document(other))

    panel.remeasure_active_detection()
    win._on_selection(other, 0)                    # move on while it runs
    _pump(qtbot, panel)

    res = win._session.results.get(doc_id, 0, "detect.schmitt")
    assert res.baseline_fingerprint == DEFAULT_BASELINE.fingerprint(FS)


def test_a_click_made_while_the_re_measure_runs_is_not_discarded(qapp, qtbot, tmp_path):
    """The table stays live for the seconds the job takes (~3.3 s on the owner's own
    file). Silently overwriting the clicks made in that window is the same failure as
    losing them on reload, on a shorter fuse."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    assert panel.remeasure_active_detection() == "started"

    live = panel.current_detection().payload.df       # curate WHILE it runs
    live.loc[3, "accepted"] = False
    live.loc[7, "flag"] = "typed_during_the_job"
    _pump(qtbot, panel)

    after = panel.current_detection().payload.df
    assert not bool(after.loc[3, "accepted"])
    assert has_flag(after.loc[7, "flag"], "typed_during_the_job")
    assert panel.current_detection().baseline_fingerprint == DEFAULT_BASELINE.fingerprint(FS)


def test_a_table_merged_while_the_re_measure_runs_drops_the_answer_and_says_so(
        qapp, qtbot, tmp_path):
    """Those rows are not the rows that were measured. Installing over them would
    undo the merge — so the answer is discarded, out loud, and the table stays
    marked stale (which is still true).

    The rows are collapsed here with the Event Inspector's "Merge with next", which is
    still synchronous. "Merge all" no longer is — it costs 3.3-8.7 s on the owner's real
    recordings and now runs on the shared controller, which means it is REFUSED while a
    re-measure holds that controller ("Another analysis is still running"), exactly as
    every other controller job already was. The invariant under test is the re-measure's,
    not the merge's: any edit that collapses the measured rows must make the answer be
    dropped out loud.
    """
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))
    assert panel.remeasure_active_detection() == "started"

    # ...and "Merge all" says why it will not start, rather than starting a second
    # controller job on top of the re-measure.
    assert panel.merge_all_subevents(0.05).status == "busy"

    ids = [int(i) for i in panel.current_detection().payload.df["event_id"].to_numpy()]
    report = panel.merge_event_with_next(ids[0], 0.05)   # collapses rows under the job
    assert report.status == "ok"
    n_after_merge = len(panel.current_detection().payload.df)
    assert n_after_merge < len(ids)
    _pump(qtbot, panel)

    assert any(t == "superseded" for t, _ in said), said
    assert len(panel.current_detection().payload.df) == n_after_merge
    assert not panel._stale.isHidden()            # still stale, and still true


def test_the_params_a_re_run_would_reproduce_name_the_new_estimator(qapp, qtbot, tmp_path):
    """``params`` is what the calibration "Re-run" reads. Left naming the replaced
    estimator, a later gain edit would silently measure the depths back down again,
    and ``params`` and ``provenance`` would disagree about one result."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    assert panel.current_detection().params["baseline_method"] == "block_median"
    panel.remeasure_active_detection()
    _pump(qtbot, panel)

    res = panel.current_detection()
    assert res.params["baseline_method"] == "masked_median"
    assert res.params["min_dwell_s"] == 1e-4          # the detection settings survive
    # ...and the DERIVED fingerprint is still kept out of params, where a value that
    # can contradict the parameters beside it does not belong.
    assert "baseline_fingerprint" not in res.params
    from nanoscope.analysis.baseline import baseline_config_from_params
    cfg, _src = baseline_config_from_params(res.params)
    assert cfg.fingerprint(FS) == res.baseline_fingerprint


def test_every_refusal_is_said_out_loud(qapp, qtbot, tmp_path):
    """A click that does nothing and explains nothing is indistinguishable from a
    hang — the rule this project already applies to every gated button."""
    win = MainWindow()
    qtbot.addWidget(win)
    said = []
    win._analysis_panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))

    assert win._analysis_panel.remeasure_active_detection() == "no_detection"
    assert win._analysis_panel.compare_baseline_estimators() == "no_detection"
    assert len(said) == 2
    assert all(m for _, m in said)
    assert "Run detection" in said[0][1]


def test_a_second_job_is_refused_while_one_is_in_flight(qapp, qtbot, tmp_path):
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    assert panel.remeasure_active_detection() == "started"
    assert panel.remeasure_active_detection() == "busy"
    _pump(qtbot, panel)


# ================================================================================================
# 4. THE REPORT
# ================================================================================================
def test_the_menu_action_is_gated_on_there_being_a_detection(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win._act_compare_baseline.isEnabled()
    win.close()

    win2, _ = _window(qtbot, tmp_path, _signal())
    win2._sync_export_action()
    assert win2._act_compare_baseline.isEnabled()


def test_the_report_installs_nothing(qapp, qtbot, tmp_path, monkeypatch):
    """Opening the evidence must not be an action. Nothing is written until
    "Re-measure now" or the banner's button is pressed."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    stored = panel.current_detection().payload.df["depth_pA"].to_numpy().copy()
    monkeypatch.setattr(EstimatorCompareDialog, "exec",
                        lambda self: QtWidgets.QDialog.Rejected)

    panel.compare_baseline_estimators()
    _pump(qtbot, panel)

    assert np.array_equal(
        panel.current_detection().payload.df["depth_pA"].to_numpy(), stored)
    assert panel.current_detection().baseline_fingerprint != DEFAULT_BASELINE.fingerprint(FS)
    assert panel.last_comparison is not None
    assert panel.last_comparison.changed


def test_re_measure_now_inside_the_report_goes_through_the_same_entry_point(
        qapp, qtbot, tmp_path, monkeypatch):
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    monkeypatch.setattr(EstimatorCompareDialog, "exec",
                        lambda self: QtWidgets.QDialog.Accepted)
    panel.compare_baseline_estimators()
    _pump(qtbot, panel)          # the compare job
    _pump(qtbot, panel)          # the re-measure it launched
    assert panel.current_detection().baseline_fingerprint == DEFAULT_BASELINE.fingerprint(FS)


def test_the_dialog_states_both_estimators_and_the_factor(qapp, qtbot, tmp_path):
    signal = _signal()
    win, _ = _window(qtbot, tmp_path, signal)
    ev = win._analysis_panel.current_detection().payload
    cmp_ = compare_estimators(ev, signal, FS, config=DEFAULT_BASELINE, polarity=-1)

    dlg = EstimatorCompareDialog(cmp_)
    qtbot.addWidget(dlg)
    assert "block_median" in dlg.old_label.text()
    assert "masked median" in dlg.new_label.text()
    assert f"{cmp_.median_ratio:.3g}x" in dlg.summary_text()
    assert dlg.remeasure_button.isEnabled()
    assert dlg.remeasure_button.toolTip()


def test_the_dialog_is_honest_about_a_clean_recording(qapp, qtbot, tmp_path):
    signal = _signal(dwell=150, period=1000)
    win, _ = _window(qtbot, tmp_path, signal)
    ev = win._analysis_panel.current_detection().payload
    cmp_ = compare_estimators(ev, signal, FS, config=DEFAULT_BASELINE, polarity=-1)
    dlg = EstimatorCompareDialog(cmp_)
    qtbot.addWidget(dlg)
    assert "re-measuring is safe" in dlg.summary_text()
    # ...and the button is still live: re-measuring a clean file is a harmless no-op
    # that stamps the new fingerprint and clears the banner.
    assert dlg.remeasure_button.isEnabled()


# ================================================================================================
# 5. NEVER REFUSE TO OPEN
# ================================================================================================
def test_a_stale_table_stays_readable_plottable_and_exportable(qapp, qtbot, tmp_path):
    """§4.2.5. Marked stale, badged — and otherwise a fully working analysis."""
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    assert not panel._stale.isHidden()

    df = panel.current_detection().payload.df
    assert len(df) and panel._table._model.rowCount() == len(df)
    win._sync_export_action()
    assert win._act_export_events.isEnabled()
    assert win._act_event_inspector.isEnabled()

    out = tmp_path / "ev.csv"
    from nanoscope.analysis.scoring import write_events_export
    events = panel.current_detection().payload
    written = write_events_export(
        events.df, out, provenance=events.provenance,
        source_file="mig.dat", detector="detect.schmitt")
    assert written["data"].exists() and written["meta"].exists()
    import json
    meta = json.loads(written["meta"].read_text())
    assert meta["provenance"]["baseline_method"] == "block_median"


# ================================================================================================
# 9. A CONFIRMED CONFIGURATION SURVIVES NAVIGATION  (FAILING BEFORE)
# ================================================================================================
def test_reselecting_the_same_channel_does_not_discard_a_confirmed_baseline(
        qapp, qtbot, tmp_path):
    """FAILING BEFORE: one click on the channel that was already selected threw away
    the user's confirmed per-recording windows AND the warning that would have said
    so.

    §0.3 of the build is "the app PROPOSES, the user CONFIRMS", and a single global
    default provably cannot exist -- the apo file needs ``rough_s = 2.0`` for its
    521 ms clog while ``ferritin_data3`` needs 0.03. So the confirmed value is the
    whole point of the surface, and it was reachable only through the widgets.

    ``_seed_form_from`` (reached from ``_load_stored_detection``, reached from every
    ``set_selection``) called ``BaselinePanel.set_params`` with the STORED
    detection's parameters, which is correct when the form has nothing better to say
    and wrong when the user has just said something. Because ``set_params`` does not
    emit ``paramsChanged``, ``_check_staleness`` never re-ran on the edit; the
    ``_show_result`` that followed then compared the stored fingerprint against the
    now-reverted form, found them equal, and hid the banner. The two wrongs cancelled
    and the user was shown "no problem" over an estimator he had replaced. Detect and
    Run-ML would both have used the reverted value.
    """
    win, doc_id = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    bp = panel._baseline

    # The user confirms this recording's own windows, through the widgets.
    bp.rough_s.setValue(2.0)
    bp.base_s.setValue(0.5)
    QtWidgets.QApplication.processEvents()
    assert bp.param_source() == "user"
    confirmed = bp.config().fingerprint(FS)
    assert not panel._stale.isHidden(), "the edit must mark the shown table stale"

    # ...and then re-selects the channel that is already selected.
    win._on_selection(doc_id, 0)
    QtWidgets.QApplication.processEvents()

    assert bp.param_source() == "user"
    assert float(bp.rough_s.value()) == 2.0
    assert float(bp.base_s.value()) == 0.5
    assert bp.config().fingerprint(FS) == confirmed
    params = panel._form_params()
    assert params["baseline_rough_s"] == 2.0 and params["baseline_base_s"] == 0.5
    assert not panel._stale.isHidden(), (
        "the banner cleared without the estimator changing back"
    )
    assert panel._table.stale_columns == REMEASURED_COLUMNS


def test_a_stored_detection_still_seeds_a_form_the_user_has_not_confirmed(
        qapp, qtbot, tmp_path):
    """The other half, and the reason the seeding exists: a form still on its
    defaults must be re-seeded from the table on screen, or a reopened project shows
    the app default over a table measured with something else and Re-run quietly
    detects something different. Only a CONFIRMED value is protected."""
    # A table measured at windows that are NOT the app defaults: the only setting in
    # which "did the seed run?" is answerable, since the form owns just these two
    # values (`BaselinePanel.config` replaces them off DEFAULT_BASELINE so the app
    # keeps one baseline-parameter literal).
    stored = DEFAULT_BASELINE.replace(rough_s=2.0, base_s=0.5)
    # select=False: the unseeded intermediate state this test needs is only reachable
    # while the panel looks elsewhere — installing a result on the CURRENT selection
    # now seeds the form at once (see the `_window` docstring).
    win, doc_id = _window(qtbot, tmp_path, _signal(), config=stored, select=False)
    panel = win._analysis_panel
    bp = panel._baseline
    assert float(bp.rough_s.value()) == DEFAULT_BASELINE.rough_s   # not seeded yet
    # Point the panel at the target the stored table belongs to -- the reopened-project
    # path, and the one `_seed_form_from` exists for.
    win._on_selection(doc_id, 0)
    QtWidgets.QApplication.processEvents()

    assert float(bp.rough_s.value()) == 2.0
    assert float(bp.base_s.value()) == 0.5
    assert bp.config().fingerprint(FS) == stored.fingerprint(FS)
    assert panel._stale.isHidden()


def test_a_confirmed_configuration_does_not_follow_the_user_to_another_recording(
        qapp, qtbot, tmp_path):
    """"Parameters are per-recording" is the spec's first sentence about them. A
    value confirmed for one file must not silently become the estimator for the next
    one the user opens -- that is the same defect as a surrogate study's answer
    landing on the wrong file, and it is why the protection is keyed on the target
    rather than on the panel."""
    win, doc_id = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    bp = panel._baseline
    bp.rough_s.setValue(2.0)
    bp.base_s.setValue(0.5)
    QtWidgets.QApplication.processEvents()

    other = win._session.add_document(_doc(tmp_path, _signal(seed=11), name="other.dat"))
    win._create_default_layers(win._session.document(other))
    df, prov = detect_events(_signal(seed=11), FS, baseline_config=OLD, polarity=-1,
                             min_dwell_s=1e-4, recording="other", pore_id="pore")
    block = OLD.to_provenance()
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=other, channel=0,
        params={"min_dwell_s": 1e-4,
                **{k: block[k] for k in BASELINE_PARAM_KEYS if k in block}},
        calibration_fingerprint=win._session.calibration(other, 0).fingerprint(),
        payload=EventSet(df, prov), baseline_fingerprint=prov.get("baseline_fingerprint")))
    win._on_selection(other, 0)
    QtWidgets.QApplication.processEvents()

    assert bp.param_source() == "default", (
        "the previous recording's confirmed estimator followed the user to a new file"
    )
    assert float(bp.rough_s.value()) == DEFAULT_BASELINE.rough_s
    assert float(bp.base_s.value()) == DEFAULT_BASELINE.base_s

    # ...and going back is a new target too, so it does not resurrect the old value:
    # a confirmation is per-visit, and the app says so in the source badge rather
    # than remembering something the user cannot see.
    win._on_selection(doc_id, 0)
    QtWidgets.QApplication.processEvents()
    assert bp.param_source() == "default"


def test_every_baseline_job_says_it_finished(qapp, qtbot, tmp_path, monkeypatch):
    """A started job leaves a sentence in the status bar; a job that never says it
    ended leaves that sentence there for the rest of the session.

    Measured before the fix: dismissing the comparison dialog produced exactly ONE
    ``baselineJobStatus`` — ('started', 'Comparing open-pore estimators on this
    recording…') — and no second signal ever arrived, so MainWindow's status bar read
    "Comparing…" indefinitely over a finished job. Accepting the dialog happened to
    emit a terminal message from the re-measure it launched, which is why the
    existing tests could not see it.
    """
    win, _ = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said: list[tuple[str, str]] = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))
    monkeypatch.setattr(EstimatorCompareDialog, "exec",
                        lambda self: QtWidgets.QDialog.Rejected)

    panel.compare_baseline_estimators()
    _pump(qtbot, panel)

    assert [t for t, _ in said][0] == "started"
    assert len(said) >= 2, f"the compare job never said it finished: {said}"
    assert said[-1][0] == "compared"
    assert "Nothing changed" in said[-1][1]

    said.clear()
    panel.remeasure_active_detection()
    _pump(qtbot, panel)
    assert [t for t, _ in said] == ["started", "remeasured"], said


def test_a_finished_job_whose_target_vanished_still_says_so(qapp, qtbot, tmp_path):
    """The other five silent exits: the recording closed, the detection replaced, the
    target lost. Each was a bare ``return``. The guarantee is written once, in the
    wrapper, so a sixth exit cannot forget it."""
    win, doc_id = _window(qtbot, tmp_path, _signal())
    panel = win._analysis_panel
    said: list[tuple[str, str]] = []
    panel.baselineJobStatus.connect(lambda t, m: said.append((t, m)))

    panel._remeasure_target = None      # the shape every early return leaves behind
    panel._on_remeasure_finished(object(), install=True)
    assert said and said[-1][0] == "dropped"
    assert "no longer on screen" in said[-1][1]


def test_only_one_function_emits_the_baseline_job_status():
    """The guarantee above is "exactly one terminal message per started job", and it
    is enforced by a flag that ``_say_baseline`` sets. A second emitter bypasses the
    flag, so this pins that there is not one — the same call-shape guard this project
    uses for `characterize` and for the flag tokens."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "nanoscope" / "gui" / "analysis_panel.py"
    tree = ast.parse(src.read_text())
    owners: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "emit"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "baselineJobStatus"):
                owners.append(fn.name)
    assert owners == ["_say_baseline"], (
        f"baselineJobStatus.emit is called from {sorted(set(owners))}; route it "
        "through _say_baseline or a job can finish without saying so"
    )
