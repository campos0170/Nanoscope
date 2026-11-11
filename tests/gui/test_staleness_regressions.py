"""Staleness / state regressions: what a result REMEMBERS, and what state belongs to whom.

Five defects, one theme — a piece of state that should have been attached to a result or a
recording was instead re-derived, dropped, or kept in a single panel-global slot:

* **C5** a detection's ``params`` were never persisted, so a reopened project's *Re-run*
  re-detected with the PANEL'S DEFAULTS and replaced the event table — and every hand
  curation on it — with something else. The detection form was not re-seeded either, so the
  tab showed defaults above a table produced by other numbers. Its separate sub-bug: the
  stale banner was hidden BEFORE the guard that returns, making a re-run that could not run
  a banner-dismissing no-op.
* **C6** the ``calibration_fingerprint`` was re-read from the LIVE calibration on load, so a
  round-trip cleared a "calibration changed" banner that was telling the truth — leaving
  gain-1 depths on screen under a gain-2 calibration with no warning.
* **C11** ``resultsChanged`` drove four refresh slots but not the Plots window's, so it kept
  showing rows from a superseded detection — with a PNG export button beside them.
* **C8** the visible-range / event-zoom mappings used the document's native x-axis, ignoring
  the layer's alignment offset: "Detect over the visible range" (the DEFAULT ML mode) ran
  over the WHOLE recording while the user framed one event.
* **C7** the recovery map and the FDR readout were panel-global, keyed by nothing, so
  selecting another recording carried them over: B's dwell fit divided by A's eta(t) and
  said so in print.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import json
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from conftest import live_doc_id
from nanoscope.analysis import ml_detect
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.analysis_workspace import fingerprint_frame
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000

#: Six resolved blockades (3 ms), then thirty brief spikes (0.4 ms) packed into the tail. At
#: the panel's DEFAULT ``min_dwell_s`` (2e-5 s = 2 samples) the detector reports all
#: thirty-six; at 0.0022 s (220 samples) only the six. That gap is the whole point: it makes
#: "which settings ran?" an observable fact about the event table rather than a claim about a
#: widget. (It is also the shape of the reported defect: n 6 -> 20 after a reload.)
_LONG = [(8_000 + 12_000 * i, 8_300 + 12_000 * i) for i in range(6)]
_SHORT = [(76_000 + 700 * j, 76_040 + 700 * j) for j in range(30)]
_TUNED_MIN_DWELL_S = 0.0022
_DEFAULT_MIN_DWELL_S = 2e-5


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.0, N)
    for s, e in (*_LONG, *_SHORT):
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the TextTraceReader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    """An in-memory doc whose meta.path IS ``path``, so its doc_id == the reopened one."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _fresh_window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class _FakeEventModel:
    """A stand-in Keras model that fires on the deep, globally-normalised event samples and
    stays quiet on the open pore — a deterministic detector with no TensorFlow involved
    (mirrors the one in ``test_ml_detection.py``; duplicated rather than imported because the
    test modules are not a package)."""

    def __init__(self, window: int) -> None:
        self.window = window

    def predict(self, arr, verbose=0):
        return (np.abs(np.asarray(arr, dtype=np.float32)) > 0.5).astype(np.float16)


def _use_fake_model(monkeypatch, window: int = 3072):
    wrapper = ml_detect.KerasEventModel(
        model=_FakeEventModel(window), window=window, path="<fake>", mtime=0.0,
        input_shape=(None, window, 1), output_shape=(None, window, 1))
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _path: wrapper)
    return wrapper


def _install(win: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    ch = next(c.index for c in doc.meta.channels
              if c.kind is ChannelKind.CURRENT and not c.is_dead)
    win._on_selection(doc_id, ch)
    return doc_id, ch


def _detect(win: MainWindow, qtbot, *, min_dwell_s: float = _TUNED_MIN_DWELL_S) -> None:
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _n_events(win: MainWindow, doc_id: str, ch: int) -> int:
    return len(win._session.results.get(doc_id, ch, "detect.schmitt").payload.df)


def _regain(win: MainWindow, doc_id: str, ch: int, gain: float) -> None:
    win._session.set_calibration(
        doc_id, ch, ChannelCalibration(gain, 0.0, "pA", Provenance.UNKNOWN))


def _tuned_project(qtbot, tmp_path) -> dict:
    """One recording, detected with a DELIBERATELY non-default ``min_dwell_s``, saved."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(csv, arr))
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    n_tuned = _n_events(win, doc_id, ch)
    assert n_tuned == len(_LONG), "fixture: the tuned floor must keep only the long events"

    proj = tmp_path / "session.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()
    return {"proj": str(proj), "doc_id": doc_id, "ch": ch, "n_tuned": n_tuned}


def _reopen(qtbot, proj: str) -> MainWindow:
    w = _fresh_window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=60_000):
        w._open_project(proj)
    qtbot.wait(50)
    return w


# ================================================================================================
# fixture sanity — the two floors really do disagree
# ================================================================================================

def test_the_fixture_separates_the_two_detection_floors(qapp, qtbot, tmp_path):
    """Guards every C5 assertion below: if the default floor did not add events, "the
    defaults ran" and "the stored settings ran" would look identical."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n_default = _n_events(win, doc_id, ch)
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    n_tuned = _n_events(win, doc_id, ch)
    assert n_tuned == len(_LONG)
    assert n_default > n_tuned
    win._analysis_controller.abort()


# ================================================================================================
# C5 — a detection has to remember the settings it ran under
# ================================================================================================

def test_a_reopened_detection_carries_the_settings_it_ran_under(qapp, qtbot, tmp_path):
    """REGRESSION: ``ResultRec`` had no params field and the restore wrote ``params={}``, so
    a reopened detection could not say what produced it."""
    ctx = _tuned_project(qtbot, tmp_path)
    w2 = _reopen(qtbot, ctx["proj"])

    # By SOURCE: a reopened file is a new LOAD with a new doc_id (core/identity.py).
    res = w2._session.results.get(
        live_doc_id(w2._session, ctx["doc_id"]), ctx["ch"], "detect.schmitt")
    assert res is not None
    assert res.params, "a restored detection came back with no params at all"
    assert res.params["min_dwell_s"] == pytest.approx(_TUNED_MIN_DWELL_S)
    w2._controller.abort()


def test_the_detection_form_is_reseeded_from_the_reopened_detection(qapp, qtbot, tmp_path):
    """REGRESSION: the tab showed the panel's DEFAULTS above a table detected at 0.0022 s —
    two contradictory statements on one screen, with no way to tell which was true."""
    ctx = _tuned_project(qtbot, tmp_path)
    w2 = _reopen(qtbot, ctx["proj"])

    form = w2._analysis_panel._detection
    assert form.min_dwell_s.value() == pytest.approx(_TUNED_MIN_DWELL_S)
    assert form.params()["min_dwell_s"] == pytest.approx(_TUNED_MIN_DWELL_S)
    w2._controller.abort()


def test_re_running_a_reopened_detection_does_not_re_detect_with_the_defaults(
        qapp, qtbot, tmp_path):
    """REGRESSION, the money case: reopen a project, and the stale banner's *Re-run* detected
    with ``min_dwell_s`` 2e-05 instead of 0.0022 — twenty events where there were six."""
    ctx = _tuned_project(qtbot, tmp_path)
    w2 = _reopen(qtbot, ctx["proj"])
    doc_id, ch = live_doc_id(w2._session, ctx["doc_id"]), ctx["ch"]

    _regain(w2, doc_id, ch, 2.0)                     # the calibration edit the banner is about
    qtbot.wait(10)
    assert not w2._analysis_panel._stale.isHidden()

    with qtbot.waitSignal(w2._analysis_controller.finished, timeout=60_000):
        w2._analysis_panel._on_rerun()
    qtbot.wait(20)

    res = w2._session.results.get(doc_id, ch, "detect.schmitt")
    assert len(res.payload.df) == ctx["n_tuned"]
    assert res.payload.provenance["min_dwell_s"] == pytest.approx(_TUNED_MIN_DWELL_S)
    w2._controller.abort()


def test_re_running_a_reopened_detection_keeps_the_curation_it_had(qapp, qtbot, tmp_path):
    """The consequence the user actually feels: a re-detection REPLACES the event table, so
    re-detecting with the wrong settings threw away every accept/reject decision."""
    ctx = _tuned_project(qtbot, tmp_path)
    w2 = _reopen(qtbot, ctx["proj"])
    doc_id, ch = live_doc_id(w2._session, ctx["doc_id"]), ctx["ch"]

    panel = w2._analysis_panel
    panel.set_event_accepted(0, False)               # one hand rejection to survive the re-run
    panel.set_event_accepted(2, False)
    _regain(w2, doc_id, ch, 2.0)
    qtbot.wait(10)
    with qtbot.waitSignal(w2._analysis_controller.finished, timeout=60_000):
        panel._on_rerun()
    qtbot.wait(20)

    df = w2._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    # The re-run is a fresh detection, so acceptance resets — but the event SET must be the
    # same six events, in the same places, so the user's curation is re-applicable at all.
    assert len(df) == ctx["n_tuned"]
    starts = sorted(int(s) for s in df["start_idx"])
    assert len(starts) == len(_LONG)
    for got, (want, _end) in zip(starts, _LONG):
        assert abs(got - want) < 50
    w2._controller.abort()


def test_a_rerun_that_cannot_run_does_not_dismiss_the_staleness_banner(qapp, qtbot, tmp_path):
    """REGRESSION (C5's sub-bug): ``_on_rerun`` hid the banner BEFORE the guard that returns.
    An ML result with no model path therefore answered the click by deleting the only warning
    that the depths on screen were computed under the old gain — and recomputing nothing."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))

    # An ML detection whose params carry no model_path — exactly what a pre-v3 project
    # restored (params={}), and what an ML result whose model has moved looks like. The
    # EventSet itself is a real detection's, so the panel has a genuine table to show.
    _detect(win, qtbot)
    events = win._session.results.get(doc_id, ch, "detect.schmitt").payload
    win._session.add_result(AnalysisResult(
        module_id="detect.keras", module_version="1.0", doc_id=doc_id, channel=ch,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events))
    qtbot.wait(10)
    panel = win._analysis_panel
    panel._active_detection = "detect.keras"
    panel._load_stored_detection(doc_id, ch)

    _regain(win, doc_id, ch, 2.0)
    qtbot.wait(10)
    assert not panel._stale.isHidden(), "fixture: the banner must be up before the click"

    panel._on_rerun()
    qtbot.wait(10)
    assert not panel._stale.isHidden(), (
        "the click dismissed the warning without recomputing anything")

    # ...and the same holds for the Schmitt path's silent refusals: the banner is dismissed
    # by a RUN, not by a CLICK, so a re-run the gate turns away leaves the warning standing.
    panel._active_detection = "detect.schmitt"
    panel._load_stored_detection(doc_id, ch)
    assert not panel._stale.isHidden()
    panel._sel = None                       # the "no target" refusal inside _run_detection
    panel._on_rerun()
    qtbot.wait(10)
    assert not panel._stale.isHidden()
    win._analysis_controller.abort()


def test_a_successful_rerun_still_clears_the_banner(qapp, qtbot, tmp_path):
    """Control: the banner must still go away when the re-run really happens."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot)
    _regain(win, doc_id, ch, 2.0)
    qtbot.wait(10)
    panel = win._analysis_panel
    assert not panel._stale.isHidden()

    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel._on_rerun()
    qtbot.wait(20)

    assert panel._stale.isHidden()
    res = win._session.results.get(doc_id, ch, "detect.schmitt")
    assert tuple(res.calibration_fingerprint) == (2.0, 0.0)   # recomputed under the new gain
    win._analysis_controller.abort()


# ================================================================================================
# C6 — a result's staleness stamp is the calibration it was COMPUTED under
# ================================================================================================

def test_a_stale_detection_is_still_stale_after_a_project_round_trip(qapp, qtbot, tmp_path):
    """REGRESSION: detect at gain 1, re-gain to 2 (banner up), save, reopen — and the banner
    was GONE while every depth on screen was still the gain-1 number, because the restore
    re-read the fingerprint off the live calibration."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(csv, arr))
    _detect(win, qtbot)
    depth_before = float(
        win._session.results.get(doc_id, ch, "detect.schmitt").payload.df["depth_pA"].iloc[0])

    _regain(win, doc_id, ch, 2.0)                    # the deferred current-gain workflow
    qtbot.wait(10)
    assert not win._analysis_panel._stale.isHidden()

    proj = tmp_path / "stale.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _reopen(qtbot, str(proj))
    live = live_doc_id(w2._session, doc_id)
    res = w2._session.results.get(live, ch, "detect.schmitt")
    assert tuple(res.calibration_fingerprint) == (1.0, 0.0), (
        "the stamp must be the calibration the result was COMPUTED under")
    assert w2._session.calibration(live, ch).gain == 2.0      # ...and the live one is the new
    assert float(res.payload.df["depth_pA"].iloc[0]) == pytest.approx(depth_before)
    assert not w2._analysis_panel._stale.isHidden(), (
        "the reopened project cleared a warning that was still true")
    w2._controller.abort()


def test_a_project_saved_in_step_with_its_calibration_reopens_not_stale(qapp, qtbot, tmp_path):
    """Control: the fix must not make everything look stale. A result saved under the same
    calibration it was computed with comes back with no banner."""
    ctx = _tuned_project(qtbot, tmp_path)
    w2 = _reopen(qtbot, ctx["proj"])
    assert w2._analysis_panel._stale.isHidden()
    res = w2._session.results.get(
        live_doc_id(w2._session, ctx["doc_id"]), ctx["ch"], "detect.schmitt")
    assert tuple(res.calibration_fingerprint) == (1.0, 0.0)
    w2._controller.abort()


# ================================================================================================
# C11 — the Plots window is a result consumer like any other
# ================================================================================================

def _plot_rows(win: MainWindow, doc_id: str, ch: int) -> int:
    """Row count of THIS (doc, channel)'s source as the open Plots window holds it."""
    key = next(s for s in win._plots_window.sources()
               if s.doc_id == doc_id and s.channel == ch)
    return int(len(key.df))


def _figure_status(win: MainWindow) -> str:
    """The open Distributions figure's freshness record: empty / current / stale / frozen."""
    owner = win._figure_workspace
    return owner.get(owner.current_id()).spec.result.status


def _chip(win: MainWindow) -> str:
    """The freshness word the user reads in the Explorer's header, once Qt has run what the
    change queued. The record's status and ``MainWindow._analysis_explorer_status`` are what
    that word is made from; a header that stopped being refreshed would pass both."""
    QtCore.QCoreApplication.processEvents()
    return win._analysis_explorer._status_label.text()


def _update_plot(win: MainWindow, qtbot) -> None:
    """The user's Update plot: the Explorer toolbar's own button."""
    win._analysis_explorer.toolbar().update_action.trigger()
    qtbot.wait(20)


def _refresh_from_recordings(win: MainWindow, qtbot) -> None:
    """Update plot ▾ ▸ Refresh from recordings: the Explorer's explicit repull."""
    win._analysis_explorer.refresh_current()
    qtbot.wait(20)


def _drawn_n(win: MainWindow, label: str) -> int:
    """The ``n`` the Details pane states for the source called ``label``: the rows the
    picture on screen was drawn from (not the catalog the window holds for its next Plot)."""
    text = win._plots_window.details_text()
    hit = re.search(rf"{re.escape(label)}: n = (\d+)", text)
    assert hit, f"the Details pane no longer states n for {label!r}:\n{text}"
    return int(hit.group(1))


def _plotted_figure(qtbot, tmp_path, name: str = "a.csv"):
    """One recording detected at the default floor (36 events), Analysis > Plots, and the
    user's first Update plot: a plotted figure that says CURRENT. Returns the window, the
    figure's id, the recording's (doc_id, channel) and the row count the figure was drawn from."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / name, _trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n = _n_events(win, doc_id, ch)
    win._open_plots()
    qtbot.wait(20)
    key = win._figure_workspace.current_id()
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert _drawn_n(win, name) == n
    return win, key, (doc_id, ch), n


# 2026-11-11: the tests below were written when the Plots window redrew itself on every
# session change. Since 8661d35 (plotting redesign, "independent named figures") Analysis >
# Plots opens a MANAGED Distributions figure, and a session change no longer redraws it: it
# only marks a plotted figure STALE (a figure that was never plotted stays NO DATA and draws
# nothing). The user's Update plot -- the Explorer toolbar's own button -- is what takes the
# session's current event tables, when a table the figure uses is no longer the session's
# (``MainWindow._stage_changed_distribution_sources``). What each pair of tests pins is still
# the original C11 promise -- the window HEARS about a replaced or added table -- in the two
# states a figure can be in. Before that step Update plot redrew the superseded rows and
# labelled them CURRENT.

def test_a_re_detection_reaches_the_open_plots_window(qapp, qtbot, tmp_path):
    """REGRESSION: ``resultsChanged`` was wired to four refresh slots but not the Plots
    window's. Curation appeared to flow through only because the panel mutates the SHARED
    frame in place; a re-detection REPLACES it, and the window went on showing the old rows —
    beside a PNG export button that would write a figure of data that no longer exists.

    This is the figure that was never plotted: the re-detection draws nothing in it (it stays
    NO DATA), and its first Update plot takes the NEW table, never the superseded one."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n_before = _n_events(win, doc_id, ch)

    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    assert _figure_status(win) == "empty", "fixture: this half is about a never-plotted figure"
    assert _plot_rows(win, doc_id, ch) == n_before

    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)   # fewer events now
    n_after = _n_events(win, doc_id, ch)
    assert n_after < n_before
    # The session changed and nothing was drawn: still never plotted, still NO DATA.
    assert _figure_status(win) == "empty" and _chip(win) == "NO DATA"
    assert page._plot_armed is False
    assert not page._plot.listDataItems()

    _update_plot(win, qtbot)                               # its first Update plot
    assert _plot_rows(win, doc_id, ch) == n_after, (
        "the Plots window kept the superseded detection's rows")
    assert _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_re_detection_marks_a_plotted_figure_stale_and_update_draws_the_new_rows(
        qapp, qtbot, tmp_path):
    """The figure that HAS been plotted keeps exactly what it drew and says so (STALE) while
    a re-detection replaces its table; the explicit Update plot then draws the replacement.
    The measured defect: the window never heard about the new table, so Update redrew the
    36 superseded rows and the header said CURRENT over a detection that no longer existed."""
    win, key, (doc_id, ch), n_before = _plotted_figure(qtbot, tmp_path)
    page = win._plots_window

    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)   # fewer events now
    n_after = _n_events(win, doc_id, ch)
    assert n_after < n_before
    # The figure says it is out of date -- on screen, not only in its record -- and has not
    # been touched behind the user's back: the notice staged nothing and drew nothing ...
    assert _figure_status(win) == "stale"
    assert win._analysis_explorer_status(key, page) == ("STALE", "stale")
    assert _chip(win) == "STALE"
    assert page._figure_pending_sources is None
    assert _plot_rows(win, doc_id, ch) == n_before
    assert _drawn_n(win, "a.csv") == n_before
    # ... and the explicit Update draws the replacement table, never the superseded one.
    _update_plot(win, qtbot)
    assert _plot_rows(win, doc_id, ch) == n_after, (
        "Update plot redrew the superseded detection's rows")
    assert _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current"
    assert win._analysis_explorer_status(key, page) == ("CURRENT", "current")
    assert _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def _shattered_trace(seed: int = 0) -> np.ndarray:
    """`_trace()` plus ONE genuinely shattered blockade: two 0.4 ms fragments split by a
    0.1 ms gap, in the quiet stretch between the long events and the spike train.

    The plain `_trace()` cannot be merged honestly. Its `_SHORT` spikes are 0.4 ms wide
    and 7 ms apart, so any window wide enough to join them produces one 20 ms "event"
    that is 94% open pore — which is the runaway `merge_by_gap`'s occupancy floor exists
    to refuse (see MIN_MERGE_OCCUPANCY). This test is about a merge REACHING the Plots
    window, not about merge quality, so it needs a merge that deserves to happen.
    """
    x = _trace(seed)
    for s, e in _SHATTERED:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


#: Two fragments of one blockade: 40 samples each, 10 samples (0.1 ms) apart. Occupancy of
#: the merged span is 80/90 = 0.89, comfortably above the floor.
_SHATTERED = [(60_000, 60_040), (60_050, 60_090)]


def _merge_the_shattered_pair(win: MainWindow, qtbot):
    """'Merge all' at 1 ms, through the panel; returns the merge report.

    'Merge all' runs on the analysis controller (it costs 3.3-8.7 s on the owner's real
    recordings), so the click returns "started" and the outcome arrives on
    ``mergeFinished``. 1 ms joins the shattered pair (0.1 ms apart) and nothing else --
    the `_SHORT` spikes are 7 ms apart, and a window wide enough for THEM is refused by
    the occupancy floor rather than reporting one 20 ms mostly-open-pore event."""
    with qtbot.waitSignal(win._analysis_panel.mergeFinished, timeout=30_000) as landed:
        started = win._analysis_panel.merge_all_subevents(0.001)
    assert started.status == "started"
    qtbot.wait(20)
    report = landed.args[0]
    assert report.status == "ok" and report.n_after < report.n_before
    return report


def test_a_merge_reaches_the_open_plots_window(qapp, qtbot, tmp_path):
    """The same hole, through the other frame-REPLACING edit: a sub-event merge. (A figure
    that was never plotted draws nothing on the merge; its first Update plot takes the merged
    table.)"""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _shattered_trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    assert _figure_status(win) == "empty", "fixture: this half is about a never-plotted figure"
    assert _plot_rows(win, doc_id, ch) == _n_events(win, doc_id, ch)

    report = _merge_the_shattered_pair(win, qtbot)
    assert _figure_status(win) == "empty" and _chip(win) == "NO DATA"
    assert page._plot_armed is False
    assert not page._plot.listDataItems()

    _update_plot(win, qtbot)
    assert _plot_rows(win, doc_id, ch) == report.n_after
    assert _drawn_n(win, "a.csv") == report.n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_merge_marks_a_plotted_figure_stale_and_update_draws_the_merged_rows(
        qapp, qtbot, tmp_path):
    """The plotted half of the merge case: STALE while the drawn rows stay as they were,
    then Update plot draws the merged table (before: the pre-merge rows, labelled CURRENT)."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _shattered_trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._open_plots()
    qtbot.wait(20)
    owner = win._figure_workspace
    key = owner.current_id()
    page = win._plots_window
    _update_plot(win, qtbot)
    n_before = _n_events(win, doc_id, ch)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert _drawn_n(win, "a.csv") == n_before

    report = _merge_the_shattered_pair(win, qtbot)
    assert report.n_before == n_before
    assert _figure_status(win) == "stale"
    assert win._analysis_explorer_status(key, page) == ("STALE", "stale")
    assert _chip(win) == "STALE"
    assert page._figure_pending_sources is None
    assert _plot_rows(win, doc_id, ch) == n_before
    assert _drawn_n(win, "a.csv") == n_before

    _update_plot(win, qtbot)
    assert _plot_rows(win, doc_id, ch) == report.n_after, (
        "Update plot redrew the pre-merge rows")
    assert _drawn_n(win, "a.csv") == report.n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_detection_on_another_recording_appears_in_the_open_plots_window(
        qapp, qtbot, tmp_path):
    """And through the addition case: a second recording's detection is a new source to
    overlay, which is half the reason the window takes a list of them. (A figure that was
    never plotted draws nothing when it arrives; its first Update plot offers it.)"""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    assert _figure_status(win) == "empty", "fixture: this half is about a never-plotted figure"
    assert not any(s.doc_id == doc_b for s in win._plots_window.sources())

    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot)
    qtbot.wait(20)
    assert _figure_status(win) == "empty" and _chip(win) == "NO DATA"
    assert page._plot_armed is False
    assert not page._plot.listDataItems()

    _update_plot(win, qtbot)
    assert any(s.doc_id == doc_b for s in win._plots_window.sources()), (
        "the open Plots window never heard about the second detection")
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


@pytest.mark.parametrize("carries", ["a histogram", "a PCA", "a cluster split"])
def test_a_detection_on_another_recording_leaves_a_plotted_figure_what_it_drew(
        qapp, qtbot, tmp_path, carries):
    """The plotted half of the addition case, and the rule that keeps Update plot from
    costing a figure its analysis: it takes the session's tables only when a table the figure
    USES changed. A detection on a recording the figure does not draw marks it STALE (the
    session changed) and nothing more: Update plot ends CURRENT with the PCA or the cluster
    split it had, the same picture, and no staged catalog. (Installing a catalog clears
    PCA and clustering, so installing one for a change that is not the figure's would blank a
    PCA figure and unsplit a coloured histogram.) Refresh from recordings is the explicit
    route that brings the new recording in."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._open_plots()
    qtbot.wait(20)
    owner = win._figure_workspace
    key = owner.current_id()
    page = win._plots_window
    if carries != "a histogram":
        page.set_feature_selection(["dwell_s", "depth_pA"])
        assert (page.run_pca() if carries == "a PCA" else page.run_clustering()) is not None
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    pca, clusters = page.pca_result(), page.cluster_result()
    assert (pca is not None) is (carries == "a PCA")
    assert (clusters is not None) is (carries == "a cluster split")
    details, items = page.details_text(), len(page._plot.listDataItems())

    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    qtbot.wait(20)
    assert _figure_status(win) == "stale"
    assert win._analysis_explorer_status(key, page) == ("STALE", "stale")
    assert _chip(win) == "STALE"
    assert page._figure_pending_sources is None
    assert not any(s.doc_id == doc_b for s in page.sources())

    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert page.pca_result() is pca, "Update plot threw away the PCA of an unchanged table"
    assert page.cluster_result() is clusters, (
        "Update plot threw away the clustering of an unchanged table")
    assert page.details_text() == details and len(page._plot.listDataItems()) == items
    assert not any(s.doc_id == doc_b for s in page.sources())

    _refresh_from_recordings(win, qtbot)
    assert any(s.doc_id == doc_b for s in page.sources()), (
        "Refresh from recordings no longer offers a new recording")
    win._analysis_controller.abort()


def test_a_dormant_figure_activated_after_a_re_detection_takes_the_new_table_at_update(
        qapp, qtbot, tmp_path):
    """A figure that was dormant when its table was replaced (more than six figures, or not
    opened yet in this session) is built later from its record, which holds the table it was
    drawn from. Opening it says STALE over that table; Update plot takes the session's."""
    win, key, (doc_id, ch), n_before = _plotted_figure(qtbot, tmp_path)
    owner = win._figure_workspace
    assert owner.suspend(key) and owner.page(key) is None

    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    n_after = _n_events(win, doc_id, ch)
    assert n_after < n_before
    assert _figure_status(win) == "stale"            # the dormant record heard about it

    assert owner.activate(key)
    qtbot.wait(20)
    assert win._plots_window is owner.page(key) and win._plots_window is not None
    assert _chip(win) == "STALE"
    assert _plot_rows(win, doc_id, ch) == n_before
    assert _drawn_n(win, "a.csv") == n_before

    _update_plot(win, qtbot)
    assert _plot_rows(win, doc_id, ch) == n_after, (
        "Update plot redrew the superseded rows of a figure that was built from its record")
    assert _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_duplicate_of_a_stale_figure_takes_the_new_table_at_update(qapp, qtbot, tmp_path):
    """Duplicate builds the copy's page from the original's record: it holds the table the
    original was drawn from and starts as stale as the original. Update plot on the copy takes
    the session's table, and so does the original's own, so two figures of one source never
    disagree about it (both used to say CURRENT over different numbers)."""
    win, key, (doc_id, ch), n_before = _plotted_figure(qtbot, tmp_path)
    owner = win._figure_workspace

    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    n_after = _n_events(win, doc_id, ch)
    assert n_after < n_before
    assert _figure_status(win) == "stale"

    copy = owner.duplicate(key)
    qtbot.wait(20)
    assert copy and owner.current_id() == copy and win._plots_window is owner.page(copy)
    assert _chip(win) == "STALE"
    assert _plot_rows(win, doc_id, ch) == n_before

    _update_plot(win, qtbot)
    assert _plot_rows(win, doc_id, ch) == n_after, (
        "Update plot redrew the superseded rows of a duplicated figure")
    assert _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"

    assert owner.activate(key)
    qtbot.wait(20)
    assert _chip(win) == "STALE" and _drawn_n(win, "a.csv") == n_before
    _update_plot(win, qtbot)
    assert _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


@pytest.mark.parametrize("re_detected", [True, False], ids=["re-detected", "unchanged"])
def test_a_reopened_saved_result_takes_a_replaced_table_at_update_and_keeps_its_pca_otherwise(
        qapp, qtbot, tmp_path, re_detected):
    """A project reopens a plotted figure as a SAVED RESULT built from the copies of its inputs
    that it saved, which are never the session's frames. Update plot compares them with the
    session's tables by CONTENT: after a re-detection the figure takes the new table (before,
    it redrew the saved copy and said CURRENT over a detection the session no longer had); on a
    reopen where nothing changed the saved copies equal the session's tables, nothing is
    staged, and a figure that carries a PCA keeps it. (The re-detected leg saves a plain
    histogram: a replaced table clears a PCA, and a PCA figure then reads NO DATA.)"""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(csv, arr))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n_saved = _n_events(win, doc_id, ch)
    win._open_plots()
    qtbot.wait(20)
    if not re_detected:
        win._plots_window.set_feature_selection(["dwell_s", "depth_pA"])
        assert win._plots_window.run_pca() is not None
    _update_plot(win, qtbot)
    proj = tmp_path / "figure.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _reopen(qtbot, str(proj))
    live = live_doc_id(w2._session, doc_id)
    if re_detected:
        w2._on_selection(live, ch)
        _detect(w2, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    n_now = _n_events(w2, live, ch)
    assert (n_now < n_saved) is re_detected, "fixture: the session's table must differ or not"

    w2._open_plots()
    qtbot.wait(20)
    page = w2._plots_window
    assert _figure_status(w2) == "frozen" and _chip(w2) == "SAVED RESULT"
    assert _plot_rows(w2, live, ch) == n_saved          # the figure holds the table it saved
    pca = page.pca_result()
    assert (pca is not None) is (not re_detected)

    _update_plot(w2, qtbot)
    assert _figure_status(w2) == "current" and _chip(w2) == "CURRENT"
    if re_detected:
        assert _plot_rows(w2, live, ch) == n_now, (
            "Update plot redrew the saved copy of a table the session replaced")
        assert _drawn_n(w2, "pore.csv") == n_now
    else:
        assert _plot_rows(w2, live, ch) == n_saved
        assert page.pca_result() is pca, (
            "Update plot threw away the PCA of a reopened figure whose table did not change")
        assert page.details_text().startswith("PCA on")
    w2._analysis_controller.abort()


def test_a_recording_the_figure_uses_that_is_no_longer_open_leaves_it_its_inputs(
        qapp, qtbot, tmp_path):
    """A closed recording leaves nothing to take: the figure drawn from it keeps the table it
    holds (its own copy is all there is of it), and its PCA with it. It says CURRENT after Update
    plot over its own inputs; Refresh from recordings is the explicit route that drops it."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)     # A is the detection on screen
    n_a = _n_events(win, doc_a, ch_a)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    key_a = next(s.key for s in page.sources() if s.doc_id == doc_a)
    assert page.selected_source_keys() == [key_a], "fixture: the figure opens on recording A"
    page.set_feature_selection(["dwell_s", "depth_pA"])
    pca = page.run_pca()
    assert pca is not None
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current"

    win._session.close_document(doc_a)
    qtbot.wait(30)
    assert key_a not in {s.key for s in win._plots_sources()}, "fixture: A is no longer offered"
    assert _figure_status(win) == "stale" and _chip(win) == "STALE"

    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert page.pca_result() is pca, "Update plot dropped the PCA of a figure it left no tables"
    assert page.selected_source_keys() == [key_a]
    assert next(len(s.df) for s in page.sources() if s.key == key_a) == n_a
    assert f"n = {n_a} kept" in page.details_text()
    win._analysis_controller.abort()


def test_a_never_plotted_figure_keeps_a_ticked_recording_that_is_no_longer_open(
        qapp, qtbot, tmp_path):
    """The same rule for the figure that was never plotted: its first Update plot offers the
    session's tables, but it does not swap a recording the figure has ticked, and that is no
    longer open, for whichever one the session has now."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot)
    n_a = _n_events(win, doc_a, ch_a)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    key_a = next(s.key for s in page.sources() if s.doc_id == doc_a)
    assert page.selected_source_keys() == [key_a], "fixture: the figure opens on recording A"
    assert _figure_status(win) == "empty", "fixture: this half is about a never-plotted figure"

    win._session.close_document(doc_a)
    qtbot.wait(30)
    assert key_a not in {s.key for s in win._plots_sources()}, "fixture: A is no longer offered"

    _update_plot(win, qtbot)
    assert page.selected_source_keys() == [key_a]
    assert _drawn_n(win, "a.csv") == n_a
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_notice_marks_figures_stale_and_stages_nothing(qapp, qtbot, tmp_path, monkeypatch):
    """A session notice (``FigureWorkspace.sources_changed``) only marks figures stale, as in
    the released build: it builds no source catalog, installs nothing in a page, draws nothing
    and is not an edit of the project. Staging a catalog on every notice for every resident
    page cost PCA and clustering after changes that were not the figure's, and built the
    session's catalog once per page per notice, hidden pages included."""
    win, plotted, (doc_id, ch), n_before = _plotted_figure(qtbot, tmp_path)
    owner = win._figure_workspace
    plotted_page = owner.page(plotted)
    never = owner.create("distributions")           # a second figure, never plotted, in front
    never_page = owner.page(never)
    qtbot.wait(30)                                    # what creating it queued has run
    assert owner.get(never).spec.result.status == "empty"

    catalogs = []
    real = win._plots_sources
    monkeypatch.setattr(win, "_plots_sources", lambda: (catalogs.append(1), real())[1])
    held = (plotted_page._sources, never_page._sources)
    items = tuple(plotted_page._plot.items)
    project = (win._dirty, win._session_revision)

    owner.sources_changed()
    qtbot.wait(10)

    assert catalogs == [], "a notice built the session's source catalog"
    assert owner.get(plotted).spec.result.status == "stale"
    assert owner.get(never).spec.result.status == "empty"
    assert plotted_page._figure_pending_sources is None
    assert never_page._figure_pending_sources is None
    assert plotted_page._sources is held[0] and never_page._sources is held[1], (
        "a notice installed a catalog in a page")
    assert tuple(plotted_page._plot.items) == items
    assert never_page._plot_armed is False and not never_page._plot.listDataItems()
    assert (win._dirty, win._session_revision) == project
    win._analysis_controller.abort()


def test_a_notice_leaves_a_legend_name_being_typed_on_a_never_plotted_figure(
        qapp, qtbot, tmp_path):
    """A detection, a batch or a merge can land while the user types a legend name in the
    Sources list of a figure that was never plotted. The notice must not rebuild that list: the
    editor stays open with the text typed so far. (Installing the new catalog on the notice
    closed the editor and put the row's own name back.)"""
    from PySide6 import QtWidgets

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._open_plots()
    qtbot.wait(30)
    lst = win._plots_window._source_list
    lst.editItem(lst.item(0))
    editing = QtWidgets.QAbstractItemView.State.EditingState
    editor = lst.findChild(QtWidgets.QLineEdit)
    assert lst.state() == editing and editor is not None, "fixture: a legend name is being typed"
    editor.setText("half typed na")

    win._figure_workspace.sources_changed()
    qtbot.wait(10)

    assert lst.state() == editing, "a notice closed the legend name being typed"
    assert lst.findChild(QtWidgets.QLineEdit) is editor and editor.text() == "half typed na"
    win._analysis_controller.abort()


def test_an_update_that_takes_a_new_table_keeps_the_ticks_names_and_look_the_user_set(
        qapp, qtbot, tmp_path):
    """What the user chose stays when Update plot takes a replaced table: the ticked sources,
    the legend name typed over a source, the title and colour (the whole style), and the Undo
    stack (arriving data is not an edit, so it is no step). Only the table, and what is
    computed from it, is replaced."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n_b = _n_events(win, doc_b, ch_b)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    key_a = next(s.key for s in page.sources() if s.doc_id == doc_a)
    key_b = next(s.key for s in page.sources() if s.doc_id == doc_b)
    page.set_sources([key_a, key_b])
    page.set_title("Kept title")
    page.set_data_color("#12ab34")
    lst = page._source_list
    row = next(i for i in range(lst.count()) if lst.item(i).data(QtCore.Qt.UserRole) == key_a)
    lst.item(row).setText("Apo")
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current"
    chosen = (page.selected_source_keys(), page.style(), page.history_mark())
    assert chosen[0] == [key_a, key_b], "fixture: both recordings are ticked"
    assert dict(chosen[1].label_overrides) == {key_a: "Apo"}, "fixture: A is named Apo"
    assert chosen[1].title == "Kept title" and chosen[1].data_color == "#12ab34", "fixture"

    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)   # recording A only is re-detected
    n_a_after = _n_events(win, doc_a, ch_a)
    assert _figure_status(win) == "stale" and _chip(win) == "STALE"
    _update_plot(win, qtbot)

    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert _drawn_n(win, "a.csv") == n_a_after, "the replaced table was not drawn"
    assert _drawn_n(win, "b.csv") == n_b
    assert (page.selected_source_keys(), page.style(), page.history_mark()) == chosen
    win._analysis_controller.abort()


def _after_a_replaced_table(qtbot, tmp_path, carries: str, route: str) -> dict:
    """What a Distributions figure carrying a PCA or a cluster split shows once its table was
    replaced and the user pressed Update plot (``route`` "update") or Update plot ▾ ▸ Refresh
    from recordings (``route`` "refresh")."""
    win, _key, (_doc_id, _ch), _n = _plotted_figure(qtbot, tmp_path)
    page = win._plots_window
    page.set_feature_selection(["dwell_s", "depth_pA"])
    assert (page.run_pca() if carries == "PCA" else page.run_clustering()) is not None
    _update_plot(win, qtbot)
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    (_update_plot if route == "update" else _refresh_from_recordings)(win, qtbot)
    message = page._message_item.toPlainText() if page._message_item is not None else None
    shown = {"status": _figure_status(win), "chip": _chip(win),
             "plot type": page._plot_type_combo.currentText(),
             "PCA kept": page.pca_result() is not None,
             "clusters kept": page.cluster_result() is not None,
             "Colour by": page.histogram_colour_by(),
             "canvas message": message, "details": page.details_text(),
             "drawn items": len(page._plot.listDataItems())}
    win._analysis_controller.abort()
    return shown


@pytest.mark.parametrize("carries", ["PCA", "cluster split"])
def test_update_ends_where_refresh_from_recordings_ends_when_a_table_was_replaced(
        qapp, qtbot, tmp_path, carries):
    """Update plot takes a replaced table the way Refresh from recordings does, so a figure
    carrying a PCA or a cluster split ends in the SAME state by either route: PCA and
    clustering are cleared with the replaced table (the rule ``PlotsWindow.set_data`` states
    and the guide describes), nothing is recomputed behind the user's back, and the header,
    the canvas and Details say the same thing either way.

    2026-11-11: what that shared state SAYS is not judged here, and is reported to the owner:
    a PCA figure reads NO DATA over "Select ≥2 feature columns on the left, then click Run
    PCA." with its features still ticked, and a histogram coloured by cluster reads CURRENT,
    unsplit, with Colour by still reading Cluster. This test only pins that Update plot does
    not end anywhere Refresh from recordings does not."""
    by_update = _after_a_replaced_table(qtbot, tmp_path, carries, "update")
    by_refresh = _after_a_replaced_table(qtbot, tmp_path, carries, "refresh")
    assert by_update == by_refresh
    assert not by_update["PCA kept"] and not by_update["clusters kept"], (
        "the replaced table did not clear what was computed from the superseded one")


def test_a_pca_computed_again_after_a_replaced_table_survives_the_next_update(
        qapp, qtbot, tmp_path):
    """A replaced table costs a figure its PCA once. The figure now holds the session's table,
    so a PCA run again on it is not thrown away by the next Update plot (it used to be, every
    time, for as long as a session notice had arrived since the last Update)."""
    win, key, (doc_id, ch), n_before = _plotted_figure(qtbot, tmp_path)
    page = win._plots_window
    page.set_feature_selection(["dwell_s", "depth_pA"])
    assert page.run_pca() is not None
    _update_plot(win, qtbot)
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    _update_plot(win, qtbot)
    assert page.pca_result() is None and _figure_status(win) == "empty", "fixture: PCA cleared"

    pca = page.run_pca()
    assert pca is not None
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert page.pca_result() is pca, "Update plot threw the PCA away again"
    assert page.details_text().startswith("PCA on")
    win._analysis_controller.abort()


def _re_detected_exponential_figure(qtbot, tmp_path):
    """The measured scenario up to the typing: a figure plotted on the 36 events of the default
    floor with an exponential fit says CURRENT, then its recording is re-detected at the tuned
    floor (6 events). It says STALE and still draws the 36 rows. Returns the window, the figure's
    id and page, the recording's (doc_id, channel) and the row counts before and after."""
    win, key, (doc_id, ch), n_before = _plotted_figure(qtbot, tmp_path)
    page = win._plots_window
    page.set_plot_type("Histogram")
    page.set_histogram_column("dwell_s")
    page.set_fit("exponential")
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)   # fewer events now
    n_after = _n_events(win, doc_id, ch)
    assert n_after < n_before
    assert _figure_status(win) == "stale" and _chip(win) == "STALE"
    assert _drawn_n(win, "a.csv") == n_before
    return win, key, page, (doc_id, ch), n_before, n_after


_BAD_RANGE = "Fit range min must be a finite number; correct or clear “abc”."


@pytest.mark.parametrize("route", ["update", "refresh"])
def test_an_update_the_page_refuses_over_an_unreadable_fit_range_leaves_the_figure_stale(
        qapp, qtbot, tmp_path, route):
    """The page does not run while a Fit range box holds text that is not a number: it says so
    in the status bar and in Details, takes nothing from the session and goes on drawing the
    table it had, while the new one waits staged. That press must not certify the figure (the
    measured defect: the header said CURRENT over the 36 superseded rows, with the 6-row table
    still staged), by Update plot or by Update plot ▾ ▸ Refresh from recordings. Once the box is
    cleared the same press draws the session's table and the figure says CURRENT."""
    press = _update_plot if route == "update" else _refresh_from_recordings
    win, key, page, (doc_id, ch), n_before, n_after = _re_detected_exponential_figure(
        qtbot, tmp_path)
    page._fit_lo.setText("abc")

    press(win, qtbot)
    assert _BAD_RANGE in page.statusBar().currentMessage()
    assert _BAD_RANGE in page.details_text()
    assert _figure_status(win) == "stale"
    assert win._analysis_explorer_status(key, page) == ("STALE", "stale")
    assert _chip(win) == "STALE", "a refused press certified a figure drawing the superseded rows"
    assert _plot_rows(win, doc_id, ch) == n_before
    assert _drawn_n(win, "a.csv") == n_before

    page._fit_lo.setText("")
    press(win, qtbot)
    assert _plot_rows(win, doc_id, ch) == n_after, "the staged table was never taken"
    assert _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_readable_fit_range_does_not_hold_the_update_back(qapp, qtbot, tmp_path):
    """The refusal is only for a Fit range box that is not a number: with a readable range typed
    Update plot takes the session's table as it always did, and the figure says CURRENT."""
    win, _key, page, (doc_id, ch), _n_before, n_after = _re_detected_exponential_figure(
        qtbot, tmp_path)
    page._fit_lo.setText("0.001")

    _update_plot(win, qtbot)
    assert "must be a finite number" not in page.statusBar().currentMessage()
    assert "Fit not run" not in page.details_text()
    assert _plot_rows(win, doc_id, ch) == n_after
    assert _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


# --- what Update plot does with a table the session no longer has, or has changed in other ways --
#
# Update plot takes the session's tables whenever a table the figure USES is no longer the
# session's: its rows differ, or its detection or calibration do, or the session has no table
# for the recording at all (``MainWindow._stage_changed_distribution_sources``). The tests below
# are the cases of that rule that are not the plain replaced table; each ends where Refresh from
# recordings ends, or keeps what the rule says it keeps.

#: A minimum dwell above every event of the fixture trace (the longest is 3 ms): the detection
#: finds nothing, so the session stores a table with no rows and offers no source for it.
_NO_EVENTS_MIN_DWELL_S = 0.05

#: A minimum dwell that still keeps all 36 events (the shortest is 0.4 ms): another detection
#: of the same recording, with the same rows and another detection limit.
_OTHER_FLOOR_MIN_DWELL_S = 3e-4


def _shown(win: MainWindow) -> dict:
    """What the open Distributions figure says and shows: its header, its Sources list and its
    canvas (the first two Details lines: the rest of the pane is filled in over time). Two
    routes that end in the same state give equal dicts."""
    page = win._plots_window
    message = page._message_item.toPlainText() if page._message_item is not None else None
    return {"status": _figure_status(win), "chip": _chip(win),
            "sources": [(s.label, len(s.df)) for s in page.sources()],
            "ticked": page.selected_source_keys(),
            "plot type": page._plot_type_combo.currentText(),
            "canvas message": message, "details": page.details_text().splitlines()[:2],
            "drawn items": len(page._plot.listDataItems())}


def _after_a_re_detection_that_finds_no_events(qtbot, tmp_path, plotted: bool, route: str) -> dict:
    """What a Distributions figure shows once its only recording was re-detected with a floor
    above every event (so the session has no table for it) and the user pressed Update plot
    (``route`` "update") or Update plot ▾ ▸ Refresh from recordings (``route`` "refresh").
    ``plotted``: whether the figure had drawn the table before, or was never plotted."""
    if plotted:
        win, _key, (doc_id, ch), _n = _plotted_figure(qtbot, tmp_path)
    else:
        win = _fresh_window(qtbot)
        doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
        _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
        win._open_plots()
        qtbot.wait(20)
        assert _figure_status(win) == "empty", "fixture: this leg is about a never-plotted figure"
    _detect(win, qtbot, min_dwell_s=_NO_EVENTS_MIN_DWELL_S)
    assert _n_events(win, doc_id, ch) == 0 and win._plots_sources() == [], (
        "fixture: the session has no table for the recording now")
    (_update_plot if route == "update" else _refresh_from_recordings)(win, qtbot)
    shown = _shown(win)
    win._analysis_controller.abort()
    return shown


@pytest.mark.parametrize("plotted", [True, False], ids=["plotted", "never plotted"])
def test_a_re_detection_that_finds_no_events_is_not_drawn_over_the_old_rows(
        qapp, qtbot, tmp_path, plotted):
    """A recording that is still open and whose detection now has no events is a table the
    session no longer has: Update plot must not draw the superseded rows and say CURRENT over
    them (the measured defect, on a plotted figure and on a figure's first Update). It ends
    where Update plot ▾ ▸ Refresh from recordings ends: NO DATA, nothing left to draw."""
    by_update = _after_a_re_detection_that_finds_no_events(qtbot, tmp_path, plotted, "update")
    by_refresh = _after_a_re_detection_that_finds_no_events(qtbot, tmp_path, plotted, "refresh")
    assert by_update == by_refresh
    assert by_update["status"] == "empty" and by_update["chip"] == "NO DATA"
    assert by_update["sources"] == [] and by_update["ticked"] == []
    assert not any("n = 36" in line for line in by_update["details"]), (
        "the superseded rows are still on the figure")


def _seed_manual_measurements(win: MainWindow, doc_id: str, spans) -> None:
    """Put rows in the manual-measurement list the way a project restore does: with the
    measurement history suspended, then adopted as its starting point."""
    from nanoscope.analysis.measure import coerce_measurements
    from nanoscope.core.segment import persistent_key

    key = persistent_key(win._session.document(doc_id))
    records = [{"measurement_id": i, "recording": "a.csv", "doc_key": key, "channel": "current",
                "channel_index": 0, "start_idx": s, "end_idx": e, "label": f"m{i}", "units": "pA",
                "cal_gain": 1.0, "cal_offset": 0.0} for i, (s, e) in enumerate(spans)]
    history = win._measure_history
    history.set_suspended(True)
    try:
        win._analysis_panel.measure_panel.set_measurements_df(
            coerce_measurements(pd.DataFrame(records)))
        history.checkpoint()
    finally:
        history.set_suspended(False)


def _after_the_manual_measurements_were_removed(qtbot, tmp_path, route: str) -> dict:
    """A figure plotted on the manual measurements alone (the recording's detection is listed,
    unticked), whose measurements are then all removed; what it shows after Update plot or
    Refresh from recordings."""
    from nanoscope.core.sources import MANUAL_SOURCE_KEY

    win = _fresh_window(qtbot)
    doc_id, _ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    _seed_manual_measurements(win, doc_id, [(10_000, 12_000), (30_000, 33_000), (50_000, 52_500)])
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    page.set_sources([MANUAL_SOURCE_KEY])
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and "Manual measurements: n = 3" in page.details_text()
    _seed_manual_measurements(win, doc_id, [])
    assert MANUAL_SOURCE_KEY not in {s.key for s in win._plots_sources()}, (
        "fixture: the session has no measurements left")
    (_update_plot if route == "update" else _refresh_from_recordings)(win, qtbot)
    shown = _shown(win)
    win._analysis_controller.abort()
    return shown


def test_manual_measurements_that_are_gone_are_not_drawn_over_the_old_rows(qapp, qtbot, tmp_path):
    """The manual measurements are the one source with no recording behind it: when the session
    has none left, a figure that used them has no table for them either (they are not a closed
    recording's copy to keep). Update plot ends where Refresh from recordings ends: the
    measurements are gone from the figure and from its Sources list, and what the session still
    has is drawn."""
    from nanoscope.core.sources import MANUAL_SOURCE_KEY

    by_update = _after_the_manual_measurements_were_removed(qtbot, tmp_path, "update")
    by_refresh = _after_the_manual_measurements_were_removed(qtbot, tmp_path, "refresh")
    assert by_update == by_refresh
    assert MANUAL_SOURCE_KEY not in by_update["ticked"]
    assert [label for label, _n in by_update["sources"]] == ["a.csv"]
    assert not any("Manual measurements" in line for line in by_update["details"])


def test_a_closed_recording_keeps_its_rows_while_the_open_one_takes_its_new_table(
        qapp, qtbot, tmp_path):
    """Two recordings are ticked and drawn. One is closed and the other re-detected: Update plot
    draws the new table of the open one and keeps the copy of the closed one it was drawn from,
    ticked, and in its own row (the Sources list does not reorder). Before, one closed recording
    kept the figure from taking ANY table, and it said CURRENT over the superseded one."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n_a = _n_events(win, doc_a, ch_a)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    key_a = next(s.key for s in page.sources() if s.doc_id == doc_a)
    key_b = next(s.key for s in page.sources() if s.doc_id == doc_b)
    page.set_sources([key_a, key_b])
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert (_drawn_n(win, "a.csv"), _drawn_n(win, "b.csv")) == (n_a, n_a)
    copy_of_a = next(s for s in page.sources() if s.key == key_a)

    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)       # B, the selected one, has fewer events
    n_b = _n_events(win, doc_b, ch_b)
    assert n_b < n_a
    win._session.close_document(doc_a)
    qtbot.wait(30)
    assert [s.key for s in win._plots_sources()] == [key_b], "fixture: the session offers only B"
    assert _figure_status(win) == "stale" and _chip(win) == "STALE"

    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert _drawn_n(win, "b.csv") == n_b, "the open recording's new table was not taken"
    assert _drawn_n(win, "a.csv") == n_a, "the closed recording's rows were not kept"
    assert page.selected_source_keys() == [key_a, key_b]
    assert [s.key for s in page.sources()] == [key_a, key_b], "the Sources list changed its order"
    assert next(s for s in page.sources() if s.key == key_a) is copy_of_a
    win._analysis_controller.abort()


def test_a_dormant_figure_whose_recording_was_closed_keeps_its_saved_copy_and_its_pca(
        qapp, qtbot, tmp_path):
    """A figure built from its record while its recording is not open holds a copy of the table
    that is bound to no recording (a restored copy carries no doc_id when its recording cannot
    be found). That copy is all there is of the table: Update plot keeps it, and the PCA computed
    from it, as it does for a recording closed under a resident figure."""
    win, key, (doc_id, _ch), n = _plotted_figure(qtbot, tmp_path)
    owner = win._figure_workspace
    page = win._plots_window
    page.set_feature_selection(["dwell_s", "depth_pA"])
    assert page.run_pca() is not None
    _update_plot(win, qtbot)
    assert owner.suspend(key) and owner.page(key) is None
    win._session.close_document(doc_id)
    qtbot.wait(30)

    assert owner.activate(key)
    qtbot.wait(20)
    page = win._plots_window
    assert page is owner.page(key) and page is not None
    copy = next(iter(page.sources()))
    assert copy.doc_id is None and win._plots_sources() == [], (
        "fixture: the copy is bound to no recording and the session offers nothing")
    pca = page.pca_result()
    assert pca is not None, "fixture: the rebuilt figure carries its PCA"

    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert page.pca_result() is pca, "Update plot threw away the PCA of a table it left alone"
    assert next(iter(page.sources())) is copy
    assert page.selected_source_keys() == [copy.key]
    assert len(copy.df) == n
    win._analysis_controller.abort()


def _after_a_re_detection_at_another_floor(qtbot, tmp_path, route: str) -> dict:
    """An exponential dwell fit is plotted on the 36 events detected at the default floor
    (2e-05 s). The recording is detected again with a floor of 3e-04 s, which still keeps all 36
    events: the same rows, another detection. Returns what the figure says after Update plot
    (``route`` "update") or Refresh from recordings (``route`` "refresh"): its status, its
    Sources, the detection limit and the fitted tau on it, and the same two before."""
    win, _key, (doc_id, ch), _n = _plotted_figure(qtbot, tmp_path)
    page = win._plots_window
    page.set_plot_type("Histogram")
    page.set_histogram_column("dwell_s")
    page.set_fit("exponential")
    _update_plot(win, qtbot)

    def limit_and_tau(text: str) -> tuple:
        return (re.search(r"detection limit t_min = [^\n]+", text).group(0),
                re.search(r"τ = (\S+ \S+)", text).group(1))

    before = limit_and_tau(page.details_text())
    copy = next(s for s in page.sources() if s.doc_id == doc_id)
    _detect(win, qtbot, min_dwell_s=_OTHER_FLOOR_MIN_DWELL_S)
    now = win._session.results.get(doc_id, ch, "detect.schmitt").payload
    assert now.df is not copy.df and fingerprint_frame(now.df) == fingerprint_frame(copy.df), (
        "fixture: the detection at the other floor has the same rows")
    assert now.provenance["min_dwell_s"] != copy.provenance["min_dwell_s"], "fixture: another floor"
    assert _figure_status(win) == "stale"
    (_update_plot if route == "update" else _refresh_from_recordings)(win, qtbot)
    shown = {"status": _figure_status(win), "chip": _chip(win),
             "sources": [(s.label, len(s.df)) for s in page.sources()],
             "before": before, "after": limit_and_tau(page.details_text())}
    win._analysis_controller.abort()
    return shown


def test_a_re_detection_that_finds_the_same_events_under_another_floor_is_taken_at_update(
        qapp, qtbot, tmp_path):
    """The rows of a re-detection can be the very rows the figure holds while the detection is
    another: the minimum dwell is the detection limit the dwell fit is truncated at, and it
    travels with the table as its provenance. Update plot must show the new limit and the refit
    tau, as Refresh from recordings does, not the superseded detection's under CURRENT."""
    by_update = _after_a_re_detection_at_another_floor(qtbot, tmp_path, "update")
    by_refresh = _after_a_re_detection_at_another_floor(qtbot, tmp_path, "refresh")
    assert by_update == by_refresh
    (limit_before, tau_before), (limit_after, tau_after) = by_update["before"], by_update["after"]
    assert limit_before == "detection limit t_min = 2e-05 s (from min_dwell_s)"
    assert limit_after == "detection limit t_min = 0.0003 s (from min_dwell_s)"
    assert tau_after != tau_before, "the fit is still the superseded detection's"
    assert by_update["status"] == "current" and by_update["chip"] == "CURRENT"


def test_a_changed_calibration_on_a_used_recording_is_taken_at_update(qapp, qtbot, tmp_path):
    """The calibration a source was offered under travels with it (it decides whether absolute
    currents can be pooled across the drawn recordings), so a recording calibrated again after
    the figure took its table makes that table another one, though its rows are the same."""
    win, _key, (doc_id, ch), _n = _plotted_figure(qtbot, tmp_path)
    page = win._plots_window
    assert next(s for s in page.sources() if s.doc_id == doc_id).calibration_fingerprint == (1.0, 0.0)

    _regain(win, doc_id, ch, 2.0)
    qtbot.wait(30)
    assert [s.calibration_fingerprint for s in win._plots_sources()] == [(2.0, 0.0)], "fixture"
    _update_plot(win, qtbot)
    assert next(s for s in page.sources() if s.doc_id == doc_id).calibration_fingerprint == (
        2.0, 0.0), "the figure still holds the calibration the recording had"
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


@pytest.mark.parametrize("carries", ["a PCA", "a cluster split"])
def test_a_re_detection_of_a_listed_but_unticked_recording_leaves_a_plotted_figure_what_it_drew(
        qapp, qtbot, tmp_path, carries):
    """The rule keeps Update plot from costing a figure its analysis: it takes the session's
    tables when a table the figure USES changed, not when any table the page HOLDS did. A
    recording that is in the Sources list unticked and is detected again changes nothing the
    figure draws, so its PCA or cluster split, its picture and its inputs are as they were, and
    nothing is staged."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._open_plots()
    qtbot.wait(20)
    key = win._figure_workspace.current_id()
    page = win._plots_window
    key_a = next(s.key for s in page.sources() if s.doc_id == doc_a)
    key_b = next(s.key for s in page.sources() if s.doc_id == doc_b)
    page.set_sources([key_a])                                 # B is listed and not ticked
    page.set_feature_selection(["dwell_s", "depth_pA"])
    assert (page.run_pca() if carries == "a PCA" else page.run_clustering()) is not None
    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    pca, clusters = page.pca_result(), page.cluster_result()
    details, items = page.details_text(), len(page._plot.listDataItems())
    copy_of_b = next(s for s in page.sources() if s.key == key_b)

    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)       # B, the selected one, is detected again
    assert _n_events(win, doc_b, ch_b) != len(copy_of_b.df), "fixture: B's table is another one"
    assert _figure_status(win) == "stale"
    assert win._analysis_explorer_status(key, page) == ("STALE", "stale")

    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert page.pca_result() is pca, "Update plot threw away the PCA of a table that did not change"
    assert page.cluster_result() is clusters, "Update plot threw away the clustering"
    assert page.details_text() == details and len(page._plot.listDataItems()) == items
    assert page._figure_pending_sources is None
    assert next(s for s in page.sources() if s.key == key_b) is copy_of_b, "a catalog was installed"
    win._analysis_controller.abort()


def test_a_staged_catalog_a_refused_update_left_is_replaced_at_the_next_update(
        qapp, qtbot, tmp_path):
    """A refused Update (an unreadable Fit range) leaves the session's table staged and the
    figure drawing the one it had. If the session then goes back to that table (Edit > Undo of
    the re-detection: the rows, the detection and the calibration equal what the figure holds),
    nothing differs any more, and the staged catalog must not be installed by the next Update:
    it holds the table the session has moved away from again, and the figure said CURRENT over
    it. The next Update takes the session's offer afresh."""
    win, _key, page, (doc_id, ch), n_before, n_after = _re_detected_exponential_figure(
        qtbot, tmp_path)
    copy = next(s for s in page.sources() if s.doc_id == doc_id)
    page._fit_lo.setText("abc")
    _update_plot(win, qtbot)
    assert _chip(win) == "STALE" and page._figure_pending_sources is not None, "fixture: refused"

    win._act_undo.trigger()                                   # the session has the old table back
    qtbot.wait(30)
    now = win._session.results.get(doc_id, ch, "detect.schmitt").payload
    assert len(now.df) == n_before and fingerprint_frame(now.df) == fingerprint_frame(copy.df), (
        "fixture: the session is back at the table the figure drew")
    assert (json.dumps(dict(now.provenance), sort_keys=True, default=str)
            == json.dumps(dict(copy.provenance), sort_keys=True, default=str)), (
        "fixture: the detection is the same too")

    page._fit_lo.setText("")
    _update_plot(win, qtbot)
    assert page._figure_pending_sources is None
    assert _plot_rows(win, doc_id, ch) == n_before, "the stale staged table was installed"
    assert _drawn_n(win, "a.csv") == n_before
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_never_plotted_figure_takes_the_new_recording_when_its_ticked_one_was_closed(
        qapp, qtbot, tmp_path):
    """A figure that was never plotted ticked the active recording by itself. When that one is
    closed and another is detected, the first Update plot draws the closed recording from the
    copy the figure holds AND lists the new recording (its Sources are the session's offer, with
    the copy kept in its row); before, the new recording stayed out of the list."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n_a = _n_events(win, doc_a, ch_a)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    key_a = next(s.key for s in page.sources() if s.doc_id == doc_a)
    assert page.selected_source_keys() == [key_a], "fixture: the figure ticked A by itself"
    assert _figure_status(win) == "empty", "fixture: this leg is about a never-plotted figure"

    win._session.close_document(doc_a)
    qtbot.wait(30)
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    key_b = next(s.key for s in win._plots_sources() if s.doc_id == doc_b)
    assert [s.key for s in win._plots_sources()] == [key_b], "fixture: the session offers only B"

    _update_plot(win, qtbot)
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    assert _drawn_n(win, "a.csv") == n_a
    assert page.selected_source_keys() == [key_a]
    assert [s.key for s in page.sources()] == [key_a, key_b]
    win._analysis_controller.abort()


def test_update_plot_builds_the_session_catalog_once_whatever_it_stages(
        qapp, qtbot, tmp_path, monkeypatch):
    """One press, one catalog: the step builds the session's catalog to judge it and hands that
    same list to the page when it stages it (with the copies of the closed recordings in it); it
    does not build it again to install it. Counted when nothing changed, when a table was
    replaced and when a closed recording's copy is kept."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._open_plots()
    qtbot.wait(20)
    page = win._plots_window
    page.set_sources([s.key for s in page.sources()])
    builds = []
    real = win._plots_sources
    monkeypatch.setattr(win, "_plots_sources", lambda: (builds.append(1), real())[1])

    def presses() -> int:
        builds.clear()
        _update_plot(win, qtbot)
        return len(builds)

    assert presses() == 1, "nothing changed"
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)        # B's table is replaced
    assert _figure_status(win) == "stale"
    assert presses() == 1, "a replaced table"
    win._session.close_document(doc_a)
    qtbot.wait(30)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)      # B's table is replaced again
    assert presses() == 1, "a replaced table beside the copy of a closed recording"
    win._analysis_controller.abort()


def test_a_failure_of_the_stage_step_is_said_and_the_figure_is_not_updated(
        qapp, qtbot, tmp_path, monkeypatch):
    """The step that takes the session's tables runs before the update and reads the session
    and hashes frames. If it raises, the press says so in the status bar and ends there: the
    figure keeps what it drew and says STALE, never CURRENT over it. The press after the cause
    is gone works."""
    win, _key, (doc_id, ch), n_before = _plotted_figure(qtbot, tmp_path)
    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)
    n_after = _n_events(win, doc_id, ch)
    assert _figure_status(win) == "stale"

    def broken():
        raise RuntimeError("the source catalog could not be built")
    with monkeypatch.context() as patched:
        patched.setattr(win, "_plots_sources", broken)
        _update_plot(win, qtbot)
    assert win.statusBar().currentMessage() == "the source catalog could not be built"
    assert _figure_status(win) == "stale" and _chip(win) == "STALE"
    assert _plot_rows(win, doc_id, ch) == n_before and _drawn_n(win, "a.csv") == n_before

    _update_plot(win, qtbot)
    assert _plot_rows(win, doc_id, ch) == n_after and _drawn_n(win, "a.csv") == n_after
    assert _figure_status(win) == "current" and _chip(win) == "CURRENT"
    win._analysis_controller.abort()


def test_a_reopened_saved_result_whose_detection_stamp_holds_tuples_and_numpy_floats_keeps_its_pca(
        qapp, qtbot, tmp_path):
    """A figure saved in a project and the session that reopens it have each been through a
    different codec, so the same detection provenance comes back spelled differently: a tuple
    on the figure's side and a list on the session's, a numpy float and a float, a NaN that is
    not equal to itself. Update plot compares the two in one canonical spelling, so a reopened
    figure whose detection did not change is not staged and keeps its PCA (it does so under a
    calibration that is not the identity, which is the other half of the comparison)."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(csv, arr))
    _regain(win, doc_id, ch, 2.0)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    stamp = win._session.results.get(doc_id, ch, "detect.schmitt").payload.provenance
    stamp["regions_s"] = [(0.5, 0.75)]
    stamp["noise_pA"] = np.float32(1.1)
    stamp["unmeasured_pA"] = float("nan")
    win._open_plots()
    qtbot.wait(20)
    win._plots_window.set_feature_selection(["dwell_s", "depth_pA"])
    assert win._plots_window.run_pca() is not None
    _update_plot(win, qtbot)
    proj = tmp_path / "figure.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = _reopen(qtbot, str(proj))
    w2._open_plots()
    qtbot.wait(20)
    page = w2._plots_window
    held = next(iter(page.sources()))
    offered = next(iter(w2._plots_sources()))
    assert isinstance(held.provenance["regions_s"][0], tuple) and isinstance(
        offered.provenance["regions_s"][0], list), "fixture: the two codecs spell the stamp differently"
    assert held.calibration_fingerprint == offered.calibration_fingerprint == (2.0, 0.0)
    pca = page.pca_result()
    assert pca is not None and _figure_status(w2) == "frozen"

    _update_plot(w2, qtbot)
    assert _figure_status(w2) == "current" and _chip(w2) == "CURRENT"
    assert page.pca_result() is pca, (
        "Update plot threw away the PCA of a reopened figure whose detection did not change")
    assert next(iter(page.sources())) is held, "a catalog was installed"
    w2._analysis_controller.abort()


# ================================================================================================
# C8 — an x on screen is the document's axis PLUS the layer's alignment offset
# ================================================================================================

_OFFSET_S = 0.5


def _align(win: MainWindow, doc_id: str, ch: int, dt_s: float = _OFFSET_S) -> None:
    win._session.set_layer_time_offset(f"{doc_id}#{ch}", dt_s)


def _frame_event(win: MainWindow, doc_id: str, ch: int, i0: int, i1: int, qtbot,
                 pad_samples: int = 8_000) -> None:
    """Zoom to a window around samples [i0, i1) IN THE COORDINATES THE USER SEES.

    The pad is generous on purpose: the ML detector's input window is 3072 samples, so a
    range narrower than that is not a question it can answer, and framing one would test the
    wrong thing. 16k samples is still a sixth of the recording."""
    x0 = win._trace_view.x_for_index(doc_id, ch, i0 - pad_samples)
    x1 = win._trace_view.x_for_index(doc_id, ch, i1 + pad_samples)
    win._trace_view.zoom_x_to(x0, x1)
    qtbot.wait(10)


def test_the_visible_sample_range_of_an_aligned_layer_is_what_the_user_framed(
        qapp, qtbot, tmp_path):
    """REGRESSION: the view range is in SCREEN x; searching the document's NATIVE axis with it
    put both searchsorted results at the same end of the array, and the degenerate
    ``i1 <= i0`` fallback quietly returned THE WHOLE RECORDING."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _align(win, doc_id, ch)
    s, e = _LONG[3]
    _frame_event(win, doc_id, ch, s, e, qtbot)

    i0, i1 = win._analysis_panel._visible_sample_range(doc_id, ch, N)
    assert (i0, i1) != (0, N), "the whole recording is not what the user framed"
    assert i0 < s and i1 > e                      # ...the framed event is inside...
    assert (i1 - i0) < N // 4                     # ...and it really is a small window
    win._analysis_controller.abort()


def test_an_ml_run_over_the_visible_range_of_an_aligned_layer_stays_in_that_range(
        qapp, qtbot, tmp_path, monkeypatch):
    """The user-visible half: 'Detect over the visible range' is the DEFAULT ML mode, and on
    an aligned layer it ran the model over every sample in the file."""
    _use_fake_model(monkeypatch)
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    win._analysis_panel._detection.set_model_path("/fake/model.keras")
    _align(win, doc_id, ch)
    s, e = _LONG[3]
    _frame_event(win, doc_id, ch, s, e, qtbot)

    win._analysis_panel._detection.ml_range.setCurrentIndex(0)   # "Visible view"
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._run_ml_detection()
    qtbot.wait(20)

    res = win._session.results.get(doc_id, ch, "detect.keras")
    assert res is not None
    assert (res.params["range_start"], res.params["range_stop"]) != (0, N)
    assert res.params["range_start"] < s and res.params["range_stop"] > e
    win._analysis_controller.abort()


def test_selecting_an_event_on_an_aligned_layer_zooms_to_that_event(qapp, qtbot, tmp_path):
    """REGRESSION: the event-table click zoomed to the event's NATIVE-axis x, which on an
    aligned layer frames a stretch of trace ``time_offset_s`` away from the event."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot)
    _align(win, doc_id, ch)

    df = win._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    row = df.iloc[3]
    event_id, start_idx = int(row["event_id"]), int(row["start_idx"])
    win._analysis_panel.select_event(event_id)
    qtbot.wait(10)

    x_event = win._trace_view.x_for_index(doc_id, ch, start_idx)
    xmin, xmax = win._trace_view.anchor_plot().vb.viewRange()[0]
    assert xmin <= x_event <= xmax, "the view does not contain the event it zoomed to"
    win._analysis_controller.abort()


def test_the_selected_event_marker_lands_on_the_event_on_an_aligned_layer(
        qapp, qtbot, tmp_path):
    """The overlay half of the same mistake: index -> x inside the renderers used the native
    axis too, so the highlight sat ``time_offset_s`` from the event it marks."""
    import pyqtgraph as pg

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot)
    _align(win, doc_id, ch)

    df = win._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    row = df.iloc[3]
    win._analysis_panel.select_event(int(row["event_id"]))
    qtbot.wait(10)

    pane = win._trace_view.pane_for_layer(f"{doc_id}#{ch}")
    regions = [it for it in pane.plot.items if isinstance(it, pg.LinearRegionItem)]
    assert regions, "no marker drawn"
    lo, hi = regions[-1].getRegion()
    want_lo = win._trace_view.x_for_index(doc_id, ch, int(row["start_idx"]))
    want_hi = win._trace_view.x_for_index(doc_id, ch, int(row["end_idx"]))
    assert lo == pytest.approx(want_lo, abs=1e-6)
    assert hi == pytest.approx(want_hi, abs=1e-6)
    win._analysis_controller.abort()


def test_the_event_spans_land_on_the_events_on_an_aligned_layer(qapp, qtbot, tmp_path):
    """...and so did the batched event spans: shaded blockades over quiet baseline."""
    from nanoscope.gui.overlays import SpanOverlayItem

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot)
    _align(win, doc_id, ch)
    win._analysis_panel._redraw_spans_from_df()
    qtbot.wait(10)

    pane = win._trace_view.pane_for_layer(f"{doc_id}#{ch}")
    spans = [it for it in pane.plot.items
             if isinstance(it, SpanOverlayItem) and it.label == "events"]
    assert spans
    df = win._session.results.get(doc_id, ch, "detect.schmitt").payload.df
    first = int(df["start_idx"].to_numpy()[df["accepted"].to_numpy(dtype=bool)][0])
    assert spans[0].x_bounds()[0] == pytest.approx(
        win._trace_view.x_for_index(doc_id, ch, first), abs=1e-6)
    win._analysis_controller.abort()


def test_an_unaligned_layer_maps_exactly_as_before(qapp, qtbot, tmp_path):
    """Control: with no alignment nudge (the overwhelmingly common case) every mapping above
    is the document's native axis, unchanged."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    s, e = _LONG[3]
    _frame_event(win, doc_id, ch, s, e, qtbot)

    native = win._session.x_axis(doc_id).values
    assert win._trace_view.x_for_index(doc_id, ch, s) == pytest.approx(float(native[s]))
    i0, i1 = win._analysis_panel._visible_sample_range(doc_id, ch, N)
    assert i0 < s and i1 > e and (i1 - i0) < N // 4
    win._analysis_controller.abort()


# ================================================================================================
# C7 — eta(t) and the FDR belong to the recording they were measured on
# ================================================================================================

def _eta_frame(scale: float) -> pd.DataFrame:
    """A recovery_map-shaped efficiency table: eta rises with dwell. ``scale`` moves the
    knee, so two frames give measurably different fits."""
    t = np.geomspace(2e-5, 5e-3, 24)
    eta = np.clip(1.0 - np.exp(-t / scale), 0.02, 1.0)
    return pd.DataFrame({"dwell_s": t, "efficiency": eta})


def _wait_for_fit(qtbot, stats, timeout_ms: int = 60_000) -> None:
    """Wait for the Inspector's fit worker on its real completion condition.

    2026-11-11: ``fit_test_helpers.wait_for_fit`` polls with ``QTest.qWait``, which keeps the GIL
    while it waits on this PySide6 (6.11.1). The dwell fit with a measured eta(t) is a bootstrap
    of many small numpy calls on a worker thread, and under that polling a fit that takes 0.9 s
    here (``qtbot.wait``) had not finished in the helper's 60 s -- on the untouched baseline too.
    ``qtbot.waitUntil`` waits inside the event loop and lets the worker run."""
    qtbot.waitUntil(lambda: not getattr(stats, "_fit_busy", False), timeout=timeout_ms)


def _deliver_surrogate(panel, kind: str, target, payload) -> None:
    """Feed a surrogate-study answer through the REAL controller sink, exactly as the
    off-thread job does — including which target it was launched on."""
    panel._pending, panel._pending_sel = kind, target
    panel._on_controller_finished(payload)


def _two_recordings(qtbot, tmp_path) -> tuple[MainWindow, tuple, tuple]:
    win = _fresh_window(qtbot)
    a = _install(win, _doc(tmp_path / "a.csv", _trace(seed=1)))
    b = _install(win, _doc(tmp_path / "b.csv", _trace(seed=2)))
    return win, a, b


def test_a_dwell_fit_uses_this_recordings_recovery_map_not_the_previous_ones(
        qapp, qtbot, tmp_path):
    """REGRESSION: ``_recovery_df`` was one panel-global slot. Measure eta(t) on A, select B,
    fit — and B's dwell distribution was divided by A's efficiency curve, with the fit
    PRINTING "measured eta(t) on [...]" as though it were B's own."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    panel = win._analysis_panel

    # The default floor, so both recordings carry enough events for the fitter's n>=20 gate.
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    _deliver_surrogate(panel, "recovery", (doc_a, ch_a), _eta_frame(3e-4))

    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    panel._on_fit_requested({"n_components_dwell": 1, "n_components_depth": 1,
                             "use_efficiency": True, "use_filter_limited": False})
    _wait_for_fit(qtbot, panel._stats)

    summary = panel._stats.results.toPlainText()
    assert "measured eta(t)" not in summary, (
        "B's dwell fit reported an efficiency curve measured on A")
    win._analysis_controller.abort()


def test_a_dwell_fit_still_uses_the_recovery_map_measured_on_its_own_recording(
        qapp, qtbot, tmp_path):
    """Control: the recording that DID measure eta(t) must still get it."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    panel = win._analysis_panel
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    _deliver_surrogate(panel, "recovery", (doc_id, ch), _eta_frame(3e-4))
    assert panel._stats.use_efficiency.isEnabled()

    panel._on_fit_requested({"n_components_dwell": 1, "n_components_depth": 1,
                             "use_efficiency": True, "use_filter_limited": False})
    _wait_for_fit(qtbot, panel._stats)
    assert "measured eta(t)" in panel._stats.results.toPlainText()
    win._analysis_controller.abort()


def test_the_efficiency_toggle_is_gated_per_recording(qapp, qtbot, tmp_path):
    """The toggle is the visible half: it stayed live on B, offering A's curve."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    panel = win._analysis_panel

    win._on_selection(doc_a, ch_a)
    _deliver_surrogate(panel, "recovery", (doc_a, ch_a), _eta_frame(3e-4))
    assert panel._stats.use_efficiency.isEnabled()

    win._on_selection(doc_b, ch_b)
    assert not panel._stats.use_efficiency.isEnabled()

    win._on_selection(doc_a, ch_a)
    assert panel._stats.use_efficiency.isEnabled(), "A's own recovery map went missing"
    win._analysis_controller.abort()


def test_the_fdr_readout_does_not_follow_you_to_another_recording(qapp, qtbot, tmp_path):
    """REGRESSION: the false-positive rate measured on A kept sitting under B's name."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    panel = win._analysis_panel

    win._on_selection(doc_a, ch_a)
    _deliver_surrogate(panel, "null", (doc_a, ch_a),
                       {"estimated_fdr": 0.037, "n_surrogates": 20})
    assert "3.7%" in panel._stats.fdr_label.text()

    win._on_selection(doc_b, ch_b)
    assert "not measured" in panel._stats.fdr_label.text()

    win._on_selection(doc_a, ch_a)
    assert "3.7%" in panel._stats.fdr_label.text(), "A's own FDR went missing"
    win._analysis_controller.abort()


def test_a_surrogate_answer_lands_on_the_recording_it_was_launched_on(qapp, qtbot, tmp_path):
    """A surrogate study runs off-thread for seconds and the user is free to move on: the
    answer belongs to the recording it was STARTED on, not the one selected when it returns."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    panel = win._analysis_panel

    win._on_selection(doc_a, ch_a)
    panel._pending, panel._pending_sel = "null", (doc_a, ch_a)
    win._on_selection(doc_b, ch_b)                 # ...the user switches while it runs
    panel._on_controller_finished({"estimated_fdr": 0.11, "n_surrogates": 20})
    qtbot.wait(10)

    assert "not measured" in panel._stats.fdr_label.text(), "A's answer was shown under B"
    win._on_selection(doc_a, ch_a)
    assert "11.0%" in panel._stats.fdr_label.text()
    win._analysis_controller.abort()


def test_a_surrogate_answer_for_a_closed_recording_is_dropped(qapp, qtbot, tmp_path):
    """Same rule the detection sink already applies: an answer that lands after its recording
    was closed is an answer about nothing, and a file's doc_id is its path — so keeping it
    would present it as a fresh measurement the next time that file is opened."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    panel = win._analysis_panel

    panel._pending, panel._pending_sel = "recovery", (doc_id, ch)
    win._session.close_document(doc_id)             # ...closed while the study ran
    panel._on_controller_finished(_eta_frame(3e-4))
    qtbot.wait(10)

    _install(win, _doc(tmp_path / "a.csv", _trace()))   # the same path, opened again
    assert not panel._stats.use_efficiency.isEnabled()
    win._analysis_controller.abort()


def test_closing_a_recording_drops_its_surrogate_state(qapp, qtbot, tmp_path):
    """A file's doc_id is its path, so a kept entry would re-attach to the same file on its
    next open — presenting a stale eta(t) as a fresh measurement."""
    win = _fresh_window(qtbot)
    doc = _doc(tmp_path / "a.csv", _trace())
    doc_id, ch = _install(win, doc)
    panel = win._analysis_panel
    _deliver_surrogate(panel, "recovery", (doc_id, ch), _eta_frame(3e-4))
    _deliver_surrogate(panel, "null", (doc_id, ch), {"estimated_fdr": 0.05, "n_surrogates": 20})

    win._session.close_document(doc_id)
    qtbot.wait(10)
    _install(win, _doc(tmp_path / "a.csv", _trace()))   # the same path, opened again
    qtbot.wait(10)

    assert not panel._stats.use_efficiency.isEnabled()
    assert "not measured" in panel._stats.fdr_label.text()
    win._analysis_controller.abort()
