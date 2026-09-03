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
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import live_doc_id
from nanoscope.analysis import ml_detect
from nanoscope.core.analysis import AnalysisResult
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


def test_a_re_detection_reaches_the_open_plots_window(qapp, qtbot, tmp_path):
    """REGRESSION: ``resultsChanged`` was wired to four refresh slots but not the Plots
    window's. Curation appeared to flow through only because the panel mutates the SHARED
    frame in place; a re-detection REPLACES it, and the window went on showing the old rows —
    beside a PNG export button that would write a figure of data that no longer exists."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    n_before = _n_events(win, doc_id, ch)

    win._open_plots()
    qtbot.wait(20)
    assert _plot_rows(win, doc_id, ch) == n_before

    _detect(win, qtbot, min_dwell_s=_TUNED_MIN_DWELL_S)   # fewer events now
    n_after = _n_events(win, doc_id, ch)
    assert n_after < n_before
    assert _plot_rows(win, doc_id, ch) == n_after, (
        "the Plots window kept the superseded detection's rows")
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


def test_a_merge_reaches_the_open_plots_window(qapp, qtbot, tmp_path):
    """The same hole, through the other frame-REPLACING edit: a sub-event merge."""
    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(tmp_path / "a.csv", _shattered_trace()))
    _detect(win, qtbot, min_dwell_s=_DEFAULT_MIN_DWELL_S)
    win._open_plots()
    qtbot.wait(20)
    assert _plot_rows(win, doc_id, ch) == _n_events(win, doc_id, ch)

    # 'Merge all' now runs on the analysis controller (it costs 3.3-8.7 s on the owner's
    # real recordings), so the click returns "started" and the outcome arrives on
    # ``mergeFinished``. 1 ms joins the shattered pair (0.1 ms apart) and nothing else --
    # the `_SHORT` spikes are 7 ms apart, and a window wide enough for THEM is refused by
    # the occupancy floor rather than reporting one 20 ms mostly-open-pore event.
    with qtbot.waitSignal(win._analysis_panel.mergeFinished, timeout=30_000) as landed:
        started = win._analysis_panel.merge_all_subevents(0.001)
    assert started.status == "started"
    qtbot.wait(20)
    report = landed.args[0]
    assert report.status == "ok" and report.n_after < report.n_before
    assert _plot_rows(win, doc_id, ch) == report.n_after
    win._analysis_controller.abort()


def test_a_detection_on_another_recording_appears_in_the_open_plots_window(
        qapp, qtbot, tmp_path):
    """And through the addition case: a second recording's detection is a new source to
    overlay, which is half the reason the window takes a list of them."""
    win, (doc_a, ch_a), (doc_b, ch_b) = _two_recordings(qtbot, tmp_path)
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot)
    win._open_plots()
    qtbot.wait(20)
    assert not any(s.doc_id == doc_b for s in win._plots_window.sources())

    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot)
    qtbot.wait(20)
    assert any(s.doc_id == doc_b for s in win._plots_window.sources()), (
        "the open Plots window never heard about the second detection")
    win._analysis_controller.abort()


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
    qtbot.wait(10)

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
    qtbot.wait(10)
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
