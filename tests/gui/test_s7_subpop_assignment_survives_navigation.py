"""Sweep 7 — the subpopulation ASSIGNMENT has a life keyed to its detection, not to where
the user is looking (skeptic item "Merely LOOKING at another open recording permanently
retires the subpopulation assignment", MAJOR; S7-POP3, major).

Measured on the pre-fix tree, on the owner's own project (docs/sweep7/scratch/SKEPTIC-POP/
k2_nav_drops_subpops.py): focus 0006, run Event subpopulations (assignment 236, gallery
combo three entries, both report buttons live); click 0005's row in the File dock; click
straight back to 0006 — the SAME frame object comes back (k3_frame_identity: id
13820809488 before and after) and the assignment is None, the gallery combo is empty,
the report's two buttons are still live, and "Colour the trace" then says "That report
fitted no mixture" under a header reading "2 subpopulation(s)". The map was dropped by
``_refresh_subpop_views`` on the mismatch and its frame anchor nulled with it, so the
method's own "same frame: re-offer it" branch could never fire again.

Now the map is HELD under the ResultStore key it was adopted on, withdrawn from the
surfaces while another detection is on screen, re-offered when its frame comes back, and
RETIRED — with the true reason on the status bar and in the report window — only when
that detection's stored frame is replaced (a merge, a re-detect, an ML run, an undo, a
project open) or its recording closes. The report window gates its two buttons on that
state, never on ``report.quantification``.

Driven through real widgets: the File dock's rows (``focus_channel`` selects the tree row
and emits ``itemSelectionChanged`` exactly as a click does), the Event Inspector's own
"Merge with next", the real gallery, the real report window.
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
from PySide6 import QtCore

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import detection_signal, register_all
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import subpop_window as SW
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N_SAMPLES = 420_000
N_EVENTS = 100
_SPACING = 4_000

_NO_MIXTURE = "fitted no mixture"


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """This module's own QSettings namespace — nothing here reads or leaves app state
    (module-scoped for the same measured reason as test_subpop_galleries)."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("S7SubpopNavPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


# ================================================================================================
# builders — two CSV-backed recordings, so a project round-trip can restore them
# ================================================================================================

def _trace(seed: int) -> np.ndarray:
    """Events of different dwells and depths, so the mixture stage has something to fit."""
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 100.0) + rng.normal(0.0, 1.5, N_SAMPLES)
    ends = {}
    for i in range(N_EVENTS):
        s = 2_000 + i * _SPACING
        length = int(rng.integers(60, 600))
        x[s:s + length] -= float(rng.uniform(15.0, 45.0))
        ends[i] = s + length
    # One CLOSE pair, so the Event Inspector's "Merge with next" has a stitch the merge
    # accepts: a 3,400-sample gap between two short events is refused as low occupancy
    # (measured: MergeReport(status='low_occupancy') on the evenly spaced grid alone).
    s = ends[40] + 120
    x[s:s + 400] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    d = tmp_path_factory.mktemp("s7_subpop_nav")
    a, b = d / "a.csv", d / "b.csv"
    return (a, _trace(3)), (b, _trace(11))


def _detect_through_the_panel(w, qtbot, doc_id: str) -> None:
    w._on_selection(doc_id, 0)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert w._session.results.get(doc_id, 0, "detect.schmitt") is not None


@pytest.fixture
def two(qapp, qtbot, recordings):
    """A window with REAL detections on two recordings, A selected. No report yet."""
    (path_a, arr_a), (path_b, arr_b) = recordings
    w = MainWindow()
    qtbot.addWidget(w)
    a = w._session.add_document(_doc(path_a, arr_a))
    w._create_default_layers(w._session.document(a))
    b = w._session.add_document(_doc(path_b, arr_b))
    w._create_default_layers(w._session.document(b))
    _detect_through_the_panel(w, qtbot, b)
    _detect_through_the_panel(w, qtbot, a)
    yield w, a, b
    w._analysis_controller.abort()
    w._controller.abort()


class _Cheap(SR.SubpopConfig):
    """The same pipeline over a small grid — see ``test_subpop_gui._Cheap``."""

    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2, 3))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        super().__init__(**kw)


_REPORTS: dict[tuple, object] = {}


def _report_for(w, doc_id: str, events: EventSet | None = None):
    """The report the worker would have produced for ``doc_id``'s stored detection —
    built by the same function, cached per (recording, event-id set) because the
    detection is deterministic and the mixture search is the slow part."""
    res = w._session.results.get(doc_id, 0, "detect.schmitt")
    events = res.payload if events is None else events
    raw = np.asarray(w._session.calibrated(doc_id, 0).data)
    prov = dict(res.payload.provenance or {})
    signal = detection_signal(raw, FS, prov)
    key = (str(w._session.document(doc_id).meta.path),
           tuple(int(e) for e in events.accepted.df["event_id"]))
    report = _REPORTS.get(key)
    if report is None:
        report = SR.run_subpopulations(
            signal, raw, FS, events.accepted, recording=Path(key[0]).name,
            duration_s=N_SAMPLES / FS,
            config=_Cheap(polarity=SR.polarity_for_features(prov.get("polarity"))))
        _REPORTS[key] = report
    return res, report, signal


def _adopt(w, qtbot, doc_id: str, events: EventSet | None = None):
    """Hand a report to the REAL landing slot, launch target and all."""
    res, report, signal = _report_for(w, doc_id, events)
    w._subpop_signal_pending = signal
    w._subpop_target = res
    w._on_subpop_finished(report)
    qtbot.wait(20)
    assert w._subpop_window is not None
    return report


def _open_gallery(w, qtbot):
    w._act_event_gallery.trigger()
    qtbot.wait(30)
    assert w._event_gallery is not None
    return w._event_gallery


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _focus(w, qtbot, doc_id: str) -> None:
    """The File dock's own row, as a click reaches it."""
    assert w._file_dock.focus_channel(doc_id, 0), "the dock has no row for this channel"
    qtbot.wait(60)
    assert w._analysis_panel.current_detection().doc_id == doc_id


# ================================================================================================
# navigation holds the map; coming back re-installs it
# ================================================================================================

def test_looking_at_another_recording_holds_the_map_and_coming_back_reinstalls_it(two, qtbot):
    w, a, b = two
    _adopt(w, qtbot, a)
    assign = dict(w._subpop_assignment or {})
    assert assign, "the fixture produced no assignment to hold"
    gal = _open_gallery(w, qtbot)
    assert gal.subpopulations() == assign
    sw = w._subpop_window
    assert sw.can_show_events()

    # STEP 1 — click the other recording's row. The map is HELD: off the surfaces, kept here.
    _focus(w, qtbot, b)
    assert w._subpop_assignment == assign, "clicking a file row threw the mixture away"
    assert w._analysis_panel.subpop_labels() is None, "B's spans coloured by A's map"
    assert gal.subpopulations() is None, "B's tiles chipped by A's map"
    state, reason = sw.assignment_state()
    assert state == SW.ASSIGNMENT_AWAY
    assert "a.csv" in reason and "not the detection on screen" in reason, reason
    assert "nothing was retired" in reason
    assert not sw.can_show_events()
    assert reason in sw.live_note()

    # STEP 2 — click straight back. Same frame object -> same map, back on every surface.
    _focus(w, qtbot, a)
    assert _frame(w, a) is w._subpop_frame
    assert w._subpop_assignment == assign
    assert w._analysis_panel.subpop_labels() == assign
    assert gal.subpopulations() == assign
    assert [gal.subpop_box.itemText(i) for i in range(gal.subpop_box.count())][0].startswith(
        "All events")
    assert sw.assignment_state() == (SW.ASSIGNMENT_INSTALLED, "")
    assert sw.can_show_events()
    assert sw.live_note() == ""

    # STEP 3 — the report's button, on the recording it was fitted on: it colours.
    sw._trace_btn.click()
    qtbot.wait(20)
    shown = w.statusBar().currentMessage()
    assert shown.startswith("Trace spans coloured by subpopulation for "), shown
    assert _NO_MIXTURE not in shown


def test_while_away_the_two_entry_points_say_where_the_map_is_and_open_nothing(two, qtbot):
    w, a, b = two
    _adopt(w, qtbot, a)
    _focus(w, qtbot, b)
    assert w._event_gallery is None

    w._colour_trace_by_subpops()
    shown = w.statusBar().currentMessage()
    assert "not the detection on screen" in shown and "a.csv" in shown, shown
    assert _NO_MIXTURE not in shown
    assert w._analysis_panel.subpop_labels() is None, "A's map was pushed onto B's spans"

    w.statusBar().clearMessage()
    w._open_gallery_on_subpops()
    qtbot.wait(20)
    assert w._event_gallery is None, "a plain gallery opened from a button promising classes"
    assert "not the detection on screen" in w.statusBar().currentMessage()


def test_a_gallery_opened_while_away_shows_no_classes_and_gets_them_on_return(two, qtbot):
    w, a, b = two
    _adopt(w, qtbot, a)
    assign = dict(w._subpop_assignment)
    _focus(w, qtbot, b)
    gal = _open_gallery(w, qtbot)
    assert gal.subpopulations() is None
    assert not gal._subpop_bar.isVisible()
    _focus(w, qtbot, a)
    assert gal.subpopulations() == assign
    assert gal.frame() is w._subpop_frame


# ================================================================================================
# retirement: only when THAT detection is replaced, and then with the true reason (S7-POP3)
# ================================================================================================

def _assert_retired_with_the_true_reason(w, qtbot, *, expect: str) -> None:
    sw = w._subpop_window
    assert not sw.can_show_events(), "the buttons outlived the assignment"
    assert w._subpop_assignment is None
    state, reason = sw.assignment_state()
    assert state == SW.ASSIGNMENT_RETIRED
    assert expect in reason and "re-run Analysis" in reason, reason
    assert reason in sw.live_note()
    assert _NO_MIXTURE not in sw.live_note()

    # The button the report offers, reached directly (the button itself is dead):
    # the TRUE reason, never "fitted no mixture" under a header saying otherwise.
    w._colour_trace_by_subpops()
    shown = w.statusBar().currentMessage()
    assert expect in shown, shown
    assert _NO_MIXTURE not in shown, shown
    assert "subpopulation(s)" in sw.header()

    w.statusBar().clearMessage()
    had_gallery = w._event_gallery is not None
    w._open_gallery_on_subpops()
    qtbot.wait(20)
    assert (w._event_gallery is not None) == had_gallery, "a plain gallery was opened"
    assert expect in w.statusBar().currentMessage()


def test_replacing_the_detection_under_the_map_retires_it_naming_the_change(two, qtbot):
    w, a, _b = two
    _adopt(w, qtbot, a)
    sw = w._subpop_window
    assert sw.can_show_events()
    df = _frame(w, a)
    fresh = EventSet(df.copy(deep=True), dict(_frame_prov(w, a)))
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=a, channel=0, params={},
        calibration_fingerprint=w._session.calibration(a, 0).fingerprint(), payload=fresh))
    qtbot.wait(60)
    # ...and the status bar said so at the moment it happened.
    assert "retired when the detection changed" in w.statusBar().currentMessage()
    _assert_retired_with_the_true_reason(w, qtbot, expect="retired when the detection changed")


def _frame_prov(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.provenance or {}


def test_a_real_merge_in_the_inspector_retires_it_the_same_way(two, qtbot):
    """The finding's gesture (S7-POP3): "Merge with next" in the Event Inspector."""
    w, a, _b = two
    _adopt(w, qtbot, a)
    df0 = _frame(w, a)
    n0 = len(df0)
    gaps = (df0["start_idx"].to_numpy()[1:] - df0["end_idx"].to_numpy()[:-1]) / FS
    k = int(np.argmin(gaps))
    w._analysis_panel._merge.window_ms.setValue(float(gaps[k]) * 1e3 * 1.2)
    w._open_event_inspector()
    qtbot.wait(40)
    insp = w._event_inspector
    insp.show_event(int(df0.iloc[k]["event_id"]))
    qtbot.wait(20)
    insp._on_merge_with_next()
    qtbot.wait(100)
    assert len(_frame(w, a)) == n0 - 1, "the merge did not happen; nothing below is about it"
    _assert_retired_with_the_true_reason(w, qtbot, expect="retired when the detection changed")


def test_closing_the_recording_retires_it_naming_the_close(two, qtbot):
    w, a, _b = two
    _adopt(w, qtbot, a)
    w._session.close_document(a)
    qtbot.wait(60)
    assert w._subpop_assignment is None
    state, reason = w._subpop_window.assignment_state()
    assert state == SW.ASSIGNMENT_RETIRED
    assert "was closed" in reason and "re-run Analysis" in reason, reason
    assert not w._subpop_window.can_show_events()
    w._colour_trace_by_subpops()
    assert "was closed" in w.statusBar().currentMessage()
    assert _NO_MIXTURE not in w.statusBar().currentMessage()


def test_opening_a_project_retires_it_and_a_restored_frame_is_not_re_adopted_by_id(
        two, qtbot, tmp_path):
    """A project open replaces every frame; the same event ids in the restored table
    name events of a different load, so the map must not come back by id."""
    w, a, _b = two
    (path_a, _arr) = None, None
    src = str(w._session.document(a).meta.path)
    _adopt(w, qtbot, a)
    proj = tmp_path / "held.nanoscope"
    w._write_project(str(proj))
    w._mark_clean()
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(100)
    assert w._subpop_assignment is None
    assert w._subpop_window.assignment_state()[0] == SW.ASSIGNMENT_RETIRED
    live_a = w._session.doc_id_for_source(str(Path(src).resolve()))
    assert live_a is not None
    w._on_selection(live_a, 0)
    qtbot.wait(60)
    assert w._session.results.get(live_a, 0, "detect.schmitt") is not None
    assert w._subpop_assignment is None, "re-attached by event id onto a different load"
    assert w._analysis_panel.subpop_labels() is None
    assert not w._subpop_window.can_show_events()


# ================================================================================================
# the one case the old sentence was true for still says it
# ================================================================================================

def test_a_report_that_fitted_no_mixture_still_says_so(two, qtbot):
    w, a, _b = two
    res = w._session.results.get(a, 0, "detect.schmitt")
    df = res.payload.df
    few = EventSet(df.iloc[:30].copy(deep=True), dict(res.payload.provenance or {}))
    report = _adopt(w, qtbot, a, events=few)
    assert report.quantification is None, "30 events fitted a mixture; the case is not built"
    sw = w._subpop_window
    assert w._subpop_assignment is None
    state, reason = sw.assignment_state()
    assert state == SW.ASSIGNMENT_NONE
    assert _NO_MIXTURE in reason and reason in sw.live_note()
    assert "retired" not in reason
    assert not sw.can_show_events()
    w._colour_trace_by_subpops()
    assert _NO_MIXTURE in w.statusBar().currentMessage()


def test_the_two_modules_spell_the_states_the_same_way():
    """``main_window`` imports the window lazily, so it spells the states itself; the
    two spellings are pinned equal here so a rename cannot split them."""
    assert MainWindow._SUBPOP_NONE == SW.ASSIGNMENT_NONE
    assert MainWindow._SUBPOP_INSTALLED == SW.ASSIGNMENT_INSTALLED
    assert MainWindow._SUBPOP_AWAY == SW.ASSIGNMENT_AWAY
    assert MainWindow._SUBPOP_RETIRED == SW.ASSIGNMENT_RETIRED
    with pytest.raises(ValueError):
        SW.SubpopReportWindow().set_assignment_state("elsewhere")
