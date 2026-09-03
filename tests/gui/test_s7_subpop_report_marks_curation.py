"""Sweep 7, S7-POP4 (minor) — a standing subpopulation report SAYS when the accepted set
has moved under it, and by how much; its own numbers stay the run's.

Measured on the owner's 0006 (docs/sweep7/repros/S7-POP4_report_not_marked_after_
curation.py): run — classes {0: 131, 1: 105}, fractions 0.555 / 0.445, rates 2.188 /
1.754 per s; reject 66 of class 0 -> 170 of 236 accepted; header, status headline, gallery
combo and population table all byte-identical, and nothing anywhere said the run was made
over a different accepted set. A re-run on the curated set returned THREE classes.

Keeping the map across curation is deliberate and documented (the frame is edited in
place; ``_refresh_subpop_views``). The missing half was the statement. The report window
now carries a standing line — built like its other notes, always in the layout so nothing
moves under the pointer — reading "accepted now: 170 of 236; this run's fractions describe
the 236 accepted at run time — re-run to update", written on every curation the app
announces (the SURFACE_MUTATION_SIGNALS the window already refreshes on) and cleared by a
re-run. The gallery combo and the trace colouring keep working, as documented.

Driven through the real doors: the events table's accept/reject (``set_event_accepted``,
which emits ``eventAcceptanceChanged``) and the gallery's Reject-all / Accept-all buttons.
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
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import subpop_window as SW
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N_SAMPLES = 420_000
N_EVENTS = 100
_SPACING = 4_000


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("S7SubpopCurationPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


def _trace(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 100.0) + rng.normal(0.0, 1.5, N_SAMPLES)
    for i in range(N_EVENTS):
        s = 2_000 + i * _SPACING
        length = int(rng.integers(60, 600))
        x[s:s + length] -= float(rng.uniform(15.0, 45.0))
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
def recording(tmp_path_factory):
    return tmp_path_factory.mktemp("s7_subpop_cur") / "a.csv", _trace()


class _Cheap(SR.SubpopConfig):
    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2, 3))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        super().__init__(**kw)


_REPORTS: dict[tuple, object] = {}


def _report_over(w, doc_id: str, events: EventSet):
    res = w._session.results.get(doc_id, 0, "detect.schmitt")
    raw = np.asarray(w._session.calibrated(doc_id, 0).data)
    prov = dict(res.payload.provenance or {})
    signal = detection_signal(raw, FS, prov)
    key = tuple(int(e) for e in events.accepted.df["event_id"])
    report = _REPORTS.get(key)
    if report is None:
        report = SR.run_subpopulations(
            signal, raw, FS, events.accepted, recording="a.csv",
            duration_s=N_SAMPLES / FS,
            config=_Cheap(polarity=SR.polarity_for_features(prov.get("polarity"))))
        _REPORTS[key] = report
    return res, report, signal


def _adopt(w, qtbot, doc_id: str):
    res = w._session.results.get(doc_id, 0, "detect.schmitt")
    res, report, signal = _report_over(w, doc_id, res.payload)
    assert report.quantification is not None, "no mixture; nothing below is about the wiring"
    w._subpop_signal_pending = signal
    w._subpop_target = res
    w._on_subpop_finished(report)
    qtbot.wait(20)
    return report


@pytest.fixture
def reported(qapp, qtbot, recording):
    """A window with a real detection AND an adopted report over all of it."""
    path, arr = recording
    w = MainWindow()
    qtbot.addWidget(w)
    a = w._session.add_document(_doc(path, arr))
    w._create_default_layers(w._session.document(a))
    w._on_selection(a, 0)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    df = w._session.results.get(a, 0, "detect.schmitt").payload.df
    assert bool(df["accepted"].all()), "the fixture starts with every event accepted"
    _adopt(w, qtbot, a)
    yield w, a
    w._analysis_controller.abort()
    w._controller.abort()


def _frame(w, a):
    return w._session.results.get(a, 0, "detect.schmitt").payload.df


def _expected(n_now: int, n_rows: int, n_run: int) -> str:
    return (f"accepted now: {n_now} of {n_rows}; this run's fractions describe the "
            f"{n_run} accepted at run time — re-run to update")


# ================================================================================================
# the table's door
# ================================================================================================

def test_rejecting_events_under_a_standing_report_writes_the_live_counts_on_the_window(
        reported, qtbot):
    w, a = reported
    sw = w._subpop_window
    ap = w._analysis_panel
    df = _frame(w, a)
    n = len(df)
    header = sw.header()
    table = sw.report().quantification.table.copy()
    assign = dict(w._subpop_assignment)
    gal = w._act_event_gallery.trigger() or w._event_gallery
    qtbot.wait(30)
    gal = w._event_gallery
    combo = [gal.subpop_box.itemText(i) for i in range(gal.subpop_box.count())]
    assert sw.curation_note() == ""

    victims = [int(e) for e in df["event_id"].to_numpy()[:7]]
    for eid in victims:
        ap.set_event_accepted(eid, False)
    qtbot.wait(40)
    assert int(df["accepted"].sum()) == n - 7

    want = _expected(n - 7, n, n)
    assert sw.curation_note() == want
    assert want in sw.live_note()
    # The run's own numbers are the run's: nothing on the document moved.
    assert sw.header() == header
    pd.testing.assert_frame_equal(sw.report().quantification.table, table)
    # ...and the map survives curation, as documented: combo and colouring keep working.
    assert w._subpop_assignment == assign
    assert [gal.subpop_box.itemText(i) for i in range(gal.subpop_box.count())] == combo
    assert sw.can_show_events()
    w._colour_trace_by_subpops()
    assert w.statusBar().currentMessage().startswith("Trace spans coloured by subpopulation")


def test_accepting_them_back_clears_the_line(reported, qtbot):
    w, a = reported
    sw = w._subpop_window
    ap = w._analysis_panel
    df = _frame(w, a)
    eids = [int(e) for e in df["event_id"].to_numpy()[:3]]
    for eid in eids:
        ap.set_event_accepted(eid, False)
    qtbot.wait(30)
    assert sw.curation_note() == _expected(len(df) - 3, len(df), len(df))
    for eid in eids:
        ap.set_event_accepted(eid, True)
    qtbot.wait(30)
    assert sw.curation_note() == ""
    assert sw.live_note() == ""


# ================================================================================================
# the gallery's doors
# ================================================================================================

def test_the_gallery_s_reject_all_and_accept_all_buttons_reach_the_line(reported, qtbot):
    w, a = reported
    sw = w._subpop_window
    w._act_event_gallery.trigger()
    qtbot.wait(30)
    gal = w._event_gallery
    gal.set_filter("All")
    qtbot.wait(20)
    n = len(_frame(w, a))
    gal.reject_all_btn.click()
    qtbot.wait(60)
    assert int(_frame(w, a)["accepted"].sum()) == 0
    assert sw.curation_note() == _expected(0, n, n)
    gal.accept_all_btn.click()
    qtbot.wait(60)
    assert sw.curation_note() == ""


# ================================================================================================
# a re-run clears it; the layout never moves
# ================================================================================================

def test_a_re_run_over_the_curated_set_clears_the_line(reported, qtbot):
    w, a = reported
    sw = w._subpop_window
    ap = w._analysis_panel
    df = _frame(w, a)
    for eid in df["event_id"].to_numpy()[:7]:
        ap.set_event_accepted(int(eid), False)
    qtbot.wait(30)
    assert sw.curation_note() != ""
    res = w._session.results.get(a, 0, "detect.schmitt")
    _res, report, signal = _report_over(w, a, res.payload)
    assert report.n_fired == len(df) - 7
    w._subpop_signal_pending = signal
    w._subpop_target = res
    w._on_subpop_finished(report)
    qtbot.wait(30)
    assert sw.curation_note() == ""
    assert sw.live_note() == ""
    assert sw.report() is report


def test_the_standing_line_moves_no_widget_when_it_appears(reported, qtbot):
    """One label, always in the layout, fixed height: the splitter (and every figure in
    it) and the button row sit exactly where they sat before the sentence arrived."""
    w, a = reported
    sw = w._subpop_window
    sw.show()
    qtbot.wait(40)
    before = (sw._split.geometry(), sw._gallery_btn.geometry(), sw._live_note.geometry(),
              sw._figures.geometry())
    for eid in _frame(w, a)["event_id"].to_numpy()[:5]:
        w._analysis_panel.set_event_accepted(int(eid), False)
    qtbot.wait(60)
    assert sw.curation_note() != ""
    after = (sw._split.geometry(), sw._gallery_btn.geometry(), sw._live_note.geometry(),
             sw._figures.geometry())
    assert after == before, f"the standing line moved the layout: {before} -> {after}"


def test_the_window_seam_alone_shows_and_clears_the_sentence(qapp, qtbot):
    win = SW.SubpopReportWindow()
    qtbot.addWidget(win)
    win.set_curation_note("accepted now: 170 of 236; this run's fractions describe the 236 "
                          "accepted at run time — re-run to update")
    assert win.curation_note().startswith("accepted now: 170 of 236")
    assert win.live_note() == win.curation_note()
    win.set_curation_note("")
    assert win.live_note() == ""
