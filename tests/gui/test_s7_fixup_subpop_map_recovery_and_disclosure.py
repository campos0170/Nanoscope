"""Sweep 7 FIX-UP — the held subpopulation map: every state it can be parked in has to be
recoverable, and every sentence about it has to be true of what happened.

Wave 1 gave the map a life keyed to its detection (it is HELD while the user looks
elsewhere, RETIRED only when its detection is replaced) and a standing line in the report
window. The skeptics then measured five holes in exactly that new machinery, all of them
on the owner's own recordings, and this file pins the repairs:

* **C1** — running ANOTHER detector on the same channel parked the map "away" with the
  sentence "…until you select that recording again in the Files panel". Following it does
  nothing: the recording is already selected, and re-selecting shows the other detector's
  result again (``_stored_to_show`` prefers the previously-active detector). Measured on
  ABF 0004: Schmitt 110 events / 83 assigned, then a CUSUM run (250 events) → state
  ``away`` forever, buttons dead, no way back. The AWAY case now tells the two apart and
  names the recovery that works.
* **X-SURF-12** — and when it IS another recording, the dock to click is titled
  "File Manager"; the app has no widget called "Files".
* **X6** — Edit ▸ Undo of a plain curation retired the map with "retired when the
  detection changed", although the curation itself had left it installed: the undo
  installs a snapshot COPY of the same 115 rows with one accept flag put back.
* **C2** — the standing line was derived from the accepted COUNT, so rejecting one event
  and accepting another back cleared it while the run's fractions described a different
  SET (measured on 0004: event 3 accepted now was not in the run, event 7 in the run is
  rejected now).
* **X5** — Export… wrote the run's header and fractions with no word of the curation
  since, although ``write_report``'s own contract is that the file is read WITHOUT the app
  beside it.
* **C4** — after the away-and-back round trip the map returned, but the gallery's tiles
  were silently redrawn from "Calibrated (as recorded)" with the representative sample
  off, both of which the report's "Event gallery…" button had turned on.

Driven through real widgets: the File dock's rows, the detection tab's own detector combo
and Detect, the events table's curation door, the Edit menu's Undo, the report window's
buttons, the real gallery and the real exporter. The fixtures are the shape
``test_s7_subpop_assignment_survives_navigation`` established (two CSV-backed recordings,
REAL Schmitt detections, the mixture run cached per accepted-id set) and are spelled out
here rather than imported: this package is imported by pytest as ``gui``, and a test file
reaching sideways into a sibling by that name is a trap for whoever next runs the suite
from another root.

Measured BEFORE (the wave-1 tree, a copy of it under a symlinked tree so no other agent's
run saw it): 10 of these 12 fail — the away sentence names a "Files panel" for a CUSUM run
on the same recording, the undo retires the map, the curation note is "" after the swap,
the export carries no caveat, and the gallery comes back with the representative sample
off. The two that pass before and after are the controls they are there to be: an
untouched run exports exactly its own caveats, and the round trip does not re-impose the
"Event gallery…" defaults over the curator's own choice (which the wave-1 tree passed by
resetting everything, and this tree passes by remembering).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import detection_signal, register_all
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import subpop_window as SW
from nanoscope.gui.event_gallery import TRACE_CALIBRATED, TRACE_DETRENDED
from nanoscope.gui.main_window import MainWindow

CUSUM = "detect.cusum"
SCHMITT = "detect.schmitt"

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
    """This module's own QSettings namespace — nothing here reads or leaves app state."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("S7FixupSubpopRecovery")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


# ================================================================================================
# builders — the navigation suite's two CSV-backed recordings, verbatim in shape: events of
# different dwells and depths so the mixture stage has something to fit
# ================================================================================================

def _trace(seed: int) -> np.ndarray:
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
def recordings(tmp_path_factory):
    d = tmp_path_factory.mktemp("s7_fixup_subpop")
    a, b = d / "a.csv", d / "b.csv"
    return (a, _trace(3)), (b, _trace(11))


def _detect_through_the_panel(w, qtbot, doc_id: str) -> None:
    w._on_selection(doc_id, 0)
    w._analysis_panel._detection.set_detector(SCHMITT)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=60_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert w._session.results.get(doc_id, 0, SCHMITT) is not None


@pytest.fixture
def two(qapp, qtbot, recordings):
    """A window with REAL Schmitt detections on two recordings, A selected. No report yet."""
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


def _report_for(w, doc_id: str):
    """The report the worker would have produced for the stored detection's ACCEPTED set,
    built by the same function and cached per (recording, accepted-id set)."""
    res = w._session.results.get(doc_id, 0, SCHMITT)
    raw = np.asarray(w._session.calibrated(doc_id, 0).data)
    prov = dict(res.payload.provenance or {})
    signal = detection_signal(raw, FS, prov)
    key = (str(w._session.document(doc_id).meta.path),
           tuple(int(e) for e in res.payload.accepted.df["event_id"]))
    report = _REPORTS.get(key)
    if report is None:
        report = SR.run_subpopulations(
            signal, raw, FS, res.payload.accepted, recording=Path(key[0]).name,
            duration_s=N_SAMPLES / FS,
            config=_Cheap(polarity=SR.polarity_for_features(prov.get("polarity"))))
        _REPORTS[key] = report
    return res, report, signal


def _adopt(w, qtbot, doc_id: str):
    """Hand a report to the REAL landing slot, launch target and all.

    Every field ``_run_subpopulations`` writes when it starts a run is written here, because
    the landing slot CONSUMES the launch: the accepted-id set is snapshotted off the very
    ``EventSet`` the job is handed and not rebuilt from the frame when the report lands, so
    that a curation made while the mixture flies cannot land inside the run's own snapshot
    (C2 close — measured: the report said 110 accepted, 109 were, and the standing line was
    ""). ``_report_for`` builds the report from that same ``payload.accepted``, so this is
    the set this fake launch handed over.
    """
    res, report, signal = _report_for(w, doc_id)
    w._subpop_signal_pending = signal
    w._subpop_accepted_ids_pending = frozenset(
        int(e) for e in res.payload.accepted.df["event_id"])
    w._subpop_target = res
    w._on_subpop_finished(report)
    qtbot.wait(20)
    assert w._subpop_window is not None
    assert w._subpop_assignment, "the fixture produced no assignment to hold"
    return report


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, SCHMITT).payload.df


def _focus(w, qtbot, doc_id: str) -> None:
    """The File dock's own row, as a click reaches it."""
    assert w._file_dock.focus_channel(doc_id, 0), "the dock has no row for this channel"
    qtbot.wait(80)
    assert w._analysis_panel.current_detection().doc_id == doc_id


# ================================================================================================
# doors
# ================================================================================================

def _run_cusum(w, qtbot) -> AnalysisResult:
    """A REAL CUSUM run on the selected channel, through the detection tab's own combo.

    The gesture C1 is about: the owner switches the detector and presses Detect, on the
    recording a Schmitt subpopulation report is standing on.
    """
    w._analysis_panel._detection.set_detector(CUSUM)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=120_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(40)
    res = w._analysis_panel.current_detection()
    assert res is not None and res.module_id == CUSUM, (
        f"the CUSUM run did not become the detection on screen ({res and res.module_id})")
    return res


def _curate(w, qtbot, event_id: int, accepted: bool) -> None:
    """The events table's own curation door (the one the checkbox drives)."""
    w._analysis_panel.set_event_accepted(int(event_id), bool(accepted))
    qtbot.wait(40)


def _accepted_ids(w, doc_id) -> set[int]:
    df = _frame(w, doc_id)
    return {int(e) for e in df.loc[df["accepted"].to_numpy().astype(bool), "event_id"]}


def _undo(w, qtbot) -> str:
    """Edit ▸ Undo, read the way the menu offers it."""
    w._on_edit_menu_about_to_show()
    label = w._act_undo.text()
    w._act_undo.trigger()
    qtbot.wait(80)
    return label


def _replace_frame(w, doc_id, df) -> None:
    """Store ``df`` as this channel's Schmitt result — a detection replaced under the map."""
    prov = dict(w._session.results.get(doc_id, 0, SCHMITT).payload.provenance or {})
    w._session.add_result(AnalysisResult(
        module_id=SCHMITT, module_version="1", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))


# ================================================================================================
# C1 + X-SURF-12 — the two ways a map is "away" are not the same way, and neither
# instruction may be a dead end
# ================================================================================================

def test_another_detectors_run_holds_the_map_and_says_which_detector_is_on_screen(two, qtbot):
    w, a, _b = two
    _adopt(w, qtbot, a)
    assign = dict(w._subpop_assignment)
    sw = w._subpop_window

    _run_cusum(w, qtbot)

    # NOTHING WAS RETIRED: the CUSUM result is filed under its own module id and this
    # map's own stored frame is untouched.
    assert w._subpop_assignment == assign, "another detector's run threw the mixture away"
    assert w._subpop_key[2] == SCHMITT
    state, reason = sw.assignment_state()
    assert state == SW.ASSIGNMENT_AWAY, (state, reason)
    assert not sw.can_show_events(), "the buttons stayed live over a map that is not on screen"
    assert w._analysis_panel.subpop_labels() is None, "Schmitt's map coloured CUSUM's spans"

    # ...and the sentence names BOTH detectors and a recovery that exists.
    assert "Schmitt" in reason and "CUSUM" in reason, reason
    assert "Re-run Analysis ▸ Generate Subpopulations" in reason, reason
    assert "nothing was retired" in reason, reason
    assert reason in sw.live_note()


def test_the_other_detector_sentence_never_sends_the_user_to_the_file_dock(two, qtbot):
    """The dead end C1 measured: the instruction was "select that recording again", and
    the recording was already selected — re-selecting it shows the other detector again."""
    w, a, _b = two
    _adopt(w, qtbot, a)
    _run_cusum(w, qtbot)
    reason = w._subpop_window.assignment_state()[1]
    assert "select that recording again" not in reason, reason
    assert "Files panel" not in reason and "File Manager" not in reason, reason

    # Follow it anyway: the recording is re-selected and the state does NOT change, which
    # is exactly why the sentence must not offer it.
    _focus(w, qtbot, a)
    assert w._analysis_panel.current_detection().module_id == CUSUM
    assert w._subpop_window.assignment_state()[0] == SW.ASSIGNMENT_AWAY
    assert w._subpop_window.assignment_state()[1] == reason


def test_the_other_recording_sentence_names_a_dock_this_window_actually_has(two, qtbot):
    """X-SURF-12: the dock is titled "File Manager"; nothing in the app is called "Files"."""
    w, a, b = two
    _adopt(w, qtbot, a)
    _focus(w, qtbot, b)
    state, reason = w._subpop_window.assignment_state()
    assert state == SW.ASSIGNMENT_AWAY
    titles = {d.windowTitle() for d in w.findChildren(QtWidgets.QDockWidget)}
    assert "File Manager" in titles, titles
    assert "File Manager" in reason, reason
    assert "Files panel" not in reason, reason
    named = [t for t in titles if t in reason]
    assert named == ["File Manager"], (named, reason)


# ================================================================================================
# X6 — a curation the map survives, undone, is still a curation
# ================================================================================================

def test_undoing_a_plain_curation_keeps_the_map_that_curation_itself_kept(two, qtbot):
    w, a, _b = two
    _adopt(w, qtbot, a)
    assign = dict(w._subpop_assignment)
    sw = w._subpop_window
    eid = sorted(_accepted_ids(w, a))[0]

    _curate(w, qtbot, eid, False)
    assert sw.assignment_state()[0] == SW.ASSIGNMENT_INSTALLED, "the curation itself retired it"

    assert _undo(w, qtbot) == "Undo Curate event", "this test is not about a curation undo"

    assert w._subpop_assignment == assign, "the undo threw the mixture away"
    assert sw.assignment_state() == (SW.ASSIGNMENT_INSTALLED, ""), sw.assignment_state()
    assert sw.can_show_events(), "both report buttons died on an undone curation"
    # RE-BOUND, not merely kept: the snapshot the undo installed is the anchor now, so the
    # next refresh does not read it as a second replacement.
    assert w._subpop_frame is _frame(w, a)
    assert w._analysis_panel.subpop_labels() == assign
    assert eid in _accepted_ids(w, a), "the undo did not put the accept flag back"


def test_a_change_that_is_not_a_curation_still_retires_the_map(two, qtbot):
    """The exemption is narrow ON PURPOSE: a frame whose spans moved is a different
    detection, whatever the map's ids happen to match."""
    w, a, _b = two
    _adopt(w, qtbot, a)
    sw = w._subpop_window
    resized = _frame(w, a).copy(deep=True)
    resized.iloc[0, resized.columns.get_loc("start_idx")] = int(
        resized.iloc[0]["start_idx"]) - 40

    _replace_frame(w, a, resized)
    qtbot.wait(80)

    assert w._subpop_assignment is None
    state, reason = sw.assignment_state()
    assert state == SW.ASSIGNMENT_RETIRED, (state, reason)
    assert "retired when the detection changed" in reason, reason
    # ...and the reason lists what actually retires a map. "ML run" was in this list and
    # is not what happens when the ML detector runs beside a Schmitt map (C1); "undo" was
    # in it unqualified and is not what happens when a curation is undone (X6).
    assert "an undo past the detection" in reason, reason
    assert "ML run" not in reason, reason


# ================================================================================================
# C2 — the standing line is about the accepted SET, not its size
# ================================================================================================

def test_the_standing_line_survives_a_swap_that_keeps_the_accepted_count(two, qtbot):
    w, a, _b = two
    ids = sorted(_accepted_ids(w, a))
    put_back, take_away = ids[0], ids[1]

    # The run happens with ONE event already rejected, so a later swap can restore the
    # count without restoring the set — the gesture the skeptic measured on 0004.
    _curate(w, qtbot, put_back, False)
    report = _adopt(w, qtbot, a)
    assert report.n_fired == len(ids) - 1, report.n_fired
    assert w._subpop_curation_note() == "", "a run is not stale the moment it lands"

    _curate(w, qtbot, put_back, True)
    assert w._subpop_curation_note(), "a changed accepted count was not disclosed at all"

    _curate(w, qtbot, take_away, False)
    now = _accepted_ids(w, a)
    assert len(now) == report.n_fired, "this test is not about a count change any more"
    assert now != set(w._subpop_accepted_ids), "the swap did not move the set"

    note = w._subpop_curation_note()
    assert note, "the line went silent because the COUNT came back, not the set"
    assert f"the {report.n_fired} accepted at run time" in note, note
    assert w._subpop_window.live_note().endswith(note), w._subpop_window.live_note()


def test_the_standing_line_is_silent_only_while_the_run_describes_what_is_accepted(two, qtbot):
    """The control for the test above: the line must not appear over an untouched run,
    and it must go once the accepted set is the run's again."""
    w, a, _b = two
    report = _adopt(w, qtbot, a)
    assert w._subpop_curation_note() == ""
    assert w._subpop_accepted_ids == frozenset(_accepted_ids(w, a))

    eid = sorted(_accepted_ids(w, a))[0]
    _curate(w, qtbot, eid, False)
    assert w._subpop_curation_note(), "a rejection under a standing report said nothing"

    _curate(w, qtbot, eid, True)
    assert w._subpop_curation_note() == "", (
        "the accepted set is the run's again and the line still cries stale")
    assert int(_frame(w, a)["accepted"].sum()) == report.n_fired


# ================================================================================================
# X5 — what the window says, the exported file says
# ================================================================================================

def _export(w, qtbot, monkeypatch, tmp_path):
    from nanoscope.gui import main_window as MW

    target = tmp_path / "subpopulations.txt"
    monkeypatch.setattr(MW, "ask_export_path", lambda *a, **k: target)
    w._export_subpop_report()
    qtbot.wait(20)
    text = target.read_text(encoding="utf-8")
    meta = json.loads((tmp_path / "subpopulations.meta.json").read_text(encoding="utf-8"))
    return text, meta


def test_the_exported_report_carries_the_curation_the_window_is_showing(
        two, qtbot, monkeypatch, tmp_path):
    w, a, _b = two
    report = _adopt(w, qtbot, a)
    _curate(w, qtbot, sorted(_accepted_ids(w, a))[0], False)
    note = w._subpop_curation_note()
    assert note, "nothing to carry — the window itself said nothing"

    text, meta = _export(w, qtbot, monkeypatch, tmp_path)

    assert note in text, "the file drops the one sentence the window shows"
    assert "CURATION SINCE THE RUN" in text
    assert any(note in str(wmsg) for wmsg in meta["warnings"]), meta["warnings"]
    # THE NUMBERS DID NOT MOVE: the header is still the run's, and every caveat the run
    # itself raised is still there, in order, with the disclosure appended after them.
    assert report.headline().split(" · ")[0] in text
    assert list(meta["warnings"])[:len(report.warnings)] == list(report.warnings)
    assert len(meta["warnings"]) == len(report.warnings) + 1


def test_an_untouched_run_exports_exactly_the_caveats_it_raised(
        two, qtbot, monkeypatch, tmp_path):
    """The control: the disclosure is appended only when there is something to disclose."""
    w, a, _b = two
    report = _adopt(w, qtbot, a)
    assert w._subpop_curation_note() == ""

    text, meta = _export(w, qtbot, monkeypatch, tmp_path)

    assert list(meta["warnings"]) == list(report.warnings)
    assert "CURATION SINCE THE RUN" not in text


# ================================================================================================
# C4 — the round trip gives the tiles back the way the curator left them
# ================================================================================================

def _open_gallery_on_the_report(w, qtbot):
    w._subpop_window._gallery_btn.click()
    qtbot.wait(60)
    gal = w._event_gallery
    assert gal is not None, "the report's own button opened no gallery"
    return gal


def test_the_round_trip_puts_back_the_trace_and_the_sample_the_report_turned_on(two, qtbot):
    w, a, b = two
    _adopt(w, qtbot, a)
    gal = _open_gallery_on_the_report(w, qtbot)
    assert gal.trace_name() == TRACE_DETRENDED
    assert gal.is_representative(), "the report's button no longer turns the sample on"

    _focus(w, qtbot, b)
    assert gal.subpopulations() is None
    assert gal.trace_name() == TRACE_CALIBRATED, "the detrended trace was not withdrawn"

    _focus(w, qtbot, a)
    assert gal.subpopulations() == dict(w._subpop_assignment)
    assert gal.trace_name() == TRACE_DETRENDED, (
        "the tiles came back drawn from another trace, with no word that they had changed")
    assert gal.is_representative(), (
        "the representative sample came back off, with no word that it had changed")
    assert "representative sample" in gal.status.text(), gal.status.text()


def test_the_round_trip_restores_the_curators_choice_and_not_the_buttons_default(two, qtbot):
    """What comes back is what the CURATOR left on screen — the restore remembers, it does
    not re-assert the "Event gallery…" defaults."""
    w, a, b = two
    _adopt(w, qtbot, a)
    gal = _open_gallery_on_the_report(w, qtbot)
    gal.set_representative(False)
    gal.set_trace(TRACE_CALIBRATED)
    qtbot.wait(40)

    _focus(w, qtbot, b)
    _focus(w, qtbot, a)

    assert gal.subpopulations() == dict(w._subpop_assignment)
    assert gal.trace_name() == TRACE_CALIBRATED, "the round trip re-imposed the detrended trace"
    assert not gal.is_representative(), "the round trip re-imposed the representative sample"


def test_a_gallery_opened_after_the_round_trip_is_not_told_a_remembered_choice_it_never_had(
        two, qtbot):
    """A fresh adoption remembers nothing, so the report's button keeps choosing for it."""
    w, a, _b = two
    _adopt(w, qtbot, a)
    assert w._subpop_gallery_trace is None
    assert w._subpop_gallery_representative is False
    gal = _open_gallery_on_the_report(w, qtbot)
    assert gal.trace_name() == TRACE_DETRENDED and gal.is_representative()
