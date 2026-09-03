"""The roads BETWEEN the three curation surfaces — the events table, the Event Inspector and
the Event gallery — and the fitted sub-state levels that have to travel with them.

Four defects, all of them "the app did the work and the user could not see it":

* **A fit that never reached the window it was fitted for.** Both curation windows null their
  level table inside ``set_events`` whenever the frame differs, and nothing put it back. So
  Analysis ▸ Fit sub-state levels… followed by Analysis ▸ Event inspector… gave a window with
  no CUSUM steps drawn and an ``n_levels``-only readout — the fit looked like it had not run.
  Selecting another recording and coming back did the same to a window that had them, and the
  gallery's per-tile level bars were never installed by anything at all.

* **A valid fit thrown away for looking somewhere else.** ``_sync_export_action`` retired the
  fit whenever the CURRENT selection had no detection, so one click on a raw recording in the
  Files panel discarded it — out of this window's hands AND out of the ResultStore, i.e. out
  of the next project save.

* **"Go to this event in the trace" moved nothing.** Both curation windows are
  ``X(session, self)`` — QMainWindows with a transient parent — and on macOS a transient child
  stays above its parent whatever the parent does, so ``raise_()`` alone was invisible.

* **The events table was a dead end.** The surface a detection actually lands on had no
  double-click handler and no context menu, so the only routes to either curation window were
  items 5 and 13 of a native ~14-item menu at the top of the screen.

The detector is driven DIRECTLY (``analysis.detect.detect_events``) and its EventSet installed
through ``Session.add_result``, exactly as ``test_level_fit_reconcile`` does: these are tests
about wiring, and running the detection through the panel would make them tests about the
detection instead.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
#: Six events, each with two clean sub-states (a 300 pA step then a 500 pA one) so CUSUM has
#: something real to resolve and the gallery has more than a tile or two to lay out.
_SPANS = [(5_000, 5_600), (12_000, 12_600), (20_000, 20_600),
          (28_000, 28_600), (36_000, 36_600), (44_000, 44_600)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 5.0, N)
    for s, e in _SPANS:
        mid = (s + e) // 2
        x[s:mid] -= 300.0
        x[mid:e] -= 500.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _flat_signal() -> np.ndarray:
    """An open-pore recording with nothing in it — a second document to select AWAY to."""
    rng = np.random.default_rng(7)
    return np.ascontiguousarray(
        np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 5.0, N), dtype=np.float32)


def _doc(tmp_path, signal, name: str = "cusum.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
                         n_samples=int(signal.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _install(window: MainWindow, tmp_path, signal, name: str) -> str:
    doc_id = window._session.add_document(_doc(tmp_path, signal, name))
    window._create_default_layers(window._session.document(doc_id))
    return doc_id


def _install_detection(window: MainWindow, doc_id: str, signal) -> "pd.DataFrame":  # noqa: F821
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4},
        calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    return df


@pytest.fixture
def detected(window, qtbot, tmp_path):
    """A window with one recording selected and a detection stored on it. NO fit, NO windows."""
    signal = _signal()
    doc_id = _install(window, tmp_path, signal, "cusum.dat")
    window._on_selection(doc_id, 0)
    df = _install_detection(window, doc_id, signal)
    qtbot.wait(10)
    assert len(df) >= 4, "the fixture recording must produce several events"
    return window, doc_id, df


# ================================================================================================
# 1. the fitted levels reach EVERY surface that draws them, whenever it is fed
# ================================================================================================

def test_an_inspector_opened_after_the_fit_still_gets_the_levels(detected, qtbot):
    """*** THE PRIMARY DEFECT. *** Fit first, open the browser second — the order a user who
    has just read "Fitted 12 level(s)…" in the status bar naturally takes.

    ``_refresh_event_inspector`` calls ``set_events``, which nulls the window's private level
    table because the frame it is being handed is new to it. Nothing re-established them, so
    the CUSUM steps were absent from the plot and the readout listed no L0/L1 — on a fit that
    had in fact succeeded."""
    window, _doc_id, _df = detected
    window._act_fit_levels.trigger()
    qtbot.wait(10)
    assert window._level_fit is not None and not window._level_fit.payload.empty

    window._act_event_inspector.trigger()
    qtbot.wait(10)
    insp = window._event_inspector
    assert insp is not None
    assert len(insp._cusum_items) > 0, "the fit is installed but no level was drawn"
    assert "L0" in insp._readout.text()


def test_the_levels_survive_selecting_away_and_back(detected, qtbot, tmp_path):
    """The mirror: a window that HAS the levels is re-fed on every selection change, and each
    re-feed nulls them again. Landing back on the fitted recording must restore them."""
    window, doc_id, _df = detected
    window._act_event_inspector.trigger()
    window._act_fit_levels.trigger()
    qtbot.wait(10)
    insp = window._event_inspector
    assert len(insp._cusum_items) > 0

    other = _install(window, tmp_path, _flat_signal(), "empty.dat")
    window._on_selection(other, 0)
    qtbot.wait(10)

    window._on_selection(doc_id, 0)
    qtbot.wait(10)
    assert len(insp._cusum_items) > 0, "the levels did not come back with the recording"
    assert "L0" in insp._readout.text()


def test_the_gallery_gets_the_same_level_table(detected, qtbot):
    """``EventGalleryWindow.set_levels`` existed and painted, and NOTHING called it — the
    per-tile CUSUM bars were dormant code. One push now feeds both surfaces."""
    window, _doc_id, _df = detected
    window._act_fit_levels.trigger()
    qtbot.wait(10)

    window._act_event_gallery.trigger()
    qtbot.wait(10)
    gal = window._event_gallery
    assert gal is not None
    assert gal._levels is not None and not gal._levels.empty
    # ...and it is the very frame this window holds, not a second copy that can drift.
    assert gal._levels is window._level_fit.payload


def test_a_fit_is_never_printed_under_another_recordings_events(detected, qtbot, tmp_path):
    """The hazard the ``set_events`` reset exists for, and therefore the one the re-push has to
    answer: level rows are matched by ``event_id``, which every event table has and which means
    nothing across recordings."""
    window, _doc_id, _df = detected
    window._act_event_inspector.trigger()
    window._act_fit_levels.trigger()
    qtbot.wait(10)
    insp = window._event_inspector

    other = _install(window, tmp_path, _signal(seed=3), "other.dat")
    window._on_selection(other, 0)
    _install_detection(window, other, _signal(seed=3))
    window._on_selection(other, 0)
    qtbot.wait(10)

    assert insp._levels is None, "recording A's levels were printed under recording B's events"
    assert len(insp._cusum_items) == 0


def test_looking_at_another_recording_does_not_destroy_the_fit(detected, qtbot, tmp_path):
    """``_sync_export_action`` used to lead with "the CURRENT selection has no detection", so
    one click on a raw recording retired a perfectly good fit — from this window AND from the
    ResultStore, which is what project save reads. Selecting elsewhere is not a statement about
    the levels."""
    window, doc_id, _df = detected
    window._act_fit_levels.trigger()
    qtbot.wait(10)
    fitted = window._level_fit
    assert fitted is not None

    other = _install(window, tmp_path, _flat_signal(), "raw.dat")
    window._on_selection(other, 0)          # a recording with no detection at all
    qtbot.wait(10)

    assert window._level_fit is fitted, "a valid level fit was discarded by looking away"
    assert window._session.results.get(doc_id, 0, "fit.levels") is not None
    assert window._act_export_levels.isEnabled()


def test_a_re_detection_still_retires_the_fit(detected, qtbot):
    """The narrowing must not cost the guarantee it was narrowed around: a NEW frame on the
    fit's own recording and channel is real staleness and still retires it, store included."""
    window, doc_id, _df = detected
    window._act_fit_levels.trigger()
    qtbot.wait(10)
    assert window._level_fit is not None

    _install_detection(window, doc_id, _signal(seed=1))   # a different frame, same target
    window._on_selection(doc_id, 0)
    qtbot.wait(10)

    assert window._level_fit is None
    assert window._session.results.get(doc_id, 0, "fit.levels") is None


# ================================================================================================
# 2. "show me this event in the trace" actually shows it
# ================================================================================================

def test_going_to_an_event_in_the_trace_lowers_the_window_in_front_of_it(detected, qtbot):
    """``raise_()`` on the parent is a no-op against a transient child on macOS. Lowering the
    child is what moves, so that is what the slot does — for the gallery's double-click and the
    inspector's "Show in trace" alike."""
    window, _doc_id, df = detected
    window.show()
    window._act_event_gallery.trigger()
    window._act_event_inspector.trigger()
    qtbot.wait(10)
    gal, insp = window._event_gallery, window._event_inspector
    gal.show()
    insp.show()
    qtbot.wait(10)

    lowered: list[str] = []
    gal.lower = lambda: lowered.append("gallery")          # type: ignore[method-assign]
    insp.lower = lambda: lowered.append("inspector")       # type: ignore[method-assign]

    window._goto_event_in_trace(int(df["event_id"].iloc[2]))

    assert set(lowered) == {"gallery", "inspector"}
    # ...and the gesture still does the thing it was always supposed to do.
    assert int(window._analysis_panel._selected_id) == int(df["event_id"].iloc[2])


def test_the_inspectors_show_in_trace_signal_is_connected(detected, qtbot):
    """``traceRequested`` (the "Show in trace" button / T) emitted into nothing until it was
    wired — the same slot the gallery's double-click uses, so both windows say it one way."""
    window, _doc_id, df = detected
    window._act_event_inspector.trigger()
    qtbot.wait(10)

    seen: list[int] = []
    window._analysis_panel.eventSelected.connect(lambda eid: seen.append(int(eid)))
    target = int(df["event_id"].iloc[3])
    window._event_inspector.traceRequested.emit(target)

    assert int(window._analysis_panel._selected_id) == target
    assert seen and seen[-1] == target


def test_a_plain_tile_click_says_what_it_did(detected, qtbot):
    """Everything a single click moves — the trace, the table row, the marker — is on the
    window the gallery is standing in front of, so from the user's chair it changed nothing.
    One sentence names what happened and what the next gesture is."""
    window, _doc_id, df = detected
    window._act_event_gallery.trigger()
    qtbot.wait(10)

    target = int(df["event_id"].iloc[1])
    window._event_gallery.eventSelected.emit(target)

    msg = window.statusBar().currentMessage()
    assert f"Event {target}" in msg
    assert "Double-click" in msg


# ================================================================================================
# 3. opening a curation window ON an event actually lands on that event
# ================================================================================================

def test_opening_the_inspector_at_an_event_lands_on_that_event(detected, qtbot):
    """``select_event`` is a NO-OP when the requested row is already the table's current row —
    which it always is for a gesture on the focused tile, because the gallery's own
    ``eventSelected`` relay put it there the moment the tile took focus. So the freshly opened
    inspector received no selection and sat on the detection's first event (measured: asked for
    event 5, got event 0)."""
    window, _doc_id, df = detected
    target = int(df["event_id"].iloc[3])
    window._analysis_panel.select_event(target)      # the row is ALREADY current
    qtbot.wait(10)
    assert window._event_inspector is None

    window._open_inspector_at_event(target)
    qtbot.wait(10)

    assert window._event_inspector is not None
    assert window._event_inspector.current_event_id() == target


def test_opening_the_gallery_at_an_event_lands_on_that_event(detected, qtbot):
    """The mirror, and it needs the same second step for the same reason."""
    window, _doc_id, df = detected
    target = int(df["event_id"].iloc[2])
    window._analysis_panel.select_event(target)
    qtbot.wait(10)

    window._open_gallery_at_event(target)
    qtbot.wait(10)

    assert window._event_gallery is not None
    assert window._event_gallery.current_event_id() == target


# ================================================================================================
# 4. the events table is no longer a dead end
# ================================================================================================

def _right_click_row(window, df, row: int, choose: str):
    """Right-click the table's ``row`` and pick the menu entry whose text contains ``choose``.

    ``QMenu.exec`` blocks forever offscreen and is a Shiboken built-in that cannot be
    monkeypatched, so the table exposes the same ``_exec_menu`` seam the gallery's context menu
    does and the test patches THAT — the handler under test stays entirely real.
    """
    table = window._analysis_panel._table
    table.select_event(int(df["event_id"].iloc[row]))
    index = table.view.currentIndex()
    assert index.isValid()
    picked: list[str] = []

    def _fake_exec(menu, _pos):
        for act in menu.actions():
            if choose.lower() in act.text().lower().replace("&", ""):
                picked.append(act.text())
                return act
        return None

    table._exec_menu = _fake_exec      # type: ignore[method-assign]
    table._on_context_menu(table.view.visualRect(index).center())
    assert picked, f"no context-menu entry matched {choose!r}"


def test_the_events_table_offers_the_event_inspector(detected, qtbot):
    """The Analysis panel is where a detection lands, and it offered no route to either
    curation window at all — no double-click, no context menu."""
    window, _doc_id, df = detected
    seen: list[int] = []
    window._analysis_panel._table.inspectRequested.connect(lambda e: seen.append(int(e)))

    _right_click_row(window, df, 2, "inspector")
    qtbot.wait(10)

    target = int(df["event_id"].iloc[2])
    assert seen == [target]
    assert window._event_inspector is not None
    assert window._event_inspector.current_event_id() == target


def test_the_events_table_offers_the_event_gallery(detected, qtbot):
    window, _doc_id, df = detected
    seen: list[int] = []
    window._analysis_panel._table.galleryRequested.connect(lambda e: seen.append(int(e)))

    _right_click_row(window, df, 1, "gallery")
    qtbot.wait(10)

    target = int(df["event_id"].iloc[1])
    assert seen == [target]
    assert window._event_gallery is not None
    assert window._event_gallery.current_event_id() == target


def test_a_right_click_on_empty_space_opens_nothing(detected, qtbot):
    """Below the last row there is no event to open, and a menu there would be a menu about
    nothing."""
    window, _doc_id, _df = detected
    table = window._analysis_panel._table
    execed: list[bool] = []
    table._exec_menu = lambda *_a: execed.append(True)     # type: ignore[method-assign]

    table._on_context_menu(QtCore.QPoint(5, 100_000))

    assert execed == []
    assert window._event_inspector is None
    assert window._event_gallery is None
