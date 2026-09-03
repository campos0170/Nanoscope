"""chaos3-003 — a detection that RAN and found nothing is not "no detection".

``EventGalleryWindow.set_events`` collapses an empty frame to ``None`` so that every
row-indexed path in the window can ask one question — and ``_sync_controls`` then took the
never-ran branch and printed "No detection — run detection to fill the gallery." for a run
that had just completed. That sentence tells the user to repeat the run they have just
finished and withholds its answer, and zero events is a REAL answer: it is what a null
control, a conservative threshold or a quiet pore is supposed to produce
(``docs/USER_GUIDE.md`` §7 makes the AAFT surrogate exactly that test).

It also made three surfaces disagree about one run: the Event inspector, handed the
identical result, says "Detection found no events."; the analysis panel prints that run's QC
checks; the gallery denied the run happened — while its own menu action is enabled ONLY
because a detection exists, so the user walks in through a door labelled "you have a
detection" into a window saying they have none.

These tests drive the real ``EventGalleryWindow`` inside a real ``MainWindow``.
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
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0

#: The sentence the Event inspector already uses for this state (event_inspector.py:648).
#: The gallery's must begin with it -- two windows handed one result may not describe it in
#: two voices.
FOUND_NOTHING = "Detection found no events"
NEVER_RAN = "No detection"


def _trace(n_events: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 300_000
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for i in range(n_events):
        s = 5_000 + i * 20_000
        x[s:s + 400] -= 35.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _window(qtbot, tmp_path: Path, n_events: int):
    csv = tmp_path / f"rec{n_events}.csv"
    arr = _trace(n_events)
    _write_csv(csv, arr)
    w = MainWindow()
    qtbot.addWidget(w)
    doc_id = w._session.add_document(_doc(csv, arr))
    w._create_default_layers(w._session.document(doc_id))
    ch = next(c.index for c in w._session.document(doc_id).meta.channels
              if c.kind is ChannelKind.CURRENT and not c.is_dead)
    w._on_selection(doc_id, ch)
    return w, doc_id, ch


def _open_gallery(w, qtbot):
    assert w._act_event_gallery.isEnabled(), (
        "the gallery action is the door this defect is reached through; it is enabled "
        "only because a detection exists")
    w._act_event_gallery.trigger()
    qtbot.wait(20)
    return w._event_gallery


@pytest.fixture
def win_with_events(qapp, qtbot, tmp_path):
    """A real detection with real events — the control for every assertion below."""
    w, doc_id, ch = _window(qtbot, tmp_path, n_events=8)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    return w, doc_id, ch


# ================================================================================================
# the defect
# ================================================================================================

def test_a_completed_run_that_found_nothing_does_not_claim_no_detection(qapp, qtbot,
                                                                       tmp_path):
    """END TO END, through the real Analysis run: a quiet recording, a real detection, zero
    rows — and the gallery opened for the FIRST time afterwards, from the real menu action."""
    w, doc_id, ch = _window(qtbot, tmp_path, n_events=0)
    # A dwell floor no feature of pure noise can clear: the run completes and finds nothing.
    w._analysis_panel._detection.min_dwell_s.setValue(5e-2)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)

    res = w._analysis_panel.current_detection()
    assert res is not None, "the fixture must produce a COMPLETED run"
    assert len(res.payload.df) == 0, "…that found zero events"

    gal = _open_gallery(w, qtbot)
    text = gal.status.text()
    assert FOUND_NOTHING in text, text
    assert NEVER_RAN not in text, text
    assert "run detection" not in text, text


def test_the_gallery_and_the_inspector_describe_one_result_in_one_voice(win_with_events,
                                                                       qtbot):
    """The two windows are handed the SAME zero-row frame. They may not disagree about what
    it is."""
    w, doc_id, ch = win_with_events
    gal = _open_gallery(w, qtbot)
    w._act_event_inspector.trigger()
    qtbot.wait(20)
    insp = w._event_inspector

    empty = w._analysis_panel.current_detection().payload.df.iloc[0:0]
    gal.set_events(doc_id, ch, empty, "detect.schmitt", {})
    insp.set_events(doc_id, ch, empty)
    qtbot.wait(10)

    # Every QLabel the inspector holds -- by the Qt class, not by ``type(gal.status)``:
    # since sweep-7's fix-up the gallery's status line is its own fixed-height subclass,
    # which the inspector does not use (its notes have a sibling of their own).
    insp_text = " ".join(lbl.text() for lbl in insp.findChildren(QtWidgets.QLabel)
                         if lbl.text())
    assert FOUND_NOTHING in gal.status.text()
    assert FOUND_NOTHING in insp_text, insp_text


# ================================================================================================
# the state that must NOT be lost with it
# ================================================================================================

def test_the_never_ran_sentence_survives_for_the_state_it_belongs_to(win_with_events,
                                                                     qtbot):
    """``clear()`` is the genuinely-never-ran path (the detection went away). Its wording is
    pinned by tests/gui/test_event_gallery.py and must be untouched."""
    w, doc_id, ch = win_with_events
    gal = _open_gallery(w, qtbot)
    gal.clear()
    text = gal.status.text()
    assert NEVER_RAN in text and "run detection to fill the gallery" in text, text
    assert FOUND_NOTHING not in text, text


def test_a_none_frame_is_still_the_never_ran_state(win_with_events, qtbot):
    """``set_events(..., None)`` means "there is nothing here", which is not the same as a
    run that returned nothing — even immediately after one that did."""
    w, doc_id, ch = win_with_events
    gal = _open_gallery(w, qtbot)
    empty = w._analysis_panel.current_detection().payload.df.iloc[0:0]

    gal.set_events(doc_id, ch, empty, "detect.schmitt", {})
    assert FOUND_NOTHING in gal.status.text()

    gal.set_events(doc_id, ch, None, "detect.schmitt", {})
    text = gal.status.text()
    assert NEVER_RAN in text, text
    assert FOUND_NOTHING not in text, (
        "the emptiness of the PREVIOUS run must not describe the absence of this one")


def test_a_real_detection_after_an_empty_one_is_reported_normally(win_with_events, qtbot):
    """The flag is re-derived on every feed, never left over: an empty run followed by a real
    one shows counts, not either empty-state sentence."""
    w, doc_id, ch = win_with_events
    gal = _open_gallery(w, qtbot)
    df = w._analysis_panel.current_detection().payload.df

    gal.set_events(doc_id, ch, df.iloc[0:0], "detect.schmitt", {})
    assert FOUND_NOTHING in gal.status.text()

    gal.set_events(doc_id, ch, df, "detect.schmitt", {})
    text = gal.status.text()
    assert FOUND_NOTHING not in text and NEVER_RAN not in text, text
    assert "accepted" in text, text
    assert gal.event_count() == len(df)
