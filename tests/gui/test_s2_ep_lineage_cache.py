"""EP2-03 — ONE CACHE SLOT PER LINEAGE, NOT PER (doc, channel, module).

``MainWindow._event_refusals`` held ONE entry per (doc, channel, module) with the
lineage token stored in the VALUE: resolving the ledger for a DIFFERENT lineage's frame
overwrote the slot with a fresh empty ledger. Edit ▸ Undo past a re-detection resolves
the old lineage's restored frame as a matter of course — so a single
"Undo Re-run detection" silently and permanently destroyed the newer (or, redone back,
the older) lineage's refusal record: the frame came back with its rows still refused
and its curation exact, while the standing "N kept rejected" disclosure was gone from
the gallery, the events table and the inspector, with no gesture that retired it.
USER_GUIDE 9b promises "An undo restores the table exactly", and
tests/gui/test_bl_refusal_note_lineage.py promises the note survives Edit ▸ Undo/Redo —
neither test ever crossed the "Re-run detection" boundary, which is how the single-slot
overwrite survived.

THE FIX THIS PINS. The lineage token is part of the cache KEY, so lineages coexist;
entries are pruned on documentClosed exactly as before (``_drop_refusals_for_doc``
matches on the doc id, which stays the key's first element).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

import nanoscope.gui.event_gallery as gal_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
_LONG = [(10_000, 20_000), (30_000, 40_000)]
_CLEAN = [(50_000, 50_300), (60_000, 60_250), (70_000, 70_300),
          (90_000, 90_250), (110_000, 110_300)]
_MAX_DWELL_S = 5e-3


def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    x[0:300] -= 30.0
    for s, e in _LONG + _CLEAN:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_s2s5.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _drain(qtbot, ms: int = 150) -> None:
    qtbot.wait(ms)
    QtCore.QCoreApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


def _token(win) -> str | None:
    prov = win._analysis_panel.current_detection().payload.provenance or {}
    return prov.get("detected_at")


def _detect(win, qtbot):
    panel = win._analysis_panel
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    _drain(qtbot)
    return panel.current_detection().payload.df


def _undo_until_token(win, qtbot, want: str | None, limit: int = 8) -> None:
    for _ in range(limit):
        if _token(win) == want:
            return
        assert win._act_undo.isEnabled(), "the undo stack ran out before the target lineage"
        win._act_undo.trigger()
        _drain(qtbot)
    assert _token(win) == want, "undo never reached the target lineage"


def _redo_until_token(win, qtbot, want: str | None, limit: int = 8) -> None:
    for _ in range(limit):
        if _token(win) == want:
            return
        assert win._act_redo.isEnabled(), "the redo stack ran out before the target lineage"
        win._act_redo.trigger()
        _drain(qtbot)
    assert _token(win) == want, "redo never reached the target lineage"


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _doc(tmp_path)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    panel._detection.max_dwell_s.setValue(_MAX_DWELL_S)
    try:
        yield win
    finally:
        win._analysis_controller.abort()


def test_a_single_undo_of_a_rerun_restores_the_standing_disclosure(wired, qtbot):
    """The minimal trigger: refusals under lineage #1, one re-detection, one Edit ▸
    Undo. Resolving #2's fresh ledger must not have destroyed #1's record."""
    win = wired
    df1 = _detect(win, qtbot)
    tok1 = _token(win)
    assert tok1, "the detector no longer stamps detected_at; the lineage has no token"

    win._act_event_gallery.trigger()
    _drain(qtbot)
    gal = win._event_gallery
    gal.reject_all_btn.click()
    _drain(qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()

    df2 = _detect(win, qtbot)              # lineage #2 — a fresh ledger, empty note
    tok2 = _token(win)
    assert tok2 and tok2 != tok1, "the re-run did not mint a new lineage token"
    assert "kept rejected" not in gal.status.text(), \
        "a new detection must take the old note down (BL-2), or this test drives nothing"

    _undo_until_token(win, qtbot, tok1)    # a single "Undo Re-run detection" in practice
    tbl = win._analysis_panel._table
    assert "3 tiles kept rejected" in gal.status.text(), (
        "the undo restored #1's frame but its standing disclosure was destroyed by "
        "resolving #2's ledger:\n" + gal.status.text())
    assert "3 events kept rejected" in tbl.refusal_label.text(), \
        repr(tbl.refusal_label.text())


def test_undo_to_the_old_lineage_and_redo_back_keeps_the_new_lineages_record(
        wired, qtbot):
    """The round trip: refusals under lineage #2, undo to #1 (no cross-lineage bleed),
    redo back to #2 — the record must come back with the frame it describes."""
    win = wired
    _detect(win, qtbot)
    tok1 = _token(win)

    df2 = _detect(win, qtbot)
    tok2 = _token(win)
    assert tok2 != tok1

    win._act_event_gallery.trigger()
    _drain(qtbot)
    gal = win._event_gallery
    win._act_event_inspector.trigger()
    _drain(qtbot)
    insp = win._event_inspector
    gal.reject_all_btn.click()
    _drain(qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()

    _undo_until_token(win, qtbot, tok1)
    # #1 never had a refusal gesture: #2's note over #1's frame would be bleed.
    assert "kept rejected" not in gal.status.text(), (
        "cross-lineage bleed: #2's refusals disclosed over #1's frame:\n"
        + gal.status.text())

    _redo_until_token(win, qtbot, tok2)
    restored = win._analysis_panel.current_detection().payload.df
    refused = [i for i in range(len(restored))
               if not bool(restored["accepted"].iloc[i])
               and not gal_mod._derived_accepted(restored["flag"].iloc[i])]
    assert len(refused) == 3, "the redo did not restore #2's curation over refused rows"
    tbl = win._analysis_panel._table
    assert "3 tiles kept rejected" in gal.status.text(), (
        "the round trip destroyed the newer lineage's refusal record:\n"
        + gal.status.text())
    assert "3 events kept rejected" in tbl.refusal_label.text(), \
        repr(tbl.refusal_label.text())
    assert "3 events kept rejected" in insp._batch_note.text(), \
        repr(insp._batch_note.text())


def test_the_cache_still_prunes_on_document_close(wired, qtbot):
    """The bound the fix must keep: closing the recording drops every lineage's entry
    for that doc — the key change must not have broken the doc-id match."""
    win = wired
    _detect(win, qtbot)
    _detect(win, qtbot)                    # two lineages now cached for this doc
    win._act_event_gallery.trigger()
    _drain(qtbot)
    gal = win._event_gallery
    gal.reject_all_btn.click()
    _drain(qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    keys_before = [k for k in win._event_refusals]
    assert keys_before, "nothing cached; the test drives nothing"
    the_doc = keys_before[0][0]
    win._drop_refusals_for_doc(the_doc)
    assert not [k for k in win._event_refusals if k[0] == the_doc], \
        "documentClosed no longer prunes the lineage cache"
