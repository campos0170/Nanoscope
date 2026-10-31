"""Edit ▸ Undo takes back calibration, the time base and the acquisition bandwidth: design
§3.2-3.5 and §5.4, the owner's decisions 1, 2 and 6.

Before this, none of the three could be taken back. A calibration Apply or Reset rewrote or
deleted ``<datafile>.npcal.json`` -- a user-owned file beside the raw data, whose note a
project cannot carry -- and the code said so: a Reset that deleted it "is not undoable". A
sample-rate declaration moved every seconds-based number on screen with no way back but
re-typing the old one, and an acquisition override likewise.

What is pinned here:

* each of the three is one named step that Ctrl+Z takes back and Ctrl+Shift+Z puts forward;
  a calibration's undo puts the sidecar back EXACTLY as it was -- the user's own bytes, a
  deleted file and its note included -- WITHOUT asking (decision 6), the status line and the
  Calibration panel's own line name what happened to the file, and a file that changed since
  the step is left alone, said so;
* the first change on a freshly opened recording is a step too: the session announces a
  setting only after it changes, so the recording's settings are adopted when it opens; a
  project restore is no step at all, and closing a recording takes its steps with it;
* one journal orders settings and curation, most recent first, and redo replays in order;
* a settings step ends its own redo branch and no other (decision 2);
* Ctrl+Z takes back the newest step of the recording on screen and never changes one that is
  not on screen (decision 1), and the menu says so;
* the 30-step ceiling is announced for a setting as it is for a table;
* Re-measure refuses a table whose calibration moved (defect 8), and undoing the calibration
  makes the table current again: undo never re-measures, the stamps do the rest;
* a Nyquist clamp on arrival, or after a declaration, is said on the status line (§5.3).

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``, so every sidecar
lands in the test's own folder. The Calibration panel's two confirmations are answered
through their injectable seams, never by a live modal.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.core import atomic_write
from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.calibration import sidecar_path
from nanoscope.core.document import TraceDocument
from nanoscope.core.history import MAX_DEPTH
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.segment import SegmentDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.sample_rate_dialog import SampleRateDialog

from .test_dataloss_regressions import _doc as _csv_document, _event_trace, _write_csv

FS = 50_000.0
I0 = -1_000.0
N = 60_000
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600), (30_000, 30_500)]
NOTE = "gain measured against the 1 GΩ standard, 2026-10-30"


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _open(w: MainWindow, tmp_path, name: str = "pore.dat", *, fs: float = FS,
          detect: bool = False, channels: int = 1) -> str:
    """Open a synthetic recording of ``channels`` current channels at ``tmp_path / name`` and
    put its channel 0 on screen, with a Schmitt table when ``detect``."""
    infos = tuple(ChannelInfo(index=i, name="current" if i == 0 else f"current {i}",
                              raw_units="pA", kind=ChannelKind.CURRENT)
                  for i in range(channels))
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META,
                         channels=infos)
    doc = TraceDocument(meta=meta, channels=tuple(_signal(i) for i in range(channels)))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    if detect:
        # Through the detector module with the form's own parameters, so the table carries
        # every stamp a run gives it (rate, calibration, estimator) and starts current.
        session = w._session
        w._session.add_result(DetectSchmitt().run(SimpleAnalysisContext(
            session.calibrated(doc_id, 0), session.document(doc_id).meta,
            w._analysis_panel._form_params())))
        assert len(w._analysis_panel.current_detection().payload.df) == len(_SPANS)
    return doc_id


def _write_sidecar(path, *entries: tuple[float, str, str]):
    """A user's own sidecar beside a recording that is about to open -- version 1, as their
    files are, hand-indented -- holding ``(gain, units, note)`` for channel 0, 1, ..."""
    sc = sidecar_path(path)
    sc.write_text(json.dumps({"version": 1, "channels": {
        str(ch): {"gain": gain, "offset": 0.0, "target_units": units, "note": note}
        for ch, (gain, units, note) in enumerate(entries)}}, indent=4))
    return sc


def _panel(w: MainWindow) -> tuple:
    """The main window's own Calibration panel with both confirmations answered yes, and
    every question it asked recorded."""
    panel = w._calibration_panel
    asked: list = []
    panel._confirm_sidecar_removal = lambda path, **kw: (
        asked.append(("remove", kw.get("action"))), True)[1]
    panel._confirm_note_loss = lambda path, lost, **kw: (
        asked.append(("note", kw.get("action"))), True)[1]
    return panel, asked


def _apply(panel, gain: str) -> None:
    panel._gain.setText(gain)
    panel._apply()


def _gain(w: MainWindow, doc_id: str) -> float:
    return w._session.calibration(doc_id, 0).gain


def _undo(w: MainWindow) -> None:
    """Edit ▸ Undo, the door ⌘Z reaches (``test_editing_undo_reach`` pins the keys)."""
    w._act_undo.trigger()


def _redo(w: MainWindow) -> None:
    w._act_redo.trigger()


def _status(w: MainWindow) -> str:
    return w.statusBar().currentMessage()


def _accepted(w: MainWindow, event_id: int) -> bool:
    df = w._analysis_panel.current_detection().payload.df
    return bool(df.loc[df["event_id"] == event_id, "accepted"].iloc[0])


def _reject(w: MainWindow, event_id: int) -> None:
    assert w._analysis_panel._table.set_accepted_by_id(event_id, False)
    assert not _accepted(w, event_id)


# ================================================================================================
# calibration, and the sidecar with it
# ================================================================================================

def test_apply_then_undo_puts_back_the_calibration_and_the_sidecar_byte_for_byte(
        window, tmp_path, blocked_modals):
    doc_id = _open(window, tmp_path)
    panel, asked = _panel(window)
    sidecar = sidecar_path(tmp_path / "pore.dat")
    _apply(panel, "0.5")
    first = sidecar.read_bytes()
    _apply(panel, "2")
    second = sidecar.read_bytes()
    assert second != first
    assert window._act_undo.isEnabled() and window._act_undo.text() == "Undo Apply calibration"

    _undo(window)
    assert _gain(window, doc_id) == 0.5
    assert sidecar.read_bytes() == first, "the undo left the sidecar as the undone Apply wrote it"
    assert _status(window) == f"Undone: Apply calibration · saved to {sidecar.name}."
    assert panel._gain.text() == "0.5", "the Calibration panel kept showing the undone gain"
    # ...and the panel's own sidecar line, which outlasts the status bar, says what the undo
    # did: it used to go on reading "Applied · saved to …" whatever the undo had done.
    assert panel.status_text() == f"Undone: Apply calibration · saved to {sidecar.name}"
    assert window._act_redo.text() == "Redo Apply calibration"

    _redo(window)
    assert _gain(window, doc_id) == 2.0 and sidecar.read_bytes() == second
    assert _status(window) == f"Redone: Apply calibration · saved to {sidecar.name}."
    assert panel.status_text() == f"Redone: Apply calibration · saved to {sidecar.name}"

    _undo(window)
    _undo(window)
    assert _gain(window, doc_id) == 1.0
    assert not sidecar.exists(), "back before the first Apply there was no sidecar"
    assert _status(window) == (f"Undone: Apply calibration · {sidecar.name} deleted (all "
                               f"channels are back at default).")
    assert panel.status_text() == (f"Undone: Apply calibration · {sidecar.name} deleted (all "
                                   f"channels are back at default)"), (
        "the panel still described a file the undo had just deleted")
    assert asked == [] and blocked_modals == [], "an undo asked before rewriting the sidecar"


def test_undoing_a_reset_that_deleted_the_users_sidecar_puts_their_own_file_back(
        window, tmp_path, blocked_modals):
    """Design §5.4. The Reset asks before it deletes the user's file (it always has); the
    undo of that Reset puts the file back EXACTLY as it was -- their own version-1 file, in
    their own layout, with the note no project can carry -- and the redo deletes it again,
    neither of them asking. The undo used to write the app's re-derivation of it instead."""
    sidecar = _write_sidecar(tmp_path / "pore.dat", (0.5, "pA", NOTE))
    theirs = sidecar.read_bytes()
    doc_id = _open(window, tmp_path)
    assert (_gain(window, doc_id), window._session.calibration(doc_id, 0).note) == (0.5, NOTE)
    panel, asked = _panel(window)

    panel._reset()
    assert asked == [("remove", "Reset")]
    assert not sidecar.exists(), "setup: Reset to the opened default deletes the sidecar"
    assert window._act_undo.text() == "Undo Reset calibration"

    _undo(window)
    assert sidecar.read_bytes() == theirs, "the user's file came back rewritten"
    assert _gain(window, doc_id) == 0.5
    assert _status(window) == f"Undone: Reset calibration · saved to {sidecar.name}."

    _redo(window)
    assert not sidecar.exists(), "the redo of the Reset left the file in place"
    assert _gain(window, doc_id) == 1.0
    assert window._session.calibration(doc_id, 0).note == NOTE, "Reset keeps the note"
    assert _status(window) == (f"Redone: Reset calibration · {sidecar.name} deleted (all "
                               f"channels are back at default).")
    assert asked == [("remove", "Reset")], "the undo or the redo asked again"
    assert blocked_modals == []


def test_an_undo_does_not_delete_a_users_file_whose_note_sits_at_default_numbers(
        window, tmp_path, blocked_modals):
    """The user's file holds the channel at the very numbers it opens with, and a note. Apply
    keeps the note in the file; its undo returned the channel to numbers the app never
    stores, so re-deriving the file from the calibration deleted it, note and all, unasked,
    where Apply would have asked first (measured in B1's review). The undo puts it back."""
    sidecar = _write_sidecar(tmp_path / "pore.dat", (1.0, "", NOTE))
    theirs = sidecar.read_bytes()
    doc_id = _open(window, tmp_path)
    panel, asked = _panel(window)
    _apply(panel, "2")
    assert json.loads(sidecar.read_text())["channels"]["0"]["note"] == NOTE

    _undo(window)
    assert _gain(window, doc_id) == 1.0
    assert sidecar.exists(), "the undo deleted the user's file and its note"
    assert sidecar.read_bytes() == theirs
    assert _status(window) == f"Undone: Apply calibration · saved to {sidecar.name}."
    assert asked == [] and blocked_modals == []


def test_an_undo_keeps_a_sibling_channels_note_in_the_users_file(window, tmp_path):
    """Two channels in one file, channel 1 at its opened-default numbers with a note. An
    Apply on channel 1 kept both notes; re-deriving the file for its undo left channel 1
    out (its numbers are the default again), so the note went from disk under "saved to
    …", nothing asked (B1's review). The undo puts the user's file back as it was."""
    sidecar = _write_sidecar(tmp_path / "two.dat", (0.5, "pA", "channel 0, 1 GΩ standard"),
                             (1.0, "", NOTE))
    theirs = sidecar.read_bytes()
    doc_id = _open(window, tmp_path, "two.dat", channels=2)
    window._on_selection(doc_id, 1)
    panel, asked = _panel(window)
    _apply(panel, "2")

    _undo(window)
    assert window._session.calibration(doc_id, 1).gain == 1.0
    assert sidecar.read_bytes() == theirs, "the undo dropped channel 1's note out of the file"
    assert asked == []
    _redo(window)
    assert json.loads(sidecar.read_text())["channels"]["1"]["gain"] == 2.0


def test_a_sidecar_that_changed_since_the_step_is_left_as_it_is_and_the_line_says_so(
        window, tmp_path):
    """The file is put back only when it is as the step left it. Redo is per scope (the
    owner's decision 2): undo channel 1's Apply, then channel 0's, Apply again on channel 0,
    and channel 1's redo survives -- but the file now holds channel 0's NEW gain, while the
    file channel 1's step left holds channel 0's OLD one. Put back, it would have stored a
    gain the screen no longer shows. The redo moves channel 1 in the session, leaves the
    file alone and says so; an edit made outside the app is left alone the same way."""
    doc_id = _open(window, tmp_path, "two.dat", channels=2)
    sidecar = sidecar_path(tmp_path / "two.dat")
    panel, _asked = _panel(window)
    _apply(panel, "2")                                     # channel 0
    window._on_selection(doc_id, 1)
    _apply(panel, "3")                                     # channel 1
    _undo(window)
    _undo(window)
    assert not sidecar.exists()
    window._on_selection(doc_id, 0)
    _apply(panel, "4")                                     # channel 0 again: ITS redo ends
    current = sidecar.read_bytes()
    assert window._act_redo.text() == "Redo Apply calibration"

    _redo(window)                                          # channel 1's, which survived
    cal = window._session.calibration
    assert (cal(doc_id, 0).gain, cal(doc_id, 1).gain) == (4.0, 3.0)
    assert sidecar.read_bytes() == current, "the redo rewrote a file that changed since"

    def left(name: str) -> str:
        return (f"{name} left as it is: it has changed since this step. The calibration just "
                f"put back is in this session only (and in a saved project); Apply it in the "
                f"Calibration panel to store it in {name}.")

    assert _status(window) == f"Redone: Apply calibration · {left(sidecar.name)}"

    other = _open(window, tmp_path, "pore.dat")
    _apply(panel, "2")
    by_hand = _write_sidecar(tmp_path / "pore.dat", (5.0, "pA", "typed in a text editor"))
    edited = by_hand.read_bytes()
    _undo(window)
    assert _gain(window, other) == 1.0
    assert by_hand.read_bytes() == edited, "the undo overwrote an edit made outside the app"
    assert _status(window) == f"Undone: Apply calibration · {left(by_hand.name)}"


def test_an_undo_that_cannot_write_the_sidecar_says_so_and_a_redo_finds_it_in_place(
        window, tmp_path, monkeypatch):
    """A failed write is said in the panel's words, led by what the press did rather than
    by a step called "… to this session only"; the file keeps the undone step's values. A
    redo then finds the file already as it puts it, and has nothing to write."""
    doc_id = _open(window, tmp_path)
    panel, _asked = _panel(window)
    sidecar = sidecar_path(tmp_path / "pore.dat")
    _apply(panel, "0.5")
    _apply(panel, "2")
    applied = sidecar.read_bytes()

    def disk_full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(atomic_write.os, "replace", disk_full)
        _undo(window)
    assert _gain(window, doc_id) == 0.5 and sidecar.read_bytes() == applied
    assert _status(window) == (
        f"Undone: Apply calibration, in this session only · could NOT update {sidecar.name} "
        f"— No space left on device. The calibration is NOT stored next to your data and "
        f"will be gone when you reopen the recording.")

    _redo(window)
    assert _gain(window, doc_id) == 2.0 and sidecar.read_bytes() == applied
    assert _status(window) == "Redone: Apply calibration."


def test_the_first_apply_on_a_freshly_opened_recording_is_undoable(window, tmp_path):
    """The session announces a calibration only AFTER it changes. A history that first saw
    the channel on that announcement would take the first Apply as its starting point, and
    the first Apply -- the one most likely to be a mistake -- could never be undone."""
    doc_id = _open(window, tmp_path)
    panel, _asked = _panel(window)
    _apply(panel, "2")
    assert window._act_undo.isEnabled(), window._act_undo.text()
    assert window._act_undo.text() == "Undo Apply calibration"
    _undo(window)
    assert _gain(window, doc_id) == 1.0
    assert not sidecar_path(tmp_path / "pore.dat").exists()
    assert not window._act_undo.isEnabled()


def test_an_apply_that_moves_no_number_is_no_step(window, tmp_path):
    """A step is a change of the calibration (design §3.2); what its commit did to the file
    rides along and is not compared. Compared, Apply pressed twice on the same values -- a
    file write, no number moved -- made a second "Apply calibration" that took nothing back."""
    doc_id = _open(window, tmp_path)
    panel, _asked = _panel(window)
    _apply(panel, "2")
    _apply(panel, "2")
    assert window._settings_history.undo_depth(("calibration", doc_id, 0)) == 1
    _undo(window)
    assert _gain(window, doc_id) == 1.0
    assert not sidecar_path(tmp_path / "pore.dat").exists()


def test_a_calibration_change_nobody_announced_is_named_from_what_moved(window, tmp_path):
    """Only the Calibration panel names its steps ("Apply", "Reset"); a change that came
    another way is named from the two calibrations, so the menu never says nothing."""
    doc_id = _open(window, tmp_path)
    session = window._session
    base = session.calibration(doc_id, 0)
    session.set_calibration(doc_id, 0, replace(base, gain=2.0), write_sidecar=False)
    assert window._act_undo.text() == "Undo Change calibration (gain 1 → 2)"
    session.set_calibration(doc_id, 0, replace(base, gain=2.0, kind=ChannelKind.CURRENT),
                            write_sidecar=False)
    assert window._act_undo.text() == "Undo Treat channel as current"
    session.set_calibration(doc_id, 0, replace(base, gain=2.0, kind=ChannelKind.CURRENT,
                                               note=NOTE), write_sidecar=False)
    assert window._act_undo.text() == "Undo Edit calibration note"
    _undo(window)
    assert window._act_undo.text() == "Undo Treat channel as current"


# ================================================================================================
# the time base and the acquisition bandwidth
# ================================================================================================

def _declare(w: MainWindow, doc_id: str, mode: str, rate: str = "") -> None:
    """The declaration dialog's Apply, as the user drives it (never exec'd)."""
    dialog = SampleRateDialog(w._session, doc_id)
    dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
    if rate:
        dialog.rate.setText(rate)
    dialog.apply()
    dialog.deleteLater()


def test_a_time_base_declaration_undoes_and_redoes_to_and_from_the_files_own(window, tmp_path):
    doc_id = _open(window, tmp_path)
    session = window._session
    _declare(window, doc_id, "sample_rate", "25000")
    declared = session.sample_rate_declaration(doc_id)
    assert session.document(doc_id).meta.sample_rate_hz == 25_000.0
    assert window._act_undo.text() == "Undo Declare 25,000 Hz"

    _undo(window)
    assert session.sample_rate_declaration(doc_id) is None
    assert session.document(doc_id).meta.sample_rate_hz == FS, "the file's own rate is back"
    assert _status(window) == "Undone: Declare 25,000 Hz."
    assert window._act_redo.text() == "Redo Declare 25,000 Hz"

    _redo(window)
    assert session.sample_rate_declaration(doc_id) == declared
    assert session.document(doc_id).meta.sample_rate_hz == 25_000.0

    _declare(window, doc_id, "file")
    assert session.sample_rate_declaration(doc_id) is None
    assert window._act_undo.text() == "Undo Use the file's own time base"
    _undo(window)
    assert session.sample_rate_declaration(doc_id) == declared, (
        "the way back to the file's own time base is a step too")
    assert session.document(doc_id).meta.sample_rate_hz == 25_000.0


def test_an_acquisition_override_undoes_and_redoes(window, tmp_path):
    doc_id = _open(window, tmp_path)
    session = window._session
    override = AcquisitionBandwidth(cutoff_hz=10_000.0, status="reported", source="user")
    session.set_acquisition_bandwidth(doc_id, 0, override)
    assert window._act_undo.text() == "Undo Change acquisition bandwidth"

    _undo(window)
    assert session.acquisition_bandwidth_override(doc_id, 0) is None
    _redo(window)
    assert session.acquisition_bandwidth_override(doc_id, 0) == override

    session.set_acquisition_bandwidth(doc_id, 0, None)
    assert window._act_undo.text() == "Undo Use the file's acquisition bandwidth"
    _undo(window)
    assert session.acquisition_bandwidth_override(doc_id, 0) == override


# ================================================================================================
# one journal, separate redo
# ================================================================================================

def test_one_journal_orders_settings_and_curation_and_redo_replays_in_order(window, tmp_path):
    """Reject an event, then Apply a calibration: the first Ctrl+Z takes back the calibration
    (the most recent thing that happened), the second the rejection; redo replays them in
    the order they were made."""
    doc_id = _open(window, tmp_path, detect=True)
    panel, _asked = _panel(window)
    event_id = int(window._analysis_panel.current_detection().payload.df["event_id"].iloc[1])
    _reject(window, event_id)
    _apply(panel, "2")
    assert window._act_undo.text() == "Undo Apply calibration"

    _undo(window)
    assert _gain(window, doc_id) == 1.0 and not _accepted(window, event_id)
    assert window._act_undo.text() == "Undo Curate event"
    _undo(window)
    assert _accepted(window, event_id)

    assert window._act_redo.text() == "Redo Curate event"
    _redo(window)
    assert not _accepted(window, event_id) and _gain(window, doc_id) == 1.0
    assert window._act_redo.text() == "Redo Apply calibration"
    _redo(window)
    assert _gain(window, doc_id) == 2.0


def test_a_settings_step_leaves_an_event_redo_alone_and_the_other_way_round(window, tmp_path):
    """Decision 2: redo is separate for each scope. A calibration applied after an undone
    curation does not throw the curation's redo away, nor a curation an undone calibration;
    a new calibration does end its own scope's redo."""
    doc_id = _open(window, tmp_path, detect=True)
    panel, _asked = _panel(window)
    ids = window._analysis_panel.current_detection().payload.df["event_id"].to_numpy()

    _reject(window, int(ids[0]))
    _undo(window)
    _apply(panel, "2")
    assert window._act_redo.text() == "Redo Curate event", window._act_redo.text()
    _redo(window)
    assert not _accepted(window, int(ids[0])), "the calibration step ended the curation's redo"

    _undo(window)                                  # the curation's redo again
    _undo(window)                                  # and the calibration
    assert _gain(window, doc_id) == 1.0 and _accepted(window, int(ids[0]))
    _reject(window, int(ids[2]))                   # a new event step
    assert window._act_redo.text() == "Redo Apply calibration", window._act_redo.text()
    _redo(window)
    assert _gain(window, doc_id) == 2.0, "the curation step ended the calibration's redo"

    _undo(window)
    _apply(panel, "3")                             # a new step in the SAME scope
    assert window._settings_history.redo_depth(("calibration", doc_id, 0)) == 0
    assert window._act_redo.text() != "Redo Apply calibration"


def test_ctrl_z_never_changes_a_recording_that_is_not_on_screen(window, tmp_path):
    """Decision 1. The step on A is offered only while A is on screen; on B the menu says
    there is nothing on THIS recording, and a press (from any window) changes nothing."""
    doc_a = _open(window, tmp_path, "a.dat")
    panel, _asked = _panel(window)
    _apply(panel, "2")
    doc_b = _open(window, tmp_path, "b.dat")
    assert not window._act_undo.isEnabled()
    assert window._act_undo.text() == "Undo — nothing on this recording"

    window._undo_events()                          # what ⌘Z in another window reaches
    assert _gain(window, doc_a) == 2.0, "Ctrl+Z changed a recording that is not on screen"
    assert _gain(window, doc_b) == 1.0

    window._on_selection(doc_a, 0)
    assert window._act_undo.isEnabled() and window._act_undo.text() == "Undo Apply calibration"
    _undo(window)
    assert _gain(window, doc_a) == 1.0


def test_ctrl_z_takes_back_the_newest_step_of_the_recording_on_screen(window, tmp_path):
    """Decision 1 and design §3.1: a press takes back the most recent step that belongs to
    the recording on screen. The journal's event entries named no table, so a curation made
    on B after A's calibration stood in for A's own table: back on A, the first Ctrl+Z
    re-accepted A's OLDER event and left the calibration where it was (B1's review)."""
    doc_a = _open(window, tmp_path, "a.dat", detect=True)
    panel, _asked = _panel(window)
    on_a = int(window._analysis_panel.current_detection().payload.df["event_id"].iloc[0])
    _reject(window, on_a)
    _apply(panel, "2")                             # A's newest step
    doc_b = _open(window, tmp_path, "b.dat", detect=True)
    on_b = int(window._analysis_panel.current_detection().payload.df["event_id"].iloc[0])
    _reject(window, on_b)                          # the newest step of all, on B

    window._on_selection(doc_a, 0)
    assert window._act_undo.text() == "Undo Apply calibration"
    _undo(window)
    assert _gain(window, doc_a) == 1.0 and not _accepted(window, on_a), (
        "Ctrl+Z on A passed over A's newest step")
    assert window._act_undo.text() == "Undo Curate event"
    _undo(window)
    assert _accepted(window, on_a)
    assert not window._act_undo.isEnabled(), "B's curation was offered on A"

    window._on_selection(doc_b, 0)
    assert not _accepted(window, on_b)
    assert window._act_undo.text() == "Undo Curate event"


def test_the_30_step_ceiling_is_announced_for_a_calibration(window, tmp_path):
    doc_id = _open(window, tmp_path)
    session = window._session
    base = session.calibration(doc_id, 0)
    for i in range(MAX_DEPTH + 1):
        session.set_calibration(doc_id, 0, replace(base, gain=1.0 + 0.5 * (i + 1)),
                                write_sidecar=False)
    assert _status(window) == ("Undo history for “pore.dat” is at its 30-step ceiling — its "
                               "oldest step has been dropped and cannot be undone.")
    for _ in range(MAX_DEPTH):
        _undo(window)
    assert _gain(window, doc_id) == 1.5, "the oldest step was dropped, so this is the floor"
    assert not window._act_undo.isEnabled()
    assert window._act_undo.text() == ("Undo — this recording's oldest settings history was "
                                       "dropped at the 30-step ceiling (1 step lost)")


def _cut_segment(w: MainWindow, parent: str) -> str:
    """Cut a segment out of ``parent`` through the window's own door, as the user does."""
    w._analysis_panel.measure_panel.set_active(True)
    x = w._session.x_axis(parent).values
    w._trace_view.set_measure_region(float(x[len(x) // 6]), float(x[len(x) // 2]))
    w._ask_segment_name = lambda default: "cut"
    w._create_segment_from_selection()
    w._analysis_panel.measure_panel.set_active(False)
    return w._session.segment_documents()[0].doc_id


def test_a_segment_inherits_its_parents_calibration_without_a_step(window, tmp_path):
    """A segment is handed its parent's calibration after it is announced, silently, by
    design (no sidecar, no signal). That inheritance is not an edit, and the next commit
    must not record it as one."""
    parent = _open(window, tmp_path)
    panel, _asked = _panel(window)
    _apply(panel, "2")
    segment = _cut_segment(window, parent)
    assert _gain(window, segment) == 2.0, "setup: the segment inherits the parent's gain"

    window._on_edit_menu_about_to_show()           # commit everything, as the menu does
    assert not window._settings_history.can_undo(("calibration", segment, 0)), (
        "the inherited calibration was recorded as an edit of the segment")
    window._on_selection(segment, 0)
    assert window._act_undo.text() == "Undo — nothing on this recording"


# ================================================================================================
# the stack's lifetime: closing a recording, opening a project
# ================================================================================================

def test_closing_a_recording_takes_its_settings_steps_and_journal_entries_with_it(
        window, tmp_path):
    """``drop_doc`` on close (design §3.3): a step that outlived its recording could be
    written nowhere, and the menu would go on saying "nothing on this recording" about it.
    The journal lets go of the recording's entries with it."""
    doc_a = _open(window, tmp_path, "a.dat")
    panel, _asked = _panel(window)
    _apply(panel, "2")
    _apply(panel, "3")
    _undo(window)                                  # A holds a step each way
    _open(window, tmp_path, "b.dat")
    assert window._act_undo.text() == "Undo — nothing on this recording"

    window._session.close_document(doc_a)
    assert window._settings_history.scopes_with_history() == ()
    assert (window._act_undo.text(), window._act_redo.text()) == ("Undo", "Redo")
    assert not [e for e in window._undo_journal + window._redo_journal if doc_a in str(e)], (
        "the journal kept entries of a recording that is closed")


def test_a_project_restore_is_no_step_and_open_project_starts_the_stack_afresh(
        window, tmp_path, qtbot, blocked_modals):
    """Recording is suspended while a project restore rebuilds every setting, and Open
    Project clears the stack (design §3.3). Without the suspension a save and reopen left
    three steps nobody made -- the parent's bandwidth override, the segment's calibration and
    its override -- and Edit ▸ Undo offered "Undo Change acquisition bandwidth" straight
    after Open Project (B1's review). The first Apply after the restore is a step, and its
    undo puts the user's file back as the file was, not as the project's calibration."""
    csv = tmp_path / "pore.csv"
    trace = _event_trace()
    _write_csv(csv, trace)
    parent = window._session.add_document(_csv_document(csv, trace))
    window._create_default_layers(window._session.document(parent))
    window._on_selection(parent, 0)
    panel, _asked = _panel(window)
    _apply(panel, "0.5")
    window._session.set_acquisition_bandwidth(parent, 0, AcquisitionBandwidth(
        cutoff_hz=10_000.0, status="reported", source="user"))
    _cut_segment(window, parent)
    history = window._settings_history
    assert len(history.scopes_with_history()) == 2, "setup: the parent holds two steps"
    project = tmp_path / "pore.nanoscope"
    assert window._write_project(str(project))
    # Between sessions the user edits their own file: the project says 0.5, the file 0.75.
    sidecar = _write_sidecar(csv, (0.75, "pA", NOTE))
    theirs = sidecar.read_bytes()

    with qtbot.waitSignal(window.projectRestored, timeout=30_000):
        window._open_project(str(project))
    qtbot.wait(50)
    assert history.scopes_with_history() == (), (
        "after Open Project: the old session's steps survived, or the restore made some")
    window._on_edit_menu_about_to_show()           # commit everything, as the menu does
    assert history.scopes_with_history() == (), "the restore was recorded as edits"
    assert (window._act_undo.isEnabled(), window._act_undo.text()) == (False, "Undo")

    restored = next(d.doc_id for d in window._session.documents()
                    if not isinstance(d, SegmentDocument))
    window._on_selection(restored, 0)
    assert _gain(window, restored) == 0.5, "setup: the project's gain, over the file's own"
    _apply(panel, "3")
    assert window._act_undo.text() == "Undo Apply calibration"
    _undo(window)
    assert _gain(window, restored) == 0.5
    assert sidecar.read_bytes() == theirs, "the undo wrote the project's gain into the file"
    assert blocked_modals == []


# ================================================================================================
# Re-measure and the calibration stamp (defect 8)
# ================================================================================================

def test_remeasure_refuses_a_table_whose_calibration_moved_and_undo_makes_it_current(
        window, tmp_path):
    """Re-measure measured with the live calibration and kept the table's OLD stamp: depths
    under the new gain, stamped as the old one, under a line still calling them stale. It
    now refuses, and says the way forward. Undoing the calibration re-measures nothing: the
    table's stamp matches again, and the line goes."""
    doc_id = _open(window, tmp_path, detect=True)
    ap = window._analysis_panel
    res = ap.current_detection()
    assert ap._stale.isHidden()
    panel, _asked = _panel(window)
    _apply(panel, "2")
    assert not ap._stale.isHidden(), "setup: the calibration change raises the amber line"
    before = res.payload.df.copy(deep=True)

    assert ap.remeasure_active_detection() == "stale_calibration"
    assert _status(window) == ("Nothing was re-measured. Re-run first — the calibration "
                               "changed since this table was detected.")
    assert ap.current_detection().payload.df.equals(before), "the refusal wrote something"

    _undo(window)
    assert _gain(window, doc_id) == 1.0
    assert ap._stale.isHidden(), "the table's stamp matches again, so its line must go"
    assert ap.current_detection() is res, "an undo of a setting re-measured the table"


# ================================================================================================
# a Nyquist clamp is said (design §5.3)
# ================================================================================================

_CLAMP = ("Low-pass 20 kHz is above this recording's Nyquist (12.5 kHz); the form now "
          "reads 12.4999 kHz.")


def _type_low_pass(qtbot, det, text: str) -> None:
    det.filter_enabled.setChecked(True)
    det.filter_cutoff_hz.selectAll()
    qtbot.keyClicks(det.filter_cutoff_hz, text)
    qtbot.keyClick(det.filter_cutoff_hz, Qt.Key_Return)


def test_a_clamp_on_arrival_is_said_and_a_clean_arrival_leaves_the_line_alone(
        window, tmp_path, qtbot):
    fast = _open(window, tmp_path, "fast.dat")
    slow = _open(window, tmp_path, "slow.dat", fs=25_000.0)
    window._on_selection(fast, 0)
    det = window._analysis_panel.detection_panel
    _type_low_pass(qtbot, det, "20000")

    window._on_selection(slow, 0)
    assert _status(window) == _CLAMP

    window.statusBar().showMessage("what the user was reading")
    window._on_selection(fast, 0)
    assert det.filter_cutoff_hz.value() == pytest.approx(20_000.0)
    assert _status(window) == "what the user was reading", (
        "an arrival that clamps nothing cleared the status line")


def test_a_declaration_that_clamps_the_form_says_so(window, tmp_path, qtbot):
    doc_id = _open(window, tmp_path)
    det = window._analysis_panel.detection_panel
    _type_low_pass(qtbot, det, "20000")
    window._session.set_sample_rate_declaration(
        doc_id, SampleRateDeclaration("sample_rate", sample_rate_hz=25_000.0))
    assert _status(window) == _CLAMP
