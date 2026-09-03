"""CS5-01 — a project save never silently reverts somebody else's.

Two windows on one ``.nanoscope`` are a supported shape (packaging/nanoscope_app.py keeps
its log lock alive precisely so a later instance can run), and the save path was silent
last-writer-wins: window A saves a rejection (disk verified), window B presses Ctrl+S,
and A's committed curation leaves the file with zero dialogs while both status bars claim
their save is the file. The atomic layer itself held (49k contended saves, 0 corrupt) —
only the honesty was missing.

The fix is the stamp-lock-and-ask: the window remembers the path, metadata, and SHA-256
of the project file at its last load/save, and ``_write_project`` raises
``_confirm_overwrite_newer_project`` before replacing a file that no longer matches.
Cancel costs nothing (the newer file survives, this window stays dirty); the destructive
answer is never the accident. The final comparison and atomic replacement share one
inter-process lock.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import ProjectState, ResultRec, load_project, save_project
from nanoscope.core.project_guard import project_disk_stamp
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def win(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_A.dat", format_name="synthetic",
                         n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4},
        calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    yield w, doc_id, tmp_path / "shared.nanoscope"
    w._analysis_controller.abort()


def _accepted0(path: Path) -> bool:
    return bool(load_project(path).results[0].events["accepted"].iloc[0])


def _other_window_saves(path: Path) -> None:
    """A concurrent writer, through the app's own pure save: the same project with event 0
    rejected — committed curation this window has never seen."""
    state = load_project(path)
    df = state.results[0].events.copy()
    df.loc[df.index[0], "accepted"] = False
    state.results[0] = ResultRec(
        state.results[0].doc_id, state.results[0].channel,
        state.results[0].module_id, df, provenance=state.results[0].provenance)
    time.sleep(0.02)   # even a same-mtime-quantum filesystem sees the size/mtime move
    save_project(path, state)


def test_a_save_over_a_newer_file_asks_first_and_cancel_preserves_it(win, monkeypatch):
    w, doc_id, proj = win
    assert w._write_project(str(proj))
    assert _accepted0(proj) is True

    _other_window_saves(proj)
    assert _accepted0(proj) is False, "the other window's curation is on disk"

    asked: list[str] = []
    monkeypatch.setattr(
        MainWindow, "_confirm_overwrite_newer_project",
        lambda self, path: (asked.append(path), False)[1])
    ok = w._write_project(str(proj))

    assert asked, "the conflict must raise the question — silence was the defect"
    assert ok is False, "a cancelled save reports no success"
    assert _accepted0(proj) is False, (
        "Cancel must leave the other window's committed curation on disk")


def test_confirming_overwrites_and_restamps(win, monkeypatch):
    w, doc_id, proj = win
    assert w._write_project(str(proj))
    _other_window_saves(proj)

    monkeypatch.setattr(MainWindow, "_confirm_overwrite_newer_project",
                        lambda self, path: True)
    assert w._write_project(str(proj)) is True
    assert _accepted0(proj) is True, "the user chose to overwrite; this window's state wins"

    # ...and the overwrite re-stamped, so the very next save conflicts with nothing.
    monkeypatch.setattr(
        MainWindow, "_confirm_overwrite_newer_project",
        lambda self, path: pytest.fail("no conflict exists after our own save"))
    assert w._write_project(str(proj)) is True


def test_ordinary_resaves_never_see_the_dialog(win, monkeypatch):
    """CONTROL: the guard must be invisible on the app's most ordinary path — save,
    curate, save again, same window."""
    w, doc_id, proj = win
    monkeypatch.setattr(
        MainWindow, "_confirm_overwrite_newer_project",
        lambda self, path: pytest.fail("no concurrent writer, no dialog"))
    assert w._write_project(str(proj))
    assert w._write_project(str(proj))


def test_saving_to_a_fresh_path_is_not_a_conflict(win, monkeypatch, tmp_path):
    """A Save As elsewhere is the overwrite prompt's business, not this guard's."""
    w, doc_id, proj = win
    assert w._write_project(str(proj))
    _other_window_saves(proj)
    monkeypatch.setattr(
        MainWindow, "_confirm_overwrite_newer_project",
        lambda self, path: pytest.fail("a different target path conflicts with nothing"))
    assert w._write_project(str(tmp_path / "elsewhere.nanoscope"))


def test_opening_a_project_stamps_it(win, tmp_path):
    """The stamp is taken at BOTH moments the window's picture of the file is true —
    a completed save (above) and a completed load."""
    w, doc_id, proj = win
    assert w._write_project(str(proj))
    stamp_after_save = w._project_disk_stamp
    assert stamp_after_save is not None

    w._open_project(str(proj))
    assert w._project_disk_stamp is not None
    assert w._project_disk_stamp[0] == str(Path(os.path.realpath(str(proj))))


def test_same_size_same_mtime_content_change_still_asks(win, monkeypatch):
    """Metadata can be forged/restored; the remembered content digest is authoritative."""

    w, doc_id, proj = win
    assert w._write_project(str(proj))
    remembered = w._project_disk_stamp
    assert remembered is not None
    old_stat = proj.stat()
    changed = bytearray(proj.read_bytes())
    changed[len(changed) // 2] ^= 0x01
    proj.write_bytes(changed)
    os.utime(proj, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    forged = project_disk_stamp(proj)
    assert forged is not None
    assert (forged.mtime_ns, forged.size) == (remembered.mtime_ns, remembered.size)
    assert forged.sha256 != remembered.sha256

    asked: list[str] = []
    monkeypatch.setattr(
        MainWindow, "_confirm_overwrite_newer_project",
        lambda self, path: (asked.append(path), False)[1])

    assert w._write_project(str(proj)) is False
    assert asked == [str(proj)]
    assert proj.read_bytes() == bytes(changed)
    assert w._project_disk_stamp == remembered


def test_first_save_detects_file_created_after_absent_preflight(win, monkeypatch, tmp_path):
    """Two first-time Save As attempts cannot both treat the shared name as absent."""

    w, doc_id, _ = win
    target = tmp_path / "new-shared-name.nanoscope"
    original_gather = w._gather_project_state
    other = ProjectState()
    other_bytes: bytes | None = None
    gather_calls = 0

    def _gather_after_other_writer():
        nonlocal gather_calls, other_bytes
        gather_calls += 1
        if gather_calls == 1:
            save_project(target, other)
            other_bytes = target.read_bytes()
        return original_gather()

    asked: list[str] = []
    monkeypatch.setattr(w, "_gather_project_state", _gather_after_other_writer)
    monkeypatch.setattr(
        MainWindow, "_confirm_overwrite_existing_project",
        lambda self, path: (asked.append(path), False)[1])

    assert w._write_project(str(target)) is False
    assert asked == [str(target)]
    assert other_bytes is not None and target.read_bytes() == other_bytes


def test_approval_does_not_cover_a_third_revision(win, monkeypatch):
    """A writer landing while the first question is open requires a fresh answer."""

    w, doc_id, proj = win
    assert w._write_project(str(proj))
    remembered = w._project_disk_stamp
    _other_window_saves(proj)

    third = b"a third process replaced the project while the dialog was open"
    asked = 0

    def _answer(self, path):
        nonlocal asked
        asked += 1
        if asked == 1:
            Path(path).write_bytes(third)
            return True
        return False

    monkeypatch.setattr(MainWindow, "_confirm_overwrite_newer_project", _answer)

    assert w._write_project(str(proj)) is False
    assert asked == 2
    assert proj.read_bytes() == third
    assert w._project_disk_stamp == remembered


def test_open_adopts_the_stamp_returned_with_the_locked_read(win, monkeypatch):
    """Never re-stamp after unlock: that could bless a revision we did not load."""

    w, doc_id, proj = win
    assert w._write_project(str(proj))
    loaded_state = load_project(proj)
    loaded_stamp = project_disk_stamp(proj)
    assert loaded_stamp is not None
    proj.write_bytes(b"writer landed after the read lock was released")
    assert project_disk_stamp(proj) != loaded_stamp

    monkeypatch.setattr(
        "nanoscope.gui.main_window.locked_project_read",
        lambda path, read: (loaded_state, loaded_stamp))
    monkeypatch.setattr(w, "_offer_to_save_unsaved_work", lambda: True)
    monkeypatch.setattr(w, "_clear_session", lambda: None)
    monkeypatch.setattr(w, "_add_recent_project", lambda path: None)
    monkeypatch.setattr(w, "_mark_clean", lambda: None)
    monkeypatch.setattr(w, "_begin_restore", lambda state: None)

    w._open_project(str(proj))

    assert w._project_disk_stamp == loaded_stamp
