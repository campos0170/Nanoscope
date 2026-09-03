"""P11 (sweep-4 leftover) — an UNSTAMPED project restored over a shrunken recording still
counts the events that no longer fit.

:func:`nanoscope.core.identity.describe_restored_source` was built (and unit-pinned by
``tests/core/test_s4_identity_restored_source.py``) as "the seam both surfaces pass
through", splitting the two claims a restore can make: "this file changed" needs EVIDENCE
(a saved fingerprint), while "9 of your 11 restored events lie past the end of this
recording" is a comparison against the sample count in hand and needs no stamp at all.

But the seam had NO CALLER: ``MainWindow._note_source_change`` still gated everything on
the stamp — ``saved is None`` returned early, and even with a stamp the past-end count
was composed only inside the fingerprint-mismatch branch. Measured through a wired
MainWindow: strip the fingerprint (exactly what every pre-v4 project carries —
``MIN_READ_VERSION`` is frozen at 1 so they all still open), shrink the source 60k→20k,
reopen — 3 events reattached, 1 past EOF, info bar EMPTY. The GUI now composes through
the seam; the stamped sentence is unchanged to the character (the seam's own contract).
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

from nanoscope.core.project import load_project, save_project
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0


def _write_csv(path: Path, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]:
        if e < n:
            x[s:e] -= 30.0
    t = np.arange(n, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": x}).to_csv(path, index=False)


def _drain(qapp, n=8):
    from PySide6 import QtCore
    for _ in range(n):
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 25)


def _wait(qapp, cond, timeout=30.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        _drain(qapp, 2)
        if cond():
            return True
        time.sleep(0.002)
    return False


@pytest.fixture
def saved_project(qapp, qtbot, tmp_path):
    """A real project — CSV opened through the real pipeline, real detection, real save."""
    w = MainWindow()
    qtbot.addWidget(w)
    csv = tmp_path / "rec.csv"
    _write_csv(csv, 60_000)
    w.open_paths([str(csv)])
    assert _wait(qapp, lambda: len(w._session.documents()) == 1), "open timed out"
    doc = w._session.documents()[0]
    w._analysis_panel.set_selection(doc.doc_id, 0)
    _drain(qapp)
    w._analysis_panel.trigger_detection()
    assert _wait(qapp, lambda: any(True for _ in w._session.results.all()), 60), (
        "detection timed out")
    _drain(qapp)
    proj = tmp_path / "study.nanoscope"
    assert w._write_project(str(proj))
    yield w, csv, proj
    w._analysis_controller.abort()


def _restore(qapp, w, proj) -> list[str]:
    notes: list[str] = []
    bar_type = type(w._info_bar)
    real = bar_type.show_message

    def spy(self, text, *a, **k):
        notes.append(text)
        return real(self, text, *a, **k)

    bar_type.show_message = spy
    try:
        w._open_project(str(proj))
        assert _wait(qapp, lambda: not w._restoring, 60), "restore timed out"
        _drain(qapp, 20)
    finally:
        bar_type.show_message = real
    return notes


def _strip_fingerprint(proj: Path) -> None:
    state = load_project(proj)
    for d in state.documents:
        d.source = None
    save_project(proj, state)
    assert load_project(proj).documents[0].source is None


def test_an_unstamped_project_still_names_the_events_past_the_end(qapp, saved_project):
    w, csv, proj = saved_project
    _strip_fingerprint(proj)
    time.sleep(0.02)
    _write_csv(csv, 20_000, seed=1)          # the source shrank under the project

    notes = _restore(qapp, w, proj)

    said = " ".join(notes)
    assert "lie past the end of this recording" in said, (
        f"the past-end count needs no fingerprint and must be said: {notes!r}")
    assert "carries no fingerprint" in said, (
        "the sentence must say WHY the file itself goes unnamed")
    assert "is not the file this project was saved from" not in said, (
        "with no stamp there is no evidence for a claim about the FILE")


def test_a_stamped_project_keeps_its_sentence_to_the_character(qapp, saved_project):
    """The seam's own contract: the stamped wording did not move."""
    w, csv, proj = saved_project
    time.sleep(0.02)
    _write_csv(csv, 20_000, seed=1)

    notes = _restore(qapp, w, proj)

    said = " ".join(notes)
    assert "is not the file this project was saved from" in said, notes
    assert "now holds 20,000 samples where the project recorded 60,000" in said, notes
    assert "lie past the end of this recording" in said, notes
    assert "Re-detect on it, or point the project at the original recording" in said


def test_an_unchanged_source_restores_in_silence(qapp, saved_project):
    """CONTROL both ways: silence still means 'nothing to say', stamped or not."""
    w, csv, proj = saved_project
    notes = _restore(qapp, w, proj)
    assert not any("past the end" in n or "not the file" in n for n in notes), notes

    _strip_fingerprint(proj)
    notes = _restore(qapp, w, proj)
    assert not any("past the end" in n or "not the file" in n or "fingerprint" in n
                   for n in notes), notes
