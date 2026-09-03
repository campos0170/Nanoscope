"""chaos3-004 — "one load per file" survives two case-variant spellings of one path.

``MainWindow.open_paths`` states the policy on the line itself — *"one load per file; never
double the RAM"* — and ``_already_open`` is documented as *"THE ONE PLACE THE OPEN PIPELINE
ASKS 'IS THIS FILE ALREADY OPEN?'"*. It asked with ``str(path.resolve())`` and an exact
string compare. ``Path.resolve()`` canonicalises ``sub/../`` and follows symlinks but does
NOT fold case, and the owner's disk (macOS's default APFS) is case-INSENSITIVE, so on the
tree as found:

    open …/relocated_copy.abf   -> 1 document
    open …/Relocated_Copy.abf   -> 2 documents      (os.path.samefile(A, B) is True)

The recording was decoded and held in RAM twice, the File Manager showed two rows (with
byte-identical display names when the case variant is in a parent DIRECTORY), and the two
loads carried two independent in-session calibrations behind ONE ``.npcal.json``: one Apply
wrote gain 9.0 to the sidecar and left the other load at 3.0, so every ``depth_pA``, ECD and
conductance read off it was wrong by the gain ratio with nothing on screen saying the two
were one file. Reachable without trying — the Recent menu accumulates both spellings, a
project stores a path, a drag from a shell that spelled a parent directory differently.

The test that should have caught it, ``tests/gui/test_document_identity.py::
test_opening_a_file_that_is_already_open_still_focuses_it``, re-opens with the
byte-identical spelling. Nothing in ``nanoscope/``, ``tests/`` or ``docs/`` mentioned case
at all (grep for ``casefold|normcase|samefile|case-insensitiv``: zero hits).

The SCOPING CONTROLS are in this file too, because the fix must not merge two genuinely
different files: identity is asked of the filesystem (``os.path.samefile`` — device +
inode), never guessed by folding the string, so a case-SENSITIVE volume still gets two
documents for two files.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.identity import same_source
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0


def _write_csv(path: Path, n: int = 4_000) -> None:
    t = np.arange(n) / FS
    x = -1000.0 + np.random.default_rng(0).normal(0.0, 5.0, n)
    path.write_text("time,current\n" + "\n".join(f"{a:.8g},{b:.6g}" for a, b in zip(t, x)))


def _case_insensitive(tmp_path: Path) -> bool:
    probe = tmp_path / "CaseProbe.tmp"
    probe.write_text("x")
    try:
        return (tmp_path / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


def _open(w, qtbot, path: Path) -> None:
    w.open_paths([str(path)])
    qtbot.waitUntil(lambda: w._active_open is None and not w._open_queue, timeout=15_000)
    qtbot.wait(20)


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._controller.abort()
    w._analysis_controller.abort()


# ================================================================================================
# the defect
# ================================================================================================
def test_two_case_variant_spellings_of_one_file_load_once(win, qtbot, tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("this volume is case-sensitive: the two spellings are two real files")
    w = win
    lower = tmp_path / "relocated_copy.csv"
    _write_csv(lower)
    upper = tmp_path / "Relocated_Copy.csv"
    assert os.path.samefile(lower, upper), "premise: one file under two spellings"

    _open(w, qtbot, lower)
    assert len(w._session.documents()) == 1
    _open(w, qtbot, upper)
    assert len(w._session.documents()) == 1, (
        f"the same recording was loaded twice: "
        f"{[d.display_name for d in w._session.documents()]}")


def test_the_second_spelling_does_not_fork_the_calibration(win, qtbot, tmp_path):
    """The wrong-result core. Two loads meant two in-session calibrations behind one
    sidecar, and one Apply then left the other load reporting a different current for the
    same samples."""
    if not _case_insensitive(tmp_path):
        pytest.skip("this volume is case-sensitive")
    w = win
    lower = tmp_path / "fork_check.csv"
    _write_csv(lower)
    _open(w, qtbot, lower)
    _open(w, qtbot, tmp_path / "Fork_Check.csv")

    from dataclasses import replace as _replace

    docs = w._session.documents()
    assert len(docs) == 1, [d.display_name for d in docs]
    doc_id = docs[0].doc_id
    w._session.set_calibration(
        doc_id, 0, _replace(w._session.calibration(doc_id, 0), gain=7.0),
        write_sidecar=True)

    # ONE sidecar on disk, and every open load of this file agrees with it. Before the fix
    # the second load kept gain 1.0 while the sidecar said 7.0.
    gains = {w._session.calibration(d.doc_id, 0).gain for d in w._session.documents()}
    assert gains == {7.0}, f"two loads of one file disagree about its calibration: {gains}"
    sidecars = sorted(p.name for p in tmp_path.glob("*.npcal.json"))
    assert len(sidecars) == 1, sidecars
    first = [float(w._session.calibrated(d.doc_id, 0).data[0])
             for d in w._session.documents()]
    assert len(set(first)) == 1, f"the same sample reads differently per load: {first}"


def test_a_case_variant_in_the_directory_is_the_same_file_too(win, qtbot, tmp_path):
    """The worst spelling of the defect: the FILENAME is byte-identical, so the two File
    Manager rows were indistinguishable ('rec.csv', 'rec.csv')."""
    if not _case_insensitive(tmp_path):
        pytest.skip("this volume is case-sensitive")
    w = win
    (tmp_path / "DATA").mkdir()
    rec = tmp_path / "DATA" / "rec.csv"
    _write_csv(rec)

    _open(w, qtbot, rec)
    _open(w, qtbot, tmp_path / "data" / "rec.csv")
    names = [d.display_name for d in w._session.documents()]
    assert len(names) == 1, names


def test_two_case_variants_dropped_together_are_queued_once(win, qtbot, tmp_path):
    """The other door onto the same question. ``open_paths`` de-duplicates its QUEUE with
    the string compare ``_already_open`` used to be, so two spellings arriving in ONE
    gesture (a multi-file drop, a project's document list) were both queued and the guard
    at drain time was the only thing left between them and a second decode."""
    if not _case_insensitive(tmp_path):
        pytest.skip("this volume is case-sensitive")
    w = win
    rec = tmp_path / "dropped.csv"
    _write_csv(rec)
    w.open_paths([str(rec), str(tmp_path / "Dropped.csv")])
    assert len(w._open_queue) + (1 if w._active_open else 0) == 1, (
        f"one recording was queued twice: queue={w._open_queue} active={w._active_open}")
    qtbot.waitUntil(lambda: w._active_open is None and not w._open_queue, timeout=15_000)
    assert len(w._session.documents()) == 1


# ================================================================================================
# the scoping controls — only case was the hole, and only case may be closed
# ================================================================================================
def test_a_dot_dot_spelling_and_a_symlink_still_adopt_the_open_load(win, qtbot, tmp_path):
    """These already worked (``resolve()`` handles both) and must keep working — the
    verifier's own control set."""
    w = win
    d = tmp_path / "d"
    d.mkdir()
    (tmp_path / "other").mkdir()
    rec = d / "rec.csv"
    _write_csv(rec)
    link = tmp_path / "link.csv"
    link.symlink_to(rec)

    _open(w, qtbot, rec)
    assert len(w._session.documents()) == 1
    _open(w, qtbot, tmp_path / "other" / ".." / "d" / "rec.csv")
    assert len(w._session.documents()) == 1
    _open(w, qtbot, link)
    assert len(w._session.documents()) == 1


def test_two_different_files_are_still_two_documents(win, qtbot, tmp_path):
    """The control the fix could have broken: merging by a folded STRING would have been a
    guess about the volume, and on a case-sensitive one it would merge two real recordings.
    Asking the filesystem cannot make that mistake — and two plainly different names must
    never collapse either."""
    w = win
    a = tmp_path / "rec_a.csv"
    b = tmp_path / "rec_b.csv"
    _write_csv(a)
    _write_csv(b)
    _open(w, qtbot, a)
    _open(w, qtbot, b)
    assert len(w._session.documents()) == 2


def test_same_source_never_raises_on_a_path_that_is_not_on_disk():
    """A segment's source key is a synthetic path and a recording's volume can go away.
    Identity has to degrade to the string compare there rather than raise inside the open
    pipeline — 'cannot tell' is answered as 'not the same', which is what it said before."""
    assert same_source("/nowhere/segment#1", "/nowhere/segment#1") is True
    assert same_source("/nowhere/segment#1", "/nowhere/segment#2") is False
    assert same_source("", "") is True
