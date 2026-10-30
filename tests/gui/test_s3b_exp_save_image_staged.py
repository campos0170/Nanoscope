"""EXP3-06 + EXP3-02 — the five *Save image…* exports stage, and disclose.

All five pyqtgraph image exports (Plots ▸ Save image…, and the I-V / PSD / event-plot /
voltage-dependence twins) wrote the user's chosen path DIRECTLY with
``ImageExporter.export(str(path))`` and shared one copied refusal sentence. Two harms
came out of that, and both are pinned here:

EXP3-06 (data-loss + misleading). ``QImage.save`` opens the TARGET and truncates it, and
reports failure by RETURNING ``False``. So a name whose extension Qt cannot write —
reachable from the same dialog through its "All files (*)" filter, typed over a ``.pdf``
an earlier *Export figure…* left there — zeroed the user's file, and the dialog then said
"The file was not created". Even where nothing was destroyed (a target the user had
chmod-ed 444) that sentence is false: the file of that name IS in the folder. It is the
PREVIOUS export, and a researcher who goes and looks finds the right name with a
plausible timestamp and takes it for the figure they just saved.
``docs/USER_GUIDE.md`` §12 promises these five the opposite by name: "Every export
replaces its target the same way a project save does: staged beside it and moved into
place, so a failure leaves the previous file whole rather than truncated."

EXP3-02 (silent-failure). ``core.atomic_write`` computes the file-identity notes at every
seam and ledgers them, but nothing outside ``core`` drained the ledger — so an export
over a hard-linked file broke the link, and over a symlink wrote through to somewhere
else, in silence, while the identical situation on a ``.nanoscope`` produced a full
dialog in the same session. USER_GUIDE §11 scopes the promise to no single writer:
"Every file this app writes over your own data — the project, the calibration sidecar,
exported events, exported figures".

Everything here is driven through the panel's REAL save button and the REAL
``ask_export_path`` seam wherever the button is what a user presses; the message boxes
are captured, never shown (the static ``QMessageBox`` builds its own modal in C++, so an
unpatched regression would HANG rather than fail).
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6 import QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import (event_plot_panel, iv_panel, plot_export, plots_panel, psd_panel,
                           voltage_dep_panel)
from nanoscope.gui.plot_export_options import ExportOptionsDialog
from nanoscope.gui.event_plot_panel import EventPlotWindow
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.gui.psd_panel import PSDWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow

#: (window class, the attribute holding its real "Save image…" button).
_PANELS = (
    (PlotsWindow, "_save_btn"),
    (PSDWindow, "_save_btn"),
    (IVWindow, "_save_btn"),
    (VDepWindow, "_save_btn"),
    (EventPlotWindow, "_d_save"),
)
_IDS = [cls.__name__ for cls, _b in _PANELS]

#: Every module whose ``save_image`` reaches the file dialog through this name.
_PANEL_MODULES = (plots_panel, psd_panel, iv_panel, voltage_dep_panel, event_plot_panel,
                  plot_export)


# ================================================================================================
# fixtures
# ================================================================================================

@pytest.fixture(autouse=True)
def drained_ledger():
    """Start and finish each test with the identity ledger empty.

    The ledger is process-global by design (it holds what nobody has shown the user
    yet, across the whole session) and it is CAPPED — when more than ``LEDGER_MAX``
    sentences go unread, the next drain of ANY scope leads with "N earlier
    file-identity note(s) … were discarded". That is correct behaviour and it is
    exactly what would make "a clean export says nothing" flicker depending on which
    other test module ran first. Isolating here keeps these assertions about the
    panels rather than about the rest of the suite.
    """
    from nanoscope.core import atomic_write

    atomic_write.take_identity_notes()
    yield
    atomic_write.take_identity_notes()


@pytest.fixture
def boxes(monkeypatch):
    """Capture ``(kind, title, text)`` of every QMessageBox static instead of showing it."""
    shown: list[tuple[str, str, str]] = []
    for kind in ("warning", "information", "critical", "question"):
        monkeypatch.setattr(
            QtWidgets.QMessageBox, kind,
            staticmethod(lambda _p, t, txt="", *a, _k=kind, **k: shown.append((_k, t, txt))))
    return shown


@pytest.fixture
def options_ok(monkeypatch):
    """Press OK in the "Export current figure" options dialog, keeping its defaults."""
    def _ok(self, *_a, **_k):
        self.accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(ExportOptionsDialog, "exec", _ok)


@pytest.fixture
def picks(monkeypatch, options_ok):
    """Make the file dialog answer with whatever path the test last chose.

    Returns a one-element list; assign ``picks[0] = path`` before clicking the button.
    Patching the name each panel module imported keeps the panel's own call site — and
    the button that reaches it — completely untouched.
    """
    chosen: list[Path | None] = [None]
    for mod in _PANEL_MODULES:
        monkeypatch.setattr(mod, "ask_export_path",
                            lambda *a, **k: chosen[0], raising=True)
    return chosen


def _window(cls, qtbot):
    w = cls(Session())
    qtbot.addWidget(w)
    return w


def _texts(boxes) -> str:
    return " ".join(t for _k, _t, t in boxes)


# ================================================================================================
# EXP3-06 — the refusal may not claim the name is absent when a file of that name is there
# ================================================================================================

@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_a_refused_save_over_an_existing_export_says_the_old_file_is_still_there(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """chmod 444 over a previous export — the reported case. The old message said the
    file "was not created" while it sat in the folder, byte for byte."""
    w = _window(cls, qtbot)
    target = tmp_path / "fig3b.png"

    picks[0] = target
    getattr(w, btn).click()
    assert target.is_file() and target.stat().st_size > 0
    before = target.read_bytes()

    # The user protects the figure they are about to put in the paper, then saves again.
    target.chmod(0o444)
    boxes.clear()
    try:
        picks[0] = target
        getattr(w, btn).click()
    finally:
        target.chmod(0o644)

    text = _texts(boxes)
    assert boxes and boxes[0][0] == "warning", "a refused save must be reported"
    assert "was not created" not in text, (
        "the dialog claimed the file was not created while the previous export of "
        f"exactly that name was in the folder: {text!r}")
    assert "fig3b.png" in text, "the message must name the file the user chose"
    assert "was not changed" in text or "untouched" in text, (
        "the user has to be told the file on disk is the OLD one, untouched")
    assert target.read_bytes() == before, "the previous export must survive intact"


@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_an_unwritable_format_leaves_the_previous_file_whole_and_dates_it(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """The data-loss half. ``.xyz`` stands for the ``.pdf``/``.svg``/``.eps``/no-extension
    names the dialog's "All files (*)" filter lets through: Qt cannot write them, and the
    unstaged write truncated the user's file to zero bytes before finding that out."""
    w = _window(cls, qtbot)
    target = tmp_path / "fig3b.xyz"
    target.write_bytes(b"PREVIOUS CONTENT THE USER CARES ABOUT")

    w.save_image(str(target))

    assert target.read_bytes() == b"PREVIOUS CONTENT THE USER CARES ABOUT", (
        "an image export that cannot render the format truncated the user's file")
    text = _texts(boxes)
    assert "was not created" not in text
    assert "untouched" in text and "EARLIER export" in text, (
        f"the message must say the file there is the earlier one: {text!r}")
    assert "older than the figure on screen" in text


@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_a_refusal_with_nothing_of_that_name_still_says_the_file_was_not_created(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """The other branch keeps the sentence it always had — it was only ever wrong when a
    file of that name WAS there."""
    w = _window(cls, qtbot)
    target = tmp_path / "fig3b.xyz"          # nothing of that name on disk

    w.save_image(str(target))

    text = _texts(boxes)
    assert "The file was not created" in text
    assert not target.exists()


@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_a_good_save_still_lands_and_says_nothing(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """Staging must not turn a working export into a refusal, or add a dialog to it."""
    w = _window(cls, qtbot)
    target = tmp_path / "plot.png"

    picks[0] = target
    getattr(w, btn).click()

    assert target.is_file() and target.stat().st_size > 0
    assert boxes == [], f"a clean export must be silent, got {boxes!r}"
    # No staging sibling left behind, under any name.
    assert [p.name for p in tmp_path.iterdir()] == ["plot.png"]
    # The four panels with a Details pane record the save there. EventPlotWindow has none:
    # its details_text() (779d4ec) is computed figure-bundle provenance, not a pane.
    if cls in (PlotsWindow, PSDWindow, IVWindow, VDepWindow):
        assert "Saved image → plot.png" in w.details_text()


# ================================================================================================
# EXP3-02 — the write's effect on the FILE reaches the user, as it does for the project save
# ================================================================================================

@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_a_re_export_over_a_hard_linked_image_names_the_other_names(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """The manuscript-folder case. An atomic replace cannot preserve a hard link, so the
    other name silently keeps the PREVIOUS export — and, as the guide says, "nothing else
    in the app can tell you this after the fact"."""
    w = _window(cls, qtbot)
    target = tmp_path / "plot.png"

    picks[0] = target
    getattr(w, btn).click()
    assert target.is_file()

    manuscript = tmp_path / "manuscript_copy.png"
    os.link(target, manuscript)
    assert target.stat().st_nlink == 2

    boxes.clear()
    picks[0] = target
    getattr(w, btn).click()

    text = _texts(boxes)
    assert "hard link" in text, (
        "the re-export broke a hard link and said nothing; File ▸ Save Project discloses "
        f"exactly this in the same session. Dialogs: {boxes!r}")
    assert "plot.png" in text and "PREVIOUS contents" in text
    assert target.stat().st_ino != manuscript.stat().st_ino, (
        "the identity note describes a link that really was broken")


@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_an_export_onto_a_symlink_says_which_file_it_actually_wrote(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """"The save says so the first time — the file that changed is not the one you
    clicked" (USER_GUIDE §11). The link itself survives; its destination is rewritten."""
    w = _window(cls, qtbot)
    destination = tmp_path / "elsewhere" / "real.png"
    destination.parent.mkdir()
    destination.write_bytes(b"older")
    link = tmp_path / "plot.png"
    link.symlink_to(destination)

    picks[0] = link
    getattr(w, btn).click()

    text = _texts(boxes)
    assert "link" in text and "real.png" in text, (
        f"an export through a symlink must say where the bytes went: {boxes!r}")
    assert link.is_symlink(), "the link is followed, never replaced"
    assert destination.stat().st_size > len(b"older")


@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_a_note_is_shown_once_and_not_again_beside_the_next_export(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """The ledger is a disclosure channel, not a log: a sentence the user has been shown
    must not reappear under the next, unrelated save."""
    w = _window(cls, qtbot)
    target = tmp_path / "plot.png"
    picks[0] = target
    getattr(w, btn).click()
    os.link(target, tmp_path / "other_name.png")

    boxes.clear()
    picks[0] = target
    getattr(w, btn).click()
    assert "hard link" in _texts(boxes)

    boxes.clear()
    picks[0] = tmp_path / "second.png"       # a different file, nothing to disclose
    getattr(w, btn).click()
    assert boxes == [], f"the note was repeated beside an unrelated export: {boxes!r}"


@pytest.mark.parametrize("cls,btn", _PANELS, ids=_IDS)
def test_a_refused_save_discloses_nothing_because_nothing_was_replaced(
        cls, btn, qapp, qtbot, tmp_path, boxes, picks):
    """An identity note describes a write that HAPPENED. A refused export must not carry
    "those other names still hold the PREVIOUS contents" — after it, they hold the
    current ones."""
    w = _window(cls, qtbot)
    target = tmp_path / "plot.png"
    picks[0] = target
    getattr(w, btn).click()
    os.link(target, tmp_path / "other_name.png")

    target.chmod(0o444)
    boxes.clear()
    try:
        picks[0] = target
        getattr(w, btn).click()
    finally:
        target.chmod(0o644)

    assert "hard link" not in _texts(boxes), (
        "a refused save reported a link it never broke")
