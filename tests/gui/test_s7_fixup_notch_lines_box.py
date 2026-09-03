"""Sweep-7 fix-up, skeptic F-SK-1/2/3: the Preprocess tab's explicit notch-line box keeps
its refusal on screen through the run it gated, discloses without moving anything, and
reads a spreadsheet paste in the shape a spreadsheet gives it.

The S7-NOTCH1 fix taught the box to refuse by name and roll back (``_commit_notch_lines``),
with one rule for clearing the refusal -- a NEW legal list. Three defects in that fix, all
measured 2026-09-02 on the owner's -1000 mV excerpt through the real MainWindow:

* F-SK-1 (major). Type '60, abc', press Ctrl+D. ``params()`` committed the box, refused
  'abc', rolled back to '60, 120' and painted the refusal; the run read [60, 120]; when
  its result landed, ``AnalysisPanel._seed_form_from`` re-seeded the form from it --
  ``set_notch_lines([60.0, 120.0])`` -- which cleared the refusal UNCONDITIONALLY, a
  second writer with a different rule. End state: box '60, 120', note with no refusal,
  status bar '', a detection stamped [60, 120]. The user's list was discarded with no
  surviving disclosure. Now one writer (``_accept_notch_text``) judges every legal list
  against the list the refusal was raised under: a replay of the list in force keeps it.
* F-SK-2 (minor). The note was a bare word-wrapping QLabel: committing a refusal by
  clicking a control further down the tab grew it 15 -> 45 px between the mouse PRESS
  (which took focus and committed) and the RELEASE, moved the "show processed trace"
  checkbox +13 px, and the release landed at local y = -3 -- the click was lost. The
  note now owns ``NOTCH_NOTE_LINES`` lines (``_FixedLinesNote``) and keeps the refusal
  painted when the whole text does not fit.
* F-SK-3 (minor). ``_notch_line_tokens`` split only on ',', ';' and '/', so a pasted
  spreadsheet COLUMN (newline-separated) or ROW (tab-separated) was ONE token, refused
  as "1 entry is not a positive frequency in Hz -- '60\\n120\\n180...'" -- every entry
  was one -- and the 10,923-value column that motivated the cap never met the cap
  sentence. Whitespace now separates too, and a legal commit re-spells the box as its
  tokens joined by ", " (the user's own spellings: no value moves).

Driven through the real widgets: clipboard paste, editingFinished, a press/release pair
delivered at one global point the way a mouse does.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui.panels import DetectionPanel

# ``getattr`` so the file COLLECTS on the pre-fix snapshot (no cap there) and each test
# fails there on its own claim.
CAP = getattr(DetectionPanel, "NOTCH_MAX_LINES", 32)
FLOOD = [50.0 + 0.001 * i for i in range(10_923)]
FIVE = [60.0, 120.0, 180.0, 240.0, 300.0]
AT_CAP = ", ".join(str(60 * (i + 1)) for i in range(CAP))
OVER_CAP = ", ".join(str(60 * (i + 1)) for i in range(CAP + 1))


def _paste_and_commit(panel: DetectionPanel, qapp, text: str) -> None:
    """The shipped gesture: a clipboard paste into the box, then the commit that Tab /
    Return / clicking elsewhere performs (editingFinished)."""
    box = panel.notch_freqs
    QtWidgets.QApplication.clipboard().setText(text)
    box.setFocus()
    box.selectAll()
    box.paste()
    qapp.processEvents()
    box.editingFinished.emit()
    qapp.processEvents()


def _refusal(panel: DetectionPanel) -> str:
    """The standing refusal as the note carries it (its ``⚠`` tail), or ''."""
    note = panel.notch_lines_note.text()
    return note.split("\n⚠ ", 1)[1] if "\n⚠ " in note else ""


@pytest.fixture
def panel(qtbot, qapp) -> DetectionPanel:
    p = DetectionPanel()
    qtbot.addWidget(p)
    p.show()
    p.notch_use_lines.setChecked(True)
    _paste_and_commit(p, qapp, "60, 120")
    assert p.params()["notch_freqs"] == [60.0, 120.0]
    assert "Refused" not in p.notch_lines_note.text()
    return p


# ================================================================================================
# F-SK-1: the refusal outlives the run it gated
# ================================================================================================

def test_a_run_that_replays_the_list_in_force_keeps_the_refusal_on_the_note(panel, qapp):
    """The Ctrl+D path, as the app performs it: the run reads ``params()`` off an
    uncommitted box (commit, refuse, roll back), then its result re-seeds the form with
    the very list that ran. That replay is not a new list; the refusal stays."""
    panel.notch_freqs.setText("60, abc")                    # no editingFinished: Ctrl+D
    assert panel.params()["notch_freqs"] == [60.0, 120.0]   # what the run reads
    assert "‘abc’" in _refusal(panel)

    # The result lands: AnalysisPanel._seed_form_from -> set_params -> set_notch_lines.
    panel.set_params({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    assert "‘abc’" in _refusal(panel), "a replay of the list in force erased the refusal"
    assert "restored to its previous contents (60, 120)" in _refusal(panel)
    assert panel.notch_freqs.text() == "60, 120"
    assert panel.params()["notch_freqs"] == [60.0, 120.0]
    assert "2 line(s) — 60, 120 Hz" in panel.notch_lines_note.text()

    # A selection round trip re-seeds again; still the same list, still the refusal.
    panel.set_params({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    assert "‘abc’" in _refusal(panel)


def test_a_replay_or_a_commit_of_a_different_list_clears_the_refusal(panel, qapp):
    """The rule the S7-NOTCH1 fix wrote for the commit road, now on both roads."""
    panel.notch_freqs.setText("60, abc")
    panel.params()
    assert _refusal(panel)
    panel.set_params({"notch_enabled": True, "notch_freqs": [60.0, 180.0]})
    assert not _refusal(panel)
    assert panel.params()["notch_freqs"] == [60.0, 180.0]
    assert panel.notch_freqs.text() == "60, 180"

    panel.notch_freqs.setText("60, abc")
    panel.params()
    assert _refusal(panel)
    _paste_and_commit(panel, qapp, "60, 240")
    assert not _refusal(panel)
    assert panel.params()["notch_freqs"] == [60.0, 240.0]


def test_a_respelling_of_the_list_in_force_is_not_a_new_list(panel, qapp):
    """Judged on the parsed list, not the text: '60,120', '60.0, 120.0' and a pasted
    column are the list already in force, so they keep the refusal -- and the box is
    re-spelled as the list, so the rollback target never carries a line break."""
    panel.notch_freqs.setText("60, abc")
    panel.params()
    assert _refusal(panel)
    for same in ("60,120", "60.0, 120.0", "60\n120"):
        _paste_and_commit(panel, qapp, same)
        assert panel.params()["notch_freqs"] == [60.0, 120.0], same
        assert _refusal(panel), f"{same!r} was mistaken for a new list"
    assert "\n" not in panel.notch_freqs.text()
    assert panel.parse_notch_lines(panel.notch_freqs.text()) == [60.0, 120.0]


def test_a_stored_list_refusal_is_cleared_when_any_list_goes_live_again(panel):
    """A refused STORED list leaves the series spelling live and says so; that claim is
    false the moment any explicit list is replayed -- even the list the box held before
    the flood -- so the refusal is cleared then, whatever the list."""
    panel.set_params({"notch_enabled": True, "notch_freqs": FLOOD})
    assert "stored line list was refused" in _refusal(panel)
    assert not panel.notch_use_lines.isChecked()
    assert panel.params()["notch_freqs"] is None

    panel.set_params({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    assert not _refusal(panel), "the note still claimed the series spelling was live"
    assert panel.notch_use_lines.isChecked()
    assert panel.params()["notch_freqs"] == [60.0, 120.0]


def test_every_refusal_is_announced_once_on_the_panels_own_signal(panel, qtbot):
    """``notchLinesRefused`` carries the note's sentence for the status line -- from the
    commit road (a shortcut-driven run needs no editingFinished) and the replay road
    alike -- once per refusal, never on a re-commit or a replay of the list in force."""
    with qtbot.waitSignal(panel.notchLinesRefused, timeout=500) as refused:
        panel.notch_freqs.setText("60, abc")
        panel.params()
    assert refused.args[0] == _refusal(panel)

    with qtbot.assertNotEmitted(panel.notchLinesRefused, wait=50):
        panel.params()
        panel.notch_freqs.editingFinished.emit()
        panel.set_params({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    assert _refusal(panel)

    with qtbot.waitSignal(panel.notchLinesRefused, timeout=500) as stored:
        panel.set_params({"notch_enabled": True, "notch_freqs": FLOOD})
    assert "10,923 lines" in stored.args[0]
    assert stored.args[0] == _refusal(panel)


# ================================================================================================
# F-SK-2: the note owns its height; the click that commits the box lands
# ================================================================================================

def _shown_on_the_notch_page(qtbot, qapp, width: int) -> DetectionPanel:
    p = DetectionPanel()
    qtbot.addWidget(p)
    p.resize(width, 1000)
    p.show()
    p.activateWindow()
    qapp.processEvents()
    p.mode_tabs.setCurrentIndex(0)
    p.filter_selector.setCurrentIndex(p.filter_stack.indexOf(p.notch_freqs.parentWidget()))
    p.notch_enabled.setChecked(True)
    p.notch_use_lines.setChecked(True)
    _paste_and_commit(p, qapp, "60, 120")
    qtbot.wait(40)
    assert p.params()["notch_freqs"] == [60.0, 120.0]
    assert p.notch_lines_note.isVisible() and p.filter_preview.isVisible()
    return p


def _click_like_a_pointer(panel, qapp, qtbot, cbx, uncommitted: str):
    """Type ``uncommitted`` into the box (no Enter), then press on ``cbx`` and release
    at the SAME global point, as a mouse does. The press takes focus and commits the
    box; whatever that paints must not move ``cbx`` before the release."""
    box = panel.notch_freqs
    box.setFocus()
    box.selectAll()
    QtTest.QTest.keyClicks(box, uncommitted)
    qapp.processEvents()
    assert box.text() == uncommitted
    y0 = cbx.mapTo(panel, QtCore.QPoint(0, 0)).y()
    h0 = panel.notch_lines_note.height()
    was = cbx.isChecked()
    local = QtCore.QPoint(8, cbx.height() // 2)
    glob = cbx.mapToGlobal(local)
    QtTest.QTest.mousePress(cbx, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, local)
    qtbot.wait(60)                                   # let a layout request land (or not)
    y1 = cbx.mapTo(panel, QtCore.QPoint(0, 0)).y()
    h1 = panel.notch_lines_note.height()
    QtTest.QTest.mouseRelease(cbx, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                              cbx.mapFromGlobal(glob))
    qtbot.wait(30)
    return y1 - y0, h1 - h0, cbx.isChecked() != was


@pytest.mark.parametrize("edit, refused", [
    ("60, abc", True),        # the skeptic's case: a token refusal
    (OVER_CAP, True),         # the cap refusal, the longest sentence the note carries
    (AT_CAP, False),          # a LEGAL 32-line list: its description alone wraps to 3 lines
])
def test_the_click_that_commits_the_box_moves_nothing_under_the_pointer_and_lands(
        qtbot, qapp, edit, refused):
    panel = _shown_on_the_notch_page(qtbot, qapp, 900)
    moved, grew, toggled = _click_like_a_pointer(panel, qapp, qtbot, panel.filter_preview, edit)
    assert (_refusal(panel) != "") is refused, "the press did not commit the box"
    assert grew == 0, f"the note grew {grew:+d} px on the commit"
    assert moved == 0, f"the checkbox moved {moved:+d} px between press and release"
    assert toggled, "the click was lost"


@pytest.mark.parametrize("width", [900, 420])
def test_the_note_owns_its_height_in_every_state_and_keeps_the_refusal_painted(
        qtbot, qapp, width):
    """Every sentence the note can carry, at the dock's width and at a narrow one:
    the note, the filter stack and the control below them never move, and when the
    whole text does not fit the refusal's verdict is what stays painted, with the
    whole text in the tooltip."""
    panel = _shown_on_the_notch_page(qtbot, qapp, width)
    note = panel.notch_lines_note
    height, stack, cbx_y = (note.height(), panel.filter_stack.height(),
                            panel.filter_preview.mapTo(panel, QtCore.QPoint(0, 0)).y())
    min_w = panel.minimumSizeHint().width()

    def settled(label: str) -> None:
        qtbot.wait(40)
        assert note.height() == height, label
        assert panel.filter_stack.height() == stack, label
        assert panel.filter_preview.mapTo(panel, QtCore.QPoint(0, 0)).y() == cbx_y, label
        assert panel.minimumSizeHint().width() == min_w, label
        full, painted = note.text(), QtWidgets.QLabel.text(note)
        if painted == full:
            assert note.toolTip() == "", label
        else:
            assert "…" in painted and note.toolTip() == full, label
        if "\n⚠ " in full:
            verdict = full.rsplit("\n⚠ ", 1)[1].split(". ", 1)[0]
            assert verdict in painted, (label, painted)

    _paste_and_commit(panel, qapp, AT_CAP);            settled("a 32-line list")
    _paste_and_commit(panel, qapp, "60, abc");         settled("a token refusal under it")
    _paste_and_commit(panel, qapp, "60, 120");         settled("two lines, refusal cleared")
    assert not _refusal(panel)
    _paste_and_commit(panel, qapp, OVER_CAP);          settled("the cap refusal")
    _paste_and_commit(panel, qapp, "");                settled("an empty list")
    panel.notch_use_lines.setChecked(False);           settled("the series spelling")
    panel.set_params({"notch_enabled": True, "notch_freqs": FLOOD})
    settled("a stored list refused under the series spelling")
    panel.set_params({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    settled("a legal replay")


# ================================================================================================
# F-SK-3: a paste in the shape a spreadsheet gives it
# ================================================================================================

@pytest.mark.parametrize("shape, text", [
    ("newline column", "60\n120\n180\n240\n300"),
    ("CRLF column", "60\r\n120\r\n180\r\n240\r\n300"),
    ("tab row", "60\t120\t180\t240\t300"),
    ("space row", "60 120 180 240 300"),
    ("every separator at once", "60, 120;180\n240 / 300"),
])
def test_a_spreadsheet_column_or_row_paste_commits_as_its_lines_and_is_respelled(
        panel, qapp, shape, text):
    _paste_and_commit(panel, qapp, text)
    assert panel.params()["notch_freqs"] == FIVE, shape
    assert not _refusal(panel), shape
    assert panel.notch_freqs.text() == "60, 120, 180, 240, 300", shape
    assert "5 line(s) — 60, 120, 180, 240, 300 Hz" in panel.notch_lines_note.text(), shape


def test_the_column_paste_that_motivated_the_cap_now_meets_the_cap_sentence(panel, qapp):
    """The k16 'realistic spreadsheet paste', 10,923 values one per line: refused by
    its count against the cap -- not as one entry that 'is not a positive frequency'
    -- and the sentence never carries the column's line breaks."""
    _paste_and_commit(panel, qapp, "\n".join(f"{v:g}" for v in FLOOD))
    why = _refusal(panel)
    assert "10,923 lines" in why and f"at most {CAP}" in why, why
    assert "not a positive frequency" not in why
    assert "\n" not in why and "\t" not in why
    assert "restored to its previous contents (60, 120)" in why
    assert panel.params()["notch_freqs"] == [60.0, 120.0]
    assert panel.notch_freqs.text() == "60, 120"


def test_a_bad_entry_in_a_pasted_column_is_named_alone_without_the_column_around_it(
        panel, qapp):
    _paste_and_commit(panel, qapp, "60\nabc\n120\n-180")
    why = _refusal(panel)
    assert "2 entries are not a positive frequency" in why, why
    assert "‘abc’" in why and "‘-180’" in why
    assert "\n" not in why
    assert panel.params()["notch_freqs"] == [60.0, 120.0]


def test_the_boxs_tooltip_names_the_separators_a_paste_carries(panel):
    tip = panel.notch_freqs.toolTip()
    assert "spaces" in tip and "line" in tip and "spreadsheet" in tip, tip
