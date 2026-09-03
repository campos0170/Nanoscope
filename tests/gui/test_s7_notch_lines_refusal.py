"""Sweep-7 S7-NOTCH1 (+ the skeptic's typo-policy item): the Preprocess tab's explicit
notch-line box refuses what it may not run, BY NAME, and never truncates or drops.

Measured on the pre-fix tree (docs/sweep7/repros/S7-NOTCH1_paste_truncation.py): a
10,923-value spreadsheet column (86,157 characters) pasted into the box was cut by
Qt's default ``maxLength`` (32,767) to 4,155 lines, mid-number — the user's value
#4154, 54.153999…, became '54.' and parsed to a 54.0 Hz notch nobody typed, a duplicate
of line #4000 that the kernel then refused by name ("duplicate notch frequencies
[54.0] Hz") — and there was no line cap at all: 4,155 lines cost 42 s per detection on
the -600 mV file. Separately, '60, -120, abc, 180' parsed to [60, 180] with the note
saying "2 line(s)" and provenance stamping [60, 180] while the box held four entries.
The ML tab's sibling box refused above ``_ML_NOTCH_MAX_LINES`` (32) with the cap named
and the box rolled back; this one had neither guard.

The fix (gui/panels.py, the DetectionPanel's own notch region): ``NOTCH_MAX_LINES``
(pinned to the ML constant), a box that holds 250,000 characters so a real paste is
refused by its TRUE count, refusal of any token that is not a positive finite number,
and rollback to the last legal list — so the list the run reads is always the list the
box shows. Driven through the real widget: clipboard paste, then editingFinished.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.gui.analysis_panel import _ML_NOTCH_MAX_LINES
from nanoscope.gui.panels import DetectionPanel


def _paste_and_commit(panel: DetectionPanel, qapp, text: str) -> None:
    """The shipped gesture: a clipboard paste into the box, then the commit that Tab /
    Return / clicking Detect performs (editingFinished)."""
    box = panel.notch_freqs
    QtWidgets.QApplication.clipboard().setText(text)
    box.setFocus()
    box.selectAll()
    box.paste()
    qapp.processEvents()
    box.editingFinished.emit()
    qapp.processEvents()


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


def test_a_spreadsheet_column_paste_is_refused_by_its_true_count_and_rolled_back_not_truncated(
        panel, qapp):
    """The dossier's exact paste. The box must hold the WHOLE paste (so the count it
    names is the paste's, not a cut's), refuse it naming the cap and that count, roll
    back to the previous list, and never manufacture the 54.0 Hz line."""
    pasted = [50.0 + 0.001 * i for i in range(10_923)]
    text = ", ".join(f"{v:g}" for v in pasted)
    assert panel.notch_freqs.maxLength() >= len(text), "the paste must fit, or the count lies"

    _paste_and_commit(panel, qapp, text)

    assert panel.params()["notch_freqs"] == [60.0, 120.0]
    assert panel.notch_freqs.text() == "60, 120"
    note = panel.notch_lines_note.text()
    assert "Refused" in note
    assert "10,923 lines" in note                       # the paste's true count
    assert f"at most {DetectionPanel.NOTCH_MAX_LINES}" in note
    assert "restored to its previous contents (60, 120)" in note
    # The skeptic's point stands: the note still lists what IS in force.
    assert "2 line(s) — 60, 120 Hz" in note
    assert "54" not in panel.notch_freqs.text()


def test_a_paste_longer_than_the_box_holds_is_refused_as_a_floor_with_the_limit_named(
        panel, qapp):
    """Beyond even the enlarged box Qt still truncates — so the refusal must say the
    count is a floor and name the limit, never present the cut as the paste."""
    _paste_and_commit(panel, qapp, ",".join(["60"] * 1_000_000))

    assert panel.params()["notch_freqs"] == [60.0, 120.0]
    note = panel.notch_lines_note.text()
    assert "at least" in note
    assert f"{DetectionPanel.NOTCH_BOX_MAX_CHARS:,} characters" in note
    assert f"at most {DetectionPanel.NOTCH_MAX_LINES}" in note


def test_the_cap_is_the_ml_boxs_cap_and_one_past_it_is_the_boundary(panel, qapp):
    """Two sibling boxes on one panel, one number: a list AT the cap commits exactly as
    typed with no refusal; one past it is refused with the count named."""
    assert DetectionPanel.NOTCH_MAX_LINES == _ML_NOTCH_MAX_LINES

    at_cap = ", ".join(str(60 * (i + 1)) for i in range(DetectionPanel.NOTCH_MAX_LINES))
    _paste_and_commit(panel, qapp, at_cap)
    assert len(panel.params()["notch_freqs"]) == DetectionPanel.NOTCH_MAX_LINES
    assert "Refused" not in panel.notch_lines_note.text()

    over = ", ".join(str(60 * (i + 1)) for i in range(DetectionPanel.NOTCH_MAX_LINES + 1))
    _paste_and_commit(panel, qapp, over)
    assert len(panel.params()["notch_freqs"]) == DetectionPanel.NOTCH_MAX_LINES
    note = panel.notch_lines_note.text()
    assert f"{DetectionPanel.NOTCH_MAX_LINES + 1} lines" in note
    assert f"at most {DetectionPanel.NOTCH_MAX_LINES}" in note


def test_a_token_that_is_not_a_positive_frequency_is_refused_by_name_never_dropped(
        panel, qapp):
    """The skeptic's '60, -120, abc, 180': the kernel refuses duplicates by name so a
    typo cannot pass silently, and the panel now holds its own tokens to the same
    standard. Both bad tokens are quoted, the previous list stays in force, and the
    note still lists it. ``1e400`` is ``inf`` and used to pass ``> 0`` — refused too."""
    _paste_and_commit(panel, qapp, "60, -120, abc, 180")

    assert panel.params()["notch_freqs"] == [60.0, 120.0]      # NOT [60, 180]
    assert panel.notch_freqs.text() == "60, 120"
    note = panel.notch_lines_note.text()
    assert "Refused" in note
    assert "‘-120’" in note and "‘abc’" in note
    assert "2 entries are not a positive frequency" in note
    assert "2 line(s) — 60, 120 Hz" in note

    _paste_and_commit(panel, qapp, "60, 1e400")
    assert panel.params()["notch_freqs"] == [60.0, 120.0]
    assert "‘1e400’" in panel.notch_lines_note.text()


def test_an_uncommitted_illegal_box_is_committed_and_refused_before_a_run_reads_it(
        panel, qapp):
    """A Detect click that did not move focus, or a programmatic setText: the run
    reads params(), and params() must never see a refused list — the commit happens
    at the read, the box is rolled back, and the note says so."""
    panel.notch_freqs.setText("60, abc")          # no editingFinished

    assert panel.params()["notch_freqs"] == [60.0, 120.0]
    assert panel.notch_freqs.text() == "60, 120"
    assert "‘abc’" in panel.notch_lines_note.text()


def test_a_stored_over_cap_list_is_refused_on_restore_and_the_series_spelling_is_named(
        panel, qapp):
    """A protocol or session that stored thousands of lines must not be written into
    the box for the next commit to roll back: it is refused by name on restore, the
    series spelling is what runs, and the note says both. A legal stored list then
    restores normally and clears the refusal."""
    panel.set_params({"notch_enabled": True,
                      "notch_freqs": [60.0 + i for i in range(10_923)]})

    assert panel.params()["notch_freqs"] is None
    assert not panel.notch_use_lines.isChecked()
    note = panel.notch_lines_note.text()
    assert "stored line list was refused" in note
    assert "10,923 lines" in note and "series spelling" in note

    panel.set_params({"notch_enabled": True, "notch_freqs": [60.0, 120.0, 180.0]})
    assert panel.params()["notch_freqs"] == [60.0, 120.0, 180.0]
    assert "Refused" not in panel.notch_lines_note.text()


def test_a_refusal_outlives_the_rollback_and_only_a_new_legal_edit_clears_it(panel, qapp):
    """Every note refresh re-commits the rolled-back text; that must not erase the
    refusal it describes. Re-committing the restored list keeps the refusal on screen;
    committing a different legal list clears it."""
    _paste_and_commit(panel, qapp, "60, abc")
    assert "Refused" in panel.notch_lines_note.text()

    panel.notch_freqs.editingFinished.emit()          # Enter again on the restored text
    qapp.processEvents()
    assert "Refused" in panel.notch_lines_note.text()

    _paste_and_commit(panel, qapp, "60, 180")
    assert panel.params()["notch_freqs"] == [60.0, 180.0]
    assert "Refused" not in panel.notch_lines_note.text()
