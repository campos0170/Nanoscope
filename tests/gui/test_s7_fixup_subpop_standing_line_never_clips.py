"""Sweep 7 fix-up C3 (minor) — the subpopulation report's standing line is never drawn
cut off: its fixed height holds the longest text it is asked to carry, and a text past
that budget is elided at a word with the whole sentence in the tooltip.

What the skeptic measured (verify/C-subpops/repro_C3_livenote_clipped.py, 2026-09-01):
with one rejection under a standing report, clicking another recording's row in the Files
dock puts TWO sentences on the line — the AWAY reason, which names the recording by file
name, and the S7-POP4 count sentence. With the owner's own file name
("…0004_process _ -1000 mV_ 385-400 s time trace.abf") the pair needs 45 px (three wrapped
lines at 15 px/line) at 1600, at 1440 and at the 900 px minimum width, and the wave-1
label was a word-wrapped QLabel fixed at 34 px (two lines + 4): the 'accepted now…'
sentence was reduced to the tops of its glyphs (C3_label_1600.png). ``live_note()``
returned the full text, so no test saw it.

The invariant kept from wave 1: one label, always in the layout, fixed height — nothing
moves under the pointer when a sentence arrives. The invariant added here: what the label
DRAWS is never a clipped line. Both are measured below against the real widget at the
window's default, a MacBook-width and the minimum size, with the resolver's own wording
for the owner's file (copied verbatim from the repro's output; ``main_window._subpop_state``
builds it).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.gui import subpop_window as SW

#: The AWAY reason exactly as the resolver printed it for the owner's quick ABF (channel 2)
#: while 0006 was on screen — the longest text the line has been measured to carry.
_AWAY_0004 = (
    "This report describes “500 pM Apo ferritin_1 x PBS_22 nm_0004_process _ -1000 mV_ "
    "385-400 s time trace.abf” (channel 2), which is not the detection on screen — its "
    "colouring is held, not shown, until you select that recording again in the File "
    "Manager; nothing was retired.")
#: The S7-POP4 sentence that stood under it after one rejection in the gallery.
_CURATED = ("accepted now: 109 of 110; this run's fractions describe the 110 accepted at "
            "run time — re-run to update")
_BOTH = _AWAY_0004 + "\n" + _CURATED

#: The window's default size, a MacBook-width window, and its minimum.
_SIZES = ((1600, 980), (1440, 900), (900, 620))


def _fit(label: QtWidgets.QLabel) -> tuple[int, int, int]:
    """(width, fixed height, height the text needs at that width) — the repro's oracle:
    needed > height means Qt draws a cut-off line.

    ``QLabel.heightForWidth`` is clamped up to the widget's minimum height, which
    ``setFixedHeight`` also sets, so a text that fits reports exactly the budget rather
    than its true height (measured: the same 377-character text reports 45 px on an
    unconstrained label, 45 px on one fixed at 34 and 49 px on one fixed at 49). The
    comparison ``needed > budget`` is unaffected — it is true exactly when the true
    height exceeds the budget — which is why this is the oracle and not the raw height.
    """
    w, h = label.width(), label.height()
    return w, h, label.heightForWidth(w)


def _settle(win: QtWidgets.QWidget, qtbot, size: tuple[int, int]) -> None:
    win.resize(*size)
    qtbot.wait(60)
    assert win.width() == size[0], "the offscreen platform refused the window size"


@pytest.fixture
def window(qtbot):
    win = SW.SubpopReportWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    return win


# ================================================================================================
# the owner's own two sentences fit whole at every width
# ================================================================================================

@pytest.mark.parametrize("size", _SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_away_reason_and_the_curation_sentence_are_drawn_whole_at_every_window_width(
        window, qtbot, size):
    _settle(window, qtbot, size)
    window.set_assignment_state(SW.ASSIGNMENT_AWAY, _AWAY_0004)
    window.set_curation_note(_CURATED)
    qtbot.wait(30)
    lab = window._live_note
    w, h, need = _fit(lab)
    assert need <= h, (f"at {size[0]} px the standing line needs {need} px for its text "
                       f"but is fixed at {h} px: the last line is drawn cut off")
    # Nothing was elided away: the line says the whole of both sentences.
    assert window.live_note() == _BOTH
    assert window.live_note_full() == _BOTH
    assert lab.toolTip() == ""


@pytest.mark.parametrize("size", _SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_three_line_text_moves_no_widget_when_it_arrives(window, qtbot, size):
    """Wave 1's rule, re-pinned for the longest text: the splitter (every figure in it),
    the button row and the line itself sit exactly where they sat while the line was
    empty."""
    _settle(window, qtbot, size)
    parts = (window._split, window._figures, window._gallery_btn, window._live_note)
    before = [p.geometry() for p in parts]
    window.set_assignment_state(SW.ASSIGNMENT_AWAY, _AWAY_0004)
    window.set_curation_note(_CURATED)
    qtbot.wait(60)
    after = [p.geometry() for p in parts]
    assert after == before, f"the standing line moved the layout: {before} -> {after}"


# ================================================================================================
# past the budget: elided at a word, whole in the tooltip, re-fitted on every resize
# ================================================================================================

def _away_for(name: str) -> str:
    return (f"This report describes “{name}” (channel 2), which is not the detection on "
            f"screen — its colouring is held, not shown, until you select that recording "
            f"again in the File Manager; nothing was retired.")


def test_a_text_past_the_budget_is_elided_at_a_word_and_the_tooltip_carries_the_whole(
        window, qtbot):
    _settle(window, qtbot, _SIZES[0])
    name = " ".join(f"segment{i:03d}" for i in range(160))     # ~1.7 k characters
    reason = _away_for(name)
    window.set_assignment_state(SW.ASSIGNMENT_AWAY, reason)
    window.set_curation_note(_CURATED)
    qtbot.wait(30)
    lab = window._live_note
    full = reason + "\n" + _CURATED
    w, h, need = _fit(lab)
    assert need <= h, f"elided text still needs {need} px in a {h} px label"
    shown = window.live_note()
    assert shown != full and shown.endswith("…")
    body = shown[:-1]
    assert full.startswith(body) and body, "the drawn text is a prefix of the whole"
    assert full[len(body)].isspace(), "the cut falls at a word boundary, never mid-word"
    assert window.live_note_full() == full
    assert lab.toolTip() == full
    # Clearing the line clears the tooltip with it: no stale sentence behind an empty label.
    window.set_assignment_state(SW.ASSIGNMENT_INSTALLED, "")
    window.set_curation_note("")
    assert window.live_note() == "" and window.live_note_full() == ""
    assert lab.toolTip() == ""


def test_a_window_dragged_wider_un_elides_the_line_and_narrower_re_elides_it(window, qtbot):
    """The fit is a function of the label's width, so it is re-done on every resize —
    a text that needs four lines at the 900 px minimum and three at the default width is
    elided at 900, whole at 1600, elided again at 900."""
    lab = window._live_note
    _settle(window, qtbot, _SIZES[2])
    narrow_w = lab.width()
    _settle(window, qtbot, _SIZES[0])
    wide_w = lab.width()
    budget = lab.height()
    probe = QtWidgets.QLabel()
    probe.setWordWrap(True)
    probe.setFont(lab.font())
    text = None
    for k in range(4, 400):
        candidate = _away_for(" ".join(["ab"] * k)) + "\n" + _CURATED
        probe.setText(candidate)
        if probe.heightForWidth(narrow_w) > budget >= probe.heightForWidth(wide_w):
            text = candidate
            break
    assert text is not None, "no text separates the two widths on this font"
    reason, _, note = text.partition("\n")

    _settle(window, qtbot, _SIZES[2])
    window.set_assignment_state(SW.ASSIGNMENT_AWAY, reason)
    window.set_curation_note(note)
    qtbot.wait(30)
    assert window.live_note() != text and window.live_note().endswith("…")
    assert _fit(lab)[2] <= budget

    _settle(window, qtbot, _SIZES[0])
    assert window.live_note() == text, "wider: the whole text fits again and is drawn whole"
    assert lab.toolTip() == ""
    assert _fit(lab)[2] <= budget

    _settle(window, qtbot, _SIZES[2])
    assert window.live_note() != text and window.live_note().endswith("…")
    assert lab.toolTip() == text
    assert _fit(lab)[2] <= budget
