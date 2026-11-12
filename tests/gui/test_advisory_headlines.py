"""Advisories read as headlines on screen; the sentences are one click away and never change.

The longest advisory the app writes (``baseline.level_occupancy_warning``) is over 300 words,
and a detection routinely carries four or five more. Printed whole under the Summary table and
again in the toolbar link's tooltip they filled the dock and pushed the numbers they qualify
out of it. So the READING copy is cut down, in two places, by one pure helper
(:func:`nanoscope.gui.advisory_text.advisory_headline`):

* the Summary's "About these numbers" block shows one headline bullet per advisory, and a
  switch under it ("Show full text" / "Show first lines") shows today's whole text in the same
  label;
* the toolbar link's tooltip is its lead line plus one headline bullet per advisory.

What must NOT move is the text itself. ``AnalysisResult.warnings``, the provenance, the
exports and the saved project carry the full sentences, ``StatsPanel.warnings_text()`` still
returns today's complete block byte for byte (``warnings_shown_text()`` is the new name for
what the label reads right now), and the label's tooltip is the full text. The tests below pin
each of those, and the helper's one promise: a headline is a prefix of the stored text.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import os
import re
import runpy
from copy import deepcopy
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest

from nanoscope.analysis.baseline import (
    DEFAULT_BASELINE,
    UNPROBED_DEFAULT_WARNING,
    batch_copied_warning,
    level_occupancy_warning,
)
from nanoscope.analysis.measurement_quality import (
    build_measurement_quality,
    report_recording_messages,
)
from nanoscope.core.acquisition import acquisition_snapshot_messages
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.session import Session
from nanoscope.gui.advisory_text import advisory_headline
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.analysis_worker import AnalysisController
from nanoscope.gui.panels import DetectionPanel, StatsPanel
from nanoscope.gui.trace_view import TraceView

_helpers = runpy.run_path(str(Path(__file__).with_name("test_event_plot.py")))

ELLIPSIS = "…"

# ---- real advisories: the strings the app's own builders produce --------------------------------
#: The one every un-probed detection carries (about 90 words).
UNPROBED = UNPROBED_DEFAULT_WARNING
#: The one every two-level recording carries (over 300 words).
TWO_LEVEL = level_occupancy_warning(
    0.61, alt_occupancy=0.37, alt_offset_pA=91.9, tracked_pA=-600.0)
#: The one a batch carries for windows copied from another recording's form.
BATCH = batch_copied_warning(DEFAULT_BASELINE)
#: Two sentences, as the acquisition note reads for a result with no stored bandwidth.
ACQUISITION = acquisition_snapshot_messages({})[0]
#: One sentence that fits whole (the quality check's polarity note): no headline is shorter.
SHORT = ("Excursion polarity is not independently determined from this recording; the chosen "
         "direction is not confirmation.")

#: Four more real sentences, each one a cut used to land badly in (see the cases below). Copied
#: from the module named beside it at this commit, so that the headline the rule gives each is
#: pinned as text and does not move with the app's wording.
SHORT_RECORDING = (      # baseline.py: the health check on a recording shorter than rough_s
    "the recording (0.2 s) is SHORTER than rough_s (0.5 s), so pass 1 sees one reflected "
    "window and the mask is arbitrary: measured mask fractions of 1.0000, 0.4520 and 0.0902 "
    "for rough windows of 0.001, 0.01 and 0.1 s on the same trace, with no error at any "
    "length. Shorten rough_s or analyse more signal.")
CLAMPED = (              # baseline.py: the fallback note when the windows were clamped
    "60000 samples (1.2 s) is shorter than the 2 s tracking window (100000 samples), so "
    "'masked_median' ran with its window(s) CLAMPED to this recording: masked median, rough "
    "1200 ms / base 480 ms, decim auto, mask 3.5 sigma, dilate 1 ms. The ratio between the "
    "two passes is unchanged; only the widths are, and only because no wider window exists "
    "here.")
RECONCILE = (            # reconcile.py: the coverage note of a reconciled table
    "At least one input lacks an explicit analysed sample span, so the combined coverage "
    "cannot be computed without guessing; event rate and analysed span are omitted from the "
    "reconciled table.")
INTRUSION = (            # measurement_quality.py: ten events flagged for baseline intrusion
    "10 event(s) are flagged for local baseline intrusion: either the near-flank fraction "
    "itself exceeds the 10% advisory threshold, or the Baseline health block's own "
    "wide-reference checks (mask vs a wide-window mask, base window vs state dwell, baseline "
    "vs the open pore beside each state, or level occupancy) already found the estimator's "
    "state or level tracking unreliable -- in which case a clean-looking near-flank fraction "
    "is not independent evidence, since both flanks can share the very ramp or locked-in "
    "level the wide check condemns. This suggests depth bias, not a calibrated estimate of "
    "true amplitude error. Inspect the reference and the Baseline health block.")

REAL = (UNPROBED, TWO_LEVEL, BATCH, ACQUISITION, SHORT,
        SHORT_RECORDING, CLAMPED, RECONCILE, INTRUSION)


def _full_block(items) -> str:
    """The advisory block as it was written before headlines existed: heading, then one
    bullet per advisory, whole. Spelt out here rather than read from the app, so this file
    pins today's bytes and not whatever the panel builds now."""
    return "⚠ About these numbers\n• " + "\n• ".join(items)


def _headline_block(items) -> str:
    return "⚠ About these numbers\n• " + "\n• ".join(advisory_headline(t) for t in items)


# ================================================================================================
# THE HELPER: a capped prefix of the stored text
# ================================================================================================
_EDGE_CASES = (
    "",
    "   ",
    "No full stop at all, just a long run of words that goes on and on " * 6,
    "A" * 400,
    "Word " * 80,
    "Ends in a sentence. And another one, with a clause: and a dash -- and more text. " * 5,
    "Tail after a line break.\nSecond line that is not part of the headline.",
    "  Leading and trailing space is not part of a headline.  ",
)


@pytest.mark.parametrize("limit", [230, 180, 100, 40])
@pytest.mark.parametrize("text", REAL + _EDGE_CASES)
def test_a_headline_is_a_capped_prefix_of_the_stored_text(text, limit):
    head = advisory_headline(text, limit)
    kept = head[:-1] if head.endswith(ELLIPSIS) else head
    assert len(head) <= limit, "a headline is never longer than its cap, ellipsis included"
    assert text.strip().startswith(kept), (
        "a headline is the advisory's own opening words, none reworded, none added")
    if text.strip():
        assert kept, "a non-empty advisory has a non-empty headline"


def test_the_headline_of_a_real_advisory_is_its_first_sentence():
    for text in (UNPROBED, TWO_LEVEL, ACQUISITION):
        first = text.split(". ")[0] + "."
        assert len(first) < 180 and len(first) < len(text), "premise: a short first sentence"
        assert advisory_headline(text) == first, text
    assert advisory_headline(SHORT) == SHORT, "a one-sentence advisory is its own headline"


def test_a_real_first_sentence_over_the_cap_is_cut_at_a_clause_break():
    """The batch advisory's opening sentence is longer than a 180-character cap: it is cut
    where the sentence pauses, not in the middle of a word."""
    first = BATCH.split(". ")[0] + "."
    assert len(first) > 180, "premise"
    head = advisory_headline(BATCH, 180)
    assert head.endswith(ELLIPSIS) and len(head) <= 180
    assert BATCH[len(head) - 1:].startswith((" -- ", ": ", ", "))


def test_a_real_first_sentence_over_the_default_cap_is_cut_at_a_clause_break():
    """The same at the cap the app uses: the health check's opening sentence is the longer."""
    first = SHORT_RECORDING.split(". ")[0] + "."
    assert len(first) > 230, "premise"
    head = advisory_headline(SHORT_RECORDING)
    assert head.endswith(ELLIPSIS) and len(head) <= 230
    assert SHORT_RECORDING[len(head) - 1:].startswith((" -- ", ": ", ", "))


def test_the_longest_advisory_is_one_short_line_as_a_headline():
    """The reason for all of it, as a number: what the longest advisory adds to the block."""
    head = advisory_headline(TWO_LEVEL)
    assert len(TWO_LEVEL.split()) > 100 and len(head.split()) < 15
    assert len(head) * 4 < len(TWO_LEVEL)


def test_a_stop_inside_an_abbreviation_does_not_end_the_headline():
    text = "The baseline moved 3 pA -- i.e. less than the noise. Read it as signal."
    assert advisory_headline(text) == "The baseline moved 3 pA -- i.e. less than the noise."
    assert advisory_headline("It sat off by 2 pA, e.g. a step. Then it came back.") == (
        "It sat off by 2 pA, e.g. a step.")
    assert advisory_headline("Baseline vs. trace differ. Check both.") == (
        "Baseline vs. trace differ.")


def test_a_stop_closed_over_by_a_quotation_mark_or_bracket_still_ends_the_sentence():
    assert advisory_headline("Open the tab and press \u201cMeasure.\u201d Then wait.") == (
        "Open the tab and press \u201cMeasure.\u201d")
    assert advisory_headline("The ratio is high (8.0x.) Read it as signal.") == (
        "The ratio is high (8.0x.)")


def test_a_decimal_point_does_not_end_the_headline():
    assert advisory_headline("The window is 0.5 s long. Next.") == "The window is 0.5 s long."
    assert advisory_headline("Holds 96.4% of it, over 3.3x the level. Next.") == (
        "Holds 96.4% of it, over 3.3x the level.")


def test_a_line_break_ends_the_headline():
    assert advisory_headline("First line without a stop\nSecond line.") == (
        "First line without a stop")
    assert advisory_headline("  padded   ") == "padded"


_LONG_FIRST_SENTENCE = (
    "The tracked baseline sits 4.1 pA from this recording's own level across its first 0.05 s "
    "(1.2 sigma, against the interior), and that is only 1.5x the drift -- the boundary is off "
    "by no more than the baseline moves elsewhere: a baseline that never followed the trace, "
    "which is what the end of a recording looks like when the estimator pads by reflection.")


def test_a_long_sentence_is_cut_at_the_last_clause_break_that_fits():
    limit = 180
    text = _LONG_FIRST_SENTENCE + " A second sentence."
    assert len(_LONG_FIRST_SENTENCE) > limit, "premise"
    head = advisory_headline(text, limit)
    assert head.endswith(ELLIPSIS) and len(head) <= limit
    cut = len(head) - 1
    assert text[cut:].startswith((" -- ", ": ", ", ")), (
        f"cut at {text[cut:cut + 6]!r}: not a clause break")
    # ...and the LAST one that fits: no later break leaves room for the ellipsis.
    later = [m.start() for m in re.finditer(r" -- |: |, ", text) if cut < m.start() <= limit - 1]
    assert not later, f"a later break at {later} would have kept more"
    # The same sentence under a tighter cap is cut earlier, at an earlier break.
    tighter = advisory_headline(text, 120)
    assert len(tighter) <= 120 and len(tighter) < len(head)
    assert text.startswith(tighter[:-1])


def test_a_headline_is_never_cut_inside_a_bracket_or_quotation():
    text = ("These depths were measured under windows (rough 0.5 s, base 0.1 s, both left at the "
            "app's own defaults and never measured on this recording, so they are only as good as "
            "those defaults are for a pore like this one) and nothing here can tell otherwise.")
    head = advisory_headline(text, 150)
    kept = head[:-1]
    assert head.endswith(ELLIPSIS) and len(text) > 150, "premise: this one is cut"
    assert kept.count("(") <= kept.count(")"), f"cut inside the parenthesis: {head!r}"
    quoted = ("Open the Preprocess tab and press “Measure this recording, then confirm "
              "the windows it proposes, one by one” before trusting a depth from this table.")
    head = advisory_headline(quoted, 80)
    assert head.endswith(ELLIPSIS) and head[:-1].count("“") <= head[:-1].count("”"), head


def test_a_sentence_that_fits_exactly_is_not_cut_and_one_more_character_is():
    exact = "x" * 179 + "."
    assert len(exact) == 180 and advisory_headline(exact, 180) == exact
    over = "y" * 100 + " " + "z" * 80
    assert len(over) == 181
    cut = advisory_headline(over, 180)
    assert cut == "y" * 100 + ELLIPSIS, "no clause break: the last whole word that fits"


def test_the_default_cap_is_230_characters():
    """The cap the app uses: a sentence that fits whole is not cut, one more character is."""
    exact = "x" * 229 + "."
    assert len(exact) == 230 and advisory_headline(exact) == exact
    over = "y" * 150 + " " + "z" * 80
    assert len(over) == 231
    assert advisory_headline(over) == "y" * 150 + ELLIPSIS, "no clause break: the last word"


# Where a cut may land. The contract above (a capped prefix) says nothing about WHERE, and a cut in
# the wrong place reads as broken: inside a dash aside, in the middle of a list of numbers, one
# item after a colon, on an article. Each real sentence below is one the first rule cut badly,
# at the cap that showed it; the headline is the exact string the rule gives, and says what the
# first rule did. (The ones that read whole at 230 are pinned at 230 too.)
_ASIDE = ("These depths were measured with open-pore windows COPIED FROM THE BATCH FORM "
          "(0.5 s / 0.1 s): set for a DIFFERENT recording")
_REAL_CUTS = (
    # A cut inside the " -- ... -- " aside ("... recording -- the one selected when the batch
    # form was filled in…"): a headline stops before an aside it cannot finish.
    pytest.param(BATCH, 180, _ASIDE + ELLIPSIS, id="batch-aside-at-180"),
    pytest.param(BATCH, 230, _ASIDE + " -- the one selected when the batch form was filled in "
                 "-- and this recording was never measured.", id="batch-whole-at-230"),
    # A cut inside a list of three numbers read as one measured value ("fractions of 1.0000…").
    pytest.param(SHORT_RECORDING, 180,
                 "the recording (0.2 s) is SHORTER than rough_s (0.5 s), so pass 1 sees one "
                 "reflected window and the mask is arbitrary" + ELLIPSIS, id="short-recording-180"),
    pytest.param(SHORT_RECORDING, 230,
                 "the recording (0.2 s) is SHORTER than rough_s (0.5 s), so pass 1 sees one "
                 "reflected window and the mask is arbitrary" + ELLIPSIS, id="short-recording-230"),
    # A cut two words after a colon, the parameter list it announces gone ("...: masked median…").
    pytest.param(CLAMPED, 180,
                 "60000 samples (1.2 s) is shorter than the 2 s tracking window (100000 "
                 "samples), so 'masked_median' ran with its window(s) CLAMPED to this "
                 "recording" + ELLIPSIS, id="clamped-180"),
    pytest.param(CLAMPED, 230,
                 "60000 samples (1.2 s) is shorter than the 2 s tracking window (100000 "
                 "samples), so 'masked_median' ran with its window(s) CLAMPED to this "
                 "recording: masked median, rough 1200 ms / base 480 ms, decim auto, mask "
                 "3.5 sigma" + ELLIPSIS, id="clamped-230"),
    # A cut on an article ("...are omitted from the…").
    pytest.param(RECONCILE, 180,
                 "At least one input lacks an explicit analysed sample span, so the combined "
                 "coverage cannot be computed without guessing; event rate and analysed span "
                 "are omitted" + ELLIPSIS, id="reconcile-180"),
    pytest.param(RECONCILE, 230, RECONCILE, id="reconcile-whole-at-230"),
    # KNOWN LIMIT: the headline is a sound clause but its "either" is left without the "or" that
    # the same sentence goes on to give. Nothing in a cut rule can know that, and the stored text
    # is not reworded here (an owner decision); both caps stop at the comma before it.
    pytest.param(INTRUSION, 180,
                 "10 event(s) are flagged for local baseline intrusion: either the near-flank "
                 "fraction itself exceeds the 10% advisory threshold" + ELLIPSIS, id="intrusion-180"),
    pytest.param(INTRUSION, 230,
                 "10 event(s) are flagged for local baseline intrusion: either the near-flank "
                 "fraction itself exceeds the 10% advisory threshold" + ELLIPSIS, id="intrusion-230"),
)


@pytest.mark.parametrize("text, limit, expected", _REAL_CUTS)
def test_a_real_sentence_is_cut_where_it_pauses(text, limit, expected):
    assert advisory_headline(text, limit) == expected


def _clamped_sentence(items: str) -> str:
    return ("The estimator ran with its windows CLAMPED to this recording: " + items
            + ", rough 1200 ms and base 480 ms, which is the only width that fits it here.")


_CLAMPED_HEAD = "The estimator ran with its windows CLAMPED to this recording"

#: Each of the four guards on its own, on a sentence made to need it. What the first rule gave is
#: in the comment: the guard is what makes the difference.
_GUARD_CASES = (
    # Inside the aside: the first rule took the second dash, which keeps "-- the one selected
    # when the form was filled in" and stops there.
    pytest.param(
        "The depths here were measured with a window set for a different recording -- the one "
        "selected when the form was filled in -- and this recording was never measured at all, "
        "so read them with care.", 150,
        "The depths here were measured with a window set for a different recording" + ELLIPSIS,
        id="never-inside-an-aside"),
    # On a dash: the first rule gave "Words before --…" (a lone dash) and "…the rest…" (a pair).
    pytest.param("Words before -- " + "q" * 120, 60, "Words before" + ELLIPSIS,
                 id="never-on-a-lone-dash"),
    pytest.param("Words before -- the rest -- " + "q" * 120, 60, "Words before" + ELLIPSIS,
                 id="never-on-the-closing-dash"),
    # Between two numbers: the first rule took the comma after 1.0000 ("...as follows: 1.0000…").
    pytest.param(
        "The mask fractions were measured as follows: 1.0000, 0.4520 and 0.0902 for windows of "
        "0.001, 0.01 and 0.1 s on the same trace, with no error at any length.", 70,
        "The mask fractions were measured as follows" + ELLIPSIS, id="numbers-stay-together"),
    # A comma a few words past a colon: the colon is where the sentence paused. Three words or
    # fewer take the colon; a fourth is a clause of its own and keeps the comma.
    pytest.param(_clamped_sentence("masked median"), 105, _CLAMPED_HEAD + ELLIPSIS,
                 id="two-words-after-a-colon"),
    pytest.param(_clamped_sentence("masked median filter"), 105, _CLAMPED_HEAD + ELLIPSIS,
                 id="three-words-after-a-colon"),
    pytest.param(_clamped_sentence("masked median filter bank"), 105,
                 _CLAMPED_HEAD + ": masked median filter bank" + ELLIPSIS,
                 id="four-words-after-a-colon-keep-the-comma"),
)


@pytest.mark.parametrize("text, limit, expected", _GUARD_CASES)
def test_the_cut_rule_keeps_clear_of_the_places_that_read_as_broken(text, limit, expected):
    assert advisory_headline(text, limit) == expected


# The contract, over everything the analysis layer says in sentences (the advisories are composed
# there): any text, at the caps in use and at a tighter one.
def _long_strings_of_the_analysis_modules() -> list[str]:
    root = Path(nanoscope.__file__).resolve().parent
    found: set[str] = set()
    for path in [*sorted((root / "analysis").glob("*.py")), root / "core" / "acquisition.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) >= 60 and " " in node.value:
                    found.add(node.value)
    return sorted(found)


def test_the_contract_holds_for_every_long_string_the_analysis_modules_write():
    texts = _long_strings_of_the_analysis_modules()
    assert len(texts) > 500, "premise: a real corpus"
    for limit in (230, 180, 100):
        for text in texts:
            head = advisory_headline(text, limit)
            stored = text.strip()
            cut = head.endswith(ELLIPSIS) and head != stored
            kept = head[:-1] if cut else head
            assert len(head) <= limit, (limit, text[:80], head)
            assert stored.startswith(kept), (limit, text[:80], head)
            assert kept, (limit, text[:80])
            assert not (cut and re.search(r"(^|\s)--$", kept)), (
                f"a headline never stops on a dash: {head[-40:]!r}")


# ================================================================================================
# THE SUMMARY BLOCK
# ================================================================================================
@pytest.fixture
def stats(qapp, qtbot):
    panel = StatsPanel()
    qtbot.addWidget(panel)
    return panel


ADVISORIES = (SHORT, UNPROBED, TWO_LEVEL, BATCH)


def test_the_summary_shows_one_headline_bullet_per_advisory(stats):
    stats.set_warnings(ADVISORIES)
    shown = stats.warnings_shown_text()
    heading, *bullets = shown.split("\n")
    assert heading == "⚠ About these numbers"
    assert bullets == [f"• {advisory_headline(t)}" for t in ADVISORIES], (
        "one bullet per advisory, each its headline and nothing else")
    assert shown == _headline_block(ADVISORIES)
    assert len(shown) < len(_full_block(ADVISORIES)) / 3, "the point: the block is a fraction as long"


def test_the_switch_shows_exactly_the_text_the_block_always_had_and_back(stats):
    stats.set_warnings(ADVISORIES)
    toggle = stats.warnings_toggle
    assert toggle.isCheckable() and not toggle.isChecked()
    assert toggle.isVisibleTo(stats) and toggle.text() == "Show full text"
    assert stats.warnings_label.text() == _headline_block(ADVISORIES)

    toggle.click()
    assert toggle.isChecked() and toggle.text() == "Show first lines"
    assert stats.warnings_label.text() == _full_block(ADVISORIES), (
        "the same label, now reading today's text byte for byte")
    assert stats.warnings_shown_text() == _full_block(ADVISORIES)

    toggle.click()
    assert not toggle.isChecked() and toggle.text() == "Show full text"
    assert stats.warnings_label.text() == _headline_block(ADVISORIES)


def test_warnings_text_is_the_full_block_whichever_way_the_switch_is_set(stats):
    stats.set_warnings(ADVISORIES)
    full = _full_block(ADVISORIES)
    tooltip = "\n\n".join(ADVISORIES)
    for pressed in (False, True, False):
        stats.warnings_toggle.setChecked(pressed)
        assert stats.warnings_text() == full, f"warnings_text() moved with the switch ({pressed})"
        assert stats.warnings_label.toolTip() == tooltip, "the label's tooltip is the full text"
    stats.set_warnings(())
    assert stats.warnings_text() == "" and stats.warnings_shown_text() == ""


def test_the_switch_is_hidden_when_nothing_was_cut(stats):
    stats.set_warnings([SHORT])
    assert stats.warnings_label.isVisibleTo(stats)
    assert not stats.warnings_toggle.isVisibleTo(stats), "nothing to expand: no switch"
    assert stats.warnings_shown_text() == stats.warnings_text() == _full_block([SHORT])

    stats.set_warnings([SHORT, UNPROBED])
    assert stats.warnings_toggle.isVisibleTo(stats)

    stats.set_warnings(())
    assert not stats.warnings_label.isVisibleTo(stats)
    assert not stats.warnings_toggle.isVisibleTo(stats)


def test_the_switch_keeps_its_setting_while_the_advisories_change(stats):
    """It is a view setting, like the other disclosures: a reader who asked for the full text
    keeps getting it for the next recording, until they press it again."""
    stats.set_warnings(ADVISORIES)
    stats.warnings_toggle.click()
    other = (UNPROBED, ACQUISITION)
    stats.set_warnings(other)
    assert stats.warnings_toggle.isChecked()
    assert stats.warnings_shown_text() == _full_block(other)
    stats.set_warnings([SHORT])                      # nothing cut: no switch, same text either way
    assert stats.warnings_shown_text() == _full_block([SHORT])
    stats.set_warnings(other)
    assert stats.warnings_shown_text() == _full_block(other), "still open when it is offered again"


def test_the_switch_offers_no_way_to_silence_the_block(stats):
    stats.set_warnings(ADVISORIES)
    forbidden = ("dismiss", "don't show", "do not show", "hide", "ignore", "×")
    for pressed in (False, True):
        stats.warnings_toggle.setChecked(pressed)
        label = stats.warnings_toggle.text().lower()
        assert not any(word in label for word in forbidden), label
        assert stats.warnings_label.isVisibleTo(stats), "pressing the switch never hides the block"


@pytest.fixture
def stats_after_a_field(qapp, qtbot):
    """A ``StatsPanel`` under a line edit: the stand-in for whatever the reader was on before the
    Summary tab. Yields ``(host, before, stats)``; the host owns the panel."""
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    column = QtWidgets.QVBoxLayout(host)
    before = QtWidgets.QLineEdit()
    panel = StatsPanel()
    column.addWidget(before)
    column.addWidget(panel)
    yield host, before, panel


def test_the_switch_is_reachable_by_tab_and_has_a_name(stats_after_a_field, qtbot):
    host, before, stats = stats_after_a_field
    stats.set_warnings(ADVISORIES)
    host.resize(520, 700)
    host.show()
    qtbot.waitExposed(host)
    host.activateWindow()
    QTest.qWaitForWindowActive(host)
    toggle = stats.warnings_toggle
    table = stats.summary_table

    assert toggle.focusPolicy() & QtCore.Qt.TabFocus, "Tab must be able to land on the switch"
    assert toggle.isEnabled() and toggle.isVisibleTo(stats)

    # A name a screen reader reads, kept in step with the text on the button...
    assert toggle.accessibleName() == "Show full text"
    assert toggle.accessibleDescription()
    iface = QAccessible.queryAccessibleInterface(toggle)
    assert iface is not None and iface.text(QAccessible.Name) == "Show full text"
    # ...and it says what pressing it does, in both states.
    toggle.setChecked(True)
    assert toggle.accessibleName() == "Show first lines"
    assert QAccessible.queryAccessibleInterface(toggle).text(QAccessible.Name) == "Show first lines"
    toggle.setChecked(False)

    # The real keys. The switch sits under the table but comes BEFORE it in the tab order: Tab
    # from the widget before the panel lands on it, Space presses it, and the next Tab is the table.
    before.setFocus()
    QTest.qWait(10)
    assert QtWidgets.QApplication.focusWidget() is before
    QTest.keyClick(before, QtCore.Qt.Key_Tab)
    assert QtWidgets.QApplication.focusWidget() is toggle
    QTest.keyClick(toggle, QtCore.Qt.Key_Space)
    assert toggle.isChecked()
    assert stats.warnings_shown_text() == _full_block(ADVISORIES)
    QTest.keyClick(toggle, QtCore.Qt.Key_Tab)
    assert QtWidgets.QApplication.focusWidget() is table

    # The table's own keys are the ones it always had, on purpose: Tab steps down its rows and
    # wraps from the last to the first, which is why the switch had to come before it.
    assert table.tabKeyNavigation()
    table.setCurrentCell(table.rowCount() - 1, 0)
    QTest.keyClick(table, QtCore.Qt.Key_Tab)
    assert QtWidgets.QApplication.focusWidget() is table and table.currentRow() == 0
    table.setCurrentCell(0, 0)
    QTest.keyClick(table, QtCore.Qt.Key_Backtab)
    assert QtWidgets.QApplication.focusWidget() is table and table.currentRow() == table.rowCount() - 1

    # Nothing cut, no switch: Tab goes from the widget before straight to the table, as it did.
    stats.set_warnings([SHORT])
    assert not toggle.isVisibleTo(stats)
    before.setFocus()
    QTest.qWait(10)
    QTest.keyClick(before, QtCore.Qt.Key_Tab)
    assert QtWidgets.QApplication.focusWidget() is table


# ================================================================================================
# THE TOOLBAR LINK AND THE STORED TEXT, THROUGH THE REAL PANEL
# ================================================================================================
@pytest.fixture
def shown(qapp, qtbot, tmp_path):
    """An ``AnalysisPanel`` on a real session with a result installed through the store, so the
    panel repaints by its own path. ``install(warnings)`` returns the installed result."""
    app = QtWidgets.QApplication.instance()
    org, name = app.organizationName(), app.applicationName()
    app.setOrganizationName("NanoScopeTest")
    app.setApplicationName("AdvisoryHeadlines")
    QtCore.QSettings().clear()
    session = Session()
    trace = TraceView(session)
    qtbot.addWidget(trace)
    controller = AnalysisController(session)
    panel = AnalysisPanel(session, controller, trace)
    qtbot.addWidget(panel)
    doc, events = _helpers["_doc_and_events"](tmp_path, n_events=3)
    doc_id = session.add_document(doc)
    session.add_layer(doc_id, 0)
    panel.set_selection(doc_id, 0)
    events.provenance.update(baseline_method="constant", polarity=1,
                             polarity_evidence={"verdict": "down"})
    events.provenance["measurement_quality"] = build_measurement_quality(
        doc.channels[0], np.full(doc.meta.n_samples, _helpers["BASE"]), events.df,
        doc.meta.sample_rate_hz, events.provenance)

    def install(warnings):
        result = AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={}, calibration_fingerprint=(1.0, 0.0), payload=events,
            warnings=tuple(warnings))
        session.add_result(result)
        return result

    try:
        yield panel, session, install
    finally:
        controller.abort()
        QtCore.QSettings().clear()
        app.setOrganizationName(org)
        app.setApplicationName(name)


def _all_advisories(result: AnalysisResult) -> tuple[str, ...]:
    """What ``AnalysisPanel._show_result_advisories`` shows for ``result``: the module's own
    warnings, then the quality and acquisition messages, each once."""
    events = result.payload
    return tuple(dict.fromkeys(str(item) for item in (
        *result.warnings,
        *report_recording_messages(events.df, events.provenance or {}),
        *acquisition_snapshot_messages(events.provenance or {}),
    ) if str(item)))


def test_the_toolbar_tooltip_is_the_lead_line_and_one_headline_bullet_each(shown):
    panel, _session, install = shown
    result = install((UNPROBED, TWO_LEVEL, BATCH))
    items = _all_advisories(result)
    assert len(items) >= 4, "premise: the three above plus the quality and acquisition notes"

    notice = panel._result_notice
    n = len(items)
    lead = f"Analysis advisories ({n}) — click to review them on the Summary tab."
    assert notice.toolTip() == lead + "\n\n" + "\n".join(
        f"• {advisory_headline(t)}" for t in items)
    assert notice.toolTip().count("\n• ") == n and notice.toolTip().count("•") == n
    # Link text, accessible name and count are the ones they always were.
    assert f"⚠ {n} advisories" in notice.text()
    assert notice.accessibleName() == f"Analysis advisories ({n})"
    # ...and it is a fraction of what the tooltip used to be.
    assert len(notice.toolTip()) < len(lead + "\n\n" + "\n\n".join(items)) / 2


def test_the_summary_and_the_tooltip_list_the_same_advisories(shown):
    panel, _session, install = shown
    install((UNPROBED, TWO_LEVEL))
    stats = panel._stats
    full = stats.warnings_text().split("\n• ")[1:]
    bullets = panel._result_notice.toolTip().split("\n\n", 1)[1].split("\n")
    assert bullets == [f"• {advisory_headline(t)}" for t in full]
    assert stats.warnings_shown_text().split("\n")[1:] == bullets


def test_showing_the_advisories_leaves_the_stored_text_exactly_as_it_was(shown):
    panel, session, install = shown
    stored = (UNPROBED, TWO_LEVEL, BATCH)
    result = install(stored)
    provenance = deepcopy(result.payload.provenance)

    assert result.warnings == stored
    assert session.results.get(result.doc_id, 0, "detect.schmitt").warnings == stored
    assert panel._stats.warnings_text() == _full_block(_all_advisories(result)), (
        "warnings_text() is today's block: heading, then every advisory whole")
    for pressed in (True, False, True):
        panel._stats.warnings_toggle.setChecked(pressed)
        assert panel._stats.warnings_text() == _full_block(_all_advisories(result))
        assert result.warnings == stored
    assert result.payload.provenance == provenance


def test_clearing_the_warnings_still_leaves_the_recorded_notes_and_their_headlines(shown):
    panel, _session, install = shown
    result = install(())
    items = _all_advisories(result)
    assert items, "premise: a result always carries its quality and acquisition notes"
    assert panel._stats.warnings_text() == _full_block(items)
    assert panel._result_notice.toolTip().count("•") == len(items)


def _press_the_link(panel, viewport_height):
    """The "N advisories" link (``show_result_summary``), pressed from another tab with the Summary
    tab's viewport ``viewport_height`` px tall. Returns that viewport and where the block's top
    and the switch's bottom then are in it, in pixels (0 is the top edge)."""
    tabs = panel.results_tabs
    summary = tabs.widget(1)
    tabs.setCurrentIndex(1)
    tabs.setFixedHeight(400)
    QTest.qWait(30)
    chrome = tabs.height() - summary.viewport().height()
    tabs.setFixedHeight(viewport_height + chrome)
    tabs.setCurrentIndex(0)
    QTest.qWait(30)
    panel.show_result_summary()
    QTest.qWait(30)
    assert panel.results_tabs.currentIndex() == 1
    viewport = summary.viewport()
    assert viewport.height() == viewport_height, "premise: the dock is the height asked for"
    stats = panel._stats
    top = stats.warnings_label.mapTo(viewport, QtCore.QPoint(0, 0)).y()
    bottom = stats.warnings_toggle.mapTo(viewport, QtCore.QPoint(0, 0)).y() + stats.warnings_toggle.height()
    return viewport, top, bottom


@pytest.fixture
def block_and_switch(shown, qtbot):
    """The panel with a three-advisory result installed and its Summary tab laid out, and the
    block's height and the distance from the block's top to the switch's bottom, which is what a
    viewport has to hold to show both."""
    panel, _session, install = shown
    install((UNPROBED, TWO_LEVEL, BATCH))
    panel.resize(900, 600)
    panel.show()
    qtbot.waitExposed(panel)
    panel.results_tabs.setCurrentIndex(1)
    QTest.qWait(30)
    stats = panel._stats
    label, toggle = stats.warnings_label, stats.warnings_toggle
    assert toggle.isVisibleTo(stats) and not toggle.isChecked()
    return panel, label.height(), toggle.y() + toggle.height() - label.y()


def test_the_advisories_link_shows_the_switch_too_when_both_fit(block_and_switch):
    """A dock with room for the block and the switch under it shows both, whole."""
    panel, block, both = block_and_switch
    viewport, top, bottom = _press_the_link(panel, both + 4 + 40)
    assert 0 <= top, f"the block's heading is at y={top}"
    assert bottom <= viewport.height(), f"the switch ends at y={bottom} in {viewport.height()} px"


@pytest.mark.parametrize("room", ["block_only", "half_the_block"])
def test_the_advisories_link_lands_on_the_heading_when_they_cannot_both_fit(block_and_switch, room):
    """Too short for the block and the switch: the HEADING wins. The reader sent here by "N
    advisories" lands on the heading and the first advisory, not on the tail of the third with the
    switch under it."""
    panel, block, both = block_and_switch
    height = {"block_only": block + 8, "half_the_block": block // 2}[room]
    assert height < both, "premise: the switch does not fit with the block"
    assert height >= 3 * panel._stats.warnings_label.fontMetrics().lineSpacing(), (
        "premise: room for the heading and the first advisory")
    viewport, top, _bottom = _press_the_link(panel, height)
    assert top == 0, f"the block's heading is at y={top}, not at the top of the dock"


def test_the_advisories_link_lands_on_the_heading_of_a_block_that_is_open(block_and_switch):
    """Once the full text is open the block is long and the switch is at its far end: the link
    still lands on the heading."""
    panel, _block, _both = block_and_switch
    stats = panel._stats
    stats.warnings_toggle.setChecked(True)
    QTest.qWait(30)
    viewport, top, _bottom = _press_the_link(panel, stats.warnings_label.height() // 3)
    assert top == 0, f"the block's heading is at y={top}, not at the top of the dock"


# ================================================================================================
# THE DETECTION FORM'S ROW LABELS (the same pass: plain words, one style)
# ================================================================================================
def _row_label(field: QtWidgets.QWidget) -> str:
    """The words the form puts beside ``field``. Rows can move to another form (the advanced
    ones live in a disclosure), so every form above the field is asked."""
    parent = field.parentWidget()
    while parent is not None:
        for form in (parent.layout(), *parent.findChildren(QtWidgets.QFormLayout)):
            if isinstance(form, QtWidgets.QFormLayout) and form.labelForField(field) is not None:
                return form.labelForField(field).text()
        parent = parent.parentWidget()
    raise AssertionError(f"{field!r} is in no form row")


def test_the_detection_form_names_its_rows_in_plain_words(qapp, qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    rows = {
        "detector_combo": "Detector",
        "k_in": "Start (k_in)",
        "k_out": "End (k_out)",
        "cusum_k": "Drift (k)",
        "cusum_h": "Threshold (h)",
        "min_dwell_s": "Min dwell",
        "max_dwell_s": "Max dwell",
        "merge_gap_ms": "Merge gap",
        "range_readout": "Runs over",
        "polarity_readout": "Found in data",
        "polarity_override": "Override",
    }
    for attribute, label in rows.items():
        assert _row_label(getattr(panel, attribute)) == label, attribute
        assert len(label) <= 16, "the Detection page is laid out for labels up to 'Noise thresholds'"


def test_the_essential_summary_line_names_the_rows_as_the_form_does(qapp, qtbot):
    """The line over the form lists the settings Essential keeps out of sight, under the words
    their rows have: 'Max dwell', 'Merge gap', not 'max dwell' and 'merge'."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_params({"max_dwell_s": 0.07, "merge_gap_s": 0.002})
    panel.set_control_level("essential")
    summary = panel.control_level_summary()
    assert f"{_row_label(panel.max_dwell_s)} 70 ms" in summary, summary
    assert f"{_row_label(panel.merge_gap_ms)} 2 ms" in summary, summary
