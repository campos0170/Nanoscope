"""Phase 0 follow-up: the documents say what the follow-up's code does, once.

The follow-up merged G6B (Normalize and Align join the reset rule; the sentence stands only while
it is true; the Undo tooltips name the step; Log scale carries a range across) and the delta
review's code fixes (R1 a recalled template's promise, R2 the Y range of a density histogram, R3
the PCA's features, R4 opening a project, R6 the Reset figure tool tip, FE1 the SVG of a stand-in
font, FE2 a composition an older Mac build saved) into the Phase 0 entries of the CHANGELOG, the
guide and the inventory. Each sentence below is pinned where it was written, and pinned to occur
once in the CHANGELOG, so a merge cannot leave it twice or leave an older sentence beside it.

Reads the three documents and one module docstring; no Qt, no window.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _flat(relative: str) -> str:
    return " ".join((ROOT / relative).read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def changelog() -> str:
    return _flat("CHANGELOG.md")


@pytest.fixture(scope="module")
def guide() -> str:
    return _flat("docs/USER_GUIDE.md")


@pytest.fixture(scope="module")
def inventory() -> list:
    return json.loads((ROOT / "docs" / "inventory.json").read_text(encoding="utf-8"))


def _notes(inventory, label, group) -> str:
    hits = [c for area in inventory for c in area["controls"]
            if c.get("label") == label and c.get("group") == group]
    assert len(hits) == 1, (label, group, len(hits))
    return hits[0].get("notes", "")


#: Sentences the CHANGELOG now carries, each once.
CHANGELOG_SENTENCES = (
    # G6B: Undo and Redo name the step the rule records.
    "the **Undo** and **Redo** tooltips name the step by what it took back (*Undo: reset title "
    "and Y label*, *Undo: reset title, Y label and Y range*, *Undo: reset X range and Y range*)",
    "Distributions' **Configuration** Undo counts steps and names none (*Undo the last change to "
    "this plot's configuration (3 to step back).*)",
    # G6B: Align, Normalize, and the veto the owner is offered.
    "**Align** (start, center, extremum), which is what t = 0 means on the ensemble's X axis once "
    "time is in seconds, so another alignment takes the words and that range too",
    "the **Normalize** you pick (count, density, probability per bin), which is what the "
    "histogram's Y axis counts, so it takes the words and the Y range in the step of the pick",
    "**Normalize and Align are a choice made beyond the owner's explicit list** (plot type, "
    "column, metric, measurement and view), for the owner to veto.",
    "and so do Distributions' Time unit and the switch to density that ticking a second source "
    "makes by itself (a Normalize you pick takes them, above).",
    # G6B: the sentence stands only while it is true; Log scale carries a range across.
    "It stands only while it is true: it says that what it names is automatic, so it comes off",
    "A range left as the dialog opened with it is carried across to the new units instead, log10 "
    "of it when Log scale is ticked and ten to the power when it is unticked",
    "A fit label whose wording a change of quantity took back is the untouched label again unless "
    "its numbers, position, size or colour were set, so Reset figure pressed straight after such "
    "a change, on a figure with nothing else customised, finds nothing to take back and records "
    "no step.",
    "and for a figure whose only customisation is typed words and ranges the words of the reset a "
    "change of quantity makes, *Undo: reset title and Y label*",
    # R6: the tool tip of Reset figure.
    "a range or tick spacing typed there goes with the look, and the button's tool tip says so",
    # R1: a recalled template's promise.
    "They are kept only while the words (and, in Distribution summaries, the ranges) are still "
    "the template's: **Undo** on the recall, a word or range typed or pinned after it, another "
    "quantity picked after it, or saving the project and opening it again before the Generate or "
    "**Apply fit** makes that an ordinary change of quantity, and they go back to automatic (one "
    "Undo step and the sentence, as for any change).",
    # R2: a density histogram's Y axis.
    "(the axis of the column that changed, and the Y axis too while Normalize is *density*, whose "
    "unit is the column's; both axes on a plot-type change)",
    "While Normalize is *density* the Y axis is per the column's unit (*density (1/s)* over a "
    "dwell time, plain *density* over a fractional depth), so a new column takes the Y range too; "
    "with *count* or *probability per bin* the Y axis does not depend on the column and keeps its "
    "range.",
    # R3: the PCA's features.
    "On the PCA page the **Features** you tick are what its two axes are made of, so another set "
    "takes the words and both ranges (PC1 and PC2 are other combinations of other columns), in "
    "the Configuration step of the tick that changed it; on any other plot type the same list is "
    "a clustering setting and takes nothing.",
    # R4: opening a project.
    "**Subpopulations: opening a project no longer carries the last project's title, labels and "
    "ranges into its first report.**",
    # FE1, FE2: fonts.
    "Times New Roman and Courier New have lists of their own that end in serif and monospace, and "
    "so does the stand-in that a computer without them draws them in (Liberation Serif, "
    "Liberation Mono, Tinos…): it is named first and followed by its family's list, so a viewer "
    "without it still gets a serif or a fixed-width face",
    "A composition keeps the family it was saved with: one an earlier Mac build saved in the "
    "application font shows it on a row of its own, *Application font (older figure)*, so its "
    "letters keep matching its panels in the PNG and the PDF, and its SVG writes Arial for those "
    "letters; choosing Arial changes the letters everywhere.",
)

#: Older sentences those replaced, which must be gone.
CHANGELOG_GONE = (
    "and so do Distributions' Time unit and a change of its histogram normalization",
    "Times New Roman and Courier New have lists of their own, and any other family is followed by "
    "Arial's)",
    "when it also takes the look back; Distributions' Undo counts steps and names none). A source "
    "you renamed",
)


@pytest.mark.parametrize("sentence", CHANGELOG_SENTENCES)
def test_the_changelog_carries_each_sentence_once(changelog, sentence):
    assert changelog.count(sentence) == 1, sentence


@pytest.mark.parametrize("sentence", CHANGELOG_GONE)
def test_the_changelog_no_longer_carries_what_the_follow_up_replaced(changelog, sentence):
    assert sentence not in changelog, sentence


def test_the_changelog_says_the_rules_new_triggers_once_in_the_window_list(changelog):
    """Align and Normalize are named in the window-by-window list where they belong, and the veto
    is one sentence of its own, not a second copy of each description."""
    assert changelog.count("Normalize and Align are a choice made beyond the owner's explicit list") == 1
    assert changelog.count("what t = 0 means on the ensemble's X axis") == 2    # the item and the veto
    assert changelog.count("what the histogram's Y axis counts") == 2           # the item and the veto


GUIDE_SENTENCES = (
    # R2, R3: the Distributions bullet and the window list.
    "(a change of plot type moves both axes). While **Normalize** is *density* a new column "
    "moves the Y axis too, since its unit is the column's.",
    "On the PCA page the **Features** you tick are its columns: another set does the same on both "
    "axes (PC1 and PC2 are other combinations of other columns), while on any other plot type that "
    "list is a clustering setting and takes nothing.",
    "While **Normalize** is *density* the Y axis is per the column's unit (*density (1/s)* over a "
    "dwell time, plain *density* over a fractional depth), so a new column takes the Y range too; "
    "with *count* or *probability per bin* the Y axis does not depend on the column and keeps its "
    "range. The **Features** ticked on the PCA page count as well: another set takes the words and "
    "both ranges, and the tick that changed it carries the reset in its own Undo step.",
    # R1: a recalled template's promise.
    "They are kept only while the words (and, in Distribution summaries, the ranges) are still "
    "the template's: **Undo** on the recall, a word or range typed or pinned after it, another "
    "quantity picked after it, or saving the project and opening it again before the **Generate** "
    "or **Apply fit** makes that an ordinary change of quantity, and they go back to automatic "
    "(one **Undo** step and the sentence, as for any change).",
    # R6: the tool tip.
    "a range or tick spacing typed there goes with the look, and the button's tool tip says so.",
    # FE1, FE2: fonts.
    "and so does the stand-in that a computer without them draws them in, such as Liberation "
    "Serif or Liberation Mono (it is named first, then its family's list); any other family is "
    "followed by Arial's list",
    "(one that an earlier Mac build saved in the application font, a private face, shows it on a "
    "row of its own, **Application font (older figure)**, and keeps it so the letters match the "
    "saved panels; its SVG writes Arial for those letters, and choosing Arial changes them "
    "everywhere)",
)


@pytest.mark.parametrize("sentence", GUIDE_SENTENCES)
def test_the_guide_says_it_once(guide, sentence):
    assert guide.count(sentence) == 1, sentence


def test_the_guide_keeps_the_sentences_the_earlier_groups_pinned_beside_the_new_ones(guide):
    """The compose sentence about a Mac's application font, and the exception for a ticked source
    that lacks the plotted column, are untouched."""
    assert ("on a Mac, where the application font is a private face, the list has no row for it"
            in guide)
    assert "unless a ticked source then lacks the plotted column" in guide


def test_the_inventory_says_it(inventory):
    assert ("A new column also takes the words you typed and the range typed or pinned on the X "
            "axis; while Normalize is density the Y range goes too") in _notes(
        inventory, "Column", "Plot ▸ Histogram")
    assert ("While Normalize is density a new column clears the Y range as well (density is per "
            "the column's unit") in _notes(inventory, "Y min", "Axis range")
    features = _notes(inventory, "Features (clustering & PCA)", "Features (clustering & PCA)")
    assert ("On the PCA page ticking or clearing a feature also puts the words you typed (Title, X "
            "label, Y label and the fit label's wording) and both pinned ranges back to "
            "automatic") in features
    assert "On any other plot type the list is a clustering setting and takes nothing." in features
    assert ("a range or tick spacing typed there goes with the look, and the button's tool tip "
            "says so") in _notes(inventory, "Reset figure", "Analysis Explorer and Distributions workflow")
    assert ("one an earlier Mac build saved in the application font (a private face) shows it on "
            "a row of its own, Application font (older figure)") in _notes(
        inventory, "Panel labels", "Compose publication figure / Revise publication figure")
    gotchas = [g for area in inventory for g in area.get("gotchas", ())
               if g.startswith("Changing what a figure shows puts the words you typed back to "
                               "automatic, and the range pinned on each axis")]
    assert len(gotchas) == 1
    assert ("the typed range of an axis whose column moved, the Y axis included while Normalize "
            "is density), the Features ticked on the PCA page (the same words and both ranges), "
            "the Normalize you pick") in gotchas[0]


def test_the_rules_docstring_names_the_features_and_the_density_axis():
    text = " ".join((ROOT / "nanoscope" / "gui" / "quantity_reset.py")
                    .read_text(encoding="utf-8").split())
    assert ("Distributions: the plot type, a column, the feature set on the PCA page, and the "
            "applied fit") in text
    assert "A histogram's y has no column of its own (a count), except as a DENSITY" in text
    assert ("(``AnalysisIdentity.features``)") in text
    assert "while the wording is still the template's" in text
