"""X-SURF-3 — ``docs/inventory.json`` QUOTES the app, so the app is what it is checked against.

The control reference in the manual is rendered from this file (``docs/gen_manual.py``), and
its ``what_it_does`` / ``notes`` / ``correction`` columns quote tooltips, status lines, row
names and box titles verbatim. Nothing regenerates them, so every sweep that rewrites a
surface leaves the manual asserting a string the user cannot find on screen. Sweep 7 rewrote
five of them and the file was updated for none:

* the probe button's tooltip — the entry held a string written from a fix report rather than
  from the widget ("Look at this recording (usually a few seconds — measured 2 s …)"), which
  the button has never said;
* the "These values: …" provenance set — it listed a ``fallback`` wording that no longer
  exists and omitted the ``batch`` label entirely, so the one line that admits "these windows
  were confirmed for a DIFFERENT recording" was undocumented;
* the detector score box — "Its eight rows, verbatim and in order: … missed (FN), precision"
  against a panel that grew a ninth row, ``auto-rejected``, between those two;
* the gallery status line — quoted with an unconditional "1–9 to label" hint the window only
  prints for the digits a palette ARMS, and with "double-click opens it in the trace" against
  a window that says "in the Event Inspector";
* the calibration preview box — grouped under the old title "Raw vs calibrated (sampled)" with
  every row corrected to "every number comes from a sample of at most 20,000 points", while
  min, max and span are now exact over every sample and only the median is strided.

Written the way ``tests/test_docs_merge_and_format_agree_with_app.py`` is: read the app's own
strings off real widgets first, then check the document against THEM. A doc test that
hard-codes tonight's wording rots in step with the doc and cannot tell a fixed sentence from a
re-broken one; every assertion below fails the moment a surface moves again, whichever way it
moves. The inventory is located from the imported package rather than from this file, so the
same module can be pointed at another checkout (the pre-sweep-7 snapshot) and measure it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QLabel

from nanoscope.core.session import Session

_PKG_ROOT = Path(nanoscope.__file__).resolve().parents[1]
#: ``docs/inventory.json`` in a checkout; ``inventory.json`` beside ``nanoscope/`` in the
#: report snapshots, which carry the documents without the rest of ``docs/``.
_INVENTORY = next(p for p in (_PKG_ROOT / "docs" / "inventory.json",
                              _PKG_ROOT / "inventory.json") if p.exists())

_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


@pytest.fixture(scope="module")
def inventory() -> list[dict]:
    return json.loads(_INVENTORY.read_text(encoding="utf-8"))


def _controls(inventory: list[dict]) -> list[dict]:
    return [c for area in inventory for c in area["controls"]]


def _one(inventory: list[dict], label: str, group: str) -> dict:
    hits = [c for c in _controls(inventory)
            if c["label"] == label and c.get("group") == group]
    assert len(hits) == 1, f"expected exactly one {label!r} in {group!r}, got {len(hits)}"
    return hits[0]


def _prose(control: dict) -> str:
    """Every column of an entry that the manual renders as a sentence."""
    return "  ".join(str(control.get(k, "") or "")
                     for k in ("what_it_does", "notes", "correction"))


def _quoted_run_count(text: str, upto: str) -> int:
    """How many ``'…'`` spans the text opens before the end of ``upto``.

    The provenance and row enumerations are lists of quoted strings; counting the spans is
    how an enumeration that quotes a LABEL THE PANEL CANNOT PRINT is caught, which naming
    the retired wording could not do without hard-coding it.
    """
    end = text.index(upto) + len(upto) + 1
    return len(re.findall(r"'[^']*'", text[:end]))


def _in_order(haystack: str, needles) -> None:
    at = -1
    for n in needles:
        i = haystack.find(n, at + 1)
        assert i > at, f"{n!r} is missing from the entry, or is quoted out of the app's order"
        at = i


# ---------------------------------------------------------------------------------------------
# 1. The probe button: the entry IS the tooltip
# ---------------------------------------------------------------------------------------------
def test_the_measure_this_recording_entry_is_the_buttons_tooltip_byte_for_byte(
        qapp, inventory):
    """Not "says the same" — the same bytes.

    The manual quotes this tooltip in quotation marks, and the string it carried was written
    from a fix report: it promised a parenthetical timing the button states as a separate
    sentence, and dropped the health-check clause and the "it has not missed the click"
    reassurance the button actually gives. Equality is the only check that keeps a
    hand-written paraphrase out of a column the manual presents as a quotation.
    """
    from nanoscope.gui.baseline_panel import BaselinePanel

    panel = BaselinePanel()
    tooltip = panel.probe_button.toolTip()
    entry = _one(inventory, "Measure this recording", "Open-pore baseline")
    assert entry["what_it_does"] == tooltip


# ---------------------------------------------------------------------------------------------
# 2. "These values: …" — every provenance the panel can print, and no other
# ---------------------------------------------------------------------------------------------
def test_the_these_values_entry_quotes_every_provenance_label_the_panel_can_print(
        qapp, inventory):
    """The staleness story is only as good as its rarest line.

    ``batch`` is the label that says the windows were confirmed for a DIFFERENT recording;
    a reader who never sees it documented has no reason to distrust the number in front of
    them. The count check catches the other half — a wording the panel has retired, left
    behind in the list as if it were still reachable.
    """
    from nanoscope.gui.baseline_panel import _SOURCE_LABELS

    labels = list(_SOURCE_LABELS.values())
    notes = _one(inventory, "These values: …", "Open-pore baseline")["notes"]
    _in_order(notes, labels)
    assert _quoted_run_count(notes, labels[-1]) == len(labels), (
        "the enumeration quotes a provenance label BaselinePanel cannot print")


# ---------------------------------------------------------------------------------------------
# 3. The detector score box: nine rows, and the entry says nine
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("group", [
    "Score panel (same file)",
    "Statistics panel (panels.py ScorePanel) — sits with the Summary",
])
def test_the_detector_score_entries_name_every_row_the_panel_draws_in_its_order(
        qapp, inventory, group):
    """Both copies of the entry, because both reach the page.

    ``auto-rejected`` is not decoration: it is where the rows the DETECTOR rejected on its
    own now land, having previously been published as its false positives. An entry that
    skips it describes a panel whose precision means something else.
    """
    from nanoscope.gui.panels import ScorePanel

    rows = [label for label, _key in ScorePanel._ROWS]
    text = _prose(_one(inventory, "Detector score (vs your curation)", group))
    _in_order(text, rows)
    for word in re.findall(r"\bIts (\w+) rows\b", text):
        assert _COUNT_WORDS.get(word.lower()) == len(rows), (
            f"the entry says {word!r} rows; the panel draws {len(rows)}")


# ---------------------------------------------------------------------------------------------
# 4. The gallery status line: the gestures it really advertises
# ---------------------------------------------------------------------------------------------
def test_the_gallery_status_entry_quotes_the_gesture_clauses_the_window_prints(
        qapp, inventory):
    """Driven through the real window, because the line is assembled clause by clause.

    The clauses that carry no digit are exactly the gesture hints (the ones that do are the
    counts and the armed-digit span, which are per-session), so they are the part of the
    line a manual can quote at all — and every one of them must be quotable from the entry.
    On an empty palette that set includes the sentence naming ``Analysis ▸ Event labels…``
    as the place the digit keys are armed, which is the whole point of the rewrite: the line
    used to advertise nine digits that wrote nothing.
    """
    from nanoscope.gui.event_gallery import EventGalleryWindow

    gallery = EventGalleryWindow(Session())
    frame = pd.DataFrame({
        "event_id": np.arange(4),
        "start_idx": np.arange(4) * 1000,
        "end_idx": np.arange(4) * 1000 + 200,
        "dwell_s": np.full(4, 0.004),
        "depth_pA": np.full(4, 120.0),
        "accepted": np.ones(4, dtype=bool),
    })
    gallery.set_events("doc0", 0, frame, "threshold", {})
    printed = [c.strip() for c in gallery.status.text().split("·")]
    gestures = [c for c in printed if c and not any(ch.isdigit() for ch in c)]
    assert gestures, "the status line printed no digit-free clause to check"

    entry = _prose(_one(inventory, "(status line)", "Window foot"))
    missing = [c for c in gestures if c not in entry]
    assert not missing, f"the gallery prints these and the entry does not quote them: {missing}"


# ---------------------------------------------------------------------------------------------
# 5. The calibration preview: grouped under the box's own title
# ---------------------------------------------------------------------------------------------
def test_the_calibration_preview_rows_are_grouped_under_the_boxs_own_title(qapp, inventory):
    """The group name IS the title, so the reference cannot describe a box that is not there.

    The title is the one place the box states which of its four rows are exact and which is
    sampled; a reference filed under the old "(sampled)" spelling tells the reader all four
    are approximations, which is the claim the exactness work existed to retire.
    """
    from nanoscope.gui.calibration_panel import CalibrationPanel

    panel = CalibrationPanel(Session())  # held: the box dies with the panel that owns it
    title = panel._stats_box.title()
    assert any(c["label"] == title for c in _controls(inventory)), (
        f"no control is labelled with the preview box's title {title!r}")
    stale = {c.get("group") for c in _controls(inventory)
             if str(c.get("group", "")).startswith("Raw vs calibrated")} - {title}
    assert not stale, f"rows of the preview grid are filed under a title the box does not have: {stale}"


def test_the_median_row_is_documented_under_the_name_the_grid_gives_it(qapp, inventory):
    """The row label carries "(sampled)" now that the title no longer says it for all four.

    Read off the grid rather than from the panel's internals, so this stays true of any
    checkout in which the box exists at all.
    """
    from nanoscope.gui.calibration_panel import CalibrationPanel

    panel = CalibrationPanel(Session())  # held: the box dies with the panel that owns it
    box = panel._stats_box
    names = sorted({lbl.text() for lbl in box.findChildren(QLabel)
                    if lbl.text() == "median" or lbl.text().startswith("median ")})
    assert len(names) == 1, f"expected one median row name in the preview grid, got {names}"
    labels = {c["label"] for c in _controls(inventory) if c.get("group") == box.title()}
    assert names[0] in labels, (
        f"the grid's row is named {names[0]!r}; the reference files it as "
        f"{sorted(l for l in labels if l.startswith('median'))}")
