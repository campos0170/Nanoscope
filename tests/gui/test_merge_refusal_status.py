"""A MERGE REFUSAL NOBODY PRINTS IS A MERGE THAT LOOKS BROKEN.

``analysis.merge`` declines to merge a run of events whose fragments cover less than
:data:`~nanoscope.analysis.merge.MIN_MERGE_OCCUPANCY` of the span they would make: two
events 600 ms apart are not one translocation, and merging them would report a single
604 ms event 0.05 pA deep. The refusal is right. What was missing is the sentence.

``MergeReport`` grew a ``"low_occupancy"`` status and an ``n_refused_low_occupancy``
count, and ``MainWindow._post_merge_status`` had no branch for either — so a whole-window
refusal fell through to the final ``else`` and printed

    Merged 0 fragment(s): 40 → 40 events.

which is not merely unhelpful, it is the sentence that makes a working floor look like a
broken feature: the user widens the window, gets the same line, and concludes Merge does
nothing. The partial case was worse still, because "Merged 12 fragment(s)" said nothing
at all about the three runs that were left exactly where they were.

These tests drive ``_post_merge_status`` directly with the reports the panel builds. The
report itself is the panel's contract (``tests/gui/test_merge_gui.py`` and
``tests/analysis/`` pin the floor); this file pins that every outcome the panel can hand
back reaches the user as a sentence that is true of it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from nanoscope.gui.analysis_panel import MergeReport
from nanoscope.gui.main_window import MainWindow


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _post(win, report, **kw) -> str:
    win._post_merge_status(report, **kw)
    return win.statusBar().currentMessage()


# ================================================================================================
# nothing merged BECAUSE of the floor — the case that used to read "Merged 0 fragment(s)"
# ================================================================================================

def test_a_window_merge_refused_for_low_occupancy_says_so(win):
    msg = _post(win, MergeReport("low_occupancy", 40, 40, n_refused_low_occupancy=3),
                window_s=0.6)
    assert "Merged 0 fragment(s)" not in msg
    assert "3 run(s)" in msg
    assert "600 ms" in msg                    # the unit the spinbox speaks
    assert "open pore" in msg
    assert "smaller window" in msg            # ...and what to do about it


def test_a_region_merge_refused_for_low_occupancy_says_so(win):
    msg = _post(win, MergeReport("low_occupancy", 5, 5, n_refused_low_occupancy=1),
                region=True)
    assert "open pore" in msg
    assert "Nothing was changed" in msg
    assert "smaller window" not in msg        # there is no window in a region merge


def test_low_occupancy_is_not_confused_with_nothing_to_merge(win):
    """The two are DIFFERENT facts and must not share a sentence: 'noop' means no run was
    within the window at all, 'low_occupancy' means runs were and every one was declined.
    Telling a user "no sub-events within 600 ms to merge" about events she can see inside
    her 600 ms window is what made the floor look like a bug."""
    noop = _post(win, MergeReport("noop", 40, 40), window_s=0.6)
    refused = _post(win, MergeReport("low_occupancy", 40, 40, n_refused_low_occupancy=3),
                    window_s=0.6)
    assert noop != refused
    assert "No sub-events within" in noop
    assert "No sub-events within" not in refused


# ================================================================================================
# some merged, some refused — the partial case
# ================================================================================================

def test_a_partial_refusal_is_reported_beside_the_merge(win):
    msg = _post(win, MergeReport("ok", 40, 28, n_refused_low_occupancy=3), window_s=0.005)
    assert "Merged 12 fragment(s): 40 → 28 events." in msg     # the old sentence, intact
    assert "3 further run(s)" in msg
    assert "open pore" in msg


def test_an_unrefused_merge_says_nothing_extra(win):
    """The clause appears only when there is something to disclose — a clean merge keeps the
    sentence it always had."""
    msg = _post(win, MergeReport("ok", 40, 28), window_s=0.005)
    assert msg == "Merged 12 fragment(s): 40 → 28 events."


# ================================================================================================
# the other statuses are untouched
# ================================================================================================

@pytest.mark.parametrize("status,fragment", [
    ("started", "Cancel"),
    ("busy", "still running"),
    ("dropped", "discarded"),
    ("failed", "could not finish"),
    ("no_detection", "Run detection first"),
    ("no_selection", "Select a region"),
])
def test_every_other_status_keeps_its_own_sentence(win, status, fragment):
    assert fragment in _post(win, MergeReport(status, 40, 40), window_s=0.005)


def test_a_late_report_still_names_the_window_it_was_asked_for(win):
    """'Merge all' runs off-thread, so its report arrives long after the click and carries
    the window with it — the refusal branch has to read it from the report exactly as the
    no-op branch does."""
    msg = _post(win, MergeReport("low_occupancy", 40, 40, focus_id=-1, window_s=0.012,
                                 n_refused_low_occupancy=2))
    assert "12 ms" in msg
