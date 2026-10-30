"""SUB-1 — the report document wraps instead of running off to the right.

THE DEFECT, measured on the owner's 1512x900 display with the window already fitted
to it by the layout round (client 1496x856, the report pane 644 px wide with a 624 px
viewport)::

    report pane, LineWrapMode.NoWrap     hbar.maximum() = 2906 px, scrollbar VISIBLE

3530 px of document in a viewport five and a half times narrower. What was out there
was not the columns ``NoWrap`` was chosen for — it was PROSE. The four numbered
caveats in a real report are 245, 290, 326 and 488 characters long, i.e. the honesty
text this window exists to put in front of a reader, with its sentences ending up to
four screens to the right of the pane. The same 2906 px got worse, not better, on a
smaller display: 2937 at 1440x900, 3007 at 1280x800, 3166 at the window's own 900x620
floor, because the pane shrinks while the document does not.

THE FIX these tests pin: the pane is ``LineWrapMode.WidgetWidth`` with
``WrapAtWordBoundaryOrAnywhere``, so the document is laid out inside the width it has
at every size the window can be opened or dragged to.

IT IS A SOFT WRAP AND NOT A RE-FLOW, which is the second half of the contract and the
half that is easy to break later: the widget wraps, the *string* is untouched.
``window.text()`` stays byte-identical to ``report.text()``, so ``Copy report`` still
puts the report on the clipboard verbatim (tests/gui/test_subpop_gui.py pins both) and
``Export…`` still writes the same file. Hard-wrapping the text at a column would pass
the scrollbar assertions below and silently break all three.

WHAT IT COSTS, measured rather than asserted away: of the 140 lines in a real report,
9 wrap at the fitted width and only 3 of those are columnar — the population table's
header and its k rows, 160 characters against the 89 the pane holds. The other six are
prose. Six prose blocks readable in place against three table lines continued onto a
second row is the trade, and it is the one the caveat strip in the same window already
made (``_caveats`` has always been ``WidgetWidth``).

WHY THE SCREEN IS INJECTED: the offscreen platform reports an 800x800 screen, so every
size below is pushed through the module's one seam, ``available_screen_rect`` — the
same seam tests/gui/test_lay_subpop_window_fits_the_screen.py patches.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import pathlib

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet
from nanoscope.gui import subpop_window as SW

FS = 50_000.0
N_SAMPLES = 200_000
N_EVENTS = 140

#: The owner's laptop, and the three other displays this round sanity-checks against,
#: plus the window's own floor and its preferred size. A wrap that only holds at one
#: width is the failure this list exists to catch.
SCREENS = {
    "1512x900 (the owner's laptop)": QtCore.QRect(0, 0, 1512, 900),
    "1440x900": QtCore.QRect(0, 0, 1440, 900),
    "1280x800": QtCore.QRect(0, 0, 1280, 800),
    "900x620 (the window's floor)": QtCore.QRect(0, 0, 900, 620),
    "2560x1440 (preferred size)": QtCore.QRect(0, 0, 2560, 1440),
}

#: What the pane's horizontal scrollbar measured BEFORE the fix, per screen. Recorded
#: so a future reader can tell a regression from a re-measurement.
HBAR_BEFORE = {
    "1512x900 (the owner's laptop)": 2906,
    "1440x900": 2937,
    "1280x800": 3007,
    "900x620 (the window's floor)": 3166,
    "2560x1440 (preferred size)": 2860,
}


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """Passes in BOTH trees, so a failure below cannot be an import that didn't happen."""
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    for mod in (SR, SW):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents
        assert p.stat().st_size > 20_000


def _synthetic_events():
    """Two dwell/depth populations at 50 kHz, float32 — no file is read."""
    rng = np.random.default_rng(0)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 2.0, N_SAMPLES)
    spans, pos = [], 500
    for i in range(N_EVENTS):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        if pos + dwell + 400 >= N_SAMPLES:
            break
        x[pos:pos + dwell] -= 60.0 if deep else 30.0
        spans.append((pos, pos + dwell))
        pos += int(rng.integers(500, 1100))
    rows = []
    for i, (s, e) in enumerate(spans):
        depth = 60.0 if i % 2 else 30.0
        rows.append({
            "event_id": i, "recording": "r", "pore_id": "p1",
            "start_idx": s, "end_idx": e, "t_start_s": s / FS, "dwell_s": (e - s) / FS,
            "baseline_pA": 200.0, "depth_pA": depth, "frac_depth": depth / 200.0,
            "ecd_pA_s": depth * (e - s) / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 2.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
    return (np.ascontiguousarray(x, dtype=np.float32),
            EventSet(pd.DataFrame(rows)[REQUIRED],
                     {"analyzed_duration_s": N_SAMPLES / FS, "polarity": -1}))


@pytest.fixture(scope="module")
def report():
    """A REAL report through the real pipeline — never a hand-written stub.

    The long lines this file is about are written by the analysis layer; a stub would
    be a test of a string this test wrote itself.
    """
    arr, events = _synthetic_events()
    cfg = SR.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    rep = SR.run_subpopulations(arr, arr, FS, events, recording="r",
                                duration_s=N_SAMPLES / FS, config=cfg)
    assert rep.quantification is not None
    return rep


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """A profile of this test's own — nothing here may touch the owner's settings."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("SubpopWrapPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


def _pane(win) -> QtWidgets.QPlainTextEdit:
    """The report document itself.

    Reached through the attribute because there is no public accessor for the widget —
    ``window.text()`` returns its string, which is exactly the thing this file must not
    confuse with its layout.
    """
    return win._text


def _open(qtbot, monkeypatch, screen: QtCore.QRect, report=None):
    """A window opened as ``main_window._open_subpop_window`` opens it, on ``screen``,
    then turned to Section ▸ Analysis before any report is installed.

    The window opens on Data, and since b21e8ae the report pane is shown only in the
    Analysis section: on Data it is hidden and never laid out, so the measurements below
    read a 0 px viewport (two failed, and six passed without measuring anything).
    """
    monkeypatch.setattr(SW, "available_screen_rect", lambda _w: screen)
    win = SW.SubpopReportWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win._workflow.set_section("Analysis")
    if report is not None:
        win.set_report(report, subtitle="synthetic")
    qtbot.wait(30)
    return win


# -- the defect ---------------------------------------------------------------------

@pytest.mark.parametrize("label", list(SCREENS))
def test_the_report_never_scrolls_sideways_on_any_display_it_opens_on(
        qtbot, monkeypatch, report, label):
    """The whole defect, at every size the window can open at.

    Pre-fix this failed at EVERY size in ``HBAR_BEFORE`` with the numbers recorded
    there — 2906 px on the owner's own display, and 2860 px even on a 2560x1440
    display where the window opens at its full preferred 1600x980, because the
    document is 3530 px wide and no reachable pane is.
    """
    win = _open(qtbot, monkeypatch, SCREENS[label], report)
    pane = _pane(win)
    hbar = pane.horizontalScrollBar()
    assert hbar.maximum() == 0, (
        f"{label}: the report scrolls {hbar.maximum()} px sideways in a "
        f"{pane.viewport().width()} px viewport (was {HBAR_BEFORE.get(label, 0)} px "
        "before the wrap)")
    assert not hbar.isVisible()


def test_the_long_prose_lines_are_the_ones_that_wrap(qtbot, monkeypatch, report):
    """Not "no scrollbar" — the sentences are actually laid out inside the pane.

    A pane that had simply been emptied, or a document rendered at one line, would
    satisfy the scrollbar assertion above and show nothing.
    """
    win = _open(qtbot, monkeypatch, SCREENS["1512x900 (the owner's laptop)"], report)
    pane = _pane(win)
    doc = pane.document()
    longest = max((doc.findBlockByNumber(i) for i in range(doc.blockCount())),
                  key=lambda b: len(b.text()))
    assert len(longest.text()) > 200, "the report no longer has the prose this is about"
    assert longest.layout().lineCount() > 1, "the longest line is still laid out flat"
    width = max(longest.layout().lineAt(i).naturalTextWidth()
                for i in range(longest.layout().lineCount()))
    assert width <= pane.viewport().width(), (
        f"a wrapped line is still {width} px wide in a {pane.viewport().width()} px "
        "viewport")


def test_no_scroll_area_in_the_window_scrolls_sideways_with_a_report_installed(
        qtbot, monkeypatch, report):
    """The skeptics' sweep, re-run with a report in the window rather than without one.

    The empty window has never had a horizontal scrollbar; the document is only ever
    over-wide once something is in it, which is why the sweep has to be made here.
    """
    win = _open(qtbot, monkeypatch, SCREENS["1512x900 (the owner's laptop)"], report)
    over = [(type(sa).__name__, sa.horizontalScrollBar().maximum())
            for sa in win.findChildren(QtWidgets.QAbstractScrollArea)
            if sa.isVisible() and sa.horizontalScrollBar().maximum() > 0]
    assert over == [], f"scroll areas still running off to the right: {over}"


def test_a_single_token_wider_than_the_pane_does_not_bring_the_scrollbar_back(
        qtbot, monkeypatch, report):
    """Pins ``WrapAtWordBoundaryOrAnywhere`` and not ``WordWrap``.

    A report carries file paths and feature lists. With plain word wrapping one
    unbreakable token — the path of a recording, say — puts the 2906 px scrollbar
    straight back, and the pane would be exactly as unreadable as before for exactly
    the documents most likely to have long names.
    """
    win = _open(qtbot, monkeypatch, SCREENS["1512x900 (the owner's laptop)"], report)
    pane = _pane(win)
    pane.setPlainText("recording: " + "/Users/owner/a_very_long_unbroken_path" * 12)
    qtbot.wait(20)
    assert pane.horizontalScrollBar().maximum() == 0


# -- the contract the wrap must NOT break -------------------------------------------

def test_the_wrap_is_a_layout_and_the_report_text_is_untouched(
        qtbot, monkeypatch, report):
    """The document is the report, character for character — the wrap is soft.

    Hard-wrapping the string at a column would satisfy every assertion above and break
    ``Copy report``, ``Export…`` and the two contracts in tests/gui/test_subpop_gui.py
    that compare this text with ``report.text()``.
    """
    win = _open(qtbot, monkeypatch, SCREENS["1512x900 (the owner's laptop)"], report)
    assert win.text() == report.text()
    assert max(len(line) for line in win.text().split("\n")) > 200


def test_copy_report_still_copies_the_report_verbatim(qtbot, monkeypatch, report):
    """The clipboard is the surface the reader carries away; it is unchanged."""
    win = _open(qtbot, monkeypatch, SCREENS["1512x900 (the owner's laptop)"], report)
    clip = QtWidgets.QApplication.clipboard()
    clip.clear()
    win._copy()
    assert clip.text() == report.text()


def test_nothing_moves_when_the_report_lands(qtbot, monkeypatch, report):
    """A wrap mode may not shift one control.

    The row of buttons under the pane — everything this window can DO — and the
    standing disclosure line above it are at the same pixels empty and full, and the
    pane keeps its width, which is the number the wrap is measured against. Losing the
    horizontal scrollbar gives the document 18 px more height INSIDE the pane and
    moves nothing outside it.

    MEASURED AND NOT MINE: the pane is 30 px SHORTER once a report lands, because the
    honesty header above it goes from one line to two and the splitter absorbs it.
    That is the same 30 px before and after this change — pinning it here would pin
    someone else's behaviour, so this asserts the width and leaves the height to the
    header's own contract.
    """
    screen = SCREENS["1512x900 (the owner's laptop)"]
    win = _open(qtbot, monkeypatch, screen)
    before = {b.text(): b.geometry() for b in win.findChildren(QtWidgets.QPushButton)}
    pane_before = _pane(win).width()
    note_before = win._live_note.geometry()

    win.set_report(report, subtitle="synthetic")
    qtbot.wait(30)

    after = {b.text(): b.geometry() for b in win.findChildren(QtWidgets.QPushButton)}
    assert after == before, "a control moved when the report was installed"
    assert _pane(win).width() == pane_before
    assert win._live_note.geometry() == note_before


def test_the_button_row_is_still_wholly_on_the_owners_screen(
        qtbot, monkeypatch, report):
    """The floor the layout round left here, re-checked with a report installed."""
    screen = SCREENS["1512x900 (the owner's laptop)"]
    win = _open(qtbot, monkeypatch, screen, report)
    off = []
    for b in win.findChildren(QtWidgets.QPushButton):
        rect = QtCore.QRect(win.mapToGlobal(b.pos()), b.size())
        if not screen.contains(rect):
            off.append((b.text(), rect))
    assert off == [], f"off the {screen.width()}x{screen.height()} screen: {off}"


def test_the_pane_is_wrapping_and_says_so(qtbot, monkeypatch):
    """The mode itself, so a revert reads as a revert and not as a re-measurement."""
    win = _open(qtbot, monkeypatch, SCREENS["1512x900 (the owner's laptop)"])
    pane = _pane(win)
    assert pane.lineWrapMode() == QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
    assert pane.wordWrapMode() == QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
