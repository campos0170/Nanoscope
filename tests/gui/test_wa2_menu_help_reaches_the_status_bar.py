"""WAVE-A FIXUP / WA-R3 — the sentences written onto menu actions, made readable.

THE DEFECT, AS MEASURED. Thirty-three actions in this menu bar carried an explanatory
tooltip that no user could ever read. On the wired window at the owner's 1512x900
(``verify-reachable/p10_menu_tooltips.py``): File ``toolTipsVisible=False`` with 5 written
tooltips, View False with 4, Analysis False with 24; ``QMenu().toolTipsVisible()`` is False
by Qt default, and on macOS the menu bar is NATIVE — Qt paints no action tooltip there
whatever that flag says. Not one action carried a ``statusTip`` either, so hovering showed
nothing in the status bar. The specific cost: the app's ONLY browsable sentence naming
batch curation is Analysis > Event gallery...'s "sweep a box over the bad ones and reject
them together", and it had never once been readable.

HOW THE GESTURE IS DRIVEN. A real ``MainWindow`` at 1512x900 on the suite's redirected
QSettings, its real menu bar, and real ARROW-KEY navigation inside a popped-up menu. The
mouse is not an option here and that was measured, not assumed: offscreen Qt delivers no
mouse move into a popup that has grabbed it — ``QTest.mouseMove`` and a hand-built
``QMouseEvent`` over ``actionGeometry(...).center()`` both leave ``activeAction()`` None —
while the arrow key reaches the item and emits the same ``QMenu.hovered``. Keyboard
navigation also skips DISABLED items, which is why the gallery test opens a document and
stores a detection first: that is what a user who can reach the gallery has.

PROVING THE PRE-FIX FAILURE WITHOUT EDITING SOURCE. Run this file with the out-of-tree
plugin that puts the wave-A behaviour back at runtime (it makes
``MainWindow._install_menu_hover_help`` a no-op, which is exactly the state the skeptic
measured)::

    cd /Users/abrahamcampos/Desktop/nanoscope && QT_QPA_PLATFORM=offscreen \\
      PYTHONPATH=<scratch> /opt/anaconda3/bin/python3 -m pytest \\
      tests/gui/test_wa2_menu_help_reaches_the_status_bar.py -q -p wa2_menu_help_prefix_plugin

Every test in this file fails there and passes against the tree.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

GALLERY_LABEL = "Event &gallery…"
OPEN_PROJECT_LABEL = "Open &Project…"

FS = 50_000.0
I0 = -1000.0
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900) for i in range(12)]
N = _SPANS[-1][1] + 4_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(7)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def win(qapp, qtbot):
    """A real MainWindow at the display size every WA measurement was taken on."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 900)
    w.show()
    qapp.processEvents()
    return w


@pytest.fixture
def win_with_events(win, qapp, tmp_path):
    """...plus one recording with a stored detection, because arrow-key navigation only
    lands on ENABLED items and the gallery is gated on having events."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa2_menu_help.dat", format_name="synthetic",
                         n_samples=N, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    qapp.processEvents()
    return win


def _menus(w) -> dict[str, QtWidgets.QMenu]:
    """Menu title -> menu, reached through ``findChildren`` because ``QAction.menu()`` hands
    the returned QMenu Python ownership in this PySide6 and deletes it with the wrapper."""
    return {m.title(): m for m in w.menuBar().findChildren(QtWidgets.QMenu)}


def _hover(qapp, menu: QtWidgets.QMenu, label: str) -> QtGui.QAction:
    """Pop the menu up and walk the arrow key onto ``label`` — a gesture, not a signal emit."""
    menu.popup(QtCore.QPoint(0, 0))
    qapp.processEvents()
    for _ in range(len(menu.actions()) + 2):
        # Qt's offscreen platform can retire a popup after one synthetic key even though
        # it continues dispatching keys to that hidden QMenu.  Reopen it so this remains a
        # faithful menu gesture and cannot pass by relying on post-hide ``hovered`` noise.
        if not menu.isVisible():
            menu.popup(QtCore.QPoint(0, 0))
            qapp.processEvents()
        QTest.keyClick(menu, Qt.Key_Down)
        qapp.processEvents()
        active = menu.activeAction()
        if active is not None and active.text() == label:
            return active
    menu.close()
    raise AssertionError(f"never highlighted {label!r} in {menu.title()!r}")


def test_hovering_event_gallery_puts_its_batch_curation_sentence_in_the_status_bar(
        qapp, win_with_events):
    menu = _menus(win_with_events)["&Analysis"]
    try:
        act = _hover(qapp, menu, GALLERY_LABEL)
        message = win_with_events.statusBar().currentMessage()
    finally:
        menu.close()
        qapp.processEvents()

    assert message == act.toolTip(), "the whole written sentence, not a paraphrase"
    assert "sweep a box over the bad ones and reject them together" in message


def test_the_clause_naming_batch_curation_paints_inside_the_status_bar_at_1512x900(
        qapp, win_with_events):
    """The clause is the point, so the clause is what has to FIT — WA-R2 is this project's
    lesson that a caption which mid-elides over the words naming a gesture has not said them.

    Measured at 1512x900 with a recording loaded: the message area is 1512 px less 511 px of
    permanent widgets and margins = 1001 px, which paints the sentence's first 170 of 197
    characters — through "...reject them together. The inspector is for one eve". So the whole
    batch-curation clause is on screen and only the closing restatement is cut. A status bar
    clips (it does not elide), so this measures the prefix, not the full string."""
    bar = win_with_events.statusBar()
    menu = _menus(win_with_events)["&Analysis"]
    try:
        _hover(qapp, menu, GALLERY_LABEL)
        message = bar.currentMessage()
    finally:
        menu.close()
        qapp.processEvents()

    clause = "sweep a box over the bad ones and reject them together"
    assert clause in message
    painted = message[:message.index(clause) + len(clause)]
    clause_px = QtGui.QFontMetrics(bar.font()).horizontalAdvance(painted)
    permanent = win_with_events._rate_label.width() + win_with_events._calibration_chip.width()
    available = bar.width() - permanent - 24   # 24 px: the bar's own margins + item spacing
    assert clause_px <= available, f"{clause_px} px of clause in {available} px of status bar"


def test_closing_the_menu_withdraws_the_hint_but_never_a_message_from_elsewhere(qapp, win):
    bar = win.statusBar()
    menu = _menus(win)["&File"]

    act = _hover(qapp, menu, OPEN_PROJECT_LABEL)
    assert bar.currentMessage() == act.toolTip(), "the hover must have shown the sentence"
    menu.close()
    qapp.processEvents()
    assert bar.currentMessage() == "", "a menu hint may not outlive its menu"

    # ...and the withdrawal is not a blanket clear: a report from anywhere else stands.
    bar.showMessage("Detection finished: 115 events.")
    menu.popup(QtCore.QPoint(0, 0))
    qapp.processEvents()
    menu.close()
    qapp.processEvents()
    assert bar.currentMessage() == "Detection finished: 115 events."


def test_no_menu_carries_written_sentences_it_cannot_render(qapp, win):
    """The census the skeptic ran: File/View/Analysis each held written tooltips with
    ``toolTipsVisible()`` False, so Qt could not paint one even off macOS."""
    dead = []
    for title, menu in _menus(win).items():
        written = [a for a in menu.actions() if MainWindow._menu_action_help(a)]
        if written and not menu.toolTipsVisible():
            dead.append((title, len(written)))
    assert dead == []
