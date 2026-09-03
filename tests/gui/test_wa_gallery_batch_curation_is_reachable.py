"""WAVE-A/gallery — the Event gallery's BATCH curation, driven the way a user reaches it.

Every gesture here already had an implementation; what it did not have was a way in. The
audit measured, in the WIRED application on the owner's -1000 mV recording (110 events,
gallery at 1200x860):

  * ⌘/Ctrl-click — the gesture the "Accept selected" tooltip advertises — NETTED ZERO tiles:
    click tile 0 -> [0], ⌘-click tile 6 -> [], ⌘-click tile 11 -> []. Cutting the
    ``eventSelected`` relay made the same three clicks read [0] / [0,6] / [0,6,11], which is
    what identified the round trip as the writer.
  * There was no deselect-all and no inverse at all.
  * The status line, this window's only always-visible instruction, was singular throughout
    ("click to select, R to reject … double-click opens IT"), and named neither shift-click,
    nor drag-a-box, nor ⌘A, nor the two "…selected" buttons. Those two words lived only
    inside one button's tooltip, and this window has NO MENU BAR, so its sixteen shortcuts
    were unbrowsable.
  * ⌘Z did nothing here: the undo works, but its QAction is a WindowShortcut on MainWindow,
    which is the window BEHIND this one during every gallery gesture.

So these are reachability tests and they are driven as such: a real MainWindow, the real
relay, real QTest mouse and key events, and the window at the owner's 1512x900. A control
that exists but cannot be reached is what this file is for.

The pre-fix behaviour of each assertion is provable without editing source back: run this
file with the out-of-tree plugin that loads the pre-fix module in place of the tree's, e.g.
``PYTHONPATH=<scratch> pytest tests/gui/test_wa_gallery_batch_curation_is_reachable.py
-p wa_gallery_prefix_plugin``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
N_EVENTS = 30
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(7)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    """A MainWindow with a detection, its gallery open at the owner's display size, and every
    signal MainWindow wires between them live — the relay included, because the relay is what
    the ⌘-click defect was made of."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 900)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa_gallery.dat", format_name="synthetic",
                         n_samples=N, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == N_EVENTS, len(df)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    w._act_event_gallery.trigger()
    qtbot.wait(20)
    gal = w._event_gallery
    gal.page_size_box.setCurrentText("24")
    gal.set_filter("All")
    gal.resize(1200, 860)
    gal.show()
    gal.activateWindow()
    qtbot.wait(40)
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _click(gal, tile: int, mods=Qt.NoModifier) -> None:
    """Click the tile at PAGE row ``tile`` with ``mods`` held — a real QTest mouse click on
    the viewport, not ``setCurrentIndex``: the defect lived in the press handler."""
    idx = gal._model.index(tile, 0)
    assert idx.isValid(), f"tile {tile} is not on this page"
    gal.view.scrollTo(idx)
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, mods,
                     gal.view.visualRect(idx).center())


def _sel(gal) -> list[int]:
    return sorted(gal.selected_event_ids())


def _accepted(df) -> int:
    return int(df["accepted"].to_numpy(dtype=bool).sum())


# ================================================================================================
# (a) ⌘/Ctrl-click, in the wired app
# ================================================================================================
def test_cmd_click_adds_a_discontiguous_tile_instead_of_netting_zero(wired, qtbot):
    """THE MEASURED DEFECT. Qt6 emits ``currentChanged`` BEFORE it applies the press's own
    selection command, the gallery relays that as ``eventSelected``, MainWindow sends it back
    through ``show_event``, and ``_focus_event``'s ClearAndSelect pre-selected the very tile
    Qt was about to Toggle — so the toggle took it straight off again and the click selected
    NOTHING. Pre-fix this reads ``[] / [] / []``."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    ids = [int(df["event_id"].iloc[r]) for r in gal._model.page_df_rows()]

    _click(gal, 0)
    qtbot.wait(30)
    assert _sel(gal) == [ids[0]], "a plain click no longer selects one tile"

    _click(gal, 3, Qt.ControlModifier)
    qtbot.wait(30)
    assert _sel(gal) == sorted([ids[0], ids[3]]), (
        f"⌘-click did not ADD a tile: {_sel(gal)} (wanted {sorted([ids[0], ids[3]])})")

    _click(gal, 7, Qt.ControlModifier)
    qtbot.wait(30)
    assert _sel(gal) == sorted([ids[0], ids[3], ids[7]]), _sel(gal)

    # ...and the same gesture takes one back off, the way every list on macOS does.
    _click(gal, 3, Qt.ControlModifier)
    qtbot.wait(30)
    assert _sel(gal) == sorted([ids[0], ids[7]]), (
        f"⌘-click did not REMOVE the tile it landed on: {_sel(gal)}")


def test_a_discontiguous_selection_is_what_reject_selected_then_curates(wired, qtbot):
    """The gesture is only worth fixing because of what it feeds: three ⌘-clicks and one R
    must reject exactly those three events and nothing else."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    ids = [int(df["event_id"].iloc[r]) for r in gal._model.page_df_rows()]
    before = _accepted(df)

    _click(gal, 1)
    _click(gal, 5, Qt.ControlModifier)
    _click(gal, 9, Qt.ControlModifier)
    qtbot.wait(30)
    picked = _sel(gal)
    assert picked == sorted([ids[1], ids[5], ids[9]]), picked

    QTest.keyClick(gal.view, Qt.Key_R)
    qtbot.wait(60)
    assert _accepted(df) == before - 3, (
        f"R over a ⌘-click selection curated {before - _accepted(df)} events, not 3")
    still = [int(df["event_id"].iloc[i]) for i in range(len(df))
             if not bool(df["accepted"].iloc[i])]
    assert sorted(still) == picked, still


def test_the_inbound_mirror_still_moves_the_gallery_when_it_is_somewhere_else(wired, qtbot):
    """THE GUARD IS "already there", NOT "a click is in flight" — so the mirror that the
    events table and the inspector rely on is untouched: asked for an event the gallery is
    NOT on, it still turns to its page and selects it."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    far_row = gal._model.df_rows()[26]          # page size 24: this is page 2
    far_id = int(df["event_id"].iloc[far_row])
    _click(gal, 0)
    qtbot.wait(20)
    assert gal.page() == 0

    w._analysis_panel.select_event(far_id)
    qtbot.wait(60)
    assert gal.page() == 1, "the inbound mirror no longer turns to the event's page"
    assert gal.current_event_id() == far_id
    assert _sel(gal) == [far_id], f"the mirror must select the tile it moved to: {_sel(gal)}"


# ================================================================================================
# (b) deselect all
# ================================================================================================
def test_escape_clears_the_selection_and_leaves_the_cursor_and_the_frame_alone(wired, qtbot):
    """Esc is a SELECTION gesture: nothing is curated, and the keyboard cursor stays where it
    was so the arrow keys carry on from the same tile."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    before = _accepted(df)
    _click(gal, 2)
    QTest.keyClick(gal, Qt.Key_A, Qt.ControlModifier)
    qtbot.wait(30)
    assert len(_sel(gal)) == gal.tile_count() > 1
    cursor = gal.current_event_id()

    QTest.keyClick(gal, Qt.Key_Escape)
    qtbot.wait(30)
    assert _sel(gal) == [], f"Escape left {len(_sel(gal))} tiles selected"
    assert gal.current_event_id() == cursor, "Escape moved the keyboard cursor"
    assert _accepted(df) == before, "Escape curated something"

    # ...and with nothing selected the batch keys write nothing.
    QTest.keyClick(gal.view, Qt.Key_R)
    qtbot.wait(40)
    assert _accepted(df) == before


def test_the_selection_count_on_the_status_line_follows_the_selection(wired, qtbot):
    """The feedback path for every batch gesture, Escape included. The audit assumed this
    count was already on the line; it was not — the only clause that said "selected" was the
    off-page note, which appears when a selection is LOST."""
    w, doc_id, gal = wired
    _click(gal, 0)
    _click(gal, 4, Qt.ControlModifier)
    qtbot.wait(40)
    assert "2 selected" in gal.status.text(), gal.status.text()

    QTest.keyClick(gal, Qt.Key_A, Qt.ControlModifier)
    qtbot.wait(40)
    assert f"{gal.tile_count()} selected" in gal.status.text(), gal.status.text()

    QTest.keyClick(gal, Qt.Key_Escape)
    qtbot.wait(40)
    assert "selected" not in gal.status.text().replace("selected tile", ""), gal.status.text()


# ================================================================================================
# (c) select inverse — and the word "invert" stays with the Space key
# ================================================================================================
def test_select_inverse_swaps_the_selected_and_unselected_tiles_of_the_page(wired, qtbot):
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    ids = [int(df["event_id"].iloc[r]) for r in gal._model.page_df_rows()]
    before = _accepted(df)
    _click(gal, 0)
    _click(gal, 3, Qt.ControlModifier)
    qtbot.wait(30)
    assert _sel(gal) == sorted([ids[0], ids[3]])

    QTest.keyClick(gal, Qt.Key_I, Qt.ControlModifier)
    qtbot.wait(40)
    got = _sel(gal)
    assert got == sorted(set(ids) - {ids[0], ids[3]}), got
    assert len(got) == gal.tile_count() - 2
    assert _accepted(df) == before, "the selection gesture curated something"

    # An empty page-selection inverts to the whole page, and back again.
    QTest.keyClick(gal, Qt.Key_Escape)
    qtbot.wait(20)
    QTest.keyClick(gal, Qt.Key_I, Qt.ControlModifier)
    qtbot.wait(40)
    assert _sel(gal) == sorted(ids)
    QTest.keyClick(gal, Qt.Key_I, Qt.ControlModifier)
    qtbot.wait(40)
    assert _sel(gal) == []


def test_the_word_invert_still_belongs_to_the_space_key_alone(wired, qtbot):
    """"Space (invert)" is a DESTRUCTIVE per-tile gesture in this window (it writes
    ``accepted``). The selection gesture is therefore called "inverse", never "invert" — two
    gestures one keystroke apart must not share a verb when one of them curates."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    words = [gal.select_menu.actions()[i].text().lower() for i in range(3)]
    assert not any("invert" in t for t in words), words
    assert any("inverse" in t for t in words), words
    assert "invert" not in gal.status.text().lower(), gal.status.text()
    assert "invert" not in gal.select_btn.toolTip().lower()

    # ...and Space still means what it meant: invert the ACCEPTANCE of the selection.
    before = _accepted(df)
    _click(gal, 0)
    _click(gal, 2, Qt.ControlModifier)
    qtbot.wait(30)
    QTest.keyClick(gal.view, Qt.Key_Space)
    qtbot.wait(60)
    assert _accepted(df) == before - 2, (
        "Space no longer inverts the acceptance of the selected tiles")


# ================================================================================================
# (b/c/d) the affordances — reachable at the owner's display size
# ================================================================================================
def test_the_select_menu_names_all_three_gestures_with_their_keys_and_fits_the_window(
        wired, qtbot):
    """THE WINDOW HAS NO MENU BAR, so a shortcut nothing names cannot be found. One menu
    button carries all three, each with the key beside it — and the row it sits on still fits
    the owner's 1512 px display with room to spare."""
    w, doc_id, gal = wired
    assert gal.menuBar().actions() == [], "this test is about a window with no menu bar"
    gal.resize(1512, 900)
    qtbot.wait(40)

    assert gal.select_btn.isVisible()
    top_left = gal.select_btn.mapTo(gal, QtCore.QPoint(0, 0))
    assert 0 <= top_left.x() and top_left.x() + gal.select_btn.width() <= gal.width(), (
        f"the Select button is outside the window: {top_left} {gal.select_btn.size()}")
    assert 0 <= top_left.y() and top_left.y() + gal.select_btn.height() <= gal.height()

    named = {a.text(): a.shortcut().toString() for a in gal.select_menu.actions()}
    assert len(named) == 3, named
    assert all(seq for seq in named.values()), f"a gesture with no key printed: {named}"

    row = gal.centralWidget().layout().itemAt(1).layout()
    assert row.minimumSize().width() <= 1100, (
        f"the selection row demands {row.minimumSize().width()} px — over the 1100 px budget")
    assert gal.minimumSizeHint().width() <= 1100, gal.minimumSizeHint().width()


def test_the_menu_items_do_the_same_thing_their_shortcuts_do(wired, qtbot):
    """ONE QAction per gesture, shown in the menu and owned by the window — so the item and
    the key cannot drift apart."""
    w, doc_id, gal = wired
    gal.act_select_all.trigger()
    qtbot.wait(30)
    assert len(_sel(gal)) == gal.tile_count() > 1
    gal.act_select_inverse.trigger()
    qtbot.wait(30)
    assert _sel(gal) == []
    gal.act_select_all.trigger()
    gal.act_select_none.trigger()
    qtbot.wait(30)
    assert _sel(gal) == []
    assert set(gal.select_menu.actions()) <= set(gal.actions()), (
        "the menu's actions must be the window's, or the shortcut and the item are two things")


# ================================================================================================
# (d) the one sentence that says batch curation exists
# ================================================================================================
def test_the_status_line_names_the_batch_gestures_and_not_only_the_singular_ones(wired, qtbot):
    """THE HIGHEST-VALUE SENTENCE IN THE WINDOW. Pre-fix the line read "click to select, R to
    reject · … · double-click opens it …" — singular throughout, naming no batch gesture at
    all, which is why the owner believed the gallery rejects one event at a time."""
    w, doc_id, gal = wired
    line = gal.status.text()
    low = line.lower()
    for word in ("shift-click", "drag a box", "all on the page", "esc", "inverse"):
        assert word in low, f"the status line never names {word!r}: {line}"
    assert "every selected tile" in low, line
    # The modifier is named as the key Qt::ControlModifier ACTUALLY IS on this machine —
    # ⌘ on macOS, where a literal Ctrl-click is the context menu.
    from nanoscope.gui.event_gallery import _cmd_key
    assert f"{_cmd_key()}-click" in line, line
    assert f"{_cmd_key()}A" in line and f"{_cmd_key()}I" in line, line


def test_the_batch_sentence_survives_the_status_lines_own_clipping(wired, qtbot):
    """The line is cut from the TAIL when it will not fit its three lines, so a sentence that
    matters has to be ahead of the ones that matter less. Measured at the gallery's own
    1200x860: the batch clause is on the glass, not only in the tooltip."""
    w, doc_id, gal = wired
    gal.resize(1200, 860)
    # The ORDINARY state: no stale-estimator badge standing. With one up (a 380-character
    # disclosure) the line is clipped by design and the tooltip carries the rest — that is
    # _StatusLine's documented contract, and a disclosure outranks a hint. What must not
    # happen is the batch sentence being invisible on an ordinary screen.
    gal.set_stale_columns()
    qtbot.wait(60)
    shown = QtWidgets.QLabel.text(gal.status)
    assert "drag a box" in shown, (
        f"the batch gestures were clipped off the visible line: {shown!r}")
    assert "each acts on every selected tile" in shown, shown


# ================================================================================================
# (e) ⌘Z, from the window the batch gesture happens in
# ================================================================================================
def test_cmd_z_in_the_gallery_asks_the_one_undo_stack(wired, qtbot):
    """The gallery is a top-level window with no menu bar, so MainWindow's Undo QAction — a
    WindowShortcut — never saw ⌘Z here. The window now raises INTENT, and MainWindow's one
    line of wiring hands it to the one stack."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    asked: list[int] = []
    gal.undoRequested.connect(lambda: asked.append(1))

    QTest.keyClick(gal, Qt.Key_A, Qt.ControlModifier)
    qtbot.wait(30)
    n = len(_sel(gal))
    before = _accepted(df)
    QTest.keyClick(gal.view, Qt.Key_R)
    qtbot.wait(80)
    assert _accepted(df) == before - n, (before, _accepted(df), n)
    assert w._act_undo.isEnabled() and "Undo" in w._act_undo.text(), w._act_undo.text()

    QTest.keyClick(gal, Qt.Key_Z, Qt.ControlModifier)
    qtbot.wait(40)
    assert asked, "⌘Z inside the gallery raised nothing"

    # ...and through the documented one-line wiring it reaches the real stack.
    gal.undoRequested.connect(w._act_undo.trigger)
    QTest.keyClick(gal, Qt.Key_Z, Qt.ControlModifier)
    qtbot.wait(120)
    assert _accepted(_frame(w, doc_id)) == before, (
        "the wired ⌘Z did not walk the gallery's own batch back")


def test_the_undo_affordance_is_disabled_until_this_window_has_written_something(wired, qtbot):
    """A button that claims a step that does not exist is the same lie as a shortcut that
    does nothing. It arms on this window's own write — every curation here is checkpointed on
    MainWindow's stack — and :meth:`set_undo_state` replaces the guess with the real state
    and the real label."""
    w, doc_id, gal = wired
    assert not gal.undo_btn.isEnabled(), "the Undo button armed itself with nothing to undo"
    assert not gal.act_undo.isEnabled(), "the ⌘Z action armed itself with nothing to undo"
    assert "Nothing to undo" in gal.undo_btn.toolTip(), gal.undo_btn.toolTip()

    _click(gal, 0)
    qtbot.wait(20)
    QTest.keyClick(gal.view, Qt.Key_R)
    qtbot.wait(60)
    assert gal.undo_btn.isEnabled(), "a curation in this window left its Undo disabled"
    assert gal.act_undo.isEnabled()

    gal.set_undo_state(True, "Undo Curate 45 events")
    assert gal.undo_btn.text() == "Undo", (
        "the footer button's text must not grow — it would slide the two 'all shown' "
        "buttons along the footer under the pointer")
    assert "Undo Curate 45 events" in gal.undo_btn.toolTip(), gal.undo_btn.toolTip()
    gal.set_undo_state(False)
    assert not gal.undo_btn.isEnabled() and not gal.act_undo.isEnabled()


def test_the_undo_button_is_inside_the_window_and_raises_the_same_intent(wired, qtbot):
    w, doc_id, gal = wired
    gal.resize(1512, 900)
    qtbot.wait(40)
    gal.set_undo_state(True, "Undo Curate 3 events")
    asked: list[int] = []
    gal.undoRequested.connect(lambda: asked.append(1))
    top_left = gal.undo_btn.mapTo(gal, QtCore.QPoint(0, 0))
    assert gal.undo_btn.isVisible()
    assert 0 <= top_left.x() and top_left.x() + gal.undo_btn.width() <= gal.width(), (
        f"the Undo button is outside the window: {top_left} {gal.undo_btn.size()}")
    QTest.mouseClick(gal.undo_btn, Qt.LeftButton)
    qtbot.wait(30)
    assert asked == [1]
