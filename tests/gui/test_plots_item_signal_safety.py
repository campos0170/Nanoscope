"""The Plots window's checkable lists, and the rule a shipped build died on.

WHAT HAPPENED. The 10:12 build segfaulted in the owner's hands after fifteen minutes:
EXC_BAD_ACCESS inside ``QListWidgetItem_PTR_CppToPython_QListWidgetItem``, i.e. inside
PySide building a Python wrapper for a ``QListWidgetItem*`` that no longer existed. The
sequence was one emission of ``QListWidget.itemChanged``:

    a tick on "Legend entries"
      -> QStyledItemDelegate::editorEvent -> QListWidgetItem::setData -> dataChanged
      -> itemChanged(QListWidgetItem*)
           receiver 1: `_on_legend_entry_toggled` -> `_changed` -> `_redraw`
                       -> finally: `_sync_legend_entry_list` -> `clear()`
                          *** the ticked row is DESTROYED, mid-emission ***
           receiver 2: `_history_touch`, wired generically by `_wire_history`
                       -> PySide marshals the SAME (now freed) pointer -> crash

TWO INDEPENDENT MISTAKES, so two independent fixes, and one test class each below:

* **A widget must not destroy its own rows from inside its own item signal.**
  ``_sync_legend_entry_list`` now reconciles the checklist IN PLACE, and the two places
  that really do empty a list hand the rows to ``_retire_row``, which keeps them alive
  until the event has unwound.
* **A signal carrying a raw item pointer must not be handed to a generic hook.**
  ``_wire_history``'s walk no longer connects ``itemChanged``; the two handlers that
  own it call ``_history_touch`` themselves.

WHAT THESE TESTS PIN, AND WHAT THEY DELIBERATELY DO NOT. They do not assert "this does
not segfault" — a use-after-free is a coin flip on heap layout (the two shipped tests
that already walked this path crashed roughly half the time, and passed the other half),
and this suite runs with the cyclic collector DISABLED, so it is exactly the wrong place
to ask a crash to show up. They assert the STRUCTURE that makes the crash impossible:
that the rows survive a tick, that a removed row outlives the event that removed it, and
that the item-pointer signals have exactly one receiver. Every one of them fails against
the ordering the shipped build had.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import PlotsWindow


#: The role the tests stamp their own mark in. Above every role the window itself uses,
#: so a marked row is a row the WINDOW cannot have written — which is what makes "this is
#: the same object" provable without touching C++ pointers.
_MARK = Qt.UserRole + 501

#: The signal whose argument is a raw ``QListWidgetItem*``. Counting its receivers is the
#: one-line canary for this whole family of bug.
_ITEM_CHANGED = QtCore.SIGNAL("itemChanged(QListWidgetItem*)")


def _events(n: int = 200, *, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(3e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def win(qapp, qtbot):
    """A drawn histogram with a fit, so the legend checklist has two rows to tick."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"Detected events": _events()})
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    return w


@pytest.fixture
def two_sources(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([
        PlotSource(key="a", label="0007", df=_events(seed=1), colour="#1f77b4"),
        PlotSource(key="b", label="0009", df=_events(seed=2), colour="#d62728"),
    ])
    w.set_sources(["a", "b"])      # set_data checks only the first; overlay them both
    return w


def _mark_rows(widget) -> None:
    """Stamp every row with a mark only this test could have written, and CHECK IT TOOK.

    The check is half the test. ``setData`` is itself an ``itemChanged``, so against the
    shipped ordering the stamp is wiped by the rebuild that the emission triggers -- the
    marks come back as ``None`` and any later "the marks survived" comparison would be
    comparing nothing with nothing.
    """
    for i in range(widget.count()):
        widget.item(i).setData(_MARK, f"mark-{i}")
    assert _marks(widget) == [f"mark-{i}" for i in range(widget.count())], (
        "a write to one row rebuilt the whole list from under it")


def _marks(widget) -> list:
    return [widget.item(i).data(_MARK) for i in range(widget.count())]


def _legend_texts(w) -> list[str]:
    legend = getattr(w.plot_item(), "legend", None)
    return [] if legend is None else [label.text for _sample, label in legend.items]


# ================================================================================================
# 1 — nothing is destroyed under its own signal
# ================================================================================================

def test_ticking_a_legend_entry_reuses_the_rows_instead_of_rebuilding_them(win):
    """THE CRASH, pinned as a property of the ROWS rather than as an absence of a crash.

    A tick reaches ``_sync_legend_entry_list`` from inside Qt's delivery of
    ``itemChanged``, with the ticked row still live in the emission's argument array.
    The old code called ``clear()`` there, so every row — the ticked one included — was
    a freed C++ object before the emission finished walking its receivers. Marking the
    rows and finding the marks afterwards proves the objects are the SAME ones: a
    rebuild cannot carry a mark it never wrote.
    """
    lst = win._legend_entries
    assert lst.count() >= 2
    _mark_rows(lst)
    marked_before = _marks(lst)

    row = next(i for i in range(lst.count())
               if str(lst.item(i).data(Qt.UserRole)).startswith("fit:"))
    lst.item(row).setCheckState(Qt.Unchecked)

    assert _marks(lst) == marked_before, (
        "the checklist was rebuilt inside its own itemChanged — the very thing that "
        "left PySide converting a freed QListWidgetItem*")
    assert lst.item(row).checkState() == Qt.Unchecked, "and the tick still took effect"

    lst.item(row).setCheckState(Qt.Checked)
    assert _marks(lst) == marked_before, "...both ways"


def test_twenty_real_clicks_on_the_checkbox_leave_the_same_rows_in_place(win, qapp):
    """The owner's actual gesture: the delegate path, over and over.

    ``QStyledItemDelegate::editorEvent`` is the frame the crash report showed, and it is
    reached only by a real press/release on the check indicator. Twenty of them, because
    the shipped bug bit within a handful — and because a single tick is what the two
    pre-existing tests did, which is why they passed half the time.
    """
    win.show()
    qapp.processEvents()
    lst = win._legend_entries
    _mark_rows(lst)
    marked_before = _marks(lst)

    for i in range(20):
        rect = lst.visualItemRect(lst.item(0))
        QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier,
                         QtCore.QPoint(rect.left() + 10, rect.center().y()))
        qapp.processEvents()
        assert _marks(lst) == marked_before, f"row identity lost on click {i + 1}"

    assert lst.item(0).checkState() == Qt.Checked, "twenty ticks is an even number"


def test_every_offered_row_is_drawn_with_a_checkbox_whichever_way_it_arrived(win):
    """Reconciling means rows are now REUSED, so the one thing a fresh row must not
    inherit from the reconciliation is "no check state at all": a row whose
    ``CheckStateRole`` was never written is drawn without a checkbox, and an unticked
    entry could then never be ticked back on. Rebuilt from empty, with one entry already
    excluded, so a fresh row whose wanted state is Unchecked is what is inspected."""
    win._style = win._style.with_legend_include(
        str(win._legend_entries.item(0).data(Qt.UserRole)), False)
    win._legend_offered, offered = {}, dict(win._legend_offered)
    win._sync_legend_entry_list()          # empty it: every row is now a fresh one
    win._legend_offered = offered
    win._sync_legend_entry_list()

    lst = win._legend_entries
    assert lst.count() == len(offered)
    for i in range(lst.count()):
        assert lst.item(i).data(Qt.CheckStateRole) is not None, (
            "row without a check state: it would render with no checkbox")
        assert bool(lst.item(i).flags() & Qt.ItemIsUserCheckable)
    assert lst.item(0).checkState() == Qt.Unchecked, "and the excluded row is unticked"


def test_a_legend_row_that_really_goes_away_outlives_the_event_that_removed_it(win, qapp):
    """The other half: when the offered SET changes, a row does have to go.

    It is ``takeItem``-ed and parked rather than destroyed, because the removal can
    happen inside the delivery of a signal Qt is still marshalling for other receivers.
    "Parked" is observable: the Python wrapper still answers ``text()``. Against the old
    ``clear()`` this raises ``RuntimeError: Internal C++ object already deleted``.
    """
    lst = win._legend_entries
    doomed = lst.item(lst.count() - 1)
    doomed.setData(_MARK, "doomed")

    win._legend_offered = {}
    win._sync_legend_entry_list()

    assert lst.count() == 0, "the row is out of the list"
    assert doomed.data(_MARK) == "doomed", (
        "...but its C++ half is still alive, because whatever removed it may have been "
        "inside an emission still holding it")
    assert win._retired_rows, "and it is parked, not leaked into the list"

    qapp.processEvents()          # the zero-timer: the event is over
    assert not win._retired_rows, "the parking lot is drained on the next turn"


def test_pushing_new_data_does_not_destroy_the_source_rows_under_the_event(two_sources, qapp):
    """``set_data`` is reachable from inside a source row's own ``itemChanged``.

    Not today's crash — today's ``_on_source_item_changed`` rebuilds the LEGEND list,
    not its own — but one line away from being the identical one, and it is also what an
    asynchronous detection result does while a click is in flight. So the source list is
    emptied by the same parking route.
    """
    w = two_sources
    lst = w._source_list
    old = lst.item(0)
    old.setData(_MARK, "old-row-0")

    w.set_data([PlotSource(key="c", label="0011", df=_events(seed=5), colour="#2ca02c")])

    assert [lst.item(i).text() for i in range(lst.count())] == ["0011"]
    assert old.data(_MARK) == "old-row-0", (
        "set_data destroyed the rows in place — unsafe from inside itemChanged")

    qapp.processEvents()
    assert not w._retired_rows


# ================================================================================================
# 2 — the item-pointer signals have exactly one receiver
# ================================================================================================

@pytest.mark.parametrize("attr", ["_legend_entries", "_source_list"])
def test_only_one_python_receiver_marshals_the_raw_item_pointer(win, attr):
    """THE CHEAP CANARY, and the assertion that would have caught this at wiring time.

    PySide converts a signal's arguments once PER RECEIVER. With one receiver, a handler
    that destroys the item it was given can only hurt itself; with two, the second
    receiver is handed a pointer the first one freed and PySide dereferences it to build
    a wrapper. ``_wire_history``'s ``findChildren`` walk used to add exactly that second
    receiver to both of these lists.
    """
    widget = getattr(win, attr)
    assert widget.receivers(_ITEM_CHANGED) == 1, (
        f"{attr}.itemChanged has more than one Python receiver: a signal carrying a raw "
        "QListWidgetItem* must not be fanned out (see _wire_history)")


def test_the_history_walk_does_not_connect_itemChanged_anywhere(win):
    """The rule as a rule, not as two instances: a list added next month is covered."""
    from PySide6 import QtWidgets
    for w in win._controls_host.findChildren(QtWidgets.QListWidget):
        assert w.receivers(_ITEM_CHANGED) <= 1, w.objectName() or w


# ================================================================================================
# 3 — and none of the behaviour those lists provide moved
# ================================================================================================

def test_unticking_a_legend_entry_still_redraws_live(win):
    """Cosmetic, so it must NOT go behind the run gate (it never did)."""
    lst = win._legend_entries
    before = len(_legend_texts(win))
    assert before >= 2

    row = next(i for i in range(lst.count())
               if str(lst.item(i).data(Qt.UserRole)).startswith("fit:"))
    lst.item(row).setCheckState(Qt.Unchecked)

    assert len(_legend_texts(win)) == before - 1, "the legend redrew on the spot"
    assert not win._stale, "a legend entry is style, not data"
    assert lst.count() == before, "and the row stays offerable"

    lst.item(row).setCheckState(Qt.Checked)
    assert len(_legend_texts(win)) == before


def test_ticking_a_source_still_marks_the_plot_stale(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert not w._stale

    w._source_list.item(1).setCheckState(Qt.Unchecked)

    assert w._stale, "a change of WHICH ROWS are drawn waits for Plot"
    assert "data selection" in w._stale_reason
    assert tuple(w.selected_source_keys()) == ("a",)


def test_renaming_a_source_row_still_edits_the_legend(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")

    w._source_list.item(1).setText("−700 mV")

    assert w._style.label_overrides.get("b") == "−700 mV"
    assert any("−700 mV" in t for t in _legend_texts(w)), _legend_texts(w)
    assert not w._stale, "a rename is legend text, so it redraws live"


def test_a_legend_tick_is_still_one_undo_step(win):
    """``_wire_history`` no longer connects ``itemChanged``, so the handlers commit the
    step themselves. The grain must be what it was: one tick, one Ctrl+Z."""
    lst = win._legend_entries
    before = win.config()
    back, _forward = win._history.depth()

    row = next(i for i in range(lst.count())
               if str(lst.item(i).data(Qt.UserRole)).startswith("fit:"))
    lst.item(row).setCheckState(Qt.Unchecked)

    assert win._history.depth()[0] == back + 1, "the tick closed exactly one step"
    assert win.undo_config()
    assert win.config() == before, "and Ctrl+Z walks back out of it"
    assert lst.item(row).checkState() == Qt.Checked, "widgets came back with the value"


def test_a_source_tick_and_a_rename_are_still_undoable(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    before = w.config()

    w._source_list.item(1).setCheckState(Qt.Unchecked)
    assert tuple(w.selected_source_keys()) == ("a",)
    assert w.undo_config()
    assert w.config() == before
    assert w._source_list.item(1).checkState() == Qt.Checked

    w._source_list.item(1).setText("pore B")
    assert w._style.label_overrides.get("b") == "pore B"
    assert w.undo_config()
    assert not w._style.label_overrides, "the rename stepped back too"
    # ...AND THE ROW CAME BACK WITH IT. This list is the rename affordance, so its text
    # is a projection of `label_overrides` the way the Title box is of `style.title`; a
    # restore that reverted the legend and left the typed name on the row made the window
    # say two different things about one source, and it survived a re-Plot.
    assert w._source_list.item(1).text() == "0009", (
        "undo reverted the legend but not the row that names it")
    assert w._source_list.item(1).checkState() == Qt.Checked, "and did not untick it"
    assert w.redo_config()
    assert w._source_list.item(1).text() == "pore B", "redo puts the typed name back"


def test_a_rename_undone_after_the_source_set_moved_reverts_only_that_row(two_sources):
    """The rows the undo walks are TODAY's rows, not the ones the config was taken from.

    ``set_data`` rebuilds the list from ``_legend_label``, so a source that survives a
    Refresh arrives already carrying its override — and it then rebases the history base
    while KEEPING the stack, because arriving data is not a configuration change. So an
    undo taken afterwards installs an older config against a list whose membership has
    moved underneath it. The sync therefore has to walk the rows and look each row's OWN
    key up in the restored overrides: a source that survived reverts, one that arrived
    since is untouched, and an override for a source that has GONE — already dropped by
    ``with_pruned_legend`` — must simply match no row rather than blank one.
    """
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    lst = w._source_list

    lst.item(0).setText("Apo A")
    assert w._style.label_overrides.get("a") == "Apo A"

    w.set_data([
        PlotSource(key="a", label="0007", df=_events(seed=1), colour="#1f77b4"),
        PlotSource(key="b", label="0009", df=_events(seed=2), colour="#d62728"),
        PlotSource(key="c", label="0011", df=_events(seed=5), colour="#2ca02c"),
    ])
    assert [lst.item(i).text() for i in range(lst.count())] == ["Apo A", "0009", "0011"], (
        "the override is keyed by SOURCE key, so it survives the rebuild")

    assert w.undo_config()
    assert [lst.item(i).text() for i in range(lst.count())] == ["0007", "0009", "0011"], (
        "the renamed source reverted; the one that arrived since was not touched")

    # ...and now the half where the renamed source is gone by the time the undo lands.
    lst.item(0).setText("Apo A again")
    assert w._style.label_overrides.get("a") == "Apo A again"
    w.set_data([PlotSource(key="c", label="0011", df=_events(seed=5), colour="#2ca02c")])
    assert [lst.item(i).text() for i in range(lst.count())] == ["0011"]

    while w.undo_config():          # all the way to the bottom of the stack
        pass
    assert [lst.item(i).text() for i in range(lst.count())] == ["0011"], (
        "an override for a key no live row carries must match nothing, not blank a row")


def test_a_config_restore_never_frees_a_source_row(two_sources):
    """THE CRASH RULE, applied to the new writer.

    ``_set_config`` is reachable from inside this list's own ``itemChanged`` (a tick ->
    ``_on_sources_changed`` -> ``_repopulate_columns`` -> ``_rebase``), with the ticked
    row still live in the emission Qt is marshalling for its remaining receivers. Writing
    the row text is only safe because it is written IN PLACE: the marks prove the same
    C++ objects are still there, which a ``clear()``, an unparked ``takeItem`` or a fresh
    ``QListWidgetItem`` could not.
    """
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    lst = w._source_list
    _mark_rows(lst)
    marked_before = _marks(lst)

    lst.item(1).setText("pore B")
    assert _marks(lst) == marked_before, "the rename itself rebuilt the list"

    assert w.undo_config()
    assert lst.item(1).text() == "0009"
    assert _marks(lst) == marked_before, (
        "the restore rebuilt the source rows — the shape that left PySide converting a "
        "freed QListWidgetItem*")

    assert w.redo_config()
    assert lst.item(1).text() == "pore B"
    assert _marks(lst) == marked_before, "...both ways"
    assert not w._retired_rows, "nothing was taken out of the list at all"


def test_a_typed_name_that_needed_stripping_is_shown_as_it_was_stored(two_sources):
    """THE SAME RULE ON THE RENAME PATH: the row is a projection of the style.

    ``_on_source_item_changed`` stores ``text.strip()`` as the override. Until this was
    pinned it left the UNSTRIPPED text standing on the row, so the Sources list said
    ``'  pore B  '`` while the legend checklist and the drawn legend said ``'pore B'`` --
    one source under two names -- and no re-Plot reconciled them, because nothing on the
    draw path rewrites a row. Only an undo/redo did, because that goes through
    ``_set_config``, which is the writer the crash-class fix hooked.

    The write-back is IN PLACE for the reason the test above states, so the marks are
    checked here too: this write happens from inside the very emission it is answering.
    """
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    lst = w._source_list
    _mark_rows(lst)
    marked_before = _marks(lst)
    back, _forward = w._history.depth()

    lst.item(1).setText("  pore B  ")

    assert w._style.label_overrides.get("b") == "pore B", "the override is stripped"
    assert lst.item(1).text() == "pore B", "and the row shows what was stored"
    assert any("pore B" in t for t in _legend_texts(w)), _legend_texts(w)
    assert _marks(lst) == marked_before, (
        "the write-back rebuilt the row instead of writing it in place")
    assert not w._retired_rows, "nothing was taken out of the list at all"
    assert w._history.depth()[0] == back + 1, (
        "the window's own write-back re-emits itemChanged; under `_updating` it must not "
        "be read back as a second user gesture")

    # PADDING ALONE stores no override at all -- and must not leave the padding behind.
    lst.item(1).setText("   pore B   ")
    assert w._style.label_overrides.get("b") == "pore B"
    assert lst.item(1).text() == "pore B"
    assert w._history.depth()[0] == back + 1, "nothing changed, so nothing was recorded"

    # ...including padding that only differs from the source's OWN label by whitespace,
    # which removes the override and used to leave '  0009  ' standing over "0009".
    lst.item(1).setText("  0009  ")
    assert "b" not in dict(w._style.label_overrides)
    assert lst.item(1).text() == "0009"
    assert _marks(lst) == marked_before
