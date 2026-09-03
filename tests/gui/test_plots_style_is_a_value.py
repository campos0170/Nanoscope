"""``StyleSettings`` is a VALUE: frozen, immutable all the way down, safe to keep.

WHY THIS FILE EXISTS. The plot style is about to become the thing an undo stack and a
saved plot template are built out of, and both of those are nothing but a LIST OF
SNAPSHOTS. A snapshot is only worth taking if it stops describing the window the moment
it is taken -- otherwise "undo" restores whatever the window happens to look like now,
which is the current state, which is not an undo.

The pre-freeze class failed that twice over, and the second failure is the one a frozen
wrapper does NOT fix on its own:

* ``self._style.title = "x"`` wrote straight into the instance every caller was holding.
* ``dataclasses.replace()`` LOOKED like it made a copy, but its ``label_overrides`` and
  ``legend_include`` were the SAME dict objects as the original's -- measured on the
  pre-freeze class, ``replace(s, title="x").label_overrides is s.label_overrides`` was
  True. So a "copy" and its original shared their two mappings, and writing one
  override wrote it into every snapshot ever taken.

Every test below is that aliasing trap, closed: at the type level (the dataclass and
its collection type), at the panel's own mutation sites, and end-to-end through
:meth:`PlotsWindow.style`, which still hands out the live object and is now safe to.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import dataclasses
import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import FrozenMap, PlotsWindow, StyleSettings


# ================================================================================================
# builders
# ================================================================================================

def _events(n: int = 120, *, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(3e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key: str, *, colour: str, label: str, seed: int) -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(seed=seed),
                      provenance={"fs": 250_000.0}, colour=colour, doc_id=key,
                      channel=0, module_id="detect.threshold",
                      calibration_fingerprint=(1.0, 0.0))


@pytest.fixture
def two_sources(qapp, qtbot):
    """Two overlaid pores -- the case where a per-source mapping actually has entries."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([
        _source("a", colour="#0072B2", label="apo_0008_-500 mV.csv", seed=1),
        _source("b", colour="#D55E00", label="apo_0009_-700 mV.csv", seed=2),
    ])
    w.set_sources(["a", "b"])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    return w


# ================================================================================================
# the type itself: frozen fields, immutable collections
# ================================================================================================

def test_the_style_is_frozen_so_no_field_can_be_written_in_place():
    """The first half of "a snapshot is a value": no field may be rebound."""
    s = StyleSettings()
    for field_name, value in (("title", "figure 3b"), ("font_size_pt", 18),
                              ("background", "light"), ("legend_pos", (0.1, 0.2)),
                              ("label_overrides", FrozenMap({"a": "apo"}))):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(s, field_name, value)
    assert s == StyleSettings(), "not one of those writes landed"


def test_every_field_of_a_default_style_is_immutable():
    """A guard for the NEXT field somebody adds.

    The freeze is only as good as its least-frozen member, and the two mappings were
    exactly that hole. Rather than re-listing the fields here (a list that goes stale
    the day one is added), this asks the question that actually matters: is the value
    hashable? ``list``/``dict``/``set`` are not, and a frozen dataclass of hashable
    fields is itself hashable -- which the last line uses as the whole-object check.
    """
    for f in dataclasses.fields(StyleSettings):
        value = getattr(StyleSettings(), f.name)
        try:
            hash(value)
        except TypeError:                                   # pragma: no cover - the guard
            pytest.fail(
                f"{f.name} holds a mutable {type(value).__name__}; freeze it the way "
                "label_overrides/legend_include are frozen, or every snapshot leaks")
    hash(StyleSettings())   # the whole style is a hashable value


def test_replace_does_not_share_its_collection_fields():
    """THE MEASURED TRAP, closed. Pre-freeze this assertion failed on identity."""
    base = StyleSettings().with_label_override("a", "apo").with_legend_include("fit:a", False)
    other = replace(base, title="figure 3b")

    # Same CONTENTS (replace copies the field across) but writing through one must not
    # reach the other -- which is precisely what sharing one dict would have done.
    assert other.label_overrides == base.label_overrides
    assert other.legend_include == base.legend_include

    edited = other.with_label_override("b", "holo").with_legend_include("series:b", False)
    assert dict(base.label_overrides) == {"a": "apo"}, "the original grew an entry"
    assert dict(base.legend_include) == {"fit:a": False}, "the original grew an entry"
    assert dict(edited.label_overrides) == {"a": "apo", "b": "holo"}
    assert dict(edited.legend_include) == {"fit:a": False, "series:b": False}


def test_the_shared_empty_default_cannot_be_polluted():
    """Two fresh styles share one empty map (it is a constant), which is safe only
    because nothing can write to it. If it ever became a dict again, the first
    override anywhere would appear in every default style in the process."""
    a, b = StyleSettings(), StyleSettings()
    assert a.label_overrides == {} and b.label_overrides == {}
    a.with_label_override("a", "apo")           # the result is deliberately dropped
    assert b.label_overrides == {}, "an override leaked into an unrelated fresh style"
    assert StyleSettings().label_overrides == {}


def test_a_frozen_map_reads_exactly_like_the_dict_it_replaced():
    """The ~6 read sites and the tests comparing against a dict literal did not change,
    and this is why: everything they do still works."""
    m = FrozenMap({"a": "apo", "b": "holo"})
    assert m["a"] == "apo"
    assert m.get("b") == "holo"
    assert m.get("missing", "fallback") == "fallback"
    assert "a" in m and "missing" not in m
    assert len(m) == 2 and sorted(m) == ["a", "b"]
    assert m == {"a": "apo", "b": "holo"} and FrozenMap() == {}
    assert m != {"a": "apo"}
    assert dict(m) == {"a": "apo", "b": "holo"}


def test_a_frozen_map_refuses_every_way_of_writing_to_it():
    m = FrozenMap({"a": "apo"})
    with pytest.raises(TypeError):
        m["b"] = "holo"
    with pytest.raises(TypeError):
        del m["a"]
    assert not hasattr(m, "update") and not hasattr(m, "pop")
    assert dict(m) == {"a": "apo"}


def test_a_frozen_map_copies_the_dict_it_was_built_from():
    """Built FROM a dict, not built ON one -- otherwise the caller keeps a handle that
    edits the 'immutable' map from outside."""
    src = {"a": "apo"}
    m = FrozenMap(src)
    src["b"] = "holo"
    assert dict(m) == {"a": "apo"}


def test_set_and_remove_leave_the_original_alone():
    m = FrozenMap({"a": "apo"})
    assert dict(m.set("b", "holo")) == {"a": "apo", "b": "holo"}
    assert dict(m.set("a", "APO")) == {"a": "APO"}
    assert dict(m.remove("a")) == {}
    assert m.remove("nope") is m, "removing an absent key is a no-op, not an error"
    assert dict(m) == {"a": "apo"}, "the original was edited by one of the above"


# ================================================================================================
# through the window: a snapshot taken before a change does not observe the change
# ================================================================================================

def test_style_snapshot_does_not_see_later_changes(two_sources):
    """The precondition for undo/redo, stated as the test :meth:`PlotsWindow.style`
    cites. Six kinds of edit -- a scalar, a colour, the legend position, and BOTH
    mapping fields -- and the snapshot taken first must be untouched by all of them."""
    w = two_sources
    before = w.style()
    assert before.title == "" and before.label_overrides == {}

    w.set_title("Figure 3b")
    w.set_font_size(18)
    w.set_data_color("#009E73")
    w.set_background("light")
    w._source_list.item(0).setText("apo, −500 mV")          # a label override
    w._style = w._style.with_legend_include("series:b", False)
    w._style = replace(w._style, legend_pos=(0.25, 0.5))

    assert before.title == "", "the snapshot tracked the live window"
    assert before.font_size_pt == StyleSettings().font_size_pt
    assert before.data_color == StyleSettings().data_color
    assert before.background == "dark"
    assert before.legend_pos is None
    assert before.label_overrides == {}, "the shared-dict alias is back"
    assert before.legend_include == {}, "the shared-dict alias is back"

    # ...and the window really did change: the snapshot is stale, not merely detached.
    after = w.style()
    assert after is not before
    assert after.title == "Figure 3b" and after.background == "light"
    assert after.data_color == "#009E73"
    assert dict(after.label_overrides) == {"a": "apo, −500 mV"}
    assert dict(after.legend_include) == {"series:b": False}
    assert after.legend_pos == (0.25, 0.5)


def test_two_snapshots_across_one_edit_are_independent_values(two_sources):
    """What an undo stack literally does: push, edit, push, and expect two entries that
    differ in one field and share no mutable state."""
    w = two_sources
    w._source_list.item(0).setText("apo")
    first = w.style()
    w._source_list.item(1).setText("holo")
    second = w.style()

    assert dict(first.label_overrides) == {"a": "apo"}
    assert dict(second.label_overrides) == {"a": "apo", "b": "holo"}
    assert first.label_overrides is not second.label_overrides
    # Restoring the earlier entry must put the window back where it was, entry and all.
    w._style = first
    assert dict(w.style().label_overrides) == {"a": "apo"}


def test_removing_an_override_does_not_reach_back_into_an_older_snapshot(two_sources):
    """The remove path is the other half of ``_on_source_item_changed`` and it used to
    be a ``dict.pop`` on the shared mapping -- an UNDO of a rename that also erased the
    rename from every snapshot that was supposed to remember it."""
    w = two_sources
    w._source_list.item(0).setText("apo")
    renamed = w.style()
    w._source_list.item(0).setText("")      # blanking the row clears the override
    assert w.style().label_overrides == {}
    assert dict(renamed.label_overrides) == {"a": "apo"}


def test_the_legend_checklist_writes_a_new_style_rather_than_a_shared_dict(two_sources):
    """``_on_legend_entry_toggled`` was the one in-place ``legend_include[...] = ...``.
    Ticking still works, and it no longer edits older snapshots."""
    w = two_sources
    lst = w._legend_entries
    assert lst.count() >= 2, "premise: both sources offered a legend row"
    before = w.style()

    row = 0
    entry_id = str(lst.item(row).data(Qt.UserRole))
    lst.item(row).setCheckState(Qt.Unchecked)

    assert before.legend_include == {}, "the tick reached a snapshot taken before it"
    assert w.style().legend_include.get(entry_id) is False
