"""The project copy of plot presets and templates, without Qt (Plot Studio step 6, F1).

What a save writes over the previous copy, how an open names what it adds, and the one
container rule it shares with the table choices: an unreadable section is quarantined, never
overwritten.
"""
from __future__ import annotations

import pytest

from nanoscope.core.plot_preset_state import (
    APPEARANCE_PRESETS, DISTRIBUTIONS_TEMPLATES, UNREADABLE_KEY, WINDOW_TEMPLATES, WORKSPACE_KEY,
    base_name, capture_plot_presets, has_any, merge_carried, prepare_plot_preset_workspace,
)

LIGHT = {"background": "light", "title_size": 14.0}
DARK = {"background": "dark", "title_size": 12.0}


def test_an_unreadable_section_is_quarantined_not_overwritten():
    extra = {WORKSPACE_KEY: ["not", "an", "object"], UNREADABLE_KEY: "an older one"}
    notes = prepare_plot_preset_workspace(extra)
    assert extra[WORKSPACE_KEY] == {}
    assert extra[UNREADABLE_KEY] == "an older one", "an earlier quarantine is not replaced"
    assert extra[UNREADABLE_KEY + "_1"] == ["not", "an", "object"]
    assert notes and "could not be read" in notes[0]
    assert prepare_plot_preset_workspace({WORKSPACE_KEY: {}}) == ()
    assert prepare_plot_preset_workspace({}) == ()
    with pytest.raises(ValueError):
        capture_plot_presets({}, previous=["unreadable"])


def test_a_save_carries_this_computer_and_keeps_what_it_cannot_read():
    previous = {
        APPEARANCE_PRESETS: {"Deleted here": DARK, "Kept": LIGHT},
        DISTRIBUTIONS_TEMPLATES: {"Newer": {"format": "x", "version": 99}},
        WINDOW_TEMPLATES: {"noise-spectrum": {"Old": {"v": 1}},
                           "a-future-window": {"Its own": {"v": 3}}},
        "figure_templates": {"A newer build's family": {}},
    }
    local = {APPEARANCE_PRESETS: {"Kept": DARK},
             DISTRIBUTIONS_TEMPLATES: {},
             WINDOW_TEMPLATES: {"noise-spectrum": {"Welch 8k": {"v": 2}}, "event-waveforms": {}}}

    def keep(family, kind, payload):
        return payload in ({"format": "x", "version": 99}, {"v": 3})

    section = capture_plot_presets(local, previous, keep=keep)
    assert section[APPEARANCE_PRESETS] == {"Kept": DARK}, (
        "replaced, not merged: a preset deleted on this computer is not written back")
    assert section[DISTRIBUTIONS_TEMPLATES] == {"Newer": {"format": "x", "version": 99}}
    assert section[WINDOW_TEMPLATES] == {"noise-spectrum": {"Welch 8k": {"v": 2}},
                                         "a-future-window": {"Its own": {"v": 3}}}
    assert section["figure_templates"] == {"A newer build's family": {}}
    assert previous[APPEARANCE_PRESETS] == {"Deleted here": DARK, "Kept": LIGHT}, "not mutated"


def test_has_any_counts_entries_not_empty_families():
    assert not has_any({APPEARANCE_PRESETS: {}, DISTRIBUTIONS_TEMPLATES: {},
                        WINDOW_TEMPLATES: {"noise-spectrum": {}}})
    assert has_any({WINDOW_TEMPLATES: {"noise-spectrum": {"x": {}}}})
    assert has_any({APPEARANCE_PRESETS: {"x": LIGHT}})


def test_an_open_adds_only_what_is_missing_and_never_replaces_a_local_entry():
    local = {"Publication": LIGHT, "Mine": DARK}
    carried = {"Publication": LIGHT, "Mine": LIGHT, "Theirs": DARK}
    merged, added = merge_carried(local, carried)
    assert merged["Publication"] == LIGHT and merged["Mine"] == DARK, "local entries unchanged"
    assert merged["Mine (from project)"] == LIGHT
    assert merged["Theirs"] == DARK
    assert added == [("Mine", "Mine (from project)"), ("Theirs", "Theirs")]

    merged, added = merge_carried(merged, {"Mine": {"background": "light", "grid": True}})
    assert added == [("Mine", "Mine (from project 2)")]


def test_two_computers_do_not_grow_a_copy_each_trip():
    # A: Mine=LIGHT. B: Mine=DARK, and A's came in as "Mine (from project)". B saves; A opens.
    a_local = {"Mine": LIGHT}
    from_b = {"Mine": DARK, "Mine (from project)": LIGHT}
    merged, added = merge_carried(a_local, from_b)
    assert added == [("Mine", "Mine (from project)")]
    assert merged == {"Mine": LIGHT, "Mine (from project)": DARK}
    # And back to B: nothing new.
    assert merge_carried(from_b, merged)[1] == []
    assert base_name("Mine (from project 3)") == "Mine" == base_name("Mine")


def test_the_writing_build_is_provenance_not_content():
    one = {"format": "nanoscope-plot-config", "app_version": "0.9", "plot_type": "Histogram"}
    other = {**one, "app_version": "1.0"}
    assert merge_carried({"Dwell": one}, {"Dwell": other})[1] == []


def test_a_name_the_computer_shows_but_does_not_store_counts_as_taken():
    # The Distributions window lists its shipped templates ahead of a stored one of the same
    # name, so a carried entry stored under a shipped name could never be picked.
    shipped = {"Shipped": LIGHT, "— manual —": None}
    merged, added = merge_carried({}, {"Shipped": DARK, "— manual —": DARK},
                                  reserved=shipped)
    assert added == [("Shipped", "Shipped (from project)"),
                     ("— manual —", "— manual — (from project)")]
    assert "Shipped" not in merged and "— manual —" not in merged
    assert merge_carried({}, {"Shipped": LIGHT}, reserved=shipped)[1] == [], (
        "a copy of what the computer already ships is not added")
